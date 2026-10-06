use crate::{
    config::RuntimeConfig,
    metrics::Metrics,
    protocol::{ErrorCode, SessionId},
    runtime::{RuntimeError, SessionHandle},
    worker::{Command, WorkerHandle},
};
use futures_util::{StreamExt, stream::FuturesUnordered};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub(crate) enum ManagerCommand {
    Open {
        session_id: SessionId,
        reply: oneshot::Sender<Result<SessionHandle, RuntimeError>>,
    },
    Release {
        session_id: SessionId,
        key: String,
    },
}
struct Route {
    key: String,
    worker_id: usize,
    cancellation: CancellationToken,
}
struct OpenCompletion {
    reply: oneshot::Sender<Result<SessionHandle, RuntimeError>>,
    result: Result<SessionHandle, RuntimeError>,
}

pub(crate) async fn run_manager(
    mut receiver: mpsc::Receiver<ManagerCommand>,
    sender: mpsc::Sender<ManagerCommand>,
    workers: Vec<WorkerHandle>,
    config: RuntimeConfig,
    metrics: Arc<Metrics>,
    cancellation: CancellationToken,
) {
    let mut routes = HashMap::<SessionId, Route>::new();
    let mut pending = FuturesUnordered::new();
    let mut sequence = 0u64;
    let mut reap = tokio::time::interval(Duration::from_millis(100));
    loop {
        tokio::select! {
            _=cancellation.cancelled()=>break,
            _=reap.tick()=>routes.retain(|_,route|!route.cancellation.is_cancelled()),
            Some(completion)=pending.next(),if !pending.is_empty()=>{let OpenCompletion{reply,result}=completion;if result.is_err(){metrics.rejected_sessions.fetch_add(1,Ordering::Relaxed);}let _=reply.send(result);},
            command=receiver.recv()=>match command{
                Some(ManagerCommand::Release{session_id,key})=>{if routes.get(&session_id).is_some_and(|route|route.key==key){routes.remove(&session_id);}},
                Some(ManagerCommand::Open{session_id,reply})=>{
                    routes.retain(|_,route|!route.cancellation.is_cancelled());
                    if routes.contains_key(&session_id){metrics.rejected_sessions.fetch_add(1,Ordering::Relaxed);let _=reply.send(Err(RuntimeError::new(ErrorCode::SessionExists,"session ID already active")));continue;}
                    let loads=workers.iter().map(|worker|worker.load().max(routes.values().filter(|route|route.worker_id==worker.id).count())).collect::<Vec<_>>();
                    let worker=workers.iter().filter(|worker|loads[worker.id]<config.max_sessions_per_worker && worker.available()).min_by_key(|worker|(loads[worker.id],worker.id));
                    let Some(worker)=worker else{metrics.rejected_sessions.fetch_add(1,Ordering::Relaxed);let _=reply.send(Err(RuntimeError::new(ErrorCode::CapacityExceeded,"no worker has a free session slot")));continue;};
                    sequence+=1;let key=format!("{sequence}:{}",session_id.0);let session_cancellation=cancellation.child_token();let generation=Arc::new(AtomicU64::new(0));let (events,event_receiver)=mpsc::channel(config.event_capacity);let (open_reply,opened)=oneshot::channel();
                    if let Err(error)=worker.send(Command::Open{key:key.clone(),session_id:session_id.clone(),events,generation:generation.clone(),cancellation:session_cancellation.clone(),reply:open_reply}){metrics.rejected_sessions.fetch_add(1,Ordering::Relaxed);let _=reply.send(Err(error));continue;}
                    routes.insert(session_id.clone(),Route{key:key.clone(),worker_id:worker.id,cancellation:session_cancellation.clone()});
                    let handle=SessionHandle{key,worker_id:worker.id,worker:worker.clone(),events:event_receiver,generation,cancellation:session_cancellation.clone(),manager:sender.clone(),session_id};
                    pending.push(async move{
                        let result=tokio::select!{_=session_cancellation.cancelled()=>Err(RuntimeError::new(ErrorCode::Shutdown,"session opening cancelled")),result=opened=>result.unwrap_or_else(|_|Err(RuntimeError::new(ErrorCode::BackendUnavailable,"worker stopped")))};
                        OpenCompletion{reply,result:result.map(|()|handle)}
                    });
                }
                None=>break,
            }
        }
    }
    for route in routes.into_values() {
        route.cancellation.cancel();
    }
}
