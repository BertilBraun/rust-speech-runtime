use cpu_time::ProcessTime;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CpuUsage {
    pub cpu_secs: f64,
    pub elapsed_secs: f64,
    pub average_cores: f64,
}
impl CpuUsage {
    pub(crate) fn measured(cpu: Duration, elapsed: Duration) -> Self {
        Self {
            cpu_secs: cpu.as_secs_f64(),
            elapsed_secs: elapsed.as_secs_f64(),
            average_cores: cpu.as_secs_f64() / elapsed.as_secs_f64().max(f64::EPSILON),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "executor", content = "usage", rename_all = "snake_case")]
pub enum DeviceCpuUsage {
    DedicatedThread(CpuUsage),
    SharedRuntime,
}

pub(crate) struct ProcessCpuMeasurement {
    cpu: ProcessTime,
    wall: Instant,
}
impl ProcessCpuMeasurement {
    pub(crate) fn start() -> std::io::Result<Self> {
        Ok(Self {
            cpu: ProcessTime::try_now()?,
            wall: Instant::now(),
        })
    }
    pub(crate) fn finish(self) -> std::io::Result<CpuUsage> {
        Ok(CpuUsage::measured(
            self.cpu.try_elapsed()?,
            self.wall.elapsed(),
        ))
    }
}
