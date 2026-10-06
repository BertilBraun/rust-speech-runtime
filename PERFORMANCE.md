# Device pipelining and timing experiments

## Rust code quality pass, 2026-10-06

Revisions `15100c8` through `a4c95a9` refactor the implementation without changing the default workload, admission settings or fixed 12 ms batch cost. Worker admission, input validation, scheduling and completion are focused methods on one private worker owner. The simulator separates capture pacing, owned session state and reporting. Recoverable deliveries are explicit `AudioDelivery` variants; terminal transport and CLI failures use typed errors. Cache replay consumes its uploaded history during validation, and submitted device jobs retain only the current packet and verified prefix.

Scoped cancellation guards cover startup, serving and simulation failures. Calibration stops submitting probes after cancellation and lets its running probe finish. New regression tests cover cancellation during calibration and dropping a node during inference. Configuration defaults and validation stay with their canonical types; scheduler internals are private, and Tokio's paused-clock support is restricted to development builds. [The README code guide](README.md#rust-code-guide) maps these responsibilities to their modules.

All **67 tests** pass on Windows and WSL with Rust 1.99.0 and locked dependencies. Windows also passes `cargo clippy --all-targets --locked -- -D warnings`, `cargo fmt --check`, `cargo doc --no-deps --locked` with `RUSTDOCFLAGS=-D warnings`, and `cargo build --release --locked`. These checks validate the code and regression tests; they do not certify realtime capacity.

The first post-refactor 30-second Windows release smoke run offered 448 sessions and admitted 384. It **failed twelve sessions on four-miss clusters**: 216,174 echoes, 596 packets over 50 ms including 149 discarded outputs over 60 ms, and no unrecovered progress failures or lost metric samples. RTT p50/p95/p99/max were 22.1/28.2/40.7/74.0 ms, batch fill 68.0%, and modeled utilization 98.6%. Mean EDF preparation and assembly were 3.98 and 13.67 microseconds. Gateway and simulator used about 1.92 and 1.85 logical cores, respectively. The raw report is `rust-cleanup-windows-448-30s.json`, using revision `a4c95a9`.

The worst traces include overlapping client and gateway runtime delays around ten seconds, as well as elevated queue, completion and result-delivery delays. These observations do not identify their OS-level cause or exclude a refactor regression. This failed run is retained alongside the earlier passing results below; its low median RTT is not evidence of acceptable serving quality.

A sequential control then ran the pre-cleanup revision `9ab05a6` and refactored revision `a4c95a9` on this Windows host, each offering 384 sessions with a 48-session worker cap for 30 seconds. Both retained the other defaults: independent random phases, 48–55 ms packets, fixed 12 ms nonempty batches, 5 ms margin and the agreed quality policy. No builds or other benchmark runs overlapped either measurement.

| Revision | Admitted / rejected | Echoed packets | Over 50 ms / discarded over 60 ms | Failed sessions / four-miss clusters | RTT p50 / p95 / p99 / max, ms | Batch fill | Modeled GPU utilization |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Before cleanup, `9ab05a6` | 382 / 2 | 218,138 | 342 / 12 | 6 / 6 | 22.6 / 28.8 / 35.2 / 77.1 | 68.4% | 98.9% |
| After cleanup, `a4c95a9` | 384 / 0 | 221,641 | 0 / 0 | 0 / 0 | 21.9 / 27.5 / 28.7 / 44.9 | 69.2% | 97.7% |

Both controls had zero unrecovered progress failures and lost metric samples. Before/after mean EDF preparation was 3.92/4.05 microseconds; assembly was 13.66/13.52 microseconds. Gateway average CPU cores were 1.89/2.02, and simulator cores 1.76/1.79. Similar preparation costs and quality failures in the original code do not support attributing the first smoke failure solely to the refactor. However, one short pair, slightly different admitted counts and variable host timing cannot prove equivalent performance or a reliable capacity. No admission defaults were changed to obtain the passing repeat. Raw reports are `rust-cleanup-control-before-384-30s.json` and `rust-cleanup-control-after-384-30s.json` in local `benchmark-results/`.

```powershell
.\target\release\voice-scheduler.exe benchmark --sessions 384 --max-sessions-per-worker 48 --duration-secs 30 --output benchmark-results\rust-cleanup-control-after-384-30s.json
```

## Random starts and independent 48–55 ms packets

This is the intended workload: independent random initial capture phases, followed by an independent 48–55 ms interval for every packet. The readiness barrier completes connection setup before capture starts; random phases represent sessions that started at different times. Perfectly synchronized waves remain a separate stress test, rather than the expected traffic pattern. Socket transfer and echo are real, between separate gateway and simulator processes on Windows loopback. Every nonempty ordinary batch, from one to sixteen packets, still costs exactly 12 ms.

Two unnecessary policy limits were removed. Admission previously rounded the number of batches down before converting to session capacity. The scheduler now estimates the steady packet rate with `floor(compute_budget * batch_size / batch_cost)`, and separately checks how many whole batches fit a burst's playback budget. This never shortens a partial batch. Defaults use a 5 ms margin and a 56-session worker cap: `floor((48 - 5) * 16 / 12) = 57` before that cap, measured host reserve, queue pressure and cache constraints. The node's 448-session cap is an upper bound; calibration may admit fewer.

The other limit was treating one output beyond 60 ms as fatal to the entire session. The agreed target permits isolated discarded outputs while rejecting four misses within any ten-packet window for that session. Output after 50 ms is still late; output after the additional 10 ms grace is discarded. Its validated input advances the cache, allowing the following packets to recover with their complete history. Four queued captures and one active exchange bound outstanding input, and a separate 204 ms progress budget bounds recovery. This larger progress budget is excluded from admission capacity; it does not turn late packets into timely playback. Known unsustainable device slowdowns still revoke sessions, and stalled recovery or queue overflow still fails explicitly.

### Capacity measurements before the Rust cleanup

Both runs below used eight Tokio mock devices, random phases, 48–55 ms packets, sixteen slots, fixed 12 ms compute, a 56-session worker cap, admission headroom 1 and the retained 5 ms margin. All admitted sessions survived and no diagnostic samples were lost.

| Duration | Admitted / rejected | Echoed packets | Over 50 ms / discarded over 60 ms | Failed sessions / four-miss clusters | Largest miss count in ten packets | RTT p50 / p95 / p99 / max, ms | Batch fill | Modeled GPU utilization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 440 / 8 | 509,333 | 343 / 124 | 0 / 0 | 2 | 21.4 / 27.0 / 30.9 / 78.8 | 79.6% | 98.9% |
| 300 s | 408 / 40 | 2,361,062 | 781 / 161 | 0 / 0 | 2 | 21.3 / 26.8 / 30.0 / 75.1 | 73.9% | 99.5% |

Discarded outputs are a subset of late packets, and remain included in echo RTT and lateness statistics. They are not successful playback. In the sustained run, 0.033% exceeded the 50 ms target and 0.0068% exceeded 60 ms; maximum consecutive misses were two. Calibration limited five workers to 48 sessions and admitted 56 on the remaining three. The minute run sustained 440, but does not establish a five-minute 440-session guarantee. Raw reports are `recovery-windows-448-60s.json` and `recovery-windows-448-300s.json` in local `benchmark-results/`, using revision `0a6f3fd` with explicit candidate settings. Revision `8dafc10` makes these settings the defaults and adds a CLI scheduling-margin override and max RTT to the console summary.

A separate 60-second ceiling diagnostic removed the 5 ms margin and raised the hard cap to 64 sessions per worker. It admitted 493 and rejected 19, echoed 564,569 packets, and failed **seven sessions on four-miss clusters**, including consecutive misses. Of 422 late packets, 53 outputs were discarded; there were no unrecovered progress failures or lost diagnostic samples. RTT p50/p95/p99/max were 22.2/28.7/34.1/65.4 ms, batch fill 88.3%, and modeled utilization 99.0%. This failed the agreed quality criterion despite a low p99. Offered load fell as those sessions closed, so it does not establish sustained 493-session capacity. The default retains its margin and lower cap. The raw report is `recovery-windows-512-60s.json`, using revision `8dafc10` with explicit diagnostic overrides.

RTT is a packet's latency, rather than a measurement of device idle time. A GPU can answer one session in 21 ms and spend the rest of that session's interval processing other sessions. These runs kept modeled devices approximately 99% busy. More throughput therefore requires fuller batches, with additional queue residence consuming some of the available RTT slack. At full batches the ideal rate is `8 * 16 / 0.012 = 10,667` packets/second, or about 549 sessions at the 51.5 ms mean interval, before host overhead and admission reserves. The current 64 cache slots per worker also cap this implementation at 512 simultaneous sessions. Neither ideal arithmetic nor a low p99 certifies that ceiling.

In the five-minute run, mean EDF preparation took 2.19 microseconds and batch assembly 8.19 microseconds. Mean scheduler queue residence was 5.98 ms and device queue residence 1.95 ms; those are waiting intervals, rather than six milliseconds spent building a batch. Of 199,708 batches, 199,038 were submitted before their predecessor finished. Gateway and simulator each used about 1.12 logical CPU cores on average. Maximum host completion delay was 34.2 ms, so low average CPU use did not eliminate timing tails or establish their OS-level cause.

Earlier attempts at this higher capacity exposed the old fatal-on-first-expiry policy. A Windows random run admitted 392 then failed 341 sessions, including three four-miss clusters. A WSL run admitted 448 and failed 148, with no four-miss clusters; a Windows native-sleep run admitted 448 and failed 45, also with no clusters. These failed runs lose offered load and cannot establish sustained throughput. Reports are `streaming-windows-400-60s.json`, `streaming-linux-448-60s.json` and `streaming-sleep-windows-448-60s.json`. The recovery changes were tested explicitly for discarded-output history, cache replay, fatal progress expiry, slowdown revocation and complete latency accounting.

At these revisions, all 65 tests passed on Windows and WSL with Rust 1.99.0 and locked dependencies. Windows also passed `cargo clippy --all-targets --locked -- -D warnings` and `cargo fmt --check`. WSL shares the Windows host and is not bare-metal Linux; these measurements do not identify the cause of OS timing stalls.

The final five-second-per-scenario Windows suite passed ordinary low/50%/80%/95% load, overload admission, aligned/random arrivals, jitter and churn without session or quality failures. Overload admitted 424 and rejected 392; churn admitted 1,423 across successive lifetimes. Aligned traffic at 204 sessions had 39 isolated late echoes but no clusters. Forced cache replay, 30 ms slowdown and artificially blocked mailboxes deliberately failed 159, 204 and 204 sessions respectively; saturation recorded 95 full-mailbox events. These stress outcomes are explicit failures, not successful service. The raw report is `recovery-current-windows-suite-5s.json`, using revision `8dafc10` and current defaults. Short suite checks do not replace the sustained random workload results above.

```powershell
cargo run --release -- benchmark --sessions 448 --duration-secs 300 --output benchmark-results\random-current.json
cargo run --release -- benchmark --sessions 512 --duration-secs 60 --max-sessions-per-worker 64 --scheduling-margin-ms 0 --output benchmark-results\zero-margin-diagnostic.json
```

The second command deliberately removes the remaining scheduling margin and is a capacity diagnostic. The default retains that margin. Use `--lateness-grace-ms 0 --quality-miss-limit 1` to make the simulator end a session on its first missed target; a playback cutoff alone leaves cached-input recovery enabled.

## Earlier capacity experiment with fatal playback expiry

The revised serving target keeps the 50 ms capture-to-echo deadline, allows up to 10 ms of recovery, and flags four misses in any ten-packet window **within the same session**. A late echo advances both audio prefixes, so subsequent packets keep their original worker and cache. The simulator preserves the original capture schedule and records every late echo; it does not shift deadlines to make results look timely. A hard timeout/rejection remains a separate session failure. These are serving-quality proxies, not a perceptual audio test.

At this earlier revision, default admission capped each GPU at 48 sessions, or 384 across eight workers. The compute budget was `(48 - 5) * 0.9 = 38.7 ms`, rounded to three whole 12 ms batches. Grace absorbed measured host-delay reserve before excess host delay was subtracted. Queue pressure and cache availability could still lower admission. The current packet-rate estimate and independent burst check supersede that whole-batch rounding.

The following sequential runs used real loopback TCP, 1,600-byte packets, fixed 12 ms kernels, eight workers and the agreed quality policy:

| Platform / workload | Duration | Admitted / rejected | Echoed packets | Late echoes | Session failures / quality bursts | RTT p50 / p95 / p99 / max, ms | Batch fill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Windows, random; cap raised to 48 | 60 s | 336 / 64 | 389,209 | 0 | 0 / 0 | 20.9 / 26.3 / 27.2 / 37.0 | 60.8% |
| WSL, random; cap raised to 48 | 60 s | 384 / 16 | 446,810 | 0 | 0 / 0 | 20.6 / 26.0 / 26.8 / 47.3 | 69.8% |
| Windows, random; new defaults | 300 s | 368 / 32 | 2,130,422 | 13 | 0 / 0 | 20.9 / 26.3 / 27.9 / 51.3 | 66.7% |

All admitted sessions survived these runs, every attempted packet was echoed, and no client metric samples were lost. The five-minute run's thirteen late packets arrived at most 1.285 ms beyond the target. Maximum consecutive misses and maximum misses within any ten-packet window were both **one**, so there were no four-packet clusters. Calibration limited one worker to 32 sessions while seven accepted 48; the 384-node cap is an upper bound, not a forced admission count. The former 256 cap is therefore too restrictive for this revised tolerance: the sustained Windows run served 44% more sessions, and the WSL minute run served 50% more.

Perfectly synchronized 50 ms arrivals need a separate qualification. A 60-second Windows run with the new defaults admitted 320 sessions, then failed 71: 27 crossed the four-miss quality threshold and 44 hit unrecovered deadline failures. It recorded 2,811 recoverable late echoes. Its p99 of 49.4 ms and 91.4% fill do **not** make that a passing operating point. A dedicated-device-thread repeat admitted 384 but failed 97, including thirteen quality bursts and 84 unrecovered deadline failures. The scheduler still charged exactly 12 ms per nonempty batch in both modes. Burst traces include approximately 23 ms of device queue residence, 7–8 ms of host notification delay and additional client/network delays; dedicated device threads did not remove the problem. These failed runs lose offered load as sessions terminate, so their aggregate throughput and fill cannot establish sustained capacity.

The 48-session-per-GPU cap therefore targets the tested random 48–55 ms workload and is not certified for arbitrary correlated arrival patterns. Keep admission headroom and quality counters, and use a lower configured cap when traffic is synchronized. Reports for this stress check are `quality-default-windows-aligned-60s.json` and `quality-sleep-windows-aligned-60s.json`. The model has no observability into a new client's future arrival pattern at admission; changing the numeric cap does not solve that uncertainty.

A further 60-second aligned Windows run at the former 32-per-GPU cap admitted 256 sessions and echoed all 303,348 attempted packets, but recorded 2,329 late echoes and one four-miss cluster. It had no unrecovered deadline failures; maximum consecutive misses were two. RTT p99/max were 48.4/58.5 ms. This was much less severe than the higher-cap burst runs but still failed the agreed clustered-miss criterion, so the lower cap is not certified either. Its report is `quality-tokio-windows-aligned-256-60s.json`. Investigating correlated host/network scheduling remains separate from the demonstrated capacity increase for jittered traffic.

Reports are `quality-host-grace-windows-384-60s.json`, `quality-host-grace-linux-384-60s.json` and `quality-default-windows-300s.json` under local `benchmark-results/`. The first two used revision `7c0e315`; the sustained run used the same fixed-cost scheduler plus `d7677fc` recovery-budget checks and the new cap in `a4b2f24`. The earlier `quality-windows-384-60s.json` attempted a raised cap before host reserve used grace, and admitted only 256; its zero failures are not evidence of 384-session service.

These reports used the earlier cap and fatal playback-expiry behavior. Their RTT includes echoes accepted within the former 60 ms recovery budget, including late ones; lateness, unrecovered deadline failures and quality bursts are separate. Older reports below terminated sessions on the first missed deadline and cannot reveal whether misses would have clustered.

Validation covers late-prefix recovery, replay after a late packet, rejection beyond the recovery budget, bounded miss windows and consecutive/nearby clusters, alongside existing placement, cancellation, overload and device pipeline tests. All 63 tests pass on Windows and WSL. Windows validation also passes `cargo clippy --all-targets --locked -- -D warnings` and `cargo fmt --check`; both platforms run `cargo test --locked` with the same toolchain and locked dependencies.

## Previous strict-deadline experiments

Every ordinary nonempty batch costs 12 ms, whether it contains one, eight or sixteen packets. EDF selection is maintained while earlier inference runs. The scheduler prequeues full batches and the final partial batch when every assigned session has outstanding input. Other partial successors use a configurable 2 ms launch lead. Device queues hold at most two waiting jobs plus one running job. Already submitted jobs execute serially on an autonomous simulated device timeline; delayed host observation remains part of the actual TCP round trip and can fail its deadline.

The former multi-millisecond "batch overhead" included queue residence. Current profiles separate EDF selection, job assembly, scheduler waiting and device waiting. Typical EDF selection costs roughly 0.6–1.2 microseconds; assembly costs roughly 2.3–6.7 microseconds in the experiments below. This is not six milliseconds of CPU work before each kernel.

## Why a half-filled batch can keep a device busy

With 32 sessions on one GPU and a 50 ms input interval, the worker receives approximately 640 packets/second. The batch execution duration stays fixed in both rows:

| Average packets per batch | Required batches/second | Modeled device time/second | Utilization |
| --- | --- | --- | --- |
| 8 | 80 | 960 ms | 96% |
| 16 | 40 | 480 ms | 48% |

Random arrivals collect about eight new packets during a 12 ms execution. Aligned arrivals can provide sixteen immediately. Improving fill requires enough simultaneously available inputs or additional waiting within the packet budget. Utilization is simulated device occupancy, independent of CPU usage. A mock device waiting on a timer consumes little CPU even while its modeled execution is busy.

## Method

Measurements used real loopback TCP between the gateway and a separate simulator process, eight workers, a sixteen-packet maximum batch, 1,600-byte audio packets, a 50 ms capture-to-echo deadline and fixed 12 ms ordinary inference. Random workloads use independently chosen 48–55 ms capture intervals. Aligned workloads use one shared capture timestamp per 50 ms wave. Capture timing does not wait for responses and never catches up with a burst after a delayed wakeup.

Preparation and assembly profiles measure elapsed wall time around those code scopes, so preemption can inflate a tail sample. Actual CPU seconds are measured separately for each process and, in native wait mode, each device thread. Stage percentiles cannot be added to produce a round-trip percentile.

Windows runs used Windows 11 Home build 26200 and an Intel i7-11370H with four physical/eight logical cores. Linux comparisons used Ubuntu under WSL2, kernel 6.6.87.2-microsoft-standard-WSL2, with four vCPUs. Both used Rust 1.99.0 and the same locked dependencies. WSL shares the Windows host and is not a bare-metal Linux comparison. Benchmarks ran sequentially without concurrent builds or competing benchmark runs.

These historical strict-deadline runs capped admission at 32 sessions per worker, or 256 for the node, and could admit fewer after calibration. Before host-delay reserves, their compute budget was `(48 - 5) * 0.9 = 38.7 ms`. Admission also checked cache availability and queued compute/replay work. The historical raised-cap experiments used 48 sessions per worker; failures in those runs remain documented below.

## Validation with strict simulator deadline measurements

Revision `066ca33` completed another five-minute Windows run. Calibration admitted 224 sessions and rejected 176 attempts; 1,282,544 packets were echoed and three sessions failed explicit server deadline checks. Successful RTT p50/p95/p99/max were 20.7/26.1/27.0/44.5 ms, with 40.4% batch fill. The two initially reduced workers admitted sixteen sessions each, while the other six admitted thirty-two. This is a useful improvement over the original 64-session prototype, but three lost sessions still fail the requested zero-drop operating target.

Revision `a11772c` additionally rejects cache recovery that cannot fit even its minimum compute cost, before requesting a prefix upload or hashing a submitted prefix. It also reconciles worker removals before the shutdown session gauge is recorded. Its final WSL measurements were:

| Workload | Duration | Admissions | Capacity rejections | Failed sessions | Echoed packets | RTT p50 / p95 / p99 / max, ms | Batch fill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random, 400 initial attempts | 300 s | 256 | 144 | 0 | 1,489,334 | 20.6 / 26.0 / 26.6 / 31.4 | 46.5% |
| Aligned, 400 initial attempts | 60 s | 256 | 144 | 0 | 305,920 | 20.5 / 29.3 / 30.1 / 31.7 | 100% |
| Churn, 240 concurrent slots | 60 s | 9,471 total | 15 | 0 | 276,140 | 20.5 / 26.0 / 26.6 / 41.8 | 43.2% |
| Churn/replay, 32 concurrent slots | 60 s | 1,282 total | 0 | 0 | 37,165 | 19.4 / 25.6 / 27.5 / 34.6 | 8.6% |

The random five-minute run used about 0.229 gateway CPU cores and 0.326 simulator cores, with mean preparation/assembly durations of 0.62/2.49 microseconds. Of 200,004 batches, 199,968 were submitted before their predecessors completed. The churn/replay run completed 3,044 replay frames containing 53,680 preceding packets, or 85.9 MB. All four reports had zero lost client metric samples and zero physical completion overruns. The final suite again had no failures in its ordinary load/arrival/churn scenarios; explicit cache-replay, slowdown and saturation stress failed 95, 120 and 128 sessions. These clean runs coexist with the failed runs below and do not establish an unconditional zero-failure capacity.

Revision `44471d3` closes a final admission gap: current configured device cost is a lower bound on admission and published capacity estimates, including host slack, before the first slower result arrives. Steady fixed-12-ms service uses the same cost as in the table. All 57 tests pass on Windows and WSL. `cargo test --locked`, `cargo clippy --all-targets --locked -- -D warnings` and `cargo fmt --check` pass on Windows; Linux uses the same locked dependencies and passes `cargo test --locked`.

Both release binaries were rebuilt after this fix. A ten-second Windows smoke run admitted 256 sessions, rejected 144 and echoed all 49,411 packets without failures. The current WSL suite again completed every ordinary load/arrival/churn scenario without failures; deliberate replay, slowdown and saturation stress failed 92, 120 and 128 sessions. Local reports are `final-windows-smoke-10s.json` and `final-current-linux-suite-5s.json`. This short smoke run does not supersede the Windows failures in the longer runs.

Raw reports are `final-observation-windows-300s.json`, `final-linux-default-300s.json`, `final-linux-aligned-60s.json`, `final-linux-churn-60s.json`, `final-linux-cache-churn-60s.json` and `final-linux-suite-5s.json`, under the ignored local `benchmark-results/` directory. Server stage profiles are retained when a late echo is actually received; timeouts and server rejections cannot provide an unavailable echo's stage breakdown. Admission rejection, prelaunch rejection, physical completion overruns and client deadline failures are distinct outcomes.

## Earlier validation before strict simulator observation checks

Revision `45cc867` completed a five-minute Windows run with the default settings: 256 admissions, 144 capacity rejections, 1,467,301 echoes and 18 failed sessions. Successful RTT p50/p95/p99/max were 20.6/26.1/26.7/50.0 ms; batch fill was 46.0%. The 18 failures were detected by the external client's original capture deadline. The server reported zero physical deadline overruns, demonstrating why client failures must also be checked.

EDF preparation averaged 0.88 microseconds and batch assembly 3.55 microseconds, with p99 values of 3.10 and 16.30 microseconds respectively. Of 199,575 batches, 198,650 were submitted before their predecessors completed. Gateway and simulator used about 0.415 and 0.367 logical cores. Maximum host completion delay was 16.8 ms; both process runtime monitors recorded timing spikes at corresponding wall-clock times. This is evidence of host scheduling delay, but does not identify its OS-level cause.

The final five-second-per-case WSL suite had no failures in low/50%/80%/95% load, overload admission, aligned/random arrival, jitter or churn scenarios. Overload attempted 512 sessions, admitted 256 and rejected the remaining 256. Intentional long-prefix replay, 30 ms worker slowdown and channel saturation failed 92, 120 and 128 sessions respectively, with explicit reasons in the report. Those stress failures are not counted as successful realtime service.

The repeat 60-second 240-slot WSL churn run at this revision encountered an early timing spike, recorded 181 failed sessions and 13 capacity rejections, and completed 66,286 echoes. Maximum device observation delay was 22.5 ms. A separate 32-slot churn/replay run again had zero failures: 1,286 admissions, 37,169 echoes, 3,043 replay frames and 85.8 MB of replayed history. The earlier clean churn run below therefore does not establish sustained reliability.

Local reports are `validated-windows-default-300s.json`, `validated-linux-suite-5s.json`, `validated-linux-churn-60s.json` and `validated-linux-cache-churn-60s.json`. All 52 tests passed on Windows and WSL at this revision; strict Windows Clippy and formatting checks passed.

Revision `066ca33` retains server stage profiles for echoes received after the deadline. It also checks the simulator's observation time, closing a gap where host preemption after transport acceptance could put an over-deadline sample into successful RTT statistics. The older churn report contains a 58.8 ms successful sample through this gap; that sample is not evidence of successful realtime delivery. Late observations now fail the session and are excluded from successful latency distributions. Echo-less timeouts cannot provide server stage timings and remain explicitly separate failure traces.

## Additional sustained results before the queued-work admission check

These runs used revision `f4cd440`. The subsequent `45cc867` adds admission pressure from work already queued on the device; it does not shorten kernels or hide timing failures.

| Platform and workload | Duration | Initially admitted | Failed sessions | Echoed packets | Successful RTT p50 / p95 / p99 / max, ms | Batch fill |
| --- | --- | --- | --- | --- | --- | --- |
| WSL, random, default cap | 300 s | 256 | 47 | 1,287,039 | 20.6 / 26.0 / 26.6 / 49.7 | 40.3% |
| WSL, random, raised cap | 300 s | 384 | 113 | 1,725,663 | 20.6 / 26.0 / 26.7 / 49.9 | 54.7% |
| WSL, aligned, default cap | 60 s | 256 | 0 | 305,664 | 20.0 / 29.0 / 29.7 / 31.3 | 100% |
| WSL, aligned, raised cap | 60 s | 384 | 28 | 447,988 | 27.9 / 40.9 / 42.3 / 48.7 | 97.6% |

Successful RTT distributions exclude failed packets. A 26.6 ms p99 does not establish reliable service when 47 sessions failed. When a session terminates, subsequent offered load falls; aggregate utilization and latency then include time at lower load. The aligned 384-session run rejected 28 packets before launch because their projected completion could not meet the deadline; zero physical kernel overruns did not mean zero failed sessions.

In the five-minute 256-session WSL run, maximum host completion delay was 26.7 ms and the runtime monitor observed a 25.2 ms timer delay. Gateway and simulator used about 0.23 and 0.31 logical cores respectively. That is not evidence of sustained CPU saturation, nor does it identify an interrupt, hypervisor or OS scheduling mechanism. Native Linux measurements and system-level scheduling traces are still needed to attribute the stalls. No reliable 256- or 384-session zero-failure ceiling is established by these runs.

Revision `f4cd440` also completed a 60-second churn workload with 240 concurrent session slots, random one-to-two-second lifetimes, 9,568 total admissions and 279,212 echoes without failures. Its successful RTT p99/max were 26.7/33.7 ms. A separate 32-slot churn/replay run completed 1,285 admissions, 37,174 echoes and 3,043 full-prefix replay frames without failures; those replays transferred 53,610 preceding packets, or 85.8 MB. RTT p99/max were 27.5/33.2 ms. Neither report lost client metric samples.

Raw local reports for these rows are under `benchmark-results/`: `linux-default-256-300s.json`, `linux-raised-384-300s.json`, `linux-aligned-256-60s.json`, `linux-aligned-384-60s.json`, `linux-churn-240-60s.json`, and `linux-cache-churn-32-60s.json`.

## Native wait diagnostics

The native sleep, short-poll and hybrid-spin modes all use the same fixed-cost kernel model and bounded device pipeline. Earlier 60-second Windows diagnostics with 256 admissions recorded three failed sessions for `poll:500ns`, and thirty for `hybrid:1ms`. The hybrid run reduced typical observation delay but increased gateway CPU usage to about 0.87 cores versus 0.47 for polling; its eight device threads accounted for approximately 0.41 cores versus 0.07. Short sleeps and spinning did not establish reliable delivery.

A requested 500 ns sleep is not a wakeup guarantee. Expired waits bypass sleeping so a zero-duration Windows sleep cannot introduce an unnecessary yield. The default remains the event-driven Tokio completion task. Native strategies are diagnostic options, not a claim that CPU spinning accurately executes GPU work. Local reports are `final-poll-256-60s.json` and `final-hybrid-256-60s.json`; these precede the final five-millisecond idle collection and admission changes.

## Reproduction and interpretation

```powershell
cargo build --release --locked
.\target\release\voice-scheduler.exe benchmark --sessions 400 --duration-secs 300 --output benchmark-results\default-300s.json
.\target\release\voice-scheduler.exe benchmark --sessions 400 --duration-secs 300 --max-sessions-per-worker 48 --output benchmark-results\raised-300s.json
.\target\release\voice-scheduler.exe benchmark --sessions 400 --duration-secs 60 --phase aligned --min-interval-ms 50 --max-interval-ms 50 --output benchmark-results\aligned-60s.json
.\target\release\voice-scheduler.exe benchmark --sessions 240 --duration-secs 60 --churn-secs 2 --output benchmark-results\churn-60s.json
.\target\release\voice-scheduler.exe benchmark --sessions 32 --duration-secs 60 --churn-secs 2 --evict-every 10 --output benchmark-results\replay-60s.json
.\target\release\voice-scheduler.exe suite --duration-secs 5 --output benchmark-results\suite.json
```

The suite reports low, 50%, 80%, 95% and overload admission, aligned/random arrivals, jitter, churn, forced cache replay, worker slowdown and channel saturation. Slowdown and artificially blocked consumers intentionally fail work. Long-lived cache replays can also become impossible within 50 ms: the model charges 100 microseconds per preceding packet in addition to the fixed kernel duration. Short churn/replay experiments exercise recovery without treating arbitrary amounts of prefix recomputation as free.

Continuous sessions hit the default 10,000-packet prefix bound after approximately 8.6 minutes. Longer soaks must explicitly configure a larger bound in `RuntimeConfig` or use churn; prefix-limit termination must not be mistaken for a capacity failure. The fake cache records a GPU-local prefix fingerprint and bounded slot ownership, not a real model's KV tensors. Loopback transport also does not measure WAN latency or jitter.

The prototype now overlaps scheduling with inference, retains fixed kernel cost, preserves sticky cache ownership, rejects predictable overload and reports failures honestly. The remaining timing tails are measurable limitations of this mock/runtime environment, not a reason to silently accept late audio.
