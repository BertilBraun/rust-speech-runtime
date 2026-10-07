# Local pipeline validation — 2026-10-06

The Rust gateway, scheduler, Python worker protocol and ordinary WebSocket clients have been exercised together on Windows, with additional Rust tests on Ubuntu/WSL. These are synthetic serving measurements. No trained checkpoint, CUDA inference, GPU cache parity or rented GPU node was tested; see [hardware validation](HARDWARE_VALIDATION.md).

The runs below retain their original 48–55 ms packet cadence. On 7 October the default workload changed to 100 ms packets for the final 10 Hz model target. These historical reports have not been rerun or relabeled; current integration checks are recorded in [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md).

## Revisions and evidence

The broad release scenarios used Rust `9cc4da3` and the Python idle-frame fix `8f63afb`. The later Python `54ecefa` changes real-model timing attribution, without changing the fixture protocol. The final archive and transport audit used Rust `e010a42`; `055aa05` subsequently extracted session cleanup without changing behavior, and passed the final Rust gates. Test-only process cleanup followed in Rust `74a5553` and Python `0507388`, with the cross-language test explicitly repeated afterward. Raw JSON reports, configurations, logs and conversation archives remain locally under ignored `benchmark-results/local-turn-pipeline-20261006/`. They are deliberately excluded from Git because session archives contain actual input and output data.

## Validation gates

Windows Rust 1.99.0:

```powershell
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
cargo test --locked --test transport python_worker_process_to_websocket_client_full_pipeline -- --ignored
$env:RUSTDOCFLAGS = '-D warnings'
cargo doc --no-deps --locked
cargo build --release --locked
```

Final default tests: **47 passed** (14 library, 17 runtime, 16 transport), with the external Python process test ignored by default. That test was separately executed successfully, verifying four conversations, two turns each, and **104 accepted model tokens**, including EOS. Formatting, strict Clippy, Rustdoc and release builds passed.

Ubuntu/WSL ran the complete earlier Rust suite: 43 passed, one ignored. After the final production fixes and cleanup extraction, the affected transport suite passed again: **16 passed, one ignored**. Both archive library tests also passed on Linux. This is explicit affected-area coverage, rather than a claim that the final entire suite was repeated on Linux.

```powershell
wsl.exe -d Ubuntu --cd /mnt/c/Projects/Rust-Realtime-Voice-Scheduler --exec env RUSTUP_HOME=/home/ubuntu/voice-scheduler-linux-20261006/rustup CARGO_HOME=/home/ubuntu/voice-scheduler-linux-20261006/cargo CARGO_TARGET_DIR=/home/ubuntu/voice-scheduler-linux-20261006/target /home/ubuntu/voice-scheduler-linux-20261006/cargo/bin/rustup run 1.99.0 cargo test --locked --test transport
# Same environment, with: cargo test --locked --lib transport::archive
```

Existing WSL configuration warnings about unsupported `pageReporting` and `autoMemoryReclaim` keys were left unchanged. Backend validation ran `uv run ruff format`, `uv run ruff check --fix` and `uv run pytest -q`: **54 CPU tests passed, two GPU tests skipped** on final fixture commit `0507388`. The backend author repeated the full Python gates after the timing and fixture changes.

## Synthetic setup

Two independent fixture processes used ports 19100 and 19101, with a **fixed whole-batch** 20 ms prefill and 12 ms decode cost, maximum batch 16 and 192 ordinary response tokens followed by EOS. Input was 250 ms of silence per turn, sent as PCM16 packets with intervals selected between 48 and 55 ms. EOS is a model token, not a word, and is emitted/archived even with empty text.

```powershell
# From backend/; repeat on port 19101 in a second process.
uv run python tests/fixture_worker.py --port 19100 --response-tokens 192 --prefill-ms 20 --decode-ms 12 --max-sessions 64
```

`runtime-fixture.json` specified both endpoints, 16 resident sessions and eight active turns per worker, batch 16, mailboxes/event queues 256, 4 model tokens/s, 0.8 admission headroom and 2,000 ms maximum prefill wait. Its initial forward estimate was **20 ms because this fixture's cost is known**, not an estimate for a 3090 or the actual model. Model defaults remain conservative until actual profiles are measured.

```powershell
.\target\release\voice-scheduler.exe serve --runtime-config .\benchmark-results\local-turn-pipeline-20261006\runtime-fixture.json --listen 127.0.0.1:18080 --archive-directory .\benchmark-results\local-turn-pipeline-20261006\verified-archives
.\target\release\voice-scheduler.exe benchmark --url ws://127.0.0.1:18080/v1 --sessions 16 --turns 1 --utterance-ms 250 --start-spread-ms 0 --think-ms 0 --report .\benchmark-results\local-turn-pipeline-20261006\verified-aligned-16.json
.\target\release\voice-scheduler.exe benchmark --url ws://127.0.0.1:18080/v1 --sessions 80 --turns 1 --utterance-ms 250 --start-spread-ms 0 --think-ms 0 --report .\benchmark-results\local-turn-pipeline-20261006\verified-overload-80.json
```

## Client-observed results

The broader scenarios below used the default 500 ms randomized start spread except the explicitly aligned overload. Think time was zero. Every admitted healthy response lasted more than two seconds, so the benchmark exercised its two-second rolling throughput windows, sampled every 100 ms even while no tokens arrived.

| Report | Opened / rejected sessions | Completed / interrupted / rejected turns | Accepted model tokens | Unexpected client failures |
| --- | --- | --- | --- | --- |
| `final-healthy-16.json` (16 sessions, three turns) | 16 / 0 | 48 / 0 / 0 | 9,264 | 0 |
| `final-churn-8x3.json` (eight sessions, three rounds, three turns) | 24 / 0 | 72 / 0 / 0 | 13,896 | 0 |
| `final-interruption-8x2.json` (eight sessions, two rounds, four turns, interrupt at token three) | 16 / 0 | 0 / 64 / 0 | 192 | 0 |
| `final-overload-80.json` (80 aligned sessions, one turn) | 32 / 48 | 16 / 0 / 16 | 3,088 | 0 |
| `verified-aligned-16.json` (final transport, 16 aligned sessions) | 16 / 0 | 16 / 0 / 0 | 3,088 | 0 |
| `verified-overload-80.json` (final transport, 80 aligned sessions) | 32 / 48 | 16 / 0 / 16 | 3,088 | 0 |

Every row above had zero token gaps exceeding 250 ms and zero sessions violating the rolling four-model-token/s objective. Resident-session admission and active-turn admission are counted separately; admitting 32 idle conversations does not promise 32 simultaneous generations.

| Report | TTFT p50 / p95 / p99 / max (ms) | Token gap p50 / p95 / p99 / max (ms) |
| --- | --- | --- |
| Healthy 16, three turns | 35.0 / 69.8 / 70.6 / 70.6 | 13.8 / 15.2 / 34.1 / 166.0 |
| Churn 8 × 3 | 25.6 / 35.7 / 47.2 / 47.2 | 13.4 / 14.2 / 16.2 / 58.8 |
| Final aligned 16 | 90.8 / 162.2 / 162.2 / 162.2 | 14.6 / 15.4 / 15.8 / 168.7 |
| Final aligned overload 80 | 79.2 / 167.9 / 167.9 / 167.9 | 14.3 / 15.5 / 20.2 / 175.9 |

Aligned input causes simultaneous prefill demand and higher TTFT, while subsequent decode remains within its target. These host timing tails include Windows scheduling, Python IPC and WebSocket work. They do not establish any actual GPU throughput or latency.

`final-suite.json` contains all nine CLI suite scenarios: low, 50%, 80%, 95%, overload, aligned, jitter, churn and interruption. The command was `suite --url ws://127.0.0.1:18080/v1 --session-budget 16 --sessions 8 --turns 2 --utterance-ms 250 --think-ms 0 --report .../final-suite.json`. All cases had zero unexpected client failures and zero token-gap/rolling-quality violations. Its overload case opened 20 conversations and rejected four turns; the separately offered 80-session test exercised resident-session rejection as well.

## Deliberate slowdown and archival audit

A separate one-worker fixture used ten ordinary response tokens, 20 ms prefill and **350 ms decode**. `final-slowdecode-350ms.json` opened one conversation, completed its first turn with 11 tokens, and rejected its next turn after measured costs became known. It reported **ten gaps above 250 ms and one session below the rolling objective**, with token-gap p50/p95/p99/max 352.0/352.3/352.3/352.3 ms. This is an intentionally unhealthy workload; the completed EOS response was not misclassified as a backend failure. Its gateway recorded ten deadline misses, zero backend failures and one saved archive.

The final aligned-plus-overload run used the gateway's default archive queue capacity **eight**. `verified-gateway-report.json` recorded 48 admitted conversations, **48 saved archives**, **21 archive backpressure events**, zero archive rejections/disk failures, zero connection failures, zero backend failures and zero live sessions after shutdown. `verified-archive-audit.json` verified all 32 generated token sequences against the fixture's exact token IDs, text and indices, including terminal EOS, and their original silence PCM bytes: **6,176 tokens** and 256,000 audio bytes matched the accepted client/runtime total. No partial files remained. Runtime mean decode batch size was eight (50% fill). Worker utilization was about 21.5% over the entire 27.4-second server lifetime, including setup and idle time; it is not utilization measured only under load.

## Failures found and corrected

1. An idle Python backend connection originally inherited the 60-second frame-read deadline and disconnected an unused worker. `healthy-16.json` and `worker-1.stderr.log` preserve the failed first attempt. Python `8f63afb` waits indefinitely for the first header byte, then bounds the entire partially received frame; restarted workers passed every subsequent scenario.
2. An 80 ms test-only idle timeout flaked when run alongside Python CPU tests: scheduler delay exceeded its tiny margin before the first token. The regression now uses 500 ms idle timeout and 150 ms fixture steps, keeping a 750 ms generation alive through output progress and subsequently closing an actually idle session. This changed test margins, not the gateway's 120-second production default.
3. The first broad archive audit exposed **16 lost records**: 197 admitted sessions but only 181 saved, with 16 full-queue rejections. `final-gateway-report.json` preserves that evidence. `392deae` replaces lossy teardown enqueue with bounded asynchronous backpressure, retaining each connection permit until its record is queued. The deterministic full-mailbox test and final 48/48 actual CLI audit verify retention. File errors remain explicit failures rather than claimed successful saves.
4. Expected rejected Open clients previously dropped their sockets abruptly, producing 48 server-side connection failures during overload. Graceful rejection close removed those failures. The first stricter client handshake then exposed three close-time TCP resets in otherwise completed conversations; `interim-close-race-*.json` retains them. `e010a42` consumes the peer Close frame after application Closed, with one bounded timeout. Final aligned/overload reports contain zero client/server failures; dedicated tests cover an unacknowledged close and an already peer-initiated close.
5. The original ignored test killed its `uv run` launcher, leaving Windows Python child processes from five earlier runs. Only those identified test-owned processes were cleaned. The harness now resolves the environment's actual interpreter and package paths, spawns that interpreter directly with `kill_on_drop` for failed tests, and waits for the fixture's bounded natural exit after gateway disconnect. The fixture lifecycle flag is test-only and defaults to persistent operation. The final explicit cross-language run exited successfully with 104 tokens and left no fixture child process.

The gateway was stopped with Ctrl+C, allowing normal session/cache cleanup, archive draining and final JSON reporting. The surrounding PowerShell PTY reported exit code one for the interrupt itself; the emitted gateway reports confirm graceful runtime completion. Only owned fixture processes were stopped. The final process inventory contained no fixture process; ports 18080, 18081, 19100, 19101 and 19102 had no remaining listener.

## Remaining hardware work

Real model outputs, multi-turn semantic quality, Whisper/projector integration, hybrid Qwen cache parity, ragged batch equivalence, CUDA timing accuracy, long-context VRAM limits and sustainable concurrent four-token/s service remain hardware validation tasks. The local fixtures validate protocol, ordering, bounded ownership, metrics, admission, interruption and logging; they cannot predict trained-model performance.
