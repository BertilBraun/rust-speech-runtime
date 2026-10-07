# Rust code style and review guide

The core gateway and workers should be understandable by reading their entry points, then following named operations into focused modules. Formatting is automated; control-flow design and useful documentation require review.

## Formatting

Use stable `rustfmt` with the repository's [rustfmt.toml](../rustfmt.toml): Rust 2024 style, 100-column target, Unix newlines and standard shorthand. Format the entire workspace, including tests and examples:

```powershell
cargo fmt --all
cargo fmt --all -- --check
```

Leave one blank line between functions, methods, type declarations and implementation blocks. Within a function, separate meaningful stages such as validation, mutation and dispatch. Stable rustfmt preserves these item separators but does not insert them automatically. Keep macro branches readable manually: formatting a `select!` is not a substitute for simplifying its body.

## Functions and modules

- Give a function one responsibility and one level of abstraction. A public entry point should reveal the sequence of operations without embedding every validation rule or I/O detail.
- Prefer roughly 20–40 lines for ordinary function bodies. This is a review cue, not a line-count requirement: a flat exhaustive match can reasonably be longer. Extract meaningful operations rather than wrappers that only forward arguments.
- Keep nested control flow shallow. Prefer early returns, `?` and `let ... else` for rejected inputs and ended streams. Three nested control-flow blocks deserve review; do not bury state changes inside six or more levels.
- Split modules by ownership or responsibility. Keep closely related operations together. Move substantial test fixtures out of production modules, and put small inline test modules last.
- Use descriptive domain names. Avoid compressed identifiers and clever expressions when an ordinary match or named operation communicates the intent more directly.
- Name counts, durations and identities explicitly: `reserved_turn_count`, `decode_round_ms`, `session_key`, `generation_counter` and `in_flight_batch`. A reader should not have to inspect a type or follow a method to discover what `active`, `clock`, `previous` or `load` means.
- Separate decision inputs from decisions. Calculate named costs and capacity facts first, then write the acceptance condition. Avoid branching inside iterator closures when an ordinary loop or a focused predicate is easier to follow.
- Share canonical protocol and configuration types. Clone only where independent owners need the same data; explain non-obvious ownership requirements.

## Async and state ownership

One worker actor owns its sessions, preparation state and scheduling decisions. Its `select!` coordinates commands, completions, cancellation and cleanup; it delegates substantial work to named functions. Backend execution owns the TCP connection and never borrows actor scheduling state across an await.

Keep hot-path channels bounded. Document whether saturation rejects, waits or ends a session. Distinguish enqueueing, worker validation, model completion and network delivery. Preserve cancellation and acceptance ordering during refactors; a shorter expression is not an improvement if it drops accepted events or bypasses generation fencing.

The worker control loop is a coordination boundary: command receiver, completion receiver, cleanup timer and cancellation. Session lifecycle work lives in `worker/lifecycle.rs`, and backend failure transitions in `worker/failure.rs`. Keep computation out of those event branches. `BatchJob` owns the backend request/audio body; `BatchCompletion` returns proposals for the actor to accept. Neither task borrows the other's mutable state.

Explain non-obvious fairness, biased selection, cache acknowledgement and shutdown ordering with a short reason comment. Keep GPU operations and blocking archive writes outside the Tokio control loops.

## Boundary documentation

Public types and methods need Rustdoc that explains their purpose and the caller's obligations. Include relevant units, packet ordering, ownership, acknowledgement semantics, cancellation, idempotence and backpressure. Document protocol variants when their consequences differ, especially private preparation versus visible output. Show usage with compiling examples where helpful.

Use ordinary comments to explain a reason or invariant, not to restate a line of code. Names and short functions explain the steps; documentation explains the contract. Browse the generated interface documentation with:

```powershell
cargo doc --locked --no-deps --open
```

## Where to inspect the implementation

| Responsibility | Entry point |
| --- | --- |
| Node startup, rollback and shutdown | [runtime/node.rs](../src/runtime/node.rs) |
| Admission API and session command contracts | [runtime/ingress.rs](../src/runtime/ingress.rs), [runtime/session.rs](../src/runtime/session.rs) |
| Sticky placement and pending opens | [session/manager.rs](../src/session/manager.rs) |
| Worker control loop and isolated execution | [worker/actor.rs](../src/worker/actor.rs), [worker/execution.rs](../src/worker/execution.rs) |
| Compute reservation and capture transitions | [worker/admission.rs](../src/worker/admission.rs), [worker/turn.rs](../src/worker/turn.rs) |
| Batch packing and output acceptance | [worker/batch.rs](../src/worker/batch.rs), [worker/output.rs](../src/worker/output.rs) |
| WebSocket handshake, messages and cleanup | [transport/connection/](../src/transport/connection/) |
| Public events and Rust/Python wire contract | [protocol/mod.rs](../src/protocol/mod.rs), [protocol/backend.rs](../src/protocol/backend.rs) |

The [visual architecture guide](ARCHITECTURE.md) shows data flow, cache ownership and computation. This guide explains how the corresponding code should be written.

## Required validation

```powershell
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
# Requires uv and the installed backend Python environment:
cargo test --locked --test transport python_worker_process_to_websocket_client_full_pipeline -- --ignored
```

Review the diff for behavior changes even when tests pass. Readability refactors must preserve admission, stale-result handling, event order and archive cleanup. Formatting and Clippy cannot establish good architecture on their own.
