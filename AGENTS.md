# AGENTS.md

## Purpose

This file is the working guide for coding agents operating in `agent-sdk`.
It summarizes the repo's current conventions from `CLAUDE.md`, `Cargo.toml`,
`.github/workflows/ci.yml`, examples, and representative source files.

## Repo Facts

- Language: Rust
- Minimum Rust: 1.85+
- Edition: 2024
- Crate type: library crate with examples
- Optional feature: `otel`
- CI sets `RUSTFLAGS=-D warnings`, so assume warnings are failures
- No `.cursorrules` file was found
- No `.cursor/rules/` directory was found
- No `.github/copilot-instructions.md` file was found

## High-Value Commands

Use these commands from the repository root.

### Fast Validation

```bash
cargo check --all-targets
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`cargo check --all-targets` matches CI and is the fastest broad validation pass.
`cargo clippy --all-targets -- -D warnings` matches CI and should stay clean.
`cargo fmt --check` is what CI enforces.

### Before Committing

```bash
cargo fmt
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
```

`CLAUDE.md` explicitly requires `cargo fmt` before committing changes.

### Build Commands

```bash
cargo build
cargo build --examples
cargo build --features otel
```

Use `cargo build --examples` when changing example code or public APIs used there.
Use `cargo build --features otel` when changing observability code.

### Test Commands

```bash
cargo test
cargo test --lib
cargo test --features otel --test observability_integration
```

Most tests are inline unit tests inside source modules.
There is also a feature-gated integration test: `tests/observability_integration.rs`.

### Run One Unit Test

Preferred exact form for inline unit tests:

```bash
cargo test --lib primitive_tools::glob::tests::test_glob_simple_pattern -- --exact
```

This exact command was verified in this repository.

Shorter name-based filtering also works when the name is unique:

```bash
cargo test test_glob_simple_pattern -- --exact
```

### Run One Integration Test

For the `otel` integration test file, include the feature and the test target:

```bash
cargo test --features otel --test observability_integration root_span_emitted_for_simple_run -- --exact
```

### Useful Narrowing Patterns

```bash
cargo test providers::openai
cargo test web::security
cargo test primitive_tools::glob::tests
```

Use substring filters for quick local iteration, then rerun the broader suite.

## Repository Layout

- `src/lib.rs` is the crate root and public export surface
- `src/agent_loop.rs` and `src/agent_loop/` contain the core orchestration logic
- `src/providers.rs` and `src/providers/` implement model providers
- `src/primitive_tools.rs` and `src/primitive_tools/` contain file/shell tools
- `src/web.rs` and `src/web/` contain web search/fetch helpers and SSRF protection
- `src/mcp.rs` and `src/mcp/` contain Model Context Protocol support
- `src/observability.rs` and `src/observability/` contain `otel` support
- `src/skills.rs`, `src/subagent.rs`, `src/todo.rs`, and `src/user_interaction.rs`
  are major feature areas with their own inline tests

## Module Conventions

- Do not introduce `mod.rs`
- Use modern Rust module layout: `foo.rs` as the module root and `foo/` for children
- Keep module declarations close to the top of the parent file
- Re-export public API deliberately from module roots; do not re-export everything by default
- Follow the existing crate structure before adding new top-level modules

## Formatting And Imports

- Always use `cargo fmt`; do not hand-format around rustfmt
- Keep imports explicit and grouped logically
- Prefer grouped brace imports like `use anyhow::{Context, Result, bail};`
- Avoid wildcard imports in production code
- `use super::*;` is common and acceptable inside `#[cfg(test)] mod tests`
- Follow nearby import ordering instead of imposing a new style in one file
- Avoid noisy aliases unless they genuinely improve clarity

## Naming

- Types and traits: `UpperCamelCase`
- Functions, methods, modules, and fields: `snake_case`
- Constants and statics: `SCREAMING_SNAKE_CASE`
- Test names should be descriptive and read like behavior statements
- Use domain-specific names over generic placeholders like `data`, `handler`, or `manager`

## Types And API Shape

- Prefer `anyhow::Result<T>` for fallible functions and trait methods
- Use concrete structs and enums for stable payloads and public data shapes
- Use `serde_json::Value` only at JSON/tool boundaries where flexibility is required
- Use `Arc` for shared long-lived components such as providers, stores, and registries
- Add `#[must_use]` to constructors/getters/builders when ignoring the return is a bug
- Use `const fn` for trivial constructors/getters when it is natural and already supported
- Keep generic bounds explicit, especially `Send + Sync + 'static` on async/shared types

## Error Handling

- Add context at IO, parsing, network, lock, and serialization boundaries
- Prefer `anyhow::{Context, Result, bail, ensure}` patterns
- Use `bail!` for early invalid-state exits
- Use `ensure!` for precondition checks that should explain the failure clearly
- Convert poisoned `std::sync::RwLock` access with `.ok().context("lock poisoned")?`
- Do not add `unwrap()` or `expect()` in new production code
- Prefer not to add new `unwrap()` or `expect()` in tests either; return `Result<()>` and use `?`
- Some older tests still use `unwrap()` and `expect()`; do not copy that pattern into new code

## Async And Concurrency

- Match the existing style of the surrounding trait or module
- This repo uses both `#[async_trait]` and native async trait methods depending on context
- Prefer `tokio::sync` primitives for async coordination
- Use `std::sync::RwLock` where the code already uses it for in-memory stores
- Avoid holding locks across expensive work or `.await` points unless the design requires it

## Serialization And External APIs

- Derive `Serialize`/`Deserialize` explicitly where needed
- Use `#[serde(rename_all = "...")]` to match remote API casing
- Use `#[serde(untagged)]` only when the payload is truly polymorphic
- Build JSON examples and schemas with `serde_json::json!`
- Preserve wire-format compatibility when touching provider request/response types

## Logging, Time, And Observability

- The current codebase predominantly uses `log` macros such as `log::debug!` and `log::warn!`
- Match the surrounding file's logging style; do not introduce a second logging style casually
- Use the `time` crate, not `chrono`
- Keep `otel` changes feature-gated and validate them with `--features otel`

## Documentation And Comments

- Prefer self-documenting code and small helpers over explanatory comments
- Add Rustdoc to public APIs when behavior is not obvious
- For fallible public functions, include a `# Errors` section in the docs
- Add comments for non-obvious invariants, protocol quirks, or safety-related reasoning
- Do not add comments that only restate the code literally

## Testing Conventions

- Test behavior, not coverage for its own sake
- Keep most tests close to the code in `#[cfg(test)] mod tests`
- Use `#[tokio::test]` for async behavior and `#[test]` for synchronous logic
- Prefer focused fixtures and helper constructors local to the test module
- When fixing a bug, add or update a test that proves the behavior
- Do not change test expectations just to make failing tests pass
- Re-run the narrowest relevant test first, then the broader suite

## Safety And Lints

- `unsafe` is forbidden in this crate
- Do not bypass clippy with `#[allow(clippy::...)]`; fix the design instead
- Keep code warning-free under normal `cargo` and CI settings

## Security-Sensitive Areas

- Preserve SSRF protections in `src/web/security.rs`
- Fail closed on URL validation and host resolution checks
- Preserve capability checks in primitive tools before filesystem or shell access
- Be careful with path handling, environment access, and external process execution

## Change Strategy For Agents

- Prefer the smallest correct change
- Preserve existing APIs and behavior unless the task explicitly calls for a breaking change
- Follow nearby patterns before inventing new abstractions
- If a helper is only used once, keep the logic inline unless extraction clearly improves clarity
- Validate the touched surface area with the narrowest useful command, then run the standard checks
