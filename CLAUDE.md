# CLAUDE.md

## Build & Run & Dev Setup

Every development cycle must be verified by:
1. cargo test
2. cargo clippy

After Web UI change (`src/ui/`):
1. `cd src/ui`
2. `npm test`

After major changes, run e2e tests by:
1. `cargo run -- serve` in background
2. `pytest -v`

### Implementation Checklist
- When MCP schema changes, update Web UI accordingly.

## Dependencies

- Use fixed versions. Avoid `*` or `^` to prevent unintentional updates. This applies to all dependencies, including python and javascript.

## Conventions

- **Error handling:** Use `anyhow::Result` internally. At binary boundaries (CLI, MCP responses), convert to user-facing messages. No `.unwrap()` on fallible operations.
- **No panics in library code.** Reserve `panic` for unreachable states only.
- **Logging:** Use UI abstraction in `src/support/ui.rs`. Do not use `eprintln!` for CLI user-facing messages, except for error. The MCP server uses HTTP.
- **Tests:** Each module has unit tests in a `#[cfg(test)] mod tests` block. Integration-style tests are under `src/tests/` (compiled as crate unit tests, avoiding separate integration-test link overhead). E2E tests are in `e2e-tests/`. E2E tests assume the binary is built and available. No `#[ignore]` tests, test must be runable and provide coverage value.
- **Naming:** Snake_case for files and functions. Types are PascalCase. Constants are UPPER_SNAKE_CASE. Variable naming should be specific to carry their function. E.g. `token_counter` should not be `counter`, which can be confusing.
- **No unsafe code.** No `unsafe` blocks unless absolutely required by FFI (fastembed/ort handle this internally).
- **No Dead Code** No `allow(dead_code)`. Remove unused code immediately to maintain codebase health.
- **Module Interface at Top** Public types, contract, methods should be at the top of the files, private implementation details should be at the bottom. If a private function only is used in the same file, it should be below its callers.
- **Favor Object Oriented Design** Favor trait-based design over procedural design.
- **Use imports** Import at the file top. Avoid long module path in the code body. E.g. `crate::app::index::xxxx::bbbb::new`
- **Config passing** Do not split `Config` into multiple parameters for a function that consumes it.
- **Forbidden Warning Suppression** No `#[allow(clippy::*)]` or similar workaround. An issue must be addressed.

### Single file layout (from top to bottom)
1. imports
2. types
3. trait
4. factory method (Do not create `new` constructor in a concrete struct)
5. concrete implementation (struct)
6. file private functions

(Principle: public at top, private at bottom)

## Git 
### branching
- Main branch: `main`
- Feature branches: `<task_id>_<short-description>`, e.g., `IMPL-2_config-loader`
- User branches: `dev_*`, `fix_`.

### PR title - semantic-pull-request format:
<type>([optional task_id]): <description>
```yaml
types:
  - feat
  - fix
  - docs
  - style
  - refactor
  - perf
  - test
  - chore
  - ci
```

### PR Squash commit message
- Do not contain any individual commits. Use the github template.

## **Crucial** Coding Principles
- You are a coding architect. Look the code from a mid/high perspective, follow development principles, such as separation of concerns, SOLID principles, correct abstraction levels (e.g. reflected by their hierarchy, type and file layout, code reusability, etc), loose coupled code. The goal is simplicity and maintainability.
- MINIMAL visibility or public surface of a type or a module. This ensures loose coupling and separation of concerns. If this is violated, e.g. a type or a module exposes multiple pub functions, it usually means the design is wrong.
- Given a change, do not first attempt to insert into current code base. First look at it from a higher perspective, discover refactor opportunities and maintain small file sizes. If a file's prod code is more than 200 lines, consider splitting it. If a function is more than 50 lines, consider splitting it.
- Naming must reflect the abstraction level. If a newly introduced function violates this, considering renaming related types/functions/variables to maintain correct abstraction levels.
- Avoid "helper" functions, they are where code is coupled out of class hierarchy.
- A function's parameters should be data it consumes, parameters should not be its dependencies. A high-order function should only be used for transformation instead of procedural processing.
- A responsibility should belong to an earlier performer. E.g. if type `Config` can parse the configuration into ready-to-use types, it shouldn't pass raw strings to its clients. A producer should produce the best output for its consumers.


### Module visibility
- A module should only has 1 trait and its factory method that are public. All other implementations should not be exposed.
- For a single file module, all other things in the file should be file private.
- For multi-file module, since each file is its own module, all other things must be file private or `pub(super)`
- Unit tests should be collocated with its prod code.
- Integration tests outside the module should only test the exposed trait.

### SOLID principles:
- **Single Responsibility Principle**: A function, class, or module should have one, and only one, reason to change.
- **Open/Closed Principle**: Hide implementations behind interfaces. So that modifications happen without the client code needing to know.
- **Liskov Substitution Principle**: Switching implementation should not violate the interface's contract, including implicit ones like side effects and error handling.
- **Interface Segregation Principle**: A client should not be forced to depend on interfaces it does not use.
- **Dependency Inversion Principle**: High-level modules should not depend on low-level modules. Abstractions should not depend on detailed implementations.