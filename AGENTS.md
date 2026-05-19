# Project Context

**IMPORTANT** - you must follow the contents of this context document, if not possible, raise to the user to decide.

## Architecture
```
  files/git ──▼── index ──▶  MCP server  ◀──── query
                 (cache)        (HTTP)
```

## Implementation Checklist

- Every development cycle must be verified by:
1. `cargo test`
2. `cargo clippy --all-targets`

- After Web UI change (`src/ui/`):
1. `cd src/ui`
2. `npm test`

- After major changes, run e2e tests by:
1. `cargo run -- serve` in background
2. `pytest -v`

- When MCP schema changes, update Web UI accordingly.
- When adding utility functions/logic, check `src/support/` first.

## Conventions

- **Error handling:** Use `anyhow::Result` internally. At binary boundaries (CLI, MCP responses), convert to user-facing messages. No `.unwrap()` on fallible operations.
- **No panics in library code.** Reserve `panic` for unreachable states only.
- **Logging:** There is not dedicated logging framework in use. Logging simply means print messages to stdout/stderr by the UI abstraction in `src/support/ui.rs`. Do not use `eprintln!` for CLI user-facing messages, except for error and warning. The MCP server uses HTTP.
- **Tests:** Each module has unit tests in a `#[cfg(test)] mod tests` block. Integration tests are under `src/tests/`. E2E tests are in `e2e-tests/`. E2E tests assume the binary is built and available. No `#[ignore]` tests, test must be runnable and provide coverage value. `mockall` is used for mocking dependencies in unit tests. For mocks, see below section `Test Mocking`.
- **Naming:** Snake_case for files and functions. Types are PascalCase. Constants are UPPER_SNAKE_CASE. Variable naming should be specific to carry their function. E.g. `token_counter` should not be `counter`, which can be confusing.
- **No unsafe code.** No `unsafe` blocks unless absolutely required by FFI (fastembed/ort handle this internally).
- **No Dead Code** No `allow(dead_code)`. It should only be used during long incremental refactors, and must be removed once possible.
- **Module Interface at Top** Public types, contract, methods should be at the top of the files, private implementation details should be at the bottom. If a private function only is used in the same file, it should be below its callers. See below section `Single file layout`.
- **Use imports** Import at the file top. Avoid long module path in the code body. E.g. `crate::app::index::xxxx::bbbb::new`
- **Config passing** Try to give a function what it needs, but do not split `Config` into multiple parameters.
- **Forbidden Warning Suppression** No `#[allow(clippy::*)]` or similar workaround. An issue must be addressed.
- **No comments** Do not add comments except it's a consequential information and the code itself cannot tell.
- **No "new" constructors** Do not create `new` constructors in a concrete struct. Use a standalone factory method, i.e. the module constructor that creates an impl of this trait. This avoids exposing the concrete struct. The factory method should return `impl Trait` when possible, avoid `Box<dyn Trait>`. The naming pattern is `create_X`, e.g., `pub fn create_model_factory() -> impl ModelFactory`.
- **Use fixed dependency versions** Avoid `*` or `^` to prevent unintentional updates. `=` should be explicitly used. This applies to all dependencies, including python and javascript.
- **Clean mod.rs** The file should not contain anything except module definition and re-export.

### Test Mocking
* `pub fn mock_xxxx()` to create a shared mock for testing.
* `struct MockXxxxx` implements the trait, this is achieved by `mockall` crate.
* Do not test mock itself.
* Mocks should not violate visibility rules. A trait only used in its parent module should not have its mock exposed outside its parent module. A mock with manually implemented logic should be placed in a companion file, such as `abc_mock.rs` with test scope along with its counterpart file `abc.rs`.
* Use `#[cfg(test)]` to re-export a mock when needed.

Manual implementeation of a mock's behavior should be avoided as possible. Try to use mocks directly in unit tests with expected calls.

### Single file layout (ordered from top to bottom)
1. imports
2. domain types
3. 1 pub trait
4. factory method
5. concrete implementation (struct)
6. file private functions

### Git 
When involving git operations, refer to @doc/AGENTS_GIT.md.

## Coding Principles
- Follow development principles, such as separation of concerns, SOLID principles, correct abstraction levels (e.g. reflected by the type hierarchy, type and file layout, code reusability, etc), loose coupled code. The goal is simplicity and maintainability.
- Given a change, do not first attempt to insert into current code base. First look at it from a higher perspective, discover refactor opportunities and maintain small file sizes.
- Favor trait-based design over procedural design.
- Naming must reflect the abstraction level. If a newly introduced function violates this, considering renaming related types/functions/variables to maintain correct abstraction levels.
- Avoid "helper" functions, they are where code is coupled out of class hierarchy. "helper" functions are functions that are outside the abstraction hierarchy, containing domain logic, serving the only purpose of code reuse. They are different from "utility/support" functions that are purely technical without complex domain logic. Utility functions do not have a position in the abstraction hierarchy.
- A function's parameters should be data it consumes, parameters should not be its dependencies. A high-order function should only be used for transformation instead of procedural processing. Context and config types are exempted from this rule.
- A responsibility should belong to an earlier performer. E.g. if type `Config` can parse the configuration into ready-to-use types, it shouldn't pass raw strings to its clients. A producer should produce the best output for its consumers.
- A module should be easily testable with mocked dependencies. Unit tests should be done with mocks without creating actual dependency or causing any side effects.
- Logic should be put in pure functions as much as possible. Any side effects, e.g. IO, should be at the edge layers with minimal logic. This makes the code easier to test.

### Module visibility
Minimal visibility or public surface of a type or a module. This ensures loose coupling and separation of concerns. If this is violated, e.g. a type or a module exposes multiple pub functions, it usually means the design is wrong.
- A module should only have 1 trait and its factory method that are public. All other implementations should not be exposed.
- For a single file module, all other things in the file should be file private.
- For multi-file module, since each file is its own module, all other things must be file private or `pub(super)`
- Unit tests should be collocated with its prod code.
- Integration tests outside the module should only test the exposed `pub trait` or `pub(crate) trait`.
- In each module, search `MODULE.md` for its api, responsibilities, and files layout. Only types/functions with explicit `pub` should be exposed. You must follow its specifications. You cannot change the visibility. You should not modify this file. You cannot add any other public types/functions.
- all `mod` in `mod.rs` must be private. Any exposed types must use explicit re-export.
- Cross boundary domain types, config types, DTOs are exempted from the visibility rule.

`MODULE.md` format (inside parenthesis is comments):
```
# Module - MODULE_NAME
(implicitly for mod.rs)

- <any public export> (if there are entries, only mentioned exports are allowed)

## FILE.rs or DIR/ (a child module)
- <any public export> (only mentioned exports are allowed)
```

### SOLID principles:
- **Single Responsibility Principle**: A function, class, or module should have one, and only one, reason to change.
- **Open/Closed Principle**: Hide implementations behind interfaces. So that modifications happen without the client code needing to know.
- **Liskov Substitution Principle**: Switching implementation should not violate the interface's contract, including implicit ones like side effects and error handling.
- **Interface Segregation Principle**: A client should not be forced to depend on interfaces it does not use.
- **Dependency Inversion Principle**: High-level modules should not depend on low-level modules. Abstractions should not depend on detailed implementations.