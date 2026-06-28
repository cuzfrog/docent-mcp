# Context Rules

**IMPORTANT** - you must follow the contents of this context document, if not possible, raise to the user to decide.

## Architecture
```
  files ──▼── index ──▶  MCP server  ◀──── query
                 (cache)        (HTTP)
```

## Conventions

- **Error handling:** Use `anyhow::Result` internally. At binary boundaries (CLI, MCP responses), convert to user-facing messages. No `.unwrap()` on fallible operations.
- **No panics in library code.** Reserve `panic` for unreachable states only.
- **Logging:** There is not dedicated logging framework in use. Logging simply means print messages to stdout/stderr by the UI abstraction in `src/support/ui.rs`. Do not use `eprintln!` for CLI user-facing messages, except for error and warning. The MCP server uses HTTP.
- **Tests:** Each module has unit tests in a `#[cfg(test)] mod tests` block. Integration tests are under `src/tests/`. E2E tests are in `e2e-tests/`. E2E tests assume the binary is built and available. No `#[ignore]` tests, test must be runnable and provide coverage value. `mockall` is used for mocking dependencies in unit tests. For mocks, see below section `Test Mocking`.
- **Naming:** Snake_case for files and functions. Types are PascalCase. Constants are UPPER_SNAKE_CASE. Variable naming should be specific to carry their function. E.g. `token_counter` should not be `counter`, which can be confusing.
- **No unsafe code.** No `unsafe` blocks unless absolutely required by FFI (fastembed/ort handle this internally).
- **No Dead Code** No `allow(dead_code)`. It should only be used during long incremental refactors, and must be removed once possible.
- **File ordering** Public types, contract, methods, higher-level abstractions should be at the top of the files, private implementation details should be at the bottom. If a private function only is used in the same file, it should be below its callers. See below section `Single file layout`.
- **No inline imports** Import at the file top with `use`. Avoid qualified path in the code body, like `crate::app::index::xxxx::bbbb::new`. Import from `super` when possible. To access sibling modules, do not re-export in the `mod.rs`.
- **Config passing** Try to give a function what it needs, but do not split `Config` into multiple parameters.
- **Forbidden Warning Suppression** No `#[allow(clippy::*)]` or similar workaround. An issue must be addressed.
- **No comments** Do not add comments except it's a consequential information and the code itself cannot tell.
- **Full Code identifiers** (variables, parameters, class fields, function names) must be full words, no abbreviations beyond common ones (`id`, `url`, `db`, `ts`, `ctx`).
- **No "new" constructors** Do not create `new` constructors in a concrete struct. Use a standalone constructor method, i.e. the module constructor that creates an impl of this trait. This avoids exposing the concrete struct. The constructor method should return `impl Trait` when possible, avoid `Box<dyn Trait>`. The naming pattern is `create_X`, e.g., `pub fn create_model_factory() -> impl ModelFactory`. A constructor method should only be called by another constructor method, an implementation should not see the constructor method so that it can be tested with a mock.
- **Use fixed dependency versions** Avoid `*` or `^` to prevent unintentional updates. `=` should be explicitly used. This applies to all dependencies, including python and javascript.
- **Clean mod.rs** The file should not contain anything except module definition and re-export.

### Testing
- When writing mocks, refer to @doc/AGENTS_MOCKING.md.
- See @doc/Development.md for e2e test.

### Single file layout (ordered from top to bottom)
1. imports
2. domain types
3. 1 pub trait
4. constructor method
5. concrete implementation (struct)
6. file private functions

### Git 
- When involving git operations, refer to @doc/AGENTS_GIT.md.
- do not use worktree or submodule, work in sequence.

## Coding Principles
- Follow development principles, such as separation of concerns, SOLID principles, correct abstraction levels (e.g. reflected by the type hierarchy, type and file layout, code reusability, etc), loose coupled code. The goal is simplicity and maintainability.
- Given a change, do not first attempt to insert into current code base. First look at it from a higher perspective, discover refactor opportunities to avoid violating context rules.
- Favor trait-based design over procedural design.
- Naming must reflect the abstraction level. If a newly introduced function violates this, considering renaming related types/functions/variables to maintain correct abstraction levels.
- Avoid "helper" functions, they are where code is coupled out of class hierarchy. "helper" functions are functions that are outside the abstraction hierarchy, containing domain logic, serving the only purpose of code reuse. They are different from "utility/support" functions that are purely technical without complex domain logic. Utility functions do not have a position in the abstraction hierarchy.
- A function's parameters should be data it consumes, parameters should not be its dependencies. A high-order function should only be used for transformation instead of procedural processing. Context and config types are exempted from this rule.
- A responsibility should belong to an earlier performer. E.g. if type `Config` can parse the configuration into ready-to-use types, it shouldn't pass raw strings to its clients. A producer should produce the best output for its consumers.
- A module should be easily testable with mocked dependencies. Unit tests should be done with mocks without creating actual dependency or causing any side effects.
- Logic should be put in pure functions as much as possible. Any side effects, e.g. IO, should be at the edge layers with minimal logic. This makes the code easier to test.

### Module visibility
Minimal visibility or public surface of a type or a module. This ensures loose coupling and separation of concerns. If this is violated, e.g. a type or a module exposes multiple pub functions, it usually means the design is wrong.
- A module should only have 1 trait and its constructor method that are public. All other implementations should not be exposed.
- For a single file module, all other things in the file should be file private.
- For multi-file module, since each file is its own module, all other things must be file private or `pub(super)`
- Unit tests should be collocated with its prod code.
- Integration tests outside the module should only test the exposed `pub trait` or `pub(crate) trait`.
- In each module, search `MODULE.md` for its api, responsibilities, and files layout. You must follow its specifications. You cannot change the visibility. You should not modify this file. You cannot add any other public types/functions.
- all `mod` in `mod.rs` must be private. Any exposed types must use explicit re-export.
- Cross boundary domain types, config types, DTOs are exempted from the visibility rule.

`MODULE.md` frontmatter format:
```md
---
sealed: [mod.rs]
---
```
or
```md
---
visible:
  - path: Type1
    modifier: pub(super)
---
```

### SOLID principles:
- **Single Responsibility Principle**: A function, class, or module should have one, and only one, reason to change.
- **Open/Closed Principle**: Hide implementations behind interfaces. So that modifications happen without the client code needing to know.
- **Liskov Substitution Principle**: Switching implementation should not violate the interface's contract, including implicit ones like side effects and error handling.
- **Interface Segregation Principle**: A client should not be forced to depend on interfaces it does not use.
- **Dependency Inversion Principle**: High-level modules should not depend on low-level modules. Abstractions should not depend on detailed implementations.

## File Edit Checklist
Pre-action:
- Before adding utility functions/logic, check `src/support/` for reuse.
- Before adding a mock of `xxx`, check `xxx_mock.rs` for reuse.

Post-action:
- After file edit (semantic or logic change), run: `cargo test` and `cargo clippy --all-targets`
- After Web UI change (`src/ui/`), run: `cd src/ui && npm test`
- After adding new features, run e2e tests by: `cargo run -- serve` in background, then `pytest -v`
- After MCP schema changes, update Web UI accordingly.
