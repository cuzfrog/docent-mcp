# Contributing to docent-mcp

Thank you for your interest in contributing to `docent-mcp`! This document provides guidelines for contributing to the project.

## Code of Conduct

Please note that this project follows a standard Code of Conduct. By participating in this project, you agree to abide by its terms.

## Development Setup

1. Fork the repository
2. Clone your fork locally
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Make your changes
5. Follow the development cycle:
   ```bash
   cargo test
   cargo clippy
   ```
6. Commit your changes: `git commit -m 'feat: add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## Development Guidelines

### Code Style

- Follow Rust standard style
- Run `cargo fmt` before committing
- Ensure `cargo clippy` passes without warnings
- No `unsafe` code unless absolutely required
- Use snake_case for functions and variables, PascalCase for types

### Testing

- All changes must include tests
- Run `cargo test` to verify
- For UI changes, run tests in `src/ui`: `cd src/ui && npm test`
- For major changes, run e2e tests:
  ```bash
  cargo run -- serve &
  pytest -v
  ```

### Pull Request Process

1. PR title should follow semantic format: `<type>([task_id]): description`
2. Include a clear description of changes
3. Link any related issues
4. Ensure all checks pass
5. Squash commits when merging

## Types of Contributions

### Bug Fixes

- Use `fix:` prefix in commit messages
- Include test cases that reproduce the bug
- Verify the fix doesn't break existing functionality

### Features

- Use `feat:` prefix in commit messages
- Update documentation for new features
- Include appropriate tests

### Documentation

- Use `docs:` prefix for documentation updates
- Keep examples up to date
- Ensure clarity and accuracy

## Architecture Decisions

This project follows SOLID principles and favors:
- Trait-based design over procedural
- Object-oriented patterns where appropriate
- Minimal public surface area for modules
- Clear separation of concerns

## Before You Submit

- Check that your PR description clearly explains what you've done and why
- Ensure all tests pass
- Update any relevant documentation
- Check that you've followed the project's coding conventions