# Mocking

* `pub fn mock_xxxx()` to create a shared mock for testing.
* `struct MockXxxxx` implements the trait, this is achieved by `mockall` crate.
* Do not test mock itself.
* Mocks should not violate visibility rules. A trait only used in its parent module should not have its mock exposed outside its parent module. A mock with manually implemented logic should be placed in a companion file, such as `abc_mock.rs` with test scope along with its counterpart file `abc.rs`.
* Use `#[cfg_attr(test, mockall::automock)]` on a trait to create mock.
* Use `#[cfg(test)]` to re-export a mock when needed.

Manual implementeation of a mock's behavior should be avoided as possible. Try to use mocks directly in unit tests with expected calls.
