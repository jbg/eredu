include!("tests/support.rs");
include!("tests/lifecycle.rs");
include!("tests/prompt_cache.rs");
include!("tests/failures.rs");
include!("tests/persistence.rs");
include!("tests/transfer.rs");
include!("tests/storage.rs");

#[path = "tests/borrowed_storage.rs"]
mod borrowed_storage;
