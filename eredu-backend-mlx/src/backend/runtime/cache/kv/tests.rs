include!("tests/support.rs");
include!("tests/state_forms.rs");
include!("tests/blockwise_attention.rs");
include!("tests/paging.rs");
include!("tests/persistence.rs");

#[path = "tests/blockwise_policy.rs"]
mod blockwise_policy;

#[path = "tests/paged_scan.rs"]
mod paged_scan;
