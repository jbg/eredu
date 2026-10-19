//! Qwen prediction snapshot instrumentation.
use super::*;
pub(super) use super::super::super::speculative_snapshot::{state, tensor};

thread_local! {
    pub(super) static COPIES: Cell<usize> = const { Cell::new(0) };
    pub(super) static FAIL_PREDICTION: Cell<bool> = const { Cell::new(false) };
}
pub(super) fn copied() {
    COPIES.with(|c| c.set(c.get() + 1));
}
