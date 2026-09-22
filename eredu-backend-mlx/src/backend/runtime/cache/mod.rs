//! Attention-cache storage and residency.

pub(crate) mod kv;
/// Block-addressable attention-cache residency and persistence.
pub mod residency;
/// Runtime-policy-selected key/value state realization.
pub mod state;

mod original_copy;
pub(crate) use original_copy::{
    copy_completed_compressed, copy_completed_pooling, copy_original_compressed,
    copy_original_pooling, PreparedPredictionCacheCopy,
};

mod value_completion;
pub(crate) use value_completion::{
    complete_and_borrow, complete_values, completed_borrow_control_bytes,
    control_bytes as value_completion_control_bytes,
};
