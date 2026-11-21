//! Attention-cache storage and residency.

pub(crate) mod kv;
mod native;
pub use native::MlxCacheBackend;
