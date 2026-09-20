//! Policy boundary for source discovery and catalog construction.
use super::SafetensorsHeaderAdmission;
use std::{any::Any, error::Error, fmt, sync::Arc};

/// Encoded index extent inspected on its retained file handle.
#[derive(Debug, Clone, Copy)]
pub struct SafetensorsIndexRequest {
    /// Exact encoded input buffer length, excluding a stack-owned EOF probe.
    pub encoded_bytes: usize,
    /// Root and index path bytes contributing to decoded metadata estimates.
    pub path_bytes: usize,
}

/// Source construction policy installed before discovery begins.
///
/// The caller reserves initial path/control storage before constructing this
/// policy. Discovery extends it before reading an index and before constructing
/// the store catalog. Shared catalogs, files and diagnostics retain the policy.
/// The owner completes construction after opening returns and keeps accepted
/// custody with either the source or its returned failure.
pub trait SafetensorsSourceAdmission: Any + fmt::Debug + Send + Sync {
    /// Reserve the measured index buffer and estimated decoded metadata.
    fn reserve_index(
        &self,
        request: SafetensorsIndexRequest,
    ) -> Result<(), Arc<dyn Error + Send + Sync>>;
    /// Construct header admission once the distinct shard count is known.
    /// This precedes preparing any header and does not decode the index again.
    fn headers(
        &self,
        count: usize,
    ) -> Result<Arc<dyn SafetensorsHeaderAdmission>, Arc<dyn Error + Send + Sync>>;
    /// Reserve store map/cache/diagnostic metadata from borrowed names and paths.
    /// This input size is for an estimate, not the map's allocated capacity.
    fn reserve_store(
        &self,
        metadata_input_bytes: usize,
    ) -> Result<(), Arc<dyn Error + Send + Sync>>;
}

#[cfg(test)]
mod tests;
