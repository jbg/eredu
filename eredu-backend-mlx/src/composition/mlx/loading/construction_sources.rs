//! Native resources prepared before model materialization acquires its owner.
use crate::backend::{
    error::Error,
    runtime::{
        cache::residency::PreparedCacheTransferStream, execution::generic::PreparedLayerwiseManager,
    },
};
use eredu_runtime::{working_memory::MemoryLedger, CacheResidencyPolicy, SelectedStateRealization};
use safemlx::Stream;

/// One move-only handoff consumed by the selected typed binding route. The
/// layerwise manager is unique; cache managers share the actual transfer owner.
pub(crate) struct PreparedNativeConstructionSources {
    pub(crate) layerwise: Option<PreparedLayerwiseManager>,
    pub(crate) cache_transfer: Option<PreparedCacheTransferStream>,
}
impl PreparedNativeConstructionSources {
    pub(crate) fn prepare(
        state: &SelectedStateRealization,
        layerwise: Option<PreparedLayerwiseManager>,
        ledger: &MemoryLedger,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let cache_transfer = matches!(state.policy(), CacheResidencyPolicy::Paged(_))
            .then(|| PreparedCacheTransferStream::prepare_for_caller::<Self>(ledger, stream))
            .transpose()
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        Ok(Self {
            layerwise,
            cache_transfer,
        })
    }
}
