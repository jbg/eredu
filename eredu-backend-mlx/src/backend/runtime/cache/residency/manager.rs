//! Cache lifecycle, accounting, eviction, and residency orchestration.

use super::*;

#[path = "manager/state.rs"]
mod state;
use state::CacheResidencyManagerInner;
pub(super) use state::{
    CacheBlockRecord, CacheManagerState, MlxCacheBlockStorage, MlxCacheIoOperation,
};

/// Shared architecture-independent manager enforcing budgets across all layers.
///
/// Clones retain one compact shared inner allocation containing the catalog,
/// workers, options, and process-pool membership.
#[derive(Debug, Clone)]
pub struct CacheResidencyManager {
    session_id: u64,
    inner: Arc<CacheResidencyManagerInner>,
}

#[path = "manager/transitions.rs"]
mod transitions;
use transitions::{HostDemotionProgress, PendingCacheOperation, eviction_candidate};

#[path = "manager/history.rs"]
mod history;
pub(crate) use history::{CacheHistoryOwner, PreparedCacheHistory};

#[path = "manager/lifecycle.rs"]
mod lifecycle;

#[path = "manager/transfer_stream.rs"]
mod transfer_stream;
pub(crate) use transfer_stream::{CacheTransferStreamError, PreparedCacheTransferStream};

#[path = "manager/reporting.rs"]
mod reporting;
#[path = "manager/source.rs"]
mod source;
#[path = "manager/storage.rs"]
mod storage;
pub(super) use reporting::update_report_totals;
pub(crate) use source::{
    CacheBlockSource, CacheBlockSourceLoan, CacheDiskSource, CacheSourceError, CacheSourceFailure,
    CacheSourceFailureCause, IndependentCacheManagerPlan, PinnedCacheBlock, PinnedCacheBlockLease,
    PinnedCacheSource, PreparedIndependentCacheManager,
};

#[path = "manager/acquisition.rs"]
mod acquisition;
pub use acquisition::{CacheBlockLease, CacheBlockPrefetch};

#[path = "manager/prompt_cache.rs"]
mod prompt_cache;
pub(crate) use prompt_cache::PromptCacheTail;
pub use prompt_cache::{LoadedPromptCacheStateTensor, PromptCacheStateArray};
pub(crate) use prompt_cache::{PromptCacheMaterialization, load_prompt_cache_state_tensors_funded};

#[path = "manager/persistence.rs"]
mod persistence;
pub(super) use persistence::*;

fn sha256_hex(digest: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let digest = digest.as_ref();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for &byte in digest {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

#[cfg(test)]
fn cpu_stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn sync_file(path: &Path) -> Result<(), CacheResidencyError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| CacheResidencyError::Io {
            action: "synchronize cache file",
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

pub(crate) use source::{CatalogInstallFailure, InstalledManagerCatalog, PreparedManagerCatalog};

#[path = "manager/publication.rs"]
mod publication;
pub(crate) use publication::{CacheBlockMetadata, PreparedFloatingBlockMetadata};

#[path = "manager/original_append.rs"]
mod original_append;

#[path = "manager/original_discard.rs"]
mod original_discard;
pub(crate) use original_discard::PreparedCacheDiscard;
#[path = "manager/original_host.rs"]
mod original_host;
pub(crate) use original_host::PreparedCacheTransferSource;
#[path = "manager/original_scan.rs"]
mod original_scan;

pub(crate) use source::{PagedArrayCopyLayout, PreparedPagedArrayCopy};

pub(crate) use source::{
    PreparedCacheHostPromotion, PreparedCacheHostPromotionSlots, PreparedHostPromotion,
    PreparedHostReturn, PreparedOrdinaryCacheHostPromotion,
};

pub(crate) use source::{
    PreparedCacheHostDemotion, PreparedHostEviction, PreparedOrdinaryCacheHostDemotion,
    StoredCacheHostSource,
};

pub(crate) use source::{
    DiskWriteOccupancy, DiskWriteOperation, DiskWriteOperationFailure, PreparedDiskWrite,
    PreparedDiskWriteOutput,
};

pub(crate) use source::{
    CompletedDiskRead, DiskReadFinishFailure, DiskReadOccupancy, DiskReadOperation,
    DiskReadOperationFailure, PreparedDiskRead, PreparedDiskReadOutput, PreparedDiskReadSource,
};

pub(crate) use source::{OrdinaryWrittenCacheHostSource, PreparedDiskWriteHostRetirement};
pub(crate) use source::{PreparedCacheDiskWriteSource, PreparedDiskWriteDestination};

pub(crate) use source::{InstalledDiskWorker, PreparedDiskWorker};

pub(crate) use source::{DiskReadBinding, PreparedDiskReadDestination, disk_read_source_facts};

pub(crate) use source::PreparedInitialDiskReturn;

#[path = "manager/realtime_transaction.rs"]
mod realtime_transaction;

#[path = "manager/prepared_append.rs"]
mod prepared_append;

pub(crate) use source::{OrdinaryDiskReadSource, OrdinaryReadCacheHostSource};

#[cfg(test)]
pub use prompt_cache::{load_prompt_cache_state_tensors, open_prompt_cache};
