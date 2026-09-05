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
use transitions::{eviction_candidate, HostDemotionProgress, PendingCacheOperation};

#[path = "manager/lifecycle.rs"]
mod lifecycle;

#[path = "manager/reporting.rs"]
mod reporting;
pub(super) use reporting::update_report_totals;

#[path = "manager/acquisition.rs"]
mod acquisition;
pub use acquisition::{CacheBlockLease, CacheBlockPrefetch};

#[path = "manager/prompt_cache.rs"]
mod prompt_cache;
pub use prompt_cache::{
    load_prompt_cache_state_tensors, open_prompt_cache, LoadedPromptCacheStateTensor,
    PromptCacheStateArray,
};

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
