//! Source-owned fresh reader/materializer/shell construction after catalog admission.
use super::*;
use crate::store::{SourceControl, SourceHandle};
use std::{
    alloc::Layout,
    any::Any,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};
use touched_paths::TouchedEntry;

/// Requests for the actual retained builder. Checkpoint/header payload already
/// exists and is not cloned; reader/file handles and future recipe entries are
/// separate from this finite constructor.
#[derive(Clone, Copy, Debug)]
pub struct GgufSourceStorageRequest {
    buffers: usize,
    controls: usize,
    source: Layout,
    custody: Layout,
    materializers: usize,
    materializer_shared: Layout,
    physical_inventory: u64,
    prepaid_inventory: u64,
}
impl GgufSourceStorageRequest {
    /// Exact fresh Vec backings and fixed constructor controls.
    pub fn requested_bytes(&self) -> usize {
        self.buffers + self.controls
    }
    /// Shared StoreInner payload, retained also by weak identities.
    pub fn source_body(&self) -> Layout {
        self.source
    }
    /// Shared opaque constructor-custody payload.
    pub fn custody_body(&self) -> Layout {
        self.custody
    }
    /// Count and payload of the source's independently shared Checkpoint owners.
    pub fn checkpoint_bodies(&self) -> (usize, Layout) {
        (self.materializers, self.materializer_shared)
    }
    /// Complete existing source inventory and its newly constructed portion.
    pub fn inventory_bytes(&self) -> (u64, u64) {
        (self.physical_inventory, self.prepaid_inventory)
    }
    /// Three preinitialized std mutexes: reader cache and both recipe tables.
    pub fn mutex_count(&self) -> usize {
        3
    }
}

#[derive(Debug)]
enum Cause {
    Store(StoreError),
    Readers(GgufReaderBuffersFailure),
    Materializer(eredu_gguf::PreparedMaterializerFailure<Option<SourceControl>>),
    Reserve(TryReserveError),
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(e) => e.fmt(f),
            Self::Readers(e) => e.fmt(f),
            Self::Materializer(e) => e.fmt(f),
            Self::Reserve(e) => e.fmt(f),
        }
    }
}
// The current source, every completed materializer and each destination remain
// in this one fixed carrier until success. Original checkpoint vector storage
// remains in builder even after its elements move; no collect reuse is assumed.
#[derive(Debug)]
struct Construction {
    builder: GgufWeightStoreBuilder,
    materializers: Vec<TensorMaterializer>,
    buffers: Option<GgufReaderBuffers>,
    last_used: Vec<u64>,
    touched_entries: Vec<TouchedEntry>,
    touched: Option<TouchedPaths>,
    header_bytes: u64,
    control: Option<SourceControl>,
}
/// Actual failed input and successful storage prefix; fixed transports carry
/// no reconstructed strings or reopened artifact. Custody retires last.
#[derive(Debug)]
pub struct PreparedGgufSourceFailure {
    cause: Cause,
    construction: Construction,
}
impl PreparedGgufSourceFailure {
    /// Number of actual completed materializers still owned by this refusal.
    pub fn completed_materializers(&self) -> usize {
        self.construction.materializers.len()
    }
    /// Exact remaining input population, including the failed current checkpoint.
    pub fn retained_checkpoints(&self) -> usize {
        self.construction.builder.checkpoints.len()
            + self.construction.materializers.len()
            + usize::from(matches!(self.cause, Cause::Materializer(_)))
    }
}
impl fmt::Display for PreparedGgufSourceFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for PreparedGgufSourceFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Store(cause) => cause,
            Cause::Readers(cause) => cause,
            Cause::Materializer(cause) => cause,
            Cause::Reserve(cause) => cause,
        })
    }
}
impl PreparedGgufCatalog {
    /// Exact constructor requests using the source already owned by this catalog.
    /// No descriptor rebuilding, source callbacks or artifact reopen occurs.
    pub fn source_storage_request<C>(&self) -> Option<GgufSourceStorageRequest> {
        self.builder.source_storage_request::<C>()
    }
    /// One source constructor with caller-owned, already admitted custody. An
    /// external C cannot impersonate a runtime's private accounting origin.
    pub fn build_with_source_custody<C: Any + fmt::Debug + Send + Sync>(
        self,
        custody: C,
    ) -> Result<GgufWeightStore, PreparedGgufSourceFailure> {
        prepare(self.builder, Some(SourceControl::new(custody)), None)
    }
}
impl GgufWeightStoreBuilder {
    fn source_storage_request<C>(&self) -> Option<GgufSourceStorageRequest> {
        let mut original_headers = source_header_bytes(&self.checkpoints)?;
        let count = self.checkpoints.len();
        let readers = self.reader_buffer_count();
        let shards = self
            .checkpoints
            .iter()
            .try_fold(0usize, |n, c| n.checked_add(c.shards().len()))?;
        let mut buffers = Layout::array::<TensorMaterializer>(count)
            .ok()?
            .size()
            .checked_add(Layout::array::<u64>(count).ok()?.size())?
            .checked_add(Layout::array::<TouchedEntry>(shards).ok()?.size())?
            .checked_add(
                Layout::array::<eredu_gguf::ReaderBuffer>(readers)
                    .ok()?
                    .size(),
            )?
            .checked_add(readers.checked_mul(eredu_gguf::Reader::<
                std::io::BufReader<std::fs::File>,
            >::file_buffer_capacity())?)?;
        let mut materializer_shared = Layout::new::<()>();
        let mut peak = 0;
        for checkpoint in &self.checkpoints {
            let request = checkpoint.prepared_materializer_storage::<Option<SourceControl>>()?;
            buffers = buffers.checked_add(request.requested_buffer_bytes()?)?;
            peak = peak.max(request.control_bytes());
            original_headers =
                original_headers.checked_sub(u64::try_from(request.scratch_bytes()).ok()?)?;
            materializer_shared = request.shared_body();
        }
        let controls = [
            size_of::<Construction>(),
            size_of::<PreparedGgufSourceFailure>(),
            size_of::<Result<GgufWeightStore, PreparedGgufSourceFailure>>(),
            size_of::<StoreInner>(),
            size_of::<ReaderCache>(),
            size_of::<GgufReaderBuffers>(),
            size_of::<Result<GgufReaderBuffers, GgufReaderBuffersFailure>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<(), StoreError>>(),
            size_of::<Option<SourceControl>>(),
            size_of::<SourceHandle<StoreInner>>(),
            size_of::<std::sync::MutexGuard<'static, ReaderCache>>(),
            size_of::<std::sync::LockResult<std::sync::MutexGuard<'static, ReaderCache>>>(),
            crate::recipe::RecipeInferenceCache::initial_control_bytes()?,
            SourceControl::owner_control_bytes::<C>()?,
            SourceHandle::<StoreInner>::owner_control_bytes()?,
            TouchedPaths::construction_controls()?,
            GgufReaderBuffers::preparation_controls()?,
            size_of::<HeaderInventory>(),
            size_of::<&eredu_gguf::PreparedHeader>(),
            size_of::<Option<&eredu_gguf::PreparedHeader>>(),
            size_of::<std::slice::Iter<'static, Checkpoint>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'static, Checkpoint>>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'static, eredu_gguf::CatalogShard>>>(),
            size_of::<usize>(), // materializer count
            size_of::<usize>(), // shard count
            size_of::<u64>(),   // original header inventory
            size_of::<Option<u64>>(),
            size_of::<&mut Vec<TensorMaterializer>>(),
            size_of::<&mut Vec<u64>>(),
            size_of::<&mut Vec<TouchedEntry>>(),
            peak,
        ];
        let controls = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)?;
        buffers.checked_add(controls)?;
        Some(GgufSourceStorageRequest {
            buffers,
            controls,
            source: Layout::new::<StoreInner>(),
            custody: SourceControl::body_layout::<C>(),
            materializers: count,
            materializer_shared,
            physical_inventory: u64::try_from(buffers).ok()?.checked_add(original_headers)?,
            prepaid_inventory: u64::try_from(buffers).ok()?,
        })
    }
}

pub(super) fn prepare(
    builder: GgufWeightStoreBuilder,
    control: Option<SourceControl>,
    buffers: Option<GgufReaderBuffers>,
) -> Result<GgufWeightStore, PreparedGgufSourceFailure> {
    let mut state = Construction {
        builder,
        materializers: Vec::new(),
        buffers,
        last_used: Vec::new(),
        touched_entries: Vec::new(),
        touched: None,
        header_bytes: 0,
        control,
    };
    if let Err(cause) = state.build() {
        return Err(PreparedGgufSourceFailure {
            cause,
            construction: state,
        });
    }
    Ok(state.publish())
}
impl Construction {
    fn build(&mut self) -> Result<(), Cause> {
        self.builder.validate_build().map_err(Cause::Store)?;
        self.header_bytes = source_header_bytes(&self.builder.checkpoints).ok_or_else(|| {
            Cause::Store(StoreError::Overflow {
                context: "GGUF prepared header inventory".into(),
            })
        })?;
        if self.buffers.is_none() {
            self.buffers = Some(
                GgufReaderBuffers::prepare(self.builder.reader_buffer_count())
                    .map_err(Cause::Readers)?,
            );
        }
        let count = self.builder.checkpoints.len();
        let shards = self
            .builder
            .checkpoints
            .iter()
            .try_fold(0usize, |n, c| n.checked_add(c.shards().len()))
            // TouchedEntry is nonzero-sized: this asks the reached reserve
            // worker for its typed capacity-overflow error, retaining state.
            .unwrap_or(usize::MAX);
        self.materializers
            .try_reserve_exact(count)
            .map_err(Cause::Reserve)?;
        self.last_used
            .try_reserve_exact(count)
            .map_err(Cause::Reserve)?;
        self.last_used.resize(count, 0);
        self.touched_entries
            .try_reserve_exact(shards)
            .map_err(Cause::Reserve)?;
        // Source order is retained without a new input vector or collect.
        self.builder.checkpoints.reverse();
        while !self.builder.checkpoints.is_empty() {
            #[cfg(test)]
            if FAIL_AFTER.with(|at| at.get() == Some(self.materializers.len())) {
                return Err(Cause::Reserve(
                    self.last_used.try_reserve_exact(usize::MAX).unwrap_err(),
                ));
            }
            let checkpoint = self
                .builder
                .checkpoints
                .pop()
                .expect("remaining checkpoint");

            let materializer = checkpoint
                .try_into_prepared_materializer(self.control.clone())
                .map_err(Cause::Materializer)?;
            self.materializers.push(materializer);
        }
        self.touched = Some(TouchedPaths::from_prepared(
            std::mem::take(&mut self.touched_entries),
            &self.materializers,
        ));
        Ok(())
    }
    fn publish(self) -> GgufWeightStore {
        let Self {
            builder,
            materializers,
            buffers,
            last_used,
            touched,
            touched_entries,
            header_bytes,
            control,
        } = self;
        drop(touched_entries);
        let count = materializers.len();
        let touched = touched.expect("complete touched storage");
        let touched_storage_bytes = touched
            .capacity_layout()
            .expect("prechecked touched bytes")
            .size() as u64;
        let prepared_header_storage_bytes = header_bytes;
        let prepared_reader_storage_bytes =
            buffers.as_ref().and_then(GgufReaderBuffers::storage_bytes);
        let cache = ReaderCache {
            materializers,
            buffers,
            last_used,
            touched,
            tick: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        };
        let prepared_materializer_storage_bytes = Some(
            cache
                .prepared_materializer_storage_bytes()
                .expect("prechecked fresh materializer inventory"),
        );
        let recipes = crate::recipe::RecipeInferenceCache::default();
        recipes.initialize_control_storage();
        let readers = Mutex::new(cache);
        drop(readers.lock().expect("private reader cache"));
        GgufWeightStore {
            inner: SourceHandle::new(
                StoreInner {
                    recipes,
                    readers,
                    reader_slots: count,
                    touched_storage_bytes,
                    prepared_header_storage_bytes,
                    prepared_reader_storage_bytes,
                    prepared_materializer_storage_bytes,
                    max_cached_readers: if builder.max_cached_readers == 0 {
                        DEFAULT_MAX_CACHED_SHARDS
                    } else {
                        builder.max_cached_readers
                    },
                    statistics: StoreStatistics::default(),
                    catalog: builder.sealed_catalog.unwrap_or_else(|| {
                        catalog::CatalogHandle::new(
                            catalog::CatalogData {
                                rows: builder.catalog,
                                unclaimed: builder.unclaimed_keys,
                            },
                            (),
                        )
                    }),
                },
                control,
            ),
        }
    }
}

// Original captured-header payload is counted for the existing source inventory,
// not allocated by this compiler. Distinct source owners retain it without
// duplicating shared captured headers; parser scratch is new per materializer.
struct HeaderInventory {
    checkpoint: usize,
    shard: usize,
    earlier_checkpoint: usize,
    earlier_shard: usize,
    end: usize,
    bytes: u64,
    scratch: usize,
    duplicate: bool,
}
fn source_header_bytes(checkpoints: &[Checkpoint]) -> Option<u64> {
    let mut frame = HeaderInventory {
        checkpoint: 0,
        shard: 0,
        earlier_checkpoint: 0,
        earlier_shard: 0,
        end: 0,
        bytes: 0,
        scratch: 0,
        duplicate: false,
    };
    while frame.checkpoint < checkpoints.len() {
        let current = &checkpoints[frame.checkpoint];
        frame.scratch = current
            .prepared_materializer_storage::<Option<SourceControl>>()?
            .scratch_bytes();
        frame.shard = 0;
        while frame.shard < current.shards().len() {
            if let Some(header) = current.shards()[frame.shard].prepared_header() {
                frame.duplicate = false;
                frame.earlier_checkpoint = 0;
                while frame.earlier_checkpoint <= frame.checkpoint && !frame.duplicate {
                    let earlier = &checkpoints[frame.earlier_checkpoint];
                    frame.end = if frame.earlier_checkpoint == frame.checkpoint {
                        frame.shard
                    } else {
                        earlier.shards().len()
                    };
                    frame.earlier_shard = 0;
                    while frame.earlier_shard < frame.end {
                        if earlier.shards()[frame.earlier_shard]
                            .prepared_header()
                            .is_some_and(|other| std::ptr::eq(header, other))
                        {
                            frame.duplicate = true;
                            break;
                        }
                        frame.earlier_shard += 1;
                    }
                    frame.earlier_checkpoint += 1;
                }
                if !frame.duplicate {
                    frame.bytes = frame
                        .bytes
                        .checked_add(u64::try_from(header.retained_payload_bytes()?).ok()?)?;
                }
            }
            frame.shard += 1;
        }
        frame.bytes = frame
            .bytes
            .checked_add(u64::try_from(frame.scratch).ok()?)?;
        frame.checkpoint += 1;
    }
    Some(frame.bytes)
}

#[cfg(test)]
thread_local! { static FAIL_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
mod tests;
