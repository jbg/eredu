//! Exact admitted metadata detached from ordinary source/cache owners.
use super::*;
use crate::recipe::{EncodedRecipeRead, EncodedRecipeReadView, RecipeDtype, RecipeMetadata};
use std::{alloc::Layout, collections::TryReserveError, mem::size_of, sync::atomic::AtomicBool};

/// A pure constructor over exact existing admitted reads. Neither inspection nor
/// construction reads payloads, admits headers or selects another source.
pub struct DetachedEncodedReadPlan<'a, I>
where
    I: Iterator<Item = EncodedRecipeReadView<'a>> + Clone + ExactSizeIterator,
{
    reads: I,
    count: usize,
    bytes: usize,
}
impl<'a, I> DetachedEncodedReadPlan<'a, I>
where
    I: Iterator<Item = EncodedRecipeReadView<'a>> + Clone + ExactSizeIterator,
{
    pub(crate) fn inspect(reads: I) -> Option<Self> {
        let count = reads.len();
        let mut bytes = Layout::array::<DetachedRead>(count)
            .ok()?
            .size()
            .checked_add(Layout::array::<DetachedSource>(count).ok()?.size())?
            .checked_add(
                Layout::array::<&SafetensorsReadTelemetry>(count)
                    .ok()?
                    .size(),
            )?;
        for read in reads.clone() {
            let batch = read.admitted_batch();
            bytes = bytes
                .checked_add(
                    Layout::array::<TensorMetadata>(batch.tensors.len())
                        .ok()?
                        .size(),
                )?
                .checked_add(
                    Layout::array::<DetachedShard>(batch.shards.len())
                        .ok()?
                        .size(),
                )?
                .checked_add(
                    Layout::array::<usize>(read.output().shape.len())
                        .ok()?
                        .size(),
                )?;
            if let RecipeDtype::Other(name) = &read.output().dtype {
                bytes = bytes.checked_add(name.len())?;
            }
            bytes = bytes.checked_add(
                Layout::array::<memory::ReadMemory>(batch.memory.len())
                    .ok()?
                    .size(),
            )?;
            for source in &batch.memory {
                bytes = bytes
                    .checked_add(Layout::array::<ReadSpan>(source.spans.len()).ok()?.size())?;
            }
            for tensor in &batch.tensors {
                bytes = bytes.checked_add(MetadataCloneLayout::of(tensor)?.payload_bytes()?)?;
            }
            for (path, shard) in &batch.shards {
                // One shard path, one exact admitted identity path, and one
                // source-diagnostic path occurrence, all owned after admission.
                bytes = bytes
                    .checked_add(path.as_os_str().as_encoded_bytes().len().checked_mul(2)?)?
                    .checked_add(
                        shard
                            .admitted
                            .identity
                            .canonical_path
                            .as_os_str()
                            .as_encoded_bytes()
                            .len(),
                    )?
                    .checked_add(Layout::array::<ReadSpan>(shard.spans.len()).ok()?.size())?
                    .checked_add(size_of::<(PathBuf, AtomicBool)>())?;
            }
        }
        Some(Self {
            reads,
            count,
            bytes,
        })
    }
    /// All detached backing requests and named constructor/error transports.
    /// C is the caller's existing source custody, not a new grant. The caller
    /// prices any storage created by its own C construction/clone separately.
    pub fn required_bytes<C>(&self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<I>(),
            size_of::<DetachedEncodedReads<C>>(),
            size_of::<DetachedReadBuildError<C>>(),
            size_of::<DetachedReadFailure<C>>(),
            size_of::<Result<DetachedEncodedReads<C>, DetachedReadBuildError<C>>>(),
            size_of::<DetachedRead>(),
            size_of::<DetachedShard>(),
            size_of::<memory::ReadMemory>(),
            size_of::<storage::SourceHandle<MemoryTensor>>(),
            size_of::<DetachedSource>(),
            size_of::<AdmittedFile>(),
            size_of::<AdmittedFileIdentity>(),
            size_of::<Vec<&SafetensorsReadTelemetry>>(),
            size_of::<TensorMetadata>(),
            size_of::<RecipeMetadata>(),
            size_of::<EncodedRecipeReadView<'a>>(),
            size_of::<Option<EncodedRecipeReadView<'a>>>(),
            size_of::<PathBuf>(),
            size_of::<String>(),
            size_of::<Vec<usize>>(),
            size_of::<Vec<ReadSpan>>(),
            size_of::<TryReserveError>(),
        ]
        .iter()
        .try_fold(self.bytes, |sum, n| sum.checked_add(*n))
    }
    /// Same per-call read scratch derived before detachment. The actual source
    /// request and its C-owned error envelopes remain separate from this worker.
    pub fn read_layout<C>(&self) -> Option<EncodedReadLayout> {
        EncodedRecipeReadView::borrowed_read_layout(self.reads.clone())?.with_extra_controls(
            DetachedEncodedReads::<C>::read_controls()?
                .checked_add(size_of::<DetachedViews<'_, C>>().checked_mul(2)?)?,
        )
    }

    /// Construct only after accepting the exact source account. All successful
    /// metadata prefixes retire before the supplied custody on failure/unwind.
    pub fn construct<C>(
        self,
        custody: C,
    ) -> Result<DetachedEncodedReads<C>, DetachedReadBuildError<C>> {
        let mut owner = DetachedEncodedReads {
            reads: Vec::new(),
            sources: Vec::new(),
            custody,
        };
        let result = (|| {
            owner.reads.try_reserve_exact(self.count)?;
            owner.sources.try_reserve_exact(self.count)?;
            let mut source_keys: Vec<&SafetensorsReadTelemetry> = Vec::new();
            source_keys.try_reserve_exact(self.count)?;
            for read in self.reads.clone() {
                if owner.reads.len() == self.count {
                    return Err(DetachedReadBuildCause::Layout);
                }
                let batch = read.admitted_batch();
                let source = if let Some(telemetry) = batch.telemetry.as_ref() {
                    Some(
                        match source_keys
                            .iter()
                            .position(|key| std::ptr::eq(*key, telemetry.as_ref()))
                        {
                            Some(index) => index,
                            None => {
                                if owner.sources.len() == self.count {
                                    return Err(DetachedReadBuildCause::Layout);
                                }
                                let count = self
                                    .reads
                                    .clone()
                                    .filter(|other| {
                                        other
                                            .admitted_batch()
                                            .telemetry
                                            .as_ref()
                                            .is_some_and(|other| Arc::ptr_eq(other, telemetry))
                                    })
                                    .try_fold(0usize, |sum, other| {
                                        sum.checked_add(other.admitted_batch().shards.len())
                                    })
                                    .ok_or(DetachedReadBuildCause::Layout)?;
                                let mut paths = Vec::new();
                                paths.try_reserve_exact(count)?;
                                for other in self.reads.clone().filter(|other| {
                                    other
                                        .admitted_batch()
                                        .telemetry
                                        .as_ref()
                                        .is_some_and(|other| Arc::ptr_eq(other, telemetry))
                                }) {
                                    for (path, _) in &other.admitted_batch().shards {
                                        if paths.len() == count {
                                            return Err(DetachedReadBuildCause::Layout);
                                        }
                                        paths.push((path.clone(), AtomicBool::new(false)));
                                    }
                                }
                                paths.sort_unstable_by(|left, right| left.0.cmp(&right.0));
                                paths.dedup_by(|left, right| left.0 == right.0);
                                let index = owner.sources.len();
                                owner.sources.push(DetachedSource {
                                    telemetry: SafetensorsReadTelemetry::default(),
                                    paths,
                                });
                                source_keys.push(telemetry.as_ref());
                                index
                            }
                        },
                    )
                } else {
                    None
                };
                let mut memory = Vec::new();
                memory.try_reserve_exact(batch.memory.len())?;
                for row in &batch.memory {
                    let mut spans = Vec::new();
                    spans.try_reserve_exact(row.spans.len())?;
                    spans.extend(row.spans.iter().cloned());
                    memory.push(memory::ReadMemory {
                        tensor: row.tensor.clone(),
                        spans,
                    });
                }
                let mut shards = Vec::new();
                shards.try_reserve_exact(batch.shards.len())?;
                for (path, shard) in &batch.shards {
                    shards.push(DetachedShard {
                        path: path.clone(),
                        admitted: AdmittedFile {
                            identity: AdmittedFileIdentity {
                                canonical_path: shard.admitted.identity.canonical_path.clone(),
                                version: shard.admitted.identity.version,
                            },
                            source_admission: shard.admitted.source_admission.clone(),
                        },
                        spans: shard.spans.clone(),
                    });
                }
                owner.reads.push(DetachedRead {
                    output: read.output().clone(),
                    tensors: batch.tensors.clone(),
                    shards,
                    memory,
                    byte_len: batch.byte_len,
                    source,
                });
            }
            if owner.reads.len() != self.count {
                return Err(DetachedReadBuildCause::Layout);
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(owner),
            Err(cause) => Err(DetachedReadBuildError {
                cause,
                partial: owner,
            }),
        }
    }
}

/// Move-only source over detached admitted identities and finite diagnostics.
/// It exports neither ordinary cache aliases nor owning recipe-read clones.
/// All fields owning storage precede the caller's custody through final drop.
///
/// ```compile_fail
/// use eredu_checkpoint::store::DetachedEncodedReads;
/// fn copy(source: DetachedEncodedReads<()>) { let _ = source.clone(); }
/// ```
pub struct DetachedEncodedReads<C> {
    reads: Vec<DetachedRead>,
    sources: Vec<DetachedSource>,
    custody: C,
}
struct DetachedRead {
    output: RecipeMetadata,
    tensors: Vec<TensorMetadata>,
    shards: Vec<DetachedShard>,
    byte_len: usize,
    memory: Vec<memory::ReadMemory>,
    source: Option<usize>,
}
pub(super) struct DetachedShard {
    pub(super) path: PathBuf,
    pub(super) admitted: AdmittedFile,
    pub(super) spans: Vec<ReadSpan>,
}
pub(super) struct DetachedSource {
    telemetry: SafetensorsReadTelemetry,
    paths: Vec<(PathBuf, AtomicBool)>,
}
impl DetachedSource {
    pub(super) fn mark(&self, path: &Path) -> bool {
        let Ok(index) = self.paths.binary_search_by(|row| row.0.as_path().cmp(path)) else {
            return false;
        };
        self.paths[index].1.store(true, Ordering::Relaxed);
        true
    }
}
impl<C> DetachedEncodedReads<C> {
    /// Number of recipe outputs in their original destination order.
    pub fn len(&self) -> usize {
        self.reads.len()
    }
    /// Whether this source has no recipe outputs.
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty()
    }
    /// Exact original inferred output, borrowed without a source clone.
    pub fn output(&self, index: usize) -> Option<&RecipeMetadata> {
        Some(&self.reads.get(index)?.output)
    }
    /// Borrows an exact contiguous output subset without cloning its source or
    /// allocating an index list. Invalid bounds are rejected before any read.
    pub fn slice(&self, range: Range<usize>) -> Option<DetachedEncodedReadSlice<'_, C>> {
        self.reads.get(range.clone())?;
        Some(DetachedEncodedReadSlice { owner: self, range })
    }
    /// Checks a still-retained admitted read against this exact detached source.
    /// Geometry, file versions and every file/destination range must agree; no
    /// payload access, re-admission or owner clone is used. Memory spans must
    /// retain the exact immutable tensor owner; file identities remain versioned.
    /// Independent admissions of unchanged files may match. This comparison
    /// neither merges nor changes the detached source-local diagnostic groups,
    /// and does not replace the file-version checks performed by payload reads.
    pub fn matches_read<R>(&self, index: usize, other: &EncodedRecipeRead<R>) -> bool {
        self.matches_read_view(index, other.borrowed())
    }
    /// The same authenticated identity comparison through a custody-preserving
    /// loan. No source, metadata or funding owner is cloned.
    pub fn matches_read_view(&self, index: usize, other: EncodedRecipeReadView<'_>) -> bool {
        let Some(read) = self.reads.get(index) else {
            return false;
        };
        let batch = other.admitted_batch();
        read.output == *other.output()
            && read.tensors == batch.tensors
            && read.byte_len == batch.byte_len
            && read.memory.len() == batch.memory.len()
            && read
                .memory
                .iter()
                .zip(&batch.memory)
                .all(|(a, b)| a.matches(b))
            && read.shards.len() == batch.shards.len()
            && read.shards.iter().zip(&batch.shards).all(|(a, (path, b))| {
                a.path == *path
                    && a.admitted.identity == b.admitted.identity
                    && a.spans.len() == b.spans.len()
                    && a.spans
                        .iter()
                        .zip(&b.spans)
                        .all(|(a, b)| a.source == b.source && a.destination == b.destination)
            })
    }
    /// Exact encoded tensor metadata in one output's destination order.
    pub fn sources(&self, index: usize) -> Option<&[TensorMetadata]> {
        Some(&self.reads.get(index)?.tensors)
    }
    fn views(&self) -> DetachedViews<'_, C> {
        DetachedViews {
            owner: self,
            index: 0,
            end: self.reads.len(),
        }
    }
    fn read_controls() -> Option<usize> {
        [
            size_of::<C>(),
            size_of::<DetachedReadFailure<C>>(),
            size_of::<Result<(), DetachedReadFailure<C>>>(),
            size_of::<&Self>(),
        ]
        .iter()
        .try_fold(0usize, |total, n| total.checked_add(*n))
    }
    /// Scratch for the shared synchronous worker; source ownership is separate.
    pub fn read_layout(&self) -> Option<EncodedReadLayout> {
        EncodedReadLayout::inspect_views(self.views())?.with_extra_controls(Self::read_controls()?)
    }
    /// Actual detached diagnostics, borrowed without building a reporting Vec.
    /// Entries remain distinct for independently admitted original source stores.
    pub fn visit_payload_paths(&self, mut visitor: impl FnMut(usize, &Path)) {
        for (source, row) in self.sources.iter().enumerate() {
            for (path, used) in &row.paths {
                if used.load(Ordering::Relaxed) {
                    visitor(source, path);
                }
            }
        }
    }
    /// Completed physical payload bytes for one detached source, or None for an
    /// absent ordinal. Ordinary source diagnostics remain independent.
    pub fn physical_read_bytes(&self, source: usize) -> Option<u64> {
        Some(
            self.sources
                .get(source)?
                .telemetry
                .physical_read_bytes
                .load(Ordering::Relaxed),
        )
    }
}
struct DetachedViews<'a, C> {
    owner: &'a DetachedEncodedReads<C>,
    index: usize,
    end: usize,
}
impl<C> Clone for DetachedViews<'_, C> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner,
            index: self.index,
            end: self.end,
        }
    }
}
impl<'a, C> Iterator for DetachedViews<'a, C> {
    type Item = borrowed::BatchView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let read = self.owner.reads.get(self.index)?;
        self.index += 1;
        let source = read.source.map(|index| &self.owner.sources[index]);
        Some(borrowed::BatchView {
            byte_len: read.byte_len,
            shards: borrowed::ShardsView::Detached(&read.shards),
            memory: &read.memory,
            telemetry: source.map(|source| &source.telemetry),
            diagnostics: source.map(borrowed::Diagnostics::Detached),
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.index;
        (n, Some(n))
    }
}
impl<C> ExactSizeIterator for DetachedViews<'_, C> {}

/// A validated borrowed subset of one detached source. Failure batch ordinals
/// are relative to this subset; the retained source diagnostics stay unchanged.
pub struct DetachedEncodedReadSlice<'a, C> {
    owner: &'a DetachedEncodedReads<C>,
    range: Range<usize>,
}
impl<C> DetachedEncodedReadSlice<'_, C> {
    /// Snapshot of the detached source stores used by these reads. Counters are
    /// source-wide, including reads through other slices of the same owner.
    /// Each source contributes once; independent stores sharing a file remain
    /// distinct. Direct reads retain no shard cache or ordinary store counters.
    pub fn diagnostics(&self) -> WeightStoreDiagnostics {
        let mut report = WeightStoreDiagnostics {
            backend: WeightStoreBackend::Memory,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            currently_cached_shards: 0,
            touched_shard_paths: Vec::new(),
            payload_shard_paths: Vec::new(),
            physical_reads: 0,
            physical_read_bytes: 0,
            coalesced_group_hits: 0,
        };
        for (index, source) in self.owner.sources.iter().enumerate() {
            if !self.owner.reads[self.range.clone()]
                .iter()
                .any(|read| read.source == Some(index))
            {
                continue;
            }
            report.backend = WeightStoreBackend::Safetensors;
            report.physical_reads = report
                .physical_reads
                .saturating_add(source.telemetry.physical_reads.load(Ordering::Relaxed));
            report.physical_read_bytes = report
                .physical_read_bytes
                .saturating_add(source.telemetry.physical_read_bytes.load(Ordering::Relaxed));
            report.payload_shard_paths.extend(
                source
                    .paths
                    .iter()
                    .filter(|(_, used)| used.load(Ordering::Relaxed))
                    .map(|(path, _)| path.clone()),
            );
        }
        report.payload_shard_paths.sort_unstable();
        report.payload_shard_paths.dedup();
        report.touched_shard_paths = report.payload_shard_paths.clone();
        report
    }

    fn views(&self) -> DetachedViews<'_, C> {
        DetachedViews {
            owner: self.owner,
            index: self.range.start,
            end: self.range.end,
        }
    }
    /// Exact shared-reader scratch for this subset, without payload storage.
    pub fn read_layout(&self) -> Option<EncodedReadLayout> {
        EncodedReadLayout::inspect_views(self.views())?.with_extra_controls(
            DetachedEncodedReads::<C>::read_controls()?.checked_add(size_of::<Self>())?,
        )
    }
}
impl<C: Clone> DetachedEncodedReadSlice<'_, C> {
    /// Validates every destination before reading only this admitted subset.
    /// An escaped failure retains the same source custody as a full-source read.
    pub fn read_many_into(
        &self,
        destinations: &mut [&mut [u8]],
    ) -> Result<(), DetachedReadFailure<C>> {
        let custody = self.owner.custody.clone();
        borrowed::read_many_views(self.views(), destinations, || {}).map_err(|cause| {
            DetachedReadFailure {
                cause,
                _custody: custody,
            }
        })
    }
}

impl<C: Clone> DetachedEncodedReads<C> {
    /// Reads through the same validated shard worker. C's clone keeps the
    /// accepted source account with an escaped I/O cause after this owner drops.
    pub fn read_many_into(
        &self,
        destinations: &mut [&mut [u8]],
    ) -> Result<(), DetachedReadFailure<C>> {
        let custody = self.custody.clone();
        borrowed::read_many_views(self.views(), destinations, || {}).map_err(|cause| {
            DetachedReadFailure {
                cause,
                _custody: custody,
            }
        })
    }
}
/// Failed detached-source construction, retaining its successful metadata prefix.
pub struct DetachedReadBuildError<C> {
    cause: DetachedReadBuildCause,
    partial: DetachedEncodedReads<C>,
}
/// Fixed constructor failure, without source/path diagnostic cloning.
#[derive(Debug, thiserror::Error)]
pub enum DetachedReadBuildCause {
    /// One fixed backing reservation failed.
    #[error("detached read source storage reservation failed: {0}")]
    Reserve(#[from] TryReserveError),
    /// The supplied sequence no longer matches its inspected finite layout.
    #[error("detached read source layout mismatch")]
    Layout,
}
impl<C> DetachedReadBuildError<C> {
    /// Original fixed constructor cause; no partial owner can escape its custody.
    pub fn cause(&self) -> &DetachedReadBuildCause {
        &self.cause
    }
}
impl<C> std::fmt::Debug for DetachedReadBuildError<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedReadBuildError")
            .field("cause", &self.cause)
            .field("partial_reads", &self.partial.reads.len())
            .finish()
    }
}
impl<C> std::fmt::Display for DetachedReadBuildError<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl<C> std::error::Error for DetachedReadBuildError<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// A synchronous read failure whose original source charge remains retained.
pub struct DetachedReadFailure<C> {
    cause: EncodedReadFailure,
    _custody: C,
}
impl<C> DetachedReadFailure<C> {
    /// Fixed cause and exact failed source ordinals, without releasing custody.
    pub fn cause(&self) -> &EncodedReadFailure {
        &self.cause
    }
}
impl<C> std::fmt::Debug for DetachedReadFailure<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.cause, f)
    }
}
impl<C> std::fmt::Display for DetachedReadFailure<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl<C> std::error::Error for DetachedReadFailure<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
