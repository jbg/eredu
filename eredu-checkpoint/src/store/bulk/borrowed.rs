//! Borrowed grouping with one finite set of read scratch arrays.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

/// Fixed read failure. Ordinals refer to the caller's retained read plans;
/// no source path or formatted diagnostic is copied during prepared reads.
#[derive(Debug, thiserror::Error)]
#[error("encoded read failed at batch {batch:?}, shard {shard:?} after {completed_shards} shards: {cause}")]
pub struct EncodedReadFailure {
    /// Index in the supplied batch iterator, if the failure concerns a shard.
    pub batch: Option<usize>,
    /// Index in that batch's stable shard order, if applicable.
    pub shard: Option<usize>,
    /// Number of shard reads validated and published before this failure.
    pub completed_shards: usize,
    /// Original fixed or operating-system cause.
    #[source]
    pub cause: EncodedReadFailureCause,
}
/// Causes created by the common borrowed read worker.
#[derive(Debug, thiserror::Error)]
pub enum EncodedReadFailureCause {
    /// All output lengths are checked before any read.
    #[error("encoded read destinations have the wrong lengths")]
    DestinationLengths,
    /// Checked read-scratch layout overflowed.
    #[error("encoded read scratch layout overflow")]
    Layout,
    /// One exact scratch allocation could not be reserved.
    #[error("encoded read scratch reservation failed: {0}")]
    Reserve(#[source] TryReserveError),
    /// A retained admitted file no longer matches the current file.
    #[error("admitted encoded source changed")]
    Changed,
    /// File opening or metadata inspection failed.
    #[error("encoded source filesystem operation failed: {0}")]
    Filesystem(#[source] io::Error),
    /// Reading or seeking the source failed.
    #[error("encoded source read failed: {0}")]
    Read(#[source] io::Error),
    /// Existing diagnostic state could not be acquired.
    #[error("checkpoint shard cache is poisoned")]
    CachePoisoned,
    /// A source tried to publish a path outside its admitted shard universe.
    #[error("encoded source diagnostic identity mismatch")]
    DiagnosticIdentity,
}
impl EncodedReadFailure {
    fn new(cause: EncodedReadFailureCause) -> Self {
        Self {
            batch: None,
            shard: None,
            completed_shards: 0,
            cause,
        }
    }
    fn at(mut self, group: &Group<'_>, completed: usize) -> Self {
        self.batch = Some(group.batch);
        self.shard = Some(group.shard);
        self.completed_shards = completed;
        self
    }
    pub(super) fn into_ordinary(self, batches: &[EncodedReadBatch]) -> StoreError {
        let path = self
            .batch
            .zip(self.shard)
            .and_then(|(batch, shard)| batches.get(batch)?.shards.get(shard))
            .map(|row| row.0.as_path())
            .unwrap_or_else(|| Path::new(""));
        match self.cause {
            EncodedReadFailureCause::Changed => StoreError::AdmittedFileChanged {
                path: path.to_path_buf(),
            },
            EncodedReadFailureCause::Filesystem(cause) => fs_error(path, cause),
            EncodedReadFailureCause::Read(cause) => io_error(path, cause),
            cause => StoreError::Internal(cause.to_string()),
        }
    }
}

/// Requested scratch and fixed controls for one actual borrowed read sequence.
/// This excludes caller-owned outputs and retained source/plan owners. File and
/// I/O error representations are included; OS-private file/runtime allocations
/// are outside this host storage contract. It grants no source or native authority.
#[derive(Clone, Copy, Debug)]
pub struct EncodedReadLayout {
    groups: usize,
    spans: usize,
    batch_spans: usize,
    buffers: usize,
    bytes: usize,
}
impl EncodedReadLayout {
    /// Inspect exact retained spans; no path cloning, reads or cache mutation.
    pub fn inspect<'a, I>(batches: I) -> Option<Self>
    where
        I: IntoIterator<Item = &'a EncodedReadBatch>,
    {
        Self::inspect_views(batches.into_iter().map(BatchView::ordinary))
    }
    pub(super) fn inspect_views<'a, I>(batches: I) -> Option<Self>
    where
        I: IntoIterator<Item = BatchView<'a>>,
    {
        let mut groups = 0usize;
        let mut spans = 0usize;
        let mut batch_spans = 0usize;
        for batch in batches {
            if !batch.memory.is_empty() && !batch.shards.is_empty() {
                return None;
            }
            for source in batch.memory {
                source.validate(batch.byte_len)?;
            }
            groups = groups.checked_add(batch.shards.len())?;
            let mut current = 0usize;
            for shard in batch.shards.iter() {
                current = current.checked_add(shard.spans.len())?;
                for span in shard.spans {
                    let length = span.destination.end.checked_sub(span.destination.start)?;
                    let chunks =
                        length / MAX_READ_BYTES + usize::from(length % MAX_READ_BYTES != 0);
                    spans = spans.checked_add(chunks)?;
                }
            }
            batch_spans = batch_spans.max(current);
        }
        let buffers = spans.min(MAX_READ_BUFFERS);
        let requests = [
            Layout::array::<Group<'_>>(groups).ok()?.size(),
            Layout::array::<PendingSpan<'_>>(batch_spans).ok()?.size(),
            Layout::array::<GroupedSpan<'_>>(spans).ok()?.size(),
            Layout::array::<IoSliceMut<'_>>(buffers).ok()?.size(),
        ];
        let controls = [
            size_of::<Self>(),
            size_of::<I>(),
            size_of::<I::IntoIter>(),
            size_of::<BatchView<'_>>(),
            size_of::<ShardView<'_>>(),
            size_of::<memory::ReadMemory>(),
            size_of::<std::slice::Iter<'_, memory::ReadMemory>>(),
            size_of::<std::slice::Iter<'_, ReadSpan>>(),
            size_of::<[Option<&[u8]>; 2]>(),
            size_of::<Option<&mut [u8]>>(),
            size_of::<Vec<Group<'_>>>(),
            size_of::<Vec<PendingSpan<'_>>>(),
            size_of::<Vec<GroupedSpan<'_>>>(),
            size_of::<Vec<IoSliceMut<'_>>>(),
            size_of::<File>(),
            size_of::<std::fs::Metadata>(),
            size_of::<crate::artifact::file::FileVersion>(),
            size_of::<Result<File, io::Error>>(),
            size_of::<Result<std::fs::Metadata, io::Error>>(),
            size_of::<Result<crate::artifact::file::FileVersion, io::Error>>(),
            size_of::<super::super::read_bytes::SafetensorsByteError<'_>>(),
            size_of::<Result<(), super::super::read_bytes::SafetensorsByteError<'_>>>(),
            size_of::<std::iter::Peekable<SelectedSpans<'_, '_>>>(),
            size_of::<Result<(), io::Error>>(),
            size_of::<Result<(), EncodedReadFailure>>(),
            size_of::<EncodedReadFailure>(),
            size_of::<MutexGuard<'_, CacheState>>(),
            size_of::<std::iter::Peekable<std::vec::IntoIter<GroupedSpan<'_>>>>(),
        ];
        let bytes = requests
            .iter()
            .chain(&controls)
            .try_fold(0usize, |total, n| total.checked_add(*n))?;
        Some(Self {
            groups,
            spans,
            batch_spans,
            buffers,
            bytes,
        })
    }
    pub(super) fn with_extra_controls(mut self, bytes: usize) -> Option<Self> {
        self.bytes = self.bytes.checked_add(bytes)?;
        Some(self)
    }
    /// Total requested scratch plus named fixed controls, excluding allocator overhead.
    pub const fn required_bytes(self) -> usize {
        self.bytes
    }
}
impl EncodedReadBatch {
    /// Requested owned backing for one metadata clone, excluding its inline
    /// value and the already retained shared source owners. No B-tree nodes occur.
    pub fn clone_storage_bytes(&self) -> Option<usize> {
        let mut bytes = Layout::array::<TensorMetadata>(self.tensors.len())
            .ok()?
            .size()
            .checked_add(
                Layout::array::<(PathBuf, ReadShard)>(self.shards.len())
                    .ok()?
                    .size(),
            )?;
        for tensor in &self.tensors {
            bytes = bytes.checked_add(MetadataCloneLayout::of(tensor)?.payload_bytes()?)?;
        }
        for (path, shard) in &self.shards {
            bytes = bytes
                .checked_add(path.as_os_str().as_encoded_bytes().len())?
                .checked_add(Layout::array::<ReadSpan>(shard.spans.len()).ok()?.size())?;
        }
        bytes = bytes.checked_add(
            Layout::array::<memory::ReadMemory>(self.memory.len())
                .ok()?
                .size(),
        )?;
        for source in &self.memory {
            bytes =
                bytes.checked_add(Layout::array::<ReadSpan>(source.spans.len()).ok()?.size())?;
        }
        Some(bytes)
    }
    /// Named fixed clone transports; backing requests are reported separately.
    /// Shared reference-count increments create no new source owner allocation.
    pub fn clone_control_bytes() -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<TensorMetadata>(),
            size_of::<ReadShard>(),
            size_of::<ReadSpan>(),
            size_of::<memory::ReadMemory>(),
            size_of::<storage::SourceHandle<MemoryTensor>>(),
            size_of::<PathBuf>(),
            size_of::<String>(),
            size_of::<Vec<usize>>(),
            size_of::<Arc<AdmittedFile>>(),
            size_of::<Arc<SafetensorsReadTelemetry>>(),
            size_of::<Arc<Mutex<CacheState>>>(),
        ]
        .iter()
        .try_fold(0usize, |total, bytes| total.checked_add(*bytes))
    }

    /// Reuses immutable plans. All destination lengths are checked before the
    /// first write; on failure every output remains caller-owned and invalid.
    /// The caller retains these plans and its admitted scratch/error custody.
    pub fn read_many_borrowed_into<'a, I>(
        batches: I,
        destinations: &mut [&mut [u8]],
    ) -> Result<(), EncodedReadFailure>
    where
        I: Iterator<Item = &'a Self> + Clone + ExactSizeIterator,
    {
        read_many(batches, destinations, || {})
    }
}

#[derive(Clone, Copy)]
pub(super) struct BatchView<'a> {
    pub(super) byte_len: usize,
    pub(super) shards: ShardsView<'a>,
    pub(super) memory: &'a [memory::ReadMemory],
    pub(super) telemetry: Option<&'a SafetensorsReadTelemetry>,
    pub(super) diagnostics: Option<Diagnostics<'a>>,
}
impl<'a> BatchView<'a> {
    fn ordinary(batch: &'a EncodedReadBatch) -> Self {
        Self {
            byte_len: batch.byte_len,
            shards: ShardsView::Ordinary(&batch.shards),
            memory: &batch.memory,
            telemetry: batch.telemetry.as_deref(),
            diagnostics: batch.cache.as_deref().map(Diagnostics::Ordinary),
        }
    }
}
#[derive(Clone, Copy)]
pub(super) enum ShardsView<'a> {
    Ordinary(&'a [(PathBuf, ReadShard)]),
    Detached(&'a [super::detached::DetachedShard]),
}
#[derive(Clone, Copy)]
pub(super) struct ShardView<'a> {
    pub(super) path: &'a Path,
    pub(super) admitted: &'a AdmittedFile,
    pub(super) spans: &'a [ReadSpan],
}
impl<'a> ShardsView<'a> {
    fn is_empty(self) -> bool {
        self.len() == 0
    }
    fn len(self) -> usize {
        match self {
            Self::Ordinary(rows) => rows.len(),
            Self::Detached(rows) => rows.len(),
        }
    }
    fn iter(self) -> impl Iterator<Item = ShardView<'a>> + ExactSizeIterator + Clone {
        (0..self.len()).map(move |index| match self {
            Self::Ordinary(rows) => {
                let (path, row) = &rows[index];
                ShardView {
                    path,
                    admitted: &row.admitted,
                    spans: &row.spans,
                }
            }
            Self::Detached(rows) => {
                let row = &rows[index];
                ShardView {
                    path: &row.path,
                    admitted: &row.admitted,
                    spans: &row.spans,
                }
            }
        })
    }
}
#[derive(Clone, Copy)]
pub(super) enum Diagnostics<'a> {
    Ordinary(&'a Mutex<CacheState>),
    Detached(&'a super::detached::DetachedSource),
}
impl Diagnostics<'_> {
    fn mark(self, path: &Path) -> Result<(), EncodedReadFailureCause> {
        let marked = match self {
            Self::Ordinary(cache) => cache
                .lock()
                .map_err(|_| EncodedReadFailureCause::CachePoisoned)?
                .payloads
                .mark(path),
            Self::Detached(source) => source.mark(path),
        };
        if marked {
            Ok(())
        } else {
            Err(EncodedReadFailureCause::DiagnosticIdentity)
        }
    }
}
struct Group<'a> {
    path: &'a Path,
    admitted: &'a AdmittedFile,
    telemetry: &'a SafetensorsReadTelemetry,
    diagnostics: Diagnostics<'a>,
    batch: usize,
    shard: usize,
    order: usize,
}
struct PendingSpan<'a> {
    group: usize,
    span: &'a ReadSpan,
}
struct GroupedSpan<'a> {
    group: usize,
    span: DestinationSpan<'a>,
}
struct SelectedSpans<'s, 'd> {
    spans: &'s mut std::iter::Peekable<std::vec::IntoIter<GroupedSpan<'d>>>,
    group: usize,
}
impl<'d> Iterator for SelectedSpans<'_, 'd> {
    type Item = DestinationSpan<'d>;
    fn next(&mut self) -> Option<Self::Item> {
        self.spans
            .next_if(|row| row.group == self.group)
            .map(|row| row.span)
    }
}
fn reserve<T>(count: usize) -> Result<Vec<T>, EncodedReadFailure> {
    let mut value = Vec::new();
    value
        .try_reserve_exact(count)
        .map_err(|cause| EncodedReadFailure::new(EncodedReadFailureCause::Reserve(cause)))?;
    Ok(value)
}
fn matches(group: &Group<'_>, batch: &BatchView<'_>, path: &Path) -> bool {
    group.path == path
        && batch
            .telemetry
            .is_some_and(|value| std::ptr::eq(group.telemetry, value))
}
fn validate(group: &Group<'_>, file: &File) -> Result<(), EncodedReadFailureCause> {
    use super::super::read_bytes::SafetensorsByteError;
    super::super::read_bytes::fixed_validate_file(group.admitted, group.path, file).map_err(
        |cause| match cause {
            SafetensorsByteError::Changed { .. } => EncodedReadFailureCause::Changed,
            SafetensorsByteError::Filesystem { cause, .. } => {
                EncodedReadFailureCause::Filesystem(cause)
            }
            // The common file validator only creates the two variants above.
            _ => EncodedReadFailureCause::DiagnosticIdentity,
        },
    )
}
pub(super) fn read_many<'a, I>(
    batches: I,
    destinations: &mut [&mut [u8]],
    after_shard: impl FnMut(),
) -> Result<(), EncodedReadFailure>
where
    I: Iterator<Item = &'a EncodedReadBatch> + Clone + ExactSizeIterator,
{
    read_many_views(batches.map(BatchView::ordinary), destinations, after_shard)
}
pub(super) fn read_many_views<'a, I>(
    batches: I,
    destinations: &mut [&mut [u8]],
    mut after_shard: impl FnMut(),
) -> Result<(), EncodedReadFailure>
where
    I: Iterator<Item = BatchView<'a>> + Clone + ExactSizeIterator,
{
    if batches.len() != destinations.len()
        || batches
            .clone()
            .zip(destinations.iter())
            .any(|(batch, dest)| batch.byte_len != dest.len())
    {
        return Err(EncodedReadFailure::new(
            EncodedReadFailureCause::DestinationLengths,
        ));
    }
    let layout = EncodedReadLayout::inspect_views(batches.clone())
        .ok_or_else(|| EncodedReadFailure::new(EncodedReadFailureCause::Layout))?;
    let mut groups: Vec<Group<'_>> = reserve(layout.groups)?;
    let mut pending: Vec<PendingSpan<'_>> = reserve(layout.batch_spans)?;
    let mut spans: Vec<GroupedSpan<'_>> = reserve(layout.spans)?;
    let mut buffers = reserve(layout.buffers)?;
    // Source telemetry identity distinguishes independently admitted stores that
    // happen to name the same path. Every input remains borrowed through return.
    for (batch_index, batch) in batches.clone().enumerate() {
        for (shard_index, shard) in batch.shards.iter().enumerate() {
            let path = shard.path;
            if let Some(group) = groups.iter().find(|group| matches(group, &batch, path)) {
                if group.admitted.identity != shard.admitted.identity {
                    return Err(
                        EncodedReadFailure::new(EncodedReadFailureCause::Changed).at(group, 0)
                    );
                }
            } else {
                if groups.len() == layout.groups {
                    return Err(EncodedReadFailure::new(EncodedReadFailureCause::Layout));
                }
                groups.push(Group {
                    path,
                    admitted: shard.admitted,
                    telemetry: batch
                        .telemetry
                        .ok_or_else(|| EncodedReadFailure::new(EncodedReadFailureCause::Layout))?,
                    diagnostics: batch
                        .diagnostics
                        .ok_or_else(|| EncodedReadFailure::new(EncodedReadFailureCause::Layout))?,
                    batch: batch_index,
                    shard: shard_index,
                    order: groups.len(),
                });
            }
        }
    }
    groups.sort_unstable_by(|left, right| {
        left.path.cmp(right.path).then(left.order.cmp(&right.order))
    });
    for (batch_index, (batch, destination)) in batches.zip(destinations.iter_mut()).enumerate() {
        for source in batch.memory {
            source.copy_into(destination).map_err(|mut cause| {
                cause.batch = Some(batch_index);
                cause
            })?;
        }
        pending.clear();
        for shard in batch.shards.iter() {
            let path = shard.path;
            let group = groups
                .iter()
                .position(|group| matches(group, &batch, path))
                .expect("grouped above");
            for span in shard.spans {
                if pending.len() == layout.batch_spans {
                    return Err(EncodedReadFailure::new(EncodedReadFailureCause::Layout));
                }
                pending.push(PendingSpan { group, span });
            }
        }
        pending.sort_unstable_by_key(|row| row.span.destination.start);
        let mut remaining = &mut **destination;
        let mut position = 0;
        for row in &pending {
            let (_, tail) = remaining.split_at_mut(row.span.destination.start - position);
            let (bytes, tail) = tail.split_at_mut(row.span.destination.len());
            remaining = tail;
            position = row.span.destination.end;
            let mut offset = row.span.source.start;
            for bytes in bytes.chunks_mut(MAX_READ_BYTES) {
                let end = offset + bytes.len() as u64;
                if spans.len() == layout.spans {
                    return Err(EncodedReadFailure::new(EncodedReadFailureCause::Layout));
                }
                spans.push(GroupedSpan {
                    group: row.group,
                    span: DestinationSpan {
                        file: offset..end,
                        bytes,
                    },
                });
                offset = end;
            }
        }
    }
    spans.sort_unstable_by_key(|row| (row.group, row.span.file.start));
    let mut spans = spans.into_iter().peekable();
    for (index, group) in groups.iter().enumerate() {
        let result = (|| {
            if group.path != group.admitted.identity.canonical_path {
                return Err(EncodedReadFailureCause::Changed);
            }
            let mut file = File::open(group.path).map_err(EncodedReadFailureCause::Filesystem)?;
            validate(group, &file)?;
            let selected = SelectedSpans {
                spans: &mut spans,
                group: index,
            };
            read_destination_spans_with_buffers(&mut file, selected, group.telemetry, &mut buffers)
                .map_err(EncodedReadFailureCause::Read)?;
            group.diagnostics.mark(group.path)?;
            after_shard();
            validate(group, &file)
        })();
        result.map_err(|cause| EncodedReadFailure::new(cause).at(group, index))?;
    }
    Ok(())
}
