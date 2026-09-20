//! Exact encoded reads into storage owned by the materializer.

use super::*;
use std::io::{self, IoSliceMut};

mod borrowed;
mod projection;
pub(crate) use projection::EncodedRange;
mod detached;
mod memory;
pub use memory::{
    MemoryEncodedReadBuildError, MemoryEncodedReadPlan, MemoryEncodedReadPlanError,
    PreparedMemoryEncodedRead,
};
pub use borrowed::{EncodedReadFailure, EncodedReadFailureCause, EncodedReadLayout};
pub use detached::{
    DetachedEncodedReadPlan, DetachedEncodedReadSlice, DetachedEncodedReads,
    DetachedReadBuildCause, DetachedReadBuildError, DetachedReadFailure,
};
pub(super) use memory::prepare as prepare_memory;

/// Every payload path comes from the retained admitted shard set. A successful
/// read changes one flag; it never allocates a path or grows a diagnostic map.
#[derive(Debug, Default)]
pub(super) struct PayloadPaths(Vec<(PathBuf, bool)>);
impl PayloadPaths {
    pub(super) fn new(paths: &[PathBuf]) -> Self {
        let mut rows = paths
            .iter()
            .map(|path| (path.clone(), false))
            .collect::<Vec<_>>();
        rows.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        rows.dedup_by(|left, right| left.0 == right.0);
        Self(rows)
    }
    pub(super) fn mark(&mut self, path: &Path) -> bool {
        let Ok(index) = self.0.binary_search_by(|row| row.0.as_path().cmp(path)) else {
            return false;
        };
        self.0[index].1 = true;
        true
    }
    // Ordinary cache workers retain their consuming interface. Their supplied
    // path is still validated against exactly the admitted source universe.
    pub(super) fn insert(&mut self, path: PathBuf) -> bool {
        let Ok(index) = self.0.binary_search_by(|row| row.0.cmp(&path)) else {
            return false;
        };
        !std::mem::replace(&mut self.0[index].1, true)
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = &PathBuf> {
        self.0.iter().filter(|row| row.1).map(|row| &row.0)
    }
}

const MAX_READ_BYTES: usize = 64 * 1024 * 1024;
const MAX_READ_BUFFERS: usize = 1024;

#[derive(Clone)]
struct ReadSpan {
    source: Range<u64>,
    destination: Range<usize>,
}

#[derive(Clone)]
struct ReadShard {
    admitted: Arc<AdmittedFile>,
    spans: Vec<ReadSpan>,
}

/// An admitted batch of encoded ranges over exact retained source owners.
///
/// Full reads place tensor bytes consecutively in the requested order. Recipe
/// projections select and reorder the same source ranges into the destination.
/// File reads are ordered by shard and offset and combine adjacent ranges even
/// across disjoint destinations. Memory reads borrow the existing immutable
/// tensor payload directly. Neither path stages another payload buffer or pins
/// a shard-cache entry.
///
/// Cloning copies read/tensor metadata and shares the original admitted files
/// and diagnostics or the original immutable memory tensor. The latter retains
/// already registered source storage; cloning does not create source authority.
/// Each clone fills an independent destination through the same checked worker.
/// It never re-admits a changed source or retains an earlier read's output.
#[derive(Clone)]
pub struct EncodedReadBatch {
    tensors: Vec<TensorMetadata>,
    shards: Vec<(PathBuf, ReadShard)>,
    byte_len: usize,
    memory: Vec<memory::ReadMemory>,
    telemetry: Option<Arc<SafetensorsReadTelemetry>>,
    cache: Option<Arc<Mutex<CacheState>>>,
}

impl EncodedReadBatch {
    /// Metadata in destination order, including repeated source occurrences.
    pub fn tensors(&self) -> &[TensorMetadata] {
        &self.tensors
    }

    /// Exact size required of the destination buffer.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Reads directly into a caller-owned allocation. On error, the allocation
    /// may be partially written and must not be published as a valid tensor.
    /// No payload hashing or full-shard staging is performed.
    pub fn read_into(self, destination: &mut [u8]) -> Result<(), StoreError> {
        self.read_into_with_hook(destination, || {})
    }

    /// Fills several independent final allocations in shard/offset order.
    /// Adjacent file ranges use vectored reads even when their destinations
    /// belong to different tensors or occur in a different memory order.
    /// No payload staging is allocated. All destination lengths are checked
    /// before the first read; callers must discard every output on failure.
    pub fn read_many_into(
        batches: Vec<Self>,
        destinations: &mut [&mut [u8]],
    ) -> Result<(), StoreError> {
        Self::read_many_with_hook(batches, destinations, || {})
    }

    fn read_into_with_hook(
        self,
        destination: &mut [u8],
        after_shard: impl FnMut(),
    ) -> Result<(), StoreError> {
        Self::read_many_with_hook(vec![self], &mut [destination], after_shard)
    }

    fn read_many_with_hook(
        batches: Vec<Self>,
        destinations: &mut [&mut [u8]],
        after_shard: impl FnMut(),
    ) -> Result<(), StoreError> {
        borrowed::read_many(batches.iter(), destinations, after_shard)
            .map_err(|failure| failure.into_ordinary(&batches))
    }
}

struct DestinationSpan<'a> {
    file: Range<u64>,
    bytes: &'a mut [u8],
}

#[cfg(test)]
fn read_destination_spans(
    reader: &mut (impl Read + Seek),
    spans: Vec<DestinationSpan<'_>>,
    telemetry: &SafetensorsReadTelemetry,
) -> io::Result<()> {
    let mut buffers = Vec::with_capacity(MAX_READ_BUFFERS.min(spans.len()));
    read_destination_spans_with_buffers(reader, spans, telemetry, &mut buffers)
}

fn read_destination_spans_with_buffers<'a>(
    reader: &mut (impl Read + Seek),
    spans: impl IntoIterator<Item = DestinationSpan<'a>>,
    telemetry: &SafetensorsReadTelemetry,
    buffers: &mut Vec<IoSliceMut<'a>>,
) -> io::Result<()> {
    let mut spans = spans.into_iter().peekable();
    let mut position = None;
    while let Some(first) = spans.next() {
        let start = first.file.start;
        let mut end = first.file.end;
        let mut bytes = first.bytes.len();
        buffers.push(IoSliceMut::new(first.bytes));
        while let Some(next) = spans.peek() {
            if next.file.start != end
                || buffers.len() == MAX_READ_BUFFERS
                || bytes + next.bytes.len() > MAX_READ_BYTES
            {
                break;
            }
            let next = spans.next().expect("peeked span");
            bytes += next.bytes.len();
            end = next.file.end;
            buffers.push(IoSliceMut::new(next.bytes));
        }
        if position != Some(start) {
            reader.seek(SeekFrom::Start(start))?;
        }
        read_exact_vectored(reader, buffers, telemetry)?;
        position = Some(end);
        buffers.clear();
    }
    Ok(())
}

fn read_exact_vectored(
    reader: &mut impl Read,
    buffers: &mut [IoSliceMut<'_>],
    telemetry: &SafetensorsReadTelemetry,
) -> io::Result<()> {
    let mut remaining = buffers.iter().map(|buffer| buffer.len()).sum::<usize>();
    let mut buffers = buffers;
    while remaining != 0 {
        let count = match reader.read_vectored(buffers) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(count) if count <= remaining => count,
            Ok(_) => return Err(io::ErrorKind::InvalidData.into()),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        telemetry.physical_reads.fetch_add(1, Ordering::Relaxed);
        telemetry
            .physical_read_bytes
            .fetch_add(count as u64, Ordering::Relaxed);
        remaining -= count;
        IoSliceMut::advance_slices(&mut buffers, count);
    }
    Ok(())
}

pub(super) fn provenance(metadata: &TensorMetadata) -> TensorSourceProvenance {
    TensorSourceProvenance {
        catalog_key: metadata.name.clone(),
        physical_tensor: metadata.name.clone(),
        output: metadata.name.clone(),
        backing_shard: metadata.backing_shard.clone(),
        source_encoding: crate::SourceTensorEncoding::Safetensors(metadata.stored_dtype.clone()),
    }
}

pub(super) fn prepare(
    store: &SafetensorsWeightStore,
    keys: &[String],
) -> Result<EncodedReadBatch, StoreError> {
    let mut groups = BTreeMap::<PathBuf, Vec<(usize, &str)>>::new();
    for (index, key) in keys.iter().enumerate() {
        let entry = store
            .catalog
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.clone() })?;
        groups
            .entry(entry.shard.clone())
            .or_default()
            .push((index, key));
    }
    let mut entries = Vec::with_capacity(keys.len());
    for (path, group) in groups {
        let admission = store.shards.admission(&path);
        let header = admission.header(&path)?;
        store.lock_cache()?.touched.insert(path.clone());
        for (index, key) in group {
            let info = header
                .metadata
                .info(key)
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
            let metadata = header.tensors[key].clone();
            let start = header
                .payload_offset
                .checked_add(info.data_offsets.0)
                .ok_or_else(|| StoreError::Overflow {
                    context: "bulk tensor offset".into(),
                })?;
            entries.push((index, metadata, start, Arc::clone(&admission.file)));
        }
        // Header admission is independent of the payload-cache window.
    }
    entries.sort_unstable_by_key(|entry| entry.0);
    let mut shards = BTreeMap::<PathBuf, ReadShard>::new();
    let mut batch = EncodedReadBatch {
        tensors: Vec::with_capacity(keys.len()),
        shards: Vec::new(),
        byte_len: 0,
        memory: Vec::new(),
        telemetry: Some(Arc::clone(&store.read_telemetry)),
        cache: Some(Arc::clone(&store.cache)),
    };
    for (_, metadata, start, admitted) in entries {
        let length =
            usize::try_from(metadata.encoded_byte_len).map_err(|_| StoreError::Overflow {
                context: "bulk tensor length".into(),
            })?;
        let end = batch
            .byte_len
            .checked_add(length)
            .ok_or_else(|| StoreError::Overflow {
                context: "bulk output length".into(),
            })?;
        let file_start = u64::try_from(start).map_err(|_| StoreError::Overflow {
            context: "bulk file offset".into(),
        })?;
        let file_end = file_start
            .checked_add(metadata.encoded_byte_len)
            .ok_or_else(|| StoreError::Overflow {
                context: "bulk file end".into(),
            })?;
        let path = metadata
            .backing_shard
            .clone()
            .expect("SafeTensors source has a shard");
        shards
            .entry(path)
            .or_insert_with(|| ReadShard {
                admitted,
                spans: Vec::new(),
            })
            .spans
            .push(ReadSpan {
                source: file_start..file_end,
                destination: batch.byte_len..end,
            });
        batch.byte_len = end;
        batch.tensors.push(metadata);
    }
    for shard in shards.values_mut() {
        shard.spans.sort_unstable_by_key(|span| span.source.start);
        let mut merged = Vec::<ReadSpan>::with_capacity(shard.spans.len());
        for span in std::mem::take(&mut shard.spans) {
            if let Some(previous) = merged.last_mut() {
                if previous.source.end == span.source.start
                    && previous.destination.end == span.destination.start
                {
                    previous.source.end = span.source.end;
                    previous.destination.end = span.destination.end;
                    continue;
                }
            }
            merged.push(span);
        }
        shard.spans = merged;
    }
    batch.shards = shards.into_iter().collect();
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, TensorView};

    fn fixture() -> (tempfile::TempDir, PreparedCheckpointSource) {
        let directory = tempfile::tempdir().unwrap();
        let mut weight_map = BTreeMap::new();
        for (filename, values) in [
            ("a.safetensors", [("a", 1u8), ("b", 2)]),
            ("b.safetensors", [("c", 3), ("d", 4)]),
        ] {
            let data = values.map(|(name, value)| (name, vec![value; 8]));
            serialize_to_file(
                data.iter().map(|(name, bytes)| {
                    (
                        *name,
                        TensorView::new(Dtype::U8, vec![2, 4], bytes).unwrap(),
                    )
                }),
                None,
                &directory.path().join(filename),
            )
            .unwrap();
            for (name, _) in values {
                weight_map.insert(name, filename);
            }
        }
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
        let admitted =
            crate::safetensors::SafetensorsMetadataCatalog::discover(directory.path()).unwrap();
        let source = PreparedCheckpointSource::open_admitted_safetensors(
            admitted.admitted_shards(),
            admitted.tensors().clone(),
            1,
        )
        .unwrap();
        (directory, source)
    }

    #[test]
    fn detached_read_subset_uses_only_selected_files_and_retains_failure_custody() {
        use crate::recipe::{DerivedWeightRecipe as Recipe, EncodedRecipeRead};
        let (directory, source) = fixture();
        let reads = ["a", "c", "b"].map(|key| {
            Recipe::source(key, TensorSelection::Full)
                .prepare_encoded_read(&source)
                .unwrap()
                .unwrap()
        });
        let custody = Arc::new(());
        let alive = Arc::downgrade(&custody);
        let detached = EncodedRecipeRead::prepare_detached(reads.iter())
            .unwrap()
            .construct(custody)
            .unwrap();
        drop(reads);
        drop(source);
        assert!(detached.slice(2..4).is_none());
        assert!(detached.slice(2..1).is_none());
        assert!(detached
            .slice(3..3)
            .unwrap()
            .read_many_into(&mut [])
            .is_ok());
        std::fs::remove_file(directory.path().join("a.safetensors")).unwrap();
        let selected = detached.slice(1..2).unwrap();
        assert!(selected.read_layout().unwrap().required_bytes() > 0);
        let mut output = [99u8; 8];
        selected.read_many_into(&mut [&mut output]).unwrap();
        assert_eq!(output, [3; 8]);
        assert_eq!(detached.physical_read_bytes(0), Some(8));
        let mut short = [77u8; 7];
        let error = selected.read_many_into(&mut [&mut short]).unwrap_err();
        assert!(matches!(
            error.cause().cause,
            EncodedReadFailureCause::DestinationLengths
        ));
        assert_eq!(short, [77; 7]);
        assert_eq!(detached.physical_read_bytes(0), Some(8));
        drop(error);
        std::fs::remove_file(directory.path().join("b.safetensors")).unwrap();
        let error = selected.read_many_into(&mut [&mut output]).unwrap_err();
        assert_eq!(error.cause().batch, Some(0));
        drop(selected);
        drop(detached);
        assert!(alive.upgrade().is_some());
        drop(error);
        assert!(alive.upgrade().is_none());
    }

    #[test]
    fn detached_reads_release_ordinary_sources_and_keep_failure_custody() {
        use crate::recipe::{DerivedWeightRecipe as Recipe, EncodedRecipeRead};
        let (directory, source) = fixture();
        let independent = SafetensorsWeightStore::open(directory.path()).unwrap();
        let reads = [
            Recipe::Stack {
                axis: 0,
                inputs: ["b", "a"]
                    .map(|key| Recipe::source(key, TensorSelection::Full))
                    .into(),
            }
            .prepare_encoded_read(&source)
            .unwrap()
            .unwrap(),
            Recipe::source("a", TensorSelection::Full)
                .prepare_encoded_read(&independent)
                .unwrap()
                .unwrap(),
        ];
        let ordinary_cache = Arc::downgrade(reads[0].admitted_batch().cache.as_ref().unwrap());
        let ordinary_file = Arc::downgrade(&reads[0].admitted_batch().shards[0].1.admitted);
        let ordinary_telemetry =
            Arc::downgrade(reads[0].admitted_batch().telemetry.as_ref().unwrap());
        let custody = Arc::new(());
        let alive = Arc::downgrade(&custody);
        let plan = EncodedRecipeRead::prepare_detached(reads.iter()).unwrap();
        assert!(plan.required_bytes::<Arc<()>>().unwrap() > 0);
        let quoted_read = plan.read_layout::<Arc<()>>().unwrap().required_bytes();
        let detached = plan.construct(custody).unwrap();
        assert!(quoted_read >= detached.read_layout().unwrap().required_bytes());
        assert_eq!(detached.output(0), Some(reads[0].output()));
        assert!(detached.matches_read(0, &reads[0]));
        assert!(detached.matches_read(1, &reads[1]));
        assert!(!detached.matches_read(0, &reads[1]));
        assert!(!detached.matches_read(2, &reads[0]));
        // Equal immutable provenance can come from an independent source-local
        // telemetry owner. Matching it must not coalesce detached diagnostics.
        let same_admission = Recipe::source("a", TensorSelection::Full)
            .prepare_encoded_read(&source)
            .unwrap()
            .unwrap();
        assert!(!Arc::ptr_eq(
            same_admission.admitted_batch().telemetry.as_ref().unwrap(),
            reads[1].admitted_batch().telemetry.as_ref().unwrap(),
        ));
        assert!(detached.matches_read(1, &same_admission));
        drop((same_admission, reads, source, independent));
        assert!(ordinary_cache.upgrade().is_none());
        assert!(ordinary_file.upgrade().is_none());
        assert!(ordinary_telemetry.upgrade().is_none());
        let mut first = [0; 16];
        let mut second = [0; 8];
        detached
            .read_many_into(&mut [&mut first, &mut second])
            .unwrap();
        assert_eq!(&first[..8], &[2; 8]);
        assert_eq!(&first[8..], &[1; 8]);
        assert_eq!(second, [1; 8]);
        assert_eq!(detached.physical_read_bytes(0), Some(16));
        assert_eq!(detached.physical_read_bytes(1), Some(8));
        let mut marked = 0;
        detached.visit_payload_paths(|_, _| marked += 1);
        assert_eq!(marked, 2);
        // Re-admit identical tensor metadata and encoded ranges after changing
        // only the real file version. Geometry alone cannot authenticate reuse.
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(directory.path().join("a.safetensors"))
            .unwrap();
        let modified =
            file.metadata().unwrap().modified().unwrap() + std::time::Duration::from_secs(60);
        file.set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        drop(file);
        let readmitted_source = SafetensorsWeightStore::open(directory.path()).unwrap();
        let readmitted = Recipe::source("a", TensorSelection::Full)
            .prepare_encoded_read(&readmitted_source)
            .unwrap()
            .unwrap();
        assert_eq!(detached.output(1), Some(readmitted.output()));
        assert_eq!(detached.sources(1), Some(readmitted.sources()));
        assert!(!detached.matches_read(1, &readmitted));
        assert_eq!(detached.physical_read_bytes(0), Some(16));
        assert_eq!(detached.physical_read_bytes(1), Some(8));
        assert_eq!(
            readmitted_source
                .source_diagnostics()
                .unwrap()
                .physical_reads,
            0
        );
        drop((readmitted, readmitted_source));
        std::fs::OpenOptions::new()
            .write(true)
            .open(directory.path().join("a.safetensors"))
            .unwrap()
            .set_len(1)
            .unwrap();
        first.fill(55);
        second.fill(55);
        let failure = detached
            .read_many_into(&mut [&mut first, &mut second])
            .unwrap_err();
        assert!(matches!(
            failure.cause().cause,
            EncodedReadFailureCause::Changed
        ));
        assert_eq!(first, [55; 16]);
        assert_eq!(second, [55; 8]);
        drop(detached);
        assert!(alive.upgrade().is_some());
        drop(failure);
        assert!(alive.upgrade().is_none());
    }

    #[test]
    fn borrowed_reads_reuse_admission_and_preserve_failed_shard_prefix() {
        let (directory, source) = fixture();
        let batches = [["b", "c"], ["d", "a"]].map(|keys| {
            source
                .prepare_encoded_read(&keys.map(String::from))
                .unwrap()
                .unwrap()
        });
        let layout = EncodedReadLayout::inspect(batches.iter()).unwrap();
        assert!(layout.required_bytes() > 0);
        assert!(batches[0].clone_storage_bytes().unwrap() > 0);
        let metadata = batches[0].tensors().as_ptr();
        let mut first = [99; 16];
        let mut short = [99; 15];
        let failure = EncodedReadBatch::read_many_borrowed_into(
            batches.iter(),
            &mut [&mut first, &mut short],
        )
        .unwrap_err();
        assert!(matches!(
            failure.cause,
            EncodedReadFailureCause::DestinationLengths
        ));
        assert_eq!(first, [99; 16]);
        assert_eq!(short, [99; 15]);
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
        let mut second = [0; 16];
        for _ in 0..2 {
            EncodedReadBatch::read_many_borrowed_into(
                batches.iter(),
                &mut [&mut first, &mut second],
            )
            .unwrap();
            assert_eq!(&first[..8], &[2; 8]);
            assert_eq!(&first[8..], &[3; 8]);
            assert_eq!(&second[..8], &[4; 8]);
            assert_eq!(&second[8..], &[1; 8]);
        }
        assert_eq!(batches[0].tensors().as_ptr(), metadata);
        let diagnostics = source.source_diagnostics().unwrap();
        assert_eq!(diagnostics.payload_shard_paths.len(), 2);
        assert_eq!(diagnostics.physical_read_bytes, 64);
        assert_eq!(diagnostics.currently_cached_shards, 0);
        drop(source);
        first.fill(77);
        second.fill(77);
        let mut changed = false;
        let failure = borrowed::read_many(batches.iter(), &mut [&mut first, &mut second], || {
            if !changed {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(directory.path().join("b.safetensors"))
                    .unwrap()
                    .set_len(1)
                    .unwrap();
                changed = true;
            }
        })
        .unwrap_err();
        assert!(matches!(failure.cause, EncodedReadFailureCause::Changed));
        assert_eq!(
            (failure.batch, failure.shard, failure.completed_shards),
            (Some(0), Some(1), 1)
        );
        assert_eq!(&first[..8], &[2; 8]);
        assert_eq!(&first[8..], &[77; 8]);
        assert_eq!(&second[..8], &[77; 8]);
        assert_eq!(&second[8..], &[1; 8]);
        // The original admitted source is still retained, and the next read
        // refuses the changed second shard through the ordinary error adapter.
        assert!(matches!(
            batches[0].clone().read_into(&mut first),
            Err(StoreError::AdmittedFileChanged { .. })
        ));
    }

    #[test]
    fn vectored_reads_scatter_into_reversed_and_separate_allocations() {
        let (_directory, source) = fixture();
        let mut first = [0; 16];
        let mut second = [0; 16];
        let batches = [["b", "c"], ["d", "a"]].map(|keys| {
            source
                .prepare_encoded_read(&keys.map(String::from))
                .unwrap()
                .unwrap()
        });
        EncodedReadBatch::read_many_into(batches.into(), &mut [&mut first, &mut second]).unwrap();
        assert_eq!(first, [vec![2; 8], vec![3; 8]].concat().as_slice());
        assert_eq!(second, [vec![4; 8], vec![1; 8]].concat().as_slice());
        let diagnostics = source.source_diagnostics().unwrap();
        #[cfg(unix)]
        assert_eq!(
            diagnostics.physical_reads, 2,
            "one read per shard across both allocations"
        );
        assert_eq!(diagnostics.physical_read_bytes, 32);
        assert_eq!(diagnostics.currently_cached_shards, 0);
    }

    #[test]
    fn every_destination_is_checked_before_any_output_is_written() {
        let (_directory, source) = fixture();
        for wrong_count in [false, true] {
            let batches =
                ["a", "b"].map(|key| source.prepare_encoded_read(&[key.into()]).unwrap().unwrap());
            let mut first = [99; 8];
            let mut second = [99; 7];
            let mut outputs = vec![first.as_mut_slice()];
            if !wrong_count {
                outputs.push(second.as_mut_slice());
            }
            assert!(EncodedReadBatch::read_many_into(batches.into(), &mut outputs).is_err());
            assert_eq!(first, [99; 8]);
            assert_eq!(second, [99; 7]);
        }
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    }

    #[test]
    fn grouped_reads_retain_each_sources_admission_and_telemetry() {
        let (directory, a) = fixture();
        let b = SafetensorsWeightStore::open(directory.path()).unwrap();
        let memory =
            MemoryWeightStore::from_safetensors([("a".into(), Dtype::U8, vec![2, 4], vec![37; 8])])
                .unwrap();
        let sources = [&a as &dyn CheckpointSource, &b, &memory];
        let batches =
            sources.map(|source| source.prepare_encoded_read(&["a".into()]).unwrap().unwrap());
        let mut first = [0; 8];
        let mut second = [0; 8];
        let mut third = [0; 8];
        EncodedReadBatch::read_many_into(
            batches.into(),
            &mut [&mut first, &mut second, &mut third],
        )
        .unwrap();
        assert_eq!(first, [1; 8]);
        assert_eq!(second, [1; 8]);
        assert_eq!(third, [37; 8]);
        for source in &sources[..2] {
            assert_eq!(source.source_diagnostics().unwrap().physical_read_bytes, 8);
        }
        assert_eq!(memory.source_diagnostics().unwrap().physical_read_bytes, 0);
    }

    struct ShortReader {
        cursor: io::Cursor<Vec<u8>>,
        limit: usize,
        interrupt: bool,
        requests: Vec<(usize, usize)>,
        seeks: usize,
    }

    impl Read for ShortReader {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.read_vectored(&mut [IoSliceMut::new(bytes)])
        }
        fn read_vectored(&mut self, buffers: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
            if std::mem::take(&mut self.interrupt) {
                return Err(io::ErrorKind::Interrupted.into());
            }
            self.requests.push((
                buffers.len(),
                buffers.iter().map(|buffer| buffer.len()).sum(),
            ));
            let mut total = 0;
            for buffer in buffers {
                let length = buffer.len().min(self.limit - total);
                total += self.cursor.read(&mut buffer[..length])?;
                if total == self.limit {
                    break;
                }
            }
            Ok(total)
        }
    }
    impl Seek for ShortReader {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.seeks += 1;
            self.cursor.seek(position)
        }
    }
    fn reader(bytes: Vec<u8>, limit: usize) -> ShortReader {
        ShortReader {
            cursor: io::Cursor::new(bytes),
            limit,
            interrupt: false,
            requests: Vec::new(),
            seeks: 0,
        }
    }

    #[test]
    fn short_reads_and_interrupts_preserve_scatter_order_and_report_eof() {
        let telemetry = SafetensorsReadTelemetry::default();
        let mut input = reader((0..10).collect(), 3);
        input.interrupt = true;
        let mut first = [0; 4];
        let mut second = [0; 6];
        read_exact_vectored(
            &mut input,
            &mut [IoSliceMut::new(&mut first), IoSliceMut::new(&mut second)],
            &telemetry,
        )
        .unwrap();
        assert_eq!(first, [0, 1, 2, 3]);
        assert_eq!(second, [4, 5, 6, 7, 8, 9]);
        assert_eq!(telemetry.physical_reads.load(Ordering::Relaxed), 4);
        assert_eq!(telemetry.physical_read_bytes.load(Ordering::Relaxed), 10);
        let error = read_exact_vectored(&mut input, &mut [IoSliceMut::new(&mut first)], &telemetry)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn vectored_reads_support_the_scalar_read_fallback() {
        struct Scalar(io::Cursor<Vec<u8>>);
        impl Read for Scalar {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                self.0.read(bytes)
            }
        }
        let mut input = Scalar(io::Cursor::new(vec![1, 2, 3, 4]));
        let mut first = [0; 2];
        let mut second = [0; 2];
        let telemetry = SafetensorsReadTelemetry::default();
        read_exact_vectored(
            &mut input,
            &mut [IoSliceMut::new(&mut first), IoSliceMut::new(&mut second)],
            &telemetry,
        )
        .unwrap();
        assert_eq!(first, [1, 2]);
        assert_eq!(second, [3, 4]);
        assert_eq!(telemetry.physical_reads.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn bounded_vectors_continue_without_seeking_and_skip_unselected_gaps() {
        let count = MAX_READ_BUFFERS + 1;
        let mut input = reader(vec![7; count], usize::MAX);
        let mut output = vec![0; count];
        let spans = output
            .chunks_mut(1)
            .enumerate()
            .map(|(index, bytes)| DestinationSpan {
                file: index as u64..index as u64 + 1,
                bytes,
            })
            .collect();
        let telemetry = SafetensorsReadTelemetry::default();
        read_destination_spans(&mut input, spans, &telemetry).unwrap();
        assert_eq!(output, vec![7; count]);
        assert_eq!(
            input.requests,
            [(MAX_READ_BUFFERS, MAX_READ_BUFFERS), (1, 1)]
        );
        assert_eq!(input.seeks, 1);

        let mut input = reader(vec![10, 11, 12, 13, 14, 15], usize::MAX);
        let mut output = [0; 3];
        let spans = output
            .chunks_mut(1)
            .zip([0..1, 3..4, 3..4])
            .map(|(bytes, file)| DestinationSpan { file, bytes })
            .collect();
        read_destination_spans(&mut input, spans, &telemetry).unwrap();
        assert_eq!(output, [10, 13, 13]);
        assert_eq!(
            input.seeks, 3,
            "gaps and repeated sources need independent positions"
        );
        assert_eq!(input.requests, [(1, 1); 3]);
    }

    #[test]
    fn batches_coalesce_without_payload_staging_or_pinned_shards() {
        let (_directory, source) = fixture();
        let batch = source
            .prepare_encoded_read(&["c".into(), "d".into(), "a".into(), "b".into()])
            .unwrap()
            .unwrap();
        assert_eq!(batch.byte_len(), 32);
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
        let mut output = vec![0; batch.byte_len()];
        batch.read_into(&mut output).unwrap();
        assert_eq!(
            output,
            [vec![3; 8], vec![4; 8], vec![1; 8], vec![2; 8]].concat()
        );
        let diagnostics = source.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 2);
        assert_eq!(diagnostics.physical_read_bytes, 32);
        assert_eq!(diagnostics.payload_shard_paths.len(), 2);
        assert_eq!(diagnostics.currently_cached_shards, 0);
    }

    #[test]
    fn reordered_and_repeated_sources_have_exact_destination_order() {
        let (_directory, source) = fixture();
        let batch = source
            .prepare_encoded_read(&["d".into(), "a".into(), "c".into(), "a".into()])
            .unwrap()
            .unwrap();
        let mut output = vec![0; batch.byte_len()];
        batch.read_into(&mut output).unwrap();
        assert_eq!(
            output,
            [vec![4; 8], vec![1; 8], vec![3; 8], vec![1; 8]].concat()
        );
        assert_eq!(source.source_diagnostics().unwrap().physical_read_bytes, 32);
    }

    #[test]
    fn cloned_batches_repeat_exact_reads_into_independent_destinations() {
        let (_directory, source) = fixture();
        let batch = source
            .prepare_encoded_read(&["d".into(), "a".into(), "c".into(), "a".into()])
            .unwrap()
            .unwrap();
        let before = source.source_diagnostics().unwrap();
        let first_read = batch.clone();
        let second_read = first_read.clone();
        assert_eq!(first_read.tensors(), batch.tensors());
        assert_eq!(first_read.byte_len(), 32);
        assert_eq!(source.source_diagnostics().unwrap(), before);

        let expected = [vec![4; 8], vec![1; 8], vec![3; 8], vec![1; 8]].concat();
        let mut first = vec![99; batch.byte_len()];
        let mut second = vec![88; batch.byte_len()];
        EncodedReadBatch::read_many_into(
            vec![first_read, second_read],
            &mut [&mut first, &mut second],
        )
        .unwrap();
        assert_eq!(first, expected);
        assert_eq!(second, expected);
        first.fill(77);
        assert_eq!(second, expected, "outputs must not share mutable storage");

        // Reusing the plan performs a fresh read, not a cached output copy.
        batch.clone().read_into(&mut first).unwrap();
        assert_eq!(first, expected);
        let diagnostics = source.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_read_bytes, 96);
        assert_eq!(diagnostics.currently_cached_shards, 0);
        assert_eq!(diagnostics.cache_hits, before.cache_hits);
        assert_eq!(diagnostics.cache_misses, before.cache_misses);

        // The plan owns its original admission, rather than borrowing a store.
        drop(source);
        second.fill(66);
        batch.read_into(&mut second).unwrap();
        assert_eq!(second, expected);
    }

    #[test]
    fn cloned_batches_do_not_keep_payloads_or_cache_entries_leased() {
        let (directory, source) = fixture();
        let store =
            SafetensorsWeightStore::open_with_max_cached_shards(directory.path(), 1).unwrap();
        let batch = store.prepare_encoded_read(&["a".into()]).unwrap().unwrap();
        let clone = batch.clone();
        let request = |key: &str| TensorReadRequest {
            key: key.into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        };
        let lease = store.acquire(request("a")).unwrap();
        let payload = Arc::downgrade(&lease.bytes);
        let shard = Arc::downgrade(&lease.shard);
        drop(lease);
        assert!(
            payload.upgrade().is_none(),
            "read plans must not pin cached tensor bytes"
        );
        let other = store.acquire(request("c")).unwrap();
        assert!(
            shard.upgrade().is_none(),
            "read plans must not block shard-cache eviction"
        );
        assert_eq!(store.diagnostics().unwrap().evictions, 1);
        drop((other, store, source));
        let mut first = [0; 8];
        let mut second = [0; 8];
        EncodedReadBatch::read_many_into(vec![batch, clone], &mut [&mut first, &mut second])
            .unwrap();
        assert_eq!(first, [1; 8]);
        assert_eq!(second, [1; 8]);
        assert!(payload.upgrade().is_none());
    }

    #[test]
    fn cloning_after_source_removal_does_not_reopen_or_replace_admission() {
        let (directory, source) = fixture();
        let batch = source.prepare_encoded_read(&["a".into()]).unwrap().unwrap();
        let before = source.source_diagnostics().unwrap();
        let path = directory
            .path()
            .join("a.safetensors")
            .canonicalize()
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        let clone = batch.clone();
        assert_eq!(clone.tensors(), batch.tensors());
        assert_eq!(source.source_diagnostics().unwrap(), before);
        let mut output = [99; 8];
        let result = clone.read_into(&mut output);
        // Opening the retained path still uses fs_error's original NotFound
        // translation; cloning does not turn removal into fresh admission.
        assert!(
            matches!(&result, Err(StoreError::MissingShard { path: failed }) if failed == &path),
            "expected the original missing-shard error, got {result:?}"
        );
        assert_eq!(output, [99; 8]);
        assert_eq!(source.source_diagnostics().unwrap(), before);
    }

    #[test]
    fn cloned_batches_keep_the_original_contract_after_success_and_truncation() {
        let (directory, source) = fixture();
        let batch = source.prepare_encoded_read(&["a".into()]).unwrap().unwrap();
        let before_change = batch.clone();
        let mut successful = [0; 8];
        batch.clone().read_into(&mut successful).unwrap();
        assert_eq!(successful, [1; 8]);
        let path = directory
            .path()
            .join("a.safetensors")
            .canonicalize()
            .unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(1)
            .unwrap();
        let after_change = batch.clone();
        for read in [before_change, after_change, batch] {
            let mut output = [99; 8];
            assert!(
                matches!(read.read_into(&mut output), Err(StoreError::AdmittedFileChanged { path: failed }) if failed == path)
            );
            assert_eq!(output, [99; 8]);
        }
        assert_eq!(source.source_diagnostics().unwrap().physical_read_bytes, 8);
    }

    #[cfg(unix)]
    #[test]
    fn cloned_batches_reject_same_length_source_replacement() {
        let (directory, source) = fixture();
        let batch = source.prepare_encoded_read(&["a".into()]).unwrap().unwrap();
        let clone = batch.clone();
        let path = directory
            .path()
            .join("a.safetensors")
            .canonicalize()
            .unwrap();
        let replacement = directory.path().join("replacement.safetensors");
        // Allocate the replacement while the admitted inode still exists. Its
        // exact same bytes and length must not satisfy the old file identity.
        std::fs::copy(&path, &replacement).unwrap();
        std::fs::rename(replacement, &path).unwrap();
        for read in [clone, batch] {
            let mut output = [99; 8];
            assert!(
                matches!(read.read_into(&mut output), Err(StoreError::AdmittedFileChanged { path: failed }) if failed == path)
            );
            assert_eq!(output, [99; 8]);
        }
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    }

    #[test]
    fn a_cloned_read_failure_preserves_later_validation_and_partial_output_rules() {
        let (directory, source) = fixture();
        let batch = source
            .prepare_encoded_read(&["a".into(), "c".into()])
            .unwrap()
            .unwrap();
        let path = directory
            .path()
            .join("a.safetensors")
            .canonicalize()
            .unwrap();
        let mut output = [99; 16];
        let result = batch.clone().read_into_with_hook(&mut output, || {
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(1)
                .unwrap();
        });
        assert!(
            matches!(result, Err(StoreError::AdmittedFileChanged { path: failed }) if failed == path)
        );
        assert_eq!(&output[..8], &[1; 8]);
        assert_eq!(&output[8..], &[99; 8]);
        let mut retry = [88; 16];
        assert!(
            matches!(batch.read_into(&mut retry), Err(StoreError::AdmittedFileChanged { path: failed }) if failed == path)
        );
        assert_eq!(retry, [88; 16]);
        assert_eq!(source.source_diagnostics().unwrap().physical_read_bytes, 8);
    }

    #[test]
    fn cloned_recipe_reads_preserve_inference_order_and_source_validation() {
        use crate::recipe::{DerivedWeightRecipe as Recipe, EncodedRecipeRead};
        let (directory, source) = fixture();
        let recipe = Recipe::Reshape {
            input: Box::new(Recipe::Stack {
                axis: 0,
                inputs: ["b", "a", "b"]
                    .map(|key| Recipe::source(key, TensorSelection::Full))
                    .into(),
            }),
            shape: vec![6, 4],
        };
        let read = recipe.prepare_encoded_read(&source).unwrap().unwrap();
        let before = source.source_diagnostics().unwrap();
        let clone = read.clone();
        assert_eq!(clone.output(), read.output());
        assert_eq!(clone.sources(), read.sources());
        assert_eq!(clone.output().shape, [6, 4]);
        assert_eq!(
            clone
                .sources()
                .iter()
                .map(|source| source.name.as_str())
                .collect::<Vec<_>>(),
            ["b", "a", "b"]
        );
        assert_eq!(source.source_diagnostics().unwrap(), before);
        let expected = [vec![2; 8], vec![1; 8], vec![2; 8]].concat();
        let mut first = [0; 24];
        let mut second = [0; 24];
        EncodedRecipeRead::read_many_into(
            vec![clone, read.clone()],
            &mut [&mut first, &mut second],
        )
        .unwrap();
        assert_eq!(first.as_slice(), expected);
        assert_eq!(second.as_slice(), expected);
        first.fill(77);
        read.clone().read_into(&mut first).unwrap();
        assert_eq!(first.as_slice(), expected);
        assert_eq!(second.as_slice(), expected);
        assert_eq!(source.source_diagnostics().unwrap().physical_read_bytes, 72);
        assert_eq!(
            source.source_diagnostics().unwrap().currently_cached_shards,
            0
        );

        let path = directory
            .path()
            .join("a.safetensors")
            .canonicalize()
            .unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(1)
            .unwrap();
        let clone = read.clone();
        drop(source);
        for read in [clone, read] {
            let mut output = [99; 24];
            assert!(
                matches!(read.read_into(&mut output), Err(StoreError::AdmittedFileChanged { path: failed }) if failed == path)
            );
            assert_eq!(output, [99; 24]);
        }
    }

    #[test]
    fn wrong_destination_and_unauthorized_keys_do_not_read_payloads() {
        let (_directory, source) = fixture();
        let source: SharedCheckpointSource = Arc::new(source);
        let restricted = RestrictedCheckpointSource::including(
            Arc::clone(&source),
            "only a",
            BTreeSet::from(["a".into()]),
        )
        .unwrap();
        assert!(restricted
            .prepare_encoded_read(&["a".into(), "b".into()])
            .is_err());
        let batch = restricted
            .prepare_encoded_read(&["a".into()])
            .unwrap()
            .unwrap();
        assert!(batch.read_into(&mut [0; 7]).is_err());
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    }

    #[test]
    fn changed_files_are_rejected_before_and_during_read() {
        for during_read in [false, true] {
            let (directory, source) = fixture();
            let batch = source.prepare_encoded_read(&["a".into()]).unwrap().unwrap();
            let change = || {
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .open(directory.path().join("a.safetensors"))
                    .unwrap();
                file.set_len(1).unwrap();
            };
            let mut output = [0; 8];
            let result = if during_read {
                batch.read_into_with_hook(&mut output, change)
            } else {
                change();
                batch.read_into(&mut output)
            };
            assert!(matches!(
                result,
                Err(StoreError::AdmittedFileChanged { .. })
            ));
        }
    }

    #[test]
    fn recipe_read_infers_nested_expert_layout_and_falls_back_for_transforms() {
        use crate::recipe::DerivedWeightRecipe as Recipe;
        let (_directory, source) = fixture();
        struct BatchOnly(PreparedCheckpointSource);
        impl CheckpointSource for BatchOnly {
            fn source_keys(&self) -> Vec<String> {
                self.0.source_keys()
            }
            fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
                panic!("direct recipe inference must reuse admitted batch metadata")
            }
            fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
                panic!("direct reads must not allocate per-tensor payload leases")
            }
            fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
                self.0.source_diagnostics()
            }
            fn prepare_encoded_read(
                &self,
                keys: &[String],
            ) -> Result<Option<EncodedReadBatch>, StoreError> {
                self.0.prepare_encoded_read(keys)
            }
        }
        let source = BatchOnly(source);
        let recipe = Recipe::Stack {
            axis: 0,
            inputs: [("d", "b"), ("a", "c")]
                .map(|(gate, up)| Recipe::Concatenate {
                    axis: 0,
                    inputs: vec![
                        Recipe::source(gate, TensorSelection::Full),
                        Recipe::source(up, TensorSelection::Full),
                    ],
                })
                .to_vec(),
        };
        let read = recipe.prepare_encoded_read(&source).unwrap().unwrap();
        assert_eq!(read.output().shape, [2, 4, 4]);
        let mut output = vec![0; read.output().byte_len as usize];
        read.read_into(&mut output).unwrap();
        assert_eq!(
            output,
            [vec![4; 8], vec![2; 8], vec![1; 8], vec![3; 8]].concat()
        );
        let reads = source.source_diagnostics().unwrap().physical_reads;
        let transpose = Recipe::Transpose {
            input: Box::new(recipe),
            axes: vec![0, 2, 1],
        };
        assert!(transpose.prepare_encoded_read(&source).unwrap().is_none());
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
        // Qualification must not skip an earlier geometry error to report a
        // later invalid permutation from its byte-order proof.
        let competing_errors = Recipe::Concatenate {
            axis: 0,
            inputs: vec![
                Recipe::Reshape {
                    input: Box::new(Recipe::source("a", TensorSelection::Full)),
                    shape: vec![3],
                },
                Recipe::Transpose {
                    input: Box::new(Recipe::source("b", TensorSelection::Full)),
                    axes: vec![0, 0],
                },
            ],
        };
        assert!(matches!(
            competing_errors.infer(&source.0 as &dyn CheckpointSource),
            Err(crate::recipe::RecipeError::ElementCountMismatch {
                input: 8,
                output: 3
            })
        ));
        assert!(matches!(
            competing_errors.prepare_encoded_read(&source),
            Err(crate::recipe::RecipeError::ElementCountMismatch {
                input: 8,
                output: 3
            })
        ));
        // A singleton-axis permutation changes declared geometry without
        // changing bytes, including when followed by an exact packed-bit view.
        let singleton = Recipe::Transpose {
            input: Box::new(Recipe::Reshape {
                input: Box::new(Recipe::source("a", TensorSelection::Full)),
                shape: vec![2, 1, 4],
            }),
            axes: vec![0, 2, 1],
        };
        let read = singleton.prepare_encoded_read(&source).unwrap().unwrap();
        assert_eq!(read.output().shape, [2, 4, 1]);
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
        let bit_view = Recipe::View {
            input: Box::new(singleton),
            dtype: crate::recipe::RecipeDtype::U32,
            shape: vec![2],
        };
        let identity = Recipe::Cast {
            input: Box::new(bit_view.clone()),
            dtype: crate::recipe::RecipeDtype::U32,
        };
        let read = identity.prepare_encoded_read(&source).unwrap().unwrap();
        assert_eq!(read.output().shape, [2]);
        assert_eq!(read.output().dtype, crate::recipe::RecipeDtype::U32);
        let detached = crate::recipe::EncodedRecipeRead::prepare_detached(std::iter::once(&read))
            .unwrap()
            .construct(())
            .unwrap();
        let mut output = [0; 8];
        detached.read_many_into(&mut [&mut output]).unwrap();
        assert_eq!(output, [1; 8]);
        let physical_reads = source.source_diagnostics().unwrap().physical_reads;
        let conversion = Recipe::Cast {
            input: Box::new(bit_view),
            dtype: crate::recipe::RecipeDtype::F32,
        };
        assert!(conversion.prepare_encoded_read(&source).unwrap().is_none());
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            physical_reads
        );
        let invalid = Recipe::Concatenate {
            axis: 0,
            inputs: vec![],
        };
        assert!(invalid.prepare_encoded_read(&source).is_err());
    }
}
