//! Exact encoded reads into storage owned by the materializer.

use super::*;
use std::io::{self, IoSliceMut};

const MAX_READ_BYTES: usize = 64 * 1024 * 1024;
const MAX_READ_BUFFERS: usize = 1024;

struct ReadSpan {
    file: Range<u64>,
    destination: Range<usize>,
}

struct ReadShard {
    admitted: Arc<AdmittedFile>,
    spans: Vec<ReadSpan>,
}

/// An admitted, metadata-only batch of full encoded tensor reads.
///
/// Tensor bytes are placed consecutively in the requested order. File reads
/// are ordered by shard and offset, and adjacent file ranges are read together
/// even when their destinations are disjoint. This owns no payload buffer and
/// pins no shard-cache entry.
pub struct EncodedReadBatch {
    tensors: Vec<TensorMetadata>,
    shards: BTreeMap<PathBuf, ReadShard>,
    byte_len: usize,
    telemetry: Arc<SafetensorsReadTelemetry>,
    cache: Arc<Mutex<CacheState>>,
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
        mut after_shard: impl FnMut(),
    ) -> Result<(), StoreError> {
        if batches.len() != destinations.len()
            || batches
                .iter()
                .zip(destinations.iter())
                .any(|(batch, dest)| batch.byte_len != dest.len())
        {
            return Err(StoreError::Internal(
                "encoded read destinations have the wrong lengths".into(),
            ));
        }
        // A distinct store keeps its own counters and read admission, even if
        // another store happens to name the same filesystem path.
        let mut sources = BTreeMap::new();
        let mut group_indices = BTreeMap::<(PathBuf, usize), usize>::new();
        let mut groups = Vec::<DestinationShard<'_>>::new();
        for (batch, destination) in batches.into_iter().zip(destinations.iter_mut()) {
            let source_index = Arc::as_ptr(&batch.telemetry) as usize;
            // Retain even empty batches' source identities until grouping ends.
            sources
                .entry(source_index)
                .or_insert_with(|| Arc::clone(&batch.telemetry));
            let mut spans = Vec::new();
            for (path, shard) in batch.shards {
                let index = *group_indices
                    .entry((path.clone(), source_index))
                    .or_insert_with(|| {
                        let index = groups.len();
                        groups.push(DestinationShard {
                            path,
                            admitted: Arc::clone(&shard.admitted),
                            spans: Vec::new(),
                            telemetry: Arc::clone(&batch.telemetry),
                            cache: Arc::clone(&batch.cache),
                        });
                        index
                    });
                if !Arc::ptr_eq(&groups[index].admitted, &shard.admitted)
                    && groups[index].admitted.identity != shard.admitted.identity
                {
                    return Err(StoreError::AdmittedFileChanged {
                        path: groups[index].path.clone(),
                    });
                }
                spans.extend(shard.spans.into_iter().map(|span| (index, span)));
            }
            // Split each allocation in destination order to obtain disjoint
            // mutable borrows, then rearrange those borrows into file order.
            spans.sort_unstable_by_key(|(_, span)| span.destination.start);
            let mut remaining = &mut **destination;
            let mut position = 0;
            for (index, span) in spans {
                let (_, tail) = remaining.split_at_mut(span.destination.start - position);
                let (bytes, tail) = tail.split_at_mut(span.destination.len());
                remaining = tail;
                position = span.destination.end;
                let mut offset = span.file.start;
                for bytes in bytes.chunks_mut(MAX_READ_BYTES) {
                    let end = offset + bytes.len() as u64;
                    groups[index].spans.push(DestinationSpan {
                        file: offset..end,
                        bytes,
                    });
                    offset = end;
                }
            }
        }
        groups.sort_by(|left, right| left.path.cmp(&right.path));
        for mut group in groups {
            group.spans.sort_unstable_by_key(|span| span.file.start);
            let mut file = group.admitted.open_validated(&group.path)?;
            read_destination_spans(&mut file, group.spans, &group.telemetry)
                .map_err(|error| io_error(&group.path, error))?;
            group
                .cache
                .lock()
                .map_err(|_| StoreError::Internal("checkpoint shard cache is poisoned".into()))?
                .payloads
                .insert(group.path.clone());
            after_shard();
            group.admitted.validate_file(&group.path, &file)?;
        }
        Ok(())
    }
}

struct DestinationSpan<'a> {
    file: Range<u64>,
    bytes: &'a mut [u8],
}

struct DestinationShard<'a> {
    path: PathBuf,
    admitted: Arc<AdmittedFile>,
    spans: Vec<DestinationSpan<'a>>,
    telemetry: Arc<SafetensorsReadTelemetry>,
    cache: Arc<Mutex<CacheState>>,
}

fn read_destination_spans(
    reader: &mut (impl Read + Seek),
    spans: Vec<DestinationSpan<'_>>,
    telemetry: &SafetensorsReadTelemetry,
) -> io::Result<()> {
    let mut spans = spans.into_iter().peekable();
    let mut position = None;
    let mut buffers = Vec::with_capacity(MAX_READ_BUFFERS);
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
        read_exact_vectored(reader, &mut buffers, telemetry)?;
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
    let mut batch = EncodedReadBatch {
        tensors: Vec::with_capacity(keys.len()),
        shards: BTreeMap::new(),
        byte_len: 0,
        telemetry: Arc::clone(&store.read_telemetry),
        cache: Arc::clone(&store.cache),
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
        batch
            .shards
            .entry(path)
            .or_insert_with(|| ReadShard {
                admitted,
                spans: Vec::new(),
            })
            .spans
            .push(ReadSpan {
                file: file_start..file_end,
                destination: batch.byte_len..end,
            });
        batch.byte_len = end;
        batch.tensors.push(metadata);
    }
    for shard in batch.shards.values_mut() {
        shard.spans.sort_unstable_by_key(|span| span.file.start);
        let mut merged = Vec::<ReadSpan>::with_capacity(shard.spans.len());
        for span in std::mem::take(&mut shard.spans) {
            if let Some(previous) = merged.last_mut() {
                if previous.file.end == span.file.start
                    && previous.destination.end == span.destination.start
                {
                    previous.file.end = span.file.end;
                    previous.destination.end = span.destination.end;
                    continue;
                }
            }
            merged.push(span);
        }
        shard.spans = merged;
    }
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
        let sources = [&a as &dyn CheckpointSource, &b];
        let batches =
            sources.map(|source| source.prepare_encoded_read(&["a".into()]).unwrap().unwrap());
        let mut first = [0; 8];
        let mut second = [0; 8];
        EncodedReadBatch::read_many_into(batches.into(), &mut [&mut first, &mut second]).unwrap();
        assert_eq!(first, [1; 8]);
        assert_eq!(second, [1; 8]);
        for source in sources {
            assert_eq!(source.source_diagnostics().unwrap().physical_read_bytes, 8);
        }
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
        let invalid = Recipe::Concatenate {
            axis: 0,
            inputs: vec![],
        };
        assert!(invalid.prepare_encoded_read(&source).is_err());
    }
}
