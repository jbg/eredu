//! Exact encoded reads into storage owned by the materializer.

use super::*;

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
/// are ordered by shard and offset, and adjacent file/destination ranges are
/// coalesced. This owns no payload buffer and pins no shard-cache entry.
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

    fn read_into_with_hook(
        self,
        destination: &mut [u8],
        mut after_shard: impl FnMut(),
    ) -> Result<(), StoreError> {
        if destination.len() != self.byte_len {
            return Err(StoreError::Internal(
                "encoded read destination has the wrong length".into(),
            ));
        }
        for (path, shard) in self.shards {
            let mut file = shard.admitted.open_validated(&path)?;
            for span in shard.spans {
                file.seek(SeekFrom::Start(span.file.start))
                    .map_err(|error| io_error(&path, error))?;
                file.read_exact(&mut destination[span.destination.clone()])
                    .map_err(|error| io_error(&path, error))?;
                self.telemetry
                    .physical_reads
                    .fetch_add(1, Ordering::Relaxed);
                self.telemetry
                    .physical_read_bytes
                    .fetch_add(span.destination.len() as u64, Ordering::Relaxed);
            }
            self.cache
                .lock()
                .map_err(|_| StoreError::Internal("checkpoint shard cache is poisoned".into()))?
                .payloads
                .insert(path.clone());
            after_shard();
            shard.admitted.validate_file(&path, &file)?;
        }
        Ok(())
    }
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
        let shard = store.acquire_shard(&CatalogEntry {
            shard: path.clone(),
        })?;
        for (index, key) in group {
            let info = shard
                .metadata
                .info(key)
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
            let metadata = metadata_for_info(key, &path, info)?;
            let start = shard
                .payload_offset
                .checked_add(info.data_offsets.0)
                .ok_or_else(|| StoreError::Overflow {
                    context: "bulk tensor offset".into(),
                })?;
            entries.push((index, metadata, start, Arc::clone(&shard.admitted_file)));
        }
        // The next shard can be opened even with a one-shard cache bound.
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
        assert_eq!(diagnostics.currently_cached_shards, 1);
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
