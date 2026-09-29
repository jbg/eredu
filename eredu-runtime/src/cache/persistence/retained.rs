//! Bounded payload reads from the exact file admitted during cache restoration.
use super::*;
use std::sync::Mutex;

/// Retained immutable shard identity and bounded metadata, without a payload cache.
///
/// The open handle survives path replacement and unlinking. Callers provide the
/// already admitted destination buffers; reads hash those exact bytes before
/// publishing them. No payload-sized intermediate allocation is made here.
#[derive(Debug)]
pub struct RetainedCacheShard {
    path: PathBuf,
    file: Mutex<File>,
    metadata: safetensors::tensor::Metadata,
    data_start: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    fn write(path: &Path, values: &[i32]) -> String {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                (
                    "records",
                    TensorView::new(Dtype::I32, vec![values.len()], &bytes).unwrap(),
                ),
                (
                    "reserved",
                    TensorView::new(Dtype::I32, vec![0], &[]).unwrap(),
                ),
            ],
            None,
            path,
        )
        .unwrap();
        hash_prompt_cache_shard_payload(path).unwrap()
    }

    #[test]
    fn retained_shard_survives_replacement_and_unlink_without_payload_buffering() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("records.safetensors");
        let values = [i32::MAX, i32::MIN, 16_777_217, -99];
        let hash = write(&path, &values);
        let source = RetainedCacheShard::open(&path, 16).unwrap();
        fs::remove_file(&path).unwrap();
        write(&path, &[0, 0, 0, 0]);
        let mut bytes = [0; 16];
        source
            .read_into(
                &mut [("reserved", &mut []), ("records", &mut bytes)],
                Some(&hash),
            )
            .unwrap();
        assert_eq!(
            bytes,
            values
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
                .as_slice()
        );
        fs::remove_file(path).unwrap();
        source
            .read_into(
                &mut [("records", &mut bytes), ("reserved", &mut [])],
                Some(&hash),
            )
            .unwrap();
    }

    #[test]
    fn each_read_checks_copied_bytes_and_exact_destination_geometry() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("records.safetensors");
        let hash = write(&path, &[17, -23]);
        assert!(RetainedCacheShard::open(&path, 4).is_err());
        let source = RetainedCacheShard::open(&path, 8).unwrap();
        let mut bytes = [0; 8];
        assert!(source
            .read_into(&mut [("records", &mut bytes)], Some(&hash))
            .is_err());
        assert!(source
            .read_into(
                &mut [("reserved", &mut []), ("records", &mut [0; 4])],
                Some(&hash)
            )
            .is_err());
        assert!(source
            .read_into(
                &mut [("reserved", &mut []), ("reserved", &mut [])],
                Some(&hash)
            )
            .is_err());
        source
            .read_into(
                &mut [("records", &mut bytes), ("reserved", &mut [])],
                Some(&hash),
            )
            .unwrap();
        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[1]).unwrap();
        let error = source
            .read_into(
                &mut [("records", &mut bytes), ("reserved", &mut [])],
                Some(&hash),
            )
            .unwrap_err();
        assert!(error.to_string().contains("payload SHA-256 mismatch"));
    }

    #[test]
    fn opens_sparse_large_payload_with_only_bounded_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("large.safetensors");
        let payload = 1usize << 30;
        let header = serde_json::to_vec(&serde_json::json!({"records": {
            "dtype": "U8", "shape": [payload], "data_offsets": [0, payload]
        }}))
        .unwrap();
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + payload as u64)
            .unwrap();
        let source = RetainedCacheShard::open(&path, payload as u64).unwrap();
        assert_eq!(source.metadata().info("records").unwrap().shape, [payload]);
        assert!(source
            .read_into(&mut [("records", &mut [0; 8])], None)
            .is_err());
    }
}

impl RetainedCacheShard {
    /// Opens bounded metadata and checks the exact admitted payload length.
    pub fn open(path: &Path, payload_bytes: u64) -> Result<Self, PromptCachePersistenceError> {
        let mut file = File::open(path).map_err(|source| PromptCachePersistenceError::Io {
            action: "retain cache shard",
            path: path.to_path_buf(),
            source,
        })?;
        let (metadata, length, data_start) = read_shard_metadata_from_file(path, &mut file)?;
        validate_file_boundary(path, &metadata, length, data_start)?;
        if metadata.data_len() as u64 != payload_bytes {
            return Err(malformed(
                path,
                "payload length differs from admitted cache block",
            ));
        }
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            metadata,
            data_start,
        })
    }

    /// Retains a prompt shard and validates its metadata against the manifest on
    /// the same handle subsequently used for payload reads.
    pub fn open_block(
        path: &Path,
        block: &PromptCacheBlock,
    ) -> Result<Self, PromptCachePersistenceError> {
        let source = Self::open(path, block.logical_bytes)?;
        validate_block_metadata(
            path,
            block,
            &source.metadata,
            source.data_start + source.metadata.data_len() as u64,
            source.data_start,
        )?;
        Ok(source)
    }

    /// Exact admitted array declarations, available before allocating native buffers.
    pub fn metadata(&self) -> &safetensors::tensor::Metadata {
        &self.metadata
    }

    /// Reads all arrays into caller-owned buffers and checks the payload digest.
    /// Destinations may be in any order but must cover every array exactly once.
    /// On failure their contents must be discarded. Mutation of an admitted file
    /// is detected on every read, including reads after a previous successful load.
    pub fn read_into(
        &self,
        destinations: &mut [(&str, &mut [u8])],
        expected_sha256: Option<&str>,
    ) -> Result<(), PromptCachePersistenceError> {
        if destinations.len() != self.metadata.tensors().len() {
            return Err(malformed(
                &self.path,
                "destination array count differs from shard",
            ));
        }
        // Only compact descriptors are sorted. Payload bytes remain in the
        // caller's admitted transfer allocations throughout validation and I/O.
        let mut ranges = Vec::with_capacity(destinations.len());
        for (index, (name, bytes)) in destinations.iter().enumerate() {
            if destinations[..index].iter().any(|(prior, _)| prior == name) {
                return Err(malformed(&self.path, "duplicate destination array"));
            }
            let info = self
                .metadata
                .info(name)
                .ok_or_else(|| malformed(&self.path, format!("missing array {name}")))?;
            if bytes.len() != info.data_offsets.1 - info.data_offsets.0 {
                return Err(malformed(
                    &self.path,
                    "destination byte length differs from shard",
                ));
            }
            ranges.push((info.data_offsets, index));
        }
        ranges.sort_unstable();
        let mut file = self
            .file
            .lock()
            .map_err(|_| malformed(&self.path, "shard reader lock poisoned"))?;
        let mut digest = Sha256::new();
        for ((start, _), index) in ranges {
            let bytes = &mut destinations[index].1;
            file.seek(SeekFrom::Start(self.data_start + start as u64))
                .and_then(|_| file.read_exact(bytes))
                .map_err(|source| PromptCachePersistenceError::Io {
                    action: "read admitted cache payload",
                    path: self.path.clone(),
                    source,
                })?;
            digest.update(bytes);
        }
        if let Some(expected) = expected_sha256 {
            let actual = hex(digest.finalize());
            if expected != actual {
                return Err(malformed(
                    &self.path,
                    format!("payload SHA-256 mismatch: expected {expected}, computed {actual}"),
                ));
            }
        }
        Ok(())
    }
}
