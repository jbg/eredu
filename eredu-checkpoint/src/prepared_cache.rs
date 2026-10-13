//! Bounded, content-addressed prepared-checkpoint storage.
//!
//! Producers write encoded tensor tiles, never native tensors. Publication is
//! atomic; a shared file lock pins a published source until its final owner drops.

use crate::store::{
    CheckpointSource, EncodedTensorLease, ReadPolicy, SafetensorsWeightStore,
    SharedCheckpointSource, StoreError, TensorReadRequest,
};
use safetensors::tensor::Dtype;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

/// Filesystem and retention policy for reusable prepared tensors.
#[derive(Debug, Clone)]
pub struct PreparedCheckpointCachePolicy {
    /// Cache directory, separate from original checkpoint artifacts.
    pub directory: PathBuf,
    /// Maximum total published and in-progress payload bytes.
    pub max_bytes: u64,
}

impl Default for PreparedCheckpointCachePolicy {
    fn default() -> Self {
        Self {
            directory: std::env::var_os("EREDU_PREPARED_CHECKPOINT_CACHE")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    let root = std::env::var_os("XDG_CACHE_HOME")
                        .map(PathBuf::from)
                        .or_else(|| {
                            std::env::var_os("HOME").map(|home| {
                                PathBuf::from(home).join(if cfg!(target_os = "macos") {
                                    "Library/Caches"
                                } else {
                                    ".cache"
                                })
                            })
                        })
                        .unwrap_or_else(std::env::temp_dir);
                    root.join("eredu/prepared-checkpoints-v1")
                }),
            max_bytes: 32 * 1024 * 1024 * 1024,
        }
    }
}

/// Exact encoded geometry declared before a producer writes payloads.
#[derive(Debug, Clone)]
pub struct PreparedTensorLayout {
    /// Logical tensor name.
    pub name: String,
    /// SafeTensors scalar encoding (including packed integer words).
    pub dtype: Dtype,
    /// Encoded tensor shape.
    pub shape: Vec<usize>,
}

/// One admitted cache hit or unpublished bounded writer.
pub enum PreparedCheckpointEntry {
    /// An immutable prepared source; no conversion is needed.
    Cached(SharedCheckpointSource),
    /// Producer must finish every declared tensor before publication.
    Vacant(PreparedCheckpointWriter),
}

fn invalid(message: impl Into<String>) -> StoreError {
    StoreError::Internal(message.into())
}
fn io(path: &Path, error: impl std::fmt::Display) -> StoreError {
    StoreError::Io {
        path: path.into(),
        message: error.to_string(),
    }
}
fn open_lock(path: &Path) -> Result<File, StoreError> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| io(path, e))
}

/// Hashes exact selected source bytes with bounded leases, plus producer identity.
/// Identity must include the conversion mechanism/version and complete recipes.
/// Metadata or filesystem timestamps alone never authorize a cache hit.
pub fn prepared_checkpoint_recipe_key(
    source: &dyn CheckpointSource,
    recipes: &[crate::recipe::DerivedWeightRecipe],
    producer_identity: &[u8],
    max_read_bytes: u64,
) -> Result<String, StoreError> {
    let mut hash = Sha256::new();
    hash.update(b"eredu-prepared-checkpoint-v1\0");
    hash_part(&mut hash, producer_identity);
    // The tiling is part of the versioned fingerprint, so bounded leaf ordering
    // cannot alias a different recipe or read segmentation.
    hash.update(max_read_bytes.to_le_bytes());
    for recipe in recipes {
        hash_part(&mut hash, format!("{recipe:?}").as_bytes());
        let metadata = recipe.infer(source).map_err(|e| invalid(e.to_string()))?;
        let shape = metadata.shape();
        if shape.len() < 2 {
            return Err(invalid("prepared fingerprint requires matrix recipes"));
        }
        let leading = shape[..shape.len() - 2]
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("fingerprint geometry overflow"))?;
        let rows = shape[shape.len() - 2];
        for matrix in 0..leading {
            let row = recipe
                .select_bounded_matrix_rows(source, matrix, 0, 1)
                .map_err(|e| invalid(e.to_string()))?;
            let peak = row
                .peak_materialization_bytes(source)
                .map_err(|e| invalid(e.to_string()))?;
            if peak == 0 || peak > max_read_bytes {
                return Err(invalid("fingerprint bound cannot hold one selected row"));
            }
            let tile_rows = usize::try_from(max_read_bytes / peak)
                .map_err(|_| invalid("fingerprint row count overflow"))?
                .max(1);
            let mut start = 0;
            while start < rows {
                let mut count = tile_rows.min(rows - start);
                let selected = loop {
                    let candidate = recipe
                        .select_bounded_matrix_rows(source, matrix, start, start + count)
                        .map_err(|e| invalid(e.to_string()))?;
                    if candidate
                        .peak_materialization_bytes(source)
                        .map_err(|e| invalid(e.to_string()))?
                        <= max_read_bytes
                    {
                        break candidate;
                    }
                    count /= 2;
                    if count == 0 {
                        return Err(invalid("fingerprint selection exceeds its bound"));
                    }
                };
                hash_recipe_leaves(source, &selected, max_read_bytes, &mut hash)?;
                start += count;
            }
        }
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
fn hash_part(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
fn hash_recipe_leaves(
    source: &dyn CheckpointSource,
    recipe: &crate::recipe::DerivedWeightRecipe,
    bound: u64,
    hash: &mut Sha256,
) -> Result<(), StoreError> {
    use crate::recipe::DerivedWeightRecipe;
    match recipe {
        DerivedWeightRecipe::Source { key, selection } => {
            hash_part(hash, key.as_bytes());
            hash_part(hash, format!("{selection:?}").as_bytes());
            let metadata = source.source_metadata(key)?;
            hash_part(
                hash,
                format!(
                    "{:?}:{:?}:{:?}",
                    metadata.stored_dtype,
                    metadata.logical_shape,
                    source.source_provenance(key)?.source_encoding
                )
                .as_bytes(),
            );
            let request = TensorReadRequest {
                key: key.clone(),
                selection: selection.clone(),
                policy: ReadPolicy::RequireBounded,
            };
            if let Some(batch) = source.prepare_encoded_read(std::slice::from_ref(&request))? {
                if batch.byte_len() as u64 > bound {
                    return Err(invalid("fingerprint read exceeds its bound"));
                }
                let mut bytes = vec![0; batch.byte_len()];
                batch.read_into(&mut bytes)?;
                hash_part(hash, &bytes);
            } else {
                let lease = source.acquire_lease(request)?;
                if !lease.bounded_read_proof().physically_bounded
                    || lease.bounded_read_proof().length_bytes > bound
                {
                    return Err(invalid("fingerprint lease violates the read bound"));
                }
                let converted;
                let bytes = match &lease {
                    crate::store::CheckpointLease::Gguf(lease) => {
                        converted = lease.materialize_portable()?.into_converted();
                        match &converted {
                            eredu_gguf::ConvertedTensor::Dense(tensor) => tensor.data.as_slice(),
                            _ => {
                                return Err(invalid(
                                    "prepared fingerprint requires dense GGUF source storage",
                                ));
                            }
                        }
                    }
                    _ => lease
                        .encoded_bytes()
                        .ok_or_else(|| invalid("prepared source has no encoded bytes"))?,
                };
                if bytes.len() as u64 > bound {
                    return Err(invalid("fingerprint payload exceeds its bound"));
                }
                hash_part(hash, bytes);
            }
        }
        DerivedWeightRecipe::Concatenate { inputs, .. }
        | DerivedWeightRecipe::Stack { inputs, .. } => {
            for input in inputs {
                hash_recipe_leaves(source, input, bound, hash)?;
            }
        }
        DerivedWeightRecipe::Select { input, .. }
        | DerivedWeightRecipe::Reshape { input, .. }
        | DerivedWeightRecipe::Transpose { input, .. }
        | DerivedWeightRecipe::Cast { input, .. }
        | DerivedWeightRecipe::View { input, .. }
        | DerivedWeightRecipe::NegLog { input }
        | DerivedWeightRecipe::SubtractOne { input } => {
            hash_recipe_leaves(source, input, bound, hash)?
        }
    }
    Ok(())
}

struct CachedSource {
    store: SafetensorsWeightStore,
    // Held through all deferred acquisitions; eviction needs an exclusive lock.
    _pin: File,
}
impl CheckpointSource for CachedSource {
    fn recipe_cache(&self) -> Option<&crate::recipe::RecipeInferenceCache> {
        CheckpointSource::recipe_cache(&self.store)
    }
    fn source_keys(&self) -> Vec<String> {
        self.store.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<crate::store::TensorMetadata, StoreError> {
        self.store.source_metadata(key)
    }
    fn acquire_lease(
        &self,
        request: TensorReadRequest,
    ) -> Result<crate::store::CheckpointLease, StoreError> {
        self.store.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<crate::store::WeightStoreDiagnostics, StoreError> {
        self.store.source_diagnostics()
    }
    fn prepare_encoded_read(
        &self,
        requests: &[TensorReadRequest],
    ) -> Result<Option<crate::store::EncodedReadBatch>, StoreError> {
        self.store.prepare_encoded_read(requests)
    }
}

fn read_checksum(path: &Path) -> Result<[u8; 32], StoreError> {
    let mut file = File::open(path).map_err(|e| io(path, e))?;
    let mut bytes = [0; 32];
    file.read_exact(&mut bytes).map_err(|e| io(path, e))?;
    if file.read(&mut [0]).map_err(|e| io(path, e))? != 0 {
        return Err(invalid("invalid prepared checksum length"));
    }
    Ok(bytes)
}
fn digest_file(path: &Path) -> Result<[u8; 32], StoreError> {
    let mut file = File::open(path).map_err(|e| io(path, e))?;
    let mut buffer = vec![0; HASH_BUFFER_BYTES];
    let mut digest = Sha256::new();
    loop {
        let n = file.read(&mut buffer).map_err(|e| io(path, e))?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(digest.finalize().into())
}
fn pinned_source(path: &Path, pin: File) -> Result<SharedCheckpointSource, StoreError> {
    pin.lock_shared().map_err(|e| io(path, e))?;
    Ok(Arc::new(CachedSource {
        store: SafetensorsWeightStore::open_with_max_cached_shards(path, 1)?,
        _pin: pin,
    }))
}

impl PreparedCheckpointCachePolicy {
    /// Admits an entry under the cache-wide byte budget, evicting unleased oldest
    /// entries. A cache-directory lock serializes admission and publication.
    /// Readers retain per-entry shared locks. Contention never permits overflow.
    pub fn prepare(
        &self,
        key: &str,
        layouts: &[PreparedTensorLayout],
        max_write_bytes: u64,
    ) -> Result<PreparedCheckpointEntry, StoreError> {
        if key.len() != 64
            || !key.bytes().all(|c| c.is_ascii_hexdigit())
            || layouts.is_empty()
            || max_write_bytes == 0
        {
            return Err(invalid(
                "invalid prepared-cache identity, layouts or write bound",
            ));
        }
        fs::create_dir_all(&self.directory).map_err(|e| io(&self.directory, e))?;
        let guard = open_lock(&self.directory.join("cache.lock"))?;
        guard.lock().map_err(|e| io(&self.directory, e))?;
        let path = self.directory.join(format!("{key}.safetensors"));
        let checksum = self.directory.join(format!("{key}.sha256"));
        let pin = open_lock(&self.directory.join(format!("{key}.lock")))?;
        let (header, spans, payload_len) = encode_header(layouts)?;
        let file_len = payload_len
            .checked_add(header.len() as u64)
            .and_then(|n| n.checked_add(8))
            .ok_or_else(|| invalid("prepared file size overflow"))?;
        if file_len > self.max_bytes {
            return Err(StoreError::PreparedCacheCapacity {
                required_bytes: file_len,
                limit_bytes: self.max_bytes,
            });
        }
        if path.exists() {
            let valid = fs::metadata(&path).map_err(|e| io(&path, e))?.len() == file_len
                && read_checksum(&checksum).ok().is_some_and(|expected| {
                    digest_file(&path).is_ok_and(|actual| actual == expected)
                });
            if valid {
                let store = pinned_source(&path, pin)?;
                // Content hash covers geometry; reject an unexpected producer catalog.
                if store.source_keys().len() != layouts.len() {
                    return Err(invalid("prepared cache catalog changed"));
                }
                for layout in layouts {
                    let m = store.source_metadata(&layout.name)?;
                    if m.logical_shape != layout.shape
                        || m.encoded_byte_len != spans[&layout.name].1
                        || m.stored_dtype
                            != crate::store::stored_dtype_from_safetensors(layout.dtype)
                    {
                        return Err(invalid("prepared cache geometry changed"));
                    }
                }
                self.enforce_budget(0, Some(&path))?;
                let stamp = OpenOptions::new()
                    .write(true)
                    .open(&checksum)
                    .map_err(|e| io(&checksum, e))?;
                stamp
                    .set_times(std::fs::FileTimes::new().set_modified(std::time::SystemTime::now()))
                    .map_err(|e| io(&checksum, e))?;
                return Ok(PreparedCheckpointEntry::Cached(store));
            }
            pin.try_lock().map_err(|e| io(&path, e))?;
            fs::remove_file(&path).map_err(|e| io(&path, e))?;
            let _ = fs::remove_file(&checksum);
        } else {
            pin.lock().map_err(|e| io(&path, e))?;
        }
        self.enforce_budget(file_len, None)?;
        let pending = path.with_extension("pending");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&pending)
            .map_err(|e| io(&pending, e))?;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|_| file.write_all(&header))
            .and_then(|_| file.set_len(file_len))
            .map_err(|e| io(&pending, e))?;
        Ok(PreparedCheckpointEntry::Vacant(PreparedCheckpointWriter {
            file,
            pending,
            path,
            checksum,
            spans,
            written: BTreeMap::new(),
            payload_offset: 8 + header.len() as u64,
            max_write_bytes,
            _guard: guard,
            pin: Some(pin),
            published: false,
        }))
    }

    fn enforce_budget(
        &self,
        reserved_bytes: u64,
        protected: Option<&Path>,
    ) -> Result<(), StoreError> {
        let mut entries = Vec::new();
        let mut total = 0u64;
        for entry in fs::read_dir(&self.directory).map_err(|e| io(&self.directory, e))? {
            let entry = entry.map_err(|e| io(&self.directory, e))?;
            let p = entry.path();
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(key) = name.get(..64) else {
                continue;
            };
            if !key.bytes().all(|c| c.is_ascii_hexdigit()) {
                continue;
            }
            let suffix = &name[64..];
            if !matches!(
                suffix,
                ".safetensors" | ".pending" | ".sha256.pending" | ".lock" | ".sha256"
            ) {
                continue;
            }
            // Incomplete writers cannot survive the exclusive directory lock.
            if p.extension().is_some_and(|s| s == "pending") {
                fs::remove_file(&p).map_err(|e| io(&p, e))?;
                continue;
            }
            if suffix == ".lock" && !p.with_extension("safetensors").exists() {
                let lock = open_lock(&p)?;
                if lock.try_lock().is_ok() {
                    let _ = fs::remove_file(&p);
                }
                continue;
            }
            if p.extension().is_some_and(|s| s == "safetensors") {
                let m = entry.metadata().map_err(|e| io(&p, e))?;
                total = total
                    .checked_add(m.len())
                    .ok_or_else(|| invalid("cache byte count overflow"))?;
                let used = fs::metadata(p.with_extension("sha256"))
                    .and_then(|m| m.modified())
                    .or_else(|_| m.modified())
                    .map_err(|e| io(&p, e))?;
                entries.push((used, p, m.len()));
            }
        }
        entries.sort();
        for (_, victim, bytes) in entries {
            if total <= self.max_bytes - reserved_bytes {
                break;
            }
            if protected == Some(victim.as_path()) {
                continue;
            }
            let lock = open_lock(&victim.with_extension("lock"))?;
            if lock.try_lock().is_err() {
                continue;
            }
            fs::remove_file(&victim).map_err(|e| io(&victim, e))?;
            let _ = fs::remove_file(victim.with_extension("sha256"));
            let _ = fs::remove_file(victim.with_extension("lock"));
            total -= bytes;
        }
        if total > self.max_bytes - reserved_bytes {
            return Err(StoreError::PreparedCacheCapacity {
                required_bytes: total.saturating_add(reserved_bytes),
                limit_bytes: self.max_bytes,
            });
        }
        Ok(())
    }
}

type Spans = BTreeMap<String, (u64, u64)>;
fn encode_header(layouts: &[PreparedTensorLayout]) -> Result<(Vec<u8>, Spans, u64), StoreError> {
    let mut offset = 0u64;
    let mut spans = BTreeMap::new();
    let mut header = BTreeMap::new();
    for layout in layouts {
        let length = layout
            .shape
            .iter()
            .try_fold(layout.dtype.bitsize() as u64, |n, d| {
                n.checked_mul(*d as u64)
            })
            .filter(|n| n % 8 == 0)
            .map(|n| n / 8)
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("invalid prepared tensor geometry"))?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| invalid("prepared tensor offset overflow"))?;
        if layout.name.is_empty()
            || layout.name == "__metadata__"
            || spans
                .insert(layout.name.clone(), (offset, length))
                .is_some()
        {
            return Err(invalid("duplicate or invalid prepared tensor name"));
        }
        header.insert(layout.name.clone(),serde_json::json!({"dtype":layout.dtype,"shape":layout.shape,"data_offsets":[offset,end]}));
        offset = end;
    }
    let mut bytes = serde_json::to_vec(&header).map_err(|e| invalid(e.to_string()))?;
    bytes.resize(bytes.len().div_ceil(8) * 8, b' ');
    Ok((bytes, spans, offset))
}

/// A bounded tile sink whose incomplete file is removed on failure or unwind.
/// Writes for each tensor are contiguous and increasing; tensors may interleave.
pub struct PreparedCheckpointWriter {
    file: File,
    pending: PathBuf,
    path: PathBuf,
    checksum: PathBuf,
    spans: Spans,
    written: BTreeMap<String, u64>,
    payload_offset: u64,
    max_write_bytes: u64,
    _guard: File,
    pin: Option<File>,
    published: bool,
}
impl PreparedCheckpointWriter {
    /// Writes the next encoded tile for one declared tensor.
    pub fn write(&mut self, name: &str, offset: u64, bytes: &[u8]) -> Result<(), StoreError> {
        let (start, length) = *self
            .spans
            .get(name)
            .ok_or_else(|| invalid("undeclared prepared tensor"))?;
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("prepared tile offset overflow"))?;
        if bytes.is_empty()
            || bytes.len() as u64 > self.max_write_bytes
            || offset != *self.written.get(name).unwrap_or(&0)
            || end > length
        {
            return Err(invalid(
                "prepared tile is out of order, exceeds geometry or its write bound",
            ));
        }
        self.file
            .seek(SeekFrom::Start(self.payload_offset + start + offset))
            .and_then(|_| self.file.write_all(bytes))
            .map_err(|e| io(&self.pending, e))?;
        self.written.insert(name.into(), end);
        Ok(())
    }
    /// Publishes only complete storage and returns an ordinary bounded source.
    pub fn finish(mut self) -> Result<SharedCheckpointSource, StoreError> {
        if self
            .spans
            .iter()
            .any(|(key, (_, len))| self.written.get(key) != Some(len))
        {
            return Err(invalid("prepared checkpoint is incomplete"));
        }
        self.file.sync_all().map_err(|e| io(&self.pending, e))?;
        let digest = digest_file(&self.pending)?;
        let checksum_pending = self.checksum.with_extension("sha256.pending");
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&checksum_pending)
                .map_err(|e| io(&checksum_pending, e))?;
            f.write_all(&digest)
                .and_then(|_| f.sync_all())
                .map_err(|e| io(&checksum_pending, e))?;
        }
        fs::rename(&checksum_pending, &self.checksum).map_err(|e| io(&self.checksum, e))?;
        fs::rename(&self.pending, &self.path).map_err(|e| io(&self.path, e))?;
        self.published = true;
        pinned_source(&self.path, self.pin.take().expect("unpublished entry pin"))
    }
}
impl Drop for PreparedCheckpointWriter {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_file(&self.pending);
            let _ = fs::remove_file(self.checksum.with_extension("sha256.pending"));
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::store::{EncodedTensorLease, MemoryWeightStore, TensorSelection};
    fn layout() -> Vec<PreparedTensorLayout> {
        vec![PreparedTensorLayout {
            name: "weight".into(),
            dtype: Dtype::F32,
            shape: vec![2, 4],
        }]
    }
    fn policy(directory: &Path, max_bytes: u64) -> PreparedCheckpointCachePolicy {
        PreparedCheckpointCachePolicy {
            directory: directory.into(),
            max_bytes,
        }
    }
    fn write(policy: &PreparedCheckpointCachePolicy, key: &str) -> SharedCheckpointSource {
        let PreparedCheckpointEntry::Vacant(mut writer) =
            policy.prepare(key, &layout(), 16).unwrap()
        else {
            panic!("expected a miss")
        };
        let bytes = (0..8)
            .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
            .collect::<Vec<_>>();
        writer.write("weight", 0, &bytes[..16]).unwrap();
        writer.write("weight", 16, &bytes[16..]).unwrap();
        writer.finish().unwrap()
    }
    #[test]
    fn bounded_matrix_layout_preserves_bytes_and_reuses_prepared_source() {
        let temp = tempfile::tempdir().unwrap();
        let policy = policy(temp.path(), 4096);
        let source = write(&policy, &"a".repeat(64));
        let recipe = crate::recipe::DerivedWeightRecipe::source("weight", TensorSelection::Full);
        // One column per tile: both input and output fit a sixteen-byte bound.
        let transposed = prepare_transposed_matrix(source.as_ref(), &recipe, &policy, 16)
            .unwrap()
            .unwrap();
        let read = |source: &dyn CheckpointSource| {
            let lease = source
                .acquire_lease(TensorReadRequest {
                    key: "matrix".into(),
                    selection: TensorSelection::Full,
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap();
            assert_eq!(lease.output_shape(), &[4, 2]);
            lease.encoded_bytes().unwrap().to_vec()
        };
        let expected: Vec<_> = [0.5f32, 4.5, 1.5, 5.5, 2.5, 6.5, 3.5, 7.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(read(transposed.as_ref()), expected);
        let entries = || {
            fs::read_dir(temp.path())
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                .count()
        };
        assert_eq!(entries(), 2);
        let reloaded = prepare_transposed_matrix(source.as_ref(), &recipe, &policy, 16)
            .unwrap()
            .unwrap();
        assert_eq!(read(reloaded.as_ref()), expected);
        assert_eq!(entries(), 2);
        assert!(
            prepare_transposed_matrix(source.as_ref(), &recipe, &policy, 15)
                .unwrap()
                .is_none()
        );
        let small = PreparedCheckpointCachePolicy {
            directory: temp.path().into(),
            max_bytes: 16,
        };
        assert!(
            prepare_transposed_matrix(source.as_ref(), &recipe, &small, 16)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn content_identity_and_bounded_persisted_reads_survive_reload() {
        let temp = tempfile::tempdir().unwrap();
        let policy = policy(temp.path(), 4096);
        let source = MemoryWeightStore::from_safetensors([(
            "weight".into(),
            Dtype::F32,
            vec![2, 4],
            vec![1; 32],
        )])
        .unwrap();
        let key = prepared_checkpoint_recipe_key(
            &source,
            &[crate::recipe::DerivedWeightRecipe::source(
                "weight",
                TensorSelection::Full,
            )],
            b"producer-v1",
            16,
        )
        .unwrap();
        assert_eq!(
            key,
            prepared_checkpoint_recipe_key(
                &source,
                &[crate::recipe::DerivedWeightRecipe::source(
                    "weight",
                    TensorSelection::Full
                )],
                b"producer-v1",
                16
            )
            .unwrap()
        );
        assert_ne!(
            key,
            prepared_checkpoint_recipe_key(
                &source,
                &[crate::recipe::DerivedWeightRecipe::source(
                    "weight",
                    TensorSelection::Full
                )],
                b"producer-v2",
                16
            )
            .unwrap()
        );
        let changed = MemoryWeightStore::from_safetensors([(
            "weight".into(),
            Dtype::F32,
            vec![2, 4],
            vec![2; 32],
        )])
        .unwrap();
        assert_ne!(
            key,
            prepared_checkpoint_recipe_key(
                &changed,
                &[crate::recipe::DerivedWeightRecipe::source(
                    "weight",
                    TensorSelection::Full
                )],
                b"producer-v1",
                16
            )
            .unwrap()
        );
        // A selected rank never reads or fingerprints unrelated source rows.
        let selected = crate::recipe::DerivedWeightRecipe::source(
            "weight",
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
        );
        let unrelated = MemoryWeightStore::from_safetensors([(
            "weight".into(),
            Dtype::F32,
            vec![2, 4],
            [vec![1; 16], vec![3; 16]].concat(),
        )])
        .unwrap();
        assert_eq!(
            prepared_checkpoint_recipe_key(
                &source,
                std::slice::from_ref(&selected),
                b"selected",
                16
            )
            .unwrap(),
            prepared_checkpoint_recipe_key(&unrelated, &[selected], b"selected", 16).unwrap()
        );
        drop(write(&policy, &key));
        let PreparedCheckpointEntry::Cached(source) = policy.prepare(&key, &layout(), 16).unwrap()
        else {
            panic!("expected reuse")
        };
        let lease = source
            .acquire_lease(TensorReadRequest {
                key: "weight".into(),
                selection: TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        let expected = (4..8)
            .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(lease.encoded_bytes().unwrap(), expected);
        assert!(lease.bounded_read_proof().physically_bounded);
        assert_eq!(lease.bounded_read_proof().length_bytes, 16);
    }
    #[test]
    fn pinned_sources_block_eviction_and_unleased_entries_are_reclaimed() {
        let temp = tempfile::tempdir().unwrap();
        let mut policy = policy(temp.path(), 4096);
        fs::write(
            temp.path().join("original.safetensors"),
            b"original checkpoint",
        )
        .unwrap();
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let c = "c".repeat(64);
        let first = write(&policy, &a);
        let len = fs::metadata(temp.path().join(format!("{a}.safetensors")))
            .unwrap()
            .len();
        policy.max_bytes = len * 2;
        let second = write(&policy, &b);
        assert!(policy.prepare(&c, &layout(), 16).is_err());
        drop(first);
        let third = write(&policy, &c);
        assert!(!temp.path().join(format!("{a}.safetensors")).exists());
        assert!(matches!(
            policy.prepare(&b, &layout(), 16).unwrap(),
            PreparedCheckpointEntry::Cached(_)
        ));
        let bytes = fs::read_dir(temp.path())
            .unwrap()
            .map(Result::unwrap)
            .filter(|f| {
                f.path().extension().is_some_and(|s| s == "safetensors")
                    && f.file_name() != "original.safetensors"
            })
            .map(|f| f.metadata().unwrap().len())
            .sum::<u64>();
        assert_eq!(bytes, policy.max_bytes);
        assert_eq!(
            fs::read(temp.path().join("original.safetensors")).unwrap(),
            b"original checkpoint"
        );
        let tight = PreparedCheckpointCachePolicy {
            directory: temp.path().into(),
            max_bytes: len - 1,
        };
        assert!(
            matches!(tight.prepare(&c,&layout(),16),Err(StoreError::PreparedCacheCapacity {required_bytes,limit_bytes}) if required_bytes==len && limit_bytes==len-1)
        );
        drop(second);
        policy.max_bytes = len;
        assert!(matches!(
            policy.prepare(&c, &layout(), 16).unwrap(),
            PreparedCheckpointEntry::Cached(_)
        ));
        assert!(!temp.path().join(format!("{b}.safetensors")).exists());
        // Refreshing recency must not invalidate an earlier admitted source.
        assert!(third
            .acquire_lease(TensorReadRequest::from("weight"))
            .is_ok());
        drop(third);
    }
    #[test]
    fn failed_writes_are_unpublished_and_corrupt_entries_are_rebuilt() {
        let temp = tempfile::tempdir().unwrap();
        let policy = policy(temp.path(), 4096);
        let key = "a".repeat(64);
        let PreparedCheckpointEntry::Vacant(mut writer) =
            policy.prepare(&key, &layout(), 16).unwrap()
        else {
            panic!("expected miss")
        };
        assert!(writer.write("weight", 0, &[0; 32]).is_err());
        assert!(writer.write("weight", 16, &[0; 16]).is_err());
        writer.write("weight", 0, &[0; 16]).unwrap();
        assert!(writer.finish().is_err());
        assert!(!temp.path().join(format!("{key}.pending")).exists());
        assert!(!temp.path().join(format!("{key}.safetensors")).exists());
        drop(write(&policy, &key));
        let path = temp.path().join(format!("{key}.safetensors"));
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[0xff]).unwrap();
        drop(file);
        assert!(matches!(
            policy.prepare(&key, &layout(), 16).unwrap(),
            PreparedCheckpointEntry::Vacant(_)
        ));
        drop(write(&policy, &key));
        assert!(matches!(
            policy.prepare(&key, &layout(), 16).unwrap(),
            PreparedCheckpointEntry::Cached(_)
        ));
    }
}

/// Prepares a byte-preserving matrix transpose under the ordinary cache policy.
/// Only encoded recipes are supported. Two host tiles fit `max_working_set_bytes`;
/// final native allocations and logical views remain the materializer's concern.
/// Unsupported geometry or insufficient optional cache capacity returns `None`.
pub fn prepare_transposed_matrix(
    source: &dyn CheckpointSource,
    recipe: &crate::recipe::DerivedWeightRecipe,
    policy: &PreparedCheckpointCachePolicy,
    max_working_set_bytes: u64,
) -> Result<Option<SharedCheckpointSource>, StoreError> {
    use crate::{recipe::RecipeDtype, store::TensorSelection};
    let metadata = recipe.infer(source).map_err(|e| invalid(e.to_string()))?;
    let [rows, columns] = metadata.shape.as_slice() else {
        return Ok(None);
    };
    let (rows, columns) = (*rows, *columns);
    let dtype = match metadata.dtype {
        RecipeDtype::U8 => Dtype::U8,
        RecipeDtype::I8 => Dtype::I8,
        RecipeDtype::U16 => Dtype::U16,
        RecipeDtype::I16 => Dtype::I16,
        RecipeDtype::U32 => Dtype::U32,
        RecipeDtype::I32 => Dtype::I32,
        RecipeDtype::U64 => Dtype::U64,
        RecipeDtype::I64 => Dtype::I64,
        RecipeDtype::F16 => Dtype::F16,
        RecipeDtype::BF16 => Dtype::BF16,
        RecipeDtype::F32 => Dtype::F32,
        RecipeDtype::F64 => Dtype::F64,
        _ => return Ok(None),
    };
    let scalar_bytes = dtype.bitsize() / 8;
    let column_bytes = rows
        .checked_mul(scalar_bytes)
        .ok_or_else(|| invalid("transpose column overflow"))?;
    if column_bytes == 0 || columns == 0 || column_bytes as u64 > max_working_set_bytes / 2 {
        return Ok(None);
    }
    let row_bytes = columns
        .checked_mul(scalar_bytes)
        .ok_or_else(|| invalid("transpose row overflow"))?;
    if row_bytes as u64 > max_working_set_bytes {
        return Ok(None);
    }
    let tile_columns = usize::try_from(max_working_set_bytes / 2 / column_bytes as u64)
        .unwrap_or(usize::MAX)
        .min(columns);
    let selected = |start, end| {
        recipe
            .select_bounded(
                source,
                TensorSelection::Range {
                    axis: 1,
                    start,
                    end,
                },
            )
            .map_err(|e| invalid(e.to_string()))
    };
    // Establish support before admitting a writer or reading payloads.
    if selected(0, tile_columns)?
        .prepare_encoded_read(source)
        .map_err(|e| invalid(e.to_string()))?
        .is_none()
    {
        return Ok(None);
    }
    let key = prepared_checkpoint_recipe_key(
        source,
        std::slice::from_ref(recipe),
        b"encoded-matrix-transpose-v1",
        max_working_set_bytes,
    )?;
    let layout = PreparedTensorLayout {
        name: "matrix".into(),
        dtype,
        shape: vec![columns, rows],
    };
    let entry = match policy.prepare(&key, &[layout], max_working_set_bytes / 2) {
        Err(StoreError::PreparedCacheCapacity { .. }) => return Ok(None),
        entry => entry?,
    };
    match entry {
        PreparedCheckpointEntry::Cached(source) => Ok(Some(source)),
        PreparedCheckpointEntry::Vacant(mut writer) => {
            for start in (0..columns).step_by(tile_columns) {
                let count = tile_columns.min(columns - start);
                let mut input = vec![0u8; column_bytes * count];
                let read = selected(start, start + count)?
                    .prepare_encoded_read(source)
                    .map_err(|e| invalid(e.to_string()))?
                    .ok_or_else(|| invalid("encoded transpose support changed"))?;
                read.read_into(&mut input)?;
                let mut output = vec![0u8; input.len()];
                for column in 0..count {
                    for row in 0..rows {
                        let from = (row * count + column) * scalar_bytes;
                        let to = (column * rows + row) * scalar_bytes;
                        output[to..to + scalar_bytes]
                            .copy_from_slice(&input[from..from + scalar_bytes]);
                    }
                }
                writer.write("matrix", (start * column_bytes) as u64, &output)?;
            }
            Ok(Some(writer.finish()?))
        }
    }
}
