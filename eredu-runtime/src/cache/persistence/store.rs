//! Memory and directory storage for the same prompt-cache artifacts.
use super::*;
use eredu_core::cache::{
    PromptCacheOptions, PromptCacheRead, PromptCacheReader, PromptCacheStore,
    PromptCacheStoreError as Error, PromptCacheWrite,
};
use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::from_error(error)
}
fn invalid(message: &str) -> Error {
    PromptCacheError::Malformed(message.into()).into()
}
fn shard_name(name: &str) -> Result<(), Error> {
    if name.is_empty()
        || Path::new(name).components().count() != 1
        || !matches!(
            Path::new(name).components().next(),
            Some(Component::Normal(_))
        )
        || name.contains(['/', '\\'])
    {
        return Err(invalid("invalid prompt-cache shard name"));
    }
    Ok(())
}
fn payload_hash(bytes: &[u8]) -> Result<String, Error> {
    let mut reader = Cursor::new(bytes);
    let name = Path::new("prompt-cache shard");
    let (metadata, length, start) =
        read_shard_metadata_from_file(name, &mut reader).map_err(failure)?;
    validate_file_boundary(name, &metadata, length, start).map_err(failure)?;
    Ok(hex(Sha256::digest(&bytes[start as usize..])))
}
/// Common bounded manifest/shard validation, independent of the store medium.
pub(super) fn validate_artifact(read: &dyn PromptCacheRead) -> Result<(), Error> {
    let manifest = read.manifest();
    manifest.validate()?;
    for block in &manifest.blocks {
        shard_name(&block.shard)?;
        let mut reader = read.open_shard(&block.shard)?;
        let name = Path::new(&block.shard);
        let (meta, len, start) =
            read_shard_metadata_from_file(name, &mut *reader).map_err(failure)?;
        validate_block_metadata(name, block, &meta, len, start).map_err(failure)?;
    }
    for state in &manifest.state_tensors {
        shard_name(&state.shard)?;
        let mut reader = read.open_shard(&state.shard)?;
        let name = Path::new(&state.shard);
        let (meta, len, start) =
            read_shard_metadata_from_file(name, &mut *reader).map_err(failure)?;
        validate_state_metadata(name, state, &meta, len, start).map_err(failure)?;
    }
    Ok(())
}

fn validate_publication(
    manifest: &PromptCacheManifest,
    hashes: &BTreeMap<String, String>,
) -> Result<(), Error> {
    manifest.validate()?;
    let entries: Vec<_> = manifest
        .blocks
        .iter()
        .map(|b| (&b.shard, &b.payload_sha256))
        .chain(
            manifest
                .state_tensors
                .iter()
                .map(|s| (&s.shard, &s.payload_sha256)),
        )
        .collect();
    if entries.len() != hashes.len()
        || entries
            .iter()
            .any(|(name, hash)| hashes.get(*name) != Some(*hash))
    {
        return Err(invalid(
            "prompt-cache shard set or payload digest differs from publication",
        ));
    }
    Ok(())
}

/// A bounded, memory-only store. It creates no files or temporary directories.
/// Published readers pin immutable bytes and survive replacement/eviction.
pub struct MemoryPromptCacheStore {
    max_bytes: u64,
    current: Mutex<Option<Arc<MemoryGeneration>>>,
    budget: Arc<Mutex<u64>>,
}
struct MemoryGeneration {
    manifest: PromptCacheManifest,
    shards: BTreeMap<String, Arc<[u8]>>,
    _reservation: MemoryReservation,
}
struct MemoryReservation {
    budget: Arc<Mutex<u64>>,
    bytes: u64,
}
impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if let Ok(mut used) = self.budget.lock() {
            *used -= self.bytes;
        }
    }
}
impl MemoryPromptCacheStore {
    /// Limits the complete serialized capture allowance, including staging.
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            current: Mutex::new(None),
            budget: Arc::new(Mutex::new(0)),
        }
    }
}
struct MemoryRead(Arc<MemoryGeneration>);
impl PromptCacheRead for MemoryRead {
    fn manifest(&self) -> &PromptCacheManifest {
        &self.0.manifest
    }
    fn open_shard(&self, name: &str) -> Result<Box<dyn PromptCacheReader>, Error> {
        shard_name(name)?;
        Ok(Box::new(MemoryReader {
            bytes: Cursor::new(
                self.0
                    .shards
                    .get(name)
                    .ok_or_else(|| invalid("missing prompt-cache shard"))?
                    .clone(),
            ),
            _generation: self.0.clone(),
        }))
    }
}
struct MemoryReader {
    bytes: Cursor<Arc<[u8]>>,
    _generation: Arc<MemoryGeneration>,
}
impl Read for MemoryReader {
    fn read(&mut self, target: &mut [u8]) -> std::io::Result<usize> {
        self.bytes.read(target)
    }
}
impl Seek for MemoryReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.bytes.seek(pos)
    }
}
impl PromptCacheStore for MemoryPromptCacheStore {
    fn open(&self) -> Result<Option<Box<dyn PromptCacheRead>>, Error> {
        let current = self
            .current
            .lock()
            .map_err(|_| invalid("prompt-cache store lock poisoned"))?
            .clone();
        if let Some(read) = &current {
            validate_artifact(&MemoryRead(read.clone()))?;
        }
        Ok(current.map(|r| Box::new(MemoryRead(r)) as Box<dyn PromptCacheRead>))
    }
    fn begin(
        &self,
        reserved_bytes: u64,
        options: &PromptCacheOptions,
    ) -> Result<Box<dyn PromptCacheWrite + '_>, Error> {
        if let Some(limit) = options.capture_limit() {
            if reserved_bytes > limit {
                return Err(Error::Budget {
                    requested: reserved_bytes,
                    limit,
                });
            }
        }
        if self
            .current
            .lock()
            .map_err(|_| invalid("prompt-cache store lock poisoned"))?
            .is_some()
            && !options.replace_existing()
        {
            return Err(invalid("prompt cache already exists"));
        }
        let mut used = self
            .budget
            .lock()
            .map_err(|_| invalid("prompt-cache budget lock poisoned"))?;
        let requested = reserved_bytes.checked_add(*used).unwrap_or(u64::MAX);
        if requested > self.max_bytes {
            return Err(Error::Budget {
                requested,
                limit: self.max_bytes,
            });
        }
        *used = requested;
        drop(used);
        let reservation = MemoryReservation {
            budget: self.budget.clone(),
            bytes: reserved_bytes,
        };
        Ok(Box::new(MemoryWrite {
            store: self,
            shards: BTreeMap::new(),
            bytes: 0,
            hashes: BTreeMap::new(),
            reserved: reserved_bytes,
            replace_existing: options.replace_existing(),
            reservation,
        }))
    }
    fn clear(&self) -> Result<(), Error> {
        *self
            .current
            .lock()
            .map_err(|_| invalid("prompt-cache store lock poisoned"))? = None;
        Ok(())
    }
    fn retained_bytes(&self) -> u64 {
        self.budget.lock().map(|n| *n).unwrap_or(self.max_bytes)
    }
    fn max_bytes(&self) -> u64 {
        self.max_bytes
    }
}
struct MemoryWrite<'a> {
    store: &'a MemoryPromptCacheStore,
    shards: BTreeMap<String, Arc<[u8]>>,
    bytes: u64,
    hashes: BTreeMap<String, String>,
    reserved: u64,
    replace_existing: bool,
    reservation: MemoryReservation,
}
impl PromptCacheWrite for MemoryWrite<'_> {
    fn put_shard(&mut self, name: &str, bytes: Vec<u8>) -> Result<String, Error> {
        shard_name(name)?;
        if self.shards.contains_key(name) {
            return Err(invalid("duplicate prompt-cache shard"));
        }
        let total = self
            .bytes
            .checked_add(bytes.len() as u64)
            .unwrap_or(u64::MAX);
        if total > self.reserved {
            return Err(Error::Budget {
                requested: total,
                limit: self.reserved,
            });
        }
        let hash = payload_hash(&bytes)?;
        self.shards.insert(name.into(), bytes.into());
        self.hashes.insert(name.into(), hash.clone());
        self.bytes = total;
        Ok(hash)
    }
    fn commit(self: Box<Self>, manifest: &PromptCacheManifest) -> Result<(), Error> {
        validate_publication(manifest, &self.hashes)?;
        let mut reservation = self.reservation;
        let metadata_bytes = serde_json::to_vec(manifest).map_err(failure)?.len() as u64;
        let bytes = self
            .bytes
            .checked_add(metadata_bytes)
            .ok_or_else(|| invalid("prompt-cache size overflow"))?;
        if bytes > self.reserved {
            return Err(Error::Budget {
                requested: bytes,
                limit: self.reserved,
            });
        }
        {
            let mut used = reservation
                .budget
                .lock()
                .map_err(|_| invalid("prompt-cache budget lock poisoned"))?;
            *used -= reservation.bytes - bytes;
        }
        reservation.bytes = bytes;
        let generation = Arc::new(MemoryGeneration {
            manifest: manifest.clone(),
            shards: self.shards,
            _reservation: reservation,
        });
        validate_artifact(&MemoryRead(generation.clone()))?;
        let mut current = self
            .store
            .current
            .lock()
            .map_err(|_| invalid("prompt-cache store lock poisoned"))?;
        if current.is_some() && !self.replace_existing {
            return Err(invalid("prompt cache already exists"));
        }
        *current = Some(generation);
        Ok(())
    }
}

/// Existing atomic directory publication, exposed through the common store contract.
pub struct DirectoryPromptCacheStore {
    directory: PathBuf,
    max_bytes: u64,
}
impl DirectoryPromptCacheStore {
    /// Creates a store with an explicit capture allowance. No files are opened yet.
    pub fn new(directory: impl Into<PathBuf>, max_bytes: u64) -> Self {
        Self {
            directory: directory.into(),
            max_bytes,
        }
    }
}
struct FileGeneration {
    manifest: PromptCacheManifest,
    files: BTreeMap<String, (PathBuf, Arc<Mutex<File>>, u64)>,
}
struct FileReader {
    file: Arc<Mutex<File>>,
    position: u64,
    length: u64,
}
impl Read for FileReader {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let mut file = self
            .file
            .lock()
            .map_err(|_| std::io::Error::other("shard reader lock poisoned"))?;
        file.seek(SeekFrom::Start(self.position))?;
        let read = file.read(bytes)?;
        self.position += read as u64;
        Ok(read)
    }
}
impl Seek for FileReader {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let next = match position {
            SeekFrom::Start(n) => i128::from(n),
            SeekFrom::Current(n) => i128::from(self.position) + i128::from(n),
            SeekFrom::End(n) => i128::from(self.length) + i128::from(n),
        };
        self.position =
            u64::try_from(next).map_err(|_| std::io::Error::other("invalid shard seek"))?;
        Ok(self.position)
    }
}
impl PromptCacheRead for FileGeneration {
    fn manifest(&self) -> &PromptCacheManifest {
        &self.manifest
    }
    fn open_shard(&self, name: &str) -> Result<Box<dyn PromptCacheReader>, Error> {
        let (_, file, length) = self
            .files
            .get(name)
            .ok_or_else(|| invalid("missing prompt-cache shard"))?;
        Ok(Box::new(FileReader {
            file: file.clone(),
            position: 0,
            length: *length,
        }))
    }
    fn shard_path(&self, name: &str) -> Option<PathBuf> {
        self.files.get(name).map(|v| v.0.clone())
    }
}
impl PromptCacheStore for DirectoryPromptCacheStore {
    fn open(&self) -> Result<Option<Box<dyn PromptCacheRead>>, Error> {
        if !self.directory.exists() {
            return Ok(None);
        }
        let root = resolve_prompt_cache_root(&self.directory).map_err(failure)?;
        let manifest = inspect_prompt_cache_generation(&root).map_err(failure)?;
        let names = manifest
            .blocks
            .iter()
            .map(|b| &b.shard)
            .chain(manifest.state_tensors.iter().map(|s| &s.shard));
        let mut files = BTreeMap::new();
        for name in names {
            let path = safe_prompt_cache_shard_path(&root, name).map_err(failure)?;
            let file = File::open(&path).map_err(failure)?;
            let length = file.metadata().map_err(failure)?.len();
            files.insert(name.clone(), (path, Arc::new(Mutex::new(file)), length));
        }
        let generation = FileGeneration { manifest, files };
        validate_artifact(&generation)?;
        Ok(Some(Box::new(generation)))
    }
    fn begin(
        &self,
        reserved_bytes: u64,
        options: &PromptCacheOptions,
    ) -> Result<Box<dyn PromptCacheWrite + '_>, Error> {
        if let Some(limit) = options.capture_limit() {
            if reserved_bytes > limit {
                return Err(Error::Budget {
                    requested: reserved_bytes,
                    limit,
                });
            }
        }
        if reserved_bytes > self.max_bytes {
            return Err(Error::Budget {
                requested: reserved_bytes,
                limit: self.max_bytes,
            });
        }
        Ok(Box::new(DirectoryWrite {
            publication: PromptCachePublication::begin(&self.directory, options.replace_existing())
                .map_err(failure)?,
            reserved: reserved_bytes,
            bytes: 0,
            hashes: BTreeMap::new(),
        }))
    }
    fn clear(&self) -> Result<(), Error> {
        if self.directory.exists() {
            fs::remove_dir_all(&self.directory).map_err(failure)?;
        }
        Ok(())
    }
    fn retained_bytes(&self) -> u64 {
        self.open().ok().flatten().map_or(0, |r| {
            r.manifest()
                .blocks
                .iter()
                .map(|b| b.logical_bytes)
                .chain(r.manifest().state_tensors.iter().map(|s| s.logical_bytes))
                .sum()
        })
    }
    fn max_bytes(&self) -> u64 {
        self.max_bytes
    }
    fn directory(&self) -> Option<&Path> {
        Some(&self.directory)
    }
}
struct DirectoryWrite {
    publication: PromptCachePublication,
    reserved: u64,
    bytes: u64,
    hashes: BTreeMap<String, String>,
}
impl PromptCacheWrite for DirectoryWrite {
    fn put_shard(&mut self, name: &str, bytes: Vec<u8>) -> Result<String, Error> {
        shard_name(name)?;
        let total = self
            .bytes
            .checked_add(bytes.len() as u64)
            .unwrap_or(u64::MAX);
        if total > self.reserved {
            return Err(Error::Budget {
                requested: total,
                limit: self.reserved,
            });
        }
        let path = self.publication.staging_directory().join(name);
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(failure)?;
        file.write_all(&bytes).map_err(failure)?;
        file.sync_all().map_err(failure)?;
        let hash = payload_hash(&bytes)?;
        self.hashes.insert(name.into(), hash.clone());
        self.bytes = total;
        Ok(hash)
    }
    fn commit(self: Box<Self>, manifest: &PromptCacheManifest) -> Result<(), Error> {
        validate_publication(manifest, &self.hashes)?;
        self.publication.commit(manifest).map_err(failure)
    }
}
