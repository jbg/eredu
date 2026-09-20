//! Shared transitions of the existing source-owned SafeTensors cache.
use super::*;

pub(super) fn lock(cache: &Mutex<CacheState>) -> Result<MutexGuard<'_, CacheState>, StoreError> {
    cache
        .lock()
        .map_err(|_| StoreError::Internal("checkpoint shard cache is poisoned".into()))
}

pub(super) fn acquire_shard(
    cache: &Mutex<CacheState>,
    maximum: usize,
    path: &PathBuf,
    admission: impl FnOnce() -> Arc<AdmittedShard>,
) -> Result<Arc<CachedShard>, StoreError> {
    let canonical_path = path.clone();
    let mut cache = lock(cache)?;
    cache.tick = cache.tick.saturating_add(1);
    let tick = cache.tick;
    if let Some(shard) = cache
        .entries
        .get(&canonical_path)
        .map(|entry| Arc::clone(&entry.shard))
    {
        cache.hits = cache.hits.saturating_add(1);
        cache.entries.get_mut(&canonical_path).unwrap().last_used = tick;
        return Ok(shard);
    }
    cache.misses = cache.misses.saturating_add(1);
    if cache.entries.len() >= maximum {
        let victim = cache
            .entries
            .iter()
            .filter(|(_, candidate)| Arc::strong_count(&candidate.shard) == 1)
            .min_by(|(left_path, left), (right_path, right)| {
                (left.last_used, *left_path).cmp(&(right.last_used, *right_path))
            })
            .map(|(path, _)| path.clone());
        if let Some(victim) = victim {
            cache.entries.remove(&victim);
            cache.evictions = cache.evictions.saturating_add(1);
        } else {
            return Err(StoreError::CapacityExhausted {
                maximum: maximum,
                leased: cache
                    .entries
                    .values()
                    .map(|entry| entry.shard.path.clone())
                    .collect(),
            });
        }
    }
    let admission = admission();
    admission.header(&canonical_path)?;
    let shard = Arc::new(CachedShard {
        path: canonical_path.clone(),
        admitted_file: Arc::clone(&admission.file),
        admission,
        full_tensors: Mutex::new(BTreeMap::new()),
    });
    cache.paths.mark_touched(path);
    cache.entries.insert(
        canonical_path,
        CacheEntry {
            shard: Arc::clone(&shard),
            last_used: tick,
        },
    );
    Ok(shard)
}

/// Only the real shared weak map can create this payload association.
pub(super) struct CachedPayload<'a> {
    shard: &'a CachedShard,
    key: &'a str,
    bytes: Arc<Vec<u8>>,
}
pub(super) fn lookup<'a>(
    shard: &'a CachedShard,
    key: &'a str,
) -> Result<Option<CachedPayload<'a>>, StoreError> {
    let bytes = shard
        .full_tensors
        .lock()
        .map_err(|_| StoreError::Internal("checkpoint tensor cache is poisoned".into()))?
        .get(key)
        .and_then(Weak::upgrade);
    Ok(bytes.map(|bytes| CachedPayload { shard, key, bytes }))
}
impl<'a> CachedPayload<'a> {
    pub(super) fn validate(self) -> Result<ValidatedCachedPayload<'a>, StoreError> {
        drop(self.shard.admitted_file.open_validated(&self.shard.path)?);
        Ok(ValidatedCachedPayload(self))
    }
}
/// Exact cache association after its ordinary admitted-file check and close.
pub(super) struct ValidatedCachedPayload<'a>(CachedPayload<'a>);
impl ValidatedCachedPayload<'_> {
    pub(super) fn into_bytes(self) -> Arc<Vec<u8>> {
        self.0.bytes
    }
    pub(super) fn retained_bytes(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.0.bytes)
    }
    pub(super) fn bytes(&self) -> &[u8] {
        &self.0.bytes
    }
    pub(super) fn matches(&self, admitted: &AdmittedFile, key: &str, extent: usize) -> bool {
        std::ptr::eq(self.0.shard.admitted_file.as_ref(), admitted)
            && self.0.key == key
            && self.0.bytes.len() == extent
    }
}
pub(super) fn publish_full(
    shard: &CachedShard,
    key: &String,
    bytes: &Arc<Vec<u8>>,
) -> Result<(), StoreError> {
    shard
        .full_tensors
        .lock()
        .map_err(|_| StoreError::Internal("checkpoint tensor cache is poisoned".into()))?
        .insert(key.clone(), Arc::downgrade(bytes));
    Ok(())
}
pub(super) fn touch_metadata(cache: &Mutex<CacheState>, path: &PathBuf) -> Result<(), StoreError> {
    lock(cache)?.paths.mark_touched(path);
    Ok(())
}
pub(super) fn publish_payload(
    cache: &Mutex<CacheState>,
    shard: &CachedShard,
) -> Result<(), StoreError> {
    lock(cache)?.paths.mark_payload(&shard.path);
    Ok(())
}
