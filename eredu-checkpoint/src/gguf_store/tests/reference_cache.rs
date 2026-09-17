//! Independent old cache owner; production touched coordinates are not its oracle.
use super::*;

pub(super) struct ReferenceCache {
    pub materializers: Vec<TensorMaterializer>,
    pub last_used: Vec<u64>,
    pub touched: BTreeSet<PathBuf>,
    pub tick: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

pub(super) struct ReferenceStore {
    source: GgufWeightStore,
    pub cache: Arc<Mutex<ReferenceCache>>,
}

pub(super) struct ReferenceLease {
    lease: GgufLease,
    pub cache: Arc<Mutex<ReferenceCache>>,
}

impl std::ops::Deref for ReferenceStore {
    type Target = GgufWeightStore;
    fn deref(&self) -> &Self::Target {
        &self.source
    }
}
impl std::ops::Deref for ReferenceLease {
    type Target = GgufLease;
    fn deref(&self) -> &Self::Target {
        &self.lease
    }
}

impl ReferenceStore {
    pub fn new(source: GgufWeightStore) -> Self {
        let cache = {
            let mut actual = source.inner.readers.lock().unwrap();
            assert_eq!(actual.tick, 0);
            assert!(actual.touched.is_empty());
            assert!(actual
                .materializers
                .iter()
                .all(|m| m.open_shard_path().is_none()));
            ReferenceCache {
                materializers: std::mem::take(&mut actual.materializers),
                last_used: std::mem::take(&mut actual.last_used),
                touched: BTreeSet::new(),
                tick: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
            }
        };
        Self {
            source,
            cache: Arc::new(Mutex::new(cache)),
        }
    }
    pub fn acquire(&self, request: TensorReadRequest) -> Result<ReferenceLease, StoreError> {
        Ok(ReferenceLease {
            lease: self.source.acquire(request)?,
            cache: self.cache.clone(),
        })
    }
    pub fn stats(
        &self,
    ) -> (
        u64,
        u64,
        u64,
        usize,
        u64,
        u64,
        u64,
        Vec<u64>,
        BTreeSet<PathBuf>,
    ) {
        let cache = self.cache.lock().unwrap();
        (
            cache.hits,
            cache.misses,
            cache.evictions,
            cache
                .materializers
                .iter()
                .filter(|m| m.open_shard_path().is_some())
                .count(),
            self.inner.statistics.physical_reads.load(Ordering::Relaxed),
            self.inner
                .statistics
                .physical_read_bytes
                .load(Ordering::Relaxed),
            cache.tick,
            cache.last_used.clone(),
            cache.touched.clone(),
        )
    }
    pub fn snapshot(&self) -> (String, u64, Vec<u64>, Vec<Option<PathBuf>>) {
        let cache = self.cache.lock().unwrap();
        let mut diagnostics = self.source.diagnostics().unwrap();
        diagnostics.cache_hits = cache.hits;
        diagnostics.cache_misses = cache.misses;
        diagnostics.evictions = cache.evictions;
        diagnostics.currently_cached_shards = cache
            .materializers
            .iter()
            .filter(|m| m.open_shard_path().is_some())
            .count();
        diagnostics.touched_shard_paths = cache.touched.iter().cloned().collect();
        diagnostics.payload_shard_paths = cache.touched.iter().cloned().collect();
        (
            format!("{diagnostics:?}"),
            cache.tick,
            cache.last_used.clone(),
            cache
                .materializers
                .iter()
                .map(|m| m.open_shard_path().map(Path::to_path_buf))
                .collect(),
        )
    }
}
