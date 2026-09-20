//! Canonical discovery and index validation for SafeTensors checkpoints.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::Read,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use serde::{Deserialize, Deserializer, de::MapAccess};

use crate::{
    recipe::RecipeCatalog,
    store::{AdmittedShard, StoreError, TensorMetadata},
    validation::{CatalogTensorMetadata, SafetensorsCatalog},
};
use safetensors::tensor::{Metadata, TensorInfo};

pub(crate) const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Input limits for checkpoint discovery and lazy shard-header decoding.
///
/// These cap encoded input, not decoded metadata or process memory. Header
/// limits are retained with shard admissions and apply when each header is read.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SafetensorsDiscoveryLimits {
    /// Maximum encoded index length, including whitespace. Defaults to 100 MB.
    pub max_index_bytes: u64,
    /// Maximum JSON header length, excluding its eight-byte length prefix.
    /// Values above the format reader's 100 MB ceiling cannot raise that ceiling.
    pub max_header_bytes: u64,
}

impl Default for SafetensorsDiscoveryLimits {
    fn default() -> Self {
        Self {
            max_index_bytes: 100_000_000,
            max_header_bytes: MAX_HEADER_BYTES,
        }
    }
}

/// One canonically resolved SafeTensors checkpoint shard set.
///
/// Discovery parses an optional Hugging Face index exactly once, requires its
/// tensor map to exactly match the referenced shard headers, and resolves every
/// payload beneath the checkpoint access root. Snapshot payload symlinks may
/// target the sibling repository `blobs` directory, but no path may escape that
/// repository. Clones share the immutable path/tensor catalog and its shard
/// admissions; cloning does not copy those maps or their strings.
#[derive(Debug, Clone)]
pub struct SafetensorsShards {
    catalog: Arc<ShardCatalog>,
    limits: SafetensorsDiscoveryLimits,
}

#[derive(Debug)]
struct ShardCatalog {
    payload_paths: Vec<PathBuf>,
    logical_payload_paths: BTreeMap<String, PathBuf>,
    tensor_locations: Option<BTreeMap<String, PathBuf>>,
    admissions: BTreeMap<PathBuf, Arc<AdmittedShard>>,
    recipes: Arc<crate::recipe::RecipeInferenceCache>,
}

impl SafetensorsShards {
    /// Discovers, admits, and validates a SafeTensors file or checkpoint directory.
    pub fn discover(path: impl AsRef<Path>) -> Result<Self, SafetensorsShardError> {
        Self::discover_with_limits(path, SafetensorsDiscoveryLimits::default())
    }

    /// Strict discovery with explicit index and per-shard header input limits.
    pub fn discover_with_limits(
        path: impl AsRef<Path>,
        limits: SafetensorsDiscoveryLimits,
    ) -> Result<Self, SafetensorsShardError> {
        let path = path.as_ref();
        let shards = Self::discover_catalog(path, limits)?;
        shards
            .validate_headers()
            .map_err(|(payload, error)| match error {
                StoreError::ContradictoryIndexMapping { .. }
                | StoreError::UnindexedShardTensor { .. } => {
                    SafetensorsShardError::MalformedIndex {
                        path: path.join("model.safetensors.index.json"),
                        message: error.to_string(),
                    }
                }
                other => malformed_shard(payload, other.to_string()),
            })?;
        Ok(shards)
    }

    /// Strict discovery with prospective admission for every shard header.
    /// Header failures preserve their typed source and accepted custody. Initial
    /// discovery/index storage requires separate admission by the caller.
    pub fn discover_with_header_admission(
        path: impl AsRef<Path>,
        limits: SafetensorsDiscoveryLimits,
        admission: Arc<dyn SafetensorsHeaderAdmission>,
    ) -> Result<Self, StoreError> {
        let shards = Self::discover_catalog_with_header_admission(path, limits, Some(admission))?;
        shards.validate_headers().map_err(|(_, error)| error)?;
        Ok(shards)
    }

    fn validate_headers(&self) -> Result<(), (&Path, StoreError)> {
        for payload in self.payload_paths() {
            self.admission(payload)
                .header(payload)
                .map_err(|error| (payload.as_path(), error))?;
        }
        Ok(())
    }

    /// Builds an admitted tensor-to-shard catalog without reading payload headers.
    ///
    /// The neutral weight store uses this path so it can validate and buffer only
    /// shards requested by the caller. Public discovery remains strict.
    pub(crate) fn discover_catalog(
        path: impl AsRef<Path>,
        limits: SafetensorsDiscoveryLimits,
    ) -> Result<Self, SafetensorsShardError> {
        Self::discover_catalog_with_header_admission(path, limits, None)
    }

    pub(crate) fn discover_catalog_with_header_admission(
        path: impl AsRef<Path>,
        limits: SafetensorsDiscoveryLimits,
        header_admission: Option<Arc<dyn SafetensorsHeaderAdmission>>,
    ) -> Result<Self, SafetensorsShardError> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::discover_directory(path, limits, header_admission);
        }
        let payload = canonicalize(path)?;
        ShardCatalog {
            payload_paths: vec![payload.clone()],
            logical_payload_paths: BTreeMap::from([("weights".into(), payload)]),
            tensor_locations: None,
            admissions: BTreeMap::new(),
            recipes: Arc::default(),
        }
        .admit(limits, header_admission)
    }

    fn discover_directory(
        root: &Path,
        limits: SafetensorsDiscoveryLimits,
        header_admission: Option<Arc<dyn SafetensorsHeaderAdmission>>,
    ) -> Result<Self, SafetensorsShardError> {
        let access_root = canonical_checkpoint_access_root(root)?;
        let index_path = root.join("model.safetensors.index.json");
        if !index_path.exists() {
            let payload = admit_payload(&root.join("model.safetensors"), &access_root)?;
            return ShardCatalog {
                payload_paths: vec![payload.clone()],
                logical_payload_paths: BTreeMap::from([("weights".into(), payload)]),
                tensor_locations: None,
                admissions: BTreeMap::new(),
                recipes: Arc::default(),
            }
            .admit(limits, header_admission);
        }

        let mut file =
            std::fs::File::open(&index_path).map_err(|error| io_error(&index_path, error))?;
        let length = file
            .metadata()
            .map_err(|error| io_error(&index_path, error))?
            .len();
        if length > limits.max_index_bytes {
            return Err(index_too_large(&index_path, limits.max_index_bytes));
        }
        // The read limit also applies if the file grows after metadata inspection.
        let raw = read_index(&index_path, &mut file, limits.max_index_bytes)?;
        let index: SafetensorsIndex =
            serde_json::from_str(&raw).map_err(|error| SafetensorsShardError::MalformedIndex {
                path: index_path.clone(),
                message: error.to_string(),
            })?;
        if index.weight_map.0.is_empty() {
            return Err(SafetensorsShardError::MalformedIndex {
                path: index_path,
                message: "weight_map must not be empty".into(),
            });
        }

        let mut payload_paths = BTreeSet::new();
        let mut logical_payload_paths = BTreeMap::new();
        let mut tensor_locations = BTreeMap::new();
        for (tensor, relative) in index.weight_map.0 {
            if tensor.is_empty() {
                return Err(SafetensorsShardError::MalformedIndex {
                    path: index_path.clone(),
                    message: "tensor names must not be empty".into(),
                });
            }
            let payload = if let Some(payload) = logical_payload_paths.get(&relative) {
                PathBuf::clone(payload)
            } else {
                let member = validate_relative_shard_path(Path::new(&relative))?;
                let payload = admit_payload(&root.join(member), &access_root)?;
                payload_paths.insert(payload.clone());
                logical_payload_paths.insert(relative, payload.clone());
                payload
            };
            tensor_locations.insert(tensor, payload);
        }
        ShardCatalog {
            payload_paths: payload_paths.into_iter().collect(),
            logical_payload_paths,
            tensor_locations: Some(tensor_locations),
            admissions: BTreeMap::new(),
            recipes: Arc::default(),
        }
        .admit(limits, header_admission)
    }

    /// Input limits retained by this admitted shard set.
    pub const fn limits(&self) -> SafetensorsDiscoveryLimits {
        self.limits
    }

    pub(crate) fn recipe_cache(&self) -> &crate::recipe::RecipeInferenceCache {
        &self.catalog.recipes
    }

    pub(crate) fn admission(&self, path: &Path) -> &Arc<AdmittedShard> {
        &self.catalog.admissions[path]
    }

    /// Returns the canonical, deterministically ordered payload paths.
    pub fn payload_paths(&self) -> &[PathBuf] {
        &self.catalog.payload_paths
    }

    /// Returns location-independent logical shard roles paired with admitted paths.
    ///
    /// Indexed artifacts retain the relative member names from the admitted
    /// index even when snapshot symlinks resolve outside the submitted
    /// directory. Direct artifacts use the stable semantic role `weights`.
    pub fn logical_payload_paths(&self) -> &BTreeMap<String, PathBuf> {
        &self.catalog.logical_payload_paths
    }

    /// Consumes the discovery result into independently owned canonical paths.
    /// Shared catalogs copy the paths; the last catalog owner moves them out.
    pub fn into_payload_paths(self) -> Vec<PathBuf> {
        match Arc::try_unwrap(self.catalog) {
            Ok(catalog) => catalog.payload_paths,
            Err(catalog) => catalog.payload_paths.clone(),
        }
    }

    /// Returns indexed tensor-to-canonical-shard mappings.
    ///
    /// `None` denotes a direct file or an unindexed `model.safetensors` file.
    pub fn tensor_locations(&self) -> Option<&BTreeMap<String, PathBuf>> {
        self.catalog.tensor_locations.as_ref()
    }
}

impl ShardCatalog {
    fn admit(
        mut self,
        limits: SafetensorsDiscoveryLimits,
        header_admission: Option<Arc<dyn SafetensorsHeaderAdmission>>,
    ) -> Result<SafetensorsShards, SafetensorsShardError> {
        let mut expected = BTreeMap::<PathBuf, BTreeSet<String>>::new();
        if let Some(locations) = &self.tensor_locations {
            for (key, path) in locations {
                expected
                    .entry(path.clone())
                    .or_default()
                    .insert(key.clone());
            }
        }
        for path in &self.payload_paths {
            let shard = AdmittedShard::new(
                path,
                expected.remove(path),
                limits.max_header_bytes,
                header_admission.clone(),
            )
            .map_err(|error| malformed_shard(path, error.to_string()))?;
            self.admissions.insert(path.clone(), Arc::new(shard));
        }
        Ok(SafetensorsShards {
            catalog: Arc::new(self),
            limits,
        })
    }
}

// Memoization is not part of the value's catalog identity.
impl PartialEq for SafetensorsShards {
    fn eq(&self, other: &Self) -> bool {
        self.catalog.payload_paths == other.catalog.payload_paths
            && self.catalog.logical_payload_paths == other.catalog.logical_payload_paths
            && self.catalog.tensor_locations == other.catalog.tensor_locations
            && self.limits == other.limits
    }
}
impl Eq for SafetensorsShards {}

/// Exact header-only metadata for one strictly admitted SafeTensors shard set.
///
/// Construction performs strict [`SafetensorsShards`] discovery and reads only
/// each shard's eight-byte header length and JSON header. Tensor payload bytes
/// are never read. The retained shard set can later be passed literally to
/// [`crate::store::SafetensorsWeightStore::open_admitted`].
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SafetensorsMetadataCatalog {
    shards: SafetensorsShards,
    tensors: BTreeMap<String, TensorMetadata>,
}

impl SafetensorsMetadataCatalog {
    /// Discovers an exact shard set and validates all tensor metadata from headers only.
    pub fn discover(path: impl AsRef<Path>) -> Result<Self, SafetensorsShardError> {
        Self::discover_with_limits(path, SafetensorsDiscoveryLimits::default())
    }

    /// Reads exact metadata with explicit encoded index and header limits.
    pub fn discover_with_limits(
        path: impl AsRef<Path>,
        limits: SafetensorsDiscoveryLimits,
    ) -> Result<Self, SafetensorsShardError> {
        Self::from_admitted(SafetensorsShards::discover_with_limits(path, limits)?)
    }

    /// Discovers exact metadata with prospective admission for each header.
    /// The separate catalog map and source discovery need their own admission.
    pub fn discover_with_header_admission(
        path: impl AsRef<Path>,
        limits: SafetensorsDiscoveryLimits,
        admission: Arc<dyn SafetensorsHeaderAdmission>,
    ) -> Result<Self, StoreError> {
        Self::from_admitted(SafetensorsShards::discover_with_header_admission(
            path, limits, admission,
        )?)
        .map_err(StoreError::from)
    }

    /// Builds a catalog using the retained shard admissions without rereading headers.
    pub fn from_admitted(shards: SafetensorsShards) -> Result<Self, SafetensorsShardError> {
        let mut tensors = BTreeMap::new();
        for shard in shards.payload_paths() {
            let header = shards
                .admission(shard)
                .header(shard)
                .map_err(|error| malformed_shard(shard, error.to_string()))?;
            for (name, metadata) in &header.tensors {
                if tensors.insert(name.clone(), metadata.clone()).is_some() {
                    return Err(malformed_shard(
                        shard,
                        format!("tensor {name:?} occurs in more than one admitted shard"),
                    ));
                }
            }
        }
        Ok(Self { shards, tensors })
    }

    /// Returns the absolute encoded byte offset retained during header admission.
    pub fn tensor_offset(&self, name: &str) -> Option<u64> {
        let tensor = self.tensors.get(name)?;
        let path = tensor.backing_shard.as_ref()?;
        let header = self.shards.admission(path).header(path).ok()?;
        Some((header.payload_offset + header.metadata.info(name)?.data_offsets.0) as u64)
    }

    /// Returns the strictly admitted shard set used to build this catalog.
    pub const fn shards(&self) -> &SafetensorsShards {
        &self.shards
    }

    /// Clones the admitted shards for deferred payload-store construction.
    pub fn admitted_shards(&self) -> SafetensorsShards {
        self.shards.clone()
    }

    /// Consumes the catalog into its admitted shards.
    pub fn into_admitted_shards(self) -> SafetensorsShards {
        self.shards
    }

    /// Returns exact tensor metadata in deterministic name order.
    pub const fn tensors(&self) -> &BTreeMap<String, TensorMetadata> {
        &self.tensors
    }

    /// Returns exact metadata for one admitted tensor.
    pub fn tensor(&self, name: &str) -> Option<&TensorMetadata> {
        self.tensors.get(name)
    }
}

impl SafetensorsCatalog for SafetensorsMetadataCatalog {
    fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        self.tensors
            .get(key)
            .map(|metadata| CatalogTensorMetadata {
                shape: metadata.logical_shape.clone(),
                stored_dtype: metadata.stored_dtype.clone(),
            })
            .ok_or_else(|| format!("unknown SafeTensors tensor {key:?}"))
    }
}

impl RecipeCatalog for SafetensorsMetadataCatalog {
    fn recipe_cache(&self) -> Option<&crate::recipe::RecipeInferenceCache> {
        Some(self.shards.recipe_cache())
    }

    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.tensors
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
}

/// Failure to discover and admit a SafeTensors checkpoint shard set.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum SafetensorsShardError {
    /// The encoded index exceeds the configured discovery input limit.
    #[error("SafeTensors index {path} exceeds {limit_bytes} bytes", path = .path.display())]
    IndexTooLarge {
        /// Rejected index path.
        path: PathBuf,
        /// Configured maximum encoded index length.
        limit_bytes: u64,
    },
    /// A checkpoint path or referenced payload does not exist.
    #[error("SafeTensors checkpoint shard does not exist: {path}", path = .path.display())]
    MissingShard {
        /// Missing checkpoint or payload path.
        path: PathBuf,
    },
    /// The checkpoint index could not be decoded or validated.
    #[error("malformed SafeTensors index {path}: {message}", path = .path.display())]
    MalformedIndex {
        /// Index path.
        path: PathBuf,
        /// Decoder or validation detail.
        message: String,
    },
    /// A referenced payload header could not be decoded for index validation.
    #[error("malformed SafeTensors shard {path}: {message}", path = .path.display())]
    MalformedShard {
        /// Invalid payload path.
        path: PathBuf,
        /// Decoder or validation detail.
        message: String,
    },
    /// A shard name is absolute, traversing, empty, or resolves outside the access root.
    #[error("unsafe SafeTensors shard path {path}", path = .path.display())]
    UnsafeShardPath {
        /// Rejected path.
        path: PathBuf,
    },
    /// Filesystem access failed.
    #[error("SafeTensors shard discovery failed for {path}: {message}", path = .path.display())]
    Io {
        /// Affected path.
        path: PathBuf,
        /// Stable failure detail.
        message: String,
    },
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: UniqueWeightMap,
}

#[derive(Debug)]
struct UniqueWeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueWeightMap;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a tensor-to-shard object with unique names")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key, shard)) = map.next_entry::<String, String>()? {
                    if values.insert(key.clone(), shard).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate tensor mapping for {key:?}"
                        )));
                    }
                }
                Ok(UniqueWeightMap(values))
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

#[derive(Debug)]
struct UniqueHeader {
    metadata: Option<HashMap<String, String>>,
    tensors: Vec<(String, TensorInfo)>,
}

impl<'de> Deserialize<'de> for UniqueHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a SafeTensors header with unique tensor names")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut seen = BTreeSet::new();
                let mut metadata = None;
                let mut tensors = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate SafeTensors header entry for {key:?}"
                        )));
                    }
                    if key == "__metadata__" {
                        metadata = map.next_value()?;
                    } else {
                        tensors.push((key, map.next_value::<TensorInfo>()?));
                    }
                }
                Ok(UniqueHeader { metadata, tensors })
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

/// Decode JSON once, rejecting duplicate names before typed offset validation.
pub(crate) fn parse_header(bytes: &[u8]) -> Result<Metadata, serde_json::Error> {
    let mut header = serde_json::from_slice::<UniqueHeader>(bytes)?;
    header.tensors.sort_by_key(|(_, info)| info.data_offsets);
    Metadata::new(header.metadata, header.tensors).map_err(serde::de::Error::custom)
}

fn index_too_large(path: &Path, limit_bytes: u64) -> SafetensorsShardError {
    SafetensorsShardError::IndexTooLarge {
        path: path.into(),
        limit_bytes,
    }
}

fn read_index(
    path: &Path,
    reader: &mut impl Read,
    limit_bytes: u64,
) -> Result<String, SafetensorsShardError> {
    let mut bytes = Vec::new();
    reader
        .take(limit_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    if bytes.len() as u64 > limit_bytes {
        return Err(index_too_large(path, limit_bytes));
    }
    String::from_utf8(bytes).map_err(|error| {
        io_error(
            path,
            std::io::Error::new(std::io::ErrorKind::InvalidData, error),
        )
    })
}

fn malformed_shard(path: &Path, message: impl Into<String>) -> SafetensorsShardError {
    SafetensorsShardError::MalformedShard {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn validate_relative_shard_path(path: &Path) -> Result<&Path, SafetensorsShardError> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(SafetensorsShardError::UnsafeShardPath {
            path: path.to_path_buf(),
        });
    }
    Ok(path)
}

fn admit_payload(path: &Path, access_root: &Path) -> Result<PathBuf, SafetensorsShardError> {
    let canonical = canonicalize(path)?;
    if !canonical.starts_with(access_root) {
        return Err(SafetensorsShardError::UnsafeShardPath {
            path: path.to_path_buf(),
        });
    }
    Ok(canonical)
}

fn canonical_checkpoint_access_root(path: &Path) -> Result<PathBuf, SafetensorsShardError> {
    let canonical_root = canonicalize(path)?;
    let Some(snapshots) = canonical_root.parent() else {
        return Ok(canonical_root);
    };
    if snapshots.file_name().and_then(|name| name.to_str()) != Some("snapshots") {
        return Ok(canonical_root);
    }
    let Some(repository_root) = snapshots.parent() else {
        return Ok(canonical_root);
    };
    if !repository_root.join("blobs").is_dir() {
        return Ok(canonical_root);
    }
    canonicalize(repository_root)
}

fn canonicalize(path: &Path) -> Result<PathBuf, SafetensorsShardError> {
    std::fs::canonicalize(path).map_err(|error| io_error(path, error))
}

fn io_error(path: &Path, error: std::io::Error) -> SafetensorsShardError {
    if error.kind() == std::io::ErrorKind::NotFound {
        SafetensorsShardError::MissingShard {
            path: path.to_path_buf(),
        }
    } else {
        SafetensorsShardError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::StoredDtype;
    use std::fs::File;
    use std::io::Write;

    use super::*;
    use safetensors::{Dtype, tensor::TensorView, tensor::serialize_to_file};

    use crate::store::{SafetensorsWeightStore, WeightStore};

    fn write_index(root: &Path, contents: &str) {
        std::fs::write(root.join("model.safetensors.index.json"), contents).unwrap();
    }

    fn write_shard(path: &Path, name: &str) {
        serialize_to_file(
            [(name, TensorView::new(Dtype::U8, vec![1], &[0]).unwrap())],
            None,
            path,
        )
        .unwrap();
    }

    fn write_header_and_sparse_payload(path: &Path, header: serde_json::Value, payload_len: u64) {
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + payload_len).unwrap();
    }

    #[test]
    fn header_admission_rejects_duplicate_tensor_fields() {
        assert!(
            parse_header(
                br#"{"weight":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#
            )
            .is_err()
        );
        assert!(
            parse_header(
                br#"{"weight":{"dtype":"U8","shape":[1],"shape":[1],"data_offsets":[0,1]}}"#
            )
            .is_err()
        );
    }

    #[test]
    fn metadata_catalog_reads_a_huge_sparse_shard_header_without_payload_materialization() {
        let root = tempfile::tempdir().unwrap();
        let shard = root.path().join("huge.safetensors");
        let payload_len = 8_u64 << 30;
        write_header_and_sparse_payload(
            &shard,
            serde_json::json!({
                "huge": {
                    "dtype": "U8",
                    "shape": [payload_len],
                    "data_offsets": [0, payload_len]
                }
            }),
            payload_len,
        );

        let catalog = SafetensorsMetadataCatalog::discover(&shard).unwrap();
        let metadata = catalog.tensor("huge").unwrap();
        assert_eq!(metadata.logical_shape, [payload_len as usize]);
        assert_eq!(metadata.physical_shape, [payload_len as usize]);
        assert_eq!(metadata.stored_dtype, StoredDtype::U8);
        assert_eq!(metadata.encoded_byte_len, payload_len);
        assert_eq!(
            metadata.backing_shard.as_deref(),
            Some(shard.canonicalize().unwrap().as_path())
        );
    }

    #[test]
    fn metadata_catalog_retains_exact_indexed_provenance_and_admitted_shards() {
        let root = tempfile::tempdir().unwrap();
        let left = root.path().join("left.safetensors");
        let right = root.path().join("right.safetensors");
        write_shard(&left, "left");
        write_shard(&right, "right");
        write_index(
            root.path(),
            r#"{"weight_map":{"right":"right.safetensors","left":"left.safetensors"}}"#,
        );

        let catalog = SafetensorsMetadataCatalog::discover(root.path()).unwrap();
        let admitted = SafetensorsShards::discover(root.path()).unwrap();
        assert_eq!(catalog.shards(), &admitted);
        assert_eq!(catalog.admitted_shards(), admitted);
        assert_eq!(
            catalog.tensor("left").unwrap().backing_shard.as_deref(),
            Some(left.canonicalize().unwrap().as_path())
        );
        assert_eq!(
            catalog.tensor("right").unwrap().backing_shard.as_deref(),
            Some(right.canonicalize().unwrap().as_path())
        );
        assert_eq!(
            SafetensorsCatalog::keys(&catalog),
            vec!["left".to_owned(), "right".to_owned()]
        );
        assert_eq!(
            SafetensorsCatalog::metadata(&catalog, "left").unwrap(),
            CatalogTensorMetadata {
                shape: vec![1],
                stored_dtype: StoredDtype::U8,
            }
        );
        assert_eq!(
            RecipeCatalog::tensor_metadata(&catalog, "right").unwrap(),
            catalog.tensor("right").unwrap().clone()
        );
        let store = SafetensorsWeightStore::open_admitted(catalog.admitted_shards(), 1).unwrap();
        assert_eq!(
            WeightStore::keys(&store),
            vec!["left".to_owned(), "right".to_owned()]
        );
    }

    #[test]
    fn metadata_catalog_rejects_invalid_offsets_and_file_lengths() {
        let root = tempfile::tempdir().unwrap();
        let offsets = root.path().join("offsets.safetensors");
        write_header_and_sparse_payload(
            &offsets,
            serde_json::json!({
                "bad": {
                    "dtype": "U8",
                    "shape": [1],
                    "data_offsets": [1, 2]
                }
            }),
            2,
        );
        assert!(matches!(
            SafetensorsMetadataCatalog::discover(&offsets),
            Err(SafetensorsShardError::MalformedShard { .. })
        ));

        let truncated = root.path().join("truncated.safetensors");
        write_header_and_sparse_payload(
            &truncated,
            serde_json::json!({
                "bad": {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [0, 4]
                }
            }),
            3,
        );
        let error = SafetensorsMetadataCatalog::discover(&truncated).unwrap_err();
        assert!(matches!(
            error,
            SafetensorsShardError::MalformedShard { .. }
        ));
        assert!(error.to_string().contains("provides 3"));
    }

    #[test]
    fn metadata_catalog_rejects_duplicate_tensor_names_within_one_header() {
        let root = tempfile::tempdir().unwrap();
        let shard = root.path().join("duplicate.safetensors");
        let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut padded = header.to_vec();
        padded.resize(padded.len().next_multiple_of(8), b' ');
        let mut file = File::create(&shard).unwrap();
        file.write_all(&(padded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&padded).unwrap();
        file.write_all(&[0]).unwrap();

        let error = SafetensorsMetadataCatalog::discover(&shard).unwrap_err();
        assert!(matches!(
            error,
            SafetensorsShardError::MalformedShard { .. }
        ));
        assert!(error.to_string().contains("duplicate SafeTensors header"));
    }

    #[test]
    fn indexed_discovery_is_unique_canonical_and_deterministic() {
        let root = tempfile::tempdir().unwrap();
        write_shard(&root.path().join("z.safetensors"), "z");
        write_shard(&root.path().join("a.safetensors"), "a");
        write_index(
            root.path(),
            r#"{"weight_map":{"z":"z.safetensors","a":"a.safetensors"}}"#,
        );

        let shards = SafetensorsShards::discover(root.path()).unwrap();
        assert_eq!(
            shards.payload_paths(),
            [
                root.path().join("a.safetensors").canonicalize().unwrap(),
                root.path().join("z.safetensors").canonicalize().unwrap(),
            ]
        );
        assert_eq!(
            shards.tensor_locations().unwrap()["a"],
            shards.payload_paths()[0]
        );
    }

    #[test]
    fn indexed_discovery_rejects_missing_misassigned_and_unindexed_tensors() {
        let missing = tempfile::tempdir().unwrap();
        write_shard(&missing.path().join("payload.safetensors"), "actual");
        write_index(
            missing.path(),
            r#"{"weight_map":{"claimed":"payload.safetensors"}}"#,
        );
        assert!(matches!(
            SafetensorsShards::discover(missing.path()),
            Err(SafetensorsShardError::MalformedIndex { .. })
        ));

        let swapped = tempfile::tempdir().unwrap();
        write_shard(&swapped.path().join("a.safetensors"), "a");
        write_shard(&swapped.path().join("b.safetensors"), "b");
        write_index(
            swapped.path(),
            r#"{"weight_map":{"a":"b.safetensors","b":"a.safetensors"}}"#,
        );
        assert!(matches!(
            SafetensorsShards::discover(swapped.path()),
            Err(SafetensorsShardError::MalformedIndex { .. })
        ));

        let unindexed = tempfile::tempdir().unwrap();
        let first = [0_u8];
        let second = [1_u8];
        serialize_to_file(
            [
                (
                    "declared",
                    TensorView::new(Dtype::U8, vec![1], &first).unwrap(),
                ),
                (
                    "extra",
                    TensorView::new(Dtype::U8, vec![1], &second).unwrap(),
                ),
            ],
            None,
            &unindexed.path().join("payload.safetensors"),
        )
        .unwrap();
        write_index(
            unindexed.path(),
            r#"{"weight_map":{"declared":"payload.safetensors"}}"#,
        );
        assert!(matches!(
            SafetensorsShards::discover(unindexed.path()),
            Err(SafetensorsShardError::MalformedIndex { .. })
        ));
    }

    #[test]
    fn index_rejects_duplicates_empty_names_and_unsafe_paths() {
        for contents in [
            r#"{"weight_map":{"a":"one","a":"two"}}"#,
            r#"{"weight_map":{"":"one"}}"#,
            r#"{"weight_map":{}}"#,
        ] {
            let root = tempfile::tempdir().unwrap();
            write_index(root.path(), contents);
            assert!(matches!(
                SafetensorsShards::discover(root.path()),
                Err(SafetensorsShardError::MalformedIndex { .. })
            ));
        }
        for shard in ["", "../outside", "/absolute"] {
            let root = tempfile::tempdir().unwrap();
            write_index(
                root.path(),
                &serde_json::json!({"weight_map": {"a": shard}}).to_string(),
            );
            assert!(matches!(
                SafetensorsShards::discover(root.path()),
                Err(SafetensorsShardError::UnsafeShardPath { .. })
            ));
        }
    }

    #[test]
    fn discovery_rejects_missing_indexed_payloads() {
        let root = tempfile::tempdir().unwrap();
        write_index(
            root.path(),
            r#"{"weight_map":{"weight":"missing.safetensors"}}"#,
        );
        assert!(matches!(
            SafetensorsShards::discover(root.path()),
            Err(SafetensorsShardError::MissingShard { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn discovery_rejects_payload_symlinks_outside_the_access_root() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("checkpoint");
        std::fs::create_dir(&root).unwrap();
        let outside = parent.path().join("outside.safetensors");
        std::fs::write(&outside, []).unwrap();
        symlink(&outside, root.join("linked.safetensors")).unwrap();
        write_index(&root, r#"{"weight_map":{"weight":"linked.safetensors"}}"#);

        assert!(matches!(
            SafetensorsShards::discover(&root),
            Err(SafetensorsShardError::UnsafeShardPath { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn discovery_accepts_snapshot_symlinks_into_repository_blobs() {
        use std::os::unix::fs::symlink;

        let cache = tempfile::tempdir().unwrap();
        let repository = cache.path().join("models--owner--model");
        let snapshot = repository.join("snapshots/revision");
        let blobs = repository.join("blobs");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        let blob = blobs.join("payload");
        write_header_and_sparse_payload(
            &blob,
            serde_json::json!({"weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
            1,
        );
        symlink("../../blobs/payload", snapshot.join("model.safetensors")).unwrap();

        let shards = SafetensorsShards::discover(&snapshot).unwrap();
        assert_eq!(shards.payload_paths(), [blob.canonicalize().unwrap()]);
        assert_eq!(
            shards.logical_payload_paths(),
            &BTreeMap::from([("weights".into(), blob.canonicalize().unwrap())])
        );
    }
}

mod header;
pub use header::{SafetensorsHeaderError, SafetensorsHeaderPlan};

#[cfg(test)]
mod limits_tests;

#[cfg(test)]
mod shared_catalog_tests;

mod header_admission;
pub use header_admission::{
    SafetensorsHeaderAdmission, SafetensorsHeaderFailure, SafetensorsHeaderRequest,
    SafetensorsHeaderReservation,
};
