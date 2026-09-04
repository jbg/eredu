//! Canonical discovery and index validation for SafeTensors checkpoints.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

use serde::{de::MapAccess, Deserialize, Deserializer};

use crate::{
    recipe::RecipeCatalog,
    store::{StoreError, TensorMetadata},
    validation::{CatalogTensorMetadata, SafetensorsCatalog},
    StoredDtype,
};
use safetensors::tensor::{Dtype, Metadata};

const MAX_HEADER_BYTES: u64 = 100_000_000;

/// One canonically resolved SafeTensors checkpoint shard set.
///
/// Discovery parses an optional Hugging Face index exactly once, requires its
/// tensor map to exactly match the referenced shard headers, and resolves every
/// payload beneath the checkpoint access root. Snapshot payload symlinks may
/// target the sibling repository `blobs` directory, but no path may escape that
/// repository.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SafetensorsShards {
    payload_paths: Vec<PathBuf>,
    tensor_locations: Option<BTreeMap<String, PathBuf>>,
}

impl SafetensorsShards {
    /// Discovers, admits, and validates a SafeTensors file or checkpoint directory.
    pub fn discover(path: impl AsRef<Path>) -> Result<Self, SafetensorsShardError> {
        let path = path.as_ref();
        let shards = Self::discover_catalog(path)?;
        if let Some(locations) = shards.tensor_locations() {
            let mut indexed_names = BTreeMap::<PathBuf, BTreeSet<String>>::new();
            for (tensor, payload) in locations {
                indexed_names
                    .entry(payload.clone())
                    .or_default()
                    .insert(tensor.clone());
            }
            validate_indexed_shards(&path.join("model.safetensors.index.json"), &indexed_names)?;
        }
        Ok(shards)
    }

    /// Builds an admitted tensor-to-shard catalog without reading payload headers.
    ///
    /// The neutral weight store uses this path so it can validate and buffer only
    /// shards requested by the caller. Public discovery remains strict.
    pub(crate) fn discover_catalog(path: impl AsRef<Path>) -> Result<Self, SafetensorsShardError> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::discover_directory(path);
        }
        let payload = canonicalize(path)?;
        Ok(Self {
            payload_paths: vec![payload],
            tensor_locations: None,
        })
    }

    fn discover_directory(root: &Path) -> Result<Self, SafetensorsShardError> {
        let access_root = canonical_checkpoint_access_root(root)?;
        let index_path = root.join("model.safetensors.index.json");
        if !index_path.exists() {
            let payload = admit_payload(&root.join("model.safetensors"), &access_root)?;
            return Ok(Self {
                payload_paths: vec![payload],
                tensor_locations: None,
            });
        }

        let raw =
            std::fs::read_to_string(&index_path).map_err(|error| io_error(&index_path, error))?;
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
        let mut tensor_locations = BTreeMap::new();
        for (tensor, relative) in index.weight_map.0 {
            if tensor.is_empty() {
                return Err(SafetensorsShardError::MalformedIndex {
                    path: index_path.clone(),
                    message: "tensor names must not be empty".into(),
                });
            }
            let relative = validate_relative_shard_path(Path::new(&relative))?;
            let payload = admit_payload(&root.join(relative), &access_root)?;
            payload_paths.insert(payload.clone());
            tensor_locations.insert(tensor, payload);
        }
        Ok(Self {
            payload_paths: payload_paths.into_iter().collect(),
            tensor_locations: Some(tensor_locations),
        })
    }

    /// Returns the canonical, deterministically ordered payload paths.
    pub fn payload_paths(&self) -> &[PathBuf] {
        &self.payload_paths
    }

    /// Consumes the discovery result into canonical payload paths.
    pub fn into_payload_paths(self) -> Vec<PathBuf> {
        self.payload_paths
    }

    /// Returns indexed tensor-to-canonical-shard mappings.
    ///
    /// `None` denotes a direct file or an unindexed `model.safetensors` file.
    pub fn tensor_locations(&self) -> Option<&BTreeMap<String, PathBuf>> {
        self.tensor_locations.as_ref()
    }
}

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
        let shards = SafetensorsShards::discover(path)?;
        let mut tensors = BTreeMap::new();
        for shard in shards.payload_paths() {
            for (name, metadata) in read_tensor_metadata(shard)? {
                if tensors.insert(name.clone(), metadata).is_some() {
                    return Err(malformed_shard(
                        shard,
                        format!("tensor {name:?} occurs in more than one admitted shard"),
                    ));
                }
            }
        }
        if let Some(locations) = shards.tensor_locations() {
            if let Some(name) = tensors.keys().find(|name| !locations.contains_key(*name)) {
                return Err(malformed_shard(
                    tensors[name]
                        .backing_shard
                        .as_deref()
                        .expect("SafeTensors metadata retains shard provenance"),
                    format!("tensor {name:?} is absent from the admitted index"),
                ));
            }
            for (name, expected_shard) in locations {
                let actual_shard = tensors
                    .get(name)
                    .and_then(|metadata| metadata.backing_shard.as_ref());
                if actual_shard != Some(expected_shard) {
                    return Err(malformed_shard(
                        expected_shard,
                        format!(
                            "indexed tensor {name:?} retained unexpected shard provenance {actual_shard:?}"
                        ),
                    ));
                }
            }
        }
        Ok(Self { shards, tensors })
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

fn validate_indexed_shards(
    index_path: &Path,
    indexed_names: &BTreeMap<PathBuf, BTreeSet<String>>,
) -> Result<(), SafetensorsShardError> {
    for (shard, expected) in indexed_names {
        let actual = read_tensor_names(shard)?;
        if let Some(tensor) = expected.difference(&actual).next() {
            return Err(SafetensorsShardError::MalformedIndex {
                path: index_path.to_path_buf(),
                message: format!(
                    "weight_map assigns tensor {tensor:?} to {}, but that shard does not contain it",
                    shard.display()
                ),
            });
        }
        if let Some(tensor) = actual.difference(expected).next() {
            return Err(SafetensorsShardError::MalformedIndex {
                path: index_path.to_path_buf(),
                message: format!(
                    "shard {} contains tensor {tensor:?}, but weight_map does not assign it to that shard",
                    shard.display()
                ),
            });
        }
    }
    Ok(())
}

fn read_tensor_names(path: &Path) -> Result<BTreeSet<String>, SafetensorsShardError> {
    let mut file = File::open(path).map_err(|error| io_error(path, error))?;
    let file_len = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)
        .map_err(|error| malformed_shard(path, error.to_string()))?;
    let header_len = u64::from_le_bytes(length);
    if header_len > MAX_HEADER_BYTES {
        return Err(malformed_shard(
            path,
            format!("header exceeds {MAX_HEADER_BYTES} bytes"),
        ));
    }
    let payload_start = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| malformed_shard(path, "header length overflow"))?;
    if payload_start > file_len {
        return Err(malformed_shard(path, "header exceeds shard length"));
    }
    let header_len = usize::try_from(header_len)
        .map_err(|_| malformed_shard(path, "header length overflows usize"))?;
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)
        .map_err(|error| malformed_shard(path, error.to_string()))?;
    let raw = serde_json::from_slice::<BTreeMap<String, serde_json::Value>>(&header)
        .map_err(|error| malformed_shard(path, error.to_string()))?;
    Ok(raw
        .into_keys()
        .filter(|name| name != "__metadata__")
        .collect())
}

fn read_tensor_metadata(
    path: &Path,
) -> Result<BTreeMap<String, TensorMetadata>, SafetensorsShardError> {
    let mut file = File::open(path).map_err(|error| io_error(path, error))?;
    let file_len = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)
        .map_err(|error| malformed_shard(path, error.to_string()))?;
    let header_len = u64::from_le_bytes(length);
    if header_len > MAX_HEADER_BYTES {
        return Err(malformed_shard(
            path,
            format!("header exceeds {MAX_HEADER_BYTES} bytes"),
        ));
    }
    let payload_start = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| malformed_shard(path, "header length overflow"))?;
    if payload_start > file_len {
        return Err(malformed_shard(path, "header exceeds shard length"));
    }
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| malformed_shard(path, "header length overflows usize"))?;
    let mut header = vec![0_u8; header_len_usize];
    file.read_exact(&mut header)
        .map_err(|error| malformed_shard(path, error.to_string()))?;
    let metadata: Metadata = serde_json::from_slice(&header)
        .map_err(|error| malformed_shard(path, error.to_string()))?;
    let encoded_len = u64::try_from(metadata.data_len())
        .map_err(|_| malformed_shard(path, "encoded payload length overflows u64"))?;
    let expected_file_len = payload_start
        .checked_add(encoded_len)
        .ok_or_else(|| malformed_shard(path, "shard length overflow"))?;
    if expected_file_len != file_len {
        return Err(malformed_shard(
            path,
            format!(
                "header describes {encoded_len} payload bytes, but shard length provides {}",
                file_len - payload_start
            ),
        ));
    }
    metadata
        .tensors()
        .into_iter()
        .map(|(name, info)| {
            let encoded_byte_len = info
                .data_offsets
                .1
                .checked_sub(info.data_offsets.0)
                .and_then(|length| u64::try_from(length).ok())
                .ok_or_else(|| malformed_shard(path, format!("tensor {name:?} length overflow")))?;
            Ok((
                name.clone(),
                TensorMetadata {
                    name,
                    logical_shape: info.shape.clone(),
                    physical_shape: info.shape.clone(),
                    stored_dtype: stored_dtype(info.dtype),
                    encoded_byte_len,
                    backing_shard: Some(path.to_path_buf()),
                },
            ))
        })
        .collect()
}

fn stored_dtype(dtype: Dtype) -> StoredDtype {
    match dtype {
        Dtype::BOOL => StoredDtype::Bool,
        Dtype::U8 => StoredDtype::U8,
        Dtype::I8 => StoredDtype::I8,
        Dtype::I16 => StoredDtype::I16,
        Dtype::U16 => StoredDtype::U16,
        Dtype::F16 => StoredDtype::F16,
        Dtype::BF16 => StoredDtype::BF16,
        Dtype::I32 => StoredDtype::I32,
        Dtype::U32 => StoredDtype::U32,
        Dtype::F32 => StoredDtype::F32,
        Dtype::F64 => StoredDtype::F64,
        Dtype::I64 => StoredDtype::I64,
        Dtype::U64 => StoredDtype::U64,
        Dtype::C64 => StoredDtype::C64,
        Dtype::F8_E4M3 => StoredDtype::F8E4M3,
        Dtype::F4 => StoredDtype::F4,
        Dtype::F8_E8M0 => StoredDtype::F8E8M0,
        Dtype::F8_E5M2 => StoredDtype::F8E5M2,
        other => StoredDtype::Other(format!("{other:?}")),
    }
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
    use std::io::Write;

    use super::*;
    use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

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
        std::fs::write(&blob, []).unwrap();
        symlink("../../blobs/payload", snapshot.join("model.safetensors")).unwrap();

        let shards = SafetensorsShards::discover(&snapshot).unwrap();
        assert_eq!(shards.payload_paths(), [blob.canonicalize().unwrap()]);
    }
}
