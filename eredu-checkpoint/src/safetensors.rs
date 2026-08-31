//! Canonical discovery and path admission for SafeTensors checkpoints.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path, PathBuf},
};

use serde::{de::MapAccess, Deserialize, Deserializer};

/// One canonically resolved SafeTensors checkpoint shard set.
///
/// Discovery parses an optional Hugging Face index exactly once, rejects
/// duplicate or empty tensor mappings, and resolves every payload beneath the
/// checkpoint access root. Snapshot payload symlinks may target the sibling
/// repository `blobs` directory, but no path may escape that repository.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SafetensorsShards {
    payload_paths: Vec<PathBuf>,
    tensor_locations: Option<BTreeMap<String, PathBuf>>,
}

impl SafetensorsShards {
    /// Discovers and admits a SafeTensors file or checkpoint directory.
    pub fn discover(path: impl AsRef<Path>) -> Result<Self, SafetensorsShardError> {
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
    use super::*;

    fn write_index(root: &Path, contents: &str) {
        std::fs::write(root.join("model.safetensors.index.json"), contents).unwrap();
    }

    #[test]
    fn indexed_discovery_is_unique_canonical_and_deterministic() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("z.safetensors"), []).unwrap();
        std::fs::write(root.path().join("a.safetensors"), []).unwrap();
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
