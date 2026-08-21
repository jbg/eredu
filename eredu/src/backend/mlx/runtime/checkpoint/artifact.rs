//! Stable content identities for loaded checkpoint artifacts.

use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use sha2::{Digest, Sha256};

use crate::backend::mlx::error::Error;

/// Immutable identity attached to a loaded model instance.
///
/// Loaded artifacts are identified by a digest of their exact content and
/// logical file layout.
#[derive(Clone, Eq, Hash, PartialEq)]
pub(crate) enum LoadedArtifactIdentity {
    Content([u8; 32]),
}

impl std::fmt::Debug for LoadedArtifactIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Content(digest) => write!(formatter, "sha256:{}", hex(digest)),
        }
    }
}

/// One file and its stable logical name within a checkpoint artifact.
#[derive(Debug, Clone)]
pub(crate) struct ArtifactFile {
    pub(crate) logical_name: String,
    pub(crate) path: PathBuf,
}

impl ArtifactFile {
    pub(crate) fn new(logical_name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            logical_name: logical_name.into(),
            path: path.into(),
        }
    }
}

/// Hashes the exact bytes and logical layout of a selected checkpoint artifact.
///
/// Logical names make the result independent of the directory containing the
/// checkpoint while keeping distinct shard layouts and file roles distinct.
pub(crate) fn fingerprint_artifact(
    domain: &str,
    files: impl IntoIterator<Item = ArtifactFile>,
) -> Result<LoadedArtifactIdentity, Error> {
    let mut files = files.into_iter().collect::<Vec<_>>();
    files.sort_unstable_by(|left, right| left.logical_name.cmp(&right.logical_name));
    if files.is_empty() {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "checkpoint artifact contains no files",
        )));
    }
    for pair in files.windows(2) {
        if pair[0].logical_name == pair[1].logical_name {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "duplicate checkpoint artifact role {:?}",
                    pair[0].logical_name
                ),
            )));
        }
    }

    let mut hasher = Sha256::new();
    hash_component(&mut hasher, b"safemlx-checkpoint-artifact-v1");
    hash_component(&mut hasher, domain.as_bytes());
    hasher.update((files.len() as u64).to_le_bytes());
    let mut buffer = vec![0u8; 1024 * 1024];
    for file in files {
        hash_component(&mut hasher, file.logical_name.as_bytes());
        let opened =
            File::open(&file.path).map_err(|source| contextual_io("open", &file.path, source))?;
        let length = opened
            .metadata()
            .map_err(|source| contextual_io("inspect", &file.path, source))?
            .len();
        hasher.update(length.to_le_bytes());
        let mut reader = BufReader::with_capacity(buffer.len(), opened);
        let mut total = 0u64;
        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|source| contextual_io("read", &file.path, source))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            total = total.checked_add(read as u64).ok_or_else(|| {
                Error::Io(std::io::Error::other(format!(
                    "checkpoint artifact {} exceeds supported size",
                    file.path.display()
                )))
            })?;
        }
        if total != length {
            return Err(Error::Io(std::io::Error::other(format!(
                "checkpoint artifact {} changed size while being fingerprinted: expected {length} bytes, read {total}",
                file.path.display()
            ))));
        }
    }
    Ok(LoadedArtifactIdentity::Content(hasher.finalize().into()))
}

fn contextual_io(action: &str, path: &std::path::Path, source: std::io::Error) -> Error {
    Error::Io(std::io::Error::new(
        source.kind(),
        format!("{action} checkpoint artifact {}: {source}", path.display()),
    ))
}

fn hash_component(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(DIGITS[usize::from(byte >> 4)] as char);
        output.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn fingerprint_is_content_exact_and_location_independent() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(first.path().join("weights.safetensors"), b"weights-a").unwrap();
        fs::write(second.path().join("weights.safetensors"), b"weights-a").unwrap();

        let identity = |path: PathBuf| {
            fingerprint_artifact("test", [ArtifactFile::new("weights", path)]).unwrap()
        };
        let original = identity(first.path().join("weights.safetensors"));
        assert_eq!(
            original,
            identity(second.path().join("weights.safetensors"))
        );

        fs::write(second.path().join("weights.safetensors"), b"weights-b").unwrap();
        assert_ne!(
            original,
            identity(second.path().join("weights.safetensors"))
        );
    }

    #[test]
    fn fingerprint_includes_logical_file_layout() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tensor");
        fs::write(&path, b"same-bytes").unwrap();
        let weight = fingerprint_artifact("test", [ArtifactFile::new("weight", &path)]).unwrap();
        let scale = fingerprint_artifact("test", [ArtifactFile::new("scale", &path)]).unwrap();
        assert_ne!(weight, scale);
    }
}
