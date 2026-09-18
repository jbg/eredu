//! Filesystem-backed artifact content fingerprinting.

pub(crate) mod file;
mod fingerprint;
use file::FileVersion as StableFileMetadata;
pub use file::{
    ArtifactFileReadError, ArtifactFileReadFailure, ArtifactFileVersion, PreparedArtifactFileRead,
};
pub use fingerprint::{ArtifactFingerprintAllocation, ArtifactFingerprintPreparationError};

use sha2::{Digest as _, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/// One filesystem member and its location-independent logical artifact role.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArtifactFile {
    logical_role: String,
    path: PathBuf,
}

impl ArtifactFile {
    /// Creates a logical artifact member.
    pub fn new(logical_role: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            logical_role: logical_role.into(),
            path: path.into(),
        }
    }

    /// Stable role used in the artifact layout.
    pub fn logical_role(&self) -> &str {
        &self.logical_role
    }

    /// Filesystem path supplying this member's bytes.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Exact content facts for one logical artifact member.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArtifactMemberFingerprint {
    logical_role: String,
    length: u64,
    digest: [u8; 32],
}

impl ArtifactMemberFingerprint {
    /// Moves the original role backing into the canonical identity reducer.
    pub fn into_parts(self) -> (String, u64, [u8; 32]) {
        (self.logical_role, self.length, self.digest)
    }
    /// Stable logical role of the member.
    pub fn logical_role(&self) -> &str {
        &self.logical_role
    }

    /// Exact admitted byte length.
    pub const fn length(&self) -> u64 {
        self.length
    }

    /// SHA-256 of the exact admitted bytes.
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct FileContentFingerprint {
    pub(crate) length: u64,
    pub(crate) digest: [u8; 32],
}

/// Filesystem-backed artifact fingerprinting failure.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactFingerprintError {
    /// A source destination size cannot be represented.
    #[error("artifact fingerprint storage size overflow")]
    StorageOverflow,
    /// The actual host allocation was refused.
    #[error("artifact fingerprint allocation failed: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    /// The allocator returned a different destination capacity.
    #[error("artifact fingerprint destination capacity differs from its request")]
    Capacity,
    /// Opening, inspecting, or reading one member failed.
    #[error("failed to {action} artifact file {path}: {source}", path = .path.display())]
    Io {
        /// Failed operation.
        action: &'static str,
        /// Affected filesystem member.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// The source changed while a stable content snapshot was being read.
    #[error("artifact file changed while being fingerprinted: {path}", path = .path.display())]
    Changed {
        /// Affected filesystem member.
        path: PathBuf,
    },
}

/// Admitted filesystem members whose payload fingerprint can be requested later.
/// Construction records metadata without reading payloads or retaining open files.
#[derive(Debug)]
pub struct ArtifactFingerprintSource {
    files: Vec<AdmittedArtifactFile>,
}

#[derive(Debug)]
struct AdmittedArtifactFile {
    member: ArtifactFile,
    admitted: StableFileMetadata,
}

impl ArtifactFingerprintSource {
    /// Records original file identities and cheap change-detection facts.
    pub fn new(
        files: impl IntoIterator<Item = ArtifactFile>,
    ) -> Result<Self, ArtifactFingerprintError> {
        let files = files
            .into_iter()
            .map(|member| {
                let file = File::open(member.path())
                    .map_err(|source| io_error("open", member.path(), source))?;
                let admitted = StableFileMetadata::read(member.path(), &file)?;
                Ok(AdmittedArtifactFile { member, admitted })
            })
            .collect::<Result<_, ArtifactFingerprintError>>()?;
        Ok(Self { files })
    }

    /// Reads each complete file once, rejecting changes since admission.
    pub fn fingerprint(&self) -> Result<Vec<ArtifactMemberFingerprint>, ArtifactFingerprintError> {
        fingerprint::source(self, &fingerprint::Unenforced).map_err(fingerprint::ordinary)
    }
    /// Runs the same file/version/hash worker with prospective destinations.
    /// The source was already admitted by `new`; this neither adopts another
    /// file nor changes its original observable-version checks. The enclosing
    /// caller retains allocation custody in every result and escaping error.
    #[cfg(unix)]
    pub fn fingerprint_with_allocations<A: ArtifactFingerprintAllocation>(
        &self,
        allocation: &A,
    ) -> Result<Vec<ArtifactMemberFingerprint>, ArtifactFingerprintPreparationError<A::Error>> {
        fingerprint::source(self, allocation)
    }
}

/// Reads content fingerprints with one sequential pass per file.
/// Metadata checks before and after the pass detect observable file changes.
pub fn fingerprint_artifact_files(
    files: impl IntoIterator<Item = ArtifactFile>,
) -> Result<Vec<ArtifactMemberFingerprint>, ArtifactFingerprintError> {
    ArtifactFingerprintSource::new(files)?.fingerprint()
}

fn fingerprint_open_file_with_hook(
    path: &Path,
    file: &File,
    admitted: StableFileMetadata,
    after_pass: impl FnOnce(),
) -> Result<FileContentFingerprint, ArtifactFingerprintError> {
    fingerprint::open_file(path, file, admitted, after_pass, &fingerprint::Unenforced)
        .map_err(fingerprint::ordinary)
}

impl StableFileMetadata {
    fn read(path: &Path, file: &File) -> Result<Self, ArtifactFingerprintError> {
        let metadata = file
            .metadata()
            .map_err(|source| io_error("inspect", path, source))?;
        Self::from_metadata(&metadata).map_err(|source| io_error("inspect", path, source))
    }
}

fn digest_pass(
    path: &Path,
    file: &mut (impl Read + Seek),
) -> Result<FileContentFingerprint, ArtifactFingerprintError> {
    fingerprint::digest(path, file, &fingerprint::Unenforced).map_err(fingerprint::ordinary)
}

fn io_error(action: &'static str, path: &Path, source: std::io::Error) -> ArtifactFingerprintError {
    ArtifactFingerprintError::Io {
        action,
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn digest_reads_each_byte_in_one_pass() {
        struct Counted {
            cursor: std::io::Cursor<Vec<u8>>,
            bytes: usize,
            seeks: usize,
        }
        impl Read for Counted {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                let count = self.cursor.read(output)?;
                self.bytes += count;
                Ok(count)
            }
        }
        impl Seek for Counted {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                self.seeks += 1;
                self.cursor.seek(position)
            }
        }
        let payload = vec![37; 2 * 1024 * 1024 + 7];
        let expected: [u8; 32] = Sha256::digest(&payload).into();
        let mut input = Counted {
            cursor: std::io::Cursor::new(payload),
            bytes: 0,
            seeks: 0,
        };
        let fingerprint = digest_pass(Path::new("counted"), &mut input).unwrap();
        assert_eq!(fingerprint.digest, expected);
        assert_eq!(fingerprint.length as usize, input.bytes);
        assert_eq!(input.bytes, input.cursor.get_ref().len());
        assert_eq!(input.seeks, 1);
    }

    #[test]
    fn deferred_source_rejects_replaced_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("member");
        std::fs::write(&path, b"original").unwrap();
        let source = ArtifactFingerprintSource::new([ArtifactFile::new("weights", &path)]).unwrap();
        let replacement = directory.path().join("replacement");
        std::fs::write(&replacement, b"different file contents").unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        assert!(matches!(
            source.fingerprint(),
            Err(ArtifactFingerprintError::Changed { .. })
        ));
    }

    #[test]
    #[cfg(unix)]
    fn metadata_check_rejects_same_length_change_with_restored_mtime() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("member");
        std::fs::write(&path, b"first-content").unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        let admitted = File::open(&path).unwrap();
        let result = fingerprint_open_file_with_hook(
            &path,
            &admitted,
            StableFileMetadata::read(&path, &admitted).unwrap(),
            || {
                let mut attacker = File::options()
                    .write(true)
                    .truncate(true)
                    .open(&path)
                    .unwrap();
                attacker.write_all(b"other-content").unwrap();
                attacker
                    .set_times(std::fs::FileTimes::new().set_modified(modified))
                    .unwrap();
            },
        );
        assert!(matches!(
            result,
            Err(ArtifactFingerprintError::Changed { .. })
        ));
    }
}
