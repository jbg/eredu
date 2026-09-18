//! The original stable-file/hash pass with prospective source destinations.
use super::*;
use std::{alloc::Layout, convert::Infallible, ffi::OsString, mem::size_of};

/// Allocation facts supplied before the original fingerprint producer.
/// The callback must return a fixed refusal and retain accepted bytes in the
/// enclosing source account; it supplies no filesystem or execution authority.
pub trait ArtifactFingerprintAllocation {
    /// Concrete fixed refusal owned by the caller's account.
    type Error: std::error::Error + 'static;
    /// Admit the next destination and its concrete control values.
    fn reserve(&self, bytes: usize) -> Result<(), Self::Error>;
}
/// Source and storage failures remain typed; none requires formatting a refusal.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactFingerprintPreparationError<E: std::error::Error + 'static> {
    /// The same original version or filesystem failure.
    #[error("{0}")]
    Source(#[source] ArtifactFingerprintError),
    /// The original account refused this destination.
    #[error("{0}")]
    Funding(#[source] E),
    /// Destination or control size overflow.
    #[error("artifact fingerprint storage size overflow")]
    Overflow,
    /// The host allocator refused the destination.
    #[error("artifact fingerprint allocation failed: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    /// The host allocator supplied a different capacity.
    #[error("artifact fingerprint destination capacity differs from its request")]
    Capacity,
}
type Failure<A> = ArtifactFingerprintPreparationError<<A as ArtifactFingerprintAllocation>::Error>;
pub(super) struct Unenforced;
impl ArtifactFingerprintAllocation for Unenforced {
    type Error = Infallible;
    fn reserve(&self, _: usize) -> Result<(), Infallible> {
        Ok(())
    }
}
pub(super) fn ordinary(error: Failure<Unenforced>) -> ArtifactFingerprintError {
    use ArtifactFingerprintPreparationError as E;
    match error {
        E::Source(error) => error,
        E::Funding(error) => match error {},
        E::Overflow => ArtifactFingerprintError::StorageOverflow,
        E::Allocation(error) => ArtifactFingerprintError::Allocation(error),
        E::Capacity => ArtifactFingerprintError::Capacity,
    }
}
struct Destination<'a, A>(&'a A);
impl<A: ArtifactFingerprintAllocation> Destination<'_, A> {
    fn controls<T>(&self) -> Result<(), Failure<A>> {
        let bytes = size_of::<(T, &Self, Result<T, Failure<A>>, usize)>();
        self.0
            .reserve(bytes)
            .map_err(ArtifactFingerprintPreparationError::Funding)
    }
    fn vector<T>(&self, count: usize) -> Result<Vec<T>, Failure<A>> {
        self.controls::<(Vec<T>, usize, Layout, std::collections::TryReserveError)>()?;
        self.0
            .reserve(
                Layout::array::<T>(count)
                    .map_err(|_| ArtifactFingerprintPreparationError::Overflow)?
                    .size(),
            )
            .map_err(ArtifactFingerprintPreparationError::Funding)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(ArtifactFingerprintPreparationError::Allocation)?;
        if size_of::<T>() != 0 && result.capacity() != count {
            return Err(ArtifactFingerprintPreparationError::Capacity);
        }
        Ok(result)
    }
    fn text(&self, source: &str) -> Result<String, Failure<A>> {
        self.controls::<(&str, String, Result<String, std::string::FromUtf8Error>)>()?;
        let mut result = self.vector(source.len())?;
        result.extend_from_slice(source.as_bytes());
        Ok(String::from_utf8(result).expect("copied UTF-8"))
    }
    fn path(&self, source: &Path) -> Result<PathBuf, Failure<A>> {
        self.controls::<(&Path, PathBuf, OsString, std::collections::TryReserveError)>()?;
        let count = source.as_os_str().len();
        self.0
            .reserve(count)
            .map_err(ArtifactFingerprintPreparationError::Funding)?;
        let mut path = OsString::new();
        path.try_reserve_exact(count)
            .map_err(ArtifactFingerprintPreparationError::Allocation)?;
        if path.capacity() != count {
            return Err(ArtifactFingerprintPreparationError::Capacity);
        }
        path.push(source);
        Ok(path.into())
    }
    fn io(&self, action: &'static str, path: &Path, source: std::io::Error) -> Failure<A> {
        match self.path(path) {
            Ok(path) => ArtifactFingerprintPreparationError::Source(ArtifactFingerprintError::Io {
                action,
                path,
                source,
            }),
            Err(error) => error,
        }
    }
    fn changed(&self, path: &Path) -> Failure<A> {
        match self.path(path) {
            Ok(path) => {
                ArtifactFingerprintPreparationError::Source(ArtifactFingerprintError::Changed {
                    path,
                })
            }
            Err(error) => error,
        }
    }
    fn metadata(&self, path: &Path, file: &File) -> Result<StableFileMetadata, Failure<A>> {
        self.controls::<(
            &Path,
            &File,
            std::fs::Metadata,
            StableFileMetadata,
            std::io::Error,
        )>()?;
        let metadata = file
            .metadata()
            .map_err(|error| self.io("inspect", path, error))?;
        StableFileMetadata::from_metadata(&metadata)
            .map_err(|error| self.io("inspect", path, error))
    }
    fn open(&self, path: &Path) -> Result<File, Failure<A>> {
        self.controls::<(&Path, File, std::io::Error)>()?;
        #[cfg(unix)]
        {
            use std::{ffi::CStr, os::unix::ffi::OsStrExt};
            self.controls::<(&CStr, rustix::fd::OwnedFd, rustix::io::Errno)>()?;
            let bytes = path.as_os_str().as_bytes();
            let count = bytes
                .len()
                .checked_add(1)
                .ok_or(ArtifactFingerprintPreparationError::Overflow)?;
            let mut terminated = self.vector(count)?;
            terminated.extend_from_slice(bytes);
            terminated.push(0);
            // `ArtifactFingerprintSource::new` admitted this immutable path by
            // opening it already; an interior NUL could never pass that entry.
            let path_bytes =
                CStr::from_bytes_with_nul(&terminated).expect("admitted path has no NUL");
            self.open_prepared(path, || {
                rustix::fs::open(
                    path_bytes,
                    rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::CLOEXEC,
                    rustix::fs::Mode::empty(),
                )
            })
        }
        // Ordinary loaders on other hosts keep their original capability. The
        // public funded fingerprint entry is qualified only on Unix, matching
        // the existing original positional-file input contract.
        #[cfg(not(unix))]
        {
            File::open(path).map_err(|error| self.io("open", path, error))
        }
    }
    #[cfg(unix)]
    fn open_prepared<F>(&self, path: &Path, open: F) -> Result<File, Failure<A>>
    where
        F: FnMut() -> rustix::io::Result<rustix::fd::OwnedFd>,
    {
        self.controls::<(
            &Path,
            F,
            rustix::io::Result<rustix::fd::OwnedFd>,
            File,
            std::io::Error,
        )>()?;
        // File::open retries EINTR. Retain that behavior on this same syscall
        // worker without rebuilding the paid path or accumulating retry storage.
        rustix::io::retry_on_intr(open)
            .map(File::from)
            .map_err(|error| self.io("open", path, error.into()))
    }
}

pub(super) fn source<A: ArtifactFingerprintAllocation>(
    source: &ArtifactFingerprintSource,
    allocation: &A,
) -> Result<Vec<ArtifactMemberFingerprint>, Failure<A>> {
    let destination = Destination(allocation);
    destination.controls::<(
        &ArtifactFingerprintSource,
        std::slice::Iter<'_, AdmittedArtifactFile>,
        ArtifactMemberFingerprint,
        Option<&AdmittedArtifactFile>,
        Vec<u8>,
    )>()?;
    let mut members = destination.vector(source.files.len())?;
    let mut buffer = Vec::new();
    for pinned in &source.files {
        let path = pinned.member.path();
        let file = destination.open(path)?;
        let fingerprint =
            open_file_with_buffer(path, &file, pinned.admitted, || {}, allocation, &mut buffer)?;
        members.push(ArtifactMemberFingerprint {
            logical_role: destination.text(pinned.member.logical_role())?,
            length: fingerprint.length,
            digest: fingerprint.digest,
        });
    }
    Ok(members)
}
pub(super) fn open_file<A: ArtifactFingerprintAllocation>(
    path: &Path,
    file: &File,
    admitted: StableFileMetadata,
    after_pass: impl FnOnce(),
    allocation: &A,
) -> Result<FileContentFingerprint, Failure<A>> {
    let mut buffer = Vec::new();
    open_file_with_buffer(path, file, admitted, after_pass, allocation, &mut buffer)
}
fn open_file_with_buffer<A: ArtifactFingerprintAllocation, F: FnOnce()>(
    path: &Path,
    file: &File,
    admitted: StableFileMetadata,
    after_pass: F,
    allocation: &A,
    buffer: &mut Vec<u8>,
) -> Result<FileContentFingerprint, Failure<A>> {
    let destination = Destination(allocation);
    destination.controls::<(
        &Path,
        &File,
        StableFileMetadata,
        FileContentFingerprint,
        &mut Vec<u8>,
        F,
    )>()?;
    let before = destination.metadata(path, file)?;
    if before != admitted {
        return Err(destination.changed(path));
    }
    let fingerprint = digest_with_buffer(path, &mut &*file, allocation, buffer)?;
    after_pass();
    let after = destination.metadata(path, file)?;
    if before != after || fingerprint.length != before.length {
        return Err(destination.changed(path));
    }
    Ok(fingerprint)
}
pub(super) fn digest<A: ArtifactFingerprintAllocation>(
    path: &Path,
    file: &mut (impl Read + Seek),
    allocation: &A,
) -> Result<FileContentFingerprint, Failure<A>> {
    let mut buffer = Vec::new();
    digest_with_buffer(path, file, allocation, &mut buffer)
}
fn digest_with_buffer<A: ArtifactFingerprintAllocation>(
    path: &Path,
    file: &mut (impl Read + Seek),
    allocation: &A,
    buffer: &mut Vec<u8>,
) -> Result<FileContentFingerprint, Failure<A>> {
    let destination = Destination(allocation);
    destination.controls::<(
        &Path,
        Sha256,
        FileContentFingerprint,
        u64,
        usize,
        [u8; 32],
        sha2::digest::Output<Sha256>,
        std::io::Error,
        &mut Vec<u8>,
        SeekFrom,
        Result<u64, std::io::Error>,
        Result<usize, std::io::Error>,
        &[u8],
    )>()?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| destination.io("seek", path, error))?;
    let mut hasher = Sha256::new();
    let mut length = 0_u64;
    if buffer.is_empty() {
        *buffer = destination.vector(1024 * 1024)?;
        buffer.resize(1024 * 1024, 0_u8);
    }
    debug_assert_eq!(buffer.len(), 1024 * 1024);
    loop {
        let read = file
            .read(buffer)
            .map_err(|error| destination.io("read", path, error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        length = length
            .checked_add(read as u64)
            .ok_or_else(|| destination.changed(path))?;
    }
    Ok(FileContentFingerprint {
        length,
        digest: hasher.finalize().into(),
    })
}

#[cfg(all(test, unix))]
mod tests;
