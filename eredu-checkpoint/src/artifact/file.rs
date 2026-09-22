//! One retained regular-file handle and exact destination, without a payload owner.
use std::{alloc::Layout, fs::File, io, mem::size_of};

/// The existing observable file-version facts, shared with ordinary store reads.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct FileVersion {
    pub(crate) length: u64,
    modified: std::time::SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    change_time_seconds: i64,
    #[cfg(unix)]
    change_time_nanoseconds: i64,
    #[cfg(not(unix))]
    created: Option<std::time::SystemTime>,
    #[cfg(windows)]
    file_attributes: u32,
    #[cfg(windows)]
    creation_time: u64,
}
impl FileVersion {
    pub(crate) fn from_metadata(metadata: &std::fs::Metadata) -> io::Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt as _;
        Ok(Self {
            length: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            change_time_seconds: metadata.ctime(),
            #[cfg(unix)]
            change_time_nanoseconds: metadata.ctime_nsec(),
            #[cfg(not(unix))]
            created: metadata.created().ok(),
            #[cfg(windows)]
            file_attributes: {
                use std::os::windows::fs::MetadataExt as _;
                metadata.file_attributes()
            },
            #[cfg(windows)]
            creation_time: {
                use std::os::windows::fs::MetadataExt as _;
                metadata.creation_time()
            },
        })
    }
}

/// Fixed whole-file read rejection, or the unchanged operating-system error.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactFileReadError {
    /// This platform's positional read/allocation contract has not been closed.
    #[error("bounded artifact positional reads are not yet implemented for this platform")]
    IncompletePlatform,
    /// The handle does not identify a regular file with a fixed extent.
    #[error("bounded artifact input must be a regular file")]
    NotRegular,
    /// The extent cannot be represented by the host allocation or read offset.
    #[error("artifact file extent overflows the destination layout")]
    Overflow,
    /// A caller supplied a destination of a different exact length.
    #[error("artifact destination has the wrong exact length")]
    DestinationLength,
    /// Observable file-version facts changed before or during the read.
    #[error("artifact file changed during its prepared read")]
    Changed,
    /// The read reached EOF before filling the original extent.
    #[error("artifact file ended before its original extent")]
    Truncated,
    /// The one-byte probe observed data beyond the original extent.
    #[error("artifact file grew beyond its original extent")]
    Grown,
    /// Inspecting or reading the retained handle failed.
    #[error("artifact file operation failed: {0}")]
    Io(#[source] io::Error),
}

/// Immutable observable version captured from an actual regular-file handle.
///
/// This descriptor owns no file or path and grants no read/storage authority.
/// Its owner must retain the source publication; binding another handle compares
/// the same device/inode/time/extent facts used by ordinary artifact reads.
/// These checks detect observable changes, not an atomic filesystem snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactFileVersion {
    version: FileVersion,
    length: usize,
}
impl ArtifactFileVersion {
    /// Captures the actual handle after its source publication has completed.
    /// No payload or path allocation occurs. Unlike positional execution, this
    /// descriptive step is available on every supported host platform.
    pub fn capture(file: &File) -> Result<Self, ArtifactFileReadError> {
        let metadata = file.metadata().map_err(ArtifactFileReadError::Io)?;
        if !metadata.is_file() {
            return Err(ArtifactFileReadError::NotRegular);
        }
        let version = FileVersion::from_metadata(&metadata).map_err(ArtifactFileReadError::Io)?;
        let length =
            usize::try_from(version.length).map_err(|_| ArtifactFileReadError::Overflow)?;
        Layout::array::<u8>(length).map_err(|_| ArtifactFileReadError::Overflow)?;
        i64::try_from(version.length).map_err(|_| ArtifactFileReadError::Overflow)?;
        Ok(Self { version, length })
    }
    /// Exact byte extent observed on the retained source handle.
    pub fn byte_len(&self) -> usize {
        self.length
    }
    /// Fixed metadata inspection/binding controls; no file/path/destination allocation.
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<FileVersion>(),
            size_of::<std::fs::Metadata>(),
            size_of::<Result<std::fs::Metadata, io::Error>>(),
            size_of::<Result<FileVersion, io::Error>>(),
            size_of::<Result<Self, ArtifactFileReadError>>(),
            size_of::<Result<(), ArtifactFileReadError>>(),
            size_of::<Result<usize, std::num::TryFromIntError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Layout>(),
            size_of::<Result<i64, std::num::TryFromIntError>>(),
            size_of::<ArtifactFileReadError>(),
            size_of::<(&Self, &File)>(),
            size_of::<(Self, File)>(),
            size_of::<Result<PreparedArtifactFileRead, ArtifactFileReadError>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Verifies an actual handle against the captured version without taking it.
    pub fn validate(&self, file: &File) -> Result<(), ArtifactFileReadError> {
        if Self::capture(file)? != *self {
            return Err(ArtifactFileReadError::Changed);
        }
        Ok(())
    }
    /// Consumes a newly opened handle only after its exact source version agrees.
    /// The existing positional-read worker revalidates before and after transfer.
    pub fn bind(self, file: File) -> Result<PreparedArtifactFileRead, ArtifactFileReadError> {
        if !cfg!(unix) {
            return Err(ArtifactFileReadError::IncompletePlatform);
        }
        self.validate(&file)?;
        Ok(PreparedArtifactFileRead {
            file,
            version: self,
        })
    }
}

/// One move-only whole-file read, prepared from the actual retained handle.
///
/// Preparation allocates no payload, copies no path and grants no budget. The
/// caller owns opening/path preparation. Only audited Unix positional reads
/// are enabled; ordinary loaders on other platforms retain their behavior.
/// This preserves observable version checks, not an atomic filesystem snapshot.
///
/// ```compile_fail
/// # use eredu_checkpoint::artifact::PreparedArtifactFileRead;
/// fn copy(read: PreparedArtifactFileRead) { let _ = read.clone(); }
/// ```
#[derive(Debug)]
pub struct PreparedArtifactFileRead {
    file: File,
    version: ArtifactFileVersion,
}
impl PreparedArtifactFileRead {
    /// Consumes an already opened handle and seals its actual exact extent.
    pub fn new(file: File) -> Result<Self, ArtifactFileReadError> {
        if !cfg!(unix) {
            return Err(ArtifactFileReadError::IncompletePlatform);
        }
        let version = ArtifactFileVersion::capture(&file)?;
        Ok(Self { file, version })
    }
    /// Actual full-file destination length; no caller byte declaration is used.
    pub fn byte_len(&self) -> usize {
        self.version.byte_len()
    }
    /// Concrete finite read/return/control overlap, excluding the destination.
    pub fn control_bytes() -> Option<usize> {
        [
            ArtifactFileVersion::control_bytes()?,
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<FileVersion>(),
            size_of::<std::fs::Metadata>(),
            size_of::<Result<std::fs::Metadata, io::Error>>(),
            size_of::<Result<FileVersion, io::Error>>(),
            size_of::<Result<std::fs::Metadata, ArtifactFileReadError>>(),
            size_of::<Result<FileVersion, ArtifactFileReadError>>(),
            size_of::<Result<Self, ArtifactFileReadError>>(),
            size_of::<ArtifactFileReadError>(),
            size_of::<ArtifactFileReadFailure>(),
            size_of::<Result<(), ArtifactFileReadFailure>>(),
            size_of::<Result<(), ArtifactFileReadError>>(),
            size_of::<Result<usize, io::Error>>(),
            size_of::<Result<usize, ArtifactFileReadError>>(),
            size_of::<&mut [u8]>(), // exact destination/chunk borrow
            size_of::<&File>(),     // positional-read argument
            size_of::<usize>(),     // current chunk end
            size_of::<usize>(),     // returned byte count
            size_of::<u64>(),       // checked positional offset
            size_of::<ReadProgress>(),
            size_of::<[u8; 1]>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// Fixed streaming-copy frames, including its bounded stack byte buffer.
    /// The caller separately pays source binding and destination file ownership.
    pub fn copy_to_control_bytes<F>() -> Option<usize> {
        let frames = [
            Self::control_bytes()?,
            size_of::<[u8; 4096]>(),
            size_of::<F>(),
            size_of::<(usize, usize, usize, &mut File)>(),
            size_of::<Result<(), io::Error>>(),
            size_of::<Result<usize, io::Error>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Copies this exact source version through a bounded stack buffer. The
    /// observer receives source-file offsets and completed destination chunks.
    /// A failure retains the source handle and exact destination prefix length;
    /// the caller retains or discards the provisional destination file.
    pub fn copy_to_with<F>(
        self,
        destination: &mut File,
        mut observe: F,
    ) -> Result<(), ArtifactFileReadFailure>
    where
        F: FnMut(usize, &[u8]),
    {
        use std::io::Write as _;
        let mut progress = ReadProgress { filled: 0 };
        let result = (|| {
            self.validate()?;
            let mut bytes = [0u8; 4096];
            while progress.filled < self.version.length {
                let offset = progress.filled;
                let capacity = (self.version.length - offset).min(bytes.len());
                let read = read_at(&self.file, &mut bytes[..capacity], offset as u64)
                    .map_err(ArtifactFileReadError::Io)?;
                if read == 0 {
                    return Err(ArtifactFileReadError::Truncated);
                }
                let mut written = 0;
                while written < read {
                    match destination.write(&bytes[written..read]) {
                        Ok(0) => {
                            return Err(ArtifactFileReadError::Io(io::ErrorKind::WriteZero.into()));
                        }
                        Ok(count) => {
                            written += count;
                            progress.filled += count;
                        }
                        Err(cause) if cause.kind() == io::ErrorKind::Interrupted => {}
                        Err(cause) => return Err(ArtifactFileReadError::Io(cause)),
                    }
                }
                observe(offset, &bytes[..read]);
            }
            let mut probe = [0u8; 1];
            if read_at(&self.file, &mut probe, self.version.version.length)
                .map_err(ArtifactFileReadError::Io)?
                != 0
            {
                return Err(ArtifactFileReadError::Grown);
            }
            self.validate()
        })();
        result.map_err(|cause| ArtifactFileReadFailure {
            cause,
            progress,
            _source: self,
        })
    }

    fn validate(&self) -> Result<(), ArtifactFileReadError> {
        self.version.validate(&self.file)
    }
    /// Fills one exact caller-owned initialized destination without allocating.
    /// On failure the returned value retains this handle and reports the actual
    /// written prefix. The caller must retain/discard its destination accordingly.
    pub fn read_into(self, destination: &mut [u8]) -> Result<(), ArtifactFileReadFailure> {
        self.read_into_with(destination, |_| {})
    }
    fn read_into_with(
        self,
        destination: &mut [u8],
        mut after_chunk: impl FnMut(usize),
    ) -> Result<(), ArtifactFileReadFailure> {
        let mut progress = ReadProgress { filled: 0 };
        let result = (|| {
            if destination.len() != self.version.length {
                return Err(ArtifactFileReadError::DestinationLength);
            }
            self.validate()?;
            while progress.filled != destination.len() {
                let end = progress
                    .filled
                    .saturating_add(64 * 1024)
                    .min(destination.len());
                let count = read_at(
                    &self.file,
                    &mut destination[progress.filled..end],
                    progress.filled as u64,
                )
                .map_err(ArtifactFileReadError::Io)?;
                if count == 0 {
                    return Err(ArtifactFileReadError::Truncated);
                }
                progress.filled += count;
                after_chunk(progress.filled);
            }
            let mut probe = [0u8; 1];
            if read_at(&self.file, &mut probe, self.version.version.length)
                .map_err(ArtifactFileReadError::Io)?
                != 0
            {
                return Err(ArtifactFileReadError::Grown);
            }
            self.validate()
        })();
        match result {
            Ok(()) => Ok(()), // The same handle retires before the caller marks its read idle.
            Err(cause) => Err(ArtifactFileReadFailure {
                cause,
                progress,
                _source: self,
            }),
        }
    }
}

#[derive(Debug)]
struct ReadProgress {
    filled: usize,
}

/// A failed consuming read retaining the exact original handle and version.
#[derive(Debug)]
pub struct ArtifactFileReadFailure {
    cause: ArtifactFileReadError,
    progress: ReadProgress,
    _source: PreparedArtifactFileRead,
}
impl ArtifactFileReadFailure {
    /// Actual bytes written before the terminal failure; later bytes are invalid.
    pub fn filled_bytes(&self) -> usize {
        self.progress.filled
    }
    /// Borrows the unchanged fixed or operating-system cause.
    pub fn cause(&self) -> &ArtifactFileReadError {
        &self.cause
    }
}
impl std::fmt::Display for ArtifactFileReadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for ArtifactFileReadFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

fn read_at(file: &File, destination: &mut [u8], offset: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt as _;
        loop {
            match file.read_at(destination, offset) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => return result,
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (file, destination, offset);
        Err(io::ErrorKind::Unsupported.into())
    }
}

#[cfg(all(test, unix))]
mod tests;
