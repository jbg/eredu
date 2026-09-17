//! Fixed cold reader backing. Parsing and tensor payload policy stay in their
//! existing Read + Seek workers; this module only supplies private buffering.
use super::*;
use std::collections::TryReserveError;

/// Initialized source-owned file-reader scratch. Preparation is cold allocation,
/// not an admission grant. Only the fixed reader transport can borrow its bytes.
#[derive(Debug)]
pub struct ReaderBuffer {
    bytes: Vec<u8>,
}
/// Actual cold reserve failure, retaining the destination's existing backing.
#[derive(Debug)]
pub struct ReaderBufferPreparationFailure {
    cause: TryReserveError,
    buffer: ReaderBuffer,
}
impl ReaderBufferPreparationFailure {
    /// Preserve the original allocator error and complete partial owner.
    pub fn into_parts(self) -> (TryReserveError, ReaderBuffer) {
        (self.cause, self.buffer)
    }
    /// Actual backing retained by the refused preparation.
    pub fn buffer(&self) -> &ReaderBuffer {
        &self.buffer
    }
}
impl std::fmt::Display for ReaderBufferPreparationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for ReaderBufferPreparationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl ReaderBuffer {
    /// Requested fresh backing and concrete preparation/result transports.
    /// This describes the existing worker; it does not certify allocator capacity.
    pub fn preparation_request() -> Option<(std::alloc::Layout, usize)> {
        use std::mem::size_of;
        let layout =
            std::alloc::Layout::array::<u8>(Reader::<BufReader<File>>::file_buffer_capacity())
                .ok()?;
        let parts = [
            size_of::<Self>(),
            size_of::<ReaderBufferPreparationFailure>(),
            size_of::<std::result::Result<Self, ReaderBufferPreparationFailure>>(),
            size_of::<usize>(),
            size_of::<&mut Vec<u8>>(),
            size_of::<std::result::Result<(), TryReserveError>>(),
        ];
        Some((
            layout,
            parts
                .into_iter()
                .try_fold(std::mem::size_of_val(&parts), usize::checked_add)?,
        ))
    }

    #[cfg(test)]
    pub(crate) fn address(&self) -> usize {
        self.bytes.as_ptr() as usize
    }
    /// Reserve and initialize the source-derived fixed file-reader capacity.
    /// Any allocation prefix is returned on failure. No file is opened.
    pub fn prepare() -> std::result::Result<Self, ReaderBufferPreparationFailure> {
        let mut buffer = Self { bytes: Vec::new() };
        let length = Reader::<BufReader<File>>::file_buffer_capacity();
        if let Err(cause) = buffer.bytes.try_reserve_exact(length) {
            return Err(ReaderBufferPreparationFailure { cause, buffer });
        }
        buffer.bytes.resize(length, 0);
        Ok(buffer)
    }
    /// Actual owned capacity; spare capacity remains part of source inventory.
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }
    /// Whether this owner has the complete initialized reader extent.
    pub fn is_prepared(&self) -> bool {
        self.bytes.len() == Reader::<BufReader<File>>::file_buffer_capacity()
    }
}

pub(crate) enum FileReader {
    Ordinary(BufReader<File>),
    Prepared(Buffered<File>),
}
impl FileReader {
    #[cfg(test)]
    pub(crate) fn buffer_address(&self) -> Option<usize> {
        match self {
            Self::Ordinary(_) => None,
            Self::Prepared(reader) => Some(reader.buffer.address()),
        }
    }
    pub(crate) fn prepared(file: File, buffer: ReaderBuffer) -> Self {
        Self::Prepared(Buffered::new(file, buffer))
    }
    pub(crate) fn into_buffer(self) -> Option<ReaderBuffer> {
        match self {
            Self::Ordinary(reader) => {
                drop(reader);
                None
            }
            Self::Prepared(reader) => {
                let Buffered { inner, buffer, .. } = reader;
                // The actual File closes before its scratch returns for reuse.
                drop(inner);
                Some(buffer)
            }
        }
    }
}
impl Read for FileReader {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Ordinary(reader) => reader.read(bytes),
            Self::Prepared(reader) => reader.read(bytes),
        }
    }
    fn read_exact(&mut self, bytes: &mut [u8]) -> std::io::Result<()> {
        match self {
            Self::Ordinary(reader) => reader.read_exact(bytes),
            Self::Prepared(reader) => reader.read_exact(bytes),
        }
    }
}
impl Seek for FileReader {
    fn seek(&mut self, offset: SeekFrom) -> std::io::Result<u64> {
        match self {
            Self::Ordinary(reader) => reader.seek(offset),
            Self::Prepared(reader) => reader.seek(offset),
        }
    }
    fn stream_position(&mut self) -> std::io::Result<u64> {
        match self {
            Self::Ordinary(reader) => reader.stream_position(),
            Self::Prepared(reader) => reader.stream_position(),
        }
    }
}

// Only this owner sees the initialized slice. No growing Vec or byte alias
// escapes to a checkpoint lease or converted output.
pub(crate) struct Buffered<R> {
    inner: R,
    buffer: ReaderBuffer,
    position: usize,
    filled: usize,
}

#[cfg(test)]
mod tests;
impl<R> Buffered<R> {
    fn new(inner: R, buffer: ReaderBuffer) -> Self {
        assert!(buffer.is_prepared(), "checked complete reader buffer");
        Self {
            inner,
            buffer,
            position: 0,
            filled: 0,
        }
    }
    fn discard(&mut self) {
        self.position = 0;
        self.filled = 0;
    }
}
impl<R: Read> Read for Buffered<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let capacity = self.buffer.bytes.len();
        if self.position == self.filled && out.len() >= capacity {
            self.discard();
            return self.inner.read(out);
        }
        if self.position >= self.filled {
            let count = self.inner.read(&mut self.buffer.bytes)?;
            self.position = 0;
            self.filled = count;
        }
        let count = out.len().min(self.filled - self.position);
        out[..count].copy_from_slice(&self.buffer.bytes[self.position..self.position + count]);
        self.position += count;
        Ok(count)
    }
}
impl<R: Seek> Seek for Buffered<R> {
    fn seek(&mut self, target: SeekFrom) -> std::io::Result<u64> {
        let result = if let SeekFrom::Current(offset) = target {
            // Fixed reader capacity is 8192, so the buffered remainder fits i64.
            let remainder = (self.filled - self.position) as i64;
            if let Some(offset) = offset.checked_sub(remainder) {
                self.inner.seek(SeekFrom::Current(offset))?
            } else {
                self.inner.seek(SeekFrom::Current(-remainder))?;
                self.discard();
                self.inner.seek(SeekFrom::Current(offset))?
            }
        } else {
            self.inner.seek(target)?
        };
        self.discard();
        Ok(result)
    }
    fn stream_position(&mut self) -> std::io::Result<u64> {
        let remainder = (self.filled - self.position) as u64;
        self.inner.stream_position().map(|position| {
            position
                .checked_sub(remainder)
                .expect("buffered position is within underlying file")
        })
    }
}
