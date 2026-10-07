//! Bounded mono RIFF/WAVE input and lossless floating-point sample output.
//!
//! Container processing uses only the standard library and is available without
//! the `audio` feature. Sample rate conversion and timeline placement belong to
//! the caller. Unknown RIFF chunks are skipped without allocation.

use std::io::{Read, Seek, SeekFrom, Write};

/// Failure reading or writing a supported WAV container.
#[derive(Debug, thiserror::Error)]
pub enum WavError {
    /// An underlying stream operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The container or samples violate the supported WAV contract.
    #[error("invalid WAV: {0}")]
    Invalid(&'static str),
}

#[derive(Clone, Copy)]
enum Encoding {
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
}

impl Encoding {
    fn bytes(self) -> usize {
        match self {
            Self::Pcm16 => 2,
            Self::Pcm24 => 3,
            Self::Pcm32 | Self::Float32 => 4,
        }
    }
}

/// A bounded reader for mono PCM16, PCM24, PCM32 or IEEE float32 RIFF/WAVE.
///
/// Construction verifies container and data extents before exposing samples.
/// Sample values are decoded on demand into caller-owned storage; non-finite
/// floating-point samples are rejected. RF64 and extensible WAV are unsupported.
pub struct WavReader<R> {
    stream: R,
    encoding: Encoding,
    sample_rate: u32,
    sample_count: u64,
    remaining: u64,
}

impl<R: Read + Seek> WavReader<R> {
    /// Opens the WAV container at the beginning of a seekable stream.
    pub fn new(mut stream: R) -> Result<Self, WavError> {
        let length = stream.seek(SeekFrom::End(0))?;
        stream.seek(SeekFrom::Start(0))?;
        let mut header = [0; 12];
        stream.read_exact(&mut header)?;
        if &header[..4] != b"RIFF" || &header[8..] != b"WAVE" {
            return Err(WavError::Invalid("expected a RIFF/WAVE container"));
        }
        let end = u64::from(u32::from_le_bytes(header[4..8].try_into().unwrap())) + 8;
        if end < 12 || end > length {
            return Err(WavError::Invalid("truncated RIFF container"));
        }
        let mut offset = 12;
        let mut format = None;
        let mut data = None;
        while offset < end {
            if end - offset < 8 {
                return Err(WavError::Invalid("incomplete chunk header"));
            }
            stream.seek(SeekFrom::Start(offset))?;
            let mut chunk = [0; 8];
            stream.read_exact(&mut chunk)?;
            let bytes = u64::from(u32::from_le_bytes(chunk[4..].try_into().unwrap()));
            let start = offset + 8;
            let next = start + bytes + (bytes & 1);
            if next > end {
                return Err(WavError::Invalid("truncated chunk or missing padding"));
            }
            match &chunk[..4] {
                b"fmt " => {
                    if format.is_some() || bytes < 16 {
                        return Err(WavError::Invalid("duplicate or incomplete format chunk"));
                    }
                    let mut fmt = [0; 16];
                    stream.read_exact(&mut fmt)?;
                    let tag = u16::from_le_bytes(fmt[..2].try_into().unwrap());
                    let channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
                    let rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
                    let byte_rate = u32::from_le_bytes(fmt[8..12].try_into().unwrap());
                    let alignment = u16::from_le_bytes(fmt[12..14].try_into().unwrap());
                    let bits = u16::from_le_bytes(fmt[14..].try_into().unwrap());
                    let encoding = match (tag, bits) {
                        (1, 16) => Encoding::Pcm16,
                        (1, 24) => Encoding::Pcm24,
                        (1, 32) => Encoding::Pcm32,
                        (3, 32) => Encoding::Float32,
                        _ => return Err(WavError::Invalid("unsupported sample encoding")),
                    };
                    if channels != 1 || rate == 0 {
                        return Err(WavError::Invalid(
                            "expected mono audio with a nonzero sample rate",
                        ));
                    }
                    if usize::from(alignment) != encoding.bytes()
                        || rate.checked_mul(u32::from(alignment)) != Some(byte_rate)
                    {
                        return Err(WavError::Invalid("inconsistent sample geometry"));
                    }
                    format = Some((encoding, rate));
                }
                b"data" => {
                    if data.replace((start, bytes)).is_some() {
                        return Err(WavError::Invalid("duplicate data chunk"));
                    }
                }
                _ => {}
            }
            offset = next;
        }
        let (encoding, sample_rate) = format.ok_or(WavError::Invalid("missing format chunk"))?;
        let (start, bytes) = data.ok_or(WavError::Invalid("missing data chunk"))?;
        if bytes % encoding.bytes() as u64 != 0 {
            return Err(WavError::Invalid("incomplete sample in data chunk"));
        }
        let sample_count = bytes / encoding.bytes() as u64;
        stream.seek(SeekFrom::Start(start))?;
        Ok(Self {
            stream,
            encoding,
            sample_rate,
            sample_count,
            remaining: sample_count,
        })
    }

    /// Sample rate declared by the container.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Total number of mono samples declared by the data chunk.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Reads up to `output.len()` samples; zero indicates exhaustion or an empty buffer.
    ///
    /// On failure, previously decoded entries may have been written. Discard the
    /// buffer and reader after an error; stream reads are not transactional.
    pub fn read_samples(&mut self, output: &mut [f32]) -> Result<usize, WavError> {
        let count = output
            .len()
            .min(usize::try_from(self.remaining).unwrap_or(usize::MAX));
        let width = self.encoding.bytes();
        let mut buffer = [0; 4096];
        for output_chunk in output[..count].chunks_mut(buffer.len() / width) {
            let encoded = &mut buffer[..output_chunk.len() * width];
            self.stream.read_exact(encoded)?;
            for (sample, encoded) in output_chunk.iter_mut().zip(encoded.chunks_exact(width)) {
                let mut bytes = [0; 4];
                bytes[..width].copy_from_slice(encoded);
                *sample = match self.encoding {
                    Encoding::Pcm16 => {
                        f32::from(i16::from_le_bytes(bytes[..2].try_into().unwrap())) / 32768.0
                    }
                    Encoding::Pcm24 => ((i32::from_le_bytes(bytes) << 8) >> 8) as f32 / 8_388_608.0,
                    Encoding::Pcm32 => i32::from_le_bytes(bytes) as f32 / 2_147_483_648.0,
                    Encoding::Float32 => f32::from_le_bytes(bytes),
                };
                if !sample.is_finite() {
                    return Err(WavError::Invalid("non-finite sample"));
                }
                self.remaining -= 1;
            }
        }
        Ok(count)
    }
}

/// A streaming mono float32 WAV writer preserving finite samples without clipping.
///
/// Use a fresh or truncated stream. Call [`Self::finish`] to finalize the sizes;
/// dropping the writer does not silently ignore finalization errors.
pub struct WavWriter<W> {
    stream: W,
    sample_count: u64,
}

impl<W: Write + Seek> WavWriter<W> {
    /// Starts a mono IEEE float32 WAV container at stream offset zero.
    pub fn new(mut stream: W, sample_rate: u32) -> Result<Self, WavError> {
        let byte_rate = sample_rate
            .checked_mul(4)
            .filter(|_| sample_rate != 0)
            .ok_or(WavError::Invalid("invalid output sample rate"))?;
        stream.seek(SeekFrom::Start(0))?;
        let mut header = Vec::with_capacity(56);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&48u32.to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");
        header.extend_from_slice(&16u32.to_le_bytes());
        header.extend_from_slice(&3u16.to_le_bytes());
        header.extend_from_slice(&1u16.to_le_bytes());
        header.extend_from_slice(&sample_rate.to_le_bytes());
        header.extend_from_slice(&byte_rate.to_le_bytes());
        header.extend_from_slice(&4u16.to_le_bytes());
        header.extend_from_slice(&32u16.to_le_bytes());
        header.extend_from_slice(b"fact");
        header.extend_from_slice(&4u32.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(b"data");
        header.extend_from_slice(&0u32.to_le_bytes());
        stream.write_all(&header)?;
        Ok(Self {
            stream,
            sample_count: 0,
        })
    }

    /// Number of mono samples written so far.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Appends a contiguous chunk, rejecting non-finite values and RIFF overflow.
    ///
    /// Validation occurs before writing this chunk. I/O errors can still leave a
    /// partial chunk; do not reuse the writer after an I/O failure.
    pub fn write_samples(&mut self, samples: &[f32]) -> Result<(), WavError> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(WavError::Invalid("non-finite sample"));
        }
        let total = self
            .sample_count
            .checked_add(samples.len() as u64)
            .filter(|total| *total <= u64::from((u32::MAX - 48) / 4))
            .ok_or(WavError::Invalid("output exceeds RIFF size limit"))?;
        let mut buffer = [0; 4096];
        for chunk in samples.chunks(buffer.len() / 4) {
            for (sample, encoded) in chunk.iter().zip(buffer.chunks_exact_mut(4)) {
                encoded.copy_from_slice(&sample.to_le_bytes());
            }
            self.stream.write_all(&buffer[..chunk.len() * 4])?;
        }
        self.sample_count = total;
        Ok(())
    }

    /// Finalizes RIFF, fact and data sizes, flushes, and returns the stream.
    pub fn finish(mut self) -> Result<W, WavError> {
        let bytes = (self.sample_count * 4) as u32;
        self.stream.seek(SeekFrom::Start(4))?;
        self.stream.write_all(&(48 + bytes).to_le_bytes())?;
        self.stream.seek(SeekFrom::Start(44))?;
        self.stream
            .write_all(&(self.sample_count as u32).to_le_bytes())?;
        self.stream.seek(SeekFrom::Start(52))?;
        self.stream.write_all(&bytes.to_le_bytes())?;
        self.stream.seek(SeekFrom::Start(56 + u64::from(bytes)))?;
        self.stream.flush()?;
        Ok(self.stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Mono PCM24 at 16 kHz, odd JUNK and data chunks with padding. The data
    // precedes fmt, as RIFF permits. Values are -1, 0 and the largest positive.
    fn pcm24() -> Vec<u8> {
        let mut bytes = b"RIFF\x38\0\0\0WAVEJUNK\x01\0\0\0x\0data\x09\0\0\0".to_vec();
        bytes.extend_from_slice(&[0, 0, 128, 0, 0, 0, 255, 255, 127, 0]);
        bytes.extend_from_slice(b"fmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\x80\xbb\0\0\x03\0\x18\0");
        bytes
    }

    #[test]
    fn reads_independent_pcm24_across_chunks_and_rejects_truncation() {
        let bytes = pcm24();
        assert_eq!(bytes.len(), 64);
        let mut reader = WavReader::new(Cursor::new(bytes.clone())).unwrap();
        assert_eq!((reader.sample_rate(), reader.sample_count()), (16_000, 3));
        let mut sample = [2.0];
        for expected in [-1.0, 0.0, 0.999_999_9] {
            assert_eq!(reader.read_samples(&mut sample).unwrap(), 1);
            assert_eq!(sample[0], expected);
        }
        assert_eq!(reader.read_samples(&mut sample).unwrap(), 0);
        for length in 0..bytes.len() {
            assert!(WavReader::new(Cursor::new(&bytes[..length])).is_err());
        }
        let mut incomplete = bytes;
        incomplete[56..60].copy_from_slice(&32_000u32.to_le_bytes());
        incomplete[60..62].copy_from_slice(&2u16.to_le_bytes());
        incomplete[62..64].copy_from_slice(&16u16.to_le_bytes());
        assert!(matches!(
            WavReader::new(Cursor::new(incomplete)),
            Err(WavError::Invalid("incomplete sample in data chunk"))
        ));
    }

    #[test]
    fn writes_float_samples_without_clipping_and_finalizes_exact_sizes() {
        let mut writer = WavWriter::new(Cursor::new(Vec::new()), 22_050).unwrap();
        writer.write_samples(&[-2.0, -0.0]).unwrap();
        assert!(writer.write_samples(&[1.0, f32::NAN]).is_err());
        assert_eq!(writer.sample_count(), 2);
        writer.write_samples(&[0.25, 1.5]).unwrap();
        let bytes = writer.finish().unwrap().into_inner();
        assert_eq!(bytes.len(), 72);
        assert_eq!(&bytes[..12], b"RIFF\x40\0\0\0WAVE");
        assert_eq!(&bytes[20..24], &[3, 0, 1, 0]);
        assert_eq!(&bytes[24..28], &[0x22, 0x56, 0, 0]);
        assert_eq!(&bytes[36..56], b"fact\x04\0\0\0\x04\0\0\0data\x10\0\0\0");
        assert_eq!(
            &bytes[56..],
            &[0, 0, 0, 192, 0, 0, 0, 128, 0, 0, 128, 62, 0, 0, 192, 63]
        );
    }

    #[test]
    fn rejects_nonfinite_float_input() {
        let mut writer = WavWriter::new(Cursor::new(Vec::new()), 16_000).unwrap();
        writer.write_samples(&[0.0]).unwrap();
        let mut bytes = writer.finish().unwrap().into_inner();
        bytes[56..60].copy_from_slice(&f32::INFINITY.to_le_bytes());
        let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
        assert!(reader.read_samples(&mut [0.0]).is_err());
    }
}
