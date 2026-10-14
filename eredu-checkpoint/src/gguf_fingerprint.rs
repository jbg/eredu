//! Shard-layout-independent identity of exact GGUF metadata and encoded tensors.
use crate::artifact::{ArtifactFile, ArtifactFingerprintError, ArtifactFingerprintSource};
use eredu_gguf::{Checkpoint, MetadataArray, MetadataValue};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};

/// Failure before an exact logical GGUF identity can be established.
#[derive(Debug, thiserror::Error)]
pub enum GgufFingerprintError {
    /// Source membership or file identity changed during fingerprinting.
    #[error(transparent)]
    Artifact(#[from] ArtifactFingerprintError),
    /// The retained GGUF catalog is invalid.
    #[error(transparent)]
    Catalog(#[from] eredu_gguf::Error),
    /// A bounded encoded payload read failed.
    #[error("cannot fingerprint GGUF payload {path:?}")]
    Io {
        /// Exact member path.
        path: PathBuf,
        /// Original host I/O failure.
        #[source]
        source: std::io::Error,
    },
}

/// Hashes canonical metadata and sorted physical tensor names, encodings,
/// dimensions and complete encoded bytes. Split ordinals/counts, alignment and
/// physical offsets do not contribute. Every other metadata field contributes,
/// including the complete tokenizer. Endianness remains part of the identity.
///
/// Payloads stream through one 8 MiB buffer, independent of tensor or shard size.
/// Original file identities are checked before and after the complete pass.
pub fn fingerprint_gguf_catalog(checkpoint: &Checkpoint) -> Result<[u8; 32], GgufFingerprintError> {
    let guards = ArtifactFingerprintSource::new(
        checkpoint
            .shards()
            .iter()
            .enumerate()
            .map(|(index, shard)| ArtifactFile::new(format!("gguf.shard.{index}"), shard.path())),
    )?;
    // The catalog can predate this call. Match current headers to the retained
    // descriptors before using their absolute encoded ranges.
    for shard in checkpoint.shards() {
        let reader = eredu_gguf::Reader::open(shard.path())?;
        let metadata_matches = if shard.split_no() == 0 {
            reader
                .metadata()
                .iter()
                .filter(|(key, _)| key.as_str() != "split.no")
                .eq(checkpoint
                    .metadata()
                    .iter()
                    .filter(|(key, _)| key.as_str() != "split.no"))
        } else {
            // Standard splits retain semantic metadata on their first member.
            // Repeated secondary metadata must agree with that authority.
            reader
                .metadata()
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "split.no" | "split.count" | "split.tensors.count" | "general.alignment"
                    )
                })
                .all(|(key, value)| checkpoint.metadata().get(key) == Some(value))
        };
        if reader.version() != shard.version()
            || reader.endian() != shard.endian()
            || reader.alignment() != shard.alignment()
            || !reader
                .tensors()
                .iter()
                .eq(shard.tensors().iter().map(|tensor| tensor.descriptor()))
            || !metadata_matches
        {
            return Err(eredu_gguf::Error::InvalidHeader(
                "GGUF source header changed since catalog admission".into(),
            )
            .into());
        }
    }
    let mut digest = Sha256::new();
    digest.update(b"eredu.gguf.catalog.v1\0");
    let metadata: Vec<_> = checkpoint
        .metadata()
        .iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "split.no" | "split.count" | "split.tensors.count" | "general.alignment"
            )
        })
        .collect();
    count(&mut digest, metadata.len());
    for (key, value) in metadata {
        string(&mut digest, key);
        value_hash(&mut digest, value);
    }
    let mut tensors: Vec<_> = checkpoint.tensors().collect();
    tensors.sort_unstable_by(|a, b| a.descriptor().name.cmp(&b.descriptor().name));
    count(&mut digest, tensors.len());
    let materializer = checkpoint.materializer();
    let mut buffer = vec![0u8; 8 << 20];
    for tensor in tensors {
        let descriptor = tensor.descriptor();
        string(&mut digest, &descriptor.name);
        digest.update(descriptor.ggml_type.code().to_le_bytes());
        count(&mut digest, descriptor.dimensions.len());
        for dimension in &descriptor.dimensions {
            digest.update(dimension.to_le_bytes());
        }
        digest.update(descriptor.byte_len.to_le_bytes());
        let path = materializer.shard_path_for_tensor(&descriptor.name)?;
        let shard = checkpoint
            .shards()
            .iter()
            .find(|shard| shard.path() == path)
            .expect("materializer retains a catalog shard");
        digest.update([match shard.endian() {
            eredu_gguf::Endian::Little => 0,
            eredu_gguf::Endian::Big => 1,
        }]);
        let io_error = |source| GgufFingerprintError::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut file = File::open(path).map_err(io_error)?;
        file.seek(SeekFrom::Start(descriptor.data_offset))
            .map_err(io_error)?;
        let mut remaining = descriptor.byte_len;
        while remaining != 0 {
            let length = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..length]).map_err(io_error)?;
            digest.update(&buffer[..length]);
            remaining -= length as u64;
        }
    }
    guards.validate_unchanged()?;
    Ok(digest.finalize().into())
}

fn count(digest: &mut Sha256, length: usize) {
    digest.update((length as u64).to_le_bytes());
}
fn string(digest: &mut Sha256, value: &str) {
    count(digest, value.len());
    digest.update(value.as_bytes());
}
fn value_hash(digest: &mut Sha256, value: &MetadataValue) {
    macro_rules! scalar {
        ($tag:expr, $value:expr) => {{
            digest.update([$tag]);
            digest.update($value.to_le_bytes());
        }};
    }
    match value {
        MetadataValue::Uint8(x) => scalar!(0, x),
        MetadataValue::Int8(x) => scalar!(1, x),
        MetadataValue::Uint16(x) => scalar!(2, x),
        MetadataValue::Int16(x) => scalar!(3, x),
        MetadataValue::Uint32(x) => scalar!(4, x),
        MetadataValue::Int32(x) => scalar!(5, x),
        MetadataValue::Float32(x) => scalar!(6, x.to_bits()),
        MetadataValue::Bool(x) => {
            digest.update([7, u8::from(*x)]);
        }
        MetadataValue::String(x) => {
            digest.update([8]);
            string(digest, x);
        }
        MetadataValue::Array(x) => {
            digest.update([9]);
            array_hash(digest, x);
        }
        MetadataValue::Uint64(x) => scalar!(10, x),
        MetadataValue::Int64(x) => scalar!(11, x),
        MetadataValue::Float64(x) => scalar!(12, x.to_bits()),
    }
}
fn array_hash(digest: &mut Sha256, value: &MetadataArray) {
    macro_rules! array {
        ($tag:expr, $values:expr) => {{
            digest.update([$tag]);
            count(digest, $values.len());
            for value in $values {
                digest.update(value.to_le_bytes());
            }
        }};
    }
    match value {
        MetadataArray::Uint8(x) => array!(0, x),
        MetadataArray::Int8(x) => array!(1, x),
        MetadataArray::Uint16(x) => array!(2, x),
        MetadataArray::Int16(x) => array!(3, x),
        MetadataArray::Uint32(x) => array!(4, x),
        MetadataArray::Int32(x) => array!(5, x),
        MetadataArray::Float32(x) => {
            digest.update([6]);
            count(digest, x.len());
            for value in x {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
        MetadataArray::Bool(x) => {
            digest.update([7]);
            count(digest, x.len());
            for value in x {
                digest.update([u8::from(*value)]);
            }
        }
        MetadataArray::String(x) => {
            digest.update([8]);
            count(digest, x.len());
            for value in x {
                string(digest, value);
            }
        }
        MetadataArray::Array(x) => {
            digest.update([9]);
            count(digest, x.len());
            for value in x {
                array_hash(digest, value);
            }
        }
        MetadataArray::Uint64(x) => array!(10, x),
        MetadataArray::Int64(x) => array!(11, x),
        MetadataArray::Float64(x) => {
            digest.update([12]);
            count(digest, x.len());
            for value in x {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_gguf::{GgmlType, TensorInput, Writer, WriterOptions};
    use std::{collections::BTreeMap, path::Path};

    #[test]
    fn logical_identity_survives_split_order_and_alignment_but_not_token_or_weight_changes() {
        let directory = tempfile::tempdir().unwrap();
        let original = directory.path().join("model.gguf");
        let first = directory.path().join("model-00001-of-00002.gguf");
        let second = directory.path().join("model-00002-of-00002.gguf");
        let dense: Vec<_> = [1.25f32, -2.5, 0.125, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let packed: Vec<_> = (0..18).map(|x| x * 7 + 1).collect();
        let a = TensorInput {
            name: "a.weight",
            dimensions: &[2, 2],
            ggml_type: GgmlType::F32,
            data: &dense,
        };
        let b = TensorInput {
            name: "b.weight",
            dimensions: &[32, 1],
            ggml_type: GgmlType::Q4_0,
            data: &packed,
        };
        let mut metadata = BTreeMap::from([
            (
                "general.name".into(),
                MetadataValue::String("fixture".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetadataValue::Array(MetadataArray::String(vec!["one".into(), "two".into()])),
            ),
            ("fixture.norm_epsilon".into(), MetadataValue::Float32(1e-5)),
        ]);
        let write = |path: &Path,
                     metadata: &BTreeMap<String, MetadataValue>,
                     tensors: &[TensorInput<'_>],
                     alignment| {
            Writer::new(WriterOptions {
                alignment,
                ..Default::default()
            })
            .unwrap()
            .write(File::create(path).unwrap(), metadata, tensors)
            .unwrap();
        };
        write(&original, &metadata, &[a, b], 32);
        let expected = fingerprint_gguf_catalog(&Checkpoint::open(&original).unwrap()).unwrap();
        let mut split = metadata.clone();
        split.extend([
            ("split.no".into(), MetadataValue::Uint16(0)),
            ("split.count".into(), MetadataValue::Uint16(2)),
            ("split.tensors.count".into(), MetadataValue::Uint64(2)),
        ]);
        write(&first, &split, &[b], 64);
        split.insert("split.no".into(), MetadataValue::Uint16(1));
        let secondary = split
            .iter()
            .filter(|(key, _)| key.starts_with("split."))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        write(&second, &secondary, &[a], 64);
        assert_eq!(
            fingerprint_gguf_catalog(&Checkpoint::open(&first).unwrap()).unwrap(),
            expected
        );
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            MetadataValue::Array(MetadataArray::String(vec!["one".into(), "changed".into()])),
        );
        write(&original, &metadata, &[a, b], 32);
        assert_ne!(
            fingerprint_gguf_catalog(&Checkpoint::open(&original).unwrap()).unwrap(),
            expected
        );
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            MetadataValue::Array(MetadataArray::String(vec!["one".into(), "two".into()])),
        );
        let mut changed = packed.clone();
        changed[17] ^= 1;
        write(
            &original,
            &metadata,
            &[
                a,
                TensorInput {
                    data: &changed,
                    ..b
                },
            ],
            32,
        );
        assert_ne!(
            fingerprint_gguf_catalog(&Checkpoint::open(&original).unwrap()).unwrap(),
            expected
        );
    }
}
