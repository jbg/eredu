use super::*;
use sha2::{Digest, Sha256};
use std::{alloc::Layout, fmt, mem::size_of};

pub(crate) const KEYS: [InputMetadataKey; 3] = [
    InputMetadataKey::PatchGrid,
    InputMetadataKey::PatchPositions,
    InputMetadataKey::AudioMask,
];
/// Allocation-free structural rejection. Family token/grid equations are still
/// checked by actual selected architecture admission, not reproduced here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostInputPlanError {
    /// No parts were supplied.
    Empty,
    /// Checked capacity or shape arithmetic overflowed.
    Overflow,
    /// Part has incompatible or unknown modality/payload/metadata/extents.
    Structure {
        /// Original part index.
        part: usize,
    },
    /// A slot's positive shape and exact values disagree.
    Shape {
        /// Original part index.
        part: usize,
    },
    /// This source compiler has no storage for this exact value encoding.
    Encoding {
        /// Original part index.
        part: usize,
    },
}
impl fmt::Display for HostInputPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "prepared host input: {self:?}")
    }
}
impl std::error::Error for HostInputPlanError {}
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Counts(pub(crate) [usize; 8]);
impl Counts {
    fn add(&mut self, i: usize, n: usize) -> Result<(), HostInputPlanError> {
        self.0[i] = self.0[i]
            .checked_add(n)
            .ok_or(HostInputPlanError::Overflow)?;
        Ok(())
    }
}
/// Scalar, consuming plan borrowing the exact original values. No allocation or
/// source authority is created until the working-memory compiler accepts it.
///
/// ```compile_fail
/// use eredu_runtime::input::host::PreparedHostInputPlan;
/// fn reuse(plan: PreparedHostInputPlan<'_>) { let first=plan; let second=plan; }
/// ```
#[derive(Debug)]
pub struct PreparedHostInputPlan<'a> {
    pub(crate) parts: &'a [HostInputPart<'a>],
    pub(crate) counts: Counts,
    pub(crate) digest: [u8; 32],
    pub(crate) bytes: usize,
}
impl<'a> PreparedHostInputPlan<'a> {
    /// Checks structure and all eight complete destination extents by borrowing.
    /// This does not invoke a processor, native inspector, callback or allocator.
    pub fn prepare(parts: &'a [HostInputPart<'a>]) -> Result<Self, HostInputPlanError> {
        if parts.is_empty() {
            return Err(HostInputPlanError::Empty);
        }
        let mut counts = Counts::default();
        counts.add(0, parts.len())?;
        let mut digest = Sha256::new();
        digest.update(b"eredu-original-host-input-v1\0");
        word(&mut digest, parts.len())?;
        for (index, part) in parts.iter().enumerate() {
            let bad = HostInputPlanError::Structure { part: index };
            let modality = match part.modality {
                InputModality::Text => 0u8,
                InputModality::Image => 1,
                InputModality::Video => 2,
                InputModality::Audio => 3,
                _ => return Err(bad),
            };
            let kind = match part.kind {
                InputPayloadKind::TokenIds => 0u8,
                InputPayloadKind::Tensor => 1,
                InputPayloadKind::Embeddings => 2,
                _ => return Err(bad),
            };
            if !part.kind.accepts(part.modality) {
                return Err(bad);
            }
            digest.update([modality, kind]);
            slot(part.payload, index, &mut counts, &mut digest)?;
            word(&mut digest, part.metadata.len())?;
            for (i, (key, _)) in part.metadata.iter().enumerate() {
                if !KEYS.contains(key)
                    || !key.accepts(part.modality)
                    || part.metadata[..i].iter().any(|(prior, _)| prior == key)
                {
                    return Err(bad);
                }
            }
            for (key_index, key) in KEYS.into_iter().enumerate() {
                if let Some((_, value)) = part.metadata.iter().find(|(actual, _)| *actual == key) {
                    digest.update([key_index as u8]);
                    slot(*value, index, &mut counts, &mut digest)?;
                }
            }
            counts.add(3, part.extents.len())?;
            word(&mut digest, part.extents.len())?;
            for (i, extent) in part.extents.iter().enumerate() {
                if !extent.accepts(part.modality)
                    || part.extents[..i].iter().any(|prior| {
                        std::mem::discriminant(prior) == std::mem::discriminant(extent)
                    })
                {
                    return Err(bad);
                }
                match extent {
                    InputExtent::PatchGrid {
                        time,
                        height,
                        width,
                    } => {
                        digest.update([0]);
                        word(&mut digest, *time)?;
                        word(&mut digest, *height)?;
                        word(&mut digest, *width)?;
                    }
                    InputExtent::AudioValidFrames(n) => {
                        digest.update([1]);
                        word(&mut digest, *n)?;
                    }
                    InputExtent::VideoFrame { group, index, count, first_source_frame, last_source_frame, source_fps_bits } => {
                        let fps = f64::from_bits(*source_fps_bits);
                        if *count == 0 || index >= count || first_source_frame > last_source_frame
                            || !fps.is_finite() || fps <= 0.0 { return Err(bad); }
                        digest.update([2]);
                        for value in [group, index, count, first_source_frame, last_source_frame] { word(&mut digest, *value)?; }
                        digest.update(source_fps_bits.to_le_bytes());
                    }
                    _ => return Err(bad),
                }
            }
        }
        let bytes = storage::allocation_bytes(counts)?
            .checked_add(size_of::<Self>())
            .and_then(|n| n.checked_add(size_of::<Counts>()))
            .and_then(|n| n.checked_add(size_of::<Storage>()))
            .and_then(|n| n.checked_add(size_of::<CompileFailure>()))
            .and_then(|n| n.checked_add(size_of::<Result<Storage, CompileFailure>>()))
            .ok_or(HostInputPlanError::Overflow)?;
        // Reject extents impossible for the default allocator before comparison.
        Layout::array::<u8>(bytes).map_err(|_| HostInputPlanError::Overflow)?;
        let mut content = [0; 32];
        content.copy_from_slice(&digest.finalize());
        Ok(Self {
            parts,
            counts,
            digest: content,
            bytes,
        })
    }
    /// Exact requested destination and compiler representation bytes. It is not
    /// a grant, native-work allowance or transitive caller-source measurement.
    pub fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Canonical actual-content fingerprint; equal hashes confer no authority.
    pub fn content_digest(&self) -> &[u8; 32] {
        &self.digest
    }
    /// Exact number of logical numerical slots, including metadata.
    pub fn slot_count(&self) -> usize {
        self.counts.0[1]
    }
}
fn word(digest: &mut Sha256, n: usize) -> Result<(), HostInputPlanError> {
    digest.update(
        u64::try_from(n)
            .map_err(|_| HostInputPlanError::Overflow)?
            .to_le_bytes(),
    );
    Ok(())
}
fn slot(
    value: HostTensorView<'_>,
    part: usize,
    counts: &mut Counts,
    digest: &mut Sha256,
) -> Result<(), HostInputPlanError> {
    if value.shape.is_empty() || value.shape.contains(&0) {
        return Err(HostInputPlanError::Shape { part });
    }
    let len = value.shape.iter().try_fold(1usize, |n, d| {
        n.checked_mul(*d).ok_or(HostInputPlanError::Overflow)
    })?;
    if len != value.values.len() {
        return Err(HostInputPlanError::Shape { part });
    }
    counts.add(1, 1)?;
    counts.add(2, value.shape.len())?;
    word(digest, value.shape.len())?;
    for n in value.shape {
        word(digest, *n)?;
    }
    match value.values {
        HostTensorValues::U32(v) => {
            counts.add(4, v.len())?;
            digest.update([0]);
            for x in v {
                digest.update(x.to_le_bytes());
            }
        }
        HostTensorValues::I32(v) => {
            counts.add(5, v.len())?;
            digest.update([1]);
            for x in v {
                digest.update(x.to_le_bytes());
            }
        }
        HostTensorValues::F32(v) => {
            counts.add(6, v.len())?;
            digest.update([2]);
            for x in v {
                digest.update(x.to_bits().to_le_bytes());
            }
        }
        HostTensorValues::Bool(v) => {
            counts.add(7, v.len())?;
            digest.update([3]);
            for &value in v { digest.update([u8::from(value)]); }
        }
    }
    Ok(())
}
