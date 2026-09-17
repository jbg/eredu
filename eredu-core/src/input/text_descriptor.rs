use super::*;

/// Closed construction of one text-token descriptor with shape `[batch, positions]`.
/// Preparation performs no allocation. The worker uses fixed boxed payloads
/// moved into the existing vector representation without spare capacity/growth.
/// This plan measures managed payload; it grants no allocation authority.
#[derive(Debug)]
#[must_use = "construct only under the caller's prior allocation authority"]
pub struct TextTokenInputDescriptorPlan {
    batch: usize,
    positions: usize,
    tokens: usize,
    retained_bytes: u64,
}

impl TextTokenInputDescriptorPlan {
    /// Checks positive wire-compatible geometry and the complete host token
    /// extent before any descriptor storage is allocated.
    pub fn new(batch: u64, positions: u64) -> Result<Self, PreparedInputError> {
        if batch == 0 || positions == 0 {
            return Err(PreparedInputError::InvalidWireValue {
                field: "positive text shape",
                value: 0,
            });
        }
        u32::try_from(batch).map_err(|_| PreparedInputError::WireValueOverflow("text batch"))?;
        u32::try_from(positions)
            .map_err(|_| PreparedInputError::WireValueOverflow("text positions"))?;
        let token_bytes = batch
            .checked_mul(positions)
            .and_then(|count| count.checked_mul(4))
            .ok_or(PreparedInputError::WireValueOverflow("text token extent"))?;
        if token_bytes > isize::MAX as u64 {
            return Err(PreparedInputError::WireValueOverflow("text token extent"));
        }
        let tokens = usize::try_from(token_bytes / 4)
            .map_err(|_| PreparedInputError::WireValueOverflow("text token count"))?;
        let bytes = std::mem::size_of::<PreparedInputIdentity>()
            .checked_add(std::mem::size_of::<InputPartDescriptor>())
            .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<usize>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(PreparedInputError::WireValueOverflow(
                "text descriptor payload",
            ))?;
        Ok(Self {
            batch: usize::try_from(batch)
                .map_err(|_| PreparedInputError::WireValueOverflow("text batch"))?,
            positions: usize::try_from(positions)
                .map_err(|_| PreparedInputError::WireValueOverflow("text positions"))?,
            tokens,
            retained_bytes: bytes,
        })
    }

    /// Exact number of token IDs represented by the descriptor.
    pub const fn token_count(&self) -> usize {
        self.tokens
    }

    /// Exact inline descriptor plus its fixed part and shape allocations.
    /// Empty metadata/extent maps own no allocation. Allocator and ownership
    /// bookkeeping are outside this payload measure.
    pub const fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    /// Conservative explicit descriptor construction scratch, in addition to
    /// final owned payload. Covers the initialized fixed arrays and moved record;
    /// it is not a claim about compiler stack frames or allocator internals.
    pub const fn scratch_bytes(&self) -> u64 {
        self.retained_bytes
    }

    /// Constructs the validated descriptor once under the caller's authority.
    /// No tokenizer, native tensor, canonical word vector or map builder is used.
    pub fn construct(self) -> PreparedInputIdentity {
        #[cfg(test)]
        tests::before_construct();
        let shape: Box<[usize]> = Box::new([self.batch, self.positions]);
        let part = InputPartDescriptor {
            modality: InputModality::Text,
            payload_kind: InputPayloadKind::TokenIds,
            payload: InputTensorIdentity {
                dtype: TensorDtype::U32,
                shape: shape.into_vec(),
            },
            metadata: InputIdentityMap::empty(),
            extents: InputIdentityMap::empty(),
        };
        let parts: Box<[InputPartDescriptor]> = Box::new([part]);
        PreparedInputIdentity {
            parts: parts.into_vec(),
        }
    }
}

#[cfg(test)]
mod tests;
