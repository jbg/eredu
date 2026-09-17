use super::{PreparedInputCacheIdentity, SharedPreparedInputCacheIdentity};
use eredu_core::{
    input::{InputWordWriteError, TextTokenInputDescriptorPlan},
    PreparedInputError,
};
use sha2::{Digest, Sha256};
use std::{convert::Infallible, mem::size_of};

/// Checked construction failure for a single text-token cache identity.
#[derive(Debug, thiserror::Error)]
pub enum TextInputIdentityError {
    /// Fixed original metadata storage could not be prepared.
    #[error("original text identity storage: {0}")]
    Storage(#[from] crate::working_memory::WorkingMemoryError),
    /// Geometry cannot be represented by the canonical prepared-input descriptor.
    #[error("{0}")]
    Descriptor(#[from] PreparedInputError),
    /// An exact payload or construction-scratch sum exceeds its representation.
    #[error("text input identity {0} overflow")]
    Overflow(&'static str),
    /// The supplied immutable token source does not match the planned geometry.
    #[error("text input identity requires {expected} tokens, received {actual}")]
    TokenCount {
        /// Exact count required by the planned batch and sequence dimensions.
        expected: usize,
        /// Count supplied by the borrowed token source.
        actual: usize,
    },
}

/// Closed payload plan for one U32 text tensor and its two SHA-256 fingerprints.
///
/// Preparing and binding perform no allocation. The worker constructs fixed
/// boxed payloads and streams canonical words directly into SHA-256. Measures
/// managed inline/owned payload and explicit algorithm scratch, excluding
/// allocator, shared-owner and custody bookkeeping. This is no funding grant.
#[derive(Debug)]
#[must_use = "bind the original token slice before executing the identity worker"]
pub struct TextInputIdentityPlan {
    descriptor: TextTokenInputDescriptorPlan,
    retained: u64,
    peak: u64,
}

/// One construction worker coupled to the exact immutable token source.
#[derive(Debug)]
#[must_use = "construct under the caller's prior preparation authority"]
pub struct BoundTextInputIdentityPlan<'a> {
    plan: TextInputIdentityPlan,
    tokens: &'a [u32],
}

impl TextInputIdentityPlan {
    /// Validates positive `[batch, positions]` geometry before constructing any
    /// descriptor, fingerprint or shared owner.
    pub fn new(batch: u64, positions: u64) -> Result<Self, TextInputIdentityError> {
        let descriptor = TextTokenInputDescriptorPlan::new(batch, positions)?;
        let descriptor_heap = descriptor
            .retained_bytes()
            .checked_sub(size_of::<eredu_core::PreparedInputIdentity>() as u64)
            .ok_or(TextInputIdentityError::Overflow("descriptor payload"))?;
        let retained = (size_of::<PreparedInputCacheIdentity>() as u64)
            .checked_add(descriptor_heap)
            .and_then(|bytes| bytes.checked_add(128))
            .ok_or(TextInputIdentityError::Overflow("retained payload"))?;
        // Hashes run sequentially. Two explicit SHA states conservatively cover
        // movement/finalization, plus the digest, initialized hex array, length
        // and word buffers. Descriptor scratch is priced even across that phase.
        let hash_scratch = size_of::<Sha256>()
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(32 + 64 + 8 + 4))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(TextInputIdentityError::Overflow("hash scratch"))?;
        let peak = retained
            .checked_add(descriptor.scratch_bytes())
            .and_then(|bytes| bytes.checked_add(hash_scratch))
            .ok_or(TextInputIdentityError::Overflow("construction peak"))?;
        Ok(Self {
            descriptor,
            retained,
            peak,
        })
    }

    /// Exact final inline cache descriptor plus fixed part, shape and strings.
    pub const fn retained_bytes(&self) -> u64 {
        self.retained
    }

    /// Final payload plus conservative explicit construction scratch.
    pub const fn peak_bytes(&self) -> u64 {
        self.peak
    }

    /// Couples geometry to the original immutable source without copying tokens.
    /// Native input construction can consume the same `tokens()` borrow.
    pub fn bind(
        self,
        tokens: &[u32],
    ) -> Result<BoundTextInputIdentityPlan<'_>, TextInputIdentityError> {
        if tokens.len() != self.descriptor.token_count() {
            return Err(TextInputIdentityError::TokenCount {
                expected: self.descriptor.token_count(),
                actual: tokens.len(),
            });
        }
        Ok(BoundTextInputIdentityPlan { plan: self, tokens })
    }
}

impl<'a> BoundTextInputIdentityPlan<'a> {
    /// Exact source whose content determines the resulting fingerprints.
    pub const fn tokens(&self) -> &'a [u32] {
        self.tokens
    }

    /// Constructs the immutable owner once, without an encoded-word vector or
    /// variable-capacity fingerprint builder. Publication/accounting belongs to
    /// the caller's existing preparation scope and must precede its settlement.
    pub fn construct(self) -> Result<SharedPreparedInputCacheIdentity, TextInputIdentityError> {
        self.construct_with_custody(None)
    }
    /// Same immutable text compiler with a closed one-slot attachment owner.
    /// Custody is the caller's already accepted raw metadata hold.
    pub fn construct_original(
        self,
        custody: crate::working_memory::OriginalTextMetadataCustody,
    ) -> Result<SharedPreparedInputCacheIdentity, TextInputIdentityError> {
        self.construct_with_custody(Some(custody))
    }
    fn construct_with_custody(
        self,
        custody: Option<crate::working_memory::OriginalTextMetadataCustody>,
    ) -> Result<SharedPreparedInputCacheIdentity, TextInputIdentityError> {
        #[cfg(test)]
        tests::before_construct();
        let prepared = self.plan.descriptor.construct();
        let mut semantic = Sha256::new();
        for token in self.tokens {
            semantic.update(token.to_le_bytes());
        }
        let semantic = fixed_hex(semantic.finalize().into());

        let count = u64::try_from(prepared.encoded_word_count()?)
            .map_err(|_| TextInputIdentityError::Overflow("encoded word count"))?;
        let mut prefix = Sha256::new();
        prefix.update(b"eredu-prepared-input-cache-v1\0");
        prefix.update(count.to_le_bytes());
        prepared
            .visit_encoded_words(|word| {
                prefix.update(word.to_le_bytes());
                Ok::<_, Infallible>(())
            })
            .map_err(|error| match error {
                InputWordWriteError::Encoding(error) => TextInputIdentityError::Descriptor(error),
                InputWordWriteError::Sink(never) => match never {},
            })?;
        prefix.update(64u64.to_le_bytes());
        prefix.update(semantic.as_bytes());
        let prefix = fixed_hex(prefix.finalize().into());
        let payload =
            PreparedInputCacheIdentity::from_validated_text_parts(prepared, semantic, prefix);
        debug_assert_eq!(payload.capacity_bytes(), Some(self.plan.retained));
        match custody {
            Some(custody) => Ok(SharedPreparedInputCacheIdentity::new_original_text(
                payload, custody,
            )?),
            None => Ok(SharedPreparedInputCacheIdentity::new_text(payload)),
        }
    }
}

pub(super) fn fixed_hex(digest: [u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut bytes: Box<[u8]> = Box::new([0; 64]);
    for (index, byte) in digest.into_iter().enumerate() {
        bytes[2 * index] = DIGITS[usize::from(byte >> 4)];
        bytes[2 * index + 1] = DIGITS[usize::from(byte & 15)];
    }
    String::from_utf8(bytes.into_vec()).expect("hex digits are UTF-8")
}

#[cfg(test)]
mod tests;
