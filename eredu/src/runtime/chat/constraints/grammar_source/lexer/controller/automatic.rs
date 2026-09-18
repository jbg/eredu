//! Paid preactivation mask and fixed trigger prefix over the actual grammar.
use super::super::{
    OriginalGrammarState, OriginalGrammarStateCopyError, OriginalGrammarStateError,
};
use crate::runtime::chat::constraints::{trigger, vocabulary};
use eredu_core::{
    HostPreparationAuthority, PackedTokenFilter, PackedTokenFilterError, SharedTokenFilter,
    SpeculativeBuffer, SpeculativeBufferAllocationError,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

#[derive(Debug)]
struct Mask {
    words: SpeculativeBuffer<u32>,
    vocabulary: usize,
}
#[derive(Debug)]
pub(super) struct Automatic {
    pending: trigger::TriggerPrefix,
    mask: Option<Mask>,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("automatic grammar source or destination changed")]
    Source,
    #[error("automatic grammar control extent overflow")]
    Overflow,
    #[error("token crosses the activation boundary with bytes outside the grammar")]
    Rejected,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Buffer(#[from] SpeculativeBufferAllocationError),
    #[error(transparent)]
    Copy(#[from] OriginalGrammarStateCopyError),
    #[error(transparent)]
    State(#[from] OriginalGrammarStateError),
}
/// Any partially filled mask and owning parser failure retire before funding.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(super) struct Error {
    #[source]
    cause: Cause,
    mask: Option<Mask>,
    funding: HostMetadataFunding,
}
fn controls() -> Option<usize> {
    let parts = [
        size_of::<Automatic>(),
        size_of::<Mask>(),
        size_of::<Option<Mask>>(),
        size_of::<Cause>(),
        size_of::<Error>(),
        size_of::<HostMetadataFunding>(),
        size_of::<Result<Automatic, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<(), Cause>>(),
        size_of::<Result<Mask, Cause>>(),
        size_of::<Result<Option<OriginalGrammarState>, Error>>(),
        size_of::<Result<Option<OriginalGrammarState>, Cause>>(),
        size_of::<Option<OriginalGrammarState>>(),
        size_of::<Result<OriginalGrammarState, OriginalGrammarStateCopyError>>(),
        size_of::<Result<(OriginalGrammarState, bool), OriginalGrammarStateError>>(),
        size_of::<Result<SpeculativeBuffer<u32>, SpeculativeBufferAllocationError>>(),
        size_of::<Result<(), eredu_core::GenerationError>>(),
        size_of::<Result<(), HostMetadataFundingError>>(),
        size_of::<(
            &mut Automatic,
            &OriginalGrammarState,
            u32,
            &HostMetadataFunding,
        )>(),
        size_of::<(
            &OriginalGrammarState,
            trigger::TriggerPrefix,
            u32,
            &HostMetadataFunding,
        )>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::slice::Iter<'_, u32>>(),
        size_of::<(usize, usize, usize, u32, bool)>(),
        size_of::<Result<(&llguidance::toktrie::TokTrie, &[u8]), Cause>>(),
        size_of::<Option<&str>>(),
        eredu_core::speculative::byte_trigger::control_bytes()?,
        eredu_core::HostMetadataFunding::reservation_control_bytes(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn mask_bytes(capacity: usize) -> Option<usize> {
    SpeculativeBuffer::<u32>::retained_control_bytes(capacity)?
        .checked_add(HostPreparationAuthority::retention_bytes::<
            HostMetadataFunding,
        >()?)?
        .checked_add(size_of::<HostPreparationAuthority>())
}
fn source(
    state: &OriginalGrammarState,
    pending: trigger::TriggerPrefix,
) -> Result<(&llguidance::toktrie::TokTrie, &[u8]), Cause> {
    let vocabulary = state.parser().vocabulary();
    let trigger = vocabulary.recipe.trigger().ok_or(Cause::Source)?.as_bytes();
    if !pending.is_valid_for(trigger) {
        return Err(Cause::Source);
    }
    Ok((vocabulary.trie_source().trie(), trigger))
}
fn allocate_mask(
    vocabulary: usize,
    capacity: usize,
    funding: &HostMetadataFunding,
) -> Result<Mask, Cause> {
    funding.reserve_metadata(mask_bytes(capacity).ok_or(Cause::Overflow)?)?;
    Ok(Mask {
        vocabulary,
        words: SpeculativeBuffer::try_new_retained(
            capacity,
            HostPreparationAuthority::retain(funding.clone()),
        )?,
    })
}
impl Automatic {
    pub(super) fn prepare(
        state: &OriginalGrammarState,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let result = (|| -> Result<(), Cause> {
            funding.reserve_metadata(controls().ok_or(Cause::Overflow)?)?;
            source(state, trigger::TriggerPrefix::default())?;
            Ok(())
        })();
        result
            .map(|()| Self {
                pending: trigger::TriggerPrefix::default(),
                mask: None,
            })
            .map_err(|cause| Error {
                cause,
                mask: None,
                funding: funding.clone(),
            })
    }
    pub(super) fn copy_required_bytes(&self) -> Option<usize> {
        controls()?.checked_add(match self.mask.as_ref() {
            Some(mask) => mask_bytes(mask.words.capacity())?,
            None => 0,
        })
    }
    pub(super) fn try_copy(&self, funding: &HostMetadataFunding) -> Result<Self, Error> {
        let mut mask = None;
        let result = (|| -> Result<(), Cause> {
            funding.reserve_metadata(controls().ok_or(Cause::Overflow)?)?;
            if let Some(source) = self.mask.as_ref() {
                mask = Some(allocate_mask(
                    source.vocabulary,
                    source.words.capacity(),
                    funding,
                )?);
                mask.as_mut()
                    .expect("copied mask cell")
                    .words
                    .try_extend(source.words.iter().copied())
                    .map_err(|_| Cause::Source)?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(Self {
                pending: self.pending,
                mask,
            }),
            Err(cause) => Err(Error {
                cause,
                mask,
                funding: funding.clone(),
            }),
        }
    }
    /// Every activating candidate runs the same independent ordinary trial.
    /// Tokens without a complete trigger remain unrestricted by this grammar.
    pub(super) fn compute_mask(
        &mut self,
        state: &OriginalGrammarState,
        funding: &HostMetadataFunding,
    ) -> Result<(), Error> {
        let mut mask = None;
        let result = (|| -> Result<(), Cause> {
            funding.reserve_metadata(controls().ok_or(Cause::Overflow)?)?;
            let (trie, trigger) = source(state, self.pending)?;
            let vocabulary = trie.vocab_size();
            let words = vocabulary.checked_add(31).ok_or(Cause::Overflow)? / 32;
            mask = Some(allocate_mask(vocabulary, words, funding)?);
            let target = mask.as_mut().expect("mask destination");
            for _ in 0..words {
                target.words.try_push(0).map_err(|_| Cause::Source)?;
            }
            for id in 0..vocabulary {
                let token = u32::try_from(id).map_err(|_| Cause::Source)?;
                let bytes = vocabulary::token_bytes(trie, token);
                let allowed = if let Some(found) =
                    trigger::find(self.pending.bytes(trigger), bytes, trigger)
                {
                    let candidate = state.try_copy(funding)?;
                    let (_, accepted) = candidate.try_activation(token, found)?;
                    accepted
                } else {
                    true
                };
                if allowed {
                    target.words[id / 32] |= 1u32 << (id % 32);
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.mask = mask;
                Ok(())
            }
            Err(cause) => Err(Error {
                cause,
                mask,
                funding: funding.clone(),
            }),
        }
    }
    pub(super) fn commit(
        &mut self,
        state: &OriginalGrammarState,
        token: u32,
        funding: &HostMetadataFunding,
    ) -> Result<Option<OriginalGrammarState>, Error> {
        let result = (|| -> Result<Option<OriginalGrammarState>, Cause> {
            funding.reserve_metadata(controls().ok_or(Cause::Overflow)?)?;
            let (trie, trigger) = source(state, self.pending)?;
            if token as usize >= trie.vocab_size() {
                return Err(Cause::Source);
            }
            let bytes = vocabulary::token_bytes(trie, token);
            if let Some(found) = trigger::find(self.pending.bytes(trigger), bytes, trigger) {
                let copied = state.try_copy(funding)?;
                let (active, accepted) = copied.try_activation(token, found)?;
                if !accepted {
                    return Err(Cause::Rejected);
                }
                Ok(Some(active))
            } else {
                self.pending.advance(bytes, trigger);
                self.mask = None;
                Ok(None)
            }
        })();
        result.map_err(|cause| Error {
            cause,
            mask: None,
            funding: funding.clone(),
        })
    }
    pub(super) fn mask<'a>(
        &'a self,
        validity: &'a SharedTokenFilter,
    ) -> Result<PackedTokenFilter<'a>, PackedTokenFilterError> {
        let mask = self.mask.as_ref().ok_or(PackedTokenFilterError::Geometry)?;
        PackedTokenFilter::new(&mask.words, mask.vocabulary, validity)
    }
}
