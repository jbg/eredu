//! Exact paid forbidden-trigger inputs and allocation-free provisional decisions.
//! These contracts describe semantic sources, never native sampling authority.
use super::{
    PlainControllerError, PlainControllerHistory, SpeculativeBuffer,
    SpeculativeBufferAllocationError,
    byte_trigger::{self, TriggerPrefix},
};
use crate::{HostPreparationAuthority, SharedTokenFilter};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::{Arc, atomic::AtomicUsize},
};

/// Fixed refusal from the prepared forbidden-trigger worker.
#[derive(Debug, thiserror::Error)]
pub enum ForbiddenControllerError {
    /// This controller has no qualified forbidden source.
    #[error("controller has no prepared forbidden-trigger producer")]
    Unknown,
    /// Exact trigger, vocabulary or destination geometry changed.
    #[error("forbidden controller source or copy geometry changed")]
    Source,
    /// Host geometry overflowed before construction.
    #[error("forbidden controller host geometry overflow")]
    Overflow,
    /// This token would complete the forbidden byte sequence.
    #[error("token {0} would emit a forbidden activation trigger")]
    Forbidden(u32),
    /// Canonical history or tokenizer validity refused the operation.
    #[error(transparent)]
    History(#[from] PlainControllerError),
    /// A real destination allocation failed with its custody retained.
    #[error(transparent)]
    Allocation(#[from] SpeculativeBufferAllocationError),
}

fn offset(bytes: &[u8], index: usize) -> Option<usize> {
    let start = index.checked_mul(size_of::<u64>())?;
    let end = start.checked_add(size_of::<u64>())?;
    usize::try_from(u64::from_le_bytes(bytes.get(start..end)?.try_into().ok()?)).ok()
}
/// Shared decoder for the existing packed vocabulary: little-endian absolute
/// u64 offsets followed by token bytes. This grants no source/account identity.
pub fn packed_controller_token_bytes(
    bytes: &[u8],
    tokens: usize,
    maximum: usize,
    token: usize,
) -> Option<&[u8]> {
    if token >= tokens {
        return None;
    }
    let table = tokens.checked_add(1)?.checked_mul(size_of::<u64>())?;
    let start = offset(bytes, token)?;
    let end = offset(bytes, token.checked_add(1)?)?;
    if start < table {
        return None;
    }
    let value = bytes.get(start..end)?;
    (value.len() <= maximum).then_some(value)
}
fn validate_vocabulary(bytes: &[u8], tokens: usize, maximum: usize) -> bool {
    let Some(table) = tokens
        .checked_add(1)
        .and_then(|n| n.checked_mul(size_of::<u64>()))
    else {
        return false;
    };
    if offset(bytes, 0) != Some(table) || offset(bytes, tokens) != Some(bytes.len()) {
        return false;
    }
    (0..tokens).all(|token| packed_controller_token_bytes(bytes, tokens, maximum, token).is_some())
}
/// Borrowed exact immutable input copy plan. It owns no payload and grants no
/// source credit; an original compiler pays its actual destinations before copy.
#[derive(Debug, Clone, Copy)]
pub struct PreparedForbiddenInputCopy<'a> {
    vocabulary: &'a [u8],
    tokens: usize,
    maximum: usize,
    trigger: &'a [u8],
}
impl<'a> PreparedForbiddenInputCopy<'a> {
    /// Validates the same packed decoder and nonempty trigger without allocating.
    pub fn new(
        vocabulary: &'a [u8],
        tokens: usize,
        maximum: usize,
        trigger: &'a [u8],
    ) -> Result<Self, ForbiddenControllerError> {
        if trigger.is_empty() || !validate_vocabulary(vocabulary, tokens, maximum) {
            return Err(ForbiddenControllerError::Source);
        }
        Ok(Self {
            vocabulary,
            tokens,
            maximum,
            trigger,
        })
    }
    /// Actual copied bytes, shared owner, validation and plan transport frames.
    pub fn required_bytes(self) -> Option<usize> {
        let parts = [
            ForbiddenControllerInputs::copy_metadata_bytes(
                self.vocabulary.len(),
                self.trigger.len(),
            )?,
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, ForbiddenControllerError>>(),
            std::mem::size_of::<(Self, HostPreparationAuthority)>(),
            std::mem::size_of::<Result<ForbiddenControllerInputs, ForbiddenControllerError>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Executes the same copy producer under the compiler's prepaid authority.
    pub fn copy(
        self,
        host: HostPreparationAuthority,
    ) -> Result<ForbiddenControllerInputs, ForbiddenControllerError> {
        ForbiddenControllerInputs::copy_prepared(
            self.vocabulary,
            self.tokens,
            self.maximum,
            self.trigger,
            host,
        )
    }
}

struct Inputs {
    vocabulary: SpeculativeBuffer<u8>,
    trigger: SpeculativeBuffer<u8>,
    tokens: usize,
    maximum: usize,
}
/// Closed immutable paid copies of the actual vocabulary and trigger. Aliases
/// retain their buffers' real host authority; equality is owner identity.
#[derive(Clone)]
pub struct ForbiddenControllerInputs(Arc<Inputs>);
impl std::fmt::Debug for ForbiddenControllerInputs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForbiddenControllerInputs")
            .field("tokens", &self.0.tokens)
            .field("trigger_bytes", &self.0.trigger.len())
            .finish_non_exhaustive()
    }
}
impl ForbiddenControllerInputs {
    /// Exact independent byte destinations, Arc allocation, constructor and
    /// validation frames. Caller separately funds its authority/error wrapper.
    pub fn copy_metadata_bytes(vocabulary_bytes: usize, trigger_bytes: usize) -> Option<usize> {
        let shared = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Inputs>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            shared,
            SpeculativeBuffer::<u8>::retained_control_bytes(vocabulary_bytes)?,
            SpeculativeBuffer::<u8>::retained_control_bytes(trigger_bytes)?,
            Self::operation_control_bytes()?,
            size_of::<Inputs>(),
            size_of::<Self>(),
            size_of::<Result<Self, ForbiddenControllerError>>(),
            size_of::<(&[u8], usize, usize, &[u8], HostPreparationAuthority)>(),
            size_of::<std::iter::Copied<std::slice::Iter<'_, u8>>>(),
            size_of::<Result<(), crate::generation::GenerationError>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<[u8; 8]>(),
            size_of::<Option<&[u8]>>(),
            size_of::<std::num::TryFromIntError>(),
            size_of::<std::array::TryFromSliceError>(),
            size_of::<Result<usize, std::num::TryFromIntError>>(),
            size_of::<Result<[u8; 8], std::array::TryFromSliceError>>(),
            size_of::<Option<usize>>(),
            size_of::<usize>() * 7,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Copies the complete finite source under already-paid host custody.
    pub fn copy_prepared(
        vocabulary: &[u8],
        tokens: usize,
        maximum: usize,
        trigger: &[u8],
        host: HostPreparationAuthority,
    ) -> Result<Self, ForbiddenControllerError> {
        if host.is_unmanaged()
            || trigger.is_empty()
            || !validate_vocabulary(vocabulary, tokens, maximum)
        {
            return Err(ForbiddenControllerError::Source);
        }
        let mut copied_vocabulary =
            SpeculativeBuffer::try_new_retained(vocabulary.len(), host.clone())?;
        copied_vocabulary
            .try_extend(vocabulary.iter().copied())
            .map_err(|_| ForbiddenControllerError::Source)?;
        let mut copied_trigger = SpeculativeBuffer::try_new_retained(trigger.len(), host)?;
        copied_trigger
            .try_extend(trigger.iter().copied())
            .map_err(|_| ForbiddenControllerError::Source)?;
        Ok(Self(Arc::new(Inputs {
            vocabulary: copied_vocabulary,
            trigger: copied_trigger,
            tokens,
            maximum,
        })))
    }
    /// Actual packed source bytes, including the offset table.
    pub fn vocabulary(&self) -> &[u8] {
        &self.0.vocabulary
    }
    /// Number of canonical vocabulary entries.
    pub fn vocabulary_len(&self) -> usize {
        self.0.tokens
    }
    /// Actual decoded bytes of a canonical token.
    pub fn token_bytes(&self, token: usize) -> Option<&[u8]> {
        packed_controller_token_bytes(self.vocabulary(), self.0.tokens, self.0.maximum, token)
    }
    /// Actual nonempty forbidden byte sequence.
    pub fn trigger(&self) -> &[u8] {
        &self.0.trigger
    }
    /// Exact immutable owner comparison, never content substitution.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Fixed frames for source/decision/commit/identity operations. Loop state is
    /// reused per token; no mask, activation payload or token history is cloned.
    pub fn operation_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<ForbiddenControllerSource<'_>>(),
            size_of::<ForbiddenControllerDecision<'_>>(),
            size_of::<ForbiddenControllerMutation<'_>>(),
            size_of::<PreparedForbiddenControllerIdentity>(),
            size_of::<Result<ForbiddenControllerSource<'_>, ForbiddenControllerError>>(),
            size_of::<Result<ForbiddenControllerDecision<'_>, ForbiddenControllerError>>(),
            size_of::<Result<PreparedForbiddenControllerIdentity, ForbiddenControllerError>>(),
            size_of::<Result<(), ForbiddenControllerError>>(),
            size_of::<ForbiddenControllerError>(),
            size_of::<PlainControllerError>(),
            size_of::<Result<(), PlainControllerError>>(),
            size_of::<std::slice::Iter<'_, u32>>(),
            size_of::<TriggerPrefix>(),
            size_of::<Option<byte_trigger::TriggerMatch<'_>>>(),
            size_of::<Option<&[u8]>>(),
            byte_trigger::control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

/// Exact retained forbidden source. It deliberately has no plain-controller
/// projection: its candidate domain depends on the current trigger prefix.
#[derive(Debug, Clone, Copy)]
pub struct ForbiddenControllerSource<'a> {
    history: &'a PlainControllerHistory,
    validity: &'a SharedTokenFilter,
    inputs: &'a ForbiddenControllerInputs,
    prefix: TriggerPrefix,
    original: Option<crate::OriginalTokenDomainWitness<'a>>,
}
impl<'a> ForbiddenControllerSource<'a> {
    /// Borrows an actual controller; the supplied prefix belongs to these inputs.
    pub fn new(
        history: &'a PlainControllerHistory,
        validity: &'a SharedTokenFilter,
        inputs: &'a ForbiddenControllerInputs,
        prefix: TriggerPrefix,
    ) -> Result<Self, ForbiddenControllerError> {
        if !prefix.is_valid_for(inputs.trigger()) {
            return Err(ForbiddenControllerError::Source);
        }
        Ok(Self {
            history,
            validity,
            inputs,
            prefix,
            original: None,
        })
    }
    /// Adds a descriptive loan of a closed original source. Native consumers
    /// must downcast the exact recognized compiler owner and authenticate it.
    pub fn with_original_storage(
        mut self,
        original: crate::OriginalTokenDomainWitness<'a>,
    ) -> Self {
        self.original = Some(original);
        self
    }
    /// Borrowed compiler witness; its presence alone grants no authority.
    pub fn original_storage(self) -> Option<crate::OriginalTokenDomainWitness<'a>> {
        self.original
    }
    /// Actual durable canonical prefix.
    pub fn history(self) -> &'a [u32] {
        self.history
    }
    /// Existing bounded history capacity.
    pub fn capacity(self) -> usize {
        self.history.capacity()
    }
    /// Exact tokenizer-validity owner.
    pub fn validity(self) -> &'a SharedTokenFilter {
        self.validity
    }
    /// Exact immutable byte source owner.
    pub fn inputs(self) -> &'a ForbiddenControllerInputs {
        self.inputs
    }
    /// Actual pending prefix into the immutable trigger.
    pub fn prefix(self) -> TriggerPrefix {
        self.prefix
    }
    /// Evaluates one provisional suffix without changing the durable source.
    /// Errors preserve ordinary order: prefix, then each token's validity and
    /// forbidden transition before the next token is examined.
    pub fn decision_at(
        self,
        history: &[u32],
    ) -> Result<ForbiddenControllerDecision<'a>, ForbiddenControllerError> {
        if !history.starts_with(self.history) {
            return Err(PlainControllerError::History.into());
        }
        let mut decision = ForbiddenControllerDecision {
            source: self,
            prefix: self.prefix,
        };
        for &token in &history[self.history.len()..] {
            decision.advance(token)?;
        }
        Ok(decision)
    }
    /// Validates one current candidate without advancing the source.
    pub fn validate_token(self, token: u32) -> Result<(), ForbiddenControllerError> {
        ForbiddenControllerDecision {
            source: self,
            prefix: self.prefix,
        }
        .validate_token(token)
    }
    /// Verifies independent mutable history and the same immutable source owner.
    pub fn matches_copy(self, copied: Self, capacity: usize) -> bool {
        self.validity.same_storage(copied.validity)
            && self.inputs.same_source(copied.inputs)
            && self.prefix == copied.prefix
            && self.history() == copied.history()
            && copied.capacity() == capacity
            && copied.history.is_unique_prepared()
    }
    /// Retains this exact prepared source for completed-value identity checks.
    pub fn retain_prepared_identity(
        self,
    ) -> Result<PreparedForbiddenControllerIdentity, ForbiddenControllerError> {
        if !self.history.is_prepared() {
            return Err(ForbiddenControllerError::Source);
        }
        Ok(PreparedForbiddenControllerIdentity {
            history: self.history.clone(),
            validity: self.validity.clone(),
            inputs: self.inputs.clone(),
            prefix: self.prefix,
        })
    }
    /// Compares this source to an escaped exact prepared identity.
    pub fn matches_prepared_identity(self, identity: &PreparedForbiddenControllerIdentity) -> bool {
        self.history.same_prepared_source(&identity.history)
            && self.validity.same_storage(&identity.validity)
            && self.inputs.same_source(&identity.inputs)
            && self.prefix == identity.prefix
    }
}
/// Fixed candidate predicate at a provisional history. It owns no generated mask.
#[derive(Debug, Clone, Copy)]
pub struct ForbiddenControllerDecision<'a> {
    source: ForbiddenControllerSource<'a>,
    prefix: TriggerPrefix,
}
impl<'a> ForbiddenControllerDecision<'a> {
    /// Actual durable source from which this provisional decision was derived.
    pub fn source(self) -> ForbiddenControllerSource<'a> {
        self.source
    }
    /// Provisional trigger prefix; it does not mutate durable history.
    pub fn prefix(self) -> TriggerPrefix {
        self.prefix
    }
    /// Actual tokenizer domain followed by the shared forbidden byte predicate.
    pub fn allows(self, token: u32) -> bool {
        self.validate_token(token).is_ok()
    }
    /// Fixed refusal for an invalid or forbidden canonical candidate.
    pub fn validate_token(self, token: u32) -> Result<(), ForbiddenControllerError> {
        if !self.source.validity.allows(token) {
            return Err(PlainControllerError::InvalidToken(token).into());
        }
        let bytes = self
            .source
            .inputs
            .token_bytes(token as usize)
            .ok_or(ForbiddenControllerError::Source)?;
        let trigger = self.source.inputs.trigger();
        if byte_trigger::find(self.prefix.bytes(trigger), bytes, trigger).is_some() {
            return Err(ForbiddenControllerError::Forbidden(token));
        }
        Ok(())
    }
    fn advance(&mut self, token: u32) -> Result<(), ForbiddenControllerError> {
        self.validate_token(token)?;
        let bytes = self
            .source
            .inputs
            .token_bytes(token as usize)
            .ok_or(ForbiddenControllerError::Source)?;
        self.prefix.advance(bytes, self.source.inputs.trigger());
        Ok(())
    }
}
/// One lexical mutable loan of the actual durable history and trigger prefix.
pub struct ForbiddenControllerMutation<'a> {
    history: &'a mut PlainControllerHistory,
    validity: &'a SharedTokenFilter,
    inputs: &'a ForbiddenControllerInputs,
    prefix: &'a mut TriggerPrefix,
}
impl<'a> ForbiddenControllerMutation<'a> {
    /// Borrows the exact mutable fields corresponding to a forbidden source.
    pub fn new(
        history: &'a mut PlainControllerHistory,
        validity: &'a SharedTokenFilter,
        inputs: &'a ForbiddenControllerInputs,
        prefix: &'a mut TriggerPrefix,
    ) -> Result<Self, ForbiddenControllerError> {
        ForbiddenControllerSource::new(history, validity, inputs, *prefix)?;
        Ok(Self {
            history,
            validity,
            inputs,
            prefix,
        })
    }
    /// Validates, computes a prospective transition, appends to the unique paid
    /// destination, then publishes the new prefix. Every refusal is atomic.
    pub fn commit(&mut self, token: u32) -> Result<(), ForbiddenControllerError> {
        let next = {
            let source = ForbiddenControllerSource::new(
                self.history,
                self.validity,
                self.inputs,
                *self.prefix,
            )?;
            let mut decision = ForbiddenControllerDecision {
                source,
                prefix: *self.prefix,
            };
            decision.advance(token)?;
            decision.prefix
        };
        self.history.try_push_prepared(token)?;
        *self.prefix = next;
        Ok(())
    }
}
/// Retained semantic identity only; native consumers authenticate and fund their
/// actual sampling source separately. Immutable aliases prevent history mutation.
#[derive(Debug, Clone)]
pub struct PreparedForbiddenControllerIdentity {
    history: PlainControllerHistory,
    validity: SharedTokenFilter,
    inputs: ForbiddenControllerInputs,
    prefix: TriggerPrefix,
}
impl PreparedForbiddenControllerIdentity {
    /// Exact history, vocabulary, trigger, validity and prefix identity.
    pub fn same_source(&self, other: &Self) -> bool {
        self.history.same_prepared_source(&other.history)
            && self.validity.same_storage(&other.validity)
            && self.inputs.same_source(&other.inputs)
            && self.prefix == other.prefix
    }
}
