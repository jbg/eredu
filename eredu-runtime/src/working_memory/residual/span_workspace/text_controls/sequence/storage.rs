//! Concrete finite provider. No constructor accepts independent bytes or custody.
use super::*;
use crate::working_memory::funding::{
    GenerationCopyCustody, GenerationCopySource, RawSpanHostOwner,
};
mod copy;
use eredu_core::{
    BackendFailure, BackendFailureKind, GenerationError, GenerationSequenceAdmissionError,
    GenerationSequenceRequest, GenerationTokenIdStorage, GenerationTokenIds,
    GenerationTokenIdsIntoIter, RetainedGenerationSequenceStorage, RetainedGenerationStorage,
    RetainedGenerationStorageOwner, RetainedSequenceConstructionError,
    RetainedSequencePreparationError,
};
use std::{alloc::Layout, collections::TryReserveError, fmt, sync::atomic::AtomicUsize};

#[derive(Debug)]
struct Payload {
    tokens: Vec<u32>,
    eos: Vec<u32>,
    committed: usize,
    terminal: Option<TerminalText>,
    // Buffers precede the same original raw hold. No source witness/back-edge.
    _custody: PayloadCustody,
}
#[derive(Debug)]
struct TerminalText {
    bytes: Vec<u8>,
    bound: usize,
}
impl TerminalText {
    fn append(&mut self, text: &str) -> Result<(), eredu_core::GenerationDecoderError> {
        let end = self
            .bytes
            .len()
            .checked_add(text.len())
            .ok_or(eredu_core::GenerationDecoderError::InvalidStorage)?;
        if end > self.bound || end > self.bytes.capacity() {
            return Err(eredu_core::GenerationDecoderError::InvalidStorage);
        }
        // Only complete UTF-8 source fragments enter this bounded destination.
        self.bytes.extend_from_slice(text.as_bytes());
        Ok(())
    }
}
impl GenerationTokenIdStorage for Payload {
    fn terminal_text(&self) -> Option<&str> {
        self.terminal.as_ref().map(|text| {
            std::str::from_utf8(&text.bytes).expect("only complete UTF-8 fragments were appended")
        })
    }
    fn token_ids(&self) -> &[u32] {
        &self.tokens[..self.committed]
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
#[derive(Debug)]
struct PayloadOwner(Option<Arc<Payload>>);
impl PayloadOwner {
    fn get(&self) -> &Payload {
        self.0.as_deref().expect("live sequence payload")
    }
    fn get_mut(&mut self) -> &mut Payload {
        Arc::get_mut(self.0.as_mut().expect("live sequence payload"))
            .expect("sole mutable sequence payload")
    }
    fn freeze(mut self, committed: usize) -> GenerationTokenIds {
        self.get_mut().committed = committed;
        GenerationTokenIds::from_owner(self.0.take().expect("live sequence payload"))
    }
}
impl Drop for PayloadOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}
#[derive(Debug)]
struct Provider {
    #[cfg(test)]
    fail_terminal_reserve: bool,
    decoder: Option<OriginalGenerationDecoderSource>,
    maximum: usize,
    consumer: Option<eredu_core::GenerationSequenceConsumerLayout>,
    authority: ProviderAuthority,
    payload: PayloadOwner,
}
#[derive(Debug)]
enum PayloadCustody {
    Original(RawSpanHostOwner),
    Copy(GenerationCopyCustody),
}
#[derive(Debug)]
enum ProviderAuthority {
    Original {
        reservation: WorkingMemoryReservation,
        controls: OriginalTextControlGuard,
    },
    Copy(GenerationCopyCustody),
}
impl ProviderAuthority {
    fn validate(&self) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Original {
                reservation,
                controls,
            } => controls.validate_reservation(reservation),
            Self::Copy(c) => c.validate(),
        }
    }
    fn source(&self) -> GenerationCopySource<'_> {
        match self {
            Self::Original {
                reservation,
                controls,
            } => GenerationCopySource::Original(controls, reservation),
            Self::Copy(c) => GenerationCopySource::Copied(c),
        }
    }
    fn pool(&self) -> &WorkingMemoryPool {
        match self {
            Self::Original { controls, .. } => controls.custody.raw().pool(),
            Self::Copy(c) => c.pool(),
        }
    }
}
fn unbox<T>(owner: Box<T>) -> T {
    *owner
}
impl RetainedGenerationStorage for Provider {
    fn copy_source(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn consumer_layout(&self) -> Option<&eredu_core::GenerationSequenceConsumerLayout> {
        self.consumer.as_ref()
    }
    fn matches_decoder_input(
        &self,
        input: Option<&dyn eredu_core::GenerationDecoderInput>,
    ) -> bool {
        match &self.decoder {
            Some(decoder) => decoder.matches(input),
            None => input.is_none(),
        }
    }
    fn decoder_output(&self) -> eredu_core::GenerationDecoderOutput {
        self.decoder.as_ref().map_or(
            eredu_core::GenerationDecoderOutput::Suffix,
            OriginalGenerationDecoderSource::output_kind,
        )
    }
    fn project_plain_text(
        &mut self,
        id: u32,
    ) -> Result<eredu_core::GenerationPlainText<'_>, eredu_core::GenerationDecoderError> {
        let text = self
            .decoder
            .as_mut()
            .ok_or(eredu_core::GenerationDecoderError::Unavailable)?
            .project(id)?;
        if let Some(terminal) = &mut self.payload.get_mut().terminal {
            terminal.append(text.visible)?;
        }
        Ok(text)
    }
    fn finish_plain_text(&mut self) -> Result<&str, eredu_core::GenerationDecoderError> {
        let text = self
            .decoder
            .as_mut()
            .ok_or(eredu_core::GenerationDecoderError::Unavailable)?
            .finish_plain()?;
        if let Some(terminal) = &mut self.payload.get_mut().terminal {
            terminal.append(text)?;
        }
        Ok(text)
    }
    fn decode_token(
        &mut self,
        id: u32,
    ) -> Result<Option<&str>, eredu_core::GenerationDecoderError> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or(eredu_core::GenerationDecoderError::Unavailable)?;
        if decoder.stops.is_some() {
            return Err(eredu_core::GenerationDecoderError::Unavailable);
        }
        decoder.storage.step(id).map_err(decoder::transition_error)
    }
    fn finish_decoder(&mut self) -> Result<(), eredu_core::GenerationDecoderError> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or(eredu_core::GenerationDecoderError::Unavailable)?;
        if decoder.stops.is_some() {
            return Err(eredu_core::GenerationDecoderError::Unavailable);
        }
        decoder.storage.finish().map_err(decoder::transition_error)
    }
    fn max_tokens(&self) -> usize {
        self.maximum
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.payload.get().eos
    }
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure> {
        self.authority
            .validate()
            .map_err(|cause| BackendFailure::new(BackendFailureKind::InvalidSession, cause))?;
        let slots = &mut self.payload.get_mut().tokens;
        slots
            .try_reserve_exact(self.maximum)
            .map_err(|cause| BackendFailure::new(BackendFailureKind::ResourceExhausted, cause))?;
        slots.resize(self.maximum, 0);
        if let Some(decoder) = &mut self.decoder {
            decoder.storage.prepare_destinations().map_err(|cause| {
                BackendFailure::new(BackendFailureKind::ResourceExhausted, cause)
            })?;
            if let Some(stops) = &mut decoder.stops {
                stops.prepare_destination().map_err(|cause| {
                    BackendFailure::new(BackendFailureKind::ResourceExhausted, cause)
                })?;
            }
        }
        #[cfg(test)]
        if self.fail_terminal_reserve {
            assert_eq!(self.payload.get().tokens.len(), self.maximum);
            assert!(self.decoder.is_some()); // its four buffers and stop reserve succeeded above
        }
        #[cfg(test)]
        let fail_terminal = self.fail_terminal_reserve;
        if let Some(terminal) = &mut self.payload.get_mut().terminal {
            let requested = terminal.bound;
            #[cfg(test)]
            let requested = if fail_terminal { usize::MAX } else { requested };
            terminal
                .bytes
                .try_reserve_exact(requested)
                .map_err(|cause| {
                    BackendFailure::new(BackendFailureKind::ResourceExhausted, cause)
                })?;
            if terminal.bytes.capacity() > terminal.bound {
                return Err(BackendFailure::new(
                    BackendFailureKind::ResourceExhausted,
                    WorkingMemoryError::Overflow,
                ));
            }
        }
        Ok(())
    }
    fn token_slots(&self) -> &[u32] {
        &self.payload.get().tokens
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        &mut self.payload.get_mut().tokens
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds {
        let Self {
            decoder,
            payload,
            authority,
            ..
        } = unbox(self);
        // Source and all four destinations retire while full source custody and
        // the existing raw R owner still protect this exact provider extraction.
        drop(decoder);
        let result = payload.freeze(committed);
        // The existing result raw owner protects all local validation controls.
        drop(authority);
        result
    }
    fn retire(self: Box<Self>) {
        drop(unbox(self));
    }
}

#[derive(Debug)]
pub(in crate::working_memory) enum SequenceExtractionCause {
    Memory(WorkingMemoryError),
    Allocation(TryReserveError),
    InvalidStorage,
}
impl fmt::Display for SequenceExtractionCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(cause) => fmt::Display::fmt(cause, f),
            Self::Allocation(cause) => fmt::Display::fmt(cause, f),
            Self::InvalidStorage => f.write_str("invalid original sequence storage"),
        }
    }
}
impl std::error::Error for SequenceExtractionCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Memory(e) => Some(e),
            Self::Allocation(e) => Some(e),
            Self::InvalidStorage => None,
        }
    }
}
#[derive(Debug)]
enum Partial {
    Payload(PayloadOwner),
    Constructed(RetainedGenerationSequence),
    Invalid(RetainedSequenceConstructionError),
}
/// The single consuming failure. No owned cause, bank or partial extraction is
/// available. Core's concrete source owner retires its Box before this value.
#[derive(Debug)]
pub(in crate::working_memory) struct SequenceExtractionError {
    cause: SequenceExtractionCause,
    _partial: Option<Partial>,
    _bank: OriginalGenerationSequenceBank,
}
impl SequenceExtractionError {
    pub(super) fn into_failure(self) -> BackendFailure {
        let kind = match &self.cause {
            SequenceExtractionCause::Memory(_) => BackendFailureKind::InvalidSession,
            SequenceExtractionCause::Allocation(_) => BackendFailureKind::ResourceExhausted,
            SequenceExtractionCause::InvalidStorage => BackendFailureKind::InvalidInput,
        };
        BackendFailure::new(kind, self)
    }
    pub(in crate::working_memory) fn cause(&self) -> &SequenceExtractionCause {
        &self.cause
    }
    pub(super) fn memory(cause: WorkingMemoryError, bank: OriginalGenerationSequenceBank) -> Self {
        Self {
            cause: SequenceExtractionCause::Memory(cause),
            _partial: None,
            _bank: bank,
        }
    }
    pub(super) fn prepared(
        cause: WorkingMemoryError,
        sequence: RetainedGenerationSequence,
        bank: OriginalGenerationSequenceBank,
    ) -> Self {
        Self {
            cause: SequenceExtractionCause::Memory(cause),
            _partial: Some(Partial::Constructed(sequence)),
            _bank: bank,
        }
    }
}
impl fmt::Display for SequenceExtractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for SequenceExtractionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

// Temporary constructor result stays underneath the lexical consumed bank.
// It contains no independently minted custody or caller callback.
#[derive(Debug)]
pub(super) struct ConstructionFailure {
    cause: SequenceExtractionCause,
    partial: Partial,
}
impl ConstructionFailure {
    pub(super) fn with_bank(self, bank: OriginalGenerationSequenceBank) -> SequenceExtractionError {
        SequenceExtractionError {
            cause: self.cause,
            _partial: Some(self.partial),
            _bank: bank,
        }
    }
}

pub(super) fn construct(
    request: &GenerationSequenceRequest<'_>,
    bank: &mut OriginalGenerationSequenceBank,
) -> Result<RetainedGenerationSequence, ConstructionFailure> {
    let mut payload = PayloadOwner(Some(Arc::new(Payload {
        tokens: Vec::new(),
        eos: Vec::new(),
        committed: 0,
        terminal: bank.binding.terminal_bytes.map(|bound| TerminalText {
            bytes: Vec::new(),
            bound,
        }),
        _custody: PayloadCustody::Original(bank.controls.custody.raw().clone()),
    })));
    if let Err(cause) = payload
        .get_mut()
        .eos
        .try_reserve_exact(request.eos_token_ids().len())
    {
        return Err(ConstructionFailure {
            cause: SequenceExtractionCause::Allocation(cause),
            partial: Partial::Payload(payload),
        });
    }
    payload
        .get_mut()
        .eos
        .extend_from_slice(request.eos_token_ids());
    payload.get_mut().eos.sort_unstable();
    payload.get_mut().eos.dedup();
    let owner = Box::new(Provider {
        #[cfg(test)]
        fail_terminal_reserve: bank.fail_terminal_reserve,
        decoder: bank.decoder.take(),
        maximum: request.max_new_tokens(),
        // Consume the already validated original bank association, never a new
        // descriptor copied from the current caller after acceptance.
        consumer: bank.binding.consumer,
        authority: ProviderAuthority::Original {
            reservation: bank.reservation.clone(),
            controls: bank.controls.clone(),
        },
        payload,
    });
    RetainedGenerationSequence::from_retained_storage(owner).map_err(|owner| ConstructionFailure {
        cause: SequenceExtractionCause::InvalidStorage,
        partial: Partial::Invalid(owner),
    })
}

// Distinct one-owner phases: extraction or core rejection, or an outer owned
// preparation error concurrently retaining its inner provider error source.
// The mutable consuming core protocol cannot accumulate these phases.
fn error_retention_peak_bytes() -> Option<usize> {
    let preparation_source = BackendFailure::source_retention_peak_bytes::<WorkingMemoryError>()?
        .max(BackendFailure::source_retention_peak_bytes::<
            TryReserveError,
        >()?)
        .max(BackendFailure::source_retention_peak_bytes::<
            GenerationError,
        >()?)
        .max(BackendFailure::source_retention_peak_bytes::<
            eredu_text::decoder_storage::OwnedDecodePreparationError,
        >()?)
        .max(BackendFailure::source_retention_peak_bytes::<
            eredu_text::stop_storage::StopPreparationError,
        >()?);
    let preparation =
        BackendFailure::source_retention_peak_bytes::<RetainedSequencePreparationError>()?
            .checked_add(preparation_source)?;
    Some(
        BackendFailure::source_retention_peak_bytes::<SequenceExtractionError>()?
            .max(GenerationSequenceAdmissionError::rejected_sequence_retention_peak_bytes()?)
            .max(preparation),
    )
}

// Requested layouts under the same supported std Arc layout policy as original
// P. Layout::extend includes payload alignment and terminal padding explicitly.
// These are concrete allocation requests, not allocator RSS/usable-size bounds.
pub(super) fn required_bytes(maximum: usize, eos: usize) -> Result<u64, WorkingMemoryError> {
    let slots = Layout::array::<u32>(maximum)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .size();
    let policy = Layout::array::<u32>(eos)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .size();
    let arc = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<Payload>())
        .map_err(|_| WorkingMemoryError::Overflow)?
        .0
        .pad_to_align()
        .size();
    // Constructor: local payload/provider representations and the by-value
    // owner-preserving error. Retirement: extracted Arc payload and provider.
    // Core: mutable adapter, construction/preparation failure and owning result
    // /iterator carriers. Core-owned error source overlaps are measured below.
    let failure_sources = error_retention_peak_bytes().ok_or(WorkingMemoryError::Overflow)?;
    let parts = [
        size_of::<GenerationSequenceRequest<'static>>(),
        size_of::<GenerationSequencePreparation<'static, 'static>>(),
        slots,
        policy,
        arc,
        size_of::<Provider>(),
        size_of::<Payload>(),
        size_of::<TerminalText>(), // concrete destination construction/move overlap
        size_of::<Option<TerminalText>>(),
        size_of::<PayloadOwner>(),
        size_of::<Provider>(),
        size_of::<Option<Payload>>(),
        size_of::<Arc<Payload>>(),
        size_of::<OriginalGenerationSequenceBank>(),
        size_of::<Option<OriginalGenerationSequenceBank>>(),
        size_of::<ConstructionFailure>(),
        size_of::<Result<RetainedGenerationSequence, ConstructionFailure>>(),
        size_of::<Result<RetainedGenerationSequence, BackendFailure>>(),
        size_of::<WorkingMemoryReservation>(),
        size_of::<OriginalTextControlGuard>(),
        size_of::<RetainedGenerationSequence>(),
        size_of::<RetainedGenerationSequenceStorage>(),
        size_of::<RetainedGenerationStorageOwner>(),
        size_of::<RetainedSequenceConstructionError>(),
        size_of::<RetainedSequencePreparationError>(),
        size_of::<GenerationTokenIds>(),
        size_of::<GenerationTokenIdsIntoIter>(),
        failure_sources,
    ];
    let total = parts.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
    })?;
    u64::try_from(total).map_err(|_| WorkingMemoryError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_sequence_requested_layout_has_exact_token_eos_delta_and_rejects_overflow() {
        assert_eq!(
            required_bytes(31, 7).unwrap() - required_bytes(3, 2).unwrap(),
            4 * (28 + 5)
        );
        assert!(matches!(
            required_bytes(usize::MAX, 0),
            Err(WorkingMemoryError::Overflow)
        ));
        assert!(matches!(
            required_bytes(0, usize::MAX),
            Err(WorkingMemoryError::Overflow)
        ));
        assert!(
            required_bytes(0, 0).unwrap() > 0,
            "dormant/empty still owns concrete controls"
        );
    }
}
