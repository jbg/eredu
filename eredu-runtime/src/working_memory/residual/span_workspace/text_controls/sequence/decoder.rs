//! Concrete precompiled source transport; no HF compiler/tokenizer policy here.
use super::*;
use crate::working_memory::{LoadedDecodeSource, OriginalStopSource};
use eredu_core::{
    BackendFailure, GenerationDecoderError, GenerationDecoderInput, GenerationSequenceBankRejection,
};
use eredu_core::{GenerationDecoderOutput, GenerationPlainText};
use eredu_text::decoder_storage::{
    DecodeDestinations, DecodeStorageError, DecodeStreamLayout, OwnedDecodePreparationError,
    OwnedDecodeStorage, OwnedDecodeStorageError, PreparedDecodeSource,
};
use eredu_text::stop_storage::{OwnedStopStorage, PreparedStopSource, StopStorageError};
use std::{
    any::Any,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex, TryLockError,
    },
};

mod shared;
pub(super) use shared::DecoderStateCopyError;
pub(in crate::working_memory) use shared::OriginalTokenDomainBinding;
pub use shared::{AggregateGenerationDecoderInput, LoadedGenerationDecoderInput};
use shared::{DecoderStorage, StopStorage};

static NEXT_SOURCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DecoderBinding {
    // Equality only; never dereferenced or exported. The concrete unique Box
    // or closed shared lease keeps this exact source alive through the provider.
    // A checked process-unique nonce additionally rejects address reuse after a
    // low-level caller discards a detached source but retains its input header.
    source: usize,
    stop_source: Option<usize>,
    nonce: u64,
    maximum: usize,
    skip_special: bool,
    output: GenerationDecoderOutput,
    pub(super) bytes: u64,
}
impl DecoderBinding {
    pub(super) fn terminal_text_bytes(
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Option<usize>, WorkingMemoryError> {
        if !claim
            .request()
            .consumer_layout()
            .is_some_and(|c| c.terminal_text_output())
        {
            return Ok(None);
        }
        let header = claim
            .request()
            .decoder_input()
            .and_then(|input| {
                input
                    .as_any()
                    .downcast_ref::<AggregateGenerationDecoderInput>()
            })
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if Self::for_claim(claim)? != Some(header.binding)
            || header.binding.output != GenerationDecoderOutput::PlainText
            || header.stop_source().is_none()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let layout = DecodeStreamLayout::for_source(
            header.source().decode_source(),
            header.binding.maximum,
            header.binding.skip_special,
        )
        .map_err(|_| WorkingMemoryError::Overflow)?;
        // L is the complete retained-history candidate bound for this maximum,
        // not a one-token width. Each of M successful emissions is bounded by L;
        // stops only withhold/release/discard those bytes, including final flush.
        // This intentionally conservative general envelope may be quadratic in M.
        // A linear total bound would require a separate repair/compaction proof.
        let bytes = header
            .binding
            .maximum
            .checked_mul(layout.text_capacity())
            .ok_or(WorkingMemoryError::Overflow)?;
        std::alloc::Layout::array::<u8>(bytes).map_err(|_| WorkingMemoryError::Overflow)?;
        Ok(Some(bytes))
    }
    fn for_storage(
        storage: &OwnedDecodeStorage,
        stops: Option<&OwnedStopStorage>,
    ) -> Result<Self, DecodeStorageError> {
        let controls = Self::control_bytes(
            OwnedDecodeStorage::control_bytes(),
            size_of::<OriginalGenerationDecoderInput>(),
        )?;
        let stop_bytes = stops.map_or(Ok(0), |stops| {
            stops
                .source()
                .storage_bytes()
                .and_then(|n| n.checked_sub(size_of::<PreparedStopSource>()))
                .and_then(|n| n.checked_add(stops.destination_bytes()))
                .and_then(|n| n.checked_add(OwnedStopStorage::control_bytes()?))
                .ok_or(DecodeStorageError::Overflow)
        })?;
        let bytes = storage
            .source()
            .storage_bytes()
            .checked_add(storage.destination_bytes())
            .and_then(|n| n.checked_add(stop_bytes))
            .and_then(|n| n.checked_add(controls))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(DecodeStorageError::Overflow)?;
        Self::new(
            storage.source(),
            None, // Unique combined source is sealed by the checked request nonce.
            storage.token_capacity(),
            storage.skip_special_tokens(),
            if stops.is_some() {
                GenerationDecoderOutput::PlainText
            } else {
                GenerationDecoderOutput::Suffix
            },
            bytes,
        )
    }
    fn control_bytes(storage: Option<usize>, input: usize) -> Result<usize, DecodeStorageError> {
        [
            storage.ok_or(DecodeStorageError::Overflow)?,
            input,
            size_of::<StopStorage>(),
            size_of::<Option<StopStorage>>(),
            size_of::<DecoderStorage>(),
            size_of::<Result<DecoderStorage, DecodeStorageError>>(),
            size_of::<OriginalGenerationDecoderSource>(),
            size_of::<Option<OriginalGenerationDecoderSource>>(),
            size_of::<Result<Option<OriginalGenerationDecoderSource>, BackendFailure>>(),
            size_of::<Result<OriginalGenerationDecoderSource, BackendFailure>>(),
            size_of::<Result<OriginalGenerationSequenceBank, BackendFailure>>(),
            size_of::<Result<Option<&'static str>, GenerationDecoderError>>(),
            size_of::<Result<(), GenerationDecoderError>>(),
            size_of::<Result<GenerationPlainText<'static>, GenerationDecoderError>>(),
            size_of::<GenerationPlainText<'static>>(),
            size_of::<Result<&'static str, GenerationDecoderError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(DecodeStorageError::Overflow)
    }
    fn new(
        source: &PreparedDecodeSource,
        stops: Option<&PreparedStopSource>,
        maximum: usize,
        skip_special: bool,
        output: GenerationDecoderOutput,
        bytes: u64,
    ) -> Result<Self, DecodeStorageError> {
        let nonce = NEXT_SOURCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| DecodeStorageError::Overflow)?;
        Ok(Self {
            nonce,
            source: std::ptr::from_ref(source) as usize,
            stop_source: stops.map(|source| std::ptr::from_ref(source) as usize),
            maximum,
            skip_special,
            output,
            bytes,
        })
    }
    pub(super) fn for_claim(
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Option<Self>, WorkingMemoryError> {
        let Some(input) = claim.request().decoder_input() else {
            return Ok(None);
        };
        let binding = input_binding(input).ok_or(WorkingMemoryError::IdentityMismatch)?;
        if binding.maximum != claim.request().max_new_tokens()
            || claim.request().consumer_layout().is_some_and(|layout| {
                layout.plain_text_output() != (binding.output == GenerationDecoderOutput::PlainText)
            })
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Some(binding))
    }
}

/// One actual precompiled decoder source and its immutable original input.
///
/// Construction is cold/unfunded: the caller owns compiler/HF residency and
/// compilation peak. The accepting backend must take the source once before
/// adaptive candidate retries, which subsequently borrow only this fixed header.
/// No Clone, refill, source replacement, caller byte fact or custody constructor.
#[derive(Debug)]
pub struct OriginalGenerationDecoderInput {
    binding: DecoderBinding,
    pending: Mutex<Option<OriginalGenerationDecoderSource>>,
}
impl OriginalGenerationDecoderInput {
    /// Takes one actual text-compiled source and derives N/policy/buffer facts.
    pub fn new(
        source: PreparedDecodeSource,
        maximum: usize,
        skip_special: bool,
    ) -> Result<Self, DecodeStorageError> {
        Self::construct(source, maximum, skip_special, None)
    }
    /// Takes one actual compiled decoder and ordered literal-stop source before
    /// original admission. No later stop attachment, policy replacement or refill.
    /// Packing/compilation peak remains the cold caller's obligation.
    pub fn new_plain_text(
        source: PreparedDecodeSource,
        maximum: usize,
        skip_special: bool,
        stops: PreparedStopSource,
    ) -> Result<Self, DecodeStorageError> {
        Self::construct(source, maximum, skip_special, Some(stops))
    }
    fn construct(
        source: PreparedDecodeSource,
        maximum: usize,
        skip_special: bool,
        stops: Option<PreparedStopSource>,
    ) -> Result<Self, DecodeStorageError> {
        let layout = DecodeStreamLayout::for_source(&source, maximum, skip_special)?;
        let stops = stops
            .map(|source| {
                OwnedStopStorage::new(source, layout.text_capacity())
                    .map_err(|_| DecodeStorageError::Overflow)
            })
            .transpose()?;
        let storage = OwnedDecodeStorage::new(source, maximum, skip_special)?;
        // The checked nonce authenticates this complete immutable combined
        // program throughout its one cold take and final provider retirement.
        let binding = DecoderBinding::for_storage(&storage, stops.as_ref())?;
        let pending = Mutex::new(Some(OriginalGenerationDecoderSource {
            binding,
            storage: DecoderStorage::Owned(storage),
            stops: stops.map(StopStorage::Owned),
        }));
        // Keep the existing cold Darwin mutex initialization before acceptance.
        drop(pending.lock().expect("fresh decoder input mutex"));
        Ok(Self { binding, pending })
    }
}
impl GenerationDecoderInput for OriginalGenerationDecoderInput {
    fn output_kind(&self) -> GenerationDecoderOutput {
        self.binding.output
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Request source removed before the first original chunk-candidate estimate.
/// It is staging until consumed by the matching accepted sequence bank. A shared
/// lease retains its separate original cold allowance; it grants no request
/// authority. Unique storage retains its existing caller scope.
#[derive(Debug)]
pub struct OriginalGenerationDecoderSource {
    pub(super) binding: DecoderBinding,
    pub(super) storage: DecoderStorage,
    pub(super) stops: Option<StopStorage>,
}
impl OriginalGenerationDecoderSource {
    /// Takes the source once before adaptive retries and validates its actual
    /// target account before consuming a shared header. Replay, busy and foreign
    /// inputs retain their fixed typed rejection without consuming the source.
    pub fn take_original(
        claim: &GenerationSequencePreparation<'_, '_>,
        pool: &WorkingMemoryPool,
    ) -> Result<Option<Self>, BackendFailure> {
        DecoderBinding::terminal_text_bytes(claim).map_err(|_| {
            GenerationSequenceBankRejection::IdentityMismatch.into_backend_failure()
        })?;
        let Some(input) = claim.request().decoder_input() else {
            return Ok(None);
        };
        if let Some(input) = input
            .as_any()
            .downcast_ref::<LoadedGenerationDecoderInput>()
        {
            return input.take(claim, pool).map(Some);
        }
        if let Some(input) = input
            .as_any()
            .downcast_ref::<AggregateGenerationDecoderInput>()
        {
            return input.take(claim, pool).map(Some);
        }
        let input = input
            .as_any()
            .downcast_ref::<OriginalGenerationDecoderInput>()
            .filter(|i| {
                i.binding.maximum == claim.request().max_new_tokens()
                    && !claim.request().consumer_layout().is_some_and(|layout| {
                        layout.plain_text_output()
                            != (i.binding.output == GenerationDecoderOutput::PlainText)
                    })
            })
            .ok_or_else(|| {
                GenerationSequenceBankRejection::IdentityMismatch.into_backend_failure()
            })?;
        let source = input
            .pending
            .try_lock()
            .map_err(|error| {
                match error {
                    TryLockError::WouldBlock => GenerationSequenceBankRejection::Busy,
                    TryLockError::Poisoned(_) => GenerationSequenceBankRejection::IdentityMismatch,
                }
                .into_backend_failure()
            })?
            .take();
        source
            .map(Some)
            .ok_or_else(|| GenerationSequenceBankRejection::Unavailable.into_backend_failure())
    }
    pub(super) fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        self.storage.validate_pool(pool)?;
        if let Some(stops) = &self.stops {
            stops.validate_pool(pool)?;
        }
        Ok(())
    }
    pub(super) fn output_kind(&self) -> GenerationDecoderOutput {
        self.binding.output
    }
    pub(super) fn project(
        &mut self,
        id: u32,
    ) -> Result<GenerationPlainText<'_>, GenerationDecoderError> {
        let stops = self
            .stops
            .as_mut()
            .ok_or(GenerationDecoderError::Unavailable)?;
        let text = self
            .storage
            .step(id)
            .map_err(transition_error)?
            .unwrap_or("");
        let result = stops.step(text).map_err(stop_error)?;
        Ok(GenerationPlainText {
            visible: result.visible,
            stop_matched: result.matched.is_some(),
        })
    }
    pub(super) fn finish_plain(&mut self) -> Result<&str, GenerationDecoderError> {
        let stops = self
            .stops
            .as_mut()
            .ok_or(GenerationDecoderError::Unavailable)?;
        self.storage.finish().map_err(transition_error)?;
        stops.finish().map_err(stop_error)
    }
    pub(super) fn matches(&self, input: Option<&dyn GenerationDecoderInput>) -> bool {
        input
            .and_then(input_binding)
            .is_some_and(|binding| self.binding == binding)
    }
}
fn input_binding(input: &dyn GenerationDecoderInput) -> Option<DecoderBinding> {
    input
        .as_any()
        .downcast_ref::<OriginalGenerationDecoderInput>()
        .map(|i| i.binding)
        .or_else(|| {
            input
                .as_any()
                .downcast_ref::<LoadedGenerationDecoderInput>()
                .map(|i| i.binding)
        })
        .or_else(|| {
            input
                .as_any()
                .downcast_ref::<AggregateGenerationDecoderInput>()
                .map(|i| i.binding)
        })
}
pub(super) use crate::working_memory::decoder_transition::{transition_error, stop_error};

impl OriginalGenerationDecoderSource {
    pub(super) fn copy_required_bytes(&self) -> Option<usize> {
        let stops = match &self.stops {
            Some(s) => s.copy_required_bytes()?,
            None => 0,
        };
        self.storage
            .copy_required_bytes()?
            .checked_add(stops)?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<Result<Self, DecoderStateCopyError>>())
    }
    pub(super) fn copy_state(&self) -> Result<Self, DecoderStateCopyError> {
        // Pointer equality and nonce remain attached to these SAME retained
        // programs. Only the five mutable destinations and progress are copied.
        let storage = self.storage.copy_state()?;
        let stops = self
            .stops
            .as_ref()
            .map(StopStorage::copy_state)
            .transpose()?;
        Ok(Self {
            binding: self.binding,
            storage,
            stops,
        })
    }
}
