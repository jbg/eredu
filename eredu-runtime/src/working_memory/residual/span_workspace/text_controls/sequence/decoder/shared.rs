//! Same source/header/provider path for standalone and aggregate cold sources.
use super::*;
use crate::working_memory::OriginalTokenizer;
use eredu_text::stop_storage::{StopDestinations, StopOutput, StopPreparationError};

#[derive(Debug, Clone)]
enum SharedSource {
    Standalone(LoadedDecodeSource),
    Aggregate(OriginalTokenizer),
}
impl SharedSource {
    fn source(&self) -> &PreparedDecodeSource {
        match self {
            Self::Standalone(s) => s.source(),
            Self::Aggregate(s) => s.decode_source(),
        }
    }
    fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Standalone(s) => s.validate_pool(pool),
            Self::Aggregate(s) => s.validate_pool(pool),
        }
    }
}
#[derive(Clone, Copy)]
enum SourceRef<'a> {
    Standalone(&'a LoadedDecodeSource),
    Aggregate(&'a OriginalTokenizer),
}
impl<'a> SourceRef<'a> {
    fn source(self) -> &'a PreparedDecodeSource {
        match self {
            Self::Standalone(s) => s.source(),
            Self::Aggregate(s) => s.decode_source(),
        }
    }
    fn retain(self) -> SharedSource {
        match self {
            Self::Standalone(s) => SharedSource::Standalone(s.clone()),
            Self::Aggregate(s) => SharedSource::Aggregate(s.clone()),
        }
    }
    fn validate_pool(self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Standalone(s) => s.validate_pool(pool),
            Self::Aggregate(s) => s.validate_pool(pool),
        }
    }
    fn validate_stops(self, stops: &OriginalStopSource) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Standalone(s) => stops.validate_decoder_pool(s),
            Self::Aggregate(s) => stops.validate_pool(s.pool()),
        }
    }
}
struct HeaderParts {
    binding: DecoderBinding,
    source: SharedSource,
    stops: Option<OriginalStopSource>,
}
fn construct(
    source: SourceRef<'_>,
    stops: Option<&OriginalStopSource>,
    maximum: usize,
    skip_special: bool,
    input_bytes: usize,
) -> Result<HeaderParts, DecodeStorageError> {
    let layout = DecodeStreamLayout::for_source(source.source(), maximum, skip_special)?;
    let destinations = DecodeDestinations::new(source.source(), maximum, skip_special)?;
    let stop_destinations = stops
        .map(|s| {
            StopDestinations::new(s.source(), layout.text_capacity())
                .map_err(|_| DecodeStorageError::Overflow)
        })
        .transpose()?;
    let stop_bytes = stop_destinations.as_ref().map_or(Ok(0), |d| {
        d.destination_bytes()
            .checked_add(SharedStopStorage::control_bytes().ok_or(DecodeStorageError::Overflow)?)
            .ok_or(DecodeStorageError::Overflow)
    })?;
    let controls = DecoderBinding::control_bytes(SharedStorage::control_bytes(), input_bytes)?;
    let bytes = destinations
        .destination_bytes()
        .checked_add(stop_bytes)
        .and_then(|n| n.checked_add(controls))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(DecodeStorageError::Overflow)?;
    let binding = DecoderBinding::new(
        source.source(),
        stops.map(OriginalStopSource::source),
        maximum,
        skip_special,
        if stops.is_some() {
            GenerationDecoderOutput::PlainText
        } else {
            GenerationDecoderOutput::Suffix
        },
        bytes,
    )?;
    // All fallible geometry is complete before either strong source increment.
    Ok(HeaderParts {
        binding,
        source: source.retain(),
        stops: stops.cloned(),
    })
}
fn take(
    binding: DecoderBinding,
    claimed: &AtomicBool,
    source: SourceRef<'_>,
    stops: Option<&OriginalStopSource>,
    claim: &GenerationSequencePreparation<'_, '_>,
    pool: &WorkingMemoryPool,
) -> Result<OriginalGenerationDecoderSource, BackendFailure> {
    let reject = || GenerationSequenceBankRejection::IdentityMismatch.into_backend_failure();
    source.validate_pool(pool).map_err(|_| reject())?;
    if let Some(stops) = stops {
        stops.validate_pool(pool).map_err(|_| reject())?;
    }
    if DecoderBinding::for_claim(claim).map_err(|_| reject())? != Some(binding) {
        return Err(reject());
    }
    let layout =
        DecodeStreamLayout::for_source(source.source(), binding.maximum, binding.skip_special)
            .map_err(|_| reject())?;
    let destinations =
        DecodeDestinations::new(source.source(), binding.maximum, binding.skip_special)
            .map_err(|_| reject())?;
    let stop_destinations = stops
        .map(|s| StopDestinations::new(s.source(), layout.text_capacity()).map_err(|_| reject()))
        .transpose()?;
    claimed
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| GenerationSequenceBankRejection::Unavailable.into_backend_failure())?;
    // No fallible operation after the one paired claim.
    let stops = stop_destinations.zip(stops).map(|(destinations, source)| {
        StopStorage::Shared(SharedStopStorage {
            destinations,
            source: source.clone(),
        })
    });
    Ok(OriginalGenerationDecoderSource {
        binding,
        storage: DecoderStorage::Shared(SharedStorage {
            destinations,
            source: source.retain(),
        }),
        stops,
    })
}
/// One original request header retaining its exact decoder and optional stop source.
/// Construction derives N/skip/mode/destinations and keeps immutable strong leases.
/// The backend takes the pair once before adaptive candidates. Source charges
/// remain in their cold accounts; R contains only destinations and real controls.
/// No Clone, refill, source swapping or second independent stop claim exists.
#[derive(Debug)]
pub struct LoadedGenerationDecoderInput {
    pub(super) binding: DecoderBinding,
    claimed: AtomicBool,
    source: LoadedDecodeSource,
    stops: Option<OriginalStopSource>,
}
impl LoadedGenerationDecoderInput {
    /// Derives one suffix request without compiling or allocating.
    pub fn new(
        source: &LoadedDecodeSource,
        maximum: usize,
        skip_special: bool,
    ) -> Result<Self, DecodeStorageError> {
        Self::construct(source, None, maximum, skip_special)
    }
    /// Derives an explicit plain request from two already admitted source owners.
    /// Even an empty stop program requires its actual empty original owner.
    /// Foreign accounts reject before either source gains a request lease.
    pub fn new_plain_text(
        source: &LoadedDecodeSource,
        stops: &OriginalStopSource,
        maximum: usize,
        skip_special: bool,
    ) -> Result<Self, WorkingMemoryError> {
        SourceRef::Standalone(source).validate_stops(stops)?;
        Self::construct(source, Some(stops), maximum, skip_special)
            .map_err(|_| WorkingMemoryError::Overflow)
    }
    fn construct(
        source: &LoadedDecodeSource,
        stops: Option<&OriginalStopSource>,
        maximum: usize,
        skip_special: bool,
    ) -> Result<Self, DecodeStorageError> {
        let HeaderParts {
            binding,
            source,
            stops,
        } = construct(
            SourceRef::Standalone(source),
            stops,
            maximum,
            skip_special,
            size_of::<Self>(),
        )?;
        let SharedSource::Standalone(source) = source else {
            unreachable!("same concrete source")
        };
        Ok(Self {
            binding,
            claimed: AtomicBool::new(false),
            source,
            stops,
        })
    }
    /// Borrows the exact cold decoder; no extraction or replacement.
    pub fn source(&self) -> &LoadedDecodeSource {
        &self.source
    }
    /// Borrows the exact optional stop owner. Some(empty) still means plain output.
    pub fn stop_source(&self) -> Option<&OriginalStopSource> {
        self.stops.as_ref()
    }
    pub(super) fn take(
        &self,
        claim: &GenerationSequencePreparation<'_, '_>,
        pool: &WorkingMemoryPool,
    ) -> Result<OriginalGenerationDecoderSource, BackendFailure> {
        take(
            self.binding,
            &self.claimed,
            SourceRef::Standalone(&self.source),
            self.stops.as_ref(),
            claim,
            pool,
        )
    }
}
impl GenerationDecoderInput for LoadedGenerationDecoderInput {
    fn output_kind(&self) -> GenerationDecoderOutput {
        self.binding.output
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
/// One atomic request retaining the entire originally constructed tokenizer.
/// It exposes no raw HF, source replacement, refill or independently funded decoder.
#[derive(Debug)]
pub struct AggregateGenerationDecoderInput {
    pub(super) binding: DecoderBinding,
    claimed: AtomicBool,
    source: OriginalTokenizer,
    stops: Option<OriginalStopSource>,
}
/// Private retained source/binding evidence, never a source-independent byte tag.
#[derive(Debug, Clone)]
pub(in crate::working_memory) struct OriginalTokenDomainBinding {
    source: OriginalTokenizer,
    binding: OriginalTokenDomainEvidence,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OriginalTokenDomainEvidence {
    Initial(DecoderBinding),
    Retained { maximum: usize },
    Semantic { maximum: usize },
}
impl PartialEq for OriginalTokenDomainBinding {
    fn eq(&self, other: &Self) -> bool {
        self.source.same_source(&other.source) && self.binding == other.binding
    }
}
impl Eq for OriginalTokenDomainBinding {}
impl OriginalTokenDomainBinding {
    pub(in crate::working_memory) fn prepare(
        source: &OriginalTokenizer,
        claim: &GenerationSequencePreparation<'_, '_>,
        pool: &WorkingMemoryPool,
    ) -> Result<Self, WorkingMemoryError> {
        if claim.request().semantic_state().is_some() { return Err(WorkingMemoryError::IdentityMismatch); }
        let header = claim
            .request()
            .decoder_input()
            .and_then(|input| {
                input
                    .as_any()
                    .downcast_ref::<AggregateGenerationDecoderInput>()
            })
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        source.validate_pool(pool)?;
        if claim.context().attempt() != 0
            || !source.same_source(&header.source)
            || source.generation_domain().is_none()
            || header.claimed.load(Ordering::Acquire)
            || header.binding.output != GenerationDecoderOutput::PlainText
            || !claim
                .request()
                .consumer_layout()
                .is_some_and(|c| c.plain_text_output())
            || DecoderBinding::for_claim(claim)? != Some(header.binding)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        header
            .stops
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .validate_pool(pool)?;
        Ok(Self {
            source: source.clone(),
            binding: OriginalTokenDomainEvidence::Initial(header.binding),
        })
    }
    pub(in crate::working_memory) fn prepare_semantic(
        source: &OriginalTokenizer, claim: &GenerationSequencePreparation<'_, '_>, pool: &WorkingMemoryPool,
    ) -> Result<Self, WorkingMemoryError> {
        source.validate_pool(pool)?;
        if claim.context().attempt() != 0 || claim.request().max_new_tokens() == 0
            || claim.request().decoder_input().is_some() || claim.request().semantic_state().is_none()
            || !claim.request().consumer_layout().is_some_and(|c| !c.plain_text_output())
            || source.generation_domain().is_none() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Self { source: source.clone(), binding: OriginalTokenDomainEvidence::Semantic {
            maximum: claim.request().max_new_tokens() } })
    }
    pub(in crate::working_memory) fn prepare_retained(
        source: &OriginalTokenizer,
        pool: &WorkingMemoryPool,
        maximum: usize,
    ) -> Result<Self, WorkingMemoryError> {
        source.validate_pool(pool)?;
        // A saved terminal controller still authenticates its token domain,
        // with zero future decisions. This source binding grants no execution;
        // resume admission separately requires terminal StateOnly geometry.
        if source.generation_domain().is_none() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Self {
            source: source.clone(),
            binding: OriginalTokenDomainEvidence::Retained { maximum },
        })
    }
    pub(in crate::working_memory) fn source(&self) -> &OriginalTokenizer {
        &self.source
    }
    pub(in crate::working_memory) fn maximum(&self) -> usize {
        match self.binding {
            OriginalTokenDomainEvidence::Initial(binding) => binding.maximum,
            OriginalTokenDomainEvidence::Retained { maximum } | OriginalTokenDomainEvidence::Semantic { maximum } => maximum,
        }
    }
}
impl AggregateGenerationDecoderInput {
    /// Derives one suffix request without compiling or allocating.
    pub fn new(
        source: &OriginalTokenizer,
        maximum: usize,
        skip_special: bool,
    ) -> Result<Self, DecodeStorageError> {
        Self::construct(source, None, maximum, skip_special)
    }
    /// Derives explicit plain output from the original source and original stop owner.
    /// Even an empty stop program requires its actual empty owner; foreign
    /// accounts reject before either source gains a request lease.
    pub fn new_plain_text(
        source: &OriginalTokenizer,
        stops: &OriginalStopSource,
        maximum: usize,
        skip_special: bool,
    ) -> Result<Self, WorkingMemoryError> {
        SourceRef::Aggregate(source).validate_stops(stops)?;
        Self::construct(source, Some(stops), maximum, skip_special)
            .map_err(|_| WorkingMemoryError::Overflow)
    }
    fn construct(
        source: &OriginalTokenizer,
        stops: Option<&OriginalStopSource>,
        maximum: usize,
        skip_special: bool,
    ) -> Result<Self, DecodeStorageError> {
        let HeaderParts {
            binding,
            source,
            stops,
        } = construct(
            SourceRef::Aggregate(source),
            stops,
            maximum,
            skip_special,
            size_of::<Self>(),
        )?;
        let SharedSource::Aggregate(source) = source else {
            unreachable!("same concrete source")
        };
        Ok(Self {
            binding,
            claimed: AtomicBool::new(false),
            source,
            stops,
        })
    }
    /// Borrows the exact original source; no extraction or replacement.
    pub fn source(&self) -> &OriginalTokenizer {
        &self.source
    }
    /// Borrows the exact optional stop owner. Some(empty) still means plain output.
    pub fn stop_source(&self) -> Option<&OriginalStopSource> {
        self.stops.as_ref()
    }
    pub(super) fn take(
        &self,
        claim: &GenerationSequencePreparation<'_, '_>,
        pool: &WorkingMemoryPool,
    ) -> Result<OriginalGenerationDecoderSource, BackendFailure> {
        take(
            self.binding,
            &self.claimed,
            SourceRef::Aggregate(&self.source),
            self.stops.as_ref(),
            claim,
            pool,
        )
    }
}
impl GenerationDecoderInput for AggregateGenerationDecoderInput {
    fn output_kind(&self) -> GenerationDecoderOutput {
        self.binding.output
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
pub(in super::super) struct SharedStorage {
    // Every destination retires before this source lease.
    destinations: DecodeDestinations,
    source: SharedSource,
}
impl SharedStorage {
    fn control_bytes() -> Option<usize> {
        DecodeDestinations::control_bytes()?
            // The original-domain constructor and pre/post callback evidence are
            // concrete controls. C bytes remain in C; these are no source credit.
            .checked_add(size_of::<OriginalTokenDomainBinding>())?
            .checked_add(size_of::<Option<OriginalTokenDomainBinding>>())?
            .checked_add(size_of::<
                Result<OriginalTokenDomainBinding, WorkingMemoryError>,
            >())?
            .checked_add(size_of::<crate::working_memory::ControllerStorageContract>())?
            .checked_add(crate::working_memory::PreparedControllerBinding::validation_control_bytes()?)?
            .checked_add(size_of::<crate::working_memory::PreparedControllerBinding>())?
            .checked_add(size_of::<Option<crate::working_memory::PreparedControllerBinding>>())?
            .checked_add(size_of::<eredu_core::PreparedControllerSource<'static>>())?
            .checked_add(size_of::<Result<Option<crate::working_memory::PreparedControllerBinding>, WorkingMemoryError>>())?
            .checked_add(size_of::<
                Result<crate::working_memory::ControllerStorageContract, BackendFailure>,
            >())?
            .checked_add(size_of::<eredu_core::OriginalSourceWitness<'static>>())?
            .checked_add(size_of::<
                Option<eredu_core::OriginalSourceWitness<'static>>,
            >())?
            .checked_add(size_of::<eredu_core::TextControllerStorage<'static>>())?
            .checked_add(size_of::<
                Result<(), crate::working_memory::ControllerStorageError>,
            >())?
            .checked_add(size_of::<HeaderParts>())?
            .checked_add(size_of::<Result<HeaderParts, DecodeStorageError>>())?
            .checked_add(size_of::<SourceRef<'_>>())?
            .checked_add(size_of::<AggregateGenerationDecoderInput>())?
            .checked_add(size_of::<
                Result<AggregateGenerationDecoderInput, DecodeStorageError>,
            >())?
            .checked_add(size_of::<
                Result<AggregateGenerationDecoderInput, WorkingMemoryError>,
            >())?
            .checked_add(size_of::<Option<OriginalStopSource>>())?
            .checked_add(size_of::<
                Result<LoadedGenerationDecoderInput, WorkingMemoryError>,
            >())?
            .checked_add(size_of::<
                Result<Option<LoadedGenerationDecoderInput>, WorkingMemoryError>,
            >())?
            .checked_add(size_of::<Option<StopDestinations>>())?
            .checked_add(size_of::<
                Result<Option<StopDestinations>, DecodeStorageError>,
            >())?
            .checked_add(size_of::<Result<Option<StopDestinations>, BackendFailure>>())?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<SharedSource>())?
            .checked_add(size_of::<Result<(), WorkingMemoryError>>())?
            .checked_add(size_of::<
                Result<LoadedGenerationDecoderInput, DecodeStorageError>,
            >())?
            .checked_add(size_of::<
                Result<Option<LoadedGenerationDecoderInput>, DecodeStorageError>,
            >())
    }
}
#[derive(Debug)]
pub(in super::super) enum DecoderStorage {
    Owned(OwnedDecodeStorage),
    Shared(SharedStorage),
}
impl DecoderStorage {
    pub(in super::super) fn validate_pool(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Owned(_) => Ok(()),
            Self::Shared(s) => s.source.validate_pool(pool),
        }
    }
    pub(in super::super) fn prepare_destinations(
        &mut self,
    ) -> Result<(), OwnedDecodePreparationError> {
        match self {
            Self::Owned(s) => s.prepare_destinations(),
            Self::Shared(s) => s.destinations.prepare_destinations(),
        }
    }
    pub(in super::super) fn step(
        &mut self,
        id: u32,
    ) -> Result<Option<&str>, OwnedDecodeStorageError> {
        match self {
            Self::Owned(s) => s.step(id),
            Self::Shared(s) => s.destinations.step(s.source.source(), id),
        }
    }
    pub(in super::super) fn finish(&mut self) -> Result<(), OwnedDecodeStorageError> {
        match self {
            Self::Owned(s) => s.finish(),
            Self::Shared(s) => s.destinations.finish(s.source.source()),
        }
    }
}

#[derive(Debug)]
pub(in super::super) struct SharedStopStorage {
    // Scratch/progress retire before the exact original source lease.
    destinations: StopDestinations,
    source: OriginalStopSource,
}
impl SharedStopStorage {
    fn control_bytes() -> Option<usize> {
        StopDestinations::control_bytes()?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<OriginalStopSource>())?
            .checked_add(size_of::<Result<(), WorkingMemoryError>>())
    }
}
#[derive(Debug)]
pub(in super::super) enum StopStorage {
    Owned(OwnedStopStorage),
    Shared(SharedStopStorage),
}
impl StopStorage {
    pub(in super::super) fn validate_pool(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Owned(_) => Ok(()),
            Self::Shared(s) => s.source.validate_pool(pool),
        }
    }
    pub(in super::super) fn prepare_destination(&mut self) -> Result<(), StopPreparationError> {
        match self {
            Self::Owned(s) => s.prepare_destination(),
            Self::Shared(s) => s.destinations.prepare_destination(),
        }
    }
    pub(in super::super) fn step<'a>(
        &'a mut self,
        text: &'a str,
    ) -> Result<StopOutput<'a>, StopStorageError> {
        match self {
            Self::Owned(s) => s.step(text),
            Self::Shared(s) => s.destinations.step(s.source.source(), text),
        }
    }
    pub(in super::super) fn finish(&mut self) -> Result<&str, StopStorageError> {
        match self {
            Self::Owned(s) => s.finish(),
            Self::Shared(s) => s.destinations.finish(s.source.source()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(in super::super) enum DecoderStateCopyError {
    #[error(transparent)]
    Decoder(#[from] eredu_text::decoder_storage::DecodeDestinationCopyError),
    #[error(transparent)]
    Stop(#[from] eredu_text::stop_storage::StopDestinationCopyError),
}
impl DecoderStorage {
    pub(in super::super) fn copy_required_bytes(&self) -> Option<usize> {
        match self {
            // Unique compatibility programs have no retained original source
            // producer for cloning their nested program. Never adopt its Box.
            Self::Owned(_) => None,
            Self::Shared(s) => s
                .destinations
                .prepare_copy()
                .required_bytes()?
                .checked_add(SharedStorage::control_bytes()?),
        }
    }
    pub(in super::super) fn copy_state(&self) -> Result<Self, DecoderStateCopyError> {
        let Self::Shared(s) = self else {
            unreachable!("qualified original shared decoder")
        };
        Ok(Self::Shared(SharedStorage {
            destinations: s.destinations.prepare_copy().copy()?,
            source: s.source.clone(),
        }))
    }
}
impl StopStorage {
    pub(in super::super) fn copy_required_bytes(&self) -> Option<usize> {
        match self {
            Self::Owned(_) => None,
            Self::Shared(s) => s
                .destinations
                .prepare_copy()
                .required_bytes()?
                .checked_add(SharedStopStorage::control_bytes()?),
        }
    }
    pub(in super::super) fn copy_state(&self) -> Result<Self, DecoderStateCopyError> {
        let Self::Shared(s) = self else {
            unreachable!("qualified original shared stops")
        };
        Ok(Self::Shared(SharedStopStorage {
            destinations: s.destinations.prepare_copy().copy()?,
            source: s.source.clone(),
        }))
    }
}
