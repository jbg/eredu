//! Source-bound decoder, channel parser and event ownership for generation.
use super::decoder_transition::{stop_error, transition_error};
use super::{InferenceExecutionIdentity, MemoryLedger, OriginalStopSource, OriginalTokenizer};
use super::{OriginalSemanticChannelParser, OriginalSemanticChannelSource};
use eredu_core::generation::SemanticEvent;
use eredu_core::{
    FinishReason, GenerationPlainText, GenerationPlainTextEvent, GenerationPlainTextEvents,
    GenerationPlainTextProjection, HostMetadataFundingError, HostPreparationAuthority,
    SemanticState, SemanticStateOwner, SemanticText, SpeculativeBuffer, SpeculativeOutputError,
};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_text::decoder_storage::{DecodeDestinations, DecodeStreamLayout};
use eredu_text::stop_storage::StopDestinations;
use std::mem::size_of;

/// Closed source/selected-execution header for one cumulative host account.
/// Private source and account identity survive clones; no replacement, mutable
/// identity or raw identity alias escapes. Native permissions remain separate.
#[derive(Debug, Clone)]
pub struct PreparedSemanticSource {
    source: OriginalTokenizer,
    execution: InferenceExecutionIdentity,
    limits: eredu_core::MemoryLimits,
    identity: eredu_core::SpeculativeRequestIdentity,
    // Source and identity aliases retire before this final account token.
    funding: HostMetadataFunding,
}
/// Exact initial host producer failure, with allocation causes retaining their payer.
#[derive(Debug, thiserror::Error)]
pub enum OriginalSpeculativeHostError {
    /// Fixed source, funding or semantic preparation refusal.
    #[error("{0}")]
    Preparation(#[from] SpeculativeOutputError),
    /// Exact owned EOS allocation or validation failure.
    #[error("{0}")]
    Configuration(#[from] eredu_core::SpeculativeConfigurationError),
}
impl PreparedSemanticSource {
    /// Starts the existing cumulative account for this source and selected execution.
    pub fn new(
        source: &OriginalTokenizer,
        execution: &InferenceExecutionIdentity,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<Self, SpeculativeOutputError> {
        let funding = source
            .pool()
            .prepare_workspace_metadata(execution, capacity.clone())?;
        let parts = [
            size_of::<Self>(),
            size_of::<InferenceExecutionIdentity>(),
            size_of::<Result<Self, SpeculativeOutputError>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .and_then(|n| {
                n.checked_add(eredu_core::SpeculativeRequestIdentity::retained_control_bytes()?)
            })
            .and_then(|n| n.checked_add(controls()?));
        let host = pay(&funding, bytes)?;
        let identity = eredu_core::SpeculativeRequestIdentity::with_authority(host);
        Ok(Self {
            source: source.clone(),
            execution: execution.clone(),
            limits: capacity,
            identity,
            funding,
        })
    }
    /// Authenticates the exact retained selected execution and original C pool.
    /// This is allocation-free and grants no native execution permission.
    pub fn validate(
        &self,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), super::WorkingMemoryError> {
        self.source.validate_pool(pool)?;
        if !std::sync::Arc::ptr_eq(&self.execution.0, &execution.0) {
            return Err(super::WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// Equality of the actual preparation, including its original account.
    pub fn same_preparation(&self, other: &Self) -> bool {
        self.identity.same(&other.identity)
    }
    /// Checks that the immutable EOS owner was constructed by this exact header.
    pub fn validate_configuration(
        &self,
        value: &eredu_core::SpeculativeConfiguration,
    ) -> Result<(), super::WorkingMemoryError> {
        if value.matches_preparation(&self.identity) {
            Ok(())
        } else {
            Err(super::WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Checks the exact original framework callback constructor, not arbitrary
    /// allocations/work inside the caller-owned closure.
    pub fn validate_callback(
        &self,
        value: &eredu_core::SpeculativeEventCallback<'_>,
    ) -> Result<(), super::WorkingMemoryError> {
        if value.matches_preparation(&self.identity) {
            Ok(())
        } else {
            Err(super::WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Pays then copies the actual borrowed EOS rows, preserving private provenance.
    pub fn prepare_configuration(
        &self,
        maximum: usize,
        maximum_draft: usize,
        temperature: f32,
        eos: &[u32],
    ) -> Result<eredu_core::SpeculativeConfiguration, OriginalSpeculativeHostError> {
        let bytes = eredu_core::SpeculativeConfiguration::retained_control_bytes(eos.len())
            .and_then(|n| n.checked_add(size_of::<OriginalSpeculativeHostError>()))
            .and_then(|n| {
                n.checked_add(size_of::<
                    Result<eredu_core::SpeculativeConfiguration, OriginalSpeculativeHostError>,
                >())
            });
        let host = pay(&self.funding, bytes)?;
        Ok(eredu_core::SpeculativeConfiguration::try_copy_retained(
            maximum,
            maximum_draft,
            temperature,
            eos,
            host,
            self.identity.clone(),
        )?)
    }
    /// Pays the concrete new framework Box before moving caller-owned F into it.
    pub fn prepare_callback<'a, F: FnMut(SemanticEvent) + 'a>(
        &self,
        callback: F,
    ) -> Result<eredu_core::SpeculativeEventCallback<'a>, SpeculativeOutputError> {
        let bytes =
            eredu_core::SpeculativeEventCallback::retained_control_bytes::<F>().and_then(|n| {
                n.checked_add(size_of::<
                    Result<eredu_core::SpeculativeEventCallback<'a>, SpeculativeOutputError>,
                >())
            });
        let host = pay(&self.funding, bytes)?;
        Ok(eredu_core::SpeculativeEventCallback::new_retained(
            callback,
            host,
            self.identity.clone(),
        ))
    }
    /// Actual ceiling accepted when this preparation account was created.
    pub fn limits(&self) -> &eredu_core::MemoryLimits {
        &self.limits
    }
    /// Immutable backend-established domains for this preparation.
    pub fn topology(&self) -> &eredu_core::MemoryTopology {
        self.source.pool().topology()
    }
    /// Borrowed original tokenizer used for all prepared semantic transitions.
    pub fn tokenizer(&self) -> &OriginalTokenizer {
        &self.source
    }
    /// Same original account for concrete metadata producers before birth.
    pub fn metadata_funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    /// Constructs its plain decoder/stop state while retaining this exact header.
    pub fn prepare(
        &self,
        stops: &OriginalStopSource,
        maximum_tokens: usize,
        event_window: std::num::NonZeroUsize,
        skip_special: bool,
    ) -> Result<SemanticStateOwner, SpeculativeOutputError> {
        PreparedSemanticState::construct(
            self.clone(),
            stops,
            maximum_tokens,
            event_window,
            skip_special,
            None,
        )
    }
    /// Constructs the same original decoder/stop owner with its closed literal
    /// channel projection. Actual tokenizer/controller identities remain attached.
    pub fn prepare_channels(
        &self,
        stops: &OriginalStopSource,
        channels: &OriginalSemanticChannelSource,
        maximum_tokens: usize,
        event_window: std::num::NonZeroUsize,
        skip_special: bool,
    ) -> Result<SemanticStateOwner, SpeculativeOutputError> {
        PreparedSemanticState::construct(
            self.clone(),
            stops,
            maximum_tokens,
            event_window,
            skip_special,
            Some(channels),
        )
    }
}

/// Source-bound tokenizer/stop state with a closed plain or channel projection.
/// The selected backend must authenticate the retained source and pool before
/// using it in a request. This owns no model or native execution permission.
#[derive(Debug)]
pub struct PreparedSemanticState {
    decoder: DecodeDestinations,
    stops: StopDestinations,
    projection: GenerationPlainTextProjection,
    events: SpeculativeBuffer<SemanticEvent>,
    event_capacity: usize,
    channels: Option<OriginalSemanticChannelParser>,
    source: OriginalTokenizer,
    stop_source: OriginalStopSource,
    funding: HostMetadataFunding,
    preparation: PreparedSemanticSource,
    // Copied destinations can have a separate snapshot payer. It retires last.
    host: HostPreparationAuthority,
}
fn overflow() -> SpeculativeOutputError {
    HostMetadataFundingError::Overflow.into()
}
fn unbox<T>(value: Box<T>) -> T {
    *value
}
fn storage(message: &'static str) -> SpeculativeOutputError {
    SpeculativeOutputError::Storage(message)
}
fn controls() -> Option<usize> {
    let parts = [
        size_of::<PreparedSemanticState>(),
        size_of::<SemanticStateOwner>(),
        size_of::<eredu_core::SpeculativeSemanticConstraint>(),
        size_of::<Result<eredu_core::SpeculativeSemanticConstraint, SpeculativeOutputError>>(),
        size_of::<Box<PreparedSemanticState>>(),
        size_of::<Box<dyn SemanticState>>(),
        size_of::<Option<PreparedSemanticState>>(),
        size_of::<SpeculativeOutputError>(),
        size_of::<Result<SemanticStateOwner, SpeculativeOutputError>>(),
        size_of::<HostMetadataFunding>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<OriginalTokenizer>(),
        size_of::<OriginalStopSource>(),
        size_of::<(usize, usize, usize)>(),
        OriginalSemanticChannelSource::control_bytes()?,
        size_of::<Option<OriginalSemanticChannelParser>>(),
        size_of::<Result<OriginalSemanticChannelParser, super::OriginalSemanticChannelParserError>>(
        ),
        size_of::<Result<Option<OriginalSemanticChannelParser>, SpeculativeOutputError>>(),
        size_of::<Option<(&str, bool)>>(),
        size_of::<Result<(), super::WorkingMemoryError>>(),
        size_of::<Option<eredu_core::speculative::ForbiddenControllerSource<'static>>>(),
        size_of::<Option<&dyn std::any::Any>>(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}
fn pay(
    funding: &HostMetadataFunding,
    bytes: Option<usize>,
) -> Result<HostPreparationAuthority, SpeculativeOutputError> {
    let bytes = bytes
        .and_then(|n| {
            n.checked_add(HostPreparationAuthority::retention_bytes::<
                HostMetadataFunding,
            >()?)
        })
        .ok_or_else(overflow)?;
    funding.reserve_metadata(bytes)?;
    Ok(HostPreparationAuthority::retain(funding.clone()))
}
impl PreparedSemanticState {
    fn construct(
        prepared: PreparedSemanticSource,
        stops: &OriginalStopSource,
        maximum_tokens: usize,
        event_window: std::num::NonZeroUsize,
        skip_special: bool,
        channel_source: Option<&OriginalSemanticChannelSource>,
    ) -> Result<SemanticStateOwner, SpeculativeOutputError> {
        let funding = prepared.metadata_funding().clone();
        let source = prepared.tokenizer().clone();
        stops
            .validate_pool(source.pool())
            .map_err(|_| storage("semantic sources belong to different pools"))?;
        let fixed = DecodeDestinations::control_bytes()
            .and_then(|n| n.checked_add(StopDestinations::control_bytes()?))
            .and_then(|n| n.checked_add(size_of::<DecodeStreamLayout<'static>>()))
            .ok_or_else(overflow)?;
        funding.reserve_metadata(fixed)?;
        let layout =
            DecodeStreamLayout::for_source(source.decode_source(), maximum_tokens, skip_special)
                .map_err(|_| overflow())?;
        let channels = channel_source
            .map(|channels| {
                channels
                    .validate(source.pool(), &source)
                    .map_err(|_| storage("semantic channel tokenizer/source changed"))?;
                let input_bytes = maximum_tokens
                    .checked_mul(channels.maximum_structural_bytes())
                    .and_then(|n| n.checked_add(layout.text_capacity()))
                    .ok_or_else(overflow)?;
                OriginalSemanticChannelParser::prepare(channels, input_bytes, &funding)
                    .map_err(super::OriginalSemanticChannelParserError::into_output)
            })
            .transpose()?;
        let mut decoder =
            DecodeDestinations::new(source.decode_source(), maximum_tokens, skip_special)
                .map_err(|_| overflow())?;
        let mut stop_destinations = StopDestinations::new(stops.source(), layout.text_capacity())
            .map_err(|_| overflow())?;
        // The caller declares its maximum pending token-transition window. A finish may flush text
        // and emit Finished. Stops can instead emit Finished on the last push.
        let event_capacity = if channels.is_some() {
            0
        } else {
            maximum_tokens
                .min(event_window.get())
                .checked_add(2)
                .ok_or_else(overflow)?
        };
        let bytes = decoder
            .destination_bytes()
            .checked_add(stop_destinations.destination_bytes())
            .and_then(|n| {
                n.checked_add(SpeculativeBuffer::<SemanticEvent>::retained_control_bytes(
                    event_capacity,
                )?)
            })
            .and_then(|n| n.checked_add(controls()?));
        let host = pay(&funding, bytes)?;
        // Typed partial failures die under funding; returned diagnostics are fixed.
        decoder.prepare_destinations().map_err(|cause| {
            drop(cause);
            storage("decoder destination allocation failed")
        })?;
        stop_destinations.prepare_destination().map_err(|cause| {
            drop(cause);
            storage("stop destination allocation failed")
        })?;
        let events =
            SpeculativeBuffer::try_new_retained(event_capacity, host.clone()).map_err(|cause| {
                drop(cause);
                storage("semantic event allocation failed")
            })?;
        Ok(SemanticStateOwner::from_prepared(Self {
            decoder,
            stops: stop_destinations,
            projection: GenerationPlainTextProjection::default(),
            events,
            event_capacity,
            channels,
            source,
            stop_source: stops.clone(),
            funding,
            preparation: prepared,
            host,
        }))
    }
    /// Exact successful decoder-call ceiling accepted before destination construction.
    pub fn token_capacity(&self) -> usize {
        self.decoder.token_capacity()
    }
    /// Authenticates the state and its channel controller inputs without callbacks.
    pub(in crate::working_memory) fn validate_controller_source(
        &self,
        source: eredu_core::PreparedControllerSource<'_>,
    ) -> Result<(), super::WorkingMemoryError> {
        self.validate_pool(self.source.pool())?;
        if let Some(channels) = &self.channels {
            channels
                .source()
                .validate_controller_source(source, self.source.pool())?;
        }
        Ok(())
    }
    /// Exact header retained through construction and every semantic fork.
    pub fn preparation(&self) -> &PreparedSemanticSource {
        &self.preparation
    }
    /// Authenticates actual source residence without allocating or minting a hold.
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), super::WorkingMemoryError> {
        self.source.validate_pool(pool)?;
        self.stop_source.validate_pool(pool)?;
        if let Some(channels) = &self.channels {
            channels.source().validate(pool, &self.source)?;
        }
        Ok(())
    }
    /// Exact original controller inputs required by the selected channel program.
    /// Plain text retains its existing independent controller policy.
    pub fn validate_controller<C: eredu_core::SpeculativeTokenFilterController>(
        &self,
        controller: &C,
    ) -> Result<(), super::WorkingMemoryError> {
        if let Some(channels) = &self.channels {
            channels
                .source()
                .validate_controller(controller, self.source.pool())?;
        }
        Ok(())
    }
    /// Same immutable tokenizer used by every transition, for selected-source validation.
    pub fn tokenizer(&self) -> &OriginalTokenizer {
        &self.source
    }
    /// Actual cumulative account for enclosing metadata; never a native grant.
    pub fn metadata_funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    /// Exact known host destination bound for this parser's snapshot producer.
    /// This query admits no work and allocates no parser or source state.
    pub fn copy_bytes(&self) -> Option<usize> {
        self.decoder
            .prepare_copy()
            .required_bytes()?
            .checked_add(self.stops.prepare_copy().required_bytes()?)?
            .checked_add(SpeculativeBuffer::<SemanticEvent>::retained_control_bytes(
                self.event_capacity,
            )?)?
            .checked_add(controls()?)?
            .checked_add(match &self.channels {
                Some(channels) => channels.copy_bytes()?,
                None => 0,
            })
    }
    pub(in crate::working_memory) fn copy(
        &self,
        host: HostPreparationAuthority,
        keep_events: bool,
    ) -> Result<SemanticStateOwner, SpeculativeOutputError> {
        let decoder = self.decoder.prepare_copy().copy().map_err(|cause| {
            drop(cause);
            storage("semantic decoder copy failed")
        })?;
        let stops = self.stops.prepare_copy().copy().map_err(|cause| {
            drop(cause);
            storage("semantic stop copy failed")
        })?;
        let mut events = SpeculativeBuffer::try_new_retained(self.event_capacity, host.clone())
            .map_err(|cause| {
                drop(cause);
                storage("semantic event copy failed")
            })?;
        if keep_events {
            events
                .try_extend(self.events.iter().cloned())
                .map_err(|_| storage("semantic event copy capacity differs"))?;
        }
        let channels = self
            .channels
            .as_ref()
            .map(|channels| {
                channels
                    .copy_prepaid(host.clone(), keep_events)
                    .map_err(super::OriginalSemanticChannelParserError::into_output)
            })
            .transpose()?;
        Ok(SemanticStateOwner::from_prepared(Self {
            decoder,
            stops,
            projection: self.projection,
            events,
            event_capacity: self.event_capacity,
            channels,
            source: self.source.clone(),
            stop_source: self.stop_source.clone(),
            funding: self.funding.clone(),
            preparation: self.preparation.clone(),
            host,
        }))
    }
    fn prepare_transition(&mut self, maximum_events: usize) -> Result<(), SpeculativeOutputError> {
        let parts = [
            size_of::<Result<GenerationPlainText<'static>, SpeculativeOutputError>>(),
            size_of::<GenerationPlainTextEvents<'static>>(),
            size_of::<GenerationPlainTextEvent<'static>>(),
            size_of::<SemanticEvent>(),
            size_of::<Option<SemanticEvent>>(),
            size_of::<Result<(), SpeculativeOutputError>>(),
            size_of::<Result<bool, SpeculativeOutputError>>(),
        ];
        self.funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
                .ok_or_else(overflow)?,
        )?;
        if self.channels.is_some() {
            return Ok(());
        }
        if self.events.capacity() == 0 {
            let host = pay(
                &self.funding,
                SpeculativeBuffer::<SemanticEvent>::retained_control_bytes(self.event_capacity),
            )?;
            self.events = SpeculativeBuffer::try_new_retained(self.event_capacity, host).map_err(
                |cause| {
                    drop(cause);
                    storage("semantic event allocation failed")
                },
            )?;
        }
        if self
            .events
            .len()
            .checked_add(maximum_events)
            .is_none_or(|n| n > self.event_capacity)
        {
            return Err(storage("semantic event destination exhausted"));
        }
        Ok(())
    }
    fn append(
        events: &mut SpeculativeBuffer<SemanticEvent>,
        rows: GenerationPlainTextEvents<'_>,
        funding: &HostMetadataFunding,
    ) -> Result<(), SpeculativeOutputError> {
        for row in rows {
            let event = match row {
                GenerationPlainTextEvent::TextDelta(text) => {
                    let host = pay(funding, SemanticText::retained_control_bytes(text.len()))?;
                    SemanticEvent::TextDelta(SemanticText::try_copy_retained(text, host).map_err(
                        |cause| {
                            drop(cause);
                            storage("semantic text allocation failed")
                        },
                    )?)
                }
                GenerationPlainTextEvent::Finished { reason } => SemanticEvent::Finished { reason },
            };
            events
                .try_push(event)
                .map_err(|_| storage("semantic event destination exhausted"))?;
        }
        Ok(())
    }
}
impl SemanticState for PreparedSemanticState {
    fn prepared_source(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn control_snapshot_bytes(&self) -> Option<u64> {
        u64::try_from(self.copy_bytes()?).ok()
    }
    fn control_snapshot_metadata_bytes(&self) -> Option<usize> {
        self.copy_bytes()
    }
    fn fork_owned(&self) -> Result<SemanticStateOwner, SpeculativeOutputError> {
        let host = pay(&self.funding, self.copy_bytes())?;
        self.copy(host, false)
    }
    fn fork_prepared(
        &self,
        host: HostPreparationAuthority,
    ) -> Result<SemanticStateOwner, SpeculativeOutputError> {
        if host.is_unmanaged() {
            return self.fork_owned();
        }
        self.copy(host, true)
    }
    fn retire(self: Box<Self>) {
        drop(unbox(self));
    }
    fn push_token(&mut self, id: u32) -> Result<bool, SpeculativeOutputError> {
        self.prepare_transition(2)?;
        let text = self
            .decoder
            .step(self.source.decode_source(), id)
            .map_err(transition_error)?
            .unwrap_or("");
        if let Some(channels) = &mut self.channels {
            // The existing decoder still consumes every actual ID. Like ordinary
            // RawTokenDecoder, structural spellings replace its visible suffix.
            let source = channels.source().clone();
            let structural = source.structural(id);
            let output = self
                .stops
                .step(
                    self.stop_source.source(),
                    if structural.is_some() { "" } else { text },
                )
                .map_err(stop_error)?;
            let matched = output.matched.is_some();
            channels
                .push(output.visible)
                .map_err(super::OriginalSemanticChannelParserError::into_output)?;
            if matched {
                channels
                    .finish_reason(FinishReason::StopSequence)
                    .map_err(super::OriginalSemanticChannelParserError::into_output)?;
                return Ok(true);
            }
            if let Some((spelling, stop)) = structural {
                let visible = self
                    .stops
                    .finish(self.stop_source.source())
                    .map_err(stop_error)?;
                channels
                    .push(visible)
                    .map_err(super::OriginalSemanticChannelParserError::into_output)?;
                if stop {
                    channels
                        .finish_reason(FinishReason::StopSequence)
                        .map_err(super::OriginalSemanticChannelParserError::into_output)?;
                } else {
                    channels
                        .push(spelling)
                        .map_err(super::OriginalSemanticChannelParserError::into_output)?;
                }
            }
            return Ok(channels.is_finished());
        }
        let output = self
            .stops
            .step(self.stop_source.source(), text)
            .map_err(stop_error)?;
        let matched = output.matched.is_some();
        let rows = self.projection.push(GenerationPlainText {
            visible: output.visible,
            stop_matched: matched,
        });
        Self::append(&mut self.events, rows, &self.funding)?;
        Ok(matched)
    }
    fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
        self.prepare_transition(if self.projection.is_finished() { 0 } else { 2 })?;
        self.decoder
            .finish(self.source.decode_source())
            .map_err(transition_error)?;
        if let Some(channels) = &mut self.channels {
            if !channels.is_finished() {
                let text = self
                    .stops
                    .finish(self.stop_source.source())
                    .map_err(stop_error)?;
                channels
                    .push(text)
                    .map_err(super::OriginalSemanticChannelParserError::into_output)?;
            }
            // Finished is idempotent; a failed parser must still return failure.
            return channels
                .finish_reason(reason)
                .map_err(super::OriginalSemanticChannelParserError::into_output);
        }
        let text = self
            .stops
            .finish(self.stop_source.source())
            .map_err(stop_error)?;
        Self::append(
            &mut self.events,
            self.projection.finish(text, reason),
            &self.funding,
        )
    }
    fn cancel(&mut self) -> Result<(), SpeculativeOutputError> {
        self.prepare_transition(usize::from(!self.projection.is_finished()))?;
        if let Some(channels) = &mut self.channels {
            return channels
                .cancel_reason()
                .map_err(super::OriginalSemanticChannelParserError::into_output);
        }
        Self::append(&mut self.events, self.projection.cancel(), &self.funding)
    }
    fn take_events(&mut self) -> SpeculativeBuffer<SemanticEvent> {
        match &mut self.channels {
            Some(channels) => channels.drain_owner(),
            None => std::mem::take(&mut self.events),
        }
    }
    fn publish_events(&mut self, emit: &mut dyn FnMut(SemanticEvent)) {
        if let Some(channels) = &mut self.channels {
            channels.publish(emit);
            return;
        }
        for event in self.events.drain() {
            emit(event);
        }
    }
}
#[cfg(test)]
mod tests;
