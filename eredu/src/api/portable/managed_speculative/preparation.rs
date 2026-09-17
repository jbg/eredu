//! The same original semantic/controller/input producer for one or many lanes.
use super::*;
use crate::api::portable::speculative_semantic::PreparedOriginalSpeculativeHost;
use eredu_core::{SpeculativeBuffer, SpeculativeGenerationLane, TextGenerationConfig};
use std::num::NonZeroUsize;

pub(super) struct PreparedLane<'a> {
    host: PreparedOriginalSpeculativeHost<'a>,
    text: ManagedPlainTextRequest<'a>,
    generation: TextGenerationConfig,
    constraint: PreparedChatSpeculativeConstraint,
    cancellation: GenerationCancellationToken,
    // The aggregate owns all its framework fields until they retire on error.
    funding: WorkspaceMetadataFunding,
}
impl<'a> PreparedLane<'a> {
    pub(super) fn funding(&self) -> &WorkspaceMetadataFunding { &self.funding }
    pub(super) fn finish<B: OriginalTokenizerBackend + SpeculativeGenerationBackend>(self,
        model: &LoadedModel<B>,
    ) -> Result<SpeculativeGenerationLane<'a, B, PreparedChatSpeculativeConstraint>, Cause> {
        reserve(&self.funding, sum(&[
            size_of::<Self>(), size_of::<&LoadedModel<B>>(),
            size_of::<Result<B::Prompt, BackendFailure>>(),
            size_of::<Result<SpeculativeGenerationLane<'a, B, PreparedChatSpeculativeConstraint>, Cause>>(),
        ]))?;
        let encoded = B::encode_original_text_ids(&model.runtime,
            self.host.preparation().tokenizer(), self.text.input, self.text.add_special_tokens)?;
        let prompt = B::prepare_original_speculative_prompt(&model.runtime,
            self.host.preparation(), &encoded, self.text.settings.inference.prefill_chunk_positions)?;
        // The selected I/B worker copied actual E IDs before this source retires.
        drop(encoded);
        Ok(self.host.into_lane(prompt, self.generation, self.constraint, self.cancellation)?)
    }
}

pub(super) fn buffer<T>(count: usize, funding: &WorkspaceMetadataFunding) -> Result<SpeculativeBuffer<T>, Cause> {
    reserve(funding, SpeculativeBuffer::<T>::retained_control_bytes(count)
        .and_then(|n| n.checked_add(HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>()?))
        .and_then(|n| n.checked_add(size_of::<(&WorkspaceMetadataFunding, usize, Result<SpeculativeBuffer<T>, Cause>)>())))?;
    Ok(SpeculativeBuffer::try_new_retained(count, HostPreparationAuthority::retain(funding.clone()))?)
}

impl<B: OriginalTokenizerBackend + SpeculativeGenerationBackend> LoadedModel<B> {
    pub(super) fn prepare_managed_speculative_host<'a, F: FnMut(SemanticEvent) + 'a>(&self,
        source: &ManagedPlainTextSource, text: ManagedPlainTextRequest<'a>, maximum_draft: NonZeroUsize,
        cancellation: GenerationCancellationToken, on_event: F,
        retained: &mut Option<WorkspaceMetadataFunding>,
    ) -> Result<PreparedLane<'a>, Cause> {
        let capacity = text.settings.inference.managed_memory_capacity_bytes
            .ok_or(TokenInputRejection::Unsupported)?;
        let (generation, maximum) = self.resolve_text_generation_settings(text.settings)?;
        let host = self.prepare_original_speculative_plain_host(source, capacity, maximum.get(),
            maximum_draft.get(), generation.sampling().temperature, &self.eos_token_ids,
            OriginalSpeculativeStops::Refs(text.stop_sequences), text.skip_special_tokens, on_event)?;
        let funding = host.preparation().metadata_funding();
        // Retain before a reserve can fail. A later lane's own failed source
        // also remains inside its typed cause, independently of batch custody.
        if retained.is_none() { *retained = Some(funding.clone()); }
        reserve(funding, sum(&[
            size_of::<PreparedLane<'a>>(), size_of::<Result<PreparedLane<'a>, Cause>>(),
            size_of::<(ManagedPlainTextRequest<'a>, NonZeroUsize, GenerationCancellationToken, F)>(),
            size_of::<Cause>(), size_of::<ManagedPlainTextSpeculativeError>(),
            size_of::<Option<WorkspaceMetadataFunding>>(), size_of::<Option<SpeculativeControlError>>(),
            size_of::<PreparedChatSpeculativeConstraint>(),
        ]).and_then(|n| n.checked_add(BackendFailure::source_retention_peak_bytes::<B::Error>()?)))?;
        reserve(funding, ConstraintController::text_metadata_bytes().and_then(|n|
            n.checked_add(HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>()?)))?;
        let controller = ConstraintController::text_prepared(self.token_validity.clone(),
            HostPreparationAuthority::retain(funding.clone()));
        let funding = funding.clone();
        Ok(PreparedLane { host, text, generation, constraint: PreparedChatSpeculativeConstraint::new(controller),
            cancellation, funding })
    }
}
