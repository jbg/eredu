//! Original semantic producers before the shared speculative lane construction.
mod chat;
use super::{LoadedModel, ManagedPlainTextSource};
use eredu_core::{
    GenerationCancellationToken, HostPreparationAuthority, SpeculativeBuffer,
    SpeculativeBufferAllocationError, SpeculativeConfiguration, SpeculativeDraft,
    SpeculativeEventCallback, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeOutputError, SpeculativeSemanticOwner, SpeculativeTokenFilterController,
    TextGenerationBackend, TextGenerationConfig, generation::SemanticEvent,
};
use eredu_nn::workspace::WorkspaceMetadataFunding;
use eredu_runtime::working_memory::{
    OriginalSpeculativeHostError, OriginalSpeculativeSemanticPreparation, OriginalTextSourceError,
    OriginalTokenizerBackend,
};
use eredu_text::stop_storage::StopCompilePlan;
use std::mem::size_of;

/// Borrowed stop declarations use the same source compiler without collecting
/// reference rows or cloning request strings before their original admission.
#[derive(Clone, Copy)]
pub(super) enum OriginalSpeculativeStops<'a> {
    Strings(&'a [String]),
    Refs(&'a [&'a str]),
}
impl<'a> OriginalSpeculativeStops<'a> {
    fn plan(self) -> Result<StopCompilePlan<'a>, eredu_text::stop_storage::StopSourceError> {
        match self {
            Self::Strings(stops) => StopCompilePlan::prepare(stops),
            Self::Refs(stops) => StopCompilePlan::prepare_refs(stops),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Chat(#[from] chat::ChatCause),
    #[error(transparent)]
    Output(#[from] SpeculativeOutputError),
    #[error(transparent)]
    Source(#[from] OriginalTextSourceError),
    #[error("{0}")]
    Host(#[from] OriginalSpeculativeHostError),
    #[error("{0}")]
    Buffer(#[from] SpeculativeBufferAllocationError),
}
/// The exact source error retires before its preparation account.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct PreparedPlainSpeculativeSemanticError {
    #[source]
    cause: Cause,
    funding: Option<WorkspaceMetadataFunding>,
}
impl<B: OriginalTokenizerBackend> LoadedModel<B> {
    /// Builds the source-bound semantic owner used by managed speculative lanes.
    fn prepare_original_speculative_plain_semantic(
        &self,
        source: &ManagedPlainTextSource,
        capacity: u64,
        maximum: usize,
        maximum_draft: usize,
        stops: OriginalSpeculativeStops<'_>,
        skip_special: bool,
    ) -> Result<
        (
            SpeculativeSemanticOwner,
            OriginalSpeculativeSemanticPreparation,
        ),
        PreparedPlainSpeculativeSemanticError,
    > {
        if !source.original().matches_configuration(&self.tokenizer) {
            return Err(PreparedPlainSpeculativeSemanticError {
                cause: SpeculativeOutputError::Storage(
                    "semantic tokenizer does not match the loaded source",
                )
                .into(),
                funding: None,
            });
        }
        let prepared =
            B::prepare_original_speculative_semantic(&self.runtime, source.original(), capacity)
                .map_err(|cause| PreparedPlainSpeculativeSemanticError {
                    cause: cause.into(),
                    funding: None,
                })?;
        let funding = prepared.metadata_funding().clone();
        let result = (|| {
            let parts = [
                size_of::<Cause>(),
                size_of::<PreparedPlainSpeculativeSemanticError>(),
                size_of::<Option<WorkspaceMetadataFunding>>(),
                size_of::<StopCompilePlan<'static>>(),
                size_of::<OriginalSpeculativeStops<'static>>(),
                size_of::<
                    Result<StopCompilePlan<'static>, eredu_text::stop_storage::StopSourceError>,
                >(),
                size_of::<(
                    SpeculativeSemanticOwner,
                    OriginalSpeculativeSemanticPreparation,
                )>(),
                size_of::<
                    Result<
                        (
                            SpeculativeSemanticOwner,
                            OriginalSpeculativeSemanticPreparation,
                        ),
                        PreparedPlainSpeculativeSemanticError,
                    >,
                >(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
                .ok_or(SpeculativeOutputError::HostFunding(
                    eredu_core::HostMetadataFundingError::Overflow,
                ))?;
            funding
                .reserve_metadata(bytes)
                .map_err(SpeculativeOutputError::from)?;
            let plan = stops.plan().map_err(|_| {
                SpeculativeOutputError::HostFunding(eredu_core::HostMetadataFundingError::Overflow)
            })?;
            let stops = B::compile_original_text_stop_source(&self.runtime, plan)?;
            let semantic = prepared.prepare(&stops, maximum, maximum_draft, skip_special)?;
            Ok((semantic, prepared))
        })();
        result.map_err(|cause| PreparedPlainSpeculativeSemanticError {
            cause,
            funding: Some(funding),
        })
    }
}

/// Prepared facade metadata consumed by the same lane and batch contracts.
/// Every owning exit remains closed; the source header authenticates the H used
/// before these births. The callback's own application work remains caller-owned.
#[derive(Debug)]
pub(crate) struct PreparedOriginalSpeculativeHost<'a> {
    semantic: SpeculativeSemanticOwner,
    configuration: SpeculativeConfiguration,
    callback: SpeculativeEventCallback<'a>,
    preparation: OriginalSpeculativeSemanticPreparation,
}
impl<'a> PreparedOriginalSpeculativeHost<'a> {
    pub(crate) fn preparation(&self) -> &OriginalSpeculativeSemanticPreparation {
        &self.preparation
    }
    /// Creates the exact lane after paying its concrete generic transport.
    pub(crate) fn into_lane<B: TextGenerationBackend, C: SpeculativeTokenFilterController>(
        self,
        prompt: B::Prompt,
        generation: TextGenerationConfig,
        constraint: C,
        cancellation: GenerationCancellationToken,
    ) -> Result<SpeculativeGenerationLane<'a, B, C>, PreparedPlainSpeculativeSemanticError> {
        let funding = self.preparation.metadata_funding().clone();
        let result = (|| {
            let parts = [
                size_of::<SpeculativeGenerationLane<'a, B, C>>(),
                size_of::<
                    Result<
                        SpeculativeGenerationLane<'a, B, C>,
                        PreparedPlainSpeculativeSemanticError,
                    >,
                >(),
                size_of::<(
                    B::Prompt,
                    TextGenerationConfig,
                    C,
                    GenerationCancellationToken,
                )>(),
            ];
            reserve(&funding, sum(&parts))?;
            Ok(SpeculativeGenerationLane::new(
                prompt,
                generation,
                self.configuration,
                constraint,
                self.semantic,
                cancellation,
                self.callback,
            ))
        })();
        result.map_err(|cause| PreparedPlainSpeculativeSemanticError {
            cause,
            funding: Some(funding),
        })
    }
    /// Single-lane composition constructs its buffer under this same H, before
    /// the public driver can move it into backend preparation. No ordinary Vec
    /// is relabelled; batch callers can use into_lane with their own paid buffer.
    pub(crate) fn into_request<B: TextGenerationBackend, D, C: SpeculativeTokenFilterController>(
        self,
        drafting: SpeculativeDraft<'a, D>,
        prompt: B::Prompt,
        generation: TextGenerationConfig,
        constraint: C,
        cancellation: GenerationCancellationToken,
        fingerprint: [u8; 32],
    ) -> Result<SpeculativeGenerationBatchRequest<'a, B, D, C>, PreparedPlainSpeculativeSemanticError>
    {
        let funding = self.preparation.metadata_funding().clone();
        let result = (|| {
            let parts = [
                size_of::<SpeculativeGenerationBatchRequest<'a, B, D, C>>(),
                size_of::<
                    Result<
                        SpeculativeGenerationBatchRequest<'a, B, D, C>,
                        PreparedPlainSpeculativeSemanticError,
                    >,
                >(),
            ];
            let bytes = sum(&parts).and_then(|n| n.checked_add(
                SpeculativeBuffer::<SpeculativeGenerationLane<'a, B, C>>::retained_control_bytes(1)?));
            let host = host(&funding, bytes)?;
            let mut lanes = SpeculativeBuffer::try_new_retained(1, host)?;
            // Preserve the exact inner owning failure rather than reconstructing it.
            let lane = self
                .into_lane(prompt, generation, constraint, cancellation)
                .map_err(|error| error.cause)?;
            lanes.try_push(lane).map_err(|_| {
                SpeculativeOutputError::Storage("single lane destination exhausted")
            })?;
            Ok(SpeculativeGenerationBatchRequest::new(
                drafting,
                lanes,
                fingerprint,
            ))
        })();
        result.map_err(|cause| PreparedPlainSpeculativeSemanticError {
            cause,
            funding: Some(funding),
        })
    }
}
fn sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .copied()
        .try_fold(std::mem::size_of_val(parts), usize::checked_add)
}
fn reserve(funding: &WorkspaceMetadataFunding, bytes: Option<usize>) -> Result<(), Cause> {
    let bytes = bytes.ok_or(SpeculativeOutputError::HostFunding(
        eredu_core::HostMetadataFundingError::Overflow,
    ))?;
    funding
        .reserve_metadata(bytes)
        .map_err(SpeculativeOutputError::from)?;
    Ok(())
}
fn host(
    funding: &WorkspaceMetadataFunding,
    bytes: Option<usize>,
) -> Result<HostPreparationAuthority, Cause> {
    reserve(
        funding,
        bytes.and_then(|n| {
            n.checked_add(HostPreparationAuthority::retention_bytes::<
                WorkspaceMetadataFunding,
            >()?)
        }),
    )?;
    Ok(HostPreparationAuthority::retain(funding.clone()))
}
impl<B: OriginalTokenizerBackend> LoadedModel<B> {
    /// Same original semantic producer plus the actual borrowed EOS copy and
    /// concrete callback Box. Settings have already been resolved by the facade;
    /// this helper introduces no sampler or scheduling policy.
    pub(super) fn prepare_original_speculative_plain_host<'a, F: FnMut(SemanticEvent) + 'a>(
        &self,
        source: &ManagedPlainTextSource,
        capacity: u64,
        maximum: usize,
        maximum_draft: usize,
        temperature: f32,
        eos: &[u32],
        stops: OriginalSpeculativeStops<'_>,
        skip_special: bool,
        callback: F,
    ) -> Result<PreparedOriginalSpeculativeHost<'a>, PreparedPlainSpeculativeSemanticError> {
        let (semantic, preparation) = self.prepare_original_speculative_plain_semantic(
            source,
            capacity,
            maximum,
            maximum_draft,
            stops,
            skip_special,
        )?;
        let funding = preparation.metadata_funding().clone();
        let result = (|| {
            let controls = [
                size_of::<PreparedOriginalSpeculativeHost<'a>>(),
                size_of::<
                    Result<
                        PreparedOriginalSpeculativeHost<'a>,
                        PreparedPlainSpeculativeSemanticError,
                    >,
                >(),
                size_of::<Cause>(),
                size_of::<PreparedPlainSpeculativeSemanticError>(),
                size_of::<(usize, usize, f32, &[u32], F)>(),
            ];
            reserve(&funding, sum(&controls))?;
            let configuration =
                preparation.prepare_configuration(maximum, maximum_draft, temperature, eos)?;
            let callback = preparation.prepare_callback(callback)?;
            Ok(PreparedOriginalSpeculativeHost {
                semantic,
                configuration,
                callback,
                preparation,
            })
        })();
        result.map_err(|cause| PreparedPlainSpeculativeSemanticError {
            cause,
            funding: Some(funding),
        })
    }
}
