//! Funded semantic preparation shared by generation consumers.
pub(crate) mod chat;
mod invocation;
mod session;
use super::{LoadedModel, ManagedPlainTextSource};
use eredu_core::{
    generation::SemanticEvent, GenerationCancellationToken, HostPreparationAuthority,
    SemanticStateOwner, SpeculativeBuffer, SpeculativeBufferAllocationError,
    SpeculativeConfiguration, SpeculativeEventCallback, SpeculativeGenerationLane,
    SpeculativeOutputError, SpeculativeTokenFilterController, TextGenerationBackend,
    TextGenerationConfig,
};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::{
    OriginalSpeculativeHostError, OriginalTextSourceError, OriginalTokenizerBackend,
    PreparedSemanticSource,
};
use eredu_text::stop_storage::StopCompilePlan;
pub(crate) use session::PreparedSemanticInput;
pub use session::{
    PreparedChatBranch, PreparedChatOutputMode, PreparedChatPrompt, PreparedChatRequest,
    PreparedChatResumeSettings, PreparedChatSession, PreparedChatSnapshot,
};
use std::mem::size_of;

/// Borrowed stop declarations use the same source compiler without collecting
/// reference rows or cloning request strings before their original admission.
#[derive(Clone, Copy)]
pub(super) enum PreparedStops<'a> {
    Strings(&'a [String]),
    Refs(&'a [&'a str]),
}
impl<'a> PreparedStops<'a> {
    fn plan(self) -> Result<StopCompilePlan<'a>, eredu_text::stop_storage::StopSourceError> {
        match self {
            Self::Strings(stops) => StopCompilePlan::prepare(stops),
            Self::Refs(stops) => StopCompilePlan::prepare_refs(stops),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("{0}")]
    MemoryDomain(#[from] eredu_core::MemoryDomainError),
    #[error(transparent)]
    CaptureSource(#[from] eredu_runtime::working_memory::OriginalCaptureSourceError),
    #[error(transparent)]
    Record(#[from] crate::api::control::RecordConstructionError),
    #[error(transparent)]
    Sampling(
        #[from] eredu_core::execution_control::SamplingOverrideError<eredu_core::BackendFailure>,
    ),
    #[error("generation advancement produced neither a committed token nor termination")]
    MissingProgress,
    #[error(transparent)]
    Control(#[from] eredu_core::execution_control::ExecutionControlError),
    #[error(transparent)]
    Choice(#[from] eredu_runtime::execution_control::TokenChoiceError<crate::api::ConstraintError>),
    #[error(transparent)]
    PreparedChoice(
        #[from]
        eredu_runtime::execution_control::PreparedTokenChoiceError<
            crate::runtime::chat::constraints::OriginalPreparedGrammarController,
        >,
    ),
    #[error(transparent)]
    Boundary(
        eredu_core::TextContinuationError<eredu_core::BackendFailure, session::ChatControllerError>,
    ),
    #[error("{0}")]
    Generation(#[from] eredu_core::GenerationError),
    #[error("{0}")]
    Constraint(#[from] crate::api::ConstraintError),
    #[error(transparent)]
    Input(#[from] eredu_core::TokenInputRejection),
    #[error("{0}")]
    Backend(#[from] eredu_core::BackendFailure),
    #[error(transparent)]
    Chat(#[from] chat::ChatCause),
    #[error(transparent)]
    ChatInput(#[from] eredu_runtime::input::PreparedChatInputError),
    #[error(transparent)]
    Snapshot(
        #[from] eredu_runtime::execution_control::TextSnapshotError<eredu_core::BackendFailure>,
    ),
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
pub struct PreparedChatSessionError {
    #[source]
    cause: Cause,
    funding: Option<HostMetadataFunding>,
    input_funding: Option<eredu_runtime::input::OriginalModelInputCustody>,
}
impl PreparedChatSessionError {
    /// Exact committed prefix retained by an advancement failure, when one was
    /// constructed. Borrowing it preserves the error's original storage custody.
    pub fn committed_token_ids(&self) -> Option<&[u32]> {
        use std::error::Error;
        self.backend_failure()?
            .source()?
            .downcast_ref::<session::CursorFailure>()
            .map(session::CursorFailure::committed_token_ids)
    }
    /// Original attribution or record allocation refusal, with its retained payer.
    pub fn record_construction_failure(&self) -> Option<&crate::api::RecordConstructionError> {
        match &self.cause {
            Cause::Record(error) => Some(error),
            _ => None,
        }
    }
    /// Original snapshot or branch rejection, retaining its typed cause.
    pub fn snapshot_failure(
        &self,
    ) -> Option<&eredu_runtime::execution_control::TextSnapshotError<eredu_core::BackendFailure>>
    {
        match &self.cause {
            Cause::Snapshot(error) => Some(error),
            _ => None,
        }
    }
    /// Invalid prospective sampler policy, rejected without advancing state.
    pub fn sampling_rejection(&self) -> Option<&'static str> {
        match self.cause {
            Cause::Sampling(eredu_core::execution_control::SamplingOverrideError::Invalid(
                reason,
            )) => Some(reason),
            _ => None,
        }
    }
    /// Invalid lifecycle transition at a completed generation boundary.
    pub fn control_rejection(
        &self,
    ) -> Option<&eredu_core::execution_control::ExecutionControlError> {
        match &self.cause {
            Cause::Control(error) => Some(error),
            _ => None,
        }
    }
    /// Fixed prospective-choice refusal, preserving any nested source custody.
    pub fn token_choice_rejection(
        &self,
    ) -> Option<eredu_runtime::execution_control::TokenChoiceError<std::convert::Infallible>> {
        match &self.cause {
            Cause::PreparedChoice(error) => error.rejection(),
            Cause::Choice(error) => error.rejection(),
            _ => None,
        }
    }
    /// The retained template has no recognized semantic output protocol.
    pub fn semantic_output_rejection(&self) -> Option<crate::runtime::chat::SemanticSupport> {
        match self.cause {
            Cause::Chat(chat::ChatCause::SemanticOutput(reason)) => {
                Some(crate::runtime::chat::SemanticSupport::Unsupported { reason })
            }
            _ => None,
        }
    }
    /// Retained declaration policy rejected explicit literal text output.
    pub fn text_output_rejection(&self) -> Option<crate::runtime::chat::CapabilitySupport> {
        match self.cause {
            Cause::Chat(chat::ChatCause::TextOutput(reason)) => {
                Some(crate::runtime::chat::CapabilitySupport::Unsupported { reason })
            }
            _ => None,
        }
    }
    /// Neutral backend failure with the provider's exact kind, operation and
    /// original source. Its retained cursor and funding retire with this error.
    pub fn backend_failure(&self) -> Option<&eredu_core::BackendFailure> {
        match &self.cause {
            Cause::Backend(error)
            | Cause::Sampling(eredu_core::execution_control::SamplingOverrideError::Backend(
                error,
            ))
            | Cause::Boundary(eredu_core::TextContinuationError::Generation(
                eredu_core::ControlledTextGenerationError::Backend(error),
            ))
            | Cause::Snapshot(eredu_runtime::execution_control::TextSnapshotError::Backend(
                error,
            ))
            | Cause::Snapshot(
                eredu_runtime::execution_control::TextSnapshotError::HostPreparation(error),
            ) => Some(error),
            Cause::Snapshot(
                error @ eredu_runtime::execution_control::TextSnapshotError::Resume(_),
            ) => error.resume_backend_failure(),
            Cause::Snapshot(
                eredu_runtime::execution_control::TextSnapshotError::RetainedBackend(error),
            ) => std::error::Error::source(error)?.downcast_ref(),
            _ => None,
        }
    }

    /// Exact rendered-token or media-coordinate association refusal.
    pub fn chat_input_rejection(&self) -> Option<eredu_runtime::input::PreparedChatInputRejection> {
        match &self.cause {
            Cause::ChatInput(error) => error.association_rejection(),
            _ => None,
        }
    }

    /// Fixed input or source refusal, when it occurred before execution.
    pub fn input_rejection(&self) -> Option<eredu_core::TokenInputRejection> {
        match self.cause {
            Cause::Input(cause) => Some(cause),
            Cause::Chat(chat::ChatCause::Compilation(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )) => Some(eredu_core::TokenInputRejection::IdentityMismatch),
            _ => None,
        }
    }
    fn before(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
            funding: None,
            input_funding: None,
        }
    }
}
impl<B: OriginalTokenizerBackend> LoadedModel<B> {
    /// One source account for controller, decoder and optional media producers.
    /// Consumers pass this same owner through each preparation step.
    pub(crate) fn prepare_semantic_source(
        &self,
        source: &eredu_runtime::working_memory::OriginalTokenizer,
        limits: &eredu_core::MemoryLimitDeclarations,
    ) -> Result<PreparedSemanticSource, PreparedChatSessionError> {
        if !source.matches_configuration(&self.tokenizer) {
            return Err(PreparedChatSessionError::before(
                eredu_core::TokenInputRejection::IdentityMismatch,
            ));
        }
        let prepared = B::prepare_semantic_source(&self.runtime, source, limits)
            .map_err(PreparedChatSessionError::before)?;
        let funding = prepared.metadata_funding().clone();
        let result = (|| -> Result<_, Cause> {
            reserve(
                &funding,
                sum(&[
                    size_of::<(
                        &Self,
                        &eredu_runtime::working_memory::OriginalTokenizer,
                        u64,
                    )>(),
                    size_of::<PreparedSemanticSource>(),
                    size_of::<HostMetadataFunding>(),
                    size_of::<Result<(), eredu_core::TokenInputRejection>>(),
                    size_of::<Result<PreparedSemanticSource, Cause>>(),
                    size_of::<Result<PreparedSemanticSource, PreparedChatSessionError>>(),
                    size_of::<PreparedChatSessionError>(),
                ]),
            )?;
            B::validate_semantic_source(&self.runtime, &prepared)?;
            Ok(prepared)
        })();
        result.map_err(|cause| PreparedChatSessionError {
            cause,
            funding: Some(funding),
            input_funding: None,
        })
    }
    /// Builds the source-bound semantic owner used by managed speculative lanes.
    fn prepare_original_speculative_plain_semantic(
        &self,
        source: &ManagedPlainTextSource,
        limits: &eredu_core::MemoryLimitDeclarations,
        maximum: usize,
        maximum_draft: usize,
        stops: PreparedStops<'_>,
        skip_special: bool,
    ) -> Result<(SemanticStateOwner, PreparedSemanticSource), PreparedChatSessionError> {
        let prepared = self.prepare_semantic_source(source.original(), limits)?;
        let funding = prepared.metadata_funding().clone();
        let result = (|| {
            let parts = [
                size_of::<Cause>(),
                size_of::<PreparedChatSessionError>(),
                size_of::<Option<HostMetadataFunding>>(),
                size_of::<StopCompilePlan<'static>>(),
                size_of::<PreparedStops<'static>>(),
                size_of::<
                    Result<StopCompilePlan<'static>, eredu_text::stop_storage::StopSourceError>,
                >(),
                size_of::<(SemanticStateOwner, PreparedSemanticSource)>(),
                size_of::<
                    Result<(SemanticStateOwner, PreparedSemanticSource), PreparedChatSessionError>,
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
            let event_window = maximum_draft
                .checked_add(1)
                .and_then(std::num::NonZeroUsize::new)
                .ok_or(SpeculativeOutputError::HostFunding(
                    eredu_core::HostMetadataFundingError::Overflow,
                ))?;
            let semantic = prepared.prepare(&stops, maximum, event_window, skip_special)?;
            Ok((semantic, prepared))
        })();
        result.map_err(|cause| PreparedChatSessionError {
            cause,
            funding: Some(funding),
            input_funding: None,
        })
    }
}

/// Prepared facade metadata consumed by the same lane and batch contracts.
/// Every owning exit remains closed; the source header authenticates the H used
/// before these births. The callback's own application work remains caller-owned.
#[derive(Debug)]
pub(crate) struct PreparedOriginalSpeculativeHost<'a> {
    semantic: SemanticStateOwner,
    configuration: SpeculativeConfiguration,
    callback: SpeculativeEventCallback<'a>,
    preparation: PreparedSemanticSource,
}
impl<'a> PreparedOriginalSpeculativeHost<'a> {
    pub(crate) fn preparation(&self) -> &PreparedSemanticSource {
        &self.preparation
    }
    /// Creates the exact lane after paying its concrete generic transport.
    pub(crate) fn into_lane<B: TextGenerationBackend, C: SpeculativeTokenFilterController>(
        self,
        prompt: B::Prompt,
        generation: TextGenerationConfig,
        constraint: C,
        cancellation: GenerationCancellationToken,
    ) -> Result<SpeculativeGenerationLane<'a, B, C>, PreparedChatSessionError> {
        let funding = self.preparation.metadata_funding().clone();
        let result = (|| {
            let parts = [
                size_of::<SpeculativeGenerationLane<'a, B, C>>(),
                size_of::<Result<SpeculativeGenerationLane<'a, B, C>, PreparedChatSessionError>>(),
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
        result.map_err(|cause| PreparedChatSessionError {
            cause,
            funding: Some(funding),
            input_funding: None,
        })
    }
}
fn sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .copied()
        .try_fold(std::mem::size_of_val(parts), usize::checked_add)
}
fn reserve(funding: &HostMetadataFunding, bytes: Option<usize>) -> Result<(), Cause> {
    let bytes = bytes.ok_or(SpeculativeOutputError::HostFunding(
        eredu_core::HostMetadataFundingError::Overflow,
    ))?;
    funding
        .reserve_metadata(bytes)
        .map_err(SpeculativeOutputError::from)?;
    Ok(())
}
fn host(
    funding: &HostMetadataFunding,
    bytes: Option<usize>,
) -> Result<HostPreparationAuthority, Cause> {
    reserve(
        funding,
        bytes.and_then(|n| {
            n.checked_add(HostPreparationAuthority::retention_bytes::<
                HostMetadataFunding,
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
        limits: &eredu_core::MemoryLimitDeclarations,
        maximum: usize,
        maximum_draft: usize,
        temperature: f32,
        eos: &[u32],
        stops: PreparedStops<'_>,
        skip_special: bool,
        callback: F,
    ) -> Result<PreparedOriginalSpeculativeHost<'a>, PreparedChatSessionError> {
        let (semantic, preparation) = self.prepare_original_speculative_plain_semantic(
            source,
            limits,
            maximum,
            maximum_draft,
            stops,
            skip_special,
        )?;
        let funding = preparation.metadata_funding().clone();
        let result = (|| {
            let controls = [
                size_of::<PreparedOriginalSpeculativeHost<'a>>(),
                size_of::<Result<PreparedOriginalSpeculativeHost<'a>, PreparedChatSessionError>>(),
                size_of::<Cause>(),
                size_of::<PreparedChatSessionError>(),
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
        result.map_err(|cause| PreparedChatSessionError {
            cause,
            funding: Some(funding),
            input_funding: None,
        })
    }
}
