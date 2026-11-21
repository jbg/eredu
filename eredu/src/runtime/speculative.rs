//! Portable speculative route selection and resident batch composition.

use eredu_core::{
    generation::GenerationError, ModelRuntime, SpeculativeGenerationBatchOutput,
    SpeculativeGenerationLane, SpeculativeGenerationVisitor, SpeculativeTokenFilterController,
};
use eredu_runtime::speculative::autoregressive::SpeculativeGenerationBackend;

use eredu_core::{BackendError, SpeculativeDriverError};
use eredu_runtime::speculative::autoregressive::{
    execute_embedded_batch, execute_external_batch, execute_resident_batch,
    AutoregressiveMechanisms, BatchDraft, BatchSession, ExternalBatchMechanisms,
    ResidentBatchRoute, SpeculativeBatchBinding,
};

/// Binds execution mechanisms for speculative batches selected by the portable facade.
pub trait SpeculativeBatchBackend<S: eredu_core::TextExecutionSession<Self>>:
    SpeculativeGenerationBackend<S>
{
    /// Runs the selected resident route with the facade's prepared request and visitor.
    fn execute_resident_batch<C, V>(
        runtime: &mut ModelRuntime<Self, S>,
        route: ResidentBatchRoute,
        lanes: Vec<SpeculativeGenerationLane<'_, S::Prompt, C>>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Self::Error>
    where
        C: SpeculativeTokenFilterController + 'static,
        V: SpeculativeGenerationVisitor;

    /// Runs the selected external draft through portable batch execution.
    fn execute_external_batch<C, V>(
        runtime: &mut ModelRuntime<Self, S>,
        drafter: &mut Self::Drafter,
        lanes: Vec<SpeculativeGenerationLane<'_, S::Prompt, C>>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Self::Error>
    where
        C: SpeculativeTokenFilterController + 'static,
        V: SpeculativeGenerationVisitor;

    /// Runs the installed prediction extension through portable batch execution.
    fn execute_embedded_batch<C, V>(
        runtime: &mut ModelRuntime<Self, S>,
        lanes: Vec<SpeculativeGenerationLane<'_, S::Prompt, C>>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Self::Error>
    where
        C: SpeculativeTokenFilterController + 'static,
        V: SpeculativeGenerationVisitor;
}

impl<B, M> SpeculativeBatchBackend<BatchSession<B, M>> for B
where
    B: SpeculativeBatchBinding<Mechanisms = M>
        + SpeculativeGenerationBackend<BatchSession<B, M>, Drafter = BatchDraft<B, M>>,
    M: ExternalBatchMechanisms<B>,
    BatchSession<B, M>: eredu_core::TextExecutionSession<
        B,
        Prompt = eredu_runtime::text_session_driver::TextPromptFor<B>,
    >,
    B::Error: From<BackendError>
        + From<eredu_core::BackendFailure>
        + From<GenerationError>
        + From<<M as AutoregressiveMechanisms>::Error>
        + From<SpeculativeDriverError<<M as AutoregressiveMechanisms>::Error>>,
{
    fn execute_resident_batch<C, V>(
        runtime: &mut ModelRuntime<Self, BatchSession<B, M>>,
        route: ResidentBatchRoute,
        lanes: Vec<
            SpeculativeGenerationLane<'_, eredu_runtime::text_session_driver::TextPromptFor<B>, C>,
        >,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Self::Error>
    where
        C: SpeculativeTokenFilterController + 'static,
        V: SpeculativeGenerationVisitor,
    {
        execute_resident_batch::<_, M, _, _>(runtime, route, lanes, visitor)
    }
    fn execute_external_batch<C, V>(
        runtime: &mut ModelRuntime<Self, BatchSession<B, M>>,
        draft: &mut Self::Drafter,
        lanes: Vec<
            SpeculativeGenerationLane<'_, eredu_runtime::text_session_driver::TextPromptFor<B>, C>,
        >,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Self::Error>
    where
        C: SpeculativeTokenFilterController + 'static,
        V: SpeculativeGenerationVisitor,
    {
        execute_external_batch::<_, M, _, _>(runtime, draft, lanes, visitor)
    }
    fn execute_embedded_batch<C, V>(
        runtime: &mut ModelRuntime<Self, BatchSession<B, M>>,
        lanes: Vec<
            SpeculativeGenerationLane<'_, eredu_runtime::text_session_driver::TextPromptFor<B>, C>,
        >,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Self::Error>
    where
        C: SpeculativeTokenFilterController + 'static,
        V: SpeculativeGenerationVisitor,
    {
        execute_embedded_batch::<_, M, _, _>(runtime, lanes, visitor)
    }
}

/// Failure selecting a portable route or executing its native mechanism.
#[derive(Debug, thiserror::Error)]
pub(crate) enum SpeculativeBatchExecutionError<E: std::error::Error + 'static> {
    /// The request is incompatible with the installed target.
    #[error(transparent)]
    Generation(GenerationError),
    /// The selected route could not execute.
    #[error(transparent)]
    Backend(E),
}

/// Selects the executor route for an already admitted batch.
pub(crate) fn execute_speculative_batch<'a, B, S, C, V>(
    runtime: &mut eredu_core::ModelRuntime<B, S>,
    drafting: eredu_core::SpeculativeDraft<'a, B::Drafter>,
    lanes: Vec<SpeculativeGenerationLane<'a, S::Prompt, C>>,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, SpeculativeBatchExecutionError<B::Error>>
where
    B: SpeculativeBatchBackend<S>,
    S: eredu_core::TextExecutionSession<B>,
    C: SpeculativeTokenFilterController + 'static,
    V: SpeculativeGenerationVisitor,
{
    use eredu_core::{
        decoding::DecodingModeError, refinement::PredictionScoreKind, SpeculativeDraft,
    };

    if !matches!(drafting, SpeculativeDraft::Target)
        && runtime.session().generation_score_kind() != PredictionScoreKind::CausalConditional
    {
        return Err(SpeculativeBatchExecutionError::Generation(
            GenerationError::RequiresCausalConditionalScores,
        ));
    }

    match drafting {
        SpeculativeDraft::Target if !runtime.session().decoding_capabilities().speculation => {
            B::execute_resident_batch(runtime, ResidentBatchRoute::Target, lanes, visitor)
        }
        SpeculativeDraft::Target | SpeculativeDraft::SharedBackbone => {
            B::execute_resident_batch(runtime, ResidentBatchRoute::SharedBackbone, lanes, visitor)
        }
        SpeculativeDraft::Embedded => B::execute_embedded_batch(runtime, lanes, visitor),
        SpeculativeDraft::External(drafter) => {
            B::execute_external_batch(runtime, drafter, lanes, visitor)
        }
        _ => {
            return Err(SpeculativeBatchExecutionError::Generation(
                GenerationError::DecodingMode(DecodingModeError::Unsupported(
                    "unsupported speculative draft source",
                )),
            ));
        }
    }
    .map_err(SpeculativeBatchExecutionError::Backend)
}
