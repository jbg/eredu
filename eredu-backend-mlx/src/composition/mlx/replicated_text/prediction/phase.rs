//! Native realization around the same portable prediction and target drivers.
use super::super::session::MlxReplicatedTextMechanisms;
use super::*;
use crate::composition::mlx::{
    prepared_speculative::MlxEmbeddedPredictionMechanisms,
    speculative::{IndependentLogits, SpeculativeExecutionStreams},
};
use eredu_architectures::{
    prediction_extension::{MaterializedPredictionExecutor, equation::PredictionEquation},
    speculative_execution::{
        PredictionCompletionPoint, PredictionPhaseEvidence, PredictionPhaseRoots,
        PredictionPhaseState, ReplicatedPredictionPhase,
    },
};
use eredu_core::speculative::{
    SpeculativeActivationOrigin, SpeculativeActivationPhase, SpeculativePrefillSpan,
};
use eredu_runtime::{
    ActivationObserver, ExpertPass, LayeredArchitecture, ReplicatedTextExecutionStrategy,
    ReplicatedTextSession,
};

mod binding;
mod capture;
mod completion;
mod controls;
mod prediction;
mod target;
pub(crate) mod external_target;
pub(crate) use completion::complete_original_prediction_state;

#[derive(Default)]
pub(crate) struct MlxPredictionPhase {
    origin: Option<SpeculativeActivationOrigin>,
}

impl<A, S, D, P>
    ReplicatedPredictionPhase<
        A,
        MlxNeuralBackend,
        S,
        MlxReplicatedTextMechanisms<A, S>,
        D,
        P,
        MlxEmbeddedPredictionMaterializer,
        MlxEmbeddedPredictionMechanisms,
    > for MlxPredictionPhase
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Parameterized<MlxTensor>,
    A::Unit: Parameterized<MlxTensor> + 'static,
    D: ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
        + 'static,
{
    fn requires_activation_origin(&self) -> bool {
        true
    }
    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        self.origin = origin;
    }

    fn run<'context, 'observer, R, F>(
        &mut self,
        session: &mut ReplicatedTextSession<
            A,
            MlxNeuralBackend,
            MlxReplicatedTextMechanisms<A, S>,
            D,
        >,
        extension: &mut P,
        lane: PredictionPhaseState<'_, P::LaneState>,
        pass: ExpertPass,
        equation: &PredictionEquation<&MlxTensor>,
        completion: Option<PredictionCompletionPoint>,
        span: Option<SpeculativePrefillSpan>,
        context: SpeculativeExecutionStreams<'context>,
        observer: Option<&'observer mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
        execute: F,
    ) -> Result<R, Error>
    where
        R: PredictionPhaseRoots<MlxTensor, IndependentLogits>,
        F: for<'execution, 'observation> FnOnce(
            &mut ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
            &mut P,
            &mut P::LaneState,
            Option<&'observation mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
            SpeculativeExecutionStreams<'execution>,
        ) -> Result<R, Error>,
    {
        prediction::run(
            self.origin,
            session,
            extension,
            lane,
            pass,
            equation,
            completion,
            span,
            context,
            observer,
            execute,
        )
    }

    fn run_target<'context, 'observer, R, F>(
        &mut self,
        session: &mut ReplicatedTextSession<
            A,
            MlxNeuralBackend,
            MlxReplicatedTextMechanisms<A, S>,
            D,
        >,
        evidence: PredictionPhaseEvidence<'_, S>,
        tokens: Option<&MlxTensor>,
        phase: SpeculativeActivationPhase,
        demand: eredu_core::OutputDemand,
        span: Option<SpeculativePrefillSpan>,
        context: SpeculativeExecutionStreams<'context>,
        observer: Option<&'observer mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
        execute: F,
    ) -> Result<R, Error>
    where
        R: PredictionPhaseRoots<MlxTensor, IndependentLogits>,
        F: for<'execution, 'observation> FnOnce(
            &mut ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
            Option<&'observation mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
            SpeculativeExecutionStreams<'execution>,
        ) -> Result<R, Error>,
    {
        target::run::<A, S, D, R, F>(
            self.origin,
            session,
            evidence,
            tokens,
            phase,
            demand,
            span,
            context,
            observer,
            execute,
        )
    }
}
