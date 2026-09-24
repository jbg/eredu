//! Generic ordinary-decoder operations for the portable independent-draft transaction.
use super::*;
use crate::backend::runtime::media::input;
use crate::composition::mlx::{
    replicated_text::{ErasedReplicatedTextExecutable, MlxPredictionTargetState},
    MlxModelInput,
};
use eredu_runtime::speculative::autoregressive::{AutoregressiveMechanisms, AutoregressivePass};
use std::rc::Rc;

pub(crate) struct MlxAutoregressiveState {
    native: MlxPredictionTargetState,
    stream: Stream,
}
pub(crate) struct MlxAutoregressiveMechanisms;
pub(crate) struct MlxAutoregressiveOutput {
    logits: Array,
    stream: Stream,
}
fn exception(e: Error) -> Exception {
    Exception::custom(e.to_string())
}
impl AutoregressiveMechanisms for MlxAutoregressiveMechanisms {
    type Model = dyn ErasedReplicatedTextExecutable;
    type Input = MlxModelInput;
    type State = MlxAutoregressiveState;
    type Checkpoint = Rc<MlxAutoregressiveState>;
    type Output = MlxAutoregressiveOutput;
    type Logits = Array;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = scheduler::SpeculativeComponentTimings;
    type Error = Exception;
    fn empty(
        model: &mut Self::Model,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::State, Exception> {
        let stream = match pass {
            AutoregressivePass::DraftPrefill => context.draft(),
            _ => context.target(),
        };
        Ok(MlxAutoregressiveState {
            native: model.prepare_autoregressive_cache().map_err(exception)?,
            stream: stream.clone(),
        })
    }
    fn prefill(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        _: AutoregressivePass,
        _: Self::Context<'_>,
    ) -> Result<(Self::Output, usize), Exception> {
        let tokens = input.with_borrowed(|input| input::text_token_ids(input, &state.stream))?;
        let count = tokens.shape()[1] as usize;
        let output = model
            .autoregressive_forward(&tokens, &mut state.native, true, &state.stream)
            .map_err(exception)?;
        Ok((
            MlxAutoregressiveOutput {
                logits: output,
                stream: state.stream.clone(),
            },
            count,
        ))
    }
    fn decode(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        _: AutoregressivePass,
        _: Self::Context<'_>,
    ) -> Result<Self::Output, Exception> {
        let tokens = Array::from_slice(tokens, &[1, tokens.len() as i32]);
        model
            .autoregressive_forward(&tokens, &mut state.native, false, &state.stream)
            .map(|logits| MlxAutoregressiveOutput {
                logits,
                stream: state.stream.clone(),
            })
            .map_err(exception)
    }
    fn checkpoint(state: &Self::State) -> Result<Self::Checkpoint, Exception> {
        Ok(Rc::new(MlxAutoregressiveState {
            native: state.native.control_copy(&state.stream)?,
            stream: state.stream.clone(),
        }))
    }
    fn restore(saved: &Self::Checkpoint, _: Self::Context<'_>) -> Result<Self::State, Exception> {
        Ok(MlxAutoregressiveState {
            native: saved.native.control_copy(&saved.stream)?,
            stream: saved.stream.clone(),
        })
    }
    fn estimate(
        saved: &Self::Checkpoint,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        Self::estimate_state(saved)
    }
    fn checkpoint_retained_bytes(saved: &Self::Checkpoint) -> Option<u64> {
        saved
            .native
            .control_estimate()?
            .retained_bytes
            .checked_add(saved.native.control_growth(0)?)
    }
    fn estimate_state(
        state: &Self::State,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        state.native.control_estimate()
    }
    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        model: &mut Self::Model,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        model.trim_parameter_conversions()
    }

    fn parameter_conversion_retention(
        model: &Self::Model,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    > {
        model.parameter_conversion_retention()
    }

    fn memory_observation(
        model: &Self::Model,
        state: &Self::State,
        additional_positions: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeModelMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        let position = state
            .native
            .generation()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let retained = state.native.control_estimate().map(|e| e.retained_bytes);
        let current = retained.and_then(|n| n.checked_add(state.native.control_growth(0)?));
        let peak = retained
            .and_then(|n| n.checked_add(state.native.control_growth(additional_positions)?));
        let parameters = crate::composition::mlx::capability::static_memory_from_residency(
            model
                .residency_report()
                .map_err(eredu_core::BackendFailure::from_error)?,
            model
                .parameter_bank_report()
                .map_err(eredu_core::BackendFailure::from_error)?,
        )
        .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(Some(
            eredu_core::speculative::SpeculativeModelMemoryObservation {
                current_positions: position,
                current_state_bytes: current,
                peak_state_bytes: peak,
                parameters,
                available: crate::composition::mlx::capability::available_memory()
                    .map_err(eredu_core::BackendFailure::from_error)?,
                allocator_cache_limit: match crate::allocator_cache_policy() {
                    Ok(policy) => eredu_core::Observed::exact(
                        policy.limit_bytes,
                        "current MLX allocator-cache policy",
                    ),
                    Err(error) => eredu_core::Observed::unavailable(error.to_string()),
                },
            },
        ))
    }
    fn logits(
        output: &Self::Output,
        position: usize,
        _: Self::Context<'_>,
    ) -> Result<Array, Exception> {
        output
            .logits
            .try_index_device((.., position as i32, ..), &output.stream)
    }
    fn completion(
        output: &Self::Output,
        _: Self::Context<'_>,
    ) -> Result<Self::Completion, Exception> {
        MlxSpeculativeCompletion::submit([&output.logits])
    }
    fn invalid(message: &'static str) -> Exception {
        Exception::custom(message)
    }
}
