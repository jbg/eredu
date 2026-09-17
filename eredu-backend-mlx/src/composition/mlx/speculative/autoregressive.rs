//! Generic ordinary-decoder operations for the portable independent-draft transaction.
use super::*;
use crate::backend::{managed_memory::NativeMemoryOwner, submission_recovery};
use crate::composition::mlx::{
    MlxModelInput, model::Executable, replicated_text::MlxPredictionTargetState,
};
use eredu_runtime::speculative::autoregressive::{
    AutoregressiveMechanisms, AutoregressivePass, AutoregressivePrefill, AutoregressiveSource,
};
use eredu_runtime::working_memory::WorkingMemoryPool;
use std::rc::Rc;
pub(super) mod host_containers;
pub(in crate::composition::mlx::speculative) mod input_readout;
mod prefill_input;
pub(in crate::composition::mlx) mod sequence;
mod startup;
mod state_copy;
mod typed_completion;
mod verification_tokens;
use typed_completion::AutoregressiveCompletion;
mod stream;
pub(in crate::composition::mlx::speculative) use stream::StateStream;
mod workspace;
pub(crate) use workspace::{
    ActiveSpeculativeInvocation, AutoregressiveSourcePair, AutoregressiveWorkspaceRecipe,
    PreparedAutoregressiveInvocation,
};

pub(crate) struct MlxAutoregressiveState {
    native: MlxPredictionTargetState,
    stream: StateStream,
    pool: WorkingMemoryPool,
    // Exact model/parameter epoch at cache creation. A missing fixed source
    // remains ordinary-only; checkpoint/restore never manufactures an epoch.
    source_origin: Option<eredu_runtime::replicated_session::ReplicatedTextControlOrigin>,
    source_role: AutoregressiveSource,
    // Complete retained copy prerequisites and their funding retire last.
    original_copy: Option<state_copy::StateCopyContext>,
}
pub(crate) struct MlxAutoregressiveMechanisms;
/// Produced only by the actual settled final prefill span. Its private fields
/// cannot be reconstructed from a raw array and an unrelated role.
pub(crate) struct CompletedNumericalPrefill {
    value: Array,
    stream: StateStream,
    custody: eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody,
    funding: eredu_nn::workspace::WorkspaceMetadataFunding,
}
impl CompletedNumericalPrefill {
    pub(in crate::composition::mlx::speculative) fn funding(
        &self,
    ) -> &eredu_nn::workspace::WorkspaceMetadataFunding {
        &self.funding
    }
    pub(in crate::composition::mlx::speculative) fn into_parts(
        self,
    ) -> (
        Array,
        StateStream,
        eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody,
        eredu_nn::workspace::WorkspaceMetadataFunding,
    ) {
        (self.value, self.stream, self.custody, self.funding)
    }
}
pub(crate) struct MlxAutoregressiveOutput {
    logits: Array,
    stream: StateStream,
    original: Option<input_readout::CompletedReadouts>,
}
impl MlxAutoregressiveOutput {
    /// Moves exactly one completed row into the original numerical carrier.
    /// This shares the existing once-only row slot with ordinary `logits`.
    pub(crate) fn take_numerical(
        &self,
        position: usize,
    ) -> Result<super::sampling::numerical::OriginalNumericalValue, Error> {
        self.original
            .as_ref()
            .ok_or(Error::InvalidOperation(
                "numerical source requires a completed original row",
            ))?
            .take_numerical(position, &self.stream)
    }
    pub(super) fn completed_role(
        &self,
    ) -> Option<&eredu_runtime::working_memory::OriginalSpeculativeRole> {
        self.original.as_ref().map(|completed| completed.role())
    }
}

impl AutoregressiveMechanisms for MlxAutoregressiveMechanisms {
    type Model = Executable;
    type Input = MlxModelInput;
    type State = MlxAutoregressiveState;
    type Checkpoint = state_copy::StateCheckpoint;
    type Output = MlxAutoregressiveOutput;
    type Logits = super::sampling::logits::IndependentLogits;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = AutoregressiveCompletion;
    type Telemetry = scheduler::SpeculativeComponentTimings;
    type Error = Error;
    fn request_context<'a>(request: eredu_core::SpeculativeRequestId, context: Self::Context<'a>)
        -> Result<Self::Context<'a>, Self::Error> where Self: 'a {
        context.request_context(request)
    }
    fn occurrence_request(context: Self::Context<'_>) -> Option<eredu_core::SpeculativeRequestId> {
        context.selected_request()
    }
    fn driver_buffer<T>(
        capacity: usize,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeBuffer<T>, Error> {
        host_containers::buffer(capacity, context)
    }
    fn driver_host_metadata(
        bytes: Option<usize>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::HostPreparationAuthority, Self::Error> {
        host_containers::metadata(bytes, context)
    }
    fn driver_buffer_bytes<T>(capacity: usize) -> Option<usize> {
        host_containers::buffer_bytes::<T>(capacity)
    }
    fn driver_identity(
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeRequestIdentity, Error> {
        host_containers::identity(context)
    }
    fn coordinate_speculative_buffer(
        local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<
        eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        eredu_core::BackendFailure,
    > {
        context.coordinate_speculative_step(local)
    }

    fn copy_sequence(
        source: eredu_core::SpeculativeSequenceRef<'_>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeSequence, eredu_core::SpeculativeDriverError<Error>> {
        sequence::copy(source, context)
    }
    fn sequence_copy_bytes(source: &eredu_core::SpeculativeSequence) -> Option<u64> {
        sequence::copy_bytes(source)
    }

    fn take_retained_failure(error: Error) -> Result<eredu_core::BackendFailure, Error> {
        error.take_retained_backend_failure()
    }
    fn agree_text_preparation(
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        context.agree_text_preparation(stage, status)
    }
    fn coordinate_speculative_step(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        context.coordinate_speculative_step(local)
    }
    fn verification_tokens(
        tokens: &[u32],
        context: Self::Context<'_>,
    ) -> Result<eredu_core::GenerationTokenIds, Error> {
        match context.original_execution() {
            Some((sources, environment)) => {
                sources.validate_environment(environment)?;
                verification_tokens::copy(tokens, sources.metadata_funding())
            }
            None => Ok(tokens.to_vec().into()),
        }
    }
    fn empty(
        model: &mut Self::Model,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::State, Error> {
        if let Some((sources, environment)) = context.original_execution() {
            return startup::prepare(model, pass, sources, environment)
                .map_err(|cause| sources.retain_error(cause));
        }
        let stream = match pass {
            AutoregressivePass::DraftPrefill => context.draft(),
            _ => context.target(),
        };
        let owner = match context.memory_owner() {
            Some(owner) => owner.clone(),
            None => NativeMemoryOwner::acquire(&context.memory_pool())?,
        };
        submission_recovery::detached_retained(owner.clone(), || {
            let source_origin = model
                .erased()
                .resident_control_origin_fixed()
                .and_then(Result::ok);
            let mut native = model.erased_mut().prepare_autoregressive_cache()?;
            native.retain_memory_owner(&owner)?;
            Ok(MlxAutoregressiveState {
                native,
                stream: stream.clone().into(),
                pool: owner.pool().clone(),
                source_origin,
                source_role: if pass == AutoregressivePass::DraftPrefill {
                    AutoregressiveSource::Draft
                } else {
                    AutoregressiveSource::Target
                },
                original_copy: None,
            })
        })
    }
    fn prefill(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        pass: AutoregressivePass,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativePrefillOutcome<AutoregressivePrefill<Self::Logits>>, Error>
    {
        if let Some((sources, environment)) = context.original_execution() {
            return (|| {
                sources.validate_environment(environment)?;
                reserve_logits_transport::<
                    eredu_core::SpeculativePrefillOutcome<AutoregressivePrefill<Self::Logits>>,
                >(sources)?;
                sources
                    .active_prefill()?
                    .run_numerical(model, state, pass, cancellation, sources, environment)
                    .map(|outcome| {
                        outcome.map(|result| AutoregressivePrefill {
                            logits: result
                                .logits
                                .map(super::sampling::logits::IndependentLogits::Original),
                            evaluated_tokens: result.evaluated_tokens,
                        })
                    })
            })()
            .map_err(|cause| sources.retain_error(cause));
        }
        let sample = match pass {
            AutoregressivePass::TargetPrefill => true,
            AutoregressivePass::DraftPrefill => false,
            _ => {
                return Err(Error::InvalidOperation(
                    "non-prefill pass entered selected prefill",
                ));
            }
        };
        input
            .with_borrowed(|input| {
                model.erased_mut().autoregressive_prefill(
                    input,
                    &mut state.native,
                    sample,
                    cancellation,
                    &state.stream,
                )
            })
            .map(|outcome| {
                outcome.map(|result| AutoregressivePrefill {
                    logits: result
                        .logits
                        .map(super::sampling::logits::IndependentLogits::Ordinary),
                    evaluated_tokens: result.evaluated_tokens,
                })
            })
    }
    fn decode(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::Output, Error> {
        if let Some(invocation) = context.original_invocation()? {
            let mut io = invocation.take_io()?;
            let tokens = io.input(tokens, pass, invocation.observer())?;
            invocation.begin_equation_construction()?;
            let logits = model.erased_mut().autoregressive_forward_with_completion(
                &tokens,
                &mut state.native,
                &state.stream,
                &mut io.sequence_completion(&invocation),
            )?;
            return io.finish(logits, &state.stream, &invocation);
        }
        let tokens = Array::from_slice(tokens, &[1, tokens.len() as i32]);
        model
            .erased_mut()
            .autoregressive_forward(&tokens, &mut state.native, false, &state.stream)
            .map(|logits| MlxAutoregressiveOutput {
                logits,
                stream: state.stream.clone(),
                original: None,
            })
    }
    fn prepare_continuation(
        continuation: &eredu_runtime::speculative::autoregressive::AutoregressiveContinuation,
        context: Self::Context<'_>,
    ) -> Result<(), Error> {
        let Some((sources, environment)) = context.original_execution() else { return Ok(()); };
        sources.validate_environment(environment)?;
        let controls = [
            std::mem::size_of::<SpeculativeExecutionStreams<'_>>(),
            std::mem::size_of::<(&AutoregressiveSourcePair, &crate::backend::OriginalCopyEnvironment<'_>)>(),
            std::mem::size_of::<Result<(), Error>>(),
            std::mem::size_of::<Result<(), eredu_runtime::working_memory::SpeculativeContinuationError>>(),
        ];
        sources.metadata_funding().reserve_metadata(controls.into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        sources.request().prepare_continuation(continuation, sources.metadata_funding())
            .map_err(|cause| sources.retain_startup_error(cause))
    }
    fn occurrence_error(
        cause: eredu_runtime::speculative::autoregressive::AutoregressiveOccurrenceError,
    ) -> Error {
        Error::SpeculativeOccurrence(cause)
    }
    fn invocation_frontier(state: &Self::State) -> Result<Option<u64>, Error> {
        state.native.generation().map(Some)
    }
    fn invocation_input_positions(input: &Self::Input) -> Result<Option<usize>, Error> {
        inspect_plain_input(input)
            .map(|source| source.map(|(positions, _)| positions))
            .map_err(Error::from)
    }

    fn with_invocation<T>(
        model: &mut Self::Model,
        state: &mut Self::State,
        input: Option<&Self::Input>,
        claim: eredu_runtime::speculative::autoregressive::AutoregressiveOccurrenceClaim<'_>,
        context: Self::Context<'_>,
        run: impl FnOnce(&mut Self::Model, &mut Self::State) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let Some((sources, environment)) = context.original_execution() else {
            Self::before_invocation(model, state, claim, context)?;
            return run(model, state);
        };
        (|| {
            sources.validate_environment(environment)?;
            sources.validate_source(model, claim.invocation().source())?;
            if !claim.belongs_to(sources.schedule_identity())
                || state.source_role != claim.invocation().source()
                || !sources.matches_state_origin(state.source_role, state.source_origin.as_ref())
            {
                return Err(Error::InvalidOperation(
                    "prefill source differs from its claimed role",
                ));
            }
            let prepared_prefill =
                if claim.invocation().execution_pass() == eredu_runtime::ExpertPass::Prefill {
                    let input = input.ok_or(Error::InvalidOperation(
                        "prefill invocation requires its exact input",
                    ))?;
                    let initialized = model.erased().prefill_roots_runtime()?;
                    let mechanisms = model
                        .workspace_mechanisms()
                        .ok_or(Error::PrefillScopeUnavailable)?;
                    Some(prefill_input::PreparedPrefillInput::prepare_copy(
                        input,
                        environment,
                        &initialized,
                        mechanisms,
                        sources.metadata_funding(),
                        sources.request().capacity_bytes(),
                    )?)
                } else {
                    None
                };
            PreparedAutoregressiveInvocation::prepare(
                sources,
                model,
                state,
                input,
                claim,
                prepared_prefill,
            )
            .and_then(|prepared| prepared.run(environment, run))
        })()
        .map_err(|cause| sources.retain_error(cause))
    }

    fn checkpoint(state: &Self::State) -> Result<Self::Checkpoint, Error> {
        let copy = if let Some(original) = &state.original_copy {
            original.copy(state, None)?
        } else {
            MlxAutoregressiveState {
                native: state.native.control_copy(&state.stream, &state.pool)?,
                stream: state.stream.clone(),
                pool: state.pool.clone(),
                source_origin: state.source_origin.clone(),
                source_role: state.source_role,
                original_copy: None,
            }
        };
        state_copy::StateCheckpoint::new(copy)
    }
    fn restore(saved: &Self::Checkpoint, context: Self::Context<'_>) -> Result<Self::State, Error> {
        if let Some(original) = &saved.original_copy {
            return original.copy(saved, Some(context));
        }
        if let Some((sources, _)) = context.original_execution() {
            return Err(sources.retain_startup_error(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let pool = context.memory_pool();
        Ok(MlxAutoregressiveState {
            native: saved.native.control_copy(&saved.stream, &pool)?,
            stream: saved.stream.clone(),
            pool,
            source_origin: saved.source_origin.clone(),
            source_role: saved.source_role,
            original_copy: None,
        })
    }
    fn estimate(
        saved: &Self::Checkpoint,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        Self::estimate_state(saved)
    }
    fn estimate_state(
        state: &Self::State,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let mut estimate = state.native.control_estimate()?;
        let wrapper = std::mem::size_of::<Self::State>() as u64;
        estimate.retained_bytes = estimate.retained_bytes.checked_add(wrapper)?;
        estimate.copy_bytes = estimate.copy_bytes.checked_add(wrapper)?;
        Some(estimate)
    }
    fn logits(
        output: &Self::Output,
        position: usize,
        context: Self::Context<'_>,
    ) -> Result<Self::Logits, Error> {
        if let Some((sources, environment)) = context.original_execution() {
            return (|| {
                sources.validate_environment(environment)?;
                // Authenticate before consuming the one-use row or charging its
                // wrapper to the current request. Equal shapes are insufficient.
                let role = output.completed_role().ok_or(Error::InvalidOperation(
                    "original logits require a completed original output",
                ))?;
                let custody = role.budget_custody();
                if !eredu_runtime::working_memory::SpeculativeNumericalSource::Model(&custody)
                    .belongs_to_request(sources.request())
                    || &*output.stream != environment.stream()
                {
                    return Err(Error::InvalidOperation(
                        "original logits source differs from request",
                    ));
                }
                reserve_logits_transport::<Self::Logits>(sources)?;
                output
                    .take_numerical(position)
                    .map(super::sampling::logits::IndependentLogits::Original)
            })()
            .map_err(|cause| sources.retain_error(cause));
        }
        if output.original.is_some() {
            return Err(Error::InvalidOperation(
                "original logits require their execution context",
            ));
        }
        output
            .logits
            .try_index_device((.., position as i32, ..), &*output.stream)
            .map(super::sampling::logits::IndependentLogits::Ordinary)
            .map_err(Error::from)
    }
    fn completion(output: &Self::Output, _: Self::Context<'_>) -> Result<Self::Completion, Error> {
        if let Some(completed) = MlxSpeculativeCompletion::completed(output) {
            return Ok(AutoregressiveCompletion::new(completed));
        }
        MlxSpeculativeCompletion::submit([&output.logits])
            .map(AutoregressiveCompletion::new)
            .map_err(Error::from)
    }
    fn invalid(message: &'static str) -> Error {
        Error::InvalidOperation(message)
    }
}

#[cfg(test)]
mod memory_tests;

// One borrowed descriptor worker for the installed occurrence hook and the
// source-loaned equation producer. The original producer prepays descriptor
// controls; this emits no error owner or partially published source table.
fn inspect_plain_input(
    input: &MlxModelInput,
) -> Result<Option<(usize, eredu_nn::workspace::WorkspaceDtype)>, safemlx::ArrayDescriptorError> {
    let Some(tokens) = input.plain_token_array() else {
        return Ok(None);
    };
    let descriptor = tokens.try_descriptor()?;
    let dtype = match descriptor.facts().dtype() {
        safemlx::Dtype::Int32 => eredu_nn::workspace::WorkspaceDtype::Int32,
        safemlx::Dtype::Uint32 => eredu_nn::workspace::WorkspaceDtype::Uint32,
        _ => return Ok(None),
    };
    let positions = match descriptor.shape() {
        [1, positions] if *positions > 0 => usize::try_from(*positions).ok(),
        _ => None,
    };
    Ok(positions.map(|positions| (positions, dtype)))
}

// The numerical producer owns its inner value. This adapter separately pays the
// actual outer result/enum transports before either a row move or prefill work.
// Successful original values retain the same funding; retained failures do too.
fn reserve_logits_transport<T>(sources: &AutoregressiveSourcePair) -> Result<(), Error> {
    use eredu_nn::workspace::WorkspaceMetadataFundingError;
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<T>(),
        size_of::<Result<T, Error>>(),
        size_of::<super::sampling::logits::IndependentLogits>(),
        size_of::<Option<super::sampling::logits::IndependentLogits>>(),
        size_of::<eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody>(),
        size_of::<eredu_runtime::working_memory::SpeculativeNumericalSource<'_>>(),
        size_of::<(
            &AutoregressiveSourcePair,
            usize,
            SpeculativeExecutionStreams<'_>,
        )>(),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
    sources
        .metadata_funding()
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod completed_leaf_tests;
