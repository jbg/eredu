//! Source/selection composition for originally admitted speculative requests.
use super::*;
mod batch;
mod embedded;
mod external;
mod independent;
use crate::composition::mlx::speculative::autoregressive::{
    AutoregressiveSourcePair, MlxAutoregressiveMechanisms,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::{
    speculative::autoregressive::{AutoregressiveExecutor, AutoregressiveSchedulePlan},
    working_memory::WorkingMemoryError,
};
use std::{
    mem::{size_of, size_of_val},
    num::NonZeroU64,
};

/// Concrete request geometry supplied by shared planning. A lane's real prompt
/// and selected context ceiling must supply these values before this native bind.
pub(super) struct OriginalIndependentRequest<'a> {
    pub options: eredu_core::SpeculativeSchedulerOptions,
    pub input_positions: NonZeroU64,
    pub context_positions: NonZeroU64,
    pub preparation: &'a eredu_runtime::working_memory::PreparedSemanticSource,
}

/// Holds no model/source clone. The draft's actual selected realization and model
/// are lent as disjoint fields under its existing operation guard. The schedule
/// moves into the ordinary executor and all numerical consumers see the same pair.
pub(super) fn with_original_independent<'lane, 'world, C: SpeculativeTokenFilterController, T>(
    backend: &MlxBackend<'_>,
    session: &mut super::super::session::MlxModelSession,
    drafter: &mut MlxDrafter,
    lane: SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>,
    request: OriginalIndependentRequest<'_>,
    run: impl FnOnce(
        &mut AutoregressiveExecutor<'_, MlxAutoregressiveMechanisms>,
        SpeculativeExecutionStreams<'_>,
        SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>,
    ) -> Result<T, Error>,
) -> Result<T, Error> {
    let identity = session
        .original_model_source()
        .map_err(|cause| Error::PrefillControl(cause).at_speculative_stage("target source loan"))?
        .erased()
        .inference_execution_identity();
    drafter.autoregressive_source().map_err(|cause| {
        Error::PrefillControl(cause).at_speculative_stage("independent source loan")
    })?;
    // The currently qualified pair uses this exact selected stream. Split
    // transport requires its own producer; never substitute a different stream.
    if drafter.stream() != backend.stream() {
        return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound)
            .at_speculative_stage("independent source stream"));
    }
    if drafter.topology() != eredu_core::SpeculativeExecutionTopology::Single {
        return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound)
            .at_speculative_stage("independent source topology"));
    }
    // Authenticate the closed source header before lending either model.
    // Its cumulative account was opened before facade semantic/lane births.
    request
        .preparation
        .validate(backend.memory_pool(), identity)
        .map_err(|_| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let capacity_bytes = request.preparation.capacity_bytes();
    let funding = request.preparation.metadata_funding().clone();
    let parts = [
        size_of::<OriginalIndependentRequest<'_>>(),
        size_of::<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>(),
        size_of_val(&run),
        size_of::<AutoregressiveExecutor<'_, MlxAutoregressiveMechanisms>>(),
        size_of::<
            Result<
                AutoregressiveExecutor<'_, MlxAutoregressiveMechanisms>,
                eredu_runtime::speculative::autoregressive::AutoregressiveOccurrenceError,
            >,
        >(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<Result<T, Error>>(),
        size_of::<HostMetadataFunding>(),
        crate::backend::OriginalCopyEnvironment::control_bytes().ok_or(
            Error::WorkspacePlanning(HostMetadataFundingError::Overflow),
        )?,
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(
            HostMetadataFundingError::Overflow,
        ))?;
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    let environment = backend
        .original_copy_environment()
        .map_err(|cause| super::super::model::retain_planning_error(cause, funding.clone()))?;
    let intervention_declaration = session.original_intervention_declaration(&funding)?;
    session.with_model_operation_funded(funding.clone(), |target| {
        drafter.with_autoregressive_funded(funding.clone(), |draft, selected| {
            let capacity = selected.requirements().strategy().proposal_capacity();
            let schedule = AutoregressiveSchedulePlan::new(
                selected,
                capacity,
                request.input_positions,
                request.context_positions,
                lane.config(),
                request.options,
            )
            .map_err(|cause| super::super::model::retain_planning_error(cause, funding.clone()))?;
            funding
                .reserve_metadata(schedule.control_bytes().ok_or(Error::WorkspacePlanning(
                    HostMetadataFundingError::Overflow,
                ))?)
                .map_err(Error::WorkspacePlanning)?;
            let sources = AutoregressiveSourcePair::prepare_funded(
                target,
                draft.executable(),
                &schedule,
                backend.memory_pool(),
                capacity_bytes,
                funding.clone(),
            )?
            .with_intervention_declaration(intervention_declaration)?;
            let streams = SpeculativeExecutionStreams::single(backend.stream())
                .with_original_sources(&sources, &environment)?;
            let mut executor =
                AutoregressiveExecutor::new(target, draft.executable_mut(), capacity)
                    .with_schedule(schedule)
                    .map_err(|cause| sources.retain_startup_error(cause))?;
            let result = run(&mut executor, streams, lane);
            // Closing future issuance never retires existing values, copy
            // accounts or pending native completion owned by either model.
            let closed = sources
                .request()
                .close()
                .map_err(|cause| sources.retain_startup_error(cause));
            match result {
                Err(cause) => Err(sources.retain_error(cause)),
                Ok(value) => closed.map(|()| value),
            }
        })
    })
}

/// Consume the actual paid facade lane under its authenticated source pair.
/// Constructor policy stays shared; no ordinary domain lease is acquired here.
fn prepare_lane<'a, 'world, C: SpeculativeTokenFilterController>(
    mut lane: SpeculativeGenerationLane<'a, MlxBackend<'world>, C>,
    preparation: &eredu_runtime::working_memory::PreparedSemanticSource,
    streams: SpeculativeExecutionStreams<'_>,
) -> Result<MlxSpeculativeLaneRuntime<'a, C>, Error> {
    let (sources, environment) = streams
        .original_numerical()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    sources.validate_environment(environment)?;
    let funding = sources.metadata_funding();
    let parts = [
        size_of::<SpeculativeGenerationLane<'a, MlxBackend<'world>, C>>(),
        size_of::<MlxSpeculativeLaneRuntime<'a, C>>(),
        size_of::<Result<MlxSpeculativeLaneRuntime<'a, C>, Error>>(),
        size_of::<eredu_core::TextGenerationConfig>(),
        size_of::<MlxPreparedSampler<C>>(),
        size_of::<MlxTextSampler>(),
        size_of::<Option<MlxSpeculativeSeed>>(),
        size_of::<Result<MlxSpeculativeSeed, eredu_core::speculative::SpeculativeControlError>>(),
        size_of::<(
            &MlxModelInput,
            &eredu_runtime::working_memory::WorkingMemoryPool,
            &eredu_runtime::SharedPreparedInputCacheIdentity,
            Option<&eredu_runtime::input::PreparedModelInputOwner<crate::MlxTensor>>,
            Result<&eredu_runtime::input::PreparedModelInputOwner<crate::MlxTensor>, WorkingMemoryError>,
            Result<&eredu_runtime::working_memory::MediaSessionBinding, eredu_core::PreparedRequestRejection>,
        )>(),
    ];
    funding
        .reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(
                    HostMetadataFundingError::Overflow,
                ))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    preparation
        .validate_configuration(lane.configuration())
        .map_err(|cause| sources.retain_startup_error(cause))?;
    preparation
        .validate_callback(lane.event_callback())
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let generation = *lane.generation();
    if generation.inference_policy().managed_memory_capacity_bytes
        != Some(preparation.capacity_bytes())
    {
        return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch));
    }
    let semantic = lane.take_semantic();
    let state = semantic
        .prepared_source()
        .and_then(|source| {
            source.downcast_ref::<eredu_runtime::working_memory::PreparedSemanticState>()
        })
        .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::UnknownBound))?;
    if !state.preparation().same_preparation(preparation) {
        return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch));
    }
    let mut prompt = lane.take_prompt();
    // Borrow the actual completed input owner for either text or media. The
    // selected prefill worker prepares any numerical copy later; a text-only
    // copy descriptor is not the source contract for a media packet.
    prompt
        .original_prediction_source(sources.pool())
        .map_err(|cause| sources.retain_startup_error(
            Error::PrefillControl(cause).at_speculative_stage("lane original prompt source"),
        ))?;
    if let Some(chunk) = generation.inference_policy().prefill_chunk_positions {
        prompt = prompt.with_prefill_chunk_positions(chunk);
    }
    let sampler = MlxTextSampler::from_config(lane.take_generation())
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let constraint = lane.take_constraint();
    state
        .validate_pool(sources.pool())
        .map_err(|cause| sources.retain_startup_error(cause))?;
    state
        .validate_controller(&constraint)
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let sampler = ConstrainedSampler::new(sampler, constraint);
    let prng_key = if generation.sampling().temperature == 0.0 {
        None
    } else {
        Some(<MlxSpeculativeSampling<
            MlxPreparedSampler<C>, Error,
            IndependentLogits,
        > as SpeculativeSampling>::control_seed(generation.seed(), streams)
            .map_err(|cause| match cause {
                eredu_core::speculative::SpeculativeControlError::Backend(cause) => Error::StorageSource(cause),
                cause => sources.retain_startup_error(cause),
            })?)
    };
    Ok(MlxSpeculativeLaneRuntime {
        input: prompt,
        config: lane.take_config(),
        prng_key,
        sampler,
        semantic,
        cancellation: lane.take_cancellation(),
        on_event: lane.take_on_event(),
        memory_owner: None,
    })
}

/// Run the existing driver with one paid lane/cache buffer for either exact
/// model schedule. The typed executor retains its own cache and occurrence
/// authority; this common lane preparation never fabricates a second schedule.
pub(super) fn prepare_lane_buffers<'run, 'lane: 'run, 'world, 'streams: 'run, C, E>(
    executor: &'run mut E,
    lane: SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>,
    preparation: &eredu_runtime::working_memory::PreparedSemanticSource,
    streams: SpeculativeExecutionStreams<'streams>,
    new_cache: impl FnOnce(&mut E, SpeculativeExecutionStreams<'streams>) -> Result<E::Cache, Error>,
) -> Result<
    (
        eredu_core::SpeculativeBuffer<MlxSpeculativeLaneRuntime<'lane, C>>,
        eredu_core::SpeculativeBuffer<E::Cache>,
    ),
    Error,
>
where
    C: SpeculativeTokenFilterController + 'run,
    E: MlxSpeculativeRuntime<'run, Logits = IndependentLogits, Error = Error>,
{
    let (sources, _) = streams
        .original_numerical()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    // These are this caller's concrete moves and result transports. The shared
    // driver_buffer and cache/startup producers still pay their own allocations;
    // no preexisting caller payload or second buffer allowance is added here.
    let controls = [
        size_of::<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>(),
        size_of::<MlxSpeculativeLaneRuntime<'lane, C>>(),
        size_of::<Result<MlxSpeculativeLaneRuntime<'lane, C>, Error>>(),
        size_of::<eredu_core::SpeculativeBuffer<MlxSpeculativeLaneRuntime<'lane, C>>>(),
        size_of::<Result<eredu_core::SpeculativeBuffer<MlxSpeculativeLaneRuntime<'lane, C>>, Error>>(
        ),
        size_of::<E::Cache>(),
        size_of::<Result<E::Cache, Error>>(),
        size_of::<eredu_core::SpeculativeBuffer<E::Cache>>(),
        size_of::<Result<eredu_core::SpeculativeBuffer<E::Cache>, Error>>(),
        size_of::<Result<(), eredu_core::GenerationError>>(),
        size_of::<SpeculativeGenerationBatchOutput>(),
        size_of::<Result<SpeculativeGenerationBatchOutput, Error>>(),
        size_of::<(
            eredu_core::SpeculativeBuffer<MlxSpeculativeLaneRuntime<'lane, C>>,
            eredu_core::SpeculativeBuffer<E::Cache>,
        )>(),
        size_of_val(&new_cache),
        size_of::<SpeculativeExecutionStreams<'streams>>(),
        size_of::<(
            &mut E,
            &eredu_runtime::working_memory::PreparedSemanticSource,
        )>(),
        size_of::<
            Option<(
                &super::super::speculative::OriginalSpeculativeNumericalSources,
                &crate::backend::OriginalCopyEnvironment<'_>,
            )>,
        >(),
        size_of::<Result<(), HostMetadataFundingError>>(),
    ];
    sources
        .metadata_funding()
        .reserve_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(
                    HostMetadataFundingError::Overflow,
                ))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    // Any partially built lane/cache retires within this closure while the
    // source account is still borrowed. An escaping cause keeps that account.
    let result = (|| {
        let prepared = prepare_lane(lane, preparation, streams)?;
        let mut lanes = executor.driver_buffer(1, streams)?;
        lanes
            .try_push(prepared)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let cache = new_cache(executor, streams)?;
        let mut caches = executor.driver_buffer(1, streams)?;
        caches
            .try_push(cache)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        Ok((lanes, caches))
    })();
    result.map_err(|cause| sources.retain_error(cause))
}

/// The concrete executor preserves its short context lifetime while the common
/// paid buffers stay owned by this caller through the shared driver.
pub(super) fn run_lane<'lane, 'world, C, V>(
    executor: &mut AutoregressiveExecutor<'_, MlxAutoregressiveMechanisms>,
    lane: SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>,
    preparation: &eredu_runtime::working_memory::PreparedSemanticSource,
    streams: SpeculativeExecutionStreams<'_>,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, Error>
where
    C: SpeculativeTokenFilterController,
    V: SpeculativeGenerationVisitor,
{
    let (lanes, mut caches) =
        prepare_lane_buffers(executor, lane, preparation, streams, |executor, streams| {
            executor.new_cache(streams)
        })?;
    run_speculative_batch(executor, lanes, &mut caches, Ok, streams, visitor)
}

/// Source-explicit original requests enter before ordinary domain acquisition.
/// The same selected schedule and shared driver handle their actual work.
pub(super) fn run_request<'lane, 'world, C, V>(
    runtime: &mut ModelRuntime<MlxBackend<'world>>,
    drafting: SpeculativeDraft<'_, MlxDrafter>,
    lanes: eredu_core::SpeculativeBuffer<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, Error>
where
    C: SpeculativeTokenFilterController,
    V: SpeculativeGenerationVisitor,
{
    let options = visitor
        .scheduler_options()
        .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
    let (backend, session) = runtime.parts_mut();
    match drafting {
        SpeculativeDraft::Embedded => {
            embedded::run_batch(backend, session, lanes, options, visitor)
        }
        SpeculativeDraft::External(drafter) if drafter.is_autoregressive() => {
            independent::run_batch(backend, session, drafter, lanes, options, visitor)
        }
        SpeculativeDraft::External(drafter) => {
            external::run_batch(backend, session, drafter, lanes, options, visitor)
        }
        _ => Err(Error::PrefillControl(WorkingMemoryError::UnknownBound)),
    }
}
