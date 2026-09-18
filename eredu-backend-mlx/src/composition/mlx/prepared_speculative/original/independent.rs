//! Per-lane actual independent sources above the existing executor/table.
use super::*;
use super::batch::{buffer, inspect};
use crate::backend::OriginalCopyEnvironment;
use crate::composition::mlx::session::{MlxModelSession, OriginalInterventionDeclaration};
use eredu_core::SpeculativeBuffer;
use eredu_runtime::working_memory::PreparedSemanticSource;

struct Lane<'lane, 'world, C: SpeculativeTokenFilterController> {
    lane: SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>,
    input_positions: NonZeroU64,
    context_positions: NonZeroU64,
    declaration: Option<OriginalInterventionDeclaration>,
    // Retires after this lane's actual source/header-bearing payloads.
    preparation: PreparedSemanticSource,
}
struct Bound {
    sources: AutoregressiveSourcePair,
    preparation: PreparedSemanticSource,
}

pub(super) fn run_batch<'lane, 'world, C, V>(
    backend: &MlxBackend<'_>, session: &mut MlxModelSession, drafter: &mut MlxDrafter,
    lanes: SpeculativeBuffer<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>,
    options: eredu_core::SpeculativeSchedulerOptions, visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, Error>
where C: SpeculativeTokenFilterController, V: SpeculativeGenerationVisitor {
    let first = lanes.first().ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let (first, _) = inspect(backend, session, first)?;
    let funding = first.metadata_funding().clone();
    let controls = [size_of::<SpeculativeBuffer<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>>(),
        size_of::<eredu_core::SpeculativeBufferIntoIter<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>>(),
        size_of::<SpeculativeBuffer<Lane<'lane, 'world, C>>>(),
        size_of::<eredu_core::SpeculativeBufferIntoIter<Lane<'lane, 'world, C>>>(),
        size_of::<PreparedSemanticSource>(), size_of::<HostMetadataFunding>(),
        size_of::<SpeculativeDraft<'_, MlxDrafter>>(), size_of::<V>(),
        size_of::<(eredu_core::SpeculativeSchedulerOptions, usize)>(),
        size_of::<(&MlxBackend<'_>, &mut MlxModelSession, &mut MlxDrafter)>(),
        size_of::<Result<SpeculativeGenerationBatchOutput, Error>>(),
        OriginalCopyEnvironment::control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
    ];
    funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
    let run = || {
        // Qualify every actual source before either model is lent for work.
        for lane in &lanes { inspect(backend, session, lane)?; }
        drafter.autoregressive_source().map_err(Error::PrefillControl)?;
        // These are the same applicable transport checks as the controlled
        // single-request producer; batching supplies no substitute transport.
        if drafter.stream() != backend.stream()
            || drafter.topology() != eredu_core::SpeculativeExecutionTopology::Single {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        let capacity = drafter.selected().requirements().strategy().proposal_capacity();
        let environment = backend.original_copy_environment()
            .map_err(|cause| super::super::super::model::retain_planning_error(cause, funding.clone()))?;
        let count = lanes.len();
        let mut prepared = buffer(count, &funding)?;
        for lane in lanes {
            let (preparation, input_positions) = inspect(backend, session, &lane)?;
            let local = preparation.metadata_funding();
            let controls = [size_of::<Lane<'lane, 'world, C>>(),
                size_of::<eredu_core::speculative::SpeculativeRequestGeometry>(),
                size_of::<Result<Lane<'lane, 'world, C>, Error>>(),
                size_of::<(NonZeroU64, NonZeroU64)>(),
                size_of::<Option<OriginalInterventionDeclaration>>(),
                size_of::<Result<Option<OriginalInterventionDeclaration>, Error>>(),
            ];
            local.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
            let geometry = eredu_core::speculative::SpeculativeRequestGeometry::new(lane.config(), capacity.get());
            let context_positions = u64::try_from(geometry.output_positions()).ok()
                .and_then(|output| input_positions.get().checked_add(output)).and_then(NonZeroU64::new)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
            let declaration = session.original_intervention_declaration(local)?;
            prepared.try_push(Lane { lane, input_positions, context_positions, declaration, preparation })
                .map_err(|cause| super::super::super::model::retain_planning_error(cause, funding.clone()))?;
        }
        let with_target = |target: &mut crate::composition::mlx::model::Executable| {
            let with_draft = |draft: &mut crate::backend::MlxModel, selected: &eredu_runtime::SelectedSpeculativeRealization| {
                if selected.requirements().strategy().proposal_capacity() != capacity {
                    return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                }
                let controls = [size_of::<AutoregressiveExecutor<'_, MlxAutoregressiveMechanisms>>(),
                    size_of::<Result<AutoregressiveExecutor<'_, MlxAutoregressiveMechanisms>, Error>>(),
                    size_of::<SpeculativeBuffer<AutoregressiveSchedulePlan<'_>>>(),
                    size_of::<SpeculativeBuffer<Bound>>(),
                    size_of::<SpeculativeBuffer<SpeculativeExecutionStreams<'_>>>(),
                    size_of::<SpeculativeBuffer<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>>(),
                    size_of::<Result<(), Error>>(), size_of::<Result<SpeculativeGenerationBatchOutput, Error>>(),
                ];
                funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
                let mut plans = buffer(count, &funding)?;
                let mut sources = buffer(count, &funding)?;
                let mut lanes = buffer(count, &funding)?;
                for value in prepared {
                    let local = value.preparation.metadata_funding();
                    let controls = [size_of::<Bound>(), size_of::<AutoregressiveSchedulePlan<'_>>(),
                        size_of::<Result<AutoregressiveSchedulePlan<'_>, eredu_runtime::speculative::autoregressive::AutoregressiveOccurrenceError>>(),
                        size_of::<Result<Bound, Error>>(), size_of::<SpeculativeExecutionStreams<'_>>(),
                    ];
                    local.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
                        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
                    let schedule = AutoregressiveSchedulePlan::new(selected, capacity,
                        value.input_positions, value.context_positions, value.lane.config(), options)
                        .map_err(|cause| super::super::super::model::retain_planning_error(cause, local.clone()))?;
                    local.reserve_metadata(schedule.control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
                        .map_err(Error::WorkspacePlanning)?;
                    let pair = AutoregressiveSourcePair::prepare_funded(target, draft.executable(), &schedule,
                        backend.memory_pool(), value.preparation.capacity_bytes(), local.clone())?
                        .with_intervention_declaration(value.declaration)?;
                    sources.try_push(Bound { sources: pair, preparation: value.preparation })
                        .map_err(|cause| super::super::super::model::retain_planning_error(cause, funding.clone()))?;
                    plans.try_push(schedule).map_err(|cause| super::super::super::model::retain_planning_error(cause, funding.clone()))?;
                    lanes.try_push(value.lane).map_err(|cause| super::super::super::model::retain_planning_error(cause, funding.clone()))?;
                }
                let invoke = || {
                    let mut assignments = buffer(count, &funding)?;
                    for source in &sources {
                        let streams = SpeculativeExecutionStreams::single(environment.stream())
                            .with_original_sources(&source.sources, &environment)?;
                        assignments.try_push(streams)
                            .map_err(|cause| source.sources.retain_startup_error(cause))?;
                    }
                    let outer = assignments[0].with_request_assignments(&assignments)?;
                    let mut executor = AutoregressiveExecutor::<MlxAutoregressiveMechanisms>::new(
                        target, draft.executable_mut(), capacity).with_schedules(plans, outer)?;
                    let mut runtime_lanes = executor.driver_buffer(count, outer)?;
                    let mut caches = executor.driver_buffer(count, outer)?;
                    for (index, (lane, source)) in lanes.into_iter().zip(sources.iter()).enumerate() {
                        let context = outer.request_context(eredu_core::SpeculativeRequestId::new(index))?;
                        let (one_lane, one_cache) = prepare_lane_buffers(&mut executor, lane, &source.preparation, context,
                            |executor, context| executor.new_cache(context))?;
                        runtime_lanes.try_extend(one_lane).map_err(|cause| source.sources.retain_startup_error(cause))?;
                        caches.try_extend(one_cache).map_err(|cause| source.sources.retain_startup_error(cause))?;
                    }
                    run_speculative_batch(&mut executor, runtime_lanes, &mut caches, Ok, outer, visitor)
                };
                funding.reserve_metadata(size_of_val(&invoke)).map_err(Error::WorkspacePlanning)?;
                let result = invoke();
                // Closing issuance is separate from all escaped native and host
                // payload custody. A late lane error must close every source.
                let mut closed = Ok(());
                for source in &sources {
                    if let Err(cause) = source.sources.request().close() {
                        if closed.is_ok() { closed = Err(source.sources.retain_startup_error(cause)); }
                    }
                }
                match result { Err(cause) => Err(cause), Ok(value) => closed.map(|()| value) }
            };
            funding.reserve_metadata(size_of_val(&with_draft)).map_err(Error::WorkspacePlanning)?;
            drafter.with_autoregressive_funded(funding.clone(), with_draft)
        };
        funding.reserve_metadata(size_of_val(&with_target)).map_err(Error::WorkspacePlanning)?;
        session.with_model_operation_funded(funding.clone(), with_target)
    };
    funding.reserve_metadata(size_of_val(&run)).map_err(Error::WorkspacePlanning)?;
    let result = run();
    result.map_err(|cause: Error| super::super::super::model::retain_planning_error(cause, funding.clone()))
}
