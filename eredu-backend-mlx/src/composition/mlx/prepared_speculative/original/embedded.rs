//! Exact Embedded lane sources above the same typed executor and fair driver.
use super::*;
use crate::backend::OriginalCopyEnvironment;
use crate::composition::mlx::{
    replicated_text::{OriginalEmbeddedCachePreparation, OriginalPredictionLane, OriginalPredictionTarget},
    session::{MlxModelSession, OriginalInterventionDeclaration},
    speculative::{OriginalEmbeddedSources, OriginalSpeculativeNumericalPreparation},
};
use eredu_runtime::{prefill::PrefillControlPlan,
    speculative::embedded_occurrence::EmbeddedSchedulePlan,
    working_memory::OriginalSpeculativeSemanticPreparation};
use eredu_core::SpeculativeBuffer;
use super::batch::{buffer, inspect};


struct Lane<'lane, 'world, C: SpeculativeTokenFilterController> {
    lane: SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>,
    prediction: OriginalPredictionLane,
    target: OriginalPredictionTarget,
    cached_positions: u64,
    output_positions: u64,
    context: u64,
    input_positions: NonZeroU64,
    chunk: u64,
    source: OriginalSpeculativeNumericalPreparation,
    declaration: Option<OriginalInterventionDeclaration>,
    preparation: OriginalSpeculativeSemanticPreparation,
}
struct Bound<'selected> {
    cache: OriginalEmbeddedCachePreparation,
    sources: OriginalEmbeddedSources<'selected>,
    preparation: OriginalSpeculativeSemanticPreparation,
}


/// Single and multi-lane requests use the same source/copy constructors. Native
/// model execution begins only after every independent lane has been prepared.
pub(super) fn run_batch<'lane, 'world, C, V>(backend: &MlxBackend<'_>, session: &mut MlxModelSession,
    lanes: SpeculativeBuffer<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>,
    options: eredu_core::SpeculativeSchedulerOptions, visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, Error>
where C: SpeculativeTokenFilterController, V: SpeculativeGenerationVisitor {
    let first = lanes.first().ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let (first, _) = inspect(backend, session, first)?;
    let funding = first.metadata_funding().clone();
    let controls = [
        size_of::<Continuation<'_, 'lane, 'world, C, V>>(),
        size_of::<SpeculativeBuffer<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>>(),
        size_of::<eredu_core::SpeculativeBufferIntoIter<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>>(),
        size_of::<OriginalSpeculativeSemanticPreparation>(),
        size_of::<Result<(OriginalSpeculativeSemanticPreparation, NonZeroU64), Error>>(),
        size_of::<Result<SpeculativeGenerationBatchOutput, Error>>(),
        OriginalCopyEnvironment::control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
    ];
    funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
    let result = (|| {
        // Validate the complete source batch before making independent state copies.
        for lane in &lanes { inspect(backend, session, lane)?; }
        let environment = backend.original_copy_environment()
            .map_err(|cause| super::super::super::model::retain_planning_error(cause, funding.clone()))?;
        let mut prepared = buffer(lanes.len(), &funding)?;
        for lane in lanes {
            let (preparation, input_positions) = inspect(backend, session, &lane)?;
            let value = prepare(backend, session, lane, preparation, input_positions, &environment)?;
            prepared.try_push(value).map_err(|cause|
                super::super::super::model::retain_planning_error(cause, funding.clone()))?;
        }
        let mut continuation = Continuation { lanes: Some(prepared), funding: funding.clone(),
            environment: &environment, options, visitor: Some(visitor) };
        session.with_model_operation_funded(funding.clone(), |target|
            target.erased_mut().with_embedded_prediction(&mut continuation)
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?)
    })();
    result.map_err(|cause| super::super::super::model::retain_planning_error(cause, funding.clone()))
}

fn prepare<'lane, 'world, C: SpeculativeTokenFilterController>(backend: &MlxBackend<'_>, session: &MlxModelSession,
    lane: SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>, preparation: OriginalSpeculativeSemanticPreparation,
    input_positions: NonZeroU64, environment: &OriginalCopyEnvironment<'_>,
) -> Result<Lane<'lane, 'world, C>, Error> {
    let funding = preparation.metadata_funding();
    let controls = [size_of::<Lane<'lane, 'world, C>>(), size_of::<Result<Lane<'lane, 'world, C>, Error>>(),
        size_of::<OriginalPredictionLane>(), size_of::<Option<OriginalPredictionLane>>(),
        size_of::<OriginalPredictionTarget>(), size_of::<Option<OriginalPredictionTarget>>(),
        size_of::<eredu_core::speculative::SpeculativeRequestGeometry>(), size_of::<(u64,u64,u64)>(),
        size_of::<OriginalSpeculativeNumericalPreparation>(), size_of::<Option<OriginalInterventionDeclaration>>(),
        size_of::<(&MlxBackend<'_>, &MlxModelSession, &OriginalCopyEnvironment<'_>, NonZeroU64)>(),
        size_of::<(OriginalPredictionLane, OriginalPredictionTarget, u64, u64, u64,
            OriginalSpeculativeNumericalPreparation, Option<OriginalInterventionDeclaration>, u64)>(),
        size_of::<Result<(OriginalPredictionLane, OriginalPredictionTarget, u64, u64, u64,
            OriginalSpeculativeNumericalPreparation, Option<OriginalInterventionDeclaration>, u64), Error>>(),
    ];
    funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
    let result = (|| {
        let model = session.original_model_source().map_err(Error::PrefillControl)?;
        preparation.validate(backend.memory_pool(), model.erased().inference_execution_identity()).map_err(Error::PrefillControl)?;
        let frontier = model.erased().original_text_frontier()
            .map_err(|cause| super::super::super::model::retain_planning_error(cause, funding.clone()))?
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let source = OriginalSpeculativeNumericalPreparation::prepare(model, backend.memory_pool(), funding.clone())?;
        let prediction = session.prepare_original_prediction_lane(&preparation, environment)?
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let geometry = eredu_core::speculative::SpeculativeRequestGeometry::new(lane.config(), prediction.proposal_capacity());
        let output_positions = u64::try_from(geometry.output_positions()).map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let context = frontier.checked_add(input_positions.get()).and_then(|n| n.checked_add(output_positions))
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let target = session.prepare_original_prediction_target(&prediction, &preparation, environment, frontier)?;
        let declaration = session.original_intervention_declaration(funding)?;
        let chunk = lane.prompt().with_borrowed(|input| input.prefill_chunk_positions()).map(NonZeroU64::get)
            .unwrap_or(eredu_runtime::prefill::DEFAULT_PREFILL_CHUNK_POSITIONS).min(input_positions.get());
        Ok((prediction, target, frontier, output_positions, context, source, declaration, chunk))
    })().map_err(|cause: Error| super::super::super::model::retain_planning_error(cause, funding.clone()))?;
    let (prediction,target,cached_positions,output_positions,context,source,declaration,chunk) = result;
    Ok(Lane { lane,prediction,target,cached_positions,output_positions,context,input_positions,chunk,source,declaration,preparation })
}

struct Continuation<'request, 'lane, 'world, C: SpeculativeTokenFilterController, V> {
    lanes: Option<SpeculativeBuffer<Lane<'lane, 'world, C>>>,
    environment: &'request OriginalCopyEnvironment<'request>,
    options: eredu_core::SpeculativeSchedulerOptions,
    visitor: Option<V>,
    // Retires after all provisional lane and visitor transport fields.
    funding: WorkspaceMetadataFunding,
}
impl<C: SpeculativeTokenFilterController, V: SpeculativeGenerationVisitor> MlxEmbeddedExecutorContinuation
    for Continuation<'_, '_, '_, C, V> {
    fn construction_controls(&self, bytes: Option<usize>) -> Result<(), Error> {
        self.funding.reserve_metadata(bytes.ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)
    }
    fn execute(&mut self, selected: &eredu_runtime::SelectedSpeculativeRealization,
        executor: &mut DynEmbeddedExecutor<'_, MlxEmbeddedExecutorTypes>,
    ) -> Result<SpeculativeGenerationBatchOutput, Error> {
        let parts = [size_of::<Bound<'_>>(), size_of::<EmbeddedSchedulePlan<'_>>(), size_of::<PrefillControlPlan>(),
            size_of::<eredu_runtime::replicated_session::PrefillScoreLayout>(),
            size_of::<eredu_core::InferenceGeometry>(), size_of::<SpeculativeExecutionStreams<'_>>(),
            size_of::<Result<SpeculativeGenerationBatchOutput, Error>>(),
        ];
        self.construction_controls(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add))?;
        let prepared = self.lanes.take().ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let count = prepared.len();
        let mut sources = buffer(count, &self.funding)?;
        let mut lanes = buffer(count, &self.funding)?;
        for value in prepared {
            if value.target.frontier() != value.cached_positions { return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)); }
            let prefill = PrefillControlPlan::new(eredu_core::InferenceGeometry { batch_size:1,
                cached_positions:value.cached_positions, input_positions:value.input_positions.get(),
                max_output_tokens:value.output_positions, prefill_chunk_positions:value.chunk,
                output:eredu_core::OutputDemand::LastPosition }, false).map_err(Error::PrefillControl)?;
            let schedule = EmbeddedSchedulePlan::new(selected, value.prediction.occurrence_shape(),
                value.prediction.prefill_alignment(), value.prediction.initial_frontier(), prefill,
                value.context, value.lane.config(), self.options)
                .map_err(|cause| super::super::super::model::retain_planning_error(cause, value.preparation.metadata_funding().clone()))?;
            let source = OriginalEmbeddedSources::prepare_from_source(value.source, schedule, value.preparation.capacity_bytes())?
                .with_intervention_declaration(value.declaration)?;
            let cache = OriginalEmbeddedCachePreparation::new(value.prediction, value.target, source.numerical_sources())
                .map_err(Error::StorageSource)?;
            sources.try_push(Bound { cache, sources:source, preparation:value.preparation })
                .map_err(|cause| super::super::super::model::retain_planning_error(cause,self.funding.clone()))?;
            lanes.try_push(value.lane).map_err(|cause|
                super::super::super::model::retain_planning_error(cause,self.funding.clone()))?;
        }
        let result = (|| {
            let mut assignments = buffer(count, &self.funding)?;
            for source in &sources {
                let streams = SpeculativeExecutionStreams::single(self.environment.stream())
                    .with_original_embedded_sources(&source.sources, self.environment)?
                    .with_original_cache_preparation(&source.cache).map_err(Error::StorageSource)?;
                assignments.try_push(streams).map_err(|cause|
                    super::super::super::model::retain_planning_error(cause,self.funding.clone()))?;
            }
            let outer = assignments[0].with_request_assignments(&assignments)?;
            let mut runtime_lanes = executor.driver_buffer(count, outer)?;
            let mut caches = executor.driver_buffer(count, outer)?;
            for (index, (lane, source)) in lanes.into_iter().zip(sources.iter()).enumerate() {
                let streams = outer.request_context(eredu_core::SpeculativeRequestId::new(index))?;
                let (one_lane, one_cache) = prepare_lane_buffers(executor, lane, &source.preparation, streams,
                    |executor, streams| executor.new_cache_with_context(streams).map_err(Error::StorageSource))?;
                runtime_lanes.try_extend(one_lane).map_err(|cause| source.sources.numerical_sources().retain_startup_error(cause))?;
                caches.try_extend(one_cache).map_err(|cause| source.sources.numerical_sources().retain_startup_error(cause))?;
            }
            let visitor = self.visitor.take().ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            run_speculative_batch(executor, runtime_lanes, &mut caches, Ok, outer, visitor)
        })();
        // Every finite request closes issuance even if a later lane fails. The
        // arrays/results retain their own completed or unresolved custody.
        let mut closed = Ok(());
        for source in &sources {
            if let Err(cause) = source.sources.numerical_sources().request().close() {
                if closed.is_ok() { closed = Err(source.sources.numerical_sources().retain_startup_error(cause)); }
            }
        }
        match result { Err(cause) => Err(cause), Ok(value) => closed.map(|()| value) }
    }
}
