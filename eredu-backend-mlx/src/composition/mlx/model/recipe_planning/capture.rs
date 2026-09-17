//! The existing capture observer and native recorder consume the same equation.
use super::*;
use crate::backend::array_copy::CaptureNativePopulation;
use eredu_architectures::prepared_execution::{
    InferenceEquationTraceObserver, PreparedExecutionError,
};
use eredu_nn::workspace::{WorkspaceStoragePopulation, WorkspaceTraceReport, WorkspaceFloatingType};
use eredu_runtime::working_memory::{InferenceWorkspaceSpan, SamplingWorkspacePhase};
use std::cell::Cell;
use crate::backend::nn::workspace::ParallelRecipeRecorder;

type State = eredu_runtime::DeviceState<
    eredu_nn::workspace::WorkspaceBackend,
    eredu_runtime::working_memory::WorkspaceResidentLayerState,
>;

pub(super) fn validate(
    executable: &Executable,
    geometry: InferenceGeometry,
    bound: BoundCaptureSelection<'_>,
    context: &WorkspaceContext,
    placement: Option<(&eredu_architectures::component_partition::ComponentPartitionLayouts, usize)>,
) -> Result<(), eredu_nn::Error> {
    if bound.geometry() != geometry {
        return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
    }
    validate_selection(executable, bound.selection(), context, placement)
}

fn validate_selection(
    executable: &Executable,
    selection: &eredu_runtime::layered::PreparedCaptureSelection,
    context: &WorkspaceContext,
    placement: Option<(&eredu_architectures::component_partition::ComponentPartitionLayouts, usize)>,
) -> Result<(), eredu_nn::Error> {
    let paths = executable
        .erased()
        .shared_observation_paths()
        .ok_or_else(|| context.metadata_source(WorkingMemoryError::UnknownBound))?;
    selection
        .validate_sources(selection.source(), paths)
        .map_err(|cause| context.metadata_source(cause))?;
    // The retained declaration owner authenticates the existing callback. Routed
    // resident callbacks use the five-source observer and native carrier; their
    // providers are checked at the actual invocation. Partitioned callbacks still
    // require their distinct contribution/transport recipe.
    for (index, entry) in selection.source().admission().plan().selections.iter().enumerate() {
        let routed = placement.is_none()
            && matches!(entry.transform, eredu_core::capture::CaptureTransform::RoutedUnits)
            && matches!(selection.source().admission().points()[index].value_type,
                eredu_core::ObservationValueType::RoutedUnits { .. });
        if routed {
            // Revalidate the architecture declaration for every scheduled prefill
            // hook; decode-only selections retain the admitted invocation source.
            selection.declaration(index).map_err(|cause| context.metadata_source(cause))?;
            continue;
        }
        let selection = entry;
        if let Some(placement)=placement {
            crate::composition::mlx::session::capture_workspace::validate_partition_capture_source(
                selection,placement,context)?;
            continue;
        }
        if selection.path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH
            || !matches!(
                selection.transform,
                eredu_core::capture::CaptureTransform::FullTensor
                    | eredu_core::capture::CaptureTransform::Slice
                    | eredu_core::capture::CaptureTransform::Preview { .. }
                    | eredu_core::capture::CaptureTransform::TokenScores { .. }
                    | eredu_core::capture::CaptureTransform::Summary
                    | eredu_core::capture::CaptureTransform::Histogram { .. }
                    | eredu_core::capture::CaptureTransform::TopCandidates { .. }
            )
        {
            return Err(context.metadata_source(WorkingMemoryError::UnknownBound));
        }
    }
    Ok(())
}

pub(super) fn quote(
    blueprint: &PreparedInferenceBlueprint,
    geometry: InferenceGeometry,
    state: &State,
    context: &WorkspaceContext,
    config: TextGenerationConfig,
    filter: TextFilterWorkspace<'_>,
    bound: BoundCaptureSelection<'_>,
    interventions: Option<TextInterventionQuote<'_>>,
    recorder: &mut ResidentRecipeRecorder,
    parameters: Option<&dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>,
) -> Result<
    PreparedTextGenerationWorkspace,
    eredu_architectures::prepared_execution::PreparedExecutionError<eredu_nn::Error>,
> {
    with_capture_trace(
        bound, interventions, context, recorder,
        |cause| PreparedExecutionError::Metadata(cause.into()),
        PreparedExecutionError::Backend,
        |observer, trace| blueprint.quote_replicated_text_with_sampling_observed_and_trace(
            geometry, state, context, config, filter, parameters,
            bound.selection().paths(), observer, trace,
        ),
    )
}

// Both original media and token input use the same span-aware capture observer
// and native recorder adapter. The operation census is emitted by the actual
// observed equations, including generated tensors and completed output roots.
fn with_capture_trace<T, E>(
    bound: BoundCaptureSelection<'_>,
    interventions: Option<TextInterventionQuote<'_>>,
    context: &WorkspaceContext,
    recorder: &mut ResidentRecipeRecorder,
    metadata_error: fn(eredu_nn::workspace::WorkspaceMetadataError) -> E,
    neural_error: fn(eredu_nn::Error) -> E,
    worker: impl FnOnce(
        &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
        &mut dyn InferenceEquationTraceObserver,
    ) -> Result<T, E>,
) -> Result<T, E> {
    let controls = size_of::<(
        Cell<CaptureNativePopulation>, Trace<'_, ResidentRecipeRecorder>,
        crate::composition::mlx::session::capture_workspace::CaptureWorkspaceObserver<'_>,
        eredu_runtime::working_memory::CaptureRunHostPlan<'_>,
        BoundCaptureSelection<'_>, Option<TextInterventionQuote<'_>>, &WorkspaceContext, &mut ResidentRecipeRecorder,
        fn(eredu_nn::workspace::WorkspaceMetadataError) -> E,
        fn(eredu_nn::Error) -> E, Result<T, E>,
    )>().checked_add(size_of_val(&worker))
        .ok_or_else(|| metadata_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
    context.charge_metadata(controls).map_err(metadata_error)?;
    let transfers = Cell::new(CaptureNativePopulation::default());
    let (mut observer, host) =
        crate::composition::mlx::session::capture_workspace::CaptureWorkspaceObserver::with_prefill_transfers(
            bound, context, &transfers,
        ).map_err(neural_error)?;
    if !host.source().same_storage(bound.selection().source()) {
        return Err(neural_error(context.metadata_source(WorkingMemoryError::IdentityMismatch)));
    }
    if let Some(interventions) = interventions {
        interventions.prepare(context).map_err(neural_error)?;
        observer = observer.with_text_interventions(interventions.rows).map_err(neural_error)?;
    }
    let mut trace = Trace { recorder, transfers: &transfers, scalar: None };
    worker(&mut observer, &mut trace)
}

impl Executable {
    /// Reduces the authenticated prepared media input with the same native
    /// observation recorder used by original token input. Source publication
    /// and completed capture destinations stay with the caller's quote owner.
    pub(in crate::composition::mlx) fn quote_original_media_capture(
        &self,
        input: eredu_architectures::prepared_execution::OriginalMediaWorkspaceInput,
        current: &eredu_runtime::working_memory::MediaSessionBinding,
        geometry: InferenceGeometry,
        state: &State,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: TextFilterWorkspace<'_>,
        bound: BoundCaptureSelection<'_>,
        interventions: Option<TextInterventionQuote<'_>>,
        recorder: &mut ResidentRecipeRecorder,
    ) -> Result<eredu_architectures::prepared_execution::OriginalMediaWorkspaceReport, Error> {
        validate(self, geometry, bound, context, None).map_err(Error::Neural)?;
        let blueprint = self.inference_blueprint().ok_or_else(||
            Error::Neural(context.metadata_source(WorkingMemoryError::UnknownBound)))?;
        with_capture_trace(bound, interventions, context, recorder,
            |cause| Error::Neural(cause.into()), Error::Neural,
            |observer, trace| blueprint.quote_original_media_with_sampling_observed_and_trace(
                input, current, geometry, state, context, None, config, filter,
                bound.selection().paths(), observer, trace,
            ).map_err(|cause| Error::Neural(context.metadata_source(cause.into_failure()))),
        )
    }
}

/// Prepared partition source calls this worker only after the enclosing native
/// receipt owner is qualified. It shares the exact capture observer/recorder
/// adapter with resident execution; this function grants no transport itself.
#[allow(clippy::too_many_arguments)]
pub(super) fn quote_partitioned(
    blueprint:&PreparedInferenceBlueprint,geometry:InferenceGeometry,state:&State,
    context:&WorkspaceContext,config:TextGenerationConfig,filter:TextFilterWorkspace<'_>,
    bound:BoundCaptureSelection<'_>,interventions:Option<TextInterventionQuote<'_>>,recorder:&mut ParallelRecipeRecorder,
    communication:&eredu_runtime::RetainedCommunicationSource,
    placement:(&eredu_architectures::component_partition::ComponentPartitionLayouts,usize),
    parameters:Option<&dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>,
)->Result<PreparedTextGenerationWorkspace,PreparedExecutionError<eredu_nn::Error>> {
    context.charge_metadata(size_of::<(Cell<CaptureNativePopulation>,Option<TextInterventionQuote<'_>>,Vec<Cell<Option<WorkspaceFloatingType>>>,Trace<'_,ParallelRecipeRecorder>,
        crate::composition::mlx::session::capture_workspace::CaptureWorkspaceObserver<'_>,
        eredu_runtime::working_memory::CaptureRunHostPlan<'_>,
        Option<&dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>)>())
        .map_err(|cause|PreparedExecutionError::Metadata(cause.into()))?;
    validate_partition_selection(bound.selection(), placement, context).map_err(PreparedExecutionError::Backend)?;
    let transfers=Cell::new(CaptureNativePopulation::default());
    context.charge_metadata(size_of::<(usize, std::ops::Range<usize>, Cell<Option<WorkspaceFloatingType>>)>() )
        .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
    let count = bound.selection().source().admission().plan().selections.len();
    let mut scalar = context.metadata_vec::<Cell<Option<WorkspaceFloatingType>>>(count)
        .map_err(PreparedExecutionError::Backend)?;
    scalar.resize_with(count, || Cell::new(None));
    let (mut observer,host)=crate::composition::mlx::session::capture_workspace::CaptureWorkspaceObserver::with_prefill_source_transfers(
        bound,context,&transfers,&scalar,placement).map_err(PreparedExecutionError::Backend)?;
    if !host.source().same_storage(bound.selection().source()) {
        return Err(PreparedExecutionError::Backend(context.metadata_source(WorkingMemoryError::IdentityMismatch)));
    }
    if let Some(interventions) = interventions {
        interventions.prepare(context).map_err(PreparedExecutionError::Backend)?;
        observer = observer.with_partition_interventions(interventions.rows,communication)
            .map_err(PreparedExecutionError::Backend)?;
    }
    let mut trace=Trace{recorder,transfers:&transfers,scalar:Some(&scalar)};
    blueprint.quote_partitioned_text_with_sampling_observed_and_trace(geometry,state,context,config,filter,
        bound.selection().paths(),&mut observer,&mut trace,Some(communication),parameters)
}

fn validate_partition_selection(
    selection: &eredu_runtime::layered::PreparedCaptureSelection,
    placement: (&eredu_architectures::component_partition::ComponentPartitionLayouts, usize),
    context: &WorkspaceContext,
) -> Result<(), eredu_nn::Error> {
    for entry in &selection.source().admission().plan().selections {
        crate::composition::mlx::session::capture_workspace::validate_partition_capture_source(
            entry,placement,context)?;
    }
    Ok(())
}

pub(in crate::composition::mlx::model) fn quote_saved(
    executable: &Executable,
    blueprint: &PreparedInferenceBlueprint,
    geometry: InferenceGeometry,
    state: &State,
    context: &WorkspaceContext,
    sampling: eredu_architectures::prepared_execution::BorrowedTextSamplingWorkspace<'_>,
    checkpoint: &eredu_runtime::capture::FundedCaptureCheckpoint,
    selection: &eredu_runtime::layered::PreparedCaptureSelection,
    interventions: Option<TextInterventionQuote<'_>>,
    recorder: &mut ResidentRecipeRecorder,
    parameters: Option<&dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>,
) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<eredu_nn::Error>> {
    quote_saved_impl(executable, blueprint, geometry, state, context, sampling, checkpoint,
        selection, interventions, recorder, None, parameters)
}

type ParallelSavedCaptureSource<'a> = (
    &'a eredu_runtime::RetainedCommunicationSource,
    (&'a eredu_architectures::component_partition::ComponentPartitionLayouts, usize),
    Option<&'a dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>,
);

impl Executable {
    /// Reuses the ordinary saved capture observer under the exact retained
    /// partition source. The source tables remain borrowed from the session.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::composition::mlx) fn quote_partitioned_saved_capture(
        &self, blueprint: &PreparedInferenceBlueprint, geometry: InferenceGeometry,
        state: &State, context: &WorkspaceContext,
        sampling: eredu_architectures::prepared_execution::BorrowedTextSamplingWorkspace<'_>,
        checkpoint: &eredu_runtime::capture::FundedCaptureCheckpoint,
        selection: &eredu_runtime::layered::PreparedCaptureSelection,
        interventions: Option<TextInterventionQuote<'_>>,
        recorder: &mut ParallelRecipeRecorder,
        communication: &eredu_runtime::RetainedCommunicationSource,
        placement: (&eredu_architectures::component_partition::ComponentPartitionLayouts, usize),
        parameters: Option<&dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<eredu_nn::Error>> {
        quote_saved_impl(self, blueprint, geometry, state, context, sampling, checkpoint,
            selection, interventions, recorder, Some((communication, placement, parameters)), None)
    }
}

#[allow(clippy::too_many_arguments)]
fn quote_saved_impl<R: CaptureRecorder>(
    executable: &Executable, blueprint: &PreparedInferenceBlueprint,
    geometry: InferenceGeometry, state: &State, context: &WorkspaceContext,
    sampling: eredu_architectures::prepared_execution::BorrowedTextSamplingWorkspace<'_>,
    checkpoint: &eredu_runtime::capture::FundedCaptureCheckpoint,
    selection: &eredu_runtime::layered::PreparedCaptureSelection,
    interventions: Option<TextInterventionQuote<'_>>,
    recorder: &mut R, parallel: Option<ParallelSavedCaptureSource<'_>>,
    parameters: Option<&dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>,
) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<eredu_nn::Error>> {
    with_saved_capture_trace(executable, geometry, context, checkpoint, selection, interventions, recorder, parallel,
        |observer, trace| match parallel {
            Some((communication, _, parameters)) => blueprint.quote_partitioned_text_with_existing_sampling_observed_and_trace(
                geometry, state, context, sampling, selection.paths(), observer, trace,
                Some(communication), parameters,
            ),
            None => blueprint.quote_replicated_text_with_existing_sampling_observed_and_trace(
                geometry, state, context, sampling, parameters, selection.paths(), observer, trace,
            ),
        },
    )
}

// Shared checkpoint observer, physical transformations and recorder. Both
// sources keep the actual history and partial transfer population on failure.
fn with_saved_capture_trace<R: CaptureRecorder, T>(
    executable: &Executable,
    geometry: InferenceGeometry,
    context: &WorkspaceContext,
    checkpoint: &eredu_runtime::capture::FundedCaptureCheckpoint,
    selection: &eredu_runtime::layered::PreparedCaptureSelection,
    interventions: Option<TextInterventionQuote<'_>>,
    recorder: &mut R,
    parallel: Option<ParallelSavedCaptureSource<'_>>,
    worker: impl FnOnce(
        &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
        &mut dyn InferenceEquationTraceObserver,
    ) -> Result<T, PreparedExecutionError<eredu_nn::Error>>,
) -> Result<T, PreparedExecutionError<eredu_nn::Error>> {
    validate_selection(executable, selection, context, parallel.map(|source| source.1)).map_err(PreparedExecutionError::Backend)?;
    if !selection.source().same_storage(checkpoint.source())
        || selection.physical_output(geometry.output) != geometry.output
    {
        return Err(PreparedExecutionError::Backend(
            context.metadata_source(WorkingMemoryError::IdentityMismatch),
        ));
    }
    context.charge_metadata(size_of_val(&worker)
        .checked_add(size_of::<(
            Result<T, PreparedExecutionError<eredu_nn::Error>>,
            &Executable, InferenceGeometry, &WorkspaceContext,
            &eredu_runtime::capture::FundedCaptureCheckpoint,
            &eredu_runtime::layered::PreparedCaptureSelection, &mut R,
            Option<ParallelSavedCaptureSource<'_>>, Option<TextInterventionQuote<'_>>,
        )>()).ok_or_else(|| PreparedExecutionError::Metadata(
            eredu_nn::workspace::WorkspaceMetadataError::Overflow.into()))?)
        .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
    context
        .charge_metadata(size_of::<(
            Cell<CaptureNativePopulation>,
            Trace<'_,R>,
            Option<Vec<Cell<Option<WorkspaceFloatingType>>>>,
            Option<ParallelSavedCaptureSource<'_>>, Option<TextInterventionQuote<'_>>, usize, std::ops::Range<usize>,
            [(&Executable, &PreparedInferenceBlueprint, InferenceGeometry, &State, &WorkspaceContext,
                eredu_architectures::prepared_execution::BorrowedTextSamplingWorkspace<'_>,
                &eredu_runtime::capture::FundedCaptureCheckpoint,
                &eredu_runtime::layered::PreparedCaptureSelection, &mut R); 2],
            [Result<PreparedTextGenerationWorkspace, PreparedExecutionError<eredu_nn::Error>>; 2],
            crate::composition::mlx::session::capture_workspace::CaptureWorkspaceObserver<'_>,
            eredu_runtime::working_memory::CaptureRunHostPlan<'_>,
            Option<BoundCaptureSelection<'_>>,
        )>())
        .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
    // Only the original pending prompt uses row fragments. A saved decode token
    // is a whole physical prefill span at its later logical decode coordinate.
    let bound = if checkpoint.next_prediction() == 0 {
        Some(
            selection
                .bind_prompt_prefix(geometry)
                .map_err(|cause| PreparedExecutionError::Backend(context.metadata_source(cause)))?,
        )
    } else {
        None
    };
    if let Some((_, placement, _)) = parallel {
        validate_partition_selection(selection, placement, context).map_err(PreparedExecutionError::Backend)?;
    }
    let scalars = if parallel.is_some() {
        let count = selection.source().admission().plan().selections.len();
        let mut rows = context.metadata_vec::<Cell<Option<WorkspaceFloatingType>>>(count)
            .map_err(PreparedExecutionError::Backend)?;
        rows.resize_with(count, || Cell::new(None));
        Some(rows)
    } else { None };
    let transfers = Cell::new(CaptureNativePopulation::default());
    let (mut observer, host) = crate::composition::mlx::session::capture_workspace::CaptureWorkspaceObserver::with_checkpoint_transfers(
        checkpoint, geometry, bound, context, &transfers,
    ).map_err(PreparedExecutionError::Backend)?;
    if let Some((_, placement, _)) = parallel {
        observer.bind_partition_sources(scalars.as_deref().ok_or_else(||
            PreparedExecutionError::Backend(context.metadata_source(WorkingMemoryError::IdentityMismatch)))?, placement)
            .map_err(PreparedExecutionError::Backend)?;
    }
    if !host.source().same_storage(selection.source()) {
        return Err(PreparedExecutionError::Backend(
            context.metadata_source(WorkingMemoryError::IdentityMismatch),
        ));
    }
    match (checkpoint.intervention_source(), interventions) {
        (Some(source), Some(interventions)) if source.same_source(interventions.source) => {
            interventions.prepare_range(checkpoint.next_prediction(), geometry.max_output_tokens, context)
                .map_err(PreparedExecutionError::Backend)?;
            observer = match parallel {
                Some((communication,_,_)) => observer.with_partition_interventions(interventions.rows,communication),
                None => observer.with_text_interventions(interventions.rows),
            }.map_err(PreparedExecutionError::Backend)?;
        }
        (None, None) => {},
        _ => return Err(PreparedExecutionError::Backend(
            context.metadata_source(WorkingMemoryError::IdentityMismatch))),
    }
    let mut trace = Trace {
        recorder,
        transfers: &transfers,
        scalar: scalars.as_deref(),
    };
    worker(&mut observer, &mut trace)
}

impl Executable {
    pub(in crate::composition::mlx) fn quote_saved_original_media_capture(
        &self,
        input: eredu_architectures::prepared_execution::OriginalMediaWorkspaceInput,
        current: &eredu_runtime::working_memory::MediaSessionBinding,
        geometry: InferenceGeometry,
        state: &State,
        context: &WorkspaceContext,
        sampling: eredu_architectures::prepared_execution::BorrowedTextSamplingWorkspace<'_>,
        checkpoint: &eredu_runtime::capture::FundedCaptureCheckpoint,
        selection: &eredu_runtime::layered::PreparedCaptureSelection,
        interventions: Option<TextInterventionQuote<'_>>,
        recorder: &mut ResidentRecipeRecorder,
    ) -> Result<eredu_architectures::prepared_execution::OriginalMediaWorkspaceReport,
        PreparedExecutionError<eredu_nn::Error>> {
        let blueprint = self.inference_blueprint().ok_or_else(||
            PreparedExecutionError::Backend(context.metadata_source(WorkingMemoryError::UnknownBound)))?;
        if !selection.is_prepared_media() || checkpoint.next_prediction() != 0 {
            return Err(PreparedExecutionError::Backend(
                context.metadata_source(WorkingMemoryError::IdentityMismatch)));
        }
        with_saved_capture_trace(self, geometry, context, checkpoint, selection, interventions, recorder, None,
            |observer, trace| blueprint.quote_original_media_with_existing_sampling_observed_and_trace(
                input, current, geometry, state, context, None, sampling,
                selection.paths(), observer, trace,
            ).map_err(|cause| PreparedExecutionError::Backend(context.metadata_source(cause.into_failure()))),
        )
    }
}

trait CaptureRecorder: InferenceEquationTraceObserver {
    fn capture_population(&mut self,population:CaptureNativePopulation)->Result<(),eredu_nn::Error>;
    fn capture_scalars(&mut self,_scalar:Option<&[Cell<Option<WorkspaceFloatingType>>]>)->Result<(),eredu_nn::Error> { Ok(()) }
}
impl CaptureRecorder for ResidentRecipeRecorder {
    fn capture_population(&mut self,population:CaptureNativePopulation)->Result<(),eredu_nn::Error> {
        self.record_capture_population(population)
    }
}
impl CaptureRecorder for ParallelRecipeRecorder {
    fn capture_scalars(&mut self,scalar:Option<&[Cell<Option<WorkspaceFloatingType>>]>)->Result<(),eredu_nn::Error> {
        match scalar { Some(scalar) => self.record_capture_scalars(scalar), None => Ok(()) }
    }
    fn capture_population(&mut self,population:CaptureNativePopulation)->Result<(),eredu_nn::Error> {
        self.record_capture_population(population)
    }
}
struct Trace<'a,R:CaptureRecorder> {
    recorder: &'a mut R,
    transfers: &'a Cell<CaptureNativePopulation>,
    scalar: Option<&'a [Cell<Option<WorkspaceFloatingType>>]>,
}
impl<R:CaptureRecorder> InferenceEquationTraceObserver for Trace<'_,R> {
    fn observe(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
    ) -> Result<(), eredu_nn::Error> {
        self.recorder
            .observe(span, report, retained_roots, output_roots, input_operation)?;
        self.recorder.capture_population(self.transfers.get())?;
        self.recorder.capture_scalars(self.scalar)
    }
    fn observe_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
        output: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), eredu_nn::Error> {
        self.recorder.observe_with_storage(
            span,
            report,
            retained_roots,
            output_roots,
            input_operation,
            output,
        )?;
        self.recorder.capture_population(self.transfers.get())?;
        self.recorder.capture_scalars(self.scalar)
    }
    fn observe_prepared_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: Option<usize>,
        output: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), eredu_nn::Error> {
        self.recorder.observe_prepared_with_storage(
            span,
            report,
            retained_roots,
            output_roots,
            input_operation,
            output,
        )?;
        self.recorder.capture_population(self.transfers.get())?;
        self.recorder.capture_scalars(self.scalar)
    }
    fn observe_sampling(
        &mut self,
        phase: SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
    ) -> Result<(), eredu_nn::Error> {
        self.recorder.observe_sampling(phase, report)
    }
    fn observe_sampling_with_storage(
        &mut self,
        phase: SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
        closing: WorkspaceStoragePopulation,
    ) -> Result<(), eredu_nn::Error> {
        self.recorder
            .observe_sampling_with_storage(phase, report, closing)
    }
}
