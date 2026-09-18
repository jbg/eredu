//! The actual selected media traversal under an ordinary diagnostic owner.
use super::*;
use crate::{
    composite_execution::{
        CompositeMediaIngressArchitecture, PreparedCompositeArchitecture,
        PreparedCompositeInput,
    },
    media_plan::BoundPreparedMediaSemantics,
};
use eredu_nn::workspace::{
    HostMetadataFunding, WorkspaceReportScalars, WorkspaceStateSpanReport,
    WorkspaceTensorBufferReport, WorkspaceTraceReport,
};
use eredu_runtime::working_memory::{
    InferenceWorkspaceError, WorkspaceReportMetadata, quote_inference_workspace_with_report_owner,
};
use eredu_runtime::{
    PreparedInputPart, PreparedInputPayload, PreparedModelInput,
    input::OriginalPreparedInputProjection, working_memory::WorkingMemoryUnquotedLease,
};
use std::{borrow::Borrow, cell::RefCell, ops::Deref, rc::Rc};

#[derive(Clone, Copy)]
pub(super) struct MediaEquationRef<'a> {
    pub(super) input: &'a RefCell<Option<OriginalMediaWorkspaceInput>>,
    pub(super) intervals: Option<&'a RefCell<Vec<MediaEquationInterval>>>,
}
struct Cut {
    report: WorkspaceReportScalars,
    operations: usize,
    roots: Vec<WorkspaceTensor>,
}

// Only these private aliases escape into an interval or the shared reducer.
// The final Rc shell retires before its independent planning-account alias.
struct TraceOwner {
    report: Option<Rc<WorkspaceTraceReport>>,
    _funding: Option<HostMetadataFunding>,
}
impl TraceOwner {
    fn new(report: WorkspaceTraceReport, context: &WorkspaceContext) -> Result<Self, Error> {
        let metadata = WorkspaceReportMetadata::new(context);
        metadata
            .admit::<Self>()
            .map_err(|cause| metadata.error(cause))?;
        Ok(Self {
            report: Some(context.metadata_rc(report)?),
            _funding: context.metadata_funding(),
        })
    }
}
impl Clone for TraceOwner {
    fn clone(&self) -> Self {
        Self {
            report: self.report.clone(),
            _funding: self._funding.clone(),
        }
    }
}
impl Deref for TraceOwner {
    type Target = WorkspaceTraceReport;
    fn deref(&self) -> &Self::Target {
        self.report.as_deref().expect("live media trace")
    }
}
impl Borrow<WorkspaceTraceReport> for TraceOwner {
    fn borrow(&self) -> &WorkspaceTraceReport {
        self
    }
}
impl Drop for TraceOwner {
    fn drop(&mut self) {
        if let Some(report) = self.report.take() {
            drop(Rc::into_inner(report));
        }
    }
}
/// One complete equation interval. This is ordinary diagnostic metadata, not
/// source registration, a host-control bound or native completion evidence.
pub struct MediaEquationInterval {
    span: InferenceWorkspaceSpan,
    report: TraceOwner,
    cut: Option<Cut>,
    closing: Vec<WorkspaceTensor>,
}
impl MediaEquationInterval {
    pub fn span(&self) -> &InferenceWorkspaceSpan {
        &self.span
    }
    /// Borrows the actual operation at a diagnostic ordinal. This is descriptive
    /// workspace metadata and never an operation receipt or native permission.
    pub fn operation(&self, index: usize) -> Option<&eredu_nn::workspace::WorkspaceOperation> {
        self.report.operations.get(index)
    }
    pub fn operations(&self) -> usize {
        self.report.operations.len()
    }
    pub fn cut_operation(&self) -> Option<usize> {
        self.cut.as_ref().map(|cut| cut.operations)
    }
    pub fn compact_root_count(&self) -> usize {
        self.cut.as_ref().map_or(0, |cut| cut.roots.len())
    }
    pub fn compact_backing_bytes(&self) -> Option<u64> {
        self.cut.as_ref()?.report.state.as_ref()?.retained_bytes
    }
    pub fn closing_root_count(&self) -> usize {
        self.closing.len()
    }
    pub fn closing_backing_bytes(&self) -> Option<u64> {
        self.report.state.as_ref()?.retained_bytes
    }
    pub fn tensor_bytes(&self) -> Option<u64> {
        self.report.tensor_buffers.total_bytes
    }
    /// Complete numerical backing union: closing roots plus new transient
    /// tensor/scratch and displaced opening roots. Host controls are excluded.
    pub fn tensor_interval_bytes(&self) -> Option<u64> {
        tensor_interval_bytes(self.report.state.as_ref(), &self.report.tensor_buffers)
    }
    pub fn encoder_cut_tensor_bytes(&self) -> Option<u64> {
        self.cut.as_ref()?.report.tensor_buffers.total_bytes
    }
    pub fn encoder_cut_interval_bytes(&self) -> Option<u64> {
        let report = &self.cut.as_ref()?.report;
        tensor_interval_bytes(report.state.as_ref(), &report.tensor_buffers)
    }
    pub fn unpriced_tensor_operations(
        &self,
    ) -> impl Iterator<Item = &eredu_nn::workspace::WorkspaceOperation> {
        self.report
            .unpriced_operations
            .iter()
            .map(|&index| &self.report.operations[index])
    }
    pub fn prepared_rotary_operations(&self) -> usize {
        self.report
            .operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation.kind,
                    eredu_nn::workspace::WorkspaceOperationKind::PreparedMultiAxisRotary(_)
                )
            })
            .count()
    }
}
fn tensor_interval_bytes(
    state: Option<&WorkspaceStateSpanReport>,
    tensors: &WorkspaceTensorBufferReport,
) -> Option<u64> {
    let state = state?;
    state
        .retained_bytes?
        .checked_add(tensors.transient_bytes?)?
        .checked_add(state.displaced_bytes?)
}

/// The complete report and all its library-owned source/trace controls retain
/// the genuine ordinary lease. Original B is held only as its actual source alias.
pub struct OriginalMediaWorkspaceReport {
    equations: InferenceWorkspaceReport,
    sampling: Option<SamplingWorkspaceReport>,
    intervals: Vec<MediaEquationInterval>,
    original: BoundPreparedMediaSemantics,
    tables: OriginalPreparedInputProjection,
    source_storage: eredu_runtime::input::OriginalPreparedWorkspaceSource,
    ordinary: Option<WorkingMemoryUnquotedLease>,
    _planning: Option<HostMetadataFunding>,
}
impl OriginalMediaWorkspaceReport {
    pub(super) fn into_equations(self) -> InferenceWorkspaceReport { self.equations }
    /// Moves the completed equation/sampling reports into the shared generation
    /// composer. The caller already retains the exact source registration; no
    /// report, source tensor or sampling history is cloned by this conversion.
    pub fn into_generation(
        self,
        context: &WorkspaceContext,
    ) -> Result<PreparedTextGenerationWorkspace, Error> {
        let metadata = WorkspaceReportMetadata::new(context);
        metadata
            .admit::<(
                PreparedTextGenerationWorkspace,
                Result<PreparedTextGenerationWorkspace, Error>,
            )>()
            .map_err(|cause| metadata.error(cause))?;
        let Some(sampling) = self.sampling else {
            return Err(context.metadata_error(format_args!(
                "media generation requires its traced sampling report"
            )));
        };
        Ok(PreparedTextGenerationWorkspace {
            equations: self.equations,
            sampling,
        })
    }

    /// Actual input-source roots and their separately retained original B account.
    pub fn source_storage(&self) -> &eredu_runtime::input::OriginalPreparedWorkspaceSource {
        &self.source_storage
    }
    /// Sampling from the same score layouts and backing unions, when requested.
    pub fn sampling(&self) -> Option<&SamplingWorkspaceReport> {
        self.sampling.as_ref()
    }
    pub fn intervals(&self) -> &[MediaEquationInterval] {
        &self.intervals
    }
    pub fn equations(&self) -> &InferenceWorkspaceReport {
        &self.equations
    }
}
impl std::fmt::Debug for OriginalMediaWorkspaceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalMediaWorkspaceReport")
            .field("intervals", &self.intervals.len())
            .finish_non_exhaustive()
    }
}
/// Failed ordinary diagnostic preparation keeps partial reports/source and
/// original ordinary custody until its actual error and controls retire.
pub struct OriginalMediaWorkspaceTraceError {
    cause: PreparedExecutionError<Error>,
    intervals: Vec<MediaEquationInterval>,
    original: BoundPreparedMediaSemantics,
    tables: OriginalPreparedInputProjection,
    source_storage: eredu_runtime::input::OriginalPreparedWorkspaceSource,
    ordinary: Option<WorkingMemoryUnquotedLease>,
    _planning: Option<HostMetadataFunding>,
}
impl std::fmt::Display for OriginalMediaWorkspaceTraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::fmt::Debug for OriginalMediaWorkspaceTraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalMediaWorkspaceTraceError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl std::error::Error for OriginalMediaWorkspaceTraceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            PreparedExecutionError::Backend(cause) => Some(cause),
            cause => Some(cause),
        }
    }
}

/// Sendable failure after its non-Send trace prefix has retired. The original
/// source and all host planning custody remain live through the cause shell.
pub struct OriginalMediaWorkspaceTraceFailure {
    cause: PreparedExecutionError<Error>,
    original: BoundPreparedMediaSemantics,
    tables: OriginalPreparedInputProjection,
    ordinary: Option<WorkingMemoryUnquotedLease>,
    _planning: Option<HostMetadataFunding>,
}
impl std::fmt::Debug for OriginalMediaWorkspaceTraceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalMediaWorkspaceTraceFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for OriginalMediaWorkspaceTraceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for OriginalMediaWorkspaceTraceFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl OriginalMediaWorkspaceTraceError {
    /// Retires the partial trace before transporting its exact cause. Source and
    /// funding remain inside the returned closed owner, including on unwind.
    pub fn into_failure(self) -> OriginalMediaWorkspaceTraceFailure {
        let Self {
            cause,
            intervals,
            original,
            tables,
            source_storage,
            ordinary,
            _planning,
        } = self;
        let failure = OriginalMediaWorkspaceTraceFailure {
            cause,
            original,
            tables,
            ordinary,
            _planning,
        };
        drop(intervals);
        drop(source_storage);
        failure
    }
}

impl PreparedInferenceBlueprint {
    /// Runs the selected metadata equations for a genuine compiled media source.
    /// The native caller supplies this same executable's current binding, exact
    /// projected state/backing and selected parameter provider. The input was
    /// projected under an already held ordinary lease; this is never pre-admission
    /// allocating inspection. Distributed quoting and managed use remain separate.
    pub fn quote_original_media_ordinary(
        &self,
        input: OriginalMediaWorkspaceInput,
        current: &eredu_runtime::working_memory::MediaSessionBinding,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
    ) -> Result<OriginalMediaWorkspaceReport, OriginalMediaWorkspaceTraceError> {
        self.quote_original_media_with_trace(
            input, current, geometry, state, context, parameters, None, None, None, None,
        )
    }

    /// Reduces the actual encoder-once and decoder intervals plus ordinary
    /// sampling through one source-bound observer. This grants no native work.
    pub fn quote_original_media_with_sampling_and_trace<'a>(
        &self,
        input: OriginalMediaWorkspaceInput,
        current: &eredu_runtime::working_memory::MediaSessionBinding,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        observer: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
    ) -> Result<OriginalMediaWorkspaceReport, OriginalMediaWorkspaceTraceError> {
        let observer = RefCell::new(observer);
        self.quote_original_media_with_trace(
            input,
            current,
            geometry,
            state,
            context,
            parameters,
            Some(TextSamplingInput::Configured(config, filter.into())),
            Some(EquationTraceRef(&observer)),
            None,
            communication,
        )
    }

    /// Quotes an actual saved sampler through the same media equation worker.
    /// Its history/key remain borrowed from the independently copied source.
    pub fn quote_original_media_with_existing_sampling_and_trace(
        &self,
        input: OriginalMediaWorkspaceInput,
        current: &eredu_runtime::working_memory::MediaSessionBinding,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        observer: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
    ) -> Result<OriginalMediaWorkspaceReport, OriginalMediaWorkspaceTraceError> {
        let observer = RefCell::new(observer);
        self.quote_original_media_with_trace(
            input,
            current,
            geometry,
            state,
            context,
            parameters,
            Some(TextSamplingInput::Borrowed(sampling)),
            Some(EquationTraceRef(&observer)),
            None,
            communication,
        )
    }

    /// Quotes original media, full-sequence capture and sampling in the same
    /// retained-ingress traversal. The caller supplies its actual path source
    /// and span-aware observer; this method grants no native authority.
    pub fn quote_original_media_with_sampling_observed_and_trace<'a>(
        &self,
        input: OriginalMediaWorkspaceInput,
        current: &eredu_runtime::working_memory::MediaSessionBinding,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        paths: &eredu_runtime::SharedLayeredObservationPaths,
        observer: &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
        trace: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
    ) -> Result<OriginalMediaWorkspaceReport, OriginalMediaWorkspaceTraceError> {
        let observer = RefCell::new(observer);
        let trace = RefCell::new(trace);
        self.quote_original_media_with_trace(
            input, current, geometry, state, context, parameters,
            Some(TextSamplingInput::Configured(config, filter.into())),
            Some(EquationTraceRef(&trace)),
            Some(observed::ObservationRef { paths, observer: &observer, invocation_prediction: None }),
            communication,
        )
    }

    /// Quotes independently copied sampler and capture history against the
    /// saved original media source through the same retained-ingress worker.
    pub fn quote_original_media_with_existing_sampling_observed_and_trace(
        &self,
        input: OriginalMediaWorkspaceInput,
        current: &eredu_runtime::working_memory::MediaSessionBinding,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        paths: &eredu_runtime::SharedLayeredObservationPaths,
        observer: &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
        trace: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
    ) -> Result<OriginalMediaWorkspaceReport, OriginalMediaWorkspaceTraceError> {
        let observer = RefCell::new(observer);
        let trace = RefCell::new(trace);
        self.quote_original_media_with_trace(
            input, current, geometry, state, context, parameters,
            Some(TextSamplingInput::Borrowed(sampling)),
            Some(EquationTraceRef(&trace)),
            Some(observed::ObservationRef { paths, observer: &observer, invocation_prediction: None }),
            communication,
        )
    }

    pub(super) fn quote_original_media_with_trace(
        &self,
        input: OriginalMediaWorkspaceInput,
        current: &eredu_runtime::working_memory::MediaSessionBinding,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        sampling: Option<TextSamplingInput<'_>>,
        trace: Option<EquationTraceRef<'_, '_>>,
        observation: Option<observed::ObservationRef<'_, '_>>,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
    ) -> Result<OriginalMediaWorkspaceReport, OriginalMediaWorkspaceTraceError> {
        let original = input.original.clone();
        let tables = input.tables.clone();
        let source_storage = input.source_storage.clone();
        let ordinary = input.ordinary.clone();
        let inputs = RefCell::new(Some(input));
        let intervals = RefCell::new(Vec::new());
        let planning = context.metadata_funding();
        let metadata = WorkspaceReportMetadata::new(context);
        let result = (|| {
            metadata
                .admit::<OriginalMediaWorkspaceReport>()
                .map_err(|cause| PreparedExecutionError::Backend(metadata.error(cause)))?;
            metadata
                .admit::<(
                    OriginalMediaWorkspaceTraceError,
                    OriginalMediaWorkspaceTraceFailure,
                )>()
                .map_err(|cause| PreparedExecutionError::Backend(metadata.error(cause)))?;
            if !original.binding().matches(current)
                || current.frontier() != geometry.cached_positions
            {
                return Err(preparation_message(
                    context,
                    format_args!("media diagnostic source differs from current selected session"),
                ));
            }
            if let Some(observation) = observation {
                metadata.admit::<(
                    observed::ObservationRef<'_, '_>,
                    RefCell<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
                    RefCell<&mut dyn InferenceEquationTraceObserver>,
                )>().map_err(|cause| PreparedExecutionError::Backend(metadata.error(cause)))?;
                observation.validate_selection(geometry, context)
                    .map_err(PreparedExecutionError::Backend)?;
                if geometry.max_output_tokens > 0 && observation.requires_sequence_readout()
                    && geometry.output != eredu_core::OutputDemand::Sequence
                {
                    return Err(preparation_message(context,
                        format_args!("media observation requires sequence readout")));
                }
            }
            if sampling.is_some()
                && (geometry.batch_size != 1
                    || geometry.output == eredu_core::OutputDemand::StateOnly
                    || (observation.is_none()
                        && geometry.output != eredu_core::OutputDemand::LastPosition))
            {
                return Err(preparation_message(
                    context,
                    format_args!(
                        "configured text sampling requires single-sequence final-position generation"
                    ),
                ));
            }
            geometry
                .validate()
                .map_err(|e| preparation_message(context, format_args!("{e}")))?;
            // The actual projected layer variants, context and page geometry
            // must match the retained selection. Native source pins and span
            // catalogs stay with the enclosing completed-input quote owner.
            if self.selected().execution().parallel_topology().is_none() {
                eredu_runtime::working_memory::validate_workspace_state_realization(
                    state, self.selected().text_realization().state(), context,
                ).map_err(PreparedExecutionError::Metadata)?;
            }
            if parameters.is_some()
                && !matches!(
                    self.selected().text_realization().residency(),
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                        | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                )
            {
                return Err(preparation_message(
                    context,
                    format_args!("media parameter provider differs from selected residency"),
                ));
            }
            let visitor = EquationVisitor {
                geometry,
                state,
                context,
                sampling,
                input_dtype: None,
                target_capture: false,
                routed_pass: None,
                external_target: None,
                parameters,
                unpriced_execution: None,
                observation,
                trace,
                media: Some(MediaEquationRef {
                    input: &inputs,
                    intervals: Some(&intervals),
                }),
            }
            ;
            if self.selected().execution().parallel_topology().is_some() {
                let source = self.sources.construction_semantics().direct_partition.get()
                    .ok_or_else(|| preparation_message(context, format_args!("completed partition media source is unavailable")))?;
                source.quote_media(&self.sources, communication, visitor).map_err(PreparedExecutionError::Metadata)
            } else {
                visitor.construct(self)
            }
        })();
        // Any unconsumed input retires before its outer ordinary custody.
        drop(inputs);
        let intervals = intervals.into_inner();
        match result {
            Ok((equations, sampling)) => Ok(OriginalMediaWorkspaceReport {
                equations,
                sampling,
                intervals,
                original,
                tables,
                source_storage,
                ordinary,
                _planning: planning,
            }),
            Err(cause) => Err(OriginalMediaWorkspaceTraceError {
                cause,
                intervals,
                original,
                tables,
                source_storage,
                ordinary,
                _planning: planning,
            }),
        }
    }
}
pub(super) struct CutHook {
    cut: Option<Cut>,
}
impl<C> eredu_runtime::LayeredTraversalHook<WorkspaceBackend, C, Error> for CutHook {
    fn retained_media_cut(
        &mut self,
        visit: &mut dyn FnMut(&mut dyn FnMut(&WorkspaceTensor)),
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        if self.cut.is_some() {
            return Err(
                context.metadata_error(format_args!("media source produced a second initial cut"))
            );
        }
        let mut roots = context.metadata_vec(0)?;
        append_roots(&mut roots, context, |visitor| visit(visitor))?;
        let metadata = WorkspaceReportMetadata::new(context);
        metadata
            .admit::<Cut>()
            .map_err(|cause| metadata.error(cause))?;
        let operations = context.operation_count();
        let report = context.report_scalars(&roots)?;
        self.cut = Some(Cut {
            report,
            operations,
            roots,
        });
        Ok(())
    }
}
fn append_roots(
    roots: &mut Vec<WorkspaceTensor>,
    context: &WorkspaceContext,
    visit: impl FnOnce(&mut dyn FnMut(&WorkspaceTensor)),
) -> Result<(), Error> {
    let mut failure = None;
    visit(&mut |value| {
        if failure.is_some() {
            return;
        }
        if let Err(cause) = context.reserve_metadata_vec(roots, 1) {
            failure = Some(cause);
        } else {
            roots.push(value.clone());
        }
    });
    failure.map_or(Ok(()), Err)
}
fn copy_roots(
    roots: &[WorkspaceTensor],
    context: &WorkspaceContext,
) -> Result<Vec<WorkspaceTensor>, Error> {
    let mut copied = context.metadata_vec(roots.len())?;
    copied.extend(roots.iter().cloned());
    Ok(copied)
}
impl EquationVisitor<'_, '_, '_> {
    pub(super) fn quote_composite_media<A>(
        self,
        mut modules: crate::replicated_text::PreparedReplicatedTextModules<
            PreparedCompositeArchitecture<A>,
        >,
        admission: A::AdmissionConfig,
        provider:super::EquationRoutedProvider,
    ) -> Result<EquationQuote, Error>
    where
        A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend,ResidentState> + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        if self.state.layout() != modules.contract().selected().state().layout() {
            return Err(self.context.metadata_error(format_args!(
                "media equation state differs from selected layout",
            )));
        }
        let reference = self.media.expect("typed media visitor");
        if reference.intervals.is_none() {
            return self.quote_composite_whole_media(modules, provider);
        }
        let (original_roots, plan) = self.prepare_media_interval_source::<A>()?;
        modules.retain_composite_graph(self.context)?;
        let runtime =
            EquationRuntime::from_prepared(
                modules, self.parameters, self.context,
                self.observation.map(|observation| observation.paths), false,
            )?;
        self.context.charge_metadata(std::mem::size_of::<runtime::Serial<'_, A>>())?;
        let mut driver = runtime::Serial { runtime, provider };
        self.quote_media_intervals::<A, _>(admission, original_roots, plan, &mut driver)
    }

    pub(super) fn prepare_media_interval_source<A>(&self) -> Result<(
        Vec<WorkspaceTensor>, A::IngressPlan,
    ), Error>
    where A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error> + 'static,
        A::InputPartPlan: 'static,
    {
        let reference = self.media.expect("typed media visitor");
        let input = reference.input.borrow_mut().take().ok_or_else(|| {
            self.context.metadata_error(format_args!("media equation source consumed more than once"))
        })?;
        let mut roots = self.context.metadata_vec(0)?;
        for value in input.prepared.parts().iter().flat_map(|part| {
            std::iter::once(part.payload().value()).chain(part.metadata().values())
        }) {
            self.context.reserve_metadata_vec(&mut roots, 1)?;
            roots.push(value.clone());
        }
        let plan = A::prepare_original_workspace_ingress_plan_with_metadata(input, self.geometry, self.context)?;
        Ok((roots, plan))
    }

    pub(super) fn quote_media_intervals<A, D>(self, admission: A::AdmissionConfig,
        original_roots: Vec<WorkspaceTensor>, plan: A::IngressPlan, driver: &mut D,
    ) -> Result<EquationQuote, Error>
    where A: CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error> + 'static,
        A::InputPartPlan: 'static,
        D: runtime::Driver<A>,
    {
        self.context.charge_metadata(std::mem::size_of::<(
            &mut D, A::AdmissionConfig, A::IngressPlan, Vec<WorkspaceTensor>,
            Result<EquationQuote, Error>,
        )>())?;
        let intervals = self.media.expect("typed media visitor").intervals.expect("retained media intervals");
        let observation_host_peak = driver.observation_host_peak_bytes(self.context)?;
        let mut source = driver.prepare_source(plan, self.context)?;
        let mut state = self.state.try_clone_workspace(self.context)?;
        let mut score_source = sampling::SamplingScores::default();
        let equations =
            quote_inference_workspace_with_report_owner(self.geometry, self.context, |span| {
                let metadata = WorkspaceReportMetadata::new(self.context);
                metadata
                    .admit::<MediaEquationInterval>()
                    .map_err(|cause| metadata.error(cause))?;
                {
                    let mut intervals = intervals.borrow_mut();
                    self.context.reserve_metadata_vec(&mut intervals, 1)?;
                }
                let (position, count, mut demand) = match span {
                    InferenceWorkspaceSpan::Sampling(_) => unreachable!("model equation scheduler emits only prefill/decode spans"),
                    InferenceWorkspaceSpan::Prefill(chunk) => {
                        (chunk.position, chunk.input.end - chunk.input.start, chunk.output)
                    }
                    InferenceWorkspaceSpan::Decode { position, output, .. } => (*position, 1, *output),
                };
                validate_frontier_with_context(&state, position, self.context)?;
                let active = match self.observation {
                    Some(observation) => observation.begin_span(self.geometry, span, self.context)?,
                    None => false,
                };
                if active && self.observation.expect("active observation").requires_sequence_readout() {
                    demand = eredu_core::OutputDemand::Sequence;
                }
                metadata.admit::<(
                    bool, eredu_core::OutputDemand, eredu_runtime::prefill::PrefillChunk,
                    Option<WorkspaceTensor>, Option<A::ForwardContext>, CutHook,
                )>().map_err(|cause| metadata.error(cause))?;
                let mut opening = copy_roots(&original_roots, self.context)?;
                for value in state
                    .as_ref()
                    .iter()
                    .flat_map(|lane| lane.retained_values())
                {
                    self.context.reserve_metadata_vec(&mut opening, 1)?;
                    opening.push(value.clone());
                }
                append_roots(&mut opening, self.context, |visitor| {
                    source.visit_roots(visitor)
                })?;
                if let Some(observation) = self.observation {
                    observation.append_retained(&mut opening, self.context)?;
                }
                self.context.begin_state_span(opening.iter())?;
                let mut checkpoint = self.context.metadata_vec(state.as_ref().len())?;
                for lane in state.as_ref() {
                    checkpoint.push(lane.checkpoint_for_transaction(self.context)?);
                }
                let mut hook = CutHook { cut: None };
                let mut forward = None;
                let mut input_operation = None;
                let scores = match span {
                    InferenceWorkspaceSpan::Sampling(_) => unreachable!("model equation scheduler emits only prefill/decode spans"),
                    InferenceWorkspaceSpan::Prefill(chunk) => {
                        let mut chunk = chunk.clone();
                        chunk.output = demand;
                        let (scores, context) = if active {
                            let mut observer = self.observation.expect("active observation").observer.borrow_mut();
                            driver.prefill(&mut source, &chunk, &mut state, self.context,
                                &mut hook, Some(&mut **observer))?
                        } else {
                            driver.prefill(&mut source, &chunk, &mut state, self.context,
                                &mut hook, None)?
                        };
                        forward = Some(context);
                        scores
                    }
                    InferenceWorkspaceSpan::Decode { .. } => {
                        metadata
                            .admit::<(
                                PreparedInputPart<WorkspaceTensor>,
                                PreparedModelInput<WorkspaceTensor>,
                                crate::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
                                PreparedCompositeInput<'_, WorkspaceTensor, A::InputPartPlan>,
                            )>()
                            .map_err(|cause| metadata.error(cause))?;
                        input_operation = Some(self.context.operation_count());
                        let tokens = WorkspaceTensor::full_u32(
                            0,
                            &[self.geometry.batch_size as i32, 1],
                            self.context,
                        )?;
                        let part = PreparedInputPart::new(
                            eredu_core::InputModality::Text,
                            PreparedInputPayload::TokenIds(tokens),
                            [],
                        )
                        .map_err(|cause| self.context.metadata_source(cause))?;
                        let mut parts = self.context.metadata_vec(1)?;
                        parts.push(part);
                        let prepared =
                            PreparedModelInput::new_with_metadata(parts, self.context, |value| {
                                super::composite::TextInspector::identity_with_metadata(
                                    value,
                                    self.context,
                                )
                            })?;
                        let admitted = A::admit_prepared_input_with_metadata(
                            &admission,
                            &prepared,
                            &super::composite::TextInspector,
                            self.context,
                        )?;
                        let input = PreparedCompositeInput::new_with_diagnostic(
                            &prepared, &admitted,
                            |message| self.context.metadata_error(format_args!("{message}")),
                        )?;
                        if active {
                            let mut observer = self.observation.expect("active observation").observer.borrow_mut();
                            driver.decode(input, &mut state, self.context, demand, Some(&mut **observer))?
                        } else {
                            driver.decode(input, &mut state, self.context, demand, None)?
                        }
                    }
                };
                let scores = if active {
                    let mut observer = self.observation.expect("active observation").observer.borrow_mut();
                    driver.finish_logits(scores, demand, Some(&mut **observer), self.context)?
                } else { driver.finish_logits(scores, demand, None, self.context)? };
                // Match the native text output selection after complete logits
                // observation; its source backing remains in the observer roots.
                let scores = if self.observation.is_some() && self.sampling.is_some() {
                    scores.map(|scores| observed::sampling_row(&scores, self.context)).transpose()?
                } else { scores };
                let mut rollback = self.context.metadata_vec(checkpoint.len())?;
                for lane in &checkpoint {
                    rollback.push(lane.checkpoint_for_transaction(self.context)?);
                }
                let output_population = if self.sampling.is_some() || self.trace.is_some() {
                    score_source.observe_prepared(
                        scores.as_ref(),
                        self.sampling.is_some(),
                        self.context,
                    )?
                } else {
                    Some(eredu_nn::workspace::WorkspaceStoragePopulation::EMPTY)
                };
                validate_frontier_with_context(&state, position + count, self.context)?;
                let mut closing = copy_roots(&original_roots, self.context)?;
                for value in state
                    .as_ref()
                    .iter()
                    .flat_map(|lane| lane.retained_values())
                {
                    self.context.reserve_metadata_vec(&mut closing, 1)?;
                    closing.push(value.clone());
                }
                append_roots(&mut closing, self.context, |visitor| {
                    source.visit_roots(visitor)
                })?;
                if let Some(observation) = self.observation {
                    observation.append_retained(&mut closing, self.context)?;
                }
                let mut report = self.context.finish_report(&closing)?;
                if self.observation.is_some() {
                    report = observed::with_hook_workspace(report, observation_host_peak, self.context)?;
                }
                if let Some(reason) = self.unpriced_execution {
                    report.state.as_mut().expect("span opened").transient_bytes = None;
                    report.tensor_buffers.transient_bytes = None;
                    report.host_workspace_bytes = None;
                    self.context
                        .reserve_metadata_vec(&mut report.assumptions, 1)?;
                    report
                        .assumptions
                        .push(self.context.metadata_string(format_args!("{reason}"))?);
                }
                if active {
                    self.observation.expect("active observation").observer.borrow_mut()
                        .end_span(span, self.context)?;
                }
                if let Some(trace) = self.trace {
                    trace.0.borrow_mut().observe_prepared_with_storage(
                        span,
                        &report,
                        closing.len(),
                        usize::from(scores.is_some()),
                        input_operation,
                        output_population,
                    )?;
                }
                let report = TraceOwner::new(report, self.context)?;
                let mut intervals = intervals.borrow_mut();
                intervals.push(MediaEquationInterval {
                    span: span.clone(),
                    report: report.clone(),
                    cut: hook.cut,
                    closing,
                });
                drop(intervals);
                // All encoder/decoder/rollback allocations and both contexts have
                // been recorded in the same interval. Only now settle metadata.
                if matches!(span, InferenceWorkspaceSpan::Prefill(_)) {
                    source.finish_equation_span()?;
                }
                drop(forward);
                Ok::<_, Error>(report)
            })
            .map_err(|cause| match cause {
                InferenceWorkspaceError::Metadata(cause) => cause,
                cause => self.context.metadata_source(cause),
            })?;
        let mut trace = self.trace;
        let sampling = score_source.finish(
            self.sampling,
            self.geometry.max_output_tokens,
            self.context,
            trace.as_mut().map(|trace| {
                trace as &mut dyn eredu_runtime::working_memory::SamplingWorkspaceObserver
            }),
        )?;
        Ok((equations, sampling))
    }
}

pub(super) mod runtime;

#[cfg(test)]
mod tests;
