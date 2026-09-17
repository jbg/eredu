//! Retained architecture/source construction for inference workspace inspection.

use super::*;
use crate::replicated_text::{
    PreparedReplicatedTextArchitecture, ReplicatedTextArchitectureVisitor,
    SharedReplicatedTextVisitor,
};
use eredu_core::{InferenceGeometry, TextFilterWorkspace, TextGenerationConfig};
use eredu_nn::{
    Error, Tensor,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceLayout, WorkspaceTensor},
};
use eredu_runtime::{
    DeviceState, ReplicatedTextArchitecture, ResidentRuntime, RuntimeLayerState, RuntimeState,
    RuntimeStateComponents,
    working_memory::{
        InferenceWorkspaceReport, InferenceWorkspaceSpan, SamplingWorkspaceReport,
        WorkspaceResidentLayerState, WorkspaceResidentStateFactory, quote_inference_workspace,
        quote_inference_workspace_with_context,
    },
};

mod autoregressive;
mod embedded;
mod external;
use external::{EquationCapture, ExternalTargetQuote};
mod prediction;
pub use embedded::EmbeddedTargetWorkspaceObservation;
pub use prediction::{EmbeddedPredictionWorkspaceObservation, WorkspacePredictionEquationTails};
mod composite;
mod destinations;
pub use destinations::{
    ReplicatedTextBindingDestinations, project_replicated_text_binding_destinations,
    PartitionedTextBindingDestinations, project_partitioned_text_binding_destinations,
    AddressableBindingDestinations, project_addressable_binding_destinations,
};
mod media_input;
mod media_trace;
pub use media_input::{
    OriginalMediaWorkspaceInput, OriginalMediaWorkspaceInputError, PreparedMediaWorkspaceTensor,
};
use media_trace::MediaEquationRef;
pub use media_trace::{
    MediaEquationInterval, OriginalMediaWorkspaceReport, OriginalMediaWorkspaceTraceError,
    OriginalMediaWorkspaceTraceFailure,
};
pub(crate) mod layerwise;
mod routed_provider;
use routed_provider::EquationRoutedProvider;
mod observed;
pub(crate) mod parallel;
mod sampling;
use layerwise::EquationRuntime;
pub use layerwise::WorkspaceLayerwiseParameters;
use observed::ObservationRef;
pub use sampling::BorrowedTextSamplingWorkspace;
use sampling::TextSamplingInput;

type ResidentState = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;

/// Borrows each complete ordinary equation trace before its storage retires.
/// A selected mechanism may reduce these exact operations into its own finite
/// recipe. No native handle, capacity, submission authority or successful fit
/// is supplied by this callback. Errors preserve the ordinary quoted prefix.
pub trait InferenceEquationTraceObserver {
    /// Observes one actual prefill/decode span after all its equations and
    /// transaction rollback overlap have been recorded. Root populations name
    /// actual retained state and the optional semantic output; additional
    /// mechanism-owned roots must come from that mechanism's lowering recipe.
    fn observe(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &eredu_nn::workspace::WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
    ) -> Result<(), Error>;

    /// The same completed trace plus the actual current semantic output's full
    /// backing union. None preserves missing output coverage; zero is an actual
    /// absent output. Existing observers keep their ordinary callback behavior.
    fn observe_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &eredu_nn::workspace::WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
        _output: Option<eredu_nn::workspace::WorkspaceStoragePopulation>,
    ) -> Result<(), Error> {
        self.observe(span, report, retained_roots, output_roots, input_operation)
    }

    /// Exact embedded-target output roots. Hidden capture and vocabulary scores
    /// remain distinct even when they alias. The population is their full union;
    /// default observers retain the existing equation callback behavior.
    fn observe_target_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &eredu_nn::workspace::WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
        output: Option<eredu_nn::workspace::WorkspaceStoragePopulation>,
        _capture: &WorkspaceTensor,
        _scores: Option<&WorkspaceTensor>,
    ) -> Result<(), Error> {
        self.observe_with_storage(
            span,
            report,
            retained_roots,
            output_roots,
            input_operation,
            output,
        )
    }

    /// Exact external target outputs, including every architecture-published
    /// context root. The population is the union with any requested scores.
    fn observe_external_target_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &eredu_nn::workspace::WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: Option<usize>,
        output: Option<eredu_nn::workspace::WorkspaceStoragePopulation>,
        _capture: &crate::composite_execution::ExternalPredictionTargetCapture<WorkspaceTensor>,
        _scores: Option<&WorkspaceTensor>,
    ) -> Result<(), Error> {
        self.observe_prepared_with_storage(
            span,
            report,
            retained_roots,
            output_roots,
            input_operation,
            output,
        )
    }

    /// Observes the same equation for an already prepared input source. None
    /// means the source entered as retained roots, so no operation may be
    /// skipped as a synthetic token placeholder. Some keeps the token contract.
    fn observe_prepared_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &eredu_nn::workspace::WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: Option<usize>,
        output: Option<eredu_nn::workspace::WorkspaceStoragePopulation>,
    ) -> Result<(), Error> {
        match input_operation {
            Some(operation) => self.observe_with_storage(
                span,
                report,
                retained_roots,
                output_roots,
                operation,
                output,
            ),
            None => Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into()),
        }
    }

    /// Observes the actual sampling preparation or step trace from the shared
    /// runtime driver. The default keeps equation-only observers unchanged.
    /// Completed sampler roots include the advanced RNG key and every emitted
    /// token's full possible backing union, not merely scalar token geometry.
    fn observe_sampling_with_storage(
        &mut self,
        phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &eredu_nn::workspace::WorkspaceTraceReport,
        _closing: eredu_nn::workspace::WorkspaceStoragePopulation,
    ) -> Result<(), Error> {
        self.observe_sampling(phase, report)
    }

    fn observe_sampling(
        &mut self,
        _phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        _report: &eredu_nn::workspace::WorkspaceTraceReport,
    ) -> Result<(), Error> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct EquationTraceRef<'a, 'observer>(
    &'a std::cell::RefCell<&'observer mut dyn InferenceEquationTraceObserver>,
);

impl eredu_runtime::working_memory::SamplingWorkspaceObserver for EquationTraceRef<'_, '_> {
    fn observe(
        &mut self,
        phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &eredu_nn::workspace::WorkspaceTraceReport,
    ) -> Result<(), Error> {
        self.0.borrow_mut().observe_sampling(phase, report)
    }
    fn observe_with_storage(
        &mut self,
        phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &eredu_nn::workspace::WorkspaceTraceReport,
        closing: eredu_nn::workspace::WorkspaceStoragePopulation,
    ) -> Result<(), Error> {
        self.0
            .borrow_mut()
            .observe_sampling_with_storage(phase, report, closing)
    }
}

/// The exact preparation retained through executable construction.
/// Clones share the original source graph, reader caches, leases and provenance;
/// quoting never requires reopening an artifact or selecting another execution.
/// Native devices, state and submission authority are supplied nowhere here.
#[derive(Clone)]
pub struct PreparedInferenceBlueprint {
    pub(super) sources: PreparedModelSources,
}

impl PreparedInferenceBlueprint {
    /// Compiles media semantics from the exact sources retained by this loaded
    /// execution. The loan neither reopens artifacts nor copies source tables.
    pub fn plan_original_media_semantics<'s, 'h>(
        &'s self,
        source: &'h eredu_runtime::working_memory::OriginalPreparedHostInput,
    ) -> Result<crate::media_plan::PreparedMediaSemanticCompile<'s, 'h>, crate::media_plan::MediaSemanticError> {
        self.sources.plan_original_media_semantics(source)
    }

    /// Source-owned host payload capacity retained by this exact execution.
    /// Native parameter buffers and materialization workspace remain separate.
    pub fn source_storage(
        &self,
    ) -> Result<Option<eredu_checkpoint::store::SourceStorage>, eredu_checkpoint::store::StoreError>
    {
        self.sources.source_storage()
    }

    /// Borrows actual retained source owners without constructing an inventory.
    /// Aliases are left to the caller's supplied destination to deduplicate.
    pub fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(eredu_checkpoint::store::SourceStorageRef<'_>),
    ) -> Result<bool, eredu_checkpoint::store::StoreError> {
        self.sources.visit_source_storage(visitor)
    }

    pub(crate) fn new(sources: PreparedModelSources) -> Self {
        Self { sources }
    }

    /// Compares the actual original cold source graph and every selected branch
    /// field against this constructed execution without cloning either plan.
    pub fn has_selected_sources(&self, sources: &PreparedModelSources) -> bool {
        self.sources.same_selected_sources(sources)
    }

    /// Total retained execution and mechanism selection.
    pub fn selected(&self) -> &crate::SelectedPreparation {
        self.sources.selected()
    }

    /// Architecture plan inseparably paired with the prepared source graph.
    pub fn architecture(&self) -> &ArtifactArchitecturePlan {
        self.sources.architecture()
    }

    /// Stable selected execution identity, without resolving artifact content.
    pub fn execution_identity(&self) -> &str {
        self.sources.execution_identity()
    }
}

impl PreparedInferenceBlueprint {
    /// Inspects every text equation span using the retained replicated selection.
    /// `state` must be a fresh projection of the actual resident cache into
    /// `context`, including its full backing capacities and alias identities.
    /// The projection is cloned and advanced; neither native nor supplied state
    /// is mutated. No checkpoint payload is read and no artifact is reopened.
    ///
    /// This quotes equations and selected resident transaction checkpoint and
    /// rollback copies. Materialization, input preparation, sampling, isolated
    /// snapshots and other enclosing resources must still be composed explicitly.
    /// A retained prediction selection is authenticated but its extension is
    /// not constructed or invoked by these target equations. Prediction work
    /// and partitioned execution require their corresponding quote routes.
    /// This is not a full-request admission or capability report.
    pub fn quote_replicated_resident_text(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        self.quote_text_impl(geometry, state, context, None, None, None)
            .map(|(equations, _)| equations)
    }

    /// Adds ordinary configured sampling across the complete output allowance.
    /// The selected architecture supplies actual score geometry; shared runtime
    /// policy supplies filtering, history growth and adaptive commitment. This
    /// requires single-sequence final-position generation. Materialization,
    /// prompt preparation and enclosing capture/speculation remain separate.
    pub fn quote_replicated_resident_text_with_sampling<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        if geometry.batch_size != 1 || geometry.output != eredu_core::OutputDemand::LastPosition {
            return Err(preparation_message(
                context,
                format_args!(
                    "configured text sampling requires single-sequence final-position generation"
                ),
            ));
        }
        let (equations, sampling) = self.quote_text_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Configured(config, filter.into())),
            None,
            None,
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("configured inspection produces a sampling report"),
        })
    }

    /// Traces the selected host- or disk-layerwise equations, constructing and binding
    /// each unit inside its executed span. The provider supplies exact prepared
    /// parameter backing; no checkpoint payload is read. Transfers, destination
    /// windows and other materialization costs still require separate bounds.
    pub fn quote_replicated_layerwise_text(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: &dyn WorkspaceLayerwiseParameters,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        self.quote_text_impl(geometry, state, context, None, Some(parameters), None)
            .map(|(equations, _)| equations)
    }

    /// Adds the same cumulative configured sampling as resident inspection to
    /// the selected layerwise equation traversal. Construction allocations
    /// remain conservatively live through each complete forward span.
    pub fn quote_replicated_layerwise_text_with_sampling<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        parameters: &dyn WorkspaceLayerwiseParameters,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        if geometry.batch_size != 1 || geometry.output != eredu_core::OutputDemand::LastPosition {
            return Err(preparation_message(
                context,
                format_args!(
                    "configured text sampling requires single-sequence final-position generation"
                ),
            ));
        }
        let (equations, sampling) = self.quote_text_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Configured(config, filter.into())),
            Some(parameters),
            None,
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("configured inspection produces a sampling report"),
        })
    }

    /// Inspects the existing sampler's full future output allowance through the
    /// same resident equations and selected score backing as configured sampling.
    /// The sampler and metadata random key are borrowed; their values and history
    /// are neither copied nor advanced. Random metadata must belong to `context`.
    /// Static-policy compatibility, source custody, copying and fresh admission
    /// remain separate requirements of the enclosing consumer.
    pub fn quote_replicated_resident_text_with_existing_sampling(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_text_with_existing_sampling(geometry, state, context, sampling, None, None)
    }

    /// Inspects borrowed sampling state through the existing selected host- or
    /// disk-layerwise traversal. Parameter construction and binding stay inside
    /// each equation span; materialization and transfer bounds remain separate.
    /// This has the same source and admission obligations as resident inspection.
    pub fn quote_replicated_layerwise_text_with_existing_sampling(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        parameters: &dyn WorkspaceLayerwiseParameters,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_text_with_existing_sampling(
            geometry,
            state,
            context,
            sampling,
            Some(parameters),
            None,
        )
    }

    /// Inspects the same retained sampler and selected resident/layerwise
    /// equations while lending every completed trace to a native recipe owner.
    /// Parameters select the existing layerwise binding mechanism; no copied
    /// state, native resource or admission authority is constructed here.
    pub fn quote_replicated_text_with_existing_sampling_and_trace(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observer: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        let observer = std::cell::RefCell::new(observer);
        self.quote_text_with_existing_sampling(
            geometry,
            state,
            context,
            sampling,
            parameters,
            Some(EquationTraceRef(&observer)),
        )
    }

    fn quote_text_with_existing_sampling(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        trace: Option<EquationTraceRef<'_, '_>>,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        if geometry.batch_size != 1 || geometry.output != eredu_core::OutputDemand::LastPosition {
            return Err(preparation_message(
                context,
                format_args!(
                    "configured text sampling requires single-sequence final-position generation"
                ),
            ));
        }
        let (equations, sampling) = self.quote_text_with_trace_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Borrowed(sampling)),
            parameters,
            None,
            trace,
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("borrowed inspection produces a sampling report"),
        })
    }

    fn quote_text_impl(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: Option<TextSamplingInput<'_>>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observation: Option<ObservationRef<'_, '_>>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        self.quote_text_with_trace_impl(
            geometry,
            state,
            context,
            sampling,
            parameters,
            observation,
            None,
        )
    }

    /// Runs the same selected resident equations and sampling quotation while
    /// borrowing each completed span into a mechanism's finite recipe producer.
    /// This performs no native work and cannot grant a missing bound.
    pub fn quote_replicated_resident_text_with_sampling_and_trace<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        observer: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        if geometry.batch_size != 1 || geometry.output != eredu_core::OutputDemand::LastPosition {
            return Err(preparation_message(
                context,
                format_args!(
                    "configured text sampling requires single-sequence final-position generation"
                ),
            ));
        }
        let observer = std::cell::RefCell::new(observer);
        let (equations, sampling) = self.quote_text_with_trace_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Configured(config, filter.into())),
            None,
            None,
            Some(EquationTraceRef(&observer)),
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("configured inspection produces a sampling report"),
        })
    }

    fn quote_text_with_trace_impl(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: Option<TextSamplingInput<'_>>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observation: Option<ObservationRef<'_, '_>>,
        trace: Option<EquationTraceRef<'_, '_>>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        self.quote_text_with_input_dtype(
            geometry,
            state,
            context,
            sampling,
            parameters,
            observation,
            trace,
            None,
            None,
        )
    }

    fn quote_text_with_input_dtype(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: Option<TextSamplingInput<'_>>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observation: Option<ObservationRef<'_, '_>>,
        trace: Option<EquationTraceRef<'_, '_>>,
        input_dtype: Option<eredu_nn::workspace::WorkspaceDtype>,
        routed_pass: Option<eredu_runtime::ExpertPass>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        self.quote_text_with_target_capture(
            geometry,
            state,
            context,
            sampling,
            parameters,
            observation,
            trace,
            input_dtype,
            false,
            routed_pass,
        )
    }

    fn quote_text_with_target_capture(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: Option<TextSamplingInput<'_>>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observation: Option<ObservationRef<'_, '_>>,
        trace: Option<EquationTraceRef<'_, '_>>,
        input_dtype: Option<eredu_nn::workspace::WorkspaceDtype>,
        target_capture: bool,
        routed_pass: Option<eredu_runtime::ExpertPass>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        context
            .charge_metadata(std::mem::size_of::<(bool,Option<eredu_runtime::ExpertPass>)>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        geometry
            .validate()
            .map_err(|error| preparation_message(context, format_args!("{error}")))?;
        eredu_runtime::working_memory::validate_workspace_state_realization(
            state,
            self.selected().text_realization().state(),
            context,
        )
        .map_err(PreparedExecutionError::Metadata)?;
        if parameters.is_some()
            && !matches!(
                self.selected().text_realization().residency(),
                eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                    | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
            )
        {
            return Err(preparation_message(
                context,
                format_args!(
                    "layerwise equation inspection requires selected host or disk residency"
                ),
            ));
        }
        let visitor = EquationVisitor {
            geometry,
            state,
            context,
            sampling,
            parameters,
            unpriced_execution: None,
            observation,
            trace,
            media: None,
            input_dtype,
            target_capture,
            routed_pass,
            external_target: None,
        };
        visitor.construct(self)
    }
}

#[derive(Clone, Copy)]
struct EquationVisitor<'a, 'observer, 'trace> {
    geometry: InferenceGeometry,
    state: &'a ResidentState,
    context: &'a WorkspaceContext,
    sampling: Option<TextSamplingInput<'a>>,
    parameters: Option<&'a dyn WorkspaceLayerwiseParameters>,
    unpriced_execution: Option<&'static str>,
    observation: Option<ObservationRef<'a, 'observer>>,
    trace: Option<EquationTraceRef<'a, 'trace>>,
    media: Option<MediaEquationRef<'a>>,
    input_dtype: Option<eredu_nn::workspace::WorkspaceDtype>,
    target_capture: bool,
    // An actual independent invocation can use a Prefill report span while
    // executing Decode. The span is a storage itinerary, not semantic authority.
    routed_pass: Option<eredu_runtime::ExpertPass>,
    external_target: Option<ExternalTargetQuote<'a>>,
}
impl ReplicatedTextArchitectureVisitor<WorkspaceBackend, ResidentState>
    for EquationVisitor<'_, '_, '_>
{
    type Output = EquationQuote;
    type Error = Error;
    fn construction_started(&mut self) {}
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        _store: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error> + 'static,
        A::StaticModules: Clone,
    {
        self.quote_modules(prepared.into_modules())
    }
}

impl EquationVisitor<'_, '_, '_> {
    fn execution_pass(&self,span:&InferenceWorkspaceSpan)->eredu_runtime::ExpertPass {
        self.routed_pass.unwrap_or(match span {
            InferenceWorkspaceSpan::Prefill(_)=>eredu_runtime::ExpertPass::Prefill,
            InferenceWorkspaceSpan::Decode{..}=>eredu_runtime::ExpertPass::Decode,
        })
    }

    fn construct(
        self,
        blueprint: &PreparedInferenceBlueprint,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        // Embedded capture still requires its actual selected extension. Plain
        // target quotation also accepts the exact all-absent source tuple.
        if self.target_capture && blueprint.selected().prediction_extension().is_none() {
            return Err(PreparedExecutionError::PredictionSourceMismatch);
        }
        let routes = PreparedExecutionRoutes::new()
            .with_replicated(ReplicatedRoute::<WorkspaceBackend, _>::new(
                self.context,
                self.context,
                SharedReplicatedTextVisitor::<WorkspaceResidentStateFactory, _>::new(self),
            ))
            .with_routed(RoutedRoute::<
                WorkspaceBackend,
                ResidentState,
                ResidentState,
                _,
                _,
                _,
            >::new(
                self.context, self.context, self, self, self
            ))
            .with_composite(CompositeRoute::<WorkspaceBackend, ResidentState, _>::new(
                self.context,
                self.context,
                self,
            ));
        construct_prepared_execution_impl(
            blueprint.sources.clone(),
            None::<()>,
            routes,
            QuoteAssembler(
                blueprint
                    .selected()
                    .text_realization()
                    .state()
                    .floating_dtype(),
                self.context,
            ),
            // Both plain text and embedded target capture execute only the
            // selected target. Extension presence does not add prediction work
            // or republish its already retained materialization placement.
            ConstructionPurpose::TargetEquations,
        )
    }

    fn with_bank_residency(mut self, residency: eredu_runtime::ParameterBankResidency) -> Self {
        if matches!(
            residency,
            eredu_runtime::ParameterBankResidency::IndependentCache(_)
        ) {
            self.unpriced_execution = Some(
                "addressable grouped execution needs selected gather, member-chunk and scatter workspace; resident equations alone are incomplete",
            );
        }
        self
    }
    fn quote_modules<A>(
        self,
        modules: crate::replicated_text::PreparedReplicatedTextModules<A>,
    ) -> Result<EquationQuote, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error> + 'static,
        A::StaticModules: Clone,
    {
        if self.media.is_some() || self.external_target.is_some() {
            return Err(workspace_message(
                self.context,
                format_args!("selected equation route lost its typed media capability"),
            ));
        }
        if self.state.layout() != modules.contract().selected().state().layout() {
            return Err(workspace_message(
                self.context,
                format_args!("workspace state projection differs from the selected layout"),
            ));
        }
        let mut runtime = EquationRuntime::from_prepared(
            modules,
            self.parameters,
            self.context,
            self.observation.map(|observer| observer.paths),
            self.target_capture,
        )?;
        let hook_bytes = runtime.observation_host_peak_bytes()?;
        self.quote_spans(hook_bytes, |tokens, state, demand, observer| {
            runtime
                .forward_with_capture(
                    A::text_input(tokens, None),
                    state,
                    self.context,
                    demand,
                    observer,
                    self.target_capture,
                )
                .map(|(scores, capture)| (scores, capture.map(EquationCapture::Embedded)))
        })
    }

    fn quote_routed_modules<A>(
        self,
        modules: crate::replicated_text::PreparedReplicatedTextModules<A>,
        mut provider:EquationRoutedProvider,
    ) -> Result<EquationQuote, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState> + 'static,
        A::StaticModules: Clone,
    {
        if self.media.is_some() || self.external_target.is_some() {
            return Err(workspace_message(
                self.context,
                format_args!("selected equation route lost its typed media capability"),
            ));
        }
        if self.state.layout() != modules.contract().selected().state().layout() {
            return Err(workspace_message(
                self.context,
                format_args!("workspace state projection differs from the selected layout"),
            ));
        }
        let mut runtime = EquationRuntime::from_prepared(
            modules,
            self.parameters,
            self.context,
            self.observation.map(|observer| observer.paths),
            self.target_capture,
        )?;
        let hook_bytes = runtime.observation_host_peak_bytes()?;
        self.quote_spans_with_span(hook_bytes, |tokens, state, demand, observer, span| {
            runtime
                .forward_routed_with_capture(
                    A::text_input(tokens, None),
                    state,
                    self.context,
                    demand,
                    observer,
                    self.target_capture,
                    self.execution_pass(span),
                    &mut provider,
                )
                .map(|(scores, capture)| (scores, capture.map(EquationCapture::Embedded)))
        })
    }

    fn quote_spans(
        self,
        hook_bytes: u64,
        mut forward: impl FnMut(
            &WorkspaceTensor,
            &mut ResidentState,
            eredu_core::OutputDemand,
            Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
        )
            -> Result<(Option<WorkspaceTensor>, Option<EquationCapture>), Error>,
    ) -> Result<EquationQuote, Error> {
        self.quote_spans_with_span(hook_bytes, |tokens,state,demand,observer,_span|
            forward(tokens,state,demand,observer))
    }

    fn quote_spans_with_span(
        self,
        hook_bytes: u64,
        forward: impl FnMut(
            &WorkspaceTensor, &mut ResidentState, eredu_core::OutputDemand,
            Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
            &InferenceWorkspaceSpan,
        ) -> Result<(Option<WorkspaceTensor>, Option<EquationCapture>), Error>,
    ) -> Result<EquationQuote, Error> {
        self.quote_spans_with_output_observation(hook_bytes, false, forward)
    }

    // A retained partition callback owns the selected output observation before
    // publication. Everything else (spans, checkpoints, roots, sampling and
    // observer completion) remains in this same visitor.
    fn quote_spans_with_prepublication_observation(
        self,
        hook_bytes: u64,
        forward: impl FnMut(
            &WorkspaceTensor, &mut ResidentState, eredu_core::OutputDemand,
            Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
            &InferenceWorkspaceSpan,
        ) -> Result<(Option<WorkspaceTensor>, Option<EquationCapture>), Error>,
    ) -> Result<EquationQuote, Error> {
        self.quote_spans_with_output_observation(hook_bytes, true, forward)
    }

    fn quote_spans_with_output_observation(
        self,
        hook_bytes: u64,
        output_observed_in_forward: bool,
        mut forward: impl FnMut(
            &WorkspaceTensor,
            &mut ResidentState,
            eredu_core::OutputDemand,
            Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
            &InferenceWorkspaceSpan,
        )
            -> Result<(Option<WorkspaceTensor>, Option<EquationCapture>), Error>,
    ) -> Result<EquationQuote, Error> {
        self.context.charge_metadata(std::mem::size_of::<(Self,u64,bool)>()
            .checked_add(std::mem::size_of_val(&forward))
            .ok_or_else(||self.context.metadata_error(format_args!("observation adapter controls overflow")))?)?;
        let mut state = self.state.try_clone_workspace(self.context)?;
        let batch = i32::try_from(self.geometry.batch_size).map_err(|_| {
            workspace_message(
                self.context,
                format_args!("workspace batch exceeds tensor extent"),
            )
        })?;
        let mut score_source = sampling::SamplingScores::default();
        let equations =
            quote_inference_workspace_with_context(self.geometry, self.context, |span| {
                let (position, count, mut demand) = match span {
                    InferenceWorkspaceSpan::Prefill(chunk) => (
                        chunk.position,
                        chunk.input.end - chunk.input.start,
                        chunk.output,
                    ),
                    InferenceWorkspaceSpan::Decode {
                        position, output, ..
                    } => (*position, 1, *output),
                };
                validate_frontier_with_context(&state, position, self.context)?;
                let active = match self.observation {
                    Some(observation) => {
                        observation.begin_span(self.geometry, span, self.context)?
                    }
                    None => false,
                };
                if active
                    && self
                        .observation
                        .expect("active observation")
                        .requires_sequence_readout()
                {
                    demand = eredu_core::OutputDemand::Sequence;
                }
                if let Some(observation) = self.observation {
                    let mut opening = self.context.metadata_vec(0)?;
                    for value in state
                        .as_ref()
                        .iter()
                        .flat_map(|lane| lane.retained_values())
                    {
                        self.context.reserve_metadata_vec(&mut opening, 1)?;
                        opening.push(value.clone());
                    }
                    observation.append_retained(&mut opening, self.context)?;
                    self.context.begin_state_span(opening.iter())?;
                } else {
                    self.context.begin_state_span(
                        state
                            .as_ref()
                            .iter()
                            .flat_map(|lane| lane.retained_values()),
                    )?;
                }
                let mut checkpoint = self.context.metadata_vec(state.as_ref().len())?;
                for lane in state.as_ref() {
                    checkpoint.push(lane.checkpoint_for_transaction(self.context)?);
                }
                let count = i32::try_from(count).map_err(|_| {
                    workspace_message(
                        self.context,
                        format_args!("workspace input exceeds tensor extent"),
                    )
                })?;
                // External invocations already own the exact prepared token
                // source. Keep that source in the common span driver rather
                // than invent an allocation that the native reducer must skip.
                self.context.charge_metadata(std::mem::size_of::<(
                    Option<usize>, WorkspaceTensor, &WorkspaceTensor, [i32; 2],
                    Option<ExternalTargetQuote<'_>>,
                )>())?;
                let (input_operation, tokens) = match self.external_target {
                    Some(ExternalTargetQuote::Capture { input, .. }) => {
                        if input.shape() != [batch, count]
                            || !matches!(input.layout().dtype(),
                                eredu_nn::workspace::WorkspaceDtype::Int32
                                    | eredu_nn::workspace::WorkspaceDtype::Uint32)
                        {
                            return Err(self.context.metadata_error(format_args!(
                                "external prepared token source differs from the exact invocation")));
                        }
                        (None, input.clone())
                    }
                    _ => {
                        let operation = self.context.operation_count();
                        let tokens = prepared_text_tokens(
                            &[batch, count], self.input_dtype, self.context,
                        )?;
                        (Some(operation), tokens)
                    }
                };
                let (scores, capture) = if active {
                    let observation = self.observation.expect("active observation");
                    let mut observer = observation.observer.borrow_mut();
                    let (output, capture) =
                        forward(&tokens, &mut state, demand, Some(&mut **observer), span)?;
                    let output = if output_observed_in_forward {
                        if output.is_none() && demand != eredu_core::OutputDemand::StateOnly {
                            return Err(self.context.metadata_error(format_args!("partition output observation has no logits")));
                        }
                        observer.finish()?;
                        output
                    } else {
                        observed::finish_logits(&mut **observer, output, demand, self.context)?
                    };
                    (output, capture)
                } else {
                    forward(&tokens, &mut state, demand, None, span)?
                };
                // The native text contract indexes complete logits after observation.
                // Trace that actual view for observed sampling, including inactive
                // spans, so every possible sampling input has uniform row geometry.
                let scores = if self.observation.is_some() && self.sampling.is_some() {
                    scores
                        .map(|scores| observed::sampling_row(&scores, self.context))
                        .transpose()?
                } else {
                    scores
                };
                // A late transaction failure can require restoring all checkpoint
                // buffers while the forward graph remains live. Include that path
                // conservatively without changing the successful metadata frontier.
                let mut _rollback = self.context.metadata_vec(checkpoint.len())?;
                for lane in &checkpoint {
                    _rollback.push(lane.checkpoint_for_transaction(self.context)?);
                }
                let output_roots = usize::from(scores.is_some())
                    .checked_add(capture.as_ref().map_or(0, EquationCapture::root_count))
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                let output_population = if self.sampling.is_some() || self.trace.is_some() {
                    score_source.observe(scores.as_ref(), self.sampling.is_some(), self.context)?
                } else {
                    Some(eredu_nn::workspace::WorkspaceStoragePopulation::EMPTY)
                };
                // A target hidden capture is independent of vocabulary demand.
                // Reduce the actual union, including aliasing with score backing;
                // keep both values alive through trace completion below.
                let output_population = match capture.as_ref() {
                    Some(capture) => {
                        let mut values = self.context.metadata_vec(output_roots)?;
                        if let Some(scores) = scores.as_ref() {
                            values.push(scores.clone());
                        }
                        capture.visit_values(&mut |value| values.push(value.clone()));
                        Some(self.context.report_scalars(&values)?.closing_storage)
                    }
                    None => output_population,
                };
                validate_frontier_with_context(&state, position + count as u64, self.context)?;
                let mut roots = Vec::new();
                for value in state
                    .as_ref()
                    .iter()
                    .flat_map(|lane| lane.retained_values())
                {
                    self.context.reserve_metadata_vec(&mut roots, 1)?;
                    roots.push(value.clone());
                }
                let mut report = self.context.finish_report(&roots)?;
                if active {
                    report = observed::with_hook_workspace(report, hook_bytes, self.context)?;
                }
                if let Some(reason) = self.unpriced_execution {
                    // Resident equations do not bound a distinct addressable-bank
                    // execution strategy. Preserve diagnostics, never completeness.
                    report
                        .state
                        .as_mut()
                        .expect("state span opened")
                        .transient_bytes = None;
                    report.tensor_buffers.transient_bytes = None;
                    report.host_workspace_bytes = None;
                    self.context
                        .reserve_metadata_vec(&mut report.assumptions, 1)?;
                    report
                        .assumptions
                        .push(self.context.metadata_string(format_args!("{reason}"))?);
                }
                if active {
                    self.observation
                        .expect("active observation")
                        .observer
                        .borrow_mut()
                        .end_span(span, self.context)?;
                }
                if let Some(trace) = self.trace {
                    let mut observer = trace.0.borrow_mut();
                    match capture.as_ref() {
                        Some(EquationCapture::Embedded(capture)) => observer
                            .observe_target_with_storage(
                                span,
                                &report,
                                roots.len(),
                                output_roots,
                                input_operation.ok_or_else(|| self.context.metadata_error(
                                    format_args!("embedded target lost its initialization operation")))?,
                                output_population,
                                capture,
                                scores.as_ref(),
                            )?,
                        Some(EquationCapture::External(capture)) => observer
                            .observe_external_target_with_storage(
                                span,
                                &report,
                                roots.len(),
                                output_roots,
                                input_operation,
                                output_population,
                                capture,
                                scores.as_ref(),
                            )?,
                        // This branch owns the ordinary I32/U32 token
                        // constructor recorded above. Provider-backed expert
                        // regions do not turn those tokens into a prepared
                        // media source or change the state's opening union.
                        None => observer.observe_with_storage(
                            span,
                            &report,
                            roots.len(),
                            output_roots,
                            input_operation.ok_or_else(|| self.context.metadata_error(
                                format_args!("ordinary text equation lost its token input source")))?,
                            output_population,
                        )?,
                    }
                }
                Ok::<_, Error>(report)
            })
            .map_err(|error| {
                if self.context.uses_checked_metadata() {
                    match error {
                        eredu_runtime::working_memory::InferenceWorkspaceError::Metadata(cause) => {
                            cause
                        }
                        cause => self.context.metadata_source(cause),
                    }
                } else if self.observation.is_some() {
                    Error::backend_source(error)
                } else {
                    Error::backend(error.to_string())
                }
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

impl crate::routed_text::RoutedTextArchitectureVisitor<WorkspaceBackend, ResidentState>
    for EquationVisitor<'_, '_, '_>
{
    type Output = EquationQuote;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: crate::routed_text::PreparedRoutedTextArchitecture<A>,
        _: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::StaticModules: Clone,
    {
        let (modules, residency, banks) = prepared.into_shared_parts();
        let provider=EquationRoutedProvider::new(banks,residency,self.context)?;
        self.quote_routed_modules(modules,provider)
    }
}
impl crate::routed_text::Relu2RoutedTextArchitectureVisitor<WorkspaceBackend, ResidentState>
    for EquationVisitor<'_, '_, '_>
{
    type Output = EquationQuote;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: crate::routed_text::PreparedRoutedTextArchitecture<A>,
        _: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::StaticModules: Clone,
    {
        let (modules, residency, banks) = prepared.into_shared_parts();
        let provider=EquationRoutedProvider::new(banks,residency,self.context)?;
        self.quote_routed_modules(modules,provider)
    }
}

type EquationQuote = (InferenceWorkspaceReport, Option<SamplingWorkspaceReport>);

/// Equation and ordinary sampling costs for one exact prepared request.
#[derive(Debug, Clone)]
pub struct PreparedTextGenerationWorkspace {
    /// All prompt chunks and cached decode equations, including score storage.
    pub equations: InferenceWorkspaceReport,
    /// All committed sampling steps and the sampler's retained payloads.
    pub sampling: SamplingWorkspaceReport,
}

impl PreparedTextGenerationWorkspace {
    /// Composes fresh host-token preparation, all equation and sampling spans,
    /// selected decoder backing and the cold controller payload. The caller
    /// must still supply materialization, capture, snapshot and other enclosing
    /// bounds. This does not register existing weights or reserve a domain.
    pub fn compose_preparation(
        &self,
        state: eredu_core::RuntimeStateEstimate,
        prompt: &eredu_runtime::working_memory::TextPromptWorkspaceReport,
        controller: eredu_core::TextControllerWorkspace<'_>,
        outside: eredu_core::ExecutionWorkspaceEstimate,
    ) -> Result<eredu_core::RuntimeStateEstimate, eredu_core::CapabilityError> {
        self.compose_preparation_metadata(
            state,
            prompt,
            controller,
            outside,
            eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary(),
        )
        .map_err(eredu_runtime::working_memory::WorkspaceReportError::into_capability)
    }
    /// Ordinary enclosing report adapter over the same composition worker.
    pub fn enclosing_preparation_workspace(
        &self,
        prompt: &eredu_runtime::working_memory::TextPromptWorkspaceReport,
        controller: eredu_core::TextControllerWorkspace<'_>,
        outside: eredu_core::ExecutionWorkspaceEstimate,
    ) -> Result<eredu_core::ExecutionWorkspaceEstimate, eredu_core::CapabilityError> {
        self.enclosing_preparation_workspace_metadata(
            prompt,
            controller,
            outside,
            eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary(),
        )
        .map_err(eredu_runtime::working_memory::WorkspaceReportError::into_capability)
    }
    /// Ordinary prompt/sampling adapter before controller source composition.
    pub fn enclosing_preparation_without_controller(
        &self,
        prompt: &eredu_runtime::working_memory::TextPromptWorkspaceReport,
        outside: eredu_core::ExecutionWorkspaceEstimate,
    ) -> Result<eredu_core::ExecutionWorkspaceEstimate, eredu_core::CapabilityError> {
        self.enclosing_preparation_without_controller_metadata(
            prompt,
            outside,
            eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary(),
        )
        .map_err(eredu_runtime::working_memory::WorkspaceReportError::into_capability)
    }
    /// The same prompt/controller/equation composition with counted destinations.
    pub fn compose_preparation_metadata(
        &self,
        state: eredu_core::RuntimeStateEstimate,
        prompt: &eredu_runtime::working_memory::TextPromptWorkspaceReport,
        controller: eredu_core::TextControllerWorkspace<'_>,
        outside: eredu_core::ExecutionWorkspaceEstimate,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<eredu_core::RuntimeStateEstimate, eredu_runtime::working_memory::WorkspaceReportError>
    {
        let outside =
            self.enclosing_preparation_workspace_metadata(prompt, controller, outside, metadata)?;
        let state = self
            .equations
            .refine_state_backing_metadata(state, metadata)?;
        self.equations.compose_metadata(state, outside, metadata)
    }

    /// Complete enclosing preparation/controller/sampling contribution, without
    /// equation storage. Runtime can pair this same composition with either full
    /// state diagnostics or a registered-root residual equation proof.
    pub fn enclosing_preparation_workspace_metadata(
        &self,
        prompt: &eredu_runtime::working_memory::TextPromptWorkspaceReport,
        controller: eredu_core::TextControllerWorkspace<'_>,
        mut outside: eredu_core::ExecutionWorkspaceEstimate,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<
        eredu_core::ExecutionWorkspaceEstimate,
        eredu_runtime::working_memory::WorkspaceReportError,
    > {
        if let eredu_core::WorkspaceBound::Bounded { bytes, assumptions } = &mut outside.retained {
            *bytes = bytes.checked_add(controller.additional_host_bytes).ok_or(
                eredu_core::AdmissionPolicyError::ArithmeticOverflow {
                    operation: "controller and retained request payload",
                },
            )?;
            metadata.append(
                assumptions,
                "; complete cold controller payload beyond the emitted filter priced by sampling",
            )?;
        }
        self.enclosing_preparation_without_controller_metadata(prompt, outside, metadata)
    }

    /// Composes prompt preparation and sampling before adding controller storage.
    /// Runtime can pair this component with an identity-bound controller
    /// contribution to preserve full diagnostics while crediting registered
    /// shared sources. This component alone is not a complete controller quote.
    pub fn enclosing_preparation_without_controller_metadata(
        &self,
        prompt: &eredu_runtime::working_memory::TextPromptWorkspaceReport,
        outside: eredu_core::ExecutionWorkspaceEstimate,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<
        eredu_core::ExecutionWorkspaceEstimate,
        eredu_runtime::working_memory::WorkspaceReportError,
    > {
        self.enclosing_sampling_workspace_metadata(
            prompt.compose_metadata(outside, metadata)?,
            metadata,
        )
    }

    /// Composes both inspected components with mandatory enclosing costs.
    /// Unknown preparation/materialization/copy/observation bounds stay unknown.
    pub fn compose(
        &self,
        state: eredu_core::RuntimeStateEstimate,
        outside: eredu_core::ExecutionWorkspaceEstimate,
    ) -> Result<eredu_core::RuntimeStateEstimate, eredu_core::CapabilityError> {
        self.equations
            .compose(state, self.enclosing_sampling_workspace(outside)?)
    }

    /// Adds the existing populated sampling report to complete enclosing costs.
    /// This excludes equation storage, allowing runtime to compose the same
    /// sampling term with a sealed registered-copy preparation contribution.
    pub fn enclosing_sampling_workspace(
        &self,
        outside: eredu_core::ExecutionWorkspaceEstimate,
    ) -> Result<eredu_core::ExecutionWorkspaceEstimate, eredu_core::CapabilityError> {
        self.enclosing_sampling_workspace_metadata(
            outside,
            eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary(),
        )
        .map_err(eredu_runtime::working_memory::WorkspaceReportError::into_capability)
    }

    /// Uses the same sampling composition with the enclosing Context's owning
    /// diagnostic destinations. This changes neither workspace nor source credit.
    pub fn enclosing_sampling_workspace_metadata(
        &self,
        outside: eredu_core::ExecutionWorkspaceEstimate,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<
        eredu_core::ExecutionWorkspaceEstimate,
        eredu_runtime::working_memory::WorkspaceReportError,
    > {
        self.sampling
            .enclosing_workspace_metadata(outside, metadata)
    }
}

struct QuoteAssembler<'a>(
    Option<eredu_runtime::StateStorageDtype>,
    &'a WorkspaceContext,
);
impl PreparedExecutableAssembler<()> for QuoteAssembler<'_> {
    type Executable = EquationQuote;
    type Output = EquationQuote;
    type Error = Error;
    fn floating_state_dtype(
        &mut self,
        _: &FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, Error> {
        // This is the retained native representation, not the metadata tensor's
        // conservative f32 arithmetic. Stateless selections may omit it.
        Ok(self.0.unwrap_or(eredu_runtime::StateStorageDtype::F32))
    }
    fn validate_communication(&mut self, _: &CommunicationManifest, _: &()) -> Result<(), Error> {
        Err(workspace_message(
            self.1,
            format_args!("replicated equation inspection has no communication"),
        ))
    }
    fn finish(
        self,
        parts: PreparedExecutableParts<Self::Executable, ()>,
    ) -> Result<Self::Output, Error> {
        Ok(parts.into_executable())
    }
}

struct EquationTraversal;
impl<C> eredu_runtime::LayeredTraversalHook<WorkspaceBackend, C, Error> for EquationTraversal {}

fn validate_frontier(state: &ResidentState, position: u64) -> Result<(), Error> {
    validate_frontier_impl(state, position, || {
        Error::backend("workspace state frontier differs from the scheduled span")
    })
}

fn validate_frontier_with_context(
    state: &ResidentState,
    position: u64,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    validate_frontier_impl(state, position, || {
        workspace_message(
            context,
            format_args!("workspace state frontier differs from the scheduled span"),
        )
    })
}

fn preparation_message(
    context: &WorkspaceContext,
    args: std::fmt::Arguments<'_>,
) -> PreparedExecutionError<Error> {
    if context.uses_checked_metadata() {
        PreparedExecutionError::Metadata(context.metadata_error(args))
    } else {
        PreparedExecutionError::Architecture(args.to_string())
    }
}

fn workspace_message(context: &WorkspaceContext, args: std::fmt::Arguments<'_>) -> Error {
    if context.uses_checked_metadata() {
        context.metadata_error(args)
    } else {
        Error::backend(args.to_string())
    }
}

fn validate_frontier_impl(
    state: &ResidentState,
    position: u64,
    failure: impl Fn() -> Error,
) -> Result<(), Error> {
    for (index, lane) in state.as_ref().iter().enumerate() {
        if !matches!(
            state.layout().layer(index),
            Some(eredu_core::cache::LayerCachePolicy::NoState)
        ) && u64::try_from(lane.position()).ok() != Some(position)
        {
            return Err(failure());
        }
    }
    Ok(())
}

// Source-specific external/assistant inputs take precedence over the selected
// ordinary token producer. Portable contexts without that producer preserve
// their existing I32 caller convention. The worker validates integer dtype.
fn prepared_text_tokens(shape:&[i32],explicit:Option<eredu_nn::workspace::WorkspaceDtype>,
    context:&WorkspaceContext)->Result<WorkspaceTensor,Error> {
    context.charge_metadata(std::mem::size_of::<[Option<eredu_nn::workspace::WorkspaceDtype>;2]>()
        +std::mem::size_of::<eredu_nn::workspace::WorkspaceDtype>())?;
    let dtype=explicit.or_else(||context.prepared_text_input_dtype())
        .unwrap_or(eredu_nn::workspace::WorkspaceDtype::Int32);
    WorkspaceTensor::prepared_token_input(shape,dtype,context)
}
