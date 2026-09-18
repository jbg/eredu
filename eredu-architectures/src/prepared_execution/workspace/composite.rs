use super::*;
use crate::{
    composite_execution::{
        CompositeArchitecture, PreparedCompositeArchitecture, PreparedCompositeInput,
    },
    replicated_text::{
        CompositeTextArchitectureVisitor, PreparedCompositeTextArchitecture,
        PreparedRoutedCompositeTextArchitecture,
    },
};
use eredu_runtime::{
    PreparedInputInspector, PreparedInputPart, PreparedInputPayload, PreparedModelInput,
};

/// Text quoting needs tensor geometry only. Media metadata is not fabricated.
pub(super) struct TextInspector;
impl TextInspector {
    fn dtype(tensor: &WorkspaceTensor) -> eredu_core::checkpoint::TensorDtype {
        use eredu_core::checkpoint::TensorDtype;
        use eredu_nn::workspace::WorkspaceDtype;
        match tensor.layout().dtype() {
            WorkspaceDtype::Bool => TensorDtype::Bool,
            WorkspaceDtype::Int32 => TensorDtype::I32,
            WorkspaceDtype::Uint32 => TensorDtype::U32,
            WorkspaceDtype::Uint8 => TensorDtype::U8,
            WorkspaceDtype::Float32 => TensorDtype::F32,
        }
    }
    pub(super) fn identity_with_metadata(
        tensor: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<eredu_core::InputTensorIdentity, Error> {
        context.charge_metadata(std::mem::size_of::<(
            eredu_core::InputTensorIdentity,
            Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError>,
        )>())?;
        let mut shape = context.metadata_vec(tensor.shape().len())?;
        shape.extend(tensor.shape().iter().map(|&n| n as usize));
        eredu_core::InputTensorIdentity::new(Self::dtype(tensor), shape)
            .map_err(|cause| context.metadata_source(cause))
    }
}
impl PreparedInputInspector<WorkspaceTensor> for TextInspector {
    fn identity(
        &self,
        tensor: &WorkspaceTensor,
    ) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
        eredu_core::InputTensorIdentity::new(
            Self::dtype(tensor),
            tensor.shape().iter().map(|&n| n as usize).collect(),
        )
    }
    fn identity_with_metadata(
        &self, tensor: &WorkspaceTensor, context: &WorkspaceContext,
    ) -> Result<eredu_core::InputTensorIdentity, Error> {
        Self::identity_with_metadata(tensor, context)
    }
    fn i32_values_with_metadata(
        &self, _: &WorkspaceTensor, context: &WorkspaceContext,
    ) -> Result<Vec<i32>, Error> {
        Err(metadata_unavailable_with_context(context))
    }
    fn bool_values_with_metadata(
        &self, _: &WorkspaceTensor, context: &WorkspaceContext,
    ) -> Result<Vec<bool>, Error> {
        Err(metadata_unavailable_with_context(context))
    }
    fn i32_values(&self, _: &WorkspaceTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        Err(metadata_unavailable())
    }
    fn bool_values(&self, _: &WorkspaceTensor) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        Err(metadata_unavailable())
    }
}
fn metadata_unavailable() -> eredu_core::CapabilityError {
    eredu_core::CapabilityError::InvalidConfiguration {
        field: "workspace_input",
        detail: "text equation inspection has no evaluated media metadata".into(),
    }
}

fn metadata_unavailable_with_context(context: &WorkspaceContext) -> Error {
    match context.metadata_string(format_args!("text equation inspection has no evaluated media metadata")) {
        Ok(detail) => context.metadata_source(eredu_core::CapabilityError::InvalidConfiguration {
            field: "workspace_input", detail,
        }),
        Err(cause) => cause,
    }
}

impl EquationVisitor<'_, '_, '_> {
    fn quote_composite<A>(
        self,
        architecture: PreparedCompositeArchitecture<A>,
        admission: A::AdmissionConfig,
        layout: &eredu_runtime::StateLayout,
        mut provider:EquationRoutedProvider,
    ) -> Result<EquationQuote, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend,ResidentState> + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        if self.media.is_some() {
            return Err(Error::backend(
                "selected equation route lost its typed media capability",
            ));
        }
        if self.state.layout() != layout {
            return Err(Error::backend(
                "workspace state projection differs from the selected composite layout",
            ));
        }
        let mut runtime = EquationRuntime::new(
            architecture,
            self.parameters,
            self.context,
            self.external_target.and_then(|source| source.paths()).or_else(|| self.observation.map(|observer| observer.paths)),
            self.target_capture,
        )?;
        if let Some(ExternalTargetQuote::Static(operation))=self.external_target {
            return self.quote_external_static(&mut runtime,operation);
        }
        let hook_bytes = runtime.observation_host_peak_bytes(self.context)?;
        self.quote_spans_with_span(hook_bytes, |tokens, state, demand, observer,span| {
            self.context.charge_metadata(std::mem::size_of::<(
                PreparedInputPart<WorkspaceTensor>,
                PreparedModelInput<WorkspaceTensor>,
                crate::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
                PreparedCompositeInput<'_, WorkspaceTensor, A::InputPartPlan>,
            )>())?;
            let part = PreparedInputPart::new(
                eredu_core::InputModality::Text,
                PreparedInputPayload::TokenIds(tokens.clone()),
                [],
            )
            .map_err(|cause| self.context.metadata_source(cause))?;
            let mut parts = self.context.metadata_vec(1)?;
            parts.push(part);
            let input = PreparedModelInput::new_with_metadata(parts, self.context, |tensor| {
                TextInspector::identity_with_metadata(tensor, self.context)
            })?;
            let admitted = A::admit_prepared_input_with_metadata(
                &admission, &input, &TextInspector, self.context,
            )?;
            let input = PreparedCompositeInput::new_with_diagnostic(&input, &admitted, |message| {
                self.context.metadata_error(format_args!("{message}"))
            })?;
            self.forward_composite_input(&mut runtime, input, state, demand, observer, span, &mut provider)
        })
    }
    fn forward_composite_input<A>(
        &self, runtime: &mut EquationRuntime<PreparedCompositeArchitecture<A>>,
        input: PreparedCompositeInput<'_, WorkspaceTensor, A::InputPartPlan>,
        state: &mut ResidentState, demand: eredu_core::OutputDemand,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
        span: &InferenceWorkspaceSpan, provider: &mut EquationRoutedProvider,
    ) -> Result<(Option<WorkspaceTensor>, Option<EquationCapture>), Error>
    where A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState> + 'static,
        A::InputPartPlan: 'static, A::StaticModules: Clone,
    {
            if let Some(ExternalTargetQuote::Capture{request,..}) = self.external_target {
                // External target observation is the same passive path capture
                // used by its ordinary target worker. Assistant observations
                // consume the resulting semantic capture in the shared lifecycle.
                let mut capture = external::CaptureObserver::new::<A>(request, self.context)?;
                let (scores, forward) = runtime.forward_routed_with_context(
                    input, state, self.context, demand, Some(&mut capture),
                    self.execution_pass(span),
                    provider,
                )?;
                let values = capture.into_values()?;
                let captured = A::external_prediction_capture_with_metadata(
                    request, &forward, values, self.context,
                )?.ok_or_else(|| self.context.metadata_error(format_args!(
                    "selected target omitted its external assistant capture"
                )))?;
                Ok((scores, Some(EquationCapture::External(captured))))
            } else {
                runtime.forward_routed_with_capture(input, state, self.context, demand, observer, self.target_capture,
                    self.execution_pass(span),provider)
                    .map(|(scores, capture)| (scores, capture.map(EquationCapture::Embedded)))
            }
    }

    /// Quotes the same whole composite input consumed by captured native prefill.
    pub(super) fn quote_composite_whole_media<A>(
        self,
        modules: crate::replicated_text::PreparedReplicatedTextModules<PreparedCompositeArchitecture<A>>,
        mut provider: EquationRoutedProvider,
    ) -> Result<EquationQuote, Error>
    where A: crate::composite_execution::CompositeMediaIngressArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState> + 'static,
        A::InputPartPlan: 'static, A::StaticModules: Clone,
    {
        if self.geometry.max_output_tokens != 0
            || self.geometry.input_positions != self.geometry.prefill_chunk_positions {
            return Err(self.context.metadata_error(format_args!("whole media capture requires its exact complete prefill span")));
        }
        let reference = self.media.expect("typed whole media route");
        let input = reference.input.borrow_mut().take().ok_or_else(||
            self.context.metadata_error(format_args!("media equation source already consumed")))?;
        self.context.charge_metadata(std::mem::size_of::<(
            A::IngressPlan, Result<A::IngressPlan, Error>,
            PreparedCompositeInput<'_, WorkspaceTensor, A::InputPartPlan>,
        )>())?;
        let plan = A::prepare_original_workspace_ingress_plan_with_metadata(input, self.geometry, self.context)?;
        let mut runtime = EquationRuntime::from_prepared(modules, self.parameters, self.context,
            self.external_target.and_then(|source| source.paths()).or_else(|| self.observation.map(|source| source.paths)),
            self.target_capture)?;
        let hook_bytes = runtime.observation_host_peak_bytes(self.context)?;
        self.quote_spans_with_span(hook_bytes, |_, state, demand, observer, span| {
            let InferenceWorkspaceSpan::Prefill(chunk) = span else {
                return Err(self.context.metadata_error(format_args!("whole media capture reached a non-prefill span")));
            };
            if chunk.input.start != 0 || chunk.input.end != self.geometry.input_positions {
                return Err(self.context.metadata_error(format_args!("whole media capture span differs from source")));
            }
            let input = A::prepared_ingress_input(&plan);
            self.forward_composite_input(&mut runtime, input, state, demand, observer, span, &mut provider)
        })
    }

}

impl EquationVisitor<'_, '_, '_> {
    #[inline(never)]
    fn quote_prepared_media<A>(self,prepared:PreparedCompositeTextArchitecture<A,A::AdmissionConfig>)
        ->Result<EquationQuote,Error>
    where A: CompositeArchitecture<WorkspaceBackend,ResidentState,Error=Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend,ResidentState>
            + crate::composite_execution::CompositeMediaIngressArchitecture<WorkspaceBackend,ResidentState>
            + 'static,
        A::InputPartPlan:'static,A::StaticModules:Clone,
    {
        let (modules,admission)=prepared.into_modules();
        self.quote_composite_media(modules,admission,EquationRoutedProvider::resident())
    }
}

impl CompositeTextArchitectureVisitor<WorkspaceBackend, ResidentState>
    for EquationVisitor<'_, '_, '_>
{
    type Output = EquationQuote;
    type Error = Error;
    fn construction_started(&mut self) {}
    fn visit_media<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + crate::composite_execution::CompositeMediaIngressArchitecture<
                WorkspaceBackend,
                ResidentState,
            > + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        if self.media.is_none() {
            return CompositeTextArchitectureVisitor::visit(self, prepared, store);
        }
        // Media-only module extraction has a large by-value handoff. Keep its
        // transports out of this chooser's frame while a text quote constructs
        // its resident units through visit(). The selected media behavior is
        // unchanged and its actual helper transports are charged before entry.
        self.context.charge_metadata(std::mem::size_of::<(
            Self, PreparedCompositeTextArchitecture<A,A::AdmissionConfig>,
            crate::replicated_text::PreparedReplicatedTextModules<PreparedCompositeArchitecture<A>>,
            A::AdmissionConfig, Result<EquationQuote,Error>,
        )>())?;
        self.quote_prepared_media(prepared)
    }
    fn visit_routed_media<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + crate::composite_execution::CompositeMediaIngressArchitecture<
                WorkspaceBackend,
                ResidentState,
            > + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        if self.media.is_none() {
            return self.visit_routed(prepared, store);
        }
        let (routed, _, admission) = prepared.into_parts();
        let (modules, residency, banks) = routed.into_shared_parts();
        let provider=EquationRoutedProvider::new(banks,residency,self.context)?;
        self.quote_composite_media(modules, admission,provider)
    }
    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        let (architecture, _, contract, _, admission) = prepared.into_parts();
        self.quote_composite(
            architecture,
            admission,
            contract.selected().state().layout(),
            EquationRoutedProvider::resident(),
        )
    }
    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        let (routed, _, admission) = prepared.into_parts();
        let (mut modules, residency, banks) = routed.into_shared_parts();
        let contract = modules.take_contract();
        let provider=EquationRoutedProvider::new(banks,residency,self.context)?;
        self.quote_composite(
            modules.take_architecture(),
            admission,
            contract.selected().state().layout(),
            provider,
        )
    }
}
