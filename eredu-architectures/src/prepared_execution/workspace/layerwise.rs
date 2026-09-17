//! The ordinary architecture traversal with explicit metadata unit population.

use super::*;
use eredu_nn::{ParameterId, Parameterized};
use eredu_runtime::{
    ExecutionUnitAddress, ExecutionUnitLayout, LayeredArchitecture, LayerwiseAcquireError,
    LayerwisePolicy, LayerwiseRuntime,
};
use std::{borrow::Cow, collections::BTreeMap};

/// Cold parameter backing from the exact selected host or disk population.
///
/// This provider must retain its prepared layout and binding identities for the
/// quote. Each returned map contains every authoritative unit parameter, with
/// its selected logical layout and conservative backing capacity in `context`.
/// Known shared storage uses shared metadata roots. A future per-name copy that
/// may allocate or alias may instead use an independent root with a capacity
/// covering both outcomes; such a prospective root grants no existing-storage
/// credit. No payload read, native
/// allocation, transfer, evaluation, or residency mutation is permitted here.
/// Parameter/window residency and transfer work are priced separately; returning
/// metadata values does not certify a complete materialization bound.
pub trait WorkspaceLayerwiseParameters {
    /// Exact group identifiers, unit counts and ordinal/address mapping.
    fn layout(&self) -> &ExecutionUnitLayout;

    /// Global architecture address associated with one local storage slot.
    /// The default preserves the ordinary unpartitioned unit layout.
    fn execution_address(&self, ordinal: usize) -> Option<ExecutionUnitAddress> {
        self.layout().address(ordinal)
    }

    /// Borrows the same selected source for allocation-free metadata count/fill.
    /// The default refuses without calling the ordinary allocating adapter.
    /// A successful loan certifies neither live residency nor request admission.
    fn parameter_source(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::WorkspaceParameterSourceLoan<'_>,
        eredu_runtime::working_memory::WorkspaceParameterSourceError,
    > {
        Err(eredu_runtime::working_memory::WorkspaceParameterSourceError::CompanionUnavailable)
    }

    /// Records one actual shared-driver acquisition against a retained source.
    /// This metadata-only observation grants no materialization or submission.
    /// The default leaves ordinary providers unchanged.
    fn observe_acquire(&self,_ordinal:usize,_address:ExecutionUnitAddress,
        _context:&WorkspaceContext)->Result<(),Error> {Ok(())}

    /// Complete selected parameter inventory for one canonical execution unit.
    /// Missing, extra or incompatible slots reject before its equations execute.
    fn parameters(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        context: &WorkspaceContext,
    ) -> Result<BTreeMap<ParameterId, WorkspaceTensor>, Error>;
}

pub(super) enum EquationRuntime<'a, A>
where
    A: LayeredArchitecture<WorkspaceBackend, ResidentState, Error = Error>,
{
    Resident(
        ResidentRuntime<A, WorkspaceBackend, ResidentState>,
        Option<eredu_runtime::PreparedLayeredObservationPaths>,
        // Retains the source-format modules until the actual runtime retires.
        #[allow(dead_code)] Option<A>,
    ),
    Layerwise(
        LayerwiseRuntime<A, WorkspaceBackend, ResidentState, WorkspaceLayerwisePolicy<'a>>,
        Option<eredu_runtime::PreparedLayeredObservationPaths>,
        // Retains the source-format modules until the actual runtime retires.
        #[allow(dead_code)] Option<A>,
    ),
}

impl<'a, A> EquationRuntime<'a, A>
where
    A: LayeredArchitecture<WorkspaceBackend, ResidentState, Error = Error>,
{
    pub(super) fn new(
        mut architecture: A,
        parameters: Option<&'a dyn WorkspaceLayerwiseParameters>,
        context: &WorkspaceContext,
        paths: Option<&eredu_runtime::SharedLayeredObservationPaths>,
        retain_target_capture: bool,
    ) -> Result<Self, Error> {
        Self::constructor_controls::<(
            A, Option<&'a dyn WorkspaceLayerwiseParameters>, &WorkspaceContext,
            Option<&eredu_runtime::SharedLayeredObservationPaths>,
            Result<Self, Error>, Option<eredu_runtime::PreparedLayeredObservationPaths>,
        )>(context)?;
        if retain_target_capture {
            architecture.retain_prediction_target_capture();
        }
        match parameters {
            None => Self::new_resident(architecture, context, paths),
            Some(parameters) => Self::new_layerwise(architecture, parameters, context, paths),
        }
    }
    #[inline(never)]
    fn new_resident(architecture: A, context: &WorkspaceContext,
        paths: Option<&eredu_runtime::SharedLayeredObservationPaths>) -> Result<Self, Error> {
        Self::constructor_controls::<(
            Result<ResidentRuntime<A,WorkspaceBackend,ResidentState>,Error>,
            ResidentRuntime<A,WorkspaceBackend,ResidentState>,
            Option<&eredu_runtime::SharedLayeredObservationPaths>,
            Option<eredu_runtime::PreparedLayeredObservationPaths>,
            Result<Self,Error>,
        )>(context)?;
        // The shared resident constructor recursively builds each actual unit.
        // Its Result extraction, observation binding and final enum transport
        // run only after that constructor returns, in the helper below.
        Self::finish_resident_result(ResidentRuntime::new_workspace(architecture,context),paths)
    }
    #[inline(never)]
    fn finish_resident_result(result:Result<ResidentRuntime<A,WorkspaceBackend,ResidentState>,Error>,
        paths:Option<&eredu_runtime::SharedLayeredObservationPaths>)->Result<Self,Error> {
        let runtime=result?;
        let binding=paths.map(|paths|runtime.bind_observation_paths(paths))
            .transpose().map_err(Error::backend_source)?;
        Ok(Self::Resident(runtime,binding,None))
    }
    #[inline(never)]
    fn new_layerwise(architecture: A, parameters: &'a dyn WorkspaceLayerwiseParameters,
        context: &WorkspaceContext, paths: Option<&eredu_runtime::SharedLayeredObservationPaths>) -> Result<Self, Error> {

                let graph = architecture.execution_graph()?;
                let mut counts = context.metadata_vec(graph.groups().len())?;
                for group in 0..graph.groups().len() {
                    counts.push(architecture.group_unit_count(group)?);
                }
                let layout = ExecutionUnitLayout::new_with_metadata(&graph, &counts, context)?;
                if parameters.layout() != &layout {
                    return Err(context.metadata_error(format_args!(
                        "workspace parameter population differs from the architecture unit layout"
                    )));
                }
                let policy = WorkspaceLayerwisePolicy::new(parameters, layout)?;
                let runtime = LayerwiseRuntime::new(architecture, policy);
                let binding = paths
                    .map(|paths| runtime.bind_observation_paths(paths))
                    .transpose()
                    .map_err(Error::backend_source)?;
                Ok(Self::Layerwise(runtime, binding, None))
    }

    pub(super) fn from_prepared(
        modules: crate::replicated_text::PreparedReplicatedTextModules<A>,
        parameters: Option<&'a dyn WorkspaceLayerwiseParameters>,
        context: &WorkspaceContext,
        paths: Option<&eredu_runtime::SharedLayeredObservationPaths>,
        retain_target_capture: bool,
    ) -> Result<Self, Error> {
        let controls = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<crate::replicated_text::PreparedReplicatedTextModules<A>>(),
            std::mem::size_of::<(
                A,
                Option<A>,
                eredu_runtime::PreparedReplicatedTextExecutionGeometry,
            )>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<bool>(),
        ];
        let amount = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(amount)?;
        Self::constructor_controls::<(
            A, Option<A>, eredu_runtime::PreparedReplicatedTextExecutionGeometry,
            Option<&'a dyn WorkspaceLayerwiseParameters>, &WorkspaceContext,
            Option<&eredu_runtime::SharedLayeredObservationPaths>,
            Result<Self, Error>, Option<eredu_runtime::PreparedLayeredObservationPaths>,
        )>(context)?;
        let (mut architecture, source_architecture, geometry) = modules.into_execution_parts();
        if retain_target_capture {
            architecture.retain_prediction_target_capture();
        }
        match parameters {
            None => Self::prepared_resident(architecture, source_architecture, geometry, context, paths),
            Some(parameters) => Self::prepared_layerwise(architecture, source_architecture, geometry, parameters, context, paths),
        }
    }
    // Both constructor selections pay the arguments/results of their one
    // selected helper before entering it. No heap object replaces these moves.
    fn constructor_controls<C>(context: &WorkspaceContext) -> Result<(), Error> {
        let bytes = std::mem::size_of::<C>()
            .checked_add(std::mem::size_of::<(&WorkspaceContext, usize, Result<(), Error>)>())
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(bytes)?;
        Ok(())
    }
    #[inline(never)]
    fn prepared_resident(architecture: A, source_architecture: Option<A>,
        geometry: eredu_runtime::PreparedReplicatedTextExecutionGeometry,
        context: &WorkspaceContext, paths: Option<&eredu_runtime::SharedLayeredObservationPaths>) -> Result<Self, Error> {

                let runtime = ResidentRuntime::new_workspace_with_prepared_geometry(
                    architecture,
                    geometry,
                    context,
                )?;
                let binding = paths
                    .map(|paths| runtime.bind_observation_paths(paths))
                    .transpose()
                    .map_err(|cause| context.metadata_source(cause))?;
                Ok(Self::Resident(runtime, binding, source_architecture))
    }
    #[inline(never)]
    fn prepared_layerwise(architecture: A, source_architecture: Option<A>,
        geometry: eredu_runtime::PreparedReplicatedTextExecutionGeometry,
        parameters: &'a dyn WorkspaceLayerwiseParameters, context: &WorkspaceContext,
        paths: Option<&eredu_runtime::SharedLayeredObservationPaths>) -> Result<Self, Error> {

                // The source owns its immutable layout; compare the moved
                // contract before retaining that existing borrow in the policy.
                if parameters.layout() != geometry.units() {
                    return Err(context.metadata_error(format_args!(
                        "workspace parameter population differs from the architecture unit layout"
                    )));
                }
                let policy = WorkspaceLayerwisePolicy {
                    parameters,
                    layout: Cow::Borrowed(parameters.layout()),
                    projection: None,
                };
                let runtime =
                    LayerwiseRuntime::new_with_prepared_geometry(architecture, policy, geometry);
                let binding = paths
                    .map(|paths| runtime.bind_observation_paths(paths))
                    .transpose()
                    .map_err(|cause| context.metadata_source(cause))?;
                Ok(Self::Layerwise(runtime, binding, source_architecture))
    }

    /// Same typed target operation used by the replicated prediction driver.
    /// This equation-only runtime has no native queue or transaction authority.
    pub(super) fn prediction_operation<O>(
        &mut self,
        state: &mut ResidentState,
        operation: O,
        context: &WorkspaceContext,
    ) -> Result<O::Output, Error>
    where O: eredu_runtime::PredictionTargetOperation<A, WorkspaceBackend, ResidentState>,
    {
        context.charge_metadata(std::mem::size_of::<(O, Result<O::Output, Error>, bool)>())?;
        match self {
            Self::Resident(runtime, _, _) => runtime.apply_prediction_target_operation(operation, state, None, context),
            Self::Layerwise(runtime, _, _) => runtime.apply_prediction_target_operation(operation, state, None, context),
        }
    }

    pub(super) fn prepared_graph(&self) -> Result<&eredu_runtime::ExecutionGraph, Error> {
        match self {
            Self::Resident(runtime, _, _) => runtime.workspace_prepared_execution_graph(),
            Self::Layerwise(runtime, _, _) => runtime.workspace_prepared_execution_graph(),
        }
    }

    pub(super) fn architecture(&self) -> &A {
        match self {
            Self::Resident(runtime, _, _) => runtime.architecture(),
            Self::Layerwise(runtime, _, _) => runtime.architecture(),
        }
    }
    pub(super) fn forward_media<H>(
        &mut self,
        source: &mut eredu_runtime::media_prefill::workspace::MediaEquationSource<A, ResidentState>,
        span: &eredu_runtime::prefill::PrefillChunk,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        hook: &mut H,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>
    where
        A: eredu_runtime::media_prefill::PrefillIngressArchitecture<
                WorkspaceBackend,
                ResidentState,
            >,
        H: eredu_runtime::LayeredTraversalHook<WorkspaceBackend, A::ForwardContext, Error> + ?Sized,
    {
        match (self, observer) {
            (Self::Resident(runtime, Some(paths), _), Some(observer)) =>
                source.forward_resident_observed(runtime, span, state, context, hook, observer, paths),
            (Self::Layerwise(runtime, Some(paths), _), Some(observer)) => source
                .forward_layerwise_observed(runtime, span, state, context, hook, observer, paths)
                .map_err(Error::backend_source),
            (_, Some(_)) => Err(context.metadata_error(format_args!(
                "media observation requires the runtime's prepared path source"))),
            (Self::Resident(runtime, _, _), None) => {
                source.forward_resident(runtime, span, state, context, hook)
            }
            (Self::Layerwise(runtime, _, _), None) => source
                .forward_layerwise(runtime, span, state, context, hook)
                .map_err(Error::backend_source),
        }
    }

    pub(super) fn forward_media_routed<P,H>(
        &mut self,
        source: &mut eredu_runtime::media_prefill::workspace::MediaEquationSource<A, ResidentState>,
        span: &eredu_runtime::prefill::PrefillChunk,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        hook: &mut H,
        provider:&mut P,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>
    where
        A: eredu_runtime::media_prefill::PrefillIngressArchitecture<
                WorkspaceBackend,
                ResidentState,
            >,
        A:eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend,ResidentState>,
        P:eredu_runtime::RoutedExpertProvider<WorkspaceBackend>,P::Error:std::fmt::Display,
        H: eredu_runtime::LayeredTraversalHook<WorkspaceBackend, A::ForwardContext, Error> + ?Sized,
    {
        match (self, observer) {
            (Self::Resident(runtime, Some(paths), _), Some(observer)) =>
                source.forward_resident_routed_observed(runtime, span, state, context, hook, provider, observer, paths),
            (Self::Layerwise(runtime, Some(paths), _), Some(observer)) => source
                .forward_layerwise_routed_observed(runtime, span, state, context, hook, provider, observer, paths)
                .map_err(Error::backend_source),
            (_, Some(_)) => Err(context.metadata_error(format_args!(
                "media observation requires the runtime's prepared path source"))),
            (Self::Resident(runtime, _, _), None) => {
                source.forward_resident_routed(runtime, span, state, context, hook,provider)
            }
            (Self::Layerwise(runtime, _, _), None) => source
                .forward_layerwise_routed(runtime, span, state, context, hook,provider)
                .map_err(Error::backend_source),
        }
    }

    pub(super) fn observation_host_peak_bytes(&self) -> Result<u64, Error> {
        let binding = match self {
            Self::Resident(_, binding, _) | Self::Layerwise(_, binding, _) => binding,
        };
        binding
            .as_ref()
            .map(|binding| {
                binding.traversal_host_peak_bytes().ok_or_else(|| {
                    Error::backend_source(
                        eredu_runtime::working_memory::InferenceObservationError::Overflow,
                    )
                })
            })
            .transpose()
            .map(|bytes| bytes.unwrap_or(0))
    }

    pub(super) fn forward<'input>(
        &mut self,
        input: A::Input<'input>,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        demand: eredu_core::OutputDemand,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        self.forward_with_capture(input, state, context, demand, observer, false)
            .map(|(scores, _)| scores)
    }

    pub(super) fn forward_with_capture<'input>(
        &mut self,
        input: A::Input<'input>,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        demand: eredu_core::OutputDemand,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
        capture: bool,
    ) -> Result<(Option<WorkspaceTensor>, Option<WorkspaceTensor>), Error> {
        let output = self.forward_with_context(
            input, state, context, demand,
            observer.map(|observer| observer as &mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor, Error>),
        )?;
        let (scores, forward) = output;
        let hidden = if capture {
            Some(
                A::prediction_target_capture(&forward)
                    .ok_or_else(|| {
                        workspace_message(
                            context,
                            format_args!(
                                "selected target equation {} did not retain its prediction capture",
                                std::any::type_name::<A>(),
                            ),
                        )
                    })?
                    .clone(),
            )
        } else {
            None
        };
        Ok((scores, hidden))
    }

    pub(super) fn forward_routed_with_capture<'input,P>(
        &mut self, input: A::Input<'input>, state: &mut ResidentState,
        context: &WorkspaceContext, demand: eredu_core::OutputDemand,
        observer: Option<&mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver>,
        capture: bool, pass: eredu_runtime::ExpertPass, provider:&mut P,
    ) -> Result<(Option<WorkspaceTensor>, Option<WorkspaceTensor>), Error>
    where A: eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>,
        P:eredu_runtime::RoutedExpertProvider<WorkspaceBackend>,P::Error:std::fmt::Display,
    {
        context.charge_metadata(std::mem::size_of::<(
            A::ForwardContext, &mut P, eredu_runtime::ExpertPass,
            Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>,
            (Option<WorkspaceTensor>, Option<WorkspaceTensor>), [WorkspaceTensor; 2], bool,
        )>())?;
        let (scores,forward)=self.forward_routed_with_context(input,state,context,demand,
            observer.map(|observer| observer as &mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor,Error>),
            pass,provider)?;
        let hidden = if capture {
            Some(A::prediction_target_capture(&forward).ok_or_else(|| workspace_message(context,
                format_args!("routed target equation did not retain prediction capture")))?.clone())
        } else { None };
        Ok((scores, hidden))
    }

    pub(super) fn forward_routed_with_context<'input,P>(
        &mut self, input:A::Input<'input>, state:&mut ResidentState,
        context:&WorkspaceContext, demand:eredu_core::OutputDemand,
        observer:Option<&mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor,Error>>,
        pass:eredu_runtime::ExpertPass, provider:&mut P,
    )->Result<(Option<WorkspaceTensor>,A::ForwardContext),Error>
    where A:eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend,ResidentState>,
        P:eredu_runtime::RoutedExpertProvider<WorkspaceBackend>,P::Error:std::fmt::Display,
    {
        context.charge_metadata(std::mem::size_of::<(
            A::ForwardContext, &mut P, eredu_runtime::ExpertPass,
            Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>,
            Option<&mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor,Error>>,
        )>())?;
        context.charge_metadata(
            eredu_runtime::ResidentExpertProvider::observation_control_bytes::<WorkspaceTensor>()
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        let output = match (self,observer) {
            (Self::Resident(runtime, Some(paths), _),Some(observer)) => runtime
                .forward_serial_routed_with_prepared_paths(
                    input, state, pass, provider, context, observer, paths, demand)
                .map_err(|cause|context.metadata_source(cause))?,
            (Self::Layerwise(runtime, Some(paths), _),Some(observer)) => runtime
                .forward_serial_routed_with_prepared_paths(
                    input, state, pass, provider, context, observer, paths, demand)
                .map_err(|cause| context.metadata_source(cause))?,
            (Self::Resident(runtime,_,_),None)=>runtime.forward_serial_routed_with_traversal_hook_with_readout(
                input,state,pass,provider,context,&mut EquationTraversal,demand).map_err(|cause|context.metadata_source(cause))?,
            (Self::Layerwise(runtime,_,_),None)=>runtime.forward_serial_routed_with_traversal_hook_with_readout(
                input,state,pass,provider,context,&mut EquationTraversal,demand).map_err(|cause|context.metadata_source(cause))?,
            _ => return Err(workspace_message(context,
                format_args!("observed routed equation runtime lacks its prepared path binding"))),
        };
        Ok(output)
    }

    pub(super) fn forward_with_context<'input>(
        &mut self,
        input: A::Input<'input>,
        state: &mut ResidentState,
        context: &WorkspaceContext,
        demand: eredu_core::OutputDemand,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor, Error>>,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), Error> {
        context.charge_metadata(std::mem::size_of::<(
            A::ForwardContext,
            Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>,
            (Option<WorkspaceTensor>, Option<WorkspaceTensor>),
            [WorkspaceTensor; 2],
            bool,
        )>())?;
        let output = if let Some(observer) = observer {
            match self {
                Self::Resident(runtime, Some(paths), _) => runtime
                    .forward_with_prepared_observer_and_context_with_readout(
                        input, state, context, observer, paths, demand,
                    )
                    .map_err(|cause| {
                        if context.uses_checked_metadata() {
                            context.metadata_source(cause)
                        } else {
                            Error::backend_source(cause)
                        }
                    }),
                Self::Layerwise(runtime, Some(paths), _) => runtime
                    .forward_with_prepared_observer_and_context_with_readout(
                        input, state, context, observer, paths, demand,
                    )
                    .map_err(|cause| {
                        if context.uses_checked_metadata() {
                            context.metadata_source(cause)
                        } else {
                            Error::backend_source(cause)
                        }
                    }),
                _ => Err(workspace_message(
                    context,
                    format_args!("observed equation runtime lacks its prepared path binding"),
                )),
            }
        } else {
            match self {
                Self::Resident(runtime, _, _) => runtime.forward_with_traversal_hook_with_readout(
                    input,
                    state,
                    context,
                    &mut EquationTraversal,
                    demand,
                ),
                Self::Layerwise(runtime, _, _) => runtime
                    .forward_with_traversal_hook_with_readout(
                        input,
                        state,
                        context,
                        &mut EquationTraversal,
                        demand,
                    )
                    .map_err(|cause| {
                        if context.uses_checked_metadata() {
                            context.metadata_source(cause)
                        } else {
                            Error::backend(cause)
                        }
                    }),
            }
        }?;
        Ok(output)
    }

}

pub(crate) struct WorkspaceLayerwisePolicy<'a> {
    parameters: &'a dyn WorkspaceLayerwiseParameters,
    layout: Cow<'a, ExecutionUnitLayout>,
    projection: Option<eredu_runtime::working_memory::WorkspaceParameterProjection<'a>>,
}

impl<'a> WorkspaceLayerwisePolicy<'a> {
    pub(super) fn new(
        parameters: &'a dyn WorkspaceLayerwiseParameters,
        layout: ExecutionUnitLayout,
    ) -> Result<Self, Error> {
        if parameters.layout() != &layout {
            return Err(Error::backend(
                "workspace parameter population differs from the architecture unit layout",
            ));
        }
        Ok(Self {
            parameters,
            layout: Cow::Owned(layout),
            projection: None,
        })
    }

    pub(crate) fn for_layout(parameters:&'a dyn WorkspaceLayerwiseParameters,
        layout:&ExecutionUnitLayout,context:&WorkspaceContext)->Result<Self,Error> {
        context.charge_metadata(std::mem::size_of::<(Self,Result<Self,Error>,
            &dyn WorkspaceLayerwiseParameters,&ExecutionUnitLayout,&WorkspaceContext)>())?;
        if parameters.layout()!=layout {
            return Err(context.metadata_error(format_args!("workspace parameter source differs from the constructor unit layout")));
        }
        Ok(Self {parameters,layout:Cow::Borrowed(parameters.layout()),projection:None})
    }

    pub(super) fn for_partition<A>(parameters:&'a dyn WorkspaceLayerwiseParameters,
        architecture:&A, addresses:&[ExecutionUnitAddress], context:&WorkspaceContext)->Result<Self,Error>
    where A:LayeredArchitecture<WorkspaceBackend,ResidentState,Error=Error> {
        context.charge_metadata(std::mem::size_of::<(Self,Result<Self,Error>,
            &A,&[ExecutionUnitAddress],eredu_runtime::ExecutionGraph,Vec<usize>,ExecutionUnitLayout,
            Result<ExecutionUnitLayout,Error>,usize)>())?;
        let graph=architecture.execution_graph_with_metadata(context)?.into_owned_with_metadata(context)?;
        let mut counts=context.metadata_vec(graph.groups().len())?;
        counts.resize(graph.groups().len(),0usize);
        for (ordinal,&address) in addresses.iter().enumerate() {
            let count=counts.get_mut(address.group()).ok_or_else(||context.metadata_error(format_args!("partition source group is absent")))?;
            *count=count.checked_add(1).ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            if parameters.execution_address(ordinal)!=Some(address) {
                return Err(context.metadata_error(format_args!("partition parameter source differs from its retained global address")));
            }
        }
        let layout=ExecutionUnitLayout::new_with_metadata(&graph,&counts,context)?;
        Self::for_layout(parameters,&layout,context)
    }

    fn validate_address(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        if self.parameters.layout() != self.layout.as_ref()
            || self.parameters.execution_address(ordinal) != Some(address)
        {
            return Err(workspace_message(
                context,
                format_args!(
                    "workspace parameter population differs from the acquired unit address"
                ),
            ));
        }
        Ok(())
    }
}

impl<U: Parameterized<WorkspaceTensor>> LayerwisePolicy<WorkspaceBackend, U>
    for WorkspaceLayerwisePolicy<'_>
{
    type Lease = Box<U>;
    type Error = Error;

    fn begin(
        &mut self,
        initial: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        context.validate_values([initial])
    }

    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &WorkspaceContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&WorkspaceContext) -> Result<U, E>,
    {
        self.validate_address(ordinal, address, context)
            .map_err(LayerwiseAcquireError::Policy)?;
        self.parameters.observe_acquire(ordinal,address,context)
            .map_err(LayerwiseAcquireError::Policy)?;
        // The actual unit lease owns a Box, independently of its tensor and
        // parameter metadata. Reserve its allocation and return transports
        // before building the unit; the cumulative Context survives retirement.
        let parts = [
            std::alloc::Layout::new::<U>().size(),
            std::mem::size_of::<U>(),
            std::mem::size_of::<Box<U>>(),
            std::mem::size_of::<Result<Box<U>, LayerwiseAcquireError<E, Error>>>(),
            std::mem::size_of::<LayerwiseAcquireError<E, Error>>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)
            .map_err(Error::from)
            .map_err(LayerwiseAcquireError::Policy)?;
        context
            .charge_metadata(controls)
            .map_err(Error::from)
            .map_err(LayerwiseAcquireError::Policy)?;
        // Borrow/count/fill the actual source before any unit constructor.
        // This same projection retains explicit trace-lifetime aliases across
        // acquires; an invocation root is recreated by its shared bind worker.
        if context.uses_checked_metadata() && self.projection.is_none() {
            context.charge_metadata(std::mem::size_of::<(
                eredu_runtime::working_memory::WorkspaceParameterSourceLoan<'_>,
                Result<eredu_runtime::working_memory::WorkspaceParameterSourceLoan<'_>,
                    eredu_runtime::working_memory::WorkspaceParameterSourceError>,
            )>()).map_err(Error::from).map_err(LayerwiseAcquireError::Policy)?;
            let parameters = self.parameters;
            let source = parameters.parameter_source()
                .map_err(|cause| LayerwiseAcquireError::Policy(context.metadata_source(cause)))?;
            self.projection = Some(source.prepare_projection(context)
                .map_err(LayerwiseAcquireError::Policy)?);
        }
        // Keep the existing whole-forward ledger. In particular, constructing
        // the next native unit can overlap the previous pending unit; dropping
        // metadata or its immediate completion must not refund those buffers.
        let mut unit = build(context).map_err(LayerwiseAcquireError::Architecture)?;
        if let Some(projection) = &mut self.projection {
            projection.bind(&mut unit, ordinal, address, context)
                .map_err(LayerwiseAcquireError::Policy)?;
        } else {
            let values = self.parameters.parameters(ordinal, address, context)
                .map_err(LayerwiseAcquireError::Policy)?;
            eredu_runtime::working_memory::bind_workspace_parameters(&mut unit, values)
                .map_err(LayerwiseAcquireError::Policy)?;
        }
        Ok(Box::new(unit))
    }

    fn complete<'value, StateValues, ContextValues>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'value WorkspaceTensor,
        state_values: StateValues,
        context_values: ContextValues,
        context: &WorkspaceContext,
    ) -> Result<(), Error>
    where
        StateValues: Iterator<Item = &'value WorkspaceTensor>,
        ContextValues: Iterator<Item = &'value WorkspaceTensor>,
    {
        self.validate_address(ordinal, address, context)?;
        context.validate_values(
            std::iter::once(output)
                .chain(state_values)
                .chain(context_values),
        )?;
        drop(lease);
        Ok(())
    }

    fn finish(
        &mut self,
        output: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        context.validate_values([output])
    }
}

#[cfg(test)]
mod tests;
