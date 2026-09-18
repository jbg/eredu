//! Metadata execution through the actual retained-ingress traversal.
//! These types are fixed to WorkspaceBackend and cannot submit native work or
//! issue original admission. The enclosing quote owns its ordinary metadata lease.
use super::*;
use crate::{LayerwisePolicy, LayerwiseRuntime, LayerwiseRuntimeError, ResidentRuntime};
use eredu_nn::{
    Error,
    workspace::{WorkspaceBackend, WorkspaceContext, HostMetadataFunding, WorkspaceTensor},
};

/// Metadata-only source with the same cut, inactive outcomes and phase changes
/// as live media. No public constructor accepts a chosen execution/revision.
pub struct MediaEquationSource<A, S>
where
    S: RuntimeState<WorkspaceBackend>,
    A: PrefillIngressArchitecture<WorkspaceBackend, S>,
{
    source: PreparedMediaPrefill<A, WorkspaceBackend, S>,
    // Source tables, flags and retained values retire before their funding.
    _funding: Option<HostMetadataFunding>,
}
impl<A, S> MediaEquationSource<A, S>
where
    S: RuntimeState<WorkspaceBackend>,
    A: PrefillIngressArchitecture<WorkspaceBackend, S, Error = Error>,
{
    /// Uses the actual bound source identities solely for ordinary diagnostics.
    /// Native storage and the preparation lease are retained by the closed caller.
    pub fn new(architecture: &A, plan: A::IngressPlan) -> Result<Self, Error> {
        Self::construct(architecture, plan, construction::Destination(None), || {
            architecture.execution_graph().map(crate::ArchitectureExecutionGraph::into_owned)
        })
    }

    /// Counts the actual cut/source tables before constructing them. The graph
    /// and ingress plan are supplied by the same architecture worker and remain
    /// separate metadata producers; this method grants no native qualification.
    /// A successful source and any constructor failure retain independent
    /// funding through their final storage and error shells.
    pub fn new_with_metadata(
        architecture: &A,
        plan: A::IngressPlan,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::construct_counted(architecture, plan, context, || {
            architecture.execution_graph()?.into_owned_with_metadata(context)
        })
    }

    /// Uses the actual graph retained by the caller's matching prepared runtime.
    /// Copies are admitted before construction and preserve every validated row.
    /// The shared ingress traversal still compares this cut with the actual
    /// architecture at each span; this loan grants no native source authority.
    pub fn new_with_prepared_graph(
        architecture: &A,
        plan: A::IngressPlan,
        graph: &ExecutionGraph,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::construct_counted(architecture, plan, context, || {
            graph.clone_with_metadata(context)
        })
    }

    /// Constructs the metadata source from the selected strategy's exact paid
    /// cut worker, including its local decoder-ingress ownership.
    pub fn new_selected<R, P, E>(runtime: &E::Runtime, plan: A::IngressPlan,
        context: &WorkspaceContext,
    ) -> Result<Self, Error>
    where R: LayerwisePolicy<WorkspaceBackend, A::Unit>,
        P: LayerwisePolicy<WorkspaceBackend, A::Unit, Error = R::Error>,
        R::Error: std::error::Error + Send + Sync + 'static,
        E: MediaTextExecutionStrategy<A, WorkspaceBackend, S, R, P>,
    {
        construction::admit_source::<A, S>(context)?;
        context.charge_metadata(std::mem::size_of::<(&E::Runtime, A::IngressPlan,
            Result<CompositePrefillCut, Error>)>())?;
        let funding = context.metadata_funding();
        let result = (|| {
            let cut = E::prepare_media_cut_with_metadata(runtime, &plan, context)?;
            Self::finish_source(plan, construction::Destination(Some(context)), cut)
        })();
        result.map_err(|cause| Error::backend_retained_source(construction::SourceFailure { cause, _funding: funding }))
    }

    /// Executes the selected strategy's original media worker with this exact
    /// metadata source. It cannot manufacture native completion or admission.
    pub fn forward_selected<R, P, E, O>(&mut self, strategy: &mut E,
        runtime: &mut E::Runtime, span: &PrefillChunk, state: &mut S,
        context: &WorkspaceContext, observer: &mut O,
        paths: Option<&crate::PreparedLayeredObservationPaths>,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>
    where R: LayerwisePolicy<WorkspaceBackend, A::Unit>,
        P: LayerwisePolicy<WorkspaceBackend, A::Unit, Error = R::Error>,
        R::Error: std::error::Error + Send + Sync + 'static,
        E: MediaTextExecutionStrategy<A, WorkspaceBackend, S, R, P>,
        O: crate::ActivationObserver<WorkspaceTensor, Error> + ?Sized,
    {
        context.charge_metadata(std::mem::size_of::<(
            &mut Self, &mut E, &mut E::Runtime, &PrefillChunk, &mut S,
            &WorkspaceContext, &mut O, Option<&crate::PreparedLayeredObservationPaths>,
            Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>,
        )>())?;
        strategy.forward_media_span(runtime, &mut self.source, span, state, context,
            observer, paths, span.output).map_err(|cause| context.metadata_source(cause))
    }

    fn construct_counted(
        architecture: &A,
        plan: A::IngressPlan,
        context: &WorkspaceContext,
        graph: impl FnOnce() -> Result<ExecutionGraph, Error>,
    ) -> Result<Self, Error> {
        construction::admit_source::<A, S>(context)?;
        let funding = context.metadata_funding();
        Self::construct(
            architecture,
            plan,
            construction::Destination(Some(context)),
            graph,
        )
        .map_err(|cause| {
            Error::backend_retained_source(construction::SourceFailure {
                cause,
                _funding: funding,
            })
        })
    }

    fn construct(
        architecture: &A,
        plan: A::IngressPlan,
        destination: construction::Destination<'_>,
        graph: impl FnOnce() -> Result<ExecutionGraph, Error>,
    ) -> Result<Self, Error> {
        Self::construct_with_cut(architecture, plan, destination,
            || destination.cut(graph()?, architecture.primary_execution_group()))
    }

    fn construct_with_cut(architecture: &A, plan: A::IngressPlan,
        destination: construction::Destination<'_>, cut: impl FnOnce() -> Result<CompositePrefillCut, Error>,
    ) -> Result<Self, Error> {
        match destination.0 {
            Some(context) => architecture.validate_ingress_plan(&plan, Some(context))?,
            None => architecture.validate_ingress_plan(&plan, None)?,
        }
        Self::finish_source(plan, destination, cut()?)
    }

    fn finish_source(plan: A::IngressPlan, destination: construction::Destination<'_>,
        cut: CompositePrefillCut,
    ) -> Result<Self, Error> {
        let binding =
            A::ingress_session_binding(&plan).ok_or_else(|| destination.missing_binding())?;
        let revision = binding.revision.clone();
        let source = destination.equation(plan, cut, revision)?;
        Ok(Self {
            source,
            _funding: destination.0.and_then(WorkspaceContext::metadata_funding),
        })
    }

    /// Complete compact/cut backing for opening or closing an equation interval.
    pub fn visit_roots(&self, visitor: &mut dyn FnMut(&WorkspaceTensor)) {
        self.source.visit_retained_roots(visitor);
    }
    /// Finishes a metadata interval after its complete report. This is the quote
    /// scheduler's synthetic completion, never evidence about native completion.
    pub fn finish_equation_span(&mut self) -> Result<(), Error> {
        let revision = self.source.revision.clone();
        self.source
            .committed(&revision)
            .map_err(|cause| match self.source.metadata.as_ref() { Some(metadata) => metadata.metadata_source(cause), None => Error::backend_retained_source(cause) })
    }
    pub fn forward_resident<H>(
        &mut self,
        runtime: &mut ResidentRuntime<A, WorkspaceBackend, S>,
        span: &PrefillChunk,
        state: &mut S,
        context: &WorkspaceContext,
        hook: &mut H,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>
    where
        H: LayeredTraversalHook<WorkspaceBackend, A::ForwardContext, Error> + ?Sized,
    {
        let initial = self.source.start(span).map_err(|cause| match self.source.metadata.as_ref() { Some(metadata) => metadata.metadata_source(cause), None => Error::backend_retained_source(cause) })?;
        runtime.forward_with_invocation_and_traversal_hook(
            MediaInvocation {
                source: &mut self.source,
                span,
                initial,
            },
            state,
            context,
            hook,
            span.output,
        )
    }
    pub fn forward_layerwise<P, H>(
        &mut self,
        runtime: &mut LayerwiseRuntime<A, WorkspaceBackend, S, P>,
        span: &PrefillChunk,
        state: &mut S,
        context: &WorkspaceContext,
        hook: &mut H,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), LayerwiseRuntimeError<Error, P::Error>>
    where
        P: LayerwisePolicy<WorkspaceBackend, A::Unit>,
        P::Error: std::fmt::Display,
        H: LayeredTraversalHook<WorkspaceBackend, A::ForwardContext, Error> + ?Sized,
    {
        let initial = self
            .source
            .start(span)
            .map_err(|e| LayerwiseRuntimeError::Architecture(context.metadata_source(e)))?;
        runtime.forward_with_unit_executor_and_invocation(
            MediaInvocation {
                source: &mut self.source,
                span,
                initial,
            },
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, hook| {
                if hook.observes_activations() {
                    architecture.forward_unit_observed(
                        group, index, unit, hidden, state, forward, context,
                        &mut crate::layered::TraversalActivationObserver::<_, WorkspaceBackend, A::ForwardContext, Error> {
                            hook, types: std::marker::PhantomData,
                        },
                    )
                } else {
                    architecture.forward_unit(group, index, unit, hidden, state, forward, context)
                }
            },
            hook,
            true,
            false,
            span.output,
        )
    }

    /// Borrows the runtime's authenticated path source alongside the same media
    /// cut hook. Observation and ingress execute in one existing traversal.
    pub fn forward_resident_observed<H>(
        &mut self,
        runtime: &mut ResidentRuntime<A, WorkspaceBackend, S>,
        span: &PrefillChunk, state: &mut S, context: &WorkspaceContext,
        hook: &mut H,
        observer: &mut dyn crate::working_memory::InferenceWorkspaceObserver,
        paths: &crate::PreparedLayeredObservationPaths,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), Error>
    where H: LayeredTraversalHook<WorkspaceBackend, A::ForwardContext, Error> + ?Sized,
    {
        runtime.validate_observation_binding(paths).map_err(|cause| context.metadata_source(cause))?;
        let mut combined = observed_hook(hook, observer, paths, span, context)?;
        self.forward_resident(runtime, span, state, context, &mut combined)
    }

    /// Uses the same layerwise acquisition and unit worker with both existing
    /// observers. A failed binding refuses before ingress or layer acquisition.
    pub fn forward_layerwise_observed<P, H>(
        &mut self,
        runtime: &mut LayerwiseRuntime<A, WorkspaceBackend, S, P>,
        span: &PrefillChunk, state: &mut S, context: &WorkspaceContext,
        hook: &mut H,
        observer: &mut dyn crate::working_memory::InferenceWorkspaceObserver,
        paths: &crate::PreparedLayeredObservationPaths,
    ) -> Result<(Option<WorkspaceTensor>, A::ForwardContext), LayerwiseRuntimeError<Error, P::Error>>
    where P: LayerwisePolicy<WorkspaceBackend, A::Unit>, P::Error: std::fmt::Display,
        H: LayeredTraversalHook<WorkspaceBackend, A::ForwardContext, Error> + ?Sized,
    {
        runtime.validate_observation_binding(paths)
            .map_err(|cause| LayerwiseRuntimeError::Architecture(context.metadata_source(cause)))?;
        let mut combined = observed_hook(hook, observer, paths, span, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        self.forward_layerwise(runtime, span, state, context, &mut combined)
    }

    /// Executes the actual routed provider inside the existing retained media
    /// invocation. The source cut and readout demand keep their ordinary owner.
    pub fn forward_resident_routed<Provider,H>(&mut self,runtime:&mut ResidentRuntime<A,WorkspaceBackend,S>,
        span:&PrefillChunk,state:&mut S,context:&WorkspaceContext,hook:&mut H,provider:&mut Provider)
        ->Result<(Option<WorkspaceTensor>,A::ForwardContext),Error>
    where A:crate::RoutedLayeredArchitecture<WorkspaceBackend,S>,
        Provider:crate::RoutedExpertProvider<WorkspaceBackend>,Provider::Error:std::fmt::Display,
        H:LayeredTraversalHook<WorkspaceBackend,A::ForwardContext,Error>+?Sized,
    {
        let initial=self.source.start(span).map_err(|cause| match self.source.metadata.as_ref() { Some(metadata) => metadata.metadata_source(cause), None => Error::backend_retained_source(cause) })?;
        runtime.forward_with_invocation_and_unit_executor(MediaInvocation{source:&mut self.source,span,initial},
            state,context,hook,span.output,
            |architecture,group,index,unit,hidden,state,forward,context,hook| {
                if hook.observes_activations(){
                    architecture.forward_unit_observed_with_provider(group,index,unit,hidden,state,forward,
                        crate::ExpertPass::Prefill,provider,context,
                        &mut crate::layered::TraversalActivationObserver::<_,WorkspaceBackend,A::ForwardContext,Error>{
                            hook,types:std::marker::PhantomData})
                }else{architecture.forward_unit_with_provider(group,index,unit,hidden,state,forward,
                    crate::ExpertPass::Prefill,provider,context)}
            })
    }
    /// Uses the same source cut and layerwise acquisition while retaining the
    /// independently selected provider for each actual primary unit.
    pub fn forward_layerwise_routed<P,Provider,H>(&mut self,runtime:&mut LayerwiseRuntime<A,WorkspaceBackend,S,P>,
        span:&PrefillChunk,state:&mut S,context:&WorkspaceContext,hook:&mut H,provider:&mut Provider)
        ->Result<(Option<WorkspaceTensor>,A::ForwardContext),LayerwiseRuntimeError<Error,P::Error>>
    where A:crate::RoutedLayeredArchitecture<WorkspaceBackend,S>,
        P:LayerwisePolicy<WorkspaceBackend,A::Unit>,P::Error:std::fmt::Display,
        Provider:crate::RoutedExpertProvider<WorkspaceBackend>,Provider::Error:std::fmt::Display,
        H:LayeredTraversalHook<WorkspaceBackend,A::ForwardContext,Error>+?Sized,
    {
        let initial=self.source.start(span).map_err(|cause|LayerwiseRuntimeError::Architecture(context.metadata_source(cause)))?;
        runtime.forward_with_unit_executor_and_invocation(MediaInvocation{source:&mut self.source,span,initial},
            state,context,|architecture,group,index,unit,hidden,state,forward,context,hook| {
                if hook.observes_activations(){
                    architecture.forward_unit_observed_with_provider(group,index,unit,hidden,state,forward,
                        crate::ExpertPass::Prefill,provider,context,
                        &mut crate::layered::TraversalActivationObserver::<_,WorkspaceBackend,A::ForwardContext,Error>{
                            hook,types:std::marker::PhantomData})
                }else{architecture.forward_unit_with_provider(group,index,unit,hidden,state,forward,
                    crate::ExpertPass::Prefill,provider,context)}
            },hook,true,false,span.output)
    }
    /// Authenticates the ordinary observation binding before routed ingress.
    pub fn forward_resident_routed_observed<Provider,H>(&mut self,runtime:&mut ResidentRuntime<A,WorkspaceBackend,S>,
        span:&PrefillChunk,state:&mut S,context:&WorkspaceContext,hook:&mut H,provider:&mut Provider,
        observer:&mut dyn crate::working_memory::InferenceWorkspaceObserver,paths:&crate::PreparedLayeredObservationPaths)
        ->Result<(Option<WorkspaceTensor>,A::ForwardContext),Error>
    where A:crate::RoutedLayeredArchitecture<WorkspaceBackend,S>,
        Provider:crate::RoutedExpertProvider<WorkspaceBackend>,Provider::Error:std::fmt::Display,
        H:LayeredTraversalHook<WorkspaceBackend,A::ForwardContext,Error>+?Sized,
    {
        runtime.validate_observation_binding(paths).map_err(|cause|context.metadata_source(cause))?;
        let mut combined=observed_hook(hook,observer,paths,span,context)?;
        self.forward_resident_routed(runtime,span,state,context,&mut combined,provider)
    }
    /// Keeps the observed media source and provider in the same bounded window.
    pub fn forward_layerwise_routed_observed<P,Provider,H>(&mut self,runtime:&mut LayerwiseRuntime<A,WorkspaceBackend,S,P>,
        span:&PrefillChunk,state:&mut S,context:&WorkspaceContext,hook:&mut H,provider:&mut Provider,
        observer:&mut dyn crate::working_memory::InferenceWorkspaceObserver,paths:&crate::PreparedLayeredObservationPaths)
        ->Result<(Option<WorkspaceTensor>,A::ForwardContext),LayerwiseRuntimeError<Error,P::Error>>
    where A:crate::RoutedLayeredArchitecture<WorkspaceBackend,S>,
        P:LayerwisePolicy<WorkspaceBackend,A::Unit>,P::Error:std::fmt::Display,
        Provider:crate::RoutedExpertProvider<WorkspaceBackend>,Provider::Error:std::fmt::Display,
        H:LayeredTraversalHook<WorkspaceBackend,A::ForwardContext,Error>+?Sized,
    {
        runtime.validate_observation_binding(paths).map_err(|cause|LayerwiseRuntimeError::Architecture(context.metadata_source(cause)))?;
        let mut combined=observed_hook(hook,observer,paths,span,context).map_err(LayerwiseRuntimeError::Architecture)?;
        self.forward_layerwise_routed(runtime,span,state,context,&mut combined,provider)
    }

}

// Only borrows existing source/observer owners. The exact pair and its returned
// transport are charged before constructing them; neither callback can outlive
// the shared equation invocation.
fn observed_hook<'a, H: ?Sized>(
    hook: &'a mut H,
    observer: &'a mut dyn crate::working_memory::InferenceWorkspaceObserver,
    paths: &'a crate::PreparedLayeredObservationPaths,
    span: &PrefillChunk,
    context: &WorkspaceContext,
) -> Result<crate::CompositeLayeredTraversalHook<
    &'a mut H,
    crate::layered::BorrowedHook<'a, dyn crate::working_memory::InferenceWorkspaceObserver + 'a>,
>, Error> {
    type Pair<'a, H> = crate::CompositeLayeredTraversalHook<
        &'a mut H,
        crate::layered::BorrowedHook<'a, dyn crate::working_memory::InferenceWorkspaceObserver + 'a>,
    >;
    context.charge_metadata(std::mem::size_of::<(
        Pair<'_, H>, Result<Pair<'_, H>, Error>,
        &crate::PreparedLayeredObservationPaths, &PrefillChunk, &WorkspaceContext,
    )>())?;
    if observer.requires_sequence_readout() && span.output != eredu_core::OutputDemand::Sequence {
        return Err(context.metadata_source(
            crate::PreparedLayeredObservationError::<std::convert::Infallible>::ReadoutDemand,
        ));
    }
    Ok(crate::CompositeLayeredTraversalHook::new(
        hook, crate::layered::BorrowedHook { observer, paths: paths.source() },
    ))
}
