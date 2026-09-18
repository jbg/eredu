//! Retained media uses the same session input transaction and publication order.

use super::*;
use crate::media_prefill::{
    CompositePrefillCut, MediaInvocation, MediaTextExecutionStrategy, PrefillIngressArchitecture,
    PreparedMediaPrefill,
};
use crate::prefill::PrefillChunk;
use crate::working_memory::{InferenceRequest, WorkingMemoryError};

impl<A, B, S, R, P> ReplicatedTextRuntime<A, B, S, R, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    fn media_cut(&self, plan: &A::IngressPlan) -> Result<CompositePrefillCut, A::Error> {
        let architecture = match &self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime.architecture(),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime.architecture(),
        };
        architecture.validate_ingress_plan(plan, None)?;
        CompositePrefillCut::new(
            architecture.execution_graph()?.into_owned(),
            architecture.primary_execution_group(),
        )
        .map_err(|cause| A::ingress_error(cause, None))
    }

    fn media_cut_with_metadata(
        &self,
        plan: &A::IngressPlan,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<CompositePrefillCut, eredu_nn::Error>
    where
        A: PrefillIngressArchitecture<B, S, Error = eredu_nn::Error>,
    {
        let architecture = match &self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime.architecture(),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime.architecture(),
        };
        architecture.validate_ingress_plan(plan, Some(context))?;
        let graph = architecture
            .ingress_execution_graph(Some(context))?
            .into_owned_with_metadata(context)?;
        crate::media_prefill::construction::original_cut(
            graph,
            architecture.primary_execution_group(),
            context,
        )
    }

    fn forward_media<E, O>(
        &mut self,
        source: &mut PreparedMediaPrefill<A, B, S>,
        span: &PrefillChunk,
        state: &mut S,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        execute: E,
        observer: &mut O,
        paths: Option<&crate::PreparedLayeredObservationPaths>,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<<B as NeuralBackend>::Tensor as Tensor>::Context,
            &mut O,
        ) -> Result<B::Tensor, A::Error>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let initial = source
            .start(span)
            .map_err(|e| ReplicatedTextSessionError::Architecture(A::ingress_error(e, None)))?;
        let invocation = MediaInvocation {
            source,
            span,
            initial,
        };
        if let Some(paths) = paths {
            let result = match &mut self.kind {
                ReplicatedTextRuntimeKind::Resident(runtime) => runtime
                    .forward_invocation_with_prepared_internal_observer(
                        invocation, state, context, execute, observer, paths, demand,
                    ),
                ReplicatedTextRuntimeKind::Bounded(runtime) => runtime
                    .forward_invocation_with_prepared_internal_observer(
                        invocation, state, context, execute, observer, paths, demand,
                    ),
            };
            result
                .map_err(|error| super::observation_paths::map_prepared(error, map_layerwise_error))
        } else {
            match &mut self.kind {
                ReplicatedTextRuntimeKind::Resident(runtime) => runtime
                    .forward_invocation_with_internal_observer(
                        invocation, state, context, execute, observer, demand, false,
                    ),
                ReplicatedTextRuntimeKind::Bounded(runtime) => runtime
                    .forward_invocation_with_internal_observer(
                        invocation, state, context, execute, observer, demand, false,
                    ),
            }
            .map_err(map_layerwise_error)
        }
    }
}

impl<A, B, S, R, P> MediaTextExecutionStrategy<A, B, S, R, P> for DirectReplicatedTextExecution
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    fn prepare_media_cut(
        runtime: &Self::Runtime,
        plan: &A::IngressPlan,
    ) -> Result<CompositePrefillCut, A::Error> {
        runtime.media_cut(plan)
    }
    fn prepare_media_cut_with_metadata(
        runtime: &Self::Runtime,
        plan: &A::IngressPlan,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<CompositePrefillCut, eredu_nn::Error>
    where
        A: PrefillIngressArchitecture<B, S, Error = eredu_nn::Error>,
    {
        runtime.media_cut_with_metadata(plan, context)
    }
    fn forward_media_span<O>(
        &mut self,
        runtime: &mut Self::Runtime,
        source: &mut PreparedMediaPrefill<A, B, S>,
        chunk: &PrefillChunk,
        state: &mut S,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
        paths: Option<&crate::PreparedLayeredObservationPaths>,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        runtime.forward_media(
            source,
            chunk,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, observer| {
                architecture.forward_unit_observed(
                    group, index, unit, hidden, state, forward, context, observer,
                )
            },
            observer,
            paths,
            demand,
        )
    }
}

impl<A, B, S, R, P, Provider> MediaTextExecutionStrategy<A, B, S, R, P>
    for RoutedReplicatedTextExecution<Provider>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::GroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S> + RoutedLayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
    Provider: RoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
{
    fn prepare_media_cut(
        runtime: &Self::Runtime,
        plan: &A::IngressPlan,
    ) -> Result<CompositePrefillCut, A::Error> {
        runtime.media_cut(plan)
    }
    fn prepare_media_cut_with_metadata(
        runtime: &Self::Runtime,
        plan: &A::IngressPlan,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<CompositePrefillCut, eredu_nn::Error>
    where
        A: PrefillIngressArchitecture<B, S, Error = eredu_nn::Error>,
    {
        runtime.media_cut_with_metadata(plan, context)
    }
    fn forward_media_span<O>(
        &mut self,
        runtime: &mut Self::Runtime,
        source: &mut PreparedMediaPrefill<A, B, S>,
        chunk: &PrefillChunk,
        state: &mut S,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
        paths: Option<&crate::PreparedLayeredObservationPaths>,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        runtime.forward_media(
            source,
            chunk,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, observer| {
                architecture.forward_unit_observed_with_provider(
                    group,
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    ExpertPass::Prefill,
                    &mut self.provider,
                    context,
                    observer,
                )
            },
            observer,
            paths,
            demand,
        )
    }
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: PrefillIngressArchitecture<B, M::State>,
    D: MediaTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Prepares one ordinary retained media source against the actual selected
    /// session and current revision. This entry grants no finite media budget.
    pub fn prepare_media_prefill_unbudgeted(
        &self,
        plan: A::IngressPlan,
    ) -> Result<
        PreparedMediaPrefill<A, B, M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.validate_media_preparation()?;
        let geometry = A::ingress_geometry(&plan);
        let request = InferenceRequest::without_memory_budget(&self.prefill_identity, geometry)
            .map_err(|error| ReplicatedTextSessionError::Contract(error.to_string()))?;
        self.prepare_media_prefill_with_request(plan, request)
    }

    fn prepare_media_prefill_with_request(
        &self,
        plan: A::IngressPlan,
        request: InferenceRequest,
    ) -> Result<
        PreparedMediaPrefill<A, B, M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        if request.memory_reservation().is_some() {
            return Err(ReplicatedTextSessionError::WorkingMemory(
                crate::working_memory::WorkingMemoryError::UnknownBound,
            ));
        }
        self.validate_media_preparation()?;
        if let Some(original) = A::ingress_session_binding(&plan) {
            let current = self.media_semantic_binding().map_err(|error| match error {
                super::MediaSemanticBindingError::Boundary(error) => {
                    ReplicatedTextSessionError::WorkingMemory(error)
                }
                super::MediaSemanticBindingError::Mechanism(error) => {
                    ReplicatedTextSessionError::Mechanism(error)
                }
            })?;
            if !original.matches(&current) {
                return Err(ReplicatedTextSessionError::WorkingMemory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
        }
        let cut = D::prepare_media_cut(&self.execution, &plan)
            .map_err(ReplicatedTextSessionError::Architecture)?;
        PreparedMediaPrefill::new(
            plan,
            cut,
            request,
            &self.prefill_identity,
            self.state.inference_retention().revision().clone(),
        )
        .map_err(ReplicatedTextSessionError::WorkingMemory)
    }

    fn prepare_media_prefill_with_metadata(
        &self,
        plan: A::IngressPlan,
        request: InferenceRequest,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<
        PreparedMediaPrefill<A, B, M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        A: PrefillIngressArchitecture<B, M::State, Error = eredu_nn::Error>,
    {
        if request.memory_reservation().is_none() || context.metadata_funding().is_none() {
            return Err(ReplicatedTextSessionError::WorkingMemory(
                WorkingMemoryError::UnknownBound,
            ));
        }
        self.validate_media_preparation()?;
        let original = A::ingress_session_binding(&plan).ok_or(
            ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::IdentityMismatch),
        )?;
        let current = self.media_semantic_binding().map_err(|error| match error {
            super::MediaSemanticBindingError::Boundary(error) => {
                ReplicatedTextSessionError::WorkingMemory(error)
            }
            super::MediaSemanticBindingError::Mechanism(error) => {
                ReplicatedTextSessionError::Mechanism(error)
            }
        })?;
        if !original.matches(&current) {
            return Err(ReplicatedTextSessionError::WorkingMemory(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let cut = D::prepare_media_cut_with_metadata(&self.execution, &plan, context)
            .map_err(ReplicatedTextSessionError::Architecture)?;
        crate::media_prefill::construction::prepared_original(
            plan,
            cut,
            request,
            &self.prefill_identity,
            self.state.inference_retention().revision().clone(),
            context,
        )
        .map_err(ReplicatedTextSessionError::Architecture)
    }

    fn validate_media_preparation(
        &self,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_control_unfenced()?;
        if !self.selected.exact_completion_available()
            || !self.mechanisms.supports_media_ingress_completion()
        {
            return Err(ReplicatedTextSessionError::Contract(
                "selected mechanism has no retained-media completion".into(),
            ));
        }
        Ok(())
    }

    /// Runs explicitly ordinary retained media through the same source selection,
    /// request claim, span executor and final output handling as text. A supplied
    /// request is preserved; any original reservation is rejected before `make_plan`.
    /// This entry grants no media allocation or observation authority.
    pub fn try_prefill_media_source_cancellable<O>(
        &mut self,
        request: Option<&InferenceRequest>,
        shape: Option<[u64; 2]>,
        identity: Option<crate::SharedPreparedInputCacheIdentity>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        make_plan: impl FnOnce(eredu_core::InferenceGeometry) -> Result<A::IngressPlan, A::Error>,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        super::PrefillSourceProgress<B::Tensor>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.try_prefill_media_factory(
            request,
            shape,
            max_chunk_positions,
            |session, request| {
                let plan = make_plan(request.geometry())
                    .map_err(ReplicatedTextSessionError::Architecture)?;
                session
                    .prepare_media_prefill_with_request(plan, request.clone())
                    .and_then(|source| {
                        source
                            .with_prompt_identity(identity)
                            .map(Some)
                            .map_err(ReplicatedTextSessionError::Architecture)
                    })
            },
            None,
            cancellation,
            context,
            observer,
        )
    }

    /// Uses the accepted request and the actual retained planning Context for
    /// source construction. The shared driver still authenticates and consumes
    /// its one-use request claim before invoking this factory.
    pub fn try_prefill_media_source_with_metadata<O>(
        &mut self,
        request: Option<&InferenceRequest>,
        shape: Option<[u64; 2]>,
        identity: Option<crate::SharedPreparedInputCacheIdentity>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        make_plan: impl FnOnce(eredu_core::InferenceGeometry) -> Result<A::IngressPlan, A::Error>,
        metadata: &eredu_nn::workspace::WorkspaceContext,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        super::PrefillSourceProgress<B::Tensor>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        A: PrefillIngressArchitecture<B, M::State, Error = eredu_nn::Error>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.try_prefill_media_factory(
            request,
            shape,
            max_chunk_positions,
            |session, request| {
                metadata
                    .charge_metadata(std::mem::size_of::<(
                        A::IngressPlan,
                        Result<A::IngressPlan, A::Error>,
                    )>())
                    .map_err(|cause| ReplicatedTextSessionError::Architecture(cause.into()))?;
                let plan = make_plan(request.geometry())
                    .map_err(ReplicatedTextSessionError::Architecture)?;
                session
                    .prepare_media_prefill_with_metadata(plan, request.clone(), metadata)
                    .and_then(|source| {
                        source
                            .with_prompt_identity(identity)
                            .map(Some)
                            .map_err(ReplicatedTextSessionError::Architecture)
                    })
            },
            Some(metadata),
            cancellation,
            context,
            observer,
        )
    }

    fn try_prefill_media_factory<O>(
        &mut self,
        request: Option<&InferenceRequest>,
        shape: Option<[u64; 2]>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        make_source: impl FnOnce(
            &Self,
            &InferenceRequest,
        ) -> Result<
            Option<PreparedMediaPrefill<A, B, M::State>>,
            ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        >,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        super::PrefillSourceProgress<B::Tensor>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let progress = self.try_prefill_source_with_lifecycle(
            request,
            shape,
            max_chunk_positions,
            observer
                .ordinary_prefill_capture()
                .map_or(eredu_core::OutputDemand::LastPosition, |capture| {
                    capture.geometry().output
                }),
            make_source,
            cancellation,
            context,
            observer,
            super::MediaPrefillSpan,
            true,
            metadata,
        )?;
        let outcome = match progress.outcome {
            super::PrefillSourceOutcome::Complete(Some(output)) => {
                super::PrefillSourceOutcome::Complete(output)
            }
            super::PrefillSourceOutcome::Cancelled => super::PrefillSourceOutcome::Cancelled,
            super::PrefillSourceOutcome::Unavailable
            | super::PrefillSourceOutcome::Complete(None) => {
                return Err(match metadata {
                    Some(_) => {
                        ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::UnknownBound)
                    }
                    None => ReplicatedTextSessionError::Contract(
                        "selected media prefill has no final scores".into(),
                    ),
                });
            }
        };
        Ok(super::PrefillSourceProgress {
            outcome,
            completed_positions: progress.completed_positions,
        })
    }

    fn execute_media_span_before_publication<O>(
        &mut self,
        source: &mut PreparedMediaPrefill<A, B, M::State>,
        input: Result<&PrefillChunk, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        checkpoint: Option<M::StateCheckpoint>,
    ) -> Result<(Option<B::Tensor>, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let session = self;
        let batch_size = source.geometry().batch_size;
            session.require_input_result_agreement(true)?;
            let (output, checkpoint, forward) = session
                .execute_input_operation_before_publication(
                    input,
                    ExpertPass::Prefill,
                    context,
                    observer,
                    demand,
                    checkpoint,
                    |span| Ok(Some([batch_size, span.input.end - span.input.start])),
                    |driver, execution, state, paths, span, observer, demand| {
                        driver
                            .forward_media_span(
                                execution,
                                source,
                                span,
                                state,
                                context,
                                observer,
                                paths,
                                demand,
                            )
                            .map_err(widen_infallible)
                            .and_then(|result| {
                                // This joins the existing execution vote, before
                                // completion/publication. No second readiness phase.
                                source.validate_complete_cut().map_err(|error| {
                                    ReplicatedTextSessionError::Architecture(A::ingress_error(
                                        error, None))
                                })?;
                                Ok(result)
                            })
                    },
                )?;
        Ok((output, checkpoint, forward))
    }

    pub(super) fn prefill_media_span<O>(
        &mut self,
        source: &mut PreparedMediaPrefill<A, B, M::State>,
        span: Result<&PrefillChunk, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        demand: eredu_core::OutputDemand,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let input = span.and_then(|span| {
            source
                .validate_span(span)
                .map_err(|e| ReplicatedTextSessionError::Architecture(A::ingress_error(e, None)))?;
            source
                .validate_revision(self.state.inference_retention().revision())
                .map_err(ReplicatedTextSessionError::WorkingMemory)?;
            if let Some(original) = source.semantic_binding() {
                let current = self.media_semantic_binding().map_err(|error| match error {
                    super::MediaSemanticBindingError::Boundary(error) => {
                        ReplicatedTextSessionError::WorkingMemory(error)
                    }
                    super::MediaSemanticBindingError::Mechanism(error) => {
                        ReplicatedTextSessionError::Mechanism(error)
                    }
                })?;
                // The source's existing revision transition above authenticates
                // committed prefixes. Parameter/executable identity is separate.
                if !original.same_origin(&current) {
                    return Err(ReplicatedTextSessionError::WorkingMemory(
                        WorkingMemoryError::IdentityMismatch,
                    ));
                }
            }

            if let Some(capture) = observer.ordinary_prefill_capture() {
                if capture.geometry() != source.geometry() {
                    return Err(ReplicatedTextSessionError::WorkingMemory(
                        crate::working_memory::WorkingMemoryError::IdentityMismatch,
                    ));
                }
            } else if let Some(capture) = observer.admitted_prefill_capture() {
                // The gateway authenticated this original request and current
                // prepared paths before invoking the media source factory. Keep
                // that exact source/geometry binding at every completed span.
                if !capture.selection().selection().is_prepared_media() {
                    return Err(ReplicatedTextSessionError::WorkingMemory(
                        WorkingMemoryError::IdentityMismatch,
                    ));
                }
                capture.validate_request(source.request().map_err(ReplicatedTextSessionError::WorkingMemory)?)
                    .map_err(ReplicatedTextSessionError::WorkingMemory)?;
            } else if let Some(capture) = observer.admitted_capture_continuation() {
                if !capture.selection().is_prepared_media() {
                    return Err(ReplicatedTextSessionError::WorkingMemory(
                        WorkingMemoryError::IdentityMismatch,
                    ));
                }
                capture.validate_request(source.request().map_err(ReplicatedTextSessionError::WorkingMemory)?)
                    .map_err(ReplicatedTextSessionError::WorkingMemory)?;
            } else if observer.requires_prepared_traversal()
                || observer.requires_sequence_readout()
                || observer.transactional()
            {
                return Err(ReplicatedTextSessionError::Contract(
                    "media capture/transaction attribution is not admitted".into(),
                ));
            }
            Ok(span)
        });
        let result = self.with_observation_transaction(observer, |session, observer| {
            let (output, checkpoint, forward) = session.execute_media_span_before_publication(
                source, input, demand, context, observer, None,
            )?;
            let (output, checkpoint, forward) = session
                .publish_observed_output_transaction_with_readout(
                    output, checkpoint, forward, context,
                )?;
            session.publish_with_media_roots(
                output,
                checkpoint,
                forward,
                context,
                observer,
                true,
                Some(&source.roots()),
            )
        });
        match result {
            Ok(output) => {
                if let Some(identity) = identity {
                    self.committed_prompt_input_identity = Some(identity);
                }
                Ok(output)
            }
            Err(error) => {
                source.failed();
                Err(error)
            }
        }
    }
}

mod speculative;
