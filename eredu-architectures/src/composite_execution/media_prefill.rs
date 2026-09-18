//! Typed retained-ingress forwarding over the existing composite architecture.

use super::*;
use eredu_runtime::media_prefill::{MediaIngressError, PrefillIngressArchitecture};

impl<A, B, S> PrefillIngressArchitecture<B, S> for PreparedCompositeArchitecture<A>
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeMediaIngressArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
{
    type IngressPlan = A::IngressPlan;
    type Ingress = A::Ingress;

    fn ingress_geometry(plan: &Self::IngressPlan) -> eredu_core::InferenceGeometry {
        A::ingress_geometry(plan)
    }
    fn ingress_session_binding(
        plan: &Self::IngressPlan,
    ) -> Option<&eredu_runtime::working_memory::MediaSessionBinding> {
        A::ingress_session_binding(plan)
    }
    fn ingress_cache_identity(
        plan: &Self::IngressPlan,
    ) -> Option<eredu_runtime::SharedPreparedInputCacheIdentity> {
        A::ingress_cache_identity(plan)
    }
    fn validate_ingress_cache_identity(
        plan: &Self::IngressPlan,
        identity: &eredu_runtime::SharedPreparedInputCacheIdentity, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<(), Self::Error> {
        if identity.prepared() != A::prepared_ingress_input(plan).prepared().identity() {
            return Err(A::ingress_error(MediaIngressError::ForeignIdentity, metadata_context));
        }
        Ok(())
    }

    fn ingress_execution_graph(&self, context: Option<&eredu_nn::workspace::WorkspaceContext>)
        -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Self::Error> {
        match &self.prepared_graph {
            PreparedCompositeGraph::Validated(selected) => Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(selected.requirements().execution_graph())),
            PreparedCompositeGraph::Invalidated => Err(A::ingress_error(MediaIngressError::ForeignGraph, context)),
            PreparedCompositeGraph::Unprepared => self.inner.ingress_execution_graph(context),
        }
    }


    fn validate_ingress_plan(&self, plan: &Self::IngressPlan, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<(), Self::Error> {
        self.inner.validate_ingress_plan(plan, metadata_context)
    }
    fn ingress_error(error: MediaIngressError, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Self::Error {
        A::ingress_error(error, metadata_context)
    }

    fn begin_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner.begin_ingress(plan, state, parallel, context)
    }
    fn begin_ingress_received(
        &mut self,
        plan: &Self::IngressPlan,
        received: &B::Tensor,
        encoder_continuation: bool,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner.begin_ingress_received(
            plan,
            received,
            encoder_continuation,
            state,
            parallel,
            context,
        )
    }
    fn retain_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Ingress, Self::Error> {
        self.inner.retain_ingress(plan, forward, context)
    }
    fn begin_ingress_span(
        &mut self,
        plan: &Self::IngressPlan,
        ingress: &Self::Ingress,
        span: &eredu_runtime::prefill::PrefillChunk,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        self.inner
            .begin_ingress_span(plan, ingress, span, state, parallel, context)
    }
    fn visit_ingress_roots(ingress: &Self::Ingress, visitor: &mut dyn FnMut(&B::Tensor)) {
        A::visit_ingress_roots(ingress, visitor)
    }
}

/// Composite metadata paired with the family-owned retained-ingress plan.
/// This optional extension preserves unsupported families' ordinary input APIs.
pub trait CompositeMediaIngressArchitecture<B, S>:
    CompositeArchitecture<B, S> + PrefillIngressArchitecture<B, S>
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Constructs the exact family plan from the retained original prepared input.
    fn prepare_ingress_plan(
        admission: &Self::AdmissionConfig,
        input: eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self::IngressPlan, Self::Error>;
    /// Consumes admission already produced for these exact native parts. The
    /// default preserves existing family preparation; supporting families reuse
    /// the pair so an ordinary selected run performs one full admission.
    fn prepare_ingress_plan_admitted(
        admission: &Self::AdmissionConfig,
        input: eredu_runtime::PreparedModelInput<B::Tensor>,
        admitted: crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self::IngressPlan, Self::Error> {
        let _ = admitted;
        Self::prepare_ingress_plan(admission, input, inspector, geometry)
    }

    /// Binds the closed cold source only while the actual family remains typed.
    fn bind_original_media_semantics(
        _admission: &Self::AdmissionConfig,
        original: crate::media_plan::OriginalPreparedMediaSemantics<'_>,
        _blueprint: &crate::prepared_execution::PreparedInferenceBlueprint,
        _source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
        _binding: eredu_runtime::working_memory::MediaSessionBinding,
    ) -> Result<
        crate::media_plan::BoundPreparedMediaSemantics,
        eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
    > {
        Err(original
            .reject_boundary(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound))
    }
    /// Consumes the exact completed upload with its authenticated semantics.
    /// The native caller retains the private upload lineage; public raw input
    /// construction cannot establish that lineage.
    fn prepare_bound_original_ingress_plan(
        _input: crate::processor_execution::OriginalHostLowering<B::Tensor>,
        original: crate::media_plan::BoundPreparedMediaSemantics,
        _geometry: eredu_core::InferenceGeometry,
    ) -> Result<
        Self::IngressPlan,
        eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
    > {
        Err(
            original.reject(crate::media_plan::MediaSemanticError::input(
                "selected family has no original media semantic consumer",
            )),
        )
    }

    /// Same completed-source plan with its actual host metadata account retained.
    /// A native caller cannot fall back to an ordinary constructor on refusal.
    fn prepare_bound_original_ingress_plan_with_metadata(
        _input: crate::processor_execution::OriginalHostLowering<B::Tensor>,
        _original: crate::media_plan::BoundPreparedMediaSemantics,
        _geometry: eredu_core::InferenceGeometry,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self::IngressPlan, eredu_nn::Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }

    /// Ordinary equation projection with authenticated original A/B table custody.
    /// This method is restricted to metadata tensors and grants no native entry.
    fn prepare_original_workspace_ingress_plan(
        input: crate::prepared_execution::OriginalMediaWorkspaceInput,
        _geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self::IngressPlan, Self::Error>
    where
        B: NeuralBackend<Tensor = eredu_nn::workspace::WorkspaceTensor>,
        Self: CompositeArchitecture<B, S, Error = eredu_nn::Error>,
    {
        Err(input.reject(eredu_nn::Error::backend_retained_source(
            MediaIngressError::ForeignIdentity,
        )))
    }

    /// Consumes the same original plan under this workspace's metadata account.
    /// The default refuses before invoking an ordinary allocating producer.
    fn prepare_original_workspace_ingress_plan_with_metadata(
        input: crate::prepared_execution::OriginalMediaWorkspaceInput,
        _geometry: eredu_core::InferenceGeometry,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self::IngressPlan, Self::Error>
    where
        B: NeuralBackend<Tensor = eredu_nn::workspace::WorkspaceTensor>,
        Self: CompositeArchitecture<B, S, Error = eredu_nn::Error>,
    {
        Err(input.reject_ingress_with_metadata(MediaIngressError::ForeignIdentity, context))
    }

    /// Exact original ordered metadata, borrowed without reconstructing artifacts.
    fn prepared_ingress_input(
        plan: &Self::IngressPlan,
    ) -> PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>;
    /// Same encoder waves, excluding deferred decoder embedding lookups.
    fn media_group_collective_waves(
        &self,
        plan: &Self::IngressPlan,
        group: usize,
        tensor_partitions: usize,
        pipeline_stages: usize,
    ) -> Result<Option<Vec<Vec<CompositeTensorCollective>>>, String>;
    /// The same encoder-wave producer with prospective metadata funding.
    fn media_group_collective_waves_with_metadata(
        &self,_plan:&Self::IngressPlan,_group:usize,_tensor_partitions:usize,_pipeline_stages:usize,
        _context:&eredu_nn::workspace::WorkspaceContext,
    )->Result<Option<Vec<Vec<CompositeTensorCollective>>>,eredu_nn::Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }
    /// Exact segmented embedding lookup waves for this decoder interval.
    fn media_primary_ingress_collectives(
        &self,
        plan: &Self::IngressPlan,
        span: &eredu_runtime::prefill::PrefillChunk,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<CompositeTensorCollective>>, String>;    /// The same segmented lookup producer with prospective metadata funding.
    fn media_primary_ingress_collectives_with_metadata(
        &self,_plan:&Self::IngressPlan,_span:&eredu_runtime::prefill::PrefillChunk,_tensor_partitions:usize,
        _context:&eredu_nn::workspace::WorkspaceContext,
    )->Result<Option<Vec<CompositeTensorCollective>>,eredu_nn::Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }

}
