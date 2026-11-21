//! Composite family hooks and their shared runtime adapter.

use super::*;
use crate::{ComponentAdapter, ComponentDefinition, ComponentEquations, ComponentResourceAccess};
use std::borrow::Borrow;

/// Family-authored composite admission and equations implemented on the local kernel.
pub trait CompositeComponentKernel<B: NeuralBackend, S: RuntimeState<B>>:
    ComponentEquations<B, S> + 'static
{
    /// Binds admitted input to the same component execution and resource lifecycle.
    fn prepared_input<'a>(
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    ) -> Self::Input<'a>;
    /// Recovers the admission proof retained by a prepared invocation.
    fn prepared_view<'a>(
        input: &Self::Input<'a>,
    ) -> Option<PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>>;

    /// Iterative output geometry, selected independently from prepared media.
    fn block_refinement_geometry(
        _config: &Self::AdmissionConfig,
    ) -> Option<crate::generation::refinement::architecture::BlockRefinementGeometry> {
        None
    }
    /// Exact request-local attention equations over prepared token rows.
    fn supports_attention_invocation(
        config: &Self::AdmissionConfig,
        invocation: eredu_core::decoding::AttentionInvocation,
    ) -> bool {
        invocation == eredu_core::decoding::AttentionInvocation::CAUSAL
            || (invocation == eredu_core::decoding::AttentionInvocation::DIFFUSION
                && Self::block_refinement_geometry(config).is_some())
    }
    /// Prepared requests implement partial invocations with retained continuations.
    fn supports_retained_prefill() -> bool {
        false
    }

    /// Completed immutable assembled decoder ingress retained by a prefill cursor.
    fn prefill_decoder_continuation(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        let _ = forward;
        None
    }

    /// Ordinary unobserved plain-text passes preserve state across prefix boundaries.
    /// This does not authorize splitting prepared media or observed invocations.
    fn supports_chunked_text_prefill() -> bool {
        false
    }

    /// One architecture-specific plan for each ordered input part, with a neutral
    /// accounting projection of the same admitted geometry.
    type InputPartPlan: Clone
        + Into<crate::executable_binding::retained_session::PreparedInputPartPlan>
        + 'static;
    /// Owns request preparation and immutable encoder continuation while the
    /// ordinary runtime session owns traversal, commitment and observation.
    type PrefillRequest: Clone;

    /// Declares an invocation-specific publication width when it differs from
    /// the selected output projection, such as an authoritative encoder result.
    fn prepared_publication_width(
        &self,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<i32>, String> {
        Ok(None)
    }

    /// Maximum target invocation for an admitted request, when the architecture
    /// has a bounded continuation contract. `None` retains one invocation.
    fn prefill_chunk_limit(
        _admitted: &AdmittedCompositeInput<Self::InputPartPlan>,
    ) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    /// Creates the architecture-selected request cursor input from exact admitted
    /// tensors. Families with shared encoders prepare their request products once.
    fn prepare_prefill_request(
        prepared: eredu_core::PreparedModelInput<B::Tensor>,
        admitted: AdmittedCompositeInput<Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Self::Error>;
    /// Minimal normalized configuration retained for repeated input admission.
    type AdmissionConfig: Clone;

    /// Clones the normalized architecture facts required by input admission.
    fn admission_config(&self) -> Self::AdmissionConfig;

    /// Whether this target exposes the invocation contract used by an external assistant.
    fn supports_external_assistant(_config: &Self::AdmissionConfig) -> bool {
        false
    }

    /// Admits exact prepared tensor identity and derives ordered ingress plans.
    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_core::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_core::input::PreparedInputInspector<B::Tensor>,
    ) -> Result<AdmittedCompositeInput<Self::InputPartPlan>, eredu_core::CapabilityError>;

    /// Reconstructs the target's exact semantic token sequence for a prediction
    /// lane, including architecture-declared media placeholders. The default
    /// accepts ordinary text token parts; media-aware families share their
    /// ingress token construction with this operation.
    fn prepared_prediction_token_ids(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        B::Tensor::concatenate(&prepared_token_parts(input, context, |_| None)?, 1, context)
    }

    /// Plans one group from admitted input and selected group-local stage ranges.
    fn prepared_group_invocation(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        stage_units: &[std::ops::Range<usize>],
    ) -> Result<CompositeGroupInvocation, String>;

    /// Decoder batch extent. Media-first inputs can have a patch or frame axis
    /// instead of a batch axis on their first payload.
    fn prepared_decoder_batch_size(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        input
            .prepared()
            .parts()
            .first()
            .map(|part| part.payload().value().dim(0))
            .ok_or_else(|| "composite request has no prepared parts".into())
    }

    /// Exact TP collectives emitted while the primary decoder ingress is assembled.
    ///
    /// `None` preserves the ordinary single decoder-extent embedding sum. Composite
    /// architectures which look up independently segmented token parts override this
    /// with their request-bounded operation shapes and order.
    fn prepared_primary_ingress_collectives(
        &self,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        _tensor_partitions: usize,
    ) -> Result<Option<Vec<CompositeTensorCollective>>, String> {
        Ok(None)
    }

    /// Whether a retained context still has decoder-ingress collectives to run.
    ///
    /// Media may create a context before deferred token lookups are resolved.
    /// Inactive pipeline ranks must mirror those lookups in the primary wave.
    /// The default describes architectures which resolve ingress at context creation.
    fn primary_ingress_collectives_pending(&self, _forward: &Self::ForwardContext) -> bool {
        false
    }

    /// Exact TP reductions surrounding one routed decoder unit.
    ///
    /// The returned typed before/after order is carried unchanged into inactive
    /// pipeline waves. It must describe the concrete family equation rather
    /// than a generic transformer default.
    fn routed_tensor_reductions(
        &self,
        _unit: usize,
        _routed: bool,
    ) -> Result<crate::RoutedTensorReductions, Self::Error> {
        Ok(crate::RoutedTensorReductions::hidden(1, 1))
    }

    /// Whether primary token embedding and final vocabulary use TP collectives.
    /// Replicated vocabulary equations omit those otherwise implicit idle waves.
    fn routed_tensor_vocabulary_sharded(&self) -> bool {
        true
    }

    /// Physical vocabulary width gathered by the routed TP output equation.
    ///
    /// `None` means the published logical width is also the physical sharded
    /// width. Architectures which gather padded rows before trimming expose
    /// that checkpoint width here so inactive PP ranks submit the exact same
    /// collective shape.
    fn routed_tensor_output_width(&self) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    /// Resolves the exact typed boundary for one continuation or dependency edge.
    ///
    /// `None` declares the ordinary primary-activation-only edge. Families with
    /// learned request context produced inside a partitioned optional root declare
    /// every additional role and its invocation geometry here.
    fn partition_boundary_schema(
        &self,
        _source_group: usize,
        _destination_group: usize,
        _selected: &crate::ResolvedBoundaryWireSchema,
        _batch: i32,
        _source_sequence: i32,
        _groups: &[CompositeGroupInvocation],
        _continuation: Option<(i32, i32)>,
    ) -> Result<Option<crate::ResolvedBoundaryWireSchema>, Self::Error> {
        Ok(None)
    }

    /// Encodes architecture-owned context for one continuation or dependency edge.
    fn partition_boundary_values(
        &self,
        _source_group: usize,
        _destination_group: usize,
        _schema: &crate::ResolvedBoundaryWireSchema,
        _hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<Vec<crate::ArchitectureBoundaryValue<B::Tensor>>>, Self::Error> {
        Ok(None)
    }

    /// Installs a typed continuation or dependency before its destination begins.
    fn accept_partition_boundary(
        &mut self,
        _source_group: usize,
        _destination_group: usize,
        _schema: &crate::ResolvedBoundaryWireSchema,
        _values: Vec<B::Tensor>,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        Ok(None)
    }

    /// Returns the exact unit-output paths required by an external-assistant capture.
    ///
    /// The default keeps prediction unavailable. Implementations must reject a request whose
    /// family or geometry does not match this target; callers never infer paths from family names.
    fn external_prediction_capture_paths(
        _request: &ExternalPredictionCaptureRequest,
    ) -> Result<Option<Vec<String>>, Self::Error> {
        Ok(None)
    }

    /// Forms a typed assistant capture from architecture-retained context and exact observed paths.
    fn external_prediction_capture(
        _request: &ExternalPredictionCaptureRequest,
        _forward: &Self::ForwardContext,
        _observed: Vec<B::Tensor>,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Self::Error> {
        Ok(None)
    }

    /// Applies one complete target-owned static operation. Contributions are
    /// combined before the architecture's finishing equations run.
    fn external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        match self.external_prediction_target_contribution(operation, parallel, context)? {
            Some(value) => {
                self.finish_external_prediction_target_operation(operation, value, context)
            }
            None => Ok(None),
        }
    }

    /// Computes raw lookup/projection values. With no parallel context a
    /// vocabulary-sharded module emits only its local contribution; normalization
    /// of embeddings and output transforms belong to the finishing stage.
    /// An ordinary parallel context combines vocabulary shards before returning.
    fn external_prediction_target_contribution(
        &mut self,
        _operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        _parallel: Option<&B::ParallelContext>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        Ok(None)
    }

    /// Finishes a complete combined value. This does not look up or project
    /// parameters, and must not receive an uncombined contribution.
    fn finish_external_prediction_target_operation(
        &self,
        _operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        _combined: B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        Ok(None)
    }
}

/// Declares that the family composite ingress supports the parallel adapter.
pub trait ParallelCompositeComponentKernel<B: NeuralBackend, S: RuntimeState<B>>:
    CompositeComponentKernel<B, S>
{
}

impl<B, D, K, S, R> CompositeArchitecture<B, S> for ComponentAdapter<D, K, R>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    R: ComponentResourceAccess<B, K> + 'static,
    S: RuntimeState<B>,
    K: CompositeComponentKernel<B, S>,
    K::PrefillRequest: ChunkedPrefillRequest<Self, B, S>,
{
    type InputPartPlan = K::InputPartPlan;
    type PrefillRequest = K::PrefillRequest;
    type AdmissionConfig = K::AdmissionConfig;

    fn prepared_input<'a>(
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    ) -> Self::Input<'a> {
        <K as CompositeComponentKernel<B, S>>::prepared_input(input)
    }

    fn prepared_view<'a>(
        input: &Self::Input<'a>,
    ) -> Option<PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>> {
        <K as CompositeComponentKernel<B, S>>::prepared_view(input)
    }

    fn block_refinement_geometry(
        config: &Self::AdmissionConfig,
    ) -> Option<crate::generation::refinement::architecture::BlockRefinementGeometry> {
        <K as CompositeComponentKernel<B, S>>::block_refinement_geometry(config)
    }

    fn supports_attention_invocation(
        config: &Self::AdmissionConfig,
        invocation: eredu_core::decoding::AttentionInvocation,
    ) -> bool {
        <K as CompositeComponentKernel<B, S>>::supports_attention_invocation(config, invocation)
    }

    fn supports_retained_prefill() -> bool {
        <K as CompositeComponentKernel<B, S>>::supports_retained_prefill()
    }

    fn prefill_decoder_continuation(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        <K as CompositeComponentKernel<B, S>>::prefill_decoder_continuation(forward)
    }

    fn supports_chunked_text_prefill() -> bool {
        <K as CompositeComponentKernel<B, S>>::supports_chunked_text_prefill()
    }

    fn prepared_publication_width(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<i32>, String> {
        <K as CompositeComponentKernel<B, S>>::prepared_publication_width(self.kernel(), input)
    }

    fn prefill_chunk_limit(
        admitted: &AdmittedCompositeInput<Self::InputPartPlan>,
    ) -> Result<Option<usize>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::prefill_chunk_limit(admitted)
    }

    fn prepare_prefill_request(
        prepared: eredu_core::PreparedModelInput<B::Tensor>,
        admitted: AdmittedCompositeInput<Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::prepare_prefill_request(prepared, admitted, context)
    }

    fn admission_config(&self) -> Self::AdmissionConfig {
        <K as CompositeComponentKernel<B, S>>::admission_config(self.kernel())
    }

    fn supports_external_assistant(config: &Self::AdmissionConfig) -> bool {
        <K as CompositeComponentKernel<B, S>>::supports_external_assistant(config)
    }

    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_core::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_core::input::PreparedInputInspector<B::Tensor>,
    ) -> Result<AdmittedCompositeInput<Self::InputPartPlan>, eredu_core::CapabilityError> {
        <K as CompositeComponentKernel<B, S>>::admit_prepared_input(config, input, inspector)
    }

    fn prepared_prediction_token_ids(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        <K as CompositeComponentKernel<B, S>>::prepared_prediction_token_ids(input, context)
    }

    fn prepared_group_invocation(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        stage_units: &[std::ops::Range<usize>],
    ) -> Result<CompositeGroupInvocation, String> {
        <K as CompositeComponentKernel<B, S>>::prepared_group_invocation(
            self.kernel(),
            group,
            input,
            tensor_partitions,
            stage_units,
        )
    }

    fn prepared_decoder_batch_size(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        <K as CompositeComponentKernel<B, S>>::prepared_decoder_batch_size(self.kernel(), input)
    }

    fn prepared_primary_ingress_collectives(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<CompositeTensorCollective>>, String> {
        <K as CompositeComponentKernel<B, S>>::prepared_primary_ingress_collectives(
            self.kernel(),
            input,
            tensor_partitions,
        )
    }

    fn primary_ingress_collectives_pending(&self, forward: &Self::ForwardContext) -> bool {
        <K as CompositeComponentKernel<B, S>>::primary_ingress_collectives_pending(
            self.kernel(),
            forward,
        )
    }

    fn routed_tensor_reductions(
        &self,
        unit: usize,
        routed: bool,
    ) -> Result<crate::RoutedTensorReductions, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::routed_tensor_reductions(self.kernel(), unit, routed)
    }

    fn routed_tensor_vocabulary_sharded(&self) -> bool {
        <K as CompositeComponentKernel<B, S>>::routed_tensor_vocabulary_sharded(self.kernel())
    }

    fn routed_tensor_output_width(&self) -> Result<Option<usize>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::routed_tensor_output_width(self.kernel())
    }

    fn partition_boundary_schema(
        &self,
        source_group: usize,
        destination_group: usize,
        selected: &crate::ResolvedBoundaryWireSchema,
        batch: i32,
        source_sequence: i32,
        groups: &[CompositeGroupInvocation],
        continuation: Option<(i32, i32)>,
    ) -> Result<Option<crate::ResolvedBoundaryWireSchema>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::partition_boundary_schema(
            self.kernel(),
            source_group,
            destination_group,
            selected,
            batch,
            source_sequence,
            groups,
            continuation,
        )
    }

    fn partition_boundary_values(
        &self,
        source_group: usize,
        destination_group: usize,
        schema: &crate::ResolvedBoundaryWireSchema,
        hidden: &B::Tensor,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<Vec<crate::ArchitectureBoundaryValue<B::Tensor>>>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::partition_boundary_values(
            self.kernel(),
            source_group,
            destination_group,
            schema,
            hidden,
            forward,
            context,
        )
    }

    fn accept_partition_boundary(
        &mut self,
        source_group: usize,
        destination_group: usize,
        schema: &crate::ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::accept_partition_boundary(
            self.kernel_mut(),
            source_group,
            destination_group,
            schema,
            values,
            forward,
            context,
        )
    }

    fn external_prediction_capture_paths(
        request: &ExternalPredictionCaptureRequest,
    ) -> Result<Option<Vec<String>>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::external_prediction_capture_paths(request)
    }

    fn external_prediction_capture(
        request: &ExternalPredictionCaptureRequest,
        forward: &Self::ForwardContext,
        observed: Vec<B::Tensor>,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::external_prediction_capture(
            request, forward, observed,
        )
    }

    fn external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::external_prediction_target_operation(
            self.kernel_mut(),
            operation,
            parallel,
            context,
        )
    }

    fn external_prediction_target_contribution(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::external_prediction_target_contribution(
            self.kernel_mut(),
            operation,
            parallel,
            context,
        )
    }

    fn finish_external_prediction_target_operation(
        &self,
        operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        combined: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        <K as CompositeComponentKernel<B, S>>::finish_external_prediction_target_operation(
            self.kernel(),
            operation,
            combined,
            context,
        )
    }
}

impl<B, D, K, S, R> ParallelCompositeArchitecture<B, S> for ComponentAdapter<D, K, R>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    K: ParallelCompositeComponentKernel<B, S>,
    Self: CompositeArchitecture<B, S> + ParallelLayeredArchitecture<B, S>,
{
}
