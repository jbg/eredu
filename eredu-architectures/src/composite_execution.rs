//! Typed prepared-input ingress for replicated composite architectures.

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_core::AttentionPolicy;
use eredu_nn::{GroupedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    ArchitectureParameters, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    ParallelLayeredArchitecture, PartitionedLayeredArchitecture, RoutedExpertProvider,
    RoutedLayeredArchitecture, StaticParameterVisitor, StaticParameterVisitorMut,
};

use crate::media_plan::AdmittedCompositeInput;

/// One request-bounded tensor collective in a composite pipeline wave.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CompositeTensorCollective {
    /// Sum one tensor with this exact logical shape across the selected TP group.
    Sum {
        /// Exact logical shape of the active or zero-work tensor.
        shape: Vec<i32>,
    },
}

/// Exact external-assistant capture requested from one ordinary target pass.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExternalPredictionCaptureRequest {
    /// Final decoder hidden state and every architecture-published shared K/V class.
    Gemma4SharedAttention {
        /// Exact final decoder unit-output observation path.
        final_hidden_path: String,
    },
    /// Ordered decoder-unit outputs consumed by a DFlash assistant.
    MuseGlimmerDFlash {
        /// Exact zero-based decoder layers, in assistant encoder order.
        target_layers: Box<[usize]>,
        /// Exact unit-output observation paths in the same order.
        target_paths: Box<[String]>,
    },
}

/// Architecture-owned values captured from one committed ordinary target pass.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ExternalPredictionTargetCapture<T> {
    /// Gemma target hidden state plus shared attention publications.
    Gemma4 {
        /// Final decoder activation before vocabulary projection.
        hidden: T,
        /// Shared K/V values keyed by their exact attention policy.
        shared_kv: Vec<(AttentionPolicy, T, T)>,
    },
    /// Muse-Glimmer decoder outputs in the requested target-layer order.
    MuseGlimmerDFlash {
        /// Exact ordered target-layer activations.
        target_states: Vec<T>,
    },
}

/// Target-owned static operation needed by an external assistant.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ExternalPredictionTargetOperation<'a, T> {
    /// Applies the ordinary target token embedding to assistant proposal IDs.
    TokenEmbeddings(&'a T),
    /// Applies the ordinary target vocabulary projection to assistant states.
    ProjectLogits(&'a T),
}

impl CompositeTensorCollective {
    /// Exact logical tensor shape submitted by active and zero-work ranks.
    pub fn shape(&self) -> &[i32] {
        match self {
            Self::Sum { shape } => shape,
        }
    }
}

/// Exact prepared tensors paired with their architecture-owned admission proof.
pub struct PreparedCompositeInput<'a, T, P> {
    prepared: &'a eredu_runtime::PreparedModelInput<T>,
    admitted: &'a AdmittedCompositeInput<P>,
}

impl<T, P> Clone for PreparedCompositeInput<'_, T, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, P> Copy for PreparedCompositeInput<'_, T, P> {}

impl<'a, T, P> PreparedCompositeInput<'a, T, P> {
    /// Couples exact prepared tensors to an admission derived from those tensors.
    pub fn new(
        prepared: &'a eredu_runtime::PreparedModelInput<T>,
        admitted: &'a AdmittedCompositeInput<P>,
    ) -> Result<Self, String> {
        if prepared.identity() != admitted.identity() {
            return Err("prepared composite input identity differs from admission".into());
        }
        if prepared.len() != admitted.parts().len() {
            return Err("prepared composite part count differs from admission".into());
        }
        Ok(Self { prepared, admitted })
    }

    /// Exact backend-native prepared tensor handles.
    pub const fn prepared(&self) -> &'a eredu_runtime::PreparedModelInput<T> {
        self.prepared
    }

    /// Exact ordered architecture admission.
    pub const fn admitted(&self) -> &'a AdmittedCompositeInput<P> {
        self.admitted
    }

    /// Derives prompt-cache identity from the exact prepared input paired with this admission.
    pub fn cache_identity(
        &self,
        semantic_content_fingerprint: impl Into<String>,
    ) -> Result<
        eredu_runtime::PreparedInputCacheIdentity,
        eredu_runtime::PreparedInputCacheIdentityError,
    > {
        self.prepared.cache_identity(semantic_content_fingerprint)
    }
}

/// Architecture-owned interpretation of admitted prepared input.
pub trait CompositeArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// One architecture-specific plan for each ordered input part.
    type InputPartPlan;
    /// Minimal normalized configuration retained for repeated input admission.
    type AdmissionConfig: Clone;

    /// Clones the normalized architecture facts required by input admission.
    fn admission_config(&self) -> Self::AdmissionConfig;

    /// Publishes the target facts required to prove external-assistant compatibility.
    fn external_assistant_target_profile(
        _config: &Self::AdmissionConfig,
    ) -> Option<crate::external_assistant::ExternalAssistantTargetProfile> {
        None
    }

    /// Admits exact prepared tensor identity and derives ordered ingress plans.
    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<AdmittedCompositeInput<Self::InputPartPlan>, eredu_core::CapabilityError>;

    /// Returns whether one request-optional execution group is active for the
    /// exact admitted input.
    ///
    /// This decision intentionally consumes the architecture-owned part plan,
    /// rather than modality presence alone: projected media embeddings retain
    /// their semantic modality but must not execute the corresponding native
    /// media tower.
    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool;

    /// Exact sequence extent emitted across one prepared group boundary.
    ///
    /// Most composite groups preserve the complete decoder extent. Media
    /// towers whose projected output occupies only their admitted placeholder
    /// span override this before the prepared input is consumed.
    fn prepared_group_boundary_sequence(
        &self,
        _group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        i32::try_from(input.admitted().decoder_positions())
            .map_err(|_| "prepared composite boundary sequence exceeds i32".to_owned())
    }

    /// Exact intermediate activation geometry for a pipeline continuation
    /// within one execution group.
    ///
    /// `None` means the group preserves the ordinary decoder-width
    /// `[batch, sequence, hidden]` boundary. Media towers whose internal
    /// workspace differs from their projected output declare its exact
    /// sequence and width here.
    fn prepared_group_continuation_geometry(
        &self,
        _group: usize,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<(i32, i32)>, String> {
        Ok(None)
    }

    /// Whether a group-local continuation transports its internal activation
    /// with an explicit leading batch dimension.
    ///
    /// Shared Qwen vision blocks normally use batched activations. Conditional
    /// Qwen keeps its flattened patch matrix unbatched between vision owners.
    fn prepared_group_continuation_batched(&self, _group: usize) -> bool {
        true
    }

    /// Exact TP collective sequence for every PP stage of one prepared group.
    ///
    /// `None` means the architecture declares no shared-world schedule for the
    /// group. A tensor-sharded optional group that is active under TP+PP must
    /// return one stage entry (possibly empty) for every pipeline stage.
    fn prepared_group_collective_waves(
        &self,
        _group: usize,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        _tensor_partitions: usize,
        _pipeline_stages: usize,
    ) -> Result<Option<Vec<Vec<CompositeTensorCollective>>>, String> {
        Ok(None)
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

    /// Exact TP reductions surrounding one routed decoder unit.
    ///
    /// The returned `(before, after)` order is carried unchanged into inactive
    /// pipeline waves. It must describe the concrete family equation rather
    /// than a generic transformer default.
    fn routed_tensor_reductions(
        &self,
        _unit: usize,
        _routed: bool,
    ) -> Result<(usize, usize), Self::Error> {
        Ok((1, 1))
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
        _selected: &eredu_runtime::ResolvedBoundaryWireSchema,
        _batch: i32,
        _source_sequence: i32,
        _group_sequences: &[i32],
        _continuation: Option<(i32, i32)>,
    ) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Self::Error> {
        Ok(None)
    }

    /// Encodes architecture-owned context for one continuation or dependency edge.
    fn partition_boundary_values(
        &self,
        _source_group: usize,
        _destination_group: usize,
        _schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        _hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>, Self::Error> {
        Ok(None)
    }

    /// Installs a typed continuation or dependency before its destination begins.
    fn accept_partition_boundary(
        &mut self,
        _source_group: usize,
        _destination_group: usize,
        _schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        _values: Vec<B::Tensor>,
        _forward: &mut Self::ForwardContext,
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

    /// Applies one target-owned static operation without transferring ordinary target ownership.
    fn external_prediction_target_operation(
        &mut self,
        _operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        Ok(None)
    }

    /// Borrows the exact target value retained for an embedded prediction extension.
    fn prediction_target_capture(_forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        None
    }

    /// Declares the exact target-capture placeholder on a non-output pipeline rank.
    fn prediction_target_placeholder_shape(
        &self,
        _forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(None)
    }

    /// Builds architecture-native ingress and enters the ordinary graph lifecycle.
    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Builds architecture-native ingress through the selected tensor-parallel
    /// embedding boundary, then enters the same graph lifecycle.
    fn begin_composite_forward_parallel<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend;
}

/// Additive input adapter over the same architecture modules and graph lifecycle.
pub struct PreparedCompositeArchitecture<A> {
    inner: A,
}

impl<A> PreparedCompositeArchitecture<A> {
    /// Wraps one constructed composite architecture.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }

    /// Borrows the underlying architecture.
    pub const fn inner(&self) -> &A {
        &self.inner
    }

    /// Mutably borrows the underlying architecture for generic partition execution.
    pub fn inner_mut(&mut self) -> &mut A {
        &mut self.inner
    }

    /// Consumes the adapter.
    pub fn into_inner(self) -> A {
        self.inner
    }
}

impl<A, B> ArchitectureParameters<B> for PreparedCompositeArchitecture<A>
where
    B: NeuralBackend,
    A: ArchitectureParameters<B>,
{
    type DefinitionError = A::DefinitionError;

    fn state_layout(&self) -> Result<eredu_runtime::StateLayout, Self::DefinitionError> {
        self.inner.state_layout()
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        self.inner.state_identity(state, topology)
    }

    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Self::DefinitionError> {
        self.inner.parameter_description(context)
    }

    fn static_parameter_recipes(
        &self,
        source: &dyn CheckpointSource,
    ) -> Result<std::collections::BTreeMap<String, DerivedWeightRecipe>, String> {
        self.inner.static_parameter_recipes(source)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitor<B>,
    {
        self.inner.visit_static_parameters(visitor)
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitorMut<B>,
    {
        self.inner.visit_static_parameters_mut(visitor)
    }
}

impl<A, B, S> LayeredArchitecture<B, S> for PreparedCompositeArchitecture<A>
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
{
    type Input<'a>
        = PreparedCompositeInput<'a, B::Tensor, A::InputPartPlan>
    where
        Self: 'a;
    type StaticModules = A::StaticModules;
    type Unit = A::Unit;
    type ForwardContext = A::ForwardContext;
    type RetainedContextValues<'a>
        = A::RetainedContextValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = A::Error;

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        self.inner.group_transport(group)
    }

    fn primary_execution_group(&self) -> &str {
        self.inner.primary_execution_group()
    }

    fn prediction_execution_groups(&self) -> Vec<String> {
        self.inner.prediction_execution_groups()
    }

    fn prediction_target_capture(context: &Self::ForwardContext) -> Option<&B::Tensor> {
        <A as CompositeArchitecture<B, S>>::prediction_target_capture(context)
    }

    fn prediction_target_placeholder_shape(
        &self,
        forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        <A as CompositeArchitecture<B, S>>::prediction_target_placeholder_shape(
            &self.inner,
            forward,
        )
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        self.inner.state_partition_plan(layout)
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Self::Error> {
        self.inner.execution_graph()
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        self.inner.group_unit_count(group)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        self.inner.unit_path(group, index)
    }

    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        self.inner.group_input_observation_path(group)
    }

    fn group_output_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        self.inner.group_output_observation_path(group)
    }

    fn static_modules(&self) -> &Self::StaticModules {
        self.inner.static_modules()
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self.inner.static_modules_mut()
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        self.inner.build_unit(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner.begin_composite_forward(input, state, context)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .begin_execution_group(group, initial, dependencies, state, forward, context)
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        self.inner.should_execute_group(group, forward)
    }

    fn state_ordinal(&self, group: usize, index: usize, ordinal: usize) -> usize {
        self.inner.state_ordinal(group, index, ordinal)
    }

    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        ordinal: usize,
    ) -> std::ops::Range<usize> {
        self.inner.retained_state_ordinals(group, index, ordinal)
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .forward_unit(group, index, unit, hidden, state, forward, context)
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .complete_execution_group(group, hidden, state, forward, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner.finish_forward(hidden, state, forward, context)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        group: usize,
        index: usize,
    ) -> Self::RetainedContextValues<'a> {
        self.inner.retained_context_values(forward, group, index)
    }
}

impl<A, B, S> RoutedLayeredArchitecture<B, S> for PreparedCompositeArchitecture<A>
where
    B: GroupedNeuralBackend,
    S: LayerRuntimeState<B>,
    A: CompositeArchitecture<B, S> + RoutedLayeredArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
{
    fn routed_observation_point(
        &self,
        group: usize,
        index: usize,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoint>, Self::Error> {
        self.inner.routed_observation_point(group, index)
    }

    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.inner.forward_unit_with_provider(
            group, index, unit, hidden, state, forward, pass, provider, context,
        )
    }
}

impl<A, B, S> ParallelLayeredArchitecture<B, S> for PreparedCompositeArchitecture<A>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeArchitecture<B, S> + ParallelLayeredArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner
            .begin_composite_forward_parallel(input, state, parallel, context)
    }

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner.forward_unit_parallel(
            group, index, unit, hidden, state, forward, parallel, context,
        )
    }

    fn begin_execution_group_parallel(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner.begin_execution_group_parallel(
            group,
            initial,
            dependencies,
            state,
            forward,
            parallel,
            context,
        )
    }

    fn complete_execution_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .complete_execution_group_parallel(group, hidden, state, forward, parallel, context)
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .finish_forward_parallel(hidden, state, forward, parallel, context)
    }
}

impl<A, B, S> PartitionedLayeredArchitecture<B, S> for PreparedCompositeArchitecture<A>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeArchitecture<B, S> + PartitionedLayeredArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
{
    type Boundary = A::Boundary;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        self.inner.boundary_schema()
    }

    fn begin_partition<'a>(
        &mut self,
        input: eredu_runtime::LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &eredu_runtime::StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner
            .begin_partition(input, mask, state, expected, first_state_ordinal, context)
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: eredu_runtime::LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &eredu_runtime::StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.inner.begin_partition_parallel(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            parallel,
            context,
        )
    }

    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .enter_partition_group(group, initial, state, forward, parallel, context)
    }

    fn leave_partition_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.inner
            .leave_partition_group(group, hidden, state, forward, parallel, context)
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        eredu_runtime::LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    > {
        self.inner
            .finish_partition(hidden, state, forward, owns_output, parallel, context)
    }
}
