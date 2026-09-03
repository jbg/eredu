//! Typed prepared-input ingress for replicated composite architectures.

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_nn::{GroupedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    ArchitectureParameters, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    RoutedExpertProvider, RoutedLayeredArchitecture, StaticParameterVisitor,
    StaticParameterVisitorMut,
};

use crate::media_plan::AdmittedCompositeInput;

/// Exact prepared tensors paired with their architecture-owned admission proof.
pub struct PreparedCompositeInput<'a, T, P> {
    prepared: &'a eredu_runtime::PreparedModelInput<T>,
    admitted: &'a AdmittedCompositeInput<P>,
}

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

    /// Admits exact prepared tensor identity and derives ordered ingress plans.
    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<AdmittedCompositeInput<Self::InputPartPlan>, eredu_core::CapabilityError>;

    /// Builds architecture-native ingress and enters the ordinary graph lifecycle.
    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;
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
