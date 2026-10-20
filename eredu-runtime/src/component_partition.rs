//! Typed partition capabilities over the ordinary component adapter.
use crate::*;
use eredu_nn::{NeuralBackend, Tensor};
use std::{borrow::Borrow, ops::Range};

/// Selected local state and execution context for a partition equation.
pub struct ComponentPartitionContext<'a, B: NeuralBackend> {
    /// Selected entry-group state geometry, absent for parameter-only ingress.
    pub expected: Option<&'a StateLayout>,
    /// Driver-selected local state origin.
    pub first_state_ordinal: usize,
    /// Selected tensor-parallel communicator, when present.
    pub parallel: Option<&'a B::ParallelContext>,
    /// Native execution context supplied by the backend.
    pub native: &'a <B::Tensor as Tensor>::Context,
}

/// Explicit typed ingress and egress capability for partitioned components.
pub trait PartitionComponentKernel<
    B: NeuralBackend,
    S: RuntimeState<B> = <Self as ComponentKernel<B>>::State,
>: ParallelComponentKernel<B, S>
{
    /// Explicit internal observation coverage for the selected partition call path.
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> crate::inspection::ObservationHookSupport {
        Default::default()
    }
    /// Enters the same validated partition with optional internal instrumentation.
    fn begin_partition_observed<O>(
        &mut self,
        input: LayeredPartitionInput<
            '_,
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: ComponentPartitionContext<'_, B>,
        _observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.begin_partition(input, mask, state, selected)
    }
    /// Publishes the same local result with optional internal instrumentation.
    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    >
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.finish_partition(hidden, state, forward, owns_output, parallel, context)
    }
    /// Family-owned transport schema and auxiliary values.
    type Boundary: ArchitectureBoundary;
    /// Complete boundary geometry before route selection.
    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error>;
    /// Enters a selected partition from original input or a typed upstream value.
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<
            '_,
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;
    /// Assembles an already transported group input without graph dependency values.
    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.begin_group(group, initial, &[], state, forward, context)
    }
    /// Publishes the completed local result or its typed boundary.
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    >;
}

/// Pipeline request geometry and semantic transport conversion.
pub trait PipelineComponentKernel<
    B: NeuralBackend,
    S: RuntimeState<B> = <Self as ComponentKernel<B>>::State,
>: PartitionComponentKernel<B, S>
{
    /// Request extents used to resolve the wire schema.
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Self::Error>;
    /// Resumes the same component equation after pipeline transport.
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        )>,
        state: &mut S,
        selected: ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized;
    /// Complete final feature width.
    fn output_width(&self) -> i32;
    /// Exact authoritative output shape.
    fn output_shape(
        &self,
        input: &Self::Input<'_>,
        batch: i32,
        sequence: i32,
    ) -> Result<Vec<i32>, Self::Error> {
        let _ = input;
        Ok(vec![batch, sequence, self.output_width()])
    }
    /// Schema for the declared directed route.
    fn route_schema(&self, _route: &PartitionBoundaryRoute) -> Result<Self::Boundary, Self::Error> {
        self.boundary_schema()
    }
    /// Resolves route extents for request cadence while preserving the typed wire roles.
    fn resolved_route_schema(
        &self,
        route: &PartitionBoundaryRoute,
        batch: i32,
        sequence: i32,
    ) -> Result<ResolvedBoundaryWireSchema, Self::Error>
    where
        Self::Error: From<eredu_nn::Error>,
    {
        self.route_schema(route)?
            .wire_schema()
            .and_then(|schema| schema.resolve(batch, sequence))
            .map_err(|error| eredu_nn::Error::backend_source(error).into())
    }
    /// Whether selected vocabulary shards require sum/gather publication.
    fn tensor_vocabulary_sharded(&self) -> bool;
}

impl<B, D, K, S> PartitionedLayeredArchitecture<B, S> for ComponentAdapter<D, K>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    S: RuntimeState<B>,
    K: PartitionComponentKernel<B, S> + 'static,
{
    fn partition_observation_hooks(
        &self,
        tensor_parallel: bool,
    ) -> crate::inspection::ObservationHookSupport {
        self.kernel().partition_observation_hooks(tensor_parallel)
    }
    fn begin_partition_observed<'a, O>(
        &mut self,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &crate::StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let expected = self.partition_entry_layout(Some(expected));
        self.validate_partition_state(state.optional_layout(), Some(expected), None)?;
        self.kernel_mut().begin_partition_observed(
            input,
            mask,
            state,
            ComponentPartitionContext {
                expected,
                first_state_ordinal,
                parallel,
                native: context,
            },
            observer,
        )
    }
    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    >
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.validate_partition_state(state.optional_layout(), None, Some(owns_output))?;
        self.kernel_mut().finish_partition_observed(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            observer,
        )
    }
    type Boundary = K::Boundary;
    fn boundary_schema(&self) -> Result<Self::Boundary, K::Error> {
        self.kernel().boundary_schema()
    }
    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error> {
        let expected = self.partition_entry_layout(Some(expected));
        self.validate_partition_state(state.optional_layout(), Some(expected), None)?;
        self.kernel_mut().begin_partition(
            input,
            mask,
            state,
            ComponentPartitionContext {
                expected,
                first_state_ordinal,
                parallel: None,
                native: context,
            },
        )
    }
    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error> {
        let expected = self.partition_entry_layout(Some(expected));
        self.validate_partition_state(state.optional_layout(), Some(expected), None)?;
        self.kernel_mut().begin_partition(
            input,
            mask,
            state,
            ComponentPartitionContext {
                expected,
                first_state_ordinal,
                parallel: Some(parallel),
                native: context,
            },
        )
    }
    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut K::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.validate_partition_group(group)?;
        self.kernel_mut()
            .enter_partition_group(group, initial, state, forward, parallel, context)
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &K::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        K::Error,
    > {
        self.validate_partition_state(state.optional_layout(), None, Some(owns_output))?;
        self.kernel_mut()
            .finish_partition(hidden, state, forward, owns_output, parallel, context)
    }
}
impl<B, D, K, S> PipelinePartitionArchitecture<B, S> for ComponentAdapter<D, K>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    S: RuntimeState<B>,
    K: PipelineComponentKernel<B, S> + 'static,
{
    fn partition_input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), K::Error> {
        self.kernel().input_dimensions(input)
    }
    fn begin_pipeline_partition<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        )>,
        state: &mut S,
        expected: Option<&StateLayout>,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error>
    where
        O: ActivationObserver<B::Tensor, K::Error> + ?Sized,
    {
        self.validate_partition_state(state.optional_layout(), Some(expected), None)?;
        self.kernel_mut().begin_pipeline(
            input,
            incoming,
            state,
            ComponentPartitionContext {
                expected,
                first_state_ordinal,
                parallel,
                native: context,
            },
            observer,
        )
    }
    fn partition_output_width(&self) -> i32 {
        self.kernel().output_width()
    }
    fn partition_output_shape(
        &self,
        input: &Self::Input<'_>,
        batch: i32,
        sequence: i32,
    ) -> Result<Vec<i32>, K::Error> {
        self.kernel().output_shape(input, batch, sequence)
    }
    fn partition_boundary_schema(
        &self,
        route: &PartitionBoundaryRoute,
    ) -> Result<Self::Boundary, K::Error> {
        self.kernel().route_schema(route)
    }
    fn partition_resolved_boundary_schema(
        &self,
        route: &PartitionBoundaryRoute,
        batch: i32,
        sequence: i32,
    ) -> Result<ResolvedBoundaryWireSchema, K::Error>
    where
        K::Error: From<eredu_nn::Error>,
    {
        self.kernel().resolved_route_schema(route, batch, sequence)
    }
    fn partition_retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        _: usize,
    ) -> Range<usize> {
        self.local_state_slots(group, index)
    }
    fn partition_tensor_vocabulary_sharded(&self) -> bool {
        self.kernel().tensor_vocabulary_sharded()
    }
}

impl<D: Borrow<ComponentDefinition>, K> ComponentAdapter<D, K> {
    /// Enters the ordinary local-stage driver with its selected optional state geometry.
    pub fn begin_prepared_partition<B, S>(
        &mut self,
        input: LayeredPartitionInput<
            '_,
            B::Tensor,
            <K::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: Option<&StateLayout>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error>
    where
        B: NeuralBackend,
        S: RuntimeState<B>,
        K: PartitionComponentKernel<B, S>,
    {
        self.validate_partition_state(state.optional_layout(), Some(expected), None)?;
        self.kernel_mut().begin_partition(
            input,
            mask,
            state,
            ComponentPartitionContext {
                expected,
                first_state_ordinal: 0,
                parallel,
                native: context,
            },
        )
    }
}
