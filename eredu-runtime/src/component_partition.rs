//! Typed partition capabilities over the ordinary component adapter.
use crate::*;
use eredu_nn::{NeuralBackend, Tensor};
use std::{borrow::Borrow, ops::Range};

/// Selected local state and execution context for a partition equation.
pub struct ComponentPartitionContext<'a, B: NeuralBackend> {
    /// Exact selected local mutable-state geometry.
    pub expected: Option<&'a StateLayout>,
    /// Driver-selected local state origin.
    pub first_state_ordinal: usize,
    /// Selected tensor-parallel communicator, when present.
    pub parallel: Option<&'a B::ParallelContext>,
    /// Native execution context supplied by the backend.
    pub native: &'a <B::Tensor as Tensor>::Context,
}

/// Explicit typed ingress and egress capability for partitioned components.
pub trait PartitionComponentKernel<B: NeuralBackend>: ParallelComponentKernel<B> {
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
        state: &mut Self::State,
        selected: ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;
    /// Publishes the completed local result or its typed boundary.
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut Self::State,
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
pub trait PipelineComponentKernel<B: NeuralBackend>: PartitionComponentKernel<B> {
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
        state: &mut Self::State,
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
    ) -> Result<Vec<i32>, Self::Error>;
    /// Schema for the declared directed route.
    fn route_schema(&self, route: &PartitionBoundaryRoute) -> Result<Self::Boundary, Self::Error>;
    /// Whether selected vocabulary shards require sum/gather publication.
    fn tensor_vocabulary_sharded(&self) -> bool;
}

impl<B, D, K> PartitionedLayeredArchitecture<B, K::State> for ComponentAdapter<D, K>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    K: PartitionComponentKernel<B> + 'static,
{
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
        state: &mut K::State,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error> {
        self.validate_partition_state(state.optional_layout(), Some(expected), None)?;
        self.kernel_mut().begin_partition(
            input,
            mask,
            state,
            ComponentPartitionContext {
                expected: Some(expected),
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
        state: &mut K::State,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error> {
        self.validate_partition_state(state.optional_layout(), Some(expected), None)?;
        self.kernel_mut().begin_partition(
            input,
            mask,
            state,
            ComponentPartitionContext {
                expected: Some(expected),
                first_state_ordinal,
                parallel: Some(parallel),
                native: context,
            },
        )
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut K::State,
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
        self.validate_partition_state(
            state.optional_layout(),
            state.optional_layout(),
            Some(owns_output),
        )?;
        self.kernel_mut()
            .finish_partition(hidden, state, forward, owns_output, parallel, context)
    }
}
impl<B, D, K> PipelinePartitionArchitecture<B, K::State> for ComponentAdapter<D, K>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    K: PipelineComponentKernel<B> + 'static,
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
        state: &mut K::State,
        expected: Option<&StateLayout>,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error>
    where
        O: ActivationObserver<B::Tensor, K::Error> + ?Sized,
    {
        self.validate_partition_state(state.optional_layout(), expected, None)?;
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
