//! Pipeline entry and exit retain the ordinary request and global unit identities.
use super::super::input::RequestBoundary;
use super::*;
use eredu_runtime::{
    ActivationObserver, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    LayeredPartitionInput, LayeredPartitionOutput, ParallelLayeredArchitecture, PartitionState,
    PartitionedLayeredArchitecture, ReplicatedTextArchitecture,
};

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TargetModel<B> {
    /// Exact collective sequence shared by text and media partition adapters.
    pub fn routed_tensor_reductions(
        &self,
        unit: usize,
        routed: bool,
    ) -> Result<crate::partitioned_execution::RoutedTensorReductions, Error> {
        use crate::partitioned_execution::{
            RoutedTensorDimension as Dimension, RoutedTensorReduction as Reduction,
            RoutedTensorReductions,
        };
        match self.spec.units.get(unit) {
            Some(UnitSpec::Lexical { spec, .. }) if !routed => {
                let table = spec.embedding.lookup_spec();
                Ok(RoutedTensorReductions {
                    before: vec![
                        Reduction::Status,
                        Reduction::Tensor {
                            shape: vec![
                                Dimension::Tokens(spec.embedding.output_width() / table.dimensions),
                                Dimension::Fixed(table.dimensions),
                            ],
                            dtype: table.output_type,
                        },
                    ],
                    after: Vec::new(),
                })
            }
            Some(UnitSpec::Decoder { .. }) if routed => Ok(RoutedTensorReductions::hidden(1, 1)),
            _ => Err(Error::backend(
                "target routed collective selection differs from unit",
            )),
        }
    }

    /// Retains exact pipeline state ownership without changing global unit addresses.
    pub fn set_partition_state(&mut self, state: &PartitionState) -> Result<(), Error> {
        let start = state.global_layer_offset();
        let end = start
            .checked_add(state.layout().len())
            .ok_or_else(|| Error::backend("target partition state range overflow"))?;
        let global = self.spec.state_layout()?;
        if state.layout().is_empty()
            || end > global.len()
            || global.slice(start..end).map_err(Error::backend)? != *state.layout()
        {
            return Err(Error::backend(
                "target partition state differs from declared geometry",
            ));
        }
        self.partition_state = state.clone();
        Ok(())
    }

    pub(super) fn state_index(&self, index: usize) -> Result<usize, Error> {
        index
            .checked_sub(self.partition_state.global_layer_offset())
            .filter(|index| *index < self.partition_state.layout().len())
            .ok_or_else(|| Error::backend("target unit is outside its partition state"))
    }

    pub(super) fn validate_partition_state<S: LayerRuntimeState<B>>(
        &self,
        state: &S,
        expected: &StateLayout,
    ) -> Result<(), Error> {
        if state.layout() != expected || expected != self.partition_state.layout() {
            return Err(Error::backend(
                "target runtime state differs from selected partition",
            ));
        }
        Ok(())
    }

    pub(super) fn partition_offset<S: LayerRuntimeState<B>>(
        &self,
        state: &mut S,
    ) -> Result<i32, Error>
    where
        S::LayerState: RuntimeStateComponents<B>,
    {
        let offset = state.layer(0).map_err(Error::backend)?.position();
        for index in 1..self.partition_state.layout().len() {
            if state.layer(index).map_err(Error::backend)?.position() != offset {
                return Err(Error::backend(
                    "target units have inconsistent prefix positions",
                ));
            }
        }
        Ok(offset)
    }

    fn validate_partition_entry<S: LayerRuntimeState<B>>(
        &self,
        state: &S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
    ) -> Result<(), Error> {
        self.validate_partition_state(state, expected)?;
        if first_state_ordinal != 0 {
            return Err(Error::backend(
                "target partition state must begin at local ordinal zero",
            ));
        }
        match parallel {
            Some(parallel) => self.validate_parallel(parallel),
            None if self
                .tensor_partition
                .as_ref()
                .is_some_and(|partition| partition.ranks() > 1) =>
            {
                Err(Error::backend(
                    "rank-local target requires parallel partition execution",
                ))
            }
            None => Ok(()),
        }
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for TargetModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    type Boundary = RequestBoundarySchema;

    fn partition_observation_hooks(
        &self,
        _: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }

    fn boundary_schema(&self) -> Result<Self::Boundary, Error> {
        self.spec.boundary_schema()
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, RequestBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_partition_observed(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            None,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, RequestBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_partition_observed(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            Some(parallel),
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }

    fn begin_partition_observed<'a, O>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, RequestBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.validate_partition_entry(state, expected, first_state_ordinal, parallel)?;
        match input {
            LayeredPartitionInput::Tokens(tokens) => self.begin_forward_observed(
                <Self as ReplicatedTextArchitecture<B, S>>::text_input(tokens, mask),
                state,
                context,
                observer,
            ),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                let offset = self.partition_offset(state)?;
                self.resume_request(hidden, auxiliary, offset, context)
            }
        }
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, RequestBoundary<B::Tensor>>, Error> {
        self.finish_partition_observed(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, RequestBoundary<B::Tensor>>, Error>
    where
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: forward.request.boundary(),
            });
        }
        let output = match parallel {
            Some(parallel) => self.finish_forward_parallel_observed(
                hidden, state, forward, parallel, context, observer,
            )?,
            None => self.finish_forward_observed(hidden, state, forward, context, observer)?,
        };
        Ok(LayeredPartitionOutput::Final {
            output,
            retained: Some(hidden.clone()),
        })
    }
}

impl<B, S> crate::partitioned_execution::TextPartitionArchitecture<B, S> for TargetModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn partition_input_dimensions(input: &Self::Input<'_>) -> Result<(i32, i32), Error> {
        if input.batch <= 0 || input.tokens <= 0 {
            return Err(Error::backend(
                "target partition requires positive input dimensions",
            ));
        }
        Ok((input.batch, input.tokens))
    }

    fn begin_text_partition<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, RequestBoundary<B::Tensor>)>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.validate_partition_entry(state, expected, first_state_ordinal, parallel)?;
        match incoming {
            Some((hidden, boundary)) => {
                if hidden.shape().get(..2) != Some(&[input.batch, input.tokens]) {
                    return Err(Error::backend_source(
                        super::super::input::RequestError::Boundary,
                    ));
                }
                let offset = self.partition_offset(state)?;
                self.resume_request(hidden, boundary, offset, context)
            }
            None => self.begin_forward_observed(input, state, context, observer),
        }
    }

    fn partition_output_width(&self) -> i32 {
        self.spec.config.vocabulary
    }

    fn partition_tensor_vocabulary_sharded(&self) -> bool {
        false
    }

    fn partition_routed_tensor_reductions(
        &self,
        unit: usize,
        routed: bool,
    ) -> Result<crate::partitioned_execution::RoutedTensorReductions, Error> {
        self.routed_tensor_reductions(unit, routed)
    }
}
