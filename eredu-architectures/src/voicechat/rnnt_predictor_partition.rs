//! Typed cut and invocation identity for recurrent-layer pipeline transport.
use super::{
    rnnt::{RnntDimensions, RnntError},
    rnnt_predictor::{
        invalid, rnnt_predictor_state_layout, validate_row, RnntPredictorForward,
        RnntPredictorKernel, RnntPredictorRuntimeState,
    },
};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor, TensorElementType};
use eredu_runtime::*;
use std::{collections::BTreeMap, ops::Range};
fn boundary_error(error: ArchitectureBoundaryError) -> RnntError {
    eredu_nn::Error::backend_source(error).into()
}
/// Original typed-zero prototype and exact cut/step alongside a recurrent activation.
#[derive(Clone, Debug)]
pub struct RnntPredictorBoundary<T> {
    /// Original F32 input prototype used by every zero-state LSTM layer.
    pub prototype: T,
    /// I32 `[3]`: completed global layer cut, low and high words of the u64 step.
    pub request: T,
}
/// Exact recurrent-width wire geometry for one globally indexed cut.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RnntPredictorBoundarySchema {
    dimensions: RnntDimensions,
    cut: usize,
}
impl RnntPredictorBoundarySchema {
    /// Declares original predictor ingress geometry.
    pub fn new(dimensions: RnntDimensions) -> Result<Self, RnntError> {
        dimensions.validate()?;
        Ok(Self { dimensions, cut: 0 })
    }
    /// Selects an intermediate globally indexed cut.
    pub fn at_cut(&self, cut: usize) -> Result<Self, RnntError> {
        if cut >= self.dimensions.predictor_layers {
            return Err(invalid("RNNT predictor cut is not intermediate"));
        }
        Ok(Self {
            dimensions: self.dimensions,
            cut,
        })
    }
    /// Resolves this fixed single-stream row for generic pipeline transport.
    pub fn resolve(
        &self,
        batch: i32,
        sequence: i32,
    ) -> Result<ResolvedBoundaryWireSchema, RnntError> {
        if batch != 1 || sequence != 1 {
            return Err(invalid(
                "RNNT predictor transport requires one recurrent step",
            ));
        }
        self.wire_schema()
            .and_then(|schema| schema.resolve(batch, sequence))
            .map_err(boundary_error)
    }
    /// Exact completed global layer count.
    pub fn cut(&self) -> usize {
        self.cut
    }
}
impl ArchitectureBoundary for RnntPredictorBoundarySchema {
    type Boundary<T> = RnntPredictorBoundary<T>;
    const IDENTITY: &'static str = "voicechat.rnnt.predictor.v1";
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        BoundaryTensorSpec::new(
            "hidden",
            [
                BoundaryTensorDimension::Fixed(1),
                BoundaryTensorDimension::Fixed(self.dimensions.predictor_width),
            ],
            BoundaryTensorDtype::Activation,
        )
    }
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        vec![
            BoundaryTensorSpec::new(
                "prototype",
                [
                    BoundaryTensorDimension::Fixed(1),
                    BoundaryTensorDimension::Fixed(self.dimensions.predictor_width),
                ],
                BoundaryTensorDtype::Activation,
            ),
            BoundaryTensorSpec::new(
                "request",
                [BoundaryTensorDimension::Fixed(3)],
                BoundaryTensorDtype::Int32,
            ),
        ]
    }
    fn encode<T>(
        &self,
        boundary: Self::Boundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        Ok(vec![
            ArchitectureBoundaryValue::new("prototype", boundary.prototype)?,
            ArchitectureBoundaryValue::new("request", boundary.request)?,
        ])
    }
    fn decode<T>(&self, tensors: Vec<T>) -> Result<Self::Boundary<T>, ArchitectureBoundaryError> {
        validate_boundary_tensor_count(self, &tensors)?;
        let mut values = tensors.into_iter();
        Ok(RnntPredictorBoundary {
            prototype: values.next().unwrap(),
            request: values.next().unwrap(),
        })
    }
}
/// Opaque routes retain their exact globally indexed recurrent cuts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RnntPredictorPipelineCuts {
    schema: RnntPredictorBoundarySchema,
    routes: BTreeMap<CommunicationRouteId, usize>,
}
impl RnntPredictorPipelineCuts {
    /// Validates every retained cut without touching parameter payloads.
    pub fn new(
        dimensions: RnntDimensions,
        routes: impl IntoIterator<Item = (CommunicationRouteId, usize)>,
    ) -> Result<Self, RnntError> {
        let schema = RnntPredictorBoundarySchema::new(dimensions)?;
        let mut selected = BTreeMap::new();
        for (route, cut) in routes {
            if cut == 0 {
                return Err(invalid("RNNT predictor route precedes recurrent execution"));
            }
            schema.at_cut(cut)?;
            if selected.insert(route, cut).is_some() {
                return Err(invalid("duplicate RNNT predictor route"));
            }
        }
        Ok(Self {
            schema,
            routes: selected,
        })
    }
    /// Exact predictor model geometry.
    pub fn dimensions(&self) -> RnntDimensions {
        self.schema.dimensions
    }
    /// Exact route-to-cut declarations for shared cold pipeline validation.
    pub fn routes(&self) -> impl Iterator<Item = (CommunicationRouteId, usize)> + '_ {
        self.routes.iter().map(|(&route, &cut)| (route, cut))
    }
    /// Resolves the typed declaration for one retained route.
    pub fn schema(
        &self,
        route: CommunicationRouteId,
    ) -> Result<RnntPredictorBoundarySchema, RnntError> {
        self.schema.at_cut(
            *self
                .routes
                .get(&route)
                .ok_or_else(|| invalid("RNNT predictor route has no retained cut"))?,
        )
    }
}
/// Validates exact parameters, state slice and input/output stage ownership.
pub fn validate_rnnt_predictor_partition<G>(
    definition: &ComponentDefinition,
    dimensions: RnntDimensions,
    partition: &ArchitecturePartition<G, RnntPredictorBoundarySchema>,
) -> Result<Range<usize>, RnntError> {
    let description = definition.parameters();
    if partition.graph() != description.graph()
        || partition.unit_layout() != description.unit_layout()
        || partition.boundary_schema() != &RnntPredictorBoundarySchema::new(dimensions)?
        || !description.matches_partition_bindings(partition)
        || partition.groups().len() != 1
        || partition.groups()[0].group_index() != 0
        || !partition.ownership().static_roles().is_empty()
        || !partition.ownership().replicated_static_roles().is_empty()
    {
        return Err(invalid(
            "RNNT predictor stage graph, source or boundary ownership differs",
        ));
    }
    let range = partition.groups()[0].global_units();
    if range.is_empty()
        || range.end > dimensions.predictor_layers
        || partition.ownership().owns_input() != (range.start == 0)
        || partition.ownership().owns_output() != (range.end == dimensions.predictor_layers)
    {
        return Err(invalid(
            "RNNT predictor stage is not a contiguous owned layer interval",
        ));
    }
    let state = partition
        .state()
        .ok_or_else(|| invalid("RNNT predictor stage lacks local carry declaration"))?;
    if state.global_layers() != range
        || state.layout() != &definition.state_layout().slice(range.clone())?
    {
        return Err(invalid(
            "RNNT predictor local carry declaration differs from owned layers",
        ));
    }
    Ok(range)
}
impl<B: NeuralBackend> RnntPredictorKernel<B> {
    fn begin_partition_input(
        &self,
        input: LayeredPartitionInput<'_, B::Tensor, RnntPredictorBoundary<B::Tensor>>,
        step: Option<u64>,
        state: &RnntPredictorRuntimeState<B::Tensor>,
        expected: Option<&StateLayout>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, RnntPredictorForward<B::Tensor>>, RnntError> {
        if self.partition.is_none()
            || expected != Some(&rnnt_predictor_state_layout(self.dimensions)?.slice(self.range())?)
        {
            return Err(invalid(
                "RNNT predictor pipeline lacks exact selected carry geometry",
            ));
        }
        match input {
            LayeredPartitionInput::Tokens(input) => {
                if self.range().start != 0 {
                    return Err(invalid(
                        "RNNT predictor embedded ingress requires first stage",
                    ));
                }
                self.begin_input(input, input, step.unwrap_or(0), state)
            }
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                if self.range().start == 0
                    || auxiliary.request.shape() != [3]
                    || auxiliary.request.element_type() != Some(TensorElementType::I32)
                {
                    return Err(invalid(
                        "RNNT predictor boundary metadata or input ownership differs",
                    ));
                }
                let words = auxiliary.request.to_i32_vec(context)?;
                if words.len() != 3 {
                    return Err(invalid("RNNT predictor boundary word count differs"));
                }
                let received = words[1] as u32 as u64 | ((words[2] as u32 as u64) << 32);
                if words[0] != self.range().start as i32
                    || step.is_some_and(|expected| expected != received)
                {
                    return Err(invalid("RNNT predictor boundary cut or exact step differs"));
                }
                self.begin_input(&hidden, &auxiliary.prototype, received, state)
            }
        }
    }
}
impl<B: DistributedNeuralBackend>
    PartitionComponentKernel<
        B,
        crate::voicechat::rnnt_predictor::RnntPredictorRuntimeState<B::Tensor>,
    > for RnntPredictorKernel<B>
{
    type Boundary = RnntPredictorBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, RnntError> {
        RnntPredictorBoundarySchema::new(self.dimensions)
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, RnntPredictorBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        selected: ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError> {
        if mask.is_some() || selected.first_state_ordinal != 0 {
            return Err(invalid(
                "RNNT predictor stage expects local state origin and no mask",
            ));
        }
        self.begin_partition_input(input, None, state, selected.expected, selected.native)
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        forward: &Self::ForwardContext,
        owns_output: bool,
        _: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, RnntPredictorBoundary<B::Tensor>>, RnntError>
    {
        if owns_output != (self.range().end == self.dimensions.predictor_layers) {
            return Err(invalid("RNNT predictor final readout ownership differs"));
        }
        self.complete(hidden, state, forward)?;
        if owns_output {
            Ok(LayeredPartitionOutput::Final {
                output: hidden.clone(),
                retained: None,
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: RnntPredictorBoundary {
                    prototype: forward.prototype.clone(),
                    request: B::Tensor::from_i32_slice(
                        &[
                            self.range().end as i32,
                            forward.step as u32 as i32,
                            (forward.step >> 32) as u32 as i32,
                        ],
                        &[3],
                        context,
                    )?,
                },
            })
        }
    }
}
impl<B: DistributedNeuralBackend>
    PipelineComponentKernel<
        B,
        crate::voicechat::rnnt_predictor::RnntPredictorRuntimeState<B::Tensor>,
    > for RnntPredictorKernel<B>
{
    fn input_dimensions(&self, _: &Self::Input<'_>) -> Result<(i32, i32), RnntError> {
        Ok((1, 1))
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, RnntPredictorBoundary<B::Tensor>)>,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        selected: ComponentPartitionContext<'_, B>,
        _: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError>
    where
        O: ActivationObserver<B::Tensor, RnntError> + ?Sized,
    {
        if selected.first_state_ordinal != 0 {
            return Err(invalid("RNNT predictor stage requires local carry origin"));
        }
        validate_row(input.input, self.dimensions)?;
        let ingress = match incoming {
            Some((hidden, auxiliary)) => LayeredPartitionInput::Hidden { hidden, auxiliary },
            None => LayeredPartitionInput::Tokens(input.input),
        };
        self.begin_partition_input(
            ingress,
            Some(input.step),
            state,
            selected.expected,
            selected.native,
        )
    }
    fn output_width(&self) -> i32 {
        self.dimensions.predictor_width
    }
    fn output_shape(&self, _: &Self::Input<'_>, _: i32, _: i32) -> Result<Vec<i32>, RnntError> {
        Ok(vec![1, self.dimensions.predictor_width])
    }
    fn route_schema(&self, route: &PartitionBoundaryRoute) -> Result<Self::Boundary, RnntError> {
        if route.source_group != 0 || route.destination_group != 0 {
            return Err(invalid("RNNT predictor route addresses another group"));
        }
        self.partition
            .as_ref()
            .ok_or_else(|| invalid("RNNT predictor route requires stage selection"))?
            .1
            .schema(route.route)
    }

    fn tensor_vocabulary_sharded(&self) -> bool {
        false
    }
}
