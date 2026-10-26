//! RNNT recurrent layers over the shared stateful pipeline traversal.
use super::{
    rnnt::{
        build_predictor_layer, step_predictor_layer, RnntDimensions, RnntError, RnntPredictionState,
    },
    rnnt_layered::rnnt_parameter_description,
    rnnt_predictor_partition::{
        validate_rnnt_predictor_partition, RnntPredictorBoundarySchema, RnntPredictorPipelineCuts,
    },
};
use eredu_core::{
    cache::{
        LayerCachePolicy, StateResidencyClass, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorPresence, StateTensorRole,
    },
    LayerSchedule,
};
use eredu_nn::{
    recurrent::{Lstm, LstmState},
    DistributedNeuralBackend, NeuralBackend, Parameterized, RetainedTensorRoots, Tensor,
    TensorElementType,
};
use eredu_runtime::*;
use std::{marker::PhantomData, num::NonZeroU32, ops::Range};

pub(crate) const GROUP: &str = "voicechat.rnnt.predictor";
pub(crate) fn invalid(message: &'static str) -> RnntError {
    RnntError::InvalidState(message)
}
fn graph() -> Result<ExecutionGraph, RnntError> {
    Ok(ExecutionGraph::new(
        vec![ExecutionGroupSpec::root(GROUP)],
        GROUP,
    )?)
}
/// Exact recurrent parameter ownership, with no embedding or joint parameters.
pub fn rnnt_predictor_parameter_description(
    dimensions: RnntDimensions,
) -> Result<ArchitectureParameterDescription, RnntError> {
    dimensions.validate()?;
    let graph = graph()?;
    let layout = ExecutionUnitLayout::new(&graph, vec![dimensions.predictor_layers])?;
    let full = rnnt_parameter_description(dimensions)?;
    let mut owned = Vec::new();
    for item in full.groups() {
        let ParameterGroupOwner::ExecutionUnit {
            group, global_unit, ..
        } = item.owner()
        else {
            continue;
        };
        if group.as_str() != GROUP {
            continue;
        }
        let spec = item.group().clone();
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(layout.group_id(0).unwrap().clone(), *global_unit),
            spec,
        ));
    }
    Ok(ArchitectureParameterDescription::from_owned(&graph, &layout, owned)?)
}
/// Global recurrent operations, independently leased layers and carry identities.
pub fn rnnt_predictor_component_definition(
    dimensions: RnntDimensions,
) -> Result<ComponentDefinition, RnntError> {
    Ok(ComponentDefinition::new(
        "voicechat.rnnt.predictor",
        rnnt_predictor_parameter_description(dimensions)?,
        (0..dimensions.predictor_layers)
            .map(|index| ComponentUnit {
                path: format!("stt_model.rnnt_decoder.prediction.dec_rnn.lstm.layer{index}"),
                state: index..index + 1,
            })
            .collect(),
        vec![ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Recurrent,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        GROUP,
        vec![],
        rnnt_predictor_state_layout(dimensions)?,
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..dimensions.predictor_layers,
        )]),
    )?)
}
/// Two optional F32 vectors per globally indexed recurrent layer.
pub fn rnnt_predictor_state_layout(dimensions: RnntDimensions) -> Result<StateLayout, RnntError> {
    dimensions.validate()?;
    let policy = LayerCachePolicy::FixedState {
        tensors: (0..2)
            .map(|slot| StateTensorPolicy {
                role: StateTensorRole::Auxiliary { slot },
                shape: vec![
                    StateTensorDimension::Fixed(NonZeroU32::new(1).unwrap()),
                    StateTensorDimension::Fixed(
                        NonZeroU32::new(dimensions.predictor_width as u32).unwrap(),
                    ),
                ],
                dtype: StateTensorDtype::Float32,
                residency: StateResidencyClass::AlwaysDeviceMutable,
                presence: StateTensorPresence::Optional,
            })
            .collect(),
    };
    Ok(StateLayout::new(LayerSchedule::new(
        dimensions.predictor_layers,
        vec![policy; dimensions.predictor_layers],
    )?)?)
}
/// Logical F32 tensor payload of a completed local prediction. Hidden/cell carry
/// is phase-owned; non-final stages additionally retain the published global readout.
pub fn rnnt_predictor_prediction_state_bytes(
    dimensions: RnntDimensions,
    range: Range<usize>,
) -> Result<u64, RnntError> {
    dimensions.validate()?;
    if range.is_empty() || range.end > dimensions.predictor_layers {
        return Err(invalid("invalid RNNT predictor state accounting range"));
    }
    let vectors = range
        .len()
        .checked_mul(2)
        .and_then(|count| count.checked_add(usize::from(range.end != dimensions.predictor_layers)))
        .ok_or_else(|| invalid("RNNT prediction state byte count overflows"))?;
    (vectors as u64)
        .checked_mul(dimensions.predictor_width as u64)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| invalid("RNNT prediction state byte count overflows"))
}
/// Exact one-step input and caller-recorded predictor invocation identity.
#[derive(Clone, Copy)]
pub struct RnntPredictorInput<'a, T> {
    /// Embedded or zero priming row `[1,predictor_width]`, in F32.
    pub input: &'a T,
    /// Full-width explicit invocation identity; this does not advance audio time.
    pub step: u64,
}
/// One canonical LSTM layer with immutable global ownership.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RnntPredictorUnit<B: NeuralBackend> {
    #[parameter(skip)]
    dimensions: RnntDimensions,
    #[parameter(skip)]
    index: usize,
    layer: Lstm<B>,
}
#[derive(Clone)]
struct PredictorParent<T> {
    dimensions: RnntDimensions,
    range: Range<usize>,
    local: LstmState<T>,
    readout_shape: Vec<i32>,
    readout_dtype: Option<TensorElementType>,
}
/// Caller-owned local carry branch, separate from learned parameter residency.
/// The outer prediction state owns its published readout; the recurrent phase
/// retains only the local hidden/cell pairs described by its state layout.
#[derive(Clone)]
pub struct RnntPredictorRuntimeState<T: Tensor> {
    dimensions: RnntDimensions,
    range: Range<usize>,
    layout: StateLayout,
    parent: Option<PredictorParent<T>>,
    next: Option<LstmState<T>>,
}
impl<T: Tensor> RnntPredictorRuntimeState<T> {
    /// Retains a parent for validation inside the shared invocation failure phase.
    /// No parameter or tensor is constructed and no rank-local parent is advanced.
    pub fn for_partition<B: NeuralBackend<Tensor = T>>(
        architecture: &RnntPredictorKernel<B>,
        parent: Option<&RnntPredictionState<T>>,
    ) -> Result<Self, RnntError> {
        let range = architecture.range();
        Ok(Self {
            dimensions: architecture.dimensions,
            layout: rnnt_predictor_state_layout(architecture.dimensions)?.slice(range.clone())?,
            range,
            parent: parent.map(|parent| PredictorParent {
                dimensions: parent.dimensions(),
                range: parent.global_layers(),
                local: parent.local_state().clone(),
                readout_shape: parent.output().shape().to_vec(),
                readout_dtype: parent.output().element_type(),
            }),
            next: None,
        })
    }
    /// Publishes the completed local carry with the authoritative final readout.
    pub fn into_prediction(
        self,
        published_readout: T,
    ) -> Result<RnntPredictionState<T>, RnntError> {
        let mut local = self
            .next
            .ok_or_else(|| invalid("RNNT predictor stage has no completed step"))?;
        validate_row(&published_readout, self.dimensions)?;
        let readout = if self.range.end == self.dimensions.predictor_layers {
            local
                .layers
                .last_mut()
                .ok_or_else(|| invalid("RNNT predictor output owner has no carry"))?
                .hidden = published_readout;
            None
        } else {
            Some(published_readout)
        };
        RnntPredictionState::new(self.dimensions, self.range, local, readout)
    }
    fn validate<B: NeuralBackend<Tensor = T>>(
        &self,
        architecture: &RnntPredictorKernel<B>,
    ) -> Result<(), RnntError> {
        if self.dimensions != architecture.dimensions
            || self.range != architecture.range()
            || self.layout
                != rnnt_predictor_state_layout(self.dimensions)?.slice(self.range.clone())?
            || self.next.is_some()
        {
            return Err(invalid(
                "RNNT predictor branch geometry, ownership or lifecycle differs",
            ));
        }
        if let Some(parent) = &self.parent {
            if parent.dimensions != self.dimensions
                || parent.range != self.range
                || parent.local.layers.len() != self.range.len()
                || parent.readout_shape != [1, self.dimensions.predictor_width]
                || parent.readout_dtype != Some(TensorElementType::F32)
            {
                return Err(invalid(
                    "RNNT predictor parent identity, readout or precision differs",
                ));
            }
            for layer in &parent.local.layers {
                validate_row(&layer.hidden, self.dimensions)?;
                validate_row(&layer.cell, self.dimensions)?;
            }
        }
        Ok(())
    }
}
impl<T: Tensor> RetainedTensorRoots<T> for RnntPredictorRuntimeState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        if let Some(parent) = &self.parent {
            for layer in &parent.local.layers {
                visitor(&layer.hidden);
                visitor(&layer.cell);
            }
        }
        if let Some(next) = &self.next {
            for layer in &next.layers {
                visitor(&layer.hidden);
                visitor(&layer.cell);
            }
        }
    }
}
impl<B: NeuralBackend> RuntimeState<B> for RnntPredictorRuntimeState<B::Tensor> {
    type RetainedValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    fn layout(&self) -> &StateLayout {
        &self.layout
    }
    fn retained_values(
        &self,
        ordinal: usize,
        _: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        if ordinal >= self.range.len() {
            return Err(StateError::UnknownLayer {
                layer: ordinal,
                count: self.range.len(),
            });
        }
        let mut values = Vec::new();
        if let Some(parent) = &self.parent {
            if let Some(layer) = parent.local.layers.get(ordinal) {
                values.extend([&layer.hidden, &layer.cell]);
            }
        }
        if let Some(next) = &self.next {
            if let Some(layer) = next.layers.get(ordinal) {
                values.extend([&layer.hidden, &layer.cell]);
            }
        }
        Ok(values.into_iter())
    }
}
/// Unpublished recurrent candidate retained across independently leased layers.
pub struct RnntPredictorForward<T> {
    pub(crate) prototype: T,
    pub(crate) step: u64,
    pub(crate) next_unit: usize,
    candidate: LstmState<T>,
}
/// Model-family equations and exact state ownership for a predictor stage.
pub struct RnntPredictorKernel<B: NeuralBackend> {
    pub(crate) dimensions: RnntDimensions,
    pub(crate) partition: Option<(Range<usize>, RnntPredictorPipelineCuts)>,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> RnntPredictorKernel<B> {
    /// Declares a complete predictor without native allocations.
    pub fn new(dimensions: RnntDimensions) -> Result<Self, RnntError> {
        dimensions.validate()?;
        Ok(Self {
            dimensions,
            partition: None,
            backend: PhantomData,
        })
    }
    /// Couples the recurrent equations to their validated global declaration.
    pub fn into_component(self) -> Result<ComponentAdapter<ComponentDefinition, Self>, RnntError> {
        Ok(ComponentAdapter::new(
            rnnt_predictor_component_definition(self.dimensions)?,
            self,
        )?)
    }
    /// Retains exactly one contiguous predictor stage.
    pub fn new_partition<G>(
        definition: ComponentDefinition,
        dimensions: RnntDimensions,
        partition: &ArchitecturePartition<G, RnntPredictorBoundarySchema>,
        cuts: &RnntPredictorPipelineCuts,
    ) -> Result<ComponentAdapter<ComponentDefinition, Self>, RnntError> {
        let range = validate_rnnt_predictor_partition(&definition, dimensions, partition)?;
        if cuts.dimensions() != dimensions {
            return Err(invalid("RNNT predictor cuts belong to another geometry"));
        }
        let mut value = Self::new(dimensions)?;
        value.partition = Some((range, cuts.clone()));
        Ok(ComponentAdapter::new(definition, value)?.select_partition(partition)?)
    }
    /// Logical F32 tensor payload of the completed prediction: local hidden/cell
    /// pairs plus a separately retained readout on every non-final stage.
    /// The phase state layout describes only hidden/cell carry; the returned
    /// prediction owns the readout after shared output publication.
    pub fn prediction_state_bytes(&self) -> Result<u64, RnntError> {
        rnnt_predictor_prediction_state_bytes(self.dimensions, self.range())
    }
    /// Exact whole predictor geometry.
    pub fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    /// Exact recurrent layers whose hidden and cell state this stage owns.
    pub fn predictor_ownership(&self) -> Range<usize> {
        self.range()
    }
    pub(crate) fn range(&self) -> Range<usize> {
        self.partition
            .as_ref()
            .map(|p| p.0.clone())
            .unwrap_or(0..self.dimensions.predictor_layers)
    }

    pub(crate) fn begin_input(
        &self,
        input: &B::Tensor,
        prototype: &B::Tensor,
        step: u64,
        state: &RnntPredictorRuntimeState<B::Tensor>,
    ) -> Result<LayeredForwardState<B::Tensor, RnntPredictorForward<B::Tensor>>, RnntError> {
        state.validate(self)?;
        validate_row(input, self.dimensions)?;
        validate_row(prototype, self.dimensions)?;
        Ok(LayeredForwardState {
            hidden: input.clone(),
            context: RnntPredictorForward {
                prototype: prototype.clone(),
                step,
                next_unit: self.range().start,
                candidate: LstmState {
                    layers: Vec::with_capacity(self.range().len()),
                },
            },
        })
    }
    pub(crate) fn complete(
        &self,
        hidden: &B::Tensor,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        forward: &RnntPredictorForward<B::Tensor>,
    ) -> Result<(), RnntError> {
        if forward.next_unit != self.range().end
            || forward.candidate.layers.len() != self.range().len()
        {
            return Err(invalid(
                "RNNT predictor stage did not complete its owned layers",
            ));
        }
        validate_row(hidden, self.dimensions)?;
        state.next = Some(forward.candidate.clone());
        Ok(())
    }
}
pub(crate) fn validate_row<T: Tensor>(
    input: &T,
    dimensions: RnntDimensions,
) -> Result<(), RnntError> {
    if input.shape() != [1, dimensions.predictor_width]
        || input.element_type() != Some(TensorElementType::F32)
    {
        return Err(invalid("RNNT predictor expects one F32 embedded row"));
    }
    Ok(())
}

impl<B: NeuralBackend> ComponentKernel<B> for RnntPredictorKernel<B> {
    type Input<'a> = RnntPredictorInput<'a, B::Tensor>;
    type Unit = RnntPredictorUnit<B>;
    type ForwardContext = RnntPredictorForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = RnntError;
    fn state_identity(
        &self,
        state: &PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<ModelStateIdentity, RnntError> {
        let range = state.global_layers();
        if range.is_empty()
            || range.end > self.dimensions.predictor_layers
            || state.layout()
                != &rnnt_predictor_state_layout(self.dimensions)?.slice(range.clone())?
            || self.partition.as_ref().is_some_and(|p| p.0 != range)
        {
            return Err(invalid("RNNT predictor state identity differs"));
        }
        Ok(ModelStateIdentity::new(
            "nemotron_voicechat",
            "rnnt_predictor",
            format!("{:?}", self.dimensions),
            self.dimensions.predictor_layers,
            range.start,
            0,
            topology,
        )?)
    }
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, RnntError> {
        let index = address.index();
        if !self.range().contains(&index) {
            return Err(invalid("RNNT predictor unit exceeds selected stage"));
        }
        Ok(RnntPredictorUnit {
            dimensions: self.dimensions,
            index,
            layer: build_predictor_layer::<B>(self.dimensions, index, context)?,
        })
    }

    fn executes_group(&self, _: usize, _: &Self::ForwardContext) -> bool {
        true
    }

    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        let mut roots = vec![&forward.prototype];
        for layer in &forward.candidate.layers {
            roots.extend([&layer.hidden, &layer.cell]);
        }
        roots.into_iter()
    }
}
impl<B: NeuralBackend> ComponentEquations<B, RnntPredictorRuntimeState<B::Tensor>>
    for RnntPredictorKernel<B>
{
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError> {
        if self.partition.is_some() {
            return Err(invalid("RNNT predictor stage requires pipeline ingress"));
        }
        self.begin_input(input.input, input.input, input.step, state)
    }
    fn begin_group(
        &mut self,
        _: usize,
        initial: &B::Tensor,
        _: &[&B::Tensor],
        _: &mut RnntPredictorRuntimeState<B::Tensor>,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        let index = address.index();
        if index != forward.next_unit
            || unit.index != index
            || unit.dimensions != self.dimensions
            || !self.range().contains(&index)
        {
            return Err(invalid(
                "RNNT predictor leased unit differs from stage order",
            ));
        }
        let local = index - self.range().start;
        let previous = state.parent.as_ref().map(|p| &p.local.layers[local]);
        let next = step_predictor_layer::<B>(
            &mut unit.layer,
            hidden,
            previous,
            &forward.prototype,
            context,
        )?;
        let output = next.hidden.clone();
        forward.candidate.layers.push(next);
        forward.next_unit += 1;
        Ok(output)
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        forward: &Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.complete(hidden, state, forward)?;
        Ok(hidden.clone())
    }
}

impl<B: DistributedNeuralBackend>
    ParallelComponentKernel<
        B,
        crate::voicechat::rnnt_predictor::RnntPredictorRuntimeState<B::Tensor>,
    > for RnntPredictorKernel<B>
{
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError> {
        ComponentEquations::begin(self, input, state, context)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        forward: &mut Self::ForwardContext,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.execute(address, unit, hidden, state, forward, context)
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut RnntPredictorRuntimeState<B::Tensor>,
        forward: &Self::ForwardContext,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.finish(hidden, state, forward, context)
    }
}
