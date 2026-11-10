//! Perception equations over the shared resident and bounded unit traversal.

use super::perception_specs::{Declaration, PerceptionDeclaration};
use super::{
    conformer::{ConformerBlock, ConformerLocalDimensions, ConformerState},
    perception::{PerceptionDimensions, PerceptionError, PerceptionOutput, PerceptionState},
    subsampling::CausalSubsampling,
};
use eredu_core::{
    cache::{
        LayerCachePolicy, StateResidencyClass, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorPresence, StateTensorRole,
    },
    BackendFailure, LayerSchedule,
};
use eredu_nn::{
    DeclareLogicalParameters, Index, LinearOperator, NeuralBackend, Parameterized, Tensor,
};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ComponentAdapter, ComponentDefinition, ComponentEquations,
    ComponentKernel, ComponentUnit, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddress,
    ExecutionUnitLayout, LayeredForwardState, LayeredInvocation, LayerwisePolicy, LayerwiseRuntime,
    LocalModelLayout, MemberSharding, OwnedParameterGroupSpec, ParallelComponentKernel,
    ParameterGroupOwner, ParameterRole, RuntimeState, StateError, StateLayout, SubmissionBackend,
};
use std::{marker::PhantomData, num::NonZeroU32};

#[path = "perception_partition.rs"]
mod partition;
pub use partition::{
    unpack_perception_pipeline_output, validate_perception_partition, PerceptionBoundary,
    PerceptionBoundarySchema,
};

pub(super) const SUBSAMPLING: &str = "voicechat.perception.subsampling";
pub(super) const ENCODER: &str = "voicechat.perception.encoder";
pub(super) const PROJECTION: &str = "voicechat.perception.projection";

pub(super) fn backend(error: impl std::error::Error + Send + Sync + 'static) -> PerceptionError {
    PerceptionError::Backend(BackendFailure::from_error(error))
}

/// One bounded learned unit; subsampling and projection are independently resident.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum PerceptionUnit<B: NeuralBackend> {
    /// Three causal striding convolutions and their projection.
    Subsampling(CausalSubsampling<B>),
    /// One causal Conformer block.
    Block(ConformerBlock<B>),
    /// Perception-to-language projection.
    Projection(B::Linear),
}

/// Perception invocation available through resident or bounded execution.
pub trait PerceptionExecution<B: NeuralBackend> {
    /// Exact connected component geometry.
    fn dimensions(&self) -> PerceptionDimensions;
    /// Produces independent candidate caches without changing the borrowed parent.
    fn forward(
        &mut self,
        mel: &B::Tensor,
        drop_preencoded: usize,
        previous: &PerceptionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionOutput<B::Tensor>, PerceptionError>;
}

/// Architecture-owned graph and cold unit construction, with no pinned learned modules.
pub struct PerceptionKernel<B: NeuralBackend> {
    precision: super::precision::SpeechPrecision,
    dimensions: PerceptionDimensions,
    declaration: Declaration,
    definition: ComponentDefinition,
    projection_precision: super::precision::SpeechPrecisionDeclaration,
    local_blocks: Option<Vec<ConformerLocalDimensions>>,
    partition: Option<partition::PerceptionOwnership>,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> PerceptionKernel<B> {
    /// Validates component geometry before constructing any parameters.
    pub fn new(declaration: PerceptionDeclaration) -> Result<Self, PerceptionError> {
        let (dimensions, declaration, definition, precision) = declaration.into_parts();
        let projection_precision = super::precision::SpeechPrecisionDeclaration::from_spec(
            &declaration.pinned.projection,
        )?;
        Ok(Self {
            precision,
            dimensions,
            declaration,
            definition,
            projection_precision,
            local_blocks: None,
            partition: None,
            backend: PhantomData,
        })
    }
    /// Constructs the selected rank's projections and complete replicated caches.
    pub fn new_parallel(
        declaration: PerceptionDeclaration,
        layout: &LocalModelLayout,
    ) -> Result<Self, PerceptionError> {
        let local_blocks = declaration.local_dimensions(layout)?;
        let mut architecture = Self::new(declaration)?;
        architecture.local_blocks = Some(local_blocks);
        Ok(architecture)
    }
    /// Retains the validated declaration alongside the family equations.
    pub fn into_component(
        self,
    ) -> Result<ComponentAdapter<ComponentDefinition, Self>, PerceptionError> {
        Ok(ComponentAdapter::new(self.definition.clone(), self)?)
    }
    /// Whether execution requires the selected tensor collective.
    pub fn is_parallel(&self) -> bool {
        self.local_blocks.is_some()
    }
    /// Connected perception geometry, independent of residency.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
    }
}

pub(super) fn declared_definition(
    dimensions: PerceptionDimensions,
    declaration: &Declaration,
) -> Result<ComponentDefinition, PerceptionError> {
    let parameters = declared_parameters(declaration)?;
    let units = (0..parameters.unit_layout().len())
        .map(|ordinal| {
            let address = parameters
                .unit_layout()
                .address(ordinal)
                .expect("canonical unit");
            let (path, state) = match address.group() {
                0 => ("stt_model.perception.encoder.pre_encode".into(), 0..0),
                1 => (
                    format!("stt_model.perception.encoder.layers.{}", address.index()),
                    address.index()..address.index() + 1,
                ),
                _ => ("stt_model.perception.proj".into(), 0..0),
            };
            ComponentUnit { path, state }
        })
        .collect();
    let transport = (0..3)
        .map(|group| ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: if group == 2 {
                ArchitectureGroupKind::Projector
            } else {
                ArchitectureGroupKind::AudioEncoder
            },
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        })
        .collect();
    Ok(ComponentDefinition::new(
        "voicechat.perception",
        parameters,
        units,
        transport,
        ENCODER,
        vec![],
        perception_state_layout(dimensions)?,
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            1,
            0..dimensions.layers,
        )]),
    )?)
}

/// Unpublished runtime caches and their exact architecture layout.
#[derive(Clone)]
pub struct PerceptionRuntimeState<T> {
    layout: StateLayout,
    value: PerceptionState<T>,
    partition_blocks: Option<std::ops::Range<usize>>,
    partition_roles: Option<(bool, bool)>,
}
impl<T: Tensor> PerceptionRuntimeState<T> {
    /// Borrows immutable tensor values into a private candidate owner.
    pub fn from_state(
        dimensions: PerceptionDimensions,
        previous: &PerceptionState<T>,
    ) -> Result<Self, PerceptionError> {
        if previous.dimensions.is_some_and(|d| d != dimensions)
            || (previous.dimensions.is_some() && previous.blocks.len() != dimensions.layers)
            || (previous.dimensions.is_none()
                && (!previous.blocks.is_empty() || previous.frames != 0))
        {
            return Err(PerceptionError::State(
                "perception state belongs to different geometry",
            ));
        }
        let mut value = previous.clone();
        if value.blocks.is_empty() {
            value
                .blocks
                .resize_with(dimensions.layers, ConformerState::default);
        }
        Ok(Self {
            layout: perception_state_layout(dimensions)?,
            value,
            partition_blocks: None,
            partition_roles: None,
        })
    }
    /// Consumes a successfully completed candidate.
    pub fn into_state(self) -> PerceptionState<T> {
        self.value
    }
}
impl<T: Tensor> super::perception_rnnt::PerceptionCacheState for PerceptionRuntimeState<T> {
    fn frames(&self) -> u64 {
        self.value.frames()
    }
}
impl<T> eredu_nn::RetainedTensorRoots<T> for PerceptionRuntimeState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        eredu_nn::RetainedTensorRoots::visit_retained_tensors(&self.value, visitor);
    }
}
impl<T: eredu_runtime::execution_control::SnapshotValue>
    eredu_runtime::execution_control::SnapshotValue for PerceptionRuntimeState<T>
{
    fn estimate_snapshot_value(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, BackendFailure> {
        use eredu_core::execution_control::{ExecutionControlError, SnapshotEstimate};
        let bytes = self
            .layout
            .logical_metadata_bytes()
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        eredu_runtime::execution_control::snapshot_value_estimate::<Self>([
            self.value.estimate_snapshot_value(),
            Ok(SnapshotEstimate {
                retained_bytes: bytes,
                copy_bytes: bytes,
            }),
        ])
    }
    fn copy_snapshot_value(
        &self,
        retention: &eredu_runtime::execution_control::SnapshotReservation,
    ) -> Result<Self, BackendFailure> {
        Ok(Self {
            layout: self.layout.clone(),
            value: self.value.copy_snapshot_value(retention)?,
            partition_blocks: self.partition_blocks.clone(),
            partition_roles: self.partition_roles,
        })
    }
}
impl<B: NeuralBackend> RuntimeState<B> for PerceptionRuntimeState<B::Tensor> {
    type RetainedValues<'a>
        =
        std::iter::Chain<std::option::IntoIter<&'a B::Tensor>, std::option::IntoIter<&'a B::Tensor>>
    where
        Self: 'a,
        B::Tensor: 'a;
    fn layout(&self) -> &StateLayout {
        &self.layout
    }
    fn optional_layout(&self) -> Option<&StateLayout> {
        if self
            .partition_blocks
            .as_ref()
            .is_some_and(|blocks| blocks.is_empty())
        {
            None
        } else {
            Some(&self.layout)
        }
    }
    fn retained_values(
        &self,
        ordinal: usize,
        _: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        // The shared traversal obtains ordinals only from this architecture's
        // Conformer range; parameter-only groups declare an empty range.
        let block = self
            .value
            .blocks
            .get(ordinal)
            .ok_or(StateError::UnknownLayer {
                layer: ordinal,
                count: self.value.blocks.len(),
            })?;
        Ok(block.channel().into_iter().chain(block.convolution()))
    }
}

/// Complete Conformer cache geometry; every tensor replica retains full-width state.
pub fn perception_state_layout(d: PerceptionDimensions) -> Result<StateLayout, PerceptionError> {
    d.validate()?;
    d.block.validate()?;
    d.subsampling.validate()?;
    let fixed = |n: i32| {
        NonZeroU32::new(n as u32)
            .map(StateTensorDimension::Fixed)
            .ok_or(PerceptionError::Configuration(
                "zero perception cache dimension",
            ))
    };
    let tensor = |role, length| {
        Ok::<_, PerceptionError>(StateTensorPolicy {
            role,
            shape: vec![
                StateTensorDimension::Batch,
                fixed(length)?,
                fixed(d.block.width)?,
            ],
            dtype: StateTensorDtype::Float32,
            residency: StateResidencyClass::AlwaysDeviceMutable,
            presence: StateTensorPresence::Optional,
        })
    };
    let layer = LayerCachePolicy::FixedState {
        tensors: vec![
            tensor(StateTensorRole::Auxiliary { slot: 0 }, d.block.left_context)?,
            tensor(
                StateTensorRole::Convolution { slot: 0 },
                d.block.conv_kernel - 1,
            )?,
        ],
    };
    Ok(StateLayout::new(LayerSchedule::new(
        d.layers,
        vec![layer; d.layers],
    )?)?)
}

/// Invocation-local mel and subsampling overlap policy.
pub struct PerceptionInput<'a, T> {
    /// Full frame-major mel tensor.
    pub mel: &'a T,
    /// Subsampled overlap discarded before every Conformer block.
    pub drop_preencoded: usize,
}
/// Transient raw encoder capture and candidate frame frontier.
pub struct PerceptionForward<T> {
    encoder: Option<T>,
    drop_preencoded: usize,
    next_frames: Option<u64>,
    initial_frames: u64,
    projected: bool,
}

fn declared_parameters(
    declaration: &Declaration,
) -> Result<ArchitectureParameterDescription, PerceptionError> {
    use eredu_runtime::ParameterGroupSpec;
    let graph = perception_graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [1, declaration.blocks.len(), 1])?;
    let mut owned = Vec::new();
    let mut add =
        |group: usize, index: usize, spec: ParameterGroupSpec| -> Result<(), PerceptionError> {
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(
                    layout.group_id(group).expect("declared group").clone(),
                    index,
                ),
                spec,
            ));
            Ok(())
        };
    add(
        0,
        0,
        eredu_runtime::logical_parameter_group(
            super::perception_specs::SUBSAMPLING_ROOT,
            ParameterRole::Replicated,
            None,
            &declaration.pinned.subsampling,
            |_| MemberSharding::Replicated,
        )?,
    )?;
    for (index, spec) in declaration.blocks.iter().enumerate() {
        for ff in &spec.feed_forward {
            let projections = ff.network.logical_parameters()?;
            add(
                1,
                index,
                eredu_runtime::logical_parameter_group(
                    format!("{}.projections", ff.root),
                    ParameterRole::FeedForwardIntermediate,
                    Some(ff.network.up.output as usize),
                    &projections,
                    |parameter| {
                        let down = parameter.parameter.id == ff.network.down.weight.id
                            || ff
                                .network
                                .down
                                .bias
                                .as_ref()
                                .is_some_and(|bias| bias.id == parameter.parameter.id);
                        MemberSharding::Partitioned {
                            axis: usize::from(down),
                        }
                    },
                )?,
            )?;
        }
        let attention = spec.attention.logical_parameters()?;
        add(
            1,
            index,
            eredu_runtime::logical_parameter_group(
                format!("{}.projections", spec.attention.root),
                ParameterRole::AttentionHeads,
                Some(spec.attention.projection.geometry.query_heads() as usize),
                &attention,
                |parameter| {
                    let output = parameter.parameter.id
                        == spec.attention.projection.output.weight.id
                        || spec
                            .attention
                            .projection
                            .output
                            .bias
                            .as_ref()
                            .is_some_and(|bias| bias.id == parameter.parameter.id);
                    MemberSharding::Partitioned {
                        axis: usize::from(output),
                    }
                },
            )?,
        )?;
        let mut replicated = spec.norms[..].logical_parameters()?;
        replicated.extend(spec.pointwise_input.logical_parameters()?);
        replicated.extend(spec.depthwise.logical_parameters()?);
        replicated.extend(spec.pointwise_output.logical_parameters()?);
        add(
            1,
            index,
            eredu_runtime::logical_parameter_group(
                format!("{}.replicated", spec.root),
                ParameterRole::Replicated,
                None,
                &replicated,
                |_| MemberSharding::Replicated,
            )?,
        )?;
    }
    add(
        2,
        0,
        eredu_runtime::logical_parameter_group(
            "stt_model.perception.proj",
            ParameterRole::Replicated,
            None,
            &declaration.pinned.projection,
            |_| MemberSharding::Replicated,
        )?,
    )?;
    Ok(ArchitectureParameterDescription::from_owned(
        &graph, &layout, owned,
    )?)
}

/// Resolves physical geometry using the shared symbolic placement compiler.
pub(super) fn declared_local_dimensions(
    dimensions: PerceptionDimensions,
    layout: &LocalModelLayout,
    declaration: &Declaration,
) -> Result<Vec<ConformerLocalDimensions>, PerceptionError> {
    let description = declared_parameters(declaration)?;
    let query = layout
        .tensor(
            declaration.blocks[0]
                .attention
                .projection
                .query
                .weight
                .id
                .as_str(),
        )
        .ok_or(PerceptionError::Configuration(
            "perception layout lacks its attention partition",
        ))?;
    let heads = declaration.blocks[0]
        .attention
        .projection
        .geometry
        .query_heads() as usize;
    let range = query.expanded_logical_range(heads)?;
    let local_heads = range.len();
    if local_heads == 0
        || !heads.is_multiple_of(local_heads)
        || range.start % local_heads != 0
        || range.end > heads
    {
        return Err(PerceptionError::Configuration(
            "perception requires equal complete attention head shards",
        ));
    }
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(heads / local_heads, 1, 1, 1)
            .map_err(|_| PerceptionError::Configuration("invalid perception tensor degree"))?,
        range.start / local_heads,
    )
    .map_err(|_| PerceptionError::Configuration("invalid perception tensor rank"))?;
    let expected =
        eredu_runtime::derive_partitioned_local_layout(&description, topology).map_err(|_| {
            PerceptionError::Configuration("perception tensor placement cannot be derived")
        })?;
    if layout != &expected {
        return Err(PerceptionError::Configuration(
            "perception physical layout differs from its declared rank",
        ));
    }
    let local = ConformerLocalDimensions {
        heads: local_heads as i32,
        ff_hidden: declaration.blocks[0].feed_forward[0].network.up.output
            / (heads / local_heads) as i32,
    };
    local.validate(dimensions.block)?;
    Ok(vec![local; declaration.blocks.len()])
}

fn perception_graph() -> Result<ExecutionGraph, PerceptionError> {
    ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(SUBSAMPLING),
            ExecutionGroupSpec::with_dependencies(ENCODER, [SUBSAMPLING]),
            ExecutionGroupSpec::with_dependencies(PROJECTION, [ENCODER]),
        ],
        PROJECTION,
    )
    .map_err(PerceptionError::from)
}

impl<B: NeuralBackend> ComponentKernel<B> for PerceptionKernel<B> {
    type Input<'a> = PerceptionInput<'a, B::Tensor>;
    type Unit = PerceptionUnit<B>;
    type ForwardContext = PerceptionForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = PerceptionError;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, PerceptionError> {
        let global = perception_state_layout(self.dimensions)?;
        let begin = state.global_layer_offset();
        let end = begin
            .checked_add(state.layout().len())
            .filter(|&end| end <= global.len())
            .ok_or(PerceptionError::State(
                "perception state partition exceeds declared layers",
            ))?;
        if global.slice(begin..end)? != *state.layout() {
            return Err(PerceptionError::State(
                "perception partition state differs from declared layout",
            ));
        }
        eredu_runtime::ModelStateIdentity::new(
            "nemotron_voicechat",
            "voicechat_perception",
            format!("{:?}:{:?}", self.dimensions, self.precision),
            global.len(),
            begin,
            0,
            topology,
        )
        .map_err(PerceptionError::from)
    }
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionUnit<B>, PerceptionError> {
        let group = address.group();
        let index = address.index();
        Ok(match group {
            0 => PerceptionUnit::Subsampling(
                CausalSubsampling::new(self.declaration.pinned.subsampling.clone(), context)?
                    .with_precision(self.precision)?,
            ),
            1 => {
                let mut spec = self
                    .declaration
                    .blocks
                    .get(index)
                    .ok_or(PerceptionError::Configuration(
                        "Conformer unit exceeds declared stack",
                    ))?
                    .clone();
                if let Some(blocks) = &self.local_blocks {
                    spec = spec.local(blocks[index])?;
                }
                PerceptionUnit::Block(
                    ConformerBlock::new(spec, context)?.with_precision(self.precision)?,
                )
            }
            _ => {
                let mut projection =
                    B::linear(self.declaration.pinned.projection.clone(), context)?;
                projection.set_compute_element(self.precision.projection_element())?;
                PerceptionUnit::Projection(projection)
            }
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
        forward.encoder.iter()
    }
}
impl<B: NeuralBackend> ComponentEquations<B, PerceptionRuntimeState<B::Tensor>>
    for PerceptionKernel<B>
{
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, PerceptionError> {
        if self.is_parallel() || self.partition.is_some() {
            return Err(PerceptionError::Configuration(
                "rank-local perception requires parallel execution",
            ));
        }
        self.begin_input(input, state, context)
    }
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        let hidden = dependencies.first().copied().unwrap_or(initial);
        if group == 2 {
            forward.encoder = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut PerceptionUnit<B>,
        hidden: &B::Tensor,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        self.execute_unit(
            address.group(),
            address.index(),
            unit,
            hidden,
            state,
            forward,
            None,
            context,
        )
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        state.value.dimensions = Some(self.dimensions);
        state.value.frames = forward.next_frames.ok_or(PerceptionError::State(
            "perception traversal omitted subsampling",
        ))?;
        Ok(hidden.clone())
    }
}

impl<B: NeuralBackend> PerceptionKernel<B> {
    fn begin_input<'a>(
        &mut self,
        input: PerceptionInput<'a, B::Tensor>,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, PerceptionForward<B::Tensor>>, PerceptionError> {
        if state.layout != perception_state_layout(self.dimensions)? {
            return Err(PerceptionError::State(
                "perception runtime layout differs from selected geometry",
            ));
        }
        Ok(LayeredForwardState {
            hidden: input.mel.clone(),
            context: PerceptionForward {
                encoder: None,
                drop_preencoded: input.drop_preencoded,
                next_frames: None,
                initial_frames: state.value.frames,
                projected: false,
            },
        })
    }
    fn execute_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut PerceptionUnit<B>,
        hidden: &B::Tensor,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &mut PerceptionForward<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        match (group, unit) {
            (0, PerceptionUnit::Subsampling(module)) => {
                let encoded = module.forward(hidden, context)?;
                let drop = i32::try_from(forward.drop_preencoded).map_err(|_| {
                    PerceptionError::State("subsampling overlap exceeds tensor indexing")
                })?;
                if drop >= encoded.dim(1) {
                    return Err(PerceptionError::State(
                        "subsampling overlap consumes every output frame",
                    ));
                }
                let encoded = if drop == 0 {
                    encoded
                } else {
                    encoded
                        .index(
                            &[Index::Full, Index::Range(drop, encoded.dim(1)), Index::Full],
                            context,
                        )?
                        .compact(context)?
                };
                forward.next_frames = Some(
                    state
                        .value
                        .frames
                        .checked_add(encoded.dim(1) as u64)
                        .ok_or(PerceptionError::State("perception frame clock overflow"))?,
                );
                Ok(encoded)
            }
            (1, PerceptionUnit::Block(module)) => {
                let index = index - self.partition.as_ref().map_or(0, |p| p.blocks.start);
                let (output, candidate) = match parallel {
                    Some(parallel) => module.forward_parallel(
                        hidden,
                        &state.value.blocks[index],
                        parallel,
                        context,
                    )?,
                    None => module.forward(hidden, &state.value.blocks[index], context)?,
                };
                state.value.blocks[index] = candidate;
                Ok(output)
            }
            (2, PerceptionUnit::Projection(module)) => {
                self.precision
                    .require(module, hidden, &self.projection_precision)
                    .map_err(PerceptionError::Precision)?;
                let output = module.forward(hidden, context)?;
                forward.projected = true;
                Ok(output)
            }
            _ => Err(PerceptionError::State(
                "perception leased unit differs from graph address",
            )),
        }
    }
}

impl<B: NeuralBackend>
    ParallelComponentKernel<
        B,
        crate::voicechat::perception_layered::PerceptionRuntimeState<B::Tensor>,
    > for PerceptionKernel<B>
{
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, PerceptionError> {
        if !self.is_parallel() || self.partition.is_some() {
            return Err(PerceptionError::Configuration(
                "parallel perception requires rank-local construction",
            ));
        }
        let local = self
            .local_blocks
            .as_ref()
            .expect("validated parallel selection")[0];
        if B::parallel_size(parallel) != (self.dimensions.block.heads / local.heads) as usize {
            return Err(PerceptionError::Configuration(
                "perception tensor degree differs from the selected communicator",
            ));
        }
        self.begin_input(input, state, context)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        self.execute_unit(
            address.group(),
            address.index(),
            unit,
            hidden,
            state,
            forward,
            Some(parallel),
            context,
        )
    }
}

pub(super) fn output<T>(
    projected: T,
    forward: PerceptionForward<T>,
    state: PerceptionRuntimeState<T>,
) -> Result<PerceptionOutput<T>, PerceptionError>
where
    T: Tensor,
{
    Ok(PerceptionOutput {
        encoder: forward.encoder.ok_or(PerceptionError::State(
            "perception traversal omitted raw encoder capture",
        ))?,
        projected,
        state: state.into_state(),
    })
}

impl<B, P> PerceptionExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, PerceptionKernel<B>>,
        B,
        PerceptionRuntimeState<B::Tensor>,
        P,
    >
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, PerceptionUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> PerceptionDimensions {
        self.architecture().kernel().dimensions()
    }
    fn forward(
        &mut self,
        mel: &B::Tensor,
        drop_preencoded: usize,
        previous: &PerceptionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionOutput<B::Tensor>, PerceptionError> {
        let mut state = PerceptionRuntimeState::from_state(self.dimensions(), previous)?;
        let (projected, forward) = self
            .invoke_with_context(
                PerceptionInput {
                    mel,
                    drop_preencoded,
                },
                &mut state,
                context,
            )
            .map_err(|error| match error {
                eredu_runtime::LayerwiseRuntimeError::Architecture(error) => error,
                eredu_runtime::LayerwiseRuntimeError::State(error) => error.into(),
                eredu_runtime::LayerwiseRuntimeError::Layout(error) => error.into(),
                eredu_runtime::LayerwiseRuntimeError::Policy(error) => backend(error),
                error @ eredu_runtime::LayerwiseRuntimeError::Submission(_) => backend(error),
            })?;
        output(projected, forward, state)
    }
}

/// Selected tensor-parallel perception with caller-owned complete-cache parents.
/// The retained native mechanisms determine admitted rank-local construction.
pub trait ParallelPerceptionExecution<B: NeuralBackend>: PerceptionExecution<B> {
    /// Returns raw encoder features, language projections and unpublished candidate caches.
    fn forward_parallel(
        &mut self,
        mel: &B::Tensor,
        drop_preencoded: usize,
        previous: &PerceptionState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionOutput<B::Tensor>, PerceptionError>;
}

impl<B, P> ParallelPerceptionExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, PerceptionKernel<B>>,
        B,
        PerceptionRuntimeState<B::Tensor>,
        P,
    >
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, PerceptionUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Executes the selected tensor partition against a private complete-cache replica.
    /// Both raw encoder features and language projections use the same invocation.
    fn forward_parallel(
        &mut self,
        mel: &B::Tensor,
        drop_preencoded: usize,
        previous: &PerceptionState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionOutput<B::Tensor>, PerceptionError> {
        let mut state = PerceptionRuntimeState::from_state(self.dimensions(), previous)?;
        let (projected, forward) = self
            .forward_parallel_with_context_hook(
                PerceptionInput {
                    mel,
                    drop_preencoded,
                },
                &mut state,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map_err(|error| match error {
                eredu_runtime::LayerwiseRuntimeError::Architecture(error) => error,
                eredu_runtime::LayerwiseRuntimeError::State(error) => error.into(),
                eredu_runtime::LayerwiseRuntimeError::Layout(error) => error.into(),
                eredu_runtime::LayerwiseRuntimeError::Policy(error) => backend(error),
                error @ eredu_runtime::LayerwiseRuntimeError::Submission(_) => backend(error),
            })?;
        output(projected, forward, state)
    }
}
