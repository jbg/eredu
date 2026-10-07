//! Perception equations over the shared resident and bounded unit traversal.

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
use eredu_nn::{Index, LinearOperator, NeuralBackend, Parameterized, Tensor};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureParameters,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, ExecutionGraph,
    ExecutionGroupSpec, ExecutionUnitAddress, ExecutionUnitLayout, LayeredArchitecture,
    LayeredForwardState, LayeredInvocation, LayerwisePolicy, LayerwiseRuntime, LocalModelLayout,
    MemberSharding, OwnedParameterGroupSpec, ParallelLayeredArchitecture, ParameterGroupOwner,
    ParameterRole, RuntimeState, StateError, StateLayout, StaticParameterVisitor,
    StaticParameterVisitorMut, SubmissionBackend,
};
use std::{marker::PhantomData, num::NonZeroU32};

#[path = "perception_partition.rs"]
mod partition;
pub use partition::{
    unpack_perception_pipeline_output, validate_perception_partition, PerceptionBoundary,
    PerceptionBoundarySchema,
};

const SUBSAMPLING: &str = "voicechat.perception.subsampling";
const ENCODER: &str = "voicechat.perception.encoder";
const PROJECTION: &str = "voicechat.perception.projection";

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

/// Perception keeps every learned parameter in a bounded execution unit.
#[derive(Debug, Default)]
pub struct PerceptionStaticModules;
impl<T: 'static> Parameterized<T> for PerceptionStaticModules {
    fn visit_parameters<'a, V>(&'a self, _: &mut V)
    where
        V: eredu_nn::ParameterVisitor<'a, T>,
    {
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, _: &mut V)
    where
        V: eredu_nn::ParameterVisitorMut<'a, T>,
    {
    }
    fn set_trainable(&mut self, _: bool) {}
}

/// Architecture-owned graph and cold unit construction, with no pinned learned modules.
pub struct PerceptionArchitecture<B: NeuralBackend> {
    dimensions: PerceptionDimensions,
    local_blocks: Option<Vec<ConformerLocalDimensions>>,
    partition: Option<partition::PerceptionOwnership>,
    empty: PerceptionStaticModules,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> PerceptionArchitecture<B> {
    /// Validates component geometry before constructing any parameters.
    pub fn new(dimensions: PerceptionDimensions) -> Result<Self, PerceptionError> {
        dimensions.validate()?;
        dimensions.block.validate()?;
        dimensions.subsampling.validate()?;
        Ok(Self {
            dimensions,
            local_blocks: None,
            partition: None,
            empty: PerceptionStaticModules,
            backend: PhantomData,
        })
    }
    /// Constructs the selected rank's projections and complete replicated caches.
    pub fn new_parallel(
        dimensions: PerceptionDimensions,
        layout: &LocalModelLayout,
    ) -> Result<Self, PerceptionError> {
        let local_blocks = perception_local_dimensions(dimensions, layout)?;
        let mut architecture = Self::new(dimensions)?;
        architecture.local_blocks = Some(local_blocks);
        Ok(architecture)
    }
    /// Whether execution requires the selected tensor collective.
    pub fn is_parallel(&self) -> bool {
        self.local_blocks.is_some()
    }
    /// Connected perception geometry, independent of residency.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
    }
    fn count(&self, group: usize) -> Result<usize, PerceptionError> {
        match group {
            0 | 2 => Ok(1),
            1 => Ok(self.dimensions.layers),
            _ => Err(PerceptionError::Configuration(
                "unknown perception execution group",
            )),
        }
    }
    fn validate_unit(&self, group: usize, index: usize) -> Result<(), PerceptionError> {
        if index >= self.count(group)? {
            return Err(PerceptionError::Configuration(
                "perception unit index exceeds group",
            ));
        }
        Ok(())
    }
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

/// Exact symbolic learned parameters and unit ownership, without backend allocation.
pub fn perception_parameter_description(
    dimensions: PerceptionDimensions,
) -> Result<ArchitectureParameterDescription, PerceptionError> {
    use eredu_runtime::{ParameterGroupSpec, ParameterMemberSpec};
    dimensions.validate()?;
    dimensions.block.validate()?;
    dimensions.subsampling.validate()?;
    let graph = perception_graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [1, dimensions.layers, 1])?;
    let mut expected = Vec::new();
    let mut owned = Vec::new();
    let mut add =
        |group: usize, index: usize, spec: ParameterGroupSpec| -> Result<(), PerceptionError> {
            expected.push(spec.clone());
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(
                    layout.group_id(group).expect("declared group").clone(),
                    index,
                ),
                spec,
            ));
            Ok(())
        };
    let member = |name: String, shape: Vec<usize>| {
        ParameterMemberSpec::new(name, shape, MemberSharding::Replicated)
    };
    let prefix = "stt_model.perception.encoder.pre_encode";
    let c = dimensions.subsampling.channels as usize;
    let width = dimensions.subsampling.width as usize;
    let flattened = dimensions
        .subsampling
        .output_frames(dimensions.subsampling.features)? as usize
        * c;
    let mut members = Vec::new();
    for (index, input, kernel) in [(0, 1, 3), (2, 1, 3), (3, c, 1), (5, 1, 3), (6, c, 1)] {
        members.push(member(
            format!("{prefix}.conv.{index}.weight"),
            vec![c, input, kernel, kernel],
        ));
        members.push(member(format!("{prefix}.conv.{index}.bias"), vec![c]));
    }
    members.push(member(
        format!("{prefix}.out.weight"),
        vec![width, flattened],
    ));
    members.push(member(format!("{prefix}.out.bias"), vec![width]));
    add(
        0,
        0,
        ParameterGroupSpec::new(prefix, ParameterRole::Replicated, members)?,
    )?;
    let d = dimensions.block.width as usize;
    let f = dimensions.block.ff_hidden as usize;
    let heads = dimensions.block.heads as usize;
    for index in 0..dimensions.layers {
        let prefix = format!("stt_model.perception.encoder.layers.{index}");
        let mut members = Vec::new();
        for norm in [
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "conv.batch_norm",
            "norm_feed_forward2",
            "norm_out",
        ] {
            members.push(member(format!("{prefix}.{norm}.weight"), vec![d]));
            members.push(member(format!("{prefix}.{norm}.bias"), vec![d]));
        }
        for name in ["feed_forward1", "feed_forward2"] {
            add(
                1,
                index,
                ParameterGroupSpec::partitioned(
                    format!("{prefix}.{name}.projections"),
                    ParameterRole::FeedForwardIntermediate,
                    f,
                    [
                        ParameterMemberSpec::new(
                            format!("{prefix}.{name}.linear1.weight"),
                            vec![f, d],
                            MemberSharding::Partitioned { axis: 0 },
                        ),
                        ParameterMemberSpec::new(
                            format!("{prefix}.{name}.linear2.weight"),
                            vec![d, f],
                            MemberSharding::Partitioned { axis: 1 },
                        ),
                    ],
                )?,
            )?;
        }
        let mut attention = Vec::new();
        for name in [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_pos",
            "linear_out",
        ] {
            attention.push(ParameterMemberSpec::new(
                format!("{prefix}.self_attn.{name}.weight"),
                vec![d, d],
                MemberSharding::Partitioned {
                    axis: usize::from(name == "linear_out"),
                },
            ));
        }
        for name in ["pos_bias_u", "pos_bias_v"] {
            attention.push(ParameterMemberSpec::new(
                format!("{prefix}.self_attn.{name}"),
                vec![heads, d / heads],
                MemberSharding::Partitioned { axis: 0 },
            ));
        }
        add(
            1,
            index,
            ParameterGroupSpec::partitioned(
                format!("{prefix}.self_attn.projections"),
                ParameterRole::AttentionHeads,
                heads,
                attention,
            )?,
        )?;
        members.push(member(
            format!("{prefix}.conv.pointwise_conv1.weight"),
            vec![
                d.checked_mul(2).ok_or(PerceptionError::Configuration(
                    "Conformer projection width overflow",
                ))?,
                d,
                1,
            ],
        ));
        members.push(member(
            format!("{prefix}.conv.depthwise_conv.weight"),
            vec![d, 1, dimensions.block.conv_kernel as usize],
        ));
        members.push(member(
            format!("{prefix}.conv.pointwise_conv2.weight"),
            vec![d, d, 1],
        ));
        add(
            1,
            index,
            ParameterGroupSpec::new(
                format!("{prefix}.replicated"),
                ParameterRole::Replicated,
                members,
            )?,
        )?;
    }
    let prefix = "stt_model.perception.proj";
    add(
        2,
        0,
        ParameterGroupSpec::new(
            prefix,
            ParameterRole::Replicated,
            vec![
                member(
                    format!("{prefix}.weight"),
                    vec![dimensions.output_width as usize, d],
                ),
                member(
                    format!("{prefix}.bias"),
                    vec![dimensions.output_width as usize],
                ),
            ],
        )?,
    )?;
    Ok(ArchitectureParameterDescription::new(
        &graph, &layout, expected, owned,
    )?)
}

/// Resolves physical geometry using the shared symbolic placement compiler.
pub(super) fn perception_local_dimensions(
    dimensions: PerceptionDimensions,
    layout: &LocalModelLayout,
) -> Result<Vec<ConformerLocalDimensions>, PerceptionError> {
    let description = perception_parameter_description(dimensions)?;
    let query = layout
        .tensor("stt_model.perception.encoder.layers.0.self_attn.linear_q.weight")
        .ok_or(PerceptionError::Configuration(
            "perception layout lacks its attention partition",
        ))?;
    let heads = dimensions.block.heads as usize;
    let range = query.expanded_logical_range(heads)?;
    let local_heads = range.len();
    if local_heads == 0
        || heads % local_heads != 0
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
        ff_hidden: dimensions.block.ff_hidden / (heads / local_heads) as i32,
    };
    local.validate(dimensions.block)?;
    Ok(vec![local; dimensions.layers])
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

impl<B: NeuralBackend> ArchitectureParameters<B> for PerceptionArchitecture<B> {
    type DefinitionError = PerceptionError;
    fn state_layout(&self) -> Result<StateLayout, PerceptionError> {
        perception_state_layout(self.dimensions)
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, PerceptionError> {
        let global = self.state_layout()?;
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
            format!("{:?}", self.dimensions),
            global.len(),
            begin,
            0,
            topology,
        )
        .map_err(PerceptionError::from)
    }
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, PerceptionError> {
        perception_parameter_description(self.dimensions)
    }
    fn visit_static_parameters<V: StaticParameterVisitor<B>>(
        &self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
}

impl<B: NeuralBackend> LayeredArchitecture<B, PerceptionRuntimeState<B::Tensor>>
    for PerceptionArchitecture<B>
{
    type Input<'a> = PerceptionInput<'a, B::Tensor>;
    type StaticModules = PerceptionStaticModules;
    type Unit = PerceptionUnit<B>;
    type ForwardContext = PerceptionForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = PerceptionError;
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        ArchitectureGroupTransport {
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
        }
    }
    fn primary_execution_group(&self) -> &str {
        ENCODER
    }
    fn state_partition_plan(&self, _: &StateLayout) -> ArchitectureStatePartitionPlan {
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            1,
            0..self.dimensions.layers,
        )])
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, PerceptionError> {
        perception_graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, PerceptionError> {
        self.count(group)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, PerceptionError> {
        self.validate_unit(group, index)?;
        Ok(match group {
            0 => "stt_model.perception.encoder.pre_encode".into(),
            1 => format!("stt_model.perception.encoder.layers.{index}"),
            _ => "stt_model.perception.proj".into(),
        })
    }
    fn static_modules(&self) -> &PerceptionStaticModules {
        &self.empty
    }
    fn static_modules_mut(&mut self) -> &mut PerceptionStaticModules {
        &mut self.empty
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionUnit<B>, PerceptionError> {
        self.validate_unit(group, index)?;
        self.require_owned(group, index)?;
        Ok(match group {
            0 => PerceptionUnit::Subsampling(CausalSubsampling::new(
                self.dimensions.subsampling,
                context,
            )?),
            1 => PerceptionUnit::Block(match &self.local_blocks {
                Some(blocks) => ConformerBlock::new_parallel(
                    self.dimensions.block,
                    blocks[index],
                    index,
                    context,
                )?,
                None => ConformerBlock::new(self.dimensions.block, index, context)?,
            }),
            _ => PerceptionUnit::Projection(self.dimensions.build_projection::<B>(context)?),
        })
    }
    fn begin_forward<'a>(
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
        self.begin(input, state, context)
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        self.count(group)?;
        let hidden = match (group, dependencies) {
            (0, []) => initial,
            (_, [one]) if group != 0 => *one,
            _ => {
                return Err(PerceptionError::State(
                    "perception group dependency mismatch",
                ))
            }
        };
        if group == 2 {
            forward.encoder = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }
    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        _: usize,
    ) -> std::ops::Range<usize> {
        if group == 1 {
            let local = index.saturating_sub(self.partition.as_ref().map_or(0, |p| p.blocks.start));
            local..local + 1
        } else {
            0..0
        }
    }
    fn state_ordinal(&self, group: usize, index: usize, _: usize) -> usize {
        if group == 1 {
            index.saturating_sub(self.partition.as_ref().map_or(0, |p| p.blocks.start))
        } else {
            0
        }
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut PerceptionUnit<B>,
        hidden: &B::Tensor,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        self.execute_unit(group, index, unit, hidden, state, forward, None, context)
    }
    fn finish_forward(
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
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.encoder.iter()
    }
}

impl<B: NeuralBackend> PerceptionArchitecture<B> {
    fn begin<'a>(
        &mut self,
        input: PerceptionInput<'a, B::Tensor>,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, PerceptionForward<B::Tensor>>, PerceptionError> {
        if state.layout != self.state_layout()? {
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
        self.validate_unit(group, index)?;
        self.require_owned(group, index)?;
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
                super::precision::require_float32(module, hidden)
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

impl<B: NeuralBackend> ParallelLayeredArchitecture<B, PerceptionRuntimeState<B::Tensor>>
    for PerceptionArchitecture<B>
{
    fn begin_forward_parallel<'a>(
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
        self.begin(input, state, context)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        self.execute_unit(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            Some(parallel),
            context,
        )
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &Self::ForwardContext,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        self.finish_forward(hidden, state, forward, context)
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

/// Ordinary shared layerwise runtime, with the same perception equations and state.
pub struct LayerwisePerception<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, PerceptionUnit<B>>,
{
    runtime: LayerwiseRuntime<PerceptionArchitecture<B>, B, PerceptionRuntimeState<B::Tensor>, P>,
}
impl<B, P> LayerwisePerception<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, PerceptionUnit<B>>,
    P::Error: std::fmt::Display,
{
    /// Adopts the existing provider without constructing resident replacement units.
    pub fn new(architecture: PerceptionArchitecture<B>, policy: P) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Shared runtime for parameter operations and diagnostics.
    pub fn runtime(
        &self,
    ) -> &LayerwiseRuntime<PerceptionArchitecture<B>, B, PerceptionRuntimeState<B::Tensor>, P> {
        &self.runtime
    }
    /// Shared runtime for bounded parameter operations.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<PerceptionArchitecture<B>, B, PerceptionRuntimeState<B::Tensor>, P>
    {
        &mut self.runtime
    }
}
impl<B, P> PerceptionExecution<B> for LayerwisePerception<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, PerceptionUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> PerceptionDimensions {
        self.runtime.architecture().dimensions()
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
            .runtime
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

impl<B, P> LayerwisePerception<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, PerceptionUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Executes the selected tensor partition against a private complete-cache replica.
    /// Both raw encoder features and language projections use the same invocation.
    pub fn forward_parallel(
        &mut self,
        mel: &B::Tensor,
        drop_preencoded: usize,
        previous: &PerceptionState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionOutput<B::Tensor>, PerceptionError> {
        let mut state = PerceptionRuntimeState::from_state(self.dimensions(), previous)?;
        let (projected, forward) = self
            .runtime
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
