//! Independent function-vocabulary projection over shared bounded execution.
use super::channels::{
    agree_parallel_projection, compute_type, head_spec, validate_logits, ChannelError,
    DualChannelLogits, FunctionHead,
};
use crate::decoder::StaticModules;
use eredu_core::{cache::LayerCachePolicy, BackendFailure, LayerSchedule};
use eredu_nn::{
    DistributedNeuralBackend, LinearFormatSpec, NeuralBackend, Parameterized, Tensor,
    VocabularyParallelRange,
};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureParameters,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, ExecutionGraph,
    ExecutionGroupSpec, ExecutionUnitAddress, ExecutionUnitLayout, LayeredArchitecture,
    LayeredForwardState, LayeredInvocation, LayeredTraversalHook, LayerwisePolicy,
    LayerwiseRuntime, LocalModelLayout, LocalTensorLayout, MemberSharding, OwnedParameterGroupSpec,
    ParallelLayeredArchitecture, ParallelLayeredInvocation, ParameterGroupOwner,
    ParameterGroupSpec, ParameterMemberSpec, ParameterRole, RuntimeState, StateError, StateLayout,
    StaticParameterVisitor, StaticParameterVisitorMut, SubmissionBackend, TensorPlacement,
};
use std::marker::PhantomData;

const FUNCTION: &str = "voicechat.function_head";
/// Logical function projection dimensions, independent of storage encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionHeadDimensions {
    /// Normalized Nemotron hidden width.
    pub width: i32,
    /// Complete function token domain.
    pub vocabulary: i32,
}
impl FunctionHeadDimensions {
    /// Validates logical dimensions without native resources.
    pub fn new(width: i32, vocabulary: i32) -> Result<Self, ChannelError> {
        let value = Self { width, vocabulary };
        value.validate()?;
        Ok(value)
    }
    /// Rejects invalid logical dimensions before parameter allocation.
    pub fn validate(self) -> Result<(), ChannelError> {
        if self.width <= 0 || self.vocabulary <= 0 {
            return Err(ChannelError::Configuration(
                "function-head dimensions must be positive",
            ));
        }
        Ok(())
    }
    /// Validates normalized hidden rank, width and floating arithmetic before acquisition.
    pub fn validate_hidden<T: Tensor>(self, hidden: &T) -> Result<(), ChannelError> {
        self.validate()?;
        if hidden.shape().len() != 3
            || hidden.dim(2) != self.width
            || hidden.shape().iter().any(|n| *n <= 0)
        {
            return Err(ChannelError::Shape(
                "function head requires normalized [batch, sequence, width] hidden states",
            ));
        }
        compute_type(hidden)?;
        Ok(())
    }
}
/// Portable independent function readout from an already normalized backbone result.
pub trait FunctionHeadExecution<B: NeuralBackend> {
    /// Logical geometry used by shared dual-head preflight.
    fn dimensions(&self) -> FunctionHeadDimensions;
    /// Projects the normalized hidden state through the independently leased head.
    fn project(
        &mut self,
        normalized: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError>;
    /// Validates replicated projection before either vocabulary head executes.
    fn validate_projection(&self, normalized: &B::Tensor) -> Result<(), ChannelError> {
        self.dimensions().validate_hidden(normalized)
    }
    /// Projects both vocabulary heads from one retained normalized backbone result.
    fn project_channels(
        &mut self,
        normalized: &B::Tensor,
        backbone: &mut StaticModules<B>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DualChannelLogits<B::Tensor>, ChannelError> {
        self.validate_projection(normalized)?;
        let agent = backbone.project_normalized(normalized, context)?;
        let function = self.project(normalized, context)?;
        validate_logits(&agent, &function, self.dimensions().vocabulary)?;
        Ok(DualChannelLogits { agent, function })
    }
}
impl<B: NeuralBackend> FunctionHeadExecution<B> for FunctionHead<B> {
    fn dimensions(&self) -> FunctionHeadDimensions {
        self.dimensions()
    }
    fn project(
        &mut self,
        normalized: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.project(normalized, context)
    }
    fn validate_projection(&self, normalized: &B::Tensor) -> Result<(), ChannelError> {
        self.validate_replicated_projection(normalized)
    }
}
/// Collective function readout over an already normalized backbone result.
/// The selected communicator belongs to this head and can differ from the backbone's.
pub trait ParallelFunctionHeadExecution<B: DistributedNeuralBackend>:
    FunctionHeadExecution<B>
{
    /// Validates hidden geometry and selected row ownership before either head executes.
    fn validate_parallel_projection(
        &self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
    ) -> Result<(), ChannelError>;
    /// Projects and gathers the independently selected function vocabulary.
    fn project_parallel(
        &mut self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError>;
}
impl<B: DistributedNeuralBackend> ParallelFunctionHeadExecution<B> for FunctionHead<B> {
    fn validate_parallel_projection(
        &self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
    ) -> Result<(), ChannelError> {
        FunctionHead::validate_parallel_projection(self, normalized, parallel)
    }
    fn project_parallel(
        &mut self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        FunctionHead::project_parallel(self, normalized, parallel, context)
    }
}

/// Borrowed collective selection consumed by the ordinary coupled execution driver.
/// It preserves the selected head's parameter provider and communicator.
pub struct ParallelFunctionHead<'a, B, F>
where
    B: DistributedNeuralBackend,
    F: ParallelFunctionHeadExecution<B> + ?Sized,
{
    head: &'a mut F,
    parallel: &'a B::ParallelContext,
}
impl<'a, B, F> ParallelFunctionHead<'a, B, F>
where
    B: DistributedNeuralBackend,
    F: ParallelFunctionHeadExecution<B> + ?Sized,
{
    /// Borrows the selected head and its exact communicator for coupled execution.
    pub fn new(head: &'a mut F, parallel: &'a B::ParallelContext) -> Self {
        Self { head, parallel }
    }
}
impl<B, F> FunctionHeadExecution<B> for ParallelFunctionHead<'_, B, F>
where
    B: DistributedNeuralBackend,
    F: ParallelFunctionHeadExecution<B> + ?Sized,
{
    fn dimensions(&self) -> FunctionHeadDimensions {
        self.head.dimensions()
    }
    fn validate_projection(&self, normalized: &B::Tensor) -> Result<(), ChannelError> {
        self.head
            .validate_parallel_projection(normalized, self.parallel)
    }
    fn project(
        &mut self,
        normalized: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.head
            .project_parallel(normalized, self.parallel, context)
    }
}
/// All learned function-head parameters belong to execution units.
pub struct FunctionHeadStaticModules;
impl<T: 'static> Parameterized<T> for FunctionHeadStaticModules {
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

/// Explicit stateless traversal owner; input and result tensors belong to the invocation.
pub struct FunctionHeadRuntimeState {
    layout: StateLayout,
}
impl FunctionHeadRuntimeState {
    /// Constructs host-only empty state metadata.
    pub fn new() -> Result<Self, ChannelError> {
        Ok(Self {
            layout: function_head_state_layout()?,
        })
    }
}
impl<B: NeuralBackend> RuntimeState<B> for FunctionHeadRuntimeState {
    type RetainedValues<'a>
        = std::iter::Empty<&'a B::Tensor>
    where
        B::Tensor: 'a;
    fn layout(&self) -> &StateLayout {
        &self.layout
    }
    fn optional_layout(&self) -> Option<&StateLayout> {
        None
    }
    fn retained_values(
        &self,
        ordinal: usize,
        _: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        if ordinal != 0 {
            return Err(StateError::UnknownLayer {
                layer: ordinal,
                count: 1,
            });
        }
        Ok(std::iter::empty())
    }
}
/// Function readout metadata declares no persistent carry.
pub fn function_head_state_layout() -> Result<StateLayout, ChannelError> {
    Ok(StateLayout::new(LayerSchedule::new(
        1,
        vec![LayerCachePolicy::NoState],
    )?)?)
}
// The callback retains the additive vocabulary mechanism without requiring ordinary
// replicated backends to implement distributed construction.
struct ParallelFunctionHeadDefinition<B: NeuralBackend> {
    range: VocabularyParallelRange,
    layout: LocalModelLayout,
    build: fn(
        i32,
        i32,
        LinearFormatSpec,
        VocabularyParallelRange,
        &<B::Tensor as Tensor>::Context,
    ) -> Result<FunctionHead<B>, ChannelError>,
}

/// Validates every physical vocabulary member and returns its common row ownership.
/// Communicator rank and degree are checked at invocation, since uneven row ranges
/// do not uniquely identify a topology.
pub fn function_head_parallel_range(
    dimensions: FunctionHeadDimensions,
    format: &LinearFormatSpec,
    layout: &LocalModelLayout,
) -> Result<VocabularyParallelRange, ChannelError> {
    let description = function_head_parameter_description(dimensions, format)?;
    let weight =
        layout
            .tensor("stt_model.function_head.weight")
            .ok_or(ChannelError::Configuration(
                "function-head layout lacks its weight",
            ))?;
    let local = match weight.placement() {
        TensorPlacement::Range {
            axis: 0,
            start,
            end,
        } => *start..*end,
        _ => {
            return Err(ChannelError::Configuration(
                "function-head layout requires exact vocabulary row placement",
            ));
        }
    };
    let range = VocabularyParallelRange {
        global_vocabulary: dimensions.vocabulary as usize,
        local,
    };
    range.validate()?;
    let mut expected = LocalModelLayout::default();
    for owned in description.groups() {
        let group = owned.group();
        for member in group.members() {
            if member.sharding() != &(MemberSharding::Balanced { axis: 0 })
                || member.global_shape().first().copied() != Some(range.global_vocabulary)
            {
                return Err(ChannelError::Configuration(
                    "function-head physical member does not preserve vocabulary rows",
                ));
            }
            let mut shape = member.global_shape().to_vec();
            shape[0] = range.local.len();
            expected.insert(
                member.target().to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    shape,
                    TensorPlacement::Range {
                        axis: 0,
                        start: range.local.start,
                        end: range.local.end,
                    },
                    None,
                    None,
                    false,
                ),
            );
        }
    }
    if layout != &expected {
        return Err(ChannelError::Configuration(
            "function-head physical layout differs from its complete row declaration",
        ));
    }
    Ok(range)
}

/// Architecture-owned independent readout and physical matrix declaration.
pub struct FunctionHeadArchitecture<B: NeuralBackend> {
    dimensions: FunctionHeadDimensions,
    format: LinearFormatSpec,
    parallel: Option<ParallelFunctionHeadDefinition<B>>,
    empty: FunctionHeadStaticModules,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> FunctionHeadArchitecture<B> {
    /// Validates dimensions and exact physical companions without allocating tensors.
    pub fn new(
        dimensions: FunctionHeadDimensions,
        format: LinearFormatSpec,
    ) -> Result<Self, ChannelError> {
        dimensions.validate()?;
        head_spec(dimensions.width, dimensions.vocabulary, format.clone())?;
        function_head_parameter_description(dimensions, &format)?;
        Ok(Self {
            dimensions,
            format,
            parallel: None,
            empty: FunctionHeadStaticModules,
            backend: PhantomData,
        })
    }
    /// Whether projection requires collective execution.
    pub fn is_parallel(&self) -> bool {
        self.parallel.is_some()
    }
    /// Validates an ordinary invocation before either head or its provider runs.
    pub fn validate_replicated_projection(&self, hidden: &B::Tensor) -> Result<(), ChannelError> {
        self.dimensions.validate_hidden(hidden)?;
        if self.is_parallel() {
            return Err(ChannelError::Configuration(
                "rank-local function head requires collective projection",
            ));
        }
        Ok(())
    }
    /// Logical selected projection geometry.
    pub fn dimensions(&self) -> FunctionHeadDimensions {
        self.dimensions
    }
    /// Exact selected physical weight/companion encoding.
    pub fn format(&self) -> &LinearFormatSpec {
        &self.format
    }
    /// Independent single-unit readout graph.
    pub fn execution_graph(&self) -> Result<ExecutionGraph, ChannelError> {
        function_head_graph()
    }
    /// The readout has exactly one independently leased unit.
    pub fn group_unit_count(&self, group: usize) -> Result<usize, ChannelError> {
        if group == 0 {
            Ok(1)
        } else {
            Err(ChannelError::Configuration("unknown function-head group"))
        }
    }
    /// Canonical source namespace of the readout.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, ChannelError> {
        self.group_unit_count(group)?;
        if index != 0 {
            return Err(ChannelError::Configuration("unknown function-head unit"));
        }
        Ok("stt_model.function_head".into())
    }
    /// Constructs the selected unloaded projection with its exact row ownership.
    pub fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<FunctionHead<B>, ChannelError> {
        self.unit_path(group, index)?;
        if let Some(parallel) = &self.parallel {
            return (parallel.build)(
                self.dimensions.width,
                self.dimensions.vocabulary,
                self.format.clone(),
                parallel.range.clone(),
                context,
            );
        }
        FunctionHead::new(
            self.dimensions.width,
            self.dimensions.vocabulary,
            self.format.clone(),
            context,
        )
    }
}
impl<B: DistributedNeuralBackend> FunctionHeadArchitecture<B> {
    /// Admits exact vocabulary rows and their physical companions without allocation.
    pub fn new_parallel(
        dimensions: FunctionHeadDimensions,
        format: LinearFormatSpec,
        layout: &LocalModelLayout,
    ) -> Result<Self, ChannelError> {
        let range = function_head_parallel_range(dimensions, &format, layout)?;
        let mut architecture = Self::new(dimensions, format)?;
        architecture.parallel = Some(ParallelFunctionHeadDefinition {
            range,
            layout: layout.clone(),
            build: FunctionHead::<B>::new_parallel,
        });
        Ok(architecture)
    }
    /// Checks hidden geometry and exact communicator ownership before parameter acquisition.
    pub fn validate_parallel_projection(
        &self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
    ) -> Result<(), ChannelError> {
        self.dimensions.validate_hidden(hidden)?;
        let selected = self.parallel.as_ref().ok_or(ChannelError::Configuration(
            "collective function projection requires rank-local construction",
        ))?;
        selected
            .range
            .balanced_peer_widths(B::parallel_size(parallel), B::parallel_rank(parallel))?;
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(B::parallel_size(parallel), 1, 1, 1)
                .map_err(|_| ChannelError::Configuration("invalid function-head tensor degree"))?,
            B::parallel_rank(parallel),
        )
        .map_err(|_| ChannelError::Configuration("invalid function-head tensor rank"))?;
        let description = function_head_parameter_description(self.dimensions, &self.format)?;
        let expected = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(|_| {
                ChannelError::Configuration("function-head tensor placement cannot be derived")
            })?;
        if expected != selected.layout {
            return Err(ChannelError::Configuration(
                "function-head layout differs from the selected communicator",
            ));
        }
        Ok(())
    }
    fn validate_parallel_unit(
        &self,
        group: usize,
        index: usize,
        unit: &FunctionHead<B>,
    ) -> Result<(), ChannelError> {
        self.unit_path(group, index)?;
        let selected = self.parallel.as_ref().ok_or(ChannelError::Configuration(
            "collective function projection requires rank-local construction",
        ))?;
        if unit.dimensions() != self.dimensions
            || unit.format() != &self.format
            || unit.parallel_range() != Some(&selected.range)
        {
            return Err(ChannelError::Configuration(
                "leased function head differs from selected geometry, format or ownership",
            ));
        }
        Ok(())
    }
}

fn function_head_graph() -> Result<ExecutionGraph, ChannelError> {
    Ok(ExecutionGraph::new(
        vec![ExecutionGroupSpec::root(FUNCTION)],
        FUNCTION,
    )?)
}
/// Cold vocabulary-axis partitioning preserving every physical format companion.
pub fn function_head_parameter_description(
    dimensions: FunctionHeadDimensions,
    format: &LinearFormatSpec,
) -> Result<ArchitectureParameterDescription, ChannelError> {
    dimensions.validate()?;
    head_spec(dimensions.width, dimensions.vocabulary, format.clone())?;
    let graph = function_head_graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [1])?;
    let logical = ParameterGroupSpec::new(
        FUNCTION,
        ParameterRole::Vocabulary,
        [ParameterMemberSpec::new(
            "stt_model.function_head.weight",
            vec![dimensions.vocabulary as usize, dimensions.width as usize],
            MemberSharding::Balanced { axis: 0 },
        )],
    )?;
    let expected = eredu_runtime::expand_linear_format_parameter_groups(vec![logical], |_| {
        Ok(Some(format.clone()))
    })?;
    let owned: Vec<_> = expected
        .iter()
        .cloned()
        .map(|spec| {
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(
                    layout.group_id(0).expect("function group").clone(),
                    0,
                ),
                spec,
            )
        })
        .collect();
    Ok(ArchitectureParameterDescription::new(
        &graph, &layout, expected, owned,
    )?)
}
impl<B: NeuralBackend> ArchitectureParameters<B> for FunctionHeadArchitecture<B> {
    type DefinitionError = ChannelError;
    fn state_layout(&self) -> Result<StateLayout, ChannelError> {
        function_head_state_layout()
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, ChannelError> {
        if state.global_layer_offset() != 0 || *state.layout() != function_head_state_layout()? {
            return Err(ChannelError::Configuration(
                "function-head stateless layout mismatch",
            ));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "nemotron_voicechat",
            "function_head",
            format!(
                "{:?}:{:?}:{:?}",
                self.dimensions,
                self.format,
                self.parallel.as_ref().map(|selected| &selected.layout)
            ),
            1,
            0,
            0,
            topology,
        )?)
    }
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, ChannelError> {
        function_head_parameter_description(self.dimensions, &self.format)
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
impl<B: NeuralBackend> LayeredArchitecture<B, FunctionHeadRuntimeState>
    for FunctionHeadArchitecture<B>
{
    type Input<'a> = &'a B::Tensor;
    type StaticModules = FunctionHeadStaticModules;
    type Unit = FunctionHead<B>;
    type ForwardContext = ();
    type RetainedContextValues<'a>
        = std::iter::Empty<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = ChannelError;
    fn group_transport(&self, _: usize) -> ArchitectureGroupTransport {
        ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Prediction,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }
    }
    fn primary_execution_group(&self) -> &str {
        FUNCTION
    }
    fn state_partition_plan(&self, _: &StateLayout) -> ArchitectureStatePartitionPlan {
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::output_owner(0..1)])
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, ChannelError> {
        self.execution_graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, ChannelError> {
        self.group_unit_count(group)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, ChannelError> {
        self.unit_path(group, index)
    }
    fn static_modules(&self) -> &FunctionHeadStaticModules {
        &self.empty
    }
    fn static_modules_mut(&mut self) -> &mut FunctionHeadStaticModules {
        &mut self.empty
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, ChannelError> {
        self.build_unit(group, index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        hidden: Self::Input<'a>,
        state: &mut FunctionHeadRuntimeState,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ()>, ChannelError> {
        if state.layout != function_head_state_layout()? {
            return Err(ChannelError::Configuration(
                "function-head runtime layout mismatch",
            ));
        }
        self.validate_replicated_projection(hidden)?;
        Ok(LayeredForwardState {
            hidden: hidden.clone(),
            context: (),
        })
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut FunctionHeadRuntimeState,
        _: &mut (),
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        if group != 0 || !dependencies.is_empty() {
            return Err(ChannelError::Configuration(
                "function-head graph dependency mismatch",
            ));
        }
        Ok(initial.clone())
    }
    fn retained_state_ordinals(&self, _: usize, _: usize, _: usize) -> std::ops::Range<usize> {
        0..0
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut FunctionHeadRuntimeState,
        _: &mut (),
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.unit_path(group, index)?;
        if unit.dimensions() != self.dimensions || unit.format() != &self.format {
            return Err(ChannelError::Configuration(
                "leased function head differs from selected geometry or format",
            ));
        }
        unit.project(hidden, context)
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _: &mut FunctionHeadRuntimeState,
        _: &(),
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        Ok(hidden.clone())
    }
    fn retained_context_values<'a>(
        &'a self,
        _: &'a (),
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}
impl<B: DistributedNeuralBackend> ParallelLayeredArchitecture<B, FunctionHeadRuntimeState>
    for FunctionHeadArchitecture<B>
{
    fn begin_forward_parallel<'a>(
        &mut self,
        hidden: Self::Input<'a>,
        state: &mut FunctionHeadRuntimeState,
        parallel: &B::ParallelContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ()>, ChannelError> {
        self.validate_parallel_projection(hidden, parallel)?;
        if state.layout != function_head_state_layout()? {
            return Err(ChannelError::Configuration(
                "function-head runtime layout mismatch",
            ));
        }
        Ok(LayeredForwardState {
            hidden: hidden.clone(),
            context: (),
        })
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut FunctionHeadRuntimeState,
        _: &mut (),
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.validate_parallel_unit(group, index, unit)?;
        unit.project_parallel(hidden, parallel, context)
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut FunctionHeadRuntimeState,
        forward: &(),
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.finish_forward(hidden, state, forward, context)
    }
}

/// Neural function head using the existing prepared parameter provider and bounded traversal.
pub struct LayerwiseFunctionHead<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, FunctionHead<B>>,
{
    runtime: LayerwiseRuntime<FunctionHeadArchitecture<B>, B, FunctionHeadRuntimeState, P>,
}
impl<B, P> LayerwiseFunctionHead<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, FunctionHead<B>>,
    P::Error: std::fmt::Display,
{
    /// Adopts an exact function head unit policy.
    pub fn new(architecture: FunctionHeadArchitecture<B>, policy: P) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Existing shared provider/runtime for diagnostics and parameter inspection.
    pub fn runtime(
        &self,
    ) -> &LayerwiseRuntime<FunctionHeadArchitecture<B>, B, FunctionHeadRuntimeState, P> {
        &self.runtime
    }
    /// Existing shared provider/runtime for bounded parameter operations.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<FunctionHeadArchitecture<B>, B, FunctionHeadRuntimeState, P> {
        &mut self.runtime
    }
}
fn runtime_error<P: std::error::Error + Send + Sync + 'static>(
    error: eredu_runtime::LayerwiseRuntimeError<ChannelError, P>,
) -> ChannelError {
    match error {
        eredu_runtime::LayerwiseRuntimeError::Architecture(error) => error,
        eredu_runtime::LayerwiseRuntimeError::State(error) => error.into(),
        eredu_runtime::LayerwiseRuntimeError::Layout(error) => error.into(),
        eredu_runtime::LayerwiseRuntimeError::Policy(error) => {
            BackendFailure::from_error(error).into()
        }
        error @ eredu_runtime::LayerwiseRuntimeError::Submission(_) => {
            BackendFailure::from_error(error).into()
        }
    }
}

impl<B, P> FunctionHeadExecution<B> for LayerwiseFunctionHead<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, FunctionHead<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> FunctionHeadDimensions {
        self.runtime.architecture().dimensions()
    }
    fn validate_projection(&self, normalized: &B::Tensor) -> Result<(), ChannelError> {
        self.runtime
            .architecture()
            .validate_replicated_projection(normalized)
    }
    fn project(
        &mut self,
        normalized: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        let mut state = FunctionHeadRuntimeState::new()?;
        self.runtime
            .invoke(normalized, &mut state, context)
            .map_err(runtime_error)
    }
}

impl<B, P> LayerwiseFunctionHead<B, P>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, FunctionHead<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Projects and gathers function logits through the existing bounded traversal.
    pub fn project_parallel(
        &mut self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        let mut state = FunctionHeadRuntimeState::new()?;
        self.runtime
            .invoke_parallel(normalized, &mut state, parallel, context)
            .map_err(runtime_error)
    }
    /// Coordinates input admission, provider lifecycle and leased ownership before
    /// the independently selected function-vocabulary collective.
    pub fn project_parallel_guarded<H, F>(
        &mut self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        mut agree: F,
    ) -> Result<B::Tensor, ChannelError>
    where
        H: LayeredTraversalHook<B, (), ChannelError> + ?Sized,
        F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, ChannelError>,
    {
        let prepared = (|| {
            self.runtime
                .architecture()
                .validate_parallel_projection(normalized, parallel)?;
            FunctionHeadRuntimeState::new()
        })();
        let agreement = agree_parallel_projection::<B, _>(
            self.dimensions(),
            normalized,
            parallel,
            prepared.is_ok(),
            context,
            &mut agree,
        );
        let mut state = prepared?;
        agreement?;
        self.runtime
            .forward_parallel_with_unit_executor_and_traversal_hook(
                normalized,
                &mut state,
                parallel,
                context,
                |architecture, group, index, unit, hidden, _, _, parallel, context| {
                    let valid = architecture.validate_parallel_unit(group, index, unit);
                    let accepted = agree(
                        [
                            0x46554e4348554e49,
                            group as u64,
                            index as u64,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        valid.is_ok(),
                        context,
                    );
                    valid?;
                    if !accepted? {
                        return Err(ChannelError::Configuration(
                            "function-head peer rejected leased unit",
                        ));
                    }
                    unit.project_parallel_guarded(hidden, parallel, context, &mut agree)
                },
                hook,
            )
            .map(|(output, _)| output)
            .map_err(runtime_error)
    }

    /// Projects both vocabulary heads from one retained normalized backbone result.
    pub fn project_channels_parallel(
        &mut self,
        normalized: &B::Tensor,
        backbone: &mut StaticModules<B>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DualChannelLogits<B::Tensor>, ChannelError> {
        self.runtime
            .architecture()
            .validate_parallel_projection(normalized, parallel)?;
        let agent = backbone.project_normalized_parallel(normalized, parallel, context)?;
        let function = self.project_parallel(normalized, parallel, context)?;
        validate_logits(&agent, &function, self.dimensions().vocabulary)?;
        Ok(DualChannelLogits { agent, function })
    }
}

impl<B, P> ParallelFunctionHeadExecution<B> for LayerwiseFunctionHead<B, P>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, FunctionHead<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn validate_parallel_projection(
        &self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
    ) -> Result<(), ChannelError> {
        self.runtime
            .architecture()
            .validate_parallel_projection(normalized, parallel)
    }
    fn project_parallel(
        &mut self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        LayerwiseFunctionHead::project_parallel(self, normalized, parallel, context)
    }
}
