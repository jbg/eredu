//! Independent function-vocabulary projection over shared bounded execution.
use super::channels::{
    agree_parallel_projection, compute_type, head_spec, validate_logits, ChannelError,
    DualChannelLogits, FunctionHead,
};
use crate::decoder::StaticModules;
use eredu_core::BackendFailure;
use eredu_nn::{
    DistributedNeuralBackend, LinearFormatSpec, NeuralBackend, Tensor, VocabularyParallelRange,
};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ComponentAdapter, ComponentDefinition, ComponentEquations,
    ComponentKernel, ComponentUnit, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddress,
    ExecutionUnitLayout, LayeredForwardState, LayeredInvocation, LayeredTraversalHook,
    LayerwisePolicy, LayerwiseRuntime, LocalModelLayout, MemberSharding, OwnedParameterGroupSpec,
    ParallelComponentKernel, ParallelLayeredInvocation, ParameterGroupOwner, ParameterGroupSpec,
    ParameterMemberSpec, ParameterRole, StatelessComponentState, SubmissionBackend,
    TensorPlacement,
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
    /// Projects both vocabulary heads from one retained normalized backbone result.
    fn project_channels_parallel(
        &mut self,
        normalized: &B::Tensor,
        backbone: &mut StaticModules<B>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DualChannelLogits<B::Tensor>, ChannelError> {
        self.validate_parallel_projection(normalized, parallel)?;
        let agent = backbone.project_normalized_parallel(normalized, parallel, context)?;
        let function = self.project_parallel(normalized, parallel, context)?;
        validate_logits(&agent, &function, self.dimensions().vocabulary)?;
        Ok(DualChannelLogits { agent, function })
    }
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
    eredu_runtime::validate_parameter_layout(
        description.groups().iter().map(|owned| owned.group()),
        layout,
        |_| {
            Ok::<_, ChannelError>(eredu_runtime::ParameterGroupSelection::BalancedAxis {
                axis: 0,
                global_extent: range.global_vocabulary,
                range: range.local.clone(),
            })
        },
    )?;
    Ok(range)
}

/// Architecture-owned independent readout and physical matrix declaration.
pub struct FunctionHeadKernel<B: NeuralBackend> {
    dimensions: FunctionHeadDimensions,
    format: LinearFormatSpec,
    parallel: Option<ParallelFunctionHeadDefinition<B>>,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> FunctionHeadKernel<B> {
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
            backend: PhantomData,
        })
    }
    /// Combines selected semantic equations with the validated component declaration.
    pub fn into_component(
        self,
    ) -> Result<ComponentAdapter<ComponentDefinition, Self>, ChannelError> {
        Ok(ComponentAdapter::new(
            function_head_component_definition(self.dimensions, &self.format)?,
            self,
        )?)
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
}
impl<B: DistributedNeuralBackend> FunctionHeadKernel<B> {
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
        let description = function_head_parameter_description(self.dimensions, &self.format)?;
        eredu_runtime::validate_tensor_parallel_layout(
            "voicechat.function_head",
            &description,
            &selected.layout,
            B::parallel_size(parallel),
            B::parallel_rank(parallel),
        )?;
        Ok(())
    }
    fn validate_parallel_unit(&self, unit: &FunctionHead<B>) -> Result<(), ChannelError> {
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

/// Validated logical invocations, independent parameter units and transport semantics.
pub fn function_head_component_definition(
    dimensions: FunctionHeadDimensions,
    format: &LinearFormatSpec,
) -> Result<ComponentDefinition, ChannelError> {
    let parameters = function_head_parameter_description(dimensions, format)?;
    let paths: Vec<String> = vec!["stt_model.function_head".into()];
    let transport = (0..parameters.graph().groups().len())
        .map(|group| {
            let _ = group;
            ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Prediction,
                first_owner_static_roles: vec![],
                last_owner_static_roles: vec![],
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }
        })
        .collect();
    Ok(ComponentDefinition::new(
        "voicechat.function_head",
        parameters,
        paths
            .into_iter()
            .map(|path| ComponentUnit { path, state: 0..0 })
            .collect(),
        transport,
        FUNCTION,
        vec![],
        StatelessComponentState::state_layout()?,
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::output_owner(0..1)]),
    )?)
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
    let group = eredu_runtime::expand_linear_format_parameter_group(logical, |_| {
        Ok(Some(format.clone()))
    })?;
    let owned = OwnedParameterGroupSpec::new(
        ParameterGroupOwner::execution_unit(
            layout.group_id(0).expect("function group").clone(),
            0,
        ),
        group,
    );
    Ok(ArchitectureParameterDescription::from_owned(&graph, &layout, [owned])?)
}

impl<B: NeuralBackend> ComponentKernel<B> for FunctionHeadKernel<B> {
    type Input<'a> = &'a B::Tensor;
    type Unit = FunctionHead<B>;
    type ForwardContext = ();
    type RetainedContextValues<'a>
        = std::iter::Empty<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = ChannelError;

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, ChannelError> {
        if state.global_layer_offset() != 0
            || *state.layout() != StatelessComponentState::state_layout()?
        {
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
    fn build_unit(
        &self,
        _address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<FunctionHead<B>, ChannelError> {
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

    fn executes_group(&self, _: usize, _: &Self::ForwardContext) -> bool {
        true
    }

    fn retained_values<'a>(
        &'a self,
        _: &'a (),
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}
impl<B: NeuralBackend> ComponentEquations<B, StatelessComponentState> for FunctionHeadKernel<B> {
    fn begin<'a>(
        &mut self,
        hidden: Self::Input<'a>,
        _: &mut StatelessComponentState,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ()>, ChannelError> {
        self.validate_replicated_projection(hidden)?;
        Ok(LayeredForwardState {
            hidden: hidden.clone(),
            context: (),
        })
    }
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut StatelessComponentState,
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
    fn execute(
        &mut self,
        _address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        _: &mut (),
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        if unit.dimensions() != self.dimensions || unit.format() != &self.format {
            return Err(ChannelError::Configuration(
                "leased function head differs from selected geometry or format",
            ));
        }
        unit.project(hidden, context)
    }
}

impl<B: DistributedNeuralBackend> ParallelComponentKernel<B, eredu_runtime::StatelessComponentState>
    for FunctionHeadKernel<B>
{
    fn begin_parallel<'a>(
        &mut self,
        hidden: Self::Input<'a>,
        _: &mut StatelessComponentState,
        parallel: &B::ParallelContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ()>, ChannelError> {
        self.validate_parallel_projection(hidden, parallel)?;
        Ok(LayeredForwardState {
            hidden: hidden.clone(),
            context: (),
        })
    }
    fn execute_parallel(
        &mut self,
        _address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        _: &mut (),
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.validate_parallel_unit(unit)?;
        unit.project_parallel(hidden, parallel, context)
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

impl<B, P> FunctionHeadExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, FunctionHeadKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, FunctionHead<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> FunctionHeadDimensions {
        self.architecture().kernel().dimensions()
    }
    fn validate_projection(&self, normalized: &B::Tensor) -> Result<(), ChannelError> {
        self.architecture()
            .kernel()
            .validate_replicated_projection(normalized)
    }
    fn project(
        &mut self,
        normalized: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        let mut state = StatelessComponentState::new()?;
        self.invoke(normalized, &mut state, context)
            .map_err(runtime_error)
    }
}

/// Cohort-guarded function readout through ordinary provider and completion hooks.
/// This capability requires explicit agreement in addition to tensor projection.
pub trait GuardedFunctionHeadExecution<B: DistributedNeuralBackend>:
    ParallelFunctionHeadExecution<B>
{
    /// Coordinates input admission, provider lifecycle and leased ownership before
    /// the independently selected function-vocabulary collective.
    fn project_parallel_guarded<H, F>(
        &mut self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        agree: F,
    ) -> Result<B::Tensor, ChannelError>
    where
        H: LayeredTraversalHook<B, (), ChannelError> + ?Sized,
        F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, ChannelError>;
}

impl<B, P> GuardedFunctionHeadExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, FunctionHeadKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, FunctionHead<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Coordinates input admission, provider lifecycle and leased ownership before
    /// the independently selected function-vocabulary collective.
    fn project_parallel_guarded<H, F>(
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
            self.architecture()
                .kernel()
                .validate_parallel_projection(normalized, parallel)?;
            Ok::<_, ChannelError>(StatelessComponentState::new()?)
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
        self.forward_parallel_with_unit_executor_and_traversal_hook(
            normalized,
            &mut state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, _, _, parallel, context| {
                let valid = architecture.kernel().validate_parallel_unit(unit);
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
}

impl<B, P> ParallelFunctionHeadExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, FunctionHeadKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >
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
        self.architecture()
            .kernel()
            .validate_parallel_projection(normalized, parallel)
    }
    fn project_parallel(
        &mut self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        let mut state = StatelessComponentState::new()?;
        self.invoke_parallel(normalized, &mut state, parallel, context)
            .map_err(runtime_error)
    }
}
