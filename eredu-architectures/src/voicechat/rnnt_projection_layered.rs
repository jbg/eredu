//! RNNT embedding and joint affines without recurrent parameter ownership.
use super::{
    rnnt::{RnntDimensions, RnntError, RnntProjectionExecution},
    rnnt_layered::{
        rnnt_local_dimensions_for_description, rnnt_parameter_description, runtime_error,
        RnntArchitecture, RnntForward, RnntInput, RnntLocalDimensions, RnntRuntimeState,
        RnntStaticModules, RnntUnit,
    },
};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::*;

const GROUPS: [&str; 4] = [
    "voicechat.rnnt.embedding",
    "voicechat.rnnt.joint.enc",
    "voicechat.rnnt.joint.pred",
    "voicechat.rnnt.joint.joint_net.2",
];
fn full_group(group: usize) -> Result<usize, RnntError> {
    [0, 2, 3, 4]
        .get(group)
        .copied()
        .ok_or(RnntError::InvalidConfiguration(
            "unknown RNNT projection group",
        ))
}
fn graph() -> Result<ExecutionGraph, RnntError> {
    Ok(ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(GROUPS[0]),
            ExecutionGroupSpec::with_dependencies(GROUPS[1], [GROUPS[0]]),
            ExecutionGroupSpec::with_dependencies(GROUPS[2], [GROUPS[1]]),
            ExecutionGroupSpec::with_dependencies(GROUPS[3], [GROUPS[2]]),
        ],
        GROUPS[3],
    )?)
}
/// Exact compact four-unit ownership with no recurrent parameter declarations.
pub fn rnnt_projection_parameter_description(
    dimensions: RnntDimensions,
) -> Result<ArchitectureParameterDescription, RnntError> {
    let full = rnnt_parameter_description(dimensions)?;
    let graph = graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [1; 4])?;
    let mut specs = Vec::new();
    let mut owned = Vec::new();
    for item in full.groups() {
        let ParameterGroupOwner::ExecutionUnit {
            group, global_unit, ..
        } = item.owner()
        else {
            continue;
        };
        let Some(index) = GROUPS.iter().position(|name| *name == group.as_str()) else {
            continue;
        };
        let spec = item.group().clone();
        specs.push(spec.clone());
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(
                layout.group_id(index).unwrap().clone(),
                *global_unit,
            ),
            spec,
        ));
    }
    Ok(ArchitectureParameterDescription::new(
        &graph, &layout, specs, owned,
    )?)
}
/// Checks vocabulary slices and joint replicas in the compact projection-only topology.
pub fn rnnt_projection_local_dimensions(
    dimensions: RnntDimensions,
    layout: &LocalModelLayout,
) -> Result<RnntLocalDimensions, RnntError> {
    rnnt_local_dimensions_for_description(
        dimensions,
        layout,
        &rnnt_projection_parameter_description(dimensions)?,
    )
}
/// Canonical RNNT unit type; projection execution cannot select its recurrent variant.
pub type RnntProjectionUnit<B> = RnntUnit<B>;
/// Stateless phase traversal metadata, independent of externally owned LSTM history.
pub type RnntProjectionRuntimeState = RnntRuntimeState;
/// One nonrecurrent phase; no predictor request can enter this graph.
pub enum RnntProjectionInput<'a, T> {
    /// Global vocabulary token, including blank.
    Embed(usize),
    /// Unprojected perception row.
    Encode(&'a T),
    /// Published final predictor readout.
    Prediction(&'a T),
    /// Rectified joint hidden row.
    Output(&'a T),
}
impl<'a, T> RnntProjectionInput<'a, T> {
    fn into_full(self) -> RnntInput<'a, T> {
        match self {
            Self::Embed(token) => RnntInput::Embed(token),
            Self::Encode(input) => RnntInput::Encode(input),
            Self::Prediction(input) => RnntInput::Prediction(input),
            Self::Output(input) => RnntInput::Output(input),
        }
    }
}
/// Compact phase graph that reuses the ordinary RNNT embedding/affine equations.
pub struct RnntProjectionArchitecture<B: NeuralBackend> {
    inner: RnntArchitecture<B>,
}
impl<B: NeuralBackend> RnntProjectionArchitecture<B> {
    /// Declares only embedding and joint phases without constructing native slots.
    pub fn new(dimensions: RnntDimensions) -> Result<Self, RnntError> {
        Ok(Self {
            inner: RnntArchitecture::new(dimensions)?,
        })
    }
    /// Exact RNNT global geometry.
    pub fn dimensions(&self) -> RnntDimensions {
        self.inner.dimensions()
    }
    /// Whether primitive execution requires a matching tensor collective context.
    pub fn is_parallel(&self) -> bool {
        self.inner.is_parallel()
    }
    /// Compact embedding/encoder/prediction/output phase graph.
    pub fn execution_graph(&self) -> Result<ExecutionGraph, RnntError> {
        graph()
    }
    /// Exactly one independently leased unit per projection phase.
    pub fn group_unit_count(&self, group: usize) -> Result<usize, RnntError> {
        full_group(group)?;
        Ok(1)
    }
    /// Canonical released checkpoint path for one compact unit.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, RnntError> {
        if index != 0 {
            return Err(RnntError::InvalidConfiguration(
                "RNNT projection unit exceeds phase",
            ));
        }
        self.inner.unit_path(full_group(group)?, index)
    }
    /// Constructs only the selected embedding or affine using shared RNNT construction.
    pub fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntProjectionUnit<B>, RnntError> {
        if index != 0 {
            return Err(RnntError::InvalidConfiguration(
                "RNNT projection unit exceeds phase",
            ));
        }
        self.inner.build_unit(full_group(group)?, index, context)
    }
}
impl<B: DistributedNeuralBackend> RnntProjectionArchitecture<B> {
    /// Retains exact projection-only vocabulary placement and replicated joint affines.
    pub fn new_parallel(
        dimensions: RnntDimensions,
        layout: &LocalModelLayout,
    ) -> Result<Self, RnntError> {
        Ok(Self {
            inner: RnntArchitecture::new_parallel_for_description(
                dimensions,
                layout,
                rnnt_projection_parameter_description(dimensions)?,
            )?,
        })
    }
    /// Validates the communicator before acquiring any learned unit.
    pub fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), RnntError> {
        self.inner.validate_parallel(parallel)
    }
}
impl<B: NeuralBackend> ArchitectureParameters<B> for RnntProjectionArchitecture<B> {
    type DefinitionError = RnntError;
    fn state_layout(&self) -> Result<StateLayout, RnntError> {
        self.inner.state_layout()
    }
    fn state_identity(
        &self,
        state: &PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<ModelStateIdentity, RnntError> {
        if state.global_layer_offset() != 0 || state.layout() != &self.state_layout()? {
            return Err(RnntError::InvalidState(
                "RNNT projection traversal state differs",
            ));
        }
        Ok(ModelStateIdentity::new(
            "nemotron_voicechat",
            "rnnt_projections",
            format!("{:?}", self.dimensions()),
            1,
            0,
            0,
            topology,
        )?)
    }
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, RnntError> {
        rnnt_projection_parameter_description(self.dimensions())
    }
    fn visit_static_parameters<V: StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.inner.visit_static_parameters(visitor)
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.inner.visit_static_parameters_mut(visitor)
    }
}
impl<B: NeuralBackend> LayeredArchitecture<B, RnntRuntimeState> for RnntProjectionArchitecture<B> {
    type Input<'a> = RnntProjectionInput<'a, B::Tensor>;
    type StaticModules = RnntStaticModules;
    type Unit = RnntProjectionUnit<B>;
    type ForwardContext = RnntForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = RnntError;
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        self.inner
            .group_transport(full_group(group).expect("validated projection group"))
    }
    fn primary_execution_group(&self) -> &str {
        GROUPS[3]
    }
    fn state_partition_plan(&self, _: &StateLayout) -> ArchitectureStatePartitionPlan {
        ArchitectureStatePartitionPlan::new([])
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, RnntError> {
        graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, RnntError> {
        self.group_unit_count(group)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, RnntError> {
        self.unit_path(group, index)
    }
    fn static_modules(&self) -> &RnntStaticModules {
        self.inner.static_modules()
    }
    fn static_modules_mut(&mut self) -> &mut RnntStaticModules {
        self.inner.static_modules_mut()
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, RnntError> {
        self.build_unit(group, index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut RnntRuntimeState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError> {
        self.inner.begin_forward(input.into_full(), state, context)
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut RnntRuntimeState,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.inner.begin_execution_group(
            full_group(group)?,
            initial,
            dependencies,
            state,
            forward,
            context,
        )
    }
    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        full_group(group).is_ok_and(|group| self.inner.should_execute_group(group, forward))
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
        state: &mut RnntRuntimeState,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.inner.forward_unit(
            full_group(group)?,
            index,
            unit,
            hidden,
            state,
            forward,
            context,
        )
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut RnntRuntimeState,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.inner.finish_forward(hidden, state, forward, context)
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        group: usize,
        index: usize,
    ) -> Self::RetainedContextValues<'a> {
        self.inner.retained_context_values(
            forward,
            full_group(group).expect("validated projection group"),
            index,
        )
    }
}
impl<B: DistributedNeuralBackend> ParallelLayeredArchitecture<B, RnntRuntimeState>
    for RnntProjectionArchitecture<B>
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut RnntRuntimeState,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError> {
        self.inner
            .begin_forward_parallel(input.into_full(), state, parallel, context)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut RnntRuntimeState,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.inner.forward_unit_parallel(
            full_group(group)?,
            index,
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
        )
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut RnntRuntimeState,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.inner
            .finish_forward_parallel(hidden, state, forward, parallel, context)
    }
}
/// Ordinary bounded or resident projection phases over the shared unit lifecycle.
pub struct LayerwiseRnntProjection<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntProjectionUnit<B>>,
{
    runtime: LayerwiseRuntime<RnntProjectionArchitecture<B>, B, RnntRuntimeState, P>,
}
impl<B, P> LayerwiseRnntProjection<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntProjectionUnit<B>>,
    P::Error: std::fmt::Display,
{
    /// Adopts the compact projection graph and its exact parameter provider.
    pub fn new(architecture: RnntProjectionArchitecture<B>, policy: P) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Retained shared runtime and cold parameter ownership.
    pub fn runtime(
        &self,
    ) -> &LayerwiseRuntime<RnntProjectionArchitecture<B>, B, RnntRuntimeState, P> {
        &self.runtime
    }
    /// Shared provider inspection and fault-recovery surface.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<RnntProjectionArchitecture<B>, B, RnntRuntimeState, P> {
        &mut self.runtime
    }
}
impl<B, P> RnntProjectionExecution<B> for LayerwiseRnntProjection<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntProjectionUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> RnntDimensions {
        self.runtime.architecture().dimensions()
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntProjectionInput::Embed(token),
                &mut RnntRuntimeState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
    fn encode_frame(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntProjectionInput::Encode(input),
                &mut RnntRuntimeState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
    fn project_prediction(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntProjectionInput::Prediction(input),
                &mut RnntRuntimeState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
    fn project_output(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntProjectionInput::Output(input),
                &mut RnntRuntimeState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
}
impl<B, P> LayerwiseRnntProjection<B, P>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntProjectionUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn invoke_parallel(
        &mut self,
        input: RnntProjectionInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .forward_parallel_with_context_hook(
                input,
                &mut RnntRuntimeState::new()?,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map(|value| value.0)
            .map_err(runtime_error)
    }
    /// Runs a projection through shared traversal with cohort failure agreement.
    /// The hook participates before parameter acquisition and collective execution.
    pub fn invoke_parallel_with_hook<H>(
        &mut self,
        input: RnntProjectionInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> Result<B::Tensor, RnntError>
    where
        H: LayeredTraversalHook<B, RnntForward<B::Tensor>, RnntError> + ?Sized,
    {
        self.runtime
            .forward_parallel_with_unit_executor_and_traversal_hook(
                input,
                &mut RnntRuntimeState::new()?,
                parallel,
                context,
                |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                    architecture.forward_unit_parallel(
                        group, index, unit, hidden, state, forward, parallel, context,
                    )
                },
                hook,
            )
            .map(|value| value.0)
            .map_err(runtime_error)
    }
    /// Vocabulary-sharded lookup with complete output on every cohort participant.
    pub fn embed_token_parallel(
        &mut self,
        token: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Embed(token), parallel, context)
    }
    /// Replicated perception projection using the selected cohort's exact authority.
    pub fn encode_frame_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Encode(input), parallel, context)
    }
    /// Replicated predictor affine over the authoritative published readout.
    pub fn project_prediction_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Prediction(input), parallel, context)
    }
    /// Vocabulary-sharded affine and exact row-order gather.
    pub fn project_output_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntProjectionInput::Output(input), parallel, context)
    }
}
