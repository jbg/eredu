//! Finite EAR encoder units over the shared resident and bounded traversal.
use crate::{
    ear_encoder::{invalid, EncoderConfig, EncoderUnit, EncoderUnitSpec},
    Error,
};
use eredu_core::BackendFailure;
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor, TensorElementType};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ComponentAdapter, ComponentDefinition, ComponentEquations, ComponentKernel, ComponentUnit,
    ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddress, ExecutionUnitLayout,
    LayeredForwardState, LayeredInvocation, LayerwisePolicy, LayerwiseRuntime, LocalModelLayout,
    LocalTensorLayout, MemberSharding, OwnedParameterGroupSpec, ParallelComponentKernel,
    ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
    StatelessComponentState, SubmissionBackend, TensorPlacement,
};
use std::marker::PhantomData;

const ENCODER: &str = "ear.encoder";
/// Exact local range within one unit's declared partition axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderUnitLocalDimensions {
    /// First global partition coordinate.
    pub start: i32,
    /// Locally materialized partition coordinates.
    pub width: i32,
}
/// Independent output or intermediate partitions in encoder execution order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderLocalDimensions {
    /// Input projection output, block intermediate, and downsample per-group output ranges.
    pub units: Vec<EncoderUnitLocalDimensions>,
}
/// Validates the complete physical declaration including replicated companions.
pub fn encoder_local_dimensions(
    config: &EncoderConfig,
    prefix: &str,
    layout: &LocalModelLayout,
) -> Result<EncoderLocalDimensions, Error> {
    let description = encoder_parameter_description(config, prefix)?;
    let mut expected = LocalModelLayout::default();
    let mut units = Vec::new();
    for owned in description.groups() {
        let group = owned.group();
        let first = group
            .members()
            .iter()
            .find(|member| {
                matches!(
                    member.sharding(),
                    MemberSharding::Partitioned { .. } | MemberSharding::PartitionedSegments { .. }
                )
            })
            .ok_or_else(|| invalid("encoder unit has no partition declaration"))?;
        let size = group
            .partition_units()
            .ok_or_else(|| invalid("encoder unit omits partition units"))?;
        let range = layout
            .tensor(first.target())
            .ok_or_else(|| invalid("encoder layout omits partitioned parameter"))?
            .expanded_logical_range(size)?;
        if range.is_empty() || range.end > size {
            return Err(invalid("invalid encoder partition range"));
        }
        units.push(EncoderUnitLocalDimensions {
            start: range.start as i32,
            width: range.len() as i32,
        });
        for member in group.members() {
            let mut shape = member.global_shape().to_vec();
            let placement = match member.sharding() {
                MemberSharding::Replicated => TensorPlacement::Replicated,
                MemberSharding::Partitioned { axis } => {
                    shape[*axis] = range.len();
                    TensorPlacement::Range {
                        axis: *axis,
                        start: range.start,
                        end: range.end,
                    }
                }
                MemberSharding::PartitionedSegments { axis, segments } => {
                    let indices: Vec<_> = segments
                        .iter()
                        .flat_map(|segment| segment.start + range.start..segment.start + range.end)
                        .collect();
                    shape[*axis] = indices.len();
                    TensorPlacement::Indices {
                        axis: *axis,
                        indices,
                    }
                }
                _ => return Err(invalid("unknown encoder partition declaration")),
            };
            expected.insert(
                member.target().into(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    shape,
                    placement,
                    group.partition_units(),
                    Some(range.clone()),
                    false,
                ),
            );
        }
    }
    if layout != &expected {
        return Err(invalid(
            "encoder physical layout differs from declared partitions",
        ));
    }
    Ok(EncoderLocalDimensions { units })
}
/// Portable finite encoder execution without persistent streaming carry.
pub trait EncoderExecution<B: NeuralBackend> {
    /// Exact selected finite encoder geometry.
    fn config(&self) -> &EncoderConfig;
    /// Converts a complete spectrum to frame-major latent vectors.
    fn forward(
        &mut self,
        spectrum: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
}
/// Finite spectral encoder operations and independently leased parameter resources.
pub fn encoder_component_definition(
    config: &EncoderConfig,
    prefix: &str,
) -> Result<ComponentDefinition, Error> {
    let parameters = encoder_parameter_description(config, prefix)?;
    let units = (0..parameters.unit_layout().len())
        .map(|index| ComponentUnit {
            path: format!("{prefix}.layers.{index}"),
            state: 0..0,
        })
        .collect();
    Ok(ComponentDefinition::new(
        "ear.encoder",
        parameters,
        units,
        vec![ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::AudioEncoder,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        ENCODER,
        vec![],
        StatelessComponentState::state_layout()?,
        ArchitectureStatePartitionPlan::new([]),
    )?)
}
/// Invocation geometry and progress retained independently of learned parameter leases.
#[derive(Debug, Clone)]
pub struct EncoderForward {
    pub(crate) batch: i32,
    pub(crate) spectral_frames: i32,
    pub(crate) next_unit: usize,
}
/// Codec-owned canonical unit construction and finite execution geometry.
pub struct EncoderKernel<B: NeuralBackend> {
    pub(crate) config: EncoderConfig,
    pub(crate) specs: Vec<EncoderUnitSpec>,
    pub(crate) prefix: String,
    pub(crate) partition: Option<crate::ear_encoder_partition::EncoderPartition>,
    parallel: Option<(EncoderLocalDimensions, LocalModelLayout)>,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> EncoderKernel<B> {
    /// Validates the complete encoder before constructing native resources.
    pub fn new(config: EncoderConfig, prefix: &str) -> Result<Self, Error> {
        let specs = config.unit_specs()?;
        if prefix.is_empty() {
            return Err(invalid("empty EAR encoder parameter prefix"));
        }
        Ok(Self {
            config,
            specs,
            prefix: prefix.into(),
            partition: None,
            parallel: None,
            backend: PhantomData,
        })
    }
    /// Constructs unloaded rank-local units after exact physical layout validation.
    pub fn new_parallel(
        config: EncoderConfig,
        prefix: &str,
        layout: &LocalModelLayout,
    ) -> Result<Self, Error> {
        let local = encoder_local_dimensions(&config, prefix, layout)?;
        let mut value = Self::new(config, prefix)?;
        value.parallel = Some((local, layout.clone()));
        Ok(value)
    }
    /// Retains the validated encoder operations and their shared execution adapter.
    pub fn into_component(self) -> Result<ComponentAdapter<ComponentDefinition, Self>, Error> {
        Ok(ComponentAdapter::new(
            encoder_component_definition(&self.config, &self.prefix)?,
            self,
        )?)
    }
    /// Whether this encoder requires tensor-parallel invocation.
    pub fn is_parallel(&self) -> bool {
        self.parallel.is_some()
    }
    fn local_unit(&self, index: usize) -> Option<EncoderUnitLocalDimensions> {
        self.parallel.as_ref().map(|(local, _)| local.units[index])
    }
    /// Selected encoder geometry.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
}
impl<B: NeuralBackend> EncoderKernel<B> {
    pub(crate) fn begin_input(
        &self,
        spectrum: &B::Tensor,
    ) -> Result<LayeredForwardState<B::Tensor, EncoderForward>, Error> {
        if !matches!(spectrum.shape(),[batch,channels,time]if *batch>0&&*channels==self.config.fft_size+2&&*time>=self.config.frame_reduction()?)
        {
            return Err(invalid(
                "EAR encoder spectrum must contain at least one complete latent frame",
            ));
        }
        if spectrum.element_type() != Some(TensorElementType::F32) {
            return Err(invalid(
                "EAR codec neural execution requires F32 input and parameters",
            ));
        }
        Ok(LayeredForwardState {
            hidden: spectrum.clone(),
            context: EncoderForward {
                batch: spectrum.dim(0),
                spectral_frames: spectrum.dim(2),
                next_unit: 0,
            },
        })
    }
}
impl<B: NeuralBackend> EncoderKernel<B> {
    fn finish_unit(
        &self,
        index: usize,
        output: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if index + 1 == self.specs.len() {
            Ok(output.swap_axes(1, 2, context)?)
        } else {
            Ok(output)
        }
    }
}
fn encoder_graph() -> Result<ExecutionGraph, Error> {
    Ok(ExecutionGraph::new(
        vec![ExecutionGroupSpec::root(ENCODER)],
        ENCODER,
    )?)
}
/// Cold canonical pointwise, ConvNeXt and downsampling parameter ownership.
pub fn encoder_parameter_description(
    config: &EncoderConfig,
    prefix: &str,
) -> Result<ArchitectureParameterDescription, Error> {
    let specs = config.unit_specs()?;
    if prefix.is_empty() {
        return Err(invalid("empty EAR encoder parameter prefix"));
    }
    let graph = encoder_graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [specs.len()])?;
    let mut expected = Vec::with_capacity(specs.len());
    let mut owned = Vec::with_capacity(specs.len());
    for spec in specs {
        let index = spec.layer();
        let name = format!("{prefix}.layers.{index}");
        let member = |suffix: &str, shape: Vec<usize>, sharding: MemberSharding| {
            ParameterMemberSpec::new(format!("{name}.{suffix}"), shape, sharding)
        };
        let members = match spec {
            EncoderUnitSpec::Input { input, output, .. } => {
                vec![member(
                    "weight",
                    vec![output as usize, input as usize, 1],
                    MemberSharding::Partitioned { axis: 0 },
                )]
            }
            EncoderUnitSpec::Downsample {
                input,
                output,
                rate,
                groups,
                ..
            } => vec![member(
                "weight",
                vec![output as usize, (input / groups) as usize, rate as usize],
                MemberSharding::PartitionedSegments {
                    axis: 0,
                    segments: (0..groups)
                        .map(|group| {
                            (group * (output / groups)) as usize
                                ..((group + 1) * (output / groups)) as usize
                        })
                        .collect(),
                },
            )],
            EncoderUnitSpec::Block {
                width,
                intermediate,
                kernel,
                ..
            } => {
                let w = width as usize;
                let h = intermediate as usize;
                vec![
                    member(
                        "dwconv.weight",
                        vec![w, 1, kernel as usize],
                        MemberSharding::Replicated,
                    ),
                    member("dwconv.bias", vec![w], MemberSharding::Replicated),
                    member("norm.weight", vec![w], MemberSharding::Replicated),
                    member("norm.bias", vec![w], MemberSharding::Replicated),
                    member(
                        "pwconv1.weight",
                        vec![h, w, 1],
                        MemberSharding::Partitioned { axis: 0 },
                    ),
                    member(
                        "pwconv1.bias",
                        vec![h],
                        MemberSharding::Partitioned { axis: 0 },
                    ),
                    member(
                        "pwconv2.weight",
                        vec![w, h, 1],
                        MemberSharding::Partitioned { axis: 1 },
                    ),
                    member("pwconv2.bias", vec![w], MemberSharding::Replicated),
                ]
            }
        };
        let (size, role) = match spec {
            EncoderUnitSpec::Input { output, .. } => (output, ParameterRole::Channels),
            EncoderUnitSpec::Block { intermediate, .. } => {
                (intermediate, ParameterRole::FeedForwardIntermediate)
            }
            EncoderUnitSpec::Downsample { output, groups, .. } => {
                (output / groups, ParameterRole::Channels)
            }
        };
        let group = ParameterGroupSpec::partitioned(&name, role, size as usize, members)?;
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(
                layout.group_id(0).expect("encoder group").clone(),
                index,
            ),
            group.clone(),
        ));
        expected.push(group);
    }
    Ok(ArchitectureParameterDescription::new(
        &graph, &layout, expected, owned,
    )?)
}
impl<B: NeuralBackend> ComponentKernel<B> for EncoderKernel<B> {
    type Input<'a> = &'a B::Tensor;
    type Unit = EncoderUnit<B::Tensor>;
    type ForwardContext = EncoderForward;
    type RetainedContextValues<'a>
        = std::iter::Empty<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        if state.global_layer_offset() != 0
            || *state.layout() != StatelessComponentState::state_layout()?
        {
            return Err(invalid("EAR finite encoder stateless layout mismatch"));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "ear_codec",
            "finite_encoder",
            format!("{:?}:{}", self.config, self.prefix),
            1,
            0,
            0,
            topology,
        )?)
    }
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<EncoderUnit<B::Tensor>, Error> {
        let index = address.index();
        EncoderUnit::new_selected(
            self.specs[index],
            &self.prefix,
            self.local_unit(index),
            context,
        )
    }
    fn executes_group(&self, _: usize, _: &EncoderForward) -> bool {
        true
    }
    fn retained_values<'a>(
        &'a self,
        _: &'a EncoderForward,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}
impl<B: NeuralBackend> ComponentEquations<B, StatelessComponentState> for EncoderKernel<B> {
    fn begin<'a>(
        &mut self,
        spectrum: Self::Input<'a>,
        _: &mut StatelessComponentState,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, EncoderForward>, Error> {
        if self.is_parallel() || self.partition.is_some() {
            return Err(invalid(
                "selected encoder requires its parallel or partition execution path",
            ));
        }
        self.begin_input(spectrum)
    }
    fn begin_group(
        &mut self,
        _: usize,
        initial: &B::Tensor,
        _: &[&B::Tensor],
        _: &mut StatelessComponentState,
        _: &mut EncoderForward,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &mut EncoderForward,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let index = address.index();
        if index != forward.next_unit {
            return Err(invalid(
                "encoder units must execute in canonical invocation order",
            ));
        }
        if unit.spec() != self.specs[index]
            || unit.prefix() != self.prefix
            || unit.local() != self.local_unit(index)
        {
            return Err(invalid(
                "leased EAR encoder unit differs from selected geometry or namespace",
            ));
        }
        let output = unit.forward(hidden, context)?;
        let output = self.finish_unit(index, output, context)?;
        forward.next_unit = index + 1;
        Ok(output)
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &EncoderForward,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if forward.next_unit != self.specs.len() {
            return Err(invalid("encoder invocation did not finish every unit"));
        }
        Ok(hidden.clone())
    }
}
impl<B: DistributedNeuralBackend> EncoderKernel<B> {
    /// Checks the actual communicator against all selected unit placements before acquisition.
    pub fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), Error> {
        let (_, selected) = self
            .parallel
            .as_ref()
            .ok_or_else(|| invalid("parallel encoder requires rank-local construction"))?;
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(B::parallel_size(parallel), 1, 1, 1)
                .map_err(|_| invalid("invalid encoder tensor degree"))?,
            B::parallel_rank(parallel),
        )
        .map_err(|_| invalid("invalid encoder tensor rank"))?;
        let expected = eredu_runtime::derive_partitioned_local_layout(
            &encoder_parameter_description(&self.config, &self.prefix)?,
            topology,
        )
        .map_err(|_| invalid("encoder tensor placement cannot be derived"))?;
        if selected != &expected {
            return Err(invalid(
                "encoder layout differs from its tensor communicator",
            ));
        }
        Ok(())
    }
    fn gather_unit(
        &self,
        index: usize,
        value: B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let local = self
            .local_unit(index)
            .expect("validated parallel construction");
        let (groups, width) = match self.specs[index] {
            EncoderUnitSpec::Downsample { groups, output, .. } => (groups, output / groups),
            EncoderUnitSpec::Input { output, .. } => (1, output),
            _ => return Err(invalid("encoder residual block does not gather channels")),
        };
        let (batch, time) = (value.dim(0), value.dim(2));
        let mut parts = Vec::with_capacity(3);
        if local.start > 0 {
            parts.push(B::Tensor::full_f32(
                0.0,
                &[batch, groups, local.start, time],
                context,
            )?);
        }
        parts.push(value.reshape(&[batch, groups, local.width, time], context)?);
        let end = local.start + local.width;
        if end < width {
            parts.push(B::Tensor::full_f32(
                0.0,
                &[batch, groups, width - end, time],
                context,
            )?);
        }
        let full = B::Tensor::concatenate(&parts, 2, context)?
            .reshape(&[batch, groups * width, time], context)?;
        Ok(B::sum_parallel(full, parallel, context)?)
    }
}
impl<B: DistributedNeuralBackend> ParallelComponentKernel<B, eredu_runtime::StatelessComponentState>
    for EncoderKernel<B>
{
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut StatelessComponentState,
        parallel: &B::ParallelContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, EncoderForward>, Error> {
        if self.partition.is_some() {
            return Err(invalid("partitioned encoder requires pipeline execution"));
        }
        self.validate_parallel(parallel)?;
        self.begin_input(input)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &mut EncoderForward,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let index = address.index();
        if index != forward.next_unit {
            return Err(invalid(
                "encoder units must execute in canonical invocation order",
            ));
        }
        if unit.spec() != self.specs[index]
            || unit.prefix() != self.prefix
            || unit.local() != self.local_unit(index)
        {
            return Err(invalid(
                "leased EAR encoder unit differs from selected placement",
            ));
        }
        let output = unit.forward_selected(
            hidden,
            context,
            |value| Ok(B::sum_parallel(value, parallel, context)?),
            |value| self.gather_unit(index, value, parallel, context),
        )?;
        let output = self.finish_unit(index, output, context)?;
        forward.next_unit = index + 1;
        Ok(output)
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut StatelessComponentState,
        forward: &EncoderForward,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish(hidden, state, forward, context)
    }
}
/// Neural encoder using the existing prepared parameter provider and bounded traversal.
pub struct LayerwiseEncoder<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, EncoderUnit<B::Tensor>>,
{
    runtime: LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, EncoderKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >,
}
impl<B, P> LayerwiseEncoder<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, EncoderUnit<B::Tensor>>,
    P::Error: std::fmt::Display,
{
    /// Adopts an exact encoder unit policy.
    pub fn new(
        architecture: ComponentAdapter<ComponentDefinition, EncoderKernel<B>>,
        policy: P,
    ) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Existing shared provider/runtime for diagnostics and parameter inspection.
    pub fn runtime(
        &self,
    ) -> &LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, EncoderKernel<B>>,
        B,
        StatelessComponentState,
        P,
    > {
        &self.runtime
    }
    /// Existing shared provider/runtime for bounded parameter operations.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, EncoderKernel<B>>,
        B,
        StatelessComponentState,
        P,
    > {
        &mut self.runtime
    }
}
fn runtime_error<P: std::error::Error + Send + Sync + 'static>(
    error: eredu_runtime::LayerwiseRuntimeError<Error, P>,
) -> Error {
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

impl<B, P> EncoderExecution<B> for LayerwiseEncoder<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, EncoderUnit<B::Tensor>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn config(&self) -> &EncoderConfig {
        self.runtime.architecture().kernel().config()
    }
    fn forward(
        &mut self,
        spectrum: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let mut state = StatelessComponentState::new()?;
        self.runtime
            .invoke(spectrum, &mut state, context)
            .map_err(runtime_error)
    }
}

impl<B, P> LayerwiseEncoder<B, P>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, EncoderUnit<B::Tensor>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Executes the finite transition with rank-local parameters and full boundary activations.
    pub fn forward_parallel(
        &mut self,
        spectrum: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let mut state = StatelessComponentState::new()?;
        self.runtime
            .forward_parallel_with_context_hook(
                spectrum,
                &mut state,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map(|(output, _)| output)
            .map_err(runtime_error)
    }
}
