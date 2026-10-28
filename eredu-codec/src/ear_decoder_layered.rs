//! EAR decoder units over the shared resident and bounded traversal.
use crate::{
    ear_decoder::{invalid, DecoderConfig, DecoderState, DecoderUnit, DecoderUnitSpec},
    streaming::StreamPhase,
    Error,
};
use eredu_core::{cache::LayerCachePolicy, BackendFailure, LayerSchedule};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor, TensorElementType};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ComponentAdapter, ComponentDefinition, ComponentEquations,
    ComponentKernel, ComponentUnit, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddress,
    ExecutionUnitLayout, LayeredForwardState, LayeredInvocation, LayerwisePolicy, LayerwiseRuntime,
    LocalModelLayout, MemberSharding, OwnedParameterGroupSpec, ParallelComponentKernel,
    ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec, ParameterRole, RuntimeState,
    StateError, StateLayout, SubmissionBackend,
};
use std::marker::PhantomData;

use eredu_core::cache::{
    StateResidencyClass, StateTensorDimension, StateTensorDtype, StateTensorPolicy,
    StateTensorPresence, StateTensorRole,
};
use std::num::NonZeroU32;
/// Stable execution group for independently leased EAR decoder units.
pub const DECODER_GROUP: &str = "ear.decoder";
/// Exact local range within one unit's declared partition axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderUnitLocalDimensions {
    /// First global partition coordinate.
    pub start: i32,
    /// Locally materialized partition coordinates.
    pub width: i32,
}
/// Independent output or intermediate partitions in decoder execution order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderLocalDimensions {
    /// Upsample per-group output, block intermediate, and spectral output ranges.
    pub units: Vec<DecoderUnitLocalDimensions>,
}
/// Validates the complete physical declaration including replicated companions.
pub fn decoder_local_dimensions(
    config: &DecoderConfig,
    prefix: &str,
    layout: &LocalModelLayout,
) -> Result<DecoderLocalDimensions, Error> {
    let description = decoder_parameter_description(config, prefix)?;
    let mut units = Vec::new();
    eredu_runtime::validate_parameter_layout(
        description.groups().iter().map(|owned| owned.group()),
        layout,
        |group| -> Result<_, Error> {
            let range = eredu_runtime::logical_parameter_group_range(group, layout)?;
            units.push(DecoderUnitLocalDimensions {
                start: range.start as i32,
                width: range.len() as i32,
            });
            Ok(eredu_runtime::ParameterGroupSelection::Logical(Some(range)))
        },
    )?;
    Ok(DecoderLocalDimensions { units })
}
/// Portable decoder execution; complete-stream and controlled steps share one transition.
pub trait DecoderExecution<B: NeuralBackend> {
    /// Exact selected geometry.
    fn config(&self) -> &DecoderConfig;
    /// Executes ordered learned units into an unpublished candidate.
    fn step(
        &mut self,
        latent: &B::Tensor,
        parent: &DecoderState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DecoderState<B::Tensor>), Error>;
    /// Creates empty carry using configuration alone.
    fn initial_state(&self) -> Result<DecoderState<B::Tensor>, Error> {
        initial_state(self.config())
    }
    /// Executes ordinary decoding through the same streamed state transition.
    fn forward(
        &mut self,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.step(latent, &self.initial_state()?, context)
            .map(|(output, _)| output)
    }
    /// Releases neural carry without accessing parameters or emitting an artificial tail.
    fn finish(&self, parent: &DecoderState<B::Tensor>) -> Result<DecoderState<B::Tensor>, Error> {
        validate_state(self.config(), parent)?;
        let mut state = initial_state(self.config())?;
        state.phase = StreamPhase::Finished;
        state.batch = parent.batch;
        for (unit, previous) in state.units.iter_mut().zip(&parent.units) {
            unit.phase = StreamPhase::Finished;
            unit.batch = previous.batch;
        }
        Ok(state)
    }
}
fn initial_state<T: Tensor>(config: &DecoderConfig) -> Result<DecoderState<T>, Error> {
    Ok(DecoderState {
        units: config
            .unit_specs()?
            .into_iter()
            .map(DecoderUnitSpec::initial_state)
            .collect::<Result<_, _>>()?,
        batch: None,
        phase: StreamPhase::AcceptingInput,
    })
}
fn validate_state<T: Tensor>(config: &DecoderConfig, state: &DecoderState<T>) -> Result<(), Error> {
    let specs = config.unit_specs()?;
    if state.units.len() != specs.len()
        || state
            .units
            .iter()
            .zip(specs)
            .any(|(state, spec)| state.spec != spec)
    {
        return Err(invalid("EAR decoder state belongs to incompatible units"));
    }
    Ok(())
}
/// Independently owned candidate caches with exact per-unit retained roots.
#[derive(Clone)]
pub struct DecoderRuntimeState<T: Tensor> {
    pub(crate) layout: StateLayout,
    pub(crate) value: DecoderState<T>,
    pub(crate) partition: Option<crate::ear_decoder_partition::DecoderPartitionStateIdentity>,
}
impl<T: Tensor> DecoderRuntimeState<T> {
    /// Copies immutable carry handles into an unpublished mutable state owner.
    pub fn from_state(config: &DecoderConfig, parent: &DecoderState<T>) -> Result<Self, Error> {
        validate_state(config, parent)?;
        Ok(Self {
            layout: decoder_state_layout(config)?,
            value: parent.clone(),
            partition: None,
        })
    }
    /// Returns the candidate only after the invocation establishes completion.
    pub fn into_state(self) -> DecoderState<T> {
        self.value
    }
}
impl<B: NeuralBackend> RuntimeState<B> for DecoderRuntimeState<B::Tensor> {
    type RetainedValues<'a>
        = std::option::IntoIter<&'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    fn layout(&self) -> &StateLayout {
        &self.layout
    }
    fn retained_values(
        &self,
        ordinal: usize,
        _: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        let unit = self
            .value
            .units
            .get(ordinal)
            .ok_or(StateError::UnknownLayer {
                layer: ordinal,
                count: self.value.units.len(),
            })?;
        Ok(unit.carry().into_iter())
    }
}
/// Fixed causal carry geometry, independent of learned parameter residency.
pub fn decoder_state_layout(config: &DecoderConfig) -> Result<StateLayout, Error> {
    let specs = config.unit_specs()?;
    let policies = specs
        .iter()
        .map(|spec| match *spec {
            DecoderUnitSpec::Block { width, kernel, .. } if kernel > 1 => {
                LayerCachePolicy::FixedState {
                    tensors: vec![StateTensorPolicy {
                        role: StateTensorRole::Convolution { slot: 0 },
                        shape: vec![
                            StateTensorDimension::Batch,
                            StateTensorDimension::Fixed(
                                NonZeroU32::new(width as u32).expect("validated width"),
                            ),
                            StateTensorDimension::Fixed(
                                NonZeroU32::new((kernel - 1) as u32).expect("positive carry"),
                            ),
                        ],
                        dtype: StateTensorDtype::Float32,
                        residency: StateResidencyClass::AlwaysDeviceMutable,
                        presence: StateTensorPresence::Optional,
                    }],
                }
            }
            _ => LayerCachePolicy::NoState,
        })
        .collect();
    Ok(StateLayout::new(LayerSchedule::new(
        specs.len(),
        policies,
    )?)?)
}
/// Streaming neural decoder operations, independent leases and per-unit carry identities.
pub fn decoder_component_definition(
    config: &DecoderConfig,
    prefix: &str,
) -> Result<ComponentDefinition, Error> {
    let parameters = decoder_parameter_description(config, prefix)?;
    let count = parameters.unit_layout().len();
    let units = (0..count)
        .map(|index| ComponentUnit {
            path: format!("{prefix}.layers.{index}"),
            state: index..index + 1,
        })
        .collect();
    Ok(ComponentDefinition::new(
        "ear.decoder",
        parameters,
        units,
        vec![ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::AudioDecoder,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        DECODER_GROUP,
        vec![],
        decoder_state_layout(config)?,
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..count,
        )]),
    )?)
}
/// Original latent request geometry and canonical unit progress for one transition.
#[derive(Debug, Clone)]
pub struct DecoderForward {
    pub(crate) batch: i32,
    pub(crate) latent_frames: i32,
    pub(crate) next_unit: usize,
}
/// Codec-owned canonical unit construction and causal state geometry.
pub struct DecoderKernel<B: NeuralBackend> {
    pub(crate) config: DecoderConfig,
    pub(crate) specs: Vec<DecoderUnitSpec>,
    pub(crate) prefix: String,
    pub(crate) partition: Option<crate::ear_decoder_partition::DecoderPartition>,
    parallel: Option<(DecoderLocalDimensions, LocalModelLayout)>,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> DecoderKernel<B> {
    /// Validates the complete decoder before constructing native resources.
    pub fn new(config: DecoderConfig, prefix: &str) -> Result<Self, Error> {
        let specs = config.unit_specs()?;
        if prefix.is_empty() {
            return Err(invalid("empty EAR decoder parameter prefix"));
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
        config: DecoderConfig,
        prefix: &str,
        layout: &LocalModelLayout,
    ) -> Result<Self, Error> {
        let local = decoder_local_dimensions(&config, prefix, layout)?;
        let mut value = Self::new(config, prefix)?;
        value.parallel = Some((local, layout.clone()));
        Ok(value)
    }
    /// Retains the validated decoder operations and their shared execution adapter.
    pub fn into_component(self) -> Result<ComponentAdapter<ComponentDefinition, Self>, Error> {
        Ok(ComponentAdapter::new(
            decoder_component_definition(&self.config, &self.prefix)?,
            self,
        )?)
    }
    /// Whether this decoder requires tensor-parallel invocation.
    pub fn is_parallel(&self) -> bool {
        self.parallel.is_some()
    }
    fn local_unit(&self, index: usize) -> Option<DecoderUnitLocalDimensions> {
        self.parallel.as_ref().map(|(local, _)| local.units[index])
    }
    /// Selected decoder geometry.
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }
}
impl<B: NeuralBackend> DecoderKernel<B> {
    fn begin_input(
        &self,
        latent: &B::Tensor,
        state: &mut DecoderRuntimeState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, DecoderForward>, Error> {
        validate_state(&self.config, &state.value)?;
        if state.partition.is_some() || state.layout != decoder_state_layout(&self.config)? {
            return Err(invalid("EAR decoder runtime layout mismatch"));
        }
        if state.value.phase == StreamPhase::Finished {
            return Err(Error::StreamFinished);
        }
        if !matches!(latent.shape(),[batch,time,channels]if *batch>0&&*time>0&&*channels==self.config.latent_size)
            || state
                .value
                .batch
                .is_some_and(|batch| batch != latent.dim(0))
        {
            return Err(invalid(
                "EAR decoder expects nonempty [batch, time, latent] with unchanged batch",
            ));
        }
        if latent.element_type() != Some(TensorElementType::F32) {
            return Err(invalid(
                "EAR codec neural execution requires F32 input and parameters",
            ));
        }
        self.config
            .rates
            .iter()
            .try_fold(latent.dim(1), |length, rate| length.checked_mul(*rate))
            .ok_or_else(|| invalid("EAR decoder spectral frame count overflows"))?;
        let hidden = latent.swap_axes(1, 2, context)?;
        state.value.batch = Some(latent.dim(0));
        Ok(LayeredForwardState {
            hidden,
            context: DecoderForward {
                batch: latent.dim(0),
                latent_frames: latent.dim(1),
                next_unit: 0,
            },
        })
    }
}
fn decoder_graph() -> Result<ExecutionGraph, Error> {
    Ok(ExecutionGraph::new(
        vec![ExecutionGroupSpec::root(DECODER_GROUP)],
        DECODER_GROUP,
    )?)
}
/// Cold canonical convolution layouts and exact independent unit ownership.
pub fn decoder_parameter_description(
    config: &DecoderConfig,
    prefix: &str,
) -> Result<ArchitectureParameterDescription, Error> {
    let specs = config.unit_specs()?;
    if prefix.is_empty() {
        return Err(invalid("empty EAR decoder parameter prefix"));
    }
    let graph = decoder_graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [specs.len()])?;
    let mut owned = Vec::with_capacity(specs.len());
    for spec in specs {
        let index = spec.layer();
        let name = format!("{prefix}.layers.{index}");
        let member = |suffix: &str, shape: Vec<usize>, axis: Option<usize>| {
            ParameterMemberSpec::new(
                format!("{name}.{suffix}"),
                shape,
                axis.map_or(MemberSharding::Replicated, |axis| {
                    MemberSharding::Partitioned { axis }
                }),
            )
        };
        let members = match spec {
            DecoderUnitSpec::Upsample {
                input,
                output,
                rate,
                groups,
                ..
            } => vec![member(
                "weight",
                vec![input as usize, (output / groups) as usize, rate as usize],
                Some(1),
            )],
            DecoderUnitSpec::Output { input, output, .. } => {
                vec![member(
                    "weight",
                    vec![output as usize, input as usize, 1],
                    Some(0),
                )]
            }
            DecoderUnitSpec::Block {
                width,
                intermediate,
                kernel,
                ..
            } => {
                let w = width as usize;
                let h = intermediate as usize;
                vec![
                    member("dwconv.weight", vec![w, 1, kernel as usize], None),
                    member("dwconv.bias", vec![w], None),
                    member("norm.weight", vec![w], None),
                    member("norm.bias", vec![w], None),
                    member("pwconv1.weight", vec![h, w, 1], Some(0)),
                    member("pwconv1.bias", vec![h], Some(0)),
                    member("pwconv2.weight", vec![w, h, 1], Some(1)),
                    member("pwconv2.bias", vec![w], None),
                ]
            }
        };
        let (size, role) = match spec {
            DecoderUnitSpec::Upsample { output, groups, .. } => {
                (output / groups, ParameterRole::Channels)
            }
            DecoderUnitSpec::Block { intermediate, .. } => {
                (intermediate, ParameterRole::FeedForwardIntermediate)
            }
            DecoderUnitSpec::Output { output, .. } => (output, ParameterRole::Channels),
        };
        let group = ParameterGroupSpec::partitioned(&name, role, size as usize, members)?;
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(
                layout.group_id(0).expect("decoder group").clone(),
                index,
            ),
            group,
        ));
    }
    Ok(ArchitectureParameterDescription::from_owned(&graph, &layout, owned)?)
}
impl<B: NeuralBackend> ComponentKernel<B> for DecoderKernel<B> {
    type Input<'a> = &'a B::Tensor;
    type Unit = DecoderUnit<B::Tensor>;
    type ForwardContext = DecoderForward;
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
        let range = state.global_layers();
        if range.is_empty()
            || range.end > self.specs.len()
            || self
                .partition
                .as_ref()
                .is_some_and(|selected| selected.range != range)
            || *state.layout() != decoder_state_layout(&self.config)?.slice(range.clone())?
        {
            return Err(invalid("EAR decoder state identity layout mismatch"));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "ear_codec",
            "decoder",
            format!("{:?}:{}", self.config, self.prefix),
            self.specs.len(),
            range.start,
            0,
            topology,
        )?)
    }
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DecoderUnit<B::Tensor>, Error> {
        let index = address.index();
        DecoderUnit::new_selected(
            self.specs[index],
            &self.prefix,
            self.local_unit(index),
            context,
        )
    }
    fn executes_group(&self, _: usize, _: &DecoderForward) -> bool {
        true
    }
    fn retained_values<'a>(
        &'a self,
        _: &'a DecoderForward,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}
impl<B: NeuralBackend> ComponentEquations<B, DecoderRuntimeState<B::Tensor>> for DecoderKernel<B> {
    fn begin<'a>(
        &mut self,
        latent: Self::Input<'a>,
        state: &mut DecoderRuntimeState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, DecoderForward>, Error> {
        if self.is_parallel() || self.partition.is_some() {
            return Err(invalid(
                "selected decoder requires its parallel or partition execution path",
            ));
        }
        self.begin_input(latent, state, context)
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut DecoderRuntimeState<B::Tensor>,
        forward: &mut DecoderForward,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let index = address.index();
        if index != forward.next_unit {
            return Err(invalid(
                "decoder units must execute in canonical invocation order",
            ));
        }
        let local = self.state_index(state, index)?;
        if unit.spec() != self.specs[index]
            || unit.prefix() != self.prefix
            || unit.local() != self.local_unit(index)
        {
            return Err(invalid(
                "leased EAR decoder unit differs from selected geometry or namespace",
            ));
        }
        let (output, next) = unit.step(hidden, &state.value.units[local], context)?;
        state.value.units[local] = next;
        forward.next_unit = index + 1;
        Ok(output)
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _: &mut DecoderRuntimeState<B::Tensor>,
        forward: &DecoderForward,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if forward.next_unit != self.specs.len() {
            return Err(invalid("decoder invocation did not finish every unit"));
        }
        Ok(hidden.clone())
    }
}
impl<B: DistributedNeuralBackend> DecoderKernel<B> {
    /// Checks the actual communicator against all selected unit placements before acquisition.
    pub fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), Error> {
        let (_, selected) = self
            .parallel
            .as_ref()
            .ok_or_else(|| invalid("parallel decoder requires rank-local construction"))?;
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(B::parallel_size(parallel), 1, 1, 1)
                .map_err(|_| invalid("invalid decoder tensor degree"))?,
            B::parallel_rank(parallel),
        )
        .map_err(|_| invalid("invalid decoder tensor rank"))?;
        let expected = eredu_runtime::derive_partitioned_local_layout(
            &decoder_parameter_description(&self.config, &self.prefix)?,
            topology,
        )
        .map_err(|_| invalid("decoder tensor placement cannot be derived"))?;
        if selected != &expected {
            return Err(invalid(
                "decoder layout differs from its tensor communicator",
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
            DecoderUnitSpec::Upsample { groups, output, .. } => (groups, output / groups),
            DecoderUnitSpec::Output { output, .. } => (1, output),
            _ => return Err(invalid("decoder residual block does not gather channels")),
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
impl<B: DistributedNeuralBackend>
    ParallelComponentKernel<B, crate::ear_decoder_layered::DecoderRuntimeState<B::Tensor>>
    for DecoderKernel<B>
{
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut DecoderRuntimeState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, DecoderForward>, Error> {
        if self.partition.is_some() {
            return Err(invalid("partitioned decoder requires pipeline execution"));
        }
        self.validate_parallel(parallel)?;
        self.begin_input(input, state, context)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut DecoderRuntimeState<B::Tensor>,
        forward: &mut DecoderForward,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let index = address.index();
        if index != forward.next_unit {
            return Err(invalid(
                "decoder units must execute in canonical invocation order",
            ));
        }
        let local = self.state_index(state, index)?;
        if unit.spec() != self.specs[index]
            || unit.prefix() != self.prefix
            || unit.local() != self.local_unit(index)
        {
            return Err(invalid(
                "leased EAR decoder unit differs from selected placement",
            ));
        }
        let (output, next) = unit.step_selected(
            hidden,
            &state.value.units[local],
            context,
            |value| Ok(B::sum_parallel(value, parallel, context)?),
            |value| self.gather_unit(index, value, parallel, context),
        )?;
        state.value.units[local] = next;
        forward.next_unit = index + 1;
        Ok(output)
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

impl<B, P> DecoderExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, DecoderKernel<B>>,
        B,
        DecoderRuntimeState<B::Tensor>,
        P,
    >
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, DecoderUnit<B::Tensor>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn config(&self) -> &DecoderConfig {
        self.architecture().kernel().config()
    }
    fn step(
        &mut self,
        latent: &B::Tensor,
        parent: &DecoderState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DecoderState<B::Tensor>), Error> {
        let mut state = DecoderRuntimeState::from_state(self.config(), parent)?;
        let output = self
            .invoke(latent, &mut state, context)
            .map_err(runtime_error)?;
        Ok((output, state.into_state()))
    }
}

/// Tensor-parallel neural decoding with caller-owned immutable parent carry.
/// Implementors publish output and candidate carry only through ordinary completion.
pub trait ParallelDecoderExecution<B: DistributedNeuralBackend>: DecoderExecution<B> {
    /// Executes ordered local parameters into a complete unpublished candidate.
    fn step_parallel(
        &mut self,
        latent: &B::Tensor,
        parent: &DecoderState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DecoderState<B::Tensor>), Error>;
}

impl<B, P> ParallelDecoderExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, DecoderKernel<B>>,
        B,
        DecoderRuntimeState<B::Tensor>,
        P,
    >
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, DecoderUnit<B::Tensor>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Executes the same ordered transition with local parameters and global candidate carry.
    fn step_parallel(
        &mut self,
        latent: &B::Tensor,
        parent: &DecoderState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DecoderState<B::Tensor>), Error> {
        let mut state = DecoderRuntimeState::from_state(self.config(), parent)?;
        let output = self
            .forward_parallel_with_context_hook(latent, &mut state, parallel, context, |_, _, _| {
                Ok(())
            })
            .map_err(runtime_error)?
            .0;
        Ok((output, state.into_state()))
    }
}
