//! Audio units over the ordinary resident and bounded traversal contracts.

use std::{collections::BTreeMap, marker::PhantomData, num::NonZeroU32};

use eredu_core::{
    cache::{
        LayerCachePolicy, StateResidencyClass, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorPresence, StateTensorRole,
    },
    LayerSchedule,
};
use eredu_nn::{Error, NeuralBackend, Tensor};
use eredu_runtime::ComponentEquations as _;
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ExecutionGraph, ExecutionUnitLayout, LayerRuntimeState,
    LayeredForwardState, MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner,
    ParameterGroupSpec, ParameterMemberSpec, ParameterRole, RuntimeStateComponents, StateLayout,
};

use super::{AudioConfig, AudioUnit, AudioUnitKind};

/// Stable component address also used when the tower is embedded in Muse.
pub const AUDIO_EXECUTION_GROUP: &str = "audio_encoder";

fn graph() -> Result<ExecutionGraph, Error> {
    ExecutionGraph::chain([AUDIO_EXECUTION_GROUP]).map_err(Error::backend)
}

/// Exact parameter declarations for one patch, convolution, or final projection.
/// Physical quantization companions remain members of their matrix's unit.
pub fn unit_parameter_groups(
    config: &AudioConfig,
    kind: AudioUnitKind,
) -> Result<Vec<ParameterGroupSpec>, Error> {
    if !config.execution_units()?.contains(&kind) {
        return Err(Error::backend(
            "Muse audio unit is outside its configured pyramid",
        ));
    }
    let members = super::super::checkpoint::audio_tensor_layouts(config)
        .map_err(Error::backend)?
        .into_iter()
        .filter(|tensor| kind.owns_parameter(&tensor.canonical))
        .map(|tensor| {
            ParameterMemberSpec::new(tensor.canonical, tensor.shape, MemberSharding::Replicated)
        })
        .collect::<Vec<_>>();
    let group = ParameterGroupSpec::new(kind.path(), ParameterRole::Replicated, members)
        .map_err(Error::backend)?;
    eredu_runtime::expand_linear_format_parameter_groups(vec![group], |member| {
        crate::linear_format::standard_parallel_linear_format(
            member,
            config.linear_format_for(member.target()),
        )
    })
    .map_err(Error::backend)
}

/// Full-width bounded histories, aligned one-for-one with the tower's small units.
pub fn state_layout(config: &AudioConfig) -> Result<StateLayout, Error> {
    let fixed = |value: i32| {
        NonZeroU32::new(value as u32)
            .map(StateTensorDimension::Fixed)
            .ok_or_else(|| Error::backend("Muse audio state dimension must be positive"))
    };
    let layers = config
        .execution_units()?
        .into_iter()
        .map(|kind| {
            Ok(match kind {
                AudioUnitKind::Convolution { stage, .. } if config.conv_kernel_size > 1 => {
                    LayerCachePolicy::FixedState {
                        tensors: vec![StateTensorPolicy {
                            role: StateTensorRole::Convolution { slot: 0 },
                            shape: vec![
                                StateTensorDimension::Batch,
                                fixed(config.conv_kernel_size - 1)?,
                                fixed(config.hidden_sizes[stage])?,
                            ],
                            dtype: StateTensorDtype::Floating,
                            residency: StateResidencyClass::AlwaysDeviceMutable,
                            presence: StateTensorPresence::Optional,
                        }],
                    }
                }
                _ => LayerCachePolicy::NoState,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    StateLayout::new(LayerSchedule::new(layers.len(), layers).map_err(Error::backend)?)
        .map_err(Error::backend)
}

/// Complete cold ownership description; no complete-tower residency unit exists.
pub fn parameter_description(
    config: &AudioConfig,
) -> Result<ArchitectureParameterDescription, Error> {
    let graph = graph()?;
    let kinds = config.execution_units()?;
    let layout = ExecutionUnitLayout::new(&graph, [kinds.len()]).map_err(Error::backend)?;
    let owner = layout.group_id(0).expect("one audio group").clone();
    let mut owned = Vec::new();
    for (index, kind) in kinds.into_iter().enumerate() {
        for group in unit_parameter_groups(config, kind)? {
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(owner.clone(), index),
                group,
            ));
        }
    }
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned).map_err(Error::backend)
}

/// Validated waveform units, independently leased parameters, and convolution histories.
pub fn component_definition(
    config: &AudioConfig,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    let kinds = config.execution_units()?;
    let count = kinds.len();
    eredu_runtime::ComponentDefinition::new(
        "muse_audio",
        parameter_description(config)?,
        kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| eredu_runtime::ComponentUnit {
                path: kind.path(),
                state: index..index + 1,
            })
            .collect(),
        vec![ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::AudioEncoder,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        AUDIO_EXECUTION_GROUP,
        vec![],
        state_layout(config)?,
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..count,
        )]),
    )
    .map_err(Error::from)
}

/// Convolution-history behavior at a declared waveform boundary.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AudioCarry {
    /// Continues the preceding chunk of the same audio stream.
    Continue,
    /// Starts an independent attachment with zero convolution history.
    NewClip,
}

/// One cadence-aligned span in the concatenated waveform tensor.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AudioSegment {
    /// Number of waveform samples per batch lane, including final padding.
    pub samples: i32,
    /// Whether convolution history crosses the beginning of this span.
    pub carry: AudioCarry,
}

/// Typed waveform boundaries, independent of the decoder's token framing.
pub enum AudioInput<'a, T> {
    /// A complete chunk continuing the current audio stream.
    Continuation(&'a T),
    /// Ordered explicit clip/chunk spans sharing one packed time axis.
    Segments {
        /// Waveform shaped `[batch, total_samples]`.
        waveform: &'a T,
        /// Exact ordered spans covering the packed time axis.
        segments: &'a [AudioSegment],
    },
}

/// Mutable advancement of one waveform chunk, separate from persistent histories.
#[derive(Debug, Clone)]
pub struct AudioForward {
    next_unit: usize,
    batch: i32,
    samples: i32,
    segments: Vec<AudioSegment>,
}

impl eredu_runtime::execution_control::SnapshotValue for AudioForward {
    fn estimate_snapshot_value(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, eredu_core::BackendFailure> {
        use eredu_core::{
            execution_control::{ExecutionControlError, SnapshotEstimate},
            BackendFailure,
        };
        let bytes = self
            .segments
            .capacity()
            .checked_mul(std::mem::size_of::<AudioSegment>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        eredu_runtime::execution_control::snapshot_value_estimate::<Self>([Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })])
    }
    fn copy_snapshot_value(
        &self,
        _: &eredu_runtime::execution_control::SnapshotReservation,
    ) -> Result<Self, eredu_core::BackendFailure> {
        Ok(self.clone())
    }
}

pub(crate) fn prepare_audio_forward<B: NeuralBackend>(
    config: &AudioConfig,
    input: AudioInput<'_, B::Tensor>,
    first_unit: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<LayeredForwardState<B::Tensor, AudioForward>, Error> {
    if first_unit >= config.execution_units()?.len() {
        return Err(Error::backend(
            "Muse audio ingress starts outside its execution units",
        ));
    }
    let (waveform, segments) = match input {
        AudioInput::Continuation(waveform) => (waveform, None),
        AudioInput::Segments { waveform, segments } => (waveform, Some(segments)),
    };
    let shape = waveform.shape();
    if shape.len() != 2
        || shape[0] <= 0
        || shape[1] <= 0
        || shape[1] % config.samples_per_token()? != 0
    {
        return Err(Error::backend(
            "Muse audio chunk or runtime state has incompatible geometry",
        ));
    }
    let segments = match segments {
        Some(segments) => {
            let cadence = config.samples_per_token()?;
            let total = segments.iter().try_fold(0i32, |total, segment| {
                if segment.samples <= 0 || segment.samples % cadence != 0 {
                    None
                } else {
                    total.checked_add(segment.samples)
                }
            });
            if total != Some(shape[1]) {
                return Err(Error::backend(
                    "Muse audio segments must exactly cover cadence-aligned waveform spans",
                ));
            }
            segments.to_vec()
        }
        None => vec![AudioSegment {
            samples: shape[1],
            carry: AudioCarry::Continue,
        }],
    };
    Ok(LayeredForwardState {
        hidden: waveform.reshape(&[shape[0], shape[1], 1], context)?,
        context: AudioForward {
            next_unit: first_unit,
            batch: shape[0],
            samples: shape[1],
            segments,
        },
    })
}

pub(crate) fn execute_audio_unit<B: NeuralBackend, L: RuntimeStateComponents<B>>(
    config: &AudioConfig,
    kind: AudioUnitKind,
    index: usize,
    unit: &mut AudioUnit<B>,
    hidden: &B::Tensor,
    layer: &mut L,
    forward: &mut AudioForward,
    advance_positions: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    if advance_positions <= 0 {
        return Err(Error::backend(
            "Muse audio position advance must be positive",
        ));
    }
    if forward.next_unit != index {
        return Err(Error::backend(
            "Muse audio unit advancement is out of order",
        ));
    }
    let output = match (kind, unit) {
        (AudioUnitKind::Patch { .. }, AudioUnit::Patch(unit)) => unit.forward(hidden, context)?,
        (AudioUnitKind::Projection, AudioUnit::Projection(unit)) => {
            unit.forward(hidden, context)?
        }
        (AudioUnitKind::Convolution { stage, .. }, AudioUnit::Convolution(unit)) => {
            if config.conv_kernel_size == 1 {
                unit.forward(hidden, None, context)?.0
            } else {
                let tokens = advance_positions;
                layer
                    .position()
                    .checked_add(tokens)
                    .ok_or_else(|| Error::backend("Muse audio frontier exceeds i32"))?;
                let role = StateTensorRole::Convolution { slot: 0 };
                let divisor = config.patch_sizes[..=stage].iter().product::<i32>();
                let mut history = layer.fixed_component(role).map_err(Error::backend)?.clone();
                let mut outputs = Vec::with_capacity(forward.segments.len());
                let mut offset = 0;
                for segment in &forward.segments {
                    let end = offset + segment.samples / divisor;
                    let input = hidden.index(
                        &[
                            eredu_nn::Index::Full,
                            eredu_nn::Index::Range(offset, end),
                            eredu_nn::Index::Full,
                        ],
                        context,
                    )?;
                    let previous = match segment.carry {
                        AudioCarry::Continue => history.as_ref(),
                        AudioCarry::NewClip => None,
                    };
                    let (output, candidate) = unit.forward(&input, previous, context)?;
                    outputs.push(output);
                    history = candidate;
                    offset = end;
                }
                let output = B::Tensor::concatenate(&outputs, 1, context)?;
                layer
                    .replace_fixed_components(vec![(role, history)])
                    .map_err(Error::backend)?;
                layer.advance_fixed(tokens).map_err(Error::backend)?;
                output
            }
        }
        _ => {
            return Err(Error::backend(
                "Muse audio lease differs from its declared operation",
            ))
        }
    };
    forward.next_unit += 1;
    Ok(output)
}

pub(crate) fn finish_audio_forward<T: Tensor>(
    config: &AudioConfig,
    hidden: &T,
    forward: &AudioForward,
) -> Result<T, Error> {
    if forward.next_unit != config.execution_units()?.len()
        || hidden.shape()
            != [
                forward.batch,
                forward.samples / config.samples_per_token()?,
                config.out_hidden_size,
            ]
    {
        return Err(Error::backend(
            "Muse audio traversal did not produce its complete chunk",
        ));
    }
    Ok(hidden.clone())
}

/// Shared execution contracts for the independently leased waveform units.
pub type AudioArchitecture<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    AudioKernel<B>,
>;

/// Waveform equations and request-specific continuation geometry.
pub struct AudioKernel<B: NeuralBackend> {
    config: AudioConfig,
    units: Vec<AudioUnitKind>,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    backend: PhantomData<fn() -> B>,
}

impl<B: NeuralBackend> AudioKernel<B> {
    /// Validates geometry without constructing any backend parameter or tensor.
    pub fn new(config: AudioConfig) -> Result<AudioArchitecture<B>, Error> {
        let definition = std::sync::Arc::new(component_definition(&config)?);
        crate::operator_requirements::require::<B>(
            "Muse audio",
            crate::operator_requirements::MUSE_AUDIO,
        )?;
        let units = config.execution_units()?;
        let kernel = Self {
            config,
            units,
            definition: std::sync::Arc::clone(&definition),
            backend: PhantomData,
        };
        eredu_runtime::ComponentAdapter::new(definition, kernel).map_err(Error::from)
    }

    /// Exact selected physical formats and waveform geometry.
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }

    /// Request-specific shape after each small audio unit. Frame reduction and
    /// width changes are retained for exact continuation transport admission.
    pub fn continuation_geometry(
        &self,
        samples: i32,
    ) -> Result<crate::composite_execution::CompositeContinuationGeometry, Error> {
        if samples <= 0 || samples % self.config.samples_per_token()? != 0 {
            return Err(Error::backend(
                "Muse audio continuation requires complete output-token cadence",
            ));
        }
        let mut sequence = samples;
        let mut width = 1;
        let shapes = self
            .units
            .iter()
            .map(|kind| {
                match *kind {
                    AudioUnitKind::Patch { stage } => {
                        sequence /= self.config.patch_sizes[stage];
                        width = self.config.hidden_sizes[stage];
                    }
                    AudioUnitKind::Convolution { .. } => {}
                    AudioUnitKind::Projection => width = self.config.out_hidden_size,
                }
                (sequence, width)
            })
            .collect();
        Ok(crate::composite_execution::CompositeContinuationGeometry::AfterUnits(shapes))
    }

    fn fingerprint(&self) -> String {
        let formats = self
            .config
            .linear_formats
            .iter()
            .collect::<BTreeMap<_, _>>();
        format!(
            "{:?}:{:?}:{:?}:{}:{:?}:{:?}:{}:{:?}:{formats:?}",
            self.config.hidden_sizes,
            self.config.depths,
            self.config.patch_sizes,
            self.config.conv_kernel_size,
            self.config.mlp_ratio,
            self.config.rms_norm_eps,
            self.config.out_hidden_size,
            self.config.input_precision
        )
    }
}

impl<B: NeuralBackend> eredu_runtime::ComponentKernel<B> for AudioKernel<B> {
    type Input<'a> = AudioInput<'a, B::Tensor>;
    type Unit = AudioUnit<B>;
    type ForwardContext = AudioForward;
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
        let complete = self.definition.state_layout();
        let start = state.global_layer_offset();
        let end = start
            .checked_add(state.layout().len())
            .filter(|&end| end <= complete.len())
            .ok_or_else(|| Error::backend("Muse audio state partition exceeds its units"))?;
        if complete.slice(start..end).map_err(Error::backend)? != *state.layout() {
            return Err(Error::backend(
                "Muse audio state partition changes declared history geometry",
            ));
        }
        eredu_runtime::ModelStateIdentity::new(
            "muse_spark",
            "muse_audio",
            self.fingerprint(),
            complete.len(),
            start,
            0,
            topology,
        )
        .map_err(Error::backend)
    }
    fn output_score_kind(_: &AudioForward) -> Option<eredu_core::refinement::PredictionScoreKind> {
        None
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<AudioUnit<B>, Error> {
        AudioUnit::new(&self.config, self.units[address.index()], context)
    }
    fn executes_group(&self, _: usize, _: &AudioForward) -> bool {
        true
    }
    fn retained_values<'a>(
        &'a self,
        _: &'a AudioForward,
        _: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}

impl<B, S> eredu_runtime::ComponentEquations<B, S> for AudioKernel<B>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B>,
{
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, AudioForward>, Error> {
        if state.layout() != self.definition.state_layout() {
            return Err(Error::backend(
                "Muse audio runtime state has incompatible geometry",
            ));
        }
        // Runtime frontiers count emitted audio tokens, independently of each
        // stage's internal frame rate. A partial candidate cannot start a chunk.
        let mut frontier = None;
        for (index, kind) in self.units.iter().enumerate() {
            match *kind {
                AudioUnitKind::Convolution { .. } if self.config.conv_kernel_size > 1 => {
                    let layer = state.layer(index).map_err(Error::backend)?;
                    let tokens = layer.position();
                    if tokens < 0 || frontier.is_some_and(|previous| previous != tokens) {
                        return Err(Error::backend(
                            "Muse audio stage frontiers describe an incomplete chunk",
                        ));
                    }
                    frontier = Some(tokens);
                }
                _ => {}
            }
        }
        prepare_audio_forward::<B>(&self.config, input, 0, context)
    }
    fn begin_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        _dependencies: &[&B::Tensor],
        _: &mut S,
        _: &mut AudioForward,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut AudioUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut AudioForward,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let index = address.index();
        let kind = self.units[index];
        let advance = forward.samples / self.config.samples_per_token()?;
        execute_audio_unit::<B, _>(
            &self.config,
            kind,
            index,
            unit,
            hidden,
            state.layer(index).map_err(Error::backend)?,
            forward,
            advance,
            context,
        )
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        forward: &AudioForward,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        finish_audio_forward(&self.config, hidden, forward)
    }
}

impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for AudioKernel<B>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B>,
{
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, AudioForward>, Error> {
        self.begin(input, state, context)
    }
    fn execute_parallel(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut AudioUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut AudioForward,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.execute(address, unit, hidden, state, forward, context)
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &AudioForward,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish(hidden, state, forward, context)
    }
}
