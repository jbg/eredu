//! Audio units over the ordinary resident and bounded traversal contracts.

use std::{collections::BTreeMap, marker::PhantomData, num::NonZeroU32};

use eredu_core::{
    cache::{
        LayerCachePolicy, StateResidencyClass, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorPresence, StateTensorRole,
    },
    LayerSchedule,
};
use eredu_nn::{Error, NeuralBackend, Parameterized, Tensor};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureParameters,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, ExecutionGraph,
    ExecutionUnitLayout, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    MemberSharding, OwnedParameterGroupSpec, ParallelLayeredArchitecture, ParameterGroupOwner,
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
    let mut expected = Vec::new();
    let mut owned = Vec::new();
    for (index, kind) in kinds.into_iter().enumerate() {
        for group in unit_parameter_groups(config, kind)? {
            expected.push(group.clone());
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(owner.clone(), index),
                group,
            ));
        }
    }
    ArchitectureParameterDescription::new(&graph, &layout, expected, owned).map_err(Error::backend)
}

/// Audio has no pinned learned parameters: the output adapter is its own lease.
#[derive(Debug, Default)]
pub struct AudioStaticModules;

impl<T: 'static> Parameterized<T> for AudioStaticModules {
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, T>>(&'a self, _: &mut V) {}
    fn visit_parameters_mut<'a, V: eredu_nn::ParameterVisitorMut<'a, T>>(&'a mut self, _: &mut V) {}
    fn set_trainable(&mut self, _: bool) {}
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

/// Architecture policy consumed by the shared layerwise engine and state factory.
pub struct AudioArchitecture<B: NeuralBackend> {
    config: AudioConfig,
    units: Vec<AudioUnitKind>,
    empty: AudioStaticModules,
    backend: PhantomData<fn() -> B>,
}

impl<B: NeuralBackend> AudioArchitecture<B> {
    /// Validates geometry without constructing any backend parameter or tensor.
    pub fn new(config: AudioConfig) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>("Muse audio", crate::operator_requirements::MUSE_AUDIO)?;
        let units = config.execution_units()?;
        Ok(Self {
            config,
            units,
            empty: AudioStaticModules,
            backend: PhantomData,
        })
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

    fn kind(&self, group: usize, index: usize) -> Result<AudioUnitKind, Error> {
        if group != 0 {
            return Err(Error::backend("unknown Muse audio execution group"));
        }
        self.units
            .get(index)
            .copied()
            .ok_or_else(|| Error::backend("Muse audio unit index exceeds its group"))
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

impl<B: NeuralBackend> ArchitectureParameters<B> for AudioArchitecture<B> {
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<StateLayout, Error> {
        state_layout(&self.config)
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        let complete = state_layout(&self.config)?;
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
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        parameter_description(&self.config)
    }
    fn visit_static_parameters<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
    fn visit_static_parameters_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
}

impl<B, S> LayeredArchitecture<B, S> for AudioArchitecture<B>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B>,
{
    type Input<'a> = AudioInput<'a, B::Tensor>;
    type StaticModules = AudioStaticModules;
    type Unit = AudioUnit<B>;
    type ForwardContext = AudioForward;
    type RetainedContextValues<'a>
        = std::iter::Empty<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn output_score_kind(_: &AudioForward) -> Option<eredu_core::refinement::PredictionScoreKind> {
        None
    }
    fn group_transport(&self, _: usize) -> ArchitectureGroupTransport {
        ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::AudioEncoder,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }
    }
    fn primary_execution_group(&self) -> &str {
        AUDIO_EXECUTION_GROUP
    }
    fn state_partition_plan(&self, _: &StateLayout) -> ArchitectureStatePartitionPlan {
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..self.units.len(),
        )])
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.kind(group, 0)?;
        Ok(self.units.len())
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        Ok(self.kind(group, index)?.path())
    }
    fn static_modules(&self) -> &AudioStaticModules {
        &self.empty
    }
    fn static_modules_mut(&mut self) -> &mut AudioStaticModules {
        &mut self.empty
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<AudioUnit<B>, Error> {
        AudioUnit::new(&self.config, self.kind(group, index)?, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, AudioForward>, Error> {
        if state.layout() != &state_layout(&self.config)? {
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
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut S,
        _: &mut AudioForward,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.kind(group, 0)?;
        if !dependencies.is_empty() {
            return Err(Error::backend("Muse audio root has no dependencies"));
        }
        Ok(initial.clone())
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut AudioUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut AudioForward,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let kind = self.kind(group, index)?;
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
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        forward: &AudioForward,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        finish_audio_forward(&self.config, hidden, forward)
    }
    fn retained_context_values<'a>(
        &'a self,
        _: &'a AudioForward,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for AudioArchitecture<B>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B>,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, AudioForward>, Error> {
        self.begin_forward(input, state, context)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut AudioUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut AudioForward,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_unit(group, index, unit, hidden, state, forward, context)
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &AudioForward,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_forward(hidden, state, forward, context)
    }
}
