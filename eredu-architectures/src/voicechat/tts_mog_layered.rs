//! Mixture prediction and selected-component evaluation on shared unit traversal.

use super::tts_mog::{
    f32_shape, MogBlock, MogComponentBank, MogDimensions, MogPrediction, MogPredictionHead,
    MogSample, TtsError,
};
use eredu_core::BackendFailure;
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Parameterized, Tensor, TensorElementType};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ComponentAdapter, ComponentDefinition, ComponentEquations,
    ComponentKernel, ComponentUnit, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddress,
    ExecutionUnitLayout, LayeredForwardState, LayeredInvocation, LayeredTraversalHook,
    LayerwisePolicy, LayerwiseRuntime, LocalModelLayout, LocalTensorLayout, MemberSharding,
    OwnedParameterGroupSpec, ParallelComponentKernel, ParameterGroupOwner, ParameterGroupSpec,
    ParameterMemberSpec, ParameterRole, StateLayout, StatelessComponentState, SubmissionBackend,
    TensorPlacement,
};
use std::marker::PhantomData;

const BLOCKS: &str = "voicechat.tts.mog.layers";
const PREDICTION: &str = "voicechat.tts.mog.prediction";
const COMPONENTS: &str = "voicechat.tts.mog.components";

/// Local contraction widths; all residual, mixture and latent dimensions remain global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MogLocalDimensions {
    /// Local gated MLP intermediate channels in every residual block.
    pub intermediate: i32,
    /// Local low-rank channels in each mixture's component matrices.
    pub rank: i32,
}

/// Checks every declared parameter, including coherent mixture segments and replicas.
/// The actual communicator is checked separately before a parallel invocation.
pub fn mog_local_dimensions(
    dimensions: MogDimensions,
    layout: &LocalModelLayout,
) -> Result<MogLocalDimensions, TtsError> {
    let description = mog_parameter_description(dimensions)?;
    let range = |name: &str, width: i32| {
        layout
            .tensor(name)
            .ok_or(TtsError::Configuration(
                "MoG local layout omits its partition",
            ))?
            .expanded_logical_range(width as usize)
            .map_err(TtsError::from)
    };
    let intermediate = range(
        "tts_model.tts_model.mog_head.mlp_stack.0.mlp.gate_proj.weight",
        dimensions.intermediate,
    )?;
    let rank = range("tts_model.tts_model.mog_head.low_mat", dimensions.rank)?;
    let mut expected = LocalModelLayout::default();
    for owned in description.groups() {
        let group = owned.group();
        let logical = group.partition_units().map(|_| {
            if group.logical_name() == format!("{COMPONENTS}.low_rank") {
                rank.clone()
            } else {
                intermediate.clone()
            }
        });
        for member in group.members() {
            let mut shape = member.global_shape().to_vec();
            let placement = match member.sharding() {
                MemberSharding::Replicated => TensorPlacement::Replicated,
                MemberSharding::Partitioned { axis } => {
                    let selected = logical.as_ref().expect("declared partition");
                    shape[*axis] = selected.len();
                    TensorPlacement::Range {
                        axis: *axis,
                        start: selected.start,
                        end: selected.end,
                    }
                }
                MemberSharding::PartitionedSegments { axis, segments } => {
                    let selected = logical.as_ref().expect("declared partition");
                    let indices: Vec<_> = segments
                        .iter()
                        .flat_map(|segment| {
                            segment.start + selected.start..segment.start + selected.end
                        })
                        .collect();
                    shape[*axis] = indices.len();
                    TensorPlacement::Indices {
                        axis: *axis,
                        indices,
                    }
                }
                _ => return Err(TtsError::Configuration("unknown MoG partition declaration")),
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
                    logical.clone(),
                    false,
                ),
            );
        }
    }
    if layout != &expected {
        return Err(TtsError::Configuration(
            "MoG physical layout differs from its declared partitions",
        ));
    }
    Ok(MogLocalDimensions {
        intermediate: intermediate.len() as i32,
        rank: rank.len() as i32,
    })
}

/// Exact independently materializable units used by both request modes.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum MogUnit<B: NeuralBackend> {
    /// One residual MLP stage.
    Block(MogBlock<B>),
    /// Final normalization and mixture logits only.
    Prediction(MogPredictionHead<B>),
    /// Selected-component residual, scale and low-rank parameter bank.
    Components(MogComponentBank<B>),
}

/// Ordinary and bounded MoG execution with externally supplied random decisions.
pub trait MogExecution<B: NeuralBackend> {
    /// Exact residual-stack and mixture geometry.
    fn dimensions(&self) -> MogDimensions;
    /// Runs the residual stack and predicts logits with post-stack guidance.
    fn predict(
        &mut self,
        hidden: &B::Tensor,
        guidance: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogPrediction<B::Tensor>, TtsError>;
    /// Evaluates supplied selected components without repeating prediction.
    fn selected(
        &mut self,
        prediction: &MogPrediction<B::Tensor>,
        mixture: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogSample<B::Tensor>, TtsError>;
}

fn mog_state_layout() -> Result<StateLayout, TtsError> {
    Ok(StatelessComponentState::state_layout()?)
}

/// One of the two explicit MoG invocation modes.
pub enum MogInput<'a, T> {
    /// Compute guidance-adjusted hidden values and mixture logits.
    Predict {
        /// Paired guidance lanes or ordinary hidden values.
        hidden: &'a T,
        /// Nonnegative classifier-free guidance scale.
        guidance: f32,
    },
    /// Read moments only for supplied mixture decisions.
    Selected {
        /// Retained result of a committed or in-progress prediction transition.
        prediction: &'a MogPrediction<T>,
        /// I32 mixture index per batch/time row.
        mixture: &'a T,
        /// F32 Gaussian draw with latent width.
        gaussian: &'a T,
        /// Nonnegative explicit noise scale.
        noise_scale: f32,
    },
}
/// Retained input and output values needed by independently leased units.
pub struct MogForward<T> {
    selected: bool,
    guidance: f32,
    noise_scale: f32,
    prediction: Option<MogPrediction<T>>,
    mixture: Option<T>,
    gaussian: Option<T>,
    sample: Option<MogSample<T>>,
}
impl<T> MogForward<T> {
    /// Consumes the prediction output without retaining the selected component bank.
    pub fn into_prediction(self) -> Result<MogPrediction<T>, TtsError> {
        self.prediction
            .ok_or(TtsError::Configuration("MoG traversal omitted prediction"))
    }
    /// Consumes all selected moments and the sampled latent.
    pub fn into_sample(self) -> Result<MogSample<T>, TtsError> {
        self.sample.ok_or(TtsError::Configuration(
            "MoG traversal omitted selected component",
        ))
    }
}

/// One architecture graph whose request policy activates only required parameter units.
pub struct MogKernel<B: NeuralBackend> {
    precision: super::precision::SpeechPrecision,
    dimensions: MogDimensions,
    parallel: Option<(MogLocalDimensions, LocalModelLayout)>,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> MogKernel<B> {
    /// Selects storage/arithmetic policy for all independently constructed units.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, TtsError> {
        precision.validate().map_err(TtsError::Configuration)?;
        self.precision = precision;
        Ok(self)
    }
    /// Validates dimensions without native allocation.
    pub fn new(dimensions: MogDimensions) -> Result<Self, TtsError> {
        dimensions.validate()?;
        Ok(Self {
            precision: Default::default(),
            dimensions,
            parallel: None,
            backend: PhantomData,
        })
    }
    /// Retains exact physical rank ownership without materializing any parameters.
    pub fn new_parallel(
        dimensions: MogDimensions,
        layout: &LocalModelLayout,
    ) -> Result<Self, TtsError> {
        let local = mog_local_dimensions(dimensions, layout)?;
        let mut architecture = Self::new(dimensions)?;
        architecture.parallel = Some((local, layout.clone()));
        Ok(architecture)
    }
    /// Combines selected semantic equations with the validated component declaration.
    pub fn into_component(self) -> Result<ComponentAdapter<ComponentDefinition, Self>, TtsError> {
        Ok(ComponentAdapter::new(
            tts_mog_component_definition(self.dimensions)?,
            self,
        )?)
    }
    /// Whether execution requires the selected tensor communicator.
    pub fn is_parallel(&self) -> bool {
        self.parallel.is_some()
    }
    fn begin_input(
        &self,
        input: MogInput<'_, B::Tensor>,
        state: &StatelessComponentState,
    ) -> Result<LayeredForwardState<B::Tensor, MogForward<B::Tensor>>, TtsError> {
        if *state.layout() != mog_state_layout()? {
            return Err(TtsError::Configuration("MoG runtime layout mismatch"));
        }
        let mut forward = MogForward {
            selected: false,
            guidance: 0.0,
            noise_scale: 0.0,
            prediction: None,
            mixture: None,
            gaussian: None,
            sample: None,
        };
        let hidden = match input {
            MogInput::Predict { hidden, guidance } => {
                f32_shape(hidden, self.dimensions.hidden)?;
                if !guidance.is_finite() || guidance < 0.0 {
                    return Err(TtsError::Configuration("invalid MoG guidance scale"));
                }
                if guidance > 0.0 && hidden.dim(0) % 2 != 0 {
                    return Err(TtsError::Shape(
                        "guidance requires paired conditional/unconditional batches",
                    ));
                }
                forward.guidance = guidance;
                hidden.clone()
            }
            MogInput::Selected {
                prediction,
                mixture,
                gaussian,
                noise_scale,
            } => {
                f32_shape(&prediction.hidden, self.dimensions.hidden)?;
                f32_shape(&prediction.logits, self.dimensions.mixtures)?;
                f32_shape(gaussian, self.dimensions.latent)?;
                let shape = prediction.hidden.shape();
                if mixture.shape() != &shape[..2]
                    || mixture.element_type() != Some(TensorElementType::I32)
                    || gaussian.shape()[..2] != shape[..2]
                    || prediction.logits.shape()[..2] != shape[..2]
                {
                    return Err(TtsError::Shape(
                        "MoG decisions/draws do not match hidden batch and time",
                    ));
                }
                if !noise_scale.is_finite() || noise_scale < 0.0 {
                    return Err(TtsError::Configuration("invalid MoG noise scale"));
                }
                forward.selected = true;
                forward.noise_scale = noise_scale;
                forward.prediction = Some(prediction.clone());
                forward.mixture = Some(mixture.clone());
                forward.gaussian = Some(gaussian.clone());
                prediction.hidden.clone()
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: forward,
        })
    }
    /// Exact connected dimensions.
    pub fn dimensions(&self) -> MogDimensions {
        self.dimensions
    }
}
/// Validated logical invocations, independent parameter units and transport semantics.
pub fn tts_mog_component_definition(
    dimensions: MogDimensions,
) -> Result<ComponentDefinition, TtsError> {
    let parameters = mog_parameter_description(dimensions)?;
    let paths: Vec<String> = (0..dimensions.layers)
        .map(|index| format!("tts_model.tts_model.mog_head.mlp_stack.{index}"))
        .chain([PREDICTION.into(), COMPONENTS.into()])
        .collect();
    let transport = (0..parameters.graph().groups().len())
        .map(|group| {
            let _ = group;
            ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: if group == 0 {
                    ArchitectureGroupKind::Decoder
                } else {
                    ArchitectureGroupKind::Projector
                },
                first_owner_static_roles: vec![],
                last_owner_static_roles: vec![],
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }
        })
        .collect();
    Ok(ComponentDefinition::new(
        "voicechat.tts_mog",
        parameters,
        paths
            .into_iter()
            .map(|path| ComponentUnit { path, state: 0..0 })
            .collect(),
        transport,
        BLOCKS,
        vec![],
        mog_state_layout()?,
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::output_owner(0..1)]),
    )?)
}

fn mog_graph() -> Result<ExecutionGraph, TtsError> {
    Ok(ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(BLOCKS),
            ExecutionGroupSpec::with_dependencies(PREDICTION, [BLOCKS]),
            ExecutionGroupSpec::with_dependencies(COMPONENTS, [PREDICTION]),
        ],
        COMPONENTS,
    )?)
}

/// Symbolic MoG parameter shapes and independent head ownership, without native work.
pub fn mog_parameter_description(
    d: MogDimensions,
) -> Result<ArchitectureParameterDescription, TtsError> {
    d.validate()?;
    let graph = mog_graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [d.layers, 1, 1])?;
    let mut expected = Vec::new();
    let mut owned = Vec::new();
    let member = |name: String, shape: Vec<usize>| {
        ParameterMemberSpec::new(name, shape, MemberSharding::Replicated)
    };
    let mut add = |group: usize, index: usize, spec: ParameterGroupSpec| -> Result<(), TtsError> {
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
    let h = d.hidden as usize;
    let i = d.intermediate as usize;
    let m = d.mixtures as usize;
    let l = d.latent as usize;
    let r = d.rank as usize;
    let root = "tts_model.tts_model.mog_head";
    for index in 0..d.layers {
        let path = format!("{root}.mlp_stack.{index}");
        add(
            0,
            index,
            ParameterGroupSpec::new(
                format!("{path}.norms"),
                ParameterRole::Replicated,
                [
                    member(format!("{path}.pre_norm.weight"), vec![h]),
                    member(format!("{path}.post_norm.weight"), vec![h]),
                ],
            )?,
        )?;
        add(
            0,
            index,
            ParameterGroupSpec::partitioned(
                format!("{path}.mlp"),
                ParameterRole::FeedForwardIntermediate,
                i,
                [
                    ParameterMemberSpec::new(
                        format!("{path}.mlp.gate_proj.weight"),
                        vec![i, h],
                        MemberSharding::Partitioned { axis: 0 },
                    ),
                    ParameterMemberSpec::new(
                        format!("{path}.mlp.up_proj.weight"),
                        vec![i, h],
                        MemberSharding::Partitioned { axis: 0 },
                    ),
                    ParameterMemberSpec::new(
                        format!("{path}.mlp.down_proj.weight"),
                        vec![h, i],
                        MemberSharding::Partitioned { axis: 1 },
                    ),
                ],
            )?,
        )?;
    }
    add(
        1,
        0,
        ParameterGroupSpec::new(
            PREDICTION,
            ParameterRole::Replicated,
            [
                member(format!("{root}.mlp_stack.{}.weight", d.layers), vec![h]),
                member(format!("{root}.proj_logits.weight"), vec![m, h]),
            ],
        )?,
    )?;
    add(
        2,
        0,
        ParameterGroupSpec::new(
            COMPONENTS,
            ParameterRole::Replicated,
            [
                member(format!("{root}.proj_else.weight"), vec![l, h]),
                member(format!("{root}.proj_logs.weight"), vec![1, h]),
            ],
        )?,
    )?;
    add(
        2,
        0,
        ParameterGroupSpec::partitioned(
            format!("{COMPONENTS}.low_rank"),
            ParameterRole::FeedForwardIntermediate,
            r,
            [
                ParameterMemberSpec::new(
                    format!("{root}.proj_mus.weight"),
                    vec![m * r, h],
                    MemberSharding::PartitionedSegments {
                        axis: 0,
                        segments: (0..m)
                            .map(|mixture| mixture * r..(mixture + 1) * r)
                            .collect(),
                    },
                ),
                ParameterMemberSpec::new(
                    format!("{root}.low_mat"),
                    vec![m, l, r],
                    MemberSharding::Partitioned { axis: 2 },
                ),
            ],
        )?,
    )?;
    Ok(ArchitectureParameterDescription::new(
        &graph, &layout, expected, owned,
    )?)
}

impl<B: NeuralBackend> ComponentKernel<B> for MogKernel<B> {
    type Input<'a> = MogInput<'a, B::Tensor>;
    type Unit = MogUnit<B>;
    type ForwardContext = MogForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Flatten<std::array::IntoIter<Option<&'a B::Tensor>, 6>>
    where
        B::Tensor: 'a;
    type Error = TtsError;

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, TtsError> {
        if state.global_layer_offset() != 0 || *state.layout() != mog_state_layout()? {
            return Err(TtsError::Configuration(
                "MoG stateless layout differs from selected identity",
            ));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "nemotron_voicechat",
            "voicechat_mog",
            format!(
                "{:?}:{:?}",
                (self.dimensions, self.precision),
                self.parallel.as_ref().map(|(_, layout)| layout)
            ),
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
    ) -> Result<MogUnit<B>, TtsError> {
        let group = address.group();
        let index = address.index();

        let local = self
            .parallel
            .as_ref()
            .map(|(local, _)| *local)
            .unwrap_or(MogLocalDimensions {
                intermediate: self.dimensions.intermediate,
                rank: self.dimensions.rank,
            });
        Ok(match group {
            0 => MogUnit::Block(
                MogBlock::new_local(self.dimensions, local.intermediate, index, context)?
                    .with_precision(self.precision)?,
            ),
            1 => MogUnit::Prediction(MogPredictionHead::new(self.dimensions, context)?),
            _ => MogUnit::Components(MogComponentBank::new_local(
                self.dimensions,
                local.rank,
                context,
            )?),
        })
    }

    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        if forward.selected {
            group == 2
        } else {
            group != 2
        }
    }

    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        [
            forward.prediction.as_ref().map(|p| &p.hidden),
            forward.prediction.as_ref().map(|p| &p.logits),
            forward.mixture.as_ref(),
            forward.gaussian.as_ref(),
            forward.sample.as_ref().map(|s| &s.mean),
            forward.sample.as_ref().map(|s| &s.log_std),
        ]
        .into_iter()
        .flatten()
    }
}
impl<B: NeuralBackend> ComponentEquations<B, StatelessComponentState> for MogKernel<B> {
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut StatelessComponentState,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, TtsError> {
        if self.is_parallel() {
            return Err(TtsError::Configuration(
                "rank-local MoG execution requires its tensor communicator",
            ));
        }
        self.begin_input(input, state)
    }
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut StatelessComponentState,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            (1 | 2, [one]) => Ok((*one).clone()),
            _ => Err(TtsError::Configuration(
                "MoG group dependencies differ from graph",
            )),
        }
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut MogUnit<B>,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        let group = address.group();

        if self.is_parallel() {
            return Err(TtsError::Configuration(
                "rank-local MoG execution requires its tensor communicator",
            ));
        }
        match (group, unit) {
            (0, MogUnit::Block(block)) => block.forward(hidden, context),
            (1, MogUnit::Prediction(head)) => {
                let prediction = head.predict(hidden, forward.guidance, context)?;
                let output = prediction.logits.clone();
                forward.prediction = Some(prediction);
                Ok(output)
            }
            (2, MogUnit::Components(bank)) => {
                let sample = bank.selected(
                    forward.prediction.as_ref().ok_or(TtsError::Configuration(
                        "selected MoG invocation lacks prediction",
                    ))?,
                    forward.mixture.as_ref().ok_or(TtsError::Configuration(
                        "selected MoG invocation lacks decisions",
                    ))?,
                    forward.gaussian.as_ref().ok_or(TtsError::Configuration(
                        "selected MoG invocation lacks Gaussian draws",
                    ))?,
                    forward.noise_scale,
                    context,
                )?;
                let output = sample.latent.clone();
                forward.sample = Some(sample);
                Ok(output)
            }
            _ => Err(TtsError::Configuration(
                "leased MoG unit differs from graph address",
            )),
        }
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        _: &Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        Ok(hidden.clone())
    }
}

impl<B: DistributedNeuralBackend> MogKernel<B> {
    /// Validates exact communicator membership before parameter acquisition or collectives.
    pub fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), TtsError> {
        let (_, selected) = self.parallel.as_ref().ok_or(TtsError::Configuration(
            "parallel MoG execution requires rank-local construction",
        ))?;
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(B::parallel_size(parallel), 1, 1, 1)
                .map_err(|_| TtsError::Configuration("invalid MoG tensor degree"))?,
            B::parallel_rank(parallel),
        )
        .map_err(|_| TtsError::Configuration("invalid MoG tensor rank"))?;
        let description = mog_parameter_description(self.dimensions)?;
        let expected = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(|_| TtsError::Configuration("MoG tensor placement cannot be derived"))?;
        if selected != &expected {
            return Err(TtsError::Configuration(
                "MoG layout differs from its tensor communicator",
            ));
        }
        Ok(())
    }
    fn forward_unit_guarded(
        &self,
        group: usize,
        index: usize,
        unit: &mut MogUnit<B>,
        hidden: &B::Tensor,
        forward: &mut MogForward<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        agree: &mut impl FnMut(
            [u64; 8],
            bool,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<bool, TtsError>,
    ) -> Result<B::Tensor, TtsError> {
        let valid = (|| {
            let local = self
                .parallel
                .as_ref()
                .ok_or(TtsError::Configuration(
                    "parallel MoG requires rank-local construction",
                ))?
                .0;
            let valid = match (group, &*unit) {
                (0, MogUnit::Block(block)) => {
                    block.geometry() == (self.dimensions, local.intermediate)
                }
                (1, MogUnit::Prediction(head)) => head.dimensions() == self.dimensions,
                (2, MogUnit::Components(bank)) => {
                    bank.geometry() == (self.dimensions, local.rank)
                        && forward.prediction.is_some()
                        && forward.mixture.is_some()
                        && forward.gaussian.is_some()
                }
                _ => false,
            };
            if !valid {
                return Err(TtsError::Configuration(
                    "MoG leased unit differs from selected invocation",
                ));
            }
            Ok(())
        })();
        let accepted = agree(
            [
                0x4d4f474c4f434c00,
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
        )?;
        valid?;
        if !accepted {
            return Err(TtsError::Configuration("MoG peer rejected leased unit"));
        }
        let mut reduce = |prepared: Result<B::Tensor, TtsError>| {
            let mut descriptor = [
                0x4d4f474c4f434c01,
                group as u64,
                index as u64,
                0,
                0,
                0,
                0,
                0,
            ];
            if let Ok(value) = &prepared {
                descriptor[4] = value.shape().len() as u64;
                for (slot, dim) in descriptor[5..].iter_mut().zip(value.shape()) {
                    *slot = *dim as u64;
                }
            }
            let accepted = agree(descriptor, prepared.is_ok(), context)?;
            let value = prepared?;
            if !accepted {
                return Err(TtsError::Configuration(
                    "MoG peer failed collective preparation",
                ));
            }
            Ok(B::sum_parallel(value, parallel, context)?)
        };
        match unit {
            MogUnit::Block(block) => block.forward_with_reduction(hidden, context, &mut reduce),
            MogUnit::Prediction(head) => {
                let prediction = head.predict(hidden, forward.guidance, context)?;
                let output = prediction.logits.clone();
                forward.prediction = Some(prediction);
                Ok(output)
            }
            MogUnit::Components(bank) => {
                let sample = bank.selected_with_reduction(
                    forward.prediction.as_ref().expect("validated prediction"),
                    forward.mixture.as_ref().expect("validated decisions"),
                    forward.gaussian.as_ref().expect("validated Gaussian"),
                    forward.noise_scale,
                    context,
                    &mut reduce,
                )?;
                let output = sample.latent.clone();
                forward.sample = Some(sample);
                Ok(output)
            }
        }
    }
}

impl<B: DistributedNeuralBackend> ParallelComponentKernel<B, eredu_runtime::StatelessComponentState>
    for MogKernel<B>
{
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut StatelessComponentState,
        parallel: &B::ParallelContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, TtsError> {
        self.validate_parallel(parallel)?;
        self.begin_input(input, state)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        let group = address.group();

        let (local, _) = self.parallel.as_ref().ok_or(TtsError::Configuration(
            "parallel MoG execution requires rank-local construction",
        ))?;
        match (group, unit) {
            (0, MogUnit::Block(block))
                if block.geometry() == (self.dimensions, local.intermediate) =>
            {
                block.forward_parallel(hidden, parallel, context)
            }
            (1, MogUnit::Prediction(head)) if head.dimensions() == self.dimensions => {
                let prediction = head.predict(hidden, forward.guidance, context)?;
                let output = prediction.logits.clone();
                forward.prediction = Some(prediction);
                Ok(output)
            }
            (2, MogUnit::Components(bank)) if bank.geometry() == (self.dimensions, local.rank) => {
                let sample = bank.selected_parallel(
                    forward.prediction.as_ref().ok_or(TtsError::Configuration(
                        "selected MoG invocation lacks prediction",
                    ))?,
                    forward.mixture.as_ref().ok_or(TtsError::Configuration(
                        "selected MoG invocation lacks decisions",
                    ))?,
                    forward.gaussian.as_ref().ok_or(TtsError::Configuration(
                        "selected MoG invocation lacks Gaussian draws",
                    ))?,
                    forward.noise_scale,
                    parallel,
                    context,
                )?;
                let output = sample.latent.clone();
                forward.sample = Some(sample);
                Ok(output)
            }
            _ => Err(TtsError::Configuration(
                "leased MoG unit differs from selected geometry or graph address",
            )),
        }
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut StatelessComponentState,
        forward: &Self::ForwardContext,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.finish(hidden, state, forward, context)
    }
}

/// MoG head using an ordinary prepared parameter policy and shared bounded traversal.
pub struct LayerwiseMog<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, MogUnit<B>>,
{
    runtime: LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, MogKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >,
}
impl<B, P> LayerwiseMog<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, MogUnit<B>>,
    P::Error: std::fmt::Display,
{
    /// Adopts the existing provider without materializing unused head parameters.
    pub fn new(
        architecture: ComponentAdapter<ComponentDefinition, MogKernel<B>>,
        policy: P,
    ) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Existing provider/runtime for diagnostics and parameter operations.
    pub fn runtime(
        &self,
    ) -> &LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, MogKernel<B>>,
        B,
        StatelessComponentState,
        P,
    > {
        &self.runtime
    }
    /// Existing provider/runtime for bounded parameter operations.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, MogKernel<B>>,
        B,
        StatelessComponentState,
        P,
    > {
        &mut self.runtime
    }
}
fn runtime_error<P: std::error::Error + Send + Sync + 'static>(
    error: eredu_runtime::LayerwiseRuntimeError<TtsError, P>,
) -> TtsError {
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
impl<B, P> MogExecution<B> for LayerwiseMog<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, MogUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> MogDimensions {
        self.runtime.architecture().kernel().dimensions()
    }
    fn predict(
        &mut self,
        hidden: &B::Tensor,
        guidance: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogPrediction<B::Tensor>, TtsError> {
        let mut state = StatelessComponentState::new()?;
        let (_, forward) = self
            .runtime
            .invoke_with_context(MogInput::Predict { hidden, guidance }, &mut state, context)
            .map_err(runtime_error)?;
        forward.into_prediction()
    }
    fn selected(
        &mut self,
        prediction: &MogPrediction<B::Tensor>,
        mixture: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogSample<B::Tensor>, TtsError> {
        let mut state = StatelessComponentState::new()?;
        let (_, forward) = self
            .runtime
            .invoke_with_context(
                MogInput::Selected {
                    prediction,
                    mixture,
                    gaussian,
                    noise_scale,
                },
                &mut state,
                context,
            )
            .map_err(runtime_error)?;
        forward.into_sample()
    }
}

impl<B, P> LayerwiseMog<B, P>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, MogUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Runs rank-local residual projections through the shared parallel traversal.
    /// Guidance and mixture logits remain complete and replicated.
    pub fn predict_parallel(
        &mut self,
        hidden: &B::Tensor,
        guidance: f32,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogPrediction<B::Tensor>, TtsError> {
        let mut state = StatelessComponentState::new()?;
        let (_, forward) = self
            .runtime
            .forward_parallel_with_context_hook(
                MogInput::Predict { hidden, guidance },
                &mut state,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map_err(runtime_error)?;
        forward.into_prediction()
    }
    /// Evaluates coherent component shards using externally supplied shared decisions.
    pub fn selected_parallel(
        &mut self,
        prediction: &MogPrediction<B::Tensor>,
        mixture: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogSample<B::Tensor>, TtsError> {
        let mut state = StatelessComponentState::new()?;
        let (_, forward) = self
            .runtime
            .forward_parallel_with_context_hook(
                MogInput::Selected {
                    prediction,
                    mixture,
                    gaussian,
                    noise_scale,
                },
                &mut state,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map_err(runtime_error)?;
        forward.into_sample()
    }
    /// Executes retained tensor shards with acquisition and pre-reduction agreement.
    pub fn invoke_parallel_guarded<H, F>(
        &mut self,
        input: MogInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        mut agree: F,
    ) -> Result<(B::Tensor, MogForward<B::Tensor>), TtsError>
    where
        H: LayeredTraversalHook<B, MogForward<B::Tensor>, TtsError> + ?Sized,
        F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, TtsError>,
    {
        let (mode, value, policy) = match &input {
            MogInput::Predict { hidden, guidance } => (0, *hidden, *guidance),
            MogInput::Selected {
                prediction,
                noise_scale,
                ..
            } => (1, &prediction.hidden, *noise_scale),
        };
        let mut descriptor = [
            0x4d4f47494e505400,
            mode,
            policy.to_bits() as u64,
            0,
            value.shape().len() as u64,
            0,
            0,
            0,
        ];
        for (slot, dim) in descriptor[5..].iter_mut().zip(value.shape()) {
            *slot = *dim as u64;
        }
        if !agree(descriptor, true, context)? {
            return Err(TtsError::Configuration(
                "MoG participants selected different invocations",
            ));
        }
        let d = self.dimensions();
        if !agree(
            [
                0x4d4f47494e505401,
                d.layers as u64,
                d.hidden as u64,
                d.intermediate as u64,
                d.mixtures as u64,
                d.latent as u64,
                d.rank as u64,
                d.min_log_std.to_bits() as u64,
            ],
            true,
            context,
        )? {
            return Err(TtsError::Configuration("MoG participant dimensions differ"));
        }
        if !agree(
            [
                0x4d4f47494e505402,
                d.epsilon.to_bits() as u64,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            true,
            context,
        )? {
            return Err(TtsError::Configuration(
                "MoG participant normalization policy differs",
            ));
        }
        self.runtime
            .forward_parallel_with_unit_executor_and_traversal_hook(
                input,
                &mut StatelessComponentState::new()?,
                parallel,
                context,
                |architecture, group, index, unit, hidden, _, forward, parallel, context| {
                    architecture.kernel().forward_unit_guarded(
                        group, index, unit, hidden, forward, parallel, context, &mut agree,
                    )
                },
                hook,
            )
            .map_err(runtime_error)
    }
}
