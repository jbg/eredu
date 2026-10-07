//! Separately leased character conditioning, speaker ingress and gated fusion.
use super::{
    config::TtsConfig,
    tts_conditioning::{CharacterBatch, CharacterDimensions, CharacterEmbedding, GatedFusion},
    tts_ingress::{
        shift_prompt_codes, validate_decode, validate_embedded, validate_prefill,
        ConditioningLanes, PromptLayout, SpeakerPrompt, TtsIngress,
    },
    tts_mog::{f32_shape, TtsError},
    tts_rvq_layered::RvqExecution,
};
use eredu_core::{cache::LayerCachePolicy, BackendFailure, LayerSchedule};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Parameterized, Tensor};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureParameters,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, ExecutionGraph,
    ExecutionGroupSpec, ExecutionUnitAddress, ExecutionUnitLayout, LayeredArchitecture,
    LayeredForwardState, LayeredInvocation, LayerwisePolicy, LayerwiseRuntime, LocalModelLayout,
    LocalTensorLayout, MemberSharding, OwnedParameterGroupSpec, ParallelLayeredArchitecture,
    ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec, ParameterRole, ResidentRuntime,
    RuntimeState, StateError, StateLayout, StaticParameterVisitor, StaticParameterVisitorMut,
    SubmissionBackend, TensorPlacement,
};
use std::marker::PhantomData;

const CHARACTER: &str = "voicechat.tts.conditioning.character";
const INGRESS: &str = "voicechat.tts.conditioning.ingress";
const FUSION: &str = "voicechat.tts.conditioning.fusion";
/// Exact geometry of the three independent learned conditioning units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditioningDimensions {
    /// Character table, projection and optional flag geometry.
    pub character: CharacterDimensions,
    /// Fixed Aria prompt length.
    pub speaker_frames: i32,
}
impl ConditioningDimensions {
    /// Validates geometry without constructing native resources.
    pub fn new(character: CharacterDimensions, speaker_frames: i32) -> Result<Self, TtsError> {
        character.validate()?;
        if speaker_frames <= 0 {
            return Err(TtsError::Configuration("invalid speaker prompt length"));
        }
        Ok(Self {
            character,
            speaker_frames,
        })
    }
    /// Resolves normalized policy and admitted character/speaker dimensions.
    pub fn from_config(
        config: &TtsConfig,
        characters: i32,
        speaker_frames: i32,
    ) -> Result<Self, TtsError> {
        let size = |n| {
            i32::try_from(n).map_err(|_| {
                TtsError::Configuration("conditioning dimension exceeds tensor indexing")
            })
        };
        Self::new(
            CharacterDimensions {
                characters,
                width: size(config.character_encoder.hidden_size)?,
                output_width: size(config.backbone.hidden_size)?,
                use_continuation: config.use_subword_flag_emb,
                use_special: config.use_bos_eos_emb,
            },
            speaker_frames,
        )
    }
}
/// Exact tensor-parallel character and speech feature slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditioningLocalDimensions {
    /// First character feature on this rank.
    pub character_start: i32,
    /// Number of character features on this rank.
    pub character: i32,
    /// First speech feature on this rank.
    pub output_start: i32,
    /// Number of speech features on this rank.
    pub output: i32,
}
/// Validates every physical parameter placement, including replicated control parameters.
pub fn conditioning_local_dimensions(
    dimensions: ConditioningDimensions,
    layout: &LocalModelLayout,
) -> Result<ConditioningLocalDimensions, TtsError> {
    let description = conditioning_parameter_description(dimensions)?;
    let range = |name, width| {
        layout
            .tensor(name)
            .ok_or(TtsError::Configuration(
                "conditioning layout omits partition",
            ))?
            .expanded_logical_range(width)
            .map_err(TtsError::from)
    };
    let character = range(
        "tts_model.tts_model.embed_subword.embed_tokens.weight",
        dimensions.character.width as usize,
    )?;
    let output = range(
        "tts_model.tts_model.embed_subword.proj_embedding.weight",
        dimensions.character.output_width as usize,
    )?;
    if character.is_empty() || output.is_empty() {
        return Err(TtsError::Configuration(
            "empty conditioning feature partition",
        ));
    }
    let mut expected = LocalModelLayout::default();
    for owned in description.groups() {
        let group = owned.group();
        let logical = group.partition_units().map(|_| {
            if group.logical_name() == format!("{CHARACTER}.tokens") {
                character.clone()
            } else {
                output.clone()
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
                _ => return Err(TtsError::Configuration("unknown conditioning partition")),
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
            "conditioning physical layout differs from declared partitions",
        ));
    }
    Ok(ConditioningLocalDimensions {
        character_start: character.start as i32,
        character: character.len() as i32,
        output_start: output.start as i32,
        output: output.len() as i32,
    })
}

/// Shared conditioning choreography; phase invocations complete before another provider runs.
pub trait ConditioningExecution<B: NeuralBackend> {
    /// Exact selected geometry.
    fn dimensions(&self) -> ConditioningDimensions;
    /// Looks up packed character embeddings.
    fn character_embed(
        &mut self,
        batch: &CharacterBatch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError>;
    /// Pools, projects, scatters and adds saved character flags.
    fn character_finish(
        &mut self,
        batch: &CharacterBatch,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError>;
    /// Applies prompt substitutions to previously projected feedback.
    #[allow(clippy::too_many_arguments)]
    fn prefill_embedded(
        &mut self,
        embedded: &B::Tensor,
        layout: &PromptLayout,
        speaker: SpeakerPrompt,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError>;
    /// Builds guidance lanes from previously projected feedback.
    fn decode_embedded(
        &mut self,
        embedded: &B::Tensor,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError>;
    /// Executes the released projected audio/text gate.
    fn fuse(
        &mut self,
        audio: &B::Tensor,
        text: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError>;
    /// Runs separate character phases around the caller's transformer; empty masks skip all phases.
    fn character_forward<F>(
        &mut self,
        batch: &CharacterBatch,
        backbone: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, TtsError>
    where
        F: FnOnce(&B::Tensor, &B::Tensor) -> Result<B::Tensor, TtsError>,
    {
        if batch.is_empty() {
            return Ok(None);
        }
        batch.validate_dimensions(self.dimensions().character)?;
        let embeddings = self.character_embed(batch, context)?;
        let mask = batch.attention_mask::<B::Tensor>(context)?;
        let hidden = backbone(&embeddings, &mask)?;
        self.character_finish(batch, &hidden, context).map(Some)
    }
    /// Shifts source codes, completes feedback execution, then acquires ingress.
    #[allow(clippy::too_many_arguments)]
    fn prefill<Z: RvqExecution<B>>(
        &mut self,
        feedback: &mut Z,
        codes: &B::Tensor,
        layout: &PromptLayout,
        speaker: SpeakerPrompt,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
        let d = self.dimensions();
        validate_prefill(
            codes,
            layout,
            speaker,
            d.speaker_frames,
            d.character.output_width,
            conditioning,
        )?;
        let shifted = shift_prompt_codes(codes, layout, context)?;
        let embedded = feedback.embed_codes(&shifted, context)?;
        self.prefill_embedded(&embedded, layout, speaker, conditioning, guidance, context)
    }
    /// Completes feedback execution before acquiring cached-decode ingress.
    fn decode<Z: RvqExecution<B>>(
        &mut self,
        feedback: &mut Z,
        codes: &B::Tensor,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
        validate_decode(
            codes,
            self.dimensions().character.output_width,
            conditioning,
        )?;
        let embedded = feedback.embed_codes(codes, context)?;
        self.decode_embedded(&embedded, conditioning, guidance, context)
    }
}
/// Independent learned conditioning units.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum ConditioningUnit<B: NeuralBackend> {
    /// Character embedding, pooling projection and saved flags.
    Character(CharacterEmbedding<B>),
    /// Speaker, BOS and null conditioning.
    Ingress(TtsIngress<B>),
    /// Final audio/text gated fusion.
    Fusion(GatedFusion<B>),
}
/// All learned conditioning parameters belong to execution units.
pub struct ConditioningStaticModules;
impl<T: 'static> Parameterized<T> for ConditioningStaticModules {
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

/// Stateless phase traversal owner; mutable transformer and refinement state belong to callers.
pub struct ConditioningRuntimeState {
    layout: StateLayout,
}
impl ConditioningRuntimeState {
    /// Constructs host-only empty state metadata.
    pub fn new() -> Result<Self, TtsError> {
        Ok(Self {
            layout: conditioning_state_layout()?,
        })
    }
}
impl<B: NeuralBackend> RuntimeState<B> for ConditioningRuntimeState {
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
fn conditioning_state_layout() -> Result<StateLayout, TtsError> {
    Ok(StateLayout::new(LayerSchedule::new(
        1,
        vec![LayerCachePolicy::NoState],
    )?)?)
}
/// One conditioning phase, with immutable inputs retained until completion.
pub enum ConditioningInput<T> {
    /// Packed character lookup.
    CharacterEmbed(CharacterBatch),
    /// Character pooling and scatter.
    CharacterFinish(CharacterBatch, T),
    /// Speaker/BOS substitutions.
    Prefill {
        /// Projected, shifted prompt-code feedback.
        embedded: T,
        /// Validated host BOS and speaker-substitution positions.
        layout: PromptLayout,
        /// Fixed Aria or projected-code speaker conditioning.
        speaker: SpeakerPrompt,
        /// Optional character output for the conditional lane.
        conditioning: Option<T>,
        /// Whether to append the unconditional guidance lane.
        guidance: bool,
    },
    /// Cached guidance lanes.
    Decode {
        /// Projected previous-frame feedback.
        embedded: T,
        /// Optional character output for the conditional lane.
        conditioning: Option<T>,
        /// Whether to append the unconditional guidance lane.
        guidance: bool,
    },
    /// Learned audio/text gate.
    Fuse(T, T),
}
impl<T> ConditioningInput<T> {
    fn group(&self) -> usize {
        match self {
            Self::CharacterEmbed(_) | Self::CharacterFinish(..) => 0,
            Self::Prefill { .. } | Self::Decode { .. } => 1,
            Self::Fuse(..) => 2,
        }
    }
}
/// Exact immutable inputs and both candidate guidance lanes.
pub struct ConditioningForward<T> {
    input: ConditioningInput<T>,
    lanes: Option<ConditioningLanes<T>>,
}
impl<T> ConditioningForward<T> {
    /// Takes both lanes only after successful exact completion.
    pub fn into_lanes(self) -> Result<ConditioningLanes<T>, TtsError> {
        self.lanes.ok_or(TtsError::Configuration(
            "conditioning traversal omitted ingress",
        ))
    }
}
/// Shared graph for resident and bounded conditioning phases.
pub struct ConditioningArchitecture<B: NeuralBackend> {
    dimensions: ConditioningDimensions,
    parallel: Option<(ConditioningLocalDimensions, LocalModelLayout)>,
    empty: ConditioningStaticModules,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> ConditioningArchitecture<B> {
    /// Validates geometry without learned parameter allocation.
    pub fn new(dimensions: ConditioningDimensions) -> Result<Self, TtsError> {
        ConditioningDimensions::new(dimensions.character, dimensions.speaker_frames)?;
        Ok(Self {
            dimensions,
            parallel: None,
            empty: ConditioningStaticModules,
            backend: PhantomData,
        })
    }
    /// Selects an exact retained rank-local parameter layout before materialization.
    pub fn new_parallel(
        dimensions: ConditioningDimensions,
        layout: &LocalModelLayout,
    ) -> Result<Self, TtsError> {
        let local = conditioning_local_dimensions(dimensions, layout)?;
        let mut architecture = Self::new(dimensions)?;
        architecture.parallel = Some((local, layout.clone()));
        Ok(architecture)
    }
    /// Whether construction selected tensor-parallel parameters.
    pub fn is_parallel(&self) -> bool {
        self.parallel.is_some()
    }
    fn local(&self) -> ConditioningLocalDimensions {
        self.parallel
            .as_ref()
            .map(|(local, _)| *local)
            .unwrap_or(ConditioningLocalDimensions {
                character_start: 0,
                character: self.dimensions.character.width,
                output_start: 0,
                output: self.dimensions.character.output_width,
            })
    }
    /// Selected geometry.
    pub fn dimensions(&self) -> ConditioningDimensions {
        self.dimensions
    }
    /// Stable phase graph.
    pub fn execution_graph(&self) -> Result<ExecutionGraph, TtsError> {
        conditioning_graph()
    }
    /// Each phase owns one parameter unit.
    pub fn group_unit_count(&self, group: usize) -> Result<usize, TtsError> {
        if group < 3 {
            Ok(1)
        } else {
            Err(TtsError::Configuration("unknown conditioning group"))
        }
    }
    /// Canonical phase identity.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, TtsError> {
        if index != 0 {
            return Err(TtsError::Configuration("unknown conditioning unit"));
        }
        self.group_unit_count(group)?;
        Ok([CHARACTER, INGRESS, FUSION][group].into())
    }
    /// Constructs only the selected unloaded learned unit.
    pub fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningUnit<B>, TtsError> {
        self.unit_path(group, index)?;
        let d = self.dimensions;
        Ok(match group {
            0 => ConditioningUnit::Character(CharacterEmbedding::new_local(
                d.character,
                self.local().character,
                self.local().output,
                context,
            )?),
            1 => ConditioningUnit::Ingress(TtsIngress::new_local(
                d.character.output_width,
                d.speaker_frames,
                self.local().output,
                context,
            )?),
            _ => ConditioningUnit::Fusion(GatedFusion::new_local(
                d.character.output_width,
                self.local().output,
                context,
            )?),
        })
    }
}
impl<B: NeuralBackend> ConditioningArchitecture<B> {
    fn begin_input(
        &mut self,
        input: ConditioningInput<B::Tensor>,
        state: &mut ConditioningRuntimeState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ConditioningForward<B::Tensor>>, TtsError> {
        if state.layout != conditioning_state_layout()? {
            return Err(TtsError::Configuration(
                "conditioning runtime layout mismatch",
            ));
        }
        let d = self.dimensions;
        let hidden = match &input {
            ConditioningInput::CharacterEmbed(batch) => {
                batch.validate_dimensions(d.character)?;
                B::Tensor::full_f32(0.0, &[], context)?
            }
            ConditioningInput::CharacterFinish(batch, hidden) => {
                batch.validate_dimensions(d.character)?;
                f32_shape(hidden, d.character.width)?;
                if hidden.shape()[..2] != batch.shape() {
                    return Err(TtsError::Shape("character hidden geometry mismatch"));
                }
                hidden.clone()
            }
            ConditioningInput::Prefill {
                embedded,
                layout,
                speaker,
                conditioning,
                ..
            } => {
                validate_embedded(embedded, d.character.output_width, conditioning.as_ref())?;
                if embedded.shape()[..2] != [layout.batch(), layout.frames()]
                    || (matches!(speaker, SpeakerPrompt::Aria)
                        && layout.frames() != d.speaker_frames)
                {
                    return Err(TtsError::Shape(
                        "speaker prompt and embedded feedback geometry disagree",
                    ));
                }
                embedded.clone()
            }
            ConditioningInput::Decode {
                embedded,
                conditioning,
                ..
            } => {
                validate_embedded(embedded, d.character.output_width, conditioning.as_ref())?;
                embedded.clone()
            }
            ConditioningInput::Fuse(audio, text) => {
                f32_shape(audio, d.character.output_width)?;
                f32_shape(text, d.character.output_width)?;
                if audio.shape() != text.shape() {
                    return Err(TtsError::Shape("audio/text fusion shape mismatch"));
                }
                audio.clone()
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: ConditioningForward { input, lanes: None },
        })
    }
}
fn conditioning_graph() -> Result<ExecutionGraph, TtsError> {
    Ok(ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(CHARACTER),
            ExecutionGroupSpec::with_dependencies(INGRESS, [CHARACTER]),
            ExecutionGroupSpec::with_dependencies(FUSION, [INGRESS]),
        ],
        FUSION,
    )?)
}
/// Cold canonical parameter shapes, partitioned by the exact learned phase.
pub fn conditioning_parameter_description(
    d: ConditioningDimensions,
) -> Result<ArchitectureParameterDescription, TtsError> {
    ConditioningDimensions::new(d.character, d.speaker_frames)?;
    let graph = conditioning_graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [1, 1, 1])?;
    let c = d.character;
    let w = c.output_width as usize;
    let h = c.width as usize;
    let member = |name: &str, shape: Vec<usize>, sharding| {
        ParameterMemberSpec::new(format!("tts_model.tts_model.{name}"), shape, sharding)
    };
    let mut expected = Vec::new();
    let mut owned = Vec::new();
    let mut push = |unit: usize, spec: ParameterGroupSpec| {
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(
                layout.group_id(unit).expect("declared group").clone(),
                0,
            ),
            spec.clone(),
        ));
        expected.push(spec);
    };
    push(
        0,
        ParameterGroupSpec::partitioned(
            format!("{CHARACTER}.tokens"),
            ParameterRole::Channels,
            h,
            [member(
                "embed_subword.embed_tokens.weight",
                vec![c.characters as usize + 1, h],
                MemberSharding::Partitioned { axis: 1 },
            )],
        )?,
    );
    push(
        0,
        ParameterGroupSpec::partitioned(
            format!("{CHARACTER}.projection"),
            ParameterRole::ColumnProjection,
            w,
            [member(
                "embed_subword.proj_embedding.weight",
                vec![w, h],
                MemberSharding::Partitioned { axis: 0 },
            )],
        )?,
    );
    let mut flags = Vec::new();
    if c.use_continuation {
        flags.push(member(
            "embed_subword.subword_flag_emb.cont_emb.weight",
            vec![2, h],
            MemberSharding::Replicated,
        ));
    }
    if c.use_special {
        flags.push(member(
            "embed_subword.bos_eos_emb.special_emb.weight",
            vec![3, h],
            MemberSharding::Replicated,
        ));
    }
    if !flags.is_empty() {
        push(
            0,
            ParameterGroupSpec::new(
                format!("{CHARACTER}.flags"),
                ParameterRole::Replicated,
                flags,
            )?,
        );
    }
    push(
        1,
        ParameterGroupSpec::partitioned(
            format!("{INGRESS}.projection"),
            ParameterRole::ColumnProjection,
            w,
            [member(
                "audio_prompt_projection_W",
                vec![w, w],
                MemberSharding::Partitioned { axis: 1 },
            )],
        )?,
    );
    push(
        1,
        ParameterGroupSpec::new(
            format!("{INGRESS}.controls"),
            ParameterRole::Replicated,
            [
                member("bos_emb", vec![w], MemberSharding::Replicated),
                member("null_emb", vec![w], MemberSharding::Replicated),
                ParameterMemberSpec::new(
                    "tts_model.audio_prompt_latents.Aria",
                    vec![1, d.speaker_frames as usize, w],
                    MemberSharding::Replicated,
                ),
            ],
        )?,
    );
    let mut projections = Vec::new();
    for prefix in ["audio_proj", "text_proj"] {
        projections.push(member(
            &format!("gated_fusion_audio_text.{prefix}.weight"),
            vec![w, w],
            MemberSharding::Partitioned { axis: 0 },
        ));
        projections.push(member(
            &format!("gated_fusion_audio_text.{prefix}.bias"),
            vec![w],
            MemberSharding::Partitioned { axis: 0 },
        ));
    }
    push(
        2,
        ParameterGroupSpec::partitioned(
            format!("{FUSION}.projections"),
            ParameterRole::ColumnProjection,
            w,
            projections,
        )?,
    );
    push(
        2,
        ParameterGroupSpec::new(
            format!("{FUSION}.controls"),
            ParameterRole::Replicated,
            [
                member(
                    "gated_fusion_audio_text.gate",
                    vec![w],
                    MemberSharding::Replicated,
                ),
                member(
                    "gated_fusion_audio_text.residual_scale",
                    vec![],
                    MemberSharding::Replicated,
                ),
                member(
                    "gated_fusion_audio_text.final_norm.weight",
                    vec![w],
                    MemberSharding::Replicated,
                ),
            ],
        )?,
    );
    Ok(ArchitectureParameterDescription::new(
        &graph, &layout, expected, owned,
    )?)
}
impl<B: NeuralBackend> ArchitectureParameters<B> for ConditioningArchitecture<B> {
    type DefinitionError = TtsError;
    fn state_layout(&self) -> Result<StateLayout, TtsError> {
        conditioning_state_layout()
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, TtsError> {
        if state.global_layer_offset() != 0 || *state.layout() != conditioning_state_layout()? {
            return Err(TtsError::Configuration(
                "conditioning stateless layout differs from selected identity",
            ));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "nemotron_voicechat",
            "voicechat_conditioning",
            format!("{:?}", self.dimensions),
            1,
            0,
            0,
            topology,
        )?)
    }
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, TtsError> {
        conditioning_parameter_description(self.dimensions)
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
impl<B: NeuralBackend> LayeredArchitecture<B, ConditioningRuntimeState>
    for ConditioningArchitecture<B>
{
    type Input<'a> = ConditioningInput<B::Tensor>;
    type StaticModules = ConditioningStaticModules;
    type Unit = ConditioningUnit<B>;
    type ForwardContext = ConditioningForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = TtsError;
    fn group_transport(&self, _: usize) -> ArchitectureGroupTransport {
        ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Projector,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }
    }
    fn primary_execution_group(&self) -> &str {
        CHARACTER
    }
    fn state_partition_plan(&self, _: &StateLayout) -> ArchitectureStatePartitionPlan {
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::output_owner(0..1)])
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, TtsError> {
        self.execution_graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, TtsError> {
        self.group_unit_count(group)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, TtsError> {
        self.unit_path(group, index)
    }
    fn static_modules(&self) -> &ConditioningStaticModules {
        &self.empty
    }
    fn static_modules_mut(&mut self) -> &mut ConditioningStaticModules {
        &mut self.empty
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningUnit<B>, TtsError> {
        self.build_unit(group, index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut ConditioningRuntimeState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, TtsError> {
        if self.is_parallel() {
            return Err(TtsError::Configuration(
                "sharded conditioning requires parallel execution",
            ));
        }
        self.begin_input(input, state, context)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut ConditioningRuntimeState,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            (1 | 2, [one]) => Ok((*one).clone()),
            _ => Err(TtsError::Configuration(
                "conditioning graph dependency mismatch",
            )),
        }
    }
    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == forward.input.group()
    }
    fn retained_state_ordinals(&self, _: usize, _: usize, _: usize) -> std::ops::Range<usize> {
        0..0
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut ConditioningUnit<B>,
        _: &B::Tensor,
        _: &mut ConditioningRuntimeState,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.unit_path(group, index)?;
        if self.is_parallel() {
            return Err(TtsError::Configuration(
                "sharded conditioning requires parallel execution",
            ));
        }
        match (&forward.input, unit) {
            (ConditioningInput::CharacterEmbed(batch), ConditioningUnit::Character(character)) => {
                character.embed(batch, context)
            }
            (
                ConditioningInput::CharacterFinish(batch, hidden),
                ConditioningUnit::Character(character),
            ) => character.finish(batch, hidden, context),
            (
                ConditioningInput::Prefill {
                    embedded,
                    layout,
                    speaker,
                    conditioning,
                    guidance,
                },
                ConditioningUnit::Ingress(ingress),
            ) => {
                let lanes = ingress.prefill_embedded(
                    embedded,
                    layout,
                    *speaker,
                    conditioning.as_ref(),
                    *guidance,
                    context,
                )?;
                let audio = lanes.audio.clone();
                forward.lanes = Some(lanes);
                Ok(audio)
            }
            (
                ConditioningInput::Decode {
                    embedded,
                    conditioning,
                    guidance,
                },
                ConditioningUnit::Ingress(ingress),
            ) => {
                let lanes = ingress.decode_embedded(
                    embedded.clone(),
                    conditioning.as_ref(),
                    *guidance,
                    context,
                )?;
                let audio = lanes.audio.clone();
                forward.lanes = Some(lanes);
                Ok(audio)
            }
            (ConditioningInput::Fuse(audio, text), ConditioningUnit::Fusion(fusion)) => {
                fusion.forward(audio, text, context)
            }
            _ => Err(TtsError::Configuration(
                "leased conditioning unit differs from selected phase",
            )),
        }
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _: &mut ConditioningRuntimeState,
        _: &Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        Ok(hidden.clone())
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut roots = Vec::with_capacity(4);
        match &forward.input {
            ConditioningInput::CharacterEmbed(_) => {}
            ConditioningInput::CharacterFinish(_, hidden) => roots.push(hidden),
            ConditioningInput::Prefill {
                embedded,
                conditioning,
                ..
            }
            | ConditioningInput::Decode {
                embedded,
                conditioning,
                ..
            } => {
                roots.push(embedded);
                roots.extend(conditioning.as_ref());
            }
            ConditioningInput::Fuse(audio, text) => {
                roots.push(audio);
                roots.push(text);
            }
        }
        if let Some(lanes) = &forward.lanes {
            roots.push(&lanes.audio);
            roots.push(&lanes.text);
        }
        roots.into_iter()
    }
}
/// Conditioning using the existing prepared parameter provider and bounded traversal.
pub struct LayerwiseConditioning<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, ConditioningUnit<B>>,
{
    runtime: LayerwiseRuntime<ConditioningArchitecture<B>, B, ConditioningRuntimeState, P>,
}
impl<B, P> LayerwiseConditioning<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, ConditioningUnit<B>>,
    P::Error: std::fmt::Display,
{
    /// Adopts an exact conditioning unit policy.
    pub fn new(architecture: ConditioningArchitecture<B>, policy: P) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Existing shared provider/runtime for diagnostics and parameter inspection.
    pub fn runtime(
        &self,
    ) -> &LayerwiseRuntime<ConditioningArchitecture<B>, B, ConditioningRuntimeState, P> {
        &self.runtime
    }
    /// Existing shared provider/runtime for bounded parameter operations.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<ConditioningArchitecture<B>, B, ConditioningRuntimeState, P> {
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

/// Resident owner of the same three phase units and graph.
pub struct ResidentConditioning<B: NeuralBackend> {
    runtime: ResidentRuntime<ConditioningArchitecture<B>, B, ConditioningRuntimeState>,
}
impl<B: NeuralBackend> ResidentConditioning<B> {
    /// Constructs the three unloaded learned units exactly once.
    pub fn new(
        dimensions: ConditioningDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Self::from_parts(
            dimensions,
            CharacterEmbedding::new(dimensions.character, context)?,
            TtsIngress::new(
                dimensions.character.output_width,
                dimensions.speaker_frames,
                context,
            )?,
            GatedFusion::new(dimensions.character.output_width, context)?,
        )
    }
    /// Adopts previously materialized modules into ordinary resident traversal.
    pub fn from_parts(
        dimensions: ConditioningDimensions,
        character: CharacterEmbedding<B>,
        ingress: TtsIngress<B>,
        fusion: GatedFusion<B>,
    ) -> Result<Self, TtsError> {
        let c = dimensions.character;
        if character.geometry() != (c, c.width, c.output_width)
            || ingress.geometry() != (c.output_width, dimensions.speaker_frames, c.output_width)
            || fusion.geometry() != (c.output_width, c.output_width)
        {
            return Err(TtsError::Configuration(
                "resident conditioning units differ from complete geometry",
            ));
        }
        let architecture = ConditioningArchitecture::new(dimensions)?;
        let runtime = ResidentRuntime::from_units(
            architecture,
            vec![
                vec![ConditioningUnit::Character(character)],
                vec![ConditioningUnit::Ingress(ingress)],
                vec![ConditioningUnit::Fusion(fusion)],
            ],
        )
        .map_err(|error| match error {
            eredu_runtime::ResidentUnitsError::Architecture(error) => error,
            _ => TtsError::Configuration("conditioning resident unit count mismatch"),
        })?;
        Ok(Self { runtime })
    }
}
impl<B: NeuralBackend> std::fmt::Debug for ResidentConditioning<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentConditioning")
            .field("dimensions", &self.runtime.architecture().dimensions())
            .finish()
    }
}
impl<B: NeuralBackend> Parameterized<B::Tensor> for ResidentConditioning<B> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitor<'a, B::Tensor>,
    {
        for group in self.runtime.units() {
            for unit in group {
                unit.visit_parameters(visitor);
            }
        }
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>,
    {
        for group in self.runtime.units_mut() {
            for unit in group {
                unit.visit_parameters_mut(visitor);
            }
        }
    }
    fn set_trainable(&mut self, value: bool) {
        for group in self.runtime.units_mut() {
            for unit in group {
                unit.set_trainable(value);
            }
        }
    }
}

macro_rules! conditioning_methods {
    ($map:expr) => {
        fn dimensions(&self) -> ConditioningDimensions {
            self.runtime.architecture().dimensions()
        }
        fn character_embed(
            &mut self,
            batch: &CharacterBatch,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, TtsError> {
            let mut state = ConditioningRuntimeState::new()?;
            self.runtime
                .invoke(
                    ConditioningInput::CharacterEmbed(batch.clone()),
                    &mut state,
                    context,
                )
                .map_err($map)
        }
        fn character_finish(
            &mut self,
            batch: &CharacterBatch,
            hidden: &B::Tensor,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, TtsError> {
            let mut state = ConditioningRuntimeState::new()?;
            self.runtime
                .invoke(
                    ConditioningInput::CharacterFinish(batch.clone(), hidden.clone()),
                    &mut state,
                    context,
                )
                .map_err($map)
        }
        fn prefill_embedded(
            &mut self,
            embedded: &B::Tensor,
            layout: &PromptLayout,
            speaker: SpeakerPrompt,
            conditioning: Option<&B::Tensor>,
            guidance: bool,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
            let mut state = ConditioningRuntimeState::new()?;
            self.runtime
                .invoke_with_context(
                    ConditioningInput::Prefill {
                        embedded: embedded.clone(),
                        layout: layout.clone(),
                        speaker,
                        conditioning: conditioning.cloned(),
                        guidance,
                    },
                    &mut state,
                    context,
                )
                .map_err($map)?
                .1
                .into_lanes()
        }
        fn decode_embedded(
            &mut self,
            embedded: &B::Tensor,
            conditioning: Option<&B::Tensor>,
            guidance: bool,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
            let mut state = ConditioningRuntimeState::new()?;
            self.runtime
                .invoke_with_context(
                    ConditioningInput::Decode {
                        embedded: embedded.clone(),
                        conditioning: conditioning.cloned(),
                        guidance,
                    },
                    &mut state,
                    context,
                )
                .map_err($map)?
                .1
                .into_lanes()
        }
        fn fuse(
            &mut self,
            audio: &B::Tensor,
            text: &B::Tensor,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, TtsError> {
            let mut state = ConditioningRuntimeState::new()?;
            self.runtime
                .invoke(
                    ConditioningInput::Fuse(audio.clone(), text.clone()),
                    &mut state,
                    context,
                )
                .map_err($map)
        }
    };
}
impl<B: NeuralBackend> ConditioningExecution<B> for ResidentConditioning<B> {
    conditioning_methods!(std::convert::identity);
}
impl<B, P> ConditioningExecution<B> for LayerwiseConditioning<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, ConditioningUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    conditioning_methods!(runtime_error);
}

impl<B: DistributedNeuralBackend> ConditioningArchitecture<B> {
    /// Checks the retained physical layout against the actual communicator before acquisition.
    pub fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), TtsError> {
        let (_, selected) = self.parallel.as_ref().ok_or(TtsError::Configuration(
            "parallel conditioning requires rank-local construction",
        ))?;
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(B::parallel_size(parallel), 1, 1, 1)
                .map_err(|_| TtsError::Configuration("invalid conditioning tensor degree"))?,
            B::parallel_rank(parallel),
        )
        .map_err(|_| TtsError::Configuration("invalid conditioning tensor rank"))?;
        let expected = eredu_runtime::derive_partitioned_local_layout(
            &conditioning_parameter_description(self.dimensions)?,
            topology,
        )
        .map_err(|_| TtsError::Configuration("conditioning placement cannot be derived"))?;
        if selected != &expected {
            return Err(TtsError::Configuration(
                "conditioning layout differs from tensor communicator",
            ));
        }
        Ok(())
    }
}

fn gather_features<B: DistributedNeuralBackend>(
    local: B::Tensor,
    start: i32,
    width: i32,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, TtsError> {
    let mut shape = local.shape().to_vec();
    let axis = shape.len().checked_sub(1).ok_or(TtsError::Shape(
        "conditioning projection must have feature axis",
    ))?;
    let end = start + shape[axis];
    if start < 0 || end > width {
        return Err(TtsError::Shape(
            "conditioning projection exceeds feature partition",
        ));
    }
    let mut pieces = Vec::with_capacity(3);
    if start > 0 {
        shape[axis] = start;
        pieces.push(B::Tensor::full_f32(0.0, &shape, context)?);
    }
    pieces.push(local);
    if end < width {
        shape[axis] = width - end;
        pieces.push(B::Tensor::full_f32(0.0, &shape, context)?);
    }
    Ok(B::sum_parallel(
        B::Tensor::concatenate(&pieces, -1, context)?,
        parallel,
        context,
    )?)
}

impl<B: DistributedNeuralBackend> ParallelLayeredArchitecture<B, ConditioningRuntimeState>
    for ConditioningArchitecture<B>
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut ConditioningRuntimeState,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, TtsError> {
        self.validate_parallel(parallel)?;
        self.begin_input(input, state, context)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        _: &B::Tensor,
        _: &mut ConditioningRuntimeState,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.unit_path(group, index)?;
        let local = self.local();
        let gather = |value| {
            gather_features::<B>(
                value,
                local.output_start,
                self.dimensions.character.output_width,
                parallel,
                context,
            )
        };
        match (&forward.input, unit) {
            (ConditioningInput::CharacterEmbed(batch), ConditioningUnit::Character(character)) => {
                gather_features::<B>(
                    character.embed(batch, context)?,
                    local.character_start,
                    self.dimensions.character.width,
                    parallel,
                    context,
                )
            }
            (
                ConditioningInput::CharacterFinish(batch, hidden),
                ConditioningUnit::Character(character),
            ) => character.finish_with_projection(batch, hidden, context, gather),
            (
                ConditioningInput::Prefill {
                    embedded,
                    layout,
                    speaker,
                    conditioning,
                    guidance,
                },
                ConditioningUnit::Ingress(ingress),
            ) => {
                let lanes = ingress.prefill_with_projection(
                    embedded,
                    layout,
                    *speaker,
                    conditioning.as_ref(),
                    *guidance,
                    context,
                    gather,
                )?;
                let audio = lanes.audio.clone();
                forward.lanes = Some(lanes);
                Ok(audio)
            }
            (
                ConditioningInput::Decode {
                    embedded,
                    conditioning,
                    guidance,
                },
                ConditioningUnit::Ingress(ingress),
            ) => {
                let lanes = ingress.decode_embedded(
                    embedded.clone(),
                    conditioning.as_ref(),
                    *guidance,
                    context,
                )?;
                let audio = lanes.audio.clone();
                forward.lanes = Some(lanes);
                Ok(audio)
            }
            (ConditioningInput::Fuse(audio, text), ConditioningUnit::Fusion(fusion)) => {
                fusion.forward_with_projection(audio, text, context, gather)
            }
            _ => Err(TtsError::Configuration(
                "leased conditioning unit differs from selected phase",
            )),
        }
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut ConditioningRuntimeState,
        forward: &Self::ForwardContext,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.finish_forward(hidden, state, forward, context)
    }
}

impl<B, P> LayerwiseConditioning<B, P>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, ConditioningUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn invoke_parallel(
        &mut self,
        input: ConditioningInput<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, ConditioningForward<B::Tensor>), TtsError> {
        let mut state = ConditioningRuntimeState::new()?;
        self.runtime
            .forward_parallel_with_context_hook(input, &mut state, parallel, context, |_, _, _| {
                Ok(())
            })
            .map_err(runtime_error)
    }
    /// Looks up and gathers the complete character backbone input.
    pub fn character_embed_parallel(
        &mut self,
        batch: &CharacterBatch,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.invoke_parallel(
            ConditioningInput::CharacterEmbed(batch.clone()),
            parallel,
            context,
        )
        .map(|(output, _)| output)
    }
    /// Gathers pooled projections before scatter and replicated flag additions.
    pub fn character_finish_parallel(
        &mut self,
        batch: &CharacterBatch,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.invoke_parallel(
            ConditioningInput::CharacterFinish(batch.clone(), hidden.clone()),
            parallel,
            context,
        )
        .map(|(output, _)| output)
    }
    /// Gathers projected speaker codes before BOS substitution and guidance duplication.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_embedded_parallel(
        &mut self,
        embedded: &B::Tensor,
        layout: &PromptLayout,
        speaker: SpeakerPrompt,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
        self.invoke_parallel(
            ConditioningInput::Prefill {
                embedded: embedded.clone(),
                layout: layout.clone(),
                speaker,
                conditioning: conditioning.cloned(),
                guidance,
            },
            parallel,
            context,
        )?
        .1
        .into_lanes()
    }
    /// Creates replicated decode guidance lanes under the selected rank-local lifecycle.
    pub fn decode_embedded_parallel(
        &mut self,
        embedded: &B::Tensor,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
        self.invoke_parallel(
            ConditioningInput::Decode {
                embedded: embedded.clone(),
                conditioning: conditioning.cloned(),
                guidance,
            },
            parallel,
            context,
        )?
        .1
        .into_lanes()
    }
    /// Gathers both affine projections before the feature gate and global RMSNorm.
    pub fn fuse_parallel(
        &mut self,
        audio: &B::Tensor,
        text: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.invoke_parallel(
            ConditioningInput::Fuse(audio.clone(), text.clone()),
            parallel,
            context,
        )
        .map(|(output, _)| output)
    }
}
