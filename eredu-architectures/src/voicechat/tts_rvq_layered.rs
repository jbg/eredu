//! Independently leased feedback bank and shared projection over ordinary traversal.

use super::{
    tts_mog::{f32_shape, TtsError},
    tts_mog_layered::MogExecution,
    tts_rvq::{RefinementPolicy, RefinementState, RvqBank, RvqDimensions, RvqProjection},
};
use eredu_core::{cache::LayerCachePolicy, BackendFailure, LayerSchedule};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Parameterized, Tensor, TensorElementType};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureParameters,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, ExecutionGraph,
    ExecutionGroupSpec, ExecutionUnitAddress, ExecutionUnitLayout, LayeredArchitecture,
    LayeredForwardState, LayeredInvocation, LayeredTraversalHook, LayerwisePolicy,
    LayerwiseRuntime, LocalModelLayout, LocalTensorLayout, MemberSharding, OwnedParameterGroupSpec,
    ParallelLayeredArchitecture, ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec,
    ParameterRole, RuntimeState, StateError, StateLayout, StaticParameterVisitor,
    StaticParameterVisitorMut, SubmissionBackend, TensorPlacement,
};
use std::marker::PhantomData;

const BANK: &str = "voicechat.tts.rvq.bank";
const PROJECTION: &str = "voicechat.tts.rvq.projection";

/// Coherent latent-channel slice shared by the bank and feedback projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RvqLocalDimensions {
    /// First globally indexed latent channel.
    pub latent_start: i32,
    /// Number of locally materialized latent channels.
    pub latent: i32,
}
impl RvqLocalDimensions {
    pub(super) fn validate(self, dimensions: RvqDimensions) -> Result<(), TtsError> {
        if self.latent_start < 0
            || self.latent <= 0
            || self
                .latent_start
                .checked_add(self.latent)
                .is_none_or(|end| end > dimensions.latent)
        {
            return Err(TtsError::Configuration("invalid RVQ latent shard"));
        }
        Ok(())
    }
}

/// Checks exact physical tensor placement and coherent bank/projection partitions.
pub fn rvq_local_dimensions(
    dimensions: RvqDimensions,
    layout: &LocalModelLayout,
) -> Result<RvqLocalDimensions, TtsError> {
    let description = rvq_parameter_description(dimensions)?;
    let range = layout
        .tensor("tts_model.tts_model.rvq_embs")
        .ok_or(TtsError::Configuration("RVQ layout omits codebook bank"))?
        .expanded_logical_range(dimensions.latent as usize)?;
    let mut expected = LocalModelLayout::default();
    for owned in description.groups() {
        let group = owned.group();
        for member in group.members() {
            let MemberSharding::Partitioned { axis } = member.sharding() else {
                return Err(TtsError::Configuration("unknown RVQ partition declaration"));
            };
            let mut shape = member.global_shape().to_vec();
            shape[*axis] = range.len();
            expected.insert(
                member.target().into(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    shape,
                    TensorPlacement::Range {
                        axis: *axis,
                        start: range.start,
                        end: range.end,
                    },
                    group.partition_units(),
                    Some(range.clone()),
                    false,
                ),
            );
        }
    }
    if layout != &expected {
        return Err(TtsError::Configuration(
            "RVQ physical layout differs from its declared partition",
        ));
    }
    let local = RvqLocalDimensions {
        latent_start: range.start as i32,
        latent: range.len() as i32,
    };
    local.validate(dimensions)?;
    Ok(local)
}

/// Exact separately materializable feedback units.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum RvqUnit<B: NeuralBackend> {
    /// Stacked canonical 31-depth codebook table.
    Bank(RvqBank<B>),
    /// Single shared embed_code projection.
    Projection(RvqProjection<B>),
}
/// All learned feedback parameters belong to execution units.
pub struct RvqStaticModules;
impl<T: 'static> Parameterized<T> for RvqStaticModules {
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

/// Shared refinement boundaries that must agree before any rank proceeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum RefinementStage {
    /// Speech predecessor validation before generator execution.
    SpeechValidation = 0,
    /// Retained generator frame geometry and policy validation.
    FrameValidation = 1,
    /// New refinement frame geometry and allocation.
    Start = 2,
    /// Whether another controlled transition is required.
    Loop = 3,
    /// Input geometry and policy before feedback parameter traversal.
    Validation = 4,
    /// Feedback embedding and guidance-lane preparation.
    Input = 5,
    /// Gaussian mixture prediction.
    Prediction = 6,
    /// Owner-supplied categorical and Gaussian decisions.
    Draw = 7,
    /// Selected Gaussian latent construction.
    Selected = 8,
    /// Ordered residual quantization and candidate state construction.
    Advance = 9,
    /// Speech output substitutions before waveform codec execution.
    RenderPreparation = 10,
    /// Candidate waveform and codec carry after codec execution.
    Render = 11,
}

/// Exact architecture-owned phase identity and geometry descriptor.
/// Integer geometry is encoded directly; floating policy uses IEEE bit patterns.
/// The first word is the next active slot, with eight denoting a complete frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementPhase {
    /// Shared transition boundary.
    pub stage: RefinementStage,
    /// Exact geometry and policy words compared by a coordination mechanism.
    pub words: Vec<u64>,
}
impl RefinementPhase {
    /// Constructs a descriptor whose words are interpreted by the architecture.
    pub fn new(stage: RefinementStage, words: Vec<u64>) -> Self {
        Self { stage, words }
    }
    pub(crate) fn at_slot(stage: RefinementStage, slot: Option<usize>) -> Self {
        Self::new(stage, vec![slot.unwrap_or(8) as u64])
    }
}

/// Sampling and agreement mechanism for the single shared refinement transition.
/// Coordinated implementations publish owner decisions and reject failed or
/// differing phases before another component traversal starts. Ordinary draw
/// callbacks implement this contract with no additional agreement operation.
pub trait RefinementSampling<B: NeuralBackend> {
    /// Supplies one mixture index and one latent Gaussian draw per batch/time row.
    fn sample(
        &mut self,
        slot: usize,
        logits: &B::Tensor,
        top_p: f64,
    ) -> Result<(B::Tensor, B::Tensor), TtsError>;
    /// Agrees exact phase identity and local success before the next operation.
    fn agree(&mut self, phase: RefinementPhase, locally_succeeded: bool) -> Result<(), TtsError>;
}
impl<B: NeuralBackend, F> RefinementSampling<B> for F
where
    F: FnMut(usize, &B::Tensor, f64) -> Result<(B::Tensor, B::Tensor), TtsError>,
{
    fn sample(
        &mut self,
        slot: usize,
        logits: &B::Tensor,
        top_p: f64,
    ) -> Result<(B::Tensor, B::Tensor), TtsError> {
        self(slot, logits, top_p)
    }
    fn agree(&mut self, _: RefinementPhase, _: bool) -> Result<(), TtsError> {
        Ok(())
    }
}

/// Borrowed sampling authority forwarded through nested shared transitions.
/// This explicit wrapper avoids overlapping callback and mutable-reference impls.
pub struct BorrowedRefinementSampling<'a, F: ?Sized>(pub &'a mut F);
impl<B: NeuralBackend, F: RefinementSampling<B> + ?Sized> RefinementSampling<B>
    for BorrowedRefinementSampling<'_, F>
{
    fn sample(
        &mut self,
        slot: usize,
        logits: &B::Tensor,
        top_p: f64,
    ) -> Result<(B::Tensor, B::Tensor), TtsError> {
        self.0.sample(slot, logits, top_p)
    }
    fn agree(&mut self, phase: RefinementPhase, locally_succeeded: bool) -> Result<(), TtsError> {
        self.0.agree(phase, locally_succeeded)
    }
}

pub(crate) fn agree_result<B: NeuralBackend, F: RefinementSampling<B> + ?Sized, T>(
    sample: &mut F,
    phase: RefinementPhase,
    result: Result<T, TtsError>,
) -> Result<T, TtsError> {
    let agreed = sample.agree(phase, result.is_ok());
    // Preserve the original local failure while still participating in agreement.
    match result {
        Err(error) => Err(error),
        Ok(value) => agreed.map(|()| value),
    }
}

fn validate_input<T: Tensor>(
    dimensions: RvqDimensions,
    state: &RefinementState<T>,
    conditional: &T,
    unconditional: Option<&T>,
    policy: RefinementPolicy,
) -> Result<(), TtsError> {
    dimensions.validate()?;
    state.validate(dimensions)?;
    policy.validate()?;
    f32_shape(conditional, dimensions.hidden)?;
    if state.next_slot().is_none()
        || state.codes.shape().len() != 3
        || state.codes.element_type() != Some(TensorElementType::I32)
        || state.codes.dim(2) != 31
        || conditional.shape()[..2] != state.codes.shape()[..2]
    {
        return Err(TtsError::Shape(
            "refinement input is complete or has incompatible hidden geometry",
        ));
    }
    if policy.guidance_scale > 0.0 {
        let uncond = unconditional.ok_or(TtsError::Shape(
            "guidance requires an unconditional hidden lane",
        ))?;
        f32_shape(uncond, dimensions.hidden)?;
        if uncond.shape() != conditional.shape() {
            return Err(TtsError::Shape("guidance hidden lanes disagree"));
        }
    }
    Ok(())
}

/// Portable feedback execution and one shared eight-slot refinement transition.
/// Parameter providers implement only the three learned operations; scheduling,
/// guidance input and explicit random draws remain shared architecture policy.
pub trait RvqExecution<B: NeuralBackend> {
    /// Exact selected feedback geometry.
    fn dimensions(&self) -> RvqDimensions;
    /// Sums codebook vectors in depth order; only PAD maps to zero.
    fn depthsum(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError>;
    /// Sums and projects codes through the single shared feedback parameter.
    fn embed_codes(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError>;
    /// Assigns the next slot's ordered codebooks from a fresh residual sample.
    fn advance(
        &mut self,
        state: &RefinementState<B::Tensor>,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError>;
    /// Creates an all-PAD candidate with the three empty scheduling slots skipped.
    fn start(
        &self,
        batch: i32,
        time: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError> {
        if batch <= 0
            || time <= 0
            || batch
                .checked_mul(time)
                .and_then(|n| n.checked_mul(31))
                .is_none()
        {
            return Err(TtsError::Shape("invalid refinement batch/time geometry"));
        }
        Ok(RefinementState {
            codes: B::Tensor::full_i32(self.dimensions().cardinality, &[batch, time, 31], context)?,
            slot: 3,
            depth: 0,
            cardinality: self.dimensions().cardinality,
            latent: self.dimensions().latent,
        })
    }
    /// Builds both MoG guidance lanes from one retained code state.
    fn input(
        &mut self,
        state: &RefinementState<B::Tensor>,
        conditional: &B::Tensor,
        unconditional: Option<&B::Tensor>,
        policy: RefinementPolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        validate_input(self.dimensions(), state, conditional, unconditional, policy)?;
        let embedded = self.embed_codes(&state.codes, context)?;
        let cond = embedded.add(conditional, context)?;
        if policy.guidance_scale > 0.0 {
            let uncond = embedded.add(unconditional.expect("validated guidance lane"), context)?;
            Ok(B::Tensor::concatenate(&[cond, uncond], 0, context)?)
        } else {
            Ok(cond)
        }
    }
    /// Repeatedly advances the same component transition, asking the shared
    /// sampler for one mixture decision and Gaussian draw per active slot.
    /// The caller retains and commits the sampler's candidate RNG state.
    #[allow(clippy::too_many_arguments)]
    fn generate<F, Y: MogExecution<B>>(
        &mut self,
        conditional: &B::Tensor,
        unconditional: Option<&B::Tensor>,
        policy: RefinementPolicy,
        head: &mut Y,
        mut sample: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError>
    where
        F: RefinementSampling<B>,
    {
        let started = (|| {
            f32_shape(conditional, self.dimensions().hidden)?;
            policy.validate()?;
            self.start(conditional.dim(0), conditional.dim(1), context)
        })();
        let mut state = agree_result::<B, _, _>(
            &mut sample,
            RefinementPhase::at_slot(RefinementStage::Start, Some(3)),
            started,
        )?;
        loop {
            sample.agree(
                RefinementPhase::at_slot(RefinementStage::Loop, state.next_slot()),
                true,
            )?;
            if state.next_slot().is_none() {
                break;
            }
            state = self.advance_sampled(
                &state,
                conditional,
                unconditional,
                policy,
                head,
                BorrowedRefinementSampling(&mut sample),
                context,
            )?;
        }
        Ok(state)
    }

    /// Proposes one active refinement slot through the same sampling transition
    /// used by ordinary generation. The caller owns candidate RNG rollback.
    #[allow(clippy::too_many_arguments)]
    fn advance_sampled<F, Y: MogExecution<B>>(
        &mut self,
        state: &RefinementState<B::Tensor>,
        conditional: &B::Tensor,
        unconditional: Option<&B::Tensor>,
        policy: RefinementPolicy,
        head: &mut Y,
        mut sample: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError>
    where
        F: RefinementSampling<B>,
    {
        let dimensions = self.dimensions();
        let mog = head.dimensions();
        let validated = validate_input(dimensions, state, conditional, unconditional, policy)
            .and_then(|()| {
                if mog.hidden != dimensions.hidden || mog.latent != dimensions.latent {
                    return Err(TtsError::Shape("refinement and MoG geometry differ"));
                }
                mog.validate()
            });
        let mut phase = RefinementPhase::at_slot(RefinementStage::Validation, state.next_slot());
        phase.words.extend([
            state.depth as u64,
            dimensions.cardinality as u64,
            dimensions.latent as u64,
            dimensions.hidden as u64,
            policy.guidance_scale.to_bits() as u64,
            policy.noise_scale.to_bits() as u64,
            policy.top_p.to_bits(),
            mog.intermediate as u64,
            mog.mixtures as u64,
            mog.rank as u64,
            mog.layers as u64,
            mog.epsilon.to_bits() as u64,
            mog.min_log_std.to_bits() as u64,
        ]);
        if validated.is_ok() {
            phase
                .words
                .extend(conditional.shape().iter().map(|&n| n as u64));
        }
        agree_result::<B, _, _>(&mut sample, phase, validated)?;
        let slot = state.next_slot().expect("validated active refinement slot");
        let input = self.input(state, conditional, unconditional, policy, context);
        let input = agree_result::<B, _, _>(
            &mut sample,
            RefinementPhase::at_slot(RefinementStage::Input, Some(slot)),
            input,
        )?;
        let prediction = head.predict(&input, policy.guidance_scale, context);
        let prediction = agree_result::<B, _, _>(
            &mut sample,
            RefinementPhase::at_slot(RefinementStage::Prediction, Some(slot)),
            prediction,
        )?;
        let draws = sample.sample(slot, &prediction.logits, policy.top_p);
        let (mixture, gaussian) = agree_result::<B, _, _>(
            &mut sample,
            RefinementPhase::at_slot(RefinementStage::Draw, Some(slot)),
            draws,
        )?;
        let sampled = head.selected(
            &prediction,
            &mixture,
            &gaussian,
            policy.noise_scale,
            context,
        );
        let sampled = agree_result::<B, _, _>(
            &mut sample,
            RefinementPhase::at_slot(RefinementStage::Selected, Some(slot)),
            sampled,
        )?;
        let advanced = self.advance(state, &sampled.latent, context);
        agree_result::<B, _, _>(
            &mut sample,
            RefinementPhase::at_slot(RefinementStage::Advance, Some(slot)),
            advanced,
        )
    }
}

/// Explicit stateless traversal owner; partial codes and RNG belong to the caller.
pub struct RvqRuntimeState {
    layout: StateLayout,
}
impl RvqRuntimeState {
    /// Constructs host-only empty state metadata.
    pub fn new() -> Result<Self, TtsError> {
        Ok(Self {
            layout: rvq_state_layout()?,
        })
    }
}
impl<B: NeuralBackend> RuntimeState<B> for RvqRuntimeState {
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
fn rvq_state_layout() -> Result<StateLayout, TtsError> {
    Ok(StateLayout::new(LayerSchedule::new(
        1,
        vec![LayerCachePolicy::NoState],
    )?)?)
}

/// Exact operation over the same independently leased parameter owners.
pub enum RvqInput<'a, T> {
    /// Ordered depth sum without hidden projection.
    Sum(&'a T),
    /// Ordered depth sum followed by the shared feedback projection.
    Embed(&'a T),
    /// Quantizes a fresh residual for the current active slot.
    Advance {
        /// Borrowed parent code state, unchanged by the invocation.
        state: &'a RefinementState<T>,
        /// Fresh sampled latent; previous codes are not subtracted.
        latent: &'a T,
    },
}
/// Input and candidate roots retained through exact parameter-unit completion.
pub struct RvqForward<T> {
    embed: bool,
    parent: Option<RefinementState<T>>,
    codes: T,
    latent: Option<T>,
    candidate: Option<RefinementState<T>>,
}
impl<T> RvqForward<T> {
    /// Consumes the candidate after a complete residual-quantization invocation.
    pub fn into_state(self) -> Result<RefinementState<T>, TtsError> {
        self.candidate.ok_or(TtsError::Configuration(
            "RVQ traversal omitted quantization",
        ))
    }
}

/// Architecture-owned bank/projection graph with operation-specific unit activity.
pub struct RvqArchitecture<B: NeuralBackend> {
    dimensions: RvqDimensions,
    parallel: Option<(RvqLocalDimensions, LocalModelLayout)>,
    empty: RvqStaticModules,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> RvqArchitecture<B> {
    /// Validates dimensions without constructing any native parameter.
    pub fn new(dimensions: RvqDimensions) -> Result<Self, TtsError> {
        dimensions.validate()?;
        Ok(Self {
            dimensions,
            parallel: None,
            empty: RvqStaticModules,
            backend: PhantomData,
        })
    }
    /// Selects exact rank-local bank and projection geometry without native allocation.
    pub fn new_parallel(
        dimensions: RvqDimensions,
        layout: &LocalModelLayout,
    ) -> Result<Self, TtsError> {
        let local = rvq_local_dimensions(dimensions, layout)?;
        let mut architecture = Self::new(dimensions)?;
        architecture.parallel = Some((local, layout.clone()));
        Ok(architecture)
    }
    /// Whether execution requires an explicitly matching tensor communicator.
    pub fn is_parallel(&self) -> bool {
        self.parallel.is_some()
    }
    fn local(&self) -> RvqLocalDimensions {
        self.parallel
            .as_ref()
            .map(|(local, _)| *local)
            .unwrap_or(RvqLocalDimensions {
                latent_start: 0,
                latent: self.dimensions.latent,
            })
    }
    fn begin_input<'a>(
        &self,
        input: RvqInput<'a, B::Tensor>,
        state: &RvqRuntimeState,
    ) -> Result<LayeredForwardState<B::Tensor, RvqForward<B::Tensor>>, TtsError> {
        if state.layout != rvq_state_layout()? {
            return Err(TtsError::Configuration("RVQ runtime layout mismatch"));
        }
        let (hidden, forward) = match input {
            RvqInput::Sum(codes) | RvqInput::Embed(codes) => {
                validate_codes(codes)?;
                (
                    codes.clone(),
                    RvqForward {
                        embed: matches!(input, RvqInput::Embed(_)),
                        parent: None,
                        codes: codes.clone(),
                        latent: None,
                        candidate: None,
                    },
                )
            }
            RvqInput::Advance { state, latent } => {
                state.validate(self.dimensions)?;
                validate_codes(state.codes())?;
                f32_shape(latent, self.dimensions.latent)?;
                if state.next_slot().is_none() || latent.shape()[..2] != state.codes().shape()[..2]
                {
                    return Err(TtsError::Shape(
                        "RVQ frame is complete or sampled latent has incompatible batch/time",
                    ));
                }
                (
                    latent.clone(),
                    RvqForward {
                        embed: false,
                        parent: Some(state.clone()),
                        codes: state.codes().clone(),
                        latent: Some(latent.clone()),
                        candidate: None,
                    },
                )
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: forward,
        })
    }
    /// Exact selected feedback geometry.
    pub fn dimensions(&self) -> RvqDimensions {
        self.dimensions
    }
    /// Stable bank-before-projection graph.
    pub fn execution_graph(&self) -> Result<ExecutionGraph, TtsError> {
        rvq_graph()
    }
    /// Each group has one separately leased parameter unit.
    pub fn group_unit_count(&self, group: usize) -> Result<usize, TtsError> {
        if group < 2 {
            Ok(1)
        } else {
            Err(TtsError::Configuration("unknown RVQ execution group"))
        }
    }
    /// Canonical parameter namespace for each unit.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, TtsError> {
        if index >= self.group_unit_count(group)? {
            return Err(TtsError::Configuration("RVQ unit exceeds group"));
        }
        Ok(if group == 0 {
            "tts_model.tts_model.rvq_embs".into()
        } else {
            "tts_model.tts_model.embed_code".into()
        })
    }
    /// Constructs only the requested unloaded unit.
    pub fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RvqUnit<B>, TtsError> {
        self.unit_path(group, index)?;
        Ok(if group == 0 {
            RvqUnit::Bank(RvqBank::new_local(self.dimensions, self.local(), context)?)
        } else {
            RvqUnit::Projection(RvqProjection::new_local(
                self.dimensions,
                self.local().latent,
                context,
            )?)
        })
    }
}
fn rvq_graph() -> Result<ExecutionGraph, TtsError> {
    Ok(ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(BANK),
            ExecutionGroupSpec::with_dependencies(PROJECTION, [BANK]),
        ],
        PROJECTION,
    )?)
}

/// Cold canonical table/projection dimensions and independent storage ownership.
pub fn rvq_parameter_description(
    d: RvqDimensions,
) -> Result<ArchitectureParameterDescription, TtsError> {
    d.validate()?;
    let graph = rvq_graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [1, 1])?;
    let bank = ParameterGroupSpec::partitioned(
        BANK,
        ParameterRole::Channels,
        d.latent as usize,
        [ParameterMemberSpec::new(
            "tts_model.tts_model.rvq_embs",
            vec![31, d.cardinality as usize, d.latent as usize],
            MemberSharding::Partitioned { axis: 2 },
        )],
    )?;
    let projection = ParameterGroupSpec::partitioned(
        PROJECTION,
        ParameterRole::RowProjection,
        d.latent as usize,
        [ParameterMemberSpec::new(
            "tts_model.tts_model.embed_code.weight",
            vec![d.hidden as usize, d.latent as usize],
            MemberSharding::Partitioned { axis: 1 },
        )],
    )?;
    let expected = vec![bank.clone(), projection.clone()];
    let owned = vec![
        OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(layout.group_id(0).expect("bank group").clone(), 0),
            bank,
        ),
        OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(
                layout.group_id(1).expect("projection group").clone(),
                0,
            ),
            projection,
        ),
    ];
    Ok(ArchitectureParameterDescription::new(
        &graph, &layout, expected, owned,
    )?)
}
impl<B: NeuralBackend> ArchitectureParameters<B> for RvqArchitecture<B> {
    type DefinitionError = TtsError;
    fn state_layout(&self) -> Result<StateLayout, TtsError> {
        rvq_state_layout()
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, TtsError> {
        if state.global_layer_offset() != 0 || *state.layout() != rvq_state_layout()? {
            return Err(TtsError::Configuration(
                "RVQ stateless layout differs from selected identity",
            ));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "nemotron_voicechat",
            "voicechat_rvq",
            format!(
                "{:?}:{:?}",
                self.dimensions,
                self.parallel.as_ref().map(|(_, layout)| layout)
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
    ) -> Result<ArchitectureParameterDescription, TtsError> {
        rvq_parameter_description(self.dimensions)
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
impl<B: NeuralBackend> LayeredArchitecture<B, RvqRuntimeState> for RvqArchitecture<B> {
    type Input<'a> = RvqInput<'a, B::Tensor>;
    type StaticModules = RvqStaticModules;
    type Unit = RvqUnit<B>;
    type ForwardContext = RvqForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Flatten<std::array::IntoIter<Option<&'a B::Tensor>, 3>>
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
        BANK
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
    fn static_modules(&self) -> &RvqStaticModules {
        &self.empty
    }
    fn static_modules_mut(&mut self) -> &mut RvqStaticModules {
        &mut self.empty
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RvqUnit<B>, TtsError> {
        self.build_unit(group, index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut RvqRuntimeState,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, TtsError> {
        if self.is_parallel() {
            return Err(TtsError::Configuration(
                "sharded RVQ requires parallel execution",
            ));
        }
        self.begin_input(input, state)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut RvqRuntimeState,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            (1, [one]) => Ok((*one).clone()),
            _ => Err(TtsError::Configuration(
                "RVQ dependencies differ from graph",
            )),
        }
    }
    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == 0 || forward.embed
    }
    fn retained_state_ordinals(&self, _: usize, _: usize, _: usize) -> std::ops::Range<usize> {
        0..0
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut RvqUnit<B>,
        hidden: &B::Tensor,
        _: &mut RvqRuntimeState,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.unit_path(group, index)?;
        if self.is_parallel() {
            return Err(TtsError::Configuration(
                "sharded RVQ requires parallel execution",
            ));
        }
        match (group, unit) {
            (0, RvqUnit::Bank(bank)) if bank.geometry() == (self.dimensions, self.local()) => {
                if let Some(parent) = &forward.parent {
                    let candidate = bank.advance(
                        parent,
                        forward.latent.as_ref().expect("quantization latent"),
                        context,
                    )?;
                    let output = candidate.codes().clone();
                    forward.candidate = Some(candidate);
                    Ok(output)
                } else {
                    bank.depthsum(&forward.codes, context)
                }
            }
            (1, RvqUnit::Projection(projection))
                if projection.geometry() == (self.dimensions, self.local().latent) =>
            {
                projection.forward(hidden, context)
            }
            _ => Err(TtsError::Configuration(
                "leased RVQ unit differs from graph address",
            )),
        }
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _: &mut RvqRuntimeState,
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
        [
            Some(&forward.codes),
            forward.latent.as_ref(),
            forward.candidate.as_ref().map(RefinementState::codes),
        ]
        .into_iter()
        .flatten()
    }
}
impl<B: DistributedNeuralBackend> RvqArchitecture<B> {
    /// Rejects a mismatched rank or communicator before parameter acquisition.
    pub fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), TtsError> {
        let (_, selected) = self.parallel.as_ref().ok_or(TtsError::Configuration(
            "parallel RVQ requires rank-local construction",
        ))?;
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(B::parallel_size(parallel), 1, 1, 1)
                .map_err(|_| TtsError::Configuration("invalid RVQ tensor degree"))?,
            B::parallel_rank(parallel),
        )
        .map_err(|_| TtsError::Configuration("invalid RVQ tensor rank"))?;
        let description = rvq_parameter_description(self.dimensions)?;
        let expected = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(|_| TtsError::Configuration("RVQ tensor placement cannot be derived"))?;
        if selected != &expected {
            return Err(TtsError::Configuration(
                "RVQ layout differs from its tensor communicator",
            ));
        }
        Ok(())
    }
}
impl<B: DistributedNeuralBackend> RvqArchitecture<B> {
    fn forward_unit_guarded(
        &self,
        group: usize,
        index: usize,
        unit: &mut RvqUnit<B>,
        hidden: &B::Tensor,
        forward: &mut RvqForward<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        agree: &mut impl FnMut(
            [u64; 8],
            bool,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<bool, TtsError>,
    ) -> Result<B::Tensor, TtsError> {
        let local = self.local();
        let valid = self.unit_path(group, index).and_then(|_| match &*unit {
            RvqUnit::Bank(bank) if group == 0 && bank.geometry() == (self.dimensions, local) => {
                Ok(())
            }
            RvqUnit::Projection(projection)
                if group == 1 && projection.geometry() == (self.dimensions, local.latent) =>
            {
                Ok(())
            }
            _ => Err(TtsError::Configuration(
                "leased RVQ unit differs from selected geometry or graph address",
            )),
        });
        if !agree(
            [
                0x525651554e495430,
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
        )? {
            return Err(valid.err().unwrap_or(TtsError::Configuration(
                "RVQ participant failed unit validation",
            )));
        }
        valid?;
        let mut gate = |phase, reduce, prepared: Result<B::Tensor, TtsError>| {
            let mut descriptor = [
                0x5256515052455030,
                group as u64,
                phase,
                u64::from(reduce),
                0,
                0,
                0,
                0,
            ];
            if reduce {
                if let Ok(value) = &prepared {
                    descriptor[4] = value.shape().len() as u64;
                    for (slot, dim) in descriptor[5..].iter_mut().zip(value.shape()) {
                        *slot = *dim as u64;
                    }
                }
            }
            if !agree(descriptor, prepared.is_ok(), context)? {
                return Err(prepared.err().unwrap_or(TtsError::Configuration(
                    "RVQ participant failed collective preparation",
                )));
            }
            prepared
        };
        match (group, unit) {
            (0, RvqUnit::Bank(bank)) if bank.geometry() == (self.dimensions, local) => {
                if let Some(parent) = &forward.parent {
                    let candidate = bank.advance_parallel_guarded(
                        parent,
                        forward.latent.as_ref().expect("quantization latent"),
                        parallel,
                        context,
                        &mut gate,
                    )?;
                    let codes = candidate.codes().clone();
                    forward.candidate = Some(candidate);
                    Ok(codes)
                } else {
                    let prepared = (|| {
                        let sum = bank.depthsum(&forward.codes, context)?;
                        if forward.embed {
                            return Ok(sum);
                        }
                        let mut pieces = Vec::with_capacity(3);
                        if local.latent_start > 0 {
                            pieces.push(B::Tensor::full_f32(
                                0.0,
                                &[sum.dim(0), sum.dim(1), local.latent_start],
                                context,
                            )?);
                        }
                        let end = local.latent_start + local.latent;
                        pieces.push(sum);
                        if end < self.dimensions.latent {
                            pieces.push(B::Tensor::full_f32(
                                0.0,
                                &[hidden.dim(0), hidden.dim(1), self.dimensions.latent - end],
                                context,
                            )?);
                        }
                        Ok(B::Tensor::concatenate(&pieces, -1, context)?)
                    })();
                    let value = gate(102, !forward.embed, prepared)?;
                    if forward.embed {
                        Ok(value)
                    } else {
                        Ok(B::sum_parallel(value, parallel, context)?)
                    }
                }
            }
            (1, RvqUnit::Projection(projection))
                if projection.geometry() == (self.dimensions, local.latent) =>
            {
                Ok(B::sum_parallel(
                    gate(103, true, projection.forward(hidden, context))?,
                    parallel,
                    context,
                )?)
            }
            _ => Err(TtsError::Configuration(
                "leased RVQ unit differs from selected geometry or graph address",
            )),
        }
    }
}
impl<B: DistributedNeuralBackend> ParallelLayeredArchitecture<B, RvqRuntimeState>
    for RvqArchitecture<B>
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut RvqRuntimeState,
        parallel: &B::ParallelContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, TtsError> {
        self.validate_parallel(parallel)?;
        self.begin_input(input, state)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut RvqRuntimeState,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.forward_unit_guarded(
            group,
            index,
            unit,
            hidden,
            forward,
            parallel,
            context,
            &mut |_, success, _| Ok(success),
        )
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut RvqRuntimeState,
        forward: &Self::ForwardContext,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.finish_forward(hidden, state, forward, context)
    }
}

fn validate_codes<T: Tensor>(codes: &T) -> Result<(), TtsError> {
    if codes.shape().len() != 3
        || codes.dim(0) <= 0
        || codes.dim(1) <= 0
        || codes.dim(2) != 31
        || codes.element_type() != Some(TensorElementType::I32)
    {
        return Err(TtsError::Shape(
            "feedback requires I32 [batch, time, 31] codes",
        ));
    }
    Ok(())
}

/// RVQ feedback using the existing prepared parameter provider and bounded traversal.
pub struct LayerwiseRvq<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RvqUnit<B>>,
{
    runtime: LayerwiseRuntime<RvqArchitecture<B>, B, RvqRuntimeState, P>,
}
impl<B, P> LayerwiseRvq<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RvqUnit<B>>,
    P::Error: std::fmt::Display,
{
    /// Adopts an exact bank/projection unit policy.
    pub fn new(architecture: RvqArchitecture<B>, policy: P) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Existing shared provider/runtime for diagnostics and parameter inspection.
    pub fn runtime(&self) -> &LayerwiseRuntime<RvqArchitecture<B>, B, RvqRuntimeState, P> {
        &self.runtime
    }
    /// Existing shared provider/runtime for bounded parameter operations.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<RvqArchitecture<B>, B, RvqRuntimeState, P> {
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
impl<B, P> RvqExecution<B> for LayerwiseRvq<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RvqUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> RvqDimensions {
        self.runtime.architecture().dimensions()
    }
    fn depthsum(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        let mut state = RvqRuntimeState::new()?;
        self.runtime
            .invoke(RvqInput::Sum(codes), &mut state, context)
            .map_err(runtime_error)
    }
    fn embed_codes(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        let mut state = RvqRuntimeState::new()?;
        self.runtime
            .invoke(RvqInput::Embed(codes), &mut state, context)
            .map_err(runtime_error)
    }
    fn advance(
        &mut self,
        state: &RefinementState<B::Tensor>,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError> {
        let mut runtime_state = RvqRuntimeState::new()?;
        self.runtime
            .invoke_with_context(
                RvqInput::Advance { state, latent },
                &mut runtime_state,
                context,
            )
            .map_err(runtime_error)?
            .1
            .into_state()
    }
}

impl<B, P> LayerwiseRvq<B, P>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RvqUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Publishes complete latent feedback from coherent channel shards.
    pub fn depthsum_parallel(
        &mut self,
        codes: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        let mut state = RvqRuntimeState::new()?;
        self.runtime
            .forward_parallel_with_context_hook(
                RvqInput::Sum(codes),
                &mut state,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map(|(output, _)| output)
            .map_err(runtime_error)
    }
    /// Reduces the shared projection's rank-local input-channel products.
    pub fn embed_codes_parallel(
        &mut self,
        codes: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        let mut state = RvqRuntimeState::new()?;
        self.runtime
            .forward_parallel_with_context_hook(
                RvqInput::Embed(codes),
                &mut state,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map(|(output, _)| output)
            .map_err(runtime_error)
    }
    /// Selects shared codes from globally reduced distances before local residual subtraction.
    pub fn advance_parallel(
        &mut self,
        state: &RefinementState<B::Tensor>,
        latent: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError> {
        let mut runtime_state = RvqRuntimeState::new()?;
        self.runtime
            .forward_parallel_with_context_hook(
                RvqInput::Advance { state, latent },
                &mut runtime_state,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map_err(runtime_error)?
            .1
            .into_state()
    }
    /// Coordinates operation identity, provider lifecycle and all local preparation
    /// before the selected tensor-parallel collectives.
    pub fn invoke_parallel_guarded<H, F>(
        &mut self,
        input: RvqInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        mut agree: F,
    ) -> Result<(B::Tensor, RvqForward<B::Tensor>), TtsError>
    where
        H: LayeredTraversalHook<B, RvqForward<B::Tensor>, TtsError> + ?Sized,
        F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, TtsError>,
    {
        let dimensions = self.dimensions();
        let (mode, codes, latent, slot, depth) = match &input {
            RvqInput::Sum(codes) => (0, *codes, None, 0, 0),
            RvqInput::Embed(codes) => (1, *codes, None, 0, 0),
            RvqInput::Advance { state, latent } => {
                (2, state.codes(), Some(*latent), state.slot, state.depth)
            }
        };
        if !agree(
            [
                0x525651494e505430,
                mode,
                dimensions.cardinality as u64,
                dimensions.latent as u64,
                dimensions.hidden as u64,
                slot as u64,
                depth as u64,
                0,
            ],
            true,
            context,
        )? {
            return Err(TtsError::Configuration(
                "RVQ participants selected different invocations",
            ));
        }
        for (kind, value) in [(0, Some(codes)), (1, latent)] {
            if let Some(value) = value {
                let mut descriptor = [
                    0x525651494e505431,
                    kind,
                    value.shape().len() as u64,
                    u64::from(value.element_type() == Some(TensorElementType::I32)),
                    0,
                    0,
                    0,
                    0,
                ];
                for (slot, dim) in descriptor[4..].iter_mut().zip(value.shape()) {
                    *slot = *dim as u64;
                }
                if !agree(descriptor, true, context)? {
                    return Err(TtsError::Configuration(
                        "RVQ participants supplied different input geometry",
                    ));
                }
            }
        }
        self.runtime
            .forward_parallel_with_unit_executor_and_traversal_hook(
                input,
                &mut RvqRuntimeState::new()?,
                parallel,
                context,
                |architecture, group, index, unit, hidden, _, forward, parallel, context| {
                    architecture.forward_unit_guarded(
                        group, index, unit, hidden, forward, parallel, context, &mut agree,
                    )
                },
                hook,
            )
            .map_err(runtime_error)
    }
}
