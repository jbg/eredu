//! EAR-TTS's eight-slot, 31-codebook refinement transition.
//!
//! This is a neural component transition, not a session scheduler. Causal Gemma
//! state, control-token/EOS policy, codec overlap and event publication are
//! separate owners. Each step creates new codes, leaving its parent untouched.

use super::{
    config::TtsConfig,
    tts_mog::{f32_shape, linear, parameter, TtsError},
};
use eredu_codec::prvq::{quantize_residual_with_norms_and_score_reduction, DistanceMetric};
use eredu_nn::{
    DistributedNeuralBackend, Index, LinearOperator, NeuralBackend, Parameter, Parameterized,
    Tensor, TensorElementType,
};

/// Released cubic masking schedule: zero-depth slots consume no random draws.
pub const REFINEMENT_DEPTHS: [usize; 8] = [0, 0, 0, 1, 1, 3, 4, 22];

/// Sampling policy for the released, EOS-head-disabled refinement component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefinementPolicy {
    /// Positive values require a separate unconditional hidden lane.
    pub guidance_scale: f32,
    /// Standard deviation multiplier for the Gaussian latent perturbation.
    pub noise_scale: f32,
    /// Nucleus probability supplied to the shared sampler.
    pub top_p: f64,
}
impl RefinementPolicy {
    /// Checks the released schedule and reads its normalized sampling defaults.
    pub fn from_config(config: &TtsConfig) -> Result<Self, TtsError> {
        if config.num_iterations != 8
            || config.num_quantizers != 31
            || config.exponent != 3.0
            || config.random_target_masking
            || !config.disable_eos_prediction
        {
            return Err(TtsError::Configuration(
                "refinement requires the released eight-slot schedule and disabled binary EOS head",
            ));
        }
        let policy = Self {
            guidance_scale: if config.guidance_enabled {
                config.guidance_scale as f32
            } else {
                0.0
            },
            noise_scale: config.noise_scale as f32,
            top_p: config.top_p,
        };
        policy.validate()?;
        Ok(policy)
    }
    pub(super) fn validate(self) -> Result<(), TtsError> {
        if !self.guidance_scale.is_finite()
            || self.guidance_scale < 0.0
            || !self.noise_scale.is_finite()
            || self.noise_scale < 0.0
            || !self.top_p.is_finite()
            || !(0.0..=1.0).contains(&self.top_p)
        {
            return Err(TtsError::Configuration(
                "invalid refinement sampling policy",
            ));
        }
        Ok(())
    }
}

/// Retained partial frame; the next active slot and complete code payload fork together.
#[derive(Debug, Clone)]
pub struct RefinementState<T> {
    pub(super) codes: T,
    pub(super) slot: usize,
    pub(super) depth: usize,
    pub(super) cardinality: i32,
    pub(super) latent: i32,
}
impl<T> RefinementState<T> {
    /// `[batch, time, 31]` codes; unpredicted depths contain the PAD sentinel.
    pub fn codes(&self) -> &T {
        &self.codes
    }
    /// Next active slot, or `None` once all 31 codebooks have been predicted.
    pub fn next_slot(&self) -> Option<usize> {
        (self.slot < 8).then_some(self.slot)
    }
    /// Number of assigned codebooks.
    pub fn completed_depth(&self) -> usize {
        self.depth
    }
}

/// Exact geometry of the canonical feedback table and its shared projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RvqDimensions {
    /// Number of ordinary entries in each codebook; PAD is the next integer.
    pub cardinality: i32,
    /// Latent codebook width.
    pub latent: i32,
    /// Causal-transformer feedback width.
    pub hidden: i32,
}
impl RvqDimensions {
    /// Resolves checked dimensions from normalized TTS policy.
    pub fn from_config(config: &TtsConfig) -> Result<Self, TtsError> {
        let size = |n| {
            i32::try_from(n)
                .map_err(|_| TtsError::Configuration("RVQ dimension exceeds tensor indexing"))
        };
        let result = Self {
            cardinality: size(config.codebook_size)?,
            latent: size(config.latent_size)?,
            hidden: size(config.backbone.hidden_size)?,
        };
        result.validate()?;
        Ok(result)
    }
    pub(super) fn validate(self) -> Result<(), TtsError> {
        if self.cardinality <= 0
            || self.cardinality.checked_add(1).is_none()
            || self.latent <= 0
            || self.hidden <= 0
        {
            return Err(TtsError::Configuration("invalid EAR-TTS feedback geometry"));
        }
        Ok(())
    }
}
impl<T> RefinementState<T> {
    pub(super) fn validate(&self, dimensions: RvqDimensions) -> Result<(), TtsError> {
        if self.cardinality != dimensions.cardinality
            || self.latent != dimensions.latent
            || !(3..=8).contains(&self.slot)
            || self.depth != REFINEMENT_DEPTHS[..self.slot.min(8)].iter().sum::<usize>()
        {
            return Err(TtsError::Shape(
                "refinement state belongs to different codebooks",
            ));
        }
        Ok(())
    }
}

/// Independently leased canonical stack of 31 feedback codebooks.
#[derive(Debug)]
pub struct RvqBank<B: NeuralBackend> {
    table: Parameter<B::Tensor>,
    dimensions: RvqDimensions,
    local: super::tts_rvq_layered::RvqLocalDimensions,
    squared_norms: std::cell::RefCell<Option<B::Tensor>>,
}
impl<B: NeuralBackend> Parameterized<B::Tensor> for RvqBank<B> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitor<'a, B::Tensor>,
    {
        self.table.visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>,
    {
        self.squared_norms.get_mut().take();
        self.table.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.squared_norms.get_mut().take();
        self.table.set_trainable(trainable);
    }
}
impl<B: NeuralBackend> RvqBank<B> {
    /// Declares the exact immutable checkpoint bank without a projection copy.
    pub fn new(
        dimensions: RvqDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Self::new_local(
            dimensions,
            super::tts_rvq_layered::RvqLocalDimensions {
                latent_start: 0,
                latent: dimensions.latent,
            },
            context,
        )
    }
    /// Declares a coherent latent-channel slice of all 31 codebooks.
    pub fn new_local(
        dimensions: RvqDimensions,
        local: super::tts_rvq_layered::RvqLocalDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        dimensions.validate()?;
        local.validate(dimensions)?;
        let mut spec = parameter("rvq_embs", "voicechat.tts.rvq")?;
        spec.trainable = false;
        Ok(Self {
            table: Parameter::unloaded(spec, &[31, dimensions.cardinality, local.latent], context)?,
            dimensions,
            local,
            squared_norms: std::cell::RefCell::new(None),
        })
    }
    fn squared_norms(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        if let Some(norms) = self.squared_norms.borrow().as_ref() {
            return Ok(norms.clone());
        }
        let norms = B::Tensor::sum_axis(&self.table.as_ref().square(context)?, -1, false, context)?;
        *self.squared_norms.borrow_mut() = Some(norms.clone());
        Ok(norms)
    }
    pub(super) fn geometry(&self) -> (RvqDimensions, super::tts_rvq_layered::RvqLocalDimensions) {
        (self.dimensions, self.local)
    }
    /// Sums codebook means in depth order. Only PAD (`cardinality`) maps to zero;
    /// BOS/EOS and out-of-domain codes are rejected by the ordinary strict gather.
    pub fn depthsum(
        &self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        if self.table.as_ref().element_type() != Some(TensorElementType::F32)
            || self.table.as_ref().shape() != [31, self.dimensions.cardinality, self.local.latent]
        {
            return Err(TtsError::Shape(
                "feedback codebook requires its declared F32 geometry",
            ));
        }
        if codes.shape().len() != 3
            || codes.dim(2) != 31
            || codes.dim(0) <= 0
            || codes.dim(1) <= 0
            || codes.element_type() != Some(TensorElementType::I32)
        {
            return Err(TtsError::Shape(
                "feedback requires I32 [batch, time, 31] codes",
            ));
        }
        Ok(B::ordered_codebook_lookup_sum(
            self.table.as_ref(),
            codes,
            context,
        )?)
    }

    /// Assigns the current slot's contiguous codebooks using a fresh sampled residual.
    /// Prefix codes remain unchanged and are not subtracted from this latent.
    pub fn advance(
        &self,
        state: &RefinementState<B::Tensor>,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError> {
        if self.local.latent != self.dimensions.latent {
            return Err(TtsError::Configuration(
                "sharded RVQ requires parallel quantization",
            ));
        }
        self.advance_with_reduction(state, latent, context, |_, _, value| value)
    }
    fn advance_with_reduction(
        &self,
        state: &RefinementState<B::Tensor>,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        mut transition: impl FnMut(
            u64,
            bool,
            Result<B::Tensor, TtsError>,
        ) -> Result<B::Tensor, TtsError>,
    ) -> Result<RefinementState<B::Tensor>, TtsError> {
        let mut squared_norms = None;
        let prepared = (|| {
            state.validate(self.dimensions)?;
            f32_shape(latent, self.local.latent)?;
            super::precision::require_float32(&self.table, latent).map_err(TtsError::Shape)?;
            if self.table.as_ref().shape() != [31, self.dimensions.cardinality, self.local.latent]
                || state.next_slot().is_none()
                || state.codes.shape().len() != 3
                || state.codes.dim(2) != 31
                || state.codes.element_type() != Some(TensorElementType::I32)
                || latent.shape()[..2] != state.codes.shape()[..2]
            {
                return Err(TtsError::Shape(
                    "refinement bank or sampled latent geometry differs",
                ));
            }
            squared_norms = Some(self.squared_norms(context)?);
            Ok(latent.clone())
        })();
        let mut residual = transition(0, false, prepared)?;
        let squared_norms = squared_norms.expect("agreed local preparation initializes norms");
        let slot = state.slot;
        let end = state.depth + REFINEMENT_DEPTHS[slot];
        let mut pieces = Vec::with_capacity(REFINEMENT_DEPTHS[slot] + 2);
        if state.depth != 0 {
            pieces.push(transition(
                1,
                false,
                state
                    .codes
                    .index(
                        &[
                            Index::Full,
                            Index::Full,
                            Index::Range(0, state.depth as i32),
                        ],
                        context,
                    )
                    .map_err(TtsError::from),
            )?);
        }
        for depth in state.depth..end {
            let phase = 2 + depth as u64 * 3;
            let mut depth_norm = None;
            let means = transition(
                phase,
                false,
                (|| {
                    depth_norm = Some(squared_norms.index(&[Index::At(depth as i32)], context)?);
                    Ok(self
                        .table
                        .as_ref()
                        .index(&[Index::At(depth as i32)], context)?)
                })(),
            )?;
            let stage = quantize_residual_with_norms_and_score_reduction(
                &residual,
                &means,
                &depth_norm.expect("agreed local preparation initializes depth norm"),
                DistanceMetric::CodebookNormMinusTwiceDot,
                context,
                |scores| {
                    transition(phase + 1, true, scores.map_err(TtsError::from)).map_err(|error| {
                        eredu_codec::Error::Backend(eredu_core::BackendFailure::from_error(error))
                    })
                },
            );
            let prepared = stage.map_err(TtsError::from).and_then(|stage| {
                residual = stage.residual;
                Ok(stage.codes.expand_dims(-1, context)?)
            });
            pieces.push(transition(phase + 2, false, prepared)?);
        }
        let codes = (|| {
            if end < 31 {
                pieces.push(state.codes.index(
                    &[Index::Full, Index::Full, Index::Range(end as i32, 31)],
                    context,
                )?);
            }
            Ok(B::Tensor::concatenate(&pieces, -1, context)?.compact(context)?)
        })();
        Ok(RefinementState {
            codes: transition(100, false, codes)?,
            slot: slot + 1,
            depth: end,
            cardinality: self.dimensions.cardinality,
            latent: self.dimensions.latent,
        })
    }
}

impl<B: DistributedNeuralBackend> RvqBank<B> {
    pub(super) fn advance_parallel_guarded(
        &self,
        state: &RefinementState<B::Tensor>,
        latent: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        mut transition: impl FnMut(
            u64,
            bool,
            Result<B::Tensor, TtsError>,
        ) -> Result<B::Tensor, TtsError>,
    ) -> Result<RefinementState<B::Tensor>, TtsError> {
        let prepared = (|| {
            f32_shape(latent, self.dimensions.latent)?;
            Ok(latent.index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(
                        self.local.latent_start,
                        self.local.latent_start + self.local.latent,
                    ),
                ],
                context,
            )?)
        })();
        let local = transition(101, false, prepared)?;
        self.advance_with_reduction(state, &local, context, |phase, reduce, value| {
            let value = transition(phase, reduce, value)?;
            if reduce {
                Ok(B::sum_parallel(value, parallel, context)?)
            } else {
                Ok(value)
            }
        })
    }
}

/// Single shared feedback projection, independently leased from the large bank.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RvqProjection<B: NeuralBackend> {
    embedding: B::Linear,
    #[parameter(skip)]
    dimensions: RvqDimensions,
    #[parameter(skip)]
    latent: i32,
}
impl<B: NeuralBackend> RvqProjection<B> {
    /// Declares the canonical embed_code weight exactly once.
    pub fn new(
        dimensions: RvqDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Self::new_local(dimensions, dimensions.latent, context)
    }
    /// Declares the projection columns corresponding to one local codebook slice.
    pub fn new_local(
        dimensions: RvqDimensions,
        latent: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        dimensions.validate()?;
        if latent <= 0 || latent > dimensions.latent {
            return Err(TtsError::Configuration("invalid RVQ projection shard"));
        }
        Ok(Self {
            embedding: linear::<B>(
                "embed_code",
                "voicechat.tts.embed_code",
                latent,
                dimensions.hidden,
                context,
            )?,
            dimensions,
            latent,
        })
    }
    pub(super) fn geometry(&self) -> (RvqDimensions, i32) {
        (self.dimensions, self.latent)
    }
    /// Projects the ordered bank sum using F32 execution arithmetic.
    pub fn forward(
        &mut self,
        feedback: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        f32_shape(feedback, self.latent)?;
        super::precision::require_float32(&self.embedding, feedback).map_err(TtsError::Shape)?;
        Ok(self.embedding.forward(feedback, context)?)
    }
}

/// Resident RVQ execution through the same bank/projection traversal as bounded providers.
pub struct RvqRefinement<B: NeuralBackend> {
    runtime: eredu_runtime::ResidentRuntime<
        eredu_runtime::ComponentAdapter<
            eredu_runtime::ComponentDefinition,
            super::tts_rvq_layered::RvqKernel<B>,
        >,
        B,
        eredu_runtime::StatelessComponentState,
    >,
}
impl<B: NeuralBackend> std::fmt::Debug for RvqRefinement<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvqRefinement")
            .field("dimensions", &self.dimensions())
            .finish_non_exhaustive()
    }
}
impl<B: NeuralBackend> Parameterized<B::Tensor> for RvqRefinement<B> {
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
    fn set_trainable(&mut self, trainable: bool) {
        for group in self.runtime.units_mut() {
            for unit in group {
                unit.set_trainable(trainable);
            }
        }
    }
}
impl<B: NeuralBackend> RvqRefinement<B> {
    /// Constructs unloaded codebook and projection units without source access.
    pub fn new(
        cardinality: i32,
        latent: i32,
        hidden: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Ok(Self {
            runtime: eredu_runtime::ResidentRuntime::new(
                super::tts_rvq_layered::RvqKernel::new(RvqDimensions {
                    cardinality,
                    latent,
                    hidden,
                })?
                .into_component()?,
                context,
            )?,
        })
    }
    /// Adopts independently materialized codebook and shared projection units.
    pub(super) fn from_parts(
        dimensions: RvqDimensions,
        bank: RvqBank<B>,
        projection: RvqProjection<B>,
    ) -> Result<Self, TtsError> {
        use super::tts_rvq_layered::{RvqKernel, RvqUnit};
        Ok(Self {
            runtime: eredu_runtime::ResidentRuntime::from_units(
                RvqKernel::new(dimensions)?.into_component()?,
                vec![
                    vec![RvqUnit::Bank(bank)],
                    vec![RvqUnit::Projection(projection)],
                ],
            )
            .map_err(|error| match error {
                eredu_runtime::ResidentUnitsError::Architecture(error) => error,
                eredu_runtime::ResidentUnitsError::GroupCount { .. }
                | eredu_runtime::ResidentUnitsError::UnitCount { .. } => {
                    TtsError::Configuration("RVQ units differ from its execution graph")
                }
            })?,
        })
    }
    /// Exact selected feedback geometry.
    pub fn dimensions(&self) -> RvqDimensions {
        self.runtime.architecture().kernel().dimensions()
    }
    /// Creates an all-PAD candidate at the first active refinement slot.
    pub fn start(
        &self,
        batch: i32,
        time: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError> {
        <Self as super::tts_rvq_layered::RvqExecution<B>>::start(self, batch, time, context)
    }
    /// Sums feedback codes through the independently leased bank.
    pub fn depthsum(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        <Self as super::tts_rvq_layered::RvqExecution<B>>::depthsum(self, codes, context)
    }
    /// Sums and projects codes through the single shared embedding weight.
    pub fn embed_codes(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        <Self as super::tts_rvq_layered::RvqExecution<B>>::embed_codes(self, codes, context)
    }
    /// Assembles guidance input using the common refinement policy.
    pub fn input(
        &mut self,
        state: &RefinementState<B::Tensor>,
        conditional: &B::Tensor,
        unconditional: Option<&B::Tensor>,
        policy: RefinementPolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        <Self as super::tts_rvq_layered::RvqExecution<B>>::input(
            self,
            state,
            conditional,
            unconditional,
            policy,
            context,
        )
    }
    /// Proposes the next ordered residual-quantization slot without changing its parent.
    pub fn advance(
        &mut self,
        state: &RefinementState<B::Tensor>,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError> {
        <Self as super::tts_rvq_layered::RvqExecution<B>>::advance(self, state, latent, context)
    }
    /// Drives the same explicit refinement transition until the frame is complete.
    #[allow(clippy::too_many_arguments)]
    pub fn generate<F, Y: super::tts_mog_layered::MogExecution<B>>(
        &mut self,
        conditional: &B::Tensor,
        unconditional: Option<&B::Tensor>,
        policy: RefinementPolicy,
        head: &mut Y,
        sample: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError>
    where
        F: super::tts_rvq_layered::RefinementSampling<B>,
    {
        <Self as super::tts_rvq_layered::RvqExecution<B>>::generate(
            self,
            conditional,
            unconditional,
            policy,
            head,
            sample,
            context,
        )
    }
    /// Advances one active slot with externally owned categorical and Gaussian draws.
    #[allow(clippy::too_many_arguments)]
    pub fn advance_sampled<F, Y: super::tts_mog_layered::MogExecution<B>>(
        &mut self,
        state: &RefinementState<B::Tensor>,
        conditional: &B::Tensor,
        unconditional: Option<&B::Tensor>,
        policy: RefinementPolicy,
        head: &mut Y,
        sample: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError>
    where
        F: super::tts_rvq_layered::RefinementSampling<B>,
    {
        <Self as super::tts_rvq_layered::RvqExecution<B>>::advance_sampled(
            self,
            state,
            conditional,
            unconditional,
            policy,
            head,
            sample,
            context,
        )
    }
    pub(super) fn execute(
        &mut self,
        input: super::tts_rvq_layered::RvqInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, super::tts_rvq_layered::RvqForward<B::Tensor>), TtsError> {
        let mut state = eredu_runtime::StatelessComponentState::new()?;
        self.runtime
            .forward_with_context(input, &mut state, context)
    }
}
impl<B: NeuralBackend> super::tts_rvq_layered::RvqExecution<B> for RvqRefinement<B> {
    fn dimensions(&self) -> RvqDimensions {
        self.dimensions()
    }
    fn depthsum(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.execute(super::tts_rvq_layered::RvqInput::Sum(codes), context)
            .map(|(output, _)| output)
    }
    fn embed_codes(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.execute(super::tts_rvq_layered::RvqInput::Embed(codes), context)
            .map(|(output, _)| output)
    }
    fn advance(
        &mut self,
        state: &RefinementState<B::Tensor>,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RefinementState<B::Tensor>, TtsError> {
        self.execute(
            super::tts_rvq_layered::RvqInput::Advance { state, latent },
            context,
        )?
        .1
        .into_state()
    }
}
