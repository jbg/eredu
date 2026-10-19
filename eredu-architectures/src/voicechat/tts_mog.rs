//! EAR-TTS mixture prediction from supplied causal-backbone hidden states.
//!
//! Sampling decisions are explicit inputs. The session's shared sampler owns
//! random draws and their commit boundary; these modules own the released
//! normalization, guidance and Gaussian-mixture equations.

use super::config::TtsConfig;

/// Stabilizer for both logarithms in the released Gumbel-Max draw transform.
pub const GUMBEL_EPSILON: f32 = 1e-8;
use eredu_nn::{
    DistributedNeuralBackend, Index, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationConstructionSpec, NormalizationOperator, NormalizationScale, Parameter,
    ParameterSpec, Parameterized, Tensor, TensorElementType,
};

/// Invalid geometry, draw payload or neural execution.
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    /// Shared component declaration or invocation is invalid.
    #[error(transparent)]
    Component(#[from] eredu_runtime::ComponentDefinitionError),
    /// Portable execution graph is invalid.
    #[error(transparent)]
    Graph(#[from] eredu_runtime::ExecutionGraphError),
    /// Graph units cannot map to one stable residency order.
    #[error(transparent)]
    UnitLayout(#[from] eredu_runtime::ExecutionUnitLayoutError),
    /// Runtime state geometry or indexing is invalid.
    #[error(transparent)]
    State(#[from] eredu_runtime::StateError),
    /// Ordered layer policy is invalid.
    #[error(transparent)]
    Schedule(#[from] eredu_core::LayerScheduleError),
    /// Cold parameter dimensions or placement are invalid.
    #[error(transparent)]
    Parameters(#[from] eredu_runtime::ParallelPlanError),
    /// Exact parameter ownership is invalid.
    #[error(transparent)]
    Ownership(#[from] eredu_runtime::ArchitectureParameterError),
    /// Portable state identity is invalid.
    #[error(transparent)]
    Identity(#[from] eredu_core::cache::PromptCacheError),
    /// Parameter provider or submission failed with its original source retained.
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendFailure),
    /// Explicit preparation capacity is insufficient; no input is consumed.
    #[error("TTS {resource} needs {required} elements; capacity is {maximum}")]
    Capacity {
        /// Bounded preparation resource.
        resource: &'static str,
        /// Number of elements required.
        required: usize,
        /// Caller-supplied capacity.
        maximum: usize,
    },
    /// A construction or generation policy is invalid.
    #[error("{0}")]
    Configuration(&'static str),
    /// A tensor does not belong to this component's geometry or precision.
    #[error("{0}")]
    Shape(&'static str),
    /// A neural mechanism failed, retaining its source.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
    /// Shared transformer execution or its parameter provider failed.
    #[error(transparent)]
    Transformer(#[from] super::tts_transformer_layered::TransformerError),
    /// The codec's residual quantizer rejected a tensor or failed to compute.
    #[error(transparent)]
    Codec(#[from] eredu_codec::Error),
}

/// Independently sized MoG residual stack and low-rank output bank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MogDimensions {
    /// Input hidden width.
    pub hidden: i32,
    /// Gated MLP intermediate width.
    pub intermediate: i32,
    /// Codec latent width.
    pub latent: i32,
    /// Number of Gaussian components.
    pub mixtures: i32,
    /// Selected component's low-rank width.
    pub rank: i32,
    /// Number of residual MLP layers.
    pub layers: usize,
    /// RMS normalization epsilon.
    pub epsilon: f32,
    /// Lower bound on the scalar log standard deviation.
    pub min_log_std: f32,
}
impl MogDimensions {
    /// Reads the admitted architecture configuration without opening artifacts.
    pub fn from_config(config: &TtsConfig) -> Result<Self, TtsError> {
        let size = |v| {
            i32::try_from(v)
                .map_err(|_| TtsError::Configuration("MoG dimension exceeds tensor indexing"))
        };
        let result = Self {
            hidden: size(config.backbone.hidden_size)?,
            intermediate: size(config.mog.intermediate_size)?,
            latent: size(config.latent_size)?,
            mixtures: size(config.mog.num_predictions)?,
            rank: size(config.mog.low_rank)?,
            layers: config.mog.num_layers,
            epsilon: config.mog.eps as f32,
            min_log_std: config.mog.min_log_std as f32,
        };
        result.validate()?;
        Ok(result)
    }
    pub(super) fn validate(self) -> Result<(), TtsError> {
        if self.hidden <= 0
            || self.intermediate <= 0
            || self.latent <= 0
            || self.mixtures <= 0
            || self.rank <= 0
            || self.rank >= self.latent
            || self.mixtures.checked_mul(self.rank).is_none()
            || self.layers == 0
            || self.layers > 256
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.min_log_std.is_finite()
        {
            return Err(TtsError::Configuration("invalid low-rank MoG geometry"));
        }
        Ok(())
    }
}

pub(crate) fn parameter(name: &str, group: &str) -> Result<ParameterSpec, eredu_nn::Error> {
    let mut spec = ParameterSpec::trainable(format!("tts_model.tts_model.{name}"))
        .map_err(eredu_nn::Error::backend)?;
    spec.group = Some(group.into());
    Ok(spec)
}
pub(crate) fn linear<B: NeuralBackend>(
    name: &str,
    group: &str,
    input: i32,
    output: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Linear, eredu_nn::Error> {
    B::linear(
        LinearSpec {
            input,
            output,
            weight: parameter(&format!("{name}.weight"), group)?,
            bias: None,
            format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
        },
        context,
    )
}
fn norm<B: NeuralBackend>(
    name: &str,
    group: &str,
    dimensions: MogDimensions,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Normalization, eredu_nn::Error> {
    B::normalization(
        NormalizationConstructionSpec {
            reduction: eredu_nn::NormalizationReduction::Native,
            dimensions: dimensions.hidden,
            epsilon: dimensions.epsilon,
            groups: Some(1),
            scale: NormalizationScale::LearnedOffset {
                weight: parameter(&format!("{name}.weight"), group)?,
                offset: 1.0,
            },
        },
        context,
    )
}
pub(crate) fn f32_shape<T: Tensor>(input: &T, width: i32) -> Result<(), TtsError> {
    if input.shape().len() != 3
        || input.dim(2) != width
        || input.shape().iter().any(|n| *n <= 0)
        || input.element_type() != Some(TensorElementType::F32)
    {
        return Err(TtsError::Shape(
            "EAR-TTS requires nonempty F32 [batch, time, width] tensors",
        ));
    }
    Ok(())
}

/// One separately materializable residual MLP stage.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MogBlock<B: NeuralBackend> {
    #[parameter(skip)]
    precision: super::precision::SpeechPrecision,
    pre: B::Normalization,
    gate: B::Linear,
    up: B::Linear,
    down: B::Linear,
    post: B::Normalization,
    #[parameter(skip)]
    dimensions: MogDimensions,
    #[parameter(skip)]
    intermediate: i32,
}
impl<B: NeuralBackend> MogBlock<B> {
    /// Selects projection arithmetic; parameters are supplied by the matching retained plan.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, TtsError> {
        precision.validate().map_err(TtsError::Shape)?;
        self.gate
            .set_compute_element(precision.projection_element())?;
        self.up
            .set_compute_element(precision.projection_element())?;
        self.down
            .set_compute_element(precision.projection_element())?;
        self.precision = precision;
        Ok(self)
    }

    /// Declares one canonical residual stage, independently of the large heads.
    pub fn new(
        dimensions: MogDimensions,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Self::new_local(dimensions, dimensions.intermediate, index, context)
    }
    /// Constructs one complete or rank-local intermediate projection.
    pub fn new_local(
        dimensions: MogDimensions,
        intermediate: i32,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        dimensions.validate()?;
        if intermediate <= 0 || intermediate > dimensions.intermediate {
            return Err(TtsError::Configuration("invalid MoG intermediate shard"));
        }
        if index >= dimensions.layers {
            return Err(TtsError::Configuration("MoG layer index exceeds stack"));
        }
        let name = format!("mog_head.mlp_stack.{index}");
        let group = format!("voicechat.tts.mog.layer.{index}");
        Ok(Self {
            precision: Default::default(),
            pre: norm::<B>(&format!("{name}.pre_norm"), &group, dimensions, context)?,
            gate: linear::<B>(
                &format!("{name}.mlp.gate_proj"),
                &group,
                dimensions.hidden,
                intermediate,
                context,
            )?,
            up: linear::<B>(
                &format!("{name}.mlp.up_proj"),
                &group,
                dimensions.hidden,
                intermediate,
                context,
            )?,
            down: linear::<B>(
                &format!("{name}.mlp.down_proj"),
                &group,
                intermediate,
                dimensions.hidden,
                context,
            )?,
            post: norm::<B>(&format!("{name}.post_norm"), &group, dimensions, context)?,
            dimensions,
            intermediate,
        })
    }
    /// Pre-normalized gated tanh-GELU, post-normalized before residual addition.
    pub fn forward(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        if self.intermediate != self.dimensions.intermediate {
            return Err(TtsError::Configuration(
                "sharded MoG block requires its tensor communicator",
            ));
        }
        self.validate_hidden(hidden)?;
        let projected = self.project(hidden, context)?;
        Ok(self.finish(hidden, &projected, context)?)
    }
    /// Runs local intermediate projections, reducing before post-normalization.
    pub fn forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError>
    where
        B: DistributedNeuralBackend,
    {
        let range = eredu_core::balanced_contiguous_range(
            self.dimensions.intermediate as usize,
            B::parallel_size(parallel),
            B::parallel_rank(parallel),
            false,
        )
        .map_err(|_| TtsError::Configuration("invalid MoG intermediate tensor topology"))?;
        if range.len() != self.intermediate as usize {
            return Err(TtsError::Configuration(
                "MoG intermediate shard differs from tensor topology",
            ));
        }
        self.forward_with_reduction(hidden, context, |value| {
            Ok(B::sum_parallel(value?, parallel, context)?)
        })
    }
    pub(super) fn geometry(&self) -> (MogDimensions, i32) {
        (self.dimensions, self.intermediate)
    }
    fn validate_hidden(&self, hidden: &B::Tensor) -> Result<(), TtsError> {
        f32_shape(hidden, self.dimensions.hidden)?;
        self.precision
            .require(self, hidden)
            .map_err(TtsError::Shape)
    }
    fn project(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        let normalized = self.pre.forward(hidden, context)?;
        let gate = B::gelu_approximate(self.gate.forward(&normalized, context)?, context)?;
        let activated = gate.multiply(&self.up.forward(&normalized, context)?, context)?;
        self.down.forward(&activated, context)
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        projected: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        hidden.add(&self.post.forward(projected, context)?, context)
    }
    pub(super) fn forward_with_reduction(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        reduce: impl FnOnce(Result<B::Tensor, TtsError>) -> Result<B::Tensor, TtsError>,
    ) -> Result<B::Tensor, TtsError> {
        let prepared = (|| {
            self.validate_hidden(hidden)?;
            Ok(self.project(hidden, context)?)
        })();
        let projected = reduce(prepared)?;
        Ok(self.finish(hidden, &projected, context)?)
    }
}

/// Mixture logits plus the post-stack guided hidden state used by selected readout.
#[derive(Debug, Clone)]
pub struct MogPrediction<T> {
    /// Guided hidden tensor `[batch, time, hidden]`.
    pub hidden: T,
    /// Unfiltered component logits `[batch, time, mixtures]`.
    pub logits: T,
}
/// Selected component moments and the noisy latent supplied to residual quantization.
#[derive(Debug, Clone)]
pub struct MogSample<T> {
    /// Low-rank mean scaled by the standard deviation plus the residual projection.
    pub mean: T,
    /// Clamped log standard deviation `[batch, time, 1]`.
    pub log_std: T,
    /// Mean plus the supplied Gaussian draw times standard deviation and noise scale.
    pub latent: T,
}

/// Small final-normalization and mixture-logit prediction unit.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MogPredictionHead<B: NeuralBackend> {
    norm: B::Normalization,
    logits: B::Linear,
    #[parameter(skip)]
    dimensions: MogDimensions,
}
impl<B: NeuralBackend> MogPredictionHead<B> {
    pub(super) fn dimensions(&self) -> MogDimensions {
        self.dimensions
    }
    /// Declares only the weights required to choose a mixture.
    pub fn new(
        dimensions: MogDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        dimensions.validate()?;
        let group = "voicechat.tts.mog.prediction";
        Ok(Self {
            norm: norm::<B>(
                &format!("mog_head.mlp_stack.{}", dimensions.layers),
                group,
                dimensions,
                context,
            )?,
            logits: linear::<B>(
                "mog_head.proj_logits",
                group,
                dimensions.hidden,
                dimensions.mixtures,
                context,
            )?,
            dimensions,
        })
    }
    /// Final normalization followed by guidance; the conditional batch lane is first.
    pub fn predict(
        &mut self,
        hidden: &B::Tensor,
        guidance: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogPrediction<B::Tensor>, TtsError> {
        f32_shape(hidden, self.dimensions.hidden)?;
        super::precision::require_float32(self, hidden).map_err(TtsError::Shape)?;
        if !guidance.is_finite() || guidance < 0.0 {
            return Err(TtsError::Configuration("invalid MoG guidance scale"));
        }
        if guidance > 0.0 && hidden.dim(0) % 2 != 0 {
            return Err(TtsError::Shape(
                "guidance requires paired conditional/unconditional batches",
            ));
        }
        let mut hidden = self.norm.forward(hidden, context)?;
        if guidance > 0.0 {
            let batch = hidden.dim(0) / 2;
            let cond = hidden.index(&[Index::Range(0, batch)], context)?;
            let uncond = hidden.index(&[Index::Range(batch, batch * 2)], context)?;
            hidden = cond.add(
                &cond
                    .subtract(&uncond, context)?
                    .multiply_scalar(guidance, context)?,
                context,
            )?;
        }
        let logits = self.logits.forward(&hidden, context)?;
        Ok(MogPrediction { hidden, logits })
    }
}

/// Selected-component moments, independently leased after a mixture is chosen.
/// Only selected components are multiplied; no dense mixture-by-latent output is built.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MogComponentBank<B: NeuralBackend> {
    residual: B::Linear,
    logs: B::Linear,
    mus: Parameter<B::Tensor>,
    low: Parameter<B::Tensor>,
    #[parameter(skip)]
    dimensions: MogDimensions,
    #[parameter(skip)]
    rank: i32,
}
impl<B: NeuralBackend> MogComponentBank<B> {
    /// Declares the residual, scale, means and low-rank component parameters.
    pub fn new(
        dimensions: MogDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Self::new_local(dimensions, dimensions.rank, context)
    }
    /// Constructs one coherent low-rank slice for every mixture.
    pub fn new_local(
        dimensions: MogDimensions,
        rank: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        dimensions.validate()?;
        if rank <= 0 || rank > dimensions.rank {
            return Err(TtsError::Configuration("invalid MoG component rank shard"));
        }
        let group = "voicechat.tts.mog.components";
        Ok(Self {
            residual: linear::<B>(
                "mog_head.proj_else",
                group,
                dimensions.hidden,
                dimensions.latent,
                context,
            )?,
            logs: linear::<B>("mog_head.proj_logs", group, dimensions.hidden, 1, context)?,
            mus: Parameter::unloaded(
                parameter("mog_head.proj_mus.weight", "voicechat.tts.mog.means")?,
                &[dimensions.mixtures * rank, dimensions.hidden],
                context,
            )?,
            low: Parameter::unloaded(
                parameter("mog_head.low_mat", "voicechat.tts.mog.low_rank")?,
                &[dimensions.mixtures, dimensions.latent, rank],
                context,
            )?,
            dimensions,
            rank,
        })
    }
    /// Evaluates only the supplied mixture IDs and applies explicit Gaussian noise.
    pub fn selected(
        &mut self,
        prediction: &MogPrediction<B::Tensor>,
        mixture: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogSample<B::Tensor>, TtsError> {
        if self.rank != self.dimensions.rank {
            return Err(TtsError::Configuration(
                "sharded MoG components require their tensor communicator",
            ));
        }
        let (mus, low) =
            self.selected_inputs(prediction, mixture, gaussian, noise_scale, context)?;
        let expanded = Self::expand(&prediction.hidden, &mus, &low, context)?;
        Ok(self.moments(
            &prediction.hidden,
            &expanded,
            gaussian,
            noise_scale,
            context,
        )?)
    }
    pub(super) fn geometry(&self) -> (MogDimensions, i32) {
        (self.dimensions, self.rank)
    }
    pub(super) fn selected_with_reduction(
        &mut self,
        prediction: &MogPrediction<B::Tensor>,
        mixture: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        context: &<B::Tensor as Tensor>::Context,
        reduce: impl FnOnce(Result<B::Tensor, TtsError>) -> Result<B::Tensor, TtsError>,
    ) -> Result<MogSample<B::Tensor>, TtsError> {
        let prepared = (|| {
            let (mus, low) =
                self.selected_inputs(prediction, mixture, gaussian, noise_scale, context)?;
            Ok(Self::expand(&prediction.hidden, &mus, &low, context)?)
        })();
        let expanded = reduce(prepared)?;
        Ok(self.moments(
            &prediction.hidden,
            &expanded,
            gaussian,
            noise_scale,
            context,
        )?)
    }
    fn selected_inputs(
        &self,
        prediction: &MogPrediction<B::Tensor>,
        mixture: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), TtsError> {
        let d = self.dimensions;
        f32_shape(&prediction.hidden, d.hidden)?;
        super::precision::require_float32(self, &prediction.hidden).map_err(TtsError::Shape)?;
        f32_shape(gaussian, d.latent)?;
        let shape = prediction.hidden.shape();
        if mixture.shape() != &shape[..2]
            || mixture.element_type() != Some(TensorElementType::I32)
            || gaussian.shape()[..2] != shape[..2]
        {
            return Err(TtsError::Shape(
                "MoG decisions/draws do not match hidden batch and time",
            ));
        }
        if !noise_scale.is_finite() || noise_scale < 0.0 {
            return Err(TtsError::Configuration("invalid MoG noise scale"));
        }
        let mus = self
            .mus
            .as_ref()
            .reshape(&[d.mixtures, self.rank, d.hidden], context)?
            .take_axis(mixture, 0, context)?;
        let low = self.low.as_ref().take_axis(mixture, 0, context)?;
        Ok((mus, low))
    }
    fn expand(
        hidden: &B::Tensor,
        mus: &B::Tensor,
        low: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        let ranked = B::Tensor::matmul(mus, &hidden.expand_dims(-1, context)?, context)?;
        B::Tensor::matmul(low, &ranked, context)?.squeeze_axes(&[-1], context)
    }
    fn moments(
        &mut self,
        hidden: &B::Tensor,
        expanded: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogSample<B::Tensor>, eredu_nn::Error> {
        let log_std = self
            .logs
            .forward(hidden, context)?
            .maximum_scalar(self.dimensions.min_log_std, context)?;
        let std = B::exp(log_std.clone(), context)?;
        let mean = expanded
            .multiply(&std, context)?
            .add(&self.residual.forward(hidden, context)?, context)?;
        let latent = mean.add(
            &std.multiply(gaussian, context)?
                .multiply_scalar(noise_scale, context)?,
            context,
        )?;
        Ok(MogSample {
            mean,
            log_std,
            latent,
        })
    }
}

impl<B: DistributedNeuralBackend> MogComponentBank<B> {
    /// Reduces the selected low-rank contribution before replicated scale and noise.
    /// Mixture IDs and Gaussian draws are supplied by the shared transition owner.
    pub fn selected_parallel(
        &mut self,
        prediction: &MogPrediction<B::Tensor>,
        mixture: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogSample<B::Tensor>, TtsError> {
        let range = eredu_core::balanced_contiguous_range(
            self.dimensions.rank as usize,
            B::parallel_size(parallel),
            B::parallel_rank(parallel),
            false,
        )
        .map_err(|_| TtsError::Configuration("invalid MoG component tensor topology"))?;
        if range.len() != self.rank as usize {
            return Err(TtsError::Configuration(
                "MoG component rank differs from tensor topology",
            ));
        }
        self.selected_with_reduction(
            prediction,
            mixture,
            gaussian,
            noise_scale,
            context,
            |value| Ok(B::sum_parallel(value?, parallel, context)?),
        )
    }
}

/// Resident MoG units use the same traversal and equations as bounded execution.
pub struct MogHead<B: NeuralBackend> {
    runtime: eredu_runtime::ResidentRuntime<
        eredu_runtime::ComponentAdapter<
            eredu_runtime::ComponentDefinition,
            super::tts_mog_layered::MogKernel<B>,
        >,
        B,
        eredu_runtime::StatelessComponentState,
    >,
}
impl<B: NeuralBackend> std::fmt::Debug for MogHead<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MogHead")
            .field("dimensions", &self.dimensions())
            .finish_non_exhaustive()
    }
}
impl<B: NeuralBackend> Parameterized<B::Tensor> for MogHead<B> {
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
impl<B: NeuralBackend> MogHead<B> {
    /// Constructs unloaded units without a backbone or generation engine.
    pub fn new(
        dimensions: MogDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Self::new_with_precision(dimensions, Default::default(), context)
    }
    /// Constructs units with the selected portable projection precision.
    pub fn new_with_precision(
        dimensions: MogDimensions,
        precision: super::precision::SpeechPrecision,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Ok(Self {
            runtime: eredu_runtime::ResidentRuntime::new(
                super::tts_mog_layered::MogKernel::new(dimensions)?
                    .with_precision(precision)?
                    .into_component()?,
                context,
            )?,
        })
    }
    /// Adopts separately bound layers, prediction and selected-component units.
    pub(super) fn from_parts(
        dimensions: MogDimensions,
        blocks: Vec<MogBlock<B>>,
        prediction: MogPredictionHead<B>,
        components: MogComponentBank<B>,
    ) -> Result<Self, TtsError> {
        use super::tts_mog_layered::{MogKernel, MogUnit};
        let units = vec![
            blocks.into_iter().map(MogUnit::Block).collect(),
            vec![MogUnit::Prediction(prediction)],
            vec![MogUnit::Components(components)],
        ];
        Ok(Self {
            runtime: eredu_runtime::ResidentRuntime::from_units(
                MogKernel::new(dimensions)?.into_component()?,
                units,
            )
            .map_err(|error| match error {
                eredu_runtime::ResidentUnitsError::Architecture(error) => error,
                eredu_runtime::ResidentUnitsError::GroupCount { .. }
                | eredu_runtime::ResidentUnitsError::UnitCount { .. } => {
                    TtsError::Configuration("MoG units differ from its execution graph")
                }
            })?,
        })
    }
    /// Complete connected mixture geometry.
    pub fn dimensions(&self) -> MogDimensions {
        self.runtime.architecture().kernel().dimensions()
    }
    /// Runs all residual layers and small prediction head, leaving component weights idle.
    pub fn predict(
        &mut self,
        hidden: &B::Tensor,
        guidance: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogPrediction<B::Tensor>, TtsError> {
        use super::tts_mog_layered::MogInput;
        use eredu_runtime::StatelessComponentState;
        let mut state = StatelessComponentState::new()?;
        let (_, forward) = self.runtime.forward_with_context(
            MogInput::Predict { hidden, guidance },
            &mut state,
            context,
        )?;
        forward.into_prediction()
    }
    /// Uses the retained prediction without rerunning residual or logit units.
    pub fn selected(
        &mut self,
        prediction: &MogPrediction<B::Tensor>,
        mixture: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogSample<B::Tensor>, TtsError> {
        use super::tts_mog_layered::MogInput;
        use eredu_runtime::StatelessComponentState;
        let mut state = StatelessComponentState::new()?;
        let (_, forward) = self.runtime.forward_with_context(
            MogInput::Selected {
                prediction,
                mixture,
                gaussian,
                noise_scale,
            },
            &mut state,
            context,
        )?;
        forward.into_sample()
    }
}
impl<B: NeuralBackend> super::tts_mog_layered::MogExecution<B> for MogHead<B> {
    fn dimensions(&self) -> MogDimensions {
        self.dimensions()
    }
    fn predict(
        &mut self,
        hidden: &B::Tensor,
        guidance: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogPrediction<B::Tensor>, TtsError> {
        self.predict(hidden, guidance, context)
    }
    fn selected(
        &mut self,
        prediction: &MogPrediction<B::Tensor>,
        mixture: &B::Tensor,
        gaussian: &B::Tensor,
        noise_scale: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MogSample<B::Tensor>, TtsError> {
        self.selected(prediction, mixture, gaussian, noise_scale, context)
    }
}
