use crate::lfm2::{self, block::ReplicatedBlock, declarations::BlockParameters, OperatorPolicy};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    schema::{
        CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    },
    store::TensorSelection,
    validation::ResolvedCheckpointPlan,
    LinearFormat,
};
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, DeclareLogicalParameters, Error, Index, LayerNorm,
    LayerNormalizationConstructionSpec, LinearArithmetic, LinearFormatSpec, LinearOperator,
    LinearSpec, LogicalParameterDeclaration, LogicalParameterRole, NeuralBackend,
    NormalizationConstructionSpec, NormalizationOperator, Parameter, ParameterSpec, Parameterized,
    Tensor, TensorElementType,
};
use eredu_runtime::{LayerRuntimeState, RuntimeStateComponents, StateLayout};
use serde_json::Value;

#[path = "encoder_policy.rs"]
mod policy;
pub use policy::{AudioEncoder, EncoderModule, PolicyEncoder, PreparedAudioEncoder};
use std::collections::BTreeMap;

const PLAN: &str = "LFM3 Audio speech input";
const ROOT: &str = "model.audio_encoder";
const PROJECTOR: &str = "model.audio_projector";
pub(super) const MEL_BINS: i32 = 128;
pub(super) const STRIDE: i32 = 8;

fn invalid(message: impl Into<String>) -> Error {
    Error::backend(message.into())
}

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name.into()).map_err(Error::backend)
}

#[derive(Debug, Clone)]
struct AffineSpec {
    weight: ParameterSpec,
    bias: ParameterSpec,
    shape: Vec<i32>,
}

impl AffineSpec {
    fn new(root: &str, shape: &[i32]) -> Result<Self, Error> {
        Ok(Self {
            weight: parameter(format!("{root}.weight"))?,
            bias: parameter(format!("{root}.bias"))?,
            shape: shape.to_vec(),
        })
    }
}

impl DeclareLogicalParameters for AffineSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![
            LogicalParameterDeclaration::new(
                &self.weight,
                &self.shape,
                LogicalParameterRole::LinearWeight,
            )?,
            LogicalParameterDeclaration::new(
                &self.bias,
                &self.shape[..1],
                LogicalParameterRole::LinearBias,
            )?,
        ])
    }
}

fn linear(root: &str, input: i32, output: i32) -> Result<LinearSpec, Error> {
    let affine = AffineSpec::new(root, &[output, input])?;
    Ok(LinearSpec {
        input,
        output,
        weight: affine.weight,
        bias: Some(affine.bias),
        format: LinearFormatSpec::unscaled(LinearFormat::Dense)?
            .with_arithmetic(LinearArithmetic::RowMajorAffine)?,
    })
}

/// Learned component ownership for the speech input path.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Unit {
    /// Spatial downsampling and embedding projection.
    Downsampler,
    /// One causal LFM2 block.
    Block(usize),
    /// Encoder final RMS normalization.
    Normalization,
    /// Affine LayerNorm and two language-width projections.
    Projector,
}

/// Retained speech input geometry, parameter scopes and state layout.
#[derive(Debug, Clone)]
pub struct Declaration {
    downsample: [AffineSpec; 5],
    downsample_projection: LinearSpec,
    blocks: Vec<BlockParameters>,
    final_norm: NormalizationConstructionSpec,
    projector_norm: LayerNormalizationConstructionSpec,
    projector_up: LinearSpec,
    projector_out: LinearSpec,
    layout: StateLayout,
    width: i32,
    context_length: i32,
}

impl Declaration {
    pub(in crate::lfm3) fn select_formats(
        &mut self,
        formats: &std::collections::BTreeMap<String, eredu_checkpoint::LinearFormat>,
    ) -> Result<(), Error> {
        let format = |name: &str| {
            formats
                .get(name)
                .copied()
                .unwrap_or(eredu_checkpoint::LinearFormat::Dense)
        };
        for spec in [
            &mut self.downsample_projection,
            &mut self.projector_up,
            &mut self.projector_out,
        ] {
            spec.format = crate::linear_format::project_matrix_format(
                &spec.weight,
                &spec.format,
                format(spec.weight.id.as_str()),
            )?;
        }
        self.blocks = self
            .blocks
            .iter()
            .cloned()
            .map(|block| block.select_formats(format))
            .collect::<Result<_, _>>()?;
        Ok(())
    }

    /// Normalizes the released Audio config and its typed speech encoder subconfig.
    pub fn from_config(config: &Value) -> Result<Self, Error> {
        if config["model_type"] != "lfm3_audio" {
            return Err(invalid(
                "speech encoder requires an LFM3 Audio configuration",
            ));
        }
        Self::from_speech_config(config)
    }

    pub(in crate::lfm3) fn from_speech_config(config: &Value) -> Result<Self, Error> {
        let (encoder, text, backbone, mel_key, channels_key, width_key) =
            if let Some(encoder) = config.get("audio_encoder_config") {
                (
                    encoder,
                    &config["text_config"],
                    &encoder["text_config"],
                    "num_mel_bins",
                    "conv_channels",
                    "hidden_size",
                )
            } else {
                let encoder = &config["encoder"];
                (
                    encoder,
                    &config["lfm"],
                    &encoder["lfm"],
                    "dim_in",
                    "dim_conv",
                    "dim_out",
                )
            };
        let positive = |value: &Value, key: &str| {
            value[key]
                .as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .filter(|n| *n > 0)
                .ok_or_else(|| invalid(format!("invalid LFM3 Audio dimension {key}")))
        };
        let mel_bins = positive(encoder, mel_key)?;
        let channels = positive(encoder, channels_key)?;
        let width = positive(encoder, width_key)?;
        let window = positive(encoder, "sliding_window")?;
        let output = positive(text, "hidden_size")?;
        let mut args =
            lfm2::model_args_from_config_value(backbone).map_err(Error::backend_source)?;
        if mel_bins != MEL_BINS
            || args.model_type != "lfm2"
            || args.hidden_size != width
            || args.weight_quantization.is_some()
            || args.quantized_weights.is_some()
            || args.quantized_weight_configs.is_some()
        {
            return Err(invalid(
                "unsupported LFM3 Audio encoder geometry or encoding",
            ));
        }
        let attention = AttentionPolicy::sliding(window as u32).map_err(Error::backend_source)?;
        let layers = args
            .layer_schedule
            .iter()
            .copied()
            .map(|mut policy| {
                if matches!(policy.operator, OperatorPolicy::SelfAttention(_)) {
                    policy.operator = OperatorPolicy::SelfAttention(attention);
                }
                policy
            })
            .collect();
        args.layer_schedule =
            LayerSchedule::new(args.layer_schedule.len(), layers).map_err(Error::backend_source)?;
        let blocks = (0..args.layer_schedule.len())
            .map(|i| BlockParameters::at(&args, i, &format!("{ROOT}.lfm.layers.{i}")))
            .collect::<Result<Vec<_>, _>>()?;
        let policies = blocks
            .iter()
            .map(|block| block.mixer.state_policy())
            .collect::<Result<Vec<_>, _>>()?;
        let layout = StateLayout::new(
            LayerSchedule::new(blocks.len(), policies).map_err(Error::backend_source)?,
        )
        .map_err(Error::backend_source)?;
        let flattened = channels
            .checked_mul(mel_bins / STRIDE)
            .ok_or_else(|| invalid("LFM3 Audio downsampler dimensions overflowed"))?;
        let affine = |name: &str, shape: &[i32]| {
            AffineSpec::new(&format!("{ROOT}.downsample.{name}"), shape)
        };
        let declaration = Self {
            downsample: [
                affine("conv_in", &[channels, 1, 3, 3])?,
                affine("conv_mid", &[channels, 1, 3, 3])?,
                affine("lin_mid", &[channels, channels, 1, 1])?,
                affine("conv_end", &[channels, 1, 3, 3])?,
                affine("lin_end", &[channels, channels, 1, 1])?,
            ],
            downsample_projection: linear(&format!("{ROOT}.downsample.lin_out"), flattened, width)?,
            blocks,
            final_norm: lfm2::block::normalization_spec(
                &args,
                &format!("{ROOT}.lfm.embedding_norm.weight"),
                width,
            )?,
            projector_norm: LayerNormalizationConstructionSpec {
                dimensions: width,
                epsilon: 1e-5,
                weight: Some(parameter(format!("{PROJECTOR}.layer_norm.weight"))?),
                bias: Some(parameter(format!("{PROJECTOR}.layer_norm.bias"))?),
                arithmetic: eredu_nn::LayerNormalizationArithmetic::Native,
            },
            projector_up: linear(&format!("{PROJECTOR}.linear_1"), width, output)?,
            projector_out: linear(&format!("{PROJECTOR}.linear_2"), output, output)?,
            layout,
            width,
            context_length: args.max_position_embeddings,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(declaration)
    }

    /// Layout of sliding KV and independent causal-convolution history.
    pub fn state_layout(&self) -> &StateLayout {
        &self.layout
    }

    pub(super) fn matches_gguf_execution(&self) -> bool {
        use crate::lfm2::declarations::MixerParameters;
        let window = AttentionPolicy::sliding(32).expect("valid published window");
        self.context_length == 128_000
            && self.final_norm.epsilon == 1e-5
            && self.blocks.iter().all(|block| match &block.mixer {
                MixerParameters::Attention(attention, policy) => {
                    *policy == window && attention.rotary.base == 1_000_000.0
                }
                MixerParameters::Convolution(_) => true,
            })
    }

    #[cfg(feature = "audio")]
    pub(super) fn context_length(&self) -> usize {
        self.context_length as usize
    }

    /// Logical invocations with independent checkpoint ownership.
    pub fn units(&self) -> impl Iterator<Item = Unit> {
        std::iter::once(Unit::Downsampler)
            .chain((0..self.blocks.len()).map(Unit::Block))
            .chain([Unit::Normalization, Unit::Projector])
    }

    /// Parameter references for an individual invocation.
    pub fn unit_parameters(
        &self,
        unit: Unit,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match unit {
            Unit::Downsampler => {
                let mut parameters = self.downsample.as_slice().logical_parameters()?;
                parameters.extend(self.downsample_projection.logical_parameters()?);
                Ok(parameters)
            }
            Unit::Block(i) => self
                .blocks
                .get(i)
                .ok_or_else(|| invalid("unknown LFM3 audio encoder block"))?
                .logical_parameters(),
            Unit::Normalization => self.final_norm.logical_parameters(),
            Unit::Projector => {
                let mut parameters = self.projector_norm.logical_parameters()?;
                parameters.extend(self.projector_up.logical_parameters()?);
                parameters.extend(self.projector_out.logical_parameters()?);
                Ok(parameters)
            }
        }
    }

    /// Admits speech input tensors from a complete or restricted SafeTensors catalog.
    pub fn safetensors_plan(&self) -> Result<SafetensorsCheckpointPlan, Error> {
        let tensors = self
            .logical_parameters()?
            .into_iter()
            .map(|p| {
                let name = p.parameter.id.to_string();
                let alternate = name.ends_with(".downsample.lin_out.weight").then(|| {
                    let mut shape = p.shape.clone();
                    shape.push(1);
                    shape
                });
                SafetensorsTensorConstraint::required(
                    name.clone(),
                    p.shape,
                    StoredDtypeConstraint::Floating,
                )
                .with_aliases(released_alias(&name))
                .with_alternate_shapes(alternate)
            })
            .collect();
        SafetensorsCheckpointPlan::new(PLAN, tensors, vec![], CatalogPolicy::non_strict())
            .map_err(Error::backend_source)
    }

    /// Projects admitted physical aliases onto canonical parameter destinations.
    pub fn safetensors_recipes(
        &self,
        resolution: &ResolvedCheckpointPlan,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        if resolution.identity() != PLAN {
            return Err(invalid(
                "checkpoint resolution does not belong to the LFM3 audio input path",
            ));
        }
        let mut recipes = BTreeMap::new();
        for parameter in self.logical_parameters()? {
            let name = parameter.parameter.id.to_string();
            let canonical = resolution.source_keys().contains(&name);
            let source = if canonical {
                name.clone()
            } else {
                released_alias(&name)
                    .filter(|alias| resolution.source_keys().contains(alias))
                    .ok_or_else(|| invalid(format!("no admitted LFM3 audio source for {name}")))?
            };
            let recipe = DerivedWeightRecipe::source(source, TensorSelection::Full);
            if name.ends_with(".downsample.lin_out.weight") {
                recipes.insert(
                    name,
                    DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: parameter.shape,
                    },
                );
            } else if !canonical {
                recipes.insert(name, recipe);
            }
        }
        Ok(recipes)
    }
}

fn released_alias(name: &str) -> Option<String> {
    [
        ("model.audio_encoder.downsample.", "encoder.downsample."),
        ("model.audio_encoder.lfm.", "encoder.lfm.lfm."),
        ("model.audio_projector.layer_norm.", "mlp.model.0."),
        ("model.audio_projector.linear_1.", "mlp.model.1."),
        ("model.audio_projector.linear_2.", "mlp.model.3."),
    ]
    .into_iter()
    .find_map(|(canonical, released)| {
        name.strip_prefix(canonical)
            .map(|suffix| format!("{released}{suffix}"))
    })
}

impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = Vec::new();
        for unit in self.units() {
            parameters.extend(self.unit_parameters(unit)?);
        }
        Ok(parameters)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct Affine<T: Tensor> {
    weight: Parameter<T>,
    bias: Parameter<T>,
}

fn affine_accumulation_dtype(dtype: TensorElementType) -> TensorElementType {
    match dtype {
        TensorElementType::Bf16 | TensorElementType::F16 => TensorElementType::F32,
        _ => dtype,
    }
}

impl<T: Tensor> Affine<T> {
    fn new(spec: &AffineSpec, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            weight: Parameter::unloaded(spec.weight.clone(), &spec.shape, context)?,
            bias: Parameter::unloaded(spec.bias.clone(), &spec.shape[..1], context)?,
        })
    }

    fn spatial(
        &self,
        input: &T,
        stride: i32,
        groups: i32,
        context: &T::Context,
    ) -> Result<T, Error> {
        let dtype = input
            .element_type()
            .ok_or_else(|| invalid("unknown downsampler input dtype"))?;
        let compute = affine_accumulation_dtype(dtype);
        let weight = self
            .weight
            .as_ref()
            .transpose_axes(&[0, 2, 3, 1], context)?
            .cast_float(compute, context)?;
        let padding = weight.dim(1) / 2;
        // A biased convolution rounds only after adding its bias.
        T::conv2d(
            &input.cast_float(compute, context)?,
            &weight,
            (stride, stride),
            (padding, padding),
            (1, 1),
            groups,
            context,
        )?
        .add(&self.bias.as_ref().cast_float(compute, context)?, context)?
        .cast_float(dtype, context)
    }
}

/// Spatial stride-eight reduction followed by the encoder input projection.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Downsampler<B: NeuralBackend> {
    stages: Vec<Affine<B::Tensor>>,
    projection: B::Linear,
}

impl<B: NeuralBackend> Downsampler<B> {
    /// Constructs the downsampler independently of the backbone and projector.
    pub fn new(
        declaration: &Declaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        B::require_operator_capabilities(
            "LFM3 Audio speech encoder",
            eredu_nn::NeuralOperatorCapabilities::CONV2D,
        )?;
        Ok(Self {
            stages: declaration
                .downsample
                .iter()
                .map(|s| Affine::new(s, context))
                .collect::<Result<_, _>>()?,
            projection: B::linear(declaration.downsample_projection.clone(), context)?,
        })
    }

    fn weight_dtype(&self) -> Result<TensorElementType, Error> {
        self.stages[0]
            .weight
            .as_ref()
            .element_type()
            .ok_or_else(|| invalid("LFM3 audio downsampler weight dtype is unavailable"))
    }

    /// Reduces equal-length, unpadded clips from [batch, mel, frames] to [batch, tokens, width].
    pub fn forward(
        &mut self,
        features: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if features.shape().len() != 3
            || features.dim(0) <= 0
            || features.dim(1) != MEL_BINS
            || features.dim(2) <= 0
        {
            return Err(invalid(
                "LFM3 audio downsampling requires nonempty [batch,128,frames]",
            ));
        }
        let weight_dtype = self.weight_dtype()?;
        let input = features
            .cast_float(weight_dtype, context)?
            .expand_dims(-1, context)?;
        let hidden = B::silu(self.stages[0].spatial(&input, 2, 1, context)?, context)?;
        let channels = hidden.dim(3);
        let hidden = self.stages[1].spatial(&hidden, 2, channels, context)?;
        let hidden = B::silu(self.stages[2].spatial(&hidden, 1, 1, context)?, context)?;
        let hidden = self.stages[3].spatial(&hidden, 2, channels, context)?;
        let hidden = B::silu(self.stages[4].spatial(&hidden, 1, 1, context)?, context)?;
        // Checkpoint projection flattens channel before frequency, per time step.
        let flattened = hidden.transpose_axes(&[0, 2, 3, 1], context)?.reshape(
            &[hidden.dim(0), hidden.dim(2), channels * hidden.dim(1)],
            context,
        )?;
        self.projection.forward(&flattened, context)
    }
}

/// Projects normalized encoder states to language-model width.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Projector<B: NeuralBackend> {
    norm: LayerNorm<B::Tensor>,
    up: B::Linear,
    out: B::Linear,
}

impl<B: NeuralBackend> Projector<B> {
    /// Constructs the projector independently of the encoder.
    pub fn new(
        declaration: &Declaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            norm: LayerNorm::new(declaration.projector_norm.clone(), context)?,
            up: B::linear(declaration.projector_up.clone(), context)?,
            out: B::linear(declaration.projector_out.clone(), context)?,
        })
    }

    /// Applies affine normalization, exact GELU and the final affine projection.
    pub fn forward(
        &mut self,
        encoded: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = match encoded.element_type() {
            Some(dtype @ (TensorElementType::Bf16 | TensorElementType::F16)) => {
                // Keep normalization and the learned affine in F32 until the output cast.
                let widen = |parameter: &Parameter<B::Tensor>| {
                    parameter
                        .as_ref()
                        .cast_float(TensorElementType::F32, context)
                };
                let weight = self.norm.weight.as_ref().map(widen).transpose()?;
                let bias = self.norm.bias.as_ref().map(widen).transpose()?;
                B::Tensor::layer_norm(
                    &encoded.cast_float(TensorElementType::F32, context)?,
                    weight.as_ref(),
                    bias.as_ref(),
                    self.norm.epsilon,
                    self.norm.arithmetic,
                    context,
                )?
                .cast_float(dtype, context)?
            }
            _ => self.norm.forward(encoded, context)?,
        };
        let hidden = self.up.forward(&hidden, context)?;
        self.out
            .forward(&B::Tensor::gelu(&hidden, context)?, context)
    }
}

/// Caller-owned speech encoder continuation, including the downsampler's feature overlap.
#[derive(Debug)]
pub struct EncoderState<S, T: Tensor> {
    backbone: S,
    features: Option<T>,
    frames: i32,
    offset: i32,
    tokens: i32,
    batch: i32,
    finished: bool,
}

impl<S, T> eredu_runtime::execution_control::SnapshotValue for EncoderState<S, T>
where
    S: eredu_runtime::execution_control::SnapshotValue,
    T: Tensor + eredu_runtime::execution_control::SnapshotValue,
{
    fn estimate_snapshot_value(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, eredu_core::BackendFailure> {
        eredu_runtime::execution_control::snapshot_value_estimate::<Self>([
            self.backbone.estimate_snapshot_value(),
            self.features.estimate_snapshot_value(),
        ])
    }

    fn copy_snapshot_value(
        &self,
        retention: &eredu_runtime::execution_control::SnapshotReservation,
    ) -> Result<Self, eredu_core::BackendFailure> {
        Ok(Self {
            backbone: self.backbone.copy_snapshot_value(retention)?,
            features: self.features.copy_snapshot_value(retention)?,
            ..*self
        })
    }
}

impl<S, T: Tensor> EncoderState<S, T> {
    pub(super) fn try_branch<E>(&self, branch: impl FnOnce(&S) -> Result<S, E>) -> Result<Self, E> {
        Ok(Self {
            backbone: branch(&self.backbone)?,
            features: self.features.clone(),
            ..*self
        })
    }

    pub(super) fn projected_positions(&self, frames: i32, final_chunk: bool) -> Option<usize> {
        let total = self.frames.checked_add(frames)?;
        usize::try_from(frame_tokens(total, final_chunk) - self.tokens).ok()
    }
    /// Returns the LFM2 layer-state realization.
    pub fn backbone(&self) -> &S {
        &self.backbone
    }

    /// Returns the compact feature overlap retained by the downsampler.
    pub fn feature_carry(&self) -> Option<&T> {
        self.features.as_ref()
    }

    /// Whether the stream has finalized or encountered a synchronous execution failure.
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

/// Resident speech encoder over architecture-declared components and a neutral layer state.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Encoder<B: NeuralBackend> {
    downsample: Downsampler<B>,
    blocks: Vec<ReplicatedBlock<B>>,
    norm: B::Normalization,
    #[parameter(skip)]
    declaration: Declaration,
}

impl<B: NeuralBackend> Encoder<B> {
    /// Builds resident execution from the same independently declared block scopes.
    pub fn new(
        declaration: &Declaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            downsample: Downsampler::new(declaration, context)?,
            blocks: declaration
                .blocks
                .iter()
                .cloned()
                .map(|block| ReplicatedBlock::from_parameters(block, context))
                .collect::<Result<_, _>>()?,
            norm: B::normalization(declaration.final_norm.clone(), context)?,
            declaration: declaration.clone(),
        })
    }

    fn run_blocks<S>(
        blocks: &mut [ReplicatedBlock<B>],
        norm: &mut B::Normalization,
        mut hidden: B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        for (i, block) in blocks.iter_mut().enumerate() {
            hidden = block.forward(
                &hidden,
                None,
                state.layer(i).map_err(Error::backend_source)?,
                context,
            )?;
        }
        norm.forward(&hidden, context)
    }

    /// Encodes equal-length clips with a fresh caller-supplied state.
    pub fn forward<S>(
        &mut self,
        features: &B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.declaration
            .forward_with::<B, S>(features, state, |features, state| {
                let hidden = self.downsample.forward(features, context)?;
                Self::run_blocks(&mut self.blocks, &mut self.norm, hidden, state, context)
            })
    }

    /// Creates a stream with an independent empty layer state and fixed batch.
    pub fn stream<S>(&self, backbone: S, batch: i32) -> Result<EncoderState<S, B::Tensor>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        self.declaration.stream::<B, S>(backbone, batch)
    }

    /// Advances whole stride-eight windows, using the final partial window only on flush.
    /// The caller establishes native completion before publishing this state.
    pub fn advance<S>(
        &mut self,
        input: &B::Tensor,
        state: &mut EncoderState<S, B::Tensor>,
        finalize: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let weight_dtype = self.downsample.weight_dtype()?;
        self.declaration.advance_with::<B, S>(
            input,
            state,
            finalize,
            context,
            |batch| {
                B::Tensor::full_f32(0.0, &[batch, 0, self.declaration.width], context)?
                    .cast_float(weight_dtype, context)
            },
            |features, state, range| {
                let hidden = self
                    .downsample
                    .forward(features, context)
                    .and_then(|hidden| {
                        hidden.index(
                            &[
                                Index::Full,
                                Index::Range(range.start, range.end),
                                Index::Full,
                            ],
                            context,
                        )
                    })
                    .map_err(AdvanceError::Input)?;
                Self::run_blocks(&mut self.blocks, &mut self.norm, hidden, state, context)
                    .map_err(AdvanceError::State)
            },
        )
    }
}

enum AdvanceError {
    Input(Error),
    State(Error),
}

impl Declaration {
    fn validate_state<B, S>(&self, state: &mut S, tokens: i32) -> Result<(), Error>
    where
        B: NeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        if state.optional_layout() != Some(self.state_layout()) {
            return Err(invalid("LFM3 audio state layout differs from the encoder"));
        }
        for i in 0..self.blocks.len() {
            if state.layer(i).map_err(Error::backend_source)?.position() != tokens {
                return Err(invalid("LFM3 audio state has inconsistent token frontiers"));
            }
        }
        Ok(())
    }

    fn forward_with<B, S>(
        &self,
        features: &B::Tensor,
        state: &mut S,
        run: impl FnOnce(&B::Tensor, &mut S) -> Result<B::Tensor, Error>,
    ) -> Result<B::Tensor, Error>
    where
        B: NeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        self.validate_state::<B, S>(state, 0)?;
        if features.shape().len() != 3 || frame_tokens(features.dim(2), true) > self.context_length
        {
            return Err(invalid(
                "LFM3 audio features exceed the encoder context or have invalid rank",
            ));
        }
        run(features, state)
    }

    /// Creates a stream with an independent empty layer state and fixed batch.
    fn stream<B, S>(&self, mut backbone: S, batch: i32) -> Result<EncoderState<S, B::Tensor>, Error>
    where
        B: NeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        if batch <= 0 {
            return Err(invalid("LFM3 audio encoder batch must be positive"));
        }
        self.validate_state::<B, S>(&mut backbone, 0)?;
        Ok(EncoderState {
            backbone,
            features: None,
            frames: 0,
            offset: 0,
            tokens: 0,
            batch,
            finished: false,
        })
    }

    fn advance_with<B, S>(
        &self,
        input: &B::Tensor,
        state: &mut EncoderState<S, B::Tensor>,
        finalize: bool,
        context: &<B::Tensor as Tensor>::Context,
        empty: impl FnOnce(i32) -> Result<B::Tensor, Error>,
        run: impl FnOnce(&B::Tensor, &mut S, std::ops::Range<i32>) -> Result<B::Tensor, AdvanceError>,
    ) -> Result<B::Tensor, Error>
    where
        B: NeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        if state.finished
            || input.shape().len() != 3
            || input.dim(0) != state.batch
            || input.dim(1) != MEL_BINS
        {
            return Err(invalid(
                "LFM3 audio stream is closed or the feature shape changed",
            ));
        }
        let frames = state
            .frames
            .checked_add(input.dim(2))
            .ok_or_else(|| invalid("LFM3 audio frame count overflowed"))?;
        let tokens = frame_tokens(frames, finalize);
        if tokens > self.context_length {
            return Err(invalid("LFM3 audio stream exceeds the encoder context"));
        }
        self.validate_state::<B, S>(&mut state.backbone, state.tokens)?;
        let features = match state.features.as_ref() {
            Some(carry) if input.dim(2) > 0 => {
                B::Tensor::concatenate(&[carry.clone(), input.clone()], 2, context)?
            }
            Some(carry) => carry.clone(),
            None => input.clone(),
        };
        // Keep the preceding stride to cover the next convolutional receptive field.
        let offset = if finalize {
            frames
        } else {
            (tokens - 1).max(0) * STRIDE
        };
        let carry = if !finalize && offset < frames {
            Some(
                features
                    .index(
                        &[
                            Index::Full,
                            Index::Full,
                            Index::Range(offset - state.offset, features.dim(2)),
                        ],
                        context,
                    )?
                    .compact(context)?,
            )
        } else {
            None
        };
        let count = tokens - state.tokens;
        let output = if count == 0 {
            empty(state.batch)?
        } else {
            let start = state.tokens - state.offset / STRIDE;
            match run(&features, &mut state.backbone, start..start + count) {
                Ok(output) => output,
                Err(AdvanceError::Input(error)) => return Err(error),
                Err(AdvanceError::State(error)) => {
                    state.finished = true;
                    return Err(error);
                }
            }
        };
        state.frames = frames;
        state.tokens = tokens;
        state.offset = offset;
        state.features = carry;
        state.finished = finalize;
        Ok(output)
    }
}

pub(in crate::lfm3) fn frame_tokens(frames: i32, finalize: bool) -> i32 {
    frames / STRIDE + i32::from(finalize && frames % STRIDE != 0)
}
