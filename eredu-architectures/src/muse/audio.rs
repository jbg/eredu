//! Portable causal waveform pyramid and caller-owned convolution carry.

use std::collections::HashMap;

use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    CausalDepthwiseConvolution, DeclareLogicalParameters, Error, LinearOperator, NeuralBackend,
    NormalizationOperator, Parameterized, Tensor,
};
use serde::Deserialize;

pub mod execution;
pub(crate) mod specs;

/// One independently materialized operation in the waveform pyramid.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AudioUnitKind {
    /// Time reduction and normalization at the start of one stage.
    Patch {
        /// Stage index in the released reduction schedule.
        stage: usize,
    },
    /// One causal convolution and its residual MLP.
    Convolution {
        /// Owning patch-reduction stage.
        stage: usize,
        /// Residual block index within that stage.
        block: usize,
    },
    /// Final normalization and projection into decoder space.
    Projection,
}

impl AudioUnitKind {
    /// Canonical logical operation path, independent of checkpoint containers.
    pub fn path(self) -> String {
        match self {
            Self::Patch { stage } => format!("model.audio_tower.layers.{stage}.patch_embedder"),
            Self::Convolution { stage, block } => {
                format!("model.audio_tower.layers.{stage}.conv_blocks.{block}")
            }
            Self::Projection => "model.audio_tower.output".into(),
        }
    }
}

/// Audio pyramid policy validated when constructing an `AudioConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioPolicy {
    /// Host-waveform precision at the first patch projection boundary.
    pub input_precision: super::config::MediaInputPrecision,
    /// Output channels at each stage.
    pub hidden_sizes: Vec<i32>,
    /// Causal residual blocks at each stage.
    pub depths: Vec<usize>,
    /// Consecutive time samples folded into each patch.
    pub patch_sizes: Vec<i32>,
    /// Width of every stride-one depthwise causal kernel.
    pub conv_kernel_size: i32,
    /// Ratio of MLP intermediate width to stage width.
    pub mlp_ratio: f64,
    /// Direct RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Language decoder output width.
    pub out_hidden_size: i32,
    /// Exact canonical physical projection formats selected during preparation.
    pub linear_formats: HashMap<String, LinearFormat>,
}

#[derive(Deserialize)]
struct AudioSource {
    model_type: String,
    hidden_sizes: Vec<i32>,
    depths: Vec<usize>,
    patch_sizes: Vec<i32>,
    num_hidden_layers: usize,
    conv_kernel_size: i32,
    mlp_ratio: f64,
    rms_norm_eps: f32,
    out_hidden_size: i32,
    hidden_act: String,
}

impl AudioPolicy {
    /// Ordered patch, convolution, and projection residency units.
    pub(super) fn execution_units(&self) -> Result<Vec<AudioUnitKind>, Error> {
        self.validate()?;
        let mut units = Vec::new();
        for (stage, &depth) in self.depths.iter().enumerate() {
            units.push(AudioUnitKind::Patch { stage });
            units.extend((0..depth).map(|block| AudioUnitKind::Convolution { stage, block }));
        }
        units.push(AudioUnitKind::Projection);
        Ok(units)
    }
    /// Normalizes the released waveform companion's explicit GGUF stage geometry.
    pub(super) fn from_gguf_metadata(
        metadata: &HashMap<String, eredu_gguf::MetadataValue>,
        decoder_width: i32,
    ) -> Result<Self, Error> {
        use eredu_gguf::{MetadataArray, MetadataValue};
        let integer = |key: &str| {
            metadata
                .get(key)
                .and_then(MetadataValue::as_i64)
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| Error::backend(format!("Muse audio GGUF {key:?} must be an i32")))
        };
        let list = |key: &str| -> Result<Vec<i32>, Error> {
            let values: Vec<i64> = match metadata.get(key) {
                Some(MetadataValue::Array(MetadataArray::Int32(values))) => {
                    values.iter().map(|&v| i64::from(v)).collect()
                }
                Some(MetadataValue::Array(MetadataArray::Uint32(values))) => {
                    values.iter().map(|&v| i64::from(v)).collect()
                }
                Some(MetadataValue::Array(MetadataArray::Int64(values))) => values.clone(),
                Some(MetadataValue::Array(MetadataArray::Uint64(values))) => values
                    .iter()
                    .map(|&v| i64::try_from(v).map_err(Error::backend))
                    .collect::<Result<_, _>>()?,
                _ => {
                    return Err(Error::backend(format!(
                        "Muse audio GGUF {key:?} must be an integer array"
                    )));
                }
            };
            values
                .into_iter()
                .map(|value| i32::try_from(value).map_err(Error::backend))
                .collect()
        };
        if metadata
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            != Some("clip")
            || metadata.get("general.type").and_then(MetadataValue::as_str) != Some("mmproj")
            || metadata
                .get("clip.audio.projector_type")
                .and_then(MetadataValue::as_str)
                != Some("muse-spark-audio")
            || metadata
                .get("clip.has_audio_encoder")
                .and_then(MetadataValue::as_bool)
                != Some(true)
            || integer("clip.audio.num_mel_bins")? != 1
            || integer("clip.audio.attention.head_count")? != 1
            || integer("clip.audio.projection_dim")? != decoder_width
        {
            return Err(Error::backend("incompatible Muse waveform GGUF companion"));
        }
        let hidden_sizes = list("clip.audio.hidden_sizes")?;
        let depths = list("clip.audio.depths")?
            .into_iter()
            .map(|depth| usize::try_from(depth).map_err(Error::backend))
            .collect::<Result<Vec<_>, _>>()?;
        let last = *hidden_sizes
            .last()
            .ok_or_else(|| Error::backend("Muse audio GGUF stages are empty"))?;
        let expansion = integer("clip.audio.feed_forward_length")?;
        if last <= 0
            || expansion <= 0
            || expansion % last != 0
            || integer("clip.audio.embedding_length")? != last
            || usize::try_from(integer("clip.audio.block_count")?).ok()
                != depths
                    .iter()
                    .try_fold(0usize, |sum, depth| sum.checked_add(*depth))
        {
            return Err(Error::backend(
                "Muse audio GGUF stage, block or MLP geometry disagrees",
            ));
        }
        let config = Self {
            input_precision: super::config::MediaInputPrecision::Preserve,
            hidden_sizes,
            depths,
            patch_sizes: list("clip.audio.patch_sizes")?,
            conv_kernel_size: integer("clip.audio.conv_kernel_size")?,
            mlp_ratio: f64::from(expansion / last),
            rms_norm_eps: metadata
                .get("clip.audio.attention.layer_norm_epsilon")
                .and_then(MetadataValue::as_f32)
                .ok_or_else(|| Error::backend("Muse audio GGUF RMS epsilon is absent"))?,
            out_hidden_size: decoder_width,
            linear_formats: HashMap::new(),
        };
        config.validate()?;
        Ok(config)
    }

    /// Normalizes the released encoder config and validates decoder geometry.
    pub(super) fn from_hf_value(
        value: &serde_json::Value,
        decoder_width: i32,
    ) -> Result<Self, Error> {
        let source: AudioSource =
            serde_json::from_value(value.clone()).map_err(Error::backend_source)?;
        if source.model_type != "muse_spark_audio"
            || source.hidden_act != "gelu"
            || source.num_hidden_layers != source.hidden_sizes.len()
            || source.out_hidden_size != decoder_width
        {
            return Err(Error::backend(
                "incompatible Muse waveform encoder configuration",
            ));
        }
        let config = Self {
            input_precision: super::config::MediaInputPrecision::Parameter,
            hidden_sizes: source.hidden_sizes,
            depths: source.depths,
            patch_sizes: source.patch_sizes,
            conv_kernel_size: source.conv_kernel_size,
            mlp_ratio: source.mlp_ratio,
            rms_norm_eps: source.rms_norm_eps,
            out_hidden_size: source.out_hidden_size,
            linear_formats: HashMap::new(),
        };
        config.validate()?;
        if let Some(declared) = value.get("samples_per_token") {
            if declared.as_i64() != Some(i64::from(config.samples_per_token()?)) {
                return Err(Error::backend(
                    "Muse audio declared cadence differs from patch geometry",
                ));
            }
        }
        Ok(config)
    }

    /// Checks all patch, MLP, state and output extents without allocating parameters.
    pub fn validate(&self) -> Result<(), Error> {
        if self.hidden_sizes.is_empty()
            || self.hidden_sizes.len() != self.depths.len()
            || self.hidden_sizes.len() != self.patch_sizes.len()
            || self.hidden_sizes.iter().any(|&size| size <= 0)
            || self.depths.contains(&0)
            || self.patch_sizes.iter().any(|&size| size <= 0)
            || self.conv_kernel_size <= 0
            || self.conv_kernel_size == i32::MAX
            || self.out_hidden_size <= 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.mlp_ratio.is_finite()
            || self.mlp_ratio <= 0.0
        {
            return Err(Error::backend("invalid Muse waveform pyramid geometry"));
        }
        self.samples_per_token()?;
        for stage in 0..self.hidden_sizes.len() {
            let input = if stage == 0 {
                1
            } else {
                self.hidden_sizes[stage - 1]
            };
            input
                .checked_mul(self.patch_sizes[stage])
                .ok_or_else(|| Error::backend("Muse audio patch width exceeds i32"))?;
            self.intermediate_size(stage)?;
        }
        for format in self.linear_formats.values() {
            format.validate().map_err(Error::backend)?;
        }
        Ok(())
    }

    /// Exact cadence of one language-width output token (1,920 in the release).
    pub fn samples_per_token(&self) -> Result<i32, Error> {
        self.patch_sizes.iter().try_fold(1i32, |total, &patch| {
            total
                .checked_mul(patch)
                .filter(|&value| value > 0)
                .ok_or_else(|| Error::backend("Muse audio sample cadence exceeds i32"))
        })
    }

    /// Exact integer MLP expansion width of one stage.
    pub fn intermediate_size(&self, stage: usize) -> Result<i32, Error> {
        let width = f64::from(
            *self
                .hidden_sizes
                .get(stage)
                .ok_or_else(|| Error::backend("missing Muse audio stage"))?,
        ) * self.mlp_ratio;
        if !width.is_finite() || width < 1.0 || width >= f64::from(i32::MAX) {
            return Err(Error::backend("Muse audio MLP width exceeds i32"));
        }
        Ok(width as i32)
    }

    /// Resolves a canonical projection's selected physical representation.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        self.linear_formats
            .get(name)
            .copied()
            .unwrap_or(LinearFormat::Dense)
    }
}

/// Validated waveform policy with retained typed residency-unit declarations.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    policy: AudioPolicy,
    declaration: std::sync::Arc<specs::Declarations>,
}
impl AudioConfig {
    /// Declares the complete source pyramid once, without backend resources.
    pub fn new(policy: AudioPolicy) -> Result<Self, Error> {
        let declaration = specs::Declarations::new(&policy)?;
        Ok(Self {
            policy,
            declaration: std::sync::Arc::new(declaration),
        })
    }
    /// Normalized cadence, arithmetic and format policy.
    pub fn policy(&self) -> &AudioPolicy {
        &self.policy
    }
    pub(crate) fn declaration(&self) -> &specs::Declarations {
        &self.declaration
    }
    /// Released GGUF normalization followed by immutable source declaration.
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, eredu_gguf::MetadataValue>,
        decoder_width: i32,
    ) -> Result<Self, Error> {
        Self::new(AudioPolicy::from_gguf_metadata(metadata, decoder_width)?)
    }
    /// Released SafeTensors normalization followed by immutable source declaration.
    pub fn from_hf_value(value: &serde_json::Value, decoder_width: i32) -> Result<Self, Error> {
        Self::new(AudioPolicy::from_hf_value(value, decoder_width)?)
    }
    /// Projects invocation identities from the retained parameter owners.
    pub fn execution_units(&self) -> Result<Vec<AudioUnitKind>, Error> {
        Ok(self
            .declaration
            .units
            .iter()
            .map(|(kind, _)| *kind)
            .collect())
    }
    /// Projects encoding extensions while preserving geometry and execution policy.
    pub fn with_formats(&self, formats: HashMap<String, LinearFormat>) -> Result<Self, Error> {
        if formats == self.policy.linear_formats {
            return Ok(self.clone());
        }
        let mut policy = self.policy.clone();
        policy.linear_formats = formats;
        policy.validate()?;
        let mut declaration = self.declaration.as_ref().clone();
        declaration.project_formats(&policy)?;
        Ok(Self {
            policy,
            declaration: std::sync::Arc::new(declaration),
        })
    }
}
impl std::ops::Deref for AudioConfig {
    type Target = AudioPolicy;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}
impl DeclareLogicalParameters for AudioConfig {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.declaration.logical_parameters()
    }
}

/// Biased patch projection followed by direct FP32 RMS normalization.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioPatchEmbedding<B: NeuralBackend> {
    /// Patch projection with exact checkpoint companions.
    pub projection: B::Linear,
    /// Direct learned RMS gain.
    pub norm: B::Normalization,
    #[parameter(skip)]
    patch_size: i32,
    #[parameter(skip)]
    input_channels: i32,
    #[parameter(skip)]
    input_width: i32,
    #[parameter(skip)]
    input_precision: super::config::MediaInputPrecision,
}

impl<B: NeuralBackend> AudioPatchEmbedding<B> {
    /// Constructs one separately addressable stage's patch parameters.
    pub fn new(
        config: &AudioConfig,
        stage: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let specs::UnitSpec::Patch(spec) = config
            .declaration()
            .unit(AudioUnitKind::Patch { stage })?
            .clone()
        else {
            unreachable!()
        };
        Self::from_spec(spec, context)
    }

    fn from_spec(
        spec: specs::PatchSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        eredu_nn::DeclareLogicalParameters::logical_parameters(&spec)?;
        Ok(Self {
            input_width: spec.projection.input,
            projection: B::linear(spec.projection, context)?,
            norm: B::normalization(spec.norm, context)?,
            patch_size: spec.patch_size,
            input_channels: spec.input_channels,
            input_precision: spec.input_precision,
        })
    }

    /// Folds consecutive time positions in time-major/channel-minor order.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let shape = input.shape();
        if shape.len() != 3
            || shape[0] <= 0
            || shape[1] <= 0
            || shape[1] % self.patch_size != 0
            || shape[2] != self.input_channels
        {
            return Err(Error::backend(
                "Muse audio patch input has incompatible geometry",
            ));
        }
        let patches = input.reshape(
            &[shape[0], shape[1] / self.patch_size, self.input_width],
            context,
        )?;
        let patches = self
            .input_precision
            .prepare(&patches, &self.projection, context)?;
        self.norm
            .forward(&self.projection.forward(&patches, context)?, context)
    }
}

/// Causal depthwise convolution, direct RMS norm, exact GELU MLP and residual.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioConvBlock<B: NeuralBackend> {
    /// Independently leasable causal convolution parameters.
    pub convolution: CausalDepthwiseConvolution<B>,
    /// Direct learned RMS norm.
    pub norm: B::Normalization,
    /// Biased expansion projection.
    pub input: B::Linear,
    /// Biased output projection.
    pub output: B::Linear,
    #[parameter(skip)]
    activation: eredu_nn::UnaryActivation,
}

impl<B: NeuralBackend> AudioConvBlock<B> {
    /// Constructs one separately addressable convolution block.
    pub fn new(
        config: &AudioConfig,
        stage: usize,
        block: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let specs::UnitSpec::Convolution(spec) = config
            .declaration()
            .unit(AudioUnitKind::Convolution { stage, block })?
            .clone()
        else {
            unreachable!()
        };
        Self::from_spec(spec, context)
    }

    fn from_spec(
        spec: specs::ConvSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        eredu_nn::DeclareLogicalParameters::logical_parameters(&spec)?;
        crate::operator_requirements::require::<B>(
            "Muse audio",
            crate::operator_requirements::MUSE_AUDIO,
        )?;
        Ok(Self {
            convolution: CausalDepthwiseConvolution::new(spec.convolution, context)?,
            norm: B::normalization(spec.norm, context)?,
            input: B::linear(spec.network.up, context)?,
            output: B::linear(spec.network.down, context)?,
            activation: spec.network.activation,
        })
    }

    /// Returns a candidate output/history; the caller owns commitment.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        history: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, Option<B::Tensor>), Error> {
        let convolution = self.convolution.forward(input, history, context)?;
        let hidden = self.norm.forward(&convolution.output, context)?;
        let hidden = self
            .activation
            .apply::<B>(self.input.forward(&hidden, context)?, context)?;
        let output = input.add(&self.output.forward(&hidden, context)?, context)?;
        Ok((output, convolution.history))
    }
}

/// Patch reduction and its ordered residual blocks.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioStage<B: NeuralBackend> {
    /// Consecutive-sample patch reduction.
    pub patch: AudioPatchEmbedding<B>,
    /// Independently addressable convolution units.
    pub blocks: Vec<AudioConvBlock<B>>,
}

/// Final direct RMS norms and biased language-width projection.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioProjection<B: NeuralBackend> {
    /// Final stage direct gain.
    pub norm: B::Normalization,
    /// Projection into the decoder width.
    pub projection: B::Linear,
    /// Direct perception gain in decoder space.
    pub post_norm: B::Normalization,
}

impl<B: NeuralBackend> AudioProjection<B> {
    /// Constructs only the final projection unit's exact parameters.
    pub fn new(
        config: &AudioConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let specs::UnitSpec::Projection(spec) = config
            .declaration()
            .unit(AudioUnitKind::Projection)?
            .clone()
        else {
            unreachable!()
        };
        Self::from_spec(spec, context)
    }

    fn from_spec(
        spec: specs::ProjectionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            norm: B::normalization(spec.norm, context)?,
            projection: B::linear(spec.projection, context)?,
            post_norm: B::normalization(spec.post_norm, context)?,
        })
    }

    /// Applies the released normalization, projection, normalization order.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.norm.forward(input, context)?;
        self.post_norm
            .forward(&self.projection.forward(&hidden, context)?, context)
    }
}

/// One bounded audio parameter lease, shared by composite and component traversal.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum AudioUnit<B: NeuralBackend> {
    /// Patch reduction.
    Patch(AudioPatchEmbedding<B>),
    /// Causal convolution and residual MLP.
    Convolution(AudioConvBlock<B>),
    /// Perception projection.
    Projection(AudioProjection<B>),
}

impl<B: NeuralBackend> AudioUnit<B> {
    /// Builds exactly one operation without allocating the remaining tower.
    pub fn new(
        config: &AudioConfig,
        kind: AudioUnitKind,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_spec(config.declaration().unit(kind)?.clone(), context)
    }

    fn from_spec(
        spec: specs::UnitSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(match spec {
            specs::UnitSpec::Patch(spec) => {
                Self::Patch(AudioPatchEmbedding::from_spec(spec, context)?)
            }
            specs::UnitSpec::Convolution(spec) => {
                Self::Convolution(AudioConvBlock::from_spec(spec, context)?)
            }
            specs::UnitSpec::Projection(spec) => {
                Self::Projection(AudioProjection::from_spec(spec, context)?)
            }
        })
    }
}

/// Complete caller-owned bounded carry, separate from checkpoint parameters.
#[derive(Debug, Clone)]
pub struct AudioState<T> {
    histories: Vec<Vec<Option<T>>>,
    widths: Vec<i32>,
    patch_sizes: Vec<i32>,
    kernel_size: i32,
    samples_per_token: i32,
    processed_samples: u64,
    batch_size: Option<i32>,
}

impl<T> AudioState<T> {
    /// Returns the number of waveform samples represented by this carry.
    pub fn processed_samples(&self) -> u64 {
        self.processed_samples
    }

    /// Iterates exact retained convolution tensors for copying and accounting.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.histories.iter().flatten().filter_map(Option::as_ref)
    }
}

/// Waveform encoder assembled from the same patch and convolution units.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioPyramid<B: NeuralBackend> {
    /// Ordered stage parameters.
    pub stages: Vec<AudioStage<B>>,
    /// Final normalization and perception projection.
    pub output: AudioProjection<B>,
    #[parameter(skip)]
    config: AudioConfig,
}

impl<B: NeuralBackend> AudioPyramid<B> {
    /// Constructs unloaded parameter topology without a native-family implementation.
    pub fn new(
        config: AudioConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let declarations = config.declaration();
        let mut stages: Vec<AudioStage<B>> = Vec::new();
        let mut output = None;
        for (_, spec) in declarations.units.iter() {
            match AudioUnit::from_spec(spec.clone(), context)? {
                AudioUnit::Patch(patch) => stages.push(AudioStage {
                    patch,
                    blocks: Vec::new(),
                }),
                AudioUnit::Convolution(block) => stages
                    .last_mut()
                    .expect("declared audio stage")
                    .blocks
                    .push(block),
                AudioUnit::Projection(projection) => output = Some(projection),
            }
        }
        Ok(Self {
            stages,
            output: output.expect("declared audio output"),
            config,
        })
    }

    /// Creates empty bounded convolution carry for this exact geometry.
    pub fn empty_state(&self) -> AudioState<B::Tensor> {
        AudioState {
            histories: self
                .config
                .depths
                .iter()
                .map(|&depth| vec![None; depth])
                .collect(),
            widths: self.config.hidden_sizes.clone(),
            patch_sizes: self.config.patch_sizes.clone(),
            kernel_size: self.config.conv_kernel_size,
            samples_per_token: self
                .config
                .samples_per_token()
                .expect("validated audio cadence"),
            processed_samples: 0,
            batch_size: None,
        }
    }

    /// Evaluates complete cadence-aligned waveform chunks and returns candidate carry.
    /// The previous state is never mutated, including on failure.
    pub fn forward(
        &mut self,
        waveform: &B::Tensor,
        previous: &AudioState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, AudioState<B::Tensor>), Error> {
        let shape = waveform.shape();
        let cadence = self.config.samples_per_token()?;
        if shape.len() != 2
            || shape[0] <= 0
            || shape[1] <= 0
            || shape[1] % cadence != 0
            || previous.widths != self.config.hidden_sizes
            || previous.patch_sizes != self.config.patch_sizes
            || previous.batch_size.is_some_and(|batch| batch != shape[0])
            || previous.kernel_size != self.config.conv_kernel_size
            || previous.samples_per_token != cadence
            || previous.histories.len() != self.stages.len()
            || previous
                .histories
                .iter()
                .zip(&self.config.depths)
                .any(|(histories, &depth)| histories.len() != depth)
        {
            return Err(Error::backend(
                "Muse audio waveform or carry has incompatible geometry",
            ));
        }
        let mut next = previous.clone();
        next.batch_size = Some(shape[0]);
        next.processed_samples = next
            .processed_samples
            .checked_add(shape[1] as u64)
            .ok_or_else(|| Error::backend("Muse audio sample position exceeds u64"))?;
        let mut hidden = waveform.reshape(&[shape[0], shape[1], 1], context)?;
        for (stage, histories) in self.stages.iter_mut().zip(&mut next.histories) {
            hidden = stage.patch.forward(&hidden, context)?;
            for (block, history) in stage.blocks.iter_mut().zip(histories) {
                let (output, carry) = block.forward(&hidden, history.as_ref(), context)?;
                hidden = output;
                *history = carry;
            }
        }
        let output = self.output.forward(&hidden, context)?;
        Ok((output, next))
    }
}
