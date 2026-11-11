//! Portable LiquidCodec declarations, neural execution, quantization and streaming.

use crate::{
    spectral::{
        AnalysisEndpoints, AnalysisExecution, OverlapAddOutput, PeriodicHannIstft,
        PeriodicHannStft, WindowArithmetic,
    },
    Error,
};
use eredu_nn::{
    DeclareLogicalParameters, LinearOperator, LinearSpec, LogicalParameterDeclaration,
    NeuralBackend, PadMode, ParameterSpec, Parameterized, Tensor, TensorElementType,
};
use serde::Deserialize;

mod gguf;
mod neural;
pub use gguf::GgufDecoderBindings;
pub use neural::{
    BackboneBlock, Declaration, DecoderState, NeuralCodec, NeuralDecoder, Tokenizer, Unit,
};

/// Released codec component geometry, independent of language-model architectures.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    /// Exact checkpoint model type (`liquid_codec`).
    pub model_type: String,
    /// Neural encoder and decoder residual width.
    pub encoder_dim: i32,
    /// Spectral front-end convolution width.
    pub encoder_conv_dim: i32,
    /// Analysis and synthesis FFT size.
    pub stft_n_fft: i32,
    /// Audio samples between spectral frames.
    pub stft_hop_length: i32,
    /// Periodic-Hann window length.
    pub stft_win_length: i32,
    /// Spectral frames represented by one codec frame.
    pub stft_frame_stack: i32,
    /// Learned latent width before finite-scalar projection.
    pub quantizer_dim: i32,
    /// Ordered scalar alphabet sizes within each codebook.
    pub fsq_levels: Vec<i32>,
    /// Ordered independent codebooks per frame.
    pub n_q: i32,
    /// Decoder attention window in spectral frames.
    pub decoder_window: i32,
    /// Audio sample rate in Hz.
    pub sampling_rate: i32,
}

impl Config {
    /// Normalizes current flat and published nested codec configuration.
    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        let mut value: serde_json::Value = serde_json::from_slice(bytes).map_err(json_error)?;
        if let Some(nested) = value.get("model").cloned() {
            let nested = nested
                .as_object()
                .ok_or_else(|| invalid("codec model config must be an object"))?;
            let outer = value
                .as_object_mut()
                .ok_or_else(|| invalid("codec config must be an object"))?;
            for (key, value) in nested {
                if matches!(
                    key.as_str(),
                    "encoder_lfm_config" | "decoder_lfm_config" | "n_output_samples"
                ) {
                    continue;
                }
                if !matches!(
                    key.as_str(),
                    "encoder_dim"
                        | "encoder_conv_dim"
                        | "stft_n_fft"
                        | "stft_hop_length"
                        | "stft_win_length"
                        | "stft_frame_stack"
                        | "quantizer_dim"
                        | "fsq_levels"
                        | "n_q"
                        | "decoder_window"
                        | "sampling_rate"
                        | "initializer_range"
                ) {
                    return Err(invalid(format!("unknown nested LiquidCodec setting {key}")));
                }
                outer.insert(key.clone(), value.clone());
            }
        }
        let config: Self = serde_json::from_value(value.clone()).map_err(json_error)?;
        config.validate()?;
        for (key, expected) in [
            ("sample_rate", config.sampling_rate),
            ("frame_samples", config.frame_samples()?),
        ] {
            if value
                .get(key)
                .is_some_and(|v| v.as_i64() != Some(i64::from(expected)))
            {
                return Err(invalid(format!(
                    "LiquidCodec {key} conflicts with its component geometry"
                )));
            }
        }
        Ok(config)
    }

    /// Checks supported spectral geometry and exact I32 scalar-code representation.
    pub fn validate(&self) -> Result<(), Error> {
        if self.model_type != "liquid_codec"
            || self.stft_n_fft < 4
            || self.stft_n_fft % 2 != 0
            || self.stft_win_length != self.stft_n_fft
            || self.stft_hop_length <= 0
            || self.stft_hop_length > self.stft_n_fft - 2
            || (self.stft_n_fft - self.stft_hop_length) % 2 != 0
            || [
                self.encoder_dim,
                self.encoder_conv_dim,
                self.stft_frame_stack,
                self.quantizer_dim,
                self.n_q,
                self.decoder_window,
                self.sampling_rate,
            ]
            .into_iter()
            .any(|n| n <= 0)
            || self.fsq_levels.is_empty()
            || self.fsq_levels.iter().any(|n| *n < 2 || *n > 1000)
        {
            return Err(invalid("invalid LiquidCodec component geometry"));
        }
        self.frame_samples()?;
        self.scalar_width()?;
        self.codebook_size()?;
        Ok(())
    }

    /// Exact audio samples represented by one complete codec frame.
    pub fn frame_samples(&self) -> Result<i32, Error> {
        self.stft_hop_length
            .checked_mul(self.stft_frame_stack)
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("LiquidCodec frame sample extent overflow"))
    }
    /// Number of discrete entries in each independent scalar-product codebook.
    pub fn codebook_size(&self) -> Result<i32, Error> {
        self.fsq_levels
            .iter()
            .try_fold(1i32, |n, level| n.checked_mul(*level))
            .filter(|n| *n > 0 && *n <= 16_777_216)
            .ok_or_else(|| invalid("LiquidCodec codes exceed exact F32 integer geometry"))
    }

    fn scalar_code(&self, mut index: i32) -> impl Iterator<Item = f32> + '_ {
        self.fsq_levels.iter().map(move |&level| {
            let half = level / 2;
            let value = (index % level - half) as f32 / half as f32;
            index /= level;
            value
        })
    }
    fn scalar_width(&self) -> Result<i32, Error> {
        i32::try_from(self.fsq_levels.len())
            .ok()
            .and_then(|n| n.checked_mul(self.n_q))
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("LiquidCodec scalar width overflow"))
    }
}

/// Reflection-padded periodic-Hann analysis and log-magnitude/phase features.
#[derive(Debug, Clone)]
pub struct Analysis<T: Tensor> {
    config: Config,
    stft: PeriodicHannStft<T>,
}

impl<T: Tensor> Analysis<T> {
    /// Optional tensor mechanisms needed by reflective codec analysis.
    pub const fn required_capabilities() -> eredu_nn::NeuralOperatorCapabilities {
        PeriodicHannStft::<T>::required_capabilities_for(
            WindowArithmetic::Float32Polynomial,
            AnalysisExecution::HostFloat32,
        )
        .union(eredu_nn::NeuralOperatorCapabilities::REQUIRE_FINITE)
        .union(eredu_nn::NeuralOperatorCapabilities::SQRT)
        .union(eredu_nn::NeuralOperatorCapabilities::LOG1P)
    }
    /// Constructs analysis constants without opening a checkpoint.
    pub fn new(config: &Config, context: &T::Context) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self {
            config: config.clone(),
            stft: PeriodicHannStft::with_policy(
                config.stft_n_fft,
                config.stft_hop_length,
                AnalysisEndpoints::Reflect,
                WindowArithmetic::Float32Polynomial,
                AnalysisExecution::HostFloat32,
                context,
            )?,
        })
    }
    /// Converts finite mono `[batch,1,samples]` PCM to F32
    /// `[batch,3*(FFT/2+1),spectral_frames]`. Final PCM is zero-padded to
    /// a complete codec frame before reflective spectral padding.
    pub fn features(&self, pcm: &T, context: &T::Context) -> Result<T, Error> {
        if pcm.shape().len() != 3 || pcm.dim(0) <= 0 || pcm.dim(1) != 1 || pcm.dim(2) <= 0 {
            return Err(invalid(
                "LiquidCodec analysis requires nonempty mono [batch,1,samples]",
            ));
        }
        pcm.require_finite(context)?;
        let frame = self.config.frame_samples()?;
        let padded = pcm
            .dim(2)
            .checked_add(frame - 1)
            .and_then(|n| (n / frame).checked_mul(frame))
            .ok_or_else(|| invalid("LiquidCodec padded waveform exceeds I32"))?;
        let pcm = T::pad(
            &pcm.cast_float(TensorElementType::F32, context)?,
            &[(0, 0), (0, 0), (0, padded - pcm.dim(2))],
            PadMode::Constant,
            context,
        )?;
        let (real, imaginary) = self.stft.analyze(&pcm, context)?;
        let magnitude = real
            .square(context)?
            .add(&imaginary.square(context)?, context)?
            .sqrt(context)?;
        let denominator = magnitude.maximum_scalar(1e-5, context)?;
        let features = T::concatenate(
            &[
                magnitude.log1p(context)?,
                real.divide(&denominator, context)?,
                imaginary.divide(&denominator, context)?,
            ],
            1,
            context,
        )?;
        features.require_finite(context)?;
        Ok(features)
    }
}

/// Learned projections surrounding a product of bounded scalar alphabets.
#[derive(Debug, Clone)]
pub struct QuantizerSpec {
    config: Config,
    input: Option<LinearSpec>,
    output: Option<LinearSpec>,
}

impl QuantizerSpec {
    /// Declares projection identities in the canonical quantizer namespace.
    pub fn new(config: &Config, root: &str) -> Result<Self, Error> {
        config.validate()?;
        if root.is_empty() {
            return Err(invalid(
                "LiquidCodec quantizer requires a parameter namespace",
            ));
        }
        let width = config.scalar_width()?;
        let linear = |name: &str, input, output| -> Result<LinearSpec, Error> {
            let root = format!("{root}.{name}");
            Ok(LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(format!("{root}.weight"))
                    .map_err(|e| invalid(e.to_string()))?,
                bias: Some(
                    ParameterSpec::trainable(format!("{root}.bias"))
                        .map_err(|e| invalid(e.to_string()))?,
                ),
                format: eredu_nn::LinearFormatSpec::unscaled(
                    eredu_checkpoint::LinearFormat::Dense,
                )?,
            })
        };
        let spec = Self {
            config: config.clone(),
            input: (config.quantizer_dim != width)
                .then(|| linear("project_in", config.quantizer_dim, width))
                .transpose()?,
            output: (config.quantizer_dim != width)
                .then(|| linear("project_out", width, config.quantizer_dim))
                .transpose()?,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(spec)
    }
    /// Exact floating SafeTensors admission for the released projection unit.
    pub fn safetensors_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, Error> {
        use eredu_checkpoint::schema::*;
        let constraints = self
            .logical_parameters()?
            .into_iter()
            .map(|p| {
                let name = p.parameter.id.to_string();
                let alias = name.replacen("quantizer.", "quantizer.fsq.", 1);
                SafetensorsTensorConstraint::required(
                    &name,
                    p.shape,
                    StoredDtypeConstraint::Floating,
                )
                .with_aliases((name != alias).then_some(alias))
            })
            .collect();
        SafetensorsCheckpointPlan::new(
            "LiquidCodec scalar projections",
            constraints,
            vec![],
            CatalogPolicy::non_strict(),
        )
        .map_err(|e| invalid(e.to_string()))
    }
}

impl DeclareLogicalParameters for QuantizerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        let mut p = Vec::new();
        if let Some(s) = &self.input {
            p.extend(s.logical_parameters()?);
        }
        if let Some(s) = &self.output {
            p.extend(s.logical_parameters()?);
        }
        Ok(p)
    }
}

/// Learned finite-scalar quantizer. Fixed tables are constants, not codebooks
/// loaded from the checkpoint and not mutable streaming state.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Quantizer<B: NeuralBackend> {
    encoder: ScalarEncoder<B>,
    decoder: ScalarDecoder<B>,
}

impl<B: NeuralBackend> Quantizer<B> {
    /// Immutable non-parameter roots retained for residency and allocation inspection.
    pub fn constants(&self) -> [&B::Tensor; 6] {
        [
            &self.encoder.half_width,
            &self.encoder.basis,
            &self.encoder.half_levels,
            &self.encoder.offset,
            &self.encoder.shift,
            &self.decoder.table,
        ]
    }

    /// Constructs unloaded learned projections and the exact finite alphabets.
    pub fn new(
        spec: &QuantizerSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            encoder: ScalarEncoder::new(spec, context)?,
            decoder: ScalarDecoder::new(spec, context)?,
        })
    }

    /// Encodes `[batch,latent_width,frames]` to exact I32 `[batch,codebooks,frames]`.
    pub fn encode(
        &mut self,
        latents: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.encoder.encode(latents, context)
    }

    /// Decodes exact nonnegative I32 codes to `[batch,latent_width,frames]`.
    pub fn decode(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.decoder.decode(codes, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct ScalarEncoder<B: NeuralBackend> {
    input: Option<B::Linear>,
    #[parameter(skip)]
    config: Config,
    #[parameter(skip)]
    half_width: B::Tensor,
    #[parameter(skip)]
    basis: B::Tensor,
    #[parameter(skip)]
    half_levels: B::Tensor,
    #[parameter(skip)]
    offset: B::Tensor,
    #[parameter(skip)]
    shift: B::Tensor,
}

impl<B: NeuralBackend> ScalarEncoder<B> {
    fn new(spec: &QuantizerSpec, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        use eredu_nn::NeuralOperatorCapabilities as Ops;
        B::require_operator_capabilities(
            "finite scalar encoder",
            Ops::ROUND_TO_I32
                .union(Ops::REQUIRE_FINITE)
                .union(Ops::TANH)
                .union(Ops::CAST_FLOAT)
                .union(Ops::FROM_I32_SLICE),
        )?;
        let config = spec.config.clone();
        let dimension = config.fsq_levels.len() as i32;
        let mut basis = Vec::new();
        let mut radix = 1;
        let mut half = Vec::new();
        let mut offsets = Vec::new();
        let mut shifts = Vec::new();
        for &level in &config.fsq_levels {
            basis.push(radix);
            radix *= level;
            let h = (level - 1) as f32 * (1.0 + 1e-3) / 2.0;
            let offset = if level % 2 == 0 { 0.5f32 } else { 0.0 };
            half.push(h);
            offsets.push(offset);
            shifts.push((offset / h).atanh());
        }
        Ok(Self {
            input: spec
                .input
                .clone()
                .map(|s| B::linear(s, context))
                .transpose()?,
            half_width: B::Tensor::from_i32_slice(
                &config.fsq_levels.iter().map(|n| n / 2).collect::<Vec<_>>(),
                &[dimension],
                context,
            )?,
            basis: B::Tensor::from_i32_slice(&basis, &[dimension], context)?,
            half_levels: B::Tensor::from_f32_slice(&half, &[dimension], context)?,
            offset: B::Tensor::from_f32_slice(&offsets, &[dimension], context)?,
            shift: B::Tensor::from_f32_slice(&shifts, &[dimension], context)?,
            config,
        })
    }

    /// Encodes `[batch,latent_width,frames]` to exact I32 `[batch,codebooks,frames]`.
    pub fn encode(
        &mut self,
        latents: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if latents.shape().len() != 3
            || latents.dim(0) <= 0
            || latents.dim(1) != self.config.quantizer_dim
            || latents.dim(2) <= 0
        {
            return Err(invalid(
                "invalid LiquidCodec scalar encoder latent geometry",
            ));
        }
        let latents = latents.transpose_axes(&[0, 2, 1], context)?;
        let latents = match &mut self.input {
            Some(p) => p.forward(&latents, context)?,
            None => latents,
        };
        let latents = latents.cast_float(TensorElementType::F32, context)?;
        latents.require_finite(context)?;
        let latent = latents.reshape(
            &[
                latents.dim(0),
                latents.dim(1),
                self.config.n_q,
                self.config.fsq_levels.len() as i32,
            ],
            context,
        )?;
        let bounded = latent
            .add(&self.shift, context)?
            .tanh(context)?
            .multiply(&self.half_levels, context)?
            .subtract(&self.offset, context)?;
        let digits = bounded
            .round_to_i32(context)?
            .add(&self.half_width, context)?
            .multiply(&self.basis, context)?;
        let codes = B::Tensor::sum_axis(&digits, -1, false, context)?;
        Ok(codes.transpose_axes(&[0, 2, 1], context)?)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct ScalarDecoder<B: NeuralBackend> {
    output: Option<B::Linear>,
    #[parameter(skip)]
    config: Config,
    #[parameter(skip)]
    table: B::Tensor,
}

impl<B: NeuralBackend> ScalarDecoder<B> {
    fn new(spec: &QuantizerSpec, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        use eredu_nn::NeuralOperatorCapabilities as Ops;
        B::require_operator_capabilities(
            "finite scalar decoder",
            Ops::FULL_I32
                .union(Ops::EQUAL_I32)
                .union(Ops::WHERE_CONDITION),
        )?;
        let config = spec.config.clone();
        let dimension = config.fsq_levels.len() as i32;
        let mut table = Vec::new();
        let count = (config.codebook_size()? as usize)
            .checked_mul(dimension as usize)
            .ok_or_else(|| invalid("finite scalar lookup extent overflow"))?;
        table
            .try_reserve_exact(count)
            .map_err(|e| invalid(e.to_string()))?;
        table.extend((0..config.codebook_size()?).flat_map(|code| config.scalar_code(code)));
        Ok(Self {
            output: spec
                .output
                .clone()
                .map(|s| B::linear(s, context))
                .transpose()?,
            table: B::Tensor::from_f32_slice(
                &table,
                &[config.codebook_size()?, dimension],
                context,
            )?,
            config,
        })
    }
    /// Decodes exact nonnegative I32 codes to `[batch,latent_width,frames]`.
    pub fn decode(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if codes.shape().len() != 3
            || codes.dim(0) <= 0
            || codes.dim(1) != self.config.n_q
            || codes.dim(2) <= 0
            || codes.element_type() != Some(TensorElementType::I32)
        {
            return Err(invalid(
                "invalid LiquidCodec scalar decoder code geometry or representation",
            ));
        }
        let codes = codes.transpose_axes(&[0, 2, 1], context)?;
        // General gathers can wrap negative indices. Map them to the first
        // out-of-range positive sentinel so the strict gather rejects them.
        let valid = codes
            .subtract(&codes.maximum_i32(0, context)?, context)?
            .equal_i32(0, context)?;
        let codes = B::Tensor::where_condition(
            &valid,
            &codes,
            &B::Tensor::full_i32(self.config.codebook_size()?, &[], context)?,
            context,
        )?;
        let latents = self.table.take_axis(&codes, 0, context)?.reshape(
            &[codes.dim(0), codes.dim(1), self.config.scalar_width()?],
            context,
        )?;
        let latents = match &mut self.output {
            Some(p) => p.forward(&latents, context)?,
            None => latents,
        };
        Ok(latents.transpose_axes(&[0, 2, 1], context)?)
    }
}

/// F32 polar spectra to unnormalized sample/envelope sums for streaming synthesis.
#[derive(Debug, Clone)]
pub struct Synthesis<B: NeuralBackend> {
    istft: PeriodicHannIstft<B::Tensor>,
}
impl<B: NeuralBackend> Synthesis<B> {
    /// Binds the retained checkpoint synthesis window without regenerating it.
    pub fn set_window(
        &mut self,
        window: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        self.istft.set_window(window, context)
    }
    /// Constructs parameter-free periodic-Hann synthesis constants.
    pub fn new(config: &Config, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        config.validate()?;
        B::require_operator_capabilities(
            "LiquidCodec spectral synthesis",
            eredu_nn::NeuralOperatorCapabilities::IRFFT
                .union(eredu_nn::NeuralOperatorCapabilities::SIN_COS)
                .union(eredu_nn::NeuralOperatorCapabilities::EXP)
                .union(eredu_nn::NeuralOperatorCapabilities::REQUIRE_FINITE)
                .union(eredu_nn::NeuralOperatorCapabilities::CAST_FLOAT)
                .union(eredu_nn::NeuralOperatorCapabilities::FULL_F32)
                .union(eredu_nn::NeuralOperatorCapabilities::BROADCAST_TO),
        )?;
        Ok(Self {
            istft: PeriodicHannIstft::new(config.stft_n_fft, config.stft_hop_length, context)?,
        })
    }
    /// Computes inverse FFT and overlap-add before endpoint trimming or division.
    pub fn overlap_add(
        &self,
        log_magnitude: &B::Tensor,
        angle: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<OverlapAddOutput<B::Tensor>, Error> {
        let (real, imaginary) = self.cartesian(log_magnitude, angle, context)?;
        self.istft.overlap_add(&real, &imaginary, false, context)
    }
    /// Reconstructs a finite spectrum with the codec's symmetric endpoint trim.
    pub fn reconstruct(
        &self,
        log_magnitude: &B::Tensor,
        angle: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (real, imaginary) = self.cartesian(log_magnitude, angle, context)?;
        self.istft.reconstruct(&real, &imaginary, false, context)
    }
    fn cartesian(
        &self,
        log_magnitude: &B::Tensor,
        angle: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        if log_magnitude.shape() != angle.shape() {
            return Err(invalid("LiquidCodec polar spectrum shapes differ"));
        }
        log_magnitude.require_finite(context)?;
        angle.require_finite(context)?;
        let magnitude = B::exp(
            log_magnitude.cast_float(TensorElementType::F32, context)?,
            context,
        )?;
        magnitude.require_finite(context)?;
        let (sin, cos) = angle
            .cast_float(TensorElementType::F32, context)?
            .sin_cos(context)?;
        Ok((
            magnitude.multiply(&cos, context)?,
            magnitude.multiply(&sin, context)?,
        ))
    }
}

fn json_error(e: serde_json::Error) -> Error {
    invalid(e.to_string())
}
fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidShape(message.into())
}
