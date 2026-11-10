//! Backend-free Mimi construction declarations and logical parameter ownership.
use super::Config;
use crate::{
    streaming::{
        Conv1dSpec, ConvTranspose1dSpec, StreamableConv1d, StreamableConvTranspose1d, TransposeTail,
    },
    Error,
};
use eredu_nn::{
    DeclareLogicalParameters, Error as ComputeError, LayerNormalizationArithmetic,
    LayerNormalizationConstructionSpec, LinearFormatSpec, LinearSpec, LogicalParameterDeclaration,
    LogicalParameterRole, PadMode, Parameter, ParameterSpec, Tensor,
};
fn parameter(root: &str, field: &str) -> Result<ParameterSpec, Error> {
    let mut p =
        ParameterSpec::trainable(format!("{root}.{field}")).map_err(ComputeError::backend)?;
    p.group = Some(root.into());
    Ok(p)
}
fn validate(s: &impl DeclareLogicalParameters) -> Result<(), Error> {
    eredu_nn::validate_logical_parameter_topology(&s.logical_parameters()?)
        .map_err(ComputeError::backend_source)?;
    Ok(())
}
fn linear(root: &str, input: i32, output: i32) -> Result<LinearSpec, Error> {
    Ok(LinearSpec {
        input,
        output,
        weight: parameter(root, "weight")?,
        bias: None,
        format: LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
    })
}
fn norm(root: &str, width: i32) -> Result<LayerNormalizationConstructionSpec, Error> {
    Ok(LayerNormalizationConstructionSpec {
        dimensions: width,
        epsilon: 1e-5,
        weight: Some(parameter(root, "weight")?),
        bias: Some(parameter(root, "bias")?),
        arithmetic: LayerNormalizationArithmetic::Native,
    })
}
/// Released physical kernel layout, independent of the native OKI geometry.
#[derive(Debug, Clone, Copy)]
enum SourceKernelLayout {
    OutputInputKernel,
    InputOutputKernel,
}
impl SourceKernelLayout {
    fn axes(self) -> [usize; 3] {
        match self {
            Self::OutputInputKernel => [0, 2, 1],
            Self::InputOutputKernel => [1, 2, 0],
        }
    }
}

/// A source extension borrows the exact logical construction slot it transforms.
pub(super) struct CheckpointParameter<'a> {
    pub logical: LogicalParameterDeclaration<'a>,
    pub axes: Option<[usize; 3]>,
}
pub(super) trait DeclareCheckpointParameters {
    fn checkpoint_parameters(&self) -> Result<Vec<CheckpointParameter<'_>>, ComputeError>;
}
impl<S: DeclareCheckpointParameters> DeclareCheckpointParameters for [S] {
    fn checkpoint_parameters(&self) -> Result<Vec<CheckpointParameter<'_>>, ComputeError> {
        let mut parameters = Vec::new();
        for scope in self {
            parameters.extend(scope.checkpoint_parameters()?);
        }
        Ok(parameters)
    }
}
macro_rules! plain_checkpoint {
    ($($ty:ty),+) => {$(impl DeclareCheckpointParameters for $ty {
        fn checkpoint_parameters(&self) -> Result<Vec<CheckpointParameter<'_>>, ComputeError> {
            Ok(self.logical_parameters()?.into_iter().map(|logical| CheckpointParameter { logical, axes: None }).collect())
        }
    })+};
}
plain_checkpoint!(Slot, LinearSpec, LayerNormalizationConstructionSpec);

/// One explicitly owned non-affine tensor slot.
#[derive(Debug, Clone)]
pub(crate) struct Slot {
    pub parameter: ParameterSpec,
    pub shape: Vec<i32>,
    pub role: LogicalParameterRole,
}
impl Slot {
    fn new(
        root: &str,
        field: &str,
        shape: Vec<i32>,
        role: LogicalParameterRole,
    ) -> Result<Self, Error> {
        Ok(Self {
            parameter: parameter(root, field)?,
            shape,
            role,
        })
    }
    pub fn construct<T: Tensor>(self, context: &T::Context) -> Result<Parameter<T>, Error> {
        Ok(Parameter::unloaded(self.parameter, &self.shape, context)?)
    }
}
impl DeclareLogicalParameters for Slot {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, ComputeError> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.parameter,
            &self.shape,
            self.role,
        )?])
    }
}
/// Canonical internal OKI kernel slots and causal state geometry.
#[derive(Debug, Clone)]
pub(crate) struct Conv {
    pub geometry: Conv1dSpec,
    pub weight: ParameterSpec,
    pub bias: Option<ParameterSpec>,
}
impl Conv {
    fn new(root: &str, geometry: Conv1dSpec) -> Result<Self, Error> {
        Ok(Self {
            geometry,
            weight: parameter(root, "weight")?,
            bias: geometry.bias.then(|| parameter(root, "bias")).transpose()?,
        })
    }
    fn shape(&self) -> [i32; 3] {
        let g = self.geometry;
        [
            g.output_channels,
            g.kernel_size,
            g.input_channels / g.groups,
        ]
    }
    pub fn construct<T: Tensor>(self, context: &T::Context) -> Result<StreamableConv1d<T>, Error> {
        let weight = Parameter::unloaded(self.weight.clone(), &self.shape(), context)?;
        let bias = self
            .bias
            .map(|p| Parameter::unloaded(p, &[self.geometry.output_channels], context))
            .transpose()?;
        StreamableConv1d::new(self.geometry, weight, bias)
    }
}
impl DeclareLogicalParameters for Conv {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, ComputeError> {
        let mut p = vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.shape(),
            LogicalParameterRole::LinearWeight,
        )?];
        if let Some(b) = &self.bias {
            p.push(LogicalParameterDeclaration::new(
                b,
                &[self.geometry.output_channels],
                LogicalParameterRole::LinearBias,
            )?);
        }
        Ok(p)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Transpose {
    pub geometry: ConvTranspose1dSpec,
    source_layout: SourceKernelLayout,
    pub weight: ParameterSpec,
    pub bias: Option<ParameterSpec>,
}
impl Transpose {
    fn new(
        root: &str,
        geometry: ConvTranspose1dSpec,
        source_layout: SourceKernelLayout,
    ) -> Result<Self, Error> {
        Ok(Self {
            geometry,
            source_layout,
            weight: parameter(root, "weight")?,
            bias: geometry.bias.then(|| parameter(root, "bias")).transpose()?,
        })
    }
    fn shape(&self) -> [i32; 3] {
        let g = self.geometry;
        [
            g.output_channels,
            g.kernel_size,
            g.input_channels / g.groups,
        ]
    }
    pub fn construct<T: Tensor>(
        self,
        context: &T::Context,
    ) -> Result<StreamableConvTranspose1d<T>, Error> {
        let weight = Parameter::unloaded(self.weight.clone(), &self.shape(), context)?;
        let bias = self
            .bias
            .map(|p| Parameter::unloaded(p, &[self.geometry.output_channels], context))
            .transpose()?;
        StreamableConvTranspose1d::new(self.geometry, weight, bias)
    }
}
impl DeclareLogicalParameters for Transpose {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, ComputeError> {
        let mut p = vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.shape(),
            LogicalParameterRole::LinearWeight,
        )?];
        if let Some(b) = &self.bias {
            p.push(LogicalParameterDeclaration::new(
                b,
                &[self.geometry.output_channels],
                LogicalParameterRole::LinearBias,
            )?);
        }
        Ok(p)
    }
}
impl DeclareCheckpointParameters for Conv {
    fn checkpoint_parameters(&self) -> Result<Vec<CheckpointParameter<'_>>, ComputeError> {
        Ok(self
            .logical_parameters()?
            .into_iter()
            .map(|logical| {
                let axes = (logical.parameter.id == self.weight.id)
                    .then(|| SourceKernelLayout::OutputInputKernel.axes());
                CheckpointParameter { logical, axes }
            })
            .collect())
    }
}
impl DeclareCheckpointParameters for Transpose {
    fn checkpoint_parameters(&self) -> Result<Vec<CheckpointParameter<'_>>, ComputeError> {
        Ok(self
            .logical_parameters()?
            .into_iter()
            .map(|logical| {
                let axes =
                    (logical.parameter.id == self.weight.id).then(|| self.source_layout.axes());
                CheckpointParameter { logical, axes }
            })
            .collect())
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Residual {
    pub block: Vec<Conv>,
}
impl Residual {
    fn new(root: &str, channels: i32) -> Result<Self, Error> {
        Ok(Self {
            block: vec![
                Conv::new(
                    &format!("{root}.block.0"),
                    Conv1dSpec::causal(channels, channels / 2, 3, 1),
                )?,
                Conv::new(
                    &format!("{root}.block.1"),
                    Conv1dSpec::causal(channels / 2, channels, 1, 1),
                )?,
            ],
        })
    }
}
#[derive(Debug, Clone)]
pub(crate) struct EncoderLayer {
    pub residuals: Vec<Residual>,
    pub downsample: Conv,
}
#[derive(Debug, Clone)]
pub(crate) struct DecoderLayer {
    pub upsample: Transpose,
    pub residuals: Vec<Residual>,
}
#[derive(Debug, Clone)]
pub(crate) struct Encoder {
    pub init: Conv,
    pub layers: Vec<EncoderLayer>,
    pub final_conv: Conv,
}
impl Encoder {
    fn new(input: i32, latent: i32) -> Result<Self, Error> {
        let mut channels = 64;
        let mut layers = vec![];
        for (index, ratio) in [4, 5, 6, 8].into_iter().enumerate() {
            let root = format!("encoder.layers.{index}");
            layers.push(EncoderLayer {
                residuals: vec![Residual::new(&format!("{root}.residuals.0"), channels)?],
                downsample: Conv::new(
                    &format!("{root}.downsample"),
                    Conv1dSpec::causal(channels, channels * 2, ratio * 2, ratio),
                )?,
            });
            channels *= 2;
        }
        Ok(Self {
            init: Conv::new("encoder.init_conv1d", Conv1dSpec::causal(input, 64, 7, 1))?,
            layers,
            final_conv: Conv::new(
                "encoder.final_conv1d",
                Conv1dSpec::causal(channels, latent, 3, 1),
            )?,
        })
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Decoder {
    pub init: Conv,
    pub layers: Vec<DecoderLayer>,
    pub final_conv: Conv,
}
impl Decoder {
    fn new(latent: i32, output: i32) -> Result<Self, Error> {
        let mut channels = 1024;
        let mut layers = vec![];
        for (index, ratio) in [8, 6, 5, 4].into_iter().enumerate() {
            let root = format!("decoder.layers.{index}");
            let output = channels / 2;
            layers.push(DecoderLayer {
                upsample: Transpose::new(
                    &format!("{root}.upsample"),
                    ConvTranspose1dSpec {
                        input_channels: channels,
                        output_channels: output,
                        kernel_size: ratio * 2,
                        stride: ratio,
                        groups: 1,
                        bias: true,
                        tail: TransposeTail::Trim,
                    },
                    SourceKernelLayout::InputOutputKernel,
                )?,
                residuals: vec![Residual::new(&format!("{root}.residuals.0"), output)?],
            });
            channels = output;
        }
        Ok(Self {
            init: Conv::new(
                "decoder.init_conv1d",
                Conv1dSpec::causal(latent, 1024, 7, 1),
            )?,
            layers,
            final_conv: Conv::new(
                "decoder.final_conv1d",
                Conv1dSpec::causal(channels, output, 3, 1),
            )?,
        })
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Attention {
    pub input: LinearSpec,
    pub output: LinearSpec,
    pub heads: i32,
    pub head_dim: i32,
    pub context: i32,
}
#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    pub up: LinearSpec,
    pub down: LinearSpec,
}
#[derive(Debug, Clone)]
pub(crate) struct TransformerLayer {
    pub norm1: LayerNormalizationConstructionSpec,
    pub norm2: LayerNormalizationConstructionSpec,
    pub attention: Attention,
    pub mlp: Mlp,
    pub scale1: Slot,
    pub scale2: Slot,
}
#[derive(Debug, Clone)]
pub(crate) struct Transformer {
    pub layers: Vec<TransformerLayer>,
}
impl Transformer {
    fn new(root: &str, width: i32) -> Result<Self, Error> {
        Ok(Self {
            layers: (0..8)
                .map(|index| {
                    let root = format!("{root}.layers.{index}");
                    Ok(TransformerLayer {
                        norm1: norm(&format!("{root}.norm1"), width)?,
                        norm2: norm(&format!("{root}.norm2"), width)?,
                        attention: Attention {
                            input: linear(&format!("{root}.self_attn.in_proj"), width, width * 3)?,
                            output: linear(&format!("{root}.self_attn.out_proj"), width, width)?,
                            heads: 8,
                            head_dim: width / 8,
                            context: 250,
                        },
                        mlp: Mlp {
                            up: linear(&format!("{root}.mlp.linear1"), width, width * 4)?,
                            down: linear(&format!("{root}.mlp.linear2"), width * 4, width)?,
                        },
                        scale1: Slot::new(
                            &format!("{root}.layer_scale_1"),
                            "scale",
                            vec![width],
                            LogicalParameterRole::ResidualScale,
                        )?,
                        scale2: Slot::new(
                            &format!("{root}.layer_scale_2"),
                            "scale",
                            vec![width],
                            LogicalParameterRole::ResidualScale,
                        )?,
                    })
                })
                .collect::<Result<_, Error>>()?,
        })
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Codebook {
    pub initialized: Slot,
    pub usage: Slot,
    pub sum: Slot,
}
impl Codebook {
    fn new(root: &str, bins: i32, dim: i32) -> Result<Self, Error> {
        Ok(Self {
            initialized: Slot::new(
                root,
                "_initialized",
                vec![1],
                LogicalParameterRole::CodebookInitialization,
            )?,
            usage: Slot::new(
                root,
                "cluster_usage",
                vec![bins],
                LogicalParameterRole::CodebookUsage,
            )?,
            sum: Slot::new(
                root,
                "embedding_sum",
                vec![bins, dim],
                LogicalParameterRole::CodebookSum,
            )?,
        })
    }
}
#[derive(Debug, Clone)]
pub(crate) struct QuantizerBranch {
    pub input: Slot,
    pub output: Slot,
    pub books: Vec<Codebook>,
}
impl QuantizerBranch {
    fn new(root: &str, latent: i32, dim: i32, bins: i32, count: i32) -> Result<Self, Error> {
        Ok(Self {
            input: Slot::new(
                &format!("{root}.input_proj"),
                "weight",
                vec![dim, latent, 1],
                LogicalParameterRole::LinearWeight,
            )?,
            output: Slot::new(
                &format!("{root}.output_proj"),
                "weight",
                vec![latent, dim, 1],
                LogicalParameterRole::LinearWeight,
            )?,
            books: (0..count)
                .map(|index| {
                    Codebook::new(&format!("{root}.vq.layers.{index}._codebook"), bins, dim)
                })
                .collect::<Result<_, _>>()?,
        })
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Quantizer {
    pub first: QuantizerBranch,
    pub rest: QuantizerBranch,
    pub count: i32,
}
impl Quantizer {
    fn new(c: &Config) -> Result<Self, Error> {
        Ok(Self {
            first: QuantizerBranch::new(
                "quantizer.rvq_first",
                c.latent_dim,
                c.quantizer_dim,
                c.bins,
                1,
            )?,
            rest: QuantizerBranch::new(
                "quantizer.rvq_rest",
                c.latent_dim,
                c.quantizer_dim,
                c.bins,
                c.num_codebooks - 1,
            )?,
            count: c.num_codebooks,
        })
    }
}
/// Validated codec-owned parameter topology, without backend tensors or execution state.
#[derive(Debug, Clone)]
pub struct MimiSpec {
    pub(crate) config: Config,
    pub(crate) quantizer: Quantizer,
    pub(crate) encoder: Encoder,
    pub(crate) encoder_transformer: Transformer,
    pub(crate) downsample: Conv,
    pub(crate) upsample: Transpose,
    pub(crate) decoder_transformer: Transformer,
    pub(crate) decoder: Decoder,
}
impl MimiSpec {
    /// Declares typed codec scopes; checkpoint admission selects active codebooks from released scopes.
    pub fn new(config: Config) -> Result<Self, Error> {
        config.validate()?;
        let spec = Self {
            quantizer: Quantizer::new(&config)?,
            encoder: Encoder::new(config.channels, config.latent_dim)?,
            encoder_transformer: Transformer::new("encoder_transformer", config.latent_dim)?,
            downsample: Conv::new(
                "downsample",
                Conv1dSpec {
                    bias: false,
                    pad_mode: PadMode::Edge,
                    ..Conv1dSpec::causal(config.latent_dim, config.latent_dim, 4, 2)
                },
            )?,
            upsample: Transpose::new(
                "upsample",
                ConvTranspose1dSpec {
                    input_channels: config.latent_dim,
                    output_channels: config.latent_dim,
                    kernel_size: 4,
                    stride: 2,
                    groups: config.latent_dim,
                    bias: false,
                    tail: TransposeTail::Trim,
                },
                SourceKernelLayout::OutputInputKernel,
            )?,
            decoder_transformer: Transformer::new("decoder_transformer", config.latent_dim)?,
            decoder: Decoder::new(config.latent_dim, config.channels)?,
            config,
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Projects active codebooks from retained released scopes without rebuilding shared parameters.
    pub fn select_codebooks(&self, count: i32) -> Result<Self, Error> {
        if count <= 0 || count > self.quantizer.count {
            return Err(Error::InvalidShape(
                "active Mimi codebooks exceed the declared scopes".into(),
            ));
        }
        let mut selected = self.clone();
        selected.quantizer.rest.books.truncate((count - 1) as usize);
        selected.quantizer.count = count;
        selected.config.num_codebooks = count;
        selected.config.validate()?;
        validate(&selected)?;
        Ok(selected)
    }

    /// Validated inference policy and selected active codebook count.
    pub fn config(&self) -> &Config {
        &self.config
    }
}
macro_rules! declare {
    ($ty:ty, $($field:ident),+) => {
        impl DeclareLogicalParameters for $ty {
            fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, ComputeError> {
                let mut p = vec![]; $(p.extend(self.$field.logical_parameters()?);)+ Ok(p)
            }
        }
        impl DeclareCheckpointParameters for $ty {
            fn checkpoint_parameters(&self) -> Result<Vec<CheckpointParameter<'_>>, ComputeError> {
                let mut p = vec![]; $(p.extend(self.$field.checkpoint_parameters()?);)+ Ok(p)
            }
        }
    }
}
declare!(Residual, block);
declare!(EncoderLayer, residuals, downsample);
declare!(DecoderLayer, upsample, residuals);
declare!(Encoder, init, layers, final_conv);
declare!(Decoder, init, layers, final_conv);
declare!(Attention, input, output);
declare!(Mlp, up, down);
declare!(
    TransformerLayer,
    norm1,
    norm2,
    attention,
    mlp,
    scale1,
    scale2
);
declare!(Transformer, layers);
declare!(Codebook, initialized, usage, sum);
declare!(QuantizerBranch, input, output, books);
declare!(Quantizer, first, rest);
declare!(
    MimiSpec,
    quantizer,
    encoder,
    encoder_transformer,
    downsample,
    upsample,
    decoder_transformer,
    decoder
);
