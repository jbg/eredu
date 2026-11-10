//! Causal depthwise FastConformer subsampling with canonical checkpoint storage.
//!
//! Raw parameters retain NVIDIA OIHW geometry. Ordinary backend convolutions
//! receive OHWI views over NHWC activations. Streaming window ownership and the
//! removal of overlapping output frames belong to the perception driver.

use eredu_nn::{Error, LinearOperator, NeuralBackend, PadMode, Parameter, Parameterized, Tensor};

use super::config::PerceptionEncoderConfig;

/// Released eightfold subsampling dimensions, independent of execution context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsamplingDimensions {
    /// Number of input mel frequency bins.
    pub features: i32,
    /// Convolutional channel count of all three stages.
    pub channels: i32,
    /// Projected encoder width.
    pub width: i32,
}

impl SubsamplingDimensions {
    /// Resolves the admitted causal `dw_striding` subsampling policy.
    pub fn from_config(config: &PerceptionEncoderConfig) -> Result<Self, SubsamplingError> {
        if !config.causal_downsampling
            || config.subsampling != "dw_striding"
            || config.subsampling_factor != 8
        {
            return Err(SubsamplingError::Configuration(
                "subsampling changes the released causal depthwise equations",
            ));
        }
        let convert = |value| {
            i32::try_from(value).map_err(|_| {
                SubsamplingError::Configuration("subsampling dimension exceeds tensor indexing")
            })
        };
        let dimensions = Self {
            features: convert(config.feat_in)?,
            channels: convert(config.subsampling_conv_channels)?,
            width: convert(config.d_model)?,
        };
        dimensions.validate()?;
        Ok(dimensions)
    }

    /// Validates dimensions and the flattened convolution projection width.
    pub fn validate(self) -> Result<(), SubsamplingError> {
        if self.features <= 0 || self.channels <= 0 || self.width <= 0 {
            return Err(SubsamplingError::Configuration(
                "subsampling dimensions must be positive",
            ));
        }
        self.flattened_width()?;
        Ok(())
    }

    /// Exact raw output length, including each stage's right padding.
    ///
    /// Each causal kernel has left padding two, right padding one, kernel three
    /// and stride two. Its output length is `floor(input / 2) + 1`.
    pub fn output_frames(self, input_frames: i32) -> Result<i32, SubsamplingError> {
        if input_frames <= 0 || input_frames.checked_add(3).is_none() {
            return Err(SubsamplingError::Input(
                "subsampling input length must be positive and allow causal padding",
            ));
        }
        let mut length = input_frames;
        for _ in 0..3 {
            length = length / 2 + 1;
        }
        Ok(length)
    }

    pub(super) fn flattened_width(self) -> Result<i32, SubsamplingError> {
        self.output_frames(self.features)?
            .checked_mul(self.channels)
            .filter(|width| *width > 0)
            .ok_or(SubsamplingError::Configuration(
                "flattened subsampling feature width overflows tensor indexing",
            ))
    }
}

/// Subsampling admission or numerical failure.
#[derive(Debug, thiserror::Error)]
pub enum SubsamplingError {
    /// The dimensions or equations differ from the selected component.
    #[error("{0}")]
    Configuration(&'static str),
    /// Input geometry cannot execute under this component's contract.
    #[error("{0}")]
    Input(&'static str),
    /// Input or parameters differ from the selected floating execution policy.
    #[error("{0}")]
    Precision(&'static str),
    /// Backend-neutral numerical failure retaining its original source.
    #[error(transparent)]
    Neural(#[from] Error),
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct Convolution<T> {
    weight: Parameter<T>,
    bias: Parameter<T>,
    #[parameter(skip)]
    stride: i32,
    #[parameter(skip)]
    groups: i32,
}

impl<T: Tensor> Convolution<T> {
    fn new(
        spec: super::perception_specs::ConvolutionSpec,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: Parameter::unloaded(spec.weight, &spec.shape, context)?,
            bias: Parameter::unloaded(spec.bias, &[spec.shape[0]], context)?,
            stride: spec.stride,
            groups: spec.groups,
        })
    }

    fn forward(
        &self,
        input: &T,
        precision: super::precision::SpeechPrecision,
        context: &T::Context,
    ) -> Result<T, Error> {
        let input = if self.stride == 2 {
            T::pad(
                input,
                &[(0, 0), (2, 1), (2, 1), (0, 0)],
                PadMode::Constant,
                context,
            )?
        } else {
            input.clone()
        };
        let value = T::conv2d(
            &precision.conv_input(&input, context)?,
            &self
                .weight
                .as_ref()
                .transpose_axes(&[0, 2, 3, 1], context)?,
            (self.stride, self.stride),
            (0, 0),
            (1, 1),
            self.groups,
            context,
        )?;
        precision
            .restore(value, context)?
            .add(self.bias.as_ref(), context)
    }
}

/// Three causal convolution stages and their final encoder projection.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct CausalSubsampling<B: NeuralBackend> {
    #[parameter(skip)]
    precision: super::precision::SpeechPrecision,
    #[parameter(skip)]
    precision_declaration: super::precision::SpeechPrecisionDeclaration,
    #[parameter(skip)]
    dimensions: SubsamplingDimensions,
    initial: Convolution<B::Tensor>,
    depthwise1: Convolution<B::Tensor>,
    pointwise1: Convolution<B::Tensor>,
    depthwise2: Convolution<B::Tensor>,
    pointwise2: Convolution<B::Tensor>,
    projection: B::Linear,
}

impl<B: NeuralBackend> Clone for CausalSubsampling<B> {
    fn clone(&self) -> Self {
        Self {
            precision: self.precision,
            precision_declaration: self.precision_declaration.clone(),
            dimensions: self.dimensions,
            initial: self.initial.clone(),
            depthwise1: self.depthwise1.clone(),
            pointwise1: self.pointwise1.clone(),
            depthwise2: self.depthwise2.clone(),
            pointwise2: self.pointwise2.clone(),
            projection: self.projection.clone(),
        }
    }
}

impl<B: NeuralBackend> CausalSubsampling<B> {
    /// Selects projection arithmetic; parameters are supplied by the matching retained plan.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, SubsamplingError> {
        precision.validate().map_err(SubsamplingError::Precision)?;
        self.projection
            .set_compute_element(precision.projection_element())?;
        self.precision = precision;
        Ok(self)
    }

    /// Constructs canonical released convolution and output parameter identities.
    pub fn new(
        spec: super::perception_specs::SubsamplingSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, SubsamplingError> {
        let precision_declaration = super::precision::SpeechPrecisionDeclaration::from_spec(&spec)?;
        let [initial, depthwise1, pointwise1, depthwise2, pointwise2] = spec.convolutions;
        Ok(Self {
            precision: Default::default(),
            precision_declaration,
            dimensions: spec.dimensions,
            initial: Convolution::new(initial, context)?,
            depthwise1: Convolution::new(depthwise1, context)?,
            pointwise1: Convolution::new(pointwise1, context)?,
            depthwise2: Convolution::new(depthwise2, context)?,
            pointwise2: Convolution::new(pointwise2, context)?,
            projection: B::linear(spec.output, context)?,
        })
    }

    /// Logical component geometry.
    pub fn dimensions(&self) -> SubsamplingDimensions {
        self.dimensions
    }

    /// Executes one supplied `[batch, mel_time, features]` window.
    ///
    /// Returns every raw encoder frame. The perception driver determines window
    /// overlap and discards corresponding output rows explicitly.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, SubsamplingError> {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] <= 0 || shape[2] != self.dimensions.features {
            return Err(SubsamplingError::Input(
                "subsampling input expects [batch, mel_time, features]",
            ));
        }
        let frames = self.dimensions.output_frames(shape[1])?;
        self.precision
            .require(self, input, &self.precision_declaration)
            .map_err(SubsamplingError::Precision)?;
        let input = input.expand_dims(-1, context)?;
        let hidden = self
            .initial
            .forward(&input, self.precision, context)?
            .maximum_scalar(0.0, context)?;
        let hidden = self.depthwise1.forward(&hidden, self.precision, context)?;
        let hidden = self
            .pointwise1
            .forward(&hidden, self.precision, context)?
            .maximum_scalar(0.0, context)?;
        let hidden = self.depthwise2.forward(&hidden, self.precision, context)?;
        let hidden = self
            .pointwise2
            .forward(&hidden, self.precision, context)?
            .maximum_scalar(0.0, context)?;
        // NVIDIA flattens channels before frequency after converting NCTF to
        // NTCF. Flattening NHWC directly would silently interchange these axes.
        let flattened = hidden.transpose_axes(&[0, 1, 3, 2], context)?.reshape(
            &[shape[0], frames, self.dimensions.flattened_width()?],
            context,
        )?;
        Ok(self.projection.forward(&flattened, context)?)
    }
}
