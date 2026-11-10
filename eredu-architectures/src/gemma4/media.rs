//! Gemma 4 media-to-text projection shared by image and audio features.

use eredu_nn::{
    DeclareLogicalParameters, Error, LinearOperator, LinearSpec, LogicalParameterDeclaration,
    NeuralBackend, ParameterSpec, Parameterized, Tensor,
};

use super::ModelArgs;

/// Encoder role, independent of the projector's actual parameter root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionKind {
    /// Image/video features entering the text decoder.
    Vision,
    /// Audio features entering the text decoder.
    Audio,
}

impl ProjectionKind {
    /// Canonical Gemma parameter identity, also referenced by source extensions.
    pub const fn gemma_weight(self) -> &'static str {
        match self {
            Self::Vision => "model.embed_vision.embedding_projection.weight",
            Self::Audio => "model.embed_audio.embedding_projection.weight",
        }
    }
}

/// Authoritative media-to-text matrix and the separate weightless input normalization.
#[derive(Debug, Clone)]
pub struct ProjectionSpec {
    /// Semantic encoder role; source aliases remain format extensions.
    pub kind: ProjectionKind,
    /// Exact matrix identity, geometry, and selected encoding.
    pub projection: LinearSpec,
    /// Epsilon used before the projection, without a learned gain.
    pub epsilon: f32,
}
impl ProjectionSpec {
    /// Diffusion image projector identity under the encoder component.
    pub const DIFFUSION_VISION_WEIGHT: &'static str =
        "model.encoder.embed_vision.embedding_projection.weight";
    /// Gemma vision projector with the tower's released alignment/format policy.
    pub(in crate::gemma4) fn vision(
        output: i32,
        config: &super::VisionConfig,
    ) -> Result<Self, Error> {
        config.require_root(super::VisionOwner::Gemma4.root())?;
        Self::build(
            ProjectionKind::Vision,
            ProjectionKind::Vision.gemma_weight(),
            config.hidden_size,
            output,
            config.rms_norm_eps,
            |name, input| config.linear_format_for(name, input),
        )
    }
    /// Gemma audio projector from the tower's output features.
    pub(in crate::gemma4) fn audio(
        output: i32,
        config: &super::AudioConfig,
    ) -> Result<Self, Error> {
        Self::build(
            ProjectionKind::Audio,
            ProjectionKind::Audio.gemma_weight(),
            config.output_proj_dims,
            output,
            config.rms_norm_eps,
            |name, input| config.linear_format_for(name, input),
        )
    }
    /// Diffusion image projector under its independently owned encoder root.
    pub(crate) fn diffusion_vision(
        text: &ModelArgs,
        config: &super::VisionConfig,
    ) -> Result<Self, Error> {
        config.require_root(super::VisionOwner::DiffusionGemma.root())?;
        Self::build(
            ProjectionKind::Vision,
            Self::DIFFUSION_VISION_WEIGHT,
            config.hidden_size,
            text.hidden_size,
            config.rms_norm_eps,
            |name, _| text.linear_format_for(name),
        )
    }
    fn build(
        kind: ProjectionKind,
        name: &str,
        input: i32,
        output: i32,
        epsilon: f32,
        format: impl FnOnce(&str, i32) -> eredu_checkpoint::LinearFormat,
    ) -> Result<Self, Error> {
        let spec = Self {
            kind,
            projection: LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(name, format(name, input))?,
            },
            epsilon,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Projects storage encoding while retaining the declared identity and equation.
    pub(crate) fn with_format(
        &self,
        encoding: eredu_checkpoint::LinearFormat,
    ) -> Result<Self, Error> {
        let mut exact = self.clone();
        let mut format = crate::linear_format::standard_linear_format(
            self.projection.weight.id.as_str(),
            encoding,
        )?
        .with_arithmetic(self.projection.format.arithmetic())?;
        if self.projection.format.exact_zero_inputs() {
            format = format.with_exact_zero_inputs();
        }
        exact.projection.format = format;
        exact.validate()?;
        Ok(exact)
    }
    /// Rejects invalid geometry, normalization, or parameter topology before native construction.
    pub fn validate(&self) -> Result<(), Error> {
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::backend("invalid modality normalization epsilon"));
        }
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        self.parameter_group("media_projection")?;
        Ok(())
    }
    /// Replicated static storage, including the actual encoding companions.
    pub fn parameter_group(
        &self,
        name: impl Into<String>,
    ) -> Result<eredu_runtime::ParameterGroupSpec, Error> {
        eredu_runtime::logical_parameter_group(
            name,
            eredu_runtime::ParameterRole::Replicated,
            None,
            self,
            |_| eredu_runtime::MemberSharding::Replicated,
        )
        .map_err(Error::backend)
    }
}
impl DeclareLogicalParameters for ProjectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.projection.logical_parameters()
    }
}

/// Weightless RMS normalization followed by a learned media-to-text projection.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ModalityProjector<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    projection: B::Linear,
    #[parameter(skip)]
    epsilon: f32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> ModalityProjector<B> {
    /// Constructs the already validated family declaration without reconstructing its slots.
    pub fn new(
        spec: ProjectionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            projection: B::linear(spec.projection, context)?,
            epsilon: spec.epsilon,
        })
    }

    /// Normalizes encoder features and projects them into decoder hidden space.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let normalized = B::rms_norm_without_weight(input, self.epsilon, context)?;
        self.projection.forward(&normalized, context)
    }
}

pub(crate) fn clipped_linear(
    prefix: &str,
    input: i32,
    output: i32,
    format: eredu_checkpoint::LinearFormat,
    bounds: bool,
) -> Result<eredu_nn::ClippedLinearSpec, Error> {
    let parameter =
        |field| ParameterSpec::trainable(format!("{prefix}.{field}")).map_err(Error::backend);
    let weight = format!("{prefix}.linear.weight");
    let spec = eredu_nn::ClippedLinearSpec {
        linear: LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(&weight, format)?,
        },
        bounds: bounds
            .then(|| {
                Ok::<_, Error>(eredu_nn::LinearClippingSpec {
                    input_min: parameter("input_min")?,
                    input_max: parameter("input_max")?,
                    output_min: parameter("output_min")?,
                    output_max: parameter("output_max")?,
                })
            })
            .transpose()?,
    };
    spec.validate()?;
    Ok(spec)
}
