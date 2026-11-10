//! Independently owned perception boundary parameters and their convolution layout contracts.
mod conformer;
pub use conformer::ConformerSpec;

use super::{
    perception::{PerceptionDimensions, PerceptionError},
    subsampling::{SubsamplingDimensions, SubsamplingError},
};
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration, LogicalParameterRole,
    ParameterSpec,
};

pub(crate) const SUBSAMPLING_ROOT: &str = "stt_model.perception.encoder.pre_encode";
#[derive(Debug, Clone)]
pub(crate) struct ConvolutionSpec {
    pub index: usize,
    pub weight: ParameterSpec,
    pub bias: ParameterSpec,
    pub shape: [i32; 4],
    pub stride: i32,
    pub groups: i32,
}
impl ConvolutionSpec {
    fn new(
        index: usize,
        input: i32,
        output: i32,
        kernel: i32,
        stride: i32,
        groups: i32,
    ) -> Result<Self, Error> {
        let parameter = |name| {
            let mut p = ParameterSpec::trainable(format!("{SUBSAMPLING_ROOT}.conv.{index}.{name}"))
                .map_err(Error::backend)?;
            p.group = Some(format!("voicechat.perception.subsampling.conv.{index}"));
            Ok::<_, Error>(p)
        };
        Ok(Self {
            index,
            weight: parameter("weight")?,
            bias: parameter("bias")?,
            shape: [output, input / groups, kernel, kernel],
            stride,
            groups,
        })
    }
}
impl DeclareLogicalParameters for ConvolutionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![
            LogicalParameterDeclaration::new(
                &self.weight,
                &self.shape,
                LogicalParameterRole::LinearWeight,
            )?,
            LogicalParameterDeclaration::new(
                &self.bias,
                &[self.shape[0]],
                LogicalParameterRole::LinearBias,
            )?,
        ])
    }
}
/// Five biased OIHW convolutions and their final encoder-width affine projection.
#[derive(Debug, Clone)]
pub struct SubsamplingSpec {
    pub(crate) dimensions: SubsamplingDimensions,
    pub(crate) convolutions: [ConvolutionSpec; 5],
    pub(crate) output: LinearSpec,
}
impl SubsamplingSpec {
    /// Validates all boundary parameter geometry without allocating backend tensors.
    pub fn new(dimensions: SubsamplingDimensions) -> Result<Self, SubsamplingError> {
        dimensions.validate()?;
        let c = dimensions.channels;
        let parameter = |name| {
            let mut p = ParameterSpec::trainable(format!("{SUBSAMPLING_ROOT}.out.{name}"))
                .map_err(Error::backend)?;
            p.group = Some("voicechat.perception.subsampling.output".into());
            Ok::<_, Error>(p)
        };
        let spec = Self {
            dimensions,
            convolutions: [
                ConvolutionSpec::new(0, 1, c, 3, 2, 1)?,
                ConvolutionSpec::new(2, c, c, 3, 2, c)?,
                ConvolutionSpec::new(3, c, c, 1, 1, 1)?,
                ConvolutionSpec::new(5, c, c, 3, 2, c)?,
                ConvolutionSpec::new(6, c, c, 1, 1, 1)?,
            ],
            output: LinearSpec {
                input: dimensions.flattened_width()?,
                output: dimensions.width,
                weight: parameter("weight")?,
                bias: Some(parameter("bias")?),
                format: eredu_nn::LinearFormatSpec::unscaled(
                    eredu_checkpoint::LinearFormat::Dense,
                )?,
            },
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for SubsamplingSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.convolutions[..].logical_parameters()?;
        p.extend(self.output.logical_parameters()?);
        Ok(p)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct PinnedSpec {
    pub subsampling: SubsamplingSpec,
    pub projection: LinearSpec,
}
impl PinnedSpec {
    pub fn new(d: PerceptionDimensions) -> Result<Self, PerceptionError> {
        d.validate()?;
        d.block.validate()?;
        let spec = Self {
            subsampling: SubsamplingSpec::new(d.subsampling)?,
            projection: projection_spec(d)?,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for PinnedSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.subsampling.logical_parameters()?;
        p.extend(self.projection.logical_parameters()?);
        Ok(p)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Declaration {
    pub pinned: PinnedSpec,
    pub blocks: Vec<ConformerSpec>,
}
impl Declaration {
    pub fn new(d: PerceptionDimensions) -> Result<Self, PerceptionError> {
        let declaration = Self {
            pinned: PinnedSpec::new(d)?,
            blocks: (0..d.layers)
                .map(|index| ConformerSpec::at(d.block, index))
                .collect::<Result<_, _>>()?,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(declaration)
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.pinned.logical_parameters()?;
        p.extend(self.blocks[..].logical_parameters()?);
        Ok(p)
    }
}

fn projection_spec(dimensions: PerceptionDimensions) -> Result<LinearSpec, PerceptionError> {
    if dimensions.block.width <= 0 || dimensions.output_width <= 0 {
        return Err(PerceptionError::Configuration(
            "invalid perception projection widths",
        ));
    }
    let parameter = |name: &str| {
        let mut spec = ParameterSpec::trainable(format!("stt_model.perception.proj.{name}"))
            .map_err(eredu_nn::Error::backend)?;
        spec.group = Some(super::perception_layered::PROJECTION.into());
        Ok::<_, eredu_nn::Error>(spec)
    };
    let spec = LinearSpec {
        input: dimensions.block.width,
        output: dimensions.output_width,
        weight: parameter("weight")?,
        bias: Some(parameter("bias")?),
        format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
    };
    use eredu_nn::DeclareLogicalParameters;
    eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
        .map_err(eredu_nn::Error::backend_source)?;
    Ok(spec)
}

/// Retained perception boundary slots, Conformer owners and component execution policy.
/// Mutable carry, stage invocation and observation permissions remain separate contracts.
#[derive(Debug, Clone)]
pub struct PerceptionDeclaration {
    dimensions: PerceptionDimensions,
    spec: Declaration,
    definition: eredu_runtime::ComponentDefinition,
    precision: super::precision::SpeechPrecision,
}
impl PerceptionDeclaration {
    /// Declares complete connected geometry without backend resources or payload access.
    pub fn new(dimensions: PerceptionDimensions) -> Result<Self, PerceptionError> {
        let spec = Declaration::new(dimensions)?;
        let definition = super::perception_layered::declared_definition(dimensions, &spec)?;
        Ok(Self {
            dimensions,
            spec,
            definition,
            precision: Default::default(),
        })
    }
    /// Connected subsampling, Conformer and output geometry.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
    }
    /// Validated parameter ownership, independent leases and mutable-state boundaries.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }
    /// Selected storage and arithmetic policy; histories and sensitive reductions stay F32.
    pub fn precision(&self) -> super::precision::SpeechPrecision {
        self.precision
    }
    /// Selects precision without changing canonical source geometry or ownership.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, PerceptionError> {
        precision
            .validate()
            .map_err(PerceptionError::Configuration)?;
        self.precision = precision;
        Ok(self)
    }
    /// Resolves coherent complete-head and feed-forward domains using retained slots.
    pub fn local_dimensions(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<Vec<super::conformer::ConformerLocalDimensions>, PerceptionError> {
        super::perception_layered::declared_local_dimensions(self.dimensions, layout, &self.spec)
    }
    pub(crate) fn spec(&self) -> &Declaration {
        &self.spec
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        PerceptionDimensions,
        Declaration,
        eredu_runtime::ComponentDefinition,
        super::precision::SpeechPrecision,
    ) {
        (self.dimensions, self.spec, self.definition, self.precision)
    }
}
impl eredu_runtime::ComponentPreparationDeclaration for PerceptionDeclaration {
    fn parameters(&self) -> &eredu_runtime::ArchitectureParameterDescription {
        self.definition.parameters()
    }
}
/// Every admitted VoiceChat configuration validates against this source-free published scope.
/// It contains no checkpoint values, backend resources or mutable invocation state.
pub(crate) fn released_declaration() -> Result<&'static PerceptionDeclaration, PerceptionError> {
    static DECLARATION: std::sync::OnceLock<Result<PerceptionDeclaration, String>> =
        std::sync::OnceLock::new();
    DECLARATION
        .get_or_init(|| {
            let config = super::config::released_configuration().map_err(|e| e.to_string())?;
            let dimensions =
                PerceptionDimensions::from_config(&config.perception).map_err(|e| e.to_string())?;
            PerceptionDeclaration::new(dimensions).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|reason| Error::backend(reason.clone()).into())
}
