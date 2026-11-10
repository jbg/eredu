//! RNNT parameter owners, distinct from phase invocations and recurrent state.
use super::{
    rnnt::{RnntDimensions, RnntError},
    rnnt_layered::RnntPhase,
};
use eredu_nn::{
    recurrent::LstmLayerSpec, DeclareLogicalParameters, EmbeddingSpec, Error, LinearSpec,
    LogicalParameterDeclaration, ParameterSpec,
};

pub(super) const EMBEDDING: &str = "stt_model.rnnt_decoder.prediction.embed.weight";
pub(super) const PREDICTOR: &str = "stt_model.rnnt_decoder.prediction.dec_rnn.lstm";

fn parameter(name: String, group: &str) -> Result<ParameterSpec, Error> {
    let mut parameter = ParameterSpec::trainable(name).map_err(Error::backend)?;
    parameter.group = Some(group.into());
    Ok(parameter)
}
fn affine(
    input: i32,
    output: i32,
    weight: String,
    bias: String,
    group: &str,
) -> Result<LinearSpec, Error> {
    Ok(LinearSpec {
        input,
        output,
        weight: parameter(weight, group)?,
        bias: Some(parameter(bias, group)?),
        format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
    })
}
#[derive(Debug, Clone)]
pub(crate) enum UnitSpec {
    Embedding(EmbeddingSpec),
    Predictor(LstmLayerSpec),
    Projection(LinearSpec),
}
impl UnitSpec {
    pub fn at(d: RnntDimensions, phase: RnntPhase, index: usize) -> Result<Self, RnntError> {
        d.validate()?;
        if index >= phase.units(d) {
            return Err(RnntError::InvalidConfiguration(
                "RNNT parameter unit exceeds its phase",
            ));
        }
        let group = phase.parameter_group(index);
        let spec = match phase {
            RnntPhase::Embedding => Self::Embedding(EmbeddingSpec {
                vocabulary: d.vocab_size + 1,
                dimensions: d.predictor_width,
                weight: parameter(EMBEDDING.into(), &group)?,
                format: eredu_nn::LinearFormatSpec::unscaled(
                    eredu_checkpoint::LinearFormat::Dense,
                )?,
            }),
            RnntPhase::Predictor => {
                let matrix = |name| {
                    affine(
                        d.predictor_width,
                        d.predictor_width * 4,
                        format!("{PREDICTOR}.weight_{name}_l{index}"),
                        format!("{PREDICTOR}.bias_{name}_l{index}"),
                        &group,
                    )
                };
                Self::Predictor(LstmLayerSpec {
                    input: matrix("ih")?,
                    recurrent: matrix("hh")?,
                })
            }
            RnntPhase::Encoder | RnntPhase::Prediction | RnntPhase::Output => {
                let (input, output) = match phase {
                    RnntPhase::Encoder => (d.encoder_width, d.joint_width),
                    RnntPhase::Prediction => (d.predictor_width, d.joint_width),
                    RnntPhase::Output => (d.joint_width, d.vocab_size + 1),
                    _ => unreachable!(),
                };
                let root = phase.path(index);
                Self::Projection(affine(
                    input,
                    output,
                    format!("{root}.weight"),
                    format!("{root}.bias"),
                    &group,
                )?)
            }
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for UnitSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Embedding(spec) => spec.logical_parameters(),
            Self::Predictor(spec) => spec.logical_parameters(),
            Self::Projection(spec) => spec.logical_parameters(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Declaration {
    phases: Vec<(RnntPhase, Vec<UnitSpec>)>,
}
impl Declaration {
    pub fn new(d: RnntDimensions, phases: &[RnntPhase]) -> Result<Self, RnntError> {
        d.validate()?;
        let declaration = Self {
            phases: phases
                .iter()
                .map(|&phase| {
                    Ok((
                        phase,
                        (0..phase.units(d))
                            .map(|i| UnitSpec::at(d, phase, i))
                            .collect::<Result<_, RnntError>>()?,
                    ))
                })
                .collect::<Result<_, RnntError>>()?,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(declaration)
    }
    fn project(&self, phases: &[RnntPhase]) -> Result<Self, RnntError> {
        let selected = Self {
            phases: phases
                .iter()
                .map(|phase| {
                    self.phases
                        .iter()
                        .find(|(p, _)| p.ordinal() == phase.ordinal())
                        .cloned()
                        .ok_or(RnntError::InvalidConfiguration(
                            "RNNT projection references an undeclared phase",
                        ))
                })
                .collect::<Result<_, _>>()?,
        };
        eredu_nn::validate_logical_parameter_topology(&selected.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(selected)
    }
    pub fn units(&self) -> impl Iterator<Item = &UnitSpec> {
        self.phases.iter().flat_map(|(_, units)| units)
    }
    pub fn phase(&self, phase: RnntPhase) -> Result<&[UnitSpec], RnntError> {
        self.phases
            .iter()
            .find(|(p, _)| p.ordinal() == phase.ordinal())
            .map(|(_, units)| units.as_slice())
            .ok_or(RnntError::InvalidConfiguration(
                "RNNT parameter phase is absent from the declaration",
            ))
    }
    pub fn unit(&self, phase: RnntPhase, index: usize) -> Result<&UnitSpec, RnntError> {
        self.phase(phase)?
            .get(index)
            .ok_or(RnntError::InvalidConfiguration(
                "RNNT parameter unit is absent from the declaration",
            ))
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = Vec::new();
        for (_, units) in &self.phases {
            parameters.extend(units[..].logical_parameters()?);
        }
        Ok(parameters)
    }
}

/// Retained neural ownership and execution projection for complete or projection-only RNNT.
/// `RECURRENT` distinguishes full recurrent ownership from the four affine/lookup phases.
#[derive(Debug, Clone)]
pub struct RnntDeclaration<const RECURRENT: bool = true> {
    dimensions: RnntDimensions,
    spec: Declaration,
    definition: eredu_runtime::ComponentDefinition,
}
impl RnntDeclaration<true> {
    /// Declares the complete transducer without native resources or invocation state.
    pub fn new(dimensions: RnntDimensions) -> Result<Self, RnntError> {
        let selection = super::rnnt_layered::RnntSelection::Complete;
        let spec = Declaration::new(dimensions, selection.phases())?;
        let definition = super::rnnt_layered::rnnt_definition_for_selection(selection, &spec)?;
        Ok(Self {
            dimensions,
            spec,
            definition,
        })
    }
    /// Projects only lookup and joint parameters; no recurrent owner remains visible.
    pub fn projections(&self) -> Result<RnntDeclaration<false>, RnntError> {
        let selection = super::rnnt_layered::RnntSelection::Projections;
        let spec = self.spec.project(selection.phases())?;
        let definition = super::rnnt_layered::rnnt_definition_for_selection(selection, &spec)?;
        Ok(RnntDeclaration {
            dimensions: self.dimensions,
            spec,
            definition,
        })
    }
    /// Projects recurrent owners while retaining global layer and carry identities.
    pub fn predictor(&self) -> Result<RnntPredictorDeclaration, RnntError> {
        let spec = self.spec.project(&[RnntPhase::Predictor])?;
        let definition = super::rnnt_predictor::declared_definition(self.dimensions, &spec)?;
        Ok(RnntPredictorDeclaration {
            dimensions: self.dimensions,
            spec,
            definition,
        })
    }
}
impl<const RECURRENT: bool> RnntDeclaration<RECURRENT> {
    /// Global neural geometry, independent of selected vocabulary placement.
    pub fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    /// Parameter, invocation and independently leased boundary projection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }
    /// Resolves coherent vocabulary rows and every replicated member.
    pub fn local_dimensions(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<super::rnnt_layered::RnntLocalDimensions, RnntError> {
        super::rnnt_layered::rnnt_local_dimensions_for_description(
            &self.spec,
            layout,
            self.definition.parameters(),
        )
    }
    pub(crate) fn spec(&self) -> &Declaration {
        &self.spec
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        RnntDimensions,
        Declaration,
        eredu_runtime::ComponentDefinition,
    ) {
        (self.dimensions, self.spec, self.definition)
    }
}
impl<const RECURRENT: bool> eredu_runtime::ComponentPreparationDeclaration
    for RnntDeclaration<RECURRENT>
{
    fn parameters(&self) -> &eredu_runtime::ArchitectureParameterDescription {
        self.definition.parameters()
    }
}

/// Recurrent-only RNNT parameters with their explicit mutable carry and stage boundaries.
#[derive(Debug, Clone)]
pub struct RnntPredictorDeclaration {
    dimensions: RnntDimensions,
    spec: Declaration,
    definition: eredu_runtime::ComponentDefinition,
}
impl RnntPredictorDeclaration {
    /// Complete transducer geometry retained with the selected recurrent scope.
    pub fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    /// Global recurrent ownership, state layout and partition semantics.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }
    pub(crate) fn spec(&self) -> &Declaration {
        &self.spec
    }
}
impl eredu_runtime::ComponentPreparationDeclaration for RnntPredictorDeclaration {
    fn parameters(&self) -> &eredu_runtime::ArchitectureParameterDescription {
        self.definition.parameters()
    }
}

/// The immutable published scope contains no payload, native resource or mutable state.
pub(crate) fn released_declaration() -> Result<&'static RnntDeclaration, RnntError> {
    static DECLARATION: std::sync::OnceLock<Result<RnntDeclaration, String>> =
        std::sync::OnceLock::new();
    DECLARATION
        .get_or_init(|| {
            let config = super::config::released_configuration().map_err(|e| e.to_string())?;
            let dimensions =
                RnntDimensions::from_config(&config.rnnt).map_err(|e| e.to_string())?;
            RnntDeclaration::new(dimensions).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|reason| Error::backend(reason.clone()).into())
}
