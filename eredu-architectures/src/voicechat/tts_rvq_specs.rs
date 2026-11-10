//! Feedback parameter ownership, separate from refinement scheduling and mutable codes.
use super::{
    tts_mog::{linear_spec, parameter, TtsError},
    tts_rvq::RvqDimensions,
    tts_rvq_layered::RvqLocalDimensions,
};
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration, LogicalParameterRole,
    ParameterSpec,
};
fn validate(spec: &impl DeclareLogicalParameters) -> Result<(), Error> {
    eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
        .map_err(Error::backend_source)
}
/// The frozen, stacked codebooks used for both feedback and nearest-code refinement.
#[derive(Debug, Clone)]
pub struct BankSpec {
    pub(crate) dimensions: RvqDimensions,
    pub(crate) local: RvqLocalDimensions,
    pub(crate) table: ParameterSpec,
}
impl BankSpec {
    /// Declares the complete bank without allocating its table or derived norms.
    pub fn new(dimensions: RvqDimensions) -> Result<Self, TtsError> {
        dimensions.validate()?;
        let mut table = parameter("rvq_embs", "voicechat.tts.rvq")?;
        table.trainable = false;
        let spec = Self {
            dimensions,
            local: RvqLocalDimensions {
                latent_start: 0,
                latent: dimensions.latent,
            },
            table,
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Selects the retained latent channels across every depth and codebook entry.
    pub fn local(mut self, local: RvqLocalDimensions) -> Result<Self, TtsError> {
        local.validate(self.dimensions)?;
        self.local = local;
        validate(&self)?;
        Ok(self)
    }
    pub(crate) fn shape(&self) -> [i32; 3] {
        [31, self.dimensions.cardinality, self.local.latent]
    }
}
impl DeclareLogicalParameters for BankSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.table,
            &self.shape(),
            LogicalParameterRole::Codebook,
        )?])
    }
}
/// The single affine owner reused by prompt feedback and refinement invocations.
#[derive(Debug, Clone)]
pub struct ProjectionSpec {
    pub(crate) dimensions: RvqDimensions,
    pub(crate) embedding: LinearSpec,
}
impl ProjectionSpec {
    /// Declares the full feedback affine over the same latent domain as the bank.
    pub fn new(dimensions: RvqDimensions) -> Result<Self, TtsError> {
        dimensions.validate()?;
        let spec = Self {
            dimensions,
            embedding: linear_spec(
                "embed_code",
                "voicechat.tts.embed_code",
                dimensions.latent,
                dimensions.hidden,
            )?,
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Selects the input columns paired with one bank channel slice.
    pub fn local(mut self, latent: i32) -> Result<Self, TtsError> {
        if latent <= 0 || latent > self.dimensions.latent {
            return Err(TtsError::Configuration("invalid RVQ projection shard"));
        }
        self.embedding.input = latent;
        validate(&self)?;
        Ok(self)
    }
}
impl DeclareLogicalParameters for ProjectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.embedding.logical_parameters()
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Declaration {
    pub bank: BankSpec,
    pub projection: ProjectionSpec,
}
impl Declaration {
    pub fn new(d: RvqDimensions) -> Result<Self, TtsError> {
        let spec = Self {
            bank: BankSpec::new(d)?,
            projection: ProjectionSpec::new(d)?,
        };
        validate(&spec)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.bank.logical_parameters()?;
        p.extend(self.projection.logical_parameters()?);
        Ok(p)
    }
}

/// Validated feedback ownership, typed neural specs and execution boundaries.
/// Clones retain source-free geometry; invocations and mutable code state remain separate.
#[derive(Debug, Clone)]
pub struct RvqDeclaration {
    spec: Declaration,
    definition: eredu_runtime::ComponentDefinition,
}
impl RvqDeclaration {
    /// Declares the bank and its single shared projection without native allocation.
    pub fn new(dimensions: RvqDimensions) -> Result<Self, TtsError> {
        let spec = Declaration::new(dimensions)?;
        let definition = super::tts_rvq_layered::declared_component_definition(&spec)?;
        Ok(Self { spec, definition })
    }
    /// Exact canonical feedback geometry.
    pub fn dimensions(&self) -> RvqDimensions {
        self.spec.bank.dimensions
    }
    /// Parameter ownership, placement semantics and independently leased boundaries.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }
    /// Resolves a coherent latent slice using this declaration's canonical parameters.
    pub fn local_dimensions(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<RvqLocalDimensions, TtsError> {
        super::tts_rvq_layered::declared_local_dimensions(&self.spec, layout)
    }
    pub(crate) fn spec(&self) -> &Declaration {
        &self.spec
    }
    pub(crate) fn into_parts(self) -> (Declaration, eredu_runtime::ComponentDefinition) {
        (self.spec, self.definition)
    }
}
impl eredu_runtime::ComponentPreparationDeclaration for RvqDeclaration {
    fn parameters(&self) -> &eredu_runtime::ArchitectureParameterDescription {
        self.definition.parameters()
    }
}

/// All admitted VoiceChat configurations validate against this published profile.
/// The retained scope contains no artifacts, mutable invocation state or backend resources.
pub(crate) fn released_declaration() -> Result<&'static RvqDeclaration, TtsError> {
    static DECLARATION: std::sync::OnceLock<Result<RvqDeclaration, String>> =
        std::sync::OnceLock::new();
    DECLARATION
        .get_or_init(|| {
            let config = super::config::released_configuration().map_err(|e| e.to_string())?;
            let dimensions = RvqDimensions::from_config(&config.tts).map_err(|e| e.to_string())?;
            RvqDeclaration::new(dimensions).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|reason| eredu_nn::Error::backend(reason.clone()).into())
}
