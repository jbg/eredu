//! Typed mixture-head parameter ownership, separate from guidance and sampling invocations.
use super::tts_mog::{linear_spec, parameter, MogDimensions, TtsError};
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration, LogicalParameterRole,
    NormalizationConstructionSpec, NormalizationScale, ParameterSpec,
};

fn norm(name: &str, group: &str, d: MogDimensions) -> Result<NormalizationConstructionSpec, Error> {
    let spec = NormalizationConstructionSpec {
        reduction: eredu_nn::NormalizationReduction::Native,
        dimensions: d.hidden,
        epsilon: d.epsilon,
        groups: Some(1),
        scale: NormalizationScale::LearnedOffset {
            weight: parameter(&format!("{name}.weight"), group)?,
            offset: 1.0,
        },
    };
    spec.validate()?;
    Ok(spec)
}
fn validate(spec: &impl DeclareLogicalParameters) -> Result<(), Error> {
    eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
        .map_err(Error::backend_source)
}
/// One independently materializable residual MLP and its normalization gains.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    pub(crate) root: String,
    pub(crate) dimensions: MogDimensions,
    pub(crate) pre: NormalizationConstructionSpec,
    pub(crate) product: eredu_nn::SplitGatedProductSpec,
    pub(crate) post: NormalizationConstructionSpec,
}
impl BlockSpec {
    const PRODUCT_POLICY: eredu_nn::GatedProductPolicy =
        eredu_nn::GatedProductPolicy::ordinary_gelu_approximate();

    /// Declares one complete global intermediate domain without native allocation.
    pub fn at(d: MogDimensions, index: usize) -> Result<Self, TtsError> {
        d.validate()?;
        if index >= d.layers {
            return Err(TtsError::Configuration("MoG layer index exceeds stack"));
        }
        let name = format!("mog_head.mlp_stack.{index}");
        let group = format!("voicechat.tts.mog.layer.{index}");
        let spec = Self {
            root: format!("tts_model.tts_model.{name}"),
            dimensions: d,
            pre: norm(&format!("{name}.pre_norm"), &group, d)?,
            product: eredu_nn::SplitGatedProductSpec::new(
                linear_spec(
                    &format!("{name}.mlp.gate_proj"),
                    &group,
                    d.hidden,
                    d.intermediate,
                )?,
                linear_spec(
                    &format!("{name}.mlp.up_proj"),
                    &group,
                    d.hidden,
                    d.intermediate,
                )?,
                linear_spec(
                    &format!("{name}.mlp.down_proj"),
                    &group,
                    d.intermediate,
                    d.hidden,
                )?,
                Self::PRODUCT_POLICY,
            )?,
            post: norm(&format!("{name}.post_norm"), &group, d)?,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Projects retained global identities into one selected intermediate width.
    pub fn local(mut self, width: i32) -> Result<Self, TtsError> {
        if width <= 0 || width > self.dimensions.intermediate {
            return Err(TtsError::Configuration("invalid MoG intermediate shard"));
        }
        self.product.gate.output = width;
        self.product.up.output = width;
        self.product.down.input = width;
        self.validate()?;
        Ok(self)
    }
    /// Validates residual geometry, the released equation and complete parameter ownership.
    pub fn validate(&self) -> Result<(), Error> {
        validate(self)
    }
    fn validate_geometry(&self) -> Result<(), Error> {
        self.dimensions.validate().map_err(Error::backend_source)?;
        if self.pre.dimensions != self.dimensions.hidden
            || self.post.dimensions != self.dimensions.hidden
            || self.product.gate.input != self.dimensions.hidden
            || self.product.down.output != self.dimensions.hidden
            || self.product.gate.output > self.dimensions.intermediate
            || self.product.policy != Self::PRODUCT_POLICY
        {
            return Err(Error::backend(
                "incompatible MoG residual gated-product equation",
            ));
        }
        Ok(())
    }
}
impl DeclareLogicalParameters for BlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut p = self.pre.logical_parameters()?;
        p.extend(self.post.logical_parameters()?);
        p.extend(self.product.logical_parameters()?);
        Ok(p)
    }
}
/// Final normalization and mixture-selection projection.
#[derive(Debug, Clone)]
pub struct PredictionSpec {
    pub(crate) dimensions: MogDimensions,
    pub(crate) norm: NormalizationConstructionSpec,
    pub(crate) logits: LinearSpec,
}
impl PredictionSpec {
    /// Declares the small prediction unit independently of the component bank.
    pub fn new(d: MogDimensions) -> Result<Self, TtsError> {
        d.validate()?;
        let group = super::tts_mog_layered::PREDICTION;
        let spec = Self {
            dimensions: d,
            norm: norm(&format!("mog_head.mlp_stack.{}", d.layers), group, d)?,
            logits: linear_spec("mog_head.proj_logits", group, d.hidden, d.mixtures)?,
        };
        validate(&spec)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for PredictionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.norm.logical_parameters()?;
        p.extend(self.logits.logical_parameters()?);
        Ok(p)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct FactorSpec {
    pub weight: ParameterSpec,
    pub mixtures: i32,
    pub latent: i32,
    pub rank: i32,
}
impl FactorSpec {
    pub fn shape(&self) -> [i32; 3] {
        [self.mixtures, self.latent, self.rank]
    }
}
impl DeclareLogicalParameters for FactorSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.shape(),
            LogicalParameterRole::LinearWeight,
        )?])
    }
}
/// Residual and log-scale projections plus the selected low-rank component factors.
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    pub(crate) dimensions: MogDimensions,
    pub(crate) residual: LinearSpec,
    pub(crate) logs: LinearSpec,
    pub(crate) means: LinearSpec,
    pub(crate) low: FactorSpec,
}
impl ComponentSpec {
    /// Declares the complete component bank without expanding mixture-by-latent activations.
    pub fn new(d: MogDimensions) -> Result<Self, TtsError> {
        d.validate()?;
        let group = super::tts_mog_layered::COMPONENTS;
        let spec = Self {
            dimensions: d,
            residual: linear_spec("mog_head.proj_else", group, d.hidden, d.latent)?,
            logs: linear_spec("mog_head.proj_logs", group, d.hidden, 1)?,
            means: linear_spec(
                "mog_head.proj_mus",
                "voicechat.tts.mog.means",
                d.hidden,
                d.mixtures
                    .checked_mul(d.rank)
                    .ok_or(TtsError::Configuration(
                        "MoG mean projection width overflows",
                    ))?,
            )?,
            low: FactorSpec {
                weight: parameter("mog_head.low_mat", "voicechat.tts.mog.low_rank")?,
                mixtures: d.mixtures,
                latent: d.latent,
                rank: d.rank,
            },
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Projects the same rank slice coherently into every mixture's mean and expansion factors.
    pub fn local(mut self, rank: i32) -> Result<Self, TtsError> {
        if rank <= 0 || rank > self.dimensions.rank {
            return Err(TtsError::Configuration("invalid MoG component rank shard"));
        }
        self.means.output = self
            .low
            .mixtures
            .checked_mul(rank)
            .ok_or(TtsError::Configuration(
                "MoG local mean projection width overflows",
            ))?;
        self.low.rank = rank;
        validate(&self)?;
        Ok(self)
    }
}
impl DeclareLogicalParameters for ComponentSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = Vec::new();
        for spec in [&self.residual, &self.logs, &self.means] {
            p.extend(spec.logical_parameters()?);
        }
        p.extend(self.low.logical_parameters()?);
        Ok(p)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Declaration {
    pub blocks: Vec<BlockSpec>,
    pub prediction: PredictionSpec,
    pub components: ComponentSpec,
}
impl Declaration {
    pub fn new(d: MogDimensions) -> Result<Self, TtsError> {
        d.validate()?;
        let declaration = Self {
            blocks: (0..d.layers)
                .map(|index| BlockSpec::at(d, index))
                .collect::<Result<_, _>>()?,
            prediction: PredictionSpec::new(d)?,
            components: ComponentSpec::new(d)?,
        };
        validate(&declaration)?;
        Ok(declaration)
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.blocks[..].logical_parameters()?;
        p.extend(self.prediction.logical_parameters()?);
        p.extend(self.components.logical_parameters()?);
        Ok(p)
    }
}

/// Retained residual-stack and mixture-bank specs with their execution projection.
/// Guidance, supplied random draws and selected-component invocations remain separate.
#[derive(Debug, Clone)]
pub struct MogDeclaration {
    spec: Declaration,
    definition: eredu_runtime::ComponentDefinition,
}
impl MogDeclaration {
    /// Declares complete mixture-head ownership without native allocation.
    pub fn new(dimensions: MogDimensions) -> Result<Self, TtsError> {
        let spec = Declaration::new(dimensions)?;
        let definition = super::tts_mog_layered::declared_definition(&spec)?;
        Ok(Self { spec, definition })
    }
    /// Connected geometry and numerical policy from the retained typed slots.
    pub fn dimensions(&self) -> MogDimensions {
        self.spec.components.dimensions
    }
    /// Parameter ownership, placement semantics and independently leased boundaries.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }
    /// Resolves coherent intermediate and mixture-rank partitions from canonical slots.
    pub fn local_dimensions(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<super::tts_mog_layered::MogLocalDimensions, TtsError> {
        super::tts_mog_layered::local_dimensions(layout, &self.spec)
    }
    pub(crate) fn spec(&self) -> &Declaration {
        &self.spec
    }
    pub(crate) fn into_parts(self) -> (Declaration, eredu_runtime::ComponentDefinition) {
        (self.spec, self.definition)
    }
}
impl eredu_runtime::ComponentPreparationDeclaration for MogDeclaration {
    fn parameters(&self) -> &eredu_runtime::ArchitectureParameterDescription {
        self.definition.parameters()
    }
}

/// Every admitted configuration validates against this source-free published profile.
/// The scope contains no artifact values, mutable invocation state or backend resources.
pub(crate) fn released_declaration() -> Result<&'static MogDeclaration, TtsError> {
    static DECLARATION: std::sync::OnceLock<Result<MogDeclaration, String>> =
        std::sync::OnceLock::new();
    DECLARATION
        .get_or_init(|| {
            let config = super::config::released_configuration().map_err(|e| e.to_string())?;
            let dimensions = MogDimensions::from_config(&config.tts).map_err(|e| e.to_string())?;
            MogDeclaration::new(dimensions).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|reason| eredu_nn::Error::backend(reason.clone()).into())
}
