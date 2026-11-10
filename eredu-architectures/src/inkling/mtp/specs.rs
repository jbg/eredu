//! Complete source-free ownership of Inkling prediction parameters.
use crate::inkling::text::{CommonParameters, DenseParameters};
use crate::inkling::{ModelPolicy, TextArgs};
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration,
    NormalizationConstructionSpec, ParameterSpec,
};

#[derive(Debug, Clone)]
pub(crate) struct FusionParameters {
    pub hidden_norm: NormalizationConstructionSpec,
    pub embedding_norm: NormalizationConstructionSpec,
    pub input_projection: LinearSpec,
}
impl FusionParameters {
    fn new_at(text: &TextArgs, root: &str) -> Result<Self, Error> {
        let norm = |field: &str| {
            Ok::<_, Error>(NormalizationConstructionSpec::learned(
                text.hidden_size,
                text.rms_norm_eps,
                ParameterSpec::trainable(format!("{root}.{field}.weight"))
                    .map_err(Error::backend)?,
            ))
        };
        let weight = format!("{root}.input_proj.weight");
        let parameters = Self {
            hidden_norm: norm("hidden_norm")?,
            embedding_norm: norm("embed_norm")?,
            input_projection: LinearSpec {
                input: text
                    .hidden_size
                    .checked_mul(2)
                    .ok_or_else(|| Error::backend("Inkling prediction fusion width overflow"))?,
                output: text.hidden_size,
                weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    &weight,
                    text.linear_format_for(&weight),
                )?,
            },
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for FusionParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.hidden_norm.logical_parameters()?;
        parameters.extend(self.embedding_norm.logical_parameters()?);
        parameters.extend(self.input_projection.logical_parameters()?);
        Ok(parameters)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DepthParameters {
    pub fusion: FusionParameters,
    pub common: CommonParameters,
    pub dense: DenseParameters,
}
impl DeclareLogicalParameters for DepthParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.fusion.logical_parameters()?;
        parameters.extend(self.common.logical_parameters()?);
        parameters.extend(self.dense.logical_parameters()?);
        Ok(parameters)
    }
}

fn chain_norm_spec(args: &ModelPolicy) -> Result<Option<NormalizationConstructionSpec>, Error> {
    if !args
        .mtp_config
        .as_ref()
        .is_some_and(|config| config.chain_hidden_post_norm)
    {
        return Ok(None);
    }
    let spec = NormalizationConstructionSpec::learned(
        args.text_config.hidden_size,
        args.text_config.rms_norm_eps,
        ParameterSpec::trainable("model.mtp.chain_norm.weight").map_err(Error::backend)?,
    );
    spec.validate()?;
    Ok(Some(spec))
}

#[derive(Debug, Clone)]
pub(crate) struct Parameters {
    pub depths: Vec<DepthParameters>,
    pub chain_norm: Option<NormalizationConstructionSpec>,
}
impl Parameters {
    pub(in crate::inkling) fn new(args: &ModelPolicy) -> Result<Self, Error> {
        let count = args.mtp_config.as_ref().map_or(Ok(0), |config| {
            usize::try_from(config.num_nextn_predict_layers).map_err(Error::backend)
        })?;
        let mut depths = Vec::with_capacity(count);
        for depth in 0..count {
            let policy = super::depth_policy(args, depth)?;
            let text = super::mtp_text_args(
                &args.text_config,
                args.mtp_config.as_ref().expect("declared depth"),
                policy,
            )?;
            let root = format!("model.mtp.layers.{depth}");
            let block = format!("{root}.transformer_block");
            depths.push(DepthParameters {
                fusion: FusionParameters::new_at(&text, &root)?,
                common: CommonParameters::new_at(&text, policy, &block)?,
                dense: DenseParameters::new_at(&text, &block)?,
            });
        }
        // Source ownership retains the configured chain norm even with zero depths.
        // Execution selects no prediction resources until at least one depth is active.
        let parameters = Self {
            depths,
            chain_norm: chain_norm_spec(args)?,
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
    pub(in crate::inkling) fn empty() -> Self {
        Self {
            depths: Vec::new(),
            chain_norm: None,
        }
    }
    pub(in crate::inkling) fn project_formats(&mut self, text: &TextArgs) -> Result<(), Error> {
        for depth in &mut self.depths {
            depth.common.project_formats(text)?;
            depth.dense.project_formats(text)?;
            let projections = [&mut depth.fusion.input_projection];
            for spec in projections {
                let mut format = crate::linear_format::standard_linear_format(
                    spec.weight.id.as_str(),
                    text.linear_format_for(spec.weight.id.as_str()),
                )?
                .with_arithmetic(spec.format.arithmetic())?;
                if spec.format.exact_zero_inputs() {
                    format = format.with_exact_zero_inputs();
                }
                spec.format = format;
            }
        }
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(())
    }
}
impl DeclareLogicalParameters for Parameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.depths.logical_parameters()?;
        if let Some(norm) = &self.chain_norm {
            parameters.extend(norm.logical_parameters()?);
        }
        Ok(parameters)
    }
}
