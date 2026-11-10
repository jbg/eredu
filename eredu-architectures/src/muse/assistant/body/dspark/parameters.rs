//! Retained context, causal block and final-normalization parameter scopes.

use crate::muse::{
    text::specs::{AttentionGeometry, DsparkBlockParameters, DsparkEdgeSpec, DsparkScalarPolicy},
    DecoderPolicy,
};
use eredu_nn::{DeclareLogicalParameters, Error, LogicalParameterDeclaration};

#[derive(Debug, Clone)]
pub(crate) struct DsparkBodyParameters {
    pub edge: DsparkEdgeSpec,
    pub blocks: Vec<DsparkBlockParameters>,
    policy: DsparkScalarPolicy,
}
impl DsparkBodyParameters {
    pub fn project_formats(
        &self,
        format: &impl Fn(&str) -> eredu_checkpoint::LinearFormat,
    ) -> Result<Self, Error> {
        let mut projected = self.clone();
        super::super::project_linear_encoding(&mut projected.edge.projection, format)?;
        for block in &mut projected.blocks {
            let p = &mut block.attention.projection;
            for linear in [
                &mut p.query,
                &mut p.key,
                &mut p.value,
                &mut p.output,
                &mut block.attention.gate,
                &mut block.dense.gate,
                &mut block.dense.up,
                &mut block.dense.down,
            ] {
                super::super::project_linear_encoding(linear, format)?;
            }
        }
        projected.validate()?;
        Ok(projected)
    }
    pub fn new(
        geometry: AttentionGeometry,
        intermediate: i32,
        layers: usize,
        captures: usize,
        format: impl Fn(&str) -> eredu_checkpoint::LinearFormat,
        policy: DsparkScalarPolicy,
    ) -> Result<Self, Error> {
        if layers == 0 {
            return Err(Error::backend("DSpark parameter definition has no blocks"));
        }
        let parameters = Self {
            edge: DsparkEdgeSpec::from_geometry(
                geometry.hidden,
                policy.epsilon,
                i32::try_from(captures).map_err(Error::backend_source)?,
                &format,
            )?,
            blocks: (0..layers)
                .map(|layer| {
                    DsparkBlockParameters::new(geometry, intermediate, layer, &format, policy)
                })
                .collect::<Result<_, _>>()?,
            policy,
        };
        parameters.validate()?;
        Ok(parameters)
    }
    pub fn from_args(args: &DecoderPolicy, captures: usize) -> Result<Self, Error> {
        Self::new(
            args.into(),
            args.intermediate_size,
            usize::try_from(args.num_hidden_layers).map_err(Error::backend_source)?,
            captures,
            |name| args.linear_format_for(name),
            DsparkScalarPolicy::from_args(args)?,
        )
    }
    pub fn validate(&self) -> Result<(), Error> {
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
    pub fn validate_binding(&self, args: &DecoderPolicy) -> Result<usize, Error> {
        self.validate()?;
        for block in &self.blocks {
            let prefix = format!("{}.", block.attention.root);
            for name in args
                .quantized_weight_configs
                .iter()
                .flat_map(|formats| formats.keys())
                .chain(args.quantized_weights.iter().flat_map(|names| names.iter()))
            {
                if name.starts_with(&prefix)
                    && !block
                        .attention
                        .projections()
                        .iter()
                        .any(|p| p.weight.id.as_str() == name)
                {
                    return Err(Error::backend(format!(
                        "Muse attention format references undeclared projection {name}"
                    )));
                }
            }
        }
        let captures = self
            .edge
            .projection
            .input
            .checked_div(args.hidden_size)
            .filter(|captures| {
                *captures > 0
                    && captures.checked_mul(args.hidden_size) == Some(self.edge.projection.input)
            })
            .ok_or_else(|| Error::backend("DSpark retained context geometry differs"))?;
        if self.edge.projection.output != args.hidden_size
            || self.blocks.len() != args.num_hidden_layers as usize
            || self.policy != DsparkScalarPolicy::from_args(args)?
            || self.blocks.iter().any(|block| {
                let heads = block.attention.projection.geometry;
                block.attention.projection.query.input != args.hidden_size
                    || heads.query_heads() != args.num_attention_heads
                    || heads.key_value_heads() != args.num_key_value_heads
                    || heads.key_dimensions() != args.head_dim
                    || block.dense.gate.output != args.intermediate_size
            })
        {
            return Err(Error::backend(
                "DSpark retained parameters differ from target-bound geometry or scalar equation",
            ));
        }
        Ok(captures as usize)
    }
    pub fn append_parameters(
        &self,
        declaration: &mut crate::external_assistant::AssistantParameterDeclarations,
    ) -> Result<(), Error> {
        use eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit;
        let mut context = self.edge.projection.logical_parameters()?;
        context.extend(self.edge.norms[0].logical_parameters()?);
        declaration.append(
            ExecutionUnit {
                group: super::super::CONTEXT_GROUP.into(),
                unit: 0,
            },
            &context,
        )?;
        declaration.append(
            ExecutionUnit {
                group: super::super::FINAL_NORM_GROUP.into(),
                unit: 0,
            },
            &self.edge.norms[1],
        )?;
        for (unit, block) in self.blocks.iter().enumerate() {
            declaration.append(
                ExecutionUnit {
                    group: super::super::DECODER_GROUP.into(),
                    unit,
                },
                block,
            )?;
        }
        Ok(())
    }
}
impl DeclareLogicalParameters for DsparkBodyParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.edge.logical_parameters()?;
        parameters.extend(self.blocks[..].logical_parameters()?);
        Ok(parameters)
    }
}
