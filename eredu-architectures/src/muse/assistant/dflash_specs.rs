//! Shared parameter scopes for direct and block-convolved DFlash bodies.

use super::{
    block_convolution::{BlockConvolutionGeometry, BlockConvolutionSpec},
    DFlashDecoderConfig, DFlashSublayerPolicy,
};
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration,
    NormalizationConstructionSpec, ParameterSpec,
};

fn linear(
    config: &DFlashDecoderConfig,
    name: String,
    input: i32,
    output: i32,
) -> Result<LinearSpec, Error> {
    Ok(LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
        bias: None,
        format: crate::linear_format::standard_linear_format(
            &name,
            config.linear_format_for(&name),
        )?,
    })
}
fn norm(
    config: &DFlashDecoderConfig,
    name: String,
    dimensions: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    Ok(config.equation.normalization(
        dimensions,
        config.rms_norm_eps,
        ParameterSpec::trainable(name).map_err(Error::backend)?,
    ))
}

#[derive(Debug, Clone)]
pub(super) struct EdgeParameters {
    pub projection: LinearSpec,
    pub context_norm: NormalizationConstructionSpec,
    pub final_norm: NormalizationConstructionSpec,
}
impl EdgeParameters {
    pub fn context_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.projection.logical_parameters()?;
        parameters.extend(self.context_norm.logical_parameters()?);
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for EdgeParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.context_parameters()?;
        parameters.extend(self.final_norm.logical_parameters()?);
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(super) struct AttentionParameters {
    pub projection: eredu_nn::SelfAttentionProjectionSpec,
    pub norms: [NormalizationConstructionSpec; 2],
}
impl AttentionParameters {
    fn validate_geometry(&self) -> Result<(), Error> {
        self.projection.validate_geometry()?;
        let heads = self.projection.geometry;
        if self.projection.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || heads.key_dimensions() != heads.value_dimensions()
            || heads.key_dimensions() % 2 != 0
            || self.projection.query.input != self.projection.output.output
            || self
                .norms
                .iter()
                .any(|norm| norm.dimensions != heads.key_dimensions())
        {
            return Err(Error::backend(
                "DFlash attention residual, rotary or normalization geometry mismatch",
            ));
        }
        Ok(())
    }
}
impl DeclareLogicalParameters for AttentionParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = self.projection.logical_parameters()?;
        parameters.extend(self.norms[..].logical_parameters()?);
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(super) struct BlockParameters {
    pub input_norm: NormalizationConstructionSpec,
    pub post_attention_norm: NormalizationConstructionSpec,
    pub attention: AttentionParameters,
    pub mlp: eredu_nn::SplitGatedProductSpec,
    pub convolutions: [Option<BlockConvolutionSpec>; 2],
}
impl BlockParameters {
    fn new(config: &DFlashDecoderConfig, index: usize) -> Result<Self, Error> {
        let root = format!("layers.{index}");
        let heads = eredu_nn::AttentionHeadGeometry::new(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.head_dim,
        )?;
        let convolution = |field: &str| match config.sublayer {
            DFlashSublayerPolicy::Direct => Ok(None),
            DFlashSublayerPolicy::BlockConvolution { taps, group_size } => {
                let prefix = format!("{root}.{field}");
                BlockConvolutionSpec::new(
                    BlockConvolutionGeometry {
                        hidden: config.hidden_size,
                        block_size: config.block_size as i32,
                        taps,
                        group_size,
                    },
                    &prefix,
                    config.linear_format_for(&format!("{prefix}.kernel_projection.weight")),
                )
                .map(Some)
            }
        };
        let h = config.hidden_size;
        Ok(Self {
            input_norm: norm(config, format!("{root}.input_layernorm.weight"), h)?,
            post_attention_norm: norm(
                config,
                format!("{root}.post_attention_layernorm.weight"),
                h,
            )?,
            attention: AttentionParameters {
                projection: eredu_nn::SelfAttentionProjectionSpec::new(
                    heads,
                    eredu_nn::AttentionQueryLayout::QueryOnly,
                    linear(
                        config,
                        format!("{root}.self_attn.q_proj.weight"),
                        h,
                        heads.query_width(),
                    )?,
                    linear(
                        config,
                        format!("{root}.self_attn.k_proj.weight"),
                        h,
                        heads.key_width(),
                    )?,
                    linear(
                        config,
                        format!("{root}.self_attn.v_proj.weight"),
                        h,
                        heads.value_width(),
                    )?,
                    linear(
                        config,
                        format!("{root}.self_attn.o_proj.weight"),
                        heads.attended_width(),
                        h,
                    )?,
                )?,
                norms: [
                    norm(
                        config,
                        format!("{root}.self_attn.q_norm.weight"),
                        config.head_dim,
                    )?,
                    norm(
                        config,
                        format!("{root}.self_attn.k_norm.weight"),
                        config.head_dim,
                    )?,
                ],
            },
            mlp: eredu_nn::SplitGatedProductSpec::new(
                linear(
                    config,
                    format!("{root}.mlp.gate_proj.weight"),
                    h,
                    config.intermediate_size,
                )?,
                linear(
                    config,
                    format!("{root}.mlp.up_proj.weight"),
                    h,
                    config.intermediate_size,
                )?,
                linear(
                    config,
                    format!("{root}.mlp.down_proj.weight"),
                    config.intermediate_size,
                    h,
                )?,
                eredu_nn::GatedProductPolicy::ordinary_silu(),
            )?,
            convolutions: [convolution("attention_conv")?, convolution("mlp_conv")?],
        })
    }
    pub fn projection_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.attention.projection.logical_parameters()?;
        parameters.extend(self.mlp.logical_parameters()?);
        Ok(parameters)
    }
    pub fn normalization_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.input_norm.logical_parameters()?;
        parameters.extend(self.post_attention_norm.logical_parameters()?);
        parameters.extend(self.attention.norms[..].logical_parameters()?);
        Ok(parameters)
    }
    pub fn convolution_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = Vec::new();
        for spec in self.convolutions.iter().flatten() {
            parameters.extend(spec.logical_parameters()?);
        }
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for BlockParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.attention.validate_geometry()?;
        let hidden = self.attention.projection.query.input;
        if self.input_norm.dimensions != hidden
            || self.post_attention_norm.dimensions != hidden
            || self.mlp.gate.input != hidden
            || self.mlp.up.input != hidden
            || self.mlp.down.output != hidden
            || self.mlp.policy != eredu_nn::GatedProductPolicy::ordinary_silu()
            || self
                .convolutions
                .iter()
                .flatten()
                .any(|spec| spec.geometry().hidden != hidden)
            || match &self.convolutions {
                [None, None] => false,
                [Some(attention), Some(mlp)] => attention.geometry() != mlp.geometry(),
                _ => true,
            }
        {
            return Err(Error::backend(
                "DFlash residual, feed-forward or block-convolution geometry mismatch",
            ));
        }
        let mut parameters = self.normalization_parameters()?;
        parameters.extend(self.projection_parameters()?);
        parameters.extend(self.convolution_parameters()?);
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(super) struct Parameters {
    pub edge: EdgeParameters,
    pub blocks: Vec<BlockParameters>,
}
impl Parameters {
    pub fn project_formats(
        &self,
        format: &impl Fn(&str) -> eredu_checkpoint::LinearFormat,
    ) -> Result<Self, Error> {
        let mut projected = self.clone();
        super::body::project_linear_encoding(&mut projected.edge.projection, format)?;
        for block in &mut projected.blocks {
            let p = &mut block.attention.projection;
            for linear in [
                &mut p.query,
                &mut p.key,
                &mut p.value,
                &mut p.output,
                &mut block.mlp.gate,
                &mut block.mlp.up,
                &mut block.mlp.down,
            ] {
                super::body::project_linear_encoding(linear, format)?;
            }
            for convolution in block.convolutions.iter_mut().flatten() {
                *convolution = convolution.project_formats(format)?;
            }
        }
        eredu_nn::validate_logical_parameter_topology(&projected.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(projected)
    }
    pub fn new(config: &DFlashDecoderConfig) -> Result<Self, Error> {
        config.validate()?;
        let (projection, context_norm) = config.context_projection.names();
        let captures = i32::try_from(config.target_layer_ids.len()).map_err(Error::backend)?;
        let input = config
            .hidden_size
            .checked_mul(captures)
            .ok_or_else(|| Error::backend("DFlash context width exceeds i32"))?;
        let parameters = Self {
            edge: EdgeParameters {
                projection: linear(config, projection.into(), input, config.hidden_size)?,
                context_norm: norm(config, context_norm.into(), config.hidden_size)?,
                final_norm: norm(config, "norm.weight".into(), config.hidden_size)?,
            },
            blocks: (0..config.num_hidden_layers as usize)
                .map(|index| BlockParameters::new(config, index))
                .collect::<Result<_, _>>()?,
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
    pub fn append_parameters(
        &self,
        declaration: &mut crate::external_assistant::AssistantParameterDeclarations,
    ) -> Result<(), Error> {
        use eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit;
        declaration.append(
            ExecutionUnit {
                group: super::body::CONTEXT_GROUP.into(),
                unit: 0,
            },
            &self.edge.context_parameters()?,
        )?;
        declaration.append(
            ExecutionUnit {
                group: super::body::FINAL_NORM_GROUP.into(),
                unit: 0,
            },
            &self.edge.final_norm,
        )?;
        for (unit, block) in self.blocks.iter().enumerate() {
            declaration.append(
                ExecutionUnit {
                    group: super::body::DECODER_GROUP.into(),
                    unit,
                },
                block,
            )?;
        }
        Ok(())
    }
}
impl DeclareLogicalParameters for Parameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.edge.logical_parameters()?;
        parameters.extend(self.blocks[..].logical_parameters()?);
        Ok(parameters)
    }
}
