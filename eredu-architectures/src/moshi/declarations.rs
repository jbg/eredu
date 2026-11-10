//! Parameter scopes projected from the temporal and depth construction specs.

use super::{
    depth::DepthConstruction, model::StaticConstruction, MoshiPolicy, MoshiTransformerConfig,
};
use crate::decoder::{topology, Config as _};
use eredu_nn::{
    DeclareLogicalParameters, Error, LogicalParameterDeclaration, NormalizationConstructionSpec,
};

#[derive(Debug, Clone)]
pub(super) struct BlockConstruction {
    pub root: String,
    pub attention: topology::AttentionParameters,
    pub gating: eredu_nn::FusedGatedProductSpec,
    pub norms: [NormalizationConstructionSpec; 2],
}
impl BlockConstruction {
    fn new(config: &MoshiTransformerConfig, layer: usize) -> Result<Self, Error> {
        config.validate_config()?;
        let root = format!("{}.layers.{layer}", config.parameter_root());
        let fields = config.block_parameter_fields();
        let spec = Self {
            root: root.clone(),
            attention: topology::AttentionParameters::new(config, layer)?,
            gating: match topology::gated_parameters(config, layer)? {
                eredu_nn::GatedProductSpec::Fused(spec) => spec,
                eredu_nn::GatedProductSpec::Split(_) => {
                    return Err(Error::backend("Moshi requires fused gated projections"));
                }
            },
            norms: [
                topology::normalization_spec(
                    config,
                    format!("{root}.{}.weight", fields.input_norm),
                )?,
                topology::normalization_spec(
                    config,
                    format!("{root}.{}.weight", fields.post_attention_norm),
                )?,
            ],
        };
        if !matches!(
            spec.attention.input,
            topology::AttentionInputParameters::Fused { .. }
        ) || spec.attention.gate.is_some()
            || spec.attention.geometry.query_heads() != spec.attention.geometry.key_value_heads()
        {
            return Err(Error::backend(
                "Moshi requires fused multi-head attention without an output gate",
            ));
        }
        spec.logical_parameters()?;
        Ok(spec)
    }
    pub(crate) fn localized(&self, heads: i32, gated: i32) -> Result<Self, Error> {
        let mut spec = self.clone();
        let geometry = eredu_nn::AttentionHeadGeometry::new(
            heads,
            heads,
            self.attention.geometry.key_dimensions(),
            self.attention.geometry.value_dimensions(),
        )?;
        let topology::AttentionInputParameters::Fused { projection, layout } =
            &mut spec.attention.input
        else {
            return Err(Error::backend(
                "Moshi local projection requires fused attention",
            ));
        };
        *layout = eredu_nn::FusedProjectionLayout::new(
            layout
                .segments()
                .iter()
                .zip([
                    geometry.query_width(),
                    geometry.key_width(),
                    geometry.value_width(),
                ])
                .map(|(segment, width)| {
                    eredu_nn::FusedProjectionSegment::new(segment.name(), width)
                })
                .collect::<Result<Vec<_>, Error>>()?,
        )?;
        projection.output = layout.output_width();
        spec.attention.geometry = geometry;
        spec.attention.output.input = geometry.attended_width();
        spec.gating.gate_up.output = gated
            .checked_mul(2)
            .ok_or_else(|| Error::backend("Moshi gated width overflow"))?;
        spec.gating.down.input = gated;
        spec.logical_parameters()?;
        Ok(spec)
    }
    pub(crate) fn decoder_parameters(&self) -> topology::BlockParameters {
        topology::BlockParameters {
            attention: self.attention.clone(),
            feed_forward: Some(eredu_nn::GatedProductSpec::Fused(self.gating.clone())),
            normalizations: topology::BlockNormalizations {
                input: self.norms[0].clone(),
                post_attention: self.norms[1].clone(),
                attention_output: None,
                feed_forward_output: None,
            },
            query_key: [None, None],
            sinks: None,
            terminal: None,
        }
    }

    pub fn fused_attention(&self) -> &eredu_nn::LinearSpec {
        let topology::AttentionInputParameters::Fused { projection, .. } = &self.attention.input
        else {
            unreachable!("validated Moshi fused QKV declaration")
        };
        projection
    }
    pub fn qkv_segments(&self) -> Vec<std::ops::Range<usize>> {
        self.attention
            .fused_segments()
            .expect("Moshi fused QKV declaration")
    }
    pub fn gated_segments(&self) -> Vec<std::ops::Range<usize>> {
        let width = self.gating.down.input as usize;
        vec![0..width, width..2 * width]
    }
}
impl DeclareLogicalParameters for BlockConstruction {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = Vec::new();
        parameters.extend(self.attention.logical_parameters()?);
        parameters.extend(self.gating.logical_parameters()?);
        for norm in &self.norms {
            parameters.extend(norm.logical_parameters()?);
        }
        Ok(parameters)
    }
}

#[derive(Debug, Clone)]
pub(super) struct DepthDeclarations {
    pub edge: DepthConstruction,
    pub blocks: Vec<BlockConstruction>,
}
impl DeclareLogicalParameters for DepthDeclarations {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.edge.logical_parameters()?;
        for block in &self.blocks {
            parameters.extend(block.logical_parameters()?);
        }
        Ok(parameters)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Declarations {
    pub(super) statics: StaticConstruction,
    pub(super) temporal: Vec<BlockConstruction>,
    pub(super) depth: Vec<DepthDeclarations>,
}
impl Declarations {
    pub(super) fn new(config: &MoshiPolicy) -> Result<Self, Error> {
        let temporal = (0..config.temporal().num_hidden_layers() as usize)
            .map(|layer| BlockConstruction::new(config.temporal(), layer))
            .collect::<Result<_, _>>()?;
        let depth = (0..config.frame_schedule().depth_audio_codebooks())
            .map(|slice| {
                let edge = DepthConstruction::new(config, slice)?;
                let blocks = (0..edge.transformer.num_hidden_layers() as usize)
                    .map(|layer| BlockConstruction::new(&edge.transformer, layer))
                    .collect::<Result<_, _>>()?;
                Ok(DepthDeclarations { edge, blocks })
            })
            .collect::<Result<_, Error>>()?;
        let spec = Self {
            statics: StaticConstruction::new(config)?,
            temporal,
            depth,
        };
        let parameters = spec.logical_parameters()?;
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
}
impl Declarations {
    pub(crate) fn temporal_parameters(
        &self,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.statics.logical_parameters()?;
        for block in &self.temporal {
            parameters.extend(block.logical_parameters()?);
        }
        Ok(parameters)
    }
    pub(crate) fn depth_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = Vec::new();
        for slice in &self.depth {
            parameters.extend(slice.logical_parameters()?);
        }
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for Declarations {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.temporal_parameters()?;
        parameters.extend(self.depth_parameters()?);
        Ok(parameters)
    }
}

impl Declarations {
    pub(crate) fn with_format(
        &self,
        format: eredu_checkpoint::LinearFormat,
    ) -> Result<Self, Error> {
        fn linear(
            spec: &mut eredu_nn::LinearSpec,
            format: eredu_checkpoint::LinearFormat,
        ) -> Result<(), Error> {
            let mut selected =
                crate::linear_format::standard_linear_format(spec.weight.id.as_str(), format)?
                    .with_arithmetic(spec.format.arithmetic())?;
            if spec.format.exact_zero_inputs() {
                selected = selected.with_exact_zero_inputs();
            }
            spec.format = selected;
            Ok(())
        }
        fn block(
            spec: &mut BlockConstruction,
            format: eredu_checkpoint::LinearFormat,
        ) -> Result<(), Error> {
            let topology::AttentionInputParameters::Fused { projection, .. } =
                &mut spec.attention.input
            else {
                return Err(Error::backend("Moshi requires a fused QKV declaration"));
            };
            linear(projection, format)?;
            linear(&mut spec.attention.output, format)?;
            linear(&mut spec.gating.gate_up, format)?;
            linear(&mut spec.gating.down, format)
        }
        let mut projected = self.clone();
        for table in &mut projected.statics.embeddings {
            table.embedding.format = crate::linear_format::standard_linear_format(
                table.embedding.weight.id.as_str(),
                format,
            )?;
        }
        linear(&mut projected.statics.text_output, format)?;
        for spec in &mut projected.temporal {
            block(spec, format)?;
        }
        for slice in &mut projected.depth {
            slice.edge.embedding.format = crate::linear_format::standard_linear_format(
                slice.edge.embedding.weight.id.as_str(),
                format,
            )?;
            linear(&mut slice.edge.input, format)?;
            linear(&mut slice.edge.output, format)?;
            for spec in &mut slice.blocks {
                block(spec, format)?;
            }
        }
        let before = self.logical_parameters()?;
        let after = projected.logical_parameters()?;
        eredu_nn::validate_logical_parameter_topology(&after).map_err(Error::backend_source)?;
        if before.len() != after.len()
            || before
                .iter()
                .zip(after.iter())
                .any(|(a, b)| a.parameter != b.parameter || a.shape != b.shape || a.role != b.role)
        {
            return Err(Error::backend(
                "Moshi storage projection changes logical declarations",
            ));
        }
        Ok(projected)
    }
}
