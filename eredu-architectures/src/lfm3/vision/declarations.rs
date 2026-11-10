//! Retained source slots for independently leased vision units.

use super::{ProjectorConfig, VisionConfig, VisionLayerKind};
use eredu_nn::{
    DeclareLogicalParameters, Error, LayerNormalizationConstructionSpec, LinearSpec,
    LogicalParameterDeclaration, LogicalParameterRole, NormalizationConstructionSpec,
    ParameterSpec, SelfAttentionProjectionSpec, SplitGatedProductSpec,
};

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name.into()).map_err(Error::backend)
}

fn linear(root: &str, input: i32, output: i32, bias: bool) -> Result<LinearSpec, Error> {
    let weight = format!("{root}.weight");
    Ok(LinearSpec {
        input,
        output,
        weight: parameter(&weight)?,
        bias: bias
            .then(|| parameter(format!("{root}.bias")))
            .transpose()?,
        format: crate::linear_format::standard_linear_format(
            &weight,
            eredu_checkpoint::LinearFormat::Dense,
        )?,
    })
}

fn rms(root: &str, dimensions: i32, eps: f32) -> Result<NormalizationConstructionSpec, Error> {
    Ok(NormalizationConstructionSpec::learned(
        dimensions,
        eps,
        parameter(format!("{root}.weight"))?,
    ))
}

/// Semantic identity of one independently constructible vision invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionUnit {
    /// Flattened-patch projection and initial RMS normalization.
    Patchifier,
    /// One gated-convolution or frame-local attention residual block.
    Block(usize),
    /// Affine normalization, spatial merger and language-width projection.
    Projector,
}

#[derive(Debug, Clone)]
pub(super) struct AttentionSpec {
    pub projections: SelfAttentionProjectionSpec,
    pub query_norm: NormalizationConstructionSpec,
    pub key_norm: NormalizationConstructionSpec,
}

#[derive(Debug, Clone)]
pub(super) struct ConvolutionSpec {
    pub input: LinearSpec,
    pub output: LinearSpec,
    pub weight: ParameterSpec,
    pub channels: i32,
    pub kernel: i32,
}

#[derive(Debug, Clone)]
pub(super) enum MixerSpec {
    Attention(AttentionSpec),
    Convolution(ConvolutionSpec),
}

#[derive(Debug, Clone)]
pub(super) struct BlockSpec {
    pub mixer: MixerSpec,
    pub operator_norm: NormalizationConstructionSpec,
    pub ffn_norm: NormalizationConstructionSpec,
    pub feed_forward: SplitGatedProductSpec,
}

impl DeclareLogicalParameters for BlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.operator_norm.logical_parameters()?;
        p.extend(self.ffn_norm.logical_parameters()?);
        p.extend(self.feed_forward.logical_parameters()?);
        match &self.mixer {
            MixerSpec::Attention(s) => {
                p.extend(s.projections.logical_parameters()?);
                p.extend(s.query_norm.logical_parameters()?);
                p.extend(s.key_norm.logical_parameters()?);
            }
            MixerSpec::Convolution(s) => {
                p.extend(s.input.logical_parameters()?);
                p.extend(s.output.logical_parameters()?);
                p.push(LogicalParameterDeclaration::new(
                    &s.weight,
                    &[s.channels, 1, s.kernel, s.kernel],
                    LogicalParameterRole::LinearWeight,
                )?);
            }
        }
        Ok(p)
    }
}

/// Source geometry and semantic unit boundaries for an LFM3 vision encoder.
#[derive(Debug, Clone)]
pub struct Declaration {
    config: VisionConfig,
    projector: ProjectorConfig,
    pub(super) patch: LinearSpec,
    pub(super) embedding_norm: NormalizationConstructionSpec,
    pub(super) blocks: Vec<BlockSpec>,
    pub(super) projector_norm: LayerNormalizationConstructionSpec,
    pub(super) projector_up: LinearSpec,
    pub(super) projector_down: LinearSpec,
}

impl Declaration {
    /// Declares one retained tower and projector in their exact checkpoint namespaces.
    pub fn new(
        config: VisionConfig,
        projector: ProjectorConfig,
        vision_root: &str,
        projector_root: &str,
    ) -> Result<Self, Error> {
        config.validate()?;
        let merged = projector.input_width(&config)?;
        if vision_root.is_empty() || projector_root.is_empty() {
            return Err(Error::backend(
                "LFM3 vision requires explicit parameter namespaces",
            ));
        }
        let blocks = config
            .layer_types
            .iter()
            .enumerate()
            .map(|(i, kind)| {
                let root = format!("{vision_root}.layers.{i}");
                let h = config.hidden_size;
                let head = h / config.num_attention_heads;
                let mixer = match kind {
                    VisionLayerKind::Attention => {
                        let p = format!("{root}.self_attn");
                        let g = eredu_nn::AttentionHeadGeometry::new(
                            config.num_attention_heads,
                            config.num_key_value_heads,
                            head,
                            head,
                        )?;
                        MixerSpec::Attention(AttentionSpec {
                            projections: SelfAttentionProjectionSpec::new(
                                g,
                                eredu_nn::AttentionQueryLayout::QueryOnly,
                                linear(&format!("{p}.q_proj"), h, g.query_width(), false)?,
                                linear(&format!("{p}.k_proj"), h, g.key_width(), false)?,
                                linear(&format!("{p}.v_proj"), h, g.value_width(), false)?,
                                linear(&format!("{p}.out_proj"), g.attended_width(), h, false)?,
                            )?,
                            query_norm: rms(&format!("{p}.q_layernorm"), head, config.norm_eps)?,
                            key_norm: rms(&format!("{p}.k_layernorm"), head, config.norm_eps)?,
                        })
                    }
                    VisionLayerKind::Convolution => MixerSpec::Convolution(ConvolutionSpec {
                        input: linear(
                            &format!("{root}.conv.in_proj"),
                            h,
                            h.checked_mul(3).ok_or_else(|| {
                                Error::backend("vision convolution input width overflow")
                            })?,
                            false,
                        )?,
                        output: linear(&format!("{root}.conv.out_proj"), h, h, false)?,
                        weight: parameter(format!("{root}.conv.spatial_conv.weight"))?,
                        channels: h,
                        kernel: config.conv_kernel_size,
                    }),
                };
                let p = format!("{root}.feed_forward");
                Ok(BlockSpec {
                    mixer,
                    operator_norm: rms(&format!("{root}.operator_norm"), h, config.norm_eps)?,
                    ffn_norm: rms(&format!("{root}.ffn_norm"), h, config.norm_eps)?,
                    feed_forward: SplitGatedProductSpec::new(
                        linear(&format!("{p}.w1"), h, config.intermediate_size, false)?,
                        linear(&format!("{p}.w3"), h, config.intermediate_size, false)?,
                        linear(&format!("{p}.w2"), config.intermediate_size, h, false)?,
                        eredu_nn::GatedProductPolicy::default(),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let norm = LayerNormalizationConstructionSpec {
            dimensions: config.hidden_size,
            epsilon: projector.layer_norm_eps,
            weight: Some(parameter(format!("{projector_root}.layer_norm.weight"))?),
            bias: Some(parameter(format!("{projector_root}.layer_norm.bias"))?),
            arithmetic: eredu_nn::LayerNormalizationArithmetic::Native,
        };
        let declaration = Self {
            patch: linear(
                &format!("{vision_root}.patchifier"),
                config.patch_width()?,
                config.hidden_size,
                false,
            )?,
            embedding_norm: rms(
                &format!("{vision_root}.embedding_norm"),
                config.hidden_size,
                config.norm_eps,
            )?,
            projector_up: linear(
                &format!("{projector_root}.linear_1"),
                merged,
                projector.hidden_size,
                projector.bias,
            )?,
            projector_down: linear(
                &format!("{projector_root}.linear_2"),
                projector.hidden_size,
                projector.text_hidden_size,
                projector.bias,
            )?,
            projector_norm: norm,
            blocks,
            config,
            projector,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(declaration)
    }

    /// Retained encoder invocation geometry.
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Retained spatial merger and projector policy.
    pub fn projector_config(&self) -> &ProjectorConfig {
        &self.projector
    }

    /// Semantic units in execution order, without merging all encoder layers.
    pub fn units(&self) -> impl Iterator<Item = VisionUnit> {
        std::iter::once(VisionUnit::Patchifier)
            .chain((0..self.blocks.len()).map(VisionUnit::Block))
            .chain(std::iter::once(VisionUnit::Projector))
    }

    /// Exact parameters required by one invocation, before backend materialization.
    pub fn unit_parameters(
        &self,
        unit: VisionUnit,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(match unit {
            VisionUnit::Patchifier => {
                let mut p = self.patch.logical_parameters()?;
                p.extend(self.embedding_norm.logical_parameters()?);
                p
            }
            VisionUnit::Block(i) => self
                .blocks
                .get(i)
                .ok_or_else(|| Error::backend("LFM3 vision block is outside its declaration"))?
                .logical_parameters()?,
            VisionUnit::Projector => {
                let mut p = self.projector_norm.logical_parameters()?;
                p.extend(self.projector_up.logical_parameters()?);
                p.extend(self.projector_down.logical_parameters()?);
                p
            }
        })
    }

    /// Dense official SafeTensors source geometry for this independent component.
    pub fn safetensors_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, Error> {
        use eredu_checkpoint::schema::*;
        let constraints = self
            .logical_parameters()?
            .into_iter()
            .map(|p| {
                SafetensorsTensorConstraint::required(
                    p.parameter.id.to_string(),
                    p.shape,
                    StoredDtypeConstraint::Floating,
                )
            })
            .collect();
        SafetensorsCheckpointPlan::new(
            "LFM3 vision",
            constraints,
            vec![],
            CatalogPolicy::non_strict(),
        )
        .map_err(Error::backend_source)
    }
}

impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = Vec::new();
        for unit in self.units() {
            p.extend(self.unit_parameters(unit)?);
        }
        Ok(p)
    }
}
