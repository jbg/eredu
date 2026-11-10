//! Parameter identities and geometry shared by vision declarations and native construction.

use super::VisionConfig;
use eredu_nn::{
    multimodal::FlattenedPatchSpec, AttentionHeadGeometry, DeclareLogicalParameters, EmbeddingSpec,
    Error, FusedProjectionLayout, FusedProjectionSegment, LayerNormalizationConstructionSpec,
    LinearSpec, ParameterSpec, UnaryActivation, UnaryFeedForwardSpec,
};

pub(super) fn rooted(root: &str, relative: &str) -> String {
    if root.is_empty() {
        relative.to_owned()
    } else {
        format!("{root}.{relative}")
    }
}

pub(super) fn norm_spec(
    prefix: &str,
    dimensions: i32,
) -> Result<eredu_nn::LayerNormalizationConstructionSpec, Error> {
    let spec = eredu_nn::LayerNormalizationConstructionSpec {
        dimensions,
        epsilon: 1e-6,
        weight: Some(ParameterSpec::trainable(format!("{prefix}.weight")).map_err(Error::backend)?),
        bias: Some(ParameterSpec::trainable(format!("{prefix}.bias")).map_err(Error::backend)?),
        arithmetic: eredu_nn::LayerNormalizationArithmetic::Native,
    };
    spec.validate()?;
    Ok(spec)
}

#[derive(Debug, Clone)]
pub(super) struct PatchSpec {
    pub geometry: FlattenedPatchSpec,
    pub weight: ParameterSpec,
    pub bias: ParameterSpec,
}
impl PatchSpec {
    pub fn weight_shape(&self) -> [i32; 5] {
        [
            self.geometry.output,
            self.geometry.channels,
            self.geometry.temporal,
            self.geometry.height,
            self.geometry.width,
        ]
    }
}
pub(super) fn patch_spec(config: &VisionConfig, root: &str) -> Result<PatchSpec, Error> {
    Ok(PatchSpec {
        geometry: FlattenedPatchSpec {
            channels: config.in_channels,
            temporal: config.temporal_patch_size,
            height: config.patch_size,
            width: config.patch_size,
            output: config.hidden_size,
        },
        weight: ParameterSpec::trainable(rooted(root, "patch_embed.proj.weight"))
            .map_err(Error::backend)?,
        bias: ParameterSpec::trainable(rooted(root, "patch_embed.proj.bias"))
            .map_err(Error::backend)?,
    })
}

pub(super) fn position_spec(config: &VisionConfig, root: &str) -> Result<EmbeddingSpec, Error> {
    let name = rooted(root, "pos_embed.weight");
    Ok(EmbeddingSpec {
        vocabulary: config.num_position_embeddings,
        dimensions: config.hidden_size,
        weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
        format: crate::linear_format::standard_linear_format(
            &name,
            eredu_checkpoint::LinearFormat::Dense,
        )?,
    })
}

fn linear_spec(
    config: &VisionConfig,
    prefix: &str,
    input: i32,
    output: i32,
) -> Result<LinearSpec, Error> {
    let weight = format!("{prefix}.weight");
    let relative = ["blocks.", "merger.", "deepstack_merger_list."]
        .into_iter()
        .find_map(|marker| weight.find(marker).map(|index| &weight[index..]))
        .unwrap_or(&weight);
    Ok(LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
        bias: Some(ParameterSpec::trainable(format!("{prefix}.bias")).map_err(Error::backend)?),
        format: crate::linear_format::standard_linear_format(
            &weight,
            config.linear_format(relative),
        )?,
    })
}

/// Checked global feature width with independently selected local head ownership.
pub(super) fn attention_geometry(
    config: &VisionConfig,
    heads: i32,
) -> Result<AttentionHeadGeometry, Error> {
    config.validate().map_err(Error::backend)?;
    let dimensions = config.hidden_size / config.num_heads;
    AttentionHeadGeometry::new(heads, heads, dimensions, dimensions)
}

/// Complete component-major Q/K/V projection and attended-head readout.
#[derive(Debug, Clone)]
pub(super) struct AttentionSpec {
    pub geometry: AttentionHeadGeometry,
    pub layout: FusedProjectionLayout,
    pub qkv: LinearSpec,
    pub output: LinearSpec,
}
impl AttentionSpec {
    pub fn segments(&self) -> Vec<std::ops::Range<usize>> {
        let mut start = 0;
        self.layout
            .segments()
            .iter()
            .map(|segment| {
                let end = start + segment.width() as usize;
                let range = start..end;
                start = end;
                range
            })
            .collect()
    }
}
impl DeclareLogicalParameters for AttentionSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let g = self.geometry;
        let expected = [
            ("query", g.query_width()),
            ("key", g.key_width()),
            ("value", g.value_width()),
        ];
        if g.query_heads() != g.key_value_heads()
            || g.key_dimensions() != g.value_dimensions()
            || self.layout.segments().len() != expected.len()
            || !self
                .layout
                .segments()
                .iter()
                .zip(expected)
                .all(|(s, (name, width))| s.name() == name && s.width() == width)
            || self.qkv.output != self.layout.output_width()
            || self.output.input != g.attended_width()
            || self.output.output != self.qkv.input
        {
            return Err(Error::backend(
                "incompatible Qwen vision attention geometry",
            ));
        }
        let mut p = self.qkv.logical_parameters()?;
        p.extend(self.output.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&p).map_err(Error::backend_source)?;
        Ok(p)
    }
}

/// One residual block's exact neural slots, independent of execution invocation and placement.
#[derive(Debug, Clone)]
pub(crate) struct BlockSpec {
    pub(super) attention: AttentionSpec,
    pub feed_forward: UnaryFeedForwardSpec,
    pub norms: [LayerNormalizationConstructionSpec; 2],
}
impl BlockSpec {
    pub fn new(
        config: &VisionConfig,
        root: &str,
        layer: usize,
        heads: i32,
        intermediate: i32,
    ) -> Result<Self, Error> {
        let geometry = attention_geometry(config, heads)?;
        if config.layer_policy(layer).is_none() {
            return Err(Error::backend(
                "Qwen vision block has no declared invocation",
            ));
        }
        let prefix = rooted(root, &format!("blocks.{layer}"));
        let layout = FusedProjectionLayout::new([
            FusedProjectionSegment::new("query", geometry.query_width())?,
            FusedProjectionSegment::new("key", geometry.key_width())?,
            FusedProjectionSegment::new("value", geometry.value_width())?,
        ])?;
        let activation = match config.hidden_act.as_str() {
            "gelu" => UnaryActivation::Gelu,
            "gelu_pytorch_tanh" => UnaryActivation::GeluApproximate,
            "silu" => UnaryActivation::Silu,
            _ => return Err(Error::backend("unsupported Qwen vision activation")),
        };
        let spec = Self {
            attention: AttentionSpec {
                geometry,
                qkv: linear_spec(
                    config,
                    &format!("{prefix}.attn.qkv"),
                    config.hidden_size,
                    layout.output_width(),
                )?,
                output: linear_spec(
                    config,
                    &format!("{prefix}.attn.proj"),
                    geometry.attended_width(),
                    config.hidden_size,
                )?,
                layout,
            },
            feed_forward: UnaryFeedForwardSpec::new(
                linear_spec(
                    config,
                    &format!("{prefix}.mlp.linear_fc1"),
                    config.hidden_size,
                    intermediate,
                )?,
                linear_spec(
                    config,
                    &format!("{prefix}.mlp.linear_fc2"),
                    intermediate,
                    config.hidden_size,
                )?,
                activation,
            )?,
            norms: [
                norm_spec(&format!("{prefix}.norm1"), config.hidden_size)?,
                norm_spec(&format!("{prefix}.norm2"), config.hidden_size)?,
            ],
        };
        spec.logical_parameters()?;
        Ok(spec)
    }
    pub fn projections(&self) -> [&LinearSpec; 4] {
        [
            &self.attention.qkv,
            &self.attention.output,
            &self.feed_forward.up,
            &self.feed_forward.down,
        ]
    }
}
impl DeclareLogicalParameters for BlockSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let hidden = self.attention.qkv.input;
        if self.norms.iter().any(|n| n.dimensions != hidden)
            || self.feed_forward.up.input != hidden
            || self.feed_forward.down.output != hidden
            || !matches!(
                self.feed_forward.activation,
                UnaryActivation::Gelu | UnaryActivation::GeluApproximate | UnaryActivation::Silu
            )
        {
            return Err(Error::backend(
                "incompatible Qwen vision block residual geometry",
            ));
        }
        let mut p = self.norms[0].logical_parameters()?;
        p.extend(self.norms[1].logical_parameters()?);
        p.extend(self.attention.logical_parameters()?);
        p.extend(self.feed_forward.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&p).map_err(Error::backend_source)?;
        Ok(p)
    }
}

pub(super) fn merger_geometry(
    config: &VisionConfig,
    postshuffle: bool,
) -> Result<(i32, i32, i32), Error> {
    let unit = config
        .spatial_merge_size
        .checked_mul(config.spatial_merge_size)
        .ok_or_else(|| Error::backend("vision merge area overflowed"))?;
    let width = config
        .hidden_size
        .checked_mul(unit)
        .ok_or_else(|| Error::backend("vision merger width overflowed"))?;
    Ok((
        unit,
        width,
        if postshuffle {
            width
        } else {
            config.hidden_size
        },
    ))
}

pub(super) fn merger_projection_specs(
    config: &VisionConfig,
    prefix: &str,
    intermediate: i32,
) -> Result<[LinearSpec; 2], Error> {
    let (_, width, _) = merger_geometry(config, false)?;
    Ok([
        linear_spec(config, &format!("{prefix}.linear_fc1"), width, intermediate)?,
        linear_spec(
            config,
            &format!("{prefix}.linear_fc2"),
            intermediate,
            config.out_hidden_size,
        )?,
    ])
}

impl eredu_nn::DeclareLogicalParameters for PatchSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        use eredu_nn::{LogicalParameterDeclaration as D, LogicalParameterRole as R};
        Ok(vec![
            D::new(&self.weight, &self.weight_shape(), R::LinearWeight)?,
            D::new(&self.bias, &[self.geometry.output], R::LinearBias)?,
        ])
    }
}

/// Typed construction declarations visited by source-format projections.
#[derive(Debug, Clone)]
pub(super) enum ParameterConstruction {
    Patch(PatchSpec),
    Position(EmbeddingSpec),
    Normalization(eredu_nn::LayerNormalizationConstructionSpec),
    Linear(LinearSpec),
    Block(BlockSpec),
}
impl ParameterConstruction {
    pub fn projections(&self) -> Vec<&LinearSpec> {
        match self {
            Self::Linear(spec) => vec![spec],
            Self::Block(spec) => spec.projections().to_vec(),
            _ => vec![],
        }
    }
    pub fn normalizations(&self) -> Vec<&LayerNormalizationConstructionSpec> {
        match self {
            Self::Normalization(spec) => vec![spec],
            Self::Block(spec) => spec.norms.iter().collect(),
            _ => vec![],
        }
    }
}
impl eredu_nn::DeclareLogicalParameters for ParameterConstruction {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Patch(spec) => spec.logical_parameters(),
            Self::Position(spec) => {
                let mut declarations = spec.logical_parameters()?;
                for declaration in &mut declarations {
                    declaration.role = eredu_nn::LogicalParameterRole::PositionEmbedding;
                }
                Ok(declarations)
            }
            Self::Normalization(spec) => spec.logical_parameters(),
            Self::Linear(spec) => spec.logical_parameters(),
            Self::Block(spec) => spec.logical_parameters(),
        }
    }
}

pub(super) fn parameter_construction(
    config: &VisionConfig,
    root: &str,
) -> Result<Vec<ParameterConstruction>, Error> {
    config.validate().map_err(Error::backend)?;
    use ParameterConstruction as P;
    let mut specs = vec![
        P::Position(position_spec(config, root)?),
        P::Patch(patch_spec(config, root)?),
    ];
    for layer in 0..config.layer_count() {
        specs.push(P::Block(BlockSpec::new(
            config,
            root,
            layer,
            config.num_heads,
            config.intermediate_size,
        )?));
    }
    for (index, postshuffle) in std::iter::once((None, false))
        .chain((0..config.deepstack_layer_count()).map(|index| (Some(index), true)))
    {
        let prefix = rooted(
            root,
            &index.map_or_else(
                || "merger".to_owned(),
                |index| format!("deepstack_merger_list.{index}"),
            ),
        );
        let (_, width, norm) = merger_geometry(config, postshuffle)?;
        specs.push(P::Normalization(norm_spec(
            &format!("{prefix}.norm"),
            norm,
        )?));
        specs.extend(
            merger_projection_specs(config, &prefix, width)?
                .into_iter()
                .map(P::Linear),
        );
    }
    let declarations = specs
        .iter()
        .map(eredu_nn::DeclareLogicalParameters::logical_parameters)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    eredu_nn::validate_logical_parameter_topology(&declarations).map_err(Error::backend_source)?;
    Ok(specs)
}
