//! One Conformer parameter owner and its coherent head/intermediate projection.
use crate::voicechat::conformer::{ConformerDimensions, ConformerError, ConformerLocalDimensions};
use eredu_nn::{
    CausalDepthwiseConvolutionSpec, DeclareLogicalParameters, Error,
    LayerNormalizationConstructionSpec, LinearSpec, LogicalParameterDeclaration,
    LogicalParameterRole, ParameterSpec,
};

#[derive(Debug, Clone)]
pub(crate) struct FeedForwardSpec {
    pub root: String,
    pub network: eredu_nn::UnaryFeedForwardSpec,
    pub residual_weight: f32,
}
impl FeedForwardSpec {
    const ACTIVATION: eredu_nn::UnaryActivation = eredu_nn::UnaryActivation::Silu;
    const RESIDUAL_WEIGHT: f32 = 0.5;
}
impl DeclareLogicalParameters for FeedForwardSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.network.up.input != self.network.down.output
            || self.network.activation != Self::ACTIVATION
            || self.residual_weight != Self::RESIDUAL_WEIGHT
        {
            return Err(Error::backend(
                "incompatible Conformer feed-forward residual geometry or policy",
            ));
        }
        self.network.logical_parameters()
    }
}
#[derive(Debug, Clone)]
pub(crate) struct HeadBiasSpec {
    pub weight: ParameterSpec,
    pub role: LogicalParameterRole,
}
#[derive(Debug, Clone)]
pub(crate) struct AttentionSpec {
    pub root: String,
    pub projection: eredu_nn::SelfAttentionProjectionSpec,
    pub position: LinearSpec,
    pub content_bias: HeadBiasSpec,
    pub position_bias: HeadBiasSpec,
}
impl AttentionSpec {
    pub fn bias_shape(&self) -> [i32; 2] {
        [
            self.projection.geometry.query_heads(),
            self.projection.geometry.key_dimensions(),
        ]
    }
    /// The family places its relative-position affine before the output slot.
    pub fn projections(&self) -> [&LinearSpec; 5] {
        [
            &self.projection.query,
            &self.projection.key,
            &self.projection.value,
            &self.position,
            &self.projection.output,
        ]
    }
}
impl DeclareLogicalParameters for AttentionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.projection.validate_geometry()?;
        let heads = self.projection.geometry;
        if self.projection.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || heads.query_heads() != heads.key_value_heads()
            || heads.key_dimensions() != heads.value_dimensions()
            || self.projection.output.output != self.projection.query.input
            || self.position.input != self.projection.query.input
            || self.position.output != heads.query_width()
            || self.content_bias.role != LogicalParameterRole::AttentionContentBias
            || self.position_bias.role != LogicalParameterRole::AttentionPositionBias
        {
            return Err(Error::backend(
                "Conformer attention, relative-position or bias geometry mismatch",
            ));
        }
        let mut parameters = Vec::new();
        for projection in self.projections() {
            parameters.extend(projection.logical_parameters()?);
        }
        for bias in [&self.content_bias, &self.position_bias] {
            parameters.push(LogicalParameterDeclaration::new(
                &bias.weight,
                &self.bias_shape(),
                bias.role,
            )?);
        }
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct PointwiseSpec {
    pub weight: ParameterSpec,
    pub input: i32,
    pub output: i32,
}
impl PointwiseSpec {
    pub fn shape(&self) -> [i32; 3] {
        [self.output, self.input, 1]
    }
}
impl DeclareLogicalParameters for PointwiseSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.shape(),
            LogicalParameterRole::LinearWeight,
        )?])
    }
}
/// Independently leased Conformer parameters, retaining global carry and residual geometry.
#[derive(Debug, Clone)]
pub struct ConformerSpec {
    pub(crate) root: String,
    pub(crate) index: usize,
    pub(crate) dimensions: ConformerDimensions,
    pub(crate) local: ConformerLocalDimensions,
    /// FF1, attention, pre-convolution, convolution, FF2, and output normalization.
    pub(crate) norms: [LayerNormalizationConstructionSpec; 6],
    pub(crate) feed_forward: [FeedForwardSpec; 2],
    pub(crate) attention: AttentionSpec,
    pub(crate) pointwise_input: PointwiseSpec,
    pub(crate) depthwise: CausalDepthwiseConvolutionSpec,
    pub(crate) pointwise_output: PointwiseSpec,
}
impl ConformerSpec {
    /// Declares the global owner for one independently addressable encoder layer.
    pub fn at(d: ConformerDimensions, index: usize) -> Result<Self, ConformerError> {
        d.validate()?;
        let root = format!("stt_model.perception.encoder.layers.{index}");
        let parameter = |local: &str| -> Result<ParameterSpec, Error> {
            let mut p =
                ParameterSpec::trainable(format!("{root}.{local}")).map_err(Error::backend)?;
            p.group = Some(format!("voicechat.perception.layer.{index}"));
            Ok(p)
        };
        let linear = |name: &str, input, output| -> Result<LinearSpec, Error> {
            Ok(LinearSpec {
                input,
                output,
                weight: parameter(&format!("{name}.weight"))?,
                bias: None,
                format: eredu_nn::LinearFormatSpec::unscaled(
                    eredu_checkpoint::LinearFormat::Dense,
                )?,
            })
        };
        let norm = |name: &str| -> Result<LayerNormalizationConstructionSpec, Error> {
            Ok(LayerNormalizationConstructionSpec {
                dimensions: d.width,
                epsilon: 1e-5,
                weight: Some(parameter(&format!("{name}.weight"))?),
                bias: Some(parameter(&format!("{name}.bias"))?),
                arithmetic: eredu_nn::LayerNormalizationArithmetic::Native,
            })
        };
        let ff = |name: &str| -> Result<FeedForwardSpec, Error> {
            Ok(FeedForwardSpec {
                root: format!("{root}.{name}"),
                network: eredu_nn::UnaryFeedForwardSpec {
                    up: linear(&format!("{name}.linear1"), d.width, d.ff_hidden)?,
                    down: linear(&format!("{name}.linear2"), d.ff_hidden, d.width)?,
                    activation: FeedForwardSpec::ACTIVATION,
                },
                residual_weight: FeedForwardSpec::RESIDUAL_WEIGHT,
            })
        };
        let bias = |name: &str, role| -> Result<HeadBiasSpec, Error> {
            Ok(HeadBiasSpec {
                weight: parameter(name)?,
                role,
            })
        };
        let heads = eredu_nn::AttentionHeadGeometry::new(
            d.heads,
            d.heads,
            d.width / d.heads,
            d.width / d.heads,
        )?;
        let spec = Self {
            root: root.clone(),
            index,
            dimensions: d,
            local: ConformerLocalDimensions {
                heads: d.heads,
                ff_hidden: d.ff_hidden,
            },
            norms: [
                norm("norm_feed_forward1")?,
                norm("norm_self_att")?,
                norm("norm_conv")?,
                norm("conv.batch_norm")?,
                norm("norm_feed_forward2")?,
                norm("norm_out")?,
            ],
            feed_forward: [ff("feed_forward1")?, ff("feed_forward2")?],
            attention: AttentionSpec {
                root: format!("{root}.self_attn"),
                projection: eredu_nn::SelfAttentionProjectionSpec::new(
                    heads,
                    eredu_nn::AttentionQueryLayout::QueryOnly,
                    linear("self_attn.linear_q", d.width, heads.query_width())?,
                    linear("self_attn.linear_k", d.width, heads.key_width())?,
                    linear("self_attn.linear_v", d.width, heads.value_width())?,
                    linear("self_attn.linear_out", heads.attended_width(), d.width)?,
                )?,
                position: linear("self_attn.linear_pos", d.width, heads.query_width())?,
                content_bias: bias(
                    "self_attn.pos_bias_u",
                    LogicalParameterRole::AttentionContentBias,
                )?,
                position_bias: bias(
                    "self_attn.pos_bias_v",
                    LogicalParameterRole::AttentionPositionBias,
                )?,
            },
            pointwise_input: PointwiseSpec {
                weight: parameter("conv.pointwise_conv1.weight")?,
                input: d.width,
                output: d.width.checked_mul(2).ok_or(ConformerError::Configuration(
                    "Conformer GLU width overflows",
                ))?,
            },
            depthwise: CausalDepthwiseConvolutionSpec {
                arithmetic: eredu_nn::ConvolutionArithmetic::Native,
                dilation: 1,
                channels: d.width,
                kernel_size: d.conv_kernel,
                weight: parameter("conv.depthwise_conv.weight")?,
                bias: None,
                activation: eredu_nn::ConvolutionActivation::Identity,
            },
            pointwise_output: PointwiseSpec {
                weight: parameter("conv.pointwise_conv2.weight")?,
                input: d.width,
                output: d.width,
            },
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Retains identities and global carry while selecting complete heads and FF channels.
    pub fn local(mut self, local: ConformerLocalDimensions) -> Result<Self, ConformerError> {
        self.validate()?;
        local.validate(self.dimensions)?;
        let global = self.attention.projection.geometry;
        let heads = eredu_nn::AttentionHeadGeometry::new(
            local.heads,
            local.heads,
            global.key_dimensions(),
            global.value_dimensions(),
        )?;
        for ff in &mut self.feed_forward {
            ff.network.up.output = local.ff_hidden;
            ff.network.down.input = local.ff_hidden;
        }
        self.attention.projection.geometry = heads;
        self.attention.projection.query.output = heads.query_width();
        self.attention.projection.key.output = heads.key_width();
        self.attention.projection.value.output = heads.value_width();
        self.attention.projection.output.input = heads.attended_width();
        self.attention.position.output = heads.query_width();
        self.local = local;
        self.validate()?;
        Ok(self)
    }
    pub(crate) fn validate(&self) -> Result<(), Error> {
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
}
impl DeclareLogicalParameters for ConformerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.local
            .validate(self.dimensions)
            .map_err(Error::backend_source)?;
        let heads = self.attention.projection.geometry;
        if self.attention.projection.query.input != self.dimensions.width
            || heads.query_heads() != self.local.heads
            || heads.key_dimensions() != self.dimensions.width / self.dimensions.heads
            || self.norms.iter().any(|norm| {
                norm.dimensions != self.dimensions.width
                    || norm.weight.is_none()
                    || norm.bias.is_none()
            })
            || self.pointwise_input.input != self.dimensions.width
            || Some(self.pointwise_input.output) != self.dimensions.width.checked_mul(2)
            || self.depthwise.channels != self.dimensions.width
            || self.depthwise.kernel_size != self.dimensions.conv_kernel
            || self.pointwise_output.input != self.dimensions.width
            || self.pointwise_output.output != self.dimensions.width
        {
            return Err(Error::backend(
                "Conformer residual, selected-head or carry geometry mismatch",
            ));
        }
        let mut p = self.norms[..].logical_parameters()?;
        for ff in &self.feed_forward {
            if ff.network.up.input != self.dimensions.width
                || ff.network.down.output != self.dimensions.width
                || ff.network.up.output != self.local.ff_hidden
            {
                return Err(Error::backend(
                    "Conformer feed-forward geometry differs from its retained width",
                ));
            }
            p.extend(ff.logical_parameters()?);
        }
        p.extend(self.attention.logical_parameters()?);
        p.extend(self.pointwise_input.logical_parameters()?);
        p.extend(self.depthwise.logical_parameters()?);
        p.extend(self.pointwise_output.logical_parameters()?);
        Ok(p)
    }
}
