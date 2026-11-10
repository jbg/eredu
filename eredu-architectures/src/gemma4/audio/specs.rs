//! Audio equations and typed parameter scopes shared by construction and cold consumers.
use super::AudioPolicy;
use eredu_nn::{
    ClippedLinearSpec, DeclareLogicalParameters, Error, LayerNormalizationConstructionSpec,
    LinearSpec, LogicalParameterDeclaration, LogicalParameterRole, NormalizationConstructionSpec,
    ParameterSpec,
};

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}
fn linear(c: &AudioPolicy, name: &str, input: i32, output: i32) -> Result<LinearSpec, Error> {
    Ok(LinearSpec {
        input,
        output,
        weight: parameter(name)?,
        bias: None,
        format: crate::linear_format::standard_linear_format(
            name,
            c.linear_format_for(name, input),
        )?,
    })
}
fn norm(c: &AudioPolicy, name: &str) -> Result<NormalizationConstructionSpec, Error> {
    let spec =
        NormalizationConstructionSpec::learned(c.hidden_size, c.rms_norm_eps, parameter(name)?);
    spec.validate()?;
    Ok(spec)
}
fn multiple(width: i32, factor: i32) -> Result<i32, Error> {
    width
        .checked_mul(factor)
        .ok_or_else(|| Error::backend("audio derived width overflows"))
}

#[derive(Debug, Clone)]
pub(crate) struct SubsampleSpec {
    pub weight: ParameterSpec,
    pub shape: [i32; 4],
    pub norm: LayerNormalizationConstructionSpec,
}
impl SubsampleSpec {
    fn new(c: &AudioPolicy, root: &str, input: i32, output: i32) -> Result<Self, Error> {
        Ok(Self {
            weight: parameter(format!("{root}.conv.weight"))?,
            shape: [output, 3, 3, input],
            norm: LayerNormalizationConstructionSpec {
                dimensions: output,
                epsilon: c.rms_norm_eps,
                weight: Some(parameter(format!("{root}.norm.weight"))?),
                bias: None,
                arithmetic: eredu_nn::LayerNormalizationArithmetic::Native,
            },
        })
    }
}
impl DeclareLogicalParameters for SubsampleSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.shape,
            LogicalParameterRole::LinearWeight,
        )?];
        parameters.extend(self.norm.logical_parameters()?);
        Ok(parameters)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StaticSpec {
    pub root: String,
    pub subsampling: [SubsampleSpec; 2],
    pub input: LinearSpec,
    pub output: LinearSpec,
    // The published bias remains an optional source slot even when this invocation omits it.
    pub output_bias_enabled: bool,
}
impl StaticSpec {
    pub(super) fn new(c: &AudioPolicy) -> Result<Self, Error> {
        c.validate().map_err(Error::backend_source)?;
        let [first, second] = c.subsampling_conv_channels.as_slice() else {
            return Err(Error::backend(
                "audio subsampling requires two channel counts",
            ));
        };
        let root = "model.audio_tower.subsample_conv_projection";
        let mut output = linear(
            c,
            "model.audio_tower.output_proj.weight",
            c.hidden_size,
            c.output_proj_dims,
        )?;
        output.bias = Some(parameter("model.audio_tower.output_proj.bias")?);
        let spec = Self {
            root: "model.audio_tower.static".into(),
            subsampling: [
                SubsampleSpec::new(c, &format!("{root}.layer0"), 1, *first)?,
                SubsampleSpec::new(c, &format!("{root}.layer1"), *first, *second)?,
            ],
            input: linear(
                c,
                &format!("{root}.input_proj_linear.weight"),
                multiple(*second, 32)?,
                c.hidden_size,
            )?,
            output,
            output_bias_enabled: c.output_projection_bias,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
    pub fn output_construction(&self) -> LinearSpec {
        let mut output = self.output.clone();
        if !self.output_bias_enabled {
            output.bias = None;
        }
        output
    }
    /// Invocation-owned parameters, excluding the optional unselected output bias.
    pub fn selected(&self) -> impl DeclareLogicalParameters + '_ {
        struct Selected<'a>(&'a StaticSpec);
        impl DeclareLogicalParameters for Selected<'_> {
            fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
                let bias = self.0.output.bias.as_ref().map(|p| &p.id);
                Ok(self
                    .0
                    .logical_parameters()?
                    .into_iter()
                    .filter(|p| self.0.output_bias_enabled || Some(&p.parameter.id) != bias)
                    .collect())
            }
        }
        Selected(self)
    }
}
impl DeclareLogicalParameters for StaticSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.subsampling[..].logical_parameters()?;
        parameters.extend(self.input.logical_parameters()?);
        parameters.extend(self.output.logical_parameters()?);
        Ok(parameters)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FeedForwardSpec {
    pub pre: NormalizationConstructionSpec,
    pub network: eredu_nn::UnaryFeedForwardSpec,
    pub first_bounds: Option<eredu_nn::LinearClippingSpec>,
    pub second_bounds: Option<eredu_nn::LinearClippingSpec>,
    pub post: NormalizationConstructionSpec,
    pub residual_weight: f32,
}
impl FeedForwardSpec {
    const ACTIVATION: eredu_nn::UnaryActivation = eredu_nn::UnaryActivation::Silu;
    fn new(c: &AudioPolicy, root: &str) -> Result<Self, Error> {
        let width = multiple(c.hidden_size, 4)?;
        let first = clipped_linear_spec(c, &format!("{root}.ffw_layer_1"), c.hidden_size, width)?;
        let second = clipped_linear_spec(c, &format!("{root}.ffw_layer_2"), width, c.hidden_size)?;
        Ok(Self {
            pre: norm(c, &format!("{root}.pre_layer_norm.weight"))?,
            network: eredu_nn::UnaryFeedForwardSpec {
                up: first.linear,
                down: second.linear,
                activation: Self::ACTIVATION,
            },
            first_bounds: first.bounds,
            second_bounds: second.bounds,
            post: norm(c, &format!("{root}.post_layer_norm.weight"))?,
            residual_weight: c.residual_weight,
        })
    }
    fn validate_geometry(&self) -> Result<(), Error> {
        if self.pre.dimensions != self.network.up.input
            || self.post.dimensions != self.network.down.output
            || self.network.up.input != self.network.down.output
            || self.network.activation != Self::ACTIVATION
            || !self.residual_weight.is_finite()
            || self.first_bounds.is_none()
            || self.second_bounds.is_none()
        {
            return Err(Error::backend(
                "incompatible Gemma audio feed-forward geometry or policy",
            ));
        }
        self.network.logical_parameters().map(|_| ())
    }
}
impl DeclareLogicalParameters for FeedForwardSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = self.pre.logical_parameters()?;
        parameters.extend(self.post.logical_parameters()?);
        for (projection, bounds) in [
            (&self.network.up, &self.first_bounds),
            (&self.network.down, &self.second_bounds),
        ] {
            parameters.extend(projection.logical_parameters()?);
            if let Some(bounds) = bounds {
                parameters.extend(bounds.logical_parameters()?);
            }
        }
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct DepthwiseSpec {
    pub weight: ParameterSpec,
    pub channels: i32,
    pub kernel_size: i32,
}
impl DepthwiseSpec {
    pub fn shape(&self) -> [i32; 3] {
        [self.channels, self.kernel_size, 1]
    }
}
impl DeclareLogicalParameters for DepthwiseSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.shape(),
            LogicalParameterRole::LinearWeight,
        )?])
    }
}
#[derive(Debug, Clone)]
pub(crate) struct ConvolutionSpec {
    pub pre: NormalizationConstructionSpec,
    pub input: ClippedLinearSpec,
    pub depthwise: DepthwiseSpec,
    pub norm: NormalizationConstructionSpec,
    pub output: ClippedLinearSpec,
}
impl ConvolutionSpec {
    fn new(c: &AudioPolicy, root: &str) -> Result<Self, Error> {
        Ok(Self {
            pre: norm(c, &format!("{root}.pre_layer_norm.weight"))?,
            input: clipped_linear_spec(
                c,
                &format!("{root}.linear_start"),
                c.hidden_size,
                multiple(c.hidden_size, 2)?,
            )?,
            depthwise: DepthwiseSpec {
                weight: parameter(format!("{root}.depthwise_conv1d.weight"))?,
                channels: c.hidden_size,
                kernel_size: c.conv_kernel_size,
            },
            norm: norm(c, &format!("{root}.conv_norm.weight"))?,
            output: clipped_linear_spec(
                c,
                &format!("{root}.linear_end"),
                c.hidden_size,
                c.hidden_size,
            )?,
        })
    }
}
#[derive(Debug, Clone)]
pub(crate) struct QueryScaleSpec {
    pub weight: ParameterSpec,
    pub dimensions: i32,
}
impl DeclareLogicalParameters for QueryScaleSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.weight,
            &[self.dimensions],
            LogicalParameterRole::AttentionScale,
        )?])
    }
}
#[derive(Debug, Clone)]
pub(crate) struct AttentionSpec {
    pub projections: eredu_nn::ClippedSelfAttentionProjectionSpec,
    pub relative_key: LinearSpec,
    pub scale: QueryScaleSpec,
    pub chunk_size: i32,
    pub past: i32,
    pub logit_cap: f32,
    pub invalid_logits: f32,
}
impl AttentionSpec {
    fn new(c: &AudioPolicy, root: &str) -> Result<Self, Error> {
        let clipped = |field| {
            clipped_linear_spec(c, &format!("{root}.{field}"), c.hidden_size, c.hidden_size)
        };
        let heads = eredu_nn::AttentionHeadGeometry::new(
            c.num_attention_heads,
            c.num_attention_heads,
            c.head_dim(),
            c.head_dim(),
        )?;
        Ok(Self {
            projections: eredu_nn::ClippedSelfAttentionProjectionSpec::new(
                heads,
                eredu_nn::AttentionQueryLayout::QueryOnly,
                clipped("q_proj")?,
                clipped("k_proj")?,
                clipped("v_proj")?,
                clipped("post")?,
            )?,
            relative_key: linear(
                c,
                &format!("{root}.relative_k_proj.weight"),
                c.hidden_size,
                c.hidden_size,
            )?,
            scale: QueryScaleSpec {
                weight: parameter(format!("{root}.per_dim_scale"))?,
                dimensions: heads.key_dimensions(),
            },
            chunk_size: c.attention_chunk_size,
            past: c.attention_context_left - 1,
            logit_cap: c.attention_logit_cap,
            invalid_logits: c.attention_invalid_logits_value,
        })
    }
    fn validate_geometry(&self) -> Result<(), Error> {
        let projection = &self.projections.projection;
        let heads = projection.geometry;
        if projection.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || heads.query_heads() != heads.key_value_heads()
            || heads.key_dimensions() != heads.value_dimensions()
            || projection.query.input != heads.query_width()
            || projection.output.output != projection.query.input
            || self.relative_key.input != projection.query.input
            || self.relative_key.output != heads.key_width()
            || self.scale.dimensions != heads.key_dimensions()
            || self
                .projections
                .clipped_projections()
                .iter()
                .any(|(_, bounds)| bounds.is_none())
        {
            return Err(Error::backend(
                "Gemma audio attention residual, relative, scale or clipping geometry mismatch",
            ));
        }
        projection.validate_geometry()
    }
}

impl DeclareLogicalParameters for AttentionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = self.projections.logical_parameters()?;
        parameters.extend(self.relative_key.logical_parameters()?);
        parameters.extend(self.scale.logical_parameters()?);
        Ok(parameters)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BlockSpec {
    pub root: String,
    pub first: FeedForwardSpec,
    pub pre_attention: NormalizationConstructionSpec,
    pub attention: AttentionSpec,
    pub post_attention: NormalizationConstructionSpec,
    pub convolution: ConvolutionSpec,
    pub second: FeedForwardSpec,
    pub output: NormalizationConstructionSpec,
}
impl BlockSpec {
    pub(super) fn at(c: &AudioPolicy, layer: usize) -> Result<Self, Error> {
        c.validate().map_err(Error::backend_source)?;
        if layer >= c.num_hidden_layers as usize {
            return Err(Error::backend("audio block outside its declared stack"));
        }
        let root = format!("model.audio_tower.layers.{layer}");
        let spec = Self {
            first: FeedForwardSpec::new(c, &format!("{root}.feed_forward1"))?,
            pre_attention: norm(c, &format!("{root}.norm_pre_attn.weight"))?,
            attention: AttentionSpec::new(c, &format!("{root}.self_attn"))?,
            post_attention: norm(c, &format!("{root}.norm_post_attn.weight"))?,
            convolution: ConvolutionSpec::new(c, &format!("{root}.lconv1d"))?,
            second: FeedForwardSpec::new(c, &format!("{root}.feed_forward2"))?,
            output: norm(c, &format!("{root}.norm_out.weight"))?,
            root,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
    fn affines(&self) -> [(&LinearSpec, Option<&eredu_nn::LinearClippingSpec>); 10] {
        let [query, key, value, output] = self.attention.projections.clipped_projections();
        [
            (&self.first.network.up, self.first.first_bounds.as_ref()),
            (&self.first.network.down, self.first.second_bounds.as_ref()),
            query,
            key,
            value,
            output,
            (
                &self.convolution.input.linear,
                self.convolution.input.bounds.as_ref(),
            ),
            (
                &self.convolution.output.linear,
                self.convolution.output.bounds.as_ref(),
            ),
            (&self.second.network.up, self.second.first_bounds.as_ref()),
            (
                &self.second.network.down,
                self.second.second_bounds.as_ref(),
            ),
        ]
    }
}
impl DeclareLogicalParameters for BlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.first.validate_geometry()?;
        self.second.validate_geometry()?;
        self.attention.validate_geometry()?;
        let hidden = self.attention.projections.projection.query.input;
        if self.first.network.up.input != hidden
            || self.second.network.up.input != hidden
            || self.pre_attention.dimensions != hidden
            || self.post_attention.dimensions != hidden
            || self.convolution.pre.dimensions != hidden
            || self.convolution.norm.dimensions != hidden
            || self.convolution.depthwise.channels != hidden
            || self.output.dimensions != hidden
        {
            return Err(Error::backend("Gemma audio residual block width mismatch"));
        }
        let mut parameters = Vec::new();
        for norm in [
            &self.first.pre,
            &self.first.post,
            &self.pre_attention,
            &self.post_attention,
            &self.convolution.pre,
            &self.convolution.norm,
            &self.second.pre,
            &self.second.post,
            &self.output,
        ] {
            parameters.extend(norm.logical_parameters()?);
        }
        for (projection, bounds) in self.affines() {
            parameters.extend(projection.logical_parameters()?);
            if let Some(bounds) = bounds {
                parameters.extend(bounds.logical_parameters()?);
            }
        }
        parameters.extend(self.attention.relative_key.logical_parameters()?);
        parameters.extend(self.attention.scale.logical_parameters()?);
        parameters.extend(self.convolution.depthwise.logical_parameters()?);
        Ok(parameters)
    }
}

/// Typed audio projection with the complete learned input/output clipping equation.
fn clipped_linear_spec(
    config: &AudioPolicy,
    prefix: &str,
    input: i32,
    output: i32,
) -> Result<eredu_nn::ClippedLinearSpec, Error> {
    super::super::media::clipped_linear(
        prefix,
        input,
        output,
        config.linear_format_for(&format!("{prefix}.linear.weight"), input),
        true,
    )
}

fn project_linear(spec: &mut LinearSpec, policy: &AudioPolicy) -> Result<(), Error> {
    let mut format = crate::linear_format::standard_linear_format(
        spec.weight.id.as_str(),
        policy.linear_format_for(spec.weight.id.as_str(), spec.input),
    )?
    .with_arithmetic(spec.format.arithmetic())?;
    if spec.format.exact_zero_inputs() {
        format = format.with_exact_zero_inputs();
    }
    spec.format = format;
    Ok(())
}
impl StaticSpec {
    pub fn project_formats(&mut self, policy: &AudioPolicy) -> Result<(), Error> {
        project_linear(&mut self.input, policy)?;
        project_linear(&mut self.output, policy)
    }
}
impl BlockSpec {
    pub fn project_formats(&mut self, policy: &AudioPolicy) -> Result<(), Error> {
        let attention = &mut self.attention.projections.projection;
        for spec in [
            &mut self.first.network.up,
            &mut self.first.network.down,
            &mut attention.query,
            &mut attention.key,
            &mut attention.value,
            &mut attention.output,
            &mut self.attention.relative_key,
            &mut self.convolution.input.linear,
            &mut self.convolution.output.linear,
            &mut self.second.network.up,
            &mut self.second.network.down,
        ] {
            project_linear(spec, policy)?;
        }
        Ok(())
    }
}
