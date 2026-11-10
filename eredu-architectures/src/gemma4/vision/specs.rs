//! Vision block equations and clipped projection slots shared with cold consumers.
use super::VisionPolicy;
use eredu_nn::{
    ClippedLinearSpec, DeclareLogicalParameters, Error, LogicalParameterDeclaration,
    NormalizationConstructionSpec, NormalizationScale, ParameterSpec,
};

fn vision_linear(
    config: &VisionPolicy,
    prefix: &str,
    input: i32,
    output: i32,
) -> Result<ClippedLinearSpec, Error> {
    super::super::media::clipped_linear(
        prefix,
        input,
        output,
        config.linear_format_for(&format!("{prefix}.linear.weight"), input),
        config.use_clipped_linears,
    )
}
fn norm(
    config: &VisionPolicy,
    name: String,
    dimensions: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    let spec = NormalizationConstructionSpec {
        reduction: eredu_nn::NormalizationReduction::Native,
        groups: None,
        dimensions,
        epsilon: config.rms_norm_eps,
        scale: NormalizationScale::LearnedOffset {
            weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
            offset: 0.0,
        },
    };
    spec.validate()?;
    Ok(spec)
}
/// Attention slots, head geometry, and explicit score arithmetic for one invocation.
#[derive(Debug, Clone)]
pub struct AttentionSpec {
    /// Shared affine geometry and typed learned-clipping extensions.
    pub projections: eredu_nn::ClippedSelfAttentionProjectionSpec,
    /// Input-rounded score/probability equation.
    pub input_score_arithmetic: bool,
    /// Learned query head gain.
    pub query_norm: NormalizationConstructionSpec,
    /// Learned key head gain.
    pub key_norm: NormalizationConstructionSpec,
}
impl DeclareLogicalParameters for AttentionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let projection = &self.projections.projection;
        let heads = projection.geometry;
        if projection.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || heads.key_dimensions() != heads.value_dimensions()
            || heads.key_dimensions() % 4 != 0
            || projection.query.input != projection.output.output
            || self.query_norm.dimensions != heads.key_dimensions()
            || self.key_norm.dimensions != heads.key_dimensions()
        {
            return Err(Error::backend(
                "Gemma vision attention residual, rotary or normalization geometry mismatch",
            ));
        }
        let mut p = self.projections.logical_parameters()?;
        p.extend(self.query_norm.logical_parameters()?);
        p.extend(self.key_norm.logical_parameters()?);
        Ok(p)
    }
}
/// GELU-tanh gated MLP with learned clipping attached to its affine slots.
#[derive(Debug, Clone)]
pub struct MlpSpec {
    /// Authoritative affine geometry, encoding and fixed gated equation.
    pub product: eredu_nn::SplitGatedProductSpec,
    /// Optional learned bounds around the gate projection.
    pub gate_bounds: Option<eredu_nn::LinearClippingSpec>,
    /// Optional learned bounds around the value projection.
    pub up_bounds: Option<eredu_nn::LinearClippingSpec>,
    /// Optional learned bounds around the contraction projection.
    pub down_bounds: Option<eredu_nn::LinearClippingSpec>,
}
impl MlpSpec {
    /// Fixed pointwise equation of the image tower.
    pub const POLICY: eredu_nn::GatedProductPolicy =
        eredu_nn::GatedProductPolicy::ordinary_gelu_approximate();

    /// Declares one clipped residual branch from its typed affine slots.
    pub fn new(
        gate: ClippedLinearSpec,
        up: ClippedLinearSpec,
        down: ClippedLinearSpec,
    ) -> Result<Self, Error> {
        let spec = Self {
            product: eredu_nn::SplitGatedProductSpec::new(
                gate.linear,
                up.linear,
                down.linear,
                Self::POLICY,
            )?,
            gate_bounds: gate.bounds,
            up_bounds: up.bounds,
            down_bounds: down.bounds,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Rejects incompatible residual geometry, equation policy and parameter references.
    pub fn validate(&self) -> Result<(), Error> {
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }

    /// Binds learned clipping to the declared gate, value and contraction, in allocation order.
    pub fn into_projections(self) -> Result<[ClippedLinearSpec; 3], Error> {
        self.validate()?;
        Ok([
            ClippedLinearSpec {
                linear: self.product.gate,
                bounds: self.gate_bounds,
            },
            ClippedLinearSpec {
                linear: self.product.up,
                bounds: self.up_bounds,
            },
            ClippedLinearSpec {
                linear: self.product.down,
                bounds: self.down_bounds,
            },
        ])
    }
}
impl DeclareLogicalParameters for MlpSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.product.gate.input != self.product.down.output
            || self.product.policy != Self::POLICY
        {
            return Err(Error::backend(
                "incompatible Gemma vision MLP residual geometry or policy",
            ));
        }
        self.product.logical_parameters()?;
        let mut parameters = Vec::new();
        for (projection, bounds) in [
            (&self.product.gate, &self.gate_bounds),
            (&self.product.up, &self.up_bounds),
            (&self.product.down, &self.down_bounds),
        ] {
            parameters.extend(projection.logical_parameters()?);
            if let Some(bounds) = bounds {
                parameters.extend(bounds.logical_parameters()?);
            }
        }
        Ok(parameters)
    }
}
/// All independently owned parameters in one bounded vision block.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    /// Architecture-owned logical unit identity.
    pub root: String,
    /// Attention parameters and execution policy.
    pub attention: AttentionSpec,
    /// Dense gated branch parameters.
    pub mlp: MlpSpec,
    /// Input, post-attention, pre-feed-forward and post-feed-forward gains, in order.
    pub norms: [NormalizationConstructionSpec; 4],
}
impl BlockSpec {
    /// Builds one validated declaration under the family's tower root.
    pub(super) fn at(config: &VisionPolicy, layer: usize, tower: &str) -> Result<Self, Error> {
        config.validate().map_err(Error::backend_source)?;
        if layer >= config.num_hidden_layers as usize {
            return Err(Error::backend("vision block outside its declared stack"));
        }
        let root = format!("{tower}.encoder.layers.{layer}");
        let a = format!("{root}.self_attn");
        let m = format!("{root}.mlp");
        let hidden = config.hidden_size;
        let heads = eredu_nn::AttentionHeadGeometry::new(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.head_dim,
        )?;
        let linear = |root: &str, field, input, output| {
            vision_linear(config, &format!("{root}.{field}"), input, output)
        };
        let spec = Self {
            root: root.clone(),
            attention: AttentionSpec {
                projections: eredu_nn::ClippedSelfAttentionProjectionSpec::new(
                    heads,
                    eredu_nn::AttentionQueryLayout::QueryOnly,
                    linear(&a, "q_proj", hidden, heads.query_width())?,
                    linear(&a, "k_proj", hidden, heads.key_width())?,
                    linear(&a, "v_proj", hidden, heads.value_width())?,
                    linear(&a, "o_proj", heads.attended_width(), hidden)?,
                )?,
                input_score_arithmetic: config.input_score_arithmetic,
                query_norm: norm(config, format!("{a}.q_norm.weight"), config.head_dim)?,
                key_norm: norm(config, format!("{a}.k_norm.weight"), config.head_dim)?,
            },
            mlp: MlpSpec::new(
                linear(&m, "gate_proj", hidden, config.intermediate_size)?,
                linear(&m, "up_proj", hidden, config.intermediate_size)?,
                linear(&m, "down_proj", config.intermediate_size, hidden)?,
            )?,
            norms: [
                norm(config, format!("{root}.input_layernorm.weight"), hidden)?,
                norm(
                    config,
                    format!("{root}.post_attention_layernorm.weight"),
                    hidden,
                )?,
                norm(
                    config,
                    format!("{root}.pre_feedforward_layernorm.weight"),
                    hidden,
                )?,
                norm(
                    config,
                    format!("{root}.post_feedforward_layernorm.weight"),
                    hidden,
                )?,
            ],
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for BlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let hidden = self.attention.projections.projection.query.input;
        if self.mlp.product.gate.input != hidden
            || self.mlp.product.down.output != hidden
            || self.norms.iter().any(|norm| norm.dimensions != hidden)
        {
            return Err(Error::backend("Gemma vision residual block width mismatch"));
        }
        let mut p = self.attention.logical_parameters()?;
        p.extend(self.mlp.logical_parameters()?);
        p.extend(self.norms[..].logical_parameters()?);
        Ok(p)
    }
}

/// Learned, independent position tables for the two spatial axes.
#[derive(Debug, Clone)]
pub struct PositionSpec {
    /// Authoritative position-table identity.
    pub table: ParameterSpec,
    /// Axis count, positions per axis, and hidden width.
    pub shape: [i32; 3],
}
impl DeclareLogicalParameters for PositionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.table,
            &self.shape,
            eredu_nn::LogicalParameterRole::PositionEmbedding,
        )?])
    }
}

/// Complete learned centering and scaling of pooled image features.
#[derive(Debug, Clone)]
pub struct StandardizationSpec {
    /// Per-channel feature center, subtracted before scaling.
    pub bias: ParameterSpec,
    /// Per-channel multiplier applied after centering.
    pub scale: ParameterSpec,
    /// Exact feature width.
    pub dimensions: i32,
}
impl DeclareLogicalParameters for StandardizationSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![
            LogicalParameterDeclaration::new(
                &self.bias,
                &[self.dimensions],
                eredu_nn::LogicalParameterRole::FeatureCenter,
            )?,
            LogicalParameterDeclaration::new(
                &self.scale,
                &[self.dimensions],
                eredu_nn::LogicalParameterRole::FeatureScale,
            )?,
        ])
    }
}

/// Independently pinned patch, position and optional pooled-standardization parameters.
#[derive(Debug, Clone)]
pub struct StaticSpec {
    /// Residency unit identity.
    pub root: String,
    /// Flattened RGB patch projection and its exact encoding.
    pub patch: eredu_nn::LinearSpec,
    /// Two-axis position tables.
    pub position: PositionSpec,
    /// Complete optional pooled-output standardization equation.
    pub standardization: Option<StandardizationSpec>,
}
impl StaticSpec {
    /// Declares the family's pinned image parameter scope without tensor allocation.
    pub(super) fn at(config: &VisionPolicy, tower: &str) -> Result<Self, Error> {
        config.validate().map_err(Error::backend_source)?;
        let input = config
            .patch_size
            .checked_mul(config.patch_size)
            .and_then(|area| area.checked_mul(3))
            .ok_or_else(|| Error::backend("vision patch width overflows"))?;
        let weight = format!("{tower}.patch_embedder.input_proj.weight");
        let spec = Self {
            root: format!("{tower}.static"),
            patch: eredu_nn::LinearSpec {
                input,
                output: config.hidden_size,
                weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    &weight,
                    config.linear_format_for(&weight, input),
                )?,
            },
            position: PositionSpec {
                table: ParameterSpec::trainable(format!(
                    "{tower}.patch_embedder.position_embedding_table"
                ))
                .map_err(Error::backend)?,
                shape: [2, config.position_embedding_size, config.hidden_size],
            },
            standardization: config
                .standardize
                .then(|| {
                    Ok::<_, Error>(StandardizationSpec {
                        bias: ParameterSpec::trainable(format!("{tower}.std_bias"))
                            .map_err(Error::backend)?,
                        scale: ParameterSpec::trainable(format!("{tower}.std_scale"))
                            .map_err(Error::backend)?,
                        dimensions: config.hidden_size,
                    })
                })
                .transpose()?,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for StaticSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.patch.logical_parameters()?;
        parameters.extend(self.position.logical_parameters()?);
        if let Some(spec) = &self.standardization {
            parameters.extend(spec.logical_parameters()?);
        }
        Ok(parameters)
    }
}

fn project_linear(spec: &mut eredu_nn::LinearSpec, policy: &VisionPolicy) -> Result<(), Error> {
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
    pub(super) fn project_formats(&mut self, policy: &VisionPolicy) -> Result<(), Error> {
        project_linear(&mut self.patch, policy)
    }
}
impl BlockSpec {
    pub(super) fn project_formats(&mut self, policy: &VisionPolicy) -> Result<(), Error> {
        let attention = &mut self.attention.projections.projection;
        for spec in [
            &mut attention.query,
            &mut attention.key,
            &mut attention.value,
            &mut attention.output,
            &mut self.mlp.product.gate,
            &mut self.mlp.product.up,
            &mut self.mlp.product.down,
        ] {
            project_linear(spec, policy)?;
        }
        Ok(())
    }
}
