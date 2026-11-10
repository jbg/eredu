//! Pixtral parameter ownership shared by source admission, construction and placement.
use super::super::{FamilyConfig, PixtralConfig, PixtralPolicy};
use crate::decoder::Config;
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration, LogicalParameterRole,
    NormalizationConstructionSpec, NormalizationScale, ParameterSpec,
};

fn parameter(root: &str) -> Result<ParameterSpec, Error> {
    let mut p = ParameterSpec::trainable(format!("{root}.weight")).map_err(Error::backend)?;
    p.group = Some(root.into());
    Ok(p)
}
fn linear(root: &str, input: i32, output: i32) -> Result<LinearSpec, Error> {
    Ok(LinearSpec {
        input,
        output,
        weight: parameter(root)?,
        bias: None,
        format: crate::linear_format::standard_linear_format(
            &format!("{root}.weight"),
            eredu_checkpoint::LinearFormat::Dense,
        )?,
    })
}
fn norm(root: &str, width: i32, epsilon: f32) -> Result<NormalizationConstructionSpec, Error> {
    Ok(NormalizationConstructionSpec {
        reduction: eredu_nn::NormalizationReduction::Native,
        groups: None,
        dimensions: width,
        epsilon,
        scale: NormalizationScale::Learned(parameter(root)?),
    })
}
fn validate(spec: &impl DeclareLogicalParameters) -> Result<(), Error> {
    eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
        .map_err(Error::backend_source)?;
    Ok(())
}
fn vision(config: &FamilyConfig) -> Result<&PixtralConfig, Error> {
    config.validate_config()?;
    config
        .vision
        .as_ref()
        .ok_or_else(|| Error::backend("text artifact has no Pixtral tower"))
}
/// Canonical OIHW patch kernel and pre-attention normalization.
#[derive(Debug, Clone)]
pub struct EmbeddingSpec {
    pub(crate) weight: ParameterSpec,
    pub(crate) shape: [i32; 4],
    pub(crate) norm: NormalizationConstructionSpec,
}
impl EmbeddingSpec {
    /// Declares independently owned image ingress.
    fn new(v: &PixtralPolicy) -> Result<Self, Error> {
        let spec = Self {
            weight: parameter("model.vision_tower.patch_conv")?,
            shape: [v.hidden_size, v.num_channels, v.patch_size, v.patch_size],
            norm: norm("model.vision_tower.ln_pre", v.hidden_size, 1e-5)?,
        };
        validate(&spec)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for EmbeddingSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.shape,
            LogicalParameterRole::LinearWeight,
        )?];
        p.extend(self.norm.logical_parameters()?);
        Ok(p)
    }
}
/// Independently leased spatial merger and image-to-text projections.
#[derive(Debug, Clone)]
pub struct ProjectorSpec {
    pub(crate) norm: NormalizationConstructionSpec,
    pub(crate) merger: LinearSpec,
    pub(crate) first: LinearSpec,
    pub(crate) second: LinearSpec,
    pub(crate) merge: i32,
}
impl ProjectorSpec {
    /// Declares exact channel-major spatial unfolding and text-width affines.
    pub(in crate::nemotron_labs_diffusion) fn new(
        v: &PixtralConfig,
        merge: u32,
        hidden: i32,
        epsilon: f32,
    ) -> Result<Self, Error> {
        let root = "model.multi_modal_projector";
        let merge = i32::try_from(merge).map_err(Error::backend)?;
        let merged = v
            .hidden_size
            .checked_mul(merge)
            .and_then(|n| n.checked_mul(merge))
            .ok_or_else(|| Error::backend("Pixtral merge width overflow"))?;
        let spec = Self {
            norm: norm(&format!("{root}.norm"), v.hidden_size, epsilon)?,
            merger: linear(
                &format!("{root}.patch_merger.merging_layer"),
                merged,
                v.hidden_size,
            )?,
            first: linear(&format!("{root}.linear_1"), v.hidden_size, hidden)?,
            second: linear(&format!("{root}.linear_2"), hidden, hidden)?,
            merge,
        };
        validate(&spec)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for ProjectorSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.merger.logical_parameters()?;
        p.extend(self.norm.logical_parameters()?);
        p.extend(self.first.logical_parameters()?);
        p.extend(self.second.logical_parameters()?);
        Ok(p)
    }
}
/// One invocation's RMS scales, attention projections and SwiGLU affines.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub(crate) root: String,
    pub(crate) attention_norm: NormalizationConstructionSpec,
    pub(crate) ffn_norm: NormalizationConstructionSpec,
    pub(crate) attention: eredu_nn::SelfAttentionProjectionSpec,
    pub(crate) feed_forward: eredu_nn::SplitGatedProductSpec,
    pub(crate) width: i32,
}
impl LayerSpec {
    const PRODUCT_POLICY: eredu_nn::GatedProductPolicy =
        eredu_nn::GatedProductPolicy::ordinary_silu();

    fn new(config: &PixtralPolicy, index: usize) -> Result<Self, Error> {
        config.validate().map_err(Error::backend)?;
        if index >= config.num_hidden_layers as usize {
            return Err(Error::backend("Pixtral layer outside tower"));
        }
        let heads = config.num_attention_heads;
        let intermediate = config.intermediate_size;
        let root = format!("model.vision_tower.transformer.layers.{index}");
        let width = config.hidden_size;
        let projection =
            |field: &str, input, output| linear(&format!("{root}.{field}"), input, output);
        let geometry =
            eredu_nn::AttentionHeadGeometry::new(heads, heads, config.head_dim, config.head_dim)?;
        let spec = Self {
            attention_norm: norm(&format!("{root}.attention_norm"), width, 1e-5)?,
            ffn_norm: norm(&format!("{root}.ffn_norm"), width, 1e-5)?,
            attention: eredu_nn::SelfAttentionProjectionSpec::new(
                geometry,
                eredu_nn::AttentionQueryLayout::QueryOnly,
                projection("attention.q_proj", width, geometry.query_width())?,
                projection("attention.k_proj", width, geometry.key_width())?,
                projection("attention.v_proj", width, geometry.value_width())?,
                projection("attention.o_proj", geometry.attended_width(), width)?,
            )?,
            feed_forward: eredu_nn::SplitGatedProductSpec::new(
                projection("feed_forward.gate_proj", width, intermediate)?,
                projection("feed_forward.up_proj", width, intermediate)?,
                projection("feed_forward.down_proj", intermediate, width)?,
                Self::PRODUCT_POLICY,
            )?,
            width,
            root,
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Projects a local invocation from immutable global slots without rebuilding parameter facts.
    pub fn project_local(&self, heads: i32, intermediate: i32) -> Result<Self, Error> {
        let source = self.attention.geometry;
        if heads <= 0
            || heads > source.query_heads()
            || intermediate <= 0
            || intermediate > self.feed_forward.gate.output
        {
            return Err(Error::backend(
                "Pixtral local geometry outside retained tower",
            ));
        }
        let mut local = self.clone();
        let geometry = eredu_nn::AttentionHeadGeometry::new(
            heads,
            heads,
            source.key_dimensions(),
            source.value_dimensions(),
        )?;
        local.attention.geometry = geometry;
        local.attention.query.output = geometry.query_width();
        local.attention.key.output = geometry.key_width();
        local.attention.value.output = geometry.value_width();
        local.attention.output.input = geometry.attended_width();
        local.feed_forward.gate.output = intermediate;
        local.feed_forward.up.output = intermediate;
        local.feed_forward.down.input = intermediate;
        validate(&local)?;
        Ok(local)
    }
    /// Projects exact global head/channel partition semantics from the declared slots.
    pub fn parameter_groups(
        &self,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, eredu_runtime::ParallelPlanError> {
        use eredu_runtime::{
            logical_parameter_members, MemberSharding, ParameterGroupSpec, ParameterRole,
        };
        let mut groups = Vec::new();
        for norm in [&self.attention_norm, &self.ffn_norm] {
            let p = norm
                .logical_parameters()
                .map_err(|e| eredu_runtime::ParallelPlanError::InvalidGroup(e.to_string()))?;
            groups.push(ParameterGroupSpec::new(
                p[0].parameter.group.as_ref().expect("declared norm owner"),
                ParameterRole::Replicated,
                logical_parameter_members(norm, |_| MemberSharding::Replicated)?,
            )?);
        }
        let mut attention = Vec::new();
        for (spec, axis) in [
            (&self.attention.query, 0),
            (&self.attention.key, 0),
            (&self.attention.value, 0),
            (&self.attention.output, 1),
        ] {
            attention.extend(logical_parameter_members(spec, |_| {
                MemberSharding::Partitioned { axis }
            })?);
        }
        groups.push(ParameterGroupSpec::partitioned(
            format!("{}.attention", self.root),
            ParameterRole::AttentionHeads,
            self.attention.geometry.query_heads() as usize,
            attention,
        )?);
        groups.push(eredu_runtime::logical_parameter_group(
            format!("{}.feed_forward", self.root),
            ParameterRole::FeedForwardIntermediate,
            Some(self.feed_forward.gate.output as usize),
            &self.feed_forward,
            |parameter| {
                if parameter.parameter.id == self.feed_forward.down.weight.id {
                    MemberSharding::Partitioned { axis: 1 }
                } else if self
                    .feed_forward
                    .down
                    .bias
                    .as_ref()
                    .is_some_and(|bias| bias.id == parameter.parameter.id)
                {
                    MemberSharding::Replicated
                } else {
                    MemberSharding::Partitioned { axis: 0 }
                }
            },
        )?);
        Ok(groups)
    }
}
impl DeclareLogicalParameters for LayerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.attention.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || self.attention.geometry.query_heads() != self.attention.geometry.key_value_heads()
            || self.attention.geometry.key_dimensions()
                != self.attention.geometry.value_dimensions()
            || self.attention.query.input != self.width
            || self.attention.output.output != self.width
            || self.attention_norm.dimensions != self.width
            || self.ffn_norm.dimensions != self.width
            || self.feed_forward.gate.input != self.width
            || self.feed_forward.down.output != self.width
            || self.feed_forward.policy != Self::PRODUCT_POLICY
        {
            return Err(Error::backend(
                "incompatible Pixtral attention or feed-forward residual geometry",
            ));
        }
        let mut p = self.attention_norm.logical_parameters()?;
        p.extend(self.ffn_norm.logical_parameters()?);
        p.extend(self.attention.logical_parameters()?);
        p.extend(self.feed_forward.logical_parameters()?);
        Ok(p)
    }
}
/// Retained patch ingress and independently acquired Pixtral blocks.
#[derive(Debug, Clone)]
pub struct TowerSpec {
    pub(crate) embedding: EmbeddingSpec,
    pub(crate) layers: std::sync::Arc<[LayerSpec]>,
}
impl TowerSpec {
    pub(in crate::nemotron_labs_diffusion) fn new(policy: &PixtralPolicy) -> Result<Self, Error> {
        let tower = Self {
            embedding: EmbeddingSpec::new(policy)?,
            layers: (0..policy.num_hidden_layers as usize)
                .map(|i| LayerSpec::new(policy, i))
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        };
        validate(&tower)?;
        Ok(tower)
    }
    /// Patch kernel and ingress normalization owned independently of image blocks.
    pub fn embedding(&self) -> &EmbeddingSpec {
        &self.embedding
    }
    /// Global parameter owners in execution order, independent of invocation state.
    pub fn layers(&self) -> &[LayerSpec] {
        &self.layers
    }
    /// Resolves a typed source block before constructing a local invocation.
    pub fn layer(&self, index: usize) -> Result<&LayerSpec, Error> {
        self.layers
            .get(index)
            .ok_or_else(|| Error::backend("Pixtral layer outside retained tower"))
    }
}
impl DeclareLogicalParameters for TowerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.embedding.logical_parameters()?;
        p.extend(self.layers[..].logical_parameters()?);
        Ok(p)
    }
}

/// Complete validated image topology with independent ingress, blocks and projection ownership.
#[derive(Debug, Clone)]
pub struct VisionSpec {
    pub(crate) embedding: EmbeddingSpec,
    pub(crate) projector: ProjectorSpec,
    pub(crate) layers: std::sync::Arc<[LayerSpec]>,
}
impl VisionSpec {
    /// Declares the complete image component without backend storage or invocation state.
    pub fn new(config: &FamilyConfig) -> Result<Self, Error> {
        let v = vision(config)?;
        let spec = Self {
            embedding: v.tower().embedding.clone(),
            projector: config
                .projector()
                .ok_or_else(|| Error::backend("text artifact has no spatial projector"))?
                .as_ref()
                .clone(),
            layers: std::sync::Arc::clone(&v.tower().layers),
        };
        validate(&spec)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for VisionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.embedding.logical_parameters()?;
        p.extend(self.projector.logical_parameters()?);
        p.extend(self.layers[..].logical_parameters()?);
        Ok(p)
    }
}
