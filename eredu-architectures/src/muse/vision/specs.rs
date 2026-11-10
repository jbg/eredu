//! Source-free vision construction and independently materialized parameter scopes.

use super::super::{VisionArithmetic, VisionPolicy};
use eredu_nn::{
    DeclareLogicalParameters, EmbeddingSpec, Error, LayerNormalizationConstructionSpec, LinearSpec,
    LogicalParameterDeclaration, ParameterSpec,
};

fn linear(
    c: &VisionPolicy,
    root: &str,
    input: i32,
    output: i32,
    biased: bool,
    block: bool,
) -> Result<LinearSpec, Error> {
    let name = format!("{root}.weight");
    let mut format =
        crate::linear_format::standard_linear_format(&name, c.linear_format_for(&name))?;
    if block {
        format = format.with_arithmetic(match c.arithmetic {
            VisionArithmetic::Native => eredu_nn::LinearArithmetic::Native,
            VisionArithmetic::Rounded => eredu_nn::LinearArithmetic::RowMajorAffine,
        })?;
    }
    Ok(LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
        bias: biased
            .then(|| ParameterSpec::trainable(format!("{root}.bias")))
            .transpose()
            .map_err(Error::backend)?,
        format,
    })
}
pub(crate) fn normalization_arithmetic(c: &VisionPolicy) -> eredu_nn::LayerNormalizationArithmetic {
    match c.arithmetic {
        VisionArithmetic::Native => eredu_nn::LayerNormalizationArithmetic::Native,
        VisionArithmetic::Rounded => eredu_nn::LayerNormalizationArithmetic::CascadedMoments,
    }
}
fn normalization_parameters(
    norm: &LayerNormalizationConstructionSpec,
    dimensions: i32,
) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
    if norm.weight.is_none() || norm.bias.is_none() || norm.dimensions != dimensions {
        return Err(Error::backend(
            "Muse vision normalization requires gain and bias at the residual width",
        ));
    }
    norm.logical_parameters()
}
fn norm(c: &VisionPolicy, root: &str) -> Result<LayerNormalizationConstructionSpec, Error> {
    Ok(LayerNormalizationConstructionSpec {
        dimensions: c.hidden_size,
        epsilon: c.layer_norm_eps,
        weight: Some(ParameterSpec::trainable(format!("{root}.weight")).map_err(Error::backend)?),
        bias: Some(ParameterSpec::trainable(format!("{root}.bias")).map_err(Error::backend)?),
        arithmetic: normalization_arithmetic(c),
    })
}
fn product(values: &[i32]) -> Result<i32, Error> {
    values
        .iter()
        .try_fold(1i32, |total, &value| {
            total.checked_mul(value).filter(|&n| value > 0 && n > 0)
        })
        .ok_or_else(|| Error::backend("Muse vision geometry exceeds positive i32 dimensions"))
}
#[derive(Debug, Clone)]
pub(crate) struct StaticSpec {
    pub patch: LinearSpec,
    pub position: EmbeddingSpec,
    pub norms: [LayerNormalizationConstructionSpec; 2],
    pub adapter: eredu_nn::UnaryFeedForwardSpec,
    pub adapter_output_activation: eredu_nn::UnaryActivation,
    pub projection: LinearSpec,
    // Source spatial axes exclude temporal folding and share the declared
    // patch projection's output channels.
    spatial_kernel_axes: [i32; 3],
}
/// Image-only source layout referencing the constructed patch weight.
pub(crate) struct SpatialPatchSource<'a> {
    pub weight: &'a ParameterSpec,
    pub kernel_shape: [usize; 4],
    pub matrix_shape: [usize; 2],
}
impl StaticSpec {
    pub fn spatial_source(&self) -> Result<SpatialPatchSource<'_>, Error> {
        let width = product(&self.spatial_kernel_axes)? as usize;
        let output = usize::try_from(self.patch.output)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::backend("Muse spatial patch output must be positive"))?;
        // The checked product establishes that all source axes are positive.
        let axes = self.spatial_kernel_axes.map(|axis| axis as usize);
        Ok(SpatialPatchSource {
            weight: &self.patch.weight,
            kernel_shape: [output, axes[0], axes[1], axes[2]],
            matrix_shape: [output, width],
        })
    }

    fn new(c: &VisionPolicy) -> Result<Self, Error> {
        let spatial_kernel_axes = [3, c.patch_size, c.patch_size];
        let input = product(&[c.temporal_patch_size, product(&spatial_kernel_axes)?])?;
        let shuffled = product(&[c.hidden_size, c.merge_size, c.merge_size])?;
        let position = "model.vision_tower.patch_embedder.position_embedding_table.weight";
        let spec = Self {
            patch: linear(
                c,
                "model.vision_tower.patch_embedder.patch_embedding",
                input,
                c.hidden_size,
                false,
                false,
            )?,
            position: EmbeddingSpec {
                vocabulary: product(&[c.position_height, c.position_width])?,
                dimensions: c.hidden_size,
                weight: ParameterSpec::trainable(position).map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(
                    position,
                    c.linear_format_for(position),
                )?,
            },
            norms: [
                norm(c, "model.vision_tower.ln_pre")?,
                norm(c, "model.vision_tower.ln_post")?,
            ],
            adapter: eredu_nn::UnaryFeedForwardSpec {
                up: linear(
                    c,
                    "model.vision_adapter.fc1",
                    shuffled,
                    c.projector_hidden_size,
                    false,
                    false,
                )?,
                down: linear(
                    c,
                    "model.vision_adapter.fc2",
                    c.projector_hidden_size,
                    c.projector_hidden_size,
                    false,
                    false,
                )?,
                activation: eredu_nn::UnaryActivation::Gelu,
            },
            adapter_output_activation: eredu_nn::UnaryActivation::Gelu,
            projection: linear(
                c,
                "model.vision_projection",
                c.projector_hidden_size,
                c.language_hidden_size,
                false,
                false,
            )?,
            spatial_kernel_axes,
        };
        spec.logical_parameters()?;
        Ok(spec)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct BlockSpec {
    pub root: String,
    pub attention: eredu_nn::SelfAttentionProjectionSpec,
    pub mlp: eredu_nn::UnaryFeedForwardSpec,
    pub norms: [LayerNormalizationConstructionSpec; 2],
    pub scale: f32,
    pub arithmetic: VisionArithmetic,
}
impl BlockSpec {
    fn new(c: &VisionPolicy, layer: usize) -> Result<Self, Error> {
        if layer >= c.layer_count()
            || c.num_heads <= 0
            || c.hidden_size <= 0
            || c.hidden_size % c.num_heads != 0
        {
            return Err(Error::backend(
                "invalid Muse vision invocation or attention geometry",
            ));
        }
        let root = format!("model.vision_tower.layers.{layer}");
        let head_dimensions = c.hidden_size / c.num_heads;
        if head_dimensions % 4 != 0 {
            return Err(Error::backend(
                "Muse axial rotary requires a head width divisible by four",
            ));
        }
        let heads = eredu_nn::AttentionHeadGeometry::new(
            c.num_heads,
            c.num_heads,
            head_dimensions,
            head_dimensions,
        )?;
        let projection = |field: &str, input, output| {
            linear(
                c,
                &format!("{root}.attn.{field}"),
                input,
                output,
                true,
                true,
            )
        };
        let spec = Self {
            root: root.clone(),
            attention: eredu_nn::SelfAttentionProjectionSpec::new(
                heads,
                eredu_nn::AttentionQueryLayout::QueryOnly,
                projection("q_proj", c.hidden_size, heads.query_width())?,
                projection("k_proj", c.hidden_size, heads.key_width())?,
                projection("v_proj", c.hidden_size, heads.value_width())?,
                projection("proj", heads.attended_width(), c.hidden_size)?,
            )?,
            mlp: eredu_nn::UnaryFeedForwardSpec {
                up: linear(
                    c,
                    &format!("{root}.mlp.fc1"),
                    c.hidden_size,
                    c.intermediate_size,
                    true,
                    true,
                )?,
                down: linear(
                    c,
                    &format!("{root}.mlp.fc2"),
                    c.intermediate_size,
                    c.hidden_size,
                    true,
                    true,
                )?,
                activation: eredu_nn::UnaryActivation::Gelu,
            },
            norms: [
                norm(c, &format!("{root}.norm1"))?,
                norm(c, &format!("{root}.norm2"))?,
            ],
            scale: match c.arithmetic {
                VisionArithmetic::Native => (head_dimensions as f32).sqrt().recip(),
                VisionArithmetic::Rounded => (head_dimensions as f64).powf(-0.5) as f32,
            },
            arithmetic: c.arithmetic,
        };
        spec.logical_parameters()?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for BlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let heads = self.attention.geometry;
        let hidden = self.attention.query.input;
        if self.attention.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || hidden != heads.query_width()
            || heads.query_heads() != heads.key_value_heads()
            || heads.key_dimensions() != heads.value_dimensions()
            || heads.key_dimensions() % 4 != 0
            || self.attention.output.output != hidden
            || self.mlp.up.input != hidden
            || self.mlp.down.output != hidden
            || self.mlp.activation != eredu_nn::UnaryActivation::Gelu
        {
            return Err(Error::backend(
                "incompatible Muse vision attention, rotary, residual or GELU geometry",
            ));
        }
        let mut parameters = Vec::new();
        for norm in &self.norms {
            parameters.extend(normalization_parameters(norm, hidden)?);
        }
        parameters.extend(self.attention.logical_parameters()?);
        parameters.extend(self.mlp.logical_parameters()?);
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for StaticSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.position.dimensions != self.patch.output
            || self.adapter.down.output != self.projection.input
            || self.adapter.activation != eredu_nn::UnaryActivation::Gelu
            || self.adapter_output_activation != eredu_nn::UnaryActivation::Gelu
        {
            return Err(Error::backend(
                "incompatible Muse vision adapter geometry or GELU policy",
            ));
        }
        let mut parameters = self.patch.logical_parameters()?;
        parameters.extend(self.position.logical_parameters()?);
        for norm in &self.norms {
            parameters.extend(normalization_parameters(norm, self.patch.output)?);
        }
        parameters.extend(self.adapter.logical_parameters()?);
        parameters.extend(self.projection.logical_parameters()?);
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Declarations {
    pub statics: StaticSpec,
    pub blocks: Vec<BlockSpec>,
}
impl Declarations {
    pub(in crate::muse) fn new(c: &VisionPolicy) -> Result<Self, Error> {
        if c.layer_count() == 0 || !c.rope_theta.is_finite() || c.rope_theta <= 0.0 {
            return Err(Error::backend(
                "Muse vision requires a nonempty schedule and positive rotary base",
            ));
        }
        let declarations = Self {
            statics: StaticSpec::new(c)?,
            blocks: (0..c.layer_count())
                .map(|layer| BlockSpec::new(c, layer))
                .collect::<Result<_, _>>()?,
        };
        eredu_nn::validate_logical_parameter_topology(&declarations.logical_parameters()?)
            .map_err(Error::backend_source)?;
        declarations.validate_formats(c)?;
        Ok(declarations)
    }
    fn validate_formats(&self, c: &VisionPolicy) -> Result<(), Error> {
        crate::format_references::validate_matrices(
            self.logical_parameters()?,
            c.quantized_weight_configs
                .iter()
                .map(|(name, format)| (name.as_str(), Some((*format).into()))),
        )
    }
    pub(in crate::muse) fn project_formats(&mut self, c: &VisionPolicy) -> Result<(), Error> {
        let mut projections = vec![
            &mut self.statics.patch,
            &mut self.statics.adapter.up,
            &mut self.statics.adapter.down,
            &mut self.statics.projection,
        ];
        for block in &mut self.blocks {
            projections.extend([
                &mut block.attention.query,
                &mut block.attention.key,
                &mut block.attention.value,
                &mut block.attention.output,
                &mut block.mlp.up,
                &mut block.mlp.down,
            ]);
        }
        for spec in projections {
            let mut format = crate::linear_format::standard_linear_format(
                spec.weight.id.as_str(),
                c.linear_format_for(spec.weight.id.as_str()),
            )?
            .with_arithmetic(spec.format.arithmetic())?;
            if spec.format.exact_zero_inputs() {
                format = format.with_exact_zero_inputs();
            }
            spec.format = format;
        }
        let position = &mut self.statics.position;
        position.format = crate::linear_format::standard_linear_format(
            position.weight.id.as_str(),
            c.linear_format_for(position.weight.id.as_str()),
        )?;
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        self.validate_formats(c)
    }
}
impl DeclareLogicalParameters for Declarations {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.statics.logical_parameters()?;
        for block in &self.blocks {
            parameters.extend(block.logical_parameters()?);
        }
        Ok(parameters)
    }
}
