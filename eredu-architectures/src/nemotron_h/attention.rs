//! Nemotron-H unpositioned grouped-query attention policy.

use eredu_nn::{
    AttentionArithmetic, DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration,
    NeuralBackend, ParameterSpec, Tensor,
};

use crate::decoder::Attention;

use super::ModelArgs;

pub(crate) fn linear_spec(
    args: &ModelArgs,
    prefix: &str,
    input: i32,
    output: i32,
    bias: bool,
) -> Result<LinearSpec, Error> {
    let weight = format!("{prefix}.weight");
    let spec = LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
        bias: bias
            .then(|| ParameterSpec::trainable(format!("{prefix}.bias")))
            .transpose()
            .map_err(Error::backend)?,
        format: crate::linear_format::standard_linear_format(
            &weight,
            args.weight_quantization_for(&weight).into(),
        )?,
    };
    eredu_nn::DeclareLogicalParameters::logical_parameters(&spec)?;
    Ok(spec)
}

/// Complete no-RoPE attention declaration for global or rank-local heads.
#[derive(Debug, Clone)]
pub(crate) struct AttentionParameters {
    pub projection: eredu_nn::SelfAttentionProjectionSpec,
    pub arithmetic: AttentionArithmetic,
}

impl AttentionParameters {
    pub fn new(
        args: &ModelArgs,
        prefix: &str,
        query_heads: i32,
        key_value_heads: i32,
    ) -> Result<Self, Error> {
        let geometry = eredu_nn::AttentionHeadGeometry::new(
            query_heads,
            key_value_heads,
            args.head_dim,
            args.head_dim,
        )?;
        let spec = Self {
            arithmetic: AttentionArithmetic::Fused,
            projection: eredu_nn::SelfAttentionProjectionSpec::new(
                geometry,
                eredu_nn::AttentionQueryLayout::QueryOnly,
                linear_spec(
                    args,
                    &format!("{prefix}.q_proj"),
                    args.hidden_size,
                    geometry.query_width(),
                    args.attention_bias,
                )?,
                linear_spec(
                    args,
                    &format!("{prefix}.k_proj"),
                    args.hidden_size,
                    geometry.key_width(),
                    args.attention_bias,
                )?,
                linear_spec(
                    args,
                    &format!("{prefix}.v_proj"),
                    args.hidden_size,
                    geometry.value_width(),
                    args.attention_bias,
                )?,
                linear_spec(
                    args,
                    &format!("{prefix}.o_proj"),
                    geometry.attended_width(),
                    args.hidden_size,
                    args.attention_bias,
                )?,
            )?,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }

    pub(crate) fn state_policy(
        &self,
        attention: eredu_core::AttentionPolicy,
    ) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        self.logical_parameters()?;
        let geometry = self.projection.geometry;
        eredu_core::cache::LayerCachePolicy::key_value(
            attention,
            geometry.key_value_heads(),
            geometry.key_dimensions(),
        )
        .map_err(Error::backend_source)
    }

    pub(crate) fn construct<B: NeuralBackend>(
        self,
        attention: eredu_core::AttentionPolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Attention<B>, Error> {
        self.logical_parameters()?;
        let geometry = self.projection.geometry;
        let window = attention.sliding_window_i32().map_err(Error::backend)?;
        let mut module = Attention::from_parts(
            geometry.query_heads(),
            geometry.key_value_heads(),
            geometry.key_dimensions(),
            B::linear(self.projection.query, context)?,
            B::linear(self.projection.key, context)?,
            B::linear(self.projection.value, context)?,
            B::linear(self.projection.output, context)?,
            None,
            None,
            None,
            window,
        )?;
        module.arithmetic = self.arithmetic;
        Ok(module)
    }
}

impl DeclareLogicalParameters for AttentionParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.projection.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || self.projection.query.input != self.projection.output.output
            || self.projection.geometry.key_dimensions()
                != self.projection.geometry.value_dimensions()
        {
            return Err(Error::backend(
                "incompatible Nemotron attention residual or value width",
            ));
        }
        self.projection.logical_parameters()
    }
}
