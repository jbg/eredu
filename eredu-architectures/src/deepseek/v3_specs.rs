//! Typed construction declarations for V3/R1 target and prediction invocations.

use super::V3Args;
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration,
    LowRankProjectionSpec, NormalizationConstructionSpec, ParameterSpec, RotarySpec,
};

fn declare_linear_spec(
    args: &V3Args,
    name: &str,
    input: i32,
    output: i32,
) -> Result<LinearSpec, Error> {
    let spec = LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
        bias: None,
        format: crate::linear_format::standard_linear_format(name, args.linear_format_for(name))?,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

pub(super) fn declare_normalization_spec(
    args: &V3Args,
    name: &str,
    dimensions: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    let spec = NormalizationConstructionSpec::learned(
        dimensions,
        args.rms_norm_eps,
        ParameterSpec::trainable(name).map_err(Error::backend)?,
    );
    spec.logical_parameters()?;
    Ok(spec)
}

pub(super) fn declare_dense_parameters(
    args: &V3Args,
    root: &str,
) -> Result<eredu_nn::SplitGatedProductSpec, Error> {
    eredu_nn::SplitGatedProductSpec::new(
        declare_linear_spec(
            args,
            &format!("{root}.gate_proj.weight"),
            args.hidden_size,
            args.intermediate_size,
        )?,
        declare_linear_spec(
            args,
            &format!("{root}.up_proj.weight"),
            args.hidden_size,
            args.intermediate_size,
        )?,
        declare_linear_spec(
            args,
            &format!("{root}.down_proj.weight"),
            args.intermediate_size,
            args.hidden_size,
        )?,
        eredu_nn::GatedProductPolicy::default(),
    )
}

#[derive(Debug, Clone)]
pub(crate) enum QuerySpec {
    Direct(LinearSpec),
    LowRank(LowRankProjectionSpec),
}
impl DeclareLogicalParameters for QuerySpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Direct(spec) => spec.logical_parameters(),
            Self::LowRank(spec) => spec.logical_parameters(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AttentionSpec {
    pub query: QuerySpec,
    pub kv_a: LinearSpec,
    pub kv_norm: NormalizationConstructionSpec,
    pub kv_b: LinearSpec,
    pub output: LinearSpec,
    pub geometry: eredu_nn::LatentAttentionGeometry,
    pub rotary: RotarySpec,
    pub scale: f32,
}
impl DeclareLogicalParameters for AttentionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let (query_input, query_output) = match &self.query {
            QuerySpec::Direct(query) => (query.input, query.output),
            QuerySpec::LowRank(query) => (
                query
                    .first
                    .as_ref()
                    .ok_or_else(|| Error::backend("V3 MLA requires its query input projection"))?
                    .input,
                query.second.output,
            ),
        };
        let geometry = self.geometry;
        if query_input != self.kv_a.input
            || query_output != geometry.query_width()
            || self.kv_a.output != geometry.ingress_width()
            || self.kv_norm.dimensions != geometry.latent_dimensions()
            || self.kv_b.input != geometry.latent_dimensions()
            || self.kv_b.output != geometry.reconstruction_width()?
            || self.output.input != geometry.value_width()
            || self.output.output != query_input
            || self.rotary.dimensions != geometry.positional_dimensions()
            || self.rotary.dimensions <= 0
        {
            return Err(Error::backend(
                "V3 MLA query, ingress, reconstruction, residual or rotary geometry mismatch",
            ));
        }
        let mut parameters = self.query.logical_parameters()?;
        parameters.extend(self.kv_a.logical_parameters()?);
        parameters.extend(self.kv_norm.logical_parameters()?);
        parameters.extend(self.kv_b.logical_parameters()?);
        parameters.extend(self.output.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

pub(super) fn declare_attention_spec(args: &V3Args, root: &str) -> Result<AttentionSpec, Error> {
    let geometry = eredu_nn::LatentAttentionGeometry::new(
        args.num_attention_heads,
        args.qk_nope_head_dim,
        args.qk_rope_head_dim,
        args.v_head_dim,
        args.kv_lora_rank,
    )?;
    let linear = |field: &str, input, output| {
        declare_linear_spec(args, &format!("{root}.{field}.weight"), input, output)
    };
    let query = if let Some(rank) = args.q_lora_rank {
        QuerySpec::LowRank(LowRankProjectionSpec {
            first: Some(linear("q_a_proj", args.hidden_size, rank)?),
            normalization: declare_normalization_spec(
                args,
                &format!("{root}.q_a_layernorm.weight"),
                rank,
            )?,
            second: linear("q_b_proj", rank, geometry.query_width())?,
        })
    } else {
        QuerySpec::Direct(linear("q_proj", args.hidden_size, geometry.query_width())?)
    };
    let spec = AttentionSpec {
        query,
        kv_a: linear(
            "kv_a_proj_with_mqa",
            args.hidden_size,
            geometry.ingress_width(),
        )?,
        kv_norm: declare_normalization_spec(
            args,
            &format!("{root}.kv_a_layernorm.weight"),
            geometry.latent_dimensions(),
        )?,
        kv_b: linear(
            "kv_b_proj",
            geometry.latent_dimensions(),
            geometry.reconstruction_width()?,
        )?,
        output: linear("o_proj", geometry.value_width(), args.hidden_size)?,
        geometry,
        rotary: RotarySpec {
            arithmetic: eredu_nn::RotaryArithmetic::Native,
            dimensions: geometry.positional_dimensions(),
            base: args.rope_theta,
            traditional: false,
            algorithm: args
                .rope_scaling
                .as_ref()
                .map_or(eredu_nn::RotaryAlgorithm::Default, |yarn| {
                    yarn.rotary_algorithm()
                }),
        },
        scale: (geometry.query_head_dimensions() as f32).sqrt().recip()
            * args
                .rope_scaling
                .as_ref()
                .map_or(1.0, |yarn| yarn.attention_multiplier()),
    };
    spec.logical_parameters()?;
    Ok(spec)
}

#[derive(Debug, Clone)]
pub(crate) struct PredictionSpec {
    pub embedding_norm: NormalizationConstructionSpec,
    pub hidden_norm: NormalizationConstructionSpec,
    pub fusion: LinearSpec,
    pub output_norm: NormalizationConstructionSpec,
    pub output: LinearSpec,
}
impl DeclareLogicalParameters for PredictionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.embedding_norm.logical_parameters()?;
        parameters.extend(self.hidden_norm.logical_parameters()?);
        parameters.extend(self.fusion.logical_parameters()?);
        parameters.extend(self.output_norm.logical_parameters()?);
        parameters.extend(self.output.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

pub(super) fn declare_prediction_spec(args: &V3Args, root: &str) -> Result<PredictionSpec, Error> {
    let input = args
        .hidden_size
        .checked_mul(2)
        .ok_or_else(|| Error::backend("V3 prediction fusion width overflowed"))?;
    let spec = PredictionSpec {
        embedding_norm: declare_normalization_spec(
            args,
            &format!("{root}.enorm.weight"),
            args.hidden_size,
        )?,
        hidden_norm: declare_normalization_spec(
            args,
            &format!("{root}.hnorm.weight"),
            args.hidden_size,
        )?,
        fusion: declare_linear_spec(
            args,
            &format!("{root}.eh_proj.weight"),
            input,
            args.hidden_size,
        )?,
        output_norm: declare_normalization_spec(
            args,
            &format!("{root}.shared_head.norm.weight"),
            args.hidden_size,
        )?,
        output: declare_linear_spec(
            args,
            &format!("{root}.shared_head.head.weight"),
            args.hidden_size,
            args.vocab_size,
        )?,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

/// Projects retained scope identities; construction factories are used only at the source boundary.
pub(crate) fn attention_spec(args: &V3Args, root: &str) -> Result<AttentionSpec, Error> {
    Ok(args
        .parameters()
        .block(
            root.strip_suffix(".self_attn")
                .ok_or_else(|| Error::backend("unknown V3 attention root"))?,
        )?
        .attention
        .clone())
}
pub(crate) fn dense_parameters(
    args: &V3Args,
    root: &str,
) -> Result<eredu_nn::SplitGatedProductSpec, Error> {
    args.parameters()
        .block(
            root.strip_suffix(".mlp")
                .ok_or_else(|| Error::backend("unknown V3 MLP root"))?,
        )?
        .dense
        .clone()
        .ok_or_else(|| Error::backend("V3 invocation has no dense scope"))
}
pub(crate) fn prediction_spec(args: &V3Args, root: &str) -> Result<PredictionSpec, Error> {
    args.parameters().prediction(root).cloned()
}
pub(crate) fn normalization_spec(
    args: &V3Args,
    name: &str,
    dimensions: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    let spec = args.parameters().normalization(name)?;
    if spec.dimensions != dimensions {
        return Err(Error::backend(
            "V3 normalization geometry differs from retained slot",
        ));
    }
    Ok(spec.clone())
}
