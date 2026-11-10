//! Typed parameter declarations for Kimi's recurrent and latent attention units.

use eredu_nn::{
    CausalDepthwiseConvolutionSpec, ConvolutionActivation, DeclareLogicalParameters, Error,
    GatedDeltaHeadGeometry, LinearSpec, LogicalParameterDeclaration, LogicalParameterRole,
    LowRankProjectionSpec, NormalizationConstructionSpec, ParameterSpec,
};

use super::ModelArgs;

pub(crate) fn linear_spec(
    args: &ModelArgs,
    name: &str,
    input: i32,
    output: i32,
) -> Result<LinearSpec, Error> {
    let spec = LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
        bias: None,
        format: crate::linear_format::standard_linear_format(
            name,
            args.weight_quantization_for(name).into(),
        )?,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

pub(crate) fn normalization_spec(
    args: &ModelArgs,
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

/// Declares a dense or shared SwiGLU branch at global or local width.
pub(crate) fn dense_parameters(
    args: &ModelArgs,
    prefix: &str,
    intermediate: i32,
) -> Result<eredu_nn::SplitGatedProductSpec, Error> {
    eredu_nn::SplitGatedProductSpec::new(
        linear_spec(
            args,
            &format!("{prefix}.gate_proj.weight"),
            args.hidden_size,
            intermediate,
        )?,
        linear_spec(
            args,
            &format!("{prefix}.up_proj.weight"),
            args.hidden_size,
            intermediate,
        )?,
        linear_spec(
            args,
            &format!("{prefix}.down_proj.weight"),
            intermediate,
            args.hidden_size,
        )?,
        eredu_nn::GatedProductPolicy::default(),
    )
}

#[derive(Debug, Clone)]
pub(crate) struct KdaSpec {
    pub q: LinearSpec,
    pub k: LinearSpec,
    pub v: LinearSpec,
    pub convolutions: [CausalDepthwiseConvolutionSpec; 3],
    pub f_a: LinearSpec,
    pub f_b: LinearSpec,
    pub beta: LinearSpec,
    pub g_a: LinearSpec,
    pub g_b: LinearSpec,
    pub a_log: ParameterSpec,
    pub dt_bias: ParameterSpec,
    pub norm: NormalizationConstructionSpec,
    pub output: LinearSpec,
    /// Parameter axes only; recurrence and history ownership remain KDA contracts.
    pub geometry: GatedDeltaHeadGeometry,
}

impl KdaSpec {
    pub fn transition_shape(&self) -> [i32; 4] {
        [1, 1, self.geometry.value_heads(), 1]
    }
    pub fn decay_bias_shape(&self) -> [i32; 1] {
        [self.geometry.value_width()]
    }
    /// Three independent causal histories and one FP32 recurrent matrix per head.
    pub fn state_policy(&self) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        use eredu_core::cache::{
            LayerCachePolicy, MutableStateResidency, StateTensorDimension as D, StateTensorDtype,
            StateTensorPolicy, StateTensorRole,
        };
        self.logical_parameters()?;
        let fixed = |value| D::fixed(value).map_err(Error::backend);
        let mut tensors = self
            .convolutions
            .iter()
            .enumerate()
            .map(|(slot, convolution)| {
                StateTensorPolicy::new(
                    StateTensorRole::Convolution { slot: slot as u32 },
                    vec![
                        D::Batch,
                        fixed(convolution.history_len()?)?,
                        fixed(convolution.channels)?,
                    ],
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .map_err(Error::backend)
            })
            .collect::<Result<Vec<_>, _>>()?;
        tensors.push(
            StateTensorPolicy::new(
                StateTensorRole::Recurrent,
                vec![
                    D::Batch,
                    fixed(self.geometry.value_heads())?,
                    fixed(self.geometry.key_dimensions())?,
                    fixed(self.geometry.value_dimensions())?,
                ],
                StateTensorDtype::Float32,
                MutableStateResidency::LayerScopedOffloadable,
            )
            .map_err(Error::backend)?,
        );
        LayerCachePolicy::fixed_only(tensors).map_err(Error::backend)
    }
    pub fn projections(&self) -> [&LinearSpec; 9] {
        [
            &self.q,
            &self.k,
            &self.v,
            &self.f_a,
            &self.f_b,
            &self.beta,
            &self.g_a,
            &self.g_b,
            &self.output,
        ]
    }
}

impl DeclareLogicalParameters for KdaSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let g = self.geometry;
        let hidden = self.q.input;
        let affine =
            |spec: &LinearSpec, input, output| spec.input == input && spec.output == output;
        if g.key_heads() != g.value_heads()
            || g.key_dimensions() != g.value_dimensions()
            || !affine(&self.q, hidden, g.key_width())
            || !affine(&self.k, hidden, g.key_width())
            || !affine(&self.v, hidden, g.value_width())
            || !affine(&self.f_a, hidden, g.key_dimensions())
            || !affine(&self.f_b, g.key_dimensions(), g.value_width())
            || !affine(&self.g_a, hidden, g.key_dimensions())
            || !affine(&self.g_b, g.key_dimensions(), g.value_width())
            || !affine(&self.beta, hidden, g.value_heads())
            || !affine(&self.output, g.value_width(), hidden)
            || self.norm.dimensions != g.value_dimensions()
            || self.norm.groups.is_some()
            || self.projections().iter().any(|p| p.bias.is_some())
            || self.convolutions.iter().any(|c| {
                c.channels != g.value_width()
                    || c.dilation != 1
                    || c.kernel_size != self.convolutions[0].kernel_size
                    || c.bias.is_some()
                    || c.activation != ConvolutionActivation::Silu
                    || c.arithmetic != eredu_nn::ConvolutionArithmetic::Native
            })
        {
            return Err(Error::backend(
                "incompatible Kimi KDA parameter geometry or convolution policy",
            ));
        }
        let mut parameters = Vec::new();
        for projection in self.projections() {
            parameters.extend(projection.logical_parameters()?);
        }
        parameters.push(LogicalParameterDeclaration::new(
            &self.dt_bias,
            &self.decay_bias_shape(),
            LogicalParameterRole::RecurrentBias,
        )?);
        parameters.extend(self.norm.logical_parameters()?);
        for convolution in &self.convolutions {
            parameters.extend(convolution.logical_parameters()?);
        }
        parameters.push(LogicalParameterDeclaration::new(
            &self.a_log,
            &self.transition_shape(),
            LogicalParameterRole::RecurrentTransition,
        )?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

pub(crate) fn kda_spec(args: &ModelArgs, root: &str, heads: i32) -> Result<KdaSpec, Error> {
    let geometry = GatedDeltaHeadGeometry::new(
        heads,
        heads,
        args.kda_config.head_dim,
        args.kda_config.head_dim,
    )?;
    let head_dim = geometry.key_dimensions();
    let projection = geometry.value_width();
    let linear = |field: &str, input, output| {
        linear_spec(args, &format!("{root}.{field}.weight"), input, output)
    };
    let parameter =
        |field: &str| ParameterSpec::trainable(format!("{root}.{field}")).map_err(Error::backend);
    let convolution = |field: &str| -> Result<_, Error> {
        Ok(CausalDepthwiseConvolutionSpec {
            arithmetic: eredu_nn::ConvolutionArithmetic::Native,
            dilation: 1,
            channels: projection,
            kernel_size: args.kda_config.short_conv_kernel_size,
            weight: parameter(&format!("{field}.weight"))?,
            bias: None,
            activation: ConvolutionActivation::Silu,
        })
    };
    let spec = KdaSpec {
        q: linear("q_proj", args.hidden_size, projection)?,
        k: linear("k_proj", args.hidden_size, projection)?,
        v: linear("v_proj", args.hidden_size, projection)?,
        convolutions: [
            convolution("q_conv1d")?,
            convolution("k_conv1d")?,
            convolution("v_conv1d")?,
        ],
        f_a: linear("f_a_proj", args.hidden_size, head_dim)?,
        f_b: linear("f_b_proj", head_dim, projection)?,
        beta: linear("b_proj", args.hidden_size, heads)?,
        g_a: linear("g_a_proj", args.hidden_size, head_dim)?,
        g_b: linear("g_b_proj", head_dim, projection)?,
        a_log: parameter("A_log")?,
        dt_bias: parameter("dt_bias")?,
        norm: normalization_spec(args, &format!("{root}.o_norm.weight"), head_dim)?,
        output: linear("o_proj", projection, args.hidden_size)?,
        geometry,
    };
    spec.logical_parameters()?;
    Ok(spec)
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
pub(crate) enum LatentSpec {
    Fused(LinearSpec),
    Split { key: LinearSpec, value: LinearSpec },
}
impl DeclareLogicalParameters for LatentSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Fused(spec) => spec.logical_parameters(),
            Self::Split { key, value } => {
                let mut parameters = key.logical_parameters()?;
                parameters.extend(value.logical_parameters()?);
                Ok(parameters)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MlaSpec {
    pub query: QuerySpec,
    pub kv_a: LinearSpec,
    pub kv_norm: NormalizationConstructionSpec,
    pub kv_b: LatentSpec,
    pub output: LinearSpec,
    pub geometry: eredu_nn::LatentAttentionGeometry,
    pub scale: f32,
}

impl DeclareLogicalParameters for MlaSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let (query_input, query_output) = match &self.query {
            QuerySpec::Direct(query) => (query.input, query.output),
            QuerySpec::LowRank(query) => (
                query
                    .first
                    .as_ref()
                    .ok_or_else(|| Error::backend("Kimi MLA requires its query input projection"))?
                    .input,
                query.second.output,
            ),
        };
        let geometry = self.geometry;
        let reconstruction_matches = match &self.kv_b {
            LatentSpec::Fused(kv) => {
                kv.input == geometry.latent_dimensions()
                    && kv.output == geometry.reconstruction_width()?
            }
            LatentSpec::Split { key, value } => {
                key.input == geometry.latent_dimensions()
                    && key.output == geometry.key_width()
                    && value.input == geometry.latent_dimensions()
                    && value.output == geometry.value_width()
            }
        };
        if query_input != self.kv_a.input
            || query_output != geometry.query_width()
            || self.kv_a.output != geometry.ingress_width()
            || self.kv_norm.dimensions != geometry.latent_dimensions()
            || !reconstruction_matches
            || self.output.input != geometry.value_width()
            || self.output.output != query_input
        {
            return Err(Error::backend(
                "Kimi MLA query, ingress, reconstruction or residual geometry mismatch",
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

pub(crate) fn mla_spec(args: &ModelArgs, root: &str, heads: i32) -> Result<MlaSpec, Error> {
    let geometry = eredu_nn::LatentAttentionGeometry::new(
        heads,
        args.qk_nope_head_dim,
        args.qk_rope_head_dim,
        args.v_head_dim,
        args.kv_lora_rank,
    )?;
    let linear = |field: &str, input, output| {
        linear_spec(args, &format!("{root}.{field}.weight"), input, output)
    };
    let query = if let Some(rank) = args.q_lora_rank {
        QuerySpec::LowRank(LowRankProjectionSpec {
            first: Some(linear("q_a_proj", args.hidden_size, rank)?),
            normalization: normalization_spec(args, &format!("{root}.q_a_layernorm.weight"), rank)?,
            second: linear("q_b_proj", rank, geometry.query_width())?,
        })
    } else {
        QuerySpec::Direct(linear("q_proj", args.hidden_size, geometry.query_width())?)
    };
    let kv_b = if args.split_kv_b {
        LatentSpec::Split {
            key: linear(
                "k_b_proj",
                geometry.latent_dimensions(),
                geometry.key_width(),
            )?,
            value: linear(
                "v_b_proj",
                geometry.latent_dimensions(),
                geometry.value_width(),
            )?,
        }
    } else {
        LatentSpec::Fused(linear(
            "kv_b_proj",
            geometry.latent_dimensions(),
            geometry.reconstruction_width()?,
        )?)
    };
    let spec = MlaSpec {
        query,
        kv_a: linear(
            "kv_a_proj_with_mqa",
            args.hidden_size,
            geometry.ingress_width(),
        )?,
        kv_norm: normalization_spec(
            args,
            &format!("{root}.kv_a_layernorm.weight"),
            geometry.latent_dimensions(),
        )?,
        kv_b,
        output: linear("o_proj", geometry.value_width(), args.hidden_size)?,
        geometry,
        scale: (geometry.query_head_dimensions() as f32).sqrt().recip(),
    };
    spec.logical_parameters()?;
    Ok(spec)
}
