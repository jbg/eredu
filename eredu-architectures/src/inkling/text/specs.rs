//! Parameter ownership and logical geometry of Inkling's shared attention body.

use eredu_core::AttentionPolicy;
use eredu_nn::{
    CausalDepthwiseConvolutionSpec, ConvolutionActivation, DeclareLogicalParameters, Error,
    LinearSpec, LogicalParameterDeclaration, LogicalParameterRole, NormalizationConstructionSpec,
    Parameter, ParameterSpec, Tensor,
};

use crate::inkling::TextArgs;

#[derive(Debug, Clone)]
pub(crate) struct RelativeProjectionSpec {
    pub weight: ParameterSpec,
    pub shape: [i32; 2],
}
impl RelativeProjectionSpec {
    pub fn unloaded<T: Tensor>(&self, context: &T::Context) -> Result<Parameter<T>, Error> {
        Parameter::unloaded(self.weight.clone(), &self.shape, context)
    }
}
impl DeclareLogicalParameters for RelativeProjectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.shape,
            LogicalParameterRole::LinearWeight,
        )?])
    }
}

fn normalization(
    args: &TextArgs,
    root: &str,
    field: &str,
    width: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    Ok(NormalizationConstructionSpec::learned(
        width,
        args.rms_norm_eps,
        ParameterSpec::trainable(format!("{root}.{field}.weight")).map_err(Error::backend)?,
    ))
}
fn convolution(
    args: &TextArgs,
    root: &str,
    field: &str,
    channels: i32,
) -> Result<CausalDepthwiseConvolutionSpec, Error> {
    Ok(CausalDepthwiseConvolutionSpec {
        arithmetic: eredu_nn::ConvolutionArithmetic::Native,
        dilation: 1,
        channels,
        kernel_size: args.sconv_kernel_size,
        weight: ParameterSpec::trainable(format!("{root}.{field}.weight"))
            .map_err(Error::backend)?,
        bias: None,
        activation: ConvolutionActivation::Identity,
    })
}

#[derive(Debug, Clone)]
pub(crate) struct AttentionParameters {
    pub policy: AttentionPolicy,
    pub log_scaling_floor: Option<i32>,
    pub log_scaling_alpha: f32,
    pub projection: eredu_nn::SelfAttentionProjectionSpec,
    pub relative: LinearSpec,
    pub query_norm: NormalizationConstructionSpec,
    pub key_norm: NormalizationConstructionSpec,
    pub relative_projection: RelativeProjectionSpec,
    pub key_convolution: CausalDepthwiseConvolutionSpec,
    pub value_convolution: CausalDepthwiseConvolutionSpec,
}
impl AttentionParameters {
    fn new_at(args: &TextArgs, policy: AttentionPolicy, block: &str) -> Result<Self, Error> {
        let local = policy.window().is_some();
        let query_heads = args.query_heads(local);
        let key_value_heads = args.key_value_heads(local);
        let head_dimensions = args.attention_head_dim(local);
        let heads = eredu_nn::AttentionHeadGeometry::new(
            query_heads,
            key_value_heads,
            head_dimensions,
            head_dimensions,
        )?;
        let extent = policy
            .window()
            .map(|window| window.get() as i32)
            .unwrap_or(args.rel_extent);
        let product = |a: i32, b: i32| {
            a.checked_mul(b)
                .ok_or_else(|| Error::backend("Inkling attention width overflow"))
        };
        let root = format!("{block}.self_attn");
        let linear = |field: &str, input, output, bias| {
            let weight = format!("{root}.{field}.weight");
            Ok::<_, Error>(LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                bias: if bias {
                    Some(
                        ParameterSpec::trainable(format!("{root}.{field}.bias"))
                            .map_err(Error::backend)?,
                    )
                } else {
                    None
                },
                format: crate::linear_format::standard_linear_format(
                    &weight,
                    args.linear_format_for(&weight),
                )?,
            })
        };
        let parameters = Self {
            policy,
            log_scaling_floor: args.log_scaling_n_floor,
            log_scaling_alpha: args.log_scaling_alpha,
            projection: eredu_nn::SelfAttentionProjectionSpec::new(
                heads,
                eredu_nn::AttentionQueryLayout::QueryOnly,
                linear("q_proj", args.hidden_size, heads.query_width(), args.q_bias)?,
                linear("k_proj", args.hidden_size, heads.key_width(), false)?,
                linear("v_proj", args.hidden_size, heads.value_width(), false)?,
                linear(
                    "o_proj",
                    heads.attended_width(),
                    args.hidden_size,
                    args.o_bias,
                )?,
            )?,
            relative: linear(
                "r_proj",
                args.hidden_size,
                product(query_heads, args.d_rel)?,
                false,
            )?,
            query_norm: normalization(args, &root, "q_norm", head_dimensions)?,
            key_norm: normalization(args, &root, "k_norm", head_dimensions)?,
            relative_projection: RelativeProjectionSpec {
                weight: ParameterSpec::trainable(format!("{root}.rel_proj"))
                    .map_err(Error::backend)?,
                shape: [args.d_rel, extent],
            },
            key_convolution: convolution(args, &root, "k_sconv", heads.key_width())?,
            value_convolution: convolution(args, &root, "v_sconv", heads.value_width())?,
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }

    pub fn replicated_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.query_norm.logical_parameters()?;
        parameters.extend(self.key_norm.logical_parameters()?);
        parameters.extend(self.relative_projection.logical_parameters()?);
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for AttentionParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.projection.validate_geometry()?;
        let heads = self.projection.geometry;
        let relative_rows = heads
            .query_heads()
            .checked_mul(self.relative_projection.shape[0])
            .ok_or_else(|| Error::backend("Inkling relative projection width overflow"))?;
        if self.projection.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || self.projection.query.input != self.projection.output.output
            || heads.key_dimensions() != heads.value_dimensions()
            || self.query_norm.dimensions != heads.key_dimensions()
            || self.key_norm.dimensions != heads.key_dimensions()
            || self.relative.input != self.projection.query.input
            || self.relative.output != relative_rows
            || self.key_convolution.channels != heads.key_width()
            || self.value_convolution.channels != heads.value_width()
        {
            return Err(Error::backend(
                "Inkling attention, relative or convolution geometry mismatch",
            ));
        }
        let mut parameters = Vec::new();
        for spec in [
            &self.projection.query,
            &self.projection.key,
            &self.projection.value,
            &self.relative,
            &self.projection.output,
        ] {
            parameters.extend(spec.logical_parameters()?);
        }
        parameters.extend(self.replicated_parameters()?);
        parameters.extend(self.key_convolution.logical_parameters()?);
        parameters.extend(self.value_convolution.logical_parameters()?);
        Ok(parameters)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CommonParameters {
    pub attention: AttentionParameters,
    pub input_norm: NormalizationConstructionSpec,
    pub post_attention_norm: NormalizationConstructionSpec,
    pub attention_convolution: CausalDepthwiseConvolutionSpec,
    pub feed_forward_convolution: CausalDepthwiseConvolutionSpec,
}
impl CommonParameters {
    pub(in crate::inkling) fn new_at(
        args: &TextArgs,
        policy: AttentionPolicy,
        block: &str,
    ) -> Result<Self, Error> {
        let parameters = Self {
            attention: AttentionParameters::new_at(args, policy, block)?,
            input_norm: normalization(args, block, "input_layernorm", args.hidden_size)?,
            post_attention_norm: normalization(
                args,
                block,
                "post_attention_layernorm",
                args.hidden_size,
            )?,
            attention_convolution: convolution(args, block, "attn_sconv", args.hidden_size)?,
            feed_forward_convolution: convolution(args, block, "mlp_sconv", args.hidden_size)?,
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }

    pub(in crate::inkling) fn project_formats(&mut self, args: &TextArgs) -> Result<(), Error> {
        let attention = &mut self.attention;
        for spec in [
            &mut attention.projection.query,
            &mut attention.projection.key,
            &mut attention.projection.value,
            &mut attention.projection.output,
            &mut attention.relative,
        ] {
            let mut format = crate::linear_format::standard_linear_format(
                spec.weight.id.as_str(),
                args.linear_format_for(spec.weight.id.as_str()),
            )?
            .with_arithmetic(spec.format.arithmetic())?;
            if spec.format.exact_zero_inputs() {
                format = format.with_exact_zero_inputs();
            }
            spec.format = format;
        }
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
    /// Projects only attention head extents; identities and replicated geometry remain source-owned.
    pub(in crate::inkling) fn localized(
        &self,
        query_heads: i32,
        kv_heads: i32,
    ) -> Result<Self, Error> {
        let original = self.attention.projection.geometry;
        if query_heads > original.query_heads() || kv_heads > original.key_value_heads() {
            return Err(Error::backend(
                "Inkling local heads exceed declared source geometry",
            ));
        }
        let geometry = eredu_nn::AttentionHeadGeometry::new(
            query_heads,
            kv_heads,
            original.key_dimensions(),
            original.value_dimensions(),
        )?;
        let mut parameters = self.clone();
        let attention = &mut parameters.attention;
        attention.projection.geometry = geometry;
        attention.projection.query.output = geometry.query_width();
        attention.projection.key.output = geometry.key_width();
        attention.projection.value.output = geometry.value_width();
        attention.projection.output.input = geometry.attended_width();
        attention.relative.output = query_heads
            .checked_mul(attention.relative_projection.shape[0])
            .ok_or_else(|| Error::backend("Inkling relative width overflow"))?;
        attention.key_convolution.channels = geometry.key_width();
        attention.value_convolution.channels = geometry.value_width();
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }

    /// Exact KV state and four histories consumed by this block invocation.
    pub fn state_policy(&self) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        use eredu_core::cache::{
            LayerCachePolicy, MutableStateResidency, StateTensorDimension as D, StateTensorDtype,
            StateTensorPolicy, StateTensorRole,
        };
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        let fixed = |n| D::fixed(n).map_err(Error::backend_source);
        let tensors = [
            &self.attention.key_convolution,
            &self.attention.value_convolution,
            &self.attention_convolution,
            &self.feed_forward_convolution,
        ]
        .into_iter()
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
            .map_err(Error::backend_source)
        })
        .collect::<Result<Vec<_>, _>>()?;
        let heads = self.attention.projection.geometry;
        LayerCachePolicy::key_value_with_state(
            self.attention.policy,
            heads.key_value_heads(),
            heads.key_dimensions(),
            tensors,
            Vec::new(),
        )
        .map_err(Error::backend_source)
    }

    pub fn replicated_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.attention.replicated_parameters()?;
        parameters.extend(self.input_norm.logical_parameters()?);
        parameters.extend(self.post_attention_norm.logical_parameters()?);
        parameters.extend(self.attention_convolution.logical_parameters()?);
        parameters.extend(self.feed_forward_convolution.logical_parameters()?);
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for CommonParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let hidden = self.attention.projection.query.input;
        if self.input_norm.dimensions != hidden
            || self.post_attention_norm.dimensions != hidden
            || self.attention_convolution.channels != hidden
            || self.feed_forward_convolution.channels != hidden
        {
            return Err(Error::backend(
                "Inkling residual normalization or convolution width mismatch",
            ));
        }
        let mut parameters = self.attention.logical_parameters()?;
        parameters.extend(self.input_norm.logical_parameters()?);
        parameters.extend(self.post_attention_norm.logical_parameters()?);
        parameters.extend(self.attention_convolution.logical_parameters()?);
        parameters.extend(self.feed_forward_convolution.logical_parameters()?);
        Ok(parameters)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SparseParameters {
    pub router: eredu_nn::JointGroupSelectorSpec,
    pub routed: eredu_nn::GroupedGatedProductSpec,
    pub shared: eredu_nn::GroupedGatedProductSpec,
}
impl SparseParameters {
    pub(in crate::inkling) fn new_at(args: &TextArgs, block: &str) -> Result<Self, Error> {
        let root = format!("{block}.moe");
        let router_root = format!("{block}.moe.router");
        let parameter = |field: &str| {
            ParameterSpec::trainable(format!("{router_root}.{field}")).map_err(Error::backend)
        };
        let router = eredu_nn::JointGroupSelectorSpec::new(
            args.hidden_size,
            parameter("weight")?,
            parameter("bias")?,
            parameter("global_scale")?,
            eredu_nn::JointGroupSelectionSpec::new(
                args.n_routed_experts,
                args.n_shared_experts,
                args.num_experts_per_tok,
                args.route_scale,
            )?,
        )?;
        let parameters = Self {
            routed: super::expert_bank_spec_at(
                args,
                &root,
                "experts",
                router.selection().selectable_groups(),
            )?,
            shared: super::expert_bank_spec_at(
                args,
                &root,
                "shared_experts",
                router.selection().always_on_groups(),
            )?,
            router,
        };
        parameters.validate()?;
        Ok(parameters)
    }
    pub(in crate::inkling) fn project_formats(&mut self, args: &TextArgs) -> Result<(), Error> {
        fn bank(
            source: &eredu_nn::GroupedGatedProductSpec,
            args: &TextArgs,
        ) -> Result<eredu_nn::GroupedGatedProductSpec, Error> {
            let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = source.layout()
            else {
                return Err(Error::backend(
                    "Inkling source bank requires its declared packed layout",
                ));
            };
            let project =
                |projection: &eredu_nn::GroupedProjectionSpec, rows| -> Result<_, Error> {
                    let previous = projection.format();
                    let mut format = crate::linear_format::standard_expert_format(
                        projection.weight().id.as_str(),
                        args.linear_format_for(projection.weight().id.as_str()),
                        rows,
                    )?
                    .with_arithmetic(previous.arithmetic())?;
                    if previous.exact_zero_inputs() {
                        format = format.with_exact_zero_inputs();
                    }
                    eredu_nn::GroupedProjectionSpec::new(
                        projection.weight().clone(),
                        projection.bias().cloned(),
                        format,
                    )
                };
            Ok(eredu_nn::GroupedGatedProductSpec::new(
                source.group_count(),
                source.input_dimensions(),
                source.intermediate_dimensions(),
                source.output_dimensions(),
                source.policy(),
                eredu_nn::GatedProductGroupLayout::Packed {
                    gate_up: project(gate_up, eredu_nn::LinearRowLayout::equal_partitions(2)?)?,
                    down: project(down, eredu_nn::LinearRowLayout::Contiguous)?,
                },
            )?
            .with_reduction(source.reduction()))
        }
        self.routed = bank(&self.routed, args)?;
        self.shared = bank(&self.shared, args)?;
        self.validate()
    }
    pub(in crate::inkling) fn localized(&self, width: i32) -> Result<Self, Error> {
        if width <= 0
            || width > self.routed.intermediate_dimensions()
            || width > self.shared.intermediate_dimensions()
        {
            return Err(Error::backend(
                "Inkling local expert width exceeds its source",
            ));
        }
        let mut parameters = self.clone();
        parameters.routed = self
            .routed
            .clone()
            .with_group_geometry(self.routed.group_count(), width)?;
        parameters.shared = self
            .shared
            .clone()
            .with_group_geometry(self.shared.group_count(), width)?;
        parameters.validate()?;
        Ok(parameters)
    }
    pub fn validate(&self) -> Result<(), Error> {
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
    /// Binds independently localized banks while retaining the global joint router.
    pub fn with_banks(
        mut self,
        routed: eredu_nn::GroupedGatedProductSpec,
        shared: eredu_nn::GroupedGatedProductSpec,
    ) -> Result<Self, Error> {
        for (source, selected) in [(&self.routed, &routed), (&self.shared, &shared)] {
            if selected.input_dimensions() != source.input_dimensions()
                || selected.output_dimensions() != source.output_dimensions()
                || selected.intermediate_dimensions() != source.intermediate_dimensions()
                || selected.group_count() > source.group_count()
            {
                return Err(Error::backend(
                    "Inkling bank selection changes prepared local geometry",
                ));
            }
            let source_parameters = source.logical_parameters()?;
            let selected_parameters = selected.logical_parameters()?;
            if source_parameters.len() != selected_parameters.len()
                || source_parameters
                    .iter()
                    .zip(&selected_parameters)
                    .any(|(a, b)| {
                        a.parameter != b.parameter
                            || a.role != b.role
                            || a.linear_format != b.linear_format
                    })
            {
                return Err(Error::backend(
                    "Inkling bank selection changes declared identities, roles or encodings",
                ));
            }
        }
        self.routed = routed;
        self.shared = shared;
        self.validate()?;
        Ok(self)
    }
}
impl DeclareLogicalParameters for SparseParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let hidden = self.router.input_dimensions();
        if self.routed.input_dimensions() != hidden
            || self.routed.output_dimensions() != hidden
            || self.shared.input_dimensions() != hidden
            || self.shared.output_dimensions() != hidden
            || self.routed.group_count() > self.router.selection().selectable_groups()
            || self.shared.group_count() > self.router.selection().always_on_groups()
            || [&self.routed, &self.shared].into_iter().any(|bank| {
                bank.policy() != eredu_nn::GatedProductPolicy::ordinary_silu()
                    || bank.reduction() != eredu_nn::GroupReduction::Sum
            })
        {
            return Err(Error::backend(
                "Inkling joint routed/shared geometry or equation mismatch",
            ));
        }
        let mut parameters = self.router.logical_parameters()?;
        parameters.extend(self.routed.logical_parameters()?);
        parameters.extend(self.shared.logical_parameters()?);
        Ok(parameters)
    }
}
