//! Typed gated-attention slots and per-invocation execution policy.

use crate::muse::{AttentionEquation, DecoderPolicy, WeightConvention};
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration,
    NormalizationConstructionSpec, ParameterSpec, RotarySpec,
};
use eredu_runtime::{MemberSharding, ParameterGroupSpec, ParameterRole, PartitionUnitAssignment};

#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionGeometry {
    pub hidden: i32,
    pub query_heads: i32,
    pub key_value_heads: i32,
    pub head_width: i32,
}
impl From<&DecoderPolicy> for AttentionGeometry {
    fn from(args: &DecoderPolicy) -> Self {
        Self {
            hidden: args.hidden_size,
            query_heads: args.num_attention_heads,
            key_value_heads: args.num_key_value_heads,
            head_width: args.head_dim,
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) struct AttentionParameters {
    pub root: String,
    pub projection: eredu_nn::SelfAttentionProjectionSpec,
    pub gate: LinearSpec,
    pub norms: Vec<NormalizationConstructionSpec>,
}
impl AttentionParameters {
    pub(in crate::muse) fn new(
        args: &DecoderPolicy,
        root: &str,
        gate: &str,
    ) -> Result<Self, Error> {
        let parameters = Self::from_geometry(
            args.into(),
            root,
            gate,
            |name| args.linear_format_for(name),
            (args.weight_convention == WeightConvention::Gguf).then_some(args.rms_norm_eps),
        )?;
        let prefix = format!("{root}.");
        let projections = parameters.projections();
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.keys())
            .chain(args.quantized_weights.iter().flat_map(|names| names.iter()))
        {
            if name.starts_with(&prefix)
                && !projections
                    .iter()
                    .any(|spec| spec.weight.id.as_str() == name)
            {
                return Err(Error::backend(format!(
                    "Muse attention format references undeclared projection {name}"
                )));
            }
        }
        Ok(parameters)
    }
    pub fn from_geometry(
        geometry: AttentionGeometry,
        root: &str,
        gate: &str,
        format: impl Fn(&str) -> eredu_checkpoint::LinearFormat,
        norm_epsilon: Option<f32>,
    ) -> Result<Self, Error> {
        let AttentionGeometry {
            hidden,
            query_heads,
            key_value_heads,
            head_width,
        } = geometry;
        let heads = eredu_nn::AttentionHeadGeometry::new(
            query_heads,
            key_value_heads,
            head_width,
            head_width,
        )?;
        let linear = |field, input, output| -> Result<LinearSpec, Error> {
            let name = format!("{root}.{field}.weight");
            Ok(LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(&name, format(&name))?,
            })
        };
        let norms = norm_epsilon
            .map(|epsilon| {
                ["q_norm", "k_norm"]
                    .into_iter()
                    .map(|field| {
                        Ok(NormalizationConstructionSpec::learned(
                            head_width,
                            epsilon,
                            ParameterSpec::trainable(format!("{root}.{field}.weight"))
                                .map_err(Error::backend)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, Error>>()
            })
            .transpose()?
            .unwrap_or_default();
        let parameters = Self {
            root: root.into(),
            projection: eredu_nn::SelfAttentionProjectionSpec::new(
                heads,
                eredu_nn::AttentionQueryLayout::QueryOnly,
                linear("q_proj", hidden, heads.query_width())?,
                linear("k_proj", hidden, heads.key_width())?,
                linear("v_proj", hidden, heads.value_width())?,
                linear("o_proj", heads.attended_width(), hidden)?,
            )?,
            gate: linear(gate, hidden, heads.attended_width())?,
            norms,
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
    pub fn projections(&self) -> [&LinearSpec; 5] {
        [
            &self.projection.query,
            &self.projection.key,
            &self.projection.value,
            &self.projection.output,
            &self.gate,
        ]
    }
    /// Target invocations repeat whole K/V heads; DSpark requires distinct heads.
    pub fn parameter_groups(&self, repeat_keys: bool) -> Result<Vec<ParameterGroupSpec>, Error> {
        self.logical_parameters()?;
        let mut queries = self.projection.query.logical_parameters()?;
        queries.extend(self.gate.logical_parameters()?);
        queries.extend(self.projection.output.logical_parameters()?);
        let mut keys = self.projection.key.logical_parameters()?;
        keys.extend(self.projection.value.logical_parameters()?);
        let mut query_group = eredu_runtime::logical_parameter_group(
            format!("{}.query_heads", self.root),
            ParameterRole::AttentionHeads,
            Some(self.projection.geometry.query_heads() as usize),
            &queries,
            |p| {
                if p.parameter.id == self.projection.output.weight.id
                    || self
                        .projection
                        .output
                        .bias
                        .as_ref()
                        .is_some_and(|b| p.parameter.id == b.id)
                {
                    MemberSharding::Partitioned { axis: 1 }
                } else {
                    MemberSharding::Partitioned { axis: 0 }
                }
            },
        )
        .map_err(Error::backend_source)?;
        let mut key_group = eredu_runtime::logical_parameter_group(
            format!("{}.key_value_heads", self.root),
            ParameterRole::AttentionHeads,
            Some(self.projection.geometry.key_value_heads() as usize),
            &keys,
            |_| MemberSharding::Partitioned { axis: 0 },
        )
        .map_err(Error::backend_source)?;
        if repeat_keys {
            query_group = query_group
                .with_partition_unit_assignment(PartitionUnitAssignment::Even)
                .map_err(Error::backend_source)?;
            key_group = key_group
                .with_partition_unit_assignment(PartitionUnitAssignment::Repeat {
                    semantic_units: self.projection.geometry.key_value_heads() as usize,
                })
                .map_err(Error::backend_source)?;
        }
        let mut groups = vec![query_group, key_group];
        if !self.norms.is_empty() {
            groups.push(
                eredu_runtime::logical_parameter_group(
                    format!("{}.qk_norm", self.root),
                    ParameterRole::Replicated,
                    None,
                    self.norms.as_slice(),
                    |_| MemberSharding::Replicated,
                )
                .map_err(Error::backend_source)?,
            );
        }
        Ok(groups)
    }
}
impl DeclareLogicalParameters for AttentionParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let heads = self.projection.geometry;
        if self.projection.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || heads.key_dimensions() != heads.value_dimensions()
            || self.projection.output.output != self.projection.query.input
            || self.gate.input != self.projection.query.input
            || self.gate.output != heads.attended_width()
            || !matches!(self.norms.len(), 0 | 2)
            || self
                .norms
                .iter()
                .any(|norm| norm.dimensions != heads.key_dimensions())
        {
            return Err(Error::backend(
                "Muse attention residual, gate or normalization geometry mismatch",
            ));
        }
        let mut parameters = self.projection.logical_parameters()?;
        parameters.extend(self.gate.logical_parameters()?);
        parameters.extend(self.norms[..].logical_parameters()?);
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct AttentionSpec {
    pub parameters: AttentionParameters,
    pub scale: f32,
    pub policy: eredu_core::AttentionPolicy,
    pub uses_rope: bool,
    pub epsilon: f32,
    pub equation: AttentionEquation,
    pub arithmetic: eredu_nn::AttentionArithmetic,
    pub qk_scale_factor: f32,
    pub query_scale: Option<f32>,
    pub rotary: RotarySpec,
}
impl AttentionSpec {
    /// Declares the source attention owner and its independent invocation policy.
    pub(in crate::muse) fn for_layer(args: &DecoderPolicy, layer: usize) -> Result<Self, Error> {
        Self::with_parameters(
            args,
            layer,
            AttentionParameters::new(
                args,
                &format!("model.layers.{layer}.self_attn"),
                "gate_proj",
            )?,
        )
    }
    /// This invocation's mutable cache, independent of parameter ownership.
    pub fn state_policy(&self) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        self.parameters.logical_parameters()?;
        let heads = self.parameters.projection.geometry;
        eredu_core::cache::LayerCachePolicy::key_value(
            self.policy,
            heads.key_value_heads(),
            heads.key_dimensions(),
        )
        .map_err(Error::backend_source)
    }

    pub fn with_parameters(
        args: &DecoderPolicy,
        layer: usize,
        parameters: AttentionParameters,
    ) -> Result<Self, Error> {
        parameters.logical_parameters()?;
        if [args.rms_norm_eps, args.qk_scale_factor, args.rope_theta]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::backend(
                "Muse attention epsilon, query scaling, and rotary base must be finite and positive",
            ));
        }
        let policy = args
            .attention_schedule
            .get(layer)
            .ok_or_else(|| Error::backend("missing Muse attention invocation"))?;
        let uses_rope = *args
            .layer_uses_rope
            .get(layer)
            .ok_or_else(|| Error::backend("missing Muse attention rotary policy"))?;
        let head_width = parameters.projection.geometry.key_dimensions();
        Ok(Self {
            parameters,
            scale: match args.attention_equation {
                AttentionEquation::QueryScaled => (head_width as f32).sqrt().recip(),
                AttentionEquation::NormalizedHeads => (head_width as f64).powf(-0.5) as f32,
                AttentionEquation::NormalizedHeadsScoreScaled => {
                    ((head_width as f64).powf(-0.5) * args.qk_scale_factor as f64) as f32
                }
            },
            policy: *policy,
            uses_rope,
            epsilon: args.rms_norm_eps,
            equation: args.attention_equation,
            arithmetic: args.attention_arithmetic(),
            qk_scale_factor: args.qk_scale_factor,
            query_scale: (args.weight_convention != WeightConvention::Gguf)
                .then_some(args.qk_scale_factor),
            rotary: RotarySpec {
                arithmetic: args.rotary_arithmetic(),
                dimensions: head_width,
                base: args.rope_theta,
                traditional: args.weight_convention.uses_traditional_rope(),
                algorithm: crate::rotary::normalize_algorithm(args.rope_scaling.as_ref())
                    .map_err(Error::backend)?,
            },
        })
    }
}

/// One SwiGLU expansion/gate pair and its contraction, shared by target and DSpark.
pub(crate) fn dense_parameters(
    hidden: i32,
    intermediate: i32,
    root: &str,
    format: impl Fn(&str) -> eredu_checkpoint::LinearFormat,
) -> Result<eredu_nn::SplitGatedProductSpec, Error> {
    let projection = |field, input, output| -> Result<LinearSpec, Error> {
        let name = format!("{root}.{field}.weight");
        Ok(LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(&name, format(&name))?,
        })
    };
    eredu_nn::SplitGatedProductSpec::new(
        projection("gate_proj", hidden, intermediate)?,
        projection("up_proj", hidden, intermediate)?,
        projection("down_proj", intermediate, hidden)?,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
    )
}
pub(crate) fn dense_parameter_group(
    root: &str,
    parameters: &eredu_nn::SplitGatedProductSpec,
) -> Result<ParameterGroupSpec, Error> {
    eredu_runtime::logical_parameter_group(
        format!("{root}.intermediate"),
        ParameterRole::FeedForwardIntermediate,
        Some(parameters.down.input as usize),
        parameters,
        |p| {
            if p.parameter.id == parameters.down.weight.id {
                MemberSharding::Partitioned { axis: 1 }
            } else if parameters
                .down
                .bias
                .as_ref()
                .is_some_and(|bias| bias.id == p.parameter.id)
            {
                MemberSharding::Replicated
            } else {
                MemberSharding::Partitioned { axis: 0 }
            }
        },
    )
    .map_err(Error::backend_source)
}

use crate::muse::equations::{
    specs::{
        BranchNormSpec, LearnedResidualSpec, LearnedRmsNormSpec, ResidualSpec, StoredResidualSpec,
    },
    RmsGain, RmsNormEquation,
};
use crate::muse::{BlockNormalization, ResidualEquation};

pub(crate) fn learned_norm(
    dimensions: i32,
    epsilon: f32,
    equation: RmsNormEquation,
    name: impl Into<String>,
) -> Result<LearnedRmsNormSpec, Error> {
    LearnedRmsNormSpec::new(
        dimensions,
        epsilon,
        equation,
        ParameterSpec::trainable(name.into()).map_err(Error::backend)?,
    )
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DsparkScalarPolicy {
    pub epsilon: f32,
    pub post_epsilon: f32,
    pub temperature: f32,
    pub minimum_variance: f32,
}
impl DsparkScalarPolicy {
    pub fn from_args(args: &DecoderPolicy) -> Result<Self, Error> {
        let ResidualEquation::Learned {
            temperature,
            minimum_variance,
        } = args.residual_equation
        else {
            return Err(Error::backend(
                "DSpark requires learned FP32 residual gates",
            ));
        };
        Ok(Self {
            epsilon: args.rms_norm_eps,
            post_epsilon: args.post_norm_eps,
            temperature,
            minimum_variance,
        })
    }
    pub const fn released() -> Self {
        Self {
            epsilon: 1e-5,
            post_epsilon: 1e-8,
            temperature: 0.3,
            minimum_variance: 1e-3,
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) struct BlockScalarSpec {
    pub input: LearnedRmsNormSpec,
    pub post_attention: BranchNormSpec,
    pub pre_feed_forward: LearnedRmsNormSpec,
    pub post_feed_forward: LearnedRmsNormSpec,
    pub residuals: [ResidualSpec; 2],
}
impl BlockScalarSpec {
    pub(in crate::muse) fn target(args: &DecoderPolicy, layer: usize) -> Result<Self, Error> {
        if layer >= args.num_hidden_layers as usize {
            return Err(Error::backend(
                "Muse scalar invocation is outside the decoder",
            ));
        }
        let root = format!("model.layers.{layer}");
        let pre = match args.block_normalization {
            BlockNormalization::OutputScaled => {
                RmsNormEquation::OutputScaled(args.weight_convention)
            }
            BlockNormalization::InputScaledFeedForward => {
                RmsNormEquation::OutputScaledFp32(args.weight_convention)
            }
        };
        let post = match args.block_normalization {
            BlockNormalization::OutputScaled => pre,
            BlockNormalization::InputScaledFeedForward => {
                RmsNormEquation::InputScaled(args.weight_convention)
            }
        };
        let norm = |field, epsilon, equation| {
            learned_norm(
                args.hidden_size,
                epsilon,
                equation,
                format!("{root}.{field}.weight"),
            )
        };
        let residual = |field| -> Result<ResidualSpec, Error> {
            let parameter = |suffix| {
                ParameterSpec::trainable(format!("{root}.{field}.{suffix}")).map_err(Error::backend)
            };
            Ok(match args.residual_equation {
                ResidualEquation::Add => ResidualSpec::Add,
                ResidualEquation::Learned {
                    temperature,
                    minimum_variance,
                } => ResidualSpec::Learned(LearnedResidualSpec {
                    dimensions: args.hidden_size,
                    temperature,
                    minimum_variance,
                    gate: parameter("gate")?,
                }),
                ResidualEquation::StoredCoefficients => ResidualSpec::Stored(StoredResidualSpec {
                    dimensions: args.hidden_size,
                    alpha: parameter("alpha")?,
                    beta: parameter("beta")?,
                }),
            })
        };
        let spec = Self {
            input: norm("input_layernorm", args.rms_norm_eps, pre)?,
            post_attention: match args.block_normalization {
                BlockNormalization::OutputScaled => BranchNormSpec::Learned(norm(
                    "post_attention_layernorm",
                    args.post_norm_eps,
                    pre,
                )?),
                BlockNormalization::InputScaledFeedForward => BranchNormSpec::Weightless {
                    epsilon: args.post_norm_eps,
                },
            },
            pre_feed_forward: norm("pre_feedforward_layernorm", args.rms_norm_eps, pre)?,
            post_feed_forward: norm("post_feedforward_layernorm", args.post_norm_eps, post)?,
            residuals: [
                residual("post_attention_residual_gate")?,
                residual("post_feedforward_residual_gate")?,
            ],
        };
        spec.validate()?;
        Ok(spec)
    }
    pub fn dspark_geometry(
        hidden: i32,
        layer: usize,
        policy: DsparkScalarPolicy,
    ) -> Result<Self, Error> {
        let root = format!("layers.{layer}");
        let residual = |name| -> Result<ResidualSpec, Error> {
            Ok(ResidualSpec::Learned(LearnedResidualSpec {
                dimensions: hidden,
                temperature: policy.temperature,
                minimum_variance: policy.minimum_variance,
                gate: ParameterSpec::trainable(format!("{root}.{name}.gate"))
                    .map_err(Error::backend)?,
            }))
        };
        let spec = Self {
            input: learned_norm(
                hidden,
                policy.epsilon,
                RmsNormEquation::OutputScaledComputeFp32(RmsGain::Centered),
                format!("{root}.self_attn.pre_qkv_proj_norm.weight"),
            )?,
            post_attention: BranchNormSpec::Weightless {
                epsilon: policy.post_epsilon,
            },
            pre_feed_forward: learned_norm(
                hidden,
                policy.epsilon,
                RmsNormEquation::OutputScaledComputeFp32(RmsGain::Centered),
                format!("{root}.pre_ffn_norm.weight"),
            )?,
            post_feed_forward: learned_norm(
                hidden,
                policy.post_epsilon,
                RmsNormEquation::InputScaledFp32(RmsGain::Centered),
                format!("{root}.post_ffn_norm.weight"),
            )?,
            residuals: [residual("add_attn")?, residual("add_ffn")?],
        };
        spec.validate()?;
        Ok(spec)
    }
    fn validate(&self) -> Result<(), Error> {
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
    pub fn norm_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.input.logical_parameters()?;
        parameters.extend(self.pre_feed_forward.logical_parameters()?);
        parameters.extend(self.post_feed_forward.logical_parameters()?);
        parameters.extend(self.post_attention.logical_parameters()?);
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for BlockScalarSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.norm_parameters()?;
        parameters.extend(self.residuals[..].logical_parameters()?);
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct DsparkEdgeSpec {
    pub projection: LinearSpec,
    pub norms: [LearnedRmsNormSpec; 2],
}
impl DsparkEdgeSpec {
    pub fn from_geometry(
        hidden: i32,
        epsilon: f32,
        captures: i32,
        format: impl Fn(&str) -> eredu_checkpoint::LinearFormat,
    ) -> Result<Self, Error> {
        let input = hidden
            .checked_mul(captures)
            .filter(|&width| captures > 0 && width > 0)
            .ok_or_else(|| {
                Error::backend("DSpark context width exceeds positive i32 dimensions")
            })?;
        let spec = Self {
            projection: LinearSpec {
                input,
                output: hidden,
                weight: ParameterSpec::trainable("fc.weight").map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    "fc.weight",
                    format("fc.weight"),
                )?,
            },
            norms: [
                learned_norm(
                    hidden,
                    epsilon,
                    RmsNormEquation::OutputScaledComputeFp32(RmsGain::Centered),
                    "hidden_norm.weight",
                )?,
                learned_norm(
                    hidden,
                    epsilon,
                    RmsNormEquation::OutputScaledComputeFp32(RmsGain::Direct),
                    "norm.weight",
                )?,
            ],
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for DsparkEdgeSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.projection.logical_parameters()?;
        parameters.extend(self.norms[..].logical_parameters()?);
        Ok(parameters)
    }
}

/// Retained causal assistant slots, independent of target invocation policy and placement.
#[derive(Debug, Clone)]
pub(crate) struct DsparkBlockParameters {
    pub attention: AttentionParameters,
    pub dense: eredu_nn::SplitGatedProductSpec,
    pub scalar: BlockScalarSpec,
}
impl DsparkBlockParameters {
    pub fn new(
        geometry: AttentionGeometry,
        intermediate: i32,
        layer: usize,
        format: impl Fn(&str) -> eredu_checkpoint::LinearFormat,
        policy: DsparkScalarPolicy,
    ) -> Result<Self, Error> {
        let root = format!("layers.{layer}");
        let parameters = Self {
            attention: AttentionParameters::from_geometry(
                geometry,
                &format!("{root}.self_attn"),
                "o_gate_proj",
                &format,
                None,
            )?,
            dense: dense_parameters(
                geometry.hidden,
                intermediate,
                &format!("{root}.feed_forward"),
                format,
            )?,
            scalar: BlockScalarSpec::dspark_geometry(geometry.hidden, layer, policy)?,
        };
        eredu_nn::validate_logical_parameter_scope(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
    /// Applies placement-selected dimensions while retaining identities, formats and equations.
    pub fn select_dimensions(
        &self,
        queries: i32,
        keys: i32,
        intermediate: i32,
    ) -> Result<Self, Error> {
        let old = self.attention.projection.geometry;
        if queries > old.query_heads()
            || keys > old.key_value_heads()
            || intermediate <= 0
            || intermediate > self.dense.gate.output
        {
            return Err(Error::backend(
                "DSpark local dimensions exceed retained parameters",
            ));
        }
        let heads = eredu_nn::AttentionHeadGeometry::new(
            queries,
            keys,
            old.key_dimensions(),
            old.value_dimensions(),
        )?;
        let mut local = self.clone();
        local.attention.projection.geometry = heads;
        local.attention.projection.query.output = heads.query_width();
        local.attention.projection.key.output = heads.key_width();
        local.attention.projection.value.output = heads.value_width();
        local.attention.projection.output.input = heads.attended_width();
        local.attention.gate.output = heads.attended_width();
        local.dense.gate.output = intermediate;
        local.dense.up.output = intermediate;
        local.dense.down.input = intermediate;
        eredu_nn::validate_logical_parameter_scope(&local.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(local)
    }
}
impl DeclareLogicalParameters for DsparkBlockParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.attention.logical_parameters()?;
        parameters.extend(self.dense.logical_parameters()?);
        parameters.extend(self.scalar.logical_parameters()?);
        Ok(parameters)
    }
}

fn project_linear(spec: &mut LinearSpec, policy: &DecoderPolicy) -> Result<(), Error> {
    let mut format = crate::linear_format::standard_linear_format(
        spec.weight.id.as_str(),
        policy.linear_format_for(spec.weight.id.as_str()),
    )?
    .with_arithmetic(spec.format.arithmetic())?;
    if spec.format.exact_zero_inputs() {
        format = format.with_exact_zero_inputs();
    }
    spec.format = format;
    Ok(())
}
impl AttentionParameters {
    pub(in crate::muse) fn project_formats(&mut self, policy: &DecoderPolicy) -> Result<(), Error> {
        for spec in [
            &mut self.projection.query,
            &mut self.projection.key,
            &mut self.projection.value,
            &mut self.projection.output,
            &mut self.gate,
        ] {
            project_linear(spec, policy)?;
        }
        self.logical_parameters().map(|_| ())
    }
    pub(in crate::muse) fn select_dimensions(
        &self,
        queries: i32,
        keys: i32,
    ) -> Result<Self, Error> {
        let source = self.projection.geometry;
        if queries > source.query_heads() || keys > source.key_value_heads() {
            return Err(Error::backend(
                "Muse selected attention heads exceed source geometry",
            ));
        }
        let heads = eredu_nn::AttentionHeadGeometry::new(
            queries,
            keys,
            source.key_dimensions(),
            source.value_dimensions(),
        )?;
        let mut local = self.clone();
        local.projection.geometry = heads;
        local.projection.query.output = heads.query_width();
        local.projection.key.output = heads.key_width();
        local.projection.value.output = heads.value_width();
        local.projection.output.input = heads.attended_width();
        local.gate.output = heads.attended_width();
        local.logical_parameters()?;
        Ok(local)
    }
}
pub(in crate::muse) fn project_dense_formats(
    spec: &mut eredu_nn::SplitGatedProductSpec,
    policy: &DecoderPolicy,
) -> Result<(), Error> {
    for spec in [&mut spec.gate, &mut spec.up, &mut spec.down] {
        project_linear(spec, policy)?;
    }
    spec.logical_parameters().map(|_| ())
}
pub(in crate::muse) fn pinned_parameters(
    args: &DecoderPolicy,
) -> Result<crate::decoder::StaticModuleConstruction, Error> {
    crate::decoder::StaticModuleSpec {
        normalization_groups: match args.vocabulary_arithmetic() {
            crate::muse::VocabularyArithmetic::Backend => None,
            _ => Some(1),
        },
        embedding_weight: "model.embed_tokens.weight".into(),
        normalization_weight: "model.norm.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: args.vocab_size,
        hidden_size: args.hidden_size,
        normalization_epsilon: args.rms_norm_eps,
        normalization_offset: 0.0,
        embedding_format: args.linear_format_for("model.embed_tokens.weight"),
        head_format: args.linear_format_for("lm_head.weight"),
        tied_head: args.tie_word_embeddings,
    }
    .construction_specs()
}
pub(in crate::muse) fn project_pinned_formats(
    source: &crate::decoder::StaticModuleConstruction,
    policy: &DecoderPolicy,
) -> Result<crate::decoder::StaticModuleConstruction, Error> {
    let mut embedding = source.embedding_spec().clone();
    let mut format = crate::linear_format::standard_linear_format(
        embedding.weight.id.as_str(),
        policy.linear_format_for(embedding.weight.id.as_str()),
    )?
    .with_arithmetic(embedding.format.arithmetic())?;
    if embedding.format.exact_zero_inputs() {
        format = format.with_exact_zero_inputs();
    }
    embedding.format = format;
    let mut head = source.head_spec().cloned();
    if let Some(spec) = &mut head {
        project_linear(spec, policy)?;
    }
    crate::decoder::StaticModuleConstruction::new(
        embedding,
        head,
        source.normalization_spec().clone(),
    )
}
