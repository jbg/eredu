//! Typed ordinary text slots and their invocation, storage, and placement projections.
use super::super::{LayerPolicy, ModelArgs};
use eredu_nn::{
    AttentionValueSource, DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration,
    LogicalParameterRole, NormalizationConstructionSpec, ParameterSpec, RotarySpec,
};
use eredu_runtime::{MemberSharding, ParameterGroupSpec, ParameterRole};

pub(crate) fn linear_spec(
    args: &ModelArgs,
    weight_name: &str,
    input: i32,
    output: i32,
    bias: bool,
) -> Result<LinearSpec, Error> {
    let prefix = weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| Error::backend("Gemma projection requires a weight identity"))?;
    Ok(LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(weight_name).map_err(Error::backend)?,
        bias: bias
            .then(|| ParameterSpec::trainable(format!("{prefix}.bias")))
            .transpose()
            .map_err(Error::backend)?,
        format: crate::linear_format::standard_linear_format(
            weight_name,
            args.linear_format_for(weight_name),
        )?,
    })
}
fn linear(
    args: &ModelArgs,
    root: &str,
    field: &str,
    input: i32,
    output: i32,
    bias: bool,
) -> Result<LinearSpec, Error> {
    linear_spec(args, &format!("{root}.{field}.weight"), input, output, bias)
}

fn norm(
    args: &ModelArgs,
    root: &str,
    field: &str,
    width: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    Ok(args.normalization(
        width,
        ParameterSpec::trainable(format!("{root}.{field}.weight")).map_err(Error::backend)?,
    ))
}
fn validate(parameters: &dyn DeclareLogicalParameters) -> Result<(), Error> {
    eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
        .map_err(Error::backend_source)
}

fn projection_parameters(
    slots: Vec<&LinearSpec>,
) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
    let mut parameters = slots
        .into_iter()
        .map(|s| s.logical_parameters())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    parameters.sort_by_key(|p| p.role == LogicalParameterRole::LinearBias);
    Ok(parameters)
}

/// Attention-owned slots, geometry, and state publication/reference semantics for one invocation.
#[derive(Debug, Clone)]
pub struct AttentionSpec {
    /// Checked query and K/V head axes, including geometry read by shared-state consumers.
    pub geometry: eredu_nn::AttentionHeadGeometry,
    /// Independent execution and mutable-state policy.
    pub policy: LayerPolicy,
    /// Query projection.
    pub query: LinearSpec,
    /// Owned key projection; absent on shared-state consumers.
    pub key: Option<LinearSpec>,
    /// Owned independent value projection; absent for shared or key-as-value state.
    pub value: Option<LinearSpec>,
    /// Output projection.
    pub output: LinearSpec,
    /// Query head gain.
    pub query_norm: NormalizationConstructionSpec,
    /// Owned key head gain.
    pub key_norm: Option<NormalizationConstructionSpec>,
    /// Invocation rotary policy, separate from parameter ownership.
    pub rotary: RotarySpec,
}
impl AttentionSpec {
    /// Declares one attention invocation under its architecture-owned layer root.
    pub(crate) fn at(args: &ModelArgs, layer: usize, layer_root: &str) -> Result<Self, Error> {
        let policy = args.layer_policy(layer).ok_or_else(|| {
            Error::backend(format!("Gemma 4 layer {layer} is outside its schedule"))
        })?;
        let head = policy.head_dim.get() as i32;
        let kv = policy.num_key_value_heads.get() as i32;
        let geometry =
            eredu_nn::AttentionHeadGeometry::new(args.num_attention_heads, kv, head, head)?;
        let root = format!("{layer_root}.{layer}.self_attn");
        let owns = policy.key_value.owns_state();
        let declaration = Self {
            geometry,
            policy,
            query: linear(
                args,
                &root,
                "q_proj",
                args.hidden_size,
                geometry.query_width(),
                args.attention_bias,
            )?,
            key: owns
                .then(|| {
                    linear(
                        args,
                        &root,
                        "k_proj",
                        args.hidden_size,
                        geometry.key_width(),
                        args.attention_bias,
                    )
                })
                .transpose()?,
            value: (policy.key_value.value() == Some(AttentionValueSource::Projected))
                .then(|| {
                    linear(
                        args,
                        &root,
                        "v_proj",
                        args.hidden_size,
                        geometry.value_width(),
                        args.attention_bias,
                    )
                })
                .transpose()?,
            output: linear(
                args,
                &root,
                "o_proj",
                geometry.attended_width(),
                args.hidden_size,
                args.attention_bias,
            )?,
            query_norm: norm(args, &root, "q_norm", head)?,
            key_norm: owns
                .then(|| norm(args, &root, "k_norm", head))
                .transpose()?,
            rotary: Self::rotary(args, policy)?,
        };
        declaration.validate_for(args, layer)?;
        declaration.validate_formats(args)?;
        Ok(declaration)
    }
    fn rotary(args: &ModelArgs, policy: LayerPolicy) -> Result<RotarySpec, Error> {
        let head = policy.head_dim.get() as i32;
        Ok(RotarySpec {
            arithmetic: args.rotary_arithmetic,
            dimensions: super::partial_rotary_dimensions(
                head,
                args.rope_scaling_for(policy.attention),
            ),
            base: args.rope_theta_for(policy.attention),
            traditional: false,
            algorithm: crate::rotary::normalize_algorithm(args.rope_scaling_for(policy.attention))
                .map_err(Error::backend)?,
        })
    }
    fn owner(&self) -> Result<(&str, usize), Error> {
        let root = self
            .query
            .weight
            .id
            .as_str()
            .strip_suffix(".q_proj.weight")
            .ok_or_else(|| Error::backend("invalid Gemma attention owner"))?;
        let layer = root
            .strip_suffix(".self_attn")
            .and_then(|root| root.rsplit_once('.'))
            .and_then(|(_, layer)| layer.parse::<usize>().ok())
            .ok_or_else(|| Error::backend("invalid Gemma attention layer"))?;
        Ok((root, layer))
    }
    fn validate_for(&self, args: &ModelArgs, layer: usize) -> Result<(), Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend("attention outside its schedule"))?;
        if self.owner()?.1 != layer
            || self.query.input != args.hidden_size
            || self.geometry.query_heads() != args.num_attention_heads
            || self.policy.attention != policy.attention
            || self.policy.key_value != policy.key_value
            || self.policy.head_dim != policy.head_dim
            || self.policy.num_key_value_heads != policy.num_key_value_heads
            || self.policy.feed_forward != policy.feed_forward
            || self.rotary != Self::rotary(args, policy)?
            || self.query_norm.epsilon != args.rms_norm_eps
            || self
                .key_norm
                .as_ref()
                .is_some_and(|norm| norm.epsilon != args.rms_norm_eps)
            || self
                .projections()
                .iter()
                .any(|spec| spec.bias.is_some() != args.attention_bias)
        {
            return Err(Error::backend(
                "Gemma attention source differs from its invocation",
            ));
        }
        validate(self)
    }
    fn validate_formats(&self, args: &ModelArgs) -> Result<(), Error> {
        let parameters = self.logical_parameters()?;
        let key = canonical_layer_name(self.query.weight.id.as_str());
        let root = key
            .strip_suffix("q_proj.weight")
            .ok_or_else(|| Error::backend("invalid attention source root"))?;
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.keys())
            .chain(args.quantized_weights.iter().flatten())
        {
            let key = canonical_layer_name(name);
            if !key.starts_with(root) {
                continue;
            }
            let parameter = parameters
                .iter()
                .find(|p| {
                    canonical_layer_name(p.parameter.id.as_str()) == key
                        && p.linear_format.is_some()
                })
                .ok_or_else(|| {
                    Error::backend(format!("undeclared Gemma attention matrix: {name}"))
                })?;
            if let Some(format) = args
                .quantized_weight_configs
                .as_ref()
                .and_then(|formats| formats.get(name))
            {
                if parameter.linear_format.expect("declared matrix").encoding() != (*format).into()
                {
                    return Err(Error::backend(format!(
                        "conflicting Gemma attention encoding alias: {name}"
                    )));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn validate_scopes(args: &ModelArgs, scopes: &[Self]) -> Result<(), Error> {
        if scopes.len() != args.num_hidden_layers() {
            return Err(Error::backend("attention count differs from its schedule"));
        }
        for (layer, spec) in scopes.iter().enumerate() {
            spec.validate_for(args, layer)?;
            let owner = args
                .attention_state_owner(layer)
                .map_err(Error::backend_source)?;
            let publisher = scopes
                .get(owner)
                .ok_or_else(|| Error::backend("missing attention state publisher"))?;
            if publisher.key.is_none()
                || publisher.geometry.key_value_heads() != spec.geometry.key_value_heads()
                || publisher.geometry.key_dimensions() != spec.geometry.key_dimensions()
                || publisher.geometry.value_dimensions() != spec.geometry.value_dimensions()
                || publisher.rotary != spec.rotary
            {
                return Err(Error::backend(
                    "shared attention publisher has incompatible retained geometry",
                ));
            }
        }
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.keys())
            .chain(args.quantized_weights.iter().flatten())
        {
            let key = canonical_layer_name(name);
            if let Some((layer, field)) = key
                .strip_prefix("model.layers.")
                .and_then(|local| local.split_once('.'))
            {
                if field.starts_with("self_attn.")
                    && layer
                        .parse::<usize>()
                        .ok()
                        .and_then(|layer| scopes.get(layer))
                        .is_none()
                {
                    return Err(Error::backend(format!(
                        "attention encoding references an undeclared layer: {name}"
                    )));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn in_namespace(&self, layer: usize, root: &str) -> Result<Self, Error> {
        let (source, owner) = self.owner()?;
        if owner != layer {
            return Err(Error::backend(
                "attention namespace changes layer ownership",
            ));
        }
        let target = format!("{root}.{layer}.self_attn");
        let namespace = ParameterNamespace {
            source,
            target: &target,
        };
        let mut projected = self.clone();
        projected.query = namespace.linear(&self.query)?;
        projected.output = namespace.linear(&self.output)?;
        projected.key = self.key.as_ref().map(|s| namespace.linear(s)).transpose()?;
        projected.value = self
            .value
            .as_ref()
            .map(|s| namespace.linear(s))
            .transpose()?;
        projected.query_norm = namespace.normalization(&self.query_norm)?;
        projected.key_norm = self
            .key_norm
            .as_ref()
            .map(|s| namespace.normalization(s))
            .transpose()?;
        validate(&projected)?;
        Ok(projected)
    }
    /// Projects only selected head axes; parameter and state ownership remain distinct.
    pub(crate) fn for_invocation(&self, args: &ModelArgs, layer: usize) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend("attention outside local schedule"))?;
        if self.owner()?.1 != layer
            || policy.head_dim != self.policy.head_dim
            || policy.attention != self.policy.attention
            || policy.key_value != self.policy.key_value
            || policy.feed_forward != self.policy.feed_forward
            || args.num_attention_heads > self.geometry.query_heads()
            || policy.num_key_value_heads.get() as i32 > self.geometry.key_value_heads()
        {
            return Err(Error::backend(
                "selected attention heads or ownership exceed their source",
            ));
        }
        let mut projected = self.clone();
        projected.geometry = eredu_nn::AttentionHeadGeometry::new(
            args.num_attention_heads,
            policy.num_key_value_heads.get() as i32,
            self.geometry.key_dimensions(),
            self.geometry.value_dimensions(),
        )?;
        projected.policy = policy;
        projected.query.output = projected.geometry.query_width();
        projected.output.input = projected.geometry.attended_width();
        if let Some(key) = &mut projected.key {
            key.output = projected.geometry.key_width();
        }
        if let Some(value) = &mut projected.value {
            value.output = projected.geometry.value_width();
        }
        projected.validate_for(args, layer)?;
        projected.parameter_groups("gemma.local.attention")?;
        Ok(projected)
    }
    pub(crate) fn project_formats(&self, args: &ModelArgs) -> Result<Self, Error> {
        let mut projected = self.clone();
        for linear in [&mut projected.query, &mut projected.output]
            .into_iter()
            .chain(projected.key.iter_mut())
            .chain(projected.value.iter_mut())
        {
            project_linear_format(linear, args)?;
        }
        projected.validate_for(args, self.owner()?.1)?;
        projected.validate_formats(args)?;
        Ok(projected)
    }
    pub(crate) fn same_formats(&self, other: &Self) -> bool {
        self.projections()
            .iter()
            .zip(other.projections())
            .all(|(s, t)| s.format == t.format)
    }
    /// Projects cache ownership and geometry; fixed invocation state remains explicit.
    pub fn state_policy(
        &self,
        range: Option<std::ops::Range<usize>>,
        tensors: Vec<eredu_core::cache::StateTensorPolicy>,
    ) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        use eredu_core::cache::LayerCachePolicy;
        validate(self)?;
        // One projection can feed distinct normalized key and value cache payloads.
        let heads = match range {
            Some(range) => self.geometry.partition_key_value_heads(range)?,
            None => self.geometry,
        };
        if !self.policy.key_value.owns_state() {
            if !tensors.is_empty() {
                return Err(Error::backend(
                    "shared Gemma attention cannot own fixed cache components",
                ));
            }
            return Ok(LayerCachePolicy::NoState);
        }
        if tensors.is_empty() {
            LayerCachePolicy::key_value(
                self.policy.attention,
                heads.key_value_heads(),
                heads.key_dimensions(),
            )
        } else {
            LayerCachePolicy::key_value_with_state(
                self.policy.attention,
                heads.key_value_heads(),
                heads.key_dimensions(),
                tensors,
                Vec::new(),
            )
        }
        .map_err(Error::backend_source)
    }

    /// Owned matrices in attention-equation order; shared state contributes no local K/V slots.
    pub fn projections(&self) -> Vec<&LinearSpec> {
        let mut slots = vec![&self.query];
        slots.extend(self.key.as_ref());
        slots.extend(self.value.as_ref());
        slots.push(&self.output);
        slots
    }
    pub(crate) fn parameter_groups(&self, root: &str) -> Result<Vec<ParameterGroupSpec>, Error> {
        validate(self)?;
        let query = projection_parameters(vec![&self.query, &self.output])?;
        let mut groups = vec![eredu_runtime::logical_parameter_group(
            format!("{root}.query_heads"),
            ParameterRole::AttentionHeads,
            Some(self.geometry.query_heads() as usize),
            &query,
            |p| {
                if p.parameter.id == self.query.weight.id
                    || self
                        .query
                        .bias
                        .as_ref()
                        .is_some_and(|b| p.parameter.id == b.id)
                {
                    MemberSharding::Partitioned { axis: 0 }
                } else if p.parameter.id == self.output.weight.id {
                    MemberSharding::Partitioned { axis: 1 }
                } else {
                    MemberSharding::Replicated
                }
            },
        )
        .map_err(Error::backend_source)?];
        if let Some(key) = &self.key {
            let mut slots = vec![key];
            slots.extend(self.value.as_ref());
            let kv = projection_parameters(slots)?;
            groups.push(
                eredu_runtime::logical_parameter_group(
                    format!("{root}.key_value_heads"),
                    ParameterRole::AttentionHeads,
                    Some(self.geometry.key_value_heads() as usize),
                    &kv,
                    |_| MemberSharding::Partitioned { axis: 0 },
                )
                .map_err(Error::backend_source)?,
            );
        }
        let mut norms = self.query_norm.logical_parameters()?;
        if let Some(norm) = &self.key_norm {
            norms.extend(norm.logical_parameters()?)
        }
        groups.push(
            eredu_runtime::logical_parameter_group(
                format!("{root}.head_norms"),
                ParameterRole::Replicated,
                None,
                &norms,
                |_| MemberSharding::Replicated,
            )
            .map_err(Error::backend_source)?,
        );
        Ok(groups)
    }
}
impl DeclareLogicalParameters for AttentionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let owned = self.policy.key_value.owns_state();
        let projected_value =
            self.policy.key_value.value() == Some(AttentionValueSource::Projected);
        let heads = self.geometry;
        if heads.key_dimensions() != heads.value_dimensions()
            || heads.key_dimensions() as u32 != self.policy.head_dim.get()
            || heads.key_value_heads() as u32 != self.policy.num_key_value_heads.get()
            || self.key.is_some() != owned
            || self.key_norm.is_some() != owned
            || self.value.is_some() != projected_value
            || self.query.output != heads.query_width()
            || self.output.input != heads.attended_width()
            || self.output.output != self.query.input
            || self.query_norm.dimensions != heads.key_dimensions()
            || self
                .key_norm
                .as_ref()
                .is_some_and(|norm| norm.dimensions != heads.key_dimensions())
            || self
                .key
                .as_ref()
                .is_some_and(|key| key.input != self.query.input || key.output != heads.key_width())
            || self.value.as_ref().is_some_and(|value| {
                value.input != self.query.input || value.output != heads.value_width()
            })
            || self.rotary.dimensions <= 0
            || self.rotary.dimensions > heads.key_dimensions()
        {
            return Err(Error::backend(
                "Gemma attention ownership, head, residual, normalization or rotary geometry mismatch",
            ));
        }
        let mut p = Vec::new();
        for slot in self.projections() {
            p.extend(slot.logical_parameters()?)
        }
        p.extend(self.query_norm.logical_parameters()?);
        if let Some(norm) = &self.key_norm {
            p.extend(norm.logical_parameters()?)
        }
        Ok(p)
    }
}

pub(super) fn canonical_layer_name(name: &str) -> String {
    let key = crate::gemma4::config::canonical_weight_name(name);
    key.strip_prefix("model.decoder.layers.")
        .map_or_else(|| key.clone(), |local| format!("model.layers.{local}"))
}

/// Named storage requests borrow policy; geometry and eligibility come from typed slots.
pub(crate) fn format_references<'a>(
    default: Option<eredu_checkpoint::WeightQuantization>,
    selected: Option<&'a std::collections::HashSet<String>>,
    formats: Option<&'a std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>>,
) -> impl Iterator<Item = (&'a str, Option<eredu_checkpoint::LinearFormat>)> {
    formats
        .into_iter()
        .flat_map(|formats| formats.iter())
        .map(|(name, format)| (name.as_str(), Some((*format).into())))
        .chain(
            selected
                .into_iter()
                .flat_map(|names| names.iter())
                .map(move |name| {
                    (
                        name.as_str(),
                        formats
                            .and_then(|f| f.get(name))
                            .copied()
                            .or(default)
                            .map(Into::into),
                    )
                }),
        )
}

/// Resolves complete family references, including explicit split expert sources.
pub(crate) fn validate_format_references<'a>(
    parameters: &[LogicalParameterDeclaration<'_>],
    routed: &[Option<super::routed::Declaration>],
    references: impl IntoIterator<Item = (&'a str, Option<eredu_checkpoint::LinearFormat>)>,
) -> Result<(), Error> {
    let canonical = |name: &str| {
        let name = canonical_layer_name(name);
        if name.contains(".experts.") && !name.contains(".experts.switch_glu.") {
            name.replace(".experts.", ".experts.switch_glu.")
        } else {
            name
        }
    };
    let split = routed
        .iter()
        .flatten()
        .map(|s| Ok(s.split_source_names()?.into_iter().zip(s.split_formats)))
        .collect::<Result<Vec<_>, Error>>()?
        .into_iter()
        .flatten()
        .map(|(name, format)| (canonical(&name), format))
        .collect::<std::collections::BTreeMap<_, _>>();
    for (name, requested) in references {
        let target = canonical(name);
        let declared = split
            .get(&target)
            .copied()
            .or_else(|| {
                parameters.iter().find_map(|p| {
                    (canonical(p.parameter.id.as_str()) == target)
                        .then_some(p.linear_format)
                        .flatten()
                        .map(|format| format.encoding())
                })
            })
            .ok_or_else(|| {
                Error::backend(format!(
                    "storage extension references an undeclared matrix: {name}"
                ))
            })?;
        if let Some(format) = requested {
            format.validate().map_err(Error::backend_source)?;
            if declared != format {
                return Err(Error::backend(format!(
                    "storage extension conflicts with retained encoding: {name}"
                )));
            }
        }
    }
    Ok(())
}

pub(super) struct ParameterNamespace<'a> {
    pub(super) source: &'a str,
    pub(super) target: &'a str,
}
impl ParameterNamespace<'_> {
    fn id(&self, id: &eredu_nn::ParameterId) -> Result<eredu_nn::ParameterId, Error> {
        let local = id
            .as_str()
            .strip_prefix(&format!("{}.", self.source))
            .ok_or_else(|| {
                Error::backend("Gemma parameter reference outside its declared owner")
            })?;
        eredu_nn::ParameterId::new(format!("{}.{local}", self.target))
            .map_err(Error::backend_source)
    }
    pub(super) fn parameter(&self, source: &ParameterSpec) -> Result<ParameterSpec, Error> {
        let mut projected = source.clone();
        projected.id = self.id(&source.id)?;
        projected.alias_of = source.alias_of.as_ref().map(|id| self.id(id)).transpose()?;
        projected.linear_companion_of = source
            .linear_companion_of
            .as_ref()
            .map(|id| self.id(id))
            .transpose()?;
        if let Some(group) = &source.group {
            if let Some(local) = group.strip_prefix(&format!("{}.", self.source)) {
                projected.group = Some(format!("{}.{local}", self.target));
            }
        }
        Ok(projected)
    }
    pub(super) fn normalization(
        &self,
        source: &NormalizationConstructionSpec,
    ) -> Result<NormalizationConstructionSpec, Error> {
        let mut projected = source.clone();
        match &mut projected.scale {
            eredu_nn::NormalizationScale::Unit => {}
            eredu_nn::NormalizationScale::Learned(weight)
            | eredu_nn::NormalizationScale::LearnedOffset { weight, .. } => {
                *weight = self.parameter(weight)?
            }
        }
        Ok(projected)
    }
    fn linear(&self, source: &LinearSpec) -> Result<LinearSpec, Error> {
        let mut projected = source.clone();
        projected.weight = self.parameter(&source.weight)?;
        projected.bias = source
            .bias
            .as_ref()
            .map(|p| self.parameter(p))
            .transpose()?;
        projected.format = source.format.map_parameters(|p| self.parameter(p))?;
        Ok(projected)
    }
}
fn project_linear_format(linear: &mut LinearSpec, args: &ModelArgs) -> Result<(), Error> {
    let old = &linear.format;
    let mut format = crate::linear_format::standard_linear_format(
        linear.weight.id.as_str(),
        args.linear_format_for(linear.weight.id.as_str()),
    )?
    .with_arithmetic(old.arithmetic())?;
    if old.exact_zero_inputs() {
        format = format.with_exact_zero_inputs();
    }
    linear.format = format;
    Ok(())
}

/// Dense gate/value/contraction slots and their exact activation equation.
#[derive(Debug, Clone)]
pub struct DenseSpec {
    /// Complete gate/value/contraction declaration and resolved GELU equation.
    pub product: eredu_nn::SplitGatedProductSpec,
}
impl DenseSpec {
    /// Declares one dense branch under its invocation's parameter root.
    pub(crate) fn at(args: &ModelArgs, layer: usize, layer_root: &str) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend("Gemma dense invocation is outside its schedule"))?;
        let width = policy.intermediate_size.get() as i32;
        let root = format!("{layer_root}.{layer}.mlp");
        let spec = Self {
            product: eredu_nn::SplitGatedProductSpec::new(
                linear(args, &root, "gate_proj", args.hidden_size, width, false)?,
                linear(args, &root, "up_proj", args.hidden_size, width, false)?,
                linear(args, &root, "down_proj", width, args.hidden_size, false)?,
                if args.approximate_gelu {
                    eredu_nn::GatedProductPolicy::ordinary_gelu_approximate()
                } else {
                    eredu_nn::GatedProductPolicy::ordinary_gelu()
                },
            )?,
        };
        spec.validate_for(args, layer)?;
        spec.validate_formats(args)?;
        Ok(spec)
    }
    fn owner(&self) -> Result<(&str, usize), Error> {
        let root = self
            .product
            .gate
            .weight
            .id
            .as_str()
            .strip_suffix(".gate_proj.weight")
            .ok_or_else(|| Error::backend("invalid Gemma dense owner"))?;
        let layer = root
            .strip_suffix(".mlp")
            .and_then(|root| root.rsplit_once('.'))
            .and_then(|(_, layer)| layer.parse::<usize>().ok())
            .ok_or_else(|| Error::backend("invalid Gemma dense layer owner"))?;
        Ok((root, layer))
    }
    fn validate_for(&self, args: &ModelArgs, layer: usize) -> Result<(), Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend("dense layer outside its schedule"))?;
        let expected = if args.approximate_gelu {
            eredu_nn::GatedProductPolicy::ordinary_gelu_approximate()
        } else {
            eredu_nn::GatedProductPolicy::ordinary_gelu()
        };
        if self.owner()?.1 != layer
            || self.product.gate.input != args.hidden_size
            || self.product.gate.output as u32 != policy.intermediate_size.get()
            || self.product.down.output != args.hidden_size
            || self.product.policy != expected
        {
            return Err(Error::backend(
                "Gemma dense source differs from invocation geometry or equation",
            ));
        }
        validate(self)
    }
    /// Resolves storage extensions against the retained dense matrix identities.
    pub(crate) fn validate_formats(&self, args: &ModelArgs) -> Result<(), Error> {
        let parameters = self.logical_parameters()?;
        let root = canonical_layer_name(self.product.gate.weight.id.as_str());
        let root = root
            .strip_suffix("gate_proj.weight")
            .ok_or_else(|| Error::backend("invalid Gemma dense root"))?;
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.keys())
            .chain(args.quantized_weights.iter().flatten())
        {
            let key = canonical_layer_name(name);
            if !key.starts_with(root) {
                continue;
            }
            let parameter = parameters
                .iter()
                .find(|p| {
                    canonical_layer_name(p.parameter.id.as_str()) == key
                        && p.linear_format.is_some()
                })
                .ok_or_else(|| Error::backend(format!("undeclared Gemma dense matrix: {name}")))?;
            if let Some(selected) = args
                .quantized_weight_configs
                .as_ref()
                .and_then(|formats| formats.get(name))
            {
                if parameter.linear_format.expect("declared matrix").encoding()
                    != (*selected).into()
                {
                    return Err(Error::backend(format!(
                        "conflicting Gemma dense encoding alias: {name}"
                    )));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn validate_scopes(args: &ModelArgs, scopes: &[Self]) -> Result<(), Error> {
        if scopes.len() != args.num_hidden_layers() {
            return Err(Error::backend(
                "dense declaration count differs from its schedule",
            ));
        }
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.keys())
            .chain(args.quantized_weights.iter().flatten())
        {
            let key = canonical_layer_name(name);
            if let Some((layer, field)) = key
                .strip_prefix("model.layers.")
                .and_then(|local| local.split_once('.'))
            {
                if field.starts_with("mlp.")
                    && layer
                        .parse::<usize>()
                        .ok()
                        .and_then(|layer| scopes.get(layer))
                        .is_none()
                {
                    return Err(Error::backend(format!(
                        "dense encoding references an undeclared layer: {name}"
                    )));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn in_namespace(&self, layer: usize, root: &str) -> Result<Self, Error> {
        let (source, owner) = self.owner()?;
        if layer != owner {
            return Err(Error::backend(
                "Gemma dense namespace changes layer ownership",
            ));
        }
        let target = format!("{root}.{layer}.mlp");
        let namespace = ParameterNamespace {
            source,
            target: &target,
        };
        let mut projected = self.clone();
        projected.product.gate = namespace.linear(&self.product.gate)?;
        projected.product.up = namespace.linear(&self.product.up)?;
        projected.product.down = namespace.linear(&self.product.down)?;
        validate(&projected)?;
        Ok(projected)
    }
    /// Reduces an intermediate axis while preserving source parameter and operator facts.
    pub(crate) fn with_intermediate(mut self, width: i32) -> Result<Self, Error> {
        if width <= 0 || width > self.product.gate.output {
            return Err(Error::backend("local Gemma dense width exceeds its source"));
        }
        self.product.gate.output = width;
        self.product.up.output = width;
        self.product.down.input = width;
        validate(&self)?;
        self.parameter_group("gemma.local.dense")?;
        Ok(self)
    }
    pub(crate) fn project_formats(&self, args: &ModelArgs) -> Result<Self, Error> {
        let mut projected = self.clone();
        for linear in [
            &mut projected.product.gate,
            &mut projected.product.up,
            &mut projected.product.down,
        ] {
            project_linear_format(linear, args)?;
        }
        projected.validate_for(args, self.owner()?.1)?;
        projected.validate_formats(args)?;
        Ok(projected)
    }
    pub(crate) fn same_formats(&self, other: &Self) -> bool {
        self.product
            .projections()
            .iter()
            .zip(other.product.projections())
            .all(|(s, t)| s.format == t.format)
    }
    pub(crate) fn parameter_group(&self, root: &str) -> Result<ParameterGroupSpec, Error> {
        eredu_runtime::logical_parameter_group(
            format!("{root}.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            Some(self.product.gate.output as usize),
            self,
            |p| {
                if p.parameter.id == self.product.down.weight.id {
                    MemberSharding::Partitioned { axis: 1 }
                } else {
                    MemberSharding::Partitioned { axis: 0 }
                }
            },
        )
        .map_err(Error::backend_source)
    }
}
impl DeclareLogicalParameters for DenseSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.product.gate.input != self.product.down.output
            || (self.product.policy != eredu_nn::GatedProductPolicy::ordinary_gelu()
                && self.product.policy != eredu_nn::GatedProductPolicy::ordinary_gelu_approximate())
        {
            return Err(Error::backend(
                "incompatible Gemma dense residual geometry or GELU policy",
            ));
        }
        self.product.logical_parameters()
    }
}

/// One learned multiplier applied to the complete residual stream.
#[derive(Debug, Clone)]
pub struct ResidualScaleSpec {
    /// Authoritative trainable scale identity, including ownership/sharing metadata.
    pub weight: ParameterSpec,
}
impl ResidualScaleSpec {
    /// Published one-element tensor geometry, shared with unloaded construction.
    pub const SHAPE: [i32; 1] = [1];
    /// Declares an independently owned residual-stream multiplier.
    pub fn new(name: impl Into<String>) -> Result<Self, Error> {
        Ok(Self {
            weight: ParameterSpec::trainable(name.into()).map_err(Error::backend)?,
        })
    }
    /// Constructs the declared floating parameter without duplicating its geometry.
    pub fn unloaded<T: eredu_nn::Tensor>(
        &self,
        context: &T::Context,
    ) -> Result<eredu_nn::Parameter<T>, Error> {
        eredu_nn::Parameter::unloaded(self.weight.clone(), &Self::SHAPE, context)
    }
}
impl DeclareLogicalParameters for ResidualScaleSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.weight,
            &Self::SHAPE,
            LogicalParameterRole::ResidualScale,
        )?])
    }
}

/// Per-layer input contribution, gated by the current decoder residual stream.
#[derive(Debug, Clone)]
pub struct PerLayerSpec {
    /// Hidden-to-input-width gate projection.
    pub gate: LinearSpec,
    /// Input-width-to-hidden contraction after applying the gate.
    pub projection: LinearSpec,
    /// Complete contribution normalization before residual addition.
    pub norm: NormalizationConstructionSpec,
}
impl DeclareLogicalParameters for PerLayerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.gate.logical_parameters()?;
        p.extend(self.projection.logical_parameters()?);
        p.extend(self.norm.logical_parameters()?);
        Ok(p)
    }
}
impl PerLayerSpec {
    pub(crate) fn parameter_group(&self, root: &str) -> Result<ParameterGroupSpec, Error> {
        let mut parameters = self.gate.logical_parameters()?;
        parameters.extend(self.projection.logical_parameters()?);
        eredu_runtime::logical_parameter_group(
            format!("{root}.media_channels"),
            ParameterRole::Channels,
            Some(self.gate.output as usize),
            &parameters,
            |p| {
                if p.parameter.id == self.gate.weight.id {
                    MemberSharding::Partitioned { axis: 0 }
                } else {
                    MemberSharding::Partitioned { axis: 1 }
                }
            },
        )
        .map_err(Error::backend_source)
    }
}

/// Block normalization, independent invocation scale, and optional per-layer input slots.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    /// Attention input gain.
    pub input: NormalizationConstructionSpec,
    /// Attention write gain.
    pub post_attention: NormalizationConstructionSpec,
    /// Dense feed-forward input gain.
    pub pre_feed_forward: NormalizationConstructionSpec,
    /// Complete feed-forward write gain.
    pub post_feed_forward: NormalizationConstructionSpec,
    /// Residual stream multiplier after sublayer and per-layer contributions.
    pub residual_scale: ResidualScaleSpec,
    /// Optional per-layer input equation, with its own placement semantics.
    pub per_layer: Option<PerLayerSpec>,
}
impl BlockSpec {
    /// Declares one block's scalar and per-layer-input equation under its parameter root.
    pub(crate) fn at(args: &ModelArgs, layer: usize, layer_root: &str) -> Result<Self, Error> {
        args.layer_policy(layer)
            .ok_or_else(|| Error::backend("Gemma block invocation is outside its schedule"))?;
        let root = format!("{layer_root}.{layer}");
        let per_layer = if args.hidden_size_per_layer_input > 0 {
            Some(PerLayerSpec {
                gate: linear(
                    args,
                    &root,
                    "per_layer_input_gate",
                    args.hidden_size,
                    args.hidden_size_per_layer_input,
                    false,
                )?,
                projection: linear(
                    args,
                    &root,
                    "per_layer_projection",
                    args.hidden_size_per_layer_input,
                    args.hidden_size,
                    false,
                )?,
                norm: norm(args, &root, "post_per_layer_input_norm", args.hidden_size)?,
            })
        } else {
            None
        };
        let spec = Self {
            input: norm(args, &root, "input_layernorm", args.hidden_size)?,
            post_attention: norm(args, &root, "post_attention_layernorm", args.hidden_size)?,
            pre_feed_forward: norm(args, &root, "pre_feedforward_layernorm", args.hidden_size)?,
            post_feed_forward: norm(args, &root, "post_feedforward_layernorm", args.hidden_size)?,
            residual_scale: ResidualScaleSpec::new(format!("{root}.layer_scalar"))?,
            per_layer,
        };
        validate(&spec)?;
        spec.validate_formats(args)?;
        Ok(spec)
    }
    fn validate_for(&self, args: &ModelArgs, layer: usize) -> Result<(), Error> {
        let owner = self
            .residual_scale
            .weight
            .id
            .as_str()
            .strip_suffix(".layer_scalar")
            .and_then(|root| root.rsplit_once('.'))
            .and_then(|(_, index)| index.parse::<usize>().ok());
        if owner != Some(layer) {
            return Err(Error::backend("Gemma block references another layer owner"));
        }
        for norm in [
            &self.input,
            &self.post_attention,
            &self.pre_feed_forward,
            &self.post_feed_forward,
        ]
        .into_iter()
        .chain(self.per_layer.iter().map(|spec| &spec.norm))
        {
            if norm.dimensions != args.hidden_size || norm.epsilon != args.rms_norm_eps {
                return Err(Error::backend(
                    "Gemma block gain differs from retained residual geometry",
                ));
            }
        }
        match &self.per_layer {
            None if args.hidden_size_per_layer_input == 0 => {}
            Some(spec)
                if spec.gate.input == args.hidden_size
                    && spec.gate.output == args.hidden_size_per_layer_input
                    && spec.projection.input == args.hidden_size_per_layer_input
                    && spec.projection.output == args.hidden_size => {}
            _ => {
                return Err(Error::backend(
                    "Gemma layer-input source geometry differs from its invocation",
                ));
            }
        }
        validate(self)
    }
    /// Checks named storage extensions against this retained block's matrix owners.
    pub(crate) fn validate_formats(&self, args: &ModelArgs) -> Result<(), Error> {
        let parameters = self.logical_parameters()?;
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.keys())
            .chain(args.quantized_weights.iter().flatten())
        {
            let canonical = canonical_layer_name(name);
            let parameter = parameters
                .iter()
                .find(|p| canonical_layer_name(p.parameter.id.as_str()) == canonical);
            let owner = canonical_layer_name(self.residual_scale.weight.id.as_str());
            let root = owner
                .strip_suffix("layer_scalar.weight")
                .ok_or_else(|| Error::backend("invalid block namespace"))?;
            let known_field = parameters.iter().any(|p| {
                let key = canonical_layer_name(p.parameter.id.as_str());
                let field = key.strip_suffix(".weight").unwrap_or(&key);
                canonical.starts_with(&format!("{field}."))
            });
            let block_extension = known_field
                || canonical.strip_prefix(root).is_some_and(|local| {
                    local.starts_with("per_layer_")
                        || local.starts_with("post_per_layer_input_norm.")
                });
            if block_extension && parameter.is_none() {
                return Err(Error::backend(format!(
                    "undeclared Gemma block format reference: {name}"
                )));
            }
            if let Some(parameter) = parameter {
                let format = parameter.linear_format.ok_or_else(|| {
                    Error::backend(format!("Gemma block scalar has no matrix format: {name}"))
                })?;
                if let Some(selected) = args
                    .quantized_weight_configs
                    .as_ref()
                    .and_then(|f| f.get(name))
                {
                    if format.encoding() != (*selected).into() {
                        return Err(Error::backend(format!(
                            "conflicting Gemma block format alias: {name}"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
    /// Source alias projection preserves geometry, metadata and equations.
    pub(crate) fn in_namespace(&self, layer: usize, root: &str) -> Result<Self, Error> {
        let source = self
            .residual_scale
            .weight
            .id
            .as_str()
            .strip_suffix(".layer_scalar")
            .ok_or_else(|| Error::backend("unresolved Gemma block owner"))?;
        if source
            .rsplit_once('.')
            .and_then(|(_, index)| index.parse::<usize>().ok())
            != Some(layer)
        {
            return Err(Error::backend(
                "Gemma namespace projection changes layer ownership",
            ));
        }
        let target = format!("{root}.{layer}");
        let namespace = ParameterNamespace {
            source,
            target: &target,
        };
        let projected = Self {
            input: namespace.normalization(&self.input)?,
            post_attention: namespace.normalization(&self.post_attention)?,
            pre_feed_forward: namespace.normalization(&self.pre_feed_forward)?,
            post_feed_forward: namespace.normalization(&self.post_feed_forward)?,
            residual_scale: ResidualScaleSpec {
                weight: namespace.parameter(&self.residual_scale.weight)?,
            },
            per_layer: self
                .per_layer
                .as_ref()
                .map(|spec| -> Result<_, Error> {
                    Ok(PerLayerSpec {
                        gate: namespace.linear(&spec.gate)?,
                        projection: namespace.linear(&spec.projection)?,
                        norm: namespace.normalization(&spec.norm)?,
                    })
                })
                .transpose()?,
        };
        validate(&projected)?;
        Ok(projected)
    }
    /// Selects a bounded local input-channel width without changing source owners.
    pub(crate) fn with_input_width(mut self, width: i32) -> Result<Self, Error> {
        match &mut self.per_layer {
            Some(spec) if width > 0 && width <= spec.gate.output => {
                spec.gate.output = width;
                spec.projection.input = width;
            }
            None if width == 0 => {}
            _ => {
                return Err(Error::backend(
                    "local Gemma layer-input width exceeds its source",
                ));
            }
        }
        validate(&self)?;
        if let Some(spec) = &self.per_layer {
            spec.parameter_group("gemma.local.layer_input")?;
        }
        Ok(self)
    }
    pub(crate) fn same_formats(&self, other: &Self) -> bool {
        match (&self.per_layer, &other.per_layer) {
            (None, None) => true,
            (Some(source), Some(target)) => {
                source.gate.format == target.gate.format
                    && source.projection.format == target.projection.format
            }
            _ => false,
        }
    }
    /// Projects selected matrix storage onto immutable block geometry and equations.
    pub(crate) fn project_formats(&self, args: &ModelArgs) -> Result<Self, Error> {
        let mut projected = self.clone();
        if let Some(spec) = &mut projected.per_layer {
            for linear in [&mut spec.gate, &mut spec.projection] {
                project_linear_format(linear, args)?;
            }
        }
        let layer = self
            .residual_scale
            .weight
            .id
            .as_str()
            .strip_suffix(".layer_scalar")
            .and_then(|root| root.rsplit_once('.'))
            .and_then(|(_, index)| index.parse::<usize>().ok())
            .ok_or_else(|| Error::backend("invalid Gemma block owner"))?;
        projected.validate_for(args, layer)?;
        projected.validate_formats(args)?;
        Ok(projected)
    }
    pub(crate) fn replicated_parameters(
        &self,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = Vec::new();
        for spec in [
            &self.input,
            &self.post_attention,
            &self.pre_feed_forward,
            &self.post_feed_forward,
        ] {
            p.extend(spec.logical_parameters()?)
        }
        p.extend(self.residual_scale.logical_parameters()?);
        Ok(p)
    }
}
impl DeclareLogicalParameters for BlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.replicated_parameters()?;
        if let Some(spec) = &self.per_layer {
            p.extend(spec.logical_parameters()?)
        }
        Ok(p)
    }
}

/// Complete typed parameter and operator definition for one decoder invocation.
#[derive(Debug, Clone)]
pub(crate) struct DecoderBlockSpec {
    pub attention: AttentionSpec,
    pub attention_arithmetic: eredu_nn::AttentionArithmetic,
    pub dense: DenseSpec,
    pub block: BlockSpec,
    pub routed: Option<super::routed::Declaration>,
}
impl DecoderBlockSpec {
    pub fn from_scopes(
        args: &ModelArgs,
        layer: usize,
        block: BlockSpec,
        dense: DenseSpec,
        attention: AttentionSpec,
        routed: Option<super::routed::Declaration>,
    ) -> Result<Self, Error> {
        dense.validate_for(args, layer)?;
        block.validate_for(args, layer)?;
        attention.validate_for(args, layer)?;
        if routed.is_some()
            != (attention.policy.feed_forward
                == super::super::FeedForwardPolicy::DenseWithSparseMoe)
        {
            return Err(Error::backend(
                "routed scope differs from invocation ownership",
            ));
        }
        if let Some(scope) = &routed {
            scope.validate_invocation(args, layer)?;
        }
        Ok(Self {
            attention,
            attention_arithmetic: args.attention_arithmetic,
            dense,
            block,
            routed,
        })
    }
}
impl DeclareLogicalParameters for DecoderBlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.attention.logical_parameters()?;
        parameters.extend(self.dense.logical_parameters()?);
        parameters.extend(self.block.logical_parameters()?);
        if let Some(routed) = &self.routed {
            parameters.extend(routed.logical_parameters()?);
        }
        Ok(parameters)
    }
}
