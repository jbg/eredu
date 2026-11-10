//! Sparse-branch construction and logical ownership for Gemma 4 invocations.
use super::super::{FeedForwardPolicy, ModelArgs};
use eredu_nn::{
    DeclareLogicalParameters, Error, GatedProductGroupLayout, GroupedGatedProductSpec,
    LogicalParameterDeclaration, NormalizationConstructionSpec, ParameterSpec,
    SelectorInputTransformSpec, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};

/// Authoritative selector, expert bank, and dense/sparse branch normalization slots.
#[derive(Debug, Clone)]
pub struct Declaration {
    /// Global selector and independently localized expert-bank parameters.
    pub routed: eredu_nn::RoutedGatedProductSpec,
    /// Dense output, sparse input, and sparse output normalization, in equation order.
    pub norms: [NormalizationConstructionSpec; 3],
    /// Physical split gate/up encodings; geometry and targets reference the fused bank above.
    pub split_formats: [eredu_checkpoint::LinearFormat; 2],
}
impl Declaration {
    const BRANCH_GAINS: [&'static str; 3] = [
        "post_feedforward_layernorm_1",
        "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm_2",
    ];
    fn branch_gain(field: &str) -> bool {
        Self::BRANCH_GAINS
            .iter()
            .any(|name| field.starts_with(&format!("{name}.")))
    }
    /// Declares the same sparse equation under an architecture-owned layer root.
    pub(crate) fn at(args: &ModelArgs, layer: usize, layer_root: &str) -> Result<Self, Error> {
        if args
            .layer_policy(layer)
            .is_none_or(|p| p.feed_forward != FeedForwardPolicy::DenseWithSparseMoe)
        {
            return Err(Error::backend(format!(
                "Gemma 4 layer {layer} has no sparse invocation"
            )));
        }
        let experts = args
            .num_experts
            .ok_or_else(|| Error::backend("Gemma 4 sparse layer has no expert count"))?;
        let width = args
            .moe_intermediate_size
            .ok_or_else(|| Error::backend("Gemma 4 sparse layer has no expert width"))?;
        let top_k = args
            .top_k_experts
            .ok_or_else(|| Error::backend("Gemma 4 sparse layer has no selection width"))?;
        let root = format!("{layer_root}.{layer}");
        let weight = format!("{root}.router.proj.weight");
        let selector = TopKGroupSelectorSpec::new(
            args.hidden_size,
            ParameterSpec::trainable(&weight).map_err(Error::backend)?,
            crate::linear_format::standard_linear_format(&weight, args.linear_format_for(&weight))?,
            TopKGroupSelectionSpec::new(
                experts,
                top_k,
                args.routing_scoring,
                args.normalize_selected_routes,
            )?,
        )?
        .with_arithmetic(args.routing_arithmetic)
        .with_input_transform(SelectorInputTransformSpec::new(
            args.rms_norm_eps,
            ParameterSpec::trainable(format!("{root}.router.scale")).map_err(Error::backend)?,
            true,
        )?)
        .with_coefficient_scale(
            ParameterSpec::trainable(format!("{root}.router.per_expert_scale"))
                .map_err(Error::backend)?,
        );
        let gate_up = format!("{root}.experts.switch_glu.gate_up_proj");
        let down = format!("{root}.experts.switch_glu.down_proj");
        let bank = GroupedGatedProductSpec::new(
            experts,
            args.hidden_size,
            width,
            args.hidden_size,
            eredu_nn::GatedProductPolicy::ordinary_gelu_approximate(),
            GatedProductGroupLayout::Packed {
                gate_up: crate::linear_format::standard_expert_projection(
                    &gate_up,
                    None,
                    args.linear_format_for(&gate_up),
                    eredu_nn::LinearRowLayout::equal_partitions(2)?,
                )?,
                down: crate::linear_format::standard_expert_projection(
                    &down,
                    None,
                    args.linear_format_for(&down),
                    eredu_nn::LinearRowLayout::Contiguous,
                )?,
            },
        )?
        .with_reduction(args.expert_reduction);
        let split_formats = Self::split_names(&bank)?.map(|name| args.linear_format_for(&name));
        let norm = |field| -> Result<_, Error> {
            Ok(args.normalization(
                args.hidden_size,
                ParameterSpec::trainable(format!("{root}.{field}.weight"))
                    .map_err(Error::backend)?,
            ))
        };
        let declaration = Self {
            routed: eredu_nn::RoutedGatedProductSpec {
                selector,
                experts: bank,
            },
            norms: [
                norm(Self::BRANCH_GAINS[0])?,
                norm(Self::BRANCH_GAINS[1])?,
                norm(Self::BRANCH_GAINS[2])?,
            ],
            split_formats,
        };
        declaration.validate_source(args, layer)?;
        declaration.validate_formats(args)?;
        Ok(declaration)
    }
    fn owner(&self) -> Result<(&str, usize), Error> {
        let root = self
            .routed
            .selector
            .weight()
            .id
            .as_str()
            .strip_suffix(".router.proj.weight")
            .ok_or_else(|| Error::backend("invalid Gemma routed owner"))?;
        let layer = root
            .rsplit_once('.')
            .and_then(|(_, layer)| layer.parse::<usize>().ok())
            .ok_or_else(|| Error::backend("invalid Gemma routed layer"))?;
        Ok((root, layer))
    }
    pub(crate) fn validate_source(&self, args: &ModelArgs, layer: usize) -> Result<(), Error> {
        self.validate_invocation(args, layer)?;
        if self.routed.experts.group_count() != self.routed.selector.selection().group_count() {
            return Err(Error::backend(
                "source bank differs from global expert ownership",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_invocation(&self, args: &ModelArgs, layer: usize) -> Result<(), Error> {
        let selection = self.routed.selector.selection();
        if self.owner()?.1 != layer
            || args
                .layer_policy(layer)
                .is_none_or(|p| p.feed_forward != FeedForwardPolicy::DenseWithSparseMoe)
            || self.routed.selector.input_dimensions() != args.hidden_size
            || Some(selection.group_count()) != args.num_experts
            || Some(selection.top_k()) != args.top_k_experts
            || selection.scoring() != args.routing_scoring
            || selection.normalize_selected() != args.normalize_selected_routes
            || self.routed.selector.arithmetic() != args.routing_arithmetic
            || self.routed.selector.logit_scale() != 1.0
            || self.routed.selector.bias().is_some()
            || self.routed.selector.correction_bias().is_some()
            || self.routed.selector.coefficient_scale().is_none()
            || Some(self.routed.experts.intermediate_dimensions()) != args.moe_intermediate_size
            || self.routed.experts.reduction() != args.expert_reduction
            || self.norms.iter().any(|norm| {
                norm.epsilon != args.rms_norm_eps
                    || matches!(norm.scale, eredu_nn::NormalizationScale::Unit)
            })
            || self
                .routed
                .selector
                .input_transform()
                .is_none_or(|t| t.epsilon() != args.rms_norm_eps || !t.inverse_sqrt_dimensions())
        {
            return Err(Error::backend(
                "routed declaration differs from its source geometry or routing equation",
            ));
        }
        self.validate()
    }
    /// Storage names are format extensions referencing declared matrices or split halves.
    fn validate_formats(&self, args: &ModelArgs) -> Result<(), Error> {
        let key = |name: &str| {
            let name = super::specs::canonical_layer_name(name);
            if name.contains(".experts.") && !name.contains(".experts.switch_glu.") {
                name.replace(".experts.", ".experts.switch_glu.")
            } else {
                name
            }
        };
        let parameters = self.logical_parameters()?;
        let (owner, _) = self.owner()?;
        let owner = key(&format!("{owner}.router.proj.weight"));
        let root = owner
            .strip_suffix("router.proj.weight")
            .ok_or_else(|| Error::backend("invalid routed root"))?;
        let split = self.split_source_names()?.map(|name| key(&name));
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|f| f.keys())
            .chain(args.quantized_weights.iter().flatten())
        {
            let target = key(name);
            let Some(local) = target.strip_prefix(root) else {
                continue;
            };
            let routed = local.starts_with("router.")
                || local.starts_with("experts.")
                || Self::branch_gain(local);
            if !routed {
                continue;
            }
            if let Some(index) = split.iter().position(|name| name == &target) {
                if let Some(format) = args
                    .quantized_weight_configs
                    .as_ref()
                    .and_then(|f| f.get(name))
                {
                    if self.split_formats[index] != (*format).into() {
                        return Err(Error::backend(format!(
                            "conflicting split expert encoding: {name}"
                        )));
                    }
                }
                continue;
            }
            let parameter = parameters
                .iter()
                .find(|p| p.linear_format.is_some() && key(p.parameter.id.as_str()) == target)
                .ok_or_else(|| {
                    Error::backend(format!("undeclared routed matrix format: {name}"))
                })?;
            if let Some(format) = args
                .quantized_weight_configs
                .as_ref()
                .and_then(|f| f.get(name))
            {
                if parameter.linear_format.expect("declared matrix").encoding() != (*format).into()
                {
                    return Err(Error::backend(format!(
                        "conflicting routed format alias: {name}"
                    )));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn split_source_names(&self) -> Result<[String; 2], Error> {
        Self::split_names(&self.routed.experts)
    }
    fn split_names(bank: &GroupedGatedProductSpec) -> Result<[String; 2], Error> {
        let packed = bank.packed_parameters()?;
        let root = packed
            .gate_up()
            .parameter
            .id
            .as_str()
            .strip_suffix(".gate_up_proj")
            .ok_or_else(|| Error::backend("invalid fused expert source root"))?;
        Ok([
            format!("{root}.gate_proj.weight"),
            format!("{root}.up_proj.weight"),
        ])
    }
    pub(crate) fn validate_scopes(args: &ModelArgs, scopes: &[Option<Self>]) -> Result<(), Error> {
        if scopes.len() != args.num_hidden_layers() {
            return Err(Error::backend("routed scope count differs from schedule"));
        }
        for (layer, scope) in scopes.iter().enumerate() {
            let sparse = args
                .layer_policy(layer)
                .is_some_and(|p| p.feed_forward == FeedForwardPolicy::DenseWithSparseMoe);
            if scope.is_some() != sparse {
                return Err(Error::backend("routed ownership differs from schedule"));
            }
        }
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|f| f.keys())
            .chain(args.quantized_weights.iter().flatten())
        {
            let name = super::specs::canonical_layer_name(name);
            if let Some((layer, field)) = name
                .strip_prefix("model.layers.")
                .and_then(|local| local.split_once('.'))
            {
                if (field.starts_with("router.")
                    || field.starts_with("experts.")
                    || Self::branch_gain(field))
                    && layer
                        .parse::<usize>()
                        .ok()
                        .and_then(|layer| scopes.get(layer))
                        .and_then(|s| s.as_ref())
                        .is_none()
                {
                    return Err(Error::backend(format!(
                        "routed format references an undeclared sparse layer: {name}"
                    )));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn in_namespace(
        &self,
        args: &ModelArgs,
        layer: usize,
        root: &str,
    ) -> Result<Self, Error> {
        self.validate_source(args, layer)?;
        let (source, owner) = self.owner()?;
        if owner != layer {
            return Err(Error::backend("routed namespace changes layer ownership"));
        }
        let target = format!("{root}.{layer}");
        let namespace = super::specs::ParameterNamespace {
            source,
            target: &target,
        };
        let mut projected = self.clone();
        projected.routed.selector = self
            .routed
            .selector
            .map_parameters(|p| namespace.parameter(p))?;
        projected.routed.experts = self
            .routed
            .experts
            .map_projections(|p| p.map_parameters(|p| namespace.parameter(p)))?;
        for (target, source) in projected.norms.iter_mut().zip(&self.norms) {
            *target = namespace.normalization(source)?;
        }
        projected.validate()?;
        Ok(projected)
    }
    pub(crate) fn project_formats(&self, args: &ModelArgs) -> Result<Self, Error> {
        let mut projected = self.clone();
        let old = self.routed.selector.format();
        let weight = self.routed.selector.weight().id.as_str();
        let mut format =
            crate::linear_format::standard_linear_format(weight, args.linear_format_for(weight))?
                .with_arithmetic(old.arithmetic())?;
        if old.exact_zero_inputs() {
            format = format.with_exact_zero_inputs();
        }
        projected.routed.selector = self.routed.selector.clone().with_format(format)?;
        let packed = self.routed.experts.packed_parameters()?;
        let gate = &packed.gate_up().parameter.id;
        projected.routed.experts = self.routed.experts.map_projections(|p| {
            let row_layout = if &p.weight().id == gate {
                eredu_nn::LinearRowLayout::equal_partitions(2)?
            } else {
                eredu_nn::LinearRowLayout::Contiguous
            };
            let mut format = crate::linear_format::standard_expert_format(
                p.weight().id.as_str(),
                args.linear_format_for(p.weight().id.as_str()),
                row_layout,
            )?
            .with_arithmetic(p.format().arithmetic())?;
            if p.format().exact_zero_inputs() {
                format = format.with_exact_zero_inputs();
            }
            p.clone().with_format(format)
        })?;
        projected.split_formats = self
            .split_source_names()?
            .map(|name| args.linear_format_for(&name));
        projected.validate_source(args, self.owner()?.1)?;
        projected.validate_formats(args)?;
        Ok(projected)
    }
    pub(crate) fn same_formats(&self, other: &Self) -> bool {
        self.split_formats == other.split_formats
            && self.routed.selector.format() == other.routed.selector.format()
            && self.routed.experts.layout() == other.routed.experts.layout()
    }
    pub(crate) fn local_bank(
        &self,
        groups: i32,
        width: i32,
    ) -> Result<GroupedGatedProductSpec, Error> {
        Ok(self
            .routed
            .clone()
            .with_group_geometry(groups, width)?
            .experts)
    }
    /// Selects only bank axes; the complete selector remains global on each invocation.
    pub(crate) fn for_invocation(
        &self,
        args: &ModelArgs,
        layer: usize,
        bank: Option<GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        if self.owner()?.1 != layer {
            return Err(Error::backend("routed invocation changes layer ownership"));
        }
        let width = args
            .moe_intermediate_size
            .ok_or_else(|| Error::backend("missing local routed width"))?;
        let selected = match bank {
            Some(bank) => bank,
            None => self.local_bank(self.routed.experts.group_count(), width)?,
        };
        if selected.intermediate_dimensions() != width
            || self.local_bank(selected.group_count(), width)? != selected
        {
            return Err(Error::backend(
                "local routed bank changes retained ownership, encoding or equation",
            ));
        }
        let mut projected = self.clone();
        projected.routed.experts = selected;
        projected.validate_invocation(args, layer)?;
        projected.parameter_groups("gemma.local.routed")?;
        Ok(projected)
    }
    /// Validates identities and adjacent geometry, including retained local bank geometry.
    pub fn validate(&self) -> Result<(), Error> {
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
    fn validate_geometry(&self) -> Result<(), Error> {
        let width = self.routed.selector.input_dimensions();
        if self.routed.experts.output_dimensions() != width
            || self.routed.experts.policy()
                != eredu_nn::GatedProductPolicy::ordinary_gelu_approximate()
            || self.norms.iter().any(|norm| norm.dimensions != width)
        {
            return Err(Error::backend(
                "Gemma 4 sparse branch references incompatible widths or equation",
            ));
        }
        Ok(())
    }
    pub(crate) fn router_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.routed.selector.logical_parameters()?;
        let scale = self
            .routed
            .selector
            .input_transform()
            .ok_or_else(|| Error::backend("Gemma 4 selector input scale is absent"))?
            .scale();
        parameters.sort_by_key(|p| {
            if p.parameter.id == self.routed.selector.weight().id {
                0
            } else if p.parameter.id == scale.id {
                1
            } else {
                2
            }
        });
        Ok(parameters)
    }
    pub(crate) fn parameter_groups(
        &self,
        root: &str,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, Error> {
        use eredu_runtime::{logical_parameter_group, MemberSharding, ParameterRole};
        let source = self.routed.experts.packed_parameters()?;
        let gate_up = source.gate_up();
        let width = self.routed.experts.intermediate_dimensions() as usize;
        Ok(vec![
            logical_parameter_group(
                format!("{root}.experts.switch_glu.intermediate"),
                ParameterRole::ExpertIntermediate,
                Some(width),
                &self.routed.experts,
                |p| {
                    if p.parameter.id == gate_up.parameter.id {
                        MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: vec![0..width, width..2 * width],
                        }
                    } else {
                        MemberSharding::Partitioned { axis: 2 }
                    }
                },
            )
            .map_err(Error::backend_source)?,
            logical_parameter_group(
                format!("{root}.router"),
                ParameterRole::Replicated,
                None,
                &self.router_parameters()?,
                |_| MemberSharding::Replicated,
            )
            .map_err(Error::backend_source)?,
        ])
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = self.routed.logical_parameters()?;
        parameters.extend(self.norms[..].logical_parameters()?);
        Ok(parameters)
    }
}
