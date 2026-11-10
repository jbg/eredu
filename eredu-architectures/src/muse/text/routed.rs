//! Routed parameters shared by construction, source admission, placement, and discovery.

use crate::muse::{
    equations::{specs::LearnedRmsNormSpec, RmsNormEquation},
    DecoderPolicy, RoutedFeedForwardEquation, WeightConvention,
};
use eredu_nn::{
    DeclareLogicalParameters, Error, GatedProductGroupLayout, GroupScoring,
    GroupedGatedProductSpec, LinearSpec, LogicalParameterDeclaration, ParameterSpec,
    TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{MemberSharding, ParameterGroupSpec, ParameterRole};

/// Decoder/latent projections and per-expert normalization around a routed bank.
#[derive(Debug, Clone)]
pub struct CompressionSpec {
    /// Decoder-to-expert projection, before normalization.
    pub input: LinearSpec,
    /// Input and complete-expert-output gains, in execution order.
    pub norms: [LearnedRmsNormSpec; 2],
    /// Projection after weighted accumulation in expert space.
    pub output: LinearSpec,
}
impl DeclareLogicalParameters for CompressionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.input.input != self.output.output
            || self.input.output != self.output.input
            || self
                .norms
                .iter()
                .any(|norm| norm.normalization().dimensions != self.input.output)
        {
            return Err(Error::backend(
                "Muse compression references incompatible geometry",
            ));
        }
        let mut parameters = self.input.logical_parameters()?;
        parameters.extend(self.norms[..].logical_parameters()?);
        parameters.extend(self.output.logical_parameters()?);
        Ok(parameters)
    }
}
/// Decoder-width routing with a normalized latent-space expert bank.
#[derive(Debug, Clone)]
pub struct CompressedDeclaration {
    /// Router consumes the original decoder-width activation.
    pub selector: TopKGroupSelectorSpec,
    /// Expert bank consumes and produces complete latent-width activations.
    pub bank: GroupedGatedProductSpec,
    /// Required ingress, egress and per-expert normalization equations.
    pub compression: CompressionSpec,
}
impl DeclareLogicalParameters for CompressedDeclaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.selector.input_dimensions() != self.compression.input.input
            || self.bank.input_dimensions() != self.compression.input.output
            || self.bank.output_dimensions() != self.compression.output.input
            || self.bank.group_count() > self.selector.selection().group_count()
            || self.bank.policy() != eredu_nn::GatedProductPolicy::ordinary_silu()
            || self.bank.reduction() != eredu_nn::GroupReduction::SequentialGroupOrder
        {
            return Err(Error::backend(
                "Muse compressed expert geometry or equation mismatch",
            ));
        }
        let mut parameters = self.selector.logical_parameters()?;
        parameters.extend(self.bank.logical_parameters()?);
        parameters.extend(self.compression.logical_parameters()?);
        Ok(parameters)
    }
}
/// Complete parameter equations for one routed invocation; ownership stays separate.
#[derive(Debug, Clone)]
pub enum Declaration {
    /// Router and experts share the original decoder-width input.
    Direct(eredu_nn::RoutedGatedProductSpec),
    /// Router and experts consume different, explicitly connected activations.
    Compressed(CompressedDeclaration),
}
impl Declaration {
    /// Declares and validates a routed invocation without allocating backend tensors.
    pub(in crate::muse) fn new(args: &DecoderPolicy, layer: usize) -> Result<Self, Error> {
        if !args.is_moe() || args.num_hidden_layers <= 0 || layer >= args.num_hidden_layers as usize
        {
            return Err(Error::backend(
                "Muse routed invocation is outside its declared decoder",
            ));
        }
        let root = format!("model.layers.{layer}.mlp");
        let linear = |field, input, output| -> Result<LinearSpec, Error> {
            let name = format!("{root}.{field}.weight");
            Ok(LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    &name,
                    args.linear_format_for(&name),
                )?,
            })
        };
        let router = linear("gate", args.hidden_size, args.num_experts)?;
        let mut selector = TopKGroupSelectorSpec::new(
            router.input,
            router.weight,
            router.format,
            Self::selection(args)?,
        )?;
        let compression = match args.routed_feed_forward {
            RoutedFeedForwardEquation::DirectSoftmax => None,
            RoutedFeedForwardEquation::CompressedSigmoid {
                expert_width,
                logit_scale,
                ..
            } => {
                let mut correction =
                    ParameterSpec::trainable(format!("{root}.gate.e_score_correction_bias"))
                        .map_err(Error::backend)?;
                correction.trainable = false;
                selector = selector
                    .with_logit_scale(logit_scale)?
                    .with_correction_bias(correction)?;
                Some(CompressionSpec {
                    input: linear("pre_expert_proj", args.hidden_size, expert_width)?,
                    norms: [
                        super::specs::learned_norm(
                            expert_width,
                            args.rms_norm_eps,
                            RmsNormEquation::OutputScaledFp32(args.weight_convention),
                            format!("{root}.pre_expert_norm.weight"),
                        )?,
                        super::specs::learned_norm(
                            expert_width,
                            args.post_norm_eps,
                            RmsNormEquation::InputScaled(args.weight_convention),
                            format!("{root}.experts.post_expert_norm.weight"),
                        )?,
                    ],
                    output: linear("post_expert_proj", expert_width, args.hidden_size)?,
                })
            }
        };
        let width = compression
            .as_ref()
            .map_or(args.hidden_size, |spec| spec.input.output);
        let gate_up = format!("{root}.experts.gate_up_proj");
        let down = format!("{root}.experts.down_proj");
        let mut bank = GroupedGatedProductSpec::new(
            args.num_experts,
            width,
            args.moe_intermediate_size,
            width,
            eredu_nn::GatedProductPolicy::ordinary_silu(),
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
        )?;
        if compression.is_some() {
            bank = bank.with_reduction(eredu_nn::GroupReduction::SequentialGroupOrder);
        }
        let declaration = match compression {
            Some(compression) => Self::Compressed(CompressedDeclaration {
                selector,
                bank,
                compression,
            }),
            None => Self::Direct(eredu_nn::RoutedGatedProductSpec {
                selector,
                experts: bank,
            }),
        };
        declaration.validate()?;
        Ok(declaration)
    }
    /// Route scoring and ordering, separate from hooks and collector support.
    fn selection(args: &DecoderPolicy) -> Result<TopKGroupSelectionSpec, Error> {
        Ok(match args.routed_feed_forward {
            RoutedFeedForwardEquation::DirectSoftmax => TopKGroupSelectionSpec::new(
                args.num_experts,
                args.num_experts_per_tok,
                GroupScoring::Softmax,
                args.norm_topk_prob,
            )?,
            RoutedFeedForwardEquation::CompressedSigmoid {
                coefficient_scale, ..
            } => TopKGroupSelectionSpec::new(
                args.num_experts,
                args.num_experts_per_tok,
                GroupScoring::Sigmoid,
                true,
            )?
            .with_weight_policy(1e-15, coefficient_scale)?
            .with_order(if args.weight_convention == WeightConvention::HuggingFace {
                eredu_nn::GroupSelectionOrder::Partition
            } else {
                eredu_nn::GroupSelectionOrder::Ranked
            }),
        })
    }
    /// Global selector, independent of prepared expert ownership and latent width.
    pub fn selector(&self) -> &TopKGroupSelectorSpec {
        match self {
            Self::Direct(parameters) => &parameters.selector,
            Self::Compressed(parameters) => &parameters.selector,
        }
    }
    /// Authoritative bank geometry, policy and encodings.
    pub fn bank(&self) -> &GroupedGatedProductSpec {
        match self {
            Self::Direct(parameters) => &parameters.experts,
            Self::Compressed(parameters) => &parameters.bank,
        }
    }
    /// Latent-space equations are present exactly for compressed declarations.
    pub fn compression(&self) -> Option<&CompressionSpec> {
        match self {
            Self::Direct(_) => None,
            Self::Compressed(parameters) => Some(&parameters.compression),
        }
    }
    /// Consumes retained parameter slots for binding, without reconstructing geometry.
    pub fn into_parts(
        self,
    ) -> (
        TopKGroupSelectorSpec,
        GroupedGatedProductSpec,
        Option<CompressionSpec>,
    ) {
        match self {
            Self::Direct(parameters) => (parameters.selector, parameters.experts, None),
            Self::Compressed(parameters) => (
                parameters.selector,
                parameters.bank,
                Some(parameters.compression),
            ),
        }
    }
    /// Binds local expert ownership while retaining the global equation and identities.
    pub fn with_bank(mut self, bank: GroupedGatedProductSpec) -> Result<Self, Error> {
        let source = self.bank();
        if bank.input_dimensions() != source.input_dimensions()
            || bank.output_dimensions() != source.output_dimensions()
            || bank.intermediate_dimensions() > source.intermediate_dimensions()
            || bank.group_count() > source.group_count()
            || bank.policy() != source.policy()
            || bank.reduction() != source.reduction()
        {
            return Err(Error::backend(
                "Muse selected bank changes retained geometry or equation",
            ));
        }
        let original = source.logical_parameters()?;
        let selected = bank.logical_parameters()?;
        if original.len() != selected.len()
            || original.iter().zip(&selected).any(|(a, b)| {
                a.parameter != b.parameter || a.role != b.role || a.linear_format != b.linear_format
            })
        {
            return Err(Error::backend(
                "Muse selected bank changes declared identities, roles or encodings",
            ));
        }
        match &mut self {
            Self::Direct(parameters) => parameters.experts = bank,
            Self::Compressed(parameters) => parameters.bank = bank,
        }
        self.validate()?;
        Ok(self)
    }
    /// Projects storage extensions without changing retained equations or ownership.
    pub(in crate::muse) fn project_formats(&mut self, args: &DecoderPolicy) -> Result<(), Error> {
        fn format(
            name: &str,
            old: &eredu_nn::LinearFormatSpec,
            args: &DecoderPolicy,
            rows: Option<eredu_nn::LinearRowLayout>,
        ) -> Result<eredu_nn::LinearFormatSpec, Error> {
            let mut format = match rows {
                Some(rows) => crate::linear_format::standard_expert_format(
                    name,
                    args.linear_format_for(name),
                    rows,
                )?,
                None => crate::linear_format::standard_linear_format(
                    name,
                    args.linear_format_for(name),
                )?,
            }
            .with_arithmetic(old.arithmetic())?;
            if old.exact_zero_inputs() {
                format = format.with_exact_zero_inputs();
            }
            Ok(format)
        }
        let selector = self.selector().clone().with_format(format(
            self.selector().weight().id.as_str(),
            self.selector().format(),
            args,
            None,
        )?)?;
        let source = self.bank();
        let GatedProductGroupLayout::Packed { gate_up, down } = source.layout() else {
            return Err(Error::backend(
                "Muse source bank requires packed parameter ownership",
            ));
        };
        let projection = |spec: &eredu_nn::GroupedProjectionSpec, rows| {
            eredu_nn::GroupedProjectionSpec::new(
                spec.weight().clone(),
                spec.bias().cloned(),
                format(spec.weight().id.as_str(), spec.format(), args, Some(rows))?,
            )
        };
        let bank = GroupedGatedProductSpec::new(
            source.group_count(),
            source.input_dimensions(),
            source.intermediate_dimensions(),
            source.output_dimensions(),
            source.policy(),
            GatedProductGroupLayout::Packed {
                gate_up: projection(gate_up, eredu_nn::LinearRowLayout::equal_partitions(2)?)?,
                down: projection(down, eredu_nn::LinearRowLayout::Contiguous)?,
            },
        )?
        .with_reduction(source.reduction());
        match self {
            Self::Direct(spec) => {
                spec.selector = selector;
                spec.experts = bank;
            }
            Self::Compressed(spec) => {
                spec.selector = selector;
                spec.bank = bank;
                for linear in [&mut spec.compression.input, &mut spec.compression.output] {
                    linear.format = format(linear.weight.id.as_str(), &linear.format, args, None)?;
                }
            }
        }
        self.validate()
    }
    /// Validates complete references and geometry before backend allocation.
    pub fn validate(&self) -> Result<(), Error> {
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }

    /// Ordinary global bank placement; invocation and expert residency owners remain separate.
    pub(crate) fn parameter_groups(&self, root: &str) -> Result<Vec<ParameterGroupSpec>, Error> {
        let GatedProductGroupLayout::Packed { gate_up, .. } = self.bank().layout() else {
            return Err(Error::backend(
                "Muse global placement requires its declared packed target bank",
            ));
        };
        let intermediate = self.bank().intermediate_dimensions() as usize;
        let mut groups = vec![eredu_runtime::logical_parameter_group(
            format!("{root}.experts.intermediate"),
            ParameterRole::ExpertIntermediate,
            Some(intermediate),
            self.bank(),
            |p| {
                if p.parameter.id == gate_up.weight().id {
                    MemberSharding::PartitionedSegments {
                        axis: 1,
                        segments: vec![0..intermediate, intermediate..2 * intermediate],
                    }
                } else {
                    MemberSharding::Partitioned { axis: 2 }
                }
            },
        )
        .map_err(Error::backend_source)?];
        if let Some(spec) = self.compression() {
            groups.push(
                eredu_runtime::logical_parameter_group(
                    format!("{root}.compression"),
                    ParameterRole::Replicated,
                    None,
                    spec,
                    |_| MemberSharding::Replicated,
                )
                .map_err(Error::backend_source)?,
            );
        }
        groups.push(
            eredu_runtime::logical_parameter_group(
                format!("{root}.router"),
                ParameterRole::Replicated,
                None,
                self.selector(),
                |_| MemberSharding::Replicated,
            )
            .map_err(Error::backend_source)?,
        );
        Ok(groups)
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Direct(parameters) => {
                if parameters.experts.output_dimensions() != parameters.selector.input_dimensions()
                    || parameters.experts.policy() != eredu_nn::GatedProductPolicy::ordinary_silu()
                    || parameters.experts.reduction() != eredu_nn::GroupReduction::Sum
                {
                    return Err(Error::backend(
                        "Muse direct expert residual geometry or equation mismatch",
                    ));
                }
                parameters.logical_parameters()
            }
            Self::Compressed(parameters) => parameters.logical_parameters(),
        }
    }
}
