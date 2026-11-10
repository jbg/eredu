//! Shared K2 parameter ownership, with source encodings and invocation state kept separate.
use super::{ExpertBank, ModelArgs};
use crate::decoder::topology::BlockParameters;
use eredu_nn::{
    DeclareLogicalParameters, Error, GroupedLinearSpec, LinearSpec, LogicalParameterDeclaration,
    ParameterSpec, TopKGroupSelectorSpec,
};

/// Declares always-on shared SwiGLU slots, formats and policy at the retained
/// global or selected local width. Invocation ownership remains with the block.
fn shared_expert_parameters(
    args: &ModelArgs,
    layer: usize,
) -> Result<Option<eredu_nn::SplitGatedProductSpec>, Error> {
    // Routing counts remain global while these affine dimensions may be
    // owner-local; a local bank can contain fewer experts than global top-k.
    if args.num_hidden_layers <= 0
        || layer >= args.num_hidden_layers as usize
        || args.num_shared_experts < 0
    {
        return Err(Error::backend("shared expert ownership outside K2 stack"));
    }
    if args.num_shared_experts == 0 {
        return Ok(None);
    }
    let width = match args.local_shared_intermediate_size {
        Some(width) => width,
        None => args
            .moe_intermediate_size
            .checked_mul(args.num_shared_experts)
            .ok_or_else(|| Error::backend("shared expert intermediate width overflow"))?,
    };
    let root = format!("model.layers.{layer}.mlp.shared_experts");
    let linear = |field: &str, input, output| {
        let name = format!("{root}.{field}.weight");
        let mut weight = ParameterSpec::trainable(&name).map_err(Error::backend)?;
        weight.group = Some(root.clone());
        Ok::<_, Error>(LinearSpec {
            input,
            output,
            weight,
            bias: None,
            format: crate::linear_format::standard_linear_format(
                &name,
                args.linear_format_for(&name),
            )?,
        })
    };
    Ok(Some(eredu_nn::SplitGatedProductSpec::new(
        linear("gate_proj", args.hidden_size, width)?,
        linear("up_proj", args.hidden_size, width)?,
        linear("down_proj", width, args.hidden_size)?,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
    )?))
}
#[derive(Debug, Clone)]
pub(crate) struct ValueSpec {
    pub router: TopKGroupSelectorSpec,
    pub experts: GroupedLinearSpec,
}
impl DeclareLogicalParameters for ValueSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.router.input_dimensions() != self.experts.input_dimensions()
            || self.experts.group_count() > self.router.selection().group_count()
            || self.experts.activation() != eredu_nn::GroupedLinearActivation::Silu
            || self.experts.reduction() != eredu_nn::GroupReduction::SequentialGroupOrder
        {
            return Err(Error::backend(
                "K2 routed value geometry or equation mismatch",
            ));
        }
        let mut p = self.router.logical_parameters()?;
        p.extend(self.experts.logical_parameters()?);
        Ok(p)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct RoutedSpec {
    pub parameters: eredu_nn::RoutedGatedProductSpec,
    pub shared: Option<eredu_nn::SplitGatedProductSpec>,
}
impl DeclareLogicalParameters for RoutedSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let experts = &self.parameters.experts;
        if experts.output_dimensions() != experts.input_dimensions()
            || experts.policy() != eredu_nn::GatedProductPolicy::ordinary_silu()
            || experts.reduction() != eredu_nn::GroupReduction::SequentialGroupOrder
            || self.shared.as_ref().is_some_and(|shared| {
                shared.gate.input != experts.input_dimensions()
                    || shared.down.output != experts.output_dimensions()
                    || shared.policy != eredu_nn::GatedProductPolicy::ordinary_silu()
            })
        {
            return Err(Error::backend(
                "K2 routed/shared feed-forward geometry or equation mismatch",
            ));
        }
        let mut p = self.parameters.logical_parameters()?;
        if let Some(shared) = &self.shared {
            p.extend(shared.logical_parameters()?);
        }
        Ok(p)
    }
}
/// Ordinary attention/normalization slots and optional independently owned banks for one invocation.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    pub(crate) common: BlockParameters,
    pub(crate) values: Option<ValueSpec>,
    pub(crate) routed: Option<RoutedSpec>,
}
impl BlockSpec {
    /// Declares the actual sparse schedule, omitting ordinary values when a routed value bank owns them.
    pub fn new(args: &ModelArgs, layer: usize) -> Result<Self, Error> {
        args.validate().map_err(Error::backend_source)?;
        if layer >= args.num_hidden_layers as usize {
            return Err(Error::backend("K2 block outside declared stack"));
        }
        let common = BlockParameters::with_feed_forward(
            args,
            layer,
            (!args.is_sparse_layer(layer))
                .then(|| crate::decoder::topology::gated_parameters(args, layer))
                .transpose()?,
        )?;
        let routed = if args.is_sparse_layer(layer) {
            Some(RoutedSpec {
                parameters: eredu_nn::RoutedGatedProductSpec {
                    selector: super::routing::router_spec(args, layer, ExpertBank::FeedForward)?,
                    experts: super::routing::feed_forward_expert_spec(args, layer)?,
                },
                shared: shared_expert_parameters(args, layer)?,
            })
        } else {
            None
        };
        let values = if args.is_mova_layer(layer) {
            Some(ValueSpec {
                router: super::routing::router_spec(args, layer, ExpertBank::AttentionValue)?,
                experts: super::routing::value_expert_spec(args, layer)?,
            })
        } else {
            None
        };
        let spec = Self {
            common,
            values,
            routed,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
    /// Projects selected local banks and shared channels from retained global parameters.
    pub(crate) fn with_local_geometry(mut self, local: &ModelArgs) -> Result<Self, Error> {
        self.values = self
            .values
            .map(|spec| spec.with_local_geometry(local))
            .transpose()?;
        self.routed = self
            .routed
            .map(|spec| spec.with_local_geometry(local))
            .transpose()?;
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(self)
    }
}
impl DeclareLogicalParameters for BlockSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.common.logical_parameters()?;
        if let Some(values) = &self.values {
            p.extend(values.logical_parameters()?);
        }
        if let Some(routed) = &self.routed {
            p.extend(routed.logical_parameters()?);
        }
        Ok(p)
    }
}

impl ValueSpec {
    pub(crate) fn with_local_geometry(mut self, local: &ModelArgs) -> Result<Self, Error> {
        let groups = local.local_value_experts.unwrap_or(local.mova_num_experts);
        if groups > self.experts.group_count() {
            return Err(Error::backend("local K2 values exceed their declared bank"));
        }
        self.experts = self
            .experts
            .clone()
            .partition_output(local.value_output_range())?
            .with_group_count(groups)?;
        self.logical_parameters()?;
        Ok(self)
    }
}
impl RoutedSpec {
    pub(crate) fn with_local_geometry(mut self, local: &ModelArgs) -> Result<Self, Error> {
        if local.moe_intermediate_size > self.parameters.experts.intermediate_dimensions() {
            return Err(Error::backend(
                "local K2 expert width exceeds its declaration",
            ));
        }
        self.parameters = self
            .parameters
            .clone()
            .with_group_geometry(local.num_experts, local.moe_intermediate_size)?;
        if let Some(shared) = &mut self.shared {
            let width = local
                .local_shared_intermediate_size
                .unwrap_or(shared.gate.output);
            if width <= 0 || width > shared.gate.output {
                return Err(Error::backend(
                    "local K2 shared width exceeds its declaration",
                ));
            }
            shared.gate.output = width;
            shared.up.output = width;
            shared.down.input = width;
        }
        self.logical_parameters()?;
        Ok(self)
    }
}
