//! K2 projection construction inside the shared decoder and ordinary KV lifecycle.

use super::{ExpertBank, ModelArgs};
use crate::decoder::{
    self, DecoderProjectionOperator, RoutedProjectionOperator, TensorParallelProjectionOperator,
    TensorParallelRoutedProjectionOperator,
};
use eredu_nn::{
    Error, GroupedNeuralBackend, LinearSpec, NormalizationConstructionSpec, ParameterSpec,
    Parameterized, Tensor,
};
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedBankId, RoutedExpertProvider, RoutedExpertRequest,
    TensorParallelRoutedExpertProvider,
};

impl ExpertBank {
    /// Stable identity shared by execution, residency, and routing observations.
    pub const fn id(self) -> RoutedBankId {
        RoutedBankId::new(match self {
            Self::FeedForward => 0,
            Self::AttentionValue => 1,
        })
    }
}

/// Activated value projections whose mixtures are cached once per token.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RoutedValues<B: GroupedNeuralBackend> {
    /// Global value selector, independent of the feed-forward selector.
    pub router: B::Selector,
    /// Independently addressable value bank with owned output rows.
    pub experts: B::LinearGroups,
    #[parameter(skip)]
    output_width: i32,
}

/// Dense or routed feed-forward equation selected for one logical block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend> {
    /// Dense prefix or configuration-selected dense block.
    Dense(decoder::Mlp<B>),
    /// Routed SwiGLU and optional always-on shared experts.
    Routed {
        /// Global feed-forward selector.
        router: B::Selector,
        /// Routed expert bank.
        experts: B::GatedProductGroups,
        /// Always-on shared SwiGLU.
        shared: Option<decoder::Mlp<B>>,
    },
}

/// Both replaceable stages of a K2 block; normalizations and residuals are shared.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Projections<B: GroupedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Optional routed attention-value stage.
    pub values: Option<RoutedValues<B>>,
    /// Dense or routed feed-forward stage.
    pub feed_forward: FeedForward<B>,
}

fn pass<T: Tensor>(input: &T) -> ExpertPass {
    if input.shape()[input.shape().len() - 2] > 1 {
        ExpertPass::Prefill
    } else {
        ExpertPass::Decode
    }
}

impl<B: GroupedNeuralBackend> Projections<B> {
    fn values<P: RoutedExpertProvider<B>>(
        &mut self,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error>
    where
        P::Error: std::fmt::Display,
    {
        let Some(values) = &mut self.values else {
            return Ok(None);
        };
        let mut shape = input.shape().to_vec();
        let hidden = *shape
            .last()
            .ok_or_else(|| Error::backend("value input is scalar"))?;
        let flat = input.reshape(&[-1, hidden], context)?;
        let routes = eredu_runtime::select_routes_with_provider::<B, P>(
            &mut values.router,
            &flat,
            context,
            provider,
            ExpertBank::AttentionValue.id(),
        )?;
        let output = provider
            .forward_linear_routed(
                &mut values.experts,
                RoutedExpertRequest {
                    layer: self.layer,
                    bank: ExpertBank::AttentionValue.id(),
                    input: &flat,
                    routes: &routes,
                    pass,
                },
                context,
            )
            .map_err(|e| Error::backend(e.to_string()))?;
        *shape.last_mut().expect("validated input rank") = values.output_width;
        Ok(Some(output.reshape(&shape, context)?))
    }

    fn feed_forward<P: RoutedExpertProvider<B>>(
        &mut self,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::fmt::Display,
    {
        match &mut self.feed_forward {
            FeedForward::Dense(mlp) => mlp.forward_feed_forward(input, context),
            FeedForward::Routed {
                router,
                experts,
                shared,
            } => {
                let shape = input.shape();
                let flat = input.reshape(
                    &[
                        -1,
                        *shape
                            .last()
                            .ok_or_else(|| Error::backend("feed-forward input is scalar"))?,
                    ],
                    context,
                )?;
                let routes = eredu_runtime::select_routes_with_provider::<B, P>(
                    router,
                    &flat,
                    context,
                    provider,
                    ExpertBank::FeedForward.id(),
                )?;
                let routed = provider
                    .forward_grouped(
                        experts,
                        RoutedExpertRequest {
                            layer: self.layer,
                            bank: ExpertBank::FeedForward.id(),
                            input: &flat,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(|e| Error::backend(e.to_string()))?
                    .reshape(shape, context)?;
                match shared {
                    Some(shared) => {
                        routed.add(&shared.forward_feed_forward(input, context)?, context)
                    }
                    None => Ok(routed),
                }
            }
        }
    }
}

impl<B: GroupedNeuralBackend> DecoderProjectionOperator<B> for Projections<B> {
    fn project_values(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        self.values(input, pass(input), &mut ResidentExpertProvider, context)
    }
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.feed_forward(input, pass(input), &mut ResidentExpertProvider, context)
    }
}
impl<B: GroupedNeuralBackend> RoutedProjectionOperator<B> for Projections<B> {
    fn project_values_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        if layer != self.layer {
            return Err(Error::backend(
                "value bank logical layer differs from request",
            ));
        }
        self.values(input, pass, provider, context)
    }
    fn forward_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        if layer != self.layer {
            return Err(Error::backend(
                "feed-forward bank logical layer differs from request",
            ));
        }
        self.feed_forward(input, pass, provider, context)
    }
}
impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelProjectionOperator<B> for Projections<B>
{
    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_parallel_with_provider(
            self.layer,
            input,
            pass(input),
            &mut ResidentExpertProvider,
            parallel,
            context,
        )
    }
}
impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelRoutedProjectionOperator<B> for Projections<B>
{
    fn forward_parallel_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        if layer != self.layer {
            return Err(Error::backend(
                "parallel bank logical layer differs from request",
            ));
        }
        match &mut self.feed_forward {
            FeedForward::Dense(mlp) => mlp.forward_feed_forward_parallel(input, parallel, context),
            FeedForward::Routed {
                router,
                experts,
                shared,
            } => {
                let shape = input.shape();
                let flat = input.reshape(
                    &[
                        -1,
                        *shape
                            .last()
                            .ok_or_else(|| Error::backend("feed-forward input is scalar"))?,
                    ],
                    context,
                )?;
                let routes = eredu_runtime::select_routes_with_provider::<B, P>(
                    router,
                    &flat,
                    context,
                    provider,
                    ExpertBank::FeedForward.id(),
                )?;
                let routed = provider
                    .forward_grouped_tensor_parallel(
                        experts,
                        RoutedExpertRequest {
                            layer,
                            bank: ExpertBank::FeedForward.id(),
                            input: &flat,
                            routes: &routes,
                            pass,
                        },
                        B::parallel_size(parallel),
                        context,
                    )
                    .map_err(|e| Error::backend(e.to_string()))?;
                let routed = eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(
                    routed, parallel, context,
                )?
                .reshape(shape, context)?;
                match shared {
                    Some(shared) => routed.add(
                        &shared.forward_feed_forward_parallel(input, parallel, context)?,
                        context,
                    ),
                    None => Ok(routed),
                }
            }
        }
    }
}

fn shared_mlp<B: GroupedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<decoder::Mlp<B>>, Error> {
    if args.num_shared_experts == 0 {
        return Ok(None);
    }
    let width = args.moe_intermediate_size * args.num_shared_experts;
    let linear = |field: &str, input, output| {
        let name = format!("model.layers.{layer}.mlp.shared_experts.{field}.weight");
        B::linear(
            LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    &name,
                    args.linear_format_for(&name),
                )?,
            },
            context,
        )
    };
    Ok(Some(decoder::Mlp::from_parts(
        linear("gate_proj", args.hidden_size, width)?,
        linear("up_proj", args.hidden_size, width)?,
        linear("down_proj", width, args.hidden_size)?,
        None,
    )))
}

/// Shared decoder block specialized to the K2 projection equation.
pub type TransformerBlock<B> = decoder::TransformerBlock<B, Projections<B>>;
/// Shared ordinary-KV layered execution specialized to K2.
pub type LayeredModel<B> = decoder::LayeredModel<B, ModelArgs, BlockFactory>;
/// Shared pipeline-local ordinary-KV execution specialized to K2.
pub type PartitionedLayeredModel<B> = decoder::PartitionedLayeredModel<B, ModelArgs, BlockFactory>;

/// Portable construction policy; no backend selects a K2 equation.
pub struct BlockFactory;
impl<B: GroupedNeuralBackend> decoder::BlockFactory<B, ModelArgs> for BlockFactory {
    type FeedForward = Projections<B>;
    fn validate(args: &ModelArgs) -> Result<(), Error> {
        args.validate().map_err(Error::backend)
    }
    fn build(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B>, Error> {
        Self::build_partitioned(args, args, layer, context)
    }
    fn build_partitioned(
        global: &ModelArgs,
        local: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B>, Error> {
        global.validate().map_err(Error::backend)?;
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    local.hidden_size,
                    local.rms_norm_eps,
                    ParameterSpec::trainable(format!("model.layers.{layer}.{field}.weight"))
                        .map_err(Error::backend)?,
                )
                .with_groups(local.layernorm_num_groups)?,
                context,
            )
        };
        let values = if global.is_mova_layer(layer) {
            let spec = super::value_expert_spec(global, layer)?
                .partition_output(local.value_output_range())?
                .with_group_count(local.mova_num_experts)?;
            Some(RoutedValues {
                router: super::new_router::<B>(global, layer, ExpertBank::AttentionValue, context)?,
                experts: B::grouped_linear_bank(spec, context)?,
                output_width: local.num_key_value_heads * local.head_dim,
            })
        } else {
            None
        };
        let feed_forward = if global.is_sparse_layer(layer) {
            FeedForward::Routed {
                router: super::new_router::<B>(global, layer, ExpertBank::FeedForward, context)?,
                experts: B::grouped_gated_product(
                    super::feed_forward_expert_spec(global, layer)?
                        .with_group_geometry(local.num_experts, local.moe_intermediate_size)?,
                    context,
                )?,
                shared: shared_mlp::<B>(local, layer, context)?,
            }
        } else {
            FeedForward::Dense(decoder::Mlp::new(local, layer, context)?)
        };
        Ok(TransformerBlock {
            self_attention: decoder::Attention::new(local, layer, context)?,
            mlp: Projections {
                layer,
                values,
                feed_forward,
            },
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
            attention_output_norm: None,
            feed_forward_output_norm: None,
            output_norm: None,
        })
    }
    fn parameter_groups(
        block: &TransformerBlock<B>,
        args: &ModelArgs,
        layer: usize,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, eredu_runtime::ParallelPlanError> {
        super::parallel::block_parameter_groups(block, args, layer)
    }
}

/// Constructs an unloaded K2 block using its normalized sparse schedule.
pub fn new_block<B: GroupedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TransformerBlock<B>, Error> {
    <BlockFactory as decoder::BlockFactory<B, ModelArgs>>::build(args, layer, context)
}

/// Shared dense decoder execution for configurations with no routed invocations.
pub type DenseLayeredModel<B> = decoder::LayeredModel<B, ModelArgs, DenseBlockFactory>;
/// Dense construction rejects configurations that require expert providers.
pub struct DenseBlockFactory;
impl<B: eredu_nn::NeuralBackend> decoder::BlockFactory<B, ModelArgs> for DenseBlockFactory {
    type FeedForward = decoder::Mlp<B>;
    fn validate(args: &ModelArgs) -> Result<(), Error> {
        args.validate().map_err(Error::backend)?;
        if args.is_moe() {
            return Err(Error::backend(
                "dense construction received routed invocations",
            ));
        }
        Ok(())
    }
    fn build(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<decoder::TransformerBlock<B>, Error> {
        decoder::TransformerBlock::new(args, layer, context)
    }
    fn parameter_groups(
        block: &decoder::TransformerBlock<B>,
        args: &ModelArgs,
        layer: usize,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, eredu_runtime::ParallelPlanError> {
        decoder::layer_parallel_parameter_groups(block, args, layer)
    }
}
