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

mod observed;

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
    #[parameter(skip)]
    points: Option<eredu_runtime::RoutedObservationPoints>,
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
        self.values_instrumented(
            input,
            pass,
            provider,
            context,
            &mut decoder::ComponentInstrumentation::disabled(),
        )
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
        self.feed_forward_instrumented(
            input,
            pass,
            provider,
            context,
            &mut decoder::ComponentInstrumentation::disabled(),
        )
    }
}

impl<B: GroupedNeuralBackend> DecoderProjectionOperator<B> for Projections<B> {
    fn project_values_observed(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<Option<B::Tensor>, Error> {
        self.values_instrumented(
            input,
            pass(input),
            &mut ResidentExpertProvider,
            context,
            instrumentation,
        )
    }
    fn residual_observation(&self) -> &'static str {
        match self.feed_forward {
            FeedForward::Dense(_) => "feed_forward.output",
            FeedForward::Routed { .. } => "feed_forward.contribution",
        }
    }
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
    fn forward_feed_forward_observed(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.feed_forward_instrumented(
            input,
            pass(input),
            &mut ResidentExpertProvider,
            context,
            instrumentation,
        )
    }
}
impl<B: GroupedNeuralBackend> RoutedProjectionOperator<B> for Projections<B> {
    const COMPONENT_OBSERVATIONS: bool = true;

    fn project_values_observed_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        _points: Option<eredu_runtime::RoutedObservationPoints>,
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
        self.values_instrumented(input, pass, provider, context, instrumentation)
    }

    fn forward_observed_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        _points: Option<eredu_runtime::RoutedObservationPoints>,
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
        self.feed_forward_instrumented(input, pass, provider, context, instrumentation)
    }
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
    fn forward_feed_forward_parallel_observed(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.feed_forward_parallel_instrumented(
            input,
            pass(input),
            &mut ResidentExpertProvider,
            parallel,
            context,
            instrumentation,
        )
    }
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
    fn forward_parallel_observed_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        _points: Option<eredu_runtime::RoutedObservationPoints>,
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
        self.feed_forward_parallel_instrumented(
            input,
            pass,
            provider,
            parallel,
            context,
            instrumentation,
        )
    }
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
        self.feed_forward_parallel_instrumented(
            input,
            pass,
            provider,
            parallel,
            context,
            &mut decoder::ComponentInstrumentation::disabled(),
        )
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
    let width = args
        .local_shared_intermediate_size
        .unwrap_or(args.moe_intermediate_size * args.num_shared_experts);
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
                points: <ModelArgs as decoder::Config>::routed_observation_points(
                    global,
                    &format!("model.layers.{layer}"),
                    layer,
                ),
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
