//! K2 projection construction inside the shared decoder and ordinary KV lifecycle.

use super::{ExpertBank, ModelArgs};
use crate::decoder::{
    self, DecoderProjectionOperator, RoutedProjectionOperator, TensorParallelProjectionOperator,
    TensorParallelRoutedProjectionOperator,
};
use eredu_nn::{Error, GroupedNeuralBackend, Parameterized, Tensor};
use eredu_runtime::{
    ExpertPass, ParameterProvider, ResidentExpertProvider, RoutedBankId, RoutedExpertRequest,
    TensorParallelParameterProvider,
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
    fn values<P: ParameterProvider<B>>(
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

    fn feed_forward<P: ParameterProvider<B>>(
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
        P: ParameterProvider<B>,
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
        P: ParameterProvider<B>,
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
        P: ParameterProvider<B>,
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
        P: ParameterProvider<B>,
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
        P: TensorParallelParameterProvider<B>,
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
        P: TensorParallelParameterProvider<B>,
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
    parameters: Option<eredu_nn::SplitGatedProductSpec>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<decoder::Mlp<B>>, Error> {
    parameters
        .map(|spec| {
            Ok(decoder::Mlp::from_parts(
                B::linear(spec.gate, context)?,
                B::linear(spec.up, context)?,
                B::linear(spec.down, context)?,
                spec.policy,
            ))
        })
        .transpose()
}

/// Shared decoder block specialized to the K2 projection equation.
pub type TransformerBlock<B> = decoder::TransformerBlock<B, Projections<B>>;
/// Shared ordinary-KV layered execution specialized to K2.
pub type LayeredModel<B> = decoder::LayeredModel<B, ModelArgs, BlockFactory>;
/// Shared pipeline-local ordinary-KV execution specialized to K2.
pub type PartitionedLayeredModel<B> = decoder::PartitionedLayeredModel<B, ModelArgs, BlockFactory>;

/// Complete dense/shared/routed stage parameters, independent of native modules.
#[derive(Clone, Debug)]
pub struct ProjectionParameters {
    pub(crate) dense: Option<eredu_nn::GatedProductSpec>,
    pub(crate) routed: Option<super::specs::RoutedSpec>,
    pub(crate) values: Option<super::specs::ValueSpec>,
    pub(crate) points: Option<eredu_runtime::RoutedObservationPoints>,
}
impl eredu_nn::DeclareLogicalParameters for ProjectionParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        if self.dense.is_some() == self.routed.is_some() {
            return Err(Error::backend(
                "K2 projections require exactly one feed-forward equation",
            ));
        }
        let mut parameters = Vec::new();
        if let Some(dense) = &self.dense {
            parameters.extend(dense.logical_parameters()?);
        }
        if let Some(routed) = &self.routed {
            parameters.extend(routed.logical_parameters()?);
        }
        if let Some(values) = &self.values {
            parameters.extend(values.logical_parameters()?);
        }
        Ok(parameters)
    }
}

/// Portable construction policy; no backend selects a K2 equation.
pub struct BlockFactory;
impl<B: GroupedNeuralBackend> decoder::BlockFactory<B, ModelArgs> for BlockFactory {
    type FeedForward = Projections<B>;
    fn validate(args: &ModelArgs) -> Result<(), Error> {
        args.validate().map_err(Error::backend)
    }
    fn validate_partitioned(global: &ModelArgs, _local: &ModelArgs) -> Result<(), Error> {
        <Self as decoder::BlockFactory<B, ModelArgs>>::validate(global)
    }
    type Parameters = ProjectionParameters;
    fn prepare_parameters(
        global: &ModelArgs,
        local: &ModelArgs,
        layer: usize,
        dense: Option<&eredu_nn::GatedProductSpec>,
        retained: Option<&Self::Parameters>,
    ) -> Result<Self::Parameters, Error> {
        ProjectionParameters::prepare(global, local, layer, dense, retained)
    }
    fn parameter_description(
        config: &ModelArgs,
        common: &decoder::DecoderDeclaration,
        projections: Option<&decoder::ProjectionDeclarations<Self::Parameters>>,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Error> {
        super::parameter_description(&super::Declaration::from_scopes(
            config.clone(),
            std::sync::Arc::new(common.clone()),
            projections.cloned(),
        )?)
        .map_err(Error::backend_source)
    }
    fn build_feed_forward(
        _global: &ModelArgs,
        _local: &ModelArgs,
        layer: usize,
        parameters: &Self::Parameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        let values = if let Some(values) = parameters.values.clone() {
            let spec = values.experts;
            let output_width = spec.output_dimensions();
            Some(RoutedValues {
                router: B::top_k_group_selector(values.router, context)?,
                experts: B::grouped_linear_bank(spec, context)?,
                output_width,
            })
        } else {
            None
        };
        let feed_forward = if let Some(routed) = parameters.routed.clone() {
            FeedForward::Routed {
                router: B::top_k_group_selector(routed.parameters.selector, context)?,
                experts: B::grouped_gated_product(routed.parameters.experts, context)?,
                shared: shared_mlp::<B>(routed.shared, context)?,
            }
        } else {
            FeedForward::Dense(decoder::Mlp::from_parameters(
                parameters
                    .dense
                    .as_ref()
                    .ok_or_else(|| Error::backend("missing prepared dense K2 slots"))?,
                context,
            )?)
        };
        Ok(Projections {
            layer,
            points: parameters.points.clone(),
            values,
            feed_forward,
        })
    }
}

/// Constructs an unloaded K2 block using its normalized sparse schedule.
pub fn new_block<B: GroupedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TransformerBlock<B>, Error> {
    let declaration = super::Declaration::new(args.clone())?;
    TransformerBlock::from_factory::<_, BlockFactory>(
        args,
        args,
        layer,
        &declaration.block(layer)?.common,
        context,
    )
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
    type Parameters = eredu_nn::GatedProductSpec;
    fn prepare_parameters(
        _global: &ModelArgs,
        local: &ModelArgs,
        _layer: usize,
        dense: Option<&eredu_nn::GatedProductSpec>,
        retained: Option<&Self::Parameters>,
    ) -> Result<Self::Parameters, Error> {
        if let Some(parameters) = retained {
            return parameters.with_intermediate_dimensions(local.intermediate_size);
        }
        dense
            .cloned()
            .ok_or_else(|| Error::backend("dense K2 requires retained feed-forward slots"))
    }
    fn build_feed_forward(
        _global: &ModelArgs,
        _local: &ModelArgs,
        _layer: usize,
        parameters: &Self::Parameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        decoder::Mlp::from_parameters(parameters, context)
    }
}

impl ProjectionParameters {
    pub(crate) fn prepare(
        global: &ModelArgs,
        local: &ModelArgs,
        _layer: usize,
        dense: Option<&eredu_nn::GatedProductSpec>,
        retained: Option<&Self>,
    ) -> Result<Self, Error> {
        if let Some(parameters) = retained {
            return parameters.clone().with_local_geometry(local);
        }
        if global.is_moe() {
            return Err(Error::backend(
                "routed K2 requires retained projection declarations",
            ));
        }
        Ok(Self {
            dense: Some(
                dense.cloned().ok_or_else(|| {
                    Error::backend("dense K2 requires retained feed-forward slots")
                })?,
            ),
            routed: None,
            values: None,
            points: None,
        })
    }
}

impl ProjectionParameters {
    fn with_local_geometry(mut self, local: &ModelArgs) -> Result<Self, Error> {
        self.dense = self
            .dense
            .map(|spec| spec.with_intermediate_dimensions(local.intermediate_size))
            .transpose()?;
        self.routed = self
            .routed
            .map(|spec| spec.with_local_geometry(local))
            .transpose()?;
        self.values = self
            .values
            .map(|spec| spec.with_local_geometry(local))
            .transpose()?;
        eredu_nn::validate_logical_parameter_scope(
            &eredu_nn::DeclareLogicalParameters::logical_parameters(&self)?,
        )
        .map_err(Error::backend_source)?;
        Ok(self)
    }
}
