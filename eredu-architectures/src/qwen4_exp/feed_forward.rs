//! Released routed/shared SwiGLU surrounded by gated residual transport.
use super::{config::Config, residual::mixer_spec};
use crate::{
    decoder::ComponentInstrumentation,
    shared_routed::{
        SharedRoutedGatedProduct, SharedRoutedGatedProductParameters, SharedRoutedGatedProductSpec,
    },
};
use eredu_nn::{
    residual_streams::{GatedResidual, GatedResidualSpec},
    DeclareLogicalParameters, DistributedNeuralBackend, Error, GatedProductPolicy, GroupReduction,
    GroupScoring, GroupedGatedProductSpec, GroupedNeuralBackend, LinearFormatSpec, LinearSpec,
    OutputGateActivation, ParameterSpec, RoutingArithmetic, RoutingPrecision, Tensor,
    TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ParameterProvider, RoutedBankId, RoutedObservationPoints, TensorParallelParameterProvider,
};

/// Canonical packed bank; physical layouts and encodings are explicit source extensions.
/// The format callback receives the projection's component row origins; source
/// adapters retain the actual stored layout independently of the logical fusion.
pub fn expert_spec(
    config: &Config,
    root: &str,
    mut format: impl FnMut(&str, eredu_nn::LinearRowLayout) -> Result<LinearFormatSpec, Error>,
) -> Result<GroupedGatedProductSpec, Error> {
    let mut projection = |suffix: &str, row_layout| {
        let name = format!("{root}.{suffix}");
        let mut parameter = ParameterSpec::trainable(&name).map_err(Error::backend)?;
        parameter.group = Some(root.into());
        eredu_nn::GroupedProjectionSpec::new(parameter, None, format(&name, row_layout)?)
    };
    GroupedGatedProductSpec::new(
        config.experts.count,
        config.hidden_size,
        config.experts.intermediate,
        config.hidden_size,
        GatedProductPolicy::ordinary_silu(),
        eredu_nn::GatedProductGroupLayout::Packed {
            gate_up: projection(
                "gate_up_proj",
                eredu_nn::LinearRowLayout::equal_partitions(2)?,
            )?,
            down: projection("down_proj", eredu_nn::LinearRowLayout::Contiguous)?,
        },
    )
}

/// Formats and independent expert layouts are retained from checkpoint preparation.
/// Invocation layer and bank ownership are supplied separately by construction.
pub fn feed_forward_parameters(
    config: &Config,
    root: &str,
    experts: GroupedGatedProductSpec,
    mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
) -> Result<SharedRoutedGatedProductParameters, Error> {
    if experts.group_count() != config.experts.count
        || experts.input_dimensions() != config.hidden_size
        || experts.output_dimensions() != config.hidden_size
        || experts.intermediate_dimensions() != config.experts.intermediate
        || experts.policy() != GatedProductPolicy::ordinary_silu()
    {
        return Err(Error::backend(
            "qwen4_exp prepared expert geometry or equation mismatch",
        ));
    }
    let parameter = |name: &str| -> Result<ParameterSpec, Error> {
        let mut parameter = ParameterSpec::trainable(name).map_err(Error::backend)?;
        parameter.group = Some(root.into());
        Ok(parameter)
    };
    let router_name = format!("{root}.gate.weight");
    let router = TopKGroupSelectorSpec::new(
        config.hidden_size,
        parameter(&router_name)?,
        format(&router_name)?,
        TopKGroupSelectionSpec::new(
            config.experts.count,
            config.experts.selected,
            GroupScoring::Softmax,
            config.experts.renormalize,
        )?,
    )?
    .with_arithmetic(RoutingArithmetic {
        projection: RoutingPrecision::Preserve,
        scores: RoutingPrecision::Float32,
        coefficients: RoutingPrecision::Input,
    });
    let mut linear = |suffix: &str, input, output| {
        let name = format!("{root}.{suffix}.weight");
        Ok::<_, Error>(LinearSpec {
            input,
            output,
            weight: parameter(&name)?,
            bias: None,
            format: format(&name)?,
        })
    };
    let spec = SharedRoutedGatedProductParameters {
        routed: eredu_nn::RoutedGatedProductSpec {
            selector: router,
            // The reference accumulates expert writes in increasing expert-ID order.
            experts: experts.with_reduction(GroupReduction::SequentialGroupOrder),
        },
        shared: eredu_nn::SplitGatedProductSpec::new(
            linear(
                "shared_expert.gate_proj",
                config.hidden_size,
                config.experts.shared_intermediate,
            )?,
            linear(
                "shared_expert.up_proj",
                config.hidden_size,
                config.experts.shared_intermediate,
            )?,
            linear(
                "shared_expert.down_proj",
                config.experts.shared_intermediate,
                config.hidden_size,
            )?,
            GatedProductPolicy::ordinary_silu(),
        )?,
        shared_gate: linear("shared_expert_gate", config.hidden_size, 1)?,
        gate_activation: OutputGateActivation::Sigmoid,
    };
    spec.validate()?;
    Ok(spec)
}

/// Small replicated residual mixer plus independently addressable expert bank.
#[derive(Debug, Clone)]
pub struct FeedForwardSublayerSpec {
    /// Full-width residual geometry retained across pipeline boundaries.
    pub residual: GatedResidualSpec,
    /// Caller may localize this specification after global artifact validation.
    pub feed_forward: SharedRoutedGatedProductSpec,
}
impl FeedForwardSublayerSpec {
    /// Constructs global target or prediction identities beneath a layer root.
    pub fn from_config(
        config: &Config,
        layer: usize,
        root: &str,
        experts: GroupedGatedProductSpec,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        let spec = Self {
            residual: mixer_spec(
                config,
                &format!("{root}.mlp_hyper_connection"),
                true,
                &mut format,
            )?,
            feed_forward: SharedRoutedGatedProductSpec {
                layer,
                bank: RoutedBankId::new(0),
                parameters: feed_forward_parameters(
                    config,
                    &format!("{root}.mlp"),
                    experts,
                    format,
                )?,
            },
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Checks both sides of the residual boundary before constructing operators.
    pub fn validate(&self) -> Result<(), Error> {
        self.logical_parameters().map(|_| ())
    }
    fn validate_geometry(&self) -> Result<(), Error> {
        self.residual.validate()?;
        self.feed_forward.validate()?;
        if self.residual.injection.is_none()
            || self.residual.geometry.hidden_size()
                != self
                    .feed_forward
                    .parameters
                    .routed
                    .experts
                    .input_dimensions()
            || self.residual.geometry.hidden_size()
                != self
                    .feed_forward
                    .parameters
                    .routed
                    .experts
                    .output_dimensions()
        {
            return Err(Error::backend(
                "qwen4_exp feed-forward residual geometry mismatch",
            ));
        }
        Ok(())
    }
}

impl DeclareLogicalParameters for FeedForwardSublayerSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = self.residual.logical_parameters()?;
        parameters.extend(self.feed_forward.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

/// Uses the common residency provider for resident, bounded and exchanged experts.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct FeedForwardSublayer<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    residual: GatedResidual<B>,
    feed_forward: SharedRoutedGatedProduct<B>,
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> FeedForwardSublayer<B> {
    /// Constructs parameters without acquiring the expert payloads.
    pub fn new(
        spec: FeedForwardSublayerSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            residual: GatedResidual::new(spec.residual, context)?,
            feed_forward: SharedRoutedGatedProduct::new(spec.feed_forward, context)?,
        })
    }
    /// Runs routing observations and component interventions on consumed values.
    pub fn forward<P>(
        &mut self,
        hidden: &B::Tensor,
        points: RoutedObservationPoints,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let residual = self.residual.forward(hidden, context)?;
        let input = instrumentation.apply("feed_forward.input", residual.mixed.clone())?;
        let output = match instrumentation.observer() {
            Some(observer) => self
                .feed_forward
                .forward_observed_with_provider(points, &input, context, provider, observer)?,
            None => self
                .feed_forward
                .forward_with_provider(&input, context, provider)?,
        };
        let output = instrumentation.apply("feed_forward.write", output)?;
        let output = instrumentation.apply("feed_forward.output", output)?;
        instrumentation.apply("feed_forward.residual", residual.inject(&output, context)?)
    }
    /// Reduces local projections before injecting the complete result into streams.
    pub fn forward_parallel<P>(
        &mut self,
        hidden: &B::Tensor,
        points: RoutedObservationPoints,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let residual = self.residual.forward(hidden, context)?;
        let input = instrumentation.apply("feed_forward.input", residual.mixed.clone())?;
        let output = match instrumentation.observer() {
            Some(observer) => self
                .feed_forward
                .forward_tensor_parallel_observed_with_provider(
                    points, &input, parallel, context, provider, observer,
                )?,
            None => eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(
                self.feed_forward
                    .forward_tensor_parallel_with_provider(&input, parallel, context, provider)?,
                parallel,
                context,
            )?,
        };
        let output = instrumentation.apply("feed_forward.write", output)?;
        let output = instrumentation.apply("feed_forward.output", output)?;
        instrumentation.apply("feed_forward.residual", residual.inject(&output, context)?)
    }
}
