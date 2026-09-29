//! Flash-Next operations plugged into the shared prediction lifecycle.
use super::*;
use crate::qwen4_exp::{mtp, target::TargetModel};
use eredu_nn::{residual_streams::ResidualStreamGeometry, EmbeddingOperator};
use eredu_runtime::{RuntimeState, TensorParallelParameterProvider};

/// Target-owned vocabulary access for prediction. The target session keeps these
/// physical parameters resident across the complete prediction operation.
pub trait Qwen4PredictionTarget<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    /// Looks up base-width embeddings without expanding target residual streams.
    fn prediction_embedding(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>;
    /// Projects prediction-collapsed hidden values without applying target readout.
    fn prediction_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error>;
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> Qwen4PredictionTarget<B>
    for TargetModel<B>
{
    fn prediction_embedding(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.decoder
            .static_modules_mut()
            .embeddings
            .forward(tokens, context)
    }
    fn prediction_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.decoder.static_modules_mut().project_instrumented(
            hidden,
            None,
            context,
            instrumentation,
        )
    }
}

impl<A, B> Qwen4PredictionTarget<B> for crate::composite_execution::PreparedCompositeArchitecture<A>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    A: Qwen4PredictionTarget<B>,
{
    fn prediction_embedding(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.inner_mut().prediction_embedding(tokens, context)
    }
    fn prediction_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.inner_mut()
            .prediction_logits(hidden, context, instrumentation)
    }
}

/// Typed prediction strategy for the common speculative driver. Vocabulary stays
/// with the target; providers and module owners are independent of lane snapshots.
/// Construction of selected owners and public family admission are separate steps.
pub struct MaterializedQwen4Prediction<B, M, P>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    units: Vec<M::Module<mtp::PredictionUnit<B>>>,
    shared: M::Module<mtp::PredictionShared<B>>,
    state: M::ModelState,
    provider: P,
    geometry: ResidualStreamGeometry,
}
impl<B, M, P> MaterializedQwen4Prediction<B, M, P>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    <M::ModelState as PredictionModelState<B>>::LayerState: eredu_runtime::RuntimeAppendStreams<B>,
    P: TensorParallelParameterProvider<B>,
    P::Error: std::fmt::Display,
{
    /// Live provider telemetry and residency ownership; lane snapshots never clone it.
    pub fn provider(&self) -> &P {
        &self.provider
    }
    /// Pairs materialized owners with the exact separately constructed state.
    /// This does not select residency or acquire/reconstruct checkpoint sources.
    pub fn new(
        mut units: Vec<M::Module<mtp::PredictionUnit<B>>>,
        mut shared: M::Module<mtp::PredictionShared<B>>,
        state: M::ModelState,
        provider: P,
    ) -> Result<Self, eredu_nn::Error> {
        let geometry = shared.as_mut().geometry();
        if units.is_empty() || state.layout().len() != units.len() {
            return Err(eredu_nn::Error::backend(
                "Qwen4 prediction owners and state differ",
            ));
        }
        for (depth, unit) in units.iter_mut().enumerate() {
            let spec = unit.as_mut().specification();
            if spec.depth != depth
                || spec.feed_forward.residual.geometry != geometry
                || state.layout().layers().get(depth) != Some(&spec.mixer.state_policy()?)
            {
                return Err(eredu_nn::Error::backend(
                    "Qwen4 prediction owner/state geometry mismatch",
                ));
            }
        }
        Ok(Self {
            units,
            shared,
            state,
            provider,
            geometry,
        })
    }
}

struct Qwen4PredictionOperation<'a, 'o, B, M, P>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    unit: &'a mut M::Module<mtp::PredictionUnit<B>>,
    shared: &'a mut M::Module<mtp::PredictionShared<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
    provider: &'a mut P,
    observer: Option<&'o mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
}
impl<A, B, S, M, P> eredu_runtime::PredictionTargetOperation<A, B, S>
    for Qwen4PredictionOperation<'_, '_, B, M, P>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error> + Qwen4PredictionTarget<B>,
    M: PredictionExtensionMaterializer<B>,
    <M::ModelState as PredictionModelState<B>>::LayerState: eredu_runtime::RuntimeAppendStreams<B>,
    P: TensorParallelParameterProvider<B>,
    P::Error: std::fmt::Display,
{
    type Output = crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>;
    fn apply(
        self,
        architecture: &mut A,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        let state = self
            .state
            .prediction_layers_mut()
            .get_mut(self.depth)
            .ok_or_else(|| eredu_nn::Error::backend("Qwen4 prediction state is too shallow"))?;
        let path = format!("mtp.layers.{}.prediction", self.depth);
        let mut instrumentation = match self.observer {
            Some(observer) => crate::decoder::ComponentInstrumentation::new(&path, observer),
            None => crate::decoder::ComponentInstrumentation::disabled(),
        };
        M::invoke_module_with_shared(self.unit, Some(self.shared), context, |unit, shared| {
            let outcome = (|| {
                let embedded = architecture.prediction_embedding(self.tokens, context)?;
                let shared = shared.expect("Qwen4 declares shared prediction weights");
                let input = mtp::PredictionInput {
                    embeddings: &embedded,
                    residual: self.hidden,
                    visible: None,
                    rotary: None,
                };
                let output = match parallel {
                    Some(parallel) => unit.forward_parallel(
                        shared,
                        input,
                        state,
                        self.provider,
                        parallel,
                        context,
                        &mut instrumentation,
                    ),
                    None => unit.forward(
                        shared,
                        input,
                        state,
                        self.provider,
                        context,
                        &mut instrumentation,
                    ),
                }?;
                let logits = instrumentation.with_scope("readout", |instrumentation| {
                    architecture.prediction_logits(&output.hidden, context, instrumentation)
                })?;
                let logits = instrumentation.apply("logits", logits)?;
                Ok(crate::speculative_execution::EmbeddedPredictionOutput {
                    logits,
                    capture: output.capture,
                    tokens: self.tokens.clone(),
                })
            })();
            prediction_invocation(
                outcome,
                eredu_runtime::RuntimeLayerState::<B>::retained_values(state),
                |output| [&output.logits, &output.capture, &output.tokens],
            )
        })
    }
}
impl<A, B, M, P> executor_sealed::Sealed<A> for MaterializedQwen4Prediction<B, M, P>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    <M::ModelState as PredictionModelState<B>>::LayerState: eredu_runtime::RuntimeAppendStreams<B>,
    P: eredu_runtime::TensorParallelParameterProvider<B>,
    P::Error: std::fmt::Display,
    A: Qwen4PredictionTarget<B>,
{
}

impl<A, B, M, P> MaterializedPredictionExecutor<A, B, M> for MaterializedQwen4Prediction<B, M, P>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    <M::ModelState as PredictionModelState<B>>::LayerState: eredu_runtime::RuntimeAppendStreams<B>,
    P: eredu_runtime::TensorParallelParameterProvider<B>,
    P::Error: std::fmt::Display,
    M::ModelState: 'static,
    A: Qwen4PredictionTarget<B>,
{
    type LaneState = M::ModelState;

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn complete_state(
        &self,
        state: &mut Self::LaneState,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        M::complete_prediction_values(
            state
                .prediction_layers_mut()
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
        )
    }

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        visitor.visit::<crate::qwen4_exp::mtp::PredictionShared<B>>(0, &mut self.shared)?;
        for (ordinal, module) in self.units.iter_mut().enumerate() {
            visitor.visit::<crate::qwen4_exp::mtp::PredictionUnit<B>>(ordinal + 1, module)?;
        }
        Ok(())
    }

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::model_snapshot_estimate(state)
    }
    fn memory_observation(
        &self,
        state: &Self::LaneState,
        additional: u64,
    ) -> Option<eredu_core::speculative::SpeculativePredictionMemoryObservation> {
        M::model_memory_observation(state, additional)
    }
    fn snapshot(
        &self,
        state: &Self::LaneState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        M::model_snapshot(state, context)
    }

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn maximum_prefill_chunk_tokens(
        &self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
    ) -> Option<usize> {
        // Joint contract construction admits this sequence ceiling against both
        // target and prediction limits before their native owners are bound.
        let capture_tokens = selected
            .requirements()
            .capture()
            .entries()
            .first()
            .and_then(|entry| entry.shape().get(1))
            .copied()
            .unwrap_or(0);
        Some(capture_tokens)
    }

    fn logical_capture_shapes(
        &self,
        physical_shape: &[i32],
    ) -> Result<Vec<Vec<usize>>, eredu_runtime::SpeculativeCaptureError> {
        self.geometry
            .validate_streams(physical_shape)
            .map_err(|_| eredu_runtime::SpeculativeCaptureError::ShapeMismatch)?;
        Ok(vec![physical_shape.iter().map(|d| *d as usize).collect()])
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prefill<S, I>(
        &mut self,
        invoker: &mut I,
        target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, target_capture, hidden, tokens, lane, None)
    }

    fn prefill_observed<S, I>(
        &mut self,
        invoker: &mut I,
        _target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        mut observer: Option<
            &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
        >,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        if self.units.len() != lane.prediction_layers_mut().len() {
            return Err(I::invalid("Qwen4 prediction units and state differ".into()));
        }
        self.units
            .iter_mut()
            .enumerate()
            .try_for_each(|(depth, unit)| {
                invoker
                    .invoke(Qwen4PredictionOperation::<B, M, P> {
                        unit,
                        shared: &mut self.shared,
                        hidden,
                        tokens,
                        depth,
                        state: lane,
                        provider: &mut self.provider,
                        observer: observer.as_mut().map(|observer| {
                            &mut **observer
                                as &mut dyn eredu_runtime::ActivationObserver<
                                    B::Tensor,
                                    eredu_nn::Error,
                                >
                        }),
                    })
                    .map(|_| ())
            })
    }

    fn logits<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.logits_observed::<S, I>(invoker, hidden, token, draft_index, lane, None)
    }

    fn logits_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        let count = self.units.len();
        let unit = self.units.get_mut(draft_index).ok_or_else(|| {
            I::invalid(format!(
                "prediction depth {draft_index} exceeds {count} units"
            ))
        })?;
        invoker
            .invoke(Qwen4PredictionOperation::<B, M, P> {
                unit,
                shared: &mut self.shared,
                hidden,
                tokens: token,
                depth: draft_index,
                state: lane,
                provider: &mut self.provider,
                observer,
            })
            .map(|output| (output.logits, output.capture))
    }
    fn advance_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, hidden, hidden, tokens, lane, observer)
    }
}

/// Failure while assembling selected prediction state and immutable owners.
#[derive(Debug, thiserror::Error)]
pub enum PredictionConstructionError<E> {
    /// Native state construction failed before parameter materialization.
    #[error("prediction state construction failed: {0}")]
    State(#[source] E),
    /// Binding a selected immutable parameter owner failed.
    #[error("prediction weight construction failed: {0}")]
    Weights(#[source] E),
    /// Constructed state or parameter geometry violated the retained contract.
    #[error(transparent)]
    Contract(#[from] eredu_nn::Error),
}

/// Exact prediction parameter handoff after joint target/auxiliary weight selection.
/// Independently selected state and append bounds remain available for generic native
/// construction; immutable parameter materialization does not allocate that state.
pub struct PreparedQwen4PredictionWeights<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    source: eredu_checkpoint::store::SharedCheckpointSource,
    global_spec: std::sync::Arc<mtp::PredictionSpec>,
    parameters: std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>,
    placement: Option<(
        super::PredictionPlacementSlot,
        std::sync::Arc<super::PreparedPredictionPlacement>,
    )>,
    spec: mtp::PredictionSpec,
    state: eredu_runtime::SelectedStateRealization,
    layout: Option<std::sync::Arc<LocalModelLayout>>,
    shared: PreparedPredictionUnit<mtp::PredictionShared<B>>,
    units: Vec<PreparedPredictionUnit<mtp::PredictionUnit<B>>>,
}

impl<B, A> PreparedRoutedPrediction<B, A> for PreparedQwen4PredictionWeights<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend
        + 'static,
    A: Qwen4PredictionTarget<B>,
{
    type Executor<M, P>
        = MaterializedQwen4Prediction<B, M, P>
    where
        M: PredictionExtensionMaterializer<B> + 'static,
        <M::ModelState as PredictionModelState<B>>::LayerState:
            eredu_runtime::RuntimeAppendStreams<B>,
        P: TensorParallelParameterProvider<B> + 'static,
        P::Error: std::fmt::Display;

    fn source(&self) -> &eredu_checkpoint::store::SharedCheckpointSource {
        &self.source
    }

    fn banks(&self) -> Vec<eredu_runtime::RoutedBankId> {
        self.spec
            .units
            .iter()
            .map(|unit| unit.feed_forward.feed_forward.bank)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn materialize<M>(
        self,
        context: &mut M::Context<'_>,
        realize_state: impl FnOnce(
            &mut M::Context<'_>,
            &eredu_runtime::SelectedStateRealization,
        ) -> Result<M::ModelState, M::Error>,
    ) -> Result<
        Self::Executor<M, eredu_runtime::ResidentExpertProvider>,
        PredictionConstructionError<M::Error>,
    >
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
        M: PredictionExtensionMaterializer<B> + 'static,
        <M::ModelState as PredictionModelState<B>>::LayerState:
            eredu_runtime::RuntimeAppendStreams<B>,
    {
        self.materialize_executor::<M, _>(
            context,
            eredu_runtime::ResidentExpertProvider,
            realize_state,
        )
    }

    fn with_provider<M, P>(
        executor: Self::Executor<M, eredu_runtime::ResidentExpertProvider>,
        provider: P,
    ) -> Self::Executor<M, P>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
        M: PredictionExtensionMaterializer<B> + 'static,
        <M::ModelState as PredictionModelState<B>>::LayerState:
            eredu_runtime::RuntimeAppendStreams<B>,
        P: TensorParallelParameterProvider<B> + 'static,
        P::Error: std::fmt::Display,
    {
        MaterializedQwen4Prediction {
            units: executor.units,
            shared: executor.shared,
            state: executor.state,
            provider,
            geometry: executor.geometry,
        }
    }
}
impl<B> PreparedQwen4PredictionWeights<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    pub(crate) fn new(
        source: eredu_checkpoint::store::SharedCheckpointSource,
        source_spec: &mtp::PredictionSpec,
        spec: mtp::PredictionSpec,
        partitions: Option<(
            &mtp::PredictionTensorPartition,
            &mtp::PredictionTensorPartition,
        )>,
        state: eredu_runtime::SelectedStateRealization,
        tasks: &[ReplicatedTextMaterializationTask],
        residency: eredu_runtime::LayerWeightResidency,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, eredu_core::artifact::ArtifactError> {
        let parameters = std::sync::Arc::new(prediction_parameters::<B>(&spec, context)?);
        let global_spec = std::sync::Arc::new(spec.clone());
        let (layout, source_layout) = match partitions {
            Some((source_partition, partition)) => {
                if source_partition.rank() != partition.rank()
                    || source_partition.ranks() != partition.ranks()
                {
                    return Err(invalid(
                        "prediction source and executable tensor coordinates differ",
                    ));
                }
                let local = prediction_parameters::<B>(partition.local_spec(), context)?;
                let layout = partition
                    .local_layout(&parameters, &local)
                    .map_err(|error| invalid(error.to_string()))?;
                let source_layout = if tasks.iter().any(|task| {
                    matches!(
                        task.lowering(),
                        eredu_runtime::WeightLoweringKind::Transform
                            | eredu_runtime::WeightLoweringKind::DerivedTransform
                    )
                }) {
                    Some(std::sync::Arc::new(
                        source_partition
                            .local_layout(
                                &prediction_parameters::<B>(source_spec, context)?,
                                &prediction_parameters::<B>(
                                    source_partition.local_spec(),
                                    context,
                                )?,
                            )
                            .map_err(|error| invalid(error.to_string()))?,
                    ))
                } else {
                    None
                };
                (Some(std::sync::Arc::new(layout)), source_layout)
            }
            None => (None, None),
        };
        let spec = partitions.map_or(spec, |(_, partition)| partition.local_spec().clone());
        if source_spec.units.len() != spec.units.len() {
            return Err(invalid("prediction source and executable depths differ"));
        }
        if &spec.state_layout().map_err(|e| invalid(e.to_string()))? != state.layout() {
            return Err(invalid(
                "selected prediction state geometry differs from its weights",
            ));
        }
        B::require_operator_capabilities("qwen4_exp prediction", spec.required_operators())
            .map_err(|e| invalid(e.to_string()))?;
        let shared = PreparedPredictionUnit::new_shared(
            mtp::PredictionShared::<B>::new(source_spec, context)
                .map_err(|e| invalid(e.to_string()))?,
            mtp::PredictionShared::<B>::new(&spec, context).map_err(|e| invalid(e.to_string()))?,
            tasks,
        )?
        .with_source_layout(source_layout.clone())
        .with_residency(residency);
        let units = source_spec
            .units
            .iter()
            .zip(&spec.units)
            .enumerate()
            .map(|(depth, (source, local))| {
                let source = match partitions.filter(|_| source_layout.is_some()) {
                    Some((source_partition, _)) => {
                        source_partition.construct_unit::<B>(depth, context)
                    }
                    None => mtp::PredictionUnit::<B>::new(source.clone(), context),
                }
                .map_err(|error| invalid(error.to_string()))?;
                let local = match partitions {
                    Some((_, partition)) => partition.construct_unit::<B>(depth, context),
                    None => mtp::PredictionUnit::<B>::new(local.clone(), context),
                }
                .map_err(|error| invalid(error.to_string()))?;
                Ok(PreparedPredictionUnit::new(source, local, tasks)?
                    .with_source_layout(source_layout.clone())
                    .with_residency(residency))
            })
            .collect::<Result<_, eredu_core::artifact::ArtifactError>>()?;
        Ok(Self {
            source,
            global_spec,
            parameters,
            placement: None,
            spec,
            state,
            layout,
            shared,
            units,
        })
    }
    pub(crate) fn with_discovery_binding(
        mut self,
        binding: &crate::prepared_execution::PredictionBinding,
    ) -> Result<Self, eredu_core::artifact::ArtifactError> {
        if let Some((slot, topology)) = &binding.placement {
            let layout = match &self.layout {
                Some(layout) => layout.clone(),
                None => std::sync::Arc::new(
                    self.global_spec
                        .tensor_partition(0, 1)
                        .and_then(|partition| partition.projected_layout(&self.parameters))
                        .map_err(|error| invalid(error.to_string()))?,
                ),
            };
            let modules = std::iter::once(self.shared.resource_module(0))
                .chain(
                    self.units
                        .iter()
                        .enumerate()
                        .map(|(index, unit)| unit.resource_module(index + 1)),
                )
                .collect();
            let placement = super::PreparedPredictionPlacement::qwen4(
                *topology,
                self.global_spec.clone(),
                self.parameters.clone(),
                layout,
                modules,
                self.state.layout(),
            );
            self.placement = Some((slot.clone(), std::sync::Arc::new(placement)));
        }
        Ok(self)
    }
    /// Original retained source; backend context construction must use this source.
    pub fn source(&self) -> &eredu_checkpoint::store::SharedCheckpointSource {
        &self.source
    }
    /// Selected formats and independent prediction state geometry.
    pub fn spec(&self) -> &mtp::PredictionSpec {
        &self.spec
    }
    /// Exact local physical placement, including selected encoding companions.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.layout.as_deref()
    }
    /// Architecture-admitted stream identities and finite bounds.
    pub fn streams(&self) -> &[eredu_runtime::AppendStreamBinding] {
        self.state.append_streams()
    }
    /// Exact selected mechanisms for independent prediction-state construction.
    pub fn state(&self) -> &eredu_runtime::SelectedStateRealization {
        &self.state
    }
    /// Realizes the independently selected state before binding immutable owners,
    /// then installs both with the provider into the shared prediction executor.
    /// The native callback consumes exact mechanisms and retained stream bounds.
    pub fn materialize_executor<M, P>(
        self,
        context: &mut M::Context<'_>,
        provider: P,
        realize_state: impl FnOnce(
            &mut M::Context<'_>,
            &eredu_runtime::SelectedStateRealization,
        ) -> Result<M::ModelState, M::Error>,
    ) -> Result<MaterializedQwen4Prediction<B, M, P>, PredictionConstructionError<M::Error>>
    where
        M: PredictionExtensionMaterializer<B>,
        <M::ModelState as PredictionModelState<B>>::LayerState:
            eredu_runtime::RuntimeAppendStreams<B>,
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let state =
            realize_state(context, &self.state).map_err(PredictionConstructionError::State)?;
        if state.layout() != self.state.layout() {
            return Err(eredu_nn::Error::backend(
                "native prediction state differs from selected geometry",
            )
            .into());
        }
        let placement = self.placement.clone();
        let (shared, units) = self
            .materialize::<M>(context)
            .map_err(PredictionConstructionError::Weights)?;
        let executor = MaterializedQwen4Prediction::new(units, shared, state, provider)?;
        if let Some((slot, placement)) = placement {
            slot.set(placement).map_err(|_| {
                eredu_nn::Error::backend(
                    "prediction placement was already bound for these prepared sources",
                )
            })?;
        }
        Ok(executor)
    }
    /// Materializes only immutable owners with the existing completion/residency
    /// mechanism. No vocabulary duplicate or native state is allocated here.
    pub fn materialize<M>(
        self,
        context: &mut M::Context<'_>,
    ) -> Result<
        (
            M::Module<mtp::PredictionShared<B>>,
            Vec<M::Module<mtp::PredictionUnit<B>>>,
        ),
        M::Error,
    >
    where
        M: PredictionExtensionMaterializer<B>,
    {
        let shared = M::materialize_module(context, self.shared, self.layout.as_deref())?;
        let units = self
            .units
            .into_iter()
            .map(|unit| M::materialize_module(context, unit, self.layout.as_deref()))
            .collect::<Result<_, _>>()?;
        Ok((shared, units))
    }
}

/// Describes the same physical prediction owners used by materialization. The
/// retained partition validates every parameter and encoding companion against
/// this independently constructed local description.
fn prediction_parameters<B>(
    spec: &mtp::PredictionSpec,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<eredu_runtime::ArchitectureParameterDescription, eredu_core::artifact::ArtifactError>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
{
    use eredu_runtime::{
        module_parameter_group, ArchitectureParameterDescription, ExecutionGraph,
        ExecutionGroupSpec, ExecutionUnitLayout, MemberSharding, OwnedParameterGroupSpec,
        ParameterGroupOwner, ParameterRole,
    };
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("prediction")], "prediction")
        .map_err(|error| invalid(error.to_string()))?;
    let layout = ExecutionUnitLayout::new(&graph, [spec.units.len()])
        .map_err(|error| invalid(error.to_string()))?;
    let shared = mtp::PredictionShared::<B>::new(spec, context)
        .map_err(|error| invalid(error.to_string()))?;
    let shared = module_parameter_group(
        "prediction.shared",
        ParameterRole::Replicated,
        &shared,
        |_, _| Ok(MemberSharding::Replicated),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let mut expected = vec![shared.clone()];
    let mut owned = vec![OwnedParameterGroupSpec::new(
        ParameterGroupOwner::static_role("prediction_shared"),
        shared,
    )];
    for (depth, unit) in spec.units.iter().enumerate() {
        let module = mtp::PredictionUnit::<B>::new(unit.clone(), context)
            .map_err(|error| invalid(error.to_string()))?;
        let group = module_parameter_group(
            &format!("mtp.layers.{depth}"),
            ParameterRole::Replicated,
            &module,
            |_, _| Ok(MemberSharding::Replicated),
        )
        .map_err(|error| invalid(error.to_string()))?;
        expected.push(group.clone());
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(layout.group_id(0).unwrap().clone(), depth),
            group,
        ));
    }
    ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
        .map_err(|error| invalid(error.to_string()))
}

/// Authors the capture and cache contract from the retained selected prediction.
pub(crate) fn speculative_contract(
    spec: &crate::qwen4_exp::mtp::PredictionSpec,
    target_identity: &str,
    target_limits: crate::qwen4_exp::target::TargetLimits,
    request: super::EmbeddedSpeculativeContractRequest,
) -> Result<super::EmbeddedSpeculativeContract, eredu_core::artifact::ArtifactError> {
    use super::*;
    if request.target.as_str() != target_identity {
        return Err(invalid(
            "prediction target identity differs from retained selection",
        ));
    }
    if request.topology.topology().data() != 1 {
        return Err(invalid("Qwen4 prediction requires data-parallel size one"));
    }
    let batch = request.maximum_batch_size.get();
    let sequence = request.maximum_sequence_length.get();
    if batch > target_limits.qsa.batch as usize
        || batch > spec.limits.qsa.batch as usize
        || sequence > target_limits.qsa.tokens as usize
        || sequence > spec.limits.qsa.tokens as usize
        || batch
            .checked_mul(sequence)
            .is_none_or(|n| n > target_limits.invocation_tokens)
        || request.maximum_draft_tokens.get() >= sequence
    {
        return Err(invalid(
            "prediction capture bounds exceed admitted invocation limits",
        ));
    }
    let geometry = spec.fusion.geometry;
    let depth = NonZeroUsize::new(spec.units.len())
        .ok_or_else(|| invalid("prediction depth must be positive"))?;
    let state = spec.state_layout().map_err(|e| invalid(e.to_string()))?;
    finish_embedded_speculative_contract(
        EmbeddedFamilyContract {
            family: "qwen4-exp-mtp",
            class: SpeculativeStrategyClass::EmbeddedSequential,
            architecture_capacity: depth,
            capture: EmbeddedCaptureContract::Single {
                path: "target.final_residual".into(),
                shape: vec![
                    batch,
                    sequence,
                    geometry.streams() as usize,
                    geometry.hidden_size() as usize,
                ],
                observation: "prediction.target_capture".into(),
            },
            state_components: vec![format!("prediction.state/{state:?}")],
            additional_mechanisms: vec![SpeculativeMechanism::GroupedNeuralOperations],
            strategy_detail: format!("spec={spec:?}"),
        },
        request,
    )
}
