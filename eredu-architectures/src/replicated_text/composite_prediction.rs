//! Prediction pairing consumes the same ordinary composite constructors/handoff.
use super::*;
use crate::prediction_extension::{
    MaterializedPredictionExtension, PredictionExtensionMaterializer,
};

struct Pair<B, M, V>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    extension: MaterializedPredictionExtension<B, M>,
    visitor: V,
}

pub(crate) fn visit_prepared_composite_prediction_target_text_architecture<B, S, M, V>(
    source: &crate::prepared_sources::PreparedModelSources,
    selected: SelectedCompositeTextRealization,
    extension: MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    M: PredictionExtensionMaterializer<B>,
    V: CompositePredictionTargetVisitor<B, S, M>,
{
    let requirements =
        composite_source::Requirements::prepared::<B, V::Error>(source, &selected, context)?;
    visit_with_requirements::<B, S, M, V>(
        requirements,
        selected,
        extension,
        store,
        context,
        visitor,
    )
}

pub(super) fn visit_with_requirements<B, S, M, V>(
    requirements: composite_source::Requirements,
    selected: SelectedCompositeTextRealization,
    extension: MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    M: PredictionExtensionMaterializer<B>,
    V: CompositePredictionTargetVisitor<B, S, M>,
{
    let pair = Pair { extension, visitor };
    with_composite_construction::<B, S, _, _, _>(
        requirements,
        selected,
        store,
        context,
        pair,
        true,
        |pair| pair.visitor.construction_started(),
        |config, construction| match config {
            CompositeConfig::Inkling(args) => {
                construction.inkling_with(args, finish::<B, S, M, V, _>)
            }
            CompositeConfig::QwenHybrid(args) => {
                construction.qwen_hybrid_with(args, finish::<B, S, M, V, _>)
            }
            CompositeConfig::Gemma4(_) | CompositeConfig::Muse(_) | CompositeConfig::QwenVl(_) => {
                // Preserve the existing architecture applicability rejection;
                // none of these targets has a MaterializedPredictionTarget.
                Err(ReplicatedTextDispatchError::Architecture(
                    "composite architecture does not admit an embedded prediction extension".into(),
                ))
            }
        },
    )
}

#[inline(never)]
fn finish<B, S, M, V, A>(
    construction: CompositeTextConstruction<'_, B, S, Pair<B, M, V>>,
    architecture: A,
    source: Option<A>,
    capability: crate::capability::CapabilityEstimate,
    model_type: String,
    cache_identity: String,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    M: PredictionExtensionMaterializer<B>,
    V: CompositePredictionTargetVisitor<B, S, M>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<B, S>
        + 'static,
    crate::composite_execution::PreparedCompositeArchitecture<A>:
        crate::prediction_extension::MaterializedPredictionTarget<B>,
    A::InputPartPlan: 'static,
    A::StaticModules: Clone,
{
    if let Some(context) = B::construction_metadata(construction.context) {
        let controls = [
            size_of_val(&construction),
            size_of::<A>(),
            size_of::<Option<A>>(),
            size_of::<crate::capability::CapabilityEstimate>(),
            size_of::<String>(),
            size_of::<String>(),
            size_of::<Pair<B, M, V>>(),
            size_of::<Result<V::Output, ReplicatedTextDispatchError<V::Error>>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| {
                ReplicatedTextDispatchError::Metadata(
                    eredu_nn::workspace::WorkspaceMetadataError::Overflow.into(),
                )
            })?;
        context
            .charge_metadata(bytes)
            .map_err(|cause| ReplicatedTextDispatchError::Metadata(cause.into()))?;
    }
    construction.visit_with(
        architecture,
        source,
        capability,
        model_type,
        cache_identity,
        |pair, prepared, store| {
            visit_composite_prediction_target_architecture::<B, S, M, A, V>(
                prepared,
                pair.extension,
                store,
                pair.visitor,
            )
        },
        |pair, prepared, store| {
            visit_routed_composite_prediction_target_architecture::<B, S, M, A, V>(
                prepared,
                pair.extension,
                store,
                pair.visitor,
            )
        },
    )
}
