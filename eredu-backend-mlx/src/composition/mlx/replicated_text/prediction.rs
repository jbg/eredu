use super::*;
use crate::backend::runtime::cache::MlxCacheBackend;
use eredu_checkpoint::store::SharedCheckpointSource;

#[doc(hidden)]
pub struct MlxEmbeddedPredictionMaterializer;

#[doc(hidden)]
pub struct MlxPredictionMaterializationContext<'a> {
    store: SharedCheckpointSource,
    stream: &'a Stream,
}

impl<'a> MlxPredictionMaterializationContext<'a> {
    pub(super) fn new(store: SharedCheckpointSource, stream: &'a Stream) -> Self {
        Self { store, stream }
    }
}

impl
    eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization<MlxNeuralBackend>
    for MlxEmbeddedPredictionMaterializer
{
    type Error = Error;
    type Parameters = MlxNeuralBackend;
    type PoolingStorage = MlxCacheBackend;
    type SequentialStorage = crate::backend::runtime::cache::MlxCacheBackend;
    type ModelCacheStorage = MlxCacheBackend;
    type Context<'a> = MlxPredictionMaterializationContext<'a>;

    fn bind_source<'a>(
        source: SharedCheckpointSource,
        context: &'a Stream,
        _source_context: &'a Stream,
    ) -> Self::Context<'a> {
        MlxPredictionMaterializationContext::new(source, context)
    }

    fn complete_prediction_values<'a>(
        values: impl IntoIterator<Item = &'a MlxTensor>,
        _context: &Stream,
    ) -> Result<(), eredu_core::BackendFailure> {
        let token_validations = active_token_validation_arrays();
        async_eval_with_event(
            values
                .into_iter()
                .map(|value| value.as_array())
                .chain(token_validations.iter()),
        )
        .and_then(|completion| completion.synchronize())
        .and_then(|()| validate_active_token_validations())
        .map_err(eredu_core::BackendFailure::from_error)
    }

    fn parameter_source<'a>(
        context: &'a Self::Context<'_>,
    ) -> (&'a SharedCheckpointSource, &'a Stream) {
        (&context.store, context.stream)
    }

    type ModuleParameters = MlxNeuralBackend;
    type ModuleScope = crate::backend::runtime::execution::generic::MlxPredictionTransfer;

    type ResidencyMechanisms = crate::backend::runtime::residency::manager::MlxResidencyResources;

    type ModuleContext = Stream;

    fn retain_module_context(context: &Stream) -> Stream {
        context.clone()
    }

    fn begin_module(
        transfer: crate::backend::runtime::residency::manager::ResidentTransfer,
        context: &Stream,
    ) -> Result<Self::ModuleScope, Error> {
        Self::ModuleScope::begin(transfer, context)
    }

    fn module_weights(scope: &mut Self::ModuleScope) -> Result<Vec<(String, MlxTensor)>, Error> {
        scope.weights()
    }

    fn finish_module(scope: Self::ModuleScope, roots: Vec<MlxTensor>) -> Result<(), Error> {
        scope.finish(roots)
    }
}

impl eredu_core::checkpoint_mechanisms::NativePredictionRead for MlxTextValues {
    type Reader<'a> = MlxPredictionMaterializationContext<'a>;

    fn prediction_reader(store: SharedCheckpointSource, queue: &Stream) -> Self::Reader<'_> {
        MlxPredictionMaterializationContext::new(store, queue)
    }
}
