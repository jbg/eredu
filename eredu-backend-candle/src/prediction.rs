//! Eager storage binding for portable prediction modules.

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{
    checkpoint_mechanisms::{prediction::NativePredictionMaterialization, NativePredictionRead},
    residency::ResidencyStorageMechanisms,
    BackendFailure,
};

use crate::{
    residency::{retained_weights, CandleWeightResidency},
    text_native::CandleTextValues,
    CandleCacheStorage, CandleContext, CandleNeuralBackend, CandleTensor,
};

/// Native parameter transfer and state storage for an admitted prediction unit.
pub struct CandlePredictionMaterializer;

/// The retained checkpoint source and exact execution context for this binding.
pub struct CandlePredictionSource<'a> {
    source: SharedCheckpointSource,
    context: &'a CandleContext,
}

type Transfer = <CandleWeightResidency as ResidencyStorageMechanisms>::Transfer;

impl NativePredictionRead for CandleTextValues {
    type Reader<'a> = CandlePredictionSource<'a>;

    fn prediction_reader(
        source: SharedCheckpointSource,
        context: &CandleContext,
    ) -> Self::Reader<'_> {
        CandlePredictionSource { source, context }
    }
}

impl NativePredictionMaterialization<CandleNeuralBackend> for CandlePredictionMaterializer {
    type Error = BackendFailure;
    type Parameters = CandleNeuralBackend;
    type PoolingStorage = CandleCacheStorage;
    type SequentialStorage = CandleCacheStorage;
    type ModelCacheStorage = CandleCacheStorage;
    type Context<'a> = CandlePredictionSource<'a>;
    type ModuleParameters = CandleNeuralBackend;
    type ResidencyMechanisms = CandleWeightResidency;
    type ModuleScope = (Transfer, CandleContext);
    type ModuleContext = CandleContext;

    fn bind_source<'a>(
        source: SharedCheckpointSource,
        context: &'a CandleContext,
        _: &'a CandleContext,
    ) -> Self::Context<'a> {
        CandleTextValues::prediction_reader(source, context)
    }

    fn retain_module_context(context: &CandleContext) -> CandleContext {
        context.clone()
    }

    fn begin_module(
        transfer: Transfer,
        context: &CandleContext,
    ) -> Result<Self::ModuleScope, BackendFailure> {
        Ok((transfer, context.clone()))
    }

    fn module_weights(
        scope: &mut Self::ModuleScope,
    ) -> Result<Vec<(String, CandleTensor)>, BackendFailure> {
        retained_weights(&scope.0, 0).map_err(BackendFailure::from_error)
    }

    fn finish_module(
        (transfer, context): Self::ModuleScope,
        roots: Vec<CandleTensor>,
    ) -> Result<(), BackendFailure> {
        if let Err(error) = context.synchronize() {
            context.quarantine((transfer, roots));
            return Err(error);
        }
        Ok(())
    }

    fn complete_prediction_values<'a>(
        roots: impl IntoIterator<Item = &'a CandleTensor>,
        _: &CandleContext,
    ) -> Result<(), BackendFailure> {
        CandleContext::settle_roots(roots)
    }

    fn parameter_source<'a>(
        context: &'a Self::Context<'_>,
    ) -> (&'a SharedCheckpointSource, &'a CandleContext) {
        (&context.source, context.context)
    }
}

use eredu_core::{
    cache::mechanisms::{CacheTensorStorage, LiveCacheBackend},
    eager_completion::EagerCompletion,
    execution_control::SnapshotEstimate,
    speculative::embedded::EmbeddedPredictionOperators,
    BackendFailureKind,
};
use eredu_nn::Tensor;

/// Native values and exact eager completion for prediction invocations.
pub struct CandlePredictionOperators;

impl EmbeddedPredictionOperators for CandlePredictionOperators {
    type Tensor = CandleTensor;
    type Logits = CandleTensor;
    type Capture = CandleTextValues;
    type Completion = EagerCompletion<BackendFailure>;
    type Error = BackendFailure;

    fn failure(error: BackendFailure) -> BackendFailure {
        error
    }

    fn invalid(message: impl Into<String>) -> BackendFailure {
        BackendFailure::new(
            BackendFailureKind::InvalidInput,
            std::io::Error::other(message.into()),
        )
    }

    fn validate<T>(
        operation: impl FnOnce() -> Result<T, BackendFailure>,
    ) -> Result<T, BackendFailure> {
        operation()
    }

    fn token(token: u32, context: &CandleContext) -> Result<CandleTensor, BackendFailure> {
        context.tokens(&[token])
    }

    fn logits_estimate(value: &CandleTensor) -> Option<SnapshotEstimate> {
        Self::tensor_estimate(value)
    }

    fn tensor_estimate(value: &CandleTensor) -> Option<SnapshotEstimate> {
        let bytes = CandleCacheStorage::snapshot_tensor_costs()?
            .bytes(CandleCacheStorage::bytes(value), value.shape().len())?;
        Some(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    fn copy_tensor(
        value: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        value.compact(context).map_err(BackendFailure::from_error)
    }

    fn logits_row(
        value: &CandleTensor,
        row: usize,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        eredu_nn::sequence::row(value, row, false, context)
            .and_then(|row| row.squeeze_axes(&[0], context))
            .map_err(BackendFailure::from_error)
    }

    fn target_tokens(
        tokens: &[u32],
        destination: &CandleContext,
        _: bool,
    ) -> Result<CandleTensor, BackendFailure> {
        destination.tokens(tokens)
    }

    fn submit_verification(roots: &[&CandleTensor]) -> Result<Self::Completion, BackendFailure> {
        CandleContext::settle_roots(roots.iter().copied())?;
        Ok(EagerCompletion::completed())
    }
}

impl eredu_core::speculative::assistant::ExternalAssistantBinder for CandleContext {
    type Backend = CandleNeuralBackend;
    type Storage = Self;
    type Logits = CandleTensor;
    type ObservationError = BackendFailure;
    type Error = BackendFailure;
    type Parameters = CandleNeuralBackend;
    type ParameterReads = Self;

    fn context(&self) -> &Self {
        self
    }

    fn storage(&self) -> Self {
        self.clone()
    }

    fn recipe_workspace(
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<u64, BackendFailure> {
        eredu_core::checkpoint_mechanisms::recipe_execution::conservative_workspace_bytes(
            recipe, source,
        )
        .map_err(|detail| {
            BackendFailure::new(
                BackendFailureKind::InvalidInput,
                std::io::Error::other(detail),
            )
        })
    }

    fn parameter_reads(&self) -> Self {
        self.clone()
    }
}

/// Eager tensor and completion operations for a prepared conditioned assistant.
pub struct CandleAssistantOperators;

impl eredu_core::speculative::assistant::ExternalAssistantOperators for CandleAssistantOperators {
    type Text = CandleTextValues;
    type NeuralBackend = CandleNeuralBackend;
    type Binder = CandleContext;
    type Cache = CandleCacheStorage;
    type Capture = CandleTextValues;
    fn placement_memory(
        queue: &CandleContext,
    ) -> Result<eredu_core::capability::ParameterMemoryFacts, BackendFailure> {
        Ok(crate::memory::placement_memory(queue))
    }
    type Tensor = CandleTensor;
    type Logits = CandleTensor;
    type Completion = EagerCompletion<BackendFailure>;
    type Error = BackendFailure;

    fn failure(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
        BackendFailure::from_error(error)
    }

    fn invalid(message: String) -> BackendFailure {
        CandlePredictionOperators::invalid(message)
    }

    fn target_input(value: &CandleTensor) -> CandleTensor {
        value.clone()
    }

    fn tensor_shape(value: &CandleTensor) -> Result<Vec<usize>, BackendFailure> {
        Ok(value.value.dims().to_vec())
    }

    fn sequence_len(value: &CandleTensor) -> Result<usize, BackendFailure> {
        eredu_nn::sequence::length(value).map_err(BackendFailure::from_error)
    }

    fn into_logits(value: CandleTensor) -> CandleTensor {
        value
    }

    fn tensor_backing_bytes(values: &[&CandleTensor]) -> Result<Option<u64>, BackendFailure> {
        use candle_core::{CpuStorage, Storage};
        let bytes = values.iter().try_fold(0_u64, |total, tensor| {
            let (storage, _) = tensor.value.storage_and_layout();
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if let Storage::Metal(storage) = &*storage {
                return total.checked_add(u64::try_from(storage.buffer().length()).ok()?);
            }
            let Storage::Cpu(storage) = &*storage else {
                return None;
            };
            let bytes = match storage {
                CpuStorage::F32(v) => v.len().checked_mul(4)?,
                CpuStorage::BF16(v) => v.len().checked_mul(2)?,
                CpuStorage::F16(v) => v.len().checked_mul(2)?,
                CpuStorage::I32(v) => v.len().checked_mul(4)?,
                CpuStorage::U32(v) => v.len().checked_mul(4)?,
                CpuStorage::I64(v) => v.len().checked_mul(8)?,
                _ => return None,
            };
            total.checked_add(u64::try_from(bytes).ok()?)
        });
        Ok(bytes)
    }

    fn control_logits_estimate(logits: &CandleTensor) -> Option<SnapshotEstimate> {
        CandlePredictionOperators::logits_estimate(logits)
    }

    fn control_tensor_bytes(value: &CandleTensor) -> Option<u64> {
        CandlePredictionOperators::tensor_estimate(value).map(|estimate| estimate.copy_bytes)
    }

    fn bound_assistant_group<'a>(
        no_transport: &'a eredu_core::backend::local_text::NoTextTransport,
        _: eredu_core::CollectiveGroupId,
        _: &eredu_core::communication::CommunicationManifest,
    ) -> Result<&'a std::convert::Infallible, BackendFailure> {
        match *no_transport {}
    }

    fn sequence_row(
        value: &CandleTensor,
        row: usize,
        keep: bool,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        eredu_nn::sequence::row(value, row, keep, context).map_err(BackendFailure::from_error)
    }

    fn sequence_suffix(
        value: &CandleTensor,
        maximum: i32,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        let length = Self::sequence_len(value)?;
        let width = usize::try_from(maximum).map_err(BackendFailure::from_error)?;
        eredu_nn::sequence::span(value, length.saturating_sub(width)..length, context)
            .map_err(BackendFailure::from_error)
    }

    fn shared_prefix(
        value: &CandleTensor,
        length: i32,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        use eredu_nn::Index;
        let shape = value.shape();
        let [_, _, available, _] = shape else {
            return Err(Self::invalid("shared state must have four axes".into()));
        };
        if length < 0 {
            return Err(Self::invalid(
                "shared state prefix must be nonnegative".into(),
            ));
        }
        value
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(0, length.min(*available)),
                ],
                context,
            )
            .map_err(BackendFailure::from_error)
    }

    fn token_prefix(
        value: &CandleTensor,
        end: usize,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        eredu_nn::sequence::span(value, 0..end, context).map_err(BackendFailure::from_error)
    }

    fn target_tokens(
        tokens: &[u32],
        destination: &CandleContext,
        _: bool,
    ) -> Result<CandleTensor, BackendFailure> {
        destination.tokens(tokens)
    }

    fn tensor_root(value: &CandleTensor) -> &CandleTensor {
        value
    }

    fn can_share_tensor(value: &CandleTensor, context: &CandleContext) -> bool {
        value.value.device().same_device(&context.device)
    }

    fn copy_tensor(
        value: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        crate::input_transfer::copy_tensor(value, context)
    }

    fn submit_completion<'a>(
        roots: impl IntoIterator<Item = &'a CandleTensor>,
    ) -> Result<Self::Completion, BackendFailure> {
        CandleContext::settle_roots(roots)?;
        Ok(EagerCompletion::completed())
    }
}

impl eredu_core::backend::local_text::HostSpeculativeScoreRead for CandleTextValues {
    type Logits = CandleTensor;

    fn read_predictor_scores(
        logits: &CandleTensor,
        context: &CandleContext,
    ) -> Result<Vec<f32>, BackendFailure> {
        logits
            .to_f32_vec(context)
            .map_err(BackendFailure::from_error)
    }
}
