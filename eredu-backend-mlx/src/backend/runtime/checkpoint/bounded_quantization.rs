//! MLX submission and readback for preflighted quantization tiles.

use std::num::NonZeroUsize;

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::{stored_dtype_from_safetensors, CheckpointSource},
    StoredDtype, WeightQuantization,
};
use eredu_core::{
    checkpoint_mechanisms::bounded_quantization::{
        BoundedQuantizationBackend, EncodedQuantizationTile, QuantizationTileCompletion,
        QuantizationTileLimits,
    },
    BackendFailure,
};
use safemlx::{memory, Array, Dtype, EvaluatedArray, Stream};

use super::{
    quantization::quantize_tensor,
    recipe::MlxWeightRecipeExt,
    store::{MlxParameterMaterializationContext, WeightMaterialization},
};
use crate::backend::{error::Error, ordinary_retirement::OrdinaryRetirement};

pub(crate) struct MlxBoundedQuantizer<'a> {
    stream: &'a Stream,
    contexts: Vec<MlxParameterMaterializationContext>,
    retained_cache_bytes: u64,
    cleanup: Option<OrdinaryRetirement<AllocatorCacheCleanup>>,
}

impl<'a> MlxBoundedQuantizer<'a> {
    pub(crate) fn new(stream: &'a Stream) -> Self {
        Self {
            stream,
            contexts: Vec::new(),
            retained_cache_bytes: 0,
            cleanup: None,
        }
    }

    fn submit_tile(
        &self,
        slot: usize,
        source: &dyn CheckpointSource,
        recipe: &DerivedWeightRecipe,
        encoding: WeightQuantization,
        companion_dtype: RecipeDtype,
    ) -> Result<WeightMaterialization, Error> {
        let context = &self.contexts[slot];
        let stream = context.source_stream();
        let pending = recipe.prepare_borrowed_materialization(source, context)?;
        let (dense, leases) = pending.into_parts();
        let prepared = WeightMaterialization::prepare_retained(vec![dense], leases)?;
        let packed = quantize_tensor(&prepared.inputs()[0], encoding, stream)?;
        let companion_dtype = match companion_dtype {
            RecipeDtype::F16 => Dtype::Float16,
            RecipeDtype::BF16 => Dtype::Bfloat16,
            RecipeDtype::F32 => Dtype::Float32,
            _ => return Err(Error::Quantization("invalid affine companion dtype".into())),
        };
        let scales = if matches!(encoding, WeightQuantization::MxFp4) {
            packed.scales
        } else {
            packed.scales.as_dtype(companion_dtype, stream)?
        };
        let mut outputs = vec![
            packed.weight.contiguous(false, stream)?,
            scales.contiguous(false, stream)?,
        ];
        if let Some(biases) = packed.biases {
            outputs.push(
                biases
                    .as_dtype(companion_dtype, stream)?
                    .contiguous(false, stream)?,
            );
        }
        Ok(prepared.submit_outputs(outputs)?)
    }
}

impl BoundedQuantizationBackend for MlxBoundedQuantizer<'_> {
    fn cache_identity(&self) -> Result<String, BackendFailure> {
        let device = self
            .stream
            .get_device()
            .map_err(BackendFailure::from_error)?;
        Ok(format!(
            "mlx-bounded-quantization-v1:{}:{}:{:?}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            device.get_type().map_err(BackendFailure::from_error)?,
        ))
    }

    fn limits(&self) -> QuantizationTileLimits {
        QuantizationTileLimits {
            slots: NonZeroUsize::new(2).unwrap(),
            elements: NonZeroUsize::new(i32::MAX as usize).unwrap(),
        }
    }

    fn begin(&mut self, slots: usize, working_set_bytes: u64) -> Result<(), BackendFailure> {
        if !cfg!(target_endian = "little") {
            return Err(BackendFailure::from_error(Error::Quantization(
                "bounded SafeTensors quantization requires a little-endian host".into(),
            )));
        }
        self.contexts.clear();
        self.contexts.push(MlxParameterMaterializationContext::new(
            self.stream,
            self.stream,
        ));
        if slots > 1 {
            let device = self
                .stream
                .get_device()
                .map_err(BackendFailure::from_error)?;
            let stream =
                Stream::try_new_with_device(&device).map_err(BackendFailure::from_error)?;
            self.contexts
                .push(MlxParameterMaterializationContext::new(&stream, &stream));
        }
        self.retained_cache_bytes = (working_set_bytes / 4).min(64 * 1024 * 1024);
        self.cleanup = Some(OrdinaryRetirement::new(AllocatorCacheCleanup {
            finished: false,
        }));
        memory::clear_cache().map_err(BackendFailure::from_error)
    }

    fn trim(&mut self, available_bytes: u64) -> Result<(), BackendFailure> {
        let cached_bytes = memory::cache_memory().map_err(BackendFailure::from_error)?;
        let cached_bytes = u64::try_from(cached_bytes).map_err(BackendFailure::from_error)?;
        if cached_bytes > self.retained_cache_bytes.min(available_bytes) {
            memory::clear_cache().map_err(BackendFailure::from_error)?;
        }
        Ok(())
    }

    fn submit(
        &mut self,
        slot: usize,
        source: &dyn CheckpointSource,
        recipe: &DerivedWeightRecipe,
        encoding: WeightQuantization,
        companion_dtype: RecipeDtype,
    ) -> Result<Box<dyn QuantizationTileCompletion>, BackendFailure> {
        let pending = self
            .submit_tile(slot, source, recipe, encoding, companion_dtype)
            .map_err(BackendFailure::from_error)?;
        Ok(Box::new(pending))
    }

    fn finish(&mut self) -> Result<(), BackendFailure> {
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim_all();
        memory::clear_cache().map_err(BackendFailure::from_error)?;
        if let Some(mut cleanup) = self.cleanup.take() {
            cleanup.finished = true;
        }
        self.contexts.clear();
        Ok(())
    }
}

impl QuantizationTileCompletion for WeightMaterialization {
    fn complete(self: Box<Self>) -> Result<Box<dyn EncodedQuantizationTile>, BackendFailure> {
        let outputs = self
            .synchronize_many()
            .map_err(BackendFailure::from_error)?;
        let evaluated = Array::into_evaluated_batch(outputs).map_err(BackendFailure::from_error)?;
        Ok(Box::new(MlxPackedTile(evaluated)))
    }
}

struct MlxPackedTile(Vec<EvaluatedArray<'static>>);

impl EncodedQuantizationTile for MlxPackedTile {
    fn output_count(&self) -> usize {
        self.0.len()
    }

    fn output(&self, index: usize) -> Result<(StoredDtype, &[u8]), BackendFailure> {
        let view = self.0[index]
            .safetensors_view()
            .map_err(BackendFailure::from_error)?;
        Ok((stored_dtype_from_safetensors(view.dtype()), view.data()))
    }
}

struct AllocatorCacheCleanup {
    finished: bool,
}

impl Drop for AllocatorCacheCleanup {
    fn drop(&mut self) {
        if !self.finished {
            let _ = memory::clear_cache();
        }
    }
}
