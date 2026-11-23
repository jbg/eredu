//! Eager preparation of one bounded checkpoint conversion tile.

use std::num::NonZeroUsize;

use candle_core::{CpuStorage, Storage};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::CheckpointSource,
    word_quantization::{self, WordTileOutput},
    StoredDtype, WeightQuantization,
};
use eredu_core::{
    checkpoint_mechanisms::{
        bounded_quantization::{
            BoundedQuantizationBackend, EncodedQuantizationTile, QuantizationTileCompletion,
            QuantizationTileLimits,
        },
        CheckpointParameterBackend,
    },
    BackendFailure,
};

use crate::{CandleContext, CandleNeuralBackend};

pub(crate) const LIMITS: QuantizationTileLimits =
    QuantizationTileLimits::signed_indexed(NonZeroUsize::MIN);

pub(crate) struct CandleBoundedQuantizer {
    source: CandleContext,
    begun: bool,
}

impl CandleBoundedQuantizer {
    pub(crate) fn new() -> Self {
        Self {
            source: CandleContext::cpu(),
            begun: false,
        }
    }
}

struct EncodedTile(Vec<WordTileOutput>);

impl QuantizationTileCompletion for EncodedTile {
    fn complete(self: Box<Self>) -> Result<Box<dyn EncodedQuantizationTile>, BackendFailure> {
        Ok(self)
    }
}

impl EncodedQuantizationTile for EncodedTile {
    fn output_count(&self) -> usize {
        self.0.len()
    }

    fn output(&self, index: usize) -> Result<(StoredDtype, &[u8]), BackendFailure> {
        let output = self
            .0
            .get(index)
            .ok_or_else(|| crate::text::invalid("unknown quantization tile output"))?;
        Ok((output.dtype.clone(), &output.bytes))
    }
}

impl BoundedQuantizationBackend for CandleBoundedQuantizer {
    fn cache_identity(&self) -> Result<String, BackendFailure> {
        Ok("candle-checkpoint-word-encoder-v1".into())
    }

    fn limits(&self) -> QuantizationTileLimits {
        LIMITS
    }

    fn begin(&mut self, slots: usize, _: u64) -> Result<(), BackendFailure> {
        if self.begun || slots != 1 {
            return Err(crate::text::invalid(
                "checkpoint conversion requires one free encoder slot",
            ));
        }
        self.begun = true;
        Ok(())
    }

    fn trim(&mut self, _: u64) -> Result<(), BackendFailure> {
        Ok(())
    }

    fn submit(
        &mut self,
        slot: usize,
        input: &dyn CheckpointSource,
        recipe: &DerivedWeightRecipe,
        encoding: WeightQuantization,
        companion_dtype: RecipeDtype,
    ) -> Result<Box<dyn QuantizationTileCompletion>, BackendFailure> {
        if !self.begun || slot != 0 {
            return Err(crate::text::invalid(
                "checkpoint conversion slot is not open",
            ));
        }
        let metadata = recipe.infer(input).map_err(BackendFailure::from_error)?;
        let columns = metadata.shape().last().copied().unwrap_or(0);
        let elements = metadata
            .shape()
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d));
        let elements = elements
            .filter(|&n| n <= LIMITS.elements.get() && n > 0 && columns > 0)
            .ok_or_else(|| {
                crate::text::invalid("checkpoint conversion exceeds the admitted tile geometry")
            })?;
        if !matches!(
            metadata.dtype(),
            RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32 | RecipeDtype::F64
        ) {
            return Err(crate::text::invalid(
                "checkpoint conversion requires floating input and packed word output",
            ));
        }
        let dense = CandleNeuralBackend::materialize_recipe(recipe, input, &self.source)
            .map_err(BackendFailure::from_error)?;
        let (storage, layout) = dense.value.storage_and_layout();
        let contiguous = layout.is_contiguous();
        let native_index = |mut index: usize| {
            let mut address = layout.start_offset();
            if contiguous {
                return address + index;
            }
            for (&dim, &stride) in layout.dims().iter().zip(layout.stride()).rev() {
                address += (index % dim) * stride;
                index /= dim;
            }
            address
        };
        let encode = |value: &dyn Fn(usize) -> f32| {
            word_quantization::encode(
                elements / columns,
                columns,
                encoding,
                companion_dtype,
                |i| value(native_index(i)),
            )
            .map_err(BackendFailure::from_error)
        };
        let outputs = match &*storage {
            Storage::Cpu(CpuStorage::F16(values)) => encode(&|i| values[i].to_f32()),
            Storage::Cpu(CpuStorage::BF16(values)) => encode(&|i| values[i].to_f32()),
            Storage::Cpu(CpuStorage::F32(values)) => encode(&|i| values[i]),
            Storage::Cpu(CpuStorage::F64(values)) => encode(&|i| values[i] as f32),
            _ => Err(crate::text::invalid(
                "checkpoint conversion expected host floating values",
            )),
        }?;
        // The CPU materializer consumes acquired bytes before returning native
        // storage. The completed tile owns its exact output through publication.
        Ok(Box::new(EncodedTile(outputs)))
    }

    fn finish(&mut self) -> Result<(), BackendFailure> {
        self.begun = false;
        Ok(())
    }
}
