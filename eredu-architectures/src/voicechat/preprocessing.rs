//! Exact host mel and window coefficients retained before frontend construction.

use std::{collections::BTreeMap, sync::Arc};

use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{
        CheckpointLease, CheckpointSource, EncodedTensorLease, ReadPolicy, StoreError,
        TensorReadRequest, TensorSelection,
    },
    StoredDtype,
};

use super::artifact::ParameterAdmission;

const FILTERS: &str = "stt_model.perception.preprocessor.featurizer.fb";
const WINDOW: &str = "stt_model.perception.preprocessor.featurizer.window";
const READ_VALUES: usize = 4096;

/// Host payload admission fails before a native resource or frontend is created.
#[derive(Debug, thiserror::Error)]
pub enum PreprocessingPreparationError {
    /// The admitted parameter cannot supply the released host geometry.
    #[error("invalid VoiceChat preprocessing parameter {0}")]
    Parameter(&'static str),
    /// Saved coefficients must be finite in their canonical host representation.
    #[error("nonfinite VoiceChat preprocessing coefficients in {0}")]
    Nonfinite(&'static str),
    /// An exact bounded source lease or portable conversion failed.
    #[error(transparent)]
    Source(#[from] StoreError),
}

/// Immutable canonical F32 coefficients, shared across prepared frontend owners.
#[derive(Debug, Clone)]
pub struct PreparedPreprocessing {
    window: Arc<[f32]>,
    filters: Arc<[f32]>,
}

impl PreparedPreprocessing {
    /// Unpadded 400-sample analysis window in source order.
    pub fn window(&self) -> &[f32] {
        &self.window
    }
    /// Row-major 128-by-257 mel filter bank (the source singleton batch is removed).
    pub fn filters(&self) -> &[f32] {
        &self.filters
    }
    /// Retained host payload bytes, excluding allocation and reference-count overhead.
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of_val(self.window()) + std::mem::size_of_val(self.filters())
    }

    pub(super) fn retain(
        parameters: &BTreeMap<String, ParameterAdmission>,
        source: &dyn CheckpointSource,
    ) -> Result<Self, PreprocessingPreparationError> {
        let read = |name, shape: &[usize]| {
            let admission = parameters
                .get(name)
                .filter(|parameter| parameter.output.shape == shape)
                .ok_or(PreprocessingPreparationError::Parameter(name))?;
            // Released recipes only remove/add a singleton batch dimension.
            // Reject other transforms rather than silently reading their inputs.
            let mut recipe = &admission.recipe;
            while let DerivedWeightRecipe::Reshape { input, .. } = recipe {
                recipe = input;
            }
            let DerivedWeightRecipe::Source {
                key,
                selection: TensorSelection::Full,
            } = recipe
            else {
                return Err(PreprocessingPreparationError::Parameter(name));
            };
            let count = shape.iter().product();
            let mut values = Vec::with_capacity(count);
            for start in (0..count).step_by(READ_VALUES) {
                values.extend(read_float_range(
                    source,
                    key,
                    start..(start + READ_VALUES).min(count),
                )?);
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(PreprocessingPreparationError::Nonfinite(name));
            }
            Ok::<Arc<[f32]>, PreprocessingPreparationError>(values.into())
        };
        Ok(Self {
            window: read(WINDOW, &[400])?,
            filters: read(FILTERS, &[1, 128, 257])?,
        })
    }
}

fn read_float_range(
    source: &dyn CheckpointSource,
    key: &str,
    range: std::ops::Range<usize>,
) -> Result<Vec<f32>, StoreError> {
    let metadata = source.source_metadata(key)?;
    let lease = source.acquire_lease(TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Contiguous {
            offset_elements: range.start,
            shape: vec![range.len()],
        },
        policy: ReadPolicy::RequireBounded,
    })?;
    let converted;
    let native_endian = matches!(lease, CheckpointLease::Gguf(_));
    let (dtype, bytes) = if let CheckpointLease::Gguf(gguf) = &lease {
        converted = gguf.materialize_portable()?.into_converted();
        let eredu_gguf::ConvertedTensor::Dense(tensor) = &converted else {
            return Err(StoreError::Internal(
                "host preprocessing requires dense coefficients".into(),
            ));
        };
        let dtype = match tensor.dtype {
            eredu_gguf::DenseDtype::F32 => StoredDtype::F32,
            eredu_gguf::DenseDtype::F16 => StoredDtype::F16,
            eredu_gguf::DenseDtype::Bf16 => StoredDtype::BF16,
            _ => {
                return Err(StoreError::Internal(
                    "non-floating host preprocessing coefficients".into(),
                ));
            }
        };
        (dtype, tensor.data.as_slice())
    } else {
        (
            metadata.stored_dtype,
            lease.encoded_bytes().ok_or_else(|| {
                StoreError::Internal("host preprocessing lease has no bytes".into())
            })?,
        )
    };
    let width = match dtype {
        StoredDtype::F32 => 4,
        StoredDtype::F16 | StoredDtype::BF16 => 2,
        _ => {
            return Err(StoreError::Internal(
                "non-floating host preprocessing coefficients".into(),
            ));
        }
    };
    if bytes.len() != range.len() * width {
        return Err(StoreError::Internal(
            "host preprocessing lease length mismatch".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(width)
        .map(|value| {
            if width == 4 {
                let bytes = value.try_into().expect("four bytes");
                if native_endian {
                    f32::from_ne_bytes(bytes)
                } else {
                    f32::from_le_bytes(bytes)
                }
            } else {
                let bytes = value.try_into().expect("two bytes");
                let bits = if native_endian {
                    u16::from_ne_bytes(bytes)
                } else {
                    u16::from_le_bytes(bytes)
                };
                if dtype == StoredDtype::F16 {
                    half::f16::from_bits(bits).to_f32()
                } else {
                    half::bf16::from_bits(bits).to_f32()
                }
            }
        })
        .collect())
}
