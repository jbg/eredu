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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{artifact_preparation::PhysicalArtifactId, voicechat::checkpoint::Component};
    use eredu_checkpoint::store::MemoryWeightStore;

    #[test]
    fn retained_preprocessing_preserves_saved_values_across_storage_precision_and_reshape() {
        let source = MemoryWeightStore::from_safetensors([
            (
                "saved.window".into(),
                safetensors::Dtype::F16,
                vec![400],
                (0..400)
                    .flat_map(|i| half::f16::from_f32((i % 9) as f32 / 8.0).to_le_bytes())
                    .collect(),
            ),
            (
                "saved.filters".into(),
                safetensors::Dtype::BF16,
                vec![128, 257],
                (0..128 * 257)
                    .flat_map(|i| half::bf16::from_f32((i % 17) as f32 / 16.0).to_le_bytes())
                    .collect(),
            ),
        ])
        .unwrap();
        let mut parameters = BTreeMap::new();
        for (name, key, shape) in [
            (WINDOW, "saved.window", vec![400]),
            (FILTERS, "saved.filters", vec![1, 128, 257]),
        ] {
            let recipe = DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::source(key, TensorSelection::Full)),
                shape,
            };
            let output = recipe.infer(&source).unwrap();
            parameters.insert(
                name.into(),
                ParameterAdmission {
                    linear_format: eredu_checkpoint::LinearFormat::Dense,
                    name: name.into(),
                    component: Component::Perception,
                    artifact: PhysicalArtifactId::Primary,
                    stored_bytes: 0,
                    conversion_workspace_bytes: 0,
                    materialization_peak_bytes: 0,
                    output,
                    recipe,
                },
            );
        }
        let prepared = PreparedPreprocessing::retain(&parameters, &source).unwrap();
        drop(source);
        assert_eq!(prepared.retained_bytes(), (400 + 128 * 257) * 4);
        for (i, value) in prepared.window().iter().enumerate() {
            assert_eq!(*value, (i % 9) as f32 / 8.0);
        }
        for (i, value) in prepared.filters().iter().enumerate() {
            assert_eq!(*value, (i % 17) as f32 / 16.0);
        }
        let invalid = MemoryWeightStore::from_safetensors([(
            "saved.window".into(),
            safetensors::Dtype::F32,
            vec![400],
            [f32::NAN; 400]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )])
        .unwrap();
        assert!(matches!(
            PreparedPreprocessing::retain(&parameters, &invalid),
            Err(PreprocessingPreparationError::Nonfinite(WINDOW))
        ));
    }

    #[test]
    fn bounded_big_endian_gguf_float_ranges_preserve_coefficient_values() {
        use eredu_checkpoint::{
            gguf_store::GgufWeightStore,
            schema::{CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint},
        };
        use eredu_gguf::{Checkpoint, Endian, GgmlType, TensorInput, Writer, WriterOptions};

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preprocessing.gguf");
        let single: Vec<u8> = [8.0f32, -0.375, 1.25, 4097.0, -2.0]
            .into_iter()
            .flat_map(f32::to_be_bytes)
            .collect();
        let half: Vec<u8> = [8.0f32, -0.375, 1.25, 4096.0, -2.0]
            .into_iter()
            .flat_map(|value| half::f16::from_f32(value).to_bits().to_be_bytes())
            .collect();
        Writer::new(WriterOptions {
            endian: Endian::Big,
            ..WriterOptions::default()
        })
        .unwrap()
        .write(
            std::fs::File::create(&path).unwrap(),
            &BTreeMap::new(),
            &[
                TensorInput {
                    name: "single",
                    dimensions: &[5],
                    ggml_type: GgmlType::F32,
                    data: &single,
                },
                TensorInput {
                    name: "half",
                    dimensions: &[5],
                    ggml_type: GgmlType::F16,
                    data: &half,
                },
            ],
        )
        .unwrap();
        let checkpoint = Checkpoint::open(&path).unwrap();
        let plan = GgufCheckpointPlan::new(
            "bounded preprocessing coefficients",
            [("single", GgmlType::F32), ("half", GgmlType::F16)]
                .into_iter()
                .map(|(name, ty)| {
                    GgufTensorConstraint::required(name, vec![5], GgufTypeConstraint::Exact(ty))
                })
                .collect(),
            Vec::new(),
            CatalogPolicy::strict(),
        )
        .unwrap();
        let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
        let source = GgufWeightStore::builder()
            .add_checkpoint(checkpoint, &plan, &mapping)
            .unwrap()
            .build()
            .unwrap();
        // A strict interior selection catches both accidental full-tensor reads
        // and decoding portable native-endian bytes as source-endian bytes.
        assert_eq!(
            read_float_range(&source, "single", 1..4).unwrap(),
            [-0.375, 1.25, 4097.0]
        );
        assert_eq!(
            read_float_range(&source, "half", 1..4).unwrap(),
            [-0.375, 1.25, 4096.0]
        );
        assert!(read_float_range(&source, "single", 3..6).is_err());
    }
}
