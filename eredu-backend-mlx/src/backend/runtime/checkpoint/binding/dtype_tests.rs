use super::*;
use eredu_checkpoint::{
    store::{CheckpointSource, SafetensorsWeightStore},
    LinearFormat,
};
use eredu_nn::{ParameterMetadata, ParameterSpec, ParameterVisitor, ParameterVisitorMut};
use eredu_runtime::{
    ReplicatedTextParameterOwner, ReplicatedTextParameterRole, ReplicatedTextPhysicalSource,
    WeightLoweringDescriptor, WeightLoweringKind,
};
use safemlx::{Device, DeviceType, Dtype};

struct Parameter(crate::MlxTensor);

impl Parameterized<crate::MlxTensor> for Parameter {
    fn visit_parameters<'a, V: ParameterVisitor<'a, crate::MlxTensor>>(&'a self, visitor: &mut V) {
        visitor.visit(
            ParameterMetadata::from_spec(&ParameterSpec::trainable("weight").unwrap(), true),
            &self.0,
        );
    }

    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, crate::MlxTensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        visitor.visit_mut(
            ParameterMetadata::from_spec(&ParameterSpec::trainable("weight").unwrap(), true),
            &mut self.0,
        );
    }

    fn set_trainable(&mut self, _: bool) {}
}

fn source(dtype: safetensors::Dtype, bytes: &[u8]) -> (tempfile::TempDir, SafetensorsWeightStore) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let tensor = safetensors::tensor::TensorView::new(dtype, vec![1, 4], bytes).unwrap();
    safetensors::tensor::serialize_to_file([("weight", tensor)], None, &path).unwrap();
    let store = SafetensorsWeightStore::open(path).unwrap();
    (directory, store)
}

fn exact_bindings(
    parameter: &Parameter,
    store: &SafetensorsWeightStore,
) -> Result<Vec<WeightBinding>, ModuleBindingError> {
    let provenance = store.source_provenance("weight").unwrap();
    let metadata = store.source_metadata("weight").unwrap();
    let encoding = provenance.source_encoding;
    let shape = vec![1, 4];
    let task = ReplicatedTextMaterializationTask::from_exact_source(
        "weight",
        ReplicatedTextPhysicalSource::new(
            provenance.catalog_key,
            provenance.physical_tensor,
            provenance.backing_shard.unwrap(),
            provenance.output,
            encoding.clone(),
            metadata.encoded_byte_len,
        )
        .unwrap(),
        Vec::new(),
        shape.clone(),
        shape.clone(),
        ReplicatedTextParameterRole::LinearWeight,
        ReplicatedTextParameterOwner::StaticRole("head".into()),
        LinearFormat::Dense,
        WeightLoweringKind::Direct,
        WeightLoweringDescriptor::new(encoding, LinearFormat::Dense, shape.clone(), shape, Some(1))
            .unwrap(),
    )
    .unwrap();
    build_mlx_exact_replicated_text_bindings(parameter, store, &[&task], &BTreeSet::new(), None)
}

#[test]
fn floating_slots_preserve_selected_f16_bf16_and_f32_storage() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let values = [0.5f32, -1.25, 2.75, -0.0];
    for (stored, native, bytes) in [
        (
            safetensors::Dtype::F16,
            Dtype::Float16,
            values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        (
            safetensors::Dtype::BF16,
            Dtype::Bfloat16,
            values
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect(),
        ),
        (
            safetensors::Dtype::F32,
            Dtype::Float32,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    ] {
        let (_directory, store) = source(stored, &bytes);
        let mut parameter = Parameter(crate::MlxTensor::from_array(Array::from_slice(
            &[0.0f32; 4],
            &[1, 4],
        )));
        let ordinary = build_module_bindings(&parameter, "", &store).unwrap();
        let exact = exact_bindings(&parameter, &store).unwrap();
        for bindings in [&ordinary, &exact] {
            assert_eq!(bindings[0].expected_bytes(), bytes.len() as u64);
            let materialized =
                materialize_module_bindings(&store, bindings, &stream, &stream).unwrap();
            populate_module_from_arrays_excluding(&mut parameter, &materialized, |_| false)
                .unwrap();
            let bound = parameter.0.as_array();
            assert_eq!(bound.dtype(), native);
            assert_eq!(bound.nbytes(), bytes.len());
            assert_eq!(bound.evaluated().unwrap().to_native_bytes(), bytes);
        }
    }
}

#[test]
fn floating_slots_reject_integer_and_unsupported_f64_sources() {
    let parameter = Parameter(crate::MlxTensor::from_array(Array::from_slice(
        &[0.0f32; 4],
        &[1, 4],
    )));
    for stored in [
        safetensors::Dtype::U32,
        safetensors::Dtype::I32,
        safetensors::Dtype::F64,
    ] {
        let (_directory, store) = source(stored, &vec![0; 4 * stored.bitsize() / 8]);
        for result in [
            build_module_bindings(&parameter, "", &store),
            exact_bindings(&parameter, &store),
        ] {
            let error = result.unwrap_err();
            assert!(error.to_string().contains("expects dtype"), "{error}");
        }
    }
}

#[test]
fn packed_integer_slots_retain_exact_dtype_matching() {
    let parameter = Parameter(crate::MlxTensor::from_array(Array::from_slice(
        &[0u32; 4],
        &[1, 4],
    )));
    for stored in [
        safetensors::Dtype::F16,
        safetensors::Dtype::BF16,
        safetensors::Dtype::F32,
        safetensors::Dtype::I32,
        safetensors::Dtype::U8,
        safetensors::Dtype::U32,
    ] {
        let (_directory, store) = source(stored, &vec![0; 4 * stored.bitsize() / 8]);
        let result = build_module_bindings(&parameter, "", &store);
        if stored == safetensors::Dtype::U32 {
            assert_eq!(result.unwrap()[0].expected_bytes(), 16);
        } else {
            let error = result.unwrap_err();
            assert!(error.to_string().contains("expects dtype"), "{error}");
        }
    }
}
