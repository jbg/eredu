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
fn byte_slots_preserve_fp8_values_and_exponent_scales() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let bytes = [0x38_u8, 0x40, 0xb8, 0];
    for stored in [safetensors::Dtype::F8_E4M3, safetensors::Dtype::F8_E8M0] {
        let (_directory, store) = source(stored, &bytes);
        let mut parameter = Parameter(crate::MlxTensor::from_array(Array::from_slice(
            &[0_u8; 4],
            &[1, 4],
        )));
        for bindings in [
            build_module_bindings(&parameter, "", &store).unwrap(),
            exact_bindings(&parameter, &store).unwrap(),
        ] {
            assert_eq!(bindings[0].expected_bytes(), 4);
            let materialized =
                materialize_module_bindings(&store, &bindings, &stream, &stream).unwrap();
            populate_module_from_arrays_excluding(&mut parameter, &materialized, |_| false)
                .unwrap();
            assert_eq!(parameter.0.as_array().dtype(), Dtype::Uint8);
            assert_eq!(
                parameter
                    .0
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .to_native_bytes(),
                bytes
            );
        }
        let (_directory, floating) = source(safetensors::Dtype::F32, &[0; 16]);
        assert!(build_module_bindings(&parameter, "", &floating).is_err());
        assert!(exact_bindings(&parameter, &floating).is_err());
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

#[cfg(not(feature = "cuda"))]
#[test]
fn grouped_expert_weights_and_scales_share_reads_and_preserve_aliases() {
    use eredu_checkpoint::store::TensorSelection;
    use safetensors::tensor::{serialize_to_file, TensorView};
    let directory = tempfile::tempdir().unwrap();
    let scales_a = 1.5f32.to_le_bytes();
    let scales_b = 2.5f32.to_le_bytes();
    serialize_to_file(
        [
            (
                "a.weight",
                TensorView::new(safetensors::Dtype::U8, vec![2], &[1, 2]).unwrap(),
            ),
            (
                "b.weight",
                TensorView::new(safetensors::Dtype::U8, vec![2], &[3, 4]).unwrap(),
            ),
            (
                "a.scale",
                TensorView::new(safetensors::Dtype::F32, vec![1], &scales_a).unwrap(),
            ),
            (
                "b.scale",
                TensorView::new(safetensors::Dtype::F32, vec![1], &scales_b).unwrap(),
            ),
        ],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(directory.path()).unwrap();
    let stacked = |suffix: &str| DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: ["b", "a"]
            .map(|expert| {
                DerivedWeightRecipe::source(format!("{expert}.{suffix}"), TensorSelection::Full)
            })
            .to_vec(),
    };
    let bindings = [
        WeightBinding::from_recipe("weight", stacked("weight"), 4).unwrap(),
        WeightBinding::from_recipe("scales", stacked("scale"), 8).unwrap(),
        WeightBinding::alias("shared", "weight", 4).unwrap(),
    ];
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let arrays = materialize_module_bindings(&store, &bindings, &stream, &stream).unwrap();
    let weights = arrays["weight"].evaluated().unwrap();
    assert_eq!(weights.as_slice::<u8>(), [3, 4, 1, 2]);
    assert_eq!(arrays["weight"].shape(), [2, 2]);
    assert_eq!(
        arrays["scales"].evaluated().unwrap().as_slice::<f32>(),
        [2.5, 1.5]
    );
    assert_eq!(
        arrays["shared"]
            .evaluated()
            .unwrap()
            .as_slice::<u8>()
            .as_ptr(),
        weights.as_slice::<u8>().as_ptr()
    );
    let diagnostics = store.source_diagnostics().unwrap();
    #[cfg(unix)]
    assert_eq!(
        diagnostics.physical_reads, 1,
        "all expert values and scales share one file read"
    );
    assert_eq!(diagnostics.physical_read_bytes, 12);
    assert_eq!(diagnostics.currently_cached_shards, 0);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn direct_groups_survive_fallbacks_and_parameter_count_boundaries() {
    use eredu_checkpoint::store::TensorSelection;
    use safetensors::tensor::{serialize_to_file, TensorView};
    let directory = tempfile::tempdir().unwrap();
    let values = (0..66u8)
        .map(|value| (format!("weight{value:02}"), vec![value; 2]))
        .collect::<Vec<_>>();
    serialize_to_file(
        values.iter().map(|(name, data)| {
            (
                name.as_str(),
                TensorView::new(safetensors::Dtype::U8, vec![1, 2], data).unwrap(),
            )
        }),
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(directory.path()).unwrap();
    let mut bindings = values
        .iter()
        .take(65)
        .map(|(name, _)| WeightBinding::new(name, name, TensorSelection::Full, 2).unwrap())
        .collect::<Vec<_>>();
    bindings.push(
        WeightBinding::from_recipe(
            "transposed",
            DerivedWeightRecipe::Transpose {
                input: Box::new(DerivedWeightRecipe::source(
                    "weight65",
                    TensorSelection::Full,
                )),
                axes: vec![1, 0],
            },
            2,
        )
        .unwrap(),
    );
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let arrays = materialize_module_bindings(&store, &bindings, &stream, &stream).unwrap();
    assert_eq!(arrays.len(), 66);
    for (name, expected) in &values[..65] {
        assert_eq!(arrays[name].evaluated().unwrap().as_slice::<u8>(), expected);
    }
    assert_eq!(arrays["transposed"].shape(), [2, 1]);
    assert_eq!(
        arrays["transposed"].evaluated().unwrap().as_slice::<u8>(),
        [65, 65]
    );
    #[cfg(unix)]
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, 3);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn unsupported_direct_reads_are_prepared_only_once() {
    use eredu_checkpoint::store::{
        CheckpointLease, EncodedReadBatch, StoreError, TensorMetadata, TensorSelection,
        WeightStoreDiagnostics,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct OrdinarySource {
        store: SafetensorsWeightStore,
        preparations: AtomicUsize,
    }
    impl CheckpointSource for OrdinarySource {
        fn source_keys(&self) -> Vec<String> {
            self.store.source_keys()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.store.source_metadata(key)
        }
        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            self.store.acquire_lease(request)
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            self.store.source_diagnostics()
        }
        fn prepare_encoded_read(
            &self,
            _: &[String],
        ) -> Result<Option<EncodedReadBatch>, StoreError> {
            assert_eq!(
                self.preparations.fetch_add(1, Ordering::Relaxed),
                0,
                "the fallback decision must be retained"
            );
            Ok(None)
        }
    }
    let (_directory, store) = source(safetensors::Dtype::U8, &[1, 2, 3, 4]);
    let source = OrdinarySource {
        store,
        preparations: AtomicUsize::new(0),
    };
    let binding = WeightBinding::from_recipe(
        "weight",
        DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::source("weight", TensorSelection::Full)),
            shape: vec![2, 2],
        },
        4,
    )
    .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let arrays = materialize_module_bindings(&source, &[binding], &stream, &stream).unwrap();
    assert_eq!(
        arrays["weight"].evaluated().unwrap().as_slice::<u8>(),
        [1, 2, 3, 4]
    );
    assert_eq!(source.preparations.load(Ordering::Relaxed), 1);
}
