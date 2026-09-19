use super::*;
use crate::backend::{nn::shared::MlxNeuralBackend, ExecutionContext};
use eredu_checkpoint::{
    store::{
        CheckpointLease, CheckpointSource, EncodedTensorLease, MemoryWeightStore, ReadPolicy,
        StoreError, TensorMetadata, TensorReadRequest, WeightStoreDiagnostics,
    },
    AffineQuantization, LinearFormat, SourceTensorEncoding, StoredDtype,
};
use eredu_nn::{
    workspace::{
        WorkspaceBackend, WorkspaceContext, WorkspaceDtype, WorkspaceLayout, WorkspaceMechanisms,
        WorkspaceOperation, WorkspaceOperationBound, WorkspaceTensor,
    },
    LinearFormatSpec, LinearSpec, NeuralBackend, ParameterSpec,
};
use eredu_runtime::{
    LocalModelLayout, LocalTensorLayout, ParameterGroupOwner, ParameterRole,
    ReplicatedTextOutputCompanion, ReplicatedTextParameterOwner, ReplicatedTextParameterRole,
    ReplicatedTextPhysicalSource, TensorPlacement, WeightLoweringDescriptor,
};
use safemlx::{Device, DeviceType};

const WEIGHT: &str = "block.projection.kernel";
const SCALE: &str = "block.codebook.scales";
const BIAS: &str = "block.codebook.zero-points";
const SOURCE: &str = "checkpoint.original";
fn affine() -> WeightQuantization {
    AffineQuantization::default().into()
}

struct NoReads(MemoryWeightStore);
impl CheckpointSource for NoReads {
    fn source_keys(&self) -> Vec<String> {
        self.0.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.0.source_metadata(key)
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("cold exact quantization must not acquire source payloads")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.0.source_diagnostics()
    }
}
#[derive(Debug)]
struct GeometryOnly;
impl WorkspaceMechanisms for GeometryOnly {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(None)
    }
}
fn parameter(name: &str) -> ParameterSpec {
    ParameterSpec::trainable(name).unwrap()
}
fn spec(rows: usize, quantization: Option<WeightQuantization>) -> LinearSpec {
    let format = match quantization {
        None => LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        Some(WeightQuantization::Affine(q)) => {
            LinearFormatSpec::affine(LinearFormat::Affine(q), parameter(SCALE), parameter(BIAS))
                .unwrap()
        }
        Some(WeightQuantization::MxFp4) => {
            LinearFormatSpec::scaled(LinearFormat::MxFp4, parameter(SCALE)).unwrap()
        }
        _ => unreachable!(),
    };
    LinearSpec {
        input: 64,
        output: i32::try_from(rows).unwrap(),
        weight: parameter(WEIGHT),
        bias: None,
        format,
    }
}
fn destinations(
    rows: usize,
    quantization: WeightQuantization,
) -> BTreeMap<String, WorkspaceLayout> {
    let context = WorkspaceContext::new(GeometryOnly);
    let linear = WorkspaceBackend::linear(spec(rows, Some(quantization)), &context).unwrap();
    struct Collector(BTreeMap<String, WorkspaceLayout>);
    impl<'a> ParameterVisitor<'a, WorkspaceTensor> for Collector {
        fn visit(
            &mut self,
            metadata: eredu_nn::ParameterMetadataView<'_>,
            value: &'a WorkspaceTensor,
        ) {
            assert!(self
                .0
                .insert(metadata.id().as_str().into(), value.layout().clone())
                .is_none());
        }
    }
    let mut collector = Collector(BTreeMap::new());
    linear.visit_parameters(&mut collector).unwrap();
    collector.0
}
fn task(
    rows: usize,
    dtype: StoredDtype,
    quantization: WeightQuantization,
) -> ReplicatedTextMaterializationTask {
    let shape = vec![rows, 64];
    let scalar = match dtype {
        StoredDtype::F16 | StoredDtype::BF16 => 2,
        StoredDtype::F32 => 4,
        _ => unreachable!(),
    };
    let encoding = SourceTensorEncoding::Safetensors(dtype);
    let format: LinearFormat = quantization.into();
    let owner = ParameterGroupOwner::static_role("projection");
    let mut companions = vec![ReplicatedTextOutputCompanion::new(
        SCALE,
        LinearCompanionRole::Scale,
        vec![rows, 64 / quantization.group_size() as usize],
        owner.clone(),
    )
    .unwrap()];
    if quantization.has_biases() {
        companions.push(
            ReplicatedTextOutputCompanion::new(
                BIAS,
                LinearCompanionRole::AffineBias,
                vec![rows, 64 / quantization.group_size() as usize],
                owner,
            )
            .unwrap(),
        );
    }
    ReplicatedTextMaterializationTask::from_exact_source(
        WEIGHT,
        ReplicatedTextPhysicalSource::new(
            SOURCE,
            SOURCE,
            "/fixture/model.safetensors",
            SOURCE,
            encoding.clone(),
            (rows * 64 * scalar) as u64,
        )
        .unwrap(),
        vec![],
        shape.clone(),
        shape.clone(),
        ReplicatedTextParameterRole::LinearWeight,
        ReplicatedTextParameterOwner::StaticRole("projection".into()),
        format,
        WeightLoweringKind::Transform,
        WeightLoweringDescriptor::new(encoding, format, shape.clone(), shape, Some(1)).unwrap(),
    )
    .unwrap()
    .with_output_companions(companions)
    .unwrap()
}
fn store(rows: usize, dtype: safetensors::Dtype) -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors([(
        SOURCE.into(),
        dtype,
        vec![rows, 64],
        vec![0; rows * 64 * dtype.bitsize() / 8],
    )])
    .unwrap()
}
fn cold_failure(result: Result<ColdQuantization, Error>) -> Error {
    match result {
        Err(error) => error,
        Ok(_) => panic!("invalid cold quantization was accepted"),
    }
}

#[test]
fn cold_plans_use_real_projection_slots_and_preserve_source_precision_without_reads() {
    for (stored, safe, scalar, expected_affine, expected_mx) in [
        (StoredDtype::F16, safetensors::Dtype::F16, 2, 164, 162),
        (StoredDtype::BF16, safetensors::Dtype::BF16, 2, 164, 162),
        (StoredDtype::F32, safetensors::Dtype::F32, 4, 296, 290),
    ] {
        for quantization in [affine(), WeightQuantization::MxFp4] {
            let source = Arc::new(NoReads(store(2, safe)));
            let task = task(2, stored.clone(), quantization);
            let slots = destinations(2, quantization);
            let parameters = mlx_workspace_binding_targets(&slots).unwrap();
            let (plan, _) =
                build(source.as_ref(), &[parameters], None, quantization, &[&task]).unwrap();
            assert_eq!(plan.targets().len(), 1);
            let target = &plan.targets()[0];
            assert_eq!(target.weight_name(), WEIGHT);
            assert_eq!(target.scales_name(), SCALE);
            assert_eq!(
                target.biases_name(),
                quantization.has_biases().then_some(BIAS)
            );
            assert_eq!(
                target.source(),
                &DerivedWeightRecipe::source(SOURCE, TensorSelection::Full)
            );
            if quantization.has_biases() {
                assert_eq!(target.affine_companion_bytes(), scalar);
            }
            assert_eq!(
                plan.max_working_set_bytes(),
                if quantization.has_biases() {
                    expected_affine
                } else {
                    expected_mx
                }
            );
            prepare_exact_quantization_from_destinations(
                source.into(),
                &[&slots],
                None,
                quantization,
                &[&task],
            )
            .unwrap();
        }
    }
}

#[test]
fn cold_plan_checks_rank_local_geometry_and_complete_target_consumption() {
    let source: RetainedCheckpointSource =
        Arc::new(NoReads(store(4, safetensors::Dtype::F32))).into();
    let task = task(4, StoredDtype::F32, affine());
    let mut layout = LocalModelLayout::default();
    layout.insert(
        WEIGHT.into(),
        LocalTensorLayout::new(
            WEIGHT,
            ParameterRole::FeedForwardIntermediate,
            vec![4, 64],
            vec![2, 64],
            TensorPlacement::Range {
                axis: 0,
                start: 2,
                end: 4,
            },
            None,
            None,
            false,
        ),
    );
    let slots = destinations(2, affine());
    let (plan, _) = build(
        source.as_ref(),
        &[mlx_workspace_binding_targets(&slots).unwrap()],
        Some(&layout),
        affine(),
        &[&task],
    )
    .unwrap();
    assert_eq!(
        plan.targets()[0]
            .source()
            .infer(source.as_ref())
            .unwrap()
            .shape(),
        [2, 64]
    );
    prepare_exact_quantization_from_destinations(
        source.clone(),
        &[&slots],
        Some(&layout),
        affine(),
        &[&task],
    )
    .unwrap();
    let error = cold_failure(prepare_exact_quantization_from_destinations(
        source.clone(),
        &[&slots],
        None,
        affine(),
        &[&task],
    ));
    assert!(error
        .to_string()
        .contains("incompatible with selected destination"));
    let error = cold_failure(prepare_exact_quantization_from_destinations(
        source.clone(),
        &[],
        Some(&layout),
        affine(),
        &[&task],
    ));
    assert!(error.to_string().contains("not consumed exactly once"));
    let error = cold_failure(prepare_exact_quantization_from_destinations(
        source,
        &[&slots, &slots],
        Some(&layout),
        affine(),
        &[&task],
    ));
    assert!(error.to_string().contains("bound more than once"));
}

#[test]
fn cold_plan_rejects_companion_shape_dtype_identity_and_format_mismatches() {
    let source: RetainedCheckpointSource =
        Arc::new(NoReads(store(2, safetensors::Dtype::F32))).into();
    let task = task(2, StoredDtype::F32, affine());
    let slots = destinations(2, affine());
    for (name, replacement) in [
        (SCALE, None),
        (
            WEIGHT,
            Some(WorkspaceLayout::new(&[2, 8], WorkspaceDtype::Float32).unwrap()),
        ),
        (
            WEIGHT,
            Some(WorkspaceLayout::new(&[2, 16], WorkspaceDtype::Uint32).unwrap()),
        ),
        (
            SCALE,
            Some(WorkspaceLayout::new(&[2, 2], WorkspaceDtype::Float32).unwrap()),
        ),
        (
            BIAS,
            Some(WorkspaceLayout::new(&[2, 1], WorkspaceDtype::Uint8).unwrap()),
        ),
    ] {
        let mut malformed = slots.clone();
        match replacement {
            Some(value) => {
                malformed.insert(name.into(), value);
            }
            None => {
                malformed.remove(name);
            }
        }
        cold_failure(prepare_exact_quantization_from_destinations(
            source.clone(),
            &[&malformed],
            None,
            affine(),
            &[&task],
        ));
    }
    let error = cold_failure(prepare_exact_quantization_from_destinations(
        source.clone(),
        &[&slots],
        None,
        affine(),
        &[&task, &task],
    ));
    assert!(error.to_string().contains("requested more than once"));
    let error = cold_failure(prepare_exact_quantization_from_destinations(
        source,
        &[&slots],
        None,
        WeightQuantization::MxFp4,
        &[&task],
    ));
    assert!(error
        .to_string()
        .contains("does not match its format group"));
}

fn output(source: &dyn CheckpointSource, key: &str) -> Vec<u8> {
    source
        .acquire_lease(TensorReadRequest {
            key: key.into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap()
        .encoded_bytes()
        .unwrap()
        .to_vec()
}
fn check_values(source: &dyn CheckpointSource, values: &[f32]) {
    let packed = output(source, WEIGHT);
    let scales = output(source, SCALE);
    let biases = output(source, BIAS);
    for (index, original) in values.iter().enumerate() {
        let row = index / 64;
        let word_offset = row * 32 + (index % 64) / 8 * 4;
        let word = u32::from_le_bytes(packed[word_offset..word_offset + 4].try_into().unwrap());
        let code = (word >> ((index % 8) * 4)) & 15;
        let scale = f32::from_le_bytes(scales[row * 4..row * 4 + 4].try_into().unwrap());
        let bias = f32::from_le_bytes(biases[row * 4..row * 4 + 4].try_into().unwrap());
        let decoded = code as f32 * scale + bias;
        assert!(
            (decoded - original).abs() <= scale.abs() / 2.0 + 0.001,
            "{index}: {decoded} vs {original}"
        );
    }
}

#[test]
fn native_and_original_cold_paths_share_the_plan_and_preserve_native_source_checks() {
    use eredu_runtime::working_memory::{DependencyMemoryPolicy, WorkingMemoryPool};
    let values = (0..128)
        .map(|index| (index % 64) as f32 / 4.0 - (index / 64) as f32 * 3.0)
        .collect::<Vec<_>>();
    let source: RetainedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([(
            SOURCE.into(),
            safetensors::Dtype::F32,
            vec![2, 64],
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )])
        .unwrap(),
    )
    .into();
    let task = task(2, StoredDtype::F32, affine());
    let slots = destinations(2, affine());
    let cold = prepare_exact_quantization_from_destinations(
        source.clone(),
        &[&slots],
        None,
        affine(),
        &[&task],
    )
    .unwrap();
    let pool = WorkingMemoryPool::new(1024 * 1024, 0).unwrap();
    let prepared = cold
        .allocate_original(
            &pool,
            DependencyMemoryPolicy {
                fixed_bytes: 1024,
                bytes_per_input_byte: 8,
            },
        )
        .unwrap();
    let native_owner = pool.acquire_unquoted().unwrap();
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let original = prepared.materialize(context.stream()).unwrap();
    let dense = MlxNeuralBackend::linear(spec(2, None), context.stream()).unwrap();
    let packed = MlxNeuralBackend::linear(spec(2, Some(affine())), context.stream()).unwrap();
    let empty = &std::slice::from_ref(&dense)[..0];
    let (ordinary, _) = quantize_exact_replicated_text_tasks(
        source.clone(),
        &dense,
        &packed,
        empty,
        empty,
        None,
        affine(),
        &[&task],
        context.stream(),
    )
    .unwrap();
    check_values(&original, &values);
    check_values(ordinary.as_ref(), &values);
    for key in [WEIGHT, SCALE, BIAS] {
        assert_eq!(output(&original, key), output(ordinary.as_ref(), key));
    }
    let wrong_shape = MlxNeuralBackend::linear(spec(3, None), context.stream()).unwrap();
    let error = quantize_exact_replicated_text_tasks(
        source,
        &wrong_shape,
        &packed,
        empty,
        empty,
        None,
        affine(),
        &[&task],
        context.stream(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("native source module requires"));
    let half_source: RetainedCheckpointSource = Arc::new(store(2, safetensors::Dtype::F16)).into();
    let half_task = self::task(2, StoredDtype::F16, affine());
    let error = quantize_exact_replicated_text_tasks(
        half_source,
        &dense,
        &packed,
        empty,
        empty,
        None,
        affine(),
        &[&half_task],
        context.stream(),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("differs from native source dtype"));
    drop((
        original,
        ordinary,
        wrong_shape,
        packed,
        dense,
        context,
        native_owner,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
