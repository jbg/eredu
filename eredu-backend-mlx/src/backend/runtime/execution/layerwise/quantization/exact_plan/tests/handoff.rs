use super::*;
use crate::backend::runtime::checkpoint::bounded_quantization::ConvertedQuantization;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct GatedSource {
    inner: MemoryWeightStore,
    frozen: AtomicBool,
    reads: AtomicUsize,
}

impl CheckpointSource for GatedSource {
    fn source_keys(&self) -> Vec<String> {
        self.inner.source_keys()
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.inner.source_metadata(key)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.inner.source_diagnostics()
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        assert!(
            !self.frozen.load(Ordering::SeqCst),
            "adoption must not read source payloads"
        );
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.acquire_lease(request)
    }
}

fn source(rows: usize) -> (Arc<GatedSource>, Vec<f32>) {
    let values = (0..rows * 64)
        .map(|index| (index % 64) as f32 / 4.0 - (index / 64) as f32 * 3.0)
        .collect::<Vec<_>>();
    let source = Arc::new(GatedSource {
        inner: MemoryWeightStore::from_safetensors([(
            SOURCE.into(),
            safetensors::Dtype::F32,
            vec![rows, 64],
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )])
        .unwrap(),
        frozen: AtomicBool::new(false),
        reads: AtomicUsize::new(0),
    });
    (source, values)
}

fn complete(
    source: &Arc<GatedSource>,
    task: &ReplicatedTextMaterializationTask,
    layout: Option<&LocalModelLayout>,
    stream: &safemlx::Stream,
) -> ConvertedQuantization {
    let slots = destinations(2, affine());
    let prepared = prepare_exact_quantization_from_destinations(
        source.clone().into(),
        &[&slots],
        layout,
        affine(),
        &[task],
    )
    .unwrap()
    .allocate_ordinary(stream)
    .unwrap();
    assert_eq!(source.reads.load(Ordering::SeqCst), 0);
    let converted = prepared.materialize_handoff(stream).unwrap();
    assert!(source.reads.load(Ordering::SeqCst) > 0);
    source.frozen.store(true, Ordering::SeqCst);
    converted
}

fn rows(start: usize) -> LocalModelLayout {
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
                start,
                end: start + 2,
            },
            None,
            None,
            false,
        ),
    );
    layout
}

fn check_selected_rows(source: &dyn CheckpointSource, values: &[f32]) {
    let weights = output(source, WEIGHT);
    let scales = output(source, SCALE);
    let biases = output(source, BIAS);
    // Selected source rows span [-6, 9.75] and [-9, 6.75]. Affine edge
    // alignment gives these two codebooks; the latter clips 6.75 to 6.
    let expected_scales = [-13.0f32 / 12.0, 1.0];
    let expected_biases = [9.75f32, -9.0];
    for row in 0..2 {
        assert_eq!(
            f32::from_le_bytes(scales[row * 4..row * 4 + 4].try_into().unwrap()),
            expected_scales[row],
        );
        assert_eq!(
            f32::from_le_bytes(biases[row * 4..row * 4 + 4].try_into().unwrap()),
            expected_biases[row],
        );
        for column in 0..64 {
            let offset = row * 32 + column / 8 * 4;
            let word = u32::from_le_bytes(weights[offset..offset + 4].try_into().unwrap());
            let code = (word >> (column % 8 * 4)) & 15;
            let expected = ((values[row * 64 + column] - expected_biases[row])
                / expected_scales[row])
                .round_ties_even()
                .clamp(0.0, 15.0) as u32;
            assert_eq!(code, expected, "row {row}, column {column}");
        }
    }
}

#[test]
fn native_adoption_reuses_the_exact_overlay_and_rank_local_payload_without_reads() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let dense = MlxNeuralBackend::linear(spec(2, None), context.stream()).unwrap();
    let packed = MlxNeuralBackend::linear(spec(2, Some(affine())), context.stream()).unwrap();
    let empty = &std::slice::from_ref(&dense)[..0];
    let (source, values) = source(4);
    let layout = rows(2);
    let task = task(4, StoredDtype::F32, affine());
    let completed = complete(&source, &task, Some(&layout), context.stream());
    let residency_source = completed.store().clone();
    let lease = residency_source
        .acquire_lease(TensorReadRequest {
            key: WEIGHT.into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let address = lease.encoded_bytes().unwrap().as_ptr();
    let reads = source.reads.load(Ordering::SeqCst);
    let (adopted, report) = adopt_exact_replicated_text_quantization(
        completed,
        source.clone().into(),
        &dense,
        &packed,
        empty,
        empty,
        Some(&layout),
        affine(),
        &[&task],
    )
    .unwrap();
    assert!(adopted.same_source(&residency_source));
    assert_eq!(source.reads.load(Ordering::SeqCst), reads);
    assert_eq!(report.transformed_weights, 1);
    assert_eq!(report.output_bytes, 80);
    assert_eq!(report.source_bytes_read, 512);
    let adopted_lease = adopted
        .acquire_lease(TensorReadRequest {
            key: WEIGHT.into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    assert_eq!(adopted_lease.encoded_bytes().unwrap().as_ptr(), address);
    check_selected_rows(adopted.as_ref(), &values[128..]);
    drop((adopted, residency_source, lease));
    assert!(adopted_lease
        .encoded_bytes()
        .unwrap()
        .iter()
        .any(|&byte| byte != 0));
}

#[test]
fn adoption_rejects_foreign_sources_changed_plans_and_invalid_native_slots_without_reads() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let dense = MlxNeuralBackend::linear(spec(2, None), context.stream()).unwrap();
    let packed = MlxNeuralBackend::linear(spec(2, Some(affine())), context.stream()).unwrap();
    let wrong_dense = MlxNeuralBackend::linear(spec(3, None), context.stream()).unwrap();
    let wrong_packed = MlxNeuralBackend::linear(spec(3, Some(affine())), context.stream()).unwrap();
    let eight_bits: WeightQuantization = AffineQuantization::new(64, 8).unwrap().into();
    let packed_eight =
        MlxNeuralBackend::linear(spec(2, Some(eight_bits)), context.stream()).unwrap();
    let empty = &std::slice::from_ref(&dense)[..0];
    for case in 0..6 {
        let row_count = if case == 5 { 4 } else { 2 };
        let (source, _) = source(row_count);
        let original_task = task(row_count, StoredDtype::F32, affine());
        let original_layout = (case == 5).then(|| rows(0));
        let completed = complete(
            &source,
            &original_task,
            original_layout.as_ref(),
            context.stream(),
        );
        let residency_source = completed.store().clone();
        let reads = source.reads.load(Ordering::SeqCst);
        let (foreign, _) = self::source(row_count);
        foreign.frozen.store(true, Ordering::SeqCst);
        let selected_source = if case == 0 {
            foreign.clone()
        } else {
            source.clone()
        };
        let selected_quantization = if case == 1 { eight_bits } else { affine() };
        let selected_task = task(row_count, StoredDtype::F32, selected_quantization);
        let selected_layout = (case == 5).then(|| rows(2));
        let target = match case {
            1 => &packed_eight,
            3 => &wrong_packed,
            _ => &packed,
        };
        let selected_tasks = if case == 4 {
            Vec::new()
        } else {
            vec![&selected_task]
        };
        let error = adopt_exact_replicated_text_quantization(
            completed,
            selected_source.into(),
            if case == 2 { &wrong_dense } else { &dense },
            target,
            empty,
            empty,
            selected_layout.as_ref(),
            selected_quantization,
            &selected_tasks,
        )
        .unwrap_err();
        let expected = match case {
            0 => "different checkpoint source",
            1 | 5 => "does not match the selected transform plan",
            2 => "native source module requires",
            3 => "incompatible with selected destination",
            4 => "received no tasks",
            _ => unreachable!(),
        };
        assert!(error.to_string().contains(expected), "case {case}: {error}");
        assert_eq!(source.reads.load(Ordering::SeqCst), reads);
        assert_eq!(foreign.reads.load(Ordering::SeqCst), 0);
        assert!(output(residency_source.as_ref(), WEIGHT)
            .iter()
            .any(|&byte| byte != 0));
    }
}
