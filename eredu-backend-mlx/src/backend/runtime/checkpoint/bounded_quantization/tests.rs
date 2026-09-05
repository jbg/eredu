use std::{collections::HashMap, sync::Arc};

use crate::{backend::runtime::checkpoint::gguf::GgufCheckpoint, backend::ExecutionContext};
use eredu_checkpoint::{
    store::{ReadPolicy as WeightReadPolicy, SafetensorsWeightStore, WeightStoreBackend},
    AffineQuantization,
};
use safemlx::{Device, DeviceType};
use safetensors::tensor::{serialize_to_file, TensorView};
use tempfile::TempDir;

use super::*;
use crate::backend::runtime::{
    checkpoint::store::test_support::open_gguf_checkpoint_source_for_test,
    residency::manager::{host_capacity_upper_bound_for_bindings, ResidencyManager},
};
use crate::tests::support::test_utils::SyntheticGguf;
use eredu_core::residency::{
    MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
};
use eredu_runtime::{OffloadUnit, WeightBinding};

fn cpu_context() -> ExecutionContext {
    ExecutionContext::new(Device::new(DeviceType::Cpu, 0))
}

fn test_target(weight_name: &str, source: DerivedWeightRecipe) -> BoundedQuantizationTarget {
    let (scales, biases) = weight_name.strip_suffix(".weight").map_or_else(
        || {
            (
                format!("{weight_name}_scales"),
                format!("{weight_name}_biases"),
            )
        },
        |prefix| (format!("{prefix}.scales"), format!("{prefix}.biases")),
    );
    BoundedQuantizationTarget::from_recipe(weight_name, scales, Some(biases), source).unwrap()
}

fn direct_test_target(weight_name: &str) -> BoundedQuantizationTarget {
    test_target(
        weight_name,
        DerivedWeightRecipe::source(weight_name, TensorSelection::Full),
    )
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn matrix_values(matrices: usize, rows: usize, columns: usize) -> Vec<f32> {
    (0..matrices * rows * columns)
        .map(|index| (index as f32 - 255.5) / 64.0)
        .collect()
}

#[test]
fn allocator_cache_reuse_respects_retained_and_working_set_bounds() {
    assert!(!allocator_cache_requires_clear(64, 64, 64));
    assert!(allocator_cache_requires_clear(65, 64, 128));
    assert!(allocator_cache_requires_clear(65, 128, 64));
}

fn direct_fixture() -> (TempDir, Arc<SafetensorsWeightStore>, Vec<f32>) {
    let directory = tempfile::tempdir().unwrap();
    let values = matrix_values(1, 8, 64);
    let bytes = float_bytes(&values);
    serialize_to_file(
        [(
            "model.proj.weight",
            TensorView::new(SafeDtype::F32, vec![8, 64], &bytes).unwrap(),
        )],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    (directory, store, values)
}

fn two_target_fixture() -> (TempDir, Arc<SafetensorsWeightStore>, [Vec<f32>; 2]) {
    let directory = tempfile::tempdir().unwrap();
    let values = [matrix_values(1, 8, 64), matrix_values(1, 8, 64)];
    let bytes = values.each_ref().map(|values| float_bytes(values));
    serialize_to_file(
        [
            (
                "model.first.weight",
                TensorView::new(SafeDtype::F32, vec![8, 64], &bytes[0]).unwrap(),
            ),
            (
                "model.second.weight",
                TensorView::new(SafeDtype::F32, vec![8, 64], &bytes[1]).unwrap(),
            ),
        ],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    (directory, store, values)
}

fn materialize(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    name: &str,
    stream: &Stream,
) -> Array {
    let lease = store
        .acquire_lease(TensorReadRequest {
            key: name.into(),
            selection: TensorSelection::Full,
            policy: WeightReadPolicy::RequireBounded,
        })
        .unwrap();
    MlxParameterMaterializationContext::new(stream, stream)
        .weight_lease(lease)
        .unwrap()
        .materialize(stream, stream)
        .unwrap()
        .synchronize()
        .unwrap()
}

fn assert_affine_outputs_match_reference(
    transformed: &BoundedQuantizedWeightStore,
    values: &[f32],
    quantization: AffineQuantization,
    context: &ExecutionContext,
) {
    let dense = Array::from_slice(values, &[8, 64]);
    let expected = quantize_tensor(&dense, quantization, context.stream()).unwrap();
    let weight = materialize(transformed, "model.proj.weight", context.stream());
    let scales = materialize(transformed, "model.proj.scales", context.stream());
    let biases = materialize(transformed, "model.proj.biases", context.stream());
    assert_eq!(
        weight.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
    assert_eq!(
        scales.evaluated().unwrap().as_slice::<f32>(),
        expected.scales.evaluated().unwrap().as_slice::<f32>()
    );
    assert_eq!(
        biases.evaluated().unwrap().as_slice::<f32>(),
        expected
            .biases
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
    );
}

fn assert_mxfp4_outputs_match_reference(
    transformed: &BoundedQuantizedWeightStore,
    values: &[f32],
    context: &ExecutionContext,
) {
    let dense = Array::from_slice(values, &[8, 64]);
    let expected = quantize_tensor(&dense, WeightQuantization::MxFp4, context.stream()).unwrap();
    let weight = materialize(transformed, "model.proj.weight", context.stream());
    let scales = materialize(transformed, "model.proj.scales", context.stream());
    assert_eq!(
        weight.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
    assert_eq!(
        scales.evaluated().unwrap().as_slice::<u8>(),
        expected.scales.evaluated().unwrap().as_slice::<u8>()
    );
}

#[test]
fn affine_conversion_is_row_bounded_and_matches_the_canonical_quantizer() {
    let (_directory, source, values) = direct_fixture();
    let context = cpu_context();
    let quantization = AffineQuantization::default();
    let target = direct_test_target("model.proj.weight");
    // One f32 source row (256 bytes) plus one packed affine output row
    // (32-byte weight, 4-byte scale, and 4-byte bias). The complete
    // quantized matrix is 320 bytes, so that runtime-sized budget is also
    // sufficient to convert the 2,048-byte dense source.
    let plan = BoundedQuantizationPlan::new(quantization, 320, [target]).unwrap();
    let transformed =
        BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

    assert_eq!(
        transformed.report(),
        &WeightMaterializationReport {
            admitted_working_set_bytes: 320,
            transformed_weights: 1,
            source_tiles: 8,
            peak_in_flight_tiles: 1,
            source_bytes_read: 2_048,
            output_bytes: 320,
            peak_planned_working_set_bytes: 296,
            largest_source_tile_bytes: 256,
            largest_output_tile_bytes: 40,
        }
    );
    assert!(
        transformed.report().peak_planned_working_set_bytes <= transformed.report().output_bytes
    );
    assert_eq!(
        transformed
            .source_metadata("model.proj.weight")
            .unwrap()
            .encoded_byte_len,
        256
    );
    assert_eq!(
        transformed
            .source_metadata("model.proj.scales")
            .unwrap()
            .encoded_byte_len,
        32
    );
    assert_eq!(
        transformed
            .source_metadata("model.proj.biases")
            .unwrap()
            .encoded_byte_len,
        32
    );

    assert_affine_outputs_match_reference(&transformed, &values, quantization, &context);
}

#[test]
fn affine_conversion_double_buffers_two_cpu_tiles_within_the_bound() {
    let (_directory, source, values) = direct_fixture();
    let context = cpu_context();
    let quantization = AffineQuantization::default();
    let plan =
        BoundedQuantizationPlan::new(quantization, 640, [direct_test_target("model.proj.weight")])
            .unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

    let report = transformed.report();
    assert_eq!(report.source_tiles, 8);
    assert_eq!(report.peak_in_flight_tiles, 2);
    assert_eq!(report.peak_planned_working_set_bytes, 592);
    assert!(report.peak_planned_working_set_bytes <= report.admitted_working_set_bytes);

    assert_affine_outputs_match_reference(&transformed, &values, quantization, &context);
}

#[test]
fn mxfp4_conversion_double_buffers_across_target_boundaries() {
    let (_directory, source, values) = two_target_fixture();
    let context = cpu_context();
    let quantization = WeightQuantization::MxFp4;
    // Each complete target needs 2,320 bytes: 2,048 source bytes plus
    // 272 packed output bytes. Both targets therefore fit as one tile in
    // the two-slot window, making cross-target overlap observable.
    let plan = BoundedQuantizationPlan::new(
        quantization,
        4_640,
        [
            direct_test_target("model.first.weight"),
            direct_test_target("model.second.weight"),
        ],
    )
    .unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

    let report = transformed.report();
    assert_eq!(report.transformed_weights, 2);
    assert_eq!(report.source_tiles, 2);
    assert_eq!(report.peak_in_flight_tiles, 2);
    assert_eq!(report.peak_planned_working_set_bytes, 4_640);
    assert_eq!(report.source_bytes_read, 4_096);
    assert_eq!(report.output_bytes, 544);

    for (name, values) in [("model.first", &values[0]), ("model.second", &values[1])] {
        let dense = Array::from_slice(values, &[8, 64]);
        let expected = quantize_tensor(&dense, quantization, context.stream()).unwrap();
        let weight = materialize(&transformed, &format!("{name}.weight"), context.stream());
        let scales = materialize(&transformed, &format!("{name}.scales"), context.stream());
        assert_eq!(
            weight.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
        assert_eq!(
            scales.evaluated().unwrap().as_slice::<u8>(),
            expected.scales.evaluated().unwrap().as_slice::<u8>()
        );
        assert!(!transformed
            .source_keys()
            .contains(&format!("{name}.biases")));
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn affine_gpu_conversion_matches_the_canonical_gpu_quantizer() {
    let (_directory, source, values) = direct_fixture();
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let quantization = AffineQuantization::default();
    let plan =
        BoundedQuantizationPlan::new(quantization, 640, [direct_test_target("model.proj.weight")])
            .unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

    assert_affine_outputs_match_reference(&transformed, &values, quantization, &context);
}

#[test]
fn insufficient_bound_fails_before_a_source_array_is_materialized() {
    let (_directory, source, _values) = direct_fixture();
    let context = cpu_context();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        295,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap();
    let error =
        BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap_err();
    assert!(error
        .to_string()
        .contains("requires at least 296 working-set bytes for one row, but the plan permits 295"));
    let diagnostics = source.source_diagnostics().unwrap();
    // Metadata/preflight may map the shard, but no selected payload was
    // converted and no GGUF physical read was issued.
    assert_eq!(diagnostics.physical_reads, 0);
}

#[test]
fn semantic_expert_recipe_is_quantized_under_its_local_target_name() {
    let directory = tempfile::tempdir().unwrap();
    let values = matrix_values(1, 4, 64);
    let bytes = float_bytes(&values);
    serialize_to_file(
        [(
            "checkpoint.expert_1.weight",
            TensorView::new(SafeDtype::F32, vec![4, 64], &bytes).unwrap(),
        )],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let context = cpu_context();
    let recipe = DerivedWeightRecipe::source("checkpoint.expert_1.weight", TensorSelection::Full);
    let target = test_target("local.expert.weight", recipe);
    let plan = BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();
    assert_eq!(transformed.report().source_tiles, 4);
    assert_eq!(transformed.report().source_bytes_read, 1_024);

    let dense = Array::from_slice(&values, &[4, 64]);
    let expected =
        quantize_tensor(&dense, AffineQuantization::default(), context.stream()).unwrap();
    let actual = materialize(&transformed, "local.expert.weight", context.stream());
    assert_eq!(
        actual.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
}

#[test]
fn expert_ownership_and_tp_row_tile_compose_into_bounded_reads() {
    let directory = tempfile::tempdir().unwrap();
    let values = matrix_values(2, 4, 64);
    let bytes = float_bytes(&values);
    serialize_to_file(
        [(
            "checkpoint.experts.weight",
            TensorView::new(SafeDtype::F32, vec![2, 4, 64], &bytes).unwrap(),
        )],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let context = cpu_context();
    let recipe = DerivedWeightRecipe::Select {
        input: Box::new(DerivedWeightRecipe::source(
            "checkpoint.experts.weight",
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            },
        )),
        selection: TensorSelection::Range {
            axis: 1,
            start: 1,
            end: 3,
        },
    };
    let target = test_target("rank.expert.weight", recipe);
    let plan = BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

    assert_eq!(transformed.report().source_tiles, 2);
    assert_eq!(transformed.report().source_bytes_read, 512);
    assert_eq!(transformed.report().output_bytes, 80);
    assert_eq!(transformed.report().peak_planned_working_set_bytes, 296);

    let selected = &values[(4 + 1) * 64..(4 + 3) * 64];
    let dense = Array::from_slice(selected, &[1, 2, 64]);
    let expected =
        quantize_tensor(&dense, AffineQuantization::default(), context.stream()).unwrap();
    let actual = materialize(&transformed, "rank.expert.weight", context.stream());
    assert_eq!(actual.shape(), &[1, 2, 8]);
    assert_eq!(
        actual.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
}

#[test]
fn complete_rank_three_expert_bank_is_tiled_without_flattening_its_output() {
    let directory = tempfile::tempdir().unwrap();
    let values = matrix_values(2, 4, 64);
    let bytes = float_bytes(&values);
    serialize_to_file(
        [(
            "model.experts.weight",
            TensorView::new(SafeDtype::F32, vec![2, 4, 64], &bytes).unwrap(),
        )],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let context = cpu_context();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        296,
        [direct_test_target("model.experts.weight")],
    )
    .unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

    assert_eq!(transformed.report().source_tiles, 8);
    assert_eq!(transformed.report().source_bytes_read, 2_048);
    assert_eq!(transformed.report().output_bytes, 320);
    assert_eq!(
        transformed
            .source_metadata("model.experts.weight")
            .unwrap()
            .logical_shape,
        vec![2, 4, 8]
    );
    let expected = quantize_tensor(
        &Array::from_slice(&values, &[2, 4, 64]),
        AffineQuantization::default(),
        context.stream(),
    )
    .unwrap();
    let actual = materialize(&transformed, "model.experts.weight", context.stream());
    assert_eq!(actual.shape(), &[2, 4, 8]);
    assert_eq!(
        actual.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
}

#[test]
fn complete_rank_three_expert_bank_uses_one_submission_when_admitted() {
    let directory = tempfile::tempdir().unwrap();
    let values = matrix_values(2, 4, 64);
    let bytes = float_bytes(&values);
    serialize_to_file(
        [(
            "model.experts.weight",
            TensorView::new(SafeDtype::F32, vec![2, 4, 64], &bytes).unwrap(),
        )],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let context = cpu_context();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        5_000,
        [direct_test_target("model.experts.weight")],
    )
    .unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

    assert_eq!(transformed.report().source_tiles, 1);
    assert_eq!(transformed.report().source_bytes_read, 2_048);
    let expected = quantize_tensor(
        &Array::from_slice(&values, &[2, 4, 64]),
        AffineQuantization::default(),
        context.stream(),
    )
    .unwrap();
    let actual = materialize(&transformed, "model.experts.weight", context.stream());
    assert_eq!(actual.shape(), &[2, 4, 8]);
    assert_eq!(
        actual.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
}

#[test]
fn complete_rank_three_bank_preserves_exact_runtime_companion_names() {
    let values = matrix_values(2, 4, 64);
    let dense = Array::from_slice(&values, &[2, 4, 64]);
    let fixture = SyntheticGguf::dense(
        &HashMap::from([("checkpoint.experts.weight".to_string(), dense)]),
        &HashMap::new(),
    );
    let source = Arc::new(
        open_gguf_checkpoint_source_for_test(
            GgufCheckpoint::open(fixture.path()).unwrap(),
            |name| name.to_string(),
        )
        .unwrap(),
    );
    let context = cpu_context();
    let target = BoundedQuantizationTarget::from_recipe(
        "model.experts.down_proj",
        "architecture.scale-table",
        Some("architecture.zero-points"),
        DerivedWeightRecipe::source("checkpoint.experts.weight", TensorSelection::Full),
    )
    .unwrap();
    assert_eq!(target.scales_name(), "architecture.scale-table");
    assert_eq!(target.biases_name(), Some("architecture.zero-points"));
    let plan = BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
    let transformed =
        BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

    assert_eq!(
        transformed
            .source_metadata("model.experts.down_proj")
            .unwrap()
            .logical_shape,
        vec![2, 4, 8]
    );
    assert_eq!(
        transformed
            .source_metadata("architecture.scale-table")
            .unwrap()
            .logical_shape,
        vec![2, 4, 1]
    );
    assert_eq!(
        transformed
            .source_metadata("architecture.zero-points")
            .unwrap()
            .logical_shape,
        vec![2, 4, 1]
    );
    let diagnostics = source.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 8);
    assert_eq!(diagnostics.physical_read_bytes, 2_048);
}

#[test]
fn dense_gguf_expert_and_row_selections_compose_without_full_bank_reads() {
    let values = matrix_values(2, 4, 64);
    let dense = Array::from_slice(&values, &[2, 4, 64]);
    let fixture = SyntheticGguf::dense(
        &HashMap::from([("checkpoint.experts.weight".to_string(), dense)]),
        &HashMap::new(),
    );
    let source = Arc::new(
        open_gguf_checkpoint_source_for_test(
            GgufCheckpoint::open(fixture.path()).unwrap(),
            |name| name.to_string(),
        )
        .unwrap(),
    );
    let context = cpu_context();
    let recipe = DerivedWeightRecipe::Select {
        input: Box::new(DerivedWeightRecipe::source(
            "checkpoint.experts.weight",
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            },
        )),
        selection: TensorSelection::Range {
            axis: 1,
            start: 1,
            end: 3,
        },
    };
    let target = test_target("rank.expert.weight", recipe);
    let plan = BoundedQuantizationPlan::new(AffineQuantization::default(), 296, [target]).unwrap();
    let transformed =
        BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

    assert_eq!(transformed.report().source_tiles, 2);
    assert_eq!(transformed.report().source_bytes_read, 512);
    assert_eq!(transformed.report().output_bytes, 80);
    assert_eq!(transformed.report().peak_planned_working_set_bytes, 296);
    let diagnostics = source.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 2);
    assert_eq!(diagnostics.physical_read_bytes, 512);

    let selected = &values[(4 + 1) * 64..(4 + 3) * 64];
    let expected = quantize_tensor(
        &Array::from_slice(selected, &[1, 2, 64]),
        AffineQuantization::default(),
        context.stream(),
    )
    .unwrap();
    let actual = materialize(&transformed, "rank.expert.weight", context.stream());
    assert_eq!(actual.shape(), &[1, 2, 8]);
    assert_eq!(
        actual.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
}

#[test]
fn complete_rank_three_dense_gguf_bank_is_read_one_matrix_row_at_a_time() {
    let values = matrix_values(2, 4, 64);
    let dense = Array::from_slice(&values, &[2, 4, 64]);
    let fixture = SyntheticGguf::dense(
        &HashMap::from([("model.experts.weight".to_string(), dense.clone())]),
        &HashMap::new(),
    );
    let source = Arc::new(
        open_gguf_checkpoint_source_for_test(
            GgufCheckpoint::open(fixture.path()).unwrap(),
            |name| name.to_string(),
        )
        .unwrap(),
    );
    let context = cpu_context();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        296,
        [direct_test_target("model.experts.weight")],
    )
    .unwrap();
    let transformed =
        BoundedQuantizedWeightStore::create(source.clone(), plan, context.stream()).unwrap();

    assert_eq!(transformed.report().source_tiles, 8);
    assert_eq!(transformed.report().source_bytes_read, 2_048);
    assert_eq!(transformed.report().output_bytes, 320);
    let diagnostics = source.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 8);
    assert_eq!(diagnostics.physical_read_bytes, 2_048);
    assert_eq!(
        transformed
            .source_metadata("model.experts.weight")
            .unwrap()
            .logical_shape,
        vec![2, 4, 8]
    );
    let expected =
        quantize_tensor(&dense, AffineQuantization::default(), context.stream()).unwrap();
    let actual = materialize(&transformed, "model.experts.weight", context.stream());
    assert_eq!(
        actual.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
}

#[test]
fn mxfp4_layout_has_byte_scales_and_no_biases() {
    let (_directory, source, values) = direct_fixture();
    let context = cpu_context();
    let plan = BoundedQuantizationPlan::new(
        WeightQuantization::MxFp4,
        290,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();
    assert_eq!(transformed.report().source_tiles, 8);
    assert_eq!(transformed.report().output_bytes, 272);
    assert_eq!(
        transformed
            .source_metadata("model.proj.scales")
            .unwrap()
            .stored_dtype,
        eredu_checkpoint::StoredDtype::U8
    );
    assert!(!transformed
        .source_keys()
        .contains(&"model.proj.biases".into()));

    assert_mxfp4_outputs_match_reference(&transformed, &values, &context);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn mxfp4_gpu_conversion_matches_the_canonical_gpu_quantizer() {
    let (_directory, source, values) = direct_fixture();
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let plan = BoundedQuantizationPlan::new(
        WeightQuantization::MxFp4,
        580,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();

    assert_mxfp4_outputs_match_reference(&transformed, &values, &context);
}

#[test]
fn dense_gguf_source_is_read_and_quantized_in_bounded_rows() {
    let values = matrix_values(1, 8, 64);
    let dense = Array::from_slice(&values, &[8, 64]);
    let fixture = SyntheticGguf::dense(
        &HashMap::from([("model.proj.weight".to_string(), dense.clone())]),
        &HashMap::new(),
    );
    let source = Arc::new(
        open_gguf_checkpoint_source_for_test(
            GgufCheckpoint::open(fixture.path()).unwrap(),
            |name| name.to_string(),
        )
        .unwrap(),
    );
    let context = cpu_context();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        320,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap();
    let transformed = BoundedQuantizedWeightStore::create(source, plan, context.stream()).unwrap();
    assert_eq!(
        transformed.source_diagnostics().unwrap().backend,
        WeightStoreBackend::Gguf
    );
    assert_eq!(transformed.report().source_tiles, 8);
    assert_eq!(transformed.report().source_bytes_read, 2_048);
    let diagnostics = transformed.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 8);
    assert_eq!(diagnostics.physical_read_bytes, 2_048);

    let expected =
        quantize_tensor(&dense, AffineQuantization::default(), context.stream()).unwrap();
    let actual = materialize(&transformed, "model.proj.weight", context.stream());
    assert_eq!(
        actual.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
}

#[test]
fn residency_budgets_and_arrays_use_only_packed_bytes() {
    let (_directory, source, _values) = direct_fixture();
    let conversion_context = cpu_context();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::default(),
        296,
        [direct_test_target("model.proj.weight")],
    )
    .unwrap();
    let transformed = Arc::new(
        BoundedQuantizedWeightStore::create(source, plan, conversion_context.stream()).unwrap(),
    );
    let id = OffloadUnitId::new("projection").unwrap();
    let bindings = [
        ("weight", "model.proj.weight", 256),
        ("scales", "model.proj.scales", 32),
        ("biases", "model.proj.biases", 32),
    ]
    .into_iter()
    .map(|(name, key, bytes)| WeightBinding::new(name, key, TensorSelection::Full, bytes).unwrap())
    .collect::<Vec<_>>();
    let host_capacity = host_capacity_upper_bound_for_bindings(&bindings).unwrap();
    let unit = OffloadUnit::new(id.clone(), bindings).unwrap();
    let spec =
        OffloadUnitSpec::new(id.clone(), 320, ResidencyPolicy::Pinned, MemoryTier::Host).unwrap();
    let offload = OffloadPlan::new(
        OffloadConfig::new(None, Some(host_capacity), 1).unwrap(),
        [spec],
    )
    .unwrap();
    let source_context = cpu_context();
    let device_context = cpu_context();
    let manager = ResidencyManager::new(
        transformed,
        offload,
        [unit],
        source_context.stream().clone(),
        device_context.stream().clone(),
    )
    .unwrap();
    manager.initialize().unwrap();
    let report = manager.report().unwrap();
    assert_eq!(report.offload().planned_bytes().get(MemoryTier::Host), 320);
    assert_eq!(
        report.offload().resident_bytes().get(MemoryTier::Host),
        host_capacity
    );
    let lease = manager.acquire(&id, MemoryTier::Host).unwrap();
    let resident_bytes = lease
        .binding_names()
        .map(|name| lease.host_value(name).unwrap().nbytes().unwrap())
        .sum::<usize>();
    assert_eq!(resident_bytes, 320);
}
