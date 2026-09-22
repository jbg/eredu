use super::*;
use crate::backend::{ExecutionContext, MlxBackend};
use eredu_core::ModelLoadingBackend as _;
use safemlx::{Array, Device, DeviceType};

fn sources(
    backend: &MlxBackend<'_>,
    root: &std::path::Path,
    residency: eredu_runtime::WeightResidency,
) -> PreparedModelSources {
    let inspection = backend.inspect_model_artifact(root).unwrap();
    let options = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::with_quantization(
            eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            },
        )
        .with_weight_residency(residency),
    );
    let policy = options.normalized().preparation_policy().unwrap();
    let selected =
        crate::composition::mlx::loading::select_preparation(&inspection, options).unwrap();
    let plan =
        eredu_core::plan_model_preparation(inspection, policy, selected.session_capabilities())
            .unwrap();
    crate::composition::mlx::loading::prepare_selected_sources(plan, selected, None)
        .unwrap()
        .0
}

#[test]
fn selected_affine_conversion_reaches_native_loading_and_matches_resident_execution() {
    if !crate::tests::support::native_process::enter("original-affine-source") {
        return;
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let backend = MlxBackend::new(stream, stream).with_memory_ledger(pool.clone());
    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
    );
    for family in ["llama", "qwen3"] {
        let root = crate::composition::mlx::replicated_text::tests::tiny_artifact(family, false);
        let mut expected = None;
        for residency in [
            eredu_runtime::WeightResidency::fully_resident(),
            eredu_runtime::WeightResidency::layerwise_host(host),
            eredu_runtime::WeightResidency::dense_disk_stream(Default::default()),
        ] {
            eprintln!("conversion loading: {family}, {residency:?}");
            let sources = sources(&backend, root.path(), residency);
            let manager = prepare_layerwise_manager(&sources, &pool, stream, stream).unwrap();
            if matches!(
                sources.selected().text_realization().residency(),
                LayerWeightResidency::FullyResident
            ) {
                assert!(manager.is_none());
            } else {
                let manager = manager
                    .as_ref()
                    .expect("selected affine producer must prepare residency");
                assert_eq!(manager.conversions.len(), 1);
                assert!(!manager.conversions.as_slice()[0]
                    .store()
                    .same_source(sources.target()));
            }
            let construction_sources = crate::composition::mlx::loading::PreparedNativeConstructionSources::prepare(
                sources.selected().text_realization().state(), manager, backend.memory_ledger(), stream,
            ).unwrap();
            let capabilities = sources.selected().session_capabilities();
            let model = backend
                .materialize_after_communication(capabilities, None, None, |_| {
                    crate::composition::mlx::loading::materialize_model_plan_with_construction_sources(
                        sources, None, stream, stream, Some(construction_sources), None,
                    )
                })
                .unwrap()
                .into_inner();
            assert!(model.materialization_report().unwrap().transformed_weights > 0);
            let mut executable = model.into_executable();
            let mut observed = Vec::new();
            for token in [1_u32, 2, 3] {
                let logits = executable
                    .erased_mut()
                    .decode(&Array::from_slice(&[token], &[1, 1]), stream)
                    .unwrap();
                let logits = logits.evaluated().unwrap().as_slice::<f32>().to_vec();
                assert!(logits.iter().all(|value| value.is_finite()));
                assert!(logits.iter().any(|value| *value != 0.0));
                observed.extend(logits);
            }
            match &expected {
                Some(expected) => assert_eq!(&observed, expected, "{family} residency parity"),
                None => expected = Some(observed),
            }
            drop(executable);
            crate::backend::submission_recovery::wait_for_retirement(|| {
                crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
                pool.unquoted_owner_count().unwrap() == 0
            });
        }
    }
}

#[test]
fn accepted_conversion_resource_failure_is_an_error_not_ordinary_fallback() {
    if !crate::tests::support::native_process::enter("original-affine-source") {
        return;
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let backend = MlxBackend::new(stream, stream).with_memory_ledger(pool.clone());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
    );
    let sources = sources(
        &backend,
        root.path(),
        eredu_runtime::WeightResidency::layerwise_host(host),
    );
    let foreign = crate::memory_fixture::ledger(1 << 30, 0).unwrap();
    assert!(
        prepare_layerwise_manager(&sources, &foreign, stream, stream)
            .unwrap()
            .is_none()
    );
    let reads = sources
        .target()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    let ordinary = pool.acquire_unquoted().unwrap();
    let error = prepare_layerwise_manager(&sources, &pool, stream, stream)
        .err()
        .expect("admitted conversion may not return None after resource rejection");
    let mut cause: &dyn std::error::Error = &error;
    loop {
        if let Some(cause) =
            cause.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>()
        {
            assert!(matches!(
                cause,
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound
            ));
            break;
        }
        cause = cause
            .source()
            .expect("typed resource admission cause is preserved");
    }
    assert_eq!(
        sources
            .target()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        reads
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(ordinary);
}
