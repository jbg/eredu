use super::*;
use crate::composition::mlx::replicated_text::tests::{
    routed_deepseek_v4_config, tiny_heterogeneous_artifact,
};

fn v4_artifact() -> tempfile::TempDir {
    let mut config = routed_deepseek_v4_config();
    config["rope_scaling"] = serde_json::json!({
        "rope_type": "yarn", "factor": 8.0,
        "original_max_position_embeddings": 128,
        "beta_fast": 32.0, "beta_slow": 1.0
    });
    tiny_heterogeneous_artifact(config)
}

fn disk_runtime(
    stream: &Stream,
    pool: &WorkingMemoryPool,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let source = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(stream, &source).with_memory_pool(pool.clone());
    let root = v4_artifact();
    let options = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::dense_disk_stream(
                eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 0, 0, 0).unwrap(),
            ),
        ),
    );
    let model = eredu_core::load_model(&backend, root.path(), options).unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    let workspace = runtime
        .session()
        .payload
        .model
        .layerwise_workspace()
        .unwrap()
        .unwrap();
    assert!(workspace.materialization().bytes().unwrap() > 0);
    let report = runtime.session().residency_report().unwrap().unwrap();
    assert!(report
        .units()
        .iter()
        .any(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Disk));
    (runtime, root)
}

fn cold_quote_and_short_rejection(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    pool: &WorkingMemoryPool,
) -> u64 {
    let controller = Controller::default();
    let baseline = pool.used_bytes().unwrap();
    let peak = pool.peak_bytes().unwrap();
    let before_paths = paths::snapshot();
    let before_inputs = paths::session_input_creation_attempts();
    let before_resets = paths::session_reset_attempts();
    let before_sources = runtime.session().residency_report().unwrap();
    let before_state = runtime.session().payload.model.erased().state_snapshot();
    let (first, width) = text_quote::quote(
        runtime.session(),
        geometry(),
        20,
        config(0.7, Some(u64::MAX)),
        controller.inference_workspace(4).unwrap(),
    )
    .unwrap();
    assert_eq!(width, 64);
    let (second, _) = text_quote::quote(
        runtime.session(),
        geometry(),
        20,
        config(0.7, Some(u64::MAX)),
        controller.inference_workspace(4).unwrap(),
    )
    .unwrap();
    assert_eq!(first, second);
    assert!(
        first
            .execution_workspace
            .as_ref()
            .unwrap()
            .materialization
            .bytes()
            .unwrap()
            > 0
    );
    assert_eq!(pool.used_bytes().unwrap(), baseline);
    assert_eq!(pool.peak_bytes().unwrap(), peak);
    let capacity = exact_capacity(runtime, pool, 0.7);
    let before_rejection_peak = pool.peak_bytes().unwrap();
    let error = ControlledTextGeneration::from_input(
        runtime,
        TextGenerationInput::TokenIds(tokens()),
        config(0.7, Some(capacity - 1)),
        controller.clone(),
    )
    .err()
    .expect("one-position chunk cannot shrink below exact capacity");
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(controller.0.get(), (0, 0));
    assert_eq!(paths::snapshot(), before_paths);
    assert_eq!(paths::session_input_creation_attempts(), before_inputs);
    assert_eq!(paths::session_reset_attempts(), before_resets);
    assert_eq!(
        runtime.session().residency_report().unwrap(),
        before_sources
    );
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        before_state
    );
    assert_eq!(pool.used_bytes().unwrap(), baseline);
    assert_eq!(pool.peak_bytes().unwrap(), before_rejection_peak);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    capacity
}

#[test]
fn v4_fixed_rotary_host_construction_is_bounded_for_both_host_windows_and_drivers() {
    if !crate::tests::support::native_process::enter("main") { return; }
    let (streams, pool, native_baseline) = crate::tests::support::native_process::metal();
    let stream = streams.execution();
    let mut reference = None;
    for depth in [None, Some(1), Some(2)] {
        for controlled in [false, true] {
            let (mut runtime, root) = runtime_from_artifact(&stream, &pool, depth, v4_artifact());
            let capacity = if depth.is_some() {
                cold_quote_and_short_rejection(&mut runtime, &pool)
            } else {
                exact_capacity(&runtime, &pool, 0.7)
            };
            let before = paths::bounded_unit_acquisitions();
            // Five prompt positions cross the ratio-four pooling frontier;
            // three cached decodes follow, using all five constructor helpers.
            let output = outputs(&mut runtime, 0.7, capacity, controlled);
            let ids = output
                .iter()
                .map(|token| token.token_id().unwrap())
                .collect::<Vec<_>>();
            assert!(ids.iter().all(|id| *id < 64));
            if let Some(reference) = &reference {
                assert_eq!(&ids, reference);
            } else {
                reference = Some(ids);
            }
            if let Some(depth) = depth {
                assert!(paths::bounded_unit_acquisitions() - before >= 3 * (5 + 3));
                let report = runtime.session().residency_report().unwrap().unwrap();
                let layers = execution_units(&report);
                assert!(layers.iter().all(|unit| unit.host_resident()));
                assert!(layers.iter().filter(|unit| unit.device_resident()).count() <= depth);
            }
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            assert!(pool.peak_bytes().unwrap() <= capacity);
            drop((output, runtime, root));
            settle(&pool, native_baseline);
        }
    }
}

#[test]
fn v4_fixed_rotary_direct_disk_construction_matches_resident_with_exact_admission() {
    if !crate::tests::support::native_process::enter("main") { return; }
    let (streams, pool, native_baseline) = crate::tests::support::native_process::metal();
    let stream = streams.execution();
    let reference = {
        let (mut runtime, root) = runtime_from_artifact(&stream, &pool, None, v4_artifact());
        let capacity = exact_capacity(&runtime, &pool, 0.7);
        let output = outputs(&mut runtime, 0.7, capacity, false);
        let ids = output
            .iter()
            .map(|token| token.token_id().unwrap())
            .collect::<Vec<_>>();
        drop((output, runtime, root));
        settle(&pool, native_baseline);
        ids
    };
    for controlled in [false, true] {
        let (mut runtime, root) = disk_runtime(&stream, &pool);
        let capacity = cold_quote_and_short_rejection(&mut runtime, &pool);
        let output = outputs(&mut runtime, 0.7, capacity, controlled);
        assert_eq!(
            output
                .iter()
                .map(|token| token.token_id().unwrap())
                .collect::<Vec<_>>(),
            reference
        );
        let report = runtime.session().residency_report().unwrap().unwrap();
        let units = report
            .units()
            .iter()
            .filter(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Disk)
            .collect::<Vec<_>>();
        assert!(!units.is_empty());
        assert!(units.iter().all(|unit| !unit.host_resident()));
        assert!(units.iter().filter(|unit| unit.device_resident()).count() <= 1);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert!(pool.peak_bytes().unwrap() <= capacity);
        drop((output, runtime, root));
        settle(&pool, native_baseline);
    }
}
