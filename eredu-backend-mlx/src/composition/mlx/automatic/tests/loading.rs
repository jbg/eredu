use super::*;

#[test]
fn factory_inspection_retains_source_admission_through_native_loading_and_decode() {
    const TEST: &str = "composition::mlx::automatic::tests::loading::factory_inspection_retains_source_admission_through_native_loading_and_decode";
    let Ok(route) = std::env::var("EREDU_FACTORY_LOADING_ROUTE") else {
        for route in ["cpu", "metal", "host", "disk"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", TEST, "--nocapture"])
                .env("EREDU_FACTORY_LOADING_ROUTE", route)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success(),
                "{route}: {stdout}\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(stdout.contains(&format!("FACTORY_LOADING_OK:{route}")));
        }
        return;
    };
    let directory = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let factory = MlxBackendFactory::default();
    let pool = crate::backend::managed_memory::domain();
    crate::tests::support::path_instrumentation::reset();
    let inspection = factory
        .inspect_loading_artifact(
            directory.path(),
            &eredu_architectures::configuration::MODEL_CONFIGURATIONS,
        )
        .unwrap();
    assert!(pool
        .safetensors_shards_have_source_admission(inspection.safetensors_shards().unwrap())
        .unwrap());
    let paths = crate::tests::support::path_instrumentation::snapshot();
    assert_eq!(paths.payload_opens, 0);
    assert_eq!(paths.materializations, 0);
    assert_eq!(crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(), 0);

    // The retained inspection must supply configuration after the sidecar retires.
    std::fs::remove_file(directory.path().join("config.json")).unwrap();
    let device = if route == "cpu" { "cpu:0" } else { "metal:0" };
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", device).unwrap());
    let plan = match route.as_str() {
        "host" => plan.with_residency(ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(1 << 30),
            host_budget_bytes: Some(1 << 30),
        }),
        "disk" => plan.with_residency(ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 30,
            host_budget_bytes: 0,
            host_lookahead: 0,
            background_queue: 0,
        }),
        "cpu" | "metal" => plan,
        _ => panic!("unexpected fixture route"),
    };
    let selected = eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
    let target = eredu_core::realize_execution_plan_target(&factory, &plan, selected).unwrap();
    assert!(target.backend().memory_pool().same_domain(&pool));
    let stream = target.backend().stream().clone();
    let weights = target.backend().weights_stream().clone();
    let mut runtime = target.into_runtime().unwrap();
    let backend = MlxBackend::new(&stream, &weights);
    for token in [1, 2, 3] {
        let output = runtime.session_mut().submit_token_decode(&backend, token).unwrap().wait().unwrap();
        let logits = output.into_logits().unwrap().into_array();
        let values = logits.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        assert_eq!(values.len(), 64);
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != 0.0));
    }
    backend.synchronize().unwrap();
    println!("FACTORY_LOADING_OK:{route}");
}
