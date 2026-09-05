use super::*;
use eredu_core::BackendProvider as _;

#[test]
fn realtime_capability_rejection_precedes_device_and_stream_realization() {
    crate::tests::support::path_instrumentation::reset();
    let directory = tempfile::tempdir().expect("tiny realtime artifact directory");
    super::super::realtime::tests::write_tiny_native_artifact(directory.path(), None);
    let preparation = eredu_architectures::moshi::prepare_realtime_model(directory.path())
        .expect("tiny realtime artifact is valid");
    let device = DevicePlan::new("mlx", "gpu:0").expect("portable device name is valid");
    let options = MlxLoadRequest::default().with_required_session_capabilities(
        eredu_core::SessionCapabilities::default().with_activation_inspection(true),
    );

    let error = match create_realtime_execution(preparation, &device, options) {
        Ok(_) => panic!("unsupported activation observation must reject selection"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("activation_inspection"));
    assert_eq!(
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(),
        0
    );
}

#[test]
fn mlx_discovery_always_reports_cpu() {
    let profile = discover_hardware();
    assert!(profile.backends.iter().any(|backend| {
        backend.backend.as_str() == "mlx"
            && backend.available
            && backend.devices.iter().any(|device| device.id == "cpu:0")
    }));
}

#[test]
fn plan_realization_rejects_distributed_topology() {
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap())
        .with_topology(eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap());
    assert!(mlx_load_options(&MlxBackendFactory::default(), &plan).is_err());
}

#[test]
fn failed_target_selection_creates_no_native_resources() {
    crate::tests::support::path_instrumentation::reset();
    let directory = tempfile::tempdir().expect("tiny inspected artifact directory");
    super::super::realtime::tests::write_tiny_native_artifact(directory.path(), None);
    let inspection = eredu_architectures::configuration::inspect_artifact(directory.path())
        .expect("tiny artifact inspection succeeds");
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap())
        .with_topology(eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap());

    let error = match eredu_core::select_execution_plan_target(
        &MlxBackendFactory::default(),
        &plan,
        inspection,
    ) {
        Ok(_) => panic!("distributed topology must fail before native realization"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("1x1x1"));
    assert_eq!(
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(),
        0
    );
}

#[test]
fn bounded_probe_selects_before_native_resources() {
    crate::tests::support::path_instrumentation::reset();
    let directory = tempfile::tempdir().expect("tiny realtime artifact directory");
    super::super::realtime::tests::write_tiny_native_artifact(directory.path(), None);
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap())
        .with_residency(ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: None,
            host_budget_bytes: None,
        });

    let inspection = eredu_architectures::configuration::inspect_artifact(directory.path())
        .expect("realtime artifact inspection");
    let error = MlxBackendFactory::default()
        .bounded_residency_requirement(&inspection, &plan)
        .expect_err("realtime architecture must not enter ordinary bounded probing");

    assert!(
        matches!(
            &error,
            AutomaticPlanningError::Backend {
                operation: "select_model_preparation",
                message,
            } if message.contains("loading protocol Realtime")
        ),
        "{error:?}"
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(),
        0
    );
}

#[test]
fn bounded_requirement_for_ordinary_text_uses_only_the_cold_selected_tasks() {
    crate::tests::support::path_instrumentation::reset();
    let directory = super::super::replicated_text::tests::tiny_artifact("llama", false);
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap())
        .with_residency(ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: None,
            host_budget_bytes: None,
        });

    let inspection = eredu_architectures::configuration::inspect_artifact(directory.path())
        .expect("ordinary artifact inspection");
    let requirement = MlxBackendFactory::default()
        .bounded_residency_requirement(&inspection, &plan)
        .expect("ordinary selected tasks have an exact bounded requirement");

    assert!(requirement.static_bytes > 0);
    assert!(requirement.window_bytes > 0);
    assert_eq!(
        requirement.required_bytes,
        requirement.static_bytes + requirement.window_bytes
    );
    assert_eq!(requirement.depth, 1);
    assert_eq!(
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(),
        0
    );
    let counts = crate::tests::support::path_instrumentation::snapshot();
    assert_eq!(counts.payload_opens, 0);
    assert_eq!(counts.architecture_constructions, 0);
    assert_eq!(counts.constructors, 0);
    assert_eq!(counts.materializations, 0);
}

#[test]
fn realized_cpu_identity_is_derived_from_the_native_device() {
    let plan = DevicePlan::new("mlx", "cpu:0").unwrap();
    let realized = mlx_device(&plan).unwrap();
    let stream = Stream::try_new_with_device(&realized.device).unwrap();
    let backend = MlxBackend::for_execution_plan(&stream, &stream, realized.identity);

    let devices = backend.devices().unwrap();
    assert_eq!(devices[0].0.id(), "cpu:0");
    assert_eq!(devices[0].0.family(), "cpu");
}

#[test]
fn realization_rejects_generic_gpu_family() {
    let plan = DevicePlan::new("mlx", "gpu:0").unwrap();
    let error = mlx_device(&plan).unwrap_err();
    assert!(error
        .to_string()
        .contains("unknown MLX device family \"gpu\""));
}

#[test]
fn realization_rejects_noncanonical_device_identity() {
    let plan = DevicePlan::new("mlx", "cpu:00").unwrap();
    let error = mlx_device(&plan).unwrap_err();
    assert!(error.to_string().contains("is not canonical"));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn realization_rejects_cuda_without_compiled_support() {
    let plan = DevicePlan::new("mlx", "cuda:0").unwrap();
    let error = mlx_device(&plan).unwrap_err();
    assert!(error
        .to_string()
        .contains("cuda device family is not compiled"));
}

#[cfg(not(all(feature = "metal", target_vendor = "apple")))]
#[test]
fn realization_rejects_metal_without_compiled_support() {
    let plan = DevicePlan::new("mlx", "metal:0").unwrap();
    let error = mlx_device(&plan).unwrap_err();
    assert!(error
        .to_string()
        .contains("metal device family is not compiled"));
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn realized_metal_plan_reports_metal_from_the_native_binding() {
    if !safemlx::metal::is_available().unwrap() {
        return;
    }
    let plan = DevicePlan::new("mlx", "metal:0").unwrap();
    let realized = mlx_device(&plan).unwrap();
    let stream = Stream::try_new_with_device(&realized.device).unwrap();
    let backend = MlxBackend::for_execution_plan(&stream, &stream, realized.identity);
    let devices = backend.devices().unwrap();

    assert_eq!(devices[0].0.id(), "metal:0");
    assert_eq!(devices[0].0.family(), "metal");
}
