use eredu::api::realtime::SessionCapabilities;
use eredu::api::{inspect_local_model, LocalInspectionOptions, LocalLoadOptions};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{DevicePlan, ExecutionPlan, QuantizationRequest};

#[path = "selected_backend_api/qwen4_exp.rs"]
mod qwen4_exp;

#[test]
fn selected_load_policy_is_facade_owned_and_portable() {
    let required = SessionCapabilities::default().with_persistent_cache(true);
    let options = LocalLoadOptions::with_quantization(QuantizationRequest::MxFp4)
        .with_required_session_capabilities(required);

    assert_eq!(options.quantization(), Some(QuantizationRequest::MxFp4));
    assert_eq!(options.required_session_capabilities(), required);
    assert_eq!(
        options.weight_residency(),
        eredu_runtime::WeightResidency::fully_resident()
    );

    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap());
    let planned =
        LocalInspectionOptions::for_execution_plan(&MlxBackendFactory::default(), &plan).unwrap();
    assert_eq!(
        planned.load().drafting(),
        eredu_runtime::DraftingLoadRequest::Disabled
    );
}

#[test]
fn selected_inspection_is_total_for_missing_artifacts() {
    let result: Result<eredu_core::ModelInspectionReport, eredu_core::BackendFailure> =
        inspect_local_model(
            "/path/that/does/not/exist/eredu-selected-backend-api",
            LocalInspectionOptions::default(),
        );
    let report = result.unwrap();
    assert_eq!(report.container, eredu_core::InspectionReadiness::Missing);
}
