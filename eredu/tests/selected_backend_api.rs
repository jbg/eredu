use eredu::api::{inspect_local_model, LocalInspectionOptions};

#[path = "selected_backend_api/qwen4_exp.rs"]
mod qwen4_exp;

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
