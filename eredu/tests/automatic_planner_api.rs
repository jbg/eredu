use eredu::{api::LocalBackendFactory, AutomaticPlanRequest, DevicePlan, ExecutionPlan};
use eredu_core::{realize_execution_plan_target, BackendProvider};

#[test]
fn portable_planner_realizes_an_owned_mlx_backend() {
    let device = DevicePlan::new("mlx", "cpu:0").unwrap();
    let request = AutomaticPlanRequest::new("model", device.clone());
    let encoded = serde_json::to_vec(&request).unwrap();
    let decoded: AutomaticPlanRequest = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, request);

    let plan = ExecutionPlan::fully_resident(device);
    let realization =
        realize_execution_plan_target(&LocalBackendFactory::default(), &plan).unwrap();
    let (backend, _) = realization.into_parts();
    assert_eq!(backend.descriptor().name, "mlx");
    assert_eq!(backend.devices().unwrap()[0].0.id, "cpu:0");
}
