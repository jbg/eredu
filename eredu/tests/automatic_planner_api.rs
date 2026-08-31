use eredu::api::{LocalBackendFactory, LocalInspectionOptions};
use eredu_core::{AutomaticPlanRequest, DevicePlan, ExecutionPlan};

#[test]
fn portable_planner_inputs_do_not_expose_the_local_backend() {
    let device = DevicePlan::new("mlx", "cpu:0").unwrap();
    let request = AutomaticPlanRequest::new("model", device.clone());
    let encoded = serde_json::to_vec(&request).unwrap();
    let decoded: AutomaticPlanRequest = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, request);

    let plan = ExecutionPlan::fully_resident(device);
    let options =
        LocalInspectionOptions::for_execution_plan(&LocalBackendFactory::default(), &plan).unwrap();
    assert_eq!(options.load, Default::default());
}
