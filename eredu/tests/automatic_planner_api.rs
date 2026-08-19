use eredu::{
    backend::mlx::automatic::MlxBackendFactory, core::realize_execution_plan_target,
    AutomaticPlanRequest, AutomaticPlanner, AutomaticPlannerPolicy, Backend, DevicePlan,
    ExecutionPlan, AUTOMATIC_SCHEMA_VERSION,
};

#[test]
fn portable_planner_realizes_an_owned_mlx_backend() {
    let device = DevicePlan::new("mlx", "cpu:0").unwrap();
    let request = AutomaticPlanRequest::new("model", device.clone());
    assert_eq!(request.schema_version, AUTOMATIC_SCHEMA_VERSION);

    let planner = AutomaticPlanner::new(AutomaticPlannerPolicy::default());
    assert_eq!(
        planner.policy().memory_headroom_percent,
        AutomaticPlannerPolicy::default().memory_headroom_percent
    );

    let encoded = serde_json::to_vec(&request).unwrap();
    let decoded: AutomaticPlanRequest = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, request);

    let plan = ExecutionPlan::fully_resident(device);
    let realization = realize_execution_plan_target(&MlxBackendFactory::default(), &plan).unwrap();
    let (backend, _) = realization.into_parts();
    assert_eq!(backend.descriptor().name, "mlx");
    assert_eq!(backend.devices().unwrap()[0].0.id, "cpu:0");
}
