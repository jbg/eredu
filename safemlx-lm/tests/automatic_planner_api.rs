use safemlx_lm::{
    execution_plan_load_options, AutomaticPlanRequest, AutomaticPlanner, AutomaticPlannerPolicy,
    DevicePlan, ExecutionPlan, AUTOMATIC_SCHEMA_VERSION,
};

#[test]
fn automatic_planner_surface_is_available_from_the_crate_root() {
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
    assert!(execution_plan_load_options(&plan).is_ok());
}
