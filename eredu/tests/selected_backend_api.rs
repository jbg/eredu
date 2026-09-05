use eredu::api::realtime::{
    RealtimeInputFrame, RealtimePreparationPlan, RealtimeSampling, RequestId, SchedulerLimits,
    SessionCapabilities,
};
use eredu::api::{
    inspect_local_model, LocalBackendError, LocalBackendFactory, LocalDevice,
    LocalInspectionOptions, LocalLoadOptions, LocalRealtimeBackendFactory, LocalRealtimeModel,
    LocalRealtimeScheduler,
};
use eredu_core::{DevicePlan, ExecutionPlan, QuantizationRequest};

fn operate_selected_realtime_backend(
    preparation: RealtimePreparationPlan,
    frame: RealtimeInputFrame,
) -> Result<(), Box<dyn std::error::Error>> {
    let factory = LocalRealtimeBackendFactory::new(LocalDevice::Cpu);
    let mut model = factory.load(preparation)?;
    assert_eq!(model.backend_name(), "mlx");
    let _ = model.speech_config();

    let mut scheduler = LocalRealtimeScheduler::new(&model, SchedulerLimits::new(1, 4)?)?;
    let request = RequestId::new(7);
    scheduler.register_request(&model, request, RealtimeSampling::greedy())?;
    let _ = scheduler.enqueue(&model, request, frame)?;
    for completed in scheduler.run_bounded(&mut model, 1)? {
        let (_work, output) = completed.into_parts();
        let _ = output.text_tokens();
    }
    scheduler.finish_request(request)?;
    Ok(())
}

#[test]
fn facade_exposes_complete_selected_realtime_operations() {
    let _: fn(
        RealtimePreparationPlan,
        RealtimeInputFrame,
    ) -> Result<(), Box<dyn std::error::Error>> = operate_selected_realtime_backend;
    let _ = std::mem::size_of::<LocalRealtimeModel>();
}

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
        LocalInspectionOptions::for_execution_plan(&LocalBackendFactory::default(), &plan).unwrap();
    assert_eq!(
        planned.load().drafting(),
        eredu_runtime::DraftingLoadRequest::Disabled
    );
}

#[test]
fn selected_inspection_is_total_for_missing_artifacts() {
    let result: Result<eredu_core::ModelInspectionReport, LocalBackendError> = inspect_local_model(
        "/path/that/does/not/exist/eredu-selected-backend-api",
        LocalInspectionOptions::default(),
    );
    let report = result.unwrap();
    assert_eq!(report.container, eredu_core::InspectionReadiness::Missing);
}
