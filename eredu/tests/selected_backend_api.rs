use eredu::{
    api::{LocalDevice, LocalRealtimeBackendFactory, LocalRealtimeModel, LocalRealtimeScheduler},
    RealtimeInputFrame, RealtimePreparationPlan, RealtimeSampling, RequestId, SchedulerLimits,
};

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
