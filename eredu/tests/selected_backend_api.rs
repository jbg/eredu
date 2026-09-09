use eredu::api::realtime::{
    RealtimeInputFrame, RealtimePreparationPlan, RealtimeSampling, RequestId, SchedulerLimits,
    SessionCapabilities,
};
use eredu::api::{inspect_local_model, LocalInspectionOptions, LocalLoadOptions};
use eredu_backend_mlx::{backend::MlxBackend, MlxBackendFactory};
use eredu_core::{DevicePlan, ExecutionPlan, QuantizationRequest};

fn operate_selected_text_control(
    model: &mut eredu::api::LoadedModel<MlxBackend<'static>>,
    request: eredu::api::PreparedObservedGeneration,
) -> Result<(), Box<dyn std::error::Error>> {
    use eredu_core::execution_control::{GenerationControlHandle, SnapshotLimits};
    use std::ops::ControlFlow;
    let mut run =
        model.start_controlled_chat(request, &[], GenerationControlHandle::default(), |_| {
            ControlFlow::Continue(())
        })?;
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 128 << 20,
        cumulative_copy_bytes: 256 << 20,
    })?;
    let snapshot: eredu::api::ControlledGenerationSnapshot<MlxBackend<'static>> =
        run.snapshot(|_| ControlFlow::Continue(()))?;
    run.step(|_| ControlFlow::Continue(()))?;
    run.pause(|_| ControlFlow::Continue(()))?;
    run.restore(&snapshot, |_| ControlFlow::Continue(()))?;
    assert_eq!(run.token_ids(), snapshot.token_ids());
    let mut branch = run.fork(
        &snapshot,
        eredu::api::GenerationBranchOptions {
            trace_limits: eredu::api::TraceLimits {
                per_record_bytes: 16384,
                total_bytes: 65536,
            },
            capture_limits: None,
            sampling: None,
            intervention: None,
        },
        |_| ControlFlow::Continue(()),
    )?;
    run.exchange(&mut branch, |_| ControlFlow::Continue(()))?;
    let _ = run.output_checkpoint();
    let _ = run.sampling_state()?;
    Ok(())
}

#[test]
#[allow(clippy::type_complexity)]
fn mlx_text_control_uses_generic_model_and_snapshot_types() {
    let _: fn(
        &mut eredu::api::LoadedModel<MlxBackend<'static>>,
        eredu::api::PreparedObservedGeneration,
    ) -> Result<(), Box<dyn std::error::Error>> = operate_selected_text_control;
}

fn operate_selected_realtime_backend(
    preparation: RealtimePreparationPlan,
    frame: RealtimeInputFrame,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = DevicePlan::new("mlx", "cpu:0")?;
    let (backend, execution) = eredu_backend_mlx::create_realtime_execution(
        preparation,
        &device,
        eredu_backend_mlx::MlxLoadRequest::default(),
    )?;
    let selected = execution.selected().clone();
    let mut model = eredu::api::realtime::PreparedRealtimeModel::new(
        execution,
        &selected,
        SessionCapabilities::new(true, true, false),
    );
    let schedule = model.session_identity().schedule().clone();
    let mut scheduler = eredu_runtime::RealtimeSessionScheduler::new(
        model.session_identity().clone(),
        SchedulerLimits::new(1, 4)?,
    )?;
    let sampling = RealtimeSampling::greedy();
    let samplers = eredu_architectures::moshi::realtime_generation_samplers(&schedule, sampling)?;
    let state = eredu_runtime::RealtimePayloadState::fresh(
        backend.new_realtime_model_state(model.mechanism())?,
        schedule.clone(),
    );
    let generation = eredu_runtime::RealtimeGenerationState::new(
        state,
        schedule,
        sampling,
        samplers,
        backend.realize_random_state(None)?,
    )?;
    let request = RequestId::new(7);
    scheduler.register(request, generation)?;
    scheduler.enqueue(request, frame)?;
    let progress =
        scheduler.run_local_bounded(std::time::Instant::now(), 1, |_, frame, branch| {
            backend.submit_realtime_frame(model.mechanism_mut(), frame, branch)
        })?;
    if let Some((_, failure)) = progress.failed.first() {
        return Err(failure.to_string().into());
    }
    for (_, _, transition) in progress.committed {
        let output = transition.into_host_output()?;
        let _ = output.text_tokens();
    }
    let session = scheduler.release(request)?;
    let _ = session.committed_batch();
    scheduler.resume(request, &mut Some(session))?;
    scheduler.finish(request)?;
    Ok(())
}

#[test]
fn mlx_realtime_uses_generic_model_scheduler_and_session_types() {
    let _: fn(
        RealtimePreparationPlan,
        RealtimeInputFrame,
    ) -> Result<(), Box<dyn std::error::Error>> = operate_selected_realtime_backend;
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
