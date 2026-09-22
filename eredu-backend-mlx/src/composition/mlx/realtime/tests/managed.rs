//! Managed frames use the same model, scheduler and native decision worker.
use super::*;
use eredu_core::residency::{MemoryTier, ResidencyPolicy, TransferDirection};
use eredu_runtime::working_memory::WorkingMemoryError;

const CAPACITY: u64 = 16u64 << 30;

fn managed_scheduler(
    model: &SelectedTestModel,
    request: RequestId,
    sampling: RealtimeSampling,
) -> MlxManagedRealtimeScheduler {
    let mut scheduler = MlxManagedRealtimeScheduler::new_with_preparation(
        RealtimeModelSessionIdentity::from_selected(model.model.selected()),
        SchedulerLimits::new(1, 1).unwrap(),
    )
    .unwrap();
    let schedule = model.model.execution_config().frame_schedule().clone();
    let samplers =
        eredu_architectures::moshi::realtime_generation_samplers(&schedule, sampling).unwrap();
    let state = RealtimePayloadState::fresh(
        model
            .backend
            .new_realtime_model_state(&model.model)
            .unwrap(),
        schedule.clone(),
    );
    let random = model
        .backend
        .realize_random_state(sampling.is_stochastic().then_some(sampling.seed()))
        .unwrap();
    scheduler
        .register(
            request,
            RealtimeGenerationState::new(state, schedule, sampling, samplers, random).unwrap(),
        )
        .unwrap();
    scheduler
}

fn load(path: &Path) -> SelectedTestModel {
    load_with_options(path, MlxLoadRequest::default())
}
fn load_with_options(path: &Path, options: MlxLoadRequest) -> SelectedTestModel {
    let (backend, model) = crate::adapter::create_realtime_execution(
        prepare(path),
        &eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
        options,
    )
    .unwrap();
    SelectedTestModel { backend, model }
}

fn inputs() -> Vec<RealtimeInputFrame> {
    vec![
        RealtimeInputFrame::new(1, vec![1]),
        RealtimeInputFrame::new(1, vec![2])
            .with_forced_text(vec![7])
            .with_forced_generated_audio(vec![9]),
        RealtimeInputFrame::new(1, vec![3]).with_forced_text(vec![11]),
        RealtimeInputFrame::new(1, vec![4]),
    ]
}

fn tokens(frame: &RealtimeOutputFrame) -> (Vec<i32>, Vec<i32>, Vec<i32>, Option<Vec<i32>>) {
    (
        frame.text_tokens().to_vec(),
        frame.decision_audio_tokens().to_vec(),
        frame.sampled_audio_tokens().to_vec(),
        frame.output_audio_tokens().map(<[i32]>::to_vec),
    )
}

fn drive_managed_frame(
    model: &mut SelectedTestModel,
    scheduler: &mut MlxManagedRealtimeScheduler,
    request: RequestId,
    frame: RealtimeInputFrame,
) -> RealtimeOutputFrame {
    scheduler.enqueue(request, frame).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let mut progress = model
            .backend
            .run_realtime_bounded(
                &mut model.model,
                scheduler,
                std::time::Instant::now(),
                1,
                &crate::memory_fixture::limits(CAPACITY),
            )
            .unwrap();
        if let Some((_, _, output)) = progress.committed.pop() {
            return output.into_host_output().unwrap();
        }
        assert!(
            std::time::Instant::now() < deadline,
            "managed frame did not finish"
        );
        std::thread::yield_now();
    }
}

#[test]
#[ignore = "requires Metal; first managed realtime frame admission regression"]
fn native_managed_realtime_frames_match_seeded_ordinary_and_refuse_before_branch() {
    let directory = tempfile::tempdir().unwrap();
    write_tiny_native_artifact_with_values(directory.path(), None, true);
    let sampling = RealtimeSampling::new(0.7, 0.8, 431).unwrap();
    let request = RequestId::new(941);
    let expected = {
        let mut model = load(directory.path());
        let mut scheduler = selected_scheduler(&model, request, sampling);
        inputs()
            .into_iter()
            .map(|frame| {
                tokens(&drive_selected_frame(
                    &mut model,
                    &mut scheduler,
                    request,
                    frame,
                ))
            })
            .collect::<Vec<_>>()
    };
    crate::backend::ordinary_retirement::reclaim();
    let mut model = load(directory.path());
    let mut scheduler = managed_scheduler(&model, request, sampling);
    let frame = RealtimeInputFrame::new(1, vec![1]);
    let original_frontier = scheduler
        .request_state(request)
        .unwrap()
        .generation()
        .schedule_state()
        .frontier();
    let error = model
        .backend
        .prepare_realtime_frame(
            &model.model,
            &frame,
            scheduler.request_state(request).unwrap(),
            &crate::memory_fixture::limits(1),
        )
        .err()
        .expect("one-byte frame allowance rejects before branch construction");
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let mut budget = false;
    while let Some(cause) = current {
        budget |= matches!(
            cause.downcast_ref::<WorkingMemoryError>(),
            Some(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        );
        current = cause.source();
    }
    assert!(budget, "expected typed budget refusal, got {error:?}");
    drop(error);
    assert_eq!(
        scheduler
            .request_state(request)
            .unwrap()
            .generation()
            .schedule_state()
            .frontier(),
        original_frontier
    );
    for (index, frame) in inputs().into_iter().enumerate() {
        let actual = drive_managed_frame(&mut model, &mut scheduler, request, frame);
        assert_eq!(tokens(&actual), expected[index], "frame {index}");
        assert_eq!(
            scheduler
                .request_state(request)
                .unwrap()
                .generation()
                .schedule_state()
                .frontier(),
            original_frontier + index + 1
        );
        if index == 1 {
            // A paid but unsubmitted branch must not change canonical KV,
            // history, samplers or RNG. Later frames still match the same
            // uninterrupted seeded ordinary reference.
            let next = RealtimeInputFrame::new(1, vec![3]).with_forced_text(vec![11]);
            let branch = model
                .backend
                .prepare_realtime_frame(
                    &model.model,
                    &next,
                    scheduler.request_state(request).unwrap(),
                    &crate::memory_fixture::limits(CAPACITY),
                )
                .unwrap();
            <MlxManagedRealtimeSessionState as eredu_core::scheduler::SemanticStateTransaction>
                ::discard_branch(branch).unwrap();
            assert_eq!(
                scheduler
                    .request_state(request)
                    .unwrap()
                    .generation()
                    .schedule_state()
                    .frontier(),
                original_frontier + index + 1
            );
            let mut released = Some(scheduler.release(request).unwrap());
            assert!(scheduler.request_state(request).is_none());
            scheduler.resume(request, &mut released).unwrap();
            assert!(released.is_none());
        }
    }
}

fn verify_bounded(residency: WeightResidency, execution: ExecutionResidency) {
    let directory = tempfile::tempdir().unwrap();
    write_tiny_native_artifact_with_values(directory.path(), None, true);
    let sampling = RealtimeSampling::new(0.7, 0.8, 431).unwrap();
    let request = RequestId::new(942);
    let expected = {
        let mut model = load(directory.path());
        let mut scheduler = selected_scheduler(&model, request, sampling);
        inputs()
            .into_iter()
            .map(|frame| {
                tokens(&drive_selected_frame(
                    &mut model,
                    &mut scheduler,
                    request,
                    frame,
                ))
            })
            .collect::<Vec<_>>()
    };
    crate::backend::ordinary_retirement::reclaim();
    let options = || {
        MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default()
                .with_weight_residency(residency.clone()),
        )
    };
    {
        let mut model = load_with_options(directory.path(), options());
        assert_eq!(model.model.executor().metadata().residency(), execution);
        let mut scheduler = selected_scheduler(&model, request, sampling);
        for (index, frame) in inputs().into_iter().enumerate() {
            assert_eq!(
                tokens(&drive_selected_frame(
                    &mut model,
                    &mut scheduler,
                    request,
                    frame
                )),
                expected[index],
                "ordinary bounded frame {index}"
            );
        }
    }
    crate::backend::ordinary_retirement::reclaim();
    let mut model = load_with_options(directory.path(), options());
    assert_eq!(model.model.executor().metadata().residency(), execution);
    let mut scheduler = managed_scheduler(&model, request, sampling);
    let disk_before = if execution == ExecutionResidency::DenseDiskStream {
        let report = model
            .model
            .executor()
            .dense_stream_report()
            .unwrap()
            .unwrap();
        let reads = report
            .residency()
            .offload()
            .transfer(TransferDirection::DiskToHost);
        Some((
            reads.count(),
            reads.bytes(),
            report.background().completed(),
        ))
    } else {
        None
    };
    for (index, frame) in inputs().into_iter().enumerate() {
        let actual = drive_managed_frame(&mut model, &mut scheduler, request, frame);
        assert_eq!(
            tokens(&actual),
            expected[index],
            "managed bounded frame {index}"
        );
        assert_eq!(
            scheduler
                .request_state(request)
                .unwrap()
                .generation()
                .schedule_state()
                .frontier(),
            index + 1
        );
    }
    let report = model.model.executor().residency_report().unwrap();
    assert!(report.initialized());
    if execution == ExecutionResidency::LayerwiseHost {
        // Checkpoint IO counters exclude reuse of the retained Host copies.
        // The residency ledger records their actual native materialization.
        let units = report
            .units()
            .iter()
            .filter(|unit| unit.policy() == ResidencyPolicy::Windowed)
            .collect::<Vec<_>>();
        assert!(
            !units.is_empty(),
            "selected Host execution retains bounded units"
        );
        assert!(units
            .iter()
            .all(|unit| unit.planned_tier() == MemoryTier::Host
                && unit.host_resident()
                && unit.host_allocated_bytes() > 0));
        let transfers = report.offload().transfer(TransferDirection::HostToDevice);
        assert!(
            transfers.count() > 0 && transfers.bytes() > 0,
            "retained Host weights were materialized for native execution"
        );
    }
    if execution == ExecutionResidency::DenseDiskStream {
        let report = model
            .model
            .executor()
            .dense_stream_report()
            .unwrap()
            .unwrap();
        assert!(report.planned_layer_count() > 0);
        assert!(
            report.decode_forwards() > 0,
            "selected disk worker executed cached frames"
        );
        // Prepared source reads bypass the ordinary checkpoint-store counter.
        // This publication is recorded only after the actual initialized read
        // receipt is consumed; exclude any load-time activity from the proof.
        let before = disk_before.expect("Disk pre-frame telemetry");
        let reads = report
            .residency()
            .offload()
            .transfer(TransferDirection::DiskToHost);
        assert!(
            reads.count() > before.0 && reads.bytes() > before.1,
            "managed frames published nonzero payload read from the retained Disk source"
        );
        let background = report.background();
        assert!(
            background.completed() > before.2,
            "joined background worker completed source reads during managed frames"
        );
        assert_eq!(background.failed(), 0);
        assert_eq!(background.queue_capacity(), 1);
        assert!(background.peak_queue_occupancy() <= background.queue_capacity());
    }
}

#[test]
#[ignore = "requires Metal; follows resident managed realtime admission"]
fn native_managed_realtime_host_layerwise_matches_ordinary_across_forced_depth_tails() {
    verify_bounded(
        WeightResidency::layerwise_host(LayerwiseLoadOptions::default()),
        ExecutionResidency::LayerwiseHost,
    );
}

#[test]
#[ignore = "requires Metal; follows resident managed realtime admission"]
fn native_managed_realtime_disk_prefetch_matches_ordinary_across_cached_frames() {
    verify_bounded(
        WeightResidency::dense_disk_stream(
            DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 1, 1).unwrap(),
        ),
        ExecutionResidency::DenseDiskStream,
    );
}

#[cfg(unix)]
#[path = "managed/tensor.rs"]
mod tensor;
