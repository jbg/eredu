//! Actual saved-source continuation with existing host/direct-disk weight policies.
use super::tests::{required_capacity, saved_source, source_config};
use super::*;
use crate::composition::mlx::session::model_session::{
    disk_layerwise_tests as disk, host_layerwise_tests as host,
    saved_array_copy::tests::{accounting, cause, reclaim, settle, words, Runtime},
};
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    residency::{MemoryTier, ResidencyPolicy},
    ControlledTextGeneration, GenerationCancellationToken, TextControllerWorkspace, TextGeneration,
    TextGenerationDriver, TokenFilter, TokenOutput,
};
use eredu_runtime::working_memory::WorkingMemoryPool;

#[derive(Clone, Copy, Debug)]
enum Route {
    Host(usize),
    Disk,
}
impl Route {
    fn load(self, pool: &WorkingMemoryPool) -> (Runtime, tempfile::TempDir) {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        match self {
            Self::Host(depth) => host::runtime(&stream, pool, Some(depth)),
            Self::Disk => disk::load_runtime(&stream, pool, true),
        }
    }
    fn disk(self) -> bool {
        matches!(self, Self::Disk)
    }
    fn assert_window(self, runtime: &Runtime) {
        if self.disk() {
            disk::assert_disk_window(runtime);
            return;
        }
        let Self::Host(depth) = self else {
            unreachable!()
        };
        let report = disk::report(runtime);
        let units = report
            .units()
            .iter()
            .filter(|unit| unit.policy() == ResidencyPolicy::Windowed)
            .collect::<Vec<_>>();
        assert_eq!(units.len(), 3);
        assert!(units
            .iter()
            .all(|unit| unit.planned_tier() == MemoryTier::Host && unit.host_resident()));
        let live = units.iter().filter(|unit| unit.device_resident()).count();
        assert!(live > 0 && live <= depth);
        let pinned = report
            .units()
            .iter()
            .filter(|unit| unit.policy() == ResidencyPolicy::Pinned)
            .count();
        assert!(
            report
                .offload()
                .peak_resident_units()
                .get(MemoryTier::Device)
                <= pinned + depth
        );
        assert!(report.offload().tier_evictions(MemoryTier::Device).count() > 0);
    }
}
fn quiescent(runtime: &Runtime) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        runtime.session().payload.active_owner_count() == 1
    });
    runtime.session().ensure_no_submission_in_flight().unwrap();
}
fn workspace() -> TextControllerWorkspace<'static> {
    TextControllerWorkspace {
        filter: (&TokenFilter::All).into(),
        additional_host_bytes: 0,
    }
}
#[derive(Debug, PartialEq)]
struct Frozen {
    sampler: String,
    history_capacity: usize,
    frontier: u64,
    ordinal: u64,
    key: Vec<u32>,
    pending: Vec<u32>,
    decoder: Vec<Vec<f32>>,
}
fn frozen(saved: &MlxSavedTextComponents) -> Frozen {
    let pair = saved.funded_source().unwrap();
    let mut decoder = Vec::new();
    pair.decoder
        .native
        .prepare_copy()
        .unwrap()
        .visit_operands(&mut |array| {
            decoder.push(array.evaluated().unwrap().try_to_vec::<f32>().unwrap());
        });
    Frozen {
        sampler: format!("{:?}", pair.sampling.sampler.as_sampler()),
        history_capacity: pair.sampling.sampler.as_sampler().history_capacity(),
        frontier: pair.sampling.frontier(),
        ordinal: pair.sampling.next_prediction,
        key: words(pair.sampling.arrays.key.as_ref().unwrap()),
        pending: words(pair.sampling.arrays.pending.as_ref().unwrap()),
        decoder,
    }
}
thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct ColdGuard;
impl ColdGuard {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.set(0);
        Self
    }
    fn assert_cold(&self) {
        assert_eq!(HOUSEKEEPING.get(), 0);
    }
}
impl Drop for ColdGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn layerwise_saved_resume_preserves_actual_stochastic_future_in_both_drivers() {
    for route in [Route::Host(1), Route::Host(2), Route::Disk] {
        for adaptive in [false, true] {
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let (mut runtime, artifact) = route.load(&pool);
            let (saved, expected) = saved_source(&mut runtime, adaptive);
            route.assert_window(&runtime);
            let original = frozen(&saved);
            assert!(original.decoder.iter().flatten().any(|value| *value != 0.0));
            let (required, exact) =
                required_capacity(&runtime, &saved, source_config(adaptive, 3, u64::MAX));
            // Both successive runs use one finite ceiling. Their exact reduced
            // threshold is checked independently below; prior physical charges
            // are never refunded merely because a successor is admitted.
            let capacity = exact.checked_add(required.checked_mul(3).unwrap()).unwrap();
            let config = source_config(adaptive, 3, capacity);
            let cancellation = GenerationCancellationToken::new();
            let ordinary =
                TextGeneration::resume_saved(&mut runtime, &saved, config, &cancellation)
                    .unwrap()
                    .unwrap()
                    .map(|token| token.unwrap().token_id().unwrap())
                    .collect::<Vec<_>>();
            assert_eq!(ordinary, expected, "{route:?}, adaptive={adaptive}");
            quiescent(&runtime);
            route.assert_window(&runtime);
            let controller = disk::Controller::default();
            let controlled = {
                let mut run = ControlledTextGeneration::resume_saved(
                    &mut runtime,
                    &saved,
                    config,
                    controller.clone(),
                    &cancellation,
                )
                .unwrap()
                .unwrap();
                let mut output = Vec::new();
                while let Some(token) = run.next_cancellable(&cancellation) {
                    output.push(token.unwrap().token_id());
                }
                output
            };
            assert_eq!(controlled, expected, "{route:?}, adaptive={adaptive}");
            assert_eq!(controller.0.get(), (3, 3));
            quiescent(&runtime);
            route.assert_window(&runtime);
            assert_eq!(frozen(&saved), original);
            assert_eq!(pool.effective_capacity().unwrap(), capacity);
            drop((saved, runtime, artifact));
            settle(&pool, 0);
        }
    }
}

#[test]
fn layerwise_saved_resume_exact_budget_is_cold_and_retains_the_matching_operation_route() {
    for route in [Route::Host(1), Route::Disk] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = route.load(&pool);
        let (saved, expected) = saved_source(&mut runtime, true);
        let original = frozen(&saved);
        let revision = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap()
            .revision()
            .clone();
        let before = (
            accounting(&pool),
            paths::snapshot(),
            disk::report(&runtime),
            paths::session_input_creation_attempts(),
        );
        let (required, capacity) = {
            let cold = ColdGuard::new();
            let quote = PreparedSavedTextResumeQuote::prepare(
                &runtime,
                saved.funded_source().unwrap(),
                source_config(true, 3, u64::MAX),
                workspace(),
            )
            .unwrap();
            let layerwise = quote
                .layerwise_workspace()
                .expect("exact selected layerwise proof");
            assert!(layerwise.materialization().bytes().unwrap() > 0);
            assert_eq!(layerwise.disk_receipt().is_some(), route.disk());
            assert_eq!(
                quote
                    .full()
                    .execution_workspace
                    .as_ref()
                    .unwrap()
                    .materialization
                    .bytes(),
                layerwise.materialization().bytes()
            );
            cold.assert_cold();
            drop(quote);
            let result = required_capacity(&runtime, &saved, source_config(true, 3, u64::MAX));
            cold.assert_cold();
            result
        };
        assert_eq!(
            (
                accounting(&pool),
                paths::snapshot(),
                disk::report(&runtime),
                paths::session_input_creation_attempts()
            ),
            before
        );
        let controller = disk::Controller::default();
        let cancellation = GenerationCancellationToken::new();
        {
            let cold = ColdGuard::new();
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let error = driver
                .resume_saved(
                    &saved,
                    source_config(true, 3, capacity - 1),
                    controller.clone(),
                    &cancellation,
                )
                .err()
                .expect("one byte short");
            assert!(
                matches!(cause::<WorkingMemoryError>(&error),Some(WorkingMemoryError::BudgetExceeded { required_bytes,available_bytes }) if *required_bytes==required && *available_bytes==required-1),
                "{error:?}"
            );
            drop(driver);
            cold.assert_cold();
        }
        assert_eq!(controller.0.get(), (0, 0));
        assert_eq!(
            (
                accounting(&pool),
                paths::snapshot(),
                disk::report(&runtime),
                paths::session_input_creation_attempts()
            ),
            before
        );
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .retained_inference_authority()
                .unwrap()
                .revision(),
            &revision
        );
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .resume_saved(
                &saved,
                source_config(true, 3, capacity),
                controller.clone(),
                &cancellation,
            )
            .unwrap()
            .unwrap();
        assert_eq!(controller.0.get(), (0, 0));
        assert_eq!(pool.peak_bytes().unwrap(), before.0 .1.max(capacity));
        {
            let mut boundary = driver.quiescent(&mut state).unwrap();
            let (_, sampling, _) = boundary.parts();
            let quote = sampling.sampling.quote.as_ref().unwrap();
            let route_guard = quote.activate_disk_route().unwrap();
            assert_eq!(route_guard.is_some(), route.disk());
            drop(route_guard);
        }
        let mut retained_outputs = Vec::new();
        for ordinal in 0..3 {
            let output = driver.advance(&mut state).unwrap().unwrap().into_output();
            driver.take_completed_delivery(&mut state).unwrap();
            assert_eq!(output.step_receipt().unwrap().attempt(), ordinal);
            assert_eq!(output.token_id().unwrap(), expected[ordinal as usize]);
            // A completed output remains alive while the next scope reuses the
            // exact receipt. Its metadata does not keep a disk route active.
            {
                let mut boundary = driver.quiescent(&mut state).unwrap();
                let (_, sampling, _) = boundary.parts();
                let guard = sampling
                    .sampling
                    .quote
                    .as_ref()
                    .unwrap()
                    .activate_disk_route()
                    .unwrap();
                assert_eq!(guard.is_some(), route.disk());
                drop(guard);
            }
            retained_outputs.push(output);
        }
        assert!(driver.advance(&mut state).unwrap().is_none());
        assert_eq!(controller.0.get(), (3, 3));
        drop((state, driver));
        quiescent(&runtime);
        route.assert_window(&runtime);
        assert_eq!(frozen(&saved), original);
        drop((retained_outputs, saved, runtime, artifact));
        settle(&pool, 0);
    }
}

#[test]
fn layerwise_resume_cancellation_preserves_source_and_releases_windows_before_raw_alias() {
    for route in [Route::Host(1), Route::Disk] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = route.load(&pool);
        let (saved, expected) = saved_source(&mut runtime, false);
        let original = frozen(&saved);
        let (_, capacity) = required_capacity(&runtime, &saved, source_config(false, 3, u64::MAX));
        let controller = disk::Controller::default();
        let cancelled = GenerationCancellationToken::new();
        cancelled.cancel();
        let before = (accounting(&pool), paths::snapshot(), disk::report(&runtime));
        assert!(ControlledTextGeneration::resume_saved(
            &mut runtime,
            &saved,
            source_config(false, 3, capacity),
            controller.clone(),
            &cancelled
        )
        .unwrap()
        .is_none());
        assert_eq!(controller.0.get(), (0, 0));
        assert_eq!(
            (accounting(&pool), paths::snapshot(), disk::report(&runtime)),
            before
        );
        let cancellation = GenerationCancellationToken::new();
        let first = {
            let mut run = ControlledTextGeneration::resume_saved(
                &mut runtime,
                &saved,
                source_config(false, 3, capacity),
                controller.clone(),
                &cancellation,
            )
            .unwrap()
            .unwrap();
            let first = run
                .next_cancellable(&cancellation)
                .unwrap()
                .unwrap()
                .into_output();
            assert_eq!(first.token_id().unwrap(), expected[0]);
            cancellation.cancel();
            assert!(run.next_cancellable(&cancellation).is_none());
            first
        };
        quiescent(&runtime);
        route.assert_window(&runtime);
        assert_eq!(controller.0.get(), (1, 1));
        assert_eq!(frozen(&saved), original);
        let escaped = first.value.clone();
        let bytes = escaped
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .unwrap()
            .bytes() as u64;
        let expected = first.token_id().unwrap();
        drop((first, saved, runtime, artifact));
        settle(&pool, bytes);
        assert_eq!(words(&escaped), vec![expected]);
        drop(escaped);
        settle(&pool, 0);
    }
}
