use super::tests::{required_capacity, saved_source, source_config};
use super::*;
use crate::composition::mlx::session::model_session::saved_array_copy::tests::{
    cause, reclaim, runtime, settle, words, Runtime,
};
use eredu_core::{BackendFailure, GenerationCancellationToken, TextGeneration, TokenOutput};
use eredu_runtime::working_memory::{InferenceStateRevision, WorkingMemoryPool};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::composition::mlx::session) enum Point {
    AfterPrompt,
    AfterSampling,
    AfterExchange,
    BeforeSamplingReadiness,
}

#[derive(Debug, thiserror::Error)]
#[error("injected native resume failure at {0:?}")]
struct InjectedResumeFailure(Point);

#[derive(Clone)]
enum Action {
    Error,
    Unwind,
    Cancel(GenerationCancellationToken),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TargetSnapshot {
    positions: Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>,
    revision: InferenceStateRevision,
    arrays: BTreeMap<safemlx::AllocationIdentity, u64>,
}

fn target(runtime: &Runtime) -> TargetSnapshot {
    let session = runtime.session();
    let revision = session
        .payload
        .model
        .erased()
        .retained_inference_authority()
        .unwrap()
        .revision()
        .clone();
    let arrays = session
        .payload
        .model
        .erased()
        .retained_decoder_state_storage()
        .unwrap()
        .array_allocation_facts();
    TargetSnapshot {
        positions: session.test_state_presence(),
        revision,
        arrays,
    }
}

#[derive(Default)]
struct Observation {
    fired: bool,
    idle: bool,
    healthy: bool,
    used: u64,
    target: Option<TargetSnapshot>,
}

struct Fault {
    point: Point,
    action: Action,
    observation: Rc<RefCell<Observation>>,
}
thread_local! {
    static FAULT: RefCell<Option<Fault>> = const { RefCell::new(None) };
}
struct FaultGuard(Rc<RefCell<Observation>>);
impl Drop for FaultGuard {
    fn drop(&mut self) {
        FAULT.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}
fn fault(point: Point, action: Action) -> FaultGuard {
    let observation = Rc::new(RefCell::new(Observation::default()));
    FAULT.with(|slot| {
        assert!(slot.borrow().is_none());
        slot.replace(Some(Fault {
            point,
            action,
            observation: Rc::clone(&observation),
        }));
    });
    FaultGuard(observation)
}

// The hook is inert unless this thread installed this exact fault. It observes
// metadata only and never retains native payload or a session Rc. Taking the
// fault before failure also makes destruction/unwind independent of TLS locks.
pub(in crate::composition::mlx::session) fn checkpoint(
    point: Point,
    runtime: &ModelRuntime<MlxBackend<'_>>,
) -> Result<(), Error> {
    let active = FAULT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.as_ref().is_some_and(|fault| fault.point == point) {
            slot.take()
        } else {
            None
        }
    });
    let Some(active) = active else {
        return Ok(());
    };
    let session = runtime.session();
    // Observe the lease before creating/dropping any inspection handles; their
    // ordinary retirement must not make an unresolved boundary look settled.
    let idle = session.authority.borrow().require_idle().is_ok();
    let healthy = session.ensure_healthy().is_ok();
    let used = runtime.backend().memory_pool().used_bytes().unwrap();
    let revision = session
        .payload
        .model
        .erased()
        .retained_inference_authority()
        .unwrap()
        .revision()
        .clone();
    let arrays = session
        .payload
        .model
        .erased()
        .retained_decoder_state_storage()
        .unwrap()
        .array_allocation_facts();
    *active.observation.borrow_mut() = Observation {
        fired: true,
        idle,
        healthy,
        used,
        target: Some(TargetSnapshot {
            positions: session.test_state_presence(),
            revision,
            arrays,
        }),
    };
    match active.action {
        Action::Error => Err(Error::Other(Box::new(InjectedResumeFailure(point)))),
        Action::Unwind => std::panic::panic_any(InjectedResumeFailure(point)),
        Action::Cancel(token) => {
            token.cancel();
            Ok(())
        }
    }
}

fn attempt(
    runtime: &mut Runtime,
    saved: &MlxSavedTextComponents,
    config: TextGenerationConfig,
    cancellation: &GenerationCancellationToken,
) -> Result<bool, BackendFailure> {
    let prepared = TextGeneration::resume_saved(runtime, saved, config, cancellation)?;
    let exists = prepared.is_some();
    drop(prepared);
    Ok(exists)
}

fn assert_reached(guard: &FaultGuard) {
    let seen = guard.0.borrow();
    assert!(
        seen.fired,
        "the requested real preparation boundary was reached"
    );
    assert!(seen.idle, "no native lease may cross preparation readiness");
    assert!(
        seen.healthy,
        "the guard fences only when failed preparation drops"
    );
}

fn assert_typed_error(result: Result<bool, BackendFailure>, point: Point) {
    let error = result.expect_err("injected hook must fail");
    assert_eq!(cause::<InjectedResumeFailure>(&error).unwrap().0, point);
}

#[test]
fn cancellation_after_settled_prompt_preserves_original_branch_and_retires_copies() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let (saved, expected) = saved_source(&mut runtime, false);
    let before = target(&runtime);
    let before_values = target_values(&runtime);
    let before_sampling = saved_sampling_values(&saved);
    let baseline = pool.used_bytes().unwrap();
    let (_, capacity) = required_capacity(&runtime, &saved, source_config(false, 3, u64::MAX));
    let cancellation = GenerationCancellationToken::new();
    let guard = fault(Point::AfterPrompt, Action::Cancel(cancellation.clone()));
    assert!(!attempt(
        &mut runtime,
        &saved,
        source_config(false, 3, capacity),
        &cancellation
    )
    .unwrap());
    assert_reached(&guard);
    assert!(
        guard.0.borrow().used > baseline,
        "actual funded prompt/decoder copies were constructed"
    );
    assert_eq!(guard.0.borrow().target.as_ref(), Some(&before));
    assert_eq!(target(&runtime), before);
    assert_eq!(target_values(&runtime), before_values);
    assert_eq!(saved_sampling_values(&saved), before_sampling);
    assert!(runtime.session().ensure_healthy().is_ok());
    assert!(runtime.session().authority.borrow().require_idle().is_ok());
    drop(guard);
    settle(&pool, baseline);
    // Cancellation is local to this preparation. The saved source and target
    // remain usable for a later actual resume; no fallback or unquoted path.
    let output = TextGeneration::resume_saved(
        &mut runtime,
        &saved,
        source_config(false, 3, capacity),
        &GenerationCancellationToken::new(),
    )
    .unwrap()
    .unwrap()
    .map(|token| token.unwrap().token_id().unwrap())
    .collect::<Vec<_>>();
    assert_eq!(output, expected);
    drop((saved, runtime, artifact));
    settle(&pool, 0);
}

#[test]
fn failure_or_unwind_after_sampler_before_exchange_keeps_target_healthy() {
    for action in [Action::Error, Action::Unwind] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let (saved, _) = saved_source(&mut runtime, true);
        let before = target(&runtime);
        let before_values = target_values(&runtime);
        let before_sampling = saved_sampling_values(&saved);
        let baseline = pool.used_bytes().unwrap();
        let (_, capacity) = required_capacity(&runtime, &saved, source_config(true, 3, u64::MAX));
        let unwind = matches!(action, Action::Unwind);
        let guard = fault(Point::AfterSampling, action);
        let result = catch_unwind(AssertUnwindSafe(|| {
            attempt(
                &mut runtime,
                &saved,
                source_config(true, 3, capacity),
                &GenerationCancellationToken::new(),
            )
        }));
        if unwind {
            let error = result.expect_err("injected panic remains an unwind");
            assert_eq!(
                error.downcast_ref::<InjectedResumeFailure>().unwrap().0,
                Point::AfterSampling
            );
        } else {
            assert_typed_error(result.unwrap(), Point::AfterSampling);
        }
        assert_reached(&guard);
        assert!(guard.0.borrow().used > baseline);
        assert_eq!(guard.0.borrow().target.as_ref(), Some(&before));
        assert_eq!(target(&runtime), before);
        assert_eq!(target_values(&runtime), before_values);
        assert_eq!(saved_sampling_values(&saved), before_sampling);
        assert!(runtime.session().ensure_healthy().is_ok());
        assert!(runtime.session().authority.borrow().require_idle().is_ok());
        drop(guard);
        settle(&pool, baseline);
        // Retry actual setup after dropping the failed fresh run. The source
        // still has its original history/key; no old permission is reused.
        assert!(attempt(
            &mut runtime,
            &saved,
            source_config(true, 3, capacity),
            &GenerationCancellationToken::new()
        )
        .unwrap());
        drop((saved, runtime, artifact));
        settle(&pool, 0);
    }
}

#[test]
fn post_exchange_error_and_unwind_fence_only_target_and_keep_escaped_native_charge() {
    for (point, action) in [
        (Point::AfterExchange, Action::Error),
        (Point::AfterExchange, Action::Unwind),
        (Point::BeforeSamplingReadiness, Action::Error),
    ] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let (saved, _) = saved_source(&mut runtime, false);
        let before = target(&runtime);
        let before_sampling = saved_sampling_values(&saved);
        let other_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (other, other_artifact) = super::super::super::tests::runtime(&other_pool);
        let other_before = target(&other);
        let other_bytes = other_pool.used_bytes().unwrap();
        let (_, capacity) = required_capacity(&runtime, &saved, source_config(false, 3, u64::MAX));
        let unwind = matches!(action, Action::Unwind);
        let guard = fault(point, action);
        let result = catch_unwind(AssertUnwindSafe(|| {
            attempt(
                &mut runtime,
                &saved,
                source_config(false, 3, capacity),
                &GenerationCancellationToken::new(),
            )
        }));
        if unwind {
            let error = result.expect_err("injected panic remains an unwind");
            assert_eq!(
                error.downcast_ref::<InjectedResumeFailure>().unwrap().0,
                point
            );
        } else {
            assert_typed_error(result.unwrap(), point);
        }
        assert_reached(&guard);
        let installed = target(&runtime);
        assert_ne!(installed.revision, before.revision);
        assert!(installed
            .positions
            .iter()
            .all(|(position, _)| *position == 6));
        assert_eq!(guard.0.borrow().target.as_ref(), Some(&installed));
        assert_ne!(installed.arrays, before.arrays);
        assert_eq!(saved_sampling_values(&saved), before_sampling);
        assert!(runtime.session().poison.get());
        assert!(runtime.session().ensure_healthy().is_err());
        assert!(runtime.session().authority.borrow().require_idle().is_ok());
        assert!(other.session().ensure_healthy().is_ok());
        assert_eq!(target(&other), other_before);
        assert_eq!(other_pool.used_bytes().unwrap(), other_bytes);
        // A settled final destination is still live in the fenced target. Its
        // raw alias keeps its actual published charge after all sessions and
        // saved sources retire; no inference request or recovery token retained.
        let escaped = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_decoder_state_storage()
            .unwrap()
            .into_retained_arrays()
            .unwrap()
            .next()
            .expect("nonzero installed KV backing");
        let allocation = escaped
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .unwrap();
        let bytes = allocation.bytes() as u64;
        assert!(bytes > 0);
        assert!(installed.arrays.contains_key(&allocation.identity()));
        assert!(pool.used_bytes().unwrap() >= bytes);
        drop(guard);
        drop((saved, runtime, artifact));
        settle(&pool, bytes);
        assert_eq!(
            escaped
                .try_metadata_snapshot()
                .unwrap()
                .allocation()
                .unwrap()
                .identity(),
            allocation.identity()
        );
        drop(escaped);
        settle(&pool, 0);
        drop((other, other_artifact));
        settle(&other_pool, 0);
    }
}

#[test]
fn exact_target_preparation_check_rejects_foreign_runtime_before_any_work() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (first, first_artifact) = runtime(&pool);
    let (second, second_artifact) = runtime(&pool);
    let before = (
        target(&first),
        target(&second),
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
    );
    MlxTextResumePreparation::validate_target(&first.session().poison, &first).unwrap();
    let error =
        MlxTextResumePreparation::validate_target(&first.session().poison, &second).unwrap_err();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(
        (
            target(&first),
            target(&second),
            pool.used_bytes().unwrap(),
            pool.peak_bytes().unwrap()
        ),
        before
    );
    assert!(first.session().authority.borrow().require_idle().is_ok());
    assert!(second.session().authority.borrow().require_idle().is_ok());
    drop((first, second, first_artifact, second_artifact));
    reclaim();
    settle(&pool, 0);
}

struct ChangingWorkspaceController {
    reads: Rc<Cell<usize>>,
    decisions: Rc<Cell<usize>>,
}
impl TokenFilterController for ChangingWorkspaceController {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        let reads = self.reads.get();
        self.reads.set(reads + 1);
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            // Admission obtains read zero. The next read belongs to the final
            // controller revalidation after actual prompt/sampler construction.
            additional_host_bytes: u64::from(reads != 0),
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.decisions.set(self.decisions.get() + 1);
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        self.decisions.set(self.decisions.get() + 1);
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.decisions.set(self.decisions.get() + 1);
        Ok(false)
    }
}

fn target_values(runtime: &Runtime) -> Vec<Vec<u32>> {
    runtime
        .session()
        .payload
        .model
        .erased()
        .retained_decoder_state_storage()
        .unwrap()
        .into_retained_arrays()
        .unwrap()
        .map(|array| {
            array
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        })
        .collect()
}

fn saved_sampling_values(saved: &MlxSavedTextComponents) -> (String, Vec<u32>, Vec<u32>) {
    let source = saved.funded_source().unwrap();
    (
        format!("{:?}", source.sampling.sampler.as_sampler()),
        words(source.sampling.arrays.key.as_ref().unwrap()),
        words(source.sampling.arrays.pending.as_ref().unwrap()),
    )
}

#[test]
fn final_controller_workspace_mismatch_rejects_before_exchange_with_source_unchanged() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let (saved, _) = saved_source(&mut runtime, true);
    let before = target(&runtime);
    let before_values = target_values(&runtime);
    assert!(before_values.iter().flatten().any(|bits| *bits != 0));
    let before_sampling = saved_sampling_values(&saved);
    let baseline = pool.used_bytes().unwrap();
    let (_, capacity) = required_capacity(&runtime, &saved, source_config(true, 3, u64::MAX));
    let reads = Rc::new(Cell::new(0));
    let decisions = Rc::new(Cell::new(0));
    let controller = ChangingWorkspaceController {
        reads: Rc::clone(&reads),
        decisions: Rc::clone(&decisions),
    };
    let guard = fault(Point::AfterExchange, Action::Error);
    let error = eredu_core::ControlledTextGeneration::resume_saved(
        &mut runtime,
        &saved,
        source_config(true, 3, capacity),
        controller,
        &GenerationCancellationToken::new(),
    )
    .err()
    .expect("changed final controller must reject");
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(reads.get(), 2);
    assert_eq!(decisions.get(), 0);
    assert!(!guard.0.borrow().fired, "exchange was never reached");
    assert_eq!(target(&runtime), before);
    assert_eq!(target_values(&runtime), before_values);
    assert_eq!(saved_sampling_values(&saved), before_sampling);
    assert!(runtime.session().ensure_healthy().is_ok());
    assert!(runtime.session().authority.borrow().require_idle().is_ok());
    drop(guard);
    settle(&pool, baseline);
    drop((saved, runtime, artifact));
    settle(&pool, 0);
}
