use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    TextGenerationContinuation, TextGenerationDriver, TextGenerationInput, TextPreparationInput,
    TokenFilterController,
};
use eredu_runtime::working_memory::{WorkingMemoryPool, WorkspaceCopyAdmissionError};

thread_local! {
    static COPIES: Cell<usize> = const { Cell::new(0) };
    static FAIL_AFTER_KEY: Cell<bool> = const { Cell::new(false) };
}

#[derive(Debug, thiserror::Error)]
#[error("injected failure after the real saved key copy")]
struct InjectedCopyFailure;

pub(super) fn after_first_copy() -> Result<(), Error> {
    COPIES.with(|copies| copies.set(copies.get() + 1));
    if FAIL_AFTER_KEY.with(|fail| fail.replace(false)) {
        return Err(Error::Other(Box::new(InjectedCopyFailure)));
    }
    Ok(())
}

fn copies() -> usize {
    COPIES.with(Cell::get)
}

pub(super) struct AllTokens;
impl TokenFilterController for AllTokens {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

pub(super) type Runtime = ModelRuntime<MlxBackend<'static>>;
pub(super) type State = TextGenerationContinuation<MlxBackend<'static>, AllTokens>;

pub(super) fn config(capacity: Option<u64>) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(3),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
                repetition_penalty: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(19)
    .with_inference_policy(eredu_core::TextInferencePolicy {
        prefill_chunk_positions: std::num::NonZeroU64::new(1),
        managed_memory_capacity_bytes: capacity,
        submission_tracking_capacity_bytes: None,
        graph_metadata_capacity_bytes: None,
    })
}

pub(super) fn tokens() -> Vec<u32> {
    vec![1, 2, 3, 4, 5]
}

pub(super) fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

pub(super) fn settle(pool: &WorkingMemoryPool, expected: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.used_bytes().unwrap() == expected && pool.unquoted_owner_count().unwrap() == 0
    });
}

pub(super) fn runtime(pool: &WorkingMemoryPool) -> (Runtime, tempfile::TempDir) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let source = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &source).with_memory_pool(pool.clone());
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model = eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
        .unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    assert!(pool.used_bytes().unwrap() > 0);
    (runtime, artifact)
}

pub(super) fn start(
    driver: &mut TextGenerationDriver<'_, MlxBackend<'static>>,
    capacity: u64,
) -> State {
    driver
        .start_input(
            TextGenerationInput::TokenIds(tokens()),
            config(Some(capacity)),
            AllTokens,
        )
        .unwrap()
}

pub(super) fn advance(
    driver: &mut TextGenerationDriver<'_, MlxBackend<'static>>,
    state: &mut State,
) -> MlxTextToken {
    let token = driver.advance(state).unwrap().unwrap().into_output();
    driver.take_completed_delivery(state).unwrap();
    token
}

pub(super) fn words(array: &Array) -> Vec<u32> {
    let dtype = array.dtype();
    let array = array.evaluated().unwrap();
    match dtype {
        Dtype::Uint32 => array.try_to_vec::<u32>().unwrap(),
        Dtype::Int32 => array
            .try_to_vec::<i32>()
            .unwrap()
            .into_iter()
            .map(|n| n as u32)
            .collect(),
        dtype => panic!("unexpected saved state dtype {dtype:?}"),
    }
}

pub(super) fn source_state(
    sampling: &super::super::super::generation::MlxTextSamplingState,
) -> (Vec<u32>, Vec<u32>, u64) {
    let history = match sampling.sampler.as_sampler() {
        MlxTextSampler::Standard(sampler) => sampler.generated_tokens(),
        MlxTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    };
    (
        words(sampling.prng.as_ref().unwrap().as_array()),
        history.to_vec(),
        sampling.next_prediction,
    )
}

fn physical_bytes<'a>(arrays: impl IntoIterator<Item = &'a Array>) -> u64 {
    let mut storage = RetainedStorage::default();
    for array in arrays {
        storage.include_array(array).unwrap();
    }
    storage
        .byte_bound()
        .unwrap()
        .expect("completed backing must be known")
}

pub(super) fn identity(array: &Array) -> safemlx::AllocationIdentity {
    array.allocation_info().unwrap().unwrap().identity()
}

pub(super) fn accounting(pool: &WorkingMemoryPool) -> (u64, u64, u64) {
    (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        pool.effective_capacity().unwrap(),
    )
}

pub(super) fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

#[test]
fn exact_saved_arrays_copy_rejects_short_limits_and_preserves_independent_values() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start(&mut driver, u64::MAX);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let (saved, original) = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let sampling = &generation.sampling;
        let original = source_state(sampling);
        let before = (accounting(&pool), paths::snapshot(), copies());
        let plan = PreparedTextArrayCopy::prepare(runtime, sampling, outputs.last()).unwrap();
        let required = plan.required_bytes();
        assert!(required > 0);
        let error = plan
            .copy(
                runtime,
                WorkspaceCopyLimits {
                    application_memory_budget_bytes: Some(required - 1),
                    ..WorkspaceCopyLimits::new(u64::MAX)
                },
            )
            .err()
            .unwrap();
        assert!(matches!(cause::<WorkspaceCopyAdmissionError>(&error),
            Some(WorkspaceCopyAdmissionError::ApplicationBudgetExceeded { required_bytes, budget_bytes })
                if *required_bytes == required && *budget_bytes == required - 1));
        let error = PreparedTextArrayCopy::prepare(runtime, sampling, outputs.last())
            .unwrap()
            .copy(
                runtime,
                WorkspaceCopyLimits::new(before.0 .0 + required - 1),
            )
            .err()
            .unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::BudgetExceeded {
                required_bytes: required,
                available_bytes: required - 1,
            })
        );
        assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
        assert_eq!(source_state(sampling), original);
        // Repeated cold preparation does not accumulate the introduced work
        // control term. A caller reserve is additional to that exact diagnostic.
        const USER_RESERVE: u64 = 13;
        let with_reserve = required.checked_add(USER_RESERVE).unwrap();
        let prepared = PreparedTextArrayCopy::prepare(runtime, sampling, outputs.last()).unwrap();
        assert_eq!(prepared.required_bytes(), required);
        let error = prepared
            .copy(
                runtime,
                WorkspaceCopyLimits {
                    capacity_bytes: before.0 .0 + with_reserve,
                    application_memory_budget_bytes: Some(with_reserve - 1),
                    safety_reserve_bytes: USER_RESERVE,
                },
            )
            .err()
            .unwrap();
        assert!(matches!(cause::<WorkspaceCopyAdmissionError>(&error),
            Some(WorkspaceCopyAdmissionError::ApplicationBudgetExceeded { required_bytes, budget_bytes })
                if *required_bytes == with_reserve && *budget_bytes == with_reserve - 1));
        assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
        assert_eq!(source_state(sampling), original);
        let prepared = PreparedTextArrayCopy::prepare(runtime, sampling, outputs.last()).unwrap();
        assert_eq!(prepared.required_bytes(), required);
        let saved = prepared
            .copy(
                runtime,
                WorkspaceCopyLimits {
                    capacity_bytes: before.0 .0 + with_reserve,
                    application_memory_budget_bytes: Some(with_reserve),
                    safety_reserve_bytes: USER_RESERVE,
                },
            )
            .unwrap();
        assert_eq!(copies(), before.2 + 1);
        assert_eq!(pool.used_bytes().unwrap(), before.0 .0 + with_reserve);
        assert_eq!(saved.custody.bytes(), with_reserve);
        assert!(saved.custody.pool().same_domain(&pool));
        let key = saved.key.as_ref().unwrap();
        let pending = saved.pending.as_ref().unwrap();
        assert_eq!(words(key), original.0);
        assert_eq!(words(pending), words(&outputs[1].value));
        assert_ne!(
            identity(key),
            identity(sampling.prng.as_ref().unwrap().as_array())
        );
        assert_ne!(identity(pending), identity(&outputs[1].value));
        assert_ne!(identity(key), identity(pending));
        assert_eq!(source_state(sampling), original);
        (saved, original)
    };
    let third = advance(&mut driver, &mut state);
    {
        let boundary = driver.quiescent(&mut state).unwrap();
        let actual = source_state(&boundary.parts().1.sampling);
        assert_eq!(actual.2, 3);
        assert_ne!(actual.0, original.0);
    }
    assert_eq!(words(saved.key.as_ref().unwrap()), original.0);
    assert_eq!(
        words(saved.pending.as_ref().unwrap()),
        words(&outputs[1].value)
    );
    drop((saved, third, outputs, state));
    drop(driver);
    drop((runtime, artifact));
    settle(&pool, 0);
}

#[test]
fn saved_array_copy_cannot_relax_the_original_source_ceiling() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let baseline = pool.used_bytes().unwrap();
    let ids = tokens();
    let evidence = TextPreparationInput::TokenIds {
        positions: ids.len() as u64,
        capacity_bytes: (ids.capacity() * std::mem::size_of::<u32>()) as u64,
    };
    let prepared =
        MlxBackend::admit_text_preparation(&runtime, &evidence, config(Some(u64::MAX)), &AllTokens)
            .unwrap();
    let ceiling = pool.used_bytes().unwrap();
    drop(prepared);
    settle(&pool, baseline);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start(&mut driver, ceiling);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let before = (
            accounting(&pool),
            copies(),
            source_state(&generation.sampling),
        );
        assert_eq!(before.0 .2, ceiling);
        let plan =
            PreparedTextArrayCopy::prepare(runtime, &generation.sampling, outputs.last()).unwrap();
        let required = plan.required_bytes();
        let available = ceiling - before.0 .0;
        assert!(required > available);
        let error = plan
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .err()
            .unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::BudgetExceeded {
                required_bytes: required,
                available_bytes: available,
            })
        );
        assert_eq!(
            (
                accounting(&pool),
                copies(),
                source_state(&generation.sampling)
            ),
            before
        );
    }
    drop((outputs, state));
    drop(driver);
    drop((runtime, artifact));
    settle(&pool, 0);
}

#[test]
fn saved_arrays_and_raw_aliases_outlive_the_model_with_exact_physical_charges() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start(&mut driver, u64::MAX);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let saved = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        PreparedTextArrayCopy::prepare(runtime, &generation.sampling, outputs.last())
            .unwrap()
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap()
    };
    let charge = saved.custody.bytes();
    let key = saved.key.as_ref().unwrap().clone();
    let key_alias = key.clone();
    let pending = saved.pending.as_ref().unwrap().clone();
    let values = (words(&key), words(&pending));
    let exact = physical_bytes([&key, &key_alias, &pending]);
    let pending_bytes = physical_bytes([&pending]);
    assert!(exact > pending_bytes && pending_bytes > 0);
    drop((outputs, state));
    drop(driver);
    drop((runtime, artifact));
    settle(&pool, charge);
    assert_eq!((words(&key), words(&pending)), values);
    drop(saved);
    settle(&pool, exact);
    drop(key);
    settle(&pool, exact);
    drop(key_alias);
    settle(&pool, pending_bytes);
    assert_eq!(words(&pending), values.1);
    drop(pending);
    settle(&pool, 0);
}

#[test]
fn stale_pending_foreign_runtime_and_unquoted_sampling_reject_before_copy() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let foreign_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let (mut foreign, foreign_artifact) = self::runtime(&foreign_pool);
    let unquoted = MlxBackend::start_text_generation(runtime.backend(), config(None)).unwrap();
    let before = (accounting(&pool), copies());
    let error = PreparedTextArrayCopy::prepare(&runtime, &unquoted.sampling, None)
        .err()
        .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert_eq!((accounting(&pool), copies()), before);
    drop(unquoted);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start(&mut driver, u64::MAX);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let before = (
            accounting(&pool),
            accounting(&foreign_pool),
            copies(),
            source_state(&generation.sampling),
        );
        let error = PreparedTextArrayCopy::prepare(runtime, &generation.sampling, outputs.first())
            .err()
            .unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        let error = PreparedTextArrayCopy::prepare(&foreign, &generation.sampling, outputs.last())
            .err()
            .unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        let error = PreparedTextArrayCopy::prepare(runtime, &generation.sampling, outputs.last())
            .unwrap()
            .copy(&mut foreign, WorkspaceCopyLimits::new(u64::MAX))
            .err()
            .unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(
            (
                accounting(&pool),
                accounting(&foreign_pool),
                copies(),
                source_state(&generation.sampling)
            ),
            before
        );
    }
    drop(advance(&mut driver, &mut state));
    drop((outputs, state));
    drop(driver);
    drop((runtime, artifact, foreign, foreign_artifact));
    settle(&pool, 0);
    settle(&foreign_pool, 0);
}

#[test]
fn settled_key_copy_failure_preserves_source_and_quarantines_destination_without_snapshot_permission(
) {
    use eredu_runtime::execution_control::TextSnapshotBackend;

    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start(&mut driver, u64::MAX);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let (quarantined, pinned_sources) = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let before = (
            accounting(&pool),
            copies(),
            source_state(&generation.sampling),
        );
        let error = MlxBackend::copy_sampling_state(runtime, &generation.sampling)
            .err()
            .unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::UnknownBound)
        );
        assert_eq!(
            (
                accounting(&pool),
                copies(),
                source_state(&generation.sampling)
            ),
            before
        );
        let pinned = physical_bytes([
            generation.sampling.prng.as_ref().unwrap().as_array(),
            &outputs[1].value,
        ]);
        let plan =
            PreparedTextArrayCopy::prepare(runtime, &generation.sampling, outputs.last()).unwrap();
        let required = plan.required_bytes();
        FAIL_AFTER_KEY.with(|flag| assert!(!flag.replace(true)));
        let error = plan
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .err()
            .unwrap();
        assert!(cause::<InjectedCopyFailure>(&error).is_some());
        assert_eq!(copies(), before.1 + 1);
        assert_eq!(source_state(&generation.sampling), before.2);
        runtime.session().ensure_no_submission_in_flight().unwrap();
        settle(&pool, before.0 .0 + required);
        (required, pinned)
    };
    drop(advance(&mut driver, &mut state));
    drop((outputs, state));
    drop(driver);
    drop((runtime, artifact));
    // The failed scope keeps its exact borrowed-source accounting pin as well
    // as its destination envelope. It retains no model weights or native roots.
    settle(&pool, quarantined + pinned_sources);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[derive(Debug, thiserror::Error)]
#[error("saved copy worker identity")]
struct WorkerIdentityError(std::sync::Arc<()>);

struct FakeProbe {
    terminal: Rc<Cell<Status>>,
    first: Option<Status>,
    calls: Rc<Cell<usize>>,
}

impl Probe for FakeProbe {
    fn seal(&mut self) {}

    fn progress(&self) -> Status {
        let index = self.calls.get();
        self.calls.set(index + 1);
        if index == 0 {
            if let Some(first) = self.first {
                return first;
            }
        }
        self.terminal.get()
    }
}

#[test]
fn saved_copy_preserves_worker_cause_across_early_and_late_recovery_failures() {
    let settled = Status {
        settled: true,
        failed: false,
        blocked: false,
    };
    let pending = Status {
        settled: false,
        failed: false,
        blocked: false,
    };
    let failed = Status {
        settled: true,
        failed: true,
        blocked: false,
    };
    let blocked = Status {
        settled: false,
        failed: false,
        blocked: true,
    };
    for (first, terminal_status, poisoned) in [
        (None, settled, false),
        (None, failed, true),
        (Some(pending), failed, true),
        (None, blocked, true),
        (Some(pending), blocked, true),
    ] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let session = runtime.session_mut();
        let poison = Rc::clone(&session.poison);
        let terminal = Rc::new(Cell::new(terminal_status));
        let calls = Rc::new(Cell::new(0));
        let lease = session.authority.borrow_mut().begin_submission().unwrap();
        let owner = SubmissionResources::with_purpose(
            lease,
            Rc::clone(&poison),
            SubmissionPurpose::SavedArrayCopy(Rc::new(RefCell::new(Vec::new()))),
        );
        let recovery = Recovery::with_probe(
            owner.ticket(),
            FakeProbe {
                terminal: Rc::clone(&terminal),
                first,
                calls: Rc::clone(&calls),
            },
        );
        let operation = SessionOperation {
            session,
            owner: owner.clone(),
            recovery: Some(recovery),
            handed_off: false,
            token_validations: Default::default(),
        };
        let identity = std::sync::Arc::new(());
        let error = finish_saved_copy::<(), _>(
            operation,
            Err(Error::Other(Box::new(WorkerIdentityError(
                std::sync::Arc::clone(&identity),
            )))),
        )
        .unwrap_err();
        let retained =
            cause::<WorkerIdentityError>(&error).expect("original worker cause survives");
        assert!(std::sync::Arc::ptr_eq(&retained.0, &identity));
        assert_eq!(poison.get(), poisoned);
        if first.is_some() {
            assert!(
                calls.get() >= 2,
                "late failure must cross the initial progress check"
            );
        }
        // Failed/blocked probes keep their session authority until terminal
        // evidence arrives; the worker error never certifies completion.
        terminal.set(settled);
        drop(owner);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            runtime.session().authority.borrow().require_idle().is_ok()
        });
        assert_eq!(poison.get(), poisoned);
        if !poisoned {
            runtime.session().ensure_no_submission_in_flight().unwrap();
        }
        drop((runtime, artifact));
        settle(&pool, 0);
    }
}

mod sampling;

mod coordinates;
