use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, PendingTextInput, StateMemoryLayout,
    TextGenerationDriver, TextGenerationInput, WorkspaceBound,
};
use eredu_runtime::working_memory::{
    InferenceRequest, InferenceRetention, MemoryLedger, WorkingMemoryError,
};

#[derive(Clone, Default)]
struct Controller(Rc<Cell<(usize, usize, usize)>>);

impl eredu_core::TokenFilterController for Controller {
    type Error = std::convert::Infallible;

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let (decisions, commits, complete) = self.0.get();
        self.0.set((decisions + 1, commits, complete));
        Ok(TokenFilter::All)
    }

    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        let (decisions, commits, complete) = self.0.get();
        self.0.set((decisions, commits + 1, complete));
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        let (decisions, commits, complete) = self.0.get();
        self.0.set((decisions, commits, complete + 1));
        Ok(false)
    }
}

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn pool() -> MemoryLedger {
    crate::memory_fixture::ledger(u64::MAX, 0).unwrap()
}

fn runtime(
    stream: &Stream,
    model_pool: &MemoryLedger,
    context_pool: &MemoryLedger,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let loader = MlxBackend::new(stream, stream).with_memory_ledger(model_pool.clone());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&loader, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let backend = MlxBackend::new(stream, stream).with_memory_ledger(context_pool.clone());
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        model_pool.unquoted_owner_count().unwrap() == 0
    });
    (runtime, root)
}

fn config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 2,
        prefill_chunk_positions: 3,
        output: OutputDemand::LastPosition,
    }
}

// Zero-byte synthetic charges are only rejection fixtures. No native inference
// is authorized by this metadata-only stateless quote; every test stops before
// native input construction, model execution, and token sampling.
fn reserved_request(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    pool: &MemoryLedger,
) -> InferenceRequest {
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![eredu_core::cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "text-step rejection fixture only");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(3),
        2,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry: geometry(),
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
        },
    ))
    .unwrap();
    pool.reserve(
        runtime
            .session()
            .payload
            .model
            .erased()
            .inference_execution_identity(),
        &crate::memory_fixture::admission(Admission {
            additional_headroom: Default::default(),
            memory_limits: Default::default(),
            requested_positions: 5,
            state,
            incremental_required_bytes: Some(0),
        }),
    )
    .unwrap()
    .into()
}

fn cause<'a, T: std::error::Error + 'static>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    let mut error = error;
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

struct Unchanged {
    paths: paths::Counts,
    inputs: usize,
    resets: usize,
    frontier: Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>,
    pools: Vec<(MemoryLedger, u64, u64, usize)>,
}

impl Unchanged {
    fn capture(runtime: &ModelRuntime<MlxBackend<'_>>, pools: &[&MemoryLedger]) -> Self {
        Self {
            paths: paths::snapshot(),
            inputs: paths::session_input_creation_attempts(),
            resets: paths::session_reset_attempts(),
            frontier: runtime.session().payload.model.erased().state_snapshot(),
            pools: pools
                .iter()
                .map(|pool| {
                    (
                        (*pool).clone(),
                        pool.fixture_host_charge().unwrap(),
                        pool.fixture_host_peak().unwrap(),
                        pool.unquoted_owner_count().unwrap(),
                    )
                })
                .collect(),
        }
    }

    fn assert(&self, runtime: &ModelRuntime<MlxBackend<'_>>, controller: &Controller) {
        assert_eq!(controller.0.get(), (0, 0, 0));
        assert_eq!(paths::snapshot(), self.paths);
        assert_eq!(paths::session_input_creation_attempts(), self.inputs);
        assert_eq!(paths::session_reset_attempts(), self.resets);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            self.frontier
        );
        assert!(runtime.session().ensure_no_submission_in_flight().is_ok());
        for (pool, bytes, peak, owners) in &self.pools {
            assert_eq!(pool.fixture_host_charge().unwrap(), *bytes);
            assert_eq!(pool.fixture_host_peak().unwrap(), *peak);
            assert_eq!(pool.unquoted_owner_count().unwrap(), *owners);
        }
    }
}

#[test]
fn stale_sampler_epoch_rejects_at_core_step_entry_before_controller_or_model_work() {
    let stream = stream();
    let pool = pool();
    let (mut runtime, _root) = runtime(&stream, &pool, &pool);
    let controller = Controller::default();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut continuation = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2, 3]),
            config(),
            controller.clone(),
        )
        .unwrap();
    {
        let mut boundary = driver.quiescent(&mut continuation).unwrap();
        let (runtime, state, _) = boundary.mechanism_parts();
        let mut epoch = None;
        runtime
            .session()
            .validate_parameter_epoch(&mut epoch)
            .unwrap();
        state.sampling.parameter_epoch = Some(epoch.unwrap().checked_add(1).unwrap());
    }
    let before = Unchanged::capture(driver.runtime(), &[&pool]);
    let error = driver.advance(&mut continuation).err().unwrap();
    // The native transparent wrapper delegates source() past BackendError;
    // inspect the returned typed variant to preserve the original rejection.
    assert!(
        matches!(
            &error,
            eredu_core::TextContinuationError::Generation(
                eredu_core::ControlledTextGenerationError::Backend(Error::Backend(
                    eredu_core::BackendError::Execution { session, operation, .. }
                ))
            ) if session == "text-generation" && operation == "validate parameter version"
        ),
        "unexpected epoch failure: {error:?}"
    );
    before.assert(driver.runtime(), &controller);
    assert!(
        driver.advance(&mut continuation).is_err(),
        "failed continuation stays fenced"
    );
}

#[test]
fn replaced_prompt_request_with_equal_geometry_rejects_before_controller_or_model_work() {
    let stream = stream();
    let pool = pool();
    let (mut runtime, _root) = runtime(&stream, &pool, &pool);
    let execution = runtime
        .session()
        .payload
        .model
        .erased()
        .inference_execution_identity();
    let canonical = reserved_request(&runtime, &pool);
    let replacement = reserved_request(&runtime, &pool);
    assert_eq!(canonical.geometry(), replacement.geometry());
    assert!(canonical.validate_same_request(&replacement).is_err());
    let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2, 3])
        .unwrap()
        .with_inference_request(canonical);
    let controller = Controller::default();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut continuation = driver.start(prompt, config(), controller.clone()).unwrap();
    {
        let mut boundary = driver.quiescent(&mut continuation).unwrap();
        let replacement = match boundary.parts().2.unwrap() {
            PendingTextInput::Prefill(prompt) => prompt.clone().with_inference_request(replacement),
            PendingTextInput::Decode(_) => panic!("unexecuted prompt"),
        };
        boundary.install_host_state(
            controller.clone(),
            Some(PendingTextInput::Prefill(replacement)),
            Some(2),
        );
    }
    let before = Unchanged::capture(driver.runtime(), &[&pool]);
    let error = driver.advance(&mut continuation).err().unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    before.assert(driver.runtime(), &controller);
}

#[test]
fn sampler_cannot_substitute_an_equally_sized_reservation_for_canonical_preparation() {
    let stream = stream();
    let pool = pool();
    let request_pool = crate::memory_fixture::ledger(0, 0).unwrap();
    let (mut runtime, _root) = runtime(&stream, &pool, &pool);
    let canonical = reserved_request(&runtime, &request_pool);
    let replacement = reserved_request(&runtime, &request_pool);
    assert_eq!(canonical.geometry(), replacement.geometry());
    let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2, 3])
        .unwrap()
        .with_inference_request(canonical);
    let controller = Controller::default();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut continuation = driver.start(prompt, config(), controller.clone()).unwrap();
    {
        let mut boundary = driver.quiescent(&mut continuation).unwrap();
        let (_, state, _) = boundary.mechanism_parts();
        let mut retained = InferenceRetention::new();
        retained.retain(&replacement);
        state.sampling.inference_retention = retained;
    }
    let before = Unchanged::capture(driver.runtime(), &[&pool, &request_pool]);
    let error = driver.advance(&mut continuation).err().unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    before.assert(driver.runtime(), &controller);
}

#[test]
fn retained_reservation_never_authorizes_unquoted_step_in_model_or_context_domain() {
    let stream = stream();
    for reserve_model in [true, false] {
        let model_pool = pool();
        let context_pool = pool();
        let source_pool = pool();
        let (mut runtime, _root) = runtime(&stream, &model_pool, &context_pool);
        let request = reserved_request(
            &runtime,
            if reserve_model {
                &model_pool
            } else {
                &context_pool
            },
        );
        let source = MlxBackend::new(&stream, &stream).with_memory_ledger(source_pool.clone());
        let prompt = MlxBackend::prepare_text_prompt(&source, vec![1, 2, 3])
            .unwrap()
            .with_inference_request(request.clone());
        let controller = Controller::default();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        // Existing prompt backing belongs to source_pool. Greedy admitted
        // sampler preparation has no native RNG allocation; the zero charge
        // is retained solely to prove it cannot authorize native step entry.
        let mut continuation = driver.start(prompt, config(), controller.clone()).unwrap();
        assert_eq!(model_pool.unquoted_owner_count().unwrap(), 0);
        assert_eq!(context_pool.unquoted_owner_count().unwrap(), 0);
        let before = Unchanged::capture(
            driver.runtime(),
            &[&model_pool, &context_pool, &source_pool],
        );
        let error = driver.advance(&mut continuation).err().unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::ReservedWorkActive)
        );
        before.assert(driver.runtime(), &controller);
        // When the second domain rejects, the temporary first-domain lease
        // must not be installed into the session's operation history.
        assert!(!driver
            .runtime()
            .session()
            .payload
            .operation_memory
            .borrow()
            .covers_pool(&model_pool));
        assert!(!driver
            .runtime()
            .session()
            .payload
            .operation_memory
            .borrow()
            .covers_pool(&context_pool));
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
