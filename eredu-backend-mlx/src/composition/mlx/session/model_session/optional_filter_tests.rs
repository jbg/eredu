#![cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]

use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, TextControllerStorage, TextFilterWorkspace, TextGeneration,
    TextGenerationDriver, TextGenerationInput, TokenFilterController,
};
use eredu_runtime::{
    execution_control::TokenChoiceController,
    working_memory::{InferenceStateRevision, WorkingMemoryError, WorkingMemoryPool},
    TokenDomain,
};

const FORCED_TOKEN: u32 = 11;

#[derive(Clone, Default)]
struct AllController(Rc<Cell<(usize, usize)>>);

impl TokenFilterController for AllController {
    type Error = std::convert::Infallible;

    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwned
    }

    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: TextFilterWorkspace::Exact(&TokenFilter::All),
            additional_host_bytes: 0,
        })
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let (decisions, commits) = self.0.get();
        self.0.set((decisions + 1, commits));
        Ok(TokenFilter::All)
    }

    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        assert!(token < 64);
        let (decisions, commits) = self.0.get();
        self.0.set((decisions, commits + 1));
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn config(capacity: Option<u64>, outputs: usize) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(outputs),
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

fn runtime(
    stream: &Stream,
    pool: &WorkingMemoryPool,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let source = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(stream, &source).with_memory_pool(pool.clone());
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model = eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
        .unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    (runtime, artifact)
}

fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}

fn ids(outputs: &[MlxTextToken]) -> Vec<u32> {
    outputs
        .iter()
        .map(|output| output.token_id().unwrap())
        .collect()
}

fn ordinary_reference(stream: &Stream, prompt: Vec<u32>, count: usize) -> Vec<u32> {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(stream, &pool);
    let outputs = TextGeneration::new(&mut runtime, prompt, config(None, count))
        .unwrap()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    let result = ids(&outputs);
    drop(outputs);
    drop(runtime);
    settle(&pool, 0);
    result
}

fn suffix_reference(stream: &Stream) -> Vec<u32> {
    ordinary_reference(stream, vec![1, 2, 3, 4, 5, FORCED_TOKEN], 2)
}

fn force_before_preparation(
    calls: Rc<Cell<(usize, usize)>>,
) -> TokenChoiceController<AllController> {
    let mut controller = TokenChoiceController::new(AllController(calls), TokenDomain::new(64));
    controller.force_next(FORCED_TOKEN).unwrap();
    controller
}

fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

struct NoWork {
    paths: paths::Counts,
    inputs: usize,
    resets: usize,
    revision: InferenceStateRevision,
    bytes: u64,
    peak: u64,
}

impl NoWork {
    fn capture(runtime: &ModelRuntime<MlxBackend<'_>>, pool: &WorkingMemoryPool) -> Self {
        Self {
            paths: paths::snapshot(),
            inputs: paths::session_input_creation_attempts(),
            resets: paths::session_reset_attempts(),
            revision: runtime
                .session()
                .payload
                .model
                .erased()
                .retained_inference_authority()
                .unwrap()
                .revision()
                .clone(),
            bytes: pool.used_bytes().unwrap(),
            peak: pool.peak_bytes().unwrap(),
        }
    }

    fn assert_unchanged(&self, runtime: &ModelRuntime<MlxBackend<'_>>, pool: &WorkingMemoryPool) {
        assert_eq!(paths::snapshot(), self.paths);
        assert_eq!(paths::session_input_creation_attempts(), self.inputs);
        assert_eq!(paths::session_reset_attempts(), self.resets);
        let model = runtime.session().payload.model.erased();
        model.validate_text_frontier(0).unwrap();
        assert_eq!(
            model.retained_inference_authority().unwrap().revision(),
            &self.revision
        );
        assert_eq!(pool.used_bytes().unwrap(), self.bytes);
        assert_eq!(pool.peak_bytes().unwrap(), self.peak);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}

#[test]
fn optional_choice_forced_then_unfiltered_matches_ordinary_and_controlled_native_runs() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let baseline = ordinary_reference(&stream, vec![1, 2, 3, 4, 5], 1)[0];
    assert_ne!(
        baseline, FORCED_TOKEN,
        "fixture must force a changed prediction"
    );
    let reference = suffix_reference(&stream);
    let mut sequences = Vec::new();
    for controlled in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = runtime(&stream, &pool);
        let calls = Rc::new(Cell::new((0, 0)));
        let controller = force_before_preparation(calls.clone());
        assert_eq!(calls.get(), (1, 0));
        let outputs = if controlled {
            ControlledTextGeneration::from_input(
                &mut runtime,
                TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
                config(Some(u64::MAX), 3),
                controller,
            )
            .unwrap()
            .map(|token| token.unwrap().into_output())
            .collect::<Vec<_>>()
        } else {
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let mut continuation = driver
                .start_input(
                    TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
                    config(Some(u64::MAX), 3),
                    controller,
                )
                .unwrap();
            let mut outputs = Vec::new();
            while let Some(token) = driver.advance(&mut continuation).unwrap() {
                outputs.push(token.into_output());
                assert!(driver
                    .take_completed_step(&mut continuation)
                    .unwrap()
                    .is_none());
            }
            assert_eq!(continuation.controller().pending_forced(), None);
            assert!(!continuation.controller().last_committed_was_forced());
            outputs
        };
        let actual = ids(&outputs);
        assert_eq!(actual.len(), 3);
        assert_eq!(actual[0], FORCED_TOKEN);
        assert_ne!(actual[0], baseline);
        assert_eq!(&actual[1..], reference.as_slice());
        assert_eq!(calls.get(), (4, 3));
        runtime
            .session()
            .payload
            .model
            .erased()
            .validate_text_frontier(7)
            .unwrap();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        sequences.push(actual);
        drop(outputs);
        drop(runtime);
        settle(&pool, 0);
    }
    assert_eq!(sequences[0], sequences[1]);
}

#[test]
fn optional_choice_rejects_one_byte_short_and_runs_at_exact_real_quote_capacity() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let baseline = ordinary_reference(&stream, vec![1, 2, 3, 4, 5], 1)[0];
    assert_ne!(
        baseline, FORCED_TOKEN,
        "fixture must force a changed prediction"
    );
    let reference = suffix_reference(&stream);
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let initial = pool.used_bytes().unwrap();
    let calls = Rc::new(Cell::new((0, 0)));
    let controller = force_before_preparation(calls.clone());
    let probe = MlxBackend::admit_text_preparation(
        &runtime,
        &eredu_core::TextPreparationInput::TokenIds {
            positions: 5,
            capacity_bytes: 20,
        },
        config(Some(u64::MAX), 3),
        &controller,
    )
    .unwrap();
    let charge = probe
        .request
        .as_ref()
        .unwrap()
        .request()
        .memory_reservation()
        .unwrap()
        .bytes();
    assert!(charge > 0);
    let exact = initial.checked_add(charge).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), exact);
    drop(probe);
    settle(&pool, initial);
    let before = NoWork::capture(&runtime, &pool);
    let error = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
        config(Some(exact - 1), 3),
        controller.clone(),
    )
    .err()
    .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::BudgetExceeded {
            required_bytes: charge,
            available_bytes: charge - 1,
        })
    );
    assert_eq!(calls.get(), (1, 0));
    before.assert_unchanged(&runtime, &pool);
    let generation = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
        config(Some(exact), 3),
        controller,
    )
    .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), exact);
    let outputs = generation
        .map(|token| token.unwrap().into_output())
        .collect::<Vec<_>>();
    let actual = ids(&outputs);
    assert_eq!(actual[0], FORCED_TOKEN);
    assert_ne!(actual[0], baseline);
    assert_eq!(&actual[1..], reference.as_slice());
    assert_eq!(calls.get(), (4, 3));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), before.peak.max(exact));
    eprintln!(
        "optional native filtering: charge={charge}, exact_capacity={exact}, baseline={baseline}, forced={FORCED_TOKEN}, outputs={actual:?}"
    );
    drop(outputs);
    drop(runtime);
    settle(&pool, 0);
}

#[test]
fn optional_filter_quote_does_not_authorize_post_preparation_controller_mutation() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let calls = Rc::new(Cell::new((0, 0)));
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut continuation = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
            config(Some(u64::MAX), 3),
            TokenChoiceController::new(AllController(calls.clone()), TokenDomain::new(64)),
        )
        .unwrap();
    continuation
        .controller_mut()
        .force_next(FORCED_TOKEN)
        .unwrap();
    assert_eq!(calls.get(), (1, 0));
    let before = NoWork::capture(driver.runtime(), &pool);
    let error = driver.advance(&mut continuation).err().unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(calls.get(), (1, 0));
    before.assert_unchanged(driver.runtime(), &pool);
    drop(continuation);
    drop(driver);
    drop(runtime);
    settle(&pool, 0);
}
