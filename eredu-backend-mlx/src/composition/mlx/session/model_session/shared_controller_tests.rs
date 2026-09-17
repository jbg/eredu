#![cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]

mod bytes;

use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, SharedTokenFilter, TextControllerStorage, TextGeneration,
    TextGenerationDriver, TextGenerationInput, TokenFilterController, TokenSamplingDecision,
};
use eredu_runtime::working_memory::{
    ControllerStorageError, WorkingMemoryError, WorkingMemoryPool,
};

struct SharedController {
    masks: [SharedTokenFilter; 1],
    calls: Rc<Cell<(usize, usize)>>,
    replace_on_decision: bool,
    additional_bytes: Option<u64>,
}

impl SharedController {
    fn new(mask: SharedTokenFilter, calls: Rc<Cell<(usize, usize)>>) -> Self {
        Self {
            masks: [mask],
            calls,
            replace_on_decision: false,
            additional_bytes: None,
        }
    }
}

impl TokenFilterController for SharedController {
    type Error = std::convert::Infallible;

    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwnedWithSharedFilters(&self.masks)
    }

    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: self.masks[0].as_ref().into(),
            // Retained shared source, visible tokenizer provenance and a
            // possible controller-owned replacement overlap. Sampling prices
            // its independently owned final filter. Counters are metadata.
            additional_host_bytes: self.additional_bytes.unwrap_or_else(|| {
                3 * self.masks[0].capacity_bytes().unwrap() + std::mem::size_of::<Self>() as u64
            }),
        })
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let (decisions, commits) = self.calls.get();
        self.calls.set((decisions + 1, commits));
        Ok(self.masks[0].as_ref().clone())
    }

    fn current_decision(&mut self) -> Result<TokenSamplingDecision<'_>, Self::Error> {
        if self.replace_on_decision {
            // Same values and capacity, but distinct ownership introduced
            // after preflight. Only this controller owns the new allocation.
            self.masks[0] = mask(self.masks[0].capacity_bytes().unwrap() as usize);
            self.replace_on_decision = false;
        }
        let filter = self.current_filter()?;
        Ok(TokenSamplingDecision::new(filter).with_shared_tokenizer_validity(&self.masks[0]))
    }

    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        assert!(self.masks[0].as_ref().allows(token));
        let (decisions, commits) = self.calls.get();
        self.calls.set((decisions, commits + 1));
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn mask(capacity: usize) -> SharedTokenFilter {
    let mut values = Vec::with_capacity(capacity);
    values.resize(64, false);
    values[11] = true;
    SharedTokenFilter::new(TokenFilter::allowed(values).unwrap())
}

fn config(capacity: Option<u64>) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.0),
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
    revision: eredu_runtime::working_memory::InferenceStateRevision,
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
fn shared_controller_matches_ordinary_output_and_preexisting_alias_owns_exact_final_charge() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let mut expected = None;
    let mut ordinary_filter = None;
    for controlled in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = runtime(&stream, &pool);
        let source = mask(193);
        let external = source.clone(); // The alias predates accounting attachment.
        let bytes = external.capacity_bytes().unwrap();
        assert!(bytes > 64);
        let calls = Rc::new(Cell::new((0, 0)));
        let controller = SharedController::new(source, calls.clone());
        let outputs = if controlled {
            ControlledTextGeneration::from_input(
                &mut runtime,
                TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
                config(Some(u64::MAX)),
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
                    config(Some(u64::MAX)),
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
            outputs
        };
        let ids = outputs
            .iter()
            .map(|token| token.token_id().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ids, [11, 11, 11]);
        if let Some(expected) = &expected {
            assert_eq!(&ids, expected);
        }
        expected = Some(ids);
        assert_eq!(calls.get(), (3, 3));
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        runtime
            .session()
            .payload
            .model
            .erased()
            .validate_text_frontier(7)
            .unwrap();
        drop((outputs, runtime));
        settle(&pool, bytes);
        ordinary_filter = Some(external.as_ref().clone());
        drop(external);
        settle(&pool, 0);
    }

    let reference_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut reference, _artifact) = self::runtime(&stream, &reference_pool);
    let reference_outputs = TextGeneration::with_token_filter(
        &mut reference,
        vec![1, 2, 3, 4, 5],
        config(Some(u64::MAX)),
        ordinary_filter.unwrap(),
    )
    .unwrap()
    .map(Result::unwrap)
    .collect::<Vec<_>>();
    assert_eq!(
        reference_outputs
            .iter()
            .map(|token| token.token_id().unwrap())
            .collect::<Vec<_>>(),
        expected.unwrap()
    );
    drop((reference_outputs, reference));
    settle(&reference_pool, 0);
}

#[test]
fn changed_shared_identity_rejects_before_native_work_at_preflight_and_decision() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for during_decision in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = runtime(&stream, &pool);
        let original = mask(193);
        let external = original.clone();
        let bytes = external.capacity_bytes().unwrap();
        let calls = Rc::new(Cell::new((0, 0)));
        let mut controller = SharedController::new(original, calls.clone());
        controller.replace_on_decision = during_decision;
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut continuation = driver
            .start_input(
                TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
                config(Some(u64::MAX)),
                controller,
            )
            .unwrap();
        if !during_decision {
            let replacement = mask(bytes as usize);
            assert_ne!(replacement.identity(), external.identity());
            assert_eq!(replacement.capacity_bytes(), Some(bytes));
            continuation.controller_mut().masks[0] = replacement;
        }
        let before = NoWork::capture(driver.runtime(), &pool);
        let error = driver.advance(&mut continuation).err().unwrap();
        assert!(cause::<ControllerStorageError>(&error).is_some(), "{error}");
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(calls.get(), (usize::from(during_decision), 0));
        before.assert_unchanged(driver.runtime(), &pool);
        drop(continuation);
        drop(driver);
        drop(runtime);
        settle(&pool, bytes);
        drop(external);
        settle(&pool, 0);
    }
}

#[test]
fn rejected_reservation_never_attaches_shared_controller_storage() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let original = mask(193);
    let external = original.clone();
    let calls = Rc::new(Cell::new((0, 0)));
    let before = NoWork::capture(&runtime, &pool);
    let error = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
        config(Some(before.bytes)),
        SharedController::new(original, calls.clone()),
    )
    .err()
    .unwrap();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(calls.get(), (0, 0));
    before.assert_unchanged(&runtime, &pool);
    drop(runtime);
    settle(&pool, 0);
    assert_eq!(external.capacity_bytes(), Some(193));
}

#[test]
fn shared_inventory_must_fit_its_own_declared_workspace_before_reservation() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let original = mask(193);
    let external = original.clone();
    let calls = Rc::new(Cell::new((0, 0)));
    let mut controller = SharedController::new(original, calls.clone());
    controller.additional_bytes = Some(external.capacity_bytes().unwrap() - 1);
    let before = NoWork::capture(&runtime, &pool);
    let error = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
        config(Some(u64::MAX)),
        controller,
    )
    .err()
    .unwrap();
    assert!(
        matches!(
            cause::<ControllerStorageError>(&error),
            Some(ControllerStorageError::UnpricedSharedStorage {
                required_bytes: 193,
                available_bytes: 192,
            })
        ),
        "{error}"
    );
    assert_eq!(calls.get(), (0, 0));
    before.assert_unchanged(&runtime, &pool);
    drop(runtime);
    settle(&pool, 0);
    assert_eq!(external.capacity_bytes(), Some(193));
}

#[test]
fn shared_controller_charge_survives_later_readiness_rejection() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let original = mask(193);
    let external = original.clone();
    let bytes = external.capacity_bytes().unwrap();
    let calls = Rc::new(Cell::new((0, 0)));
    let generation = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
        config(Some(u64::MAX)),
        SharedController::new(original, calls.clone()),
    )
    .unwrap();
    let error = generation
        .finish_text_preparation(
            eredu_core::run_preparation::TextPreparationStage::Delivery,
            Err::<(), _>(std::io::Error::other("fixture readiness rejection")),
            std::io::Error::other,
        )
        .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(calls.get(), (0, 0));
    drop(generation);
    drop(runtime);
    settle(&pool, bytes);
    drop(external);
    settle(&pool, 0);
}

#[test]
fn shared_controller_preserves_forced_choice_provenance_through_multiple_native_steps() {
    use eredu_runtime::execution_control::TokenChoiceController;

    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let mut values = Vec::with_capacity(193);
    values.resize(64, false);
    values[11] = true;
    values[17] = true;
    let original = SharedTokenFilter::new(TokenFilter::allowed(values).unwrap());
    let external = original.clone();
    let bytes = external.capacity_bytes().unwrap();
    let calls = Rc::new(Cell::new((0, 0)));
    let mut controller = TokenChoiceController::new(
        SharedController::new(original, calls.clone()),
        eredu_runtime::TokenDomain::new(64),
    );
    // Staging precedes admission and includes the wrapper's extra retained
    // pre-override mask. No post-preparation mutable policy access is needed.
    controller.force_next(17).unwrap();
    assert_eq!(calls.get(), (1, 0));
    let outputs = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
        config(Some(u64::MAX)),
        controller,
    )
    .unwrap()
    .map(|token| token.unwrap().into_output())
    .collect::<Vec<_>>();
    let ids = outputs
        .iter()
        .map(|token| token.token_id().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 3);
    assert_eq!(ids[0], 17);
    assert!(ids.iter().all(|id| [11, 17].contains(id)));
    assert_eq!(calls.get(), (4, 3));
    runtime
        .session()
        .payload
        .model
        .erased()
        .validate_text_frontier(7)
        .unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((outputs, runtime));
    settle(&pool, bytes);
    drop(external);
    settle(&pool, 0);
}

#[test]
fn loading_hook_registers_shared_mask_before_inference_and_rejects_active_run_factories() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let initial_bytes = pool.used_bytes().unwrap();
    let factory_calls = Cell::new(0);
    let prepared = MlxBackend::prepare_shared_token_filter(&runtime, || {
        factory_calls.set(factory_calls.get() + 1);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        let mut values = Vec::with_capacity(193);
        values.resize(64, false);
        values[11] = true;
        TokenFilter::allowed(values).unwrap()
    })
    .unwrap();
    assert_eq!(factory_calls.get(), 1);
    assert_eq!(prepared.capacity_bytes(), Some(193));
    assert_eq!(pool.used_bytes().unwrap(), initial_bytes + 193);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let external = prepared.clone();
    let calls = Rc::new(Cell::new((0, 0)));
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut continuation = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
            config(Some(u64::MAX)),
            SharedController::new(prepared, calls.clone()),
        )
        .unwrap();
    let before = NoWork::capture(driver.runtime(), &pool);
    let rejected_factory_calls = Cell::new(0);
    let error = MlxBackend::prepare_shared_token_filter(driver.runtime(), || {
        rejected_factory_calls.set(rejected_factory_calls.get() + 1);
        TokenFilter::All
    })
    .unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    assert_eq!(rejected_factory_calls.get(), 0);
    assert_eq!(calls.get(), (0, 0));
    before.assert_unchanged(driver.runtime(), &pool);

    // The rejected loading factory must leave the admitted continuation usable.
    let mut outputs = Vec::new();
    while let Some(token) = driver.advance(&mut continuation).unwrap() {
        outputs.push(token.into_output());
        assert!(driver
            .take_completed_step(&mut continuation)
            .unwrap()
            .is_none());
    }
    assert_eq!(
        outputs
            .iter()
            .map(|token| token.token_id().unwrap())
            .collect::<Vec<_>>(),
        [11, 11, 11]
    );
    assert_eq!(calls.get(), (3, 3));
    driver
        .runtime()
        .session()
        .payload
        .model
        .erased()
        .validate_text_frontier(7)
        .unwrap();
    drop(continuation);
    drop(driver);
    drop(outputs);
    drop(runtime);
    // Admission reused the loading-time attachment. The escaped shared owner
    // retains one exact charge after all generation and native owners retire.
    settle(&pool, 193);
    assert!(external.as_ref().allows(11));
    drop(external);
    settle(&pool, 0);
}
