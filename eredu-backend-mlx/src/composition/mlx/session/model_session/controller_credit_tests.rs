#![cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]

use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, SharedTokenFilter, TextControllerStorage, TextFilterWorkspace,
    TextGenerationDriver, TextGenerationInput, TokenFilterController, TokenSamplingDecision,
};
use eredu_runtime::{
    execution_control::TokenChoiceController,
    working_memory::{
        InferenceStateRevision, WorkingMemoryError, WorkingMemoryPool, WorkingMemoryReservation,
    },
    TokenDomain,
};

const MASK_BYTES: u64 = 193;
const FORCED_TOKEN: u32 = 11;

#[derive(Clone)]
struct Controller {
    masks: [SharedTokenFilter; 1],
    calls: Rc<Cell<(usize, usize)>>,
    optional: bool,
}

impl Controller {
    fn new(mask: SharedTokenFilter, optional: bool) -> Self {
        Self {
            masks: [mask],
            calls: Rc::new(Cell::new((0, 0))),
            optional,
        }
    }
}

impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;

    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwnedWithSharedFilters(&self.masks)
    }

    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: if self.optional {
                TextFilterWorkspace::OptionalMask {
                    max_mask_positions: 64,
                    mask_capacity_bytes: MASK_BYTES,
                }
            } else {
                self.masks[0].as_ref().into()
            },
            // The immutable source is the only additional numerical payload.
            // Sampling separately prices each owned emitted filter; the shared
            // validity reference aliases this source. Counters are metadata.
            additional_host_bytes: MASK_BYTES,
        })
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let (decisions, commits) = self.calls.get();
        self.calls.set((decisions + 1, commits));
        Ok(if self.optional && commits % 2 == 0 {
            TokenFilter::All
        } else {
            self.masks[0].as_ref().clone()
        })
    }

    fn current_decision(&mut self) -> Result<TokenSamplingDecision<'_>, Self::Error> {
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

fn filter(all: bool) -> TokenFilter {
    let mut values = Vec::with_capacity(MASK_BYTES as usize);
    values.resize(64, all);
    values[FORCED_TOKEN as usize] = true;
    TokenFilter::allowed(values).unwrap()
}

fn prepared_mask(pool: &WorkingMemoryPool, all: bool) -> SharedTokenFilter {
    let before = pool.used_bytes().unwrap();
    let mask = pool.prepare_shared_token_filter(|| filter(all)).unwrap();
    assert_eq!(mask.capacity_bytes(), Some(MASK_BYTES));
    assert_eq!(mask.as_ref().allowed_mask().unwrap().len(), 64);
    assert_eq!(pool.used_bytes().unwrap(), before + MASK_BYTES);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    mask
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

fn probe<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    prompt: &Vec<u32>,
    controller: &C,
) -> MlxTextPreparation {
    MlxBackend::admit_text_preparation(
        runtime,
        &eredu_core::TextPreparationInput::TokenIds {
            positions: prompt.len() as u64,
            capacity_bytes: (prompt.capacity() * std::mem::size_of::<u32>()) as u64,
        },
        config(Some(u64::MAX)),
        controller,
    )
    .unwrap()
}

fn reservation(preparation: &MlxTextPreparation) -> &WorkingMemoryReservation {
    preparation
        .request
        .as_ref()
        .unwrap()
        .request()
        .memory_reservation()
        .unwrap()
}

fn full_quote<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    prompt: &Vec<u32>,
    geometry: eredu_core::InferenceGeometry,
    controller: &C,
) -> eredu_core::RuntimeStateEstimate {
    let before = NoWork::capture(runtime, runtime.backend().memory_pool());
    let (state, width) = super::text_quote::quote(
        runtime.session(),
        geometry,
        (prompt.capacity() * std::mem::size_of::<u32>()) as u64,
        config(Some(u64::MAX)),
        controller.inference_workspace(3).unwrap(),
    )
    .unwrap();
    assert_eq!(width, 64);
    before.assert_unchanged(runtime, runtime.backend().memory_pool());
    state
}

fn full_bytes(state: &eredu_core::RuntimeStateEstimate) -> u64 {
    state
        .requested_state_bytes
        .checked_add(
            state
                .execution_workspace
                .as_ref()
                .unwrap()
                .peak_bytes()
                .unwrap()
                .unwrap(),
        )
        .unwrap()
}

fn generate<C: TokenFilterController>(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    prompt: Vec<u32>,
    config: TextGenerationConfig,
    controller: C,
    controlled: bool,
) -> Vec<MlxTextToken> {
    if controlled {
        ControlledTextGeneration::from_input(
            runtime,
            TextGenerationInput::TokenIds(prompt),
            config,
            controller,
        )
        .unwrap()
        .map(|token| token.unwrap().into_output())
        .collect()
    } else {
        let mut driver = TextGenerationDriver::new(runtime);
        let mut continuation = driver
            .start_input(TextGenerationInput::TokenIds(prompt), config, controller)
            .unwrap();
        let mut outputs = Vec::new();
        while let Some(token) = driver.advance(&mut continuation).unwrap() {
            outputs.push(token.into_output());
            assert!(driver
                .take_completed_delivery(&mut continuation)
                .unwrap()
                .is_none());
        }
        outputs
    }
}

fn ids(outputs: &[MlxTextToken]) -> Vec<u32> {
    outputs
        .iter()
        .map(|token| token.token_id().unwrap())
        .collect()
}

fn reference(stream: &Stream, prompt: Vec<u32>, all: bool) -> Vec<u32> {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(stream, &pool);
    let source = prepared_mask(&pool, all);
    let outputs = generate(
        &mut runtime,
        prompt,
        config(None),
        Controller::new(source, all),
        false,
    );
    let result = ids(&outputs);
    drop(outputs);
    drop(runtime);
    settle(&pool, 0);
    result
}

fn native_bytes(
    runtime: Option<&ModelRuntime<MlxBackend<'_>>>,
    outputs: &[&[MlxTextToken]],
    arrays: &[&Array],
) -> u64 {
    let mut storage = match runtime {
        Some(runtime) => {
            runtime.session().ensure_no_submission_in_flight().unwrap();
            let (mut nonstate, decoder) = runtime
                .session()
                .payload
                .retained_idle_storage()
                .unwrap()
                .into_parts();
            nonstate.merge(decoder).unwrap();
            nonstate
        }
        None => RetainedStorage::default(),
    };
    for token in outputs.iter().flat_map(|outputs| outputs.iter()) {
        storage.include_array(&token.value).unwrap();
    }
    for array in arrays {
        storage.include_array(array).unwrap();
    }
    storage.byte_bound().unwrap().unwrap()
}

fn escaped_decoder_array(runtime: &ModelRuntime<MlxBackend<'_>>) -> Array {
    let context = eredu_nn::workspace::WorkspaceContext::new(
        crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms::current_host().unwrap(),
    );
    let projected = runtime
        .session()
        .payload
        .model
        .erased()
        .project_resident_workspace_with_storage(std::num::NonZeroU32::new(1).unwrap(), &context)
        .unwrap();
    assert!(projected.storage.is_complete());
    let (identity, _, _) = projected
        .storage
        .iter()
        .find(|(_, bytes, _)| *bytes > 0)
        .unwrap();
    projected.storage.native_array(identity).unwrap().clone()
}

struct NoWork {
    paths: paths::Counts,
    inputs: usize,
    resets: usize,
    frontier: Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>,
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
            frontier: runtime.session().payload.model.erased().state_snapshot(),
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
        assert_eq!(model.state_snapshot(), self.frontier);
        assert_eq!(
            model.retained_inference_authority().unwrap().revision(),
            &self.revision
        );
        assert_eq!(pool.used_bytes().unwrap(), self.bytes);
        assert_eq!(pool.peak_bytes().unwrap(), self.peak);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
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

fn reject_one_byte_short<C: TokenFilterController + Clone>(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    pool: &WorkingMemoryPool,
    prompt: &Vec<u32>,
    controller: &C,
    charge: u64,
) -> u64 {
    let before = NoWork::capture(runtime, pool);
    let exact = before.bytes.checked_add(charge).unwrap();
    let error = ControlledTextGeneration::from_input(
        runtime,
        TextGenerationInput::TokenIds(prompt.clone()),
        config(Some(exact - 1)),
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
    before.assert_unchanged(runtime, pool);
    exact
}

#[test]
fn registered_controller_credit_preserves_full_diagnostics_and_exact_capacity_parity() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let mut sequences = Vec::new();
    for controlled in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = runtime(&stream, &pool);
        let source = prepared_mask(&pool, false);
        let controller = Controller::new(source.clone(), false);
        let calls = controller.calls.clone();
        let prompt = vec![1, 2, 3, 4, 5];
        let initial = pool.used_bytes().unwrap();
        let preparation = probe(&runtime, &prompt, &controller);
        let accepted = reservation(&preparation);
        let full = full_quote(&runtime, &prompt, accepted.geometry(), &controller);
        assert_eq!(accepted.admission().state, full);
        assert_eq!(
            accepted.admission().incremental_required_bytes,
            accepted.bytes()
        );
        let charge = accepted.bytes();
        assert_eq!(charge + MASK_BYTES, full_bytes(&full));
        assert_eq!(pool.used_bytes().unwrap(), initial + charge);
        let historical = preparation.request.as_ref().unwrap().request().clone();
        drop(preparation);
        settle(&pool, initial);

        // Equal bits and capacity without prior registration get the complete
        // envelope. Adoption after admission must not retroactively earn credit.
        let fallback = Controller::new(SharedTokenFilter::new(filter(false)), false);
        let unregistered = probe(&runtime, &prompt, &fallback);
        assert_eq!(reservation(&unregistered).admission().state, full);
        assert_eq!(reservation(&unregistered).bytes(), full_bytes(&full));
        assert_eq!(fallback.calls.get(), (0, 0));
        drop(unregistered);
        drop(fallback);
        settle(&pool, initial);

        let exact = reject_one_byte_short(&mut runtime, &pool, &prompt, &controller, charge);
        assert_eq!(calls.get(), (0, 0));
        let outputs = generate(
            &mut runtime,
            prompt,
            config(Some(exact)),
            controller,
            controlled,
        );
        let actual = ids(&outputs);
        assert_eq!(actual, [FORCED_TOKEN; 3]);
        assert_eq!(calls.get(), (3, 3));
        runtime
            .session()
            .payload
            .model
            .erased()
            .validate_text_frontier(7)
            .unwrap();
        settle(
            &pool,
            native_bytes(Some(&runtime), &[&outputs], &[]) + MASK_BYTES,
        );
        eprintln!("fresh controller credit: controlled={controlled}, full={}, incremental={charge}, source={MASK_BYTES}, outputs={actual:?}", full_bytes(&full));
        sequences.push(actual);
        drop(runtime);
        settle(&pool, native_bytes(None, &[&outputs], &[]) + MASK_BYTES);
        drop(outputs);
        settle(&pool, MASK_BYTES);
        assert!(source.as_ref().allows(FORCED_TOKEN));
        drop(source);
        // Historical request metadata contains no source payload or storage pin.
        settle(&pool, 0);
        assert_eq!(historical.memory_reservation().unwrap().bytes(), charge);
    }
    assert_eq!(sequences[0], sequences[1]);
}

#[test]
fn native_preparation_keeps_registered_host_pin_after_all_filter_owners_drop() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = runtime(&stream, &pool);
    let model_bytes = pool.used_bytes().unwrap();
    let source = prepared_mask(&pool, false);
    let controller = Controller::new(source.clone(), false);
    let calls = controller.calls.clone();
    let preparation = probe(&runtime, &vec![1, 2, 3, 4, 5], &controller);
    let charge = reservation(&preparation).bytes();
    let historical = preparation.request.as_ref().unwrap().request().clone();
    let before = NoWork::capture(&runtime, &pool);
    assert_eq!(before.bytes, model_bytes + MASK_BYTES + charge);
    drop(controller);
    drop(source);
    // The credited allocation has no filter aliases left. Its accounting pin
    // belongs to the live funding run, independent of the metadata-only proof.
    before.assert_unchanged(&runtime, &pool);
    assert_eq!(calls.get(), (0, 0));
    drop(preparation);
    settle(&pool, model_bytes);
    drop(runtime);
    settle(&pool, 0);
    assert_eq!(historical.memory_reservation().unwrap().bytes(), charge);
}

#[test]
fn cached_decoder_and_registered_controller_credits_compose_without_releasing_old_aliases() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let source = prepared_mask(&pool, false);
    let outputs_a = generate(
        &mut runtime,
        vec![1, 2, 3, 4, 5],
        config(Some(u64::MAX)),
        Controller::new(source.clone(), false),
        false,
    );
    let ids_a = ids(&outputs_a);
    assert_eq!(ids_a, [FORCED_TOKEN; 3]);
    let escaped = escaped_decoder_array(&runtime);
    let old_info = escaped.allocation_info().unwrap().unwrap();
    let old_capacity = old_info.bytes() as u64;
    let initial = native_bytes(Some(&runtime), &[&outputs_a], &[&escaped]) + MASK_BYTES;
    settle(&pool, initial);
    let controller = Controller::new(source.clone(), false);
    let calls = controller.calls.clone();
    let prompt = vec![ids_a[2], 6, 7];
    let preparation = probe(&runtime, &prompt, &controller);
    let accepted = reservation(&preparation);
    assert_eq!(accepted.geometry().cached_positions, 7);
    let full = full_quote(&runtime, &prompt, accepted.geometry(), &controller);
    assert_eq!(accepted.admission().state, full);
    let charge = accepted.bytes();
    let historical = preparation.request.as_ref().unwrap().request().clone();
    drop(preparation);
    settle(&pool, initial);

    let fallback = Controller::new(SharedTokenFilter::new(filter(false)), false);
    let without_host_credit = probe(&runtime, &prompt, &fallback);
    let decoder_only = reservation(&without_host_credit).bytes();
    assert_eq!(reservation(&without_host_credit).admission().state, full);
    assert_eq!(charge + MASK_BYTES, decoder_only);
    assert!(
        decoder_only < full_bytes(&full),
        "cached decoder must supply independent residual credit"
    );
    drop(without_host_credit);
    drop(fallback);
    settle(&pool, initial);

    let exact = reject_one_byte_short(&mut runtime, &pool, &prompt, &controller, charge);
    assert_eq!(calls.get(), (0, 0));
    assert_eq!(ids(&outputs_a), ids_a);
    let outputs_b = generate(&mut runtime, prompt, config(Some(exact)), controller, true);
    assert_eq!(calls.get(), (3, 3));
    runtime
        .session()
        .payload
        .model
        .erased()
        .validate_text_frontier(12)
        .unwrap();
    let mut full_prompt = vec![1, 2, 3, 4, 5];
    full_prompt.extend_from_slice(&ids_a);
    full_prompt.extend([6, 7]);
    assert_eq!(ids(&outputs_b), reference(&stream, full_prompt, false));
    let current_decoder = runtime
        .session()
        .payload
        .retained_idle_storage()
        .unwrap()
        .into_parts()
        .1;
    assert!(!current_decoder
        .array_allocation_facts()
        .contains_key(&old_info.identity()));
    drop(current_decoder);
    assert_eq!(escaped.allocation_info().unwrap(), Some(old_info));
    assert_eq!(
        native_bytes(Some(&runtime), &[&outputs_a, &outputs_b], &[&escaped]),
        native_bytes(Some(&runtime), &[&outputs_a, &outputs_b], &[]) + old_capacity
    );
    settle(
        &pool,
        native_bytes(Some(&runtime), &[&outputs_a, &outputs_b], &[&escaped]) + MASK_BYTES,
    );
    eprintln!("cached controller credit: full={}, decoder_only={decoder_only}, combined={charge}, source={MASK_BYTES}, escaped_decoder={old_capacity}", full_bytes(&full));
    drop(outputs_b);
    drop(runtime);
    settle(
        &pool,
        native_bytes(None, &[&outputs_a], &[&escaped]) + MASK_BYTES,
    );
    assert_eq!(ids(&outputs_a), ids_a);
    drop(outputs_a);
    settle(&pool, old_capacity + MASK_BYTES);
    drop(escaped);
    settle(&pool, MASK_BYTES);
    drop(source);
    settle(&pool, 0);
    assert_eq!(historical.memory_reservation().unwrap().bytes(), charge);
}

#[test]
fn registered_source_credit_preserves_optional_final_and_forced_preoverride_allowances() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let baseline = reference(&stream, vec![1, 2, 3, 4, 5], true)[0];
    assert_ne!(baseline, FORCED_TOKEN);
    let suffix = reference(&stream, vec![1, 2, 3, 4, 5, FORCED_TOKEN], true);
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let source = prepared_mask(&pool, true);
    let plain = Controller::new(source.clone(), true);
    let calls = plain.calls.clone();
    let mut controller = TokenChoiceController::new(plain.clone(), TokenDomain::new(64));
    controller.force_next(FORCED_TOKEN).unwrap();
    assert_eq!(calls.get(), (1, 0));
    let prompt = vec![1, 2, 3, 4, 5];
    let initial = pool.used_bytes().unwrap();
    let preparation = probe(&runtime, &prompt, &controller);
    let accepted = reservation(&preparation);
    let full = full_quote(&runtime, &prompt, accepted.geometry(), &controller);
    let plain_full = full_quote(&runtime, &prompt, accepted.geometry(), &plain);
    assert_eq!(accepted.admission().state, full);
    assert_eq!(full_bytes(&full), full_bytes(&plain_full) + MASK_BYTES);
    let charge = accepted.bytes();
    assert_eq!(charge + MASK_BYTES, full_bytes(&full));
    assert_eq!(charge, full_bytes(&plain_full));
    assert_eq!(
        controller
            .inference_workspace(3)
            .unwrap()
            .additional_host_bytes,
        2 * MASK_BYTES
    );
    assert!(matches!(
        controller.inference_workspace(3).unwrap().filter,
        TextFilterWorkspace::OptionalMask {
            max_mask_positions: 64,
            mask_capacity_bytes: MASK_BYTES
        }
    ));
    drop(preparation);
    settle(&pool, initial);
    let exact = reject_one_byte_short(&mut runtime, &pool, &prompt, &controller, charge);
    assert_eq!(calls.get(), (1, 0));
    drop(plain);
    let outputs = generate(&mut runtime, prompt, config(Some(exact)), controller, true);
    let actual = ids(&outputs);
    assert_eq!(actual[0], FORCED_TOKEN);
    assert_ne!(actual[0], baseline);
    assert_eq!(&actual[1..], &suffix[..2]);
    assert_eq!(calls.get(), (4, 3));
    eprintln!("optional controller credit: full={}, incremental={charge}, source={MASK_BYTES}, baseline={baseline}, outputs={actual:?}", full_bytes(&full));
    drop(outputs);
    drop(runtime);
    settle(&pool, MASK_BYTES);
    drop(source);
    settle(&pool, 0);
}
