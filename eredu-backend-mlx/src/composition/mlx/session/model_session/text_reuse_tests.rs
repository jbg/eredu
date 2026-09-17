#![cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]

use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, TextGeneration, TextGenerationDriver, TextGenerationInput,
    TokenFilterController,
};
use eredu_runtime::working_memory::{
    InferenceStateRevision, WorkingMemoryError, WorkingMemoryPool,
};

#[derive(Clone, Default)]
struct Controller(Rc<Cell<(usize, usize, usize)>>);

impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;

    fn inference_workspace_is_run_owned(&self) -> bool {
        // Shared counters are instrumentation, not numerical masks or history.
        true
    }

    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: (std::mem::size_of::<Self>()
                + std::mem::size_of::<Cell<(usize, usize, usize)>>()
                + 2 * std::mem::size_of::<usize>()) as u64,
        })
    }

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

fn config(managed: bool) -> TextGenerationConfig {
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
        managed_memory_capacity_bytes: managed.then_some(u64::MAX),
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
    assert!(pool.used_bytes().unwrap() > 0);
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

/// Inspect physical roots without retaining an extra inventory across drops.
/// Including all outputs in the same inventory also deduplicates shared views.
fn live_storage_bytes(
    runtime: Option<&ModelRuntime<MlxBackend<'_>>>,
    outputs: &[&[MlxTextToken]],
) -> u64 {
    live_storage_bytes_with_arrays(runtime, outputs, &[])
}

fn live_storage_bytes_with_arrays(
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
    storage
        .byte_bound()
        .unwrap()
        .expect("completed model and token storage must have a complete physical inventory")
}

/// Clone one already certified decoder root, without evaluating or introducing
/// an operation owner. The temporary projection is dropped before returning.
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
        .expect("populated decoder must retain a nonempty physical allocation");
    projected.storage.native_array(identity).unwrap().clone()
}

fn generate(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    ids: Vec<u32>,
    managed: bool,
    controlled: bool,
) -> Vec<MlxTextToken> {
    let outputs = if controlled {
        let controller = Controller::default();
        let outputs = ControlledTextGeneration::from_input(
            runtime,
            TextGenerationInput::TokenIds(ids),
            config(managed),
            controller.clone(),
        )
        .unwrap()
        .map(|token| token.unwrap().into_output())
        .collect::<Vec<_>>();
        assert_eq!(controller.0.get().0, 3);
        assert_eq!(controller.0.get().1, 3);
        outputs
    } else {
        TextGeneration::new(runtime, ids, config(managed))
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
    };
    assert_eq!(outputs.len(), 3);
    outputs
}

fn token_ids(outputs: &[MlxTextToken]) -> Vec<u32> {
    let ids = outputs
        .iter()
        .map(|token| token.token_id().unwrap())
        .collect::<Vec<_>>();
    assert!(ids.iter().all(|id| *id < 64));
    ids
}

fn assert_frontier(runtime: &ModelRuntime<MlxBackend<'_>>, position: u64) {
    let model = runtime.session().payload.model.erased();
    model.validate_text_frontier(position).unwrap();
    let frontier = model.state_snapshot();
    assert!(!frontier.is_empty());
    assert!(frontier
        .iter()
        .all(|(actual, _)| *actual as u64 == position));
}

fn reference(stream: &Stream, prompt: Vec<u32>, controlled: bool) -> Vec<u32> {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(stream, &pool);
    let outputs = generate(&mut runtime, prompt, false, controlled);
    let ids = token_ids(&outputs);
    assert_frontier(&runtime, 12);
    drop((outputs, runtime));
    settle(&pool, 0);
    ids
}

#[test]
fn funded_request_identity_cannot_replace_the_private_preparation_scope() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = runtime(&stream, &pool);
    let before = Unchanged::capture(&runtime, &pool);
    let controller = Controller::default();
    let mut preparation = MlxBackend::admit_text_preparation(
        &runtime,
        &eredu_core::TextPreparationInput::TokenIds {
            positions: 5,
            capacity_bytes: 20,
        },
        config(true),
        &controller,
    )
    .unwrap();
    assert!(preparation
        .request
        .as_ref()
        .unwrap()
        .request()
        .requires_funding_scope());
    preparation.quote = None;
    settle(&pool, before.bytes);
    let before_rejection = Unchanged::capture(&runtime, &pool);
    let error = MlxBackend::prepare_text_prompt_admitted(
        runtime.backend(),
        vec![1, 2, 3, 4, 5],
        &preparation,
    )
    .err()
    .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    let error =
        MlxBackend::start_text_generation_admitted(runtime.backend(), config(true), &preparation)
            .err()
            .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    before_rejection.assert_no_work(&runtime, &pool, &controller);
    drop((preparation, runtime));
    settle(&pool, 0);
}

#[test]
fn quoted_prefix_reuse_matches_fresh_generation_and_preserves_escaped_old_charges() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for controlled in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = runtime(&stream, &pool);
        let published_bytes = pool.used_bytes().unwrap();
        let prompt_a = vec![1, 2, 3, 4, 5];
        let outputs_a = generate(&mut runtime, prompt_a.clone(), true, controlled);
        let ids_a = token_ids(&outputs_a);
        assert_frontier(&runtime, 7);
        let (request_a, revision_a) = {
            let retained = runtime
                .session()
                .payload
                .model
                .erased()
                .retained_inference_authority()
                .unwrap();
            assert_eq!(retained.requests().len(), 1);
            let admission = retained.admission().unwrap();
            assert_eq!(admission.position(), 7);
            assert_eq!(admission.request().geometry().cached_positions, 0);
            (admission.request().clone(), retained.revision().clone())
        };
        let charge_a = request_a.memory_reservation().unwrap().bytes();
        assert!(charge_a > 0);
        let live_a = live_storage_bytes(Some(&runtime), &[&outputs_a]);
        settle(&pool, live_a);
        let retained_a = live_a.checked_sub(published_bytes).unwrap();
        assert!(retained_a > 0);
        assert!(
            retained_a < charge_a / 2,
            "completed A must retire its workspace: retained={retained_a}, original={charge_a}"
        );
        assert_eq!(
            outputs_a.last().unwrap().state_revision(),
            Some(&revision_a)
        );
        let escaped_decoder = escaped_decoder_array(&runtime);
        let escaped_decoder_info = escaped_decoder.allocation_info().unwrap().unwrap();
        assert!(escaped_decoder_info.bytes() > 0);
        settle(&pool, live_a);

        // The first two emitted tokens are already cached. The final token is
        // still pending, so the next prompt must explicitly include it once.
        let outputs_b = generate(&mut runtime, vec![ids_a[2], 6, 7], true, controlled);
        let ids_b = token_ids(&outputs_b);
        assert_frontier(&runtime, 12);
        let charge_b = {
            let retained = runtime
                .session()
                .payload
                .model
                .erased()
                .retained_inference_authority()
                .unwrap();
            assert_eq!(retained.requests().len(), 2);
            assert!(retained
                .requests()
                .any(|request| request.validate_same_request(&request_a).is_ok()));
            let admission = retained.admission().unwrap();
            assert_eq!(admission.position(), 12);
            let request_b = admission.request();
            assert_eq!(request_b.geometry().cached_positions, 7);
            assert_eq!(request_b.geometry().input_positions, 3);
            assert_eq!(request_b.geometry().max_output_tokens, 3);
            assert!(request_b.validate_same_request(&request_a).is_err());
            assert_ne!(retained.revision(), &revision_a);
            assert_eq!(
                outputs_b.last().unwrap().state_revision(),
                Some(retained.revision())
            );
            for (index, token) in outputs_b.iter().enumerate() {
                assert_eq!(token.step_receipt().unwrap().attempt(), index as u64);
                let token_retention = token.owner.inference_retention();
                for request in [&request_a, request_b] {
                    assert!(token_retention
                        .requests()
                        .any(|held| held.validate_same_request(request).is_ok()));
                }
            }
            request_b.memory_reservation().unwrap().bytes()
        };
        assert!(charge_b > 0);
        let current_decoder = runtime
            .session()
            .payload
            .retained_idle_storage()
            .unwrap()
            .into_parts()
            .1;
        assert!(!current_decoder
            .array_allocation_facts()
            .contains_key(&escaped_decoder_info.identity()));
        drop(current_decoder);
        assert_eq!(
            escaped_decoder.allocation_info().unwrap(),
            Some(escaped_decoder_info)
        );
        let without_escaped_decoder = live_storage_bytes(Some(&runtime), &[&outputs_a, &outputs_b]);
        let live_b = live_storage_bytes_with_arrays(
            Some(&runtime),
            &[&outputs_a, &outputs_b],
            &[&escaped_decoder],
        );
        assert_eq!(
            live_b,
            without_escaped_decoder + escaped_decoder_info.bytes() as u64
        );
        settle(&pool, live_b);
        let original_total = charge_a.checked_add(charge_b).unwrap();
        let retained_total = live_b.checked_sub(published_bytes).unwrap();
        assert!(retained_total > 0);
        assert!(
            retained_total < original_total / 2,
            "completed runs must retire workspace: retained={retained_total}, original={original_total}"
        );
        assert_eq!(token_ids(&outputs_a), ids_a);

        let mut full_prompt = prompt_a;
        full_prompt.extend_from_slice(&ids_a);
        full_prompt.extend([6, 7]);
        assert_eq!(ids_b, reference(&stream, full_prompt, controlled));
        eprintln!(
            "quoted prefix reuse: controlled={controlled}, A={ids_a:?}, B={ids_b:?}, \
             cached=7, final=12, published={published_bytes}, \
             original={charge_a}+{charge_b}, retained_after_A={retained_a}, \
             retained_after_B={retained_total}, live_domain={live_b}"
        );

        // No diagnostic request clone may mask the escaped token's ownership.
        drop(request_a);
        let escaped_a = live_storage_bytes(None, &[&outputs_a]);
        assert!(escaped_a > 0 && escaped_a < charge_a);
        let escaped_total =
            live_storage_bytes_with_arrays(None, &[&outputs_a], &[&escaped_decoder]);
        assert_eq!(
            escaped_total,
            escaped_a + escaped_decoder_info.bytes() as u64
        );
        drop((outputs_b, runtime));
        settle(&pool, escaped_total);
        assert_eq!(token_ids(&outputs_a), ids_a);
        drop(outputs_a);
        settle(&pool, escaped_decoder_info.bytes() as u64);
        assert_eq!(
            escaped_decoder.allocation_info().unwrap(),
            Some(escaped_decoder_info)
        );
        drop(escaped_decoder);
        settle(&pool, 0);
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

struct Unchanged {
    paths: paths::Counts,
    inputs: usize,
    resets: usize,
    frontier: Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>,
    revision: InferenceStateRevision,
    bytes: u64,
    peak: u64,
}

impl Unchanged {
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

    fn assert_no_work(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        pool: &WorkingMemoryPool,
        controller: &Controller,
    ) {
        assert_eq!(controller.0.get(), (0, 0, 0));
        assert_eq!(paths::snapshot(), self.paths);
        assert_eq!(paths::session_input_creation_attempts(), self.inputs);
        assert_eq!(paths::session_reset_attempts(), self.resets);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            self.frontier
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
            &self.revision
        );
        assert_eq!(pool.used_bytes().unwrap(), self.bytes);
        assert_eq!(pool.peak_bytes().unwrap(), self.peak);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}

#[test]
fn advancing_one_admitted_reuse_invalidates_the_other_before_controller_or_native_work() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let outputs_a = generate(&mut runtime, vec![1, 2, 3, 4, 5], true, false);
    let ids_a = token_ids(&outputs_a);
    assert_frontier(&runtime, 7);
    settle(&pool, live_storage_bytes(Some(&runtime), &[&outputs_a]));
    let before_preparation = Unchanged::capture(&runtime, &pool);
    let controller_first = Controller::default();
    let controller_stale = Controller::default();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut first = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![ids_a[2], 6, 7]),
            config(true),
            controller_first.clone(),
        )
        .unwrap();
    let mut stale = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![ids_a[2], 6, 7]),
            config(true),
            controller_stale.clone(),
        )
        .unwrap();
    assert_frontier(driver.runtime(), 7);
    assert_eq!(controller_first.0.get(), (0, 0, 0));
    assert_eq!(controller_stale.0.get(), (0, 0, 0));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert!(pool.used_bytes().unwrap() > before_preparation.bytes);
    assert_eq!(
        driver
            .runtime()
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap()
            .revision(),
        &before_preparation.revision
    );

    let mut outputs_b = vec![driver.advance(&mut first).unwrap().unwrap().into_output()];
    assert!(driver.take_completed_step(&mut first).unwrap().is_none());
    assert_frontier(driver.runtime(), 10);
    assert_eq!(controller_first.0.get().0, 1);
    assert_eq!(controller_first.0.get().1, 1);
    let before_rejection = Unchanged::capture(driver.runtime(), &pool);
    assert_ne!(before_rejection.revision, before_preparation.revision);
    let error = driver.advance(&mut stale).err().unwrap();
    assert!(
        matches!(
            cause::<WorkingMemoryError>(&error),
            Some(WorkingMemoryError::IdentityMismatch)
                | Some(WorkingMemoryError::StateFrontierMismatch {
                    expected: 7,
                    actual: 10
                })
        ),
        "stale opening must reject its branch or exact frontier: {error}"
    );
    before_rejection.assert_no_work(driver.runtime(), &pool, &controller_stale);

    // Rejecting the competing continuation must not fence or mutate the run
    // that actually advanced this native session.
    for _ in 0..2 {
        outputs_b.push(driver.advance(&mut first).unwrap().unwrap().into_output());
        assert!(driver.take_completed_step(&mut first).unwrap().is_none());
    }
    assert!(driver.advance(&mut first).unwrap().is_none());
    assert_frontier(driver.runtime(), 12);
    assert_eq!(controller_first.0.get().0, 3);
    assert_eq!(controller_first.0.get().1, 3);
    assert_eq!(token_ids(&outputs_a), ids_a);
    let ids_b = token_ids(&outputs_b);
    drop((first, stale, driver));
    settle(
        &pool,
        live_storage_bytes(Some(&runtime), &[&outputs_a, &outputs_b]),
    );
    let mut full_prompt = vec![1, 2, 3, 4, 5];
    full_prompt.extend_from_slice(&ids_a);
    full_prompt.extend([6, 7]);
    assert_eq!(ids_b, reference(&stream, full_prompt, true));
    drop((outputs_b, outputs_a, runtime));
    settle(&pool, 0);
}

#[test]
fn cached_reuse_rejects_one_byte_short_and_runs_at_the_exact_live_domain_capacity() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let published_bytes = pool.used_bytes().unwrap();
    let outputs_a = generate(&mut runtime, vec![1, 2, 3, 4, 5], true, true);
    let ids_a = token_ids(&outputs_a);
    assert_frontier(&runtime, 7);
    let charge_a = {
        let retained = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap();
        retained
            .admission()
            .unwrap()
            .request()
            .memory_reservation()
            .unwrap()
            .bytes()
    };
    let live_a = live_storage_bytes(Some(&runtime), &[&outputs_a]);
    settle(&pool, live_a);
    assert!(live_a.checked_sub(published_bytes).unwrap() < charge_a / 2);

    let prompt_b = vec![ids_a[2], 6, 7];
    let controller = Controller::default();
    let before_probe = Unchanged::capture(&runtime, &pool);
    let probe = MlxBackend::admit_text_preparation(
        &runtime,
        &eredu_core::TextPreparationInput::TokenIds {
            positions: prompt_b.len() as u64,
            capacity_bytes: (prompt_b.capacity() * std::mem::size_of::<u32>()) as u64,
        },
        config(true),
        &controller,
    )
    .unwrap();
    let request_b = probe.request.as_ref().unwrap().request();
    assert_eq!(request_b.geometry().cached_positions, 7);
    assert_eq!(request_b.geometry().input_positions, 3);
    assert_eq!(request_b.geometry().prefill_chunk_positions, 1);
    let charge_b = request_b.memory_reservation().unwrap().bytes();
    assert!(charge_b > 0);
    let before_full_quote = Unchanged::capture(&runtime, &pool);
    let (full_quote, _) = super::text_quote::quote(
        runtime.session(),
        request_b.geometry(),
        (prompt_b.capacity() * std::mem::size_of::<u32>()) as u64,
        config(true),
        controller.inference_workspace(3).unwrap(),
    )
    .unwrap();
    let full_charge_b = full_quote
        .requested_state_bytes
        .checked_add(
            full_quote
                .execution_workspace
                .as_ref()
                .unwrap()
                .peak_bytes()
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    assert!(
        charge_b < full_charge_b,
        "the actual successor must reserve only residual demand: residual={charge_b}, full={full_charge_b}"
    );
    before_full_quote.assert_no_work(&runtime, &pool, &controller);
    let exact_capacity = before_probe.bytes.checked_add(charge_b).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), exact_capacity);
    drop(probe);
    settle(&pool, before_probe.bytes);
    assert_eq!(controller.0.get(), (0, 0, 0));
    assert_eq!(paths::snapshot(), before_probe.paths);
    assert_eq!(
        paths::session_input_creation_attempts(),
        before_probe.inputs
    );
    assert_eq!(paths::session_reset_attempts(), before_probe.resets);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        before_probe.frontier
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
        &before_probe.revision
    );
    assert_eq!(
        pool.peak_bytes().unwrap(),
        before_probe.peak.max(exact_capacity)
    );

    let config_at = |capacity| {
        config(true).with_inference_policy(eredu_core::TextInferencePolicy {
            // A larger chunk could shrink and legitimately fit a lower limit.
            prefill_chunk_positions: std::num::NonZeroU64::new(1),
            managed_memory_capacity_bytes: Some(capacity),
            submission_tracking_capacity_bytes: None,
            graph_metadata_capacity_bytes: None,
        })
    };
    let before_rejection = Unchanged::capture(&runtime, &pool);
    {
        let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
            "cached request must be admitted before sampler construction".into(),
        ));
        let error = ControlledTextGeneration::from_input(
            &mut runtime,
            TextGenerationInput::TokenIds(prompt_b.clone()),
            config_at(exact_capacity - 1),
            controller.clone(),
        )
        .err()
        .unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::BudgetExceeded {
                required_bytes: charge_b,
                available_bytes: charge_b - 1,
            }),
            "one-byte-short rejection must include actual retained physical storage: {error}"
        );
        assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
        before_rejection.assert_no_work(&runtime, &pool, &controller);
    }

    let generation = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(prompt_b),
        config_at(exact_capacity),
        controller.clone(),
    )
    .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), exact_capacity);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let outputs_b = generation
        .map(|token| token.unwrap().into_output())
        .collect::<Vec<_>>();
    assert_eq!(outputs_b.len(), 3);
    assert_eq!(controller.0.get().0, 3);
    assert_eq!(controller.0.get().1, 3);
    let ids_b = token_ids(&outputs_b);
    assert_frontier(&runtime, 12);
    let live_b = live_storage_bytes(Some(&runtime), &[&outputs_a, &outputs_b]);
    settle(&pool, live_b);
    assert!(live_b < exact_capacity);
    assert!(
        live_b.checked_sub(published_bytes).unwrap() < charge_a.checked_add(charge_b).unwrap() / 2
    );
    assert_eq!(
        pool.peak_bytes().unwrap(),
        before_probe.peak.max(exact_capacity)
    );
    let mut full_prompt = vec![1, 2, 3, 4, 5];
    full_prompt.extend_from_slice(&ids_a);
    full_prompt.extend([6, 7]);
    assert_eq!(ids_b, reference(&stream, full_prompt, true));
    eprintln!(
        "finite cached reuse: published={published_bytes}, original_A={charge_a}, \
         residual_B={charge_b}, full_B={full_charge_b}, live_after_A={live_a}, exact_capacity={exact_capacity}, \
         live_after_B={live_b}, outputs={ids_b:?}"
    );

    let escaped_a = live_storage_bytes(None, &[&outputs_a]);
    assert!(escaped_a > 0 && escaped_a < charge_a);
    drop((outputs_b, runtime));
    settle(&pool, escaped_a);
    assert_eq!(token_ids(&outputs_a), ids_a);
    drop(outputs_a);
    settle(&pool, 0);
}

#[test]
fn original_recipe_binds_empty_initial_state_and_rejects_foreign_populated_roots() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let geometry = eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 3,
        prefill_chunk_positions: 1,
        output: eredu_core::OutputDemand::LastPosition,
    };
    assert!(runtime
        .session()
        .payload
        .model
        .erased()
        .retained_inference_authority()
        .unwrap()
        .admission()
        .is_none());
    let before = pool.used_bytes().unwrap();
    let paths_before = paths::snapshot();
    let (quote, roots, recipe) = runtime
        .session()
        .payload
        .model
        .quote_registered_resident_text_with_sampling_recipe(
            geometry,
            config(false),
            &TokenFilter::All,
            &pool,
        )
        .unwrap();
    assert!(roots.borrowed_storage().roots().is_empty());
    assert_eq!(roots.borrowed_storage().total_bytes(), 0);
    assert!(roots.pool().same_domain(&pool));
    assert!(recipe
        .plan()
        .same_plan(quote.equations.span_workspace_plan()));
    assert!(!recipe.records().is_empty());
    assert_eq!(recipe.sampling_records().len(), 4);
    assert_eq!(paths::snapshot(), paths_before);
    assert_eq!(pool.used_bytes().unwrap(), before);
    drop((quote, roots, recipe));

    // Populate the actual nonzero decoder through the shared ordinary driver.
    let outputs = generate(&mut runtime, vec![2, 5, 7], false, false);
    assert_frontier(&runtime, 5);
    let geometry = eredu_core::InferenceGeometry {
        cached_positions: 5,
        input_positions: 1,
        ..geometry
    };
    let before = pool.used_bytes().unwrap();
    let paths_before = paths::snapshot();
    let error = runtime
        .session()
        .payload
        .model
        .quote_registered_resident_text_with_sampling_recipe(
            geometry,
            config(false),
            &TokenFilter::All,
            &foreign,
        )
        .err()
        .expect("nonempty decoder roots require their canonical pool");
    assert!(matches!(error, Error::Other(ref cause)
        if cause.downcast_ref::<WorkingMemoryError>() == Some(&WorkingMemoryError::IdentityMismatch)));
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert_eq!(paths::snapshot(), paths_before);
    // Ordinary unquoted execution does not publish canonical decoder rows,
    // even when its live arrays are associated with this same runtime pool.
    let error = runtime
        .session()
        .payload
        .model
        .quote_registered_resident_text_with_sampling_recipe(
            geometry,
            config(false),
            &TokenFilter::All,
            &pool,
        )
        .err()
        .expect("pool identity alone cannot certify unregistered decoder roots");
    assert!(matches!(error, Error::Other(ref cause)
        if cause.downcast_ref::<WorkingMemoryError>() == Some(&WorkingMemoryError::IdentityMismatch)));
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert_eq!(paths::snapshot(), paths_before);
    drop((outputs, runtime));
    settle(&pool, 0);
}
