use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use eredu_core::{
    ControlledTextGeneration, TextGeneration, TextGenerationInput, TextPreparationInput,
};
use eredu_runtime::working_memory::{MemoryLedger, WorkingMemoryError};

struct AllTokens;
impl eredu_core::TokenFilterController for AllTokens {
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

struct UnquotedController;
impl eredu_core::TokenFilterController for UnquotedController {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("unquoted controller must reject before decisions")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("unquoted controller must reject before commitment")
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("unquoted controller must reject before completion checks")
    }
}

fn config(capacity: Option<u64>) -> TextGenerationConfig {
    config_for(capacity, 2)
}

fn config_for(capacity: Option<u64>, outputs: usize) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(outputs),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(19)
    .with_inference_policy(eredu_core::TextInferencePolicy {
        // Exact-capacity rejection uses the smallest admissible chunk; a
        // larger initial chunk could legitimately shrink to fit capacity - 1.
        prefill_chunk_positions: std::num::NonZeroU64::new(1),
        memory_limits: (capacity).map_or_else(
            eredu_core::MemoryLimitDeclarations::unlimited,
            |bytes| {
                eredu_core::MemoryLimitDeclarations::new([(
                    "host".into(),
                    eredu_core::MemoryLimit::Finite(bytes),
                )])
            },
        ),
        submission_tracking_capacity_bytes: None,
        graph_metadata_capacity_bytes: None,
    })
}

fn tokens() -> Vec<u32> {
    vec![1, 2, 3, 4, 5]
}

fn evidence() -> TextPreparationInput<'static, MlxModelInput> {
    TextPreparationInput::TokenIds {
        positions: 5,
        capacity_bytes: 20,
    }
}

fn runtime<'a>(
    stream: &'a Stream,
    pool: &MemoryLedger,
) -> (ModelRuntime<MlxBackend<'a>>, tempfile::TempDir) {
    let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(stream, &weights).with_memory_ledger(pool.clone());
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model = eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
        .unwrap();
    (
        ModelRuntime::from_prepared(backend, model).unwrap(),
        artifact,
    )
}

fn memory_error<'a>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a WorkingMemoryError> {
    let mut current = error;
    loop {
        if let Some(memory) = current.downcast_ref::<WorkingMemoryError>() {
            return Some(memory);
        }
        current = current.source()?;
    }
}

fn domain_error<'a>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a eredu_core::MemoryDomainError> {
    let mut current = error;
    loop {
        if let Some(WorkingMemoryError::Domain(domain)) =
            current.downcast_ref::<WorkingMemoryError>()
        {
            return Some(domain);
        }
        if let Some(eredu_core::HostMetadataFundingError::Domain(domain)) =
            current.downcast_ref::<eredu_core::HostMetadataFundingError>()
        {
            return Some(domain);
        }
        current = current.source()?;
    }
}

fn drain() {
    safemlx::memory::clear_cache().unwrap();
    MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
    crate::backend::ordinary_retirement::reclaim_all();
}

fn settle(pool: &MemoryLedger, bytes: u64) {
    let expected = if bytes == 0 {
        pool.snapshot()
            .unwrap()
            .domains
            .iter()
            .find(|domain| domain.domain == pool.topology().host_domain())
            .unwrap()
            .fixed_baseline
            .total()
            .unwrap()
    } else {
        bytes
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        drain();
        let actual = pool.fixture_host_current().unwrap();
        if actual == expected {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "terminal charge did not retire: actual={actual}, expected={expected}, snapshot={:?}",
            pool.snapshot().unwrap()
        );
        std::thread::yield_now();
    }
}

struct PublicCharge {
    baseline: u64,
    admission_peak: u64,
    retained_total: u64,
    preparation_bytes: u64,
    reservation_bytes: u64,
}

// Measure a real successful public admission; no synthetic state or enclosing
// workspace estimate enters these fixtures. Admission includes temporary diagnostic
// owners that retire before the returned preparation; its peak and retained
// charge are distinct. The one-byte refusal verifies the peak is applicable to
// the actual next admission rather than an unrelated historical high-water mark.
// The source Vec's actual capacity is
// the same twenty bytes used by ordinary/controlled preparation below.
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_charge(runtime: &ModelRuntime<MlxBackend<'_>>, pool: &MemoryLedger) -> PublicCharge {
    public_charge_for(runtime, pool, config(Some(u64::MAX)))
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_charge_for(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    pool: &MemoryLedger,
    config: TextGenerationConfig,
) -> PublicCharge {
    drain();
    let baseline = pool.fixture_host_current().unwrap();
    let preparation =
        MlxBackend::admit_text_preparation(runtime, &evidence(), config, &AllTokens).unwrap();
    let request = preparation.request.as_ref().unwrap().request();
    let bytes = request
        .memory_reservation()
        .requirements()
        .get(crate::memory_fixture::topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    assert!(bytes > 0);
    let retained_total = pool.fixture_host_current().unwrap();
    let admission_peak = pool.snapshot().unwrap().domains[0].historical_peak_bytes;
    assert!(admission_peak >= retained_total);
    assert!(
        retained_total > bytes,
        "resident source and parameter storage is charged separately"
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let preparation_bytes = retained_total.checked_sub(baseline).unwrap();
    assert!(
        preparation_bytes >= bytes,
        "the complete preparation also retains source and planning metadata"
    );
    drop(preparation);
    settle(pool, baseline);
    PublicCharge {
        baseline,
        admission_peak,
        retained_total,
        preparation_bytes,
        reservation_bytes: bytes,
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn ordinary_v4_publication_runs_at_its_complete_quote_and_rejects_one_byte_short() {
    if !crate::tests::support::native_process::enter("ordinary-private-ledger") {
        return;
    }
    ordinary_v4_publication_contract(false);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn ordinary_cpu_text_keeps_finite_and_unlimited_controlled_parity() {
    if !crate::tests::support::native_process::enter("ordinary-cpu-text-parity") {
        return;
    }
    ordinary_text_parity(safemlx::DeviceType::Cpu);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn ordinary_metal_text_keeps_finite_and_unlimited_controlled_parity() {
    if !crate::tests::support::native_process::enter("ordinary-metal-text-parity") {
        return;
    }
    ordinary_text_parity(safemlx::DeviceType::Gpu);
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn ordinary_text_parity(device: safemlx::DeviceType) {
    let startup = crate::tests::support::test_utils::initialize_original_sources();
    let default_stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
    let selected_cpu = (device == safemlx::DeviceType::Cpu).then(|| {
        let choice = crate::backend::nn::workspace::MlxCpuMatmulMechanism::select(
            eredu_nn::CpuMatmulImplementation::Float32Tiles,
        ).unwrap();
        crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_cpu_factory_with_matmul(
            &startup, choice,
        ).unwrap().unwrap()
    });
    let stream = selected_cpu
        .as_ref()
        .map_or(&default_stream, |source| source.execution());
    let make_config = |capacity| {
        TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    temperature: Some(0.0),
                    max_new_tokens: Some(3),
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .with_seed(19)
        .with_inference_policy(config_for(capacity, 3).inference_policy().clone())
    };
    let mut sequences = Vec::new();
    for finite in [false, true] {
        for controlled in [false, true] {
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            // The Llama fixture has deterministic nonzero weights.
            let (mut runtime, _artifact) = runtime(&stream, &pool);
            let charge = public_charge_for(&runtime, &pool, make_config(Some(u64::MAX)));
            let before = runtime.session().payload.model.erased().state_snapshot();
            let refusal = MlxBackend::admit_text_preparation(
                &runtime,
                &evidence(),
                make_config(Some(charge.admission_peak - 1)),
                &AllTokens,
            )
            .unwrap_err();
            assert!(matches!(
                memory_error(&refusal),
                Some(WorkingMemoryError::Domain(
                    eredu_core::MemoryDomainError::BudgetExceeded { .. }
                ))
            ));
            drop(refusal);
            settle(&pool, charge.baseline);
            assert_eq!(
                runtime.session().payload.model.erased().state_snapshot(),
                before
            );
            let config = make_config(finite.then_some(charge.admission_peak));
            let outputs = if controlled {
                ControlledTextGeneration::from_input(
                    &mut runtime,
                    TextGenerationInput::TokenIds(tokens()),
                    config,
                    AllTokens,
                )
                .unwrap()
                .map(|item| item.unwrap().into_output())
                .collect::<Vec<_>>()
            } else {
                TextGeneration::new(&mut runtime, tokens(), config)
                    .unwrap()
                    .map(Result::unwrap)
                    .collect::<Vec<_>>()
            };
            assert_eq!(outputs.len(), 3);
            sequences.push(
                outputs
                    .iter()
                    .map(|token| token.token_id().unwrap())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            if finite {
                assert!(pool.fixture_host_current().unwrap() <= charge.admission_peak);
                assert!(
                    pool.snapshot().unwrap().domains[0].historical_peak_bytes
                        <= charge.admission_peak
                );
            }
            drop(runtime);
            assert!(outputs.iter().all(|token| token.token_id().is_ok()));
            drop(outputs);
            settle(&pool, 0);
        }
    }
    assert!(sequences.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn ordinary_v4_allocator_birth_uses_the_same_process_ledger_allowance() {
    if !crate::tests::support::native_process::enter("ordinary-assigned-allocator") {
        return;
    }
    ordinary_v4_publication_contract(true);
}
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn ordinary_v4_publication_contract(process_ledger: bool) {
    // Backend readiness owns real native streams before ordinary request
    // admission. The borrowed adapter below has no original inference source.
    let startup = crate::tests::support::test_utils::initialize_original_sources();
    let streams =
        crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_factory(&startup)
            .unwrap()
            .unwrap();
    let stream = streams.execution();
    let weights = streams.source();
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_deepseek_v4_fixture(artifact.path(), 0);
    let process_baseline = if process_ledger {
        // Session readiness owns the shared kernel sources independently of
        // requests. Establish that real baseline through the public load path.
        let pool = crate::backend::managed_memory::ledger();
        let backend = MlxBackend::new(stream, weights);
        let model =
            eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
                .unwrap();
        let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        drop(runtime);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            drain();
            pool.unquoted_owner_count().unwrap() == 0
        });
        let mut previous = None;
        let mut baseline = 0;
        crate::backend::submission_recovery::wait_for_retirement(|| {
            drain();
            baseline = pool.fixture_host_current().unwrap();
            let settled = previous == Some(baseline);
            previous = Some(baseline);
            settled
        });
        Some(baseline)
    } else {
        None
    };
    let mut sequences = Vec::new();
    for controlled in [false, true] {
        let pool = if process_ledger {
            crate::backend::managed_memory::ledger().clone()
        } else {
            crate::memory_fixture::ledger(u64::MAX, 0).unwrap()
        };
        let backend = MlxBackend::new(stream, weights).with_memory_ledger(pool.clone());
        let model =
            eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
                .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let charge = public_charge_for(&runtime, &pool, config_for(Some(u64::MAX), 4));
        let before = runtime.session().payload.model.erased().state_snapshot();
        let rejected = MlxBackend::admit_text_preparation(
            &runtime,
            &evidence(),
            config_for(Some(charge.admission_peak - 1), 4),
            &AllTokens,
        )
        .unwrap_err();
        assert!(matches!(
            memory_error(&rejected),
            Some(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        drop(rejected);
        settle(&pool, charge.baseline);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before
        );
        let config = config_for(Some(charge.admission_peak), 4);
        let outputs = if controlled {
            ControlledTextGeneration::from_input(
                &mut runtime,
                TextGenerationInput::TokenIds(tokens()),
                config,
                AllTokens,
            )
            .unwrap()
            .map(|item| item.unwrap().into_output())
            .collect::<Vec<_>>()
        } else {
            TextGeneration::new(&mut runtime, tokens(), config)
                .unwrap()
                .map(Result::unwrap)
                .collect::<Vec<_>>()
        };
        assert_eq!(outputs.len(), 4);
        sequences.push(
            outputs
                .iter()
                .map(|value| value.token_id().unwrap())
                .collect::<Vec<_>>(),
        );
        assert!(pool.fixture_host_current().unwrap() <= charge.admission_peak);
        assert!(pool.snapshot().unwrap().domains[0].historical_peak_bytes <= charge.admission_peak);
        drop(runtime);
        assert!(outputs.iter().all(|value| value.token_id().is_ok()));
        drop(outputs);
        settle(&pool, process_baseline.unwrap_or(0));
    }
    assert_eq!(sequences[0], sequences[1]);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_resident_preparation_succeeds_at_its_exact_complete_domain_capacity() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = runtime(&stream, &pool);
    let before = runtime.session().payload.model.erased().state_snapshot();
    let charge = public_charge(&runtime, &pool);
    let capacity = charge.admission_peak;
    let request_bytes = charge.reservation_bytes;
    let preparation = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap();
    assert_eq!(pool.fixture_host_current().unwrap(), charge.retained_total);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].historical_peak_bytes,
        capacity
    );
    assert_eq!(
        preparation
            .request
            .as_ref()
            .unwrap()
            .request()
            .memory_reservation()
            .requirements()
            .get(crate::memory_fixture::topology().host_domain())
            .unwrap()
            .total()
            .unwrap(),
        request_bytes
    );
    let prompt =
        MlxBackend::prepare_text_prompt_admitted(runtime.backend(), tokens(), &preparation)
            .unwrap();
    let prompt =
        MlxBackend::bind_text_prompt_preparation(runtime.backend(), prompt, &preparation).unwrap();
    let sampler = MlxBackend::start_text_generation_admitted(
        runtime.backend(),
        config(Some(capacity)),
        &preparation,
    )
    .unwrap();
    assert_eq!(pool.fixture_host_current().unwrap(), charge.retained_total);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].historical_peak_bytes,
        capacity
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        before
    );
    drop((sampler, prompt, preparation));
    settle(&pool, charge.baseline);
    drop(runtime);
    settle(&pool, 0);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_resident_one_byte_short_rejects_before_prompt_sampling_or_state_mutation() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let capacity = public_charge(&runtime, &pool).admission_peak;
    let before = runtime.session().payload.model.erased().state_snapshot();
    let usage = pool.fixture_host_current().unwrap();
    let peak = pool.snapshot().unwrap().domains[0].historical_peak_bytes;
    let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "sampling must follow admission".into(),
    ));
    let error = TextGeneration::new(&mut runtime, tokens(), config(Some(capacity - 1)))
        .err()
        .unwrap();
    assert!(
        matches!(
            memory_error(&error),
            Some(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        ),
        "{error}"
    );
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        before
    );
    // Escaping diagnostics keep their paying metadata until the error retires.
    assert!(pool.fixture_host_current().unwrap() >= usage);
    drop(error);
    settle(&pool, usage);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].historical_peak_bytes,
        peak
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(runtime);
    settle(&pool, 0);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_preparations_share_one_domain_capacity_and_release_only_their_own_reservations() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = runtime(&stream, &pool);
    let charge = public_charge(&runtime, &pool);
    let single = charge.retained_total;
    let baseline = charge.baseline;
    let capacity = single
        .checked_add(charge.admission_peak.checked_sub(baseline).unwrap())
        .unwrap();
    let retained_pair = single.checked_add(charge.preparation_bytes).unwrap();
    let first = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap();
    let second = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap();
    assert_eq!(pool.fixture_host_current().unwrap(), retained_pair);
    let rejected = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap_err();
    assert!(
        matches!(
            domain_error(&rejected),
            Some(eredu_core::MemoryDomainError::BudgetExceeded {
                domain, limit_bytes, existing_bytes, requested_bytes,
            }) if *domain == pool.topology().host_domain()
                && *limit_bytes == capacity
                && *existing_bytes >= retained_pair
                && existing_bytes.checked_add(*requested_bytes).unwrap() > capacity
        ),
        "{rejected}"
    );
    assert!(pool.fixture_host_current().unwrap() >= retained_pair);
    drop(rejected);
    settle(&pool, retained_pair);
    drop(first);
    settle(&pool, single);
    let replacement = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap();
    assert_eq!(pool.fixture_host_current().unwrap(), retained_pair);
    drop(second);
    settle(&pool, single);
    drop(replacement);
    settle(&pool, baseline);
    drop(runtime);
    settle(&pool, 0);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_managed_generation_matches_ordinary_and_controlled_unlimited_tokens() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let mut sequences = Vec::new();
    for managed in [false, true] {
        for controlled in [false, true] {
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let (mut runtime, _artifact) = runtime(&stream, &pool);
            let capacity = managed.then(|| public_charge(&runtime, &pool).admission_peak);
            let config = config(capacity);
            let tokens = if controlled {
                ControlledTextGeneration::from_input(
                    &mut runtime,
                    TextGenerationInput::TokenIds(tokens()),
                    config,
                    AllTokens,
                )
                .unwrap()
                .map(|token| token.unwrap().into_output())
                .collect::<Vec<_>>()
            } else {
                TextGeneration::new(&mut runtime, tokens(), config)
                    .unwrap()
                    .map(Result::unwrap)
                    .collect::<Vec<_>>()
            };
            assert_eq!(tokens.len(), 2);
            sequences.push(
                tokens
                    .iter()
                    .map(|token| token.token_id().unwrap())
                    .collect::<Vec<_>>(),
            );
            if let Some(capacity) = capacity {
                assert!(pool.fixture_host_current().unwrap() <= capacity);
                assert!(pool.snapshot().unwrap().domains[0].historical_peak_bytes <= capacity);
                assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            }
            drop((tokens, runtime));
            settle(&pool, 0);
        }
    }
    assert!(sequences.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn public_managed_admission_rejects_unproven_controller_before_native_preparation() {
    for device in [safemlx::DeviceType::Cpu, safemlx::DeviceType::Gpu] {
        if device == safemlx::DeviceType::Gpu
            && !cfg!(all(
                target_vendor = "apple",
                feature = "metal",
                not(feature = "cuda")
            ))
        {
            continue;
        }
        let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let (runtime, _artifact) = runtime(&stream, &pool);
        let before = runtime.session().payload.model.erased().state_snapshot();
        let usage = pool.fixture_host_current().unwrap();
        let owners = pool.unquoted_owner_count().unwrap();
        let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
            "unproven preparation must not reach sampling".into(),
        ));
        let controller = MlxBackend::admit_text_preparation(
            &runtime,
            &evidence(),
            config(Some(u64::MAX)),
            &UnquotedController,
        )
        .unwrap_err();
        assert_eq!(
            memory_error(&controller),
            Some(&WorkingMemoryError::UnknownBound)
        );
        let admitted = MlxBackend::admit_text_preparation(
            &runtime,
            &evidence(),
            config(Some(u64::MAX)),
            &AllTokens,
        );
        if device == safemlx::DeviceType::Cpu {
            // This standalone source lacks the canonical prepared CPU facts.
            // The complete source-qualified CPU provider is exercised separately.
            assert_eq!(
                memory_error(&admitted.unwrap_err()),
                Some(&WorkingMemoryError::UnknownBound)
            );
        } else {
            let admitted = admitted.unwrap();
            assert!(admitted.request.is_some());
            drop(admitted);
        }
        settle(&pool, usage);
        assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before
        );
        assert_eq!(pool.fixture_host_current().unwrap(), usage);
        assert_eq!(pool.unquoted_owner_count().unwrap(), owners);
        drop(runtime);
        settle(&pool, 0);
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
