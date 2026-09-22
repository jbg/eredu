use super::*;
use crate::composition::mlx::session::paired_copy_fixture as copy;
use crate::tests::support::original_snapshot::NativeSnapshotProvider;
use eredu_core::execution_control::SnapshotLimits;
use eredu_evaluation::execution_control::ContinuationSnapshotProvider;
use eredu_runtime::execution_control::{SnapshotBudget, TextSnapshotBackend};

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
fn budget() -> SnapshotBudget {
    SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 16,
        max_branches: 16,
        retained_bytes: 1_000_000_000,
        cumulative_copy_bytes: 1_000_000_000,
    })
}

#[test]
fn canonical_pair_exact_limit_rejection_independence_and_escaped_root_custody() {
    if !crate::tests::support::native_process::enter("canonical-paired-copy") {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture_with_values(root.path(), fixture_value);
    let mut runtime = load(root.path());
    let pool = runtime.backend().memory_ledger().clone();
    let layout = copy::installed_layout(&runtime);
    let config = eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(20),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(47);
    let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
        NativeSnapshotProvider,
        crate::backend::error::Error,
        crate::backend::error::Error,
    >()
    .unwrap();
    let mut state = eredu_core::ControlledTextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        eredu_core::TokenIdsInputPlan::new(&[1, 2]).unwrap(),
        config.clone(),
        SnapshotController::default(),
        None,
        eredu_core::GenerationSequenceRequest::new(20, &[]).with_consumer(&consumer),
    )
    .unwrap();
    let sequence = state
        .take_prepared_sequence()
        .unwrap()
        .prepare_storage()
        .unwrap();
    let mut provider = NativeSnapshotProvider::new(sequence, config, pool.clone());
    use eredu_core::TokenOutput;
    let token = state.next().unwrap().unwrap();
    provider.observe_token(token.token_id());
    drop(token);
    let _observation = copy::observe();
    let budget = budget();
    let first = provider
        .capture(&mut state.snapshot_source().unwrap(), &budget, Some(4096))
        .unwrap();
    let first_ids = copy::identities();
    let first_layout = copy::take_layout();
    assert!(layout.same_storage(&first_layout));
    let first_alias = copy::take_alias();
    let key = first_alias
        .evaluated()
        .unwrap()
        .try_to_vec::<u32>()
        .unwrap();
    let copies = copy::copies();
    copy::limit(copy::Limit::OneShort);
    let error = provider
        .capture(&mut state.snapshot_source().unwrap(), &budget, Some(4096))
        .err()
        .unwrap();
    assert!(matches!(
        cause::<eredu_runtime::working_memory::WorkingMemoryError>(&error),
        Some(eredu_runtime::working_memory::WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(
        copy::copies(),
        copies,
        "rejected copy must not reach a native key copy"
    );
    drop(error);
    copy::limit(copy::Limit::Exact);
    let second = provider
        .capture(&mut state.snapshot_source().unwrap(), &budget, Some(4096))
        .unwrap();
    let second_ids = copy::identities();
    let second_layout = copy::take_layout();
    assert!(layout.same_storage(&second_layout));
    let escaped = copy::take_alias();
    assert_eq!(
        key,
        escaped.evaluated().unwrap().try_to_vec::<u32>().unwrap()
    );
    assert!(!first_ids.is_empty());
    assert!(
        first_ids.iter().all(|id| !second_ids.contains(id)),
        "destination backings must be independent"
    );
    assert_eq!(copy::copies(), copies + 1);
    copy::limit(copy::Limit::Unlimited);
    drop((first, second, first_alias, state, provider));
    safemlx::memory::clear_cache();
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        key,
        escaped.evaluated().unwrap().try_to_vec::<u32>().unwrap()
    );
    let retained = pool.snapshot().unwrap().domains[0].current_charge_bytes;
    drop(escaped);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::memory::clear_cache();
        safemlx::reclaim_allocation_owners();
        pool.snapshot().unwrap().domains[0].current_charge_bytes < retained
    });
    runtime.reset().unwrap();
    assert!(layout.same_storage(&copy::installed_layout(&runtime)));
    let layout_bytes = layout.capacity_bytes().unwrap();
    drop((layout, runtime));
    assert!(first_layout.same_storage(&second_layout));
    assert_eq!(second_layout.capacity_bytes(), Some(layout_bytes));
    drop(first_layout);
    let source = eredu_runtime::SharedHostMetadata::Layout(second_layout.clone());
    let layout_key = crate::backend::runtime::residency::storage::StorageIdentity::HostMetadata(
        source.identity().registry_key().clone(),
    );
    drop(source);
    assert_eq!(
        pool.registered_allocation(&layout_key)
            .unwrap()
            .unwrap()
            .capacity_bytes(),
        layout_bytes
    );
    drop(second_layout);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::memory::clear_cache();
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.registered_allocation(&layout_key).unwrap().is_none()
    });
}

#[test]
fn canonical_pair_failure_after_key_copy_preserves_source_and_allows_settled_retry() {
    if !crate::tests::support::native_process::enter("canonical-paired-copy-failure") {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture_with_values(root.path(), fixture_value);
    let mut runtime = load(root.path());
    let pool = runtime.backend().memory_ledger().clone();
    let config = eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(20),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(47);
    let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
        NativeSnapshotProvider,
        crate::backend::error::Error,
        crate::backend::error::Error,
    >()
    .unwrap();
    let mut state = eredu_core::ControlledTextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        eredu_core::TokenIdsInputPlan::new(&[1, 2]).unwrap(),
        config.clone(),
        SnapshotController::default(),
        None,
        eredu_core::GenerationSequenceRequest::new(20, &[]).with_consumer(&consumer),
    )
    .unwrap();
    let sequence = state
        .take_prepared_sequence()
        .unwrap()
        .prepare_storage()
        .unwrap();
    let mut provider = NativeSnapshotProvider::new(sequence, config, pool);
    use eredu_core::TokenOutput;
    let token = state.next().unwrap().unwrap();
    provider.observe_token(token.token_id());
    drop(token);
    let prediction = Backend::sampling_prediction(Backend::sampling_state(
        state.snapshot_source().unwrap().parts().1,
    ));
    let copies = copy::copies();
    copy::fail_next_key_copy();
    let error = provider
        .capture(&mut state.snapshot_source().unwrap(), &budget(), Some(4096))
        .err()
        .unwrap();
    assert!(cause::<copy::InjectedPairedCopyFailure>(&error).is_some());
    assert_eq!(
        copy::copies(),
        copies + 1,
        "failure is after the actual key copy"
    );
    assert_eq!(
        Backend::sampling_prediction(Backend::sampling_state(
            state.snapshot_source().unwrap().parts().1
        )),
        prediction
    );
    drop(error);
    let saved = provider
        .capture(&mut state.snapshot_source().unwrap(), &budget(), Some(4096))
        .unwrap();
    assert_eq!(copy::copies(), copies + 2);
    drop(saved);
    let next = state.next().unwrap().unwrap();
    provider.observe_token(next.token_id());
    drop((next, state, provider));
    runtime.reset().unwrap();
}

fn resume_attempt(
    runtime: &mut eredu_core::ModelRuntime<Backend>,
    saved: &eredu_runtime::execution_control::TextContinuationSnapshot<Backend, SnapshotController>,
    host: &(eredu_core::RetainedGenerationSequence, f32),
    config: &eredu_core::TextGenerationConfig,
    cancellation: &eredu_core::GenerationCancellationToken,
) -> Result<
    Option<Vec<u32>>,
    eredu_runtime::execution_control::TextSnapshotError<crate::backend::error::Error>,
> {
    use eredu_core::TokenOutput;
    let mut sampling = config.sampling();
    sampling.max_new_tokens = saved.remaining_tokens();
    sampling.temperature = host.1;
    let config = eredu_core::TextGenerationConfig::new(sampling).with_seed(config.seed());
    let options =
        eredu_core::OriginalTextResumeOptions::new(eredu_core::OriginalTextResumeKind::Branch);
    let controls = saved.original_resume_preparation_bytes(runtime, config.clone(), &options)?;
    let producer = runtime.backend().memory_ledger()
        .prepare_generation_resume_host_copy::<NativeSnapshotProvider, crate::backend::error::Error, Backend, SnapshotController>(
            &host.0, eredu_core::MemoryLimits::unlimited(runtime.backend().memory_ledger().topology()), controls,
        ).map_err(eredu_runtime::execution_control::TextSnapshotError::HostAdmission)?;
    let generation = saved.resume_original_host_with_displaced(
        runtime,
        config,
        producer,
        cancellation,
        &options,
    )?;
    Ok(generation.map(|(mut generation, displaced, host)| {
        let values = generation
            .by_ref()
            .take(3)
            .map(|token| token.unwrap().token_id())
            .collect();
        drop((generation, displaced, host));
        values
    }))
}

fn resume_failure_case(
    point: crate::composition::mlx::session::resume_failure_fixture::Point,
    unwind: bool,
    cancel: bool,
) {
    use crate::composition::mlx::session::resume_failure_fixture as failure;
    use eredu_core::{GenerationCancellationToken, TokenOutput};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    if !crate::tests::support::native_process::enter("canonical-resume-failures") {
        return;
    }
    let artifact = tempfile::tempdir().unwrap();
    write_safetensors_fixture_with_values(artifact.path(), fixture_value);
    let mut runtime = load(artifact.path());
    let pool = runtime.backend().memory_ledger().clone();
    let config = eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(20),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(47);
    let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
        NativeSnapshotProvider,
        crate::backend::error::Error,
        crate::backend::error::Error,
    >()
    .unwrap();
    let mut state = eredu_core::ControlledTextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        eredu_core::TokenIdsInputPlan::new(&[1, 2]).unwrap(),
        config.clone(),
        SnapshotController::default(),
        None,
        eredu_core::GenerationSequenceRequest::new(20, &[]).with_consumer(&consumer),
    )
    .unwrap();
    let sequence = state
        .take_prepared_sequence()
        .unwrap()
        .prepare_storage()
        .unwrap();
    let mut provider = NativeSnapshotProvider::new(sequence, config.clone(), pool.clone());
    let first = state.next().unwrap().unwrap();
    provider.observe_token(first.token_id());
    drop(first);
    let (saved, host) = provider
        .capture(&mut state.snapshot_source().unwrap(), &budget(), Some(4096))
        .unwrap();
    drop(state);
    let baseline = resume_attempt(
        &mut runtime,
        &saved,
        &host,
        &config,
        &GenerationCancellationToken::new(),
    )
    .unwrap()
    .unwrap();
    let cancellation = GenerationCancellationToken::new();
    let action = if cancel {
        failure::Action::Cancel(cancellation.clone())
    } else if unwind {
        failure::Action::Unwind
    } else {
        failure::Action::Error
    };
    let guard = failure::fault(point, action);
    let result = catch_unwind(AssertUnwindSafe(|| {
        resume_attempt(&mut runtime, &saved, &host, &config, &cancellation)
    }));
    assert!(
        guard.reached(),
        "real resumed preparation boundary must be reached"
    );
    if unwind {
        assert_eq!(
            result
                .unwrap_err()
                .downcast_ref::<failure::InjectedResumeFailure>()
                .unwrap()
                .0,
            point
        );
    } else if cancel {
        assert!(result.unwrap().unwrap().is_none());
    } else {
        let error = result.unwrap().unwrap_err();
        assert_eq!(
            cause::<failure::InjectedResumeFailure>(&error).unwrap().0,
            point
        );
    }
    drop(guard);
    let installed = matches!(
        point,
        failure::Point::AfterExchange | failure::Point::BeforeSamplingReadiness
    );
    if installed {
        assert!(
            !failure::healthy(&runtime),
            "failed installed readiness must fence its target"
        );
        let escaped = failure::escaped_installed_array(&runtime);
        let identity = escaped.allocation_info().unwrap().unwrap().identity();
        drop((saved, host, provider, runtime));
        safemlx::reclaim_allocation_owners();
        assert_eq!(
            escaped.allocation_info().unwrap().unwrap().identity(),
            identity
        );
        let retained = pool.snapshot().unwrap().domains[0].current_charge_bytes;
        drop(escaped);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            safemlx::reclaim_allocation_owners();
            pool.snapshot().unwrap().domains[0].current_charge_bytes < retained
        });
    } else {
        assert!(
            failure::healthy(&runtime),
            "uninstalled settled failure must preserve the target"
        );
        let retried = resume_attempt(
            &mut runtime,
            &saved,
            &host,
            &config,
            &GenerationCancellationToken::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            retried, baseline,
            "retry must preserve saved RNG and controller history"
        );
        drop((saved, host, provider));
        runtime.reset().unwrap();
    }
}

#[test]
fn canonical_resume_cancel_before_install_preserves_custody() {
    resume_failure_case(
        crate::composition::mlx::session::resume_failure_fixture::Point::AfterPrompt,
        false,
        true,
    );
}

#[test]
fn canonical_resume_error_before_install_preserves_custody() {
    resume_failure_case(
        crate::composition::mlx::session::resume_failure_fixture::Point::AfterSampling,
        false,
        false,
    );
}

#[test]
fn canonical_resume_unwind_before_install_preserves_custody() {
    resume_failure_case(
        crate::composition::mlx::session::resume_failure_fixture::Point::AfterSampling,
        true,
        false,
    );
}

#[test]
fn canonical_resume_error_after_install_preserves_custody() {
    resume_failure_case(
        crate::composition::mlx::session::resume_failure_fixture::Point::AfterExchange,
        false,
        false,
    );
}

#[test]
fn canonical_resume_unwind_after_install_preserves_custody() {
    resume_failure_case(
        crate::composition::mlx::session::resume_failure_fixture::Point::AfterExchange,
        true,
        false,
    );
}

#[test]
fn canonical_resume_error_before_sampling_readiness_preserves_custody() {
    resume_failure_case(
        crate::composition::mlx::session::resume_failure_fixture::Point::BeforeSamplingReadiness,
        false,
        false,
    );
}
