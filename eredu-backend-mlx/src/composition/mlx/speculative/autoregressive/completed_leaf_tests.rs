//! The leaf binder consumes real completed target-cache sources. Test oracles
//! inspect settled arrays; they never create source identities or role grants.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::composition::mlx::speculative::sampling::numerical::native_tests::{
    admitted_backend, load, settle, source_configs, REQUEST_CEILING,
};
use crate::composition::mlx::{session::MlxModelSession, MlxPreparedInputMaterializer};
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_core::{
    GenerationCancellationToken, SpeculativeConfig, SpeculativePrefillOutcome,
    SpeculativeSchedulerOptions,
};
use eredu_runtime::speculative::autoregressive::AutoregressiveSchedulePlan;
use eredu_runtime::working_memory::WorkingMemoryError;
use std::num::{NonZeroU64, NonZeroUsize};

fn complete_cache(
    target: &mut MlxModelSession,
    pair: &AutoregressiveSourcePair,
    schedule: AutoregressiveSchedulePlan<'_>,
    prompt: &MlxModelInput,
    context: SpeculativeExecutionStreams<'_>,
) -> MlxAutoregressiveState {
    let mut invocation = None;
    schedule
        .domains()
        .iter()
        .find(|domain| domain.pass() == AutoregressivePass::TargetPrefill)
        .unwrap()
        .visit(|frontier, value| {
            assert_eq!(frontier, 0);
            assert_eq!(value.positions(), 2);
            invocation = Some(value);
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();
    let mut cursor = schedule.into_cursor();
    let claim = cursor.claim(0, invocation.unwrap()).unwrap();
    let cancellation = GenerationCancellationToken::new();
    let (completed, cache) = target
        .with_model_operation_funded(pair.metadata_funding().clone(), |model| {
            let mut cache = MlxAutoregressiveMechanisms::empty(
                model,
                AutoregressivePass::TargetPrefill,
                context,
            )?;
            let completed = MlxAutoregressiveMechanisms::with_invocation(
                model,
                &mut cache,
                Some(prompt),
                claim,
                context,
                |model, cache| {
                    MlxAutoregressiveMechanisms::prefill(
                        model,
                        prompt,
                        cache,
                        AutoregressivePass::TargetPrefill,
                        &cancellation,
                        context,
                    )
                },
            )?;
            Ok((completed, cache))
        })
        .unwrap();
    let SpeculativePrefillOutcome::Complete(completed) = completed else {
        panic!("actual uncancelled prefill must complete");
    };
    assert_eq!(completed.evaluated_tokens, 2);
    // The role has completed its native roots and published cache evidence.
    // Logits are not used as cache leaves or as an ownership witness.
    drop(completed);
    cache
}

fn identity_mismatch(error: &Error) {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(cause) = current {
        if matches!(
            cause.downcast_ref::<WorkingMemoryError>(),
            Some(WorkingMemoryError::IdentityMismatch)
        ) {
            return;
        }
        current = cause.source();
    }
    panic!("expected preserved source identity refusal, got {error:?}");
}

// Reuses the real completed-copy fixture. An aggregate receipt names only its
// current roots even though it retains predecessor custody after publication.
fn registered_current_roots(
    first: &crate::backend::array_copy::RegisteredArrayCopy,
    second: &crate::backend::array_copy::RegisteredArrayCopy,
    pair: &AutoregressiveSourcePair,
    environment: &crate::backend::OriginalCopyEnvironment<'_>,
    other_stream: &safemlx::Stream,
    foreign_context: SpeculativeExecutionStreams<'_>,
) -> eredu_architectures::speculative_execution::PreparedEmbeddedEvidence {
    use crate::composition::mlx::speculative::{
        retain_external_evidence_for_roots, tensor_sources, RegisteredTensorSource,
    };
    use eredu_architectures::speculative_execution::PreparedEmbeddedEvidence;
    use eredu_core::HostPreparationAuthority;
    use eredu_nn::workspace::HostMetadataFunding;
    let funding = pair.metadata_funding();
    let retain = |copy: &crate::backend::array_copy::RegisteredArrayCopy| {
        let proof = RegisteredTensorSource::from_copy(
            copy,
            pair.request().source_identity(),
            environment.stream(),
            funding,
        )
        .unwrap();
        assert!(proof
            .matches_completed_stream(environment.stream(), funding)
            .unwrap());
        assert!(!proof
            .matches_completed_stream(other_stream, funding)
            .unwrap());
        funding
            .reserve_metadata(
                PreparedEmbeddedEvidence::retained_control_bytes::<RegisteredTensorSource>()
                    .unwrap()
                    + HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().unwrap(),
            )
            .unwrap();
        PreparedEmbeddedEvidence::from_prepared(
            proof,
            HostPreparationAuthority::retain(funding.clone()),
        )
    };
    let first_proof = retain(first);
    let second_proof = retain(second);
    let context = SpeculativeExecutionStreams::single(environment.stream())
        .with_original_sources(pair, environment)
        .unwrap();
    let selected = tensor_sources::input_array_environment(
        &first_proof,
        first.array(),
        context,
        eredu_core::speculative::SamplingPlacement::Target,
        funding,
    )
    .unwrap();
    assert!(std::ptr::eq(selected, environment));
    let foreign = tensor_sources::input_array_environment(
        &first_proof,
        first.array(),
        foreign_context,
        eredu_core::speculative::SamplingPlacement::Target,
        funding,
    )
    .err()
    .expect("foreign request is refused");
    identity_mismatch(&foreign);
    let empty = || {
        crate::backend::runtime::cache::state::CompletedResidentSource::project_array_sources(
            |_| {},
            &[],
            pair.request(),
            environment.stream(),
            funding,
        )
        .unwrap()
    };
    let both = [&first_proof, &second_proof];
    let current = retain_external_evidence_for_roots(
        empty(),
        |visit| {
            visit(first.array());
            visit(first.array());
        },
        &both,
        pair.numerical_sources(),
        environment,
    )
    .unwrap();
    assert_eq!(
        tensor_sources::registered_tensor_sources(&current).count(),
        1
    );
    assert!(
        tensor_sources::registered_source_for_array(&current, first.array(), funding)
            .unwrap()
            .is_some()
    );
    assert!(
        tensor_sources::registered_source_for_array(&current, second.array(), funding)
            .unwrap()
            .is_none()
    );
    // The second copy occurs only in predecessor custody. It does not reappear
    // as a declared root through another publication or an empty root set.
    let historical = retain_external_evidence_for_roots(
        empty(),
        |visit| visit(second.array()),
        &[&current],
        pair.numerical_sources(),
        environment,
    )
    .unwrap();
    assert_eq!(
        tensor_sources::registered_tensor_sources(&historical).count(),
        0
    );
    let none = retain_external_evidence_for_roots(
        empty(),
        |_| {},
        &[&current],
        pair.numerical_sources(),
        environment,
    )
    .unwrap();
    assert_eq!(tensor_sources::registered_tensor_sources(&none).count(), 0);
    drop((first_proof, second_proof, historical, none));
    current
}

#[test]
#[ignore = "requires native Metal execution"]
fn completed_model_cache_leaf_copies_keep_independent_backing_and_exact_source_custody() {
    if !crate::tests::support::native_process::enter("original-speculative-source") {
        return;
    }
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
    let initial = pool.fixture_host_charge().unwrap();
    let (target_config, draft_config, selected) = source_configs(&backend, artifact.path());
    let mut target = load(&backend, &target_config);
    let draft = load(&backend, &draft_config);
    let config = SpeculativeConfig {
        max_tokens: 2,
        max_draft_tokens: 1,
        temperature: 0.0,
        eos_token_ids: Vec::new(),
    };
    let make_schedule = || {
        AutoregressiveSchedulePlan::new(
            &selected,
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(32).unwrap(),
            &config,
            SpeculativeSchedulerOptions::default(),
        )
        .unwrap()
    };
    let schedule = make_schedule();
    let foreign_schedule = make_schedule();
    let (copies, values, refused, evidence) = {
        // Exact original host/native prompt admission, with two one-token spans
        // so completion also carries a prior state's source through checkpoint.
        let ids = [1u32, 3];
        let parts = [eredu_runtime::input::host::HostInputPart {
            modality: eredu_core::InputModality::Text,
            kind: eredu_core::InputPayloadKind::TokenIds,
            payload: eredu_runtime::input::host::HostTensorView {
                shape: &[1, 2],
                values: eredu_runtime::input::host::HostTensorValues::U32(&ids),
            },
            metadata: &[],
            extents: &[],
        }];
        let source = pool
            .compile_prepared_host_input(
                eredu_runtime::input::host::PreparedHostInputPlan::prepare(&parts).unwrap(),
            )
            .unwrap();
        let materializer = MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let prompt = materializer
            .text_input_plan(&source)
            .unwrap()
            .materialize(&pool)
            .unwrap()
            .into_prompt(NonZeroU64::new(1));
        drop(source);
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            crate::memory_fixture::resolved_limits(REQUEST_CEILING),
        )
        .unwrap();
        let foreign = AutoregressiveSourcePair::prepare_funded(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &foreign_schedule,
            &pool,
            crate::memory_fixture::resolved_limits(REQUEST_CEILING),
            pair.metadata_funding().clone(),
        )
        .unwrap();
        assert!(!pair
            .request()
            .source_identity()
            .belongs_to_request(foreign.request()));
        let environment = backend.original_copy_environment().unwrap();
        let context = SpeculativeExecutionStreams::single(backend.stream())
            .with_original_sources(&pair, &environment)
            .unwrap();
        let foreign_context = SpeculativeExecutionStreams::single(backend.stream())
            .with_original_sources(&foreign, &environment)
            .unwrap();
        let cache = complete_cache(&mut target, &pair, schedule, &prompt, context);
        let foreign_cache = complete_cache(
            &mut target,
            &foreign,
            foreign_schedule,
            &prompt,
            foreign_context,
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        let funding = pair.metadata_funding();
        let (roots, mechanisms) = pair.numerical_prerequisites();
        let output = cache
            .native
            .with_original_copy_source(funding, |source, completed| {
                let completed_source=completed.expect("actual completed cache source");
                assert!(completed_source.matches_completed_stream(environment.stream(),funding).unwrap());
                assert!(!completed_source.matches_completed_stream(backend.weights_stream(),funding).unwrap());
                let mut leaf = None;
                source
                    .dense_key_value()
                    .expect("resident KV fixture")
                    .visit_operands(&mut |array| {
                        if leaf.is_none() {
                            leaf = Some(array);
                        }
                    });
                let leaf = leaf.expect("actual populated cache must expose a native leaf");
                let values = leaf.evaluated().unwrap().as_slice::<f32>().to_vec();
                assert!(!values.is_empty());
                assert!(values.iter().all(|value| value.is_finite()));
                assert!(values.iter().any(|value| *value != 0.0));
                let source_identity = leaf.try_allocation_info().unwrap().unwrap().identity();
                // A role-born leaf is not silently adopted into registered storage.
                let missing = IsolatedArrayCopy::new(leaf)
                    .copy_completed(
                        None,
                        &environment,
                        roots,
                        mechanisms,
                        funding,
                        &crate::memory_fixture::resolved_limits(REQUEST_CEILING),
                    )
                    .err()
                    .expect("unregistered role-born source needs its completion evidence");
                identity_mismatch(&missing);
                foreign_cache
                    .native
                    .with_original_copy_source(funding, |other, other_completed| {
                        assert!(other_completed.is_some());
                        let mut foreign_leaf = None;
                        other
                            .dense_key_value()
                            .expect("resident KV fixture")
                            .visit_operands(&mut |array| {
                                if foreign_leaf.is_none() {
                                    foreign_leaf = Some(array);
                                }
                            });
                        let foreign_leaf = foreign_leaf.unwrap();
                        // Identical geometry and numbers are insufficient evidence.
                        assert_eq!(foreign_leaf.shape(), leaf.shape());
                        assert_eq!(foreign_leaf.dtype(), leaf.dtype());
                        assert_eq!(foreign_leaf.evaluated().unwrap().as_slice::<f32>(), values);
                        assert_ne!(
                            source_identity,
                            foreign_leaf
                                .try_allocation_info()
                                .unwrap()
                                .unwrap()
                                .identity()
                        );
                        let foreign_refusal = IsolatedArrayCopy::new(foreign_leaf)
                            .copy_completed(
                                completed,
                                &environment,
                                roots,
                                mechanisms,
                                funding,
                                &crate::memory_fixture::resolved_limits(REQUEST_CEILING),
                            )
                            .err()
                            .expect("one role's completed source cannot cover another backing");
                        identity_mismatch(&foreign_refusal);
                        // A retained ingress can hold an earlier completed
                        // backing while another span produces the current
                        // cache. Its later alias must retain the earlier
                        // native budget, not be attributed to the new span.
                        let (next_budget, next_account) = other_completed.unwrap()
                            .array_source_account(foreign_leaf, funding)?;
                        let eredu_runtime::working_memory::CompletedWorkspaceSourceAccount::Model(next_account)
                            = next_account else { panic!("model cache has model custody") };
                        let retained = crate::backend::runtime::cache::state::CompletedResidentSource
                            ::capture_array_sources_with_priors(
                                |visit| { visit(foreign_leaf); visit(leaf); },
                                &[completed_source], next_budget, next_account, funding,
                            )?;
                        let installed_alias = crate::backend::runtime::cache::state::CompletedResidentSource
                            ::capture_array_sources_with_priors(
                                |visit| visit(leaf), &[&retained], next_budget, next_account, funding,
                            )?;
                        let (original_budget, original_account) = completed_source.array_source_account(leaf, funding)?;
                        let (alias_budget, alias_account) = installed_alias.array_source_account(leaf, funding)?;
                        assert!(original_account.same_account(alias_account));
                        assert!(original_budget.inspect_array(leaf).unwrap().is_some());
                        assert!(alias_budget.inspect_array(leaf).unwrap().is_some());
                        assert!(matches!(next_budget.inspect_array(leaf), Err(safemlx::OriginalBufferCause::ForeignDomain)));
                        let first = IsolatedArrayCopy::new(leaf).copy_completed(
                            Some(&installed_alias),
                            &environment,
                            roots,
                            mechanisms,
                            funding,
                            &crate::memory_fixture::resolved_limits(REQUEST_CEILING),
                        )?;
                        let second = IsolatedArrayCopy::new(leaf).copy_completed(
                            completed,
                            &environment,
                            roots,
                            mechanisms,
                            funding,
                            &crate::memory_fixture::resolved_limits(REQUEST_CEILING),
                        )?;
                        // Published copy backing follows the existing registered path,
                        // even when an unrelated completed inventory is also supplied.
                        let branch = IsolatedArrayCopy::new(first.array()).copy_completed(
                            other_completed,
                            &environment,
                            roots,
                            mechanisms,
                            funding,
                            &crate::memory_fixture::resolved_limits(REQUEST_CEILING),
                        )?;
                        let evidence=registered_current_roots(&first,&second,&pair,&environment,backend.weights_stream(),foreign_context);
                        let copies = [first, second, branch];
                        let identities = copies.each_ref().map(|copy| {
                            copy.array()
                                .try_allocation_info()
                                .unwrap()
                                .unwrap()
                                .identity()
                        });
                        for (index, copy) in copies.iter().enumerate() {
                            assert_eq!(copy.array().evaluated().unwrap().as_slice::<f32>(), values);
                            assert_ne!(identities[index], source_identity);
                            for previous in &identities[..index] {
                                assert_ne!(&identities[index], previous);
                            }
                        }
                        assert_eq!(
                            leaf.try_allocation_info().unwrap().unwrap().identity(),
                            source_identity
                        );
                        assert_eq!(leaf.evaluated().unwrap().as_slice::<f32>(), values);
                        Ok((copies, values, [missing, foreign_refusal],evidence))
                    })
            })
            .unwrap();
        pair.request().close().unwrap();
        foreign.request().close().unwrap();
        drop((cache, foreign_cache, prompt, pair, foreign));
        output
    };
    // The actual copied payloads and failed-prefix diagnostics escape both
    // source requests and model sessions with their own Q/H retention.
    drop((target, draft, target_config, draft_config, selected));
    assert!(pool.fixture_host_charge().unwrap() > initial);
    for copy in &copies {
        assert_eq!(copy.array().evaluated().unwrap().as_slice::<f32>(), values);
    }
    for error in &refused {
        identity_mismatch(error);
    }
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(copies); // payloads retire before their closed copy custody
    drop(evidence); // current-root copy receipt remains valid through source teardown
    drop(refused); // paid failure transport releases H after the error sources
    settle(&pool, initial);
}
