//! Actual full-quote prompt work; no core context, ready seal or resumed engine.
use super::super::super::tests::{
    AllTokens, Runtime, advance, cause, config, identity, reclaim, runtime, settle, tokens, words,
};
use super::*;
use eredu_core::{
    AdmissionRequest, AdmissionResult, InputTokenCount, TextGenerationDriver, TextGenerationInput,
    TokenFilter,
};
use eredu_runtime::working_memory::{InferenceRequest, WorkingMemoryPool};
use eredu_runtime::{HostSlotAttachmentError, HostSlotMetadata};
use safemlx::Dtype;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Failure {
    None,
    Table,
    Pending,
    Native,
    Publication,
}
thread_local! {
    static FAILURE: Cell<Failure> = const { Cell::new(Failure::None) };
    static TABLES: Cell<usize> = const { Cell::new(0) };
    static PENDING: Cell<usize> = const { Cell::new(0) };
}
struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        FAILURE.set(Failure::None);
    }
}
#[derive(Debug, thiserror::Error)]
#[error("injected failure after actual pending numerical construction")]
struct LateFailure;

#[derive(Debug, thiserror::Error)]
#[error("injected final publication failure after native settlement")]
struct FinalPublicationFailure;

pub(super) fn before_table_publication(
    metadata: &HostSlotMetadata,
    funding: &WorkingMemoryFundingRun,
) -> Result<(), Error> {
    if FAILURE.get() == Failure::Table {
        // Poison the actual destination attachment lock before real publish.
        let failed = catch_unwind(AssertUnwindSafe(|| {
            let _ = metadata.try_attach(
                funding.pool().shared_storage_domain(),
                || -> Result<Box<dyn Send + Sync>, std::convert::Infallible> {
                    panic!("destination table attachment fault")
                },
            );
        }));
        assert!(failed.is_err());
    }
    Ok(())
}
pub(super) fn after_table_publication(funding: &WorkingMemoryFundingRun) -> Result<(), Error> {
    TABLES.set(TABLES.get() + 1);
    if FAILURE.get() == Failure::Pending {
        // An independently uncertified scope genuinely fences this destination
        // account. Pending host preparation must recheck it before allocation.
        drop(funding.scope().unwrap());
    }
    Ok(())
}
pub(super) fn after_pending_construction(_: &WorkingMemoryFundingRun) -> Result<(), Error> {
    PENDING.set(PENDING.get() + 1);
    if FAILURE.get() == Failure::Native {
        Err(other(LateFailure))
    } else {
        Ok(())
    }
}
pub(super) fn before_final_publication(_: &WorkingMemoryFundingRun) -> Result<(), Error> {
    if FAILURE.get() == Failure::Publication {
        // Settled storage may still be published from a quarantined account;
        // quarantining a sibling scope is therefore not a publication fault.
        // Inject a typed finalizer failure while its native scope remains live.
        return Err(other(FinalPublicationFailure));
    }
    Ok(())
}

fn source(runtime: &mut Runtime) -> CopiedTextComponentsOwner {
    let mut driver = TextGenerationDriver::new(runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(tokens()),
            config(Some(u64::MAX)),
            AllTokens,
        )
        .unwrap();
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let saved = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        PreparedTextComponentsCopy::prepare(runtime, &generation.sampling, outputs.last())
            .unwrap()
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap()
    };
    drop((outputs, state, driver));
    reclaim();
    CopiedTextComponentsOwner::new(saved)
}

fn fresh(
    runtime: &Runtime,
    source: &CopiedTextComponents,
) -> (InferenceTextPreparation, WorkingMemoryFundingRun) {
    let config = config(Some(u64::MAX));
    let quote = PreparedSavedTextResumeQuote::prepare(
        runtime,
        source,
        config,
        eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        },
    )
    .unwrap();
    let geometry = quote.geometry();
    let capabilities = runtime.session().capability_estimate().unwrap();
    // Conservative native upper bounds are admissible when every required
    // term is bounded. Use the same strict policy as the production driver.
    let AdmissionResult::Admitted(admission) = eredu_core::apply_admission_policy(
        capabilities.capabilities(),
        AdmissionRequest {
            input: InputTokenCount::text(geometry.cached_positions + geometry.input_positions),
            max_output_tokens: geometry.max_output_tokens,
            batch_size: geometry.batch_size,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        },
        quote.full().clone(),
        None,
    )
    .unwrap() else {
        panic!("actual saved prompt quote rejected")
    };
    let execution = runtime
        .session()
        .payload
        .model
        .erased()
        .inference_execution_identity();
    let reservation = runtime
        .backend()
        .memory_pool()
        .reserve_with_capacity(execution, &admission, u64::MAX)
        .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let preparation = InferenceRequest::from(reservation)
        .prepare_text(execution, geometry, config)
        .unwrap();
    (preparation, run)
}
fn values(source: &CopiedTextComponents) -> Vec<Vec<f32>> {
    let mut values = Vec::new();
    source
        .decoder
        .native
        .prepare_copy()
        .unwrap()
        .visit_operands(&mut |array| {
            values.push(array.evaluated().unwrap().try_to_vec::<f32>().unwrap());
        });
    values
}
fn run(
    runtime: &mut Runtime,
    source: &CopiedTextComponentsOwner,
    preparation: &InferenceTextPreparation,
    funding: &WorkingMemoryFundingRun,
) -> Result<ProvisionalTextResumePrompt, Error> {
    construct(
        runtime,
        PromptAuthority {
            source,
            preparation,
            funding,
        },
    )
}

#[test]
fn saved_prompt_constructs_exact_fresh_values_without_installing_or_leaving_a_lease() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let source = source(&mut runtime);
    let source_values = values(&source);
    assert!(source_values.iter().flatten().any(|value| *value != 0.));
    let source_key = source.sampling.arrays.key.as_ref().unwrap();
    let source_pending = source.sampling.arrays.pending.as_ref().unwrap();
    let source_input = source.decoder.input.as_ref().unwrap().clone();
    let before = runtime.session().payload.model.erased().state_snapshot();
    let old = runtime
        .session()
        .payload
        .model
        .erased()
        .retained_inference_authority()
        .unwrap();
    let (preparation, funding) = fresh(&runtime, &source);
    let provisional = run(&mut runtime, &source, &preparation, &funding).unwrap();
    runtime.session().authority.borrow().require_idle().unwrap();
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        before
    );
    assert_eq!(values(&source), source_values);
    assert_eq!(words(provisional.key().unwrap()), words(source_key));
    assert_ne!(identity(provisional.key().unwrap()), identity(source_key));
    let prompt = provisional.prompt();
    assert!(
        prompt.quote.is_none() && prompt.cache_identity.is_none() && prompt.memory_owner.is_none()
    );
    assert_eq!(prompt.parts.len(), 1);
    let input = prompt.parts[0].payload().value();
    assert_eq!(input.shape(), &[1, 1]);
    assert_eq!(input.dtype(), Dtype::Uint32);
    assert_eq!(words(input), words(source_pending));
    assert_ne!(identity(input), identity(source_pending));
    prompt
        .inference_request
        .as_ref()
        .unwrap()
        .validate_same_request(preparation.request())
        .unwrap();
    assert!(
        prompt
            .inference_request
            .as_ref()
            .unwrap()
            .validate_same_request(old.admission().unwrap().request())
            .is_err()
    );
    let control = provisional
        .state
        .state
        .downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<
            crate::backend::runtime::cache::state::MlxKeyValueState,
        >>()
        .unwrap();
    assert!(
        control
            .shared_prompt_input_identity()
            .unwrap()
            .same_storage(&source_input)
    );
    assert!(matches!(
        preparation.claim_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    preparation.bind_prompt().unwrap(); // constructed only; Sampling remains unclaimed
    let (mut state, key, prompt) = provisional.into_parts();
    // Separate existing installer, after proving the worker left the model alone.
    <MlxBackend<'_> as eredu_core::execution_control::NativeTextStateBackend>::exchange_native_text_state(
        &mut runtime,
        &mut state,
    )
    .unwrap();
    assert!(
        runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap()
            .is_empty()
    );
    runtime
        .session()
        .payload
        .model
        .erased()
        .validate_text_frontier(source.sampling.frontier())
        .unwrap();
    let mut installed = Vec::new();
    runtime
        .session()
        .payload
        .model
        .erased()
        .prepare_resident_decoder_copy()
        .unwrap()
        .visit_operands(&mut |array| {
            installed.push(array.evaluated().unwrap().try_to_vec::<f32>().unwrap())
        });
    assert_eq!(installed, source_values);
    drop((
        state,
        key,
        prompt,
        preparation,
        funding,
        old,
        source_input,
        source,
        runtime,
        artifact,
    ));
    settle(&pool, 0);
}

#[test]
fn repeated_claim_and_foreign_request_fail_before_any_table_or_pending_work() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let source = source(&mut runtime);
    let (preparation, funding) = fresh(&runtime, &source);
    let first = run(&mut runtime, &source, &preparation, &funding).unwrap();
    let before = (TABLES.get(), PENDING.get(), pool.used_bytes().unwrap());
    let error = run(&mut runtime, &source, &preparation, &funding)
        .err()
        .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::PreparationAlreadyStarted)
    );
    assert_eq!(
        (TABLES.get(), PENDING.get(), pool.used_bytes().unwrap()),
        before
    );
    runtime.session().authority.borrow().require_idle().unwrap();
    assert!(!runtime.session().poison.get());
    let (other_preparation, other_funding) = fresh(&runtime, &source);
    let error = run(&mut runtime, &source, &other_preparation, &funding)
        .err()
        .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!((TABLES.get(), PENDING.get()), (before.0, before.1));
    runtime.session().authority.borrow().require_idle().unwrap();
    assert!(!runtime.session().poison.get());
    drop((
        first,
        preparation,
        funding,
        other_preparation,
        other_funding,
        source,
        runtime,
        artifact,
    ));
    settle(&pool, 0);
}

#[test]
fn actual_table_and_pending_host_failures_retire_partial_values_under_original_recovery() {
    for failure in [
        Failure::Table,
        Failure::Pending,
        Failure::Native,
        Failure::Publication,
    ] {
        let _reset = Reset;
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let source = source(&mut runtime);
        let source_values = values(&source);
        let before = runtime.session().payload.model.erased().state_snapshot();
        let baseline = pool.used_bytes().unwrap();
        let (preparation, funding) = fresh(&runtime, &source);
        let counters = (TABLES.get(), PENDING.get());
        FAILURE.set(failure);
        let error = run(&mut runtime, &source, &preparation, &funding)
            .err()
            .unwrap_or_else(|| panic!("{failure:?} must reject the prompt"));
        FAILURE.set(Failure::None);
        match failure {
            Failure::Table => {
                assert!(cause::<HostSlotAttachmentError<WorkingMemoryError>>(&error).is_some());
                assert_eq!((TABLES.get(), PENDING.get()), counters);
            }
            Failure::Pending => {
                assert_eq!(
                    cause::<WorkingMemoryError>(&error),
                    Some(&WorkingMemoryError::ExecutionFenced)
                );
                assert_eq!((TABLES.get(), PENDING.get()), (counters.0 + 1, counters.1));
            }
            Failure::Native => assert!(cause::<LateFailure>(&error).is_some()),
            Failure::Publication => assert!(cause::<FinalPublicationFailure>(&error).is_some()),
            Failure::None => unreachable!(),
        }
        runtime.session().authority.borrow().require_idle().unwrap();
        assert!(
            !runtime.session().poison.get(),
            "settled copy failure preserves the unchanged live model"
        );
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before
        );
        assert_eq!(values(&source), source_values);
        assert!(matches!(
            preparation.bind_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        drop((preparation, funding));
        // Destination uncertainty remains charged. The old saved source is
        // independently reusable after all actual native copying has settled.
        reclaim();
        assert!(pool.used_bytes().unwrap() > baseline);
        let (retry_preparation, retry_funding) = fresh(&runtime, &source);
        let retry = run(&mut runtime, &source, &retry_preparation, &retry_funding).unwrap();
        runtime.session().authority.borrow().require_idle().unwrap();
        assert_eq!(values(&source), source_values);
        drop((
            retry,
            retry_preparation,
            retry_funding,
            source,
            runtime,
            artifact,
        ));
        reclaim();
        assert!(pool.used_bytes().unwrap() > 0);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
    }
}
