use super::*;
use crate::working_memory::{InferenceRequest, SubmissionTrackingFacts, TextPreparationScopeFacts};
use std::num::NonZeroU64;

fn config(capacity: u64) -> eredu_core::TextGenerationConfig {
    eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(geometry().max_output_tokens as usize),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_inference_policy(eredu_core::TextInferencePolicy {
        managed_memory_capacity_bytes: Some(1_000_000),
        submission_tracking_capacity_bytes: NonZeroU64::new(capacity),
        graph_metadata_capacity_bytes: None,
        ..Default::default()
    })
}
fn controls(quote: &IncrementalInferenceQuote) -> PreparedTextControlWorkspace {
    PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        quote.span_workspace().plan(),
        TextHostControlFacts::new(Some(17), Some(0), Some(34)),
    )
    .unwrap()
    .with_preparation_scopes(TextPreparationScopeFacts::new(Some(0), Some(0)))
    .unwrap()
    .with_submission_tracking(
        SubmissionTrackingFacts::new(NonZeroU64::new(4096).unwrap(), 5000).unwrap(),
    )
    .unwrap()
}

#[test]
fn original_tracking_control_only_quote_is_exact_nonrefillable_and_not_native_headroom() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let controls = controls(&original);
    assert!(controls.source_identity().is_none());
    assert_eq!(controls.sequence_storage_bytes(), 0);
    assert!(matches!(
        controls.clone().with_submission_tracking(
            SubmissionTrackingFacts::new(NonZeroU64::new(4096).unwrap(), 5000).unwrap()
        ),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let quote = original
        .with_span_workspace_and_text_controls(controls)
        .unwrap();
    let protected = quote.span_workspace().retention_peak_bytes().unwrap()
        + quote
            .span_workspace()
            .text_controls()
            .unwrap()
            .facts()
            .total_bytes()
            .unwrap()
            .unwrap();
    let exact = 64 + quote.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &quote, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (r, accepted) = sealed_plan(&pool, &quote, exact).unwrap();
    drop(quote);
    let (r, run) = r.into_funding().unwrap();
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    assert_eq!(span.protected_host_bytes(), protected);
    let preparation = InferenceRequest::from(&r)
        .prepare_text(&r.0.execution, geometry(), config(4096))
        .unwrap();
    let mut pair = span.take_preparation_scopes().unwrap().unwrap();
    let capsule = span
        .take_submission_tracking(&preparation)
        .unwrap()
        .unwrap();
    assert_eq!(capsule.facts().capacity().get(), 4096);
    assert!(matches!(
        span.take_submission_tracking(&preparation),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    assert!(
        matches!(
            pair.claim_prompt(&preparation),
            Err(WorkingMemoryError::TextRunUnbound)
        ),
        "host memory extraction cannot bind a core run or authorize a Scope"
    );
    let native = run.scope().unwrap();
    let free = r.bytes() - protected;
    assert!(
        matches!(native.adopt_storage_individually([(90u32, free + 1)]),
        Err(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })
        if required_bytes == free + 1 && available_bytes == free)
    );
    let payload = native.adopt_storage_individually([(90u32, free)]).unwrap();
    drop(payload);
    native.certify().unwrap();
    drop((pair, preparation, span, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(capsule);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_tracking_rejects_foreign_preparation_without_spending_the_real_capsule() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let quote = original
        .clone()
        .with_span_workspace_and_text_controls(controls(&original))
        .unwrap();
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let other = replacement_quote(&pool, geometry(), 0).into_incremental();
    let (foreign, foreign_run, other) = accept(&pool, other);
    let preparation = InferenceRequest::from(&foreign)
        .prepare_text(&foreign.0.execution, geometry(), config(4096))
        .unwrap();
    assert!(matches!(
        span.take_submission_tracking(&preparation),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let real = InferenceRequest::from(&r)
        .prepare_text(&r.0.execution, geometry(), config(4096))
        .unwrap();
    let capsule = span.take_submission_tracking(&real).unwrap().unwrap();
    assert_eq!(capsule.facts().provider_bytes(), 5000);
    let protected = span.protected_host_bytes();
    drop((
        original,
        span,
        real,
        preparation,
        other,
        foreign_run,
        foreign,
        r,
        run,
        root,
    ));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _original = capsule;
        std::panic::panic_any(73u32);
    }));
    assert_eq!(*panic.unwrap_err().downcast::<u32>().unwrap(), 73);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn selected_tracking_capacity_keeps_requested_policy_and_original_custody() {
    let ceiling = NonZeroU64::new(4096).unwrap();
    let selected = NonZeroU64::new(2048).unwrap();
    let facts = SubmissionTrackingFacts::within_ceiling(ceiling, selected, 3000).unwrap();
    assert_eq!(facts.requested_ceiling(), Some(ceiling));
    assert_eq!(facts.capacity(), selected);
    assert!(matches!(
        SubmissionTrackingFacts::within_ceiling(selected, ceiling, 5000),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let control = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        original.span_workspace().plan(),
        TextHostControlFacts::new(Some(17), Some(0), Some(34)),
    )
    .unwrap()
    .with_preparation_scopes(TextPreparationScopeFacts::new(Some(0), Some(0)))
    .unwrap()
    .with_submission_tracking(facts)
    .unwrap();
    let quote = original
        .with_span_workspace_and_text_controls(control)
        .unwrap();
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    // The exact request still declares the larger ceiling. A smaller allocation
    // does not rewrite request identity or permit post-admission expansion.
    let preparation = InferenceRequest::from(&r)
        .prepare_text(&r.0.execution, geometry(), config(ceiling.get()))
        .unwrap();
    let capsule = span
        .take_submission_tracking(&preparation)
        .unwrap()
        .unwrap();
    assert_eq!(capsule.facts(), facts);
    assert!(matches!(
        span.take_submission_tracking(&preparation),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let protected = span.protected_host_bytes();
    drop((preparation, span, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(capsule);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
