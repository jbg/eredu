use super::*;
use crate::working_memory::quote_inference_workspace_with_context;

fn terminal_geometry() -> InferenceGeometry {
    InferenceGeometry {
        input_positions: 0,
        max_output_tokens: 0,
        prefill_chunk_positions: 0,
        output: OutputDemand::StateOnly,
        ..geometry()
    }
}

fn no_equations(_: &InferenceWorkspaceSpan) -> Result<WorkspaceTraceReport, Error> {
    panic!("a terminal placement must not invoke an equation")
}

#[test]
fn terminal_quote_preserves_existing_backing_and_its_registered_owner() {
    let g = terminal_geometry();
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let storage = RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root)]).unwrap();
    let report = quote_inference_workspace_with_context(g, &context, no_equations).unwrap();
    assert_eq!(report.completed_spans(), 0);
    assert_eq!(report.transient().bytes(), Some(0));
    let retained = state(g)
        .with_selected_state_backing(g, WorkspaceBound::bounded(64, "actual retained source"))
        .unwrap();
    let refined = report.refine_state_backing(retained.clone()).unwrap();
    assert_eq!(refined, retained);
    let quote =
        ResidualInferenceQuote::compose(&report, retained, outside(g, 13), &storage).unwrap();
    assert_eq!(quote.state().requested_state_bytes, 64);
    assert_eq!(quote.incremental_bytes().unwrap(), 13);
    drop(storage);
    drop(original);
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    drop(quote);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn terminal_quote_requires_exact_selection_and_preserves_missing_state_coverage() {
    let g = terminal_geometry();
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let absent = quote_inference_workspace_with_context(g, &context, no_equations).unwrap();
    assert!(absent.residual_workspace().is_none());
    let storage = RegisteredWorkspaceStorage::bind(
        &pool,
        &context,
        std::iter::empty::<(u32, WorkspaceExistingStorage)>(),
    )
    .unwrap();
    assert!(ResidualInferenceQuote::compose(&absent, state(g), outside(g, 0), &storage).is_err());
    let report = quote_inference_workspace_with_context(g, &context, no_equations).unwrap();
    let other_context = WorkspaceContext::new(Facts::default());
    let other = RegisteredWorkspaceStorage::bind(
        &pool,
        &other_context,
        std::iter::empty::<(u32, WorkspaceExistingStorage)>(),
    )
    .unwrap();
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(g), outside(g, 0), &other),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        )),
    ));
    let unknown = state(g)
        .with_selected_state_backing(
            g,
            WorkspaceBound::Unknown {
                reason: "retained source capacity is unavailable".into(),
            },
        )
        .unwrap();
    assert_eq!(
        report.refine_state_backing(unknown.clone()).unwrap(),
        unknown
    );
    let unknown_quote =
        ResidualInferenceQuote::compose(&report, unknown, outside(g, 0), &storage).unwrap();
    assert_eq!(
        unknown_quote.state().completeness,
        EstimationCompleteness::PersistentStateOnly,
    );
    assert!(
        plan(
            &pool,
            &InferenceExecutionIdentity::default(),
            &unknown_quote,
            request(g),
            4096,
        )
        .is_err()
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(ResidualInferenceQuote::compose(&report, state(g), outside(g, 0), &storage).is_ok());
}
