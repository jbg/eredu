use super::*;
use eredu_nn::workspace::WorkspaceMetadataAccount;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Debug, Default)]
struct State {
    remaining: AtomicUsize,
    last: AtomicUsize,
    retired: AtomicBool,
    dropped: AtomicUsize,
}
#[derive(Debug)]
struct Account(Arc<State>);
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        self.0.last.store(bytes, Ordering::SeqCst);
        let available = self.0.remaining.load(Ordering::SeqCst);
        let next = available
            .checked_sub(bytes)
            .ok_or(WorkspaceMetadataFundingError::Capacity {
                required: bytes as u64,
                available: available as u64,
            })?;
        self.0.remaining.store(next, Ordering::SeqCst);
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct Cause(Arc<State>);
impl std::fmt::Display for Cause {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        panic!("typed cause was formatted")
    }
}
impl std::error::Error for Cause {}
impl Drop for Cause {
    fn drop(&mut self) {
        assert!(!self.0.retired.load(Ordering::SeqCst));
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
fn funding() -> (Arc<State>, WorkspaceMetadataFunding) {
    let state = Arc::new(State::default());
    state.remaining.store(1 << 20, Ordering::SeqCst);
    let funding = WorkspaceMetadataFunding::new(Account(state.clone())).unwrap();
    (state, funding)
}

#[test]
fn embedded_neural_error_keeps_typed_source_funding_and_exact_refusal() {
    let (state, funding) = funding();
    let error = neural_cause_funded(Cause(state.clone()), &funding);
    let alias = error.clone();
    let retained = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<NeuralCause<Cause>>()
        .unwrap();
    assert!(Arc::ptr_eq(&retained.cause.0, &state));
    drop(funding);
    drop(error);
    assert!(!state.retired.load(Ordering::SeqCst));
    assert_eq!(state.dropped.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(state.dropped.load(Ordering::SeqCst), 1);
    assert!(state.retired.load(Ordering::SeqCst));

    let (state, funding) = self::funding();
    state.remaining.store(0, Ordering::SeqCst);
    let error = neural_cause_funded(Cause(state.clone()), &funding);
    let required = state.last.load(Ordering::SeqCst) as u64;
    assert!(required > 0);
    assert!(
        matches!(error.into_metadata_funding_error(),Ok(WorkspaceMetadataFundingError::Capacity {required:actual,available:0}) if actual==required)
    );
    assert_eq!(state.dropped.load(Ordering::SeqCst), 1);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(funding);
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn prepared_session_error_keeps_original_budget_after_funding_exhaustion() {
    let (state, funding) = funding();
    let prepared = prepare_session_funding::<Error>(&funding, Some(0)).unwrap();
    state.last.store(0, Ordering::SeqCst);
    state.remaining.store(0, Ordering::SeqCst);
    let cause = Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::BudgetExceeded {
        required_bytes: 179_957_231, available_bytes: 2_236_875,
    });
    let error = prepared.retain(cause);
    assert_eq!(state.last.load(Ordering::SeqCst), 0,
        "converting the actual session error makes no new metadata request");
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let mut found = false;
    while let Some(cause) = source {
        if let Some(eredu_runtime::working_memory::WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) =
            cause.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>() {
            assert_eq!((*required_bytes, *available_bytes), (179_957_231, 2_236_875));
            found = true;
        }
        source = cause.source();
    }
    assert!(found);
    drop(funding);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(error);
    assert!(state.retired.load(Ordering::SeqCst));

    let (state, funding) = self::funding();
    state.remaining.store(0, Ordering::SeqCst);
    assert!(matches!(prepare_session_funding::<Error>(&funding, Some(0)),
        Err(WorkspaceMetadataFundingError::Capacity { available: 0, .. })));
    // A refused preparation returns before there is an operation/cause to lose.
    drop(funding);
    assert!(state.retired.load(Ordering::SeqCst));
}
