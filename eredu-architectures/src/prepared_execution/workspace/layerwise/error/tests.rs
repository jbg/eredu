use super::*;
use eredu_nn::workspace::*;
use std::{convert::Infallible, sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}}};

#[derive(Debug, Default)]
struct State { remaining: Mutex<usize>, calls: AtomicUsize, retired: AtomicBool }
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.0.calls.fetch_add(1, Ordering::SeqCst);
        let mut remaining = self.0.remaining.lock().unwrap();
        *remaining = remaining.checked_sub(bytes).ok_or(HostMetadataFundingError::Capacity {
            required: bytes as u64, available: *remaining as u64,
        })?;
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) { self.0.retired.store(true, Ordering::SeqCst); }
}
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(&self, _: &WorkspaceOperation) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("funded fixture must use the finite fact interface")
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = Infallible;
    fn operation_facts(&self, _: WorkspaceOperationView<'_>) -> Result<Option<WorkspaceOperationFacts>, Infallible> { Ok(None) }
    fn write_operation_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceEffectDestination<'_>) -> Result<Option<WorkspaceOperationFacts>, Infallible> { Ok(None) }
    fn host_facts(&self, _: WorkspaceOperationView<'_>) -> Result<Option<WorkspaceHostFacts>, Infallible> { Ok(None) }
    fn write_host_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceHostDestination<'_>) -> Result<Option<WorkspaceHostFacts>, Infallible> { Ok(None) }
}
fn context() -> (WorkspaceContext, Arc<State>) {
    let state = Arc::new(State::default());
    *state.remaining.lock().unwrap() = usize::MAX;
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
    (context, state)
}
fn propagate(error: Error, mode: usize, context: &WorkspaceContext) -> Error {
    match mode {
        0 => error.into_quote_error(context),
        1 => LayerwiseRuntimeError::Architecture(error).into_quote_error(context),
        2 => LayerwiseRuntimeError::Policy(error).into_quote_error(context),
        3 => PreparedLayeredObservationError::Execution(error).into_quote_error(context),
        4 => PreparedLayeredObservationError::Execution(LayerwiseRuntimeError::Architecture(error)).into_quote_error(context),
        5 => PreparedLayeredObservationError::Execution(LayerwiseRuntimeError::Policy(error)).into_quote_error(context),
        6 => eredu_runtime::ReplicatedTextSessionError::<Error, Infallible, Infallible>::Architecture(error).into_quote_error(context),
        _ => unreachable!(),
    }
}
#[test]
fn direct_and_observed_quote_failures_preserve_first_refusal_without_another_callback() {
    for mode in 0..7 {
        let (context, state) = context();
        *state.remaining.lock().unwrap() = 0;
        let before = state.calls.load(Ordering::SeqCst);
        let error = context.metadata_vec::<u8>(1).unwrap_err();
        let original = error.clone().into_metadata_funding_error().unwrap();
        assert_eq!(state.calls.load(Ordering::SeqCst), before + 1);
        let error = propagate(error, mode, &context);
        assert_eq!(error.into_metadata_funding_error().unwrap(), original);
        assert_eq!(state.calls.load(Ordering::SeqCst), before + 1);
        drop(context);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}
#[derive(Debug)]
struct Cause(HostMetadataFunding);
impl std::fmt::Display for Cause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("actual source") }
}
impl std::error::Error for Cause {}
#[test]
fn wrappers_move_existing_source_and_payer_but_fixed_observation_causes_are_paid() {
    for mode in 0..7 {
        let (context, state) = context();
        let error = context.metadata_source(Cause(context.metadata_funding().unwrap()));
        let alias = error.clone();
        *state.remaining.lock().unwrap() = 0;
        let before = state.calls.load(Ordering::SeqCst);
        let result = propagate(error, mode, &context);
        assert_eq!(state.calls.load(Ordering::SeqCst), before);
        assert!(std::ptr::eq(std::error::Error::source(&result).unwrap(),
            std::error::Error::source(&alias).unwrap()));
        drop(context);
        drop(result);
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(alias);
        assert!(state.retired.load(Ordering::SeqCst));
    }
    let (context, state) = context();
    *state.remaining.lock().unwrap() = 0;
    let before = state.calls.load(Ordering::SeqCst);
    let result = PreparedLayeredObservationError::<Error>::SemanticMismatch.into_quote_error(&context);
    assert!(matches!(result.into_metadata_funding_error().unwrap(), HostMetadataFundingError::Capacity { available: 0, .. }));
    assert_eq!(state.calls.load(Ordering::SeqCst), before + 1);
}
