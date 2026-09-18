use super::*;
use eredu_nn::workspace::*;
use std::sync::{Arc, atomic::{AtomicUsize, AtomicBool, Ordering}};

#[derive(Debug, Default)]
struct Ledger { calls: AtomicUsize, cut: AtomicUsize, retired: AtomicBool }
#[derive(Debug)]
struct Account(Arc<Ledger>);
impl eredu_core::HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), eredu_core::HostMetadataFundingError> {
        let call = self.0.calls.fetch_add(1, Ordering::SeqCst);
        if call == self.0.cut.load(Ordering::SeqCst) {
            return Err(eredu_core::HostMetadataFundingError::Capacity { required: bytes as u64, available: 0 });
        }
        Ok(())
    }
}
impl Drop for Account { fn drop(&mut self) { self.0.retired.store(true, Ordering::SeqCst); } }
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(&self, _: &WorkspaceOperation) -> Result<Option<WorkspaceOperationBound>, Error> { panic!("metadata only") }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = std::convert::Infallible;
    fn operation_facts(&self, _: WorkspaceOperationView<'_>) -> Result<Option<WorkspaceOperationFacts>, Self::Error> { Ok(None) }
    fn write_operation_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceEffectDestination<'_>) -> Result<Option<WorkspaceOperationFacts>, Self::Error> { Ok(None) }
    fn host_facts(&self, _: WorkspaceOperationView<'_>) -> Result<Option<WorkspaceHostFacts>, Self::Error> { Ok(None) }
    fn write_host_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceHostDestination<'_>) -> Result<Option<WorkspaceHostFacts>, Self::Error> { Ok(None) }
}
fn destination() -> (WorkspaceContext, Arc<Ledger>) {
    let ledger = Arc::new(Ledger::default());
    ledger.cut.store(usize::MAX, Ordering::SeqCst);
    let funding = eredu_core::HostMetadataFunding::new(Account(ledger.clone())).unwrap();
    (WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap(), ledger)
}
fn runtime() -> ResidentRuntime<PathsFixture, FakeBackend, State> {
    ResidentRuntime::new(PathsFixture::new(), &()).unwrap()
}

#[test]
fn paid_rebinding_preserves_source_identity_and_fingerprint_custody_at_every_refusal() {
    let original = runtime();
    let prepared = original.prepare_observation_paths().unwrap();
    let source = prepared.source();
    let target = runtime();
    let (context, ledger) = destination();
    let start = ledger.calls.load(Ordering::SeqCst);
    let bound = target.bind_observation_paths(source,
        Some(eredu_runtime::layered::LayeredMetadata::new(&context, |error| error))).unwrap();
    assert!(bound.source().same_storage(source));
    target.validate_observation_binding(&bound).unwrap();
    let requests = ledger.calls.load(Ordering::SeqCst) - start;
    assert!(requests > 2);
    let identity = bound.binding_identity();
    drop(target);
    drop(bound);
    drop(context);
    assert!(!ledger.retired.load(Ordering::SeqCst), "weak identity retains its paid allocation");
    drop(identity);
    assert!(ledger.retired.load(Ordering::SeqCst));
    for cut in 0..requests {
        let target = runtime();
        let (context, ledger) = destination();
        let start = ledger.calls.load(Ordering::SeqCst);
        ledger.cut.store(start + cut, Ordering::SeqCst);
        let error = target.bind_observation_paths(source,
            Some(eredu_runtime::layered::LayeredMetadata::new(&context, |error| error))).unwrap_err();
        let PreparedError::Execution(error) = error else { panic!("lost original funding refusal") };
        assert!(matches!(error.into_metadata_funding_error(), Ok(eredu_core::HostMetadataFundingError::Capacity { available: 0, .. })));
        assert_eq!(ledger.calls.load(Ordering::SeqCst), start + cut + 1, "cut {cut}");
        assert!(target.architecture().inner.trace.is_empty());
        drop(target);
        drop(context);
        assert!(ledger.retired.load(Ordering::SeqCst), "cut {cut}");
    }
}
