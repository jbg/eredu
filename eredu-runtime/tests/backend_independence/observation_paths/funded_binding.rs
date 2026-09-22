use super::*;
use eredu_nn::workspace::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug, Default)]
struct Ledger {
    calls: AtomicUsize,
    cut: AtomicUsize,
    retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<Ledger>);
impl eredu_core::HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), eredu_core::HostMetadataFundingError> {
        let call = self.0.calls.fetch_add(1, Ordering::SeqCst);
        if call == self.0.cut.load(Ordering::SeqCst) {
            return Err(eredu_core::HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            });
        }
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::memory::topology_ref())
    }
    fn output_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory::placement_ref())
    }
    fn scratch_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory::placement_ref())
    }

    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("metadata only")
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
}
fn destination() -> (WorkspaceContext, Arc<Ledger>) {
    let ledger = Arc::new(Ledger::default());
    ledger.cut.store(usize::MAX, Ordering::SeqCst);
    let funding = eredu_core::HostMetadataFunding::new(Account(ledger.clone())).unwrap();
    (
        WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap(),
        ledger,
    )
}
fn runtime() -> ResidentRuntime<PathsFixture, FakeBackend, State> {
    ResidentRuntime::new(PathsFixture::new(), &()).unwrap()
}

#[test]
fn finalized_parameter_binding_requires_paid_cold_rebind_and_rejects_old_fingerprint() {
    for residency in [
        eredu_runtime::LayerWeightResidency::FullyResident,
        eredu_runtime::LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let (mut session, counters) =
            super::super::prepared_session_observation::session(residency);
        let source = session.shared_observation_paths().unwrap().clone();
        let original = session
            .prepared_observation_paths()
            .unwrap()
            .binding_identity();
        let before = counters.snapshot();
        session.finalize_parameter_publication();
        assert!(session.parameter_observation_binding_pending());
        assert!(session.prepared_observation_paths().is_none());
        assert!(session
            .validate_prepared_observation_paths(&source)
            .is_err());

        let (context, ledger) = destination();
        ledger
            .cut
            .store(ledger.calls.load(Ordering::SeqCst), Ordering::SeqCst);
        let error = session
            .prepare_parameter_observation_paths(eredu_runtime::layered::LayeredMetadata::new(
                &context,
                |cause| cause,
            ))
            .unwrap_err();
        let eredu_runtime::ReplicatedTextSessionError::Architecture(error) = error else {
            panic!("lost actual metadata refusal")
        };
        assert!(matches!(
            error.into_metadata_funding_error(),
            Ok(eredu_core::HostMetadataFundingError::Capacity { available: 0, .. })
        ));
        assert!(session.prepared_observation_paths().is_none());
        assert!(session.parameter_observation_binding_pending());

        ledger.cut.store(usize::MAX, Ordering::SeqCst);
        session
            .prepare_parameter_observation_paths(eredu_runtime::layered::LayeredMetadata::new(
                &context,
                |cause| cause,
            ))
            .unwrap();
        let current = session.prepared_observation_paths().unwrap();
        assert!(!original.matches(current));
        assert!(current.source().same_storage(&source));
        session
            .validate_prepared_observation_paths(&source)
            .unwrap();
        assert!(!session.parameter_observation_binding_pending());
        assert_eq!(counters.snapshot(), before);

        session.invalidate_parameter_observations();
        assert!(session.prepared_observation_paths().is_some());
        assert!(!session.parameter_observation_binding_pending());
        let calls = ledger.calls.load(Ordering::SeqCst);
        assert!(matches!(
            session.prepare_parameter_observation_paths(
                eredu_runtime::layered::LayeredMetadata::new(&context, |cause| cause),
            ),
            Err(
                eredu_runtime::ReplicatedTextSessionError::PreparedObservation(
                    eredu_runtime::PreparedSessionObservationError::BindingMismatch
                )
            )
        ));
        assert_eq!(ledger.calls.load(Ordering::SeqCst), calls);
    }
}

#[test]
fn paid_rebinding_preserves_source_identity_and_fingerprint_custody_at_every_refusal() {
    let original = runtime();
    let prepared = original.prepare_observation_paths().unwrap();
    let source = prepared.source();
    let target = runtime();
    let (context, ledger) = destination();
    let start = ledger.calls.load(Ordering::SeqCst);
    let bound = target
        .bind_observation_paths(
            source,
            Some(eredu_runtime::layered::LayeredMetadata::new(
                &context,
                |error| error,
            )),
        )
        .unwrap();
    assert!(bound.source().same_storage(source));
    target.validate_observation_binding(&bound).unwrap();
    let requests = ledger.calls.load(Ordering::SeqCst) - start;
    assert!(requests > 2);
    let identity = bound.binding_identity();
    drop(target);
    drop(bound);
    drop(context);
    assert!(
        !ledger.retired.load(Ordering::SeqCst),
        "weak identity retains its paid allocation"
    );
    drop(identity);
    assert!(ledger.retired.load(Ordering::SeqCst));
    for cut in 0..requests {
        let target = runtime();
        let (context, ledger) = destination();
        let start = ledger.calls.load(Ordering::SeqCst);
        ledger.cut.store(start + cut, Ordering::SeqCst);
        let error = target
            .bind_observation_paths(
                source,
                Some(eredu_runtime::layered::LayeredMetadata::new(
                    &context,
                    |error| error,
                )),
            )
            .unwrap_err();
        let PreparedError::Execution(error) = error else {
            panic!("lost original funding refusal")
        };
        assert!(matches!(
            error.into_metadata_funding_error(),
            Ok(eredu_core::HostMetadataFundingError::Capacity { available: 0, .. })
        ));
        assert_eq!(
            ledger.calls.load(Ordering::SeqCst),
            start + cut + 1,
            "cut {cut}"
        );
        assert!(target.architecture().inner.trace.is_empty());
        drop(target);
        drop(context);
        assert!(ledger.retired.load(Ordering::SeqCst), "cut {cut}");
    }
}
