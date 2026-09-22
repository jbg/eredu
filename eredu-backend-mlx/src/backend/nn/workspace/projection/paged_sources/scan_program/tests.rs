use super::*;
use crate::memory_fixture::LedgerFixture;
use eredu_nn::workspace::{WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, MemoryLedger};

#[derive(Debug)]
struct InspectionOnly;
impl WorkspaceMechanisms for InspectionOnly {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        panic!("inspection slots grant no numerical execution")
    }
}

impl eredu_nn::workspace::WorkspaceFactMechanisms for InspectionOnly {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("inspection slots cannot quote equations")
    }
    fn write_operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceEffectDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("inspection slots cannot emit equations")
    }
    fn host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("inspection slots cannot quote host equations")
    }
    fn write_host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceHostDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("inspection slots cannot emit host equations")
    }
}

#[test]
#[ignore = "requires native CPU source construction"]
fn ordinary_scan_aliases_require_completed_backing_and_retain_their_metadata_payer() {
    if !crate::tests::support::native_process::enter("ordinary-scan-alias-source") {
        return;
    }
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let keys = Array::from_slice(&[1.25f32, -2.5, 3.75, 4.5], &[1, 1, 4, 1]);
    let values = keys.multiply(Array::from_f32(2.0), &stream).unwrap();
    let ledger = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let funding = ledger
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(1 << 24),
        )
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(InspectionOnly, funding.clone()).unwrap();
    let mut pair = ScanSourcePair::prepare(&context).unwrap();
    assert!(matches!(
        pair.fill_for_inspection([&keys, &values], true),
        Err(CacheSourceError::PendingStorage)
    ));
    assert!(
        pair.values().is_none(),
        "a failed readiness check consumes no alias slot"
    );
    safemlx::transforms::eval([&keys, &values]).unwrap();
    let identities =
        [&keys, &values].map(|array| array.try_allocation_info().unwrap().unwrap().identity());
    pair.fill_for_inspection([&keys, &values], true).unwrap();
    assert_eq!(
        pair.values().unwrap().map(|array| array
            .try_allocation_info()
            .unwrap()
            .unwrap()
            .identity()),
        identities
    );
    assert!(matches!(
        pair.fill_for_inspection([&keys, &values], true),
        Err(CacheSourceError::Identity)
    ));
    drop((keys, values, context, funding));
    assert!(ledger.fixture_host_charge().unwrap() > 0);
    assert_eq!(
        pair.values().unwrap()[1]
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        &[2.5, -5.0, 7.5, 9.0]
    );
    drop(pair);
    assert_eq!(ledger.fixture_host_charge().unwrap(), 0);
}
