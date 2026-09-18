use eredu_nn::{workspace::*, Error};
use eredu_runtime::ArchitectureBoundary;
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};
#[derive(Debug)]
struct Account(Arc<Mutex<usize>>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let mut available = self.0.lock().unwrap();
        *available =
            available
                .checked_sub(bytes)
                .ok_or(HostMetadataFundingError::Capacity {
                    required: bytes as u64,
                    available: *available as u64,
                })?;
        Ok(())
    }
}
#[derive(Debug)]
struct NoTensors;
impl WorkspaceMechanisms for NoTensors {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
impl WorkspaceFactMechanisms for NoTensors {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(None)
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(None)
    }
}
pub(crate) fn check<S: ArchitectureBoundary>(schema: S, values: Vec<i32>) {
    let remaining = Arc::new(Mutex::new(8 * 1024 * 1024));
    let funding = HostMetadataFunding::new(Account(remaining.clone())).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(NoTensors, funding).unwrap();
    let before = *remaining.lock().unwrap();
    let ordinary = schema
        .encode(schema.decode(values.clone()).unwrap())
        .unwrap();
    let wire = schema.wire_schema_with_metadata(&context).unwrap();
    assert_eq!(wire, schema.wire_schema().unwrap());
    assert_eq!(
        wire.resolve_with_metadata(2, 3, &context).unwrap(),
        wire.resolve(2, 3).unwrap()
    );
    let actual = schema
        .encode_with_metadata(
            schema
                .decode_with_metadata(values.clone(), &context)
                .unwrap(),
            &context,
        )
        .unwrap();
    assert_eq!(actual.len(), ordinary.len());
    for (a, e) in actual.iter().zip(&ordinary) {
        assert_eq!(a.role(), e.role());
        assert_eq!(a.tensor(), e.tensor());
    }
    let mut invalid = values.clone();
    invalid.push(103);
    assert!(schema.decode_with_metadata(invalid, &context).is_err());
    assert!(*remaining.lock().unwrap() < before);
    assert_eq!(context.operation_count(), 0);
    // A fresh budget exhaustion refuses even an otherwise valid boundary.
    *remaining.lock().unwrap() = 0;
    assert!(schema.decode_with_metadata(values, &context).is_err());
    assert_eq!(*remaining.lock().unwrap(), 0);
}
