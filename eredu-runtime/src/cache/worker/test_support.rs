use eredu_nn::{
    Error,
    workspace::{
        WorkspaceContext, WorkspaceMechanisms, WorkspaceMetadataAccount, WorkspaceMetadataFunding,
        WorkspaceMetadataFundingError, WorkspaceOperation, WorkspaceOperationBound,
    },
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
#[derive(Debug, Default)]
pub(in crate::cache) struct AccountState {
    pub(in crate::cache) remaining: Mutex<usize>,
    pub(in crate::cache) retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<AccountState>);
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        let mut remaining = self.0.remaining.lock().unwrap();
        *remaining =
            remaining
                .checked_sub(bytes)
                .ok_or(WorkspaceMetadataFundingError::Capacity {
                    required: bytes as u64,
                    available: *remaining as u64,
                })?;
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("catalog storage runs no equations")
    }
}
pub(in crate::cache) fn context() -> (WorkspaceContext, Arc<AccountState>) {
    let state = Arc::new(AccountState {
        remaining: Mutex::new(usize::MAX),
        retired: AtomicBool::new(false),
    });
    let funding = WorkspaceMetadataFunding::new(Account(state.clone())).unwrap();
    (
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding).unwrap(),
        state,
    )
}
impl eredu_nn::workspace::WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("catalog preparation cannot quote equations")
    }
    fn write_operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceEffectDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("catalog preparation cannot emit equations")
    }
    fn host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("catalog preparation cannot quote host equations")
    }
    fn write_host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceHostDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("catalog preparation cannot emit host equations")
    }
}
