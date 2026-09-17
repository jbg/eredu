//! A type witness for the ordinary strategy's intercepted route body.
//!
//! WorkspaceBackend::with_expert_route_region records the exact retained active
//! region, and with_expert_inactive_wave records the original inactive wave.
//! Neither executes the imperative movement closure. Refuse entry before any
//! host index directory or tensor movement if that contract is ever violated.
use eredu_nn::workspace::{WorkspaceMetadataError, WorkspaceTensor};
use eredu_runtime::{ExpertRouteMovementSourceError, ExpertRouteTensorMovement, PreparedExpertMovementLoan};

pub(in super::super) struct SymbolicRouteMovement;
impl ExpertRouteTensorMovement<WorkspaceTensor> for SymbolicRouteMovement {
    type Error = WorkspaceMetadataError;

    fn with_prepared_region<R, E, F>(&mut self, _: Option<PreparedExpertMovementLoan<'_>>, _: F)
        -> Result<Result<R,E>, ExpertRouteMovementSourceError<Self::Error>>
    where F: FnOnce(&mut Self) -> Result<R,E> {
        Err(ExpertRouteMovementSourceError::MissingProducer)
    }
    fn validate_population(&self, _: eredu_nn::workspace::WorkspaceExpertMovementPopulation,
        _: eredu_nn::workspace::WorkspaceExpertTransfers) -> Result<(), Self::Error> {
        Err(WorkspaceMetadataError::Unqualified)
    }
    fn index_directory(&self, _: usize) -> Result<Vec<usize>, Self::Error> {
        Err(WorkspaceMetadataError::Unqualified)
    }
    fn shape(&self, _: &WorkspaceTensor) -> Vec<usize> {
        unreachable!("symbolic route source cannot enter imperative movement")
    }
    fn checked_shape(&self, _: &WorkspaceTensor) -> Result<Vec<usize>, Self::Error> {
        Err(WorkspaceMetadataError::Unqualified)
    }
    fn gather_rows(&mut self, _: &WorkspaceTensor, _: &[usize]) -> Result<WorkspaceTensor, Self::Error> {
        Err(WorkspaceMetadataError::Unqualified)
    }
    fn gather_route_values(&mut self, _: &WorkspaceTensor, _: &[usize]) -> Result<WorkspaceTensor, Self::Error> {
        Err(WorkspaceMetadataError::Unqualified)
    }
    fn scatter_add_rows(&mut self, _: WorkspaceTensor, _: &[usize], _: usize,
        _: eredu_nn::GroupReduction) -> Result<WorkspaceTensor, Self::Error> {
        Err(WorkspaceMetadataError::Unqualified)
    }
}
