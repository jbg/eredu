//! Model control sources recorded in the same ordered equation trace.
use super::*;

/// An actual partition-driver phase, independent of session lifecycle phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceModelControlPhase {
    /// A selected group finished its local model execution.
    Execution,
    /// Sources for one selected boundary wave reached completion.
    BoundarySourceCompletion {
        /// Opaque retained route identity.
        route: u64,
    },
    /// All members accepted that boundary wave before transfer.
    BoundarySourceReady {
        /// Opaque retained route identity.
        route: u64,
    },
}
/// Exact selected group and phase. This is descriptive and has no producing authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceModelControl {
    /// Actual selected status-agreement group.
    pub group: eredu_core::CollectiveGroupId,
    /// Actual reached model phase, including its retained route when applicable.
    pub phase: WorkspaceModelControlPhase,
}
impl WorkspaceContext {
    /// Records a control occurrence without constructing tensor inputs or outputs.
    /// The selected mechanism must independently qualify its native control source.
    pub fn record_model_control(&self, control: WorkspaceModelControl) -> Result<(), Error> {
        self.charge_metadata(std::mem::size_of::<(
            &Self,
            WorkspaceModelControl,
            WorkspaceOperationKind,
            Vec<WorkspaceLayout>,
            Vec<WorkspaceTensor>,
            Result<(), Error>,
        )>())?;
        let outputs = self.execute(
            WorkspaceOperationKind::CommunicationControl(control),
            &[],
            Vec::new(),
        )?;
        debug_assert!(outputs.is_empty());
        Ok(())
    }
}
