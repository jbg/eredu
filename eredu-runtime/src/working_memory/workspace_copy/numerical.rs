//! A distinct copy source for an actual completed original numerical value.
use super::super::OriginalSpeculativeNumericalBudgetCustody;
use super::*;
use eredu_core::HostPreparationAuthority;
use std::mem::{size_of, size_of_val};

/// Closed isolated-copy demand retaining the original numerical account. It
/// cannot enter registered-source sampler aggregates or fabricate a native key.
/// The backend must authenticate every projected source root against the exact
/// native budget constructed from this custody, after successful completion.
#[derive(Debug)]
pub struct OriginalNumericalWorkspaceCopy {
    copy: OriginalCompletedWorkspaceCopy<()>,
}
impl OriginalNumericalWorkspaceCopy {
    /// Binds the sealed shared copy trace to its actual source Q. This grants no
    /// source adoption: native source/budget authentication remains mandatory.
    /// The complete alias backing, including a selected split table, must fit
    /// the retained physical population; a logical two-word shape is insufficient.
    pub fn bind(
        plan: WorkspaceIsolatedCopyPlan,
        roots: &eredu_nn::workspace::WorkspaceBorrowedStorage,
        source: OriginalSpeculativeNumericalBudgetCustody,
        host: HostPreparationAuthority,
    ) -> Result<Self, WorkspaceCopyAdmissionError> {
        let source = OriginalCompletedWorkspaceSource::numerical(roots, source, host)?;
        Ok(Self {
            copy: OriginalCompletedWorkspaceCopy::only_completed(plan, source)?,
        })
    }

    /// Exact additional source/transport frames. Actual copy account and native
    /// domain constructors retain their existing independently queried controls.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, WorkspaceCopyAdmissionError>>(),
            size_of::<OriginalSpeculativeNumericalBudgetCustody>(),
            size_of::<RegisteredStoragePin>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<(&MemoryLedger, &OriginalSpeculativeNumericalBudgetCustody)>(),
            size_of::<(&MemoryLedger, WorkspaceCopyLimits)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)?
            .checked_add(OriginalCompletedWorkspaceCopy::<()>::control_bytes().ok()?)
    }
}
impl MemoryLedger {
    /// Uses the same validated-source copy account worker and exact capacity
    /// comparison. The numerical source's cumulative request charge is unchanged.
    pub fn admit_numerical_workspace_copy(
        &self,
        copy: OriginalNumericalWorkspaceCopy,
        limits: WorkspaceCopyLimits,
    ) -> Result<AdmittedWorkspaceCopy, WorkspaceCopyAdmissionError> {
        self.admit_completed_workspace_copy(copy.copy, limits)
    }
}
