//! Checked decomposition of a fixed controller allowance into source and work.

use super::{ControllerStorageContract, ControllerStorageError, RegisteredControllerStorage};
use crate::working_memory::{
    MemoryLedger, WorkingMemoryError, WorkingMemoryStorage, WorkspaceReportError,
    WorkspaceReportMetadata, residual::RegisteredStoragePin,
};
use eredu_core::{
    ExecutionWorkspaceEstimate, InferenceGeometry, TextControllerContract, TextControllerWorkspace,
    WorkspaceBound,
};

/// Controller source failure or fixed/paid metadata construction refusal.
#[derive(Debug, thiserror::Error)]
pub enum ControllerWorkspaceMetadataError {
    /// Existing source, contract or registration failure.
    #[error(transparent)]
    Controller(#[from] ControllerStorageError),
    /// Fixed geometry or counted destination failure.
    #[error(transparent)]
    Report(#[from] WorkspaceReportError),
}
impl ControllerWorkspaceMetadataError {
    /// Preserves the existing ordinary reporting error surface.
    pub fn into_legacy(self) -> ControllerStorageError {
        match self {
            Self::Controller(error) => error,
            Self::Report(error) => ControllerStorageError::Estimate(error.into_capability()),
        }
    }
    /// Retains a typed source only after its Context error-owner debit.
    pub fn into_workspace(self, metadata: WorkspaceReportMetadata<'_>) -> eredu_nn::Error {
        match self {
            Self::Controller(error) => metadata.source(error),
            Self::Report(error) => metadata.error(error),
        }
    }
}
impl From<WorkingMemoryError> for ControllerWorkspaceMetadataError {
    fn from(error: WorkingMemoryError) -> Self {
        Self::Controller(error.into())
    }
}

/// One complete controller contribution, optionally backed by registered sources.
///
/// Shared sources are an always-live part of the declared additional allowance.
/// Only their exact pinned capacities may be removed from that fixed component;
/// equation/sampling maxima and final emitted-mask allowances remain unchanged.
/// This object retains no masks and grants no allocation or execution authority.
#[derive(Debug, Clone)]
pub struct ControllerWorkspaceContribution {
    geometry: InferenceGeometry,
    controller: TextControllerContract,
    pool: MemoryLedger,
    full_additional: u64,
    incremental_additional: u64,
    registered: Option<WorkingMemoryStorage<eredu_core::SharedStorageIdentity>>,
}

impl ControllerWorkspaceContribution {
    /// Binds the full filter/source declaration and optional existing-only pin.
    /// A missing pin grants no source credit. A supplied pin must describe the
    /// exact storage contract and domain; matching byte totals are insufficient.
    pub fn new(
        geometry: InferenceGeometry,
        workspace: TextControllerWorkspace<'_>,
        output_width: usize,
        storage_contract: &ControllerStorageContract,
        pool: &MemoryLedger,
        registered: Option<&RegisteredControllerStorage>,
    ) -> Result<Self, ControllerStorageError> {
        Self::new_metadata(
            geometry,
            workspace,
            output_width,
            storage_contract,
            pool,
            registered,
            WorkspaceReportMetadata::ordinary(),
        )
        .map_err(ControllerWorkspaceMetadataError::into_legacy)
    }
    /// The same source validation with counted metadata destinations. Only the
    /// validated existing registration handle is retained; no contract map clone.
    pub fn new_metadata(
        geometry: InferenceGeometry,
        workspace: TextControllerWorkspace<'_>,
        output_width: usize,
        storage_contract: &ControllerStorageContract,
        pool: &MemoryLedger,
        registered: Option<&RegisteredControllerStorage>,
        metadata: WorkspaceReportMetadata<'_>,
    ) -> Result<Self, ControllerWorkspaceMetadataError> {
        geometry
            .validate_fixed()
            .map_err(WorkspaceReportError::from)?;
        metadata.admit::<Self>()?;
        metadata.admit::<ControllerWorkspaceMetadataError>()?;
        let controller = TextControllerContract::from_workspace(workspace, output_width)
            .map_err(ControllerStorageError::from)?;
        storage_contract.validate_workspace(workspace)?;
        storage_contract.validate_original_pool(pool)?;
        let source_bytes = if let Some(registered) = registered {
            if !registered.pool.same_ledger(pool) || &registered.contract != storage_contract {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            registered.source_bytes()
        } else {
            0
        };
        let full_additional = workspace
            .additional_host_bytes
            .checked_add(storage_contract.publication_control_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)?;
        let incremental_additional = full_additional
            .checked_sub(source_bytes)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        Ok(Self {
            geometry,
            controller,
            pool: pool.clone(),
            full_additional,
            incremental_additional,
            registered: registered.map(|source| source.registration.clone()),
        })
    }

    /// Original complete decision contract, with no source-storage discount.
    pub fn controller_contract(&self) -> &TextControllerContract {
        &self.controller
    }

    /// Adds full and incremental controller contributions to the same enclosing
    /// workspace before its peak is computed. The input must not already include
    /// this controller contribution. All other components remain identical;
    /// unknown bounds remain unknown, including an unknown retained component.
    pub fn compose(
        &self,
        outside: ExecutionWorkspaceEstimate,
    ) -> Result<ControllerWorkspaceEstimate, ControllerStorageError> {
        self.compose_metadata(outside, WorkspaceReportMetadata::ordinary())
            .map_err(ControllerWorkspaceMetadataError::into_legacy)
    }
    /// The same full/incremental controller terms, retaining current planning
    /// funding on every new independently escapable registration wrapper.
    pub fn compose_metadata(
        &self,
        outside: ExecutionWorkspaceEstimate,
        metadata: WorkspaceReportMetadata<'_>,
    ) -> Result<ControllerWorkspaceEstimate, ControllerWorkspaceMetadataError> {
        metadata.admit::<ControllerWorkspaceEstimate>()?;
        if outside.geometry != self.geometry {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let mut full = metadata.clone_execution(&outside)?;
        if let Some(domains) = &mut full.physical_domains {
            domains
                .retained
                .add_allocation(self.full_additional, self.pool.host_placement())
                .map_err(|e| WorkingMemoryError::from(e))?;
        }
        add_retained(&mut full.retained, self.full_additional, false, metadata)?;
        let incremental = if self.full_additional == self.incremental_additional {
            metadata.clone_execution(&full)?
        } else {
            let mut incremental = outside;
            if let Some(domains) = &mut incremental.physical_domains {
                domains
                    .retained
                    .add_allocation(self.incremental_additional, self.pool.host_placement())
                    .map_err(|e| WorkingMemoryError::from(e))?;
            }
            add_retained(
                &mut incremental.retained,
                self.incremental_additional,
                true,
                metadata,
            )?;
            incremental
        };
        let pin = self
            .registered
            .as_ref()
            .map(|registration| RegisteredStoragePin::new_metadata(registration.clone(), metadata))
            .transpose()
            .map_err(WorkspaceReportError::from)?;
        Ok(ControllerWorkspaceEstimate {
            full,
            incremental,
            controller: self.controller,
            pool: self.pool.clone(),
            pin,
        })
    }
}

fn add_retained(
    bound: &mut WorkspaceBound,
    additional: u64,
    excludes_registered_sources: bool,
    metadata: WorkspaceReportMetadata<'_>,
) -> Result<(), WorkspaceReportError> {
    if let WorkspaceBound::Bounded { bytes, assumptions } = bound {
        *bytes = bytes.checked_add(additional).ok_or(
            eredu_core::AdmissionPolicyError::ArithmeticOverflow {
                operation: "controller and retained request payload",
            },
        )?;
        metadata.append(
            assumptions,
            "; complete cold controller payload beyond the emitted filter priced by sampling",
        )?;
        if excludes_registered_sources {
            metadata.append(assumptions, "; exact registered shared sources excluded from this fixed controller contribution")?;
        }
    }
    Ok(())
}

/// Full diagnostics and checked incremental enclosing workspace from one input.
///
/// Construction is private so a reduced workspace cannot be substituted for an
/// unrelated full estimate. The pin must transfer into reservation/run/scope
/// custody before temporary quote metadata retires.
#[derive(Debug, Clone)]
pub struct ControllerWorkspaceEstimate {
    full: ExecutionWorkspaceEstimate,
    incremental: ExecutionWorkspaceEstimate,
    controller: TextControllerContract,
    pool: MemoryLedger,
    pin: Option<RegisteredStoragePin>,
}

impl ControllerWorkspaceEstimate {
    /// Complete enclosing workspace with the original controller allowance.
    pub fn full(&self) -> &ExecutionWorkspaceEstimate {
        &self.full
    }

    /// Same enclosing workspace, excluding only its pinned shared-source term.
    pub fn incremental(&self) -> &ExecutionWorkspaceEstimate {
        &self.incremental
    }

    /// Exact request geometry shared by both estimates.
    pub fn geometry(&self) -> InferenceGeometry {
        self.full.geometry
    }

    /// Original complete decision limits; admission credit never weakens them.
    pub fn controller_contract(&self) -> &TextControllerContract {
        &self.controller
    }

    pub(in crate::working_memory) fn pool(&self) -> &MemoryLedger {
        &self.pool
    }

    pub(in crate::working_memory) fn pin(&self) -> Option<RegisteredStoragePin> {
        self.pin.clone()
    }
}

#[cfg(test)]
mod tests;
