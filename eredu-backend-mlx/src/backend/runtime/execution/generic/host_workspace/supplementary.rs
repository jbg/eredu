//! Source projection for exact supplementary module invocations.
use super::*;
use crate::backend::runtime::residency::manager::SupplementaryResidencySource;
use std::mem::{size_of, size_of_val};

impl LayerwiseWorkspace {
    /// Describes the manager-owned singleton source inventory. The consuming
    /// invocation bank must still authenticate actual module ID, task/layout,
    /// call order, role and selected state before any transfer or equation.
    pub(crate) fn from_supplementary_source(
        manager: &ResidencyManager,
        source: &SupplementaryResidencySource,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<(
                &ResidencyManager,
                &SupplementaryResidencySource,
                &WorkspaceContext,
            )>(),
            size_of::<LayerwiseWorkspaceIdentity>(),
            size_of::<LayerwiseGeometry>(),
            size_of::<LayerwiseCopies>(),
            size_of::<LayerwiseMaterialization>(),
            size_of::<
                Result<(), crate::backend::runtime::residency::manager::OperationSourceFailure>,
            >(),
            size_of::<
                Result<(), crate::backend::runtime::residency::manager::HostCopyWorkspaceError>,
            >(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|bytes| bytes.checked_add(source.projection_control_bytes()?))
            .ok_or_else(|| {
                Error::Neural(eredu_nn::workspace::WorkspaceMetadataError::Overflow.into())
            })?;
        context
            .charge_metadata(bytes)
            .map_err(|cause| Error::Neural(cause.into()))?;
        manager
            .validate_supplementary_source(source)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let (geometry, copies, materialization) = if let Some(host) = source.host() {
            manager
                .validate_supplementary_host(source)
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
            let identity = host.prepared_identity().ok_or_else(|| {
                Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )
            })?;
            (
                LayerwiseGeometry::PreparedHost(identity.clone()),
                LayerwiseCopies::Host(host.clone()),
                LayerwiseMaterialization::PreparedHost(identity.clone()),
            )
        } else if let Some(identity) = source.foreground() {
            (
                LayerwiseGeometry::PreparedForeground(identity.clone()),
                LayerwiseCopies::Foreground(identity.clone()),
                LayerwiseMaterialization::PreparedForeground(identity.clone()),
            )
        } else {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        };
        Ok(Self {
            manager: manager.clone(),
            identity: LayerwiseWorkspaceIdentity {
                policy: source.policy_identity().clone(),
                parameter_locations: None,
                excluded: None,
                // Supplementary ordinals address their own complete rows,
                // never the main manager's execution-unit constructor table.
                manager_unit_constructors: false,
                geometry,
            },
            copies,
            materialization,
            persistent_roots: RefCell::new(None),
            execution_trace: RefCell::new(None),
            speculative_foreground: std::cell::OnceCell::new(),
        })
    }
    pub(crate) fn validate_supplementary_policy(
        &self,
        manager: &ResidencyManager,
        source: &SupplementaryResidencySource,
    ) -> Result<(), Error> {
        manager.validate_supplementary_source(source).map_err(|_| {
            Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )
        })?;
        self.validate_original_policy(manager, source.policy_identity(), source.ids())
    }
}
