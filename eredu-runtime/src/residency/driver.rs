//! Portable ownership of an installed residency controller and its native resources.

use super::{ResidencyAccess, ResidencyInstallation, ResidencyOwner, ResidentLease};
use eredu_core::residency::ResidencyStorageMechanisms;
use eredu_core::residency::{
    conversion_claim::ConversionRetentionError, ParameterConversionRetentionTrimReport,
    ParameterConversionTrimError, ResidencyStorage,
};
use eredu_core::BackendFailure;
use std::sync::Arc;

/// Shares one portable ledger and the native resources attached at installation.
pub struct ResidencyDriver<N: ResidencyStorageMechanisms> {
    owner: Arc<ResidencyOwner<N::Storage>>,
    native: N,
}

impl<N: ResidencyStorageMechanisms + Clone> Clone for ResidencyDriver<N> {
    fn clone(&self) -> Self {
        Self {
            owner: Arc::clone(&self.owner),
            native: self.native.clone(),
        }
    }
}

impl<N: ResidencyStorageMechanisms> ResidencyDriver<N> {
    /// Observes installed native recovery and resource metadata.
    pub fn native_resources(&self) -> &N {
        &self.native
    }

    /// Read-only storage access without progressing native work.
    pub fn observation_lock(
        &self,
    ) -> Result<impl std::ops::Deref<Target = super::ResidencyState<N::Storage>> + '_, N::Error>
    {
        Ok(self.owner.observe().map_err(BackendFailure::from_error)?)
    }

    /// Attaches permanent resources to the already selected retention budget.
    pub fn enable_resident_parameter_conversions(&self) -> Result<(), N::Error> {
        let mut state = self.lock_storage()?;
        let budget = state
            .controller()
            .conversion_retention()
            .expect("installed retention budget")
            .clone();
        N::register_conversion_resources(state.native_mut(), &budget)
    }

    /// Releases optional claims after the session establishes completion.
    pub fn trim_parameter_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        let state = self.lock_storage().map_err(BackendFailure::from_error)?;
        let budget = state
            .controller()
            .conversion_retention()
            .expect("installed retention budget");
        N::trim_conversion_resources(budget)
            .map(|report| vec![report])
            .map_err(|error| match error {
                ConversionRetentionError::Busy => ParameterConversionTrimError::NotQuiescent,
                error => BackendFailure::from_error(error).into(),
            })
    }

    /// Resolves an admitted compact range without opening payloads.
    pub fn row_range(
        &self,
        prefix: &eredu_core::residency::OffloadUnitId,
    ) -> Option<&crate::RowResidencyRange> {
        self.owner.sources().row_range(prefix)
    }
}

impl<N: ResidencyStorageMechanisms> ResidencyAccess for ResidencyDriver<N> {
    type Storage = N::Storage;
    type Error = N::Error;
    type Transfer = N::Transfer;

    fn residency_owner(&self) -> &Arc<ResidencyOwner<N::Storage>> {
        &self.owner
    }

    fn progress_native_residency(&self) -> Result<(), Self::Error> {
        self.native.progress()
    }

    fn transfer(
        leases: Vec<ResidentLease<N::Storage>>,
        submitted: Option<<N::Storage as ResidencyStorage>::Transfer>,
    ) -> Self::Transfer {
        N::transfer(leases, submitted)
    }
}

impl<N: ResidencyStorageMechanisms> ResidencyInstallation for ResidencyDriver<N> {
    type Context = N::Context;
    type Resources = N;

    fn open_storage(
        context: Self::Context,
    ) -> Result<(Self::Storage, Self::Resources), Self::Error> {
        N::open_storage(context)
    }

    fn retain(owner: Arc<ResidencyOwner<N::Storage>>, native: N) -> Self {
        Self { owner, native }
    }
}
