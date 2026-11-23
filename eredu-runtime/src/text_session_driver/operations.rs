//! Portable observations of installed text residency.

use super::{TextSessionDriver, TextSessionResources};
use crate::{executable_binding::InstalledTextExecutable, expert::ParameterBanksResidencyReport};
use crate::{CacheResidencyReport, DenseDiskStreamReport, ResidencyReport};
use eredu_core::{
    backend::TextExecutableTypes, session_authority::SessionRetirement, BackendFailure,
};

/// Read-only portable observations of the installed text executable.
pub trait TextResidencyTelemetry<R>: SessionRetirement {
    /// Bounded ordinary parameter residency, when available.
    fn residency_report(resources: &R) -> Result<Option<ResidencyReport>, Self::Error>;
    /// Separately owned component residency ledgers.
    fn component_residency_reports(resources: &R) -> Result<Vec<ResidencyReport>, Self::Error>;
    /// Dense shard streaming telemetry.
    fn dense_stream_report(resources: &R) -> Result<Option<DenseDiskStreamReport>, Self::Error>;
    /// Sparse parameter bank telemetry.
    fn parameter_bank_report(
        resources: &R,
    ) -> Result<Option<ParameterBanksResidencyReport>, Self::Error>;
    /// Mutable cache residency telemetry.
    fn cache_residency_report(resources: &R) -> Result<Option<CacheResidencyReport>, Self::Error>;
}

impl<R, N, E, T, V, P> TextResidencyTelemetry<R> for N
where
    R: std::ops::Deref<Target = TextSessionResources<InstalledTextExecutable<E>, T, V, P>>,
    N: SessionRetirement<Error = E::Error>,
    E: TextExecutableTypes,
{
    fn residency_report(resources: &R) -> Result<Option<ResidencyReport>, Self::Error> {
        resources.model.residency_report()
    }

    fn component_residency_reports(resources: &R) -> Result<Vec<ResidencyReport>, Self::Error> {
        resources.model.component_residency_reports()
    }

    fn dense_stream_report(resources: &R) -> Result<Option<DenseDiskStreamReport>, Self::Error> {
        resources.model.dense_stream_report()
    }

    fn parameter_bank_report(
        resources: &R,
    ) -> Result<Option<ParameterBanksResidencyReport>, Self::Error> {
        resources.model.parameter_bank_report()
    }

    fn cache_residency_report(resources: &R) -> Result<Option<CacheResidencyReport>, Self::Error> {
        resources
            .model
            .cache_residency_report()
            .map_err(|error| BackendFailure::from_error(error).into())
    }
}

impl<R: 'static, N: TextResidencyTelemetry<R>, G> TextSessionDriver<R, N, G> {
    /// Returns bounded ordinary parameter residency when available.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, N::Error> {
        N::residency_report(self.resources())
    }

    /// Reports separately owned component residency ledgers.
    pub fn component_residency_reports(&self) -> Result<Vec<ResidencyReport>, N::Error> {
        N::component_residency_reports(self.resources())
    }

    /// Returns dense shard streaming telemetry when enabled.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, N::Error> {
        N::dense_stream_report(self.resources())
    }

    /// Returns sparse parameter bank telemetry when enabled.
    pub fn parameter_bank_report(&self) -> Result<Option<ParameterBanksResidencyReport>, N::Error> {
        N::parameter_bank_report(self.resources())
    }

    pub(super) fn text_expert_cache_telemetry(
        &self,
    ) -> Result<Option<eredu_core::ExpertCacheTelemetry>, BackendFailure> {
        self.parameter_bank_report()
            .map(|report| {
                report.map(|report| eredu_core::ExpertCacheTelemetry {
                    owned_experts: report.owned_entries(),
                    owned_bytes: report.owned_bytes(),
                    host_resident_experts: report.host_resident_entries(),
                    device_resident_experts: report.device_resident_entries(),
                    host_resident_bytes: report.host_resident_bytes(),
                    device_resident_bytes: report.device_resident_bytes(),
                    peak_host_resident_bytes: report.peak_host_resident_bytes(),
                    peak_device_resident_bytes: report.peak_device_resident_bytes(),
                })
            })
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("expert-cache telemetry")
            })
    }

    /// Returns mutable cache residency telemetry.
    pub fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, N::Error> {
        N::cache_residency_report(self.resources())
    }
}
