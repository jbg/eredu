//! Public parameter experiments use the same loaded model and generation drivers.
use super::{observed::new_identity, LoadedModel};
use eredu_core::{capture::CaptureUsage, parameters::*};

impl<B: ParameterBackend> LoadedModel<B> {
    /// Discovers actual loaded parameter access and active edit provenance.
    /// Distributed models require all ranks to participate in the same order;
    /// bounded metadata exchange contributes to the reported cumulative usage.
    pub fn parameter_discovery(&mut self) -> Result<ParameterDiscovery, ParameterError> {
        B::parameter_discovery(&mut self.runtime)
    }
    /// Obtains a bounded effective read row, write column or normalization region.
    /// Limits are cumulative for this loaded session and are never refunded by reset.
    pub fn query_parameter(
        &mut self,
        identity: &str,
        parameter: &str,
        region: ParameterRegion,
        limits: CaptureUsage,
    ) -> Result<ParameterValues, ParameterError> {
        B::query_parameter(&mut self.runtime, identity, parameter, region, limits)
    }
    /// Admits an immutable multi-parameter edit against current loaded facts.
    pub fn admit_parameter_overlay(
        &mut self,
        plan: ParameterOverlayPlan,
    ) -> Result<AdmittedParameterOverlay, ParameterError> {
        AdmittedParameterOverlay::admit(plan, &self.parameter_discovery()?)
    }
    /// Projects selected effective weights into token directions or downstream read rows.
    /// Only the contracted result is copied to the host; limits include native temporaries.
    pub fn project_parameter(
        &mut self,
        identity: &str,
        parameter: &str,
        projection: ParameterProjection,
        limits: CaptureUsage,
    ) -> Result<ParameterProjectionValues, ParameterError> {
        B::project_parameter(&mut self.runtime, identity, parameter, projection, limits)
    }
    /// Atomically installs prepared replacements and clears incompatible model state.
    /// Previously prepared generation requests become stale; prepare new exact-token trials.
    /// Distributed ranks participate in matching order with locally admitted versions
    /// of the same global plan. Returned edit provenance uses their common intent digest.
    pub fn activate_parameter_overlay(
        &mut self,
        overlay: &AdmittedParameterOverlay,
        limits: CaptureUsage,
    ) -> Result<ParameterDiscovery, ParameterError> {
        let result = B::activate_parameter_overlay(&mut self.runtime, overlay, limits)?;
        self.session_identity = new_identity("parameter-session");
        Ok(result)
    }
    /// Restores original parameters, clears incompatible state and invalidates old requests.
    /// Distributed removal participates on every rank and charges its bounded state
    /// initialization and metadata work; it does not reload weight sources.
    pub fn remove_parameter_overlay(
        &mut self,
        identity: &str,
    ) -> Result<ParameterDiscovery, ParameterError> {
        let result = B::remove_parameter_overlay(&mut self.runtime, identity)?;
        self.session_identity = new_identity("parameter-session");
        Ok(result)
    }
}
