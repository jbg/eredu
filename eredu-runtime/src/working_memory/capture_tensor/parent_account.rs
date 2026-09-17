//! Protect one closed capture destination inside its original request account.
use super::*;
use crate::working_memory::{WorkingMemoryFundingRun, WorkingMemoryReservation};

impl WorkingMemoryFundingRun {
    /// Allocates one exact host destination from this original request's reserve.
    ///
    /// `reservation` must be the opaque metadata returned with this same funding
    /// run. The checked plan derives P from actual admitted capture geometry;
    /// neither a scalar byte allowance nor an allocation callback is accepted.
    /// P becomes protected host storage under the same lock that validates the
    /// account. Native publication and other host destinations cannot spend it.
    /// No additional reservation or native work scope is created.
    ///
    /// This is physical host custody only. The enclosing quoted worker must
    /// authenticate the capture plan, current invocation and source, include all
    /// simultaneously surviving destination holds in its original quote, spend
    /// capture quota and fund native transformations/transfer separately. Calling
    /// this constructor does not install instrumentation, renew a request stage,
    /// authorize a source read, or establish native completion. Repeating it
    /// consumes further protected capacity; it does not reset logical quotas.
    ///
    /// Allocation, scalar fill and finish reject closed or quarantined parents.
    /// An already completed read-only alias may outlive the run and retains its
    /// original hold. Host cleanup releases only its internally minted host
    /// scope; the caller's independent native scope still requires settlement.
    pub fn prepare_capture_tensor<'a>(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: CaptureTensorHostPlan<'a>,
    ) -> Result<PreparedCaptureTensor<'a>, CaptureTensorConstructionError> {
        let custody = self.hold_capture_tensor(reservation, &plan)?;
        allocate(plan, custody)
    }
}
