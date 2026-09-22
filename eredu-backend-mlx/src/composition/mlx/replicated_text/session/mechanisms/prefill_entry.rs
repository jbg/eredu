use super::*;
use crate::backend::submission_recovery::prefill::{
    NativeReservationGuard, PrefillControlProjection, ReservationGuard,
};
use eredu_runtime::{prefill::PrefillControlRole, working_memory::InferenceRequest};

impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    pub(super) fn enter_prefill_retention(
        &mut self,
        reservation: InferenceRequest,
        role: Option<PrefillControlRole>,
        coordinate: bool,
    ) -> Result<ReservationGuard, Error> {
        // Release the bank's RefCell loan before waiting for a foreign runtime
        // owner. Installed roles retain their exact existing nonblocking path.
        let installed = self
            .prefill_controls
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .clone();
        let loan = if coordinate && installed.is_none() {
            Some(safemlx::OrdinarySubmissionEntry::enter()?)
        } else {
            None
        };
        let parent = crate::backend::nn::tensor::TokenValidationParent::capture()?;
        let result = match (role, installed) {
            (None, Some(PrefillControlProjection::Model(model))) => model
                .begin(&reservation)
                .map(|(guard, roots)| (NativeReservationGuard::Model(guard), Some(roots))),
            (Some(role), Some(PrefillControlProjection::Prefill(bank))) => bank
                .begin(&reservation, role)
                .map(|(guard, roots)| (NativeReservationGuard::Prefill(guard), roots)),
            (_, Some(_)) => Err(Error::PrefillScopeUnavailable),
            (_, None) => crate::backend::submission_recovery::prefill::begin_ordinary(reservation)
                .map(|(guard, roots)| (NativeReservationGuard::Prefill(guard), roots)),
        };
        // End coordination on success and failure before roots/error ownership
        // can retire or any later phase can wait on a worker or resource lock.
        drop(loan);
        let (guard, roots) = result?;
        let guard = ReservationGuard::new(guard, parent)?;
        let old = std::mem::replace(&mut self.prefill_roots, roots);
        drop(old);
        Ok(guard)
    }
}
