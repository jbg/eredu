//! Capture joins the same one-shot AR role admission and cumulative source ledger.
use super::*;
use crate::working_memory::{AutoregressiveCaptureHostPlan, PreparedAutoregressiveCapture};
impl OriginalSpeculativeRequest {
    /// Admit every exact capture frame with one genuine AR occurrence. Host
    /// frames share its source-validated request ledger; no Embedded role is made.
    pub fn reserve_role_with_capture<'a>(
        &self,
        claim: AutoregressiveOccurrenceClaim<'_>,
        requirements: SpeculativeInvocationRequirements,
        capture: AutoregressiveCaptureHostPlan<'a>,
    ) -> Result<(OriginalSpeculativeRole, PreparedAutoregressiveCapture<'a>), SpeculativeRequestError>
    {
        let (role, capture) = self.reserve_role_inner(claim, requirements, Some(capture))?;
        Ok((role, capture.expect("supplied capture plan")))
    }
    pub(super) fn reserve_role_inner<'a>(
        &self,
        claim: AutoregressiveOccurrenceClaim<'_>,
        mut requirements: SpeculativeInvocationRequirements,
        capture: Option<AutoregressiveCaptureHostPlan<'a>>,
    ) -> Result<
        (
            OriginalSpeculativeRole,
            Option<PreparedAutoregressiveCapture<'a>>,
        ),
        SpeculativeRequestError,
    > {
        if let Some(capture) = &capture {
            capture.validate(claim.invocation(), &requirements.plan, self.ticket.pool())?;
            requirements.controls = requirements
                .controls
                .checked_add(capture.initialization_peak_bytes())
                .ok_or(WorkingMemoryError::Overflow)?;
            requirements.host_bytes()?;
        }
        let ScheduleIdentity::Autoregressive(identity) = self.identity else {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        };
        let mut slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed
            || !claim.belongs_to(identity)
            || claim.ordinal() < slots.next
            || claim.ordinal() >= slots.limit
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        if let Some(capture) = &capture {
            capture.validate_lineage(slots.model_capture.as_ref())?;
        }
        let ordinal = claim.ordinal();
        slots.next = ordinal.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        let geometry = requirements.plan.geometry();
        let chunk = std::num::NonZeroU64::new(geometry.prefill_chunk_positions)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let expected = claim
            .schedule()
            .workspace_geometry(claim.frontier(), claim.invocation(), chunk)
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        let expected = capture
            .as_ref()
            .map_or(expected, |capture| capture.expected_geometry(expected));
        if geometry != expected {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let (plan, account) =
            self.accept_role_account(ordinal, requirements, role_control_bytes()?)?;
        let role = OriginalSpeculativeRole {
            plan,
            invocation: claim.invocation(),
            frontier: claim.frontier(),
            account,
        };
        let capture = capture.map(|capture| {
            if slots.model_capture.is_none() {
                slots.model_capture =
                    Some(crate::working_memory::capture_run::embedded::Cumulative {
                        source: capture.source().plan().storage_identity().clone(),
                        ledger: CaptureRunLedger::new_model(role.budget_custody()),
                    });
            }
            capture.bind(
                role.clone(),
                slots
                    .model_capture
                    .as_ref()
                    .expect("installed capture lineage")
                    .ledger
                    .clone(),
            )
        });
        Ok((role, capture))
    }
}
