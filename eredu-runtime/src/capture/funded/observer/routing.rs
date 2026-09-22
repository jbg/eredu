//! Source-bound routing controls and evidence use the ordinary funded frame.
use super::*;
use crate::working_memory::{CaptureInterventionClaim, InterventionPrefillCursor};
use eredu_core::intervention::InterventionOutcome;
use eredu_nn::routing_intervention::GroupSelectionControl;

enum Claim<'a> {
    Once(CaptureInterventionClaim<'a>),
    Prefill(
        InterventionPrefillCursor<'a>,
        crate::intervention::InterventionPrefillWindow,
    ),
}
impl<'a> Claim<'a> {
    fn claim(&self) -> &CaptureInterventionClaim<'a> {
        match self {
            Self::Once(claim) => claim,
            Self::Prefill(cursor, _) => cursor.claim(),
        }
    }
    fn window(&self) -> Option<crate::intervention::InterventionPrefillWindow> {
        match self {
            Self::Once(_) => None,
            Self::Prefill(_, window) => Some(*window),
        }
    }
}
pub(super) struct Pending<'a> {
    claim: Claim<'a>,
    rows: u64,
    charged: CaptureUsage,
    unmodified: bool,
}
impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(super) fn routing_control_value(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<GroupSelectionControl>, FundedCaptureError<E>> {
        if self.pending_routing.is_some() {
            return Err(CaptureProtocolError::Transaction.into());
        }
        let prefill = if self.prefill.is_some() {
            Some((
                self.bound
                    .ok_or(CaptureProtocolError::PrefillAttribution)?
                    .geometry(),
                self.current_fragment_chunk()?,
            ))
        } else {
            None
        };
        let Frame::Active(frame) = &mut self.frame else {
            return if matches!(self.frame, Frame::Empty) {
                Ok(None)
            } else {
                Err(CaptureProtocolError::Transaction.into())
            };
        };
        let Some(plan) = frame.intervention_admission() else {
            return Ok(None);
        };
        let Some(index) = plan
            .plan()
            .operations
            .iter()
            .zip(plan.points())
            .enumerate()
            .find_map(|(index, (operation, point))| {
                (operation.target == path
                    && point.routing.is_some()
                    && frame.interventions()[index].outcome != InterventionOutcome::Inactive)
                    .then_some(index)
            })
        else {
            return Ok(None);
        };
        if frame.interventions()[index].outcome != InterventionOutcome::Missing {
            return Err(CaptureProtocolError::Transaction.into());
        }
        let claim = match &prefill {
            Some((geometry, chunk)) => {
                let window =
                    crate::intervention::InterventionPrefillWindow::new(plan, *geometry, chunk)
                        .map_err(|_| CaptureProtocolError::PrefillAttribution)?;
                Claim::Prefill(frame.take_prefill_intervention(index, window)?, window)
            }
            None => Claim::Once(frame.take_intervention(index)?),
        };
        let mut charged = CaptureUsage::default();
        let result = (|| {
            let usage =
                self.backend
                    .routing_intervention_usage(rows, claim.claim(), claim.window())?;
            if usage.captures != 0 || usage.encoded_bytes != 0 {
                return Err(CaptureProtocolError::Transaction.into());
            }
            crate::intervention::reserve_envelope(&mut self.session.ledger, usage)?;
            charged = usage;
            self.backend
                .prepare_routing_intervention(rows, claim.claim(), claim.window())
        })();
        match result {
            Ok(control) => {
                self.pending_routing = Some(Pending {
                    claim,
                    rows,
                    charged,
                    unmodified: control.is_none(),
                });
                Ok(control)
            }
            Err(error) => {
                frame.record_intervention_failure(
                    index,
                    "routing control preparation failed",
                    charged,
                )?;
                Err(error)
            }
        }
    }
    pub(super) fn routing_unmodified_interest_value(
        &self,
        path: &str,
    ) -> crate::RoutingUnmodifiedInterest {
        if self.pending_routing.as_ref().is_some_and(|pending| {
            pending.unmodified
                && pending.claim.claim().admission().plan().operations
                    [pending.claim.claim().index()]
                .target
                    == path
        }) {
            crate::RoutingUnmodifiedInterest::Metadata
        } else {
            crate::RoutingUnmodifiedInterest::None
        }
    }
    pub(super) fn routing_applied_value(
        &mut self,
        path: &str,
        original: Option<crate::RoutingDecision<'_, T>>,
        effective: crate::RoutingDecision<'_, T>,
        unmodified: bool,
    ) -> Result<(), FundedCaptureError<E>> {
        let pending = self
            .pending_routing
            .take()
            .ok_or(CaptureProtocolError::Transaction)?;
        let index = pending.claim.claim().index();
        if pending.unmodified != unmodified
            || pending.claim.claim().admission().plan().operations[index].target != path
        {
            return Err(CaptureProtocolError::Transaction.into());
        }
        let prefill = if self.prefill.is_some() {
            Some((
                self.bound
                    .ok_or(CaptureProtocolError::PrefillAttribution)?
                    .geometry(),
                self.current_fragment_chunk()?,
            ))
        } else {
            None
        };
        let Frame::Active(frame) = &mut self.frame else {
            return Err(CaptureProtocolError::Transaction.into());
        };
        let result = (|| {
            self.backend.validate_routing_intervention_result(
                pending.rows,
                original.as_ref().map(|v| crate::RoutingDecision {
                    ids: v.ids,
                    coefficients: v.coefficients,
                }),
                crate::RoutingDecision {
                    ids: effective.ids,
                    coefficients: effective.coefficients,
                },
                pending.claim.claim(),
                pending.claim.window(),
            )?;
            let evidence = pending.claim.claim().admission().plan().operations[index].evidence
                != eredu_core::intervention::InterventionEvidence::None;
            if evidence {
                let original = original.as_ref().ok_or(CaptureProtocolError::Transaction)?;
                for (side, field, value) in [
                    (
                        InterventionEvidenceSide::Before,
                        eredu_core::RoutingObservationField::SelectedExperts,
                        original.ids,
                    ),
                    (
                        InterventionEvidenceSide::Before,
                        eredu_core::RoutingObservationField::Coefficients,
                        original.coefficients,
                    ),
                    (
                        InterventionEvidenceSide::After,
                        eredu_core::RoutingObservationField::SelectedExperts,
                        effective.ids,
                    ),
                    (
                        InterventionEvidenceSide::After,
                        eredu_core::RoutingObservationField::Coefficients,
                        effective.coefficients,
                    ),
                ] {
                    match &prefill {
                        Some((geometry, chunk)) => {
                            interventions::evidence::observe_routing_prefill(
                                self.backend,
                                frame,
                                &mut self.session.ledger,
                                index,
                                side,
                                field,
                                value,
                                *geometry,
                                chunk,
                            )?
                        }
                        None => interventions::evidence::observe_routing(
                            self.backend,
                            frame,
                            &mut self.session.ledger,
                            index,
                            side,
                            field,
                            value,
                        )?,
                    }
                }
            }
            match pending.claim {
                Claim::Once(claim) => frame.record_intervention(if unmodified {
                    claim.finish_unmatched(pending.charged)?
                } else {
                    claim.finish(pending.charged)?
                })?,
                Claim::Prefill(mut cursor, window) => {
                    let mut fragment = cursor.begin(window)?;
                    fragment.charge(pending.charged)?;
                    if unmodified {
                        fragment.finish_unmatched()?;
                    } else {
                        fragment.finish()?;
                    }
                    frame.retain_prefill_intervention(cursor)?;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = frame.record_intervention_failure(
                index,
                "routing result or evidence failed",
                pending.charged,
            );
        }
        result
    }
    pub(super) fn routing_failed_value(&mut self, path: &str, message: &str) {
        let Some(pending) = self.pending_routing.take() else {
            return;
        };
        let index = pending.claim.claim().index();
        if pending.claim.claim().admission().plan().operations[index].target != path {
            return;
        }
        if let Frame::Active(frame) = &mut self.frame {
            let _ = frame.record_intervention_failure(index, message, pending.charged);
        }
    }
}

/// Fixed protocol frames; native tensor/error values remain backend obligations.
pub(in crate::capture::funded) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<Pending<'_>>(),
        size_of::<Claim<'_>>(),
        size_of::<Option<GroupSelectionControl>>(),
        size_of::<
            Result<Option<GroupSelectionControl>, FundedCaptureError<std::convert::Infallible>>,
        >(),
        size_of::<[crate::RoutingDecision<'_, ()>; 2]>(),
        size_of::<
            [(
                InterventionEvidenceSide,
                eredu_core::RoutingObservationField,
                &(),
            ); 4],
        >(),
        size_of::<(u64, CaptureUsage, bool)>(),
        size_of::<Option<(eredu_core::InferenceGeometry, crate::prefill::PrefillChunk)>>(),
        size_of::<Result<(), FundedCaptureError<std::convert::Infallible>>>(),
        size_of::<(
            &mut FundedCaptureObserver<'_, (), std::convert::Infallible, ()>,
            &str,
            u64,
        )>(),
        size_of::<(
            &mut FundedCaptureObserver<'_, (), std::convert::Infallible, ()>,
            &str,
            Option<crate::RoutingDecision<'_, ()>>,
            crate::RoutingDecision<'_, ()>,
            bool,
        )>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
