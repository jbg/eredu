//! Existing sparse operation order with one spent cursor over actual batches.
use super::*;
use crate::intervention::InterventionPrefillWindow;
use eredu_core::intervention::InterventionOutcome;
fn same_edit(
    plan: &eredu_core::intervention::AdmittedInterventionPlan,
    first: usize,
    index: usize,
) -> bool {
    match (
        plan.points()
            .get(first)
            .and_then(|point| point.routed_units.as_ref()),
        plan.points()
            .get(index)
            .and_then(|point| point.routed_units.as_ref()),
    ) {
        (Some(a), Some(b)) => a.routing == b.routing,
        _ => false,
    }
}
impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(super) fn intervene_routed_batch(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, T>,
    ) -> Result<Option<T>, FundedCaptureError<E>> {
        let Some(first) = self.routed_intervention_selection else {
            return Ok(None);
        };
        if !self.routed_active || batch.origins.is_some() || batch.unit_coordinates.is_some() {
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
        let source = batch.capture_source()?;
        let Frame::Active(frame) = &mut self.frame else {
            return Err(CaptureProtocolError::Transaction.into());
        };
        let plan = frame
            .intervention_admission()
            .ok_or(CaptureProtocolError::Transaction)?;
        let window = prefill
            .as_ref()
            .map(|(geometry, chunk)| {
                InterventionPrefillWindow::new(plan, *geometry, chunk)
                    .map_err(|_| CaptureProtocolError::PrefillAttribution)
            })
            .transpose()?;
        let source_tokens = if let Some(window) = window {
            window.logical_positions()
        } else if let Some((_, shape)) = self.invocation {
            shape
                .batch
                .checked_mul(shape.sequence)
                .ok_or(CaptureError::Overflow)?
        } else {
            plan.request()
                .batch
                .checked_mul(if self.prediction == 0 {
                    plan.request().prompt_tokens
                } else {
                    1
                })
                .ok_or(CaptureError::Overflow)?
        };
        let mut effective = None;
        for index in 0..plan.plan().operations.len() {
            if !same_edit(plan, first, index) {
                continue;
            }
            match frame
                .interventions()
                .get(index)
                .ok_or(CaptureProtocolError::Transaction)?
                .outcome
            {
                InterventionOutcome::Inactive => continue,
                InterventionOutcome::Missing => (),
                _ => return Err(CaptureProtocolError::Transaction.into()),
            }
            let started = std::time::Instant::now();
            let mut cursor = if let Some(window) = window {
                frame.take_prefill_routed_intervention_cursor(index, window)?
            } else {
                frame.take_routed_intervention_cursor(index, source_tokens)?
            };
            let result = (|| {
                let input = effective.as_ref().unwrap_or(source.values);
                let source = RoutedUnitCaptureSource {
                    values: input,
                    token_indices: source.token_indices,
                    selection_indices: source.selection_indices,
                    coefficients: source.coefficients,
                    source_groups: source.source_groups,
                    token_offset: source.token_offset,
                    global_groups: source.global_groups,
                };
                let range = if let Some(window) = window {
                    self.backend.prefill_routed_intervention_range(
                        &source,
                        cursor.claim(),
                        window,
                    )?
                } else {
                    self.backend
                        .routed_intervention_range(&source, cursor.claim())?
                };
                if !cursor.is_charged() {
                    let usage = if let Some(window) = window {
                        self.backend.prefill_routed_intervention_usage(
                            &source,
                            cursor.claim(),
                            window,
                        )?
                    } else {
                        self.backend
                            .routed_intervention_usage(&source, cursor.claim())?
                    };
                    if usage.captures != 0 || usage.encoded_bytes != 0 {
                        return Err(CaptureProtocolError::Transaction.into());
                    }
                    crate::intervention::reserve_envelope(&mut self.session.ledger, usage)?;
                    cursor.charge(usage)?;
                }
                let batch = if let Some(window) = window {
                    cursor.begin_prefill_batch(window, range)?
                } else {
                    cursor.begin_batch(range)?
                };
                self.backend.apply_routed_intervention(&source, batch)
            })();
            self.session.capture_seconds += started.elapsed().as_secs_f64();
            match result {
                Ok(output) => {
                    frame.retain_routed_intervention_cursor(cursor)?;
                    if let Some(output) = output {
                        effective = Some(output);
                    }
                }
                Err(cause) => {
                    frame.record_intervention_failure(
                        index,
                        "admitted sparse intervention failed",
                        cursor.charged(),
                    )?;
                    return Err(cause);
                }
            }
        }
        Ok(effective)
    }
    pub(super) fn finish_routed_interventions(
        &mut self,
        success: bool,
    ) -> Result<(), FundedCaptureError<E>> {
        let Some(first) = self.routed_intervention_selection else {
            return Ok(());
        };
        if !success {
            return Ok(());
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
        let plan = frame
            .intervention_admission()
            .ok_or(CaptureProtocolError::Transaction)?;
        let window = prefill
            .as_ref()
            .map(|(geometry, chunk)| {
                InterventionPrefillWindow::new(plan, *geometry, chunk)
                    .map_err(|_| CaptureProtocolError::PrefillAttribution)
            })
            .transpose()?;
        for index in 0..plan.plan().operations.len() {
            if !same_edit(plan, first, index) {
                continue;
            }
            if frame
                .interventions()
                .get(index)
                .ok_or(CaptureProtocolError::Transaction)?
                .outcome
                == InterventionOutcome::Inactive
            {
                continue;
            }
            if let Some(window) = window {
                frame.finish_prefill_routed_intervention_chunk(index, window)?;
            } else {
                frame.finish_routed_intervention(index)?;
            }
        }
        Ok(())
    }
}
