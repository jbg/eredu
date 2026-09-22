//! Existing activation hook order, logical ledger and fixed outcome destinations.
use super::*;
pub(super) mod evidence;
mod partition;
impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(super) fn intervene_value(
        &mut self,
        path: &str,
        value: &T,
    ) -> Result<Option<T>, FundedCaptureError<E>> {
        match &self.frame {
            Frame::Empty => return Ok(None),
            Frame::Active(frame) if frame.intervention_admission().is_none() => return Ok(None),
            Frame::Active(_) => (),
            _ => return Err(CaptureProtocolError::Transaction.into()),
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
            return match self.frame {
                Frame::Empty => Ok(None),
                _ => Err(CaptureProtocolError::Transaction.into()),
            };
        };
        let Some(plan) = frame.intervention_admission() else {
            return Ok(None);
        };
        let mut effective = None;
        for (index, (operation, point)) in
            plan.plan().operations.iter().zip(plan.points()).enumerate()
        {
            let record = frame
                .interventions()
                .get(index)
                .ok_or(CaptureProtocolError::Transaction)?;
            match crate::intervention::activation_hook(operation, point, &record.outcome, path) {
                crate::intervention::ActivationHook::Unrelated => continue,
                crate::intervention::ActivationHook::Repeated => {
                    return Err(CaptureProtocolError::Transaction.into());
                }
                crate::intervention::ActivationHook::Active => (),
            }
            let window = match &prefill {
                Some((geometry, chunk))
                    if crate::intervention::InterventionPrefillWindow::row_axis(point) =>
                {
                    Some(
                        crate::intervention::InterventionPrefillWindow::new(plan, *geometry, chunk)
                            .map_err(|_| CaptureProtocolError::PrefillAttribution)?,
                    )
                }
                _ => None,
            };
            let member = match self.backend.partition_capture() {
                Some(program) => program.intervention_member(index, window)?,
                None => None,
            };
            if let Some(member) = member {
                if !member {
                    continue;
                }
                let started = std::time::Instant::now();
                let input = effective.as_ref().unwrap_or(value);
                let result = if let Some(window) = window {
                    let mut cursor = frame.take_prefill_intervention(index, window)?;
                    let result: Result<_, FundedCaptureError<E>> = (|| {
                        let mut fragment = cursor.begin(window)?;
                        let loan = partition::prepare(
                            self.backend,
                            input,
                            fragment.claim(),
                            Some(window),
                        )?;
                        let local = loan.reserved();
                        let accounting = fragment.charge(local).map_err(Into::into);
                        let output = partition::execute(
                            self.backend,
                            input,
                            fragment.claim(),
                            loan,
                            accounting,
                            frame,
                            prefill.as_ref().map(|(geometry, chunk)| (*geometry, chunk)),
                        )?;
                        fragment.finish()?;
                        Ok(output)
                    })();
                    match result {
                        Ok(output) => {
                            frame.retain_prefill_intervention(cursor)?;
                            Ok(output)
                        }
                        Err(cause) => {
                            frame.record_intervention_failure(
                                index,
                                "partition prefill intervention failed",
                                cursor.charged(),
                            )?;
                            Err(cause)
                        }
                    }
                } else {
                    let claim = frame.take_intervention(index)?;
                    let mut charged = CaptureUsage::default();
                    let result: Result<_, FundedCaptureError<E>> = (|| {
                        let loan = partition::prepare(self.backend, input, &claim, None)?;
                        charged = loan.reserved();
                        let output = partition::execute(
                            self.backend,
                            input,
                            &claim,
                            loan,
                            Ok(()),
                            frame,
                            prefill.as_ref().map(|(geometry, chunk)| (*geometry, chunk)),
                        )?;
                        Ok((output, claim.finish(charged)?))
                    })();
                    match result {
                        Ok((output, receipt)) => {
                            frame.record_intervention(receipt)?;
                            Ok(output)
                        }
                        Err(cause) => {
                            frame.record_intervention_failure(
                                index,
                                "partition intervention failed",
                                charged,
                            )?;
                            Err(cause)
                        }
                    }
                };
                self.session.capture_seconds += started.elapsed().as_secs_f64();
                if let Some(output) = result? {
                    effective = Some(output);
                }
                continue;
            }
            if let Some((geometry, chunk)) = &prefill {
                if crate::intervention::InterventionPrefillWindow::row_axis(point) {
                    let window =
                        crate::intervention::InterventionPrefillWindow::new(plan, *geometry, chunk)
                            .map_err(|_| CaptureProtocolError::PrefillAttribution)?;
                    let started = std::time::Instant::now();
                    let mut cursor = frame.take_prefill_intervention(index, window)?;
                    let result = (|| {
                        let input = effective.as_ref().unwrap_or(value);
                        let mut fragment = cursor.begin(window)?;
                        let projection = self.backend.prefill_intervention_projection_usage(
                            input,
                            fragment.claim(),
                            window,
                        )?;
                        for usage in projection {
                            if usage.captures != 0 || usage.retained_bytes != 0 {
                                return Err(CaptureProtocolError::Transaction.into());
                            }
                            crate::intervention::reserve_envelope(&mut self.session.ledger, usage)?;
                            fragment.charge(usage)?;
                        }
                        let usage = self.backend.prefill_intervention_usage(
                            input,
                            fragment.claim(),
                            window,
                        )?;
                        if usage.captures != 0 || usage.encoded_bytes != 0 {
                            return Err(CaptureProtocolError::Transaction.into());
                        }
                        crate::intervention::reserve_envelope(&mut self.session.ledger, usage)?;
                        fragment.charge(usage)?;
                        evidence::observe_prefill(
                            self.backend,
                            frame,
                            &mut self.session.ledger,
                            index,
                            InterventionEvidenceSide::Before,
                            input,
                            *geometry,
                            chunk,
                        )?;
                        let output = self
                            .backend
                            .apply_prefill_intervention(input, fragment, usage, projection)?;
                        evidence::observe_prefill(
                            self.backend,
                            frame,
                            &mut self.session.ledger,
                            index,
                            InterventionEvidenceSide::After,
                            output.as_ref().unwrap_or(input),
                            *geometry,
                            chunk,
                        )?;
                        Ok(output)
                    })();
                    self.session.capture_seconds += started.elapsed().as_secs_f64();
                    match result {
                        Ok(output) => {
                            frame.retain_prefill_intervention(cursor)?;
                            if let Some(output) = output {
                                effective = Some(output);
                            }
                        }
                        Err(cause) => {
                            frame.record_intervention_failure(
                                index,
                                "admitted prefill intervention failed",
                                cursor.charged(),
                            )?;
                            return Err(cause);
                        }
                    }
                    continue;
                }
            }
            let started = std::time::Instant::now();
            let mut charged = CaptureUsage::default();
            let claim = frame.take_intervention(index)?;
            let result = (|| {
                let input = effective.as_ref().unwrap_or(value);
                let projection = self.backend.intervention_projection_usage(input, &claim)?;
                for usage in projection {
                    if usage.captures != 0 || usage.retained_bytes != 0 {
                        return Err(CaptureProtocolError::Transaction.into());
                    }
                    crate::intervention::reserve_envelope(&mut self.session.ledger, usage)?;
                    charged = charged.checked_add(usage)?;
                }
                let usage = self.backend.intervention_usage(input, &claim)?;
                // As with ordinary activation_cost, captures/encoding belong to
                // the actual capture/evidence builders, never to an edit estimate.
                if usage.captures != 0 || usage.encoded_bytes != 0 {
                    return Err(CaptureProtocolError::Transaction.into());
                }
                crate::intervention::reserve_envelope(&mut self.session.ledger, usage)?;
                charged = charged.checked_add(usage)?;
                evidence::observe(
                    self.backend,
                    frame,
                    &mut self.session.ledger,
                    index,
                    InterventionEvidenceSide::Before,
                    input,
                )?;
                let (output, receipt) = self
                    .backend
                    .apply_intervention_projected(input, claim, usage, projection)?;
                evidence::observe(
                    self.backend,
                    frame,
                    &mut self.session.ledger,
                    index,
                    InterventionEvidenceSide::After,
                    output.as_ref().unwrap_or(input),
                )?;
                Ok((output, receipt))
            })();
            self.session.capture_seconds += started.elapsed().as_secs_f64();
            match result {
                Ok((output, receipt)) => {
                    frame.record_intervention(receipt)?;
                    if let Some(output) = output {
                        effective = Some(output);
                    }
                }
                Err(cause) => {
                    frame.record_intervention_failure(
                        index,
                        "admitted activation failed",
                        charged,
                    )?;
                    return Err(cause);
                }
            }
        }
        Ok(effective)
    }
}

pub(in crate::capture::funded) fn intervention_control_bytes() -> Option<usize> {
    partition::control_bytes()?.checked_add(evidence::prefill_control_bytes()?)
}
