//! Ordinary/decode batches retain one existing frame target per logical hook.
use super::*;
impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(super) fn routed_invocation_batch(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, T>,
        effective: bool,
        first: usize,
    ) -> Result<(), FundedCaptureError<E>> {
        let source = batch.capture_source()?;
        let policy = CaptureObservationStep::with_invocation(
            &self.session.plan,
            self.session.phase,
            self.session.prediction,
            self.invocation.map(|value| value.1),
        )?
        .with_window(self.window)?;
        let Frame::Active(frame) = &mut self.frame else {
            return Err(CaptureProtocolError::Transaction.into());
        };
        for index in 0..self.session.plan.plan().selections.len() {
            if !same_routing(&self.session.plan, first, index)
                || (self.session.plan.points()[index].position
                    == eredu_core::ObservationPosition::AfterIntervention)
                    != effective
            {
                continue;
            }
            match frame.records()[index].outcome {
                CaptureOutcome::Skipped { .. } => continue,
                CaptureOutcome::Missing => {}
                _ => return Err(CaptureProtocolError::Duplicate.into()),
            }
            let geometry = policy
                .routed_geometry(index)
                .map_err(CaptureStepError::from)
                .map_err(CaptureRunHostError::from)?;
            let dtype = self
                .backend
                .validate_routed_invocation_source(&source, &geometry)?;
            if !frame.routed_invocation_started(index) {
                let usage = self.backend.estimate_routed_prefill(&geometry)?;
                let metadata = policy.window_metadata_usage(index)?.unwrap_or_default();
                if metadata != CaptureUsage::default() {
                    if let Some(reason) =
                        policy.reserve_value(&mut self.session.ledger, metadata)?
                    {
                        frame.record_skip(index, reason, Some(dtype), CaptureUsage::default())?;
                        continue;
                    }
                }
                if let Some(reason) = policy.reserve_value(&mut self.session.ledger, usage)? {
                    frame.record_skip(index, reason, Some(dtype), metadata)?;
                    continue;
                }
                frame.begin_routed_invocation(index, dtype, metadata.checked_add(usage)?)?;
            }
            let started = std::time::Instant::now();
            let result = self
                .backend
                .transform_routed_batch(&source, frame.take_routed_batch(index)?);
            self.session.capture_seconds += started.elapsed().as_secs_f64();
            result?;
        }
        Ok(())
    }

    pub(super) fn finish_routed_ordinary_invocation(
        &mut self,
        success: bool,
        first: usize,
    ) -> Result<(), FundedCaptureError<E>> {
        let Frame::Active(frame) = &mut self.frame else {
            return Err(CaptureProtocolError::Transaction.into());
        };
        for index in 0..self.session.plan.plan().selections.len() {
            if !same_routing(&self.session.plan, first, index)
                || matches!(
                    frame.records()[index].outcome,
                    CaptureOutcome::Skipped { .. }
                )
            {
                continue;
            }
            if !success {
                frame.record_failure(
                    index,
                    CaptureFailureReason::Invalid,
                    "routed invocation did not complete",
                    None,
                    CaptureUsage::default(),
                )?;
            } else {
                frame.finish_routed_invocation(index)?;
            }
        }
        Ok(())
    }
}
