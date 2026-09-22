//! One original logical target; short physical chunks share its existing buffer.
use super::*;
use crate::capture::{
    CapturePrefillHookDecision, CapturePrefillObservationPolicy, CapturePrefillProgressError,
};
use crate::working_memory::CapturePrefillHostError;
mod generated;

pub(super) fn progress_error<E: std::error::Error + 'static>(
    error: CapturePrefillProgressError,
) -> FundedCaptureError<E> {
    CaptureRunHostError::from(CapturePrefillHostError::from(error)).into()
}
impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(super) fn observe_fragment_value(
        &mut self,
        path: &str,
        value: &T,
    ) -> Result<(), FundedCaptureError<E>> {
        let chunk = self.current_fragment_chunk()?;
        if matches!(self.frame, Frame::Empty) {
            return Ok(());
        }
        let bound = self.bound.ok_or(CaptureProtocolError::Transaction)?;
        let policy = CapturePrefillObservationPolicy::from_bound(bound).map_err(progress_error)?;
        for index in 0..bound
            .selection()
            .source()
            .admission()
            .plan()
            .selections
            .len()
        {
            if let Some(partition) = self.backend.partition_capture() {
                if !partition.produces(index)? {
                    continue;
                }
            }
            let row = policy.row(index).map_err(progress_error)?;
            let Frame::Active(frame) = &mut self.frame else {
                return Err(CaptureProtocolError::Transaction.into());
            };
            let decision = frame.begin_prefill_hook(index, &chunk, path)?;
            if decision == CapturePrefillHookDecision::Ignore {
                continue;
            }
            let started = std::time::Instant::now();
            let mut dtype = None;
            let result = observe_fragment_target(
                self.backend,
                frame,
                &mut self.session.ledger,
                &row,
                index,
                decision,
                bound.geometry(),
                &chunk,
                value,
                &mut dtype,
            );
            self.session.capture_seconds += started.elapsed().as_secs_f64();
            if let Err(error) = result {
                self.fail_fragment(index, dtype, &error);
                return Err(error);
            }
        }
        Ok(())
    }
    fn fail_fragment(
        &mut self,
        index: usize,
        dtype: Option<TensorDtype>,
        error: &FundedCaptureError<E>,
    ) {
        let reason = match error {
            FundedCaptureError::Admission(error) => policy::failure_reason(error),
            FundedCaptureError::Backend(_) => CaptureFailureReason::Native,
            _ => CaptureFailureReason::Invalid,
        };
        if let Frame::Active(frame) = &mut self.frame {
            frame.fail_prefill_hook(index);
            if matches!(frame.records()[index].outcome, CaptureOutcome::Missing) {
                // Full usage was already recorded by reserve_prefill_hook. The
                // diagnostic must not charge it again or replace the real cause.
                let _ = frame.record_failure(
                    index,
                    reason,
                    "prefill capture fragment failed",
                    dtype,
                    CaptureUsage::default(),
                );
            }
        }
    }
    pub(super) fn complete_fragment_chunk(
        &mut self,
        final_chunk: bool,
    ) -> Result<(), FundedCaptureError<E>> {
        let Some(bound) = self.bound else {
            return Ok(());
        };
        let chunk = self.current_fragment_chunk()?;
        let Frame::Active(frame) = &mut self.frame else {
            return if matches!(self.frame, Frame::Empty) {
                Ok(())
            } else {
                Err(CaptureProtocolError::Transaction.into())
            };
        };
        if let Some(plan) = frame.intervention_admission() {
            let window =
                crate::intervention::InterventionPrefillWindow::new(plan, bound.geometry(), &chunk)
                    .map_err(|_| CaptureProtocolError::PrefillAttribution)?;
            let partition = match self.backend.partition_capture() {
                Some(program) => program.validate_intervention_prefill_end(frame, window)?,
                None => false,
            };
            if !partition {
                frame.validate_prefill_intervention_end(window)?;
            }
        }
        if let Some(program) = self.backend.partition_capture() {
            program.complete_prefill_evidence(frame, bound, &chunk)?;
        } else {
            super::interventions::evidence::complete_prefill(frame, bound.geometry(), &chunk)?;
        }
        frame
            .complete_prefill_chunk(chunk.input.start / bound.geometry().prefill_chunk_positions)?;
        if final_chunk {
            if self.backend.partition_capture().is_some() {
                frame.finish_local_prefill_targets()?;
                self.complete_partition_delivery()?;
                let Frame::Active(frame) = &mut self.frame else {
                    return Err(CaptureProtocolError::Transaction.into());
                };
                frame.finish_prefill_targets()?;
            } else {
                frame.finish_prefill_targets()?;
            }
            // The shared delivery seal validates successful tensor encodings
            // before the commit vote, preserving skipped and inactive records.
        }
        Ok(())
    }
}

/// These adapter frames surround the existing callback; local native/claim and
/// owning failure controls remain in the exact projected hook source.
pub(in crate::capture::funded) fn projection_control_bytes() -> usize {
    use crate::capture::partition::{PartitionCaptureLocalHook, PartitionCaptureProgramError};
    std::mem::size_of::<(
        Option<PartitionCaptureLocalHook>,
        Result<Option<PartitionCaptureLocalHook>, PartitionCaptureProgramError>,
        PartitionCaptureLocalHook,
        Result<PartitionCaptureLocalHook, PartitionCaptureProgramError>,
        Result<(), PartitionCaptureProgramError>,
        u64,
        bool,
        (
            &mut dyn ScheduledCaptureBackend<Tensor = (), Error = std::convert::Infallible>,
            &mut crate::working_memory::ScheduledCaptureStep<'_>,
            &mut CaptureLedger,
            &crate::capture::CapturePrefillObservationRow<'_>,
            usize,
            CapturePrefillHookDecision,
            eredu_core::InferenceGeometry,
            &crate::prefill::PrefillChunk,
            &(),
            &mut Option<TensorDtype>,
        ),
    )>()
}

/// One existing fragment worker for ordinary observations and intervention
/// companions. The caller retains the original frame, ledger and native scope.
pub(super) fn observe_fragment_target<T, E: std::error::Error + Send + Sync + 'static>(
    backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
    frame: &mut crate::working_memory::ScheduledCaptureStep<'_>,
    ledger: &mut CaptureLedger,
    row: &crate::capture::CapturePrefillObservationRow<'_>,
    index: usize,
    decision: CapturePrefillHookDecision,
    inference: eredu_core::InferenceGeometry,
    chunk: &crate::prefill::PrefillChunk,
    value: &T,
    dtype: &mut Option<TensorDtype>,
) -> Result<(), FundedCaptureError<E>> {
    let projected = match backend.partition_capture() {
        Some(program) => program.take_prefill_projection(
            index,
            frame,
            decision == CapturePrefillHookDecision::First,
        )?,
        None => None,
    };
    if let Some(hook) = projected {
        let k = chunk.input.start / inference.prefill_chunk_positions;
        let hook = hook.observe(backend, value, inference, k)?;
        backend
            .partition_capture()
            .ok_or(CaptureProtocolError::Transaction)?
            .return_prefill_projection(index, hook)?;
        if let Some(plan) = row.transform_plan().filter(|plan| {
            matches!(
                plan.selection().transform,
                CaptureTransform::Summary | CaptureTransform::Histogram { .. }
            )
        }) {
            let fragment = plan.fragment(k)?;
            match plan.selection().transform {
                CaptureTransform::Summary => frame.finish_summary_prefill_hook(index, &fragment)?,
                CaptureTransform::Histogram { .. } => {
                    frame.finish_histogram_prefill_hook(index, &fragment)?
                }
                _ => return Err(CaptureProtocolError::Geometry.into()),
            }
        } else {
            let assembly = row.assembly().ok_or(CaptureProtocolError::Geometry)?;
            let fragment = assembly
                .fragment(k)
                .map_err(|error| progress_error(error.into()))?;
            frame.finish_prefill_hook(index, &fragment)?;
        }
        return Ok(());
    }
    if row.candidate().is_some() {
        let geometry = row.candidate_for_chunk(chunk).map_err(progress_error)?;
        let actual = backend.validate_candidate_source(value, &geometry)?;
        *dtype = Some(actual.clone());
        let usage = backend.estimate_candidates(&geometry)?;
        let skipped = match backend.partition_capture() {
            Some(partition) => frame.reserve_prefill_hook(
                index,
                partition.reservation(index, &actual, usage)?,
                actual.clone(),
                usage,
            )?,
            None => frame.reserve_prefill_hook(index, ledger, actual.clone(), usage)?,
        };
        if skipped.is_some() {
            return Ok(());
        }
        let claim = frame.take_candidates(index)?;
        let receipt = backend.transform_candidates(value, claim)?;
        frame.record_candidates(receipt, actual, CaptureUsage::default())?;
        frame.finish_candidate_prefill_hook(index, chunk)?;
        frame.validate_record_encoding(index)?;
        return Ok(());
    }
    if row.token_scores().is_some() {
        let geometry = row.token_scores_for_chunk(chunk).map_err(progress_error)?;
        let actual = backend.validate_token_score_source(value, &geometry)?;
        *dtype = Some(actual.clone());
        let usage = backend.estimate_token_scores(&geometry)?;
        let skipped = match backend.partition_capture() {
            Some(partition) => frame.reserve_prefill_hook(
                index,
                partition.reservation(index, &actual, usage)?,
                actual.clone(),
                usage,
            )?,
            None => frame.reserve_prefill_hook(index, ledger, actual.clone(), usage)?,
        };
        if skipped.is_some() {
            return Ok(());
        }
        let claim = frame.take_token_scores(index)?;
        let receipt = backend.transform_token_scores(value, claim)?;
        frame.record_token_scores(receipt, actual, CaptureUsage::default())?;
        frame.finish_token_scores_prefill_hook(index, chunk)?;
        frame.validate_record_encoding(index)?;
        return Ok(());
    }
    if let Some(plan) = row
        .transform_plan()
        .filter(|plan| matches!(plan.selection().transform, CaptureTransform::Summary))
    {
        let fragment = plan.fragment(chunk.input.start / inference.prefill_chunk_positions)?;
        let geometry = CaptureSummaryGeometry::prepare(
            plan.admission(),
            index,
            CapturePhase::Prefill,
            0,
            None,
        )
        .and_then(|geometry| geometry.fragment(&fragment))
        .map_err(|error| CaptureRunHostError::Step(error.into()))?;
        let actual = backend.validate_summary_source(value, &geometry)?;
        *dtype = Some(actual.clone());
        if decision == CapturePrefillHookDecision::First {
            let usage = backend.estimate_prefill_summary(plan)?;
            let skipped = match backend.partition_capture() {
                Some(partition) => frame.reserve_prefill_hook(
                    index,
                    partition.reservation(index, &actual, usage)?,
                    actual,
                    usage,
                )?,
                None => frame.reserve_prefill_hook(index, ledger, actual, usage)?,
            };
            if skipped.is_some() {
                return Ok(());
            }
        }
        let claim = frame.take_prefill_summary(index, &fragment)?;
        let receipt = if fragment.selected_elements() == 0 {
            claim.finish_empty()?
        } else {
            backend.transform_summary(value, claim)?
        };
        frame.record_prefill_summary(receipt, &fragment)?;
        frame.finish_summary_prefill_hook(index, &fragment)?;
        return Ok(());
    }
    if let Some(plan) = row.transform_plan().filter(|plan| {
        matches!(
            plan.selection().transform,
            CaptureTransform::Histogram { .. }
        )
    }) {
        let fragment = plan.fragment(chunk.input.start / inference.prefill_chunk_positions)?;
        let geometry = CaptureHistogramGeometry::prepare(
            plan.admission(),
            index,
            CapturePhase::Prefill,
            0,
            None,
        )
        .and_then(|geometry| geometry.fragment(&fragment))
        .map_err(|error| CaptureRunHostError::Step(error.into()))?;
        let actual = backend.validate_histogram_source(value, &geometry)?;
        *dtype = Some(actual.clone());
        if decision == CapturePrefillHookDecision::First {
            let usage = backend.estimate_prefill_histogram(plan)?;
            let skipped = match backend.partition_capture() {
                Some(partition) => frame.reserve_prefill_hook(
                    index,
                    partition.reservation(index, &actual, usage)?,
                    actual,
                    usage,
                )?,
                None => frame.reserve_prefill_hook(index, ledger, actual, usage)?,
            };
            if skipped.is_some() {
                return Ok(());
            }
        }
        let claim = frame.take_prefill_histogram(index, &fragment)?;
        let receipt = if fragment.selected_elements() == 0 {
            claim.prepare()?.finish(0, 0, 0)?
        } else {
            backend.transform_histogram(value, claim)?
        };
        frame.record_prefill_histogram(receipt, &fragment)?;
        frame.finish_histogram_prefill_hook(index, &fragment)?;
        return Ok(());
    }
    let assembly = row.assembly().ok_or(CaptureProtocolError::Geometry)?;
    let fragment = assembly
        .fragment(chunk.input.start / inference.prefill_chunk_positions)
        .map_err(|e| progress_error(CapturePrefillProgressError::from(e)))?;
    let actual = backend.validate_prefill_source(value, &fragment)?;
    if !matches!(
        actual,
        TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16
    ) {
        return Err(CaptureProtocolError::Geometry.into());
    }
    *dtype = Some(actual.clone());
    if decision == CapturePrefillHookDecision::First {
        let usage = backend.estimate_prefill(assembly.logical_geometry())?;
        let skipped = match backend.partition_capture() {
            Some(partition) => frame.reserve_prefill_hook(
                index,
                partition.reservation(index, &actual, usage)?,
                actual,
                usage,
            )?,
            None => frame.reserve_prefill_hook(index, ledger, actual, usage)?,
        };
        if skipped.is_some() {
            return Ok(());
        }
    }
    if fragment.output_elements() != 0 {
        let claim = frame.take_prefill_fragment(index, &fragment)?;
        backend.transform_prefill_fragment(value, claim)?;
    } else if decision == CapturePrefillHookDecision::First
        && assembly.logical_geometry().elements() == 0
    {
        // Preserve the genuine empty destination; later zero fragments
        // only satisfy hook progression and allocate no second buffer.
        frame
            .take_prefill_fragment(index, &fragment)?
            .prepare()?
            .finish()?;
    }
    frame.finish_prefill_hook(index, &fragment)?;
    Ok(())
}
