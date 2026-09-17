//! Online host aggregates. All native work remains in the ordinary capture worker.
use super::*;
use crate::capture::reduction;
pub(in crate::capture) mod preview;
use eredu_core::speculative::{
    SpeculativePrefillReduction, SpeculativePrefillReductionGeometry,
    SpeculativePrefillReductionStatus as Status, SpeculativePrefillReductions,
};

fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}
fn reduced(selection: &CaptureSelection) -> bool {
    matches!(
        selection.transform,
        CaptureTransform::Summary
            | CaptureTransform::Histogram { .. }
            | CaptureTransform::Preview { .. }
    )
}
fn phases() -> [SpeculativeActivationPhase; 2] {
    [
        SpeculativeActivationPhase::TargetPrefill,
        SpeculativeActivationPhase::PredictionPrefill,
    ]
}
fn length(geometry: SpeculativePrefillReductionGeometry, phase: SpeculativeActivationPhase) -> u64 {
    if phase == SpeculativeActivationPhase::TargetPrefill {
        geometry.target_sequence
    } else {
        geometry.prediction_sequence
    }
}

use eredu_core::capture::CaptureWindowGeometry as Geometry;

pub(super) mod construction;
mod interventions;
mod payload;
use construction::{ConstructionError, Metadata};
use payload::State;

pub(super) struct WindowReductions {
    geometry: SpeculativePrefillReductionGeometry,
    pub(super) report: SpeculativePrefillReductions,
    states: Vec<State>,
    interventions: interventions::Interventions,
    pending: Option<(SpeculativeActivationPhase, u64, u64)>,
    prepared: bool,
    sealed: bool,
}

impl WindowReductions {
    pub(super) fn create(
        session: &mut CaptureSession,
        scopes: &[SpeculativeCaptureScope],
        intervention_scopes: &[SpeculativeCaptureScope],
        geometry: SpeculativePrefillReductionGeometry,
        origin: SpeculativeActivationOrigin,
        invocation: u64,
    ) -> Result<Option<Self>, CaptureError> {
        // Preserve the ordinary validation order before inspecting extra evidence.
        Self::validate_geometry(geometry).map_err(ConstructionError::into_capture)?;
        let extra = interventions::Interventions::count(session, intervention_scopes, origin)? != 0;
        let Some(mut out) = Self::create_capture(
            &session.plan,
            &mut session.ledger,
            scopes,
            geometry,
            origin,
            invocation,
            extra,
            Metadata::ordinary(),
        )
        .map_err(ConstructionError::into_capture)?
        else {
            return Ok(None);
        };
        out.interventions = interventions::Interventions::create(
            session,
            intervention_scopes,
            geometry,
            origin,
            &mut out.report,
        )?;
        Ok(Some(out))
    }
    fn validate_geometry(
        geometry: SpeculativePrefillReductionGeometry,
    ) -> Result<(), ConstructionError> {
        if geometry.target_sequence == 0
            || geometry.prediction_sequence > geometry.target_sequence
            || geometry.target_sequence - geometry.prediction_sequence > 1
        {
            return Err(ConstructionError::Invalid(
                "selected prefill reduction extents differ",
            ));
        }
        Ok(())
    }
    pub(super) fn create_capture(
        source: &AdmittedCapturePlan,
        ledger: &mut CaptureLedger,
        scopes: &[SpeculativeCaptureScope],
        geometry: SpeculativePrefillReductionGeometry,
        origin: SpeculativeActivationOrigin,
        invocation: u64,
        extra_interventions: bool,
        metadata: Metadata<'_>,
    ) -> Result<Option<Self>, ConstructionError> {
        metadata
            .controls(Self::construction_control_bytes().ok_or(ConstructionError::Overflow)?)?;
        Self::validate_geometry(geometry)?;
        if scopes.len() != source.points().len() {
            return Err(ConstructionError::Invalid(
                "aggregate scopes differ from capture source",
            ));
        }
        let prediction = u64::try_from(origin.prediction).map_err(|_| CaptureError::Overflow)?;
        let eligible = |i: usize, phase| {
            reduced(&source.plan().selections[i])
                && scopes[i].applies(phase)
                && source.plan().selections[i]
                    .schedule
                    .includes(CapturePhase::Prefill, prediction)
        };
        metadata.controls(std::mem::size_of_val(&eligible))?;
        let count = phases().into_iter().try_fold(0usize, |count, phase| {
            count
                .checked_add((0..scopes.len()).filter(|&i| eligible(i, phase)).count())
                .ok_or(CaptureError::Overflow)
        })?;
        if count == 0 && !extra_interventions {
            return Ok(None);
        }
        // Full borrowed declaration validation precedes any aggregate construction.
        for phase in phases() {
            for i in 0..scopes.len() {
                if eligible(i, phase) && length(geometry, phase) != 0 {
                    let point = &source.points()[i];
                    let selection = &source.plan().selections[i];
                    let g = Geometry::new_fixed(point, selection, 1, length(geometry, phase))?;
                    if let CaptureTransform::Preview { max_elements } = selection.transform {
                        preview::Plan::prepare_fixed(point, max_elements, &g)?;
                    }
                }
            }
        }
        let controls = CaptureUsage {
            host_bytes: add(
                add(
                    std::mem::size_of::<Self>() as u64,
                    crate::capture::RECORD_ENCODING_CONTROL_BYTES as u64,
                )?,
                mul(
                    count as u64,
                    (std::mem::size_of::<State>()
                        + std::mem::size_of::<SpeculativePrefillReduction>())
                        as u64,
                )?,
            )?,
            // Fixed logical origin/id/usage and the bounded per-entry scalar fields.
            encoded_bytes: add(1024, mul(count as u64, 512)?)?,
            ..Default::default()
        };
        ledger.reserve_quota(controls)?;
        let mut out = Self {
            geometry,
            report: SpeculativePrefillReductions {
                logical_invocation: invocation,
                origin,
                records: metadata.vec(count)?,
                interventions: Vec::new(),
                charged: controls,
            },
            states: metadata.vec(count)?,
            interventions: interventions::Interventions::empty(),
            pending: None,
            prepared: false,
            sealed: false,
        };
        for phase in phases() {
            for i in 0..scopes.len() {
                let selection = &source.plan().selections[i];
                if !reduced(selection)
                    || !scopes[i].applies(phase)
                    || !selection
                        .schedule
                        .includes(CapturePhase::Prefill, prediction)
                {
                    continue;
                }
                let point = &source.points()[i];
                let rows = length(geometry, phase);
                let (record, state) = payload::prepare_with_metadata(
                    ledger,
                    &mut out.report.charged,
                    point,
                    selection,
                    rows,
                    metadata,
                )?;
                out.report.records.push(SpeculativePrefillReduction {
                    phase,
                    selection_index: i,
                    logical_sequence: rows,
                    covered_sequence: 0,
                    first_invocation: None,
                    last_invocation: None,
                    windows: 0,
                    status: if matches!(record.outcome, CaptureOutcome::Skipped { .. }) {
                        Status::Skipped
                    } else {
                        Status::Pending
                    },
                    record,
                });
                out.states.push(state);
            }
        }
        Ok(Some(out))
    }

    fn validate_window(
        &self,
        phase: SpeculativeActivationPhase,
        span: eredu_core::speculative::SpeculativePrefillSpan,
        origin: SpeculativeActivationOrigin,
        geometry: SpeculativePrefillReductionGeometry,
    ) -> Result<u64, ConstructionError> {
        if self.pending.is_some()
            || origin != self.report.origin
            || geometry != self.geometry
            || span.prompt_tokens != geometry.target_sequence
        {
            return Err(ConstructionError::Invalid(
                "logical prefill reduction source changed",
            ));
        }
        let end = add(span.hidden_start, span.sequence)?;
        for entry in &self.report.records {
            if entry.phase == phase {
                if entry.covered_sequence != span.hidden_start || end > entry.logical_sequence {
                    return Err(ConstructionError::Invalid(
                        "logical reduction windows are not contiguous",
                    ));
                }
            }
        }
        Ok(end)
    }
    pub(super) fn begin(
        &mut self,
        session: &mut CaptureSession,
        phase: SpeculativeActivationPhase,
        span: eredu_core::speculative::SpeculativePrefillSpan,
        origin: SpeculativeActivationOrigin,
        geometry: SpeculativePrefillReductionGeometry,
    ) -> Result<(), CaptureError> {
        let end = self
            .validate_window(phase, span, origin, geometry)
            .map_err(ConstructionError::into_capture)?;
        for entry in &self.report.records {
            if entry.phase == phase {
                if let CaptureOutcome::Skipped { reason } = &entry.record.outcome {
                    session.records.as_mut().expect("started invocation")[entry.selection_index]
                        .outcome = CaptureOutcome::Skipped {
                        reason: reason.clone(),
                    };
                }
            }
        }
        self.interventions.begin(
            &self.report.interventions,
            session,
            phase,
            span.hidden_start,
            end,
        )?;
        self.start_window(phase, span.hidden_start, end);
        Ok(())
    }

    fn start_window(&mut self, phase: SpeculativeActivationPhase, start: u64, end: u64) {
        self.prepared = false;
        self.pending = Some((phase, start, end));
    }
    pub(super) fn begin_capture_window(
        &mut self,
        phase: SpeculativeActivationPhase,
        span: eredu_core::speculative::SpeculativePrefillSpan,
        origin: SpeculativeActivationOrigin,
        geometry: SpeculativePrefillReductionGeometry,
    ) -> Result<(), ConstructionError> {
        let end = self.validate_window(phase, span, origin, geometry)?;
        self.interventions.validate_begin_fixed(&self.report.interventions,phase,span.hidden_start,end)?;
        self.start_window(phase, span.hidden_start, end);
        Ok(())
    }
    pub(super) fn prepare(
        &mut self,
        records: &[CaptureRecord],
        interventions: &[eredu_core::intervention::InterventionRecord],
    ) -> Result<(), CaptureError> {
        self.prepare_fixed(records, interventions)
            .map_err(ConstructionError::into_capture)
    }
    pub(super) fn prepare_fixed(
        &mut self,
        records: &[CaptureRecord],
        interventions: &[eredu_core::intervention::InterventionRecord],
    ) -> Result<(), ConstructionError> {
        if self.prepared {
            return Err(ConstructionError::Invalid(
                "reduction window completed twice",
            ));
        }
        let (phase, start, end) = self
            .pending
            .ok_or_else(|| ConstructionError::Invalid("no active reduction window"))?;
        // Validate every row before applying any algebra. The invocation guard
        // seals this group as failed if validation or encoding later rejects.
        for (entry, state) in self.report.records.iter_mut().zip(&mut self.states) {
            if entry.phase == phase {
                let physical = &records[entry.selection_index];
                entry.record.charged.checked_add(physical.charged)?;
                entry.windows.checked_add(1).ok_or(CaptureError::Overflow)?;
                if entry.status == Status::Skipped {
                    continue;
                }
                if let CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Limit { .. },
                } = physical.outcome
                {
                    continue;
                }
                state.validate_fixed(
                    &entry.record,
                    physical,
                    start,
                    end,
                    entry.logical_sequence,
                )?;
            }
        }
        // Host payload changes remain provisional. On any later error the
        // terminal notification discards them, without publishing/refunding.
        for (entry, state) in self.report.records.iter_mut().zip(&mut self.states) {
            if entry.phase == phase {
                let physical = &records[entry.selection_index];
                if entry.status == Status::Skipped {
                    continue;
                }
                if let CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Limit { .. },
                } = &physical.outcome
                {
                    entry.status = Status::Skipped;
                    entry.record.outcome = physical.outcome.clone();
                    entry.record.payload = None;
                    continue;
                }
                state.update_fixed(&mut entry.record, physical, start, end)?;
            }
        }
        self.interventions.prepare_fixed(
            &mut self.report.interventions,
            interventions,
            phase,
            start,
            end,
        )?;
        self.prepared = true;
        Ok(())
    }

    pub(super) fn finish_window(
        &mut self,
        records: &[CaptureRecord],
        interventions: &[eredu_core::intervention::InterventionRecord],
        invocation: u64,
        success: bool,
    ) {
        let Some((phase, _, end)) = self.pending.take() else {
            return;
        };
        let success = success && self.prepared;
        self.interventions.finish_window(
            &mut self.report.interventions,
            interventions,
            phase,
            end,
            invocation,
            success,
        );
        for (entry, state) in self.report.records.iter_mut().zip(&mut self.states) {
            if entry.phase == phase {
                if !success {
                    entry.status = Status::Failed;
                    continue;
                }
                entry.record.charged = entry
                    .record
                    .charged
                    .checked_add(records[entry.selection_index].charged)
                    .expect("validated capture usage");
                entry.first_invocation.get_or_insert(invocation);
                entry.last_invocation = Some(invocation);
                entry.windows += 1;
                entry.covered_sequence = end;
                state.commit(end, entry.logical_sequence);
            }
        }
    }

    pub(super) fn seal(&mut self) -> Result<(), CaptureError> {
        self.seal_fixed().map_err(ConstructionError::into_capture)
    }
    pub(super) fn seal_fixed(&mut self) -> Result<(), ConstructionError> {
        if self.pending.is_some()
            || self.states.iter().any(|state| !state.sealed)
            || !self.interventions.sealed()
        {
            return Err(ConstructionError::Invalid(
                "logical prefill reduction is missing a complete phase",
            ));
        }
        if !crate::capture::encoded::prefill_reductions_fit_encoding(&self.report) {
            return Err(ConstructionError::Invalid(
                "logical prefill companion encoded reservation was underestimated",
            ));
        }
        self.sealed = true;
        Ok(())
    }

    pub(super) fn finish(mut self, success: bool) -> SpeculativePrefillReductions {
        let success = success && self.sealed;
        self.interventions
            .finish(&mut self.report.interventions, success);
        for (entry, state) in self.report.records.iter_mut().zip(&self.states) {
            if success && state.sealed && entry.status == Status::Pending {
                entry.status = Status::Complete;
                entry.record.outcome =
                    crate::capture::encoded::prefill_terminal_outcome(&entry.record)
                        .expect("terminal record checked at seal");
            } else {
                if entry.status == Status::Pending {
                    entry.status = if success {
                        Status::Failed
                    } else {
                        Status::Aborted
                    };
                }
                entry.record.payload = None;
            }
        }
        self.report
    }
}

#[cfg(test)]
mod construction_tests;
