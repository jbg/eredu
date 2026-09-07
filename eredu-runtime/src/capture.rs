//! Budget admission at the observation boundary, before native handles are retained.

use eredu_core::capture::*;

#[cfg(test)]
mod tests;

/// A run owns one ledger and at most one step of host records. Consumers must drain
/// each step before another is started; there is no producer queue.
pub struct CaptureSession {
    plan: AdmittedCapturePlan,
    ledger: CaptureLedger,
    records: Option<Vec<CaptureRecord>>,
    prediction: u64,
    phase: CapturePhase,
    capture_seconds: f64,
}

impl CaptureSession {
    /// Creates an unstarted capture run owning its admission and ledger.
    pub fn new(plan: AdmittedCapturePlan) -> Self {
        Self {
            ledger: CaptureLedger::new(&plan),
            plan,
            records: None,
            prediction: 0,
            phase: CapturePhase::Prefill,
            capture_seconds: 0.0,
        }
    }

    /// Borrows this run's immutable admission.
    pub fn plan(&self) -> &AdmittedCapturePlan {
        &self.plan
    }

    /// Reserves diagnostic envelopes before execution, including scheduled skips and
    /// missing values. Exhaustion here fails the step: emitting an unaccounted skip
    /// record would itself violate the export limit.
    pub fn begin_step(&mut self, phase: CapturePhase, prediction: u64) -> Result<(), CaptureError> {
        if self.records.is_some() {
            return Err(CaptureError::Invalid(
                "previous capture step has not been consumed".into(),
            ));
        }
        if prediction >= self.plan.request().max_predictions {
            return Err(CaptureError::Invalid(
                "generation exceeds admitted prediction range".into(),
            ));
        }
        self.ledger.begin_step();
        let mut records = Vec::new();
        for (selection, point) in self.plan.plan().selections.iter().zip(self.plan.points()) {
            let charged = metadata_reservation(selection, point)?;
            if let Some(CaptureSkipReason::Limit { budget, cumulative }) =
                self.ledger.reserve(charged)?
            {
                return Err(CaptureError::Limit { budget, cumulative });
            }
            records.push(CaptureRecord {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selection_id: selection.id.clone(),
                path: selection.path.clone(),
                node_id: point.node_id.clone(),
                position: point.position,
                source_shape: None,
                selected_shape: None,
                outcome: if selection.schedule.includes(phase, prediction) {
                    CaptureOutcome::Missing
                } else {
                    CaptureOutcome::Skipped {
                        reason: CaptureSkipReason::Schedule,
                    }
                },
                payload: None,
                charged,
            });
        }
        self.records = Some(records);
        self.phase = phase;
        self.prediction = prediction;
        self.capture_seconds = 0.0;
        Ok(())
    }

    /// Invoked while the architecture borrows a tensor. A skipped point never calls
    /// the native transform and never clones a native tensor handle.
    pub fn observe<B: CaptureBackend>(
        &mut self,
        backend: &mut B,
        path: &str,
        tensor: &B::Tensor,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        let Some(records) = self.records.as_mut() else {
            return Err(CaptureError::Invalid("capture step not started".into()).into());
        };
        for ((selection, point), record) in self
            .plan
            .plan()
            .selections
            .iter()
            .zip(self.plan.points())
            .zip(records)
        {
            if selection.path != path || matches!(record.outcome, CaptureOutcome::Skipped { .. }) {
                continue;
            }
            if !matches!(record.outcome, CaptureOutcome::Missing) {
                return Err(
                    CaptureError::Invalid(format!("observation emitted twice: {path}")).into(),
                );
            }
            let started = std::time::Instant::now();
            let result = (|| -> Result<(), CaptureExecutionError<B::Error>> {
                let shape = backend
                    .shape(tensor)
                    .map_err(CaptureExecutionError::Backend)?;
                self.plan
                    .request()
                    .validate_actual(point, self.phase, self.prediction, &shape)?;
                if let Some(expected) =
                    self.plan
                        .request()
                        .resolve(point, self.phase, self.prediction)?
                {
                    if expected != shape {
                        return Err(CaptureError::Invalid(format!(
                            "runtime shape for {path}: expected {expected:?}, got {shape:?}"
                        ))
                        .into());
                    }
                }
                let slice = resolve_slice(point, selection, &shape)?;
                let usage = backend.estimate(tensor, selection, &slice)?;
                record.source_shape = Some(shape);
                record.selected_shape = Some(slice.shape.clone());
                if let Some(reason) = self.ledger.reserve(usage)? {
                    record.outcome = CaptureOutcome::Skipped { reason };
                    return Ok(());
                }
                record.charged = record.charged.checked_add(usage)?;
                let payload = backend
                    .transform(tensor, selection, &slice)
                    .map_err(CaptureExecutionError::Backend)?;
                let available = elements(&slice.shape)?;
                record.outcome = match selection.transform {
                    CaptureTransform::Preview { max_elements } if max_elements < available => {
                        CaptureOutcome::Truncated {
                            available_elements: available,
                            emitted_elements: max_elements,
                        }
                    }
                    _ => CaptureOutcome::Captured,
                };
                record.payload = Some(payload);
                // Count through a bounded sink, without allocating a second JSON buffer.
                // This is a backend-contract check, not a substitute for the pre-copy estimate.
                let mut sink = CountingWriter {
                    written: 0,
                    limit: record.charged.encoded_bytes,
                };
                serde_json::to_writer(&mut sink, record).map_err(|_| {
                    CaptureError::Invalid("backend underestimated encoded capture size".into())
                })?;
                Ok(())
            })();
            self.capture_seconds += started.elapsed().as_secs_f64();
            if let Err(error) = result {
                let reason = match &error {
                    CaptureExecutionError::Admission(CaptureError::Limit {
                        budget,
                        cumulative,
                    }) => CaptureFailureReason::Limit {
                        budget: *budget,
                        cumulative: *cumulative,
                    },
                    CaptureExecutionError::Admission(CaptureError::Unsupported(_)) => {
                        CaptureFailureReason::Unsupported
                    }
                    CaptureExecutionError::Admission(_) => CaptureFailureReason::Invalid,
                    CaptureExecutionError::Backend(_) => CaptureFailureReason::Native,
                };
                record.payload = None;
                record.outcome = CaptureOutcome::Failed {
                    reason,
                    message: bounded_diagnostic(&error),
                };
                return Err(error);
            }
        }
        Ok(())
    }

    /// Moves the current bounded record batch to the consumer.
    pub fn take_step(&mut self) -> Option<CapturedStep> {
        self.records.take().map(|records| CapturedStep {
            phase: self.phase,
            prediction_index: self.prediction,
            records,
            step_usage: self.ledger.step(),
            cumulative_usage: self.ledger.total(),
            capture_seconds: self.capture_seconds,
        })
    }
}

fn bounded_diagnostic(error: &impl std::fmt::Display) -> String {
    use std::fmt::Write;
    struct Message(String);
    impl std::fmt::Write for Message {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            let mut end = text.len().min(256 - self.0.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            self.0.push_str(&text[..end]);
            if end < text.len() {
                Err(std::fmt::Error)
            } else {
                Ok(())
            }
        }
    }
    let mut message = Message(String::with_capacity(256));
    let _ = write!(&mut message, "{error}");
    message.0
}

/// Conservative JSON envelope reservation: every UTF-8 byte can escape to at most
/// six bytes; each dimension can use twenty decimal digits. Numeric payloads are
/// reserved by the native estimator. Selection count and metadata are bounded before
/// generation begins, so missing/skipped points cannot form an unbounded queue.
pub fn metadata_reservation(
    selection: &CaptureSelection,
    point: &eredu_core::ObservationPoint,
) -> Result<CaptureUsage, CaptureError> {
    let strings = add(
        add(selection.id.len() as u64, selection.path.len() as u64)?,
        point.node_id.len() as u64,
    )?;
    let rank = point.axes.as_ref().map_or(32, |axes| axes.len() as u64);
    Ok(CaptureUsage {
        captures: 0,
        retained_bytes: 0,
        host_bytes: add(512, add(strings, mul(rank, 128)?)?)?,
        encoded_bytes: add(2048, add(mul(strings, 6)?, mul(rank, 64)?)?)?,
    })
}

/// Checks conservative known per-step and run-total costs before execution.
/// Backends supply only their side-effect-free storage/transfer estimator. Unknown
/// dimensions remain subject to exact observation-time reservation. Step estimates
/// conservatively allow every phase-enabled selection to coincide.
pub fn preflight(
    plan: &AdmittedCapturePlan,
    mut estimate: impl FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>,
) -> Result<(), CaptureError> {
    let mut base = CaptureUsage::default();
    for (selection, point) in plan.plan().selections.iter().zip(plan.points()) {
        base = base.checked_add(metadata_reservation(selection, point)?)?;
    }
    if let Some(budget) = base.exceeded(plan.plan().limits.per_step) {
        return Err(CaptureError::Limit {
            budget,
            cumulative: false,
        });
    }
    let mut total = base.checked_mul(plan.request().max_predictions)?;
    for phase in [CapturePhase::Prefill, CapturePhase::Decode] {
        if phase == CapturePhase::Decode && plan.request().max_predictions <= 1 {
            continue;
        }
        let mut step = base;
        for (selection, point) in plan.plan().selections.iter().zip(plan.points()) {
            let Some((count, last)) = selection
                .schedule
                .count_and_last(phase, plan.request().max_predictions)?
            else {
                continue;
            };
            if let Some(shape) = plan.request().resolve(point, phase, last)? {
                let slice = resolve_slice(point, selection, &shape)?;
                let cost = estimate(&shape, selection, &slice)?;
                if plan.plan().limits.on_limit == CaptureLimitPolicy::Fail {
                    step = step.checked_add(cost)?;
                    total = total.checked_add(cost.checked_mul(count)?)?;
                }
            }
        }
        if let Some(budget) = step.exceeded(plan.plan().limits.per_step) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: false,
            });
        }
    }
    if let Some(budget) = total.exceeded(plan.plan().limits.cumulative) {
        return Err(CaptureError::Limit {
            budget,
            cumulative: true,
        });
    }
    Ok(())
}

struct CountingWriter {
    written: u64,
    limit: u64,
}
impl std::io::Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .written
            .checked_add(bytes.len() as u64)
            .filter(|next| *next <= self.limit)
            .ok_or_else(|| std::io::Error::other("capture JSON budget exceeded"))?;
        self.written = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Failure before reservation or during the reserved native transformation.
#[derive(Debug, thiserror::Error)]
pub enum CaptureExecutionError<E: std::error::Error + 'static> {
    /// Invalid geometry or rejected budget/capability.
    #[error(transparent)]
    Admission(#[from] CaptureError),
    /// Native transformation failed under the backend's recovery owner.
    #[error("native capture failed: {0}")]
    Backend(E),
}
