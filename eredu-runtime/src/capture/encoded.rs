//! Streaming capture-wire validation without a secondary JSON/error buffer.
use super::*;

struct Counter {
    remaining: u64,
    fits: bool,
}
impl std::io::Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        match u64::try_from(bytes.len())
            .ok()
            .and_then(|bytes| self.remaining.checked_sub(bytes))
        {
            Some(remaining) if self.fits => self.remaining = remaining,
            _ => self.fits = false,
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
// Actual explicit counter/serializer controls plus conservative moves. Wire
// formatting uses the existing closed serializer; no dynamic output buffer or
// serializer error is produced by this accepting sink.
pub(crate) const RECORD_ENCODING_CONTROL_BYTES: usize = 3 * std::mem::size_of::<Counter>()
    + 3 * std::mem::size_of::<serde_json::Serializer<&mut Counter>>()
    + 3 * std::mem::size_of::<bool>()
    + 3 * eredu_core::capture::CaptureRecordWire::control_bytes();

/// Uses the same record serializer as ordinary capture's CountingWriter. This
/// sink records overflow but accepts all writes: a budget miss does not allocate
/// an io/serde diagnostic merely to classify a known encoded-size failure.
pub(crate) fn record_fits_encoding(record: &CaptureRecord) -> bool {
    let mut counter = Counter {
        remaining: record.charged.encoded_bytes,
        fits: true,
    };
    // CaptureRecord's closed serializer (including capture-wire nonfinite
    // numbers) supplies no caller Serialize callback. The sink cannot fail.
    serde_json::to_writer(&mut counter, record).is_ok() && counter.fits
}

pub(crate) fn intervention_fits_encoding(
    record: &eredu_core::intervention::InterventionRecord,
) -> bool {
    let Some(remaining) = record
        .evidence
        .iter()
        .try_fold(record.charged.encoded_bytes, |bytes, evidence| {
            bytes.checked_add(evidence.charged.encoded_bytes)
        })
    else {
        return false;
    };
    let mut counter = Counter {
        remaining,
        fits: true,
    };
    serde_json::to_writer(&mut counter, record).is_ok() && counter.fits
}

/// Existing companion result kind, checked before infallible terminal publication.
/// Preview keeps the original global selected extent separate from its prefix.
pub(super) fn prefill_terminal_outcome(record: &CaptureRecord) -> Option<CaptureOutcome> {
    match record.payload.as_ref()? {
        CapturePayload::Summary(_) | CapturePayload::Histogram(_) => Some(CaptureOutcome::Captured),
        payload => {
            let value = payload.as_tensor()?;
            let available = elements(record.selected_shape.as_deref()?).ok()?;
            let emitted = u64::try_from(value.data().len()).ok()?;
            if value.shape() != [value.data().len()] || emitted > available {
                return None;
            }
            Some(if emitted < available {
                CaptureOutcome::Truncated {
                    available_elements: available,
                    emitted_elements: emitted,
                }
            } else {
                CaptureOutcome::Captured
            })
        }
    }
}

pub(in crate::capture) fn encoded_size<T: serde::Serialize>(value: &T) -> Option<u64> {
    let mut counter = Counter {
        remaining: u64::MAX,
        fits: true,
    };
    (serde_json::to_writer(&mut counter, value).is_ok() && counter.fits)
        .then_some(u64::MAX - counter.remaining)
}

/// Checks the complete companion plus exact terminal scalar spelling changes.
/// No report/record/payload clone or second JSON/error buffer is constructed.
pub(super) fn prefill_reductions_fit_encoding(
    report: &eredu_core::speculative::SpeculativePrefillReductions,
) -> bool {
    use eredu_core::speculative::SpeculativePrefillReductionStatus as Status;
    let Some(terminal_growth) = report.records.iter().try_fold(0u64, |sum, entry| {
        if entry.status != Status::Pending {
            return Some(sum);
        }
        let outcome = prefill_terminal_outcome(&entry.record)?;
        let status = encoded_size(&Status::Complete)?.saturating_sub(encoded_size(&entry.status)?);
        let outcome = encoded_size(&outcome)?.saturating_sub(encoded_size(&entry.record.outcome)?);
        let remaining = entry.record.charged.encoded_bytes.checked_sub(outcome)?;
        let mut record_counter = Counter {
            remaining,
            fits: true,
        };
        if serde_json::to_writer(&mut record_counter, &entry.record).is_err()
            || !record_counter.fits
        {
            return None;
        }
        sum.checked_add(status)?.checked_add(outcome)
    }) else {
        return false;
    };
    let Some(terminal_growth) =
        report
            .interventions
            .iter()
            .try_fold(terminal_growth, |sum, entry| {
                if entry.status != Status::Pending {
                    return Some(sum);
                }
                let status =
                    encoded_size(&Status::Complete)?.saturating_sub(encoded_size(&entry.status)?);
                entry
                    .record
                    .evidence
                    .iter()
                    .try_fold(sum.checked_add(status)?, |sum, record| {
                        if matches!(record.outcome, CaptureOutcome::Skipped { .. }) {
                            return Some(sum);
                        }
                        let outcome = prefill_terminal_outcome(record)?;
                        let growth =
                            encoded_size(&outcome)?.saturating_sub(encoded_size(&record.outcome)?);
                        let remaining = record.charged.encoded_bytes.checked_sub(growth)?;
                        let mut counter = Counter {
                            remaining,
                            fits: true,
                        };
                        if serde_json::to_writer(&mut counter, record).is_err() || !counter.fits {
                            return None;
                        }
                        sum.checked_add(growth)
                    })
            })
    else {
        return false;
    };
    let Some(remaining) = report.charged.encoded_bytes.checked_sub(terminal_growth) else {
        return false;
    };
    let mut counter = Counter {
        remaining,
        fits: true,
    };
    serde_json::to_writer(&mut counter, report).is_ok() && counter.fits
}
