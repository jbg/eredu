//! Chunked sparse capture under the ordinary run ledger and commit boundary.
use super::*;

impl CaptureSession {
    pub(crate) fn wants_routed_units(&self, routing: &str) -> bool {
        self.plan.points().iter().zip(self.records.iter().flatten()).any(|(point, record)| {
            matches!(&point.value_type, eredu_core::ObservationValueType::RoutedUnits { routing: path, .. } if path == routing)
                && !matches!(record.outcome, CaptureOutcome::Skipped { .. })
        })
    }

    pub(crate) fn observe_routed_units<B: CaptureBackend>(
        &mut self,
        backend: &mut B,
        routing: &str,
        effective: bool,
        source: &RoutedUnitCaptureSource<'_, B::Tensor>,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        if self.partition.is_some() {
            return Err(CaptureError::Invalid(
                "sparse capture requires distributed route receipts".into(),
            )
            .into());
        }
        let tensor_geometry = self.tensor_geometry()?;
        let window = self.invocation_window;
        let phase = self.phase;
        let prediction = self.prediction;
        let records = self
            .records
            .as_mut()
            .ok_or_else(|| CaptureError::Invalid("capture step not started".into()))?;
        for (index, ((selection, point), record)) in self
            .plan
            .plan()
            .selections
            .iter()
            .zip(self.plan.points())
            .zip(records)
            .enumerate()
        {
            let eredu_core::ObservationValueType::RoutedUnits {
                routing: path,
                geometry,
            } = &point.value_type
            else {
                continue;
            };
            if path != routing
                || effective
                    != (point.position == eredu_core::ObservationPosition::AfterIntervention)
                || matches!(record.outcome, CaptureOutcome::Skipped { .. })
            {
                continue;
            }
            let started = std::time::Instant::now();
            let result = (|| {
                if window.is_some() && record.payload.is_none() {
                    let metadata = partition::fragment_metadata_usage(
                        selection,
                        point,
                        point.axes.as_ref().map_or(32, Vec::len),
                    )?;
                    if let Some(reason) = self.ledger.reserve(metadata)? {
                        record.outcome = CaptureOutcome::Skipped { reason };
                        return Ok(());
                    }
                    record.charged = record.charged.checked_add(metadata)?;
                }
                let (shape, slice) = routed_invocation_geometry(
                    &self.plan,
                    index,
                    phase,
                    prediction,
                    tensor_geometry,
                    window,
                )?;
                geometry.validate_slice(&slice)?;
                let dtype = backend.source_dtype(source.values);
                if record.payload.is_none() {
                    if record.outcome != CaptureOutcome::Missing {
                        return Err(CaptureError::Invalid(
                            "invalid routed-unit capture state".into(),
                        )
                        .into());
                    }
                    let usage = backend.estimate_routed_units(&shape, selection, &slice)?;
                    if let Some(reason) = self.ledger.reserve(usage)? {
                        record.outcome = CaptureOutcome::Skipped { reason };
                        return Ok(());
                    }
                    record.charged = record.charged.checked_add(usage)?;
                    record.source_shape = Some(shape);
                    record.selected_shape = Some(slice.shape.clone());
                    record.source_dtype = dtype.clone();
                    record.payload = Some(CapturePayload::RoutedUnits(RoutedUnitCapture {
                        geometry: *geometry,
                        source_token_ranges: vec![],
                        rows: vec![],
                    }));
                } else if record.source_dtype != dtype {
                    return Err(CaptureError::Invalid(
                        "routed-unit dtype changed between chunks".into(),
                    )
                    .into());
                }
                let mut fragment = backend
                    .capture_routed_units(source, *geometry, &slice)
                    .ok_or_else(|| {
                        CaptureError::Unsupported("routed-unit collector unavailable".into())
                    })?
                    .map_err(CaptureExecutionError::Backend)?;
                fragment.validate_rows(&slice)?;
                if fragment.geometry != *geometry {
                    return Err(CaptureError::Invalid(
                        "routed-unit collector changed bank geometry".into(),
                    )
                    .into());
                }
                let Some(CapturePayload::RoutedUnits(payload)) = &mut record.payload else {
                    return Err(
                        CaptureError::Invalid("routed-unit payload kind changed".into()).into(),
                    );
                };
                let source_tokens = record.source_shape.as_ref().unwrap()[0];
                if fragment.source_token_ranges.len() != 1 {
                    return Err(CaptureError::Invalid(
                        "expected one native routed-unit chunk".into(),
                    )
                    .into());
                }
                let [start, end] = fragment.source_token_ranges[0];
                if start >= end
                    || end > source_tokens
                    || payload
                        .source_token_ranges
                        .last()
                        .map_or(0, |range| range[1])
                        != start
                    || fragment
                        .rows
                        .iter()
                        .any(|row| row.token < start || row.token >= end)
                {
                    return Err(
                        CaptureError::Invalid("invalid routed-unit chunk coverage".into()).into(),
                    );
                }
                if add(payload.rows.len() as u64, fragment.rows.len() as u64)?
                    > mul(slice.shape[0], slice.shape[1])?
                {
                    return Err(CaptureError::Invalid("excess routed-unit receipts".into()).into());
                }
                payload.rows.append(&mut fragment.rows);
                payload
                    .source_token_ranges
                    .append(&mut fragment.source_token_ranges);
                Ok(())
            })();
            self.capture_seconds += started.elapsed().as_secs_f64();
            if let Err(error) = result {
                record.payload = None;
                let reason = match &error {
                    CaptureExecutionError::Backend(_) => CaptureFailureReason::Native,
                    CaptureExecutionError::Admission(error) => policy::failure_reason(error),
                };
                record.outcome = CaptureOutcome::Failed {
                    reason,
                    message: bounded_diagnostic(&error),
                };
                return Err(error);
            }
        }
        Ok(())
    }

    pub(crate) fn finish_routed_captures(&mut self) -> Result<(), CaptureError> {
        let mut first_failure = None;
        let window = self.invocation_window;
        let physical = window.map(|_| self.tensor_geometry()).transpose()?;
        for (index, ((selection, point), record)) in self
            .plan
            .plan()
            .selections
            .iter()
            .zip(self.plan.points())
            .zip(self.records.iter_mut().flatten())
            .enumerate()
        {
            if record.outcome == CaptureOutcome::Captured {
                continue;
            }
            let result = (|| {
                let Some(CapturePayload::RoutedUnits(payload)) = &mut record.payload else {
                    return Ok(());
                };
                let shape = record
                    .source_shape
                    .as_ref()
                    .ok_or_else(|| CaptureError::Invalid("missing routed source shape".into()))?;
                let slice = if window.is_some() {
                    let (expected, slice) = routed_invocation_geometry(
                        &self.plan,
                        index,
                        self.phase,
                        self.prediction,
                        physical.expect("window has physical geometry"),
                        window,
                    )?;
                    if &expected != shape {
                        return Err(CaptureError::Invalid(
                            "routed physical window changed before completion".into(),
                        ));
                    }
                    slice
                } else {
                    resolve_slice(point, selection, shape)?
                };
                payload.finish_ordinary(&slice, shape[0])?;
                record.outcome = CaptureOutcome::Captured;
                let mut sink = CountingWriter {
                    written: 0,
                    limit: record.charged.encoded_bytes,
                };
                serde_json::to_writer(&mut sink, &*record).map_err(|_| {
                    CaptureError::Invalid("routed collector underestimated encoded size".into())
                })
            })();
            if let Err(error) = result {
                record.payload = None;
                record.outcome = CaptureOutcome::Failed {
                    reason: CaptureFailureReason::Invalid,
                    message: bounded_diagnostic(&error),
                };
                first_failure.get_or_insert(error);
            }
        }
        match first_failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Ordinary and original destinations use the same selected physical sparse
/// axes. The outer speculative envelope retains the independent logical span;
/// every actual invocation still pays and completes its own captured record.
fn routed_invocation_geometry(
    source: &AdmittedCapturePlan,
    index: usize,
    phase: CapturePhase,
    prediction: u64,
    physical: CaptureInvocationShape,
    window: Option<CaptureInvocationWindow>,
) -> Result<(Vec<u64>, ResolvedCaptureSlice), CaptureError> {
    if let Some(window) = window {
        let geometry = CaptureRoutedUnitsGeometry::prepare_window(
            source, index, phase, prediction, physical, window,
        )
        .map_err(|cause| CaptureError::Invalid(cause.to_string()))?;
        Ok((
            geometry.source_shape().iter().map(|&n| n as u64).collect(),
            ResolvedCaptureSlice {
                starts: geometry.starts().to_vec(),
                ends: geometry.ends().to_vec(),
                strides: geometry.strides().to_vec(),
                shape: geometry.shape().iter().map(|&n| n as u64).collect(),
            },
        ))
    } else {
        let point = &source.points()[index];
        let shape = physical
            .resolve(point)?
            .ok_or_else(|| CaptureError::Invalid("unknown routed-unit geometry".into()))?;
        let slice = resolve_slice(point, &source.plan().selections[index], &shape)?;
        Ok((shape, slice))
    }
}
