//! Shared bounded payload algebra for ordinary capture and intervention evidence.
//! Progress/authority and record ownership remain with the enclosing collector.
use super::*;

pub(super) struct State {
    pub(super) geometry: Option<Geometry>,
    pub(super) summary: reduction::Summary,
    pub(super) pending_summary: Option<reduction::Summary>,
    pub(super) observed_elements: u64,
    pub(super) pending_elements: u64,
    pub(super) sealed: bool,
    pub(super) preview: Option<preview::Preview>,
}

impl State {
    pub(super) fn validate(
        &mut self,
        logical: &CaptureRecord,
        physical: &CaptureRecord,
        start: u64,
        end: u64,
        logical_sequence: u64,
    ) -> Result<(), CaptureError> {
        self.validate_fixed(logical, physical, start, end, logical_sequence)
            .map_err(ConstructionError::into_capture)
    }
    pub(super) fn validate_fixed(
        &mut self,
        logical: &CaptureRecord,
        physical: &CaptureRecord,
        start: u64,
        end: u64,
        logical_sequence: u64,
    ) -> Result<(), ConstructionError> {
        let g = self
            .geometry
            .as_ref()
            .ok_or_else(|| ConstructionError::Invalid("zero seed emitted a reduction hook"))?;
        let count = g.validate_record_fixed(physical, start, end)?;
        if logical.source_dtype.is_some() && logical.source_dtype != physical.source_dtype {
            return Err(ConstructionError::Invalid(
                "logical reduction source precision changed",
            ));
        }
        if matches!(
            physical.source_dtype.as_ref(),
            Some(
                eredu_core::checkpoint::TensorDtype::Complex64
                    | eredu_core::checkpoint::TensorDtype::Encoded(_)
            )
        ) {
            return Err(ConstructionError::Unsupported(
                "logical reductions require scalar numeric source precision",
            ));
        }
        if physical.source_dtype.is_none() {
            return Err(ConstructionError::Invalid(
                "logical reduction requires actual source precision",
            ));
        }
        self.pending_elements = add(self.observed_elements, count)?;
        if end == logical_sequence && self.pending_elements != g.elements() {
            return Err(ConstructionError::Invalid(
                "logical reduction coverage is incomplete",
            ));
        }
        if let Some(preview) = &mut self.preview {
            preview.validate_fixed(g, physical, logical, start, end, count)?;
            return Ok(());
        }
        if count == 0
            && matches!(
                physical.outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::NotInvoked
                }
            )
        {
            if matches!(logical.payload, Some(CapturePayload::Summary(_))) {
                self.pending_summary = Some(self.summary.clone());
            }
            return Ok(());
        }
        if physical.outcome != CaptureOutcome::Captured {
            return Err(ConstructionError::Invalid(
                "logical reduction hook did not complete",
            ));
        }
        match (&physical.payload, &logical.payload) {
            (Some(CapturePayload::Summary(value)), Some(CapturePayload::Summary(_))) => {
                self.pending_summary = Some(self.summary.appended_fixed(value, count)?);
            }
            (Some(CapturePayload::Histogram(value)), Some(CapturePayload::Histogram(total))) => {
                reduction::validate_histogram_fixed(value, &total.edges, count)?;
                reduction::check_histogram_add_fixed(total, value)?;
            }
            _ => {
                return Err(ConstructionError::Invalid(
                    "logical reduction payload differs from admission",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn update(
        &mut self,
        logical: &mut CaptureRecord,
        physical: &CaptureRecord,
        start: u64,
        end: u64,
    ) -> Result<(), CaptureError> {
        self.update_fixed(logical, physical, start, end)
            .map_err(ConstructionError::into_capture)
    }
    pub(super) fn update_fixed(
        &mut self,
        logical: &mut CaptureRecord,
        physical: &CaptureRecord,
        start: u64,
        end: u64,
    ) -> Result<(), ConstructionError> {
        if let Some(preview) = &mut self.preview {
            preview.update(
                self.geometry.as_ref().expect("active geometry"),
                physical,
                logical,
                start,
                end,
            );
        } else if let Some(next) = self.pending_summary.take() {
            self.summary = next;
            logical.payload = Some(CapturePayload::Summary(self.summary.value()));
        } else if let (
            Some(CapturePayload::Histogram(total)),
            Some(CapturePayload::Histogram(value)),
        ) = (&mut logical.payload, &physical.payload)
        {
            reduction::add_histogram_fixed(total, value)?;
        }
        logical.source_dtype = physical.source_dtype.clone();
        // Check the actual global record while still in the fallible,
        // agreed invocation completion. No second JSON buffer is built.
        if !crate::capture::record_fits_encoding(logical) {
            return Err(ConstructionError::Invalid(
                "logical reduction encoded reservation was underestimated",
            ));
        }
        Ok(())
    }

    pub(super) fn commit(&mut self, end: u64, logical_sequence: u64) {
        self.observed_elements = self.pending_elements;
        if let Some(preview) = &mut self.preview {
            preview.commit();
        }
        self.sealed = end == logical_sequence;
    }
}

/// Reserve and construct the same logical record used by capture and evidence.
/// Inline State/record storage is reserved by the enclosing finite population.
pub(super) fn prepare(
    ledger: &mut CaptureLedger,
    total: &mut CaptureUsage,
    point: &eredu_core::ObservationPoint,
    selection: &CaptureSelection,
    rows: u64,
) -> Result<(CaptureRecord, State), CaptureError> {
    prepare_with_metadata(ledger, total, point, selection, rows, Metadata::ordinary())
        .map_err(ConstructionError::into_capture)
}
pub(super) fn prepare_with_metadata(
    ledger: &mut CaptureLedger,
    total: &mut CaptureUsage,
    point: &eredu_core::ObservationPoint,
    selection: &CaptureSelection,
    rows: u64,
    metadata: Metadata<'_>,
) -> Result<(CaptureRecord, State), ConstructionError> {
    metadata.controls(construction_control_bytes().ok_or(ConstructionError::Overflow)?)?;
    let g = (rows != 0)
        .then(|| Geometry::new_fixed(point, selection, 1, rows))
        .transpose()?;
    let wire = metadata_reservation(selection, point)?;
    // Inline record/state controls are already in `controls`. Only the
    // three copied strings and two logical shape buffers are additional.
    let metadata_usage = CaptureUsage {
        host_bytes: add(
            add(
                add(selection.id.len() as u64, selection.path.len() as u64)?,
                point.node_id.len() as u64,
            )?,
            mul(
                g.map_or(0, |value| value.rank()) as u64,
                2 * std::mem::size_of::<u64>() as u64,
            )?,
        )?,
        encoded_bytes: wire.encoded_bytes,
        ..Default::default()
    };
    ledger.reserve_quota(metadata_usage)?;
    *total = total.checked_add(metadata_usage)?;
    let preview_plan = match (&selection.transform, &g) {
        (CaptureTransform::Preview { max_elements }, Some(g)) => {
            Some(preview::Plan::prepare_fixed(point, *max_elements, g)?)
        }
        _ => None,
    };
    let payload_usage = match &selection.transform {
        CaptureTransform::Preview { .. } => {
            preview_plan.map_or(Ok(CaptureUsage::default()), |plan| plan.usage())?
        }
        CaptureTransform::Summary => CaptureUsage {
            host_bytes: 0, // scalar payload is inline in the already charged record
            encoded_bytes: 512,
            ..Default::default()
        },
        CaptureTransform::Histogram { edges } => CaptureUsage {
            host_bytes: add(
                mul(edges.len() as u64, std::mem::size_of::<f32>() as u64)?,
                mul(
                    edges.len().saturating_sub(1) as u64,
                    std::mem::size_of::<u64>() as u64,
                )?,
            )?,
            encoded_bytes: add(256, mul(edges.len() as u64, 64)?)?,
            ..Default::default()
        },
        _ => unreachable!(),
    };
    let skipped = if rows == 0 {
        Some(CaptureSkipReason::NotInvoked)
    } else {
        ledger.reserve(payload_usage)?
    };
    let charged = if skipped.is_none() {
        *total = total.checked_add(payload_usage)?;
        metadata_usage.checked_add(payload_usage)?
    } else {
        metadata_usage
    };
    let mut preview = None;
    let payload = if skipped.is_none() {
        if let Some(plan) = preview_plan {
            let (state, payload) = plan.allocate_with_metadata(metadata)?;
            preview = Some(state);
            payload
        } else {
            Some(match &selection.transform {
                CaptureTransform::Summary => {
                    CapturePayload::Summary(reduction::Summary::default().value())
                }
                CaptureTransform::Histogram { edges } => {
                    CapturePayload::Histogram(CaptureHistogram {
                        edges: metadata.copy(edges)?,
                        counts: metadata.zeros(edges.len() - 1, 0u64)?,
                        below: 0,
                        above: 0,
                        non_finite: 0,
                    })
                }
                _ => unreachable!(),
            })
        }
    } else {
        None
    };
    let record = CaptureRecord {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selection_id: metadata.text(&selection.id)?,
        path: metadata.text(&selection.path)?,
        node_id: metadata.text(&point.node_id)?,
        position: point.position,
        source_shape: g
            .map(|v| metadata.copy(&v.source()[..v.rank()]))
            .transpose()?,
        source_dtype: None,
        selected_shape: g
            .map(|v| metadata.copy(&v.selected()[..v.rank()]))
            .transpose()?,
        outcome: skipped.map_or(CaptureOutcome::Missing, |reason| CaptureOutcome::Skipped {
            reason,
        }),
        payload,
        charged,
    };
    let state = State {
        geometry: g,
        summary: reduction::Summary::default(),
        pending_summary: None,
        observed_elements: 0,
        pending_elements: 0,
        sealed: rows == 0,
        preview,
    };
    Ok((record, state))
}

fn construction_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<CaptureRecord>(),
        size_of::<State>(),
        size_of::<(CaptureRecord, State)>(),
        size_of::<Result<(CaptureRecord, State), ConstructionError>>(),
        size_of::<Option<Geometry>>(),
        size_of::<Option<preview::Plan>>(),
        size_of::<Option<preview::Preview>>(),
        size_of::<Option<CapturePayload>>(),
        size_of::<Option<CaptureSkipReason>>(),
        5 * size_of::<CaptureUsage>(),
        size_of::<Result<Option<CaptureSkipReason>, CaptureError>>(),
        size_of::<(
            &mut CaptureLedger,
            &mut CaptureUsage,
            &eredu_core::ObservationPoint,
            &CaptureSelection,
            u64,
            Metadata<'_>,
        )>(),
        size_of::<Vec<u64>>(),
        size_of::<Vec<f32>>(),
        size_of::<String>(),
        size_of::<Result<String, ConstructionError>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
