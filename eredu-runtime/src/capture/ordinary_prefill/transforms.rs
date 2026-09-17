//! Host-only provisional values in the same ordinary p0 transaction.
use super::*;
use crate::capture::{reduction, speculative::reductions::preview};
pub(super) enum Aggregate {
    Summary(reduction::Summary),
    Histogram,
    Preview(preview::Preview),
    Terminal,
}
fn payload_usage(plan: &CapturePrefillTransformPlan<'_>) -> Result<CaptureUsage, CaptureError> {
    let mut usage = match &plan.selection().transform {
        CaptureTransform::Summary => CaptureUsage {
            encoded_bytes: 512,
            ..Default::default()
        },
        CaptureTransform::Preview { max_elements } => {
            let p = preview::Plan::prepare(
                &plan.admission().points()[plan.selection_index()],
                *max_elements,
                plan.window(),
            )?;
            let mut u = p.usage()?;
            u.host_bytes = add(
                u.host_bytes,
                SharedTensorObservation::retained_control_bytes::<HostPreparationAuthority>()
                    .ok_or(CaptureError::Overflow)?,
            )?;
            u
        }
        _ => return Err(invalid()),
    };
    usage.captures = 1;
    usage.host_bytes = add(
        usage.host_bytes,
        mul(plan.window().rank() as u64, 2 * size_of::<u64>() as u64)?,
    )?;
    Ok(usage)
}
fn usage<B: CaptureBackend>(
    backend: &B,
    plan: &CapturePrefillTransformPlan<'_>,
) -> Result<CaptureUsage, CaptureError> {
    if matches!(plan.selection().transform, CaptureTransform::Summary) {
        return crate::capture::summary_prefill_usage(plan, |fragment| {
            backend.estimate_capture_prefill_transform(fragment)
        });
    }
    if matches!(
        plan.selection().transform,
        CaptureTransform::Histogram { .. }
    ) {
        return crate::capture::histogram_prefill_usage(plan, |fragment| {
            backend.estimate_capture_prefill_transform(fragment)
        });
    }
    let mut total = payload_usage(plan)?;
    for chunk in 0..plan.chunk_count() {
        let mut physical = backend.estimate_capture_prefill_transform(&plan.fragment(chunk)?)?;
        physical.captures = 0;
        physical.encoded_bytes = 0;
        total = total.checked_add(physical)?;
    }
    Ok(total)
}
pub(super) fn observe<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    row: &CapturePrefillObservationRow<'_>,
    state: &mut Row,
    record: &mut CaptureRecord,
    ledger: &mut CaptureLedger,
    chunk: &PrefillChunk,
    first: bool,
) -> Result<(), CaptureExecutionError<B::Error>> {
    let plan = row.transform_plan().ok_or_else(invalid)?;
    let fragment =
        plan.fragment(chunk.input.start / plan.inference_geometry().prefill_chunk_positions)?;
    if first {
        let charged = usage(backend, plan)?;
        if let Some(reason) = row
            .reserve_first(&mut state.progress, ledger, charged)
            .map_err(progress_error)?
        {
            record.outcome = CaptureOutcome::Skipped { reason };
            return Ok(());
        }
        record.charged = record.charged.checked_add(charged)?;
        record.source_shape = Some(plan.window().source()[..plan.window().rank()].to_vec());
        record.selected_shape = Some(plan.window().selected()[..plan.window().rank()].to_vec());
        state.aggregate = Some(match &plan.selection().transform {
            CaptureTransform::Summary => {
                let value = reduction::Summary::default();
                state.payload = Some(CapturePayload::Summary(value.value()));
                Aggregate::Summary(value)
            }
            CaptureTransform::Histogram { edges } => {
                state.payload = Some(CapturePayload::Histogram(CaptureHistogram {
                    edges: edges.clone(),
                    counts: vec![0; edges.len() - 1],
                    below: 0,
                    above: 0,
                    non_finite: 0,
                }));
                Aggregate::Histogram
            }
            CaptureTransform::Preview { max_elements } => {
                let (value, payload) = preview::Plan::prepare(
                    &plan.admission().points()[plan.selection_index()],
                    *max_elements,
                    plan.window(),
                )?
                .allocate();
                state.payload = payload;
                Aggregate::Preview(value)
            }
            _ => return Err(invalid().into()),
        });
    }
    let dtype = backend
        .validate_capture_prefill_transform_source(tensor, &fragment)
        .ok_or_else(invalid)?
        .map_err(CaptureExecutionError::Backend)?;
    if !matches!(
        dtype,
        TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32
    ) || state.dtype.as_ref().is_some_and(|old| old != &dtype)
    {
        return Err(invalid().into());
    }
    state.dtype = Some(dtype.clone());
    record.source_dtype = Some(dtype);
    let count = fragment.selected_elements();
    let physical = if count == 0 {
        None
    } else {
        Some(
            backend
                .capture_prefill_transform(tensor, &fragment)
                .ok_or_else(invalid)?
                .map_err(CaptureExecutionError::Backend)?,
        )
    };
    let total = add(state.elements, count)?;
    if total > plan.window().elements() {
        return Err(invalid().into());
    }
    match state.aggregate.as_mut().ok_or_else(invalid)? {
        Aggregate::Summary(value) => {
            if let Some(payload) = physical {
                let CapturePayload::Summary(part) = payload else {
                    return Err(invalid().into());
                };
                *value = value.appended(&part, count)?;
                state.payload = Some(CapturePayload::Summary(value.value()));
            }
        }
        Aggregate::Histogram => {
            if let Some(payload) = physical {
                let CapturePayload::Histogram(part) = payload else {
                    return Err(invalid().into());
                };
                let Some(CapturePayload::Histogram(out)) = state.payload.as_mut() else {
                    return Err(invalid().into());
                };
                reduction::validate_histogram(&part, &out.edges, count)?;
                reduction::add_histogram(out, &part)?;
            }
        }
        Aggregate::Preview(value) => {
            let outcome = if count == 0 {
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::NotInvoked,
                }
            } else {
                crate::capture::completed_capture_outcome(&plan.selection().transform, count)
            };
            value.validate_value(
                plan.window(),
                state.dtype.as_ref(),
                &outcome,
                physical.as_ref(),
                state.payload.as_ref(),
                chunk.input.start,
                chunk.input.end,
                count,
            )?;
            value.update_value(
                plan.window(),
                state.dtype.as_ref(),
                physical.as_ref(),
                &mut state.payload,
                chunk.input.start,
                chunk.input.end,
            );
            value.commit();
        }
        Aggregate::Terminal => return Err(invalid().into()),
    }
    state.elements = total;
    row.finish_transform_hook(&mut state.progress, &fragment)
        .map_err(progress_error)?;
    Ok(())
}
pub(super) fn candidates<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    row: &CapturePrefillObservationRow<'_>,
    state: &mut Row,
    record: &mut CaptureRecord,
    ledger: &mut CaptureLedger,
    chunk: &PrefillChunk,
) -> Result<(), CaptureExecutionError<B::Error>> {
    let physical = row.candidate_for_chunk(chunk).map_err(progress_error)?;
    let mut usage = backend.estimate_capture_prefill_candidates(&physical)?;
    usage.host_bytes = add(usage.host_bytes, 6 * size_of::<u64>() as u64)?;
    if let Some(reason) = row
        .reserve_first(&mut state.progress, ledger, usage)
        .map_err(progress_error)?
    {
        record.outcome = CaptureOutcome::Skipped { reason };
        return Ok(());
    }
    record.charged = record.charged.checked_add(usage)?;
    let logical = row.candidate().ok_or_else(invalid)?;
    record.source_shape = Some(logical.source_shape().iter().map(|&n| n as u64).collect());
    record.selected_shape = record.source_shape.clone();
    let dtype = backend.source_dtype(tensor).ok_or_else(invalid)?;
    if !matches!(
        dtype,
        TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32
    ) {
        return Err(invalid().into());
    }
    record.source_dtype = Some(dtype);
    let value = backend
        .capture_prefill_candidates(tensor, &physical)
        .ok_or_else(invalid)?
        .map_err(CaptureExecutionError::Backend)?;
    if value.candidates.len() != physical.count()
        || value.stage != CandidateScoreStage::RawLogitsBeforeSampling
        || value.source != CandidateLogitsSource::Original
        || value
            .candidates
            .iter()
            .any(|c| c.token_id as usize >= physical.vocabulary() || !c.score.is_finite())
        || value.candidates.windows(2).any(|v| v[0].score < v[1].score)
    {
        return Err(invalid().into());
    }
    state.payload = Some(CapturePayload::Candidates(value));
    state.aggregate = Some(Aggregate::Terminal);
    row.finish_candidate_hook(&mut state.progress, chunk)
        .map_err(progress_error)?;
    Ok(())
}
pub(super) fn token_scores<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    row: &CapturePrefillObservationRow<'_>,
    state: &mut Row,
    record: &mut CaptureRecord,
    ledger: &mut CaptureLedger,
    chunk: &PrefillChunk,
) -> Result<(), CaptureExecutionError<B::Error>> {
    let physical = row.token_scores_for_chunk(chunk).map_err(progress_error)?;
    let mut usage = backend.estimate_capture_prefill_token_scores(&physical)?;
    usage.host_bytes = add(usage.host_bytes, 6 * size_of::<u64>() as u64)?;
    if let Some(reason) = row
        .reserve_first(&mut state.progress, ledger, usage)
        .map_err(progress_error)?
    {
        record.outcome = CaptureOutcome::Skipped { reason };
        return Ok(());
    }
    record.charged = record.charged.checked_add(usage)?;
    let logical = row.token_scores().ok_or_else(invalid)?;
    record.source_shape = Some(logical.source_shape().iter().map(|&n| n as u64).collect());
    record.selected_shape = record.source_shape.clone();
    let dtype = backend.source_dtype(tensor).ok_or_else(invalid)?;
    if !matches!(
        dtype,
        TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32
    ) {
        return Err(invalid().into());
    }
    record.source_dtype = Some(dtype);
    let value = backend
        .capture_prefill_token_scores(tensor, &physical)
        .ok_or_else(invalid)?
        .map_err(CaptureExecutionError::Backend)?;
    if value.scores.len() != physical.count()
        || value.vocabulary != physical.vocabulary() as u64
        || !value.log_partition.is_finite()
        || value.stage != CandidateScoreStage::RawLogitsBeforeSampling
        || value.source != CandidateLogitsSource::Original
        || value
            .scores
            .iter()
            .zip(physical.token_ids())
            .any(|(score, &id)| {
                score.target.token_id != id
                    || !score.target.score.is_finite()
                    || !score.log_probability.is_finite()
                    || score.log_probability > 0.0
                    || score.rank == 0
                    || score.rank > value.vocabulary
            })
    {
        return Err(invalid().into());
    }
    state.payload = Some(CapturePayload::TokenScores(value));
    state.aggregate = Some(Aggregate::Terminal);
    row.finish_token_scores_hook(&mut state.progress, chunk)
        .map_err(progress_error)?;
    Ok(())
}
pub(super) fn finish(
    state: &mut Row,
    record: &mut CaptureRecord,
    row: &CapturePrefillObservationRow<'_>,
    host: &HostPreparationAuthority,
) -> Result<(), CaptureError> {
    let Some(aggregate) = state.aggregate.as_ref() else {
        return Ok(());
    };
    let transform = if let Some(plan) = row.transform_plan() {
        if state.elements != plan.window().elements() {
            return Err(invalid());
        }
        Some((plan.selection(), plan.window().elements()))
    } else {
        None
    };
    let payload = state.payload.take().ok_or_else(invalid)?;
    record.payload = Some(match payload {
        CapturePayload::Tensor(tensor) => {
            CapturePayload::SharedTensor(SharedTensorObservation::retain(tensor, host.clone()))
        }
        other => other,
    });
    record.outcome = match aggregate {
        Aggregate::Terminal => CaptureOutcome::Captured,
        _ => {
            let (selection, count) = transform.ok_or_else(invalid)?;
            crate::capture::completed_capture_outcome(&selection.transform, count)
        }
    };
    let fits = record_fits_encoding(record);
    state.terminal = Some(std::mem::replace(
        &mut record.outcome,
        CaptureOutcome::Missing,
    ));
    if !fits {
        return Err(CaptureError::Invalid(
            "ordinary transform encoding exceeded reservation".into(),
        ));
    }
    Ok(())
}
