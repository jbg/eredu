//! One ordinary p0 frame over the existing guarded media transactions.
//! Geometry and logical capture quota are not original native funding.
use super::*;
mod transforms;
use crate::layered::{PreparedCaptureSelection, SharedLayeredObservationPaths};
use crate::prefill::PrefillChunk;
use eredu_core::{
    checkpoint::TensorDtype, DistributedCommitEpoch, HostPreparationAuthority, InferenceGeometry,
    OutputDemand, SharedTensorObservation, TensorObservation, TensorObservationData,
};
use std::mem::size_of;

fn invalid() -> CaptureError {
    CaptureError::Invalid("prepared-media capture source, chunk or transaction differs".into())
}
fn progress_error(e: CapturePrefillProgressError) -> CaptureError {
    CaptureError::Invalid(e.to_string())
}

/// Immutable ordinary binding, authenticated against the actual native input,
/// selected session and path token before execution. It grants no funding.
#[derive(Clone, Debug)]
pub struct OrdinaryPrefillCapture {
    selection: PreparedCaptureSelection,
    geometry: InferenceGeometry,
}
impl OrdinaryPrefillCapture {
    pub fn new(
        selection: PreparedCaptureSelection,
        geometry: InferenceGeometry,
    ) -> Result<Self, CaptureError> {
        if !selection.is_prepared_media() || geometry.max_output_tokens == 0 {
            return Err(invalid());
        }
        selection.bind_geometry(geometry).map_err(|_| invalid())?;
        Ok(Self {
            selection,
            geometry,
        })
    }
    pub fn source(&self) -> &SharedCapturePlan {
        self.selection.source()
    }
    pub fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    pub fn requires_sequence(&self) -> bool {
        self.geometry.output == OutputDemand::Sequence
    }
    pub fn validate(
        &self,
        source: &SharedCapturePlan,
        paths: &SharedLayeredObservationPaths,
        geometry: InferenceGeometry,
    ) -> Result<(), CaptureError> {
        if geometry != self.geometry {
            return Err(invalid());
        }
        self.selection
            .validate_sources(source, paths)
            .map_err(|_| invalid())?;
        self.selection
            .bind_geometry(geometry)
            .map_err(|_| invalid())?;
        Ok(())
    }
    fn policy(&self) -> Result<CapturePrefillObservationPolicy<'_>, CaptureError> {
        CapturePrefillObservationPolicy::from_bound(
            self.selection
                .bind_geometry(self.geometry)
                .map_err(|_| invalid())?,
        )
        .map_err(progress_error)
    }
    pub(super) fn rebind(&self, source: SharedCapturePlan) -> Result<Self, CaptureError> {
        let selection = self
            .selection
            .paths()
            .prepare_media_capture_selection(&source)
            .map_err(|_| invalid())?;
        let mut geometry = self.geometry;
        geometry.max_output_tokens = source.admission().request().max_predictions;
        Self::new(selection, geometry)
    }
}
struct Row {
    aggregate: Option<transforms::Aggregate>,
    payload: Option<CapturePayload>,
    elements: u64,
    terminal: Option<CaptureOutcome>,
    data: Option<Vec<f32>>,
    written: usize,
    dtype: Option<TensorDtype>,
    progress: CapturePrefillRowProgress,
}
use super::prefill_transaction::{self as transaction, Phase};
pub(super) struct Progress {
    rows: Vec<Row>,
    chunk: PrefillChunk,
    logical_epoch: Option<DistributedCommitEpoch>,
    epoch: Option<DistributedCommitEpoch>,
    phase: Phase,
    failed: bool,
    binding: OrdinaryPrefillCapture,
}
impl CaptureSession {
    /// Installs an ordinary owner even for an empty admitted capture source.
    pub fn with_ordinary_prefill(
        binding: OrdinaryPrefillCapture,
        host: &HostPreparationAuthority,
    ) -> Result<Self, CaptureError> {
        let custody = host.clone();
        let result = (|| {
            binding.source().retain_host_preparation(host)?;
            let mut result = Self::new(binding.source().clone());
            result.ordinary_error_custody = Some(host.clone());
            result.retain_host_preparation(host)?;
            result.ordinary_prefill = Some(binding);
            Ok(result)
        })();
        ordinary_error::retain_result(Some(custody), result)
    }
    /// Immutable source retained for ordinary snapshot authentication, including
    /// while restoring a pending prompt from an earlier drained boundary.
    pub fn ordinary_prefill_source_binding(&self) -> Option<&OrdinaryPrefillCapture> {
        self.ordinary_prefill.as_ref()
    }
    pub fn ordinary_prefill_capture(&self) -> Option<&OrdinaryPrefillCapture> {
        if !self.has_step || self.prediction == 0 && self.ordinary_progress.is_some() {
            self.ordinary_prefill.as_ref()
        } else {
            None
        }
    }
    pub(crate) fn begin_ordinary_prefill(
        &mut self,
        chunk: &PrefillChunk,
    ) -> Result<(), CaptureError> {
        let host = self.ordinary_error_custody.clone();
        let result = self.begin_ordinary_prefill_unretained(chunk);
        ordinary_error::retain_result(host, result)
    }

    fn begin_ordinary_prefill_unretained(
        &mut self,
        chunk: &PrefillChunk,
    ) -> Result<(), CaptureError> {
        let binding = self.ordinary_prefill.as_ref().ok_or_else(invalid)?;
        let geometry = binding.geometry;
        let next = match &self.ordinary_progress {
            Some(p) if p.phase == Phase::Between && !p.failed => p.chunk.input.end,
            None if !self.has_step && self.records.is_none() && self.transaction.is_none() => 0,
            _ => return Err(invalid()),
        };
        let end = next
            .checked_add(geometry.prefill_chunk_positions)
            .ok_or(CaptureError::Overflow)?
            .min(geometry.input_positions);
        if chunk.input != (next..end)
            || next >= end
            || geometry.cached_positions.checked_add(next) != Some(chunk.position)
            || chunk.output != geometry.output.for_chunk(end == geometry.input_positions)
        {
            return Err(invalid());
        }
        match &mut self.ordinary_progress {
            Some(p) => {
                p.chunk = chunk.clone();
                p.phase = Phase::Announced;
            }
            None => {
                self.ordinary_progress = Some(Progress {
                    binding: binding.clone(),
                    chunk: chunk.clone(),
                    logical_epoch: None,
                    epoch: None,
                    phase: Phase::Announced,
                    failed: false,
                    rows: Vec::new(),
                })
            }
        }
        Ok(())
    }
    pub(crate) fn prepare_ordinary_prefill(
        &mut self,
        epoch: DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), CaptureError> {
        let host = self.ordinary_error_custody.clone();
        let result = self.prepare_ordinary_prefill_unretained(epoch, pass);
        ordinary_error::retain_result(host, result)
    }

    fn prepare_ordinary_prefill_unretained(
        &mut self,
        epoch: DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), CaptureError> {
        let mut p = self.ordinary_progress.take().ok_or_else(invalid)?;
        let result = (|| {
            if pass != crate::ExpertPass::Prefill
                || !transaction::can_prepare(p.phase, p.failed, p.epoch, None, epoch)
            {
                return Err(invalid());
            }
            if p.logical_epoch.is_none() {
                self.claim_step_epoch(epoch, false)
                    .map_err(policy::public_error)?;
                p.logical_epoch = Some(epoch);
                self.begin_step_inner(CapturePhase::Prefill, 0)?;
                let count = p.binding.source().admission().plan().selections.len();
                let bytes = count
                    .checked_mul(std::mem::size_of::<Row>())
                    .ok_or(CaptureError::Overflow)?;
                policy::reserve_required(
                    &mut self.ledger,
                    CaptureUsage {
                        host_bytes: bytes.try_into().map_err(|_| CaptureError::Overflow)?,
                        ..CaptureUsage::default()
                    },
                )?;
                let policy = p.binding.policy()?;
                p.rows.reserve_exact(count);
                for index in 0..count {
                    p.rows.push(Row {
                        aggregate: None,
                        payload: None,
                        elements: 0,
                        terminal: None,
                        data: None,
                        written: 0,
                        dtype: None,
                        progress: policy
                            .row(index)
                            .map_err(progress_error)?
                            .initial_progress(),
                    });
                }
            } else {
                if self.transaction
                    != p.logical_epoch
                        .map(|e| (e, CaptureTransactionStatus::Pending))
                    || self.last_transaction_epoch.is_some_and(|old| old >= epoch)
                {
                    return Err(invalid());
                }
                self.last_transaction_epoch = Some(epoch);
            }
            p.epoch = Some(epoch);
            p.phase = Phase::Prepared;
            Ok(())
        })();
        if result.is_err() {
            p.failed = true;
        }
        self.ordinary_progress = Some(p);
        result
    }
    pub(super) fn observe_ordinary_prefill<B: CaptureBackend>(
        &mut self,
        backend: &mut B,
        path: &str,
        tensor: &B::Tensor,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        let mut p = self.ordinary_progress.take().ok_or_else(invalid)?;
        let started = std::time::Instant::now();
        let result = (|| {
            if p.failed || p.phase != Phase::Prepared {
                return Err(invalid().into());
            }
            let policy = p.binding.policy()?;
            let records = self.records.as_mut().ok_or_else(invalid)?;
            for (index, state) in p.rows.iter_mut().enumerate() {
                let row = policy.row(index).map_err(progress_error)?;
                let decision = row
                    .begin_hook(&mut state.progress, &p.chunk, path)
                    .map_err(progress_error)?;
                if decision == CapturePrefillHookDecision::Ignore {
                    continue;
                }
                let record = &mut records[index];
                let attempt: Result<(), CaptureExecutionError<B::Error>> = (|| {
                    if row.transform_plan().is_some() {
                        return transforms::observe(
                            backend,
                            tensor,
                            &row,
                            state,
                            record,
                            &mut self.ledger,
                            &p.chunk,
                            decision == CapturePrefillHookDecision::First,
                        );
                    }
                    if row.candidate().is_some() {
                        return transforms::candidates(
                            backend,
                            tensor,
                            &row,
                            state,
                            record,
                            &mut self.ledger,
                            &p.chunk,
                        );
                    }
                    if row.token_scores().is_some() {
                        return transforms::token_scores(
                            backend,
                            tensor,
                            &row,
                            state,
                            record,
                            &mut self.ledger,
                            &p.chunk,
                        );
                    }
                    let assembly = row.assembly().ok_or_else(invalid)?;
                    let fragment = assembly
                        .fragment(p.chunk.input.start / p.binding.geometry.prefill_chunk_positions)
                        .map_err(|_| invalid())?;
                    if decision == CapturePrefillHookDecision::First {
                        let usage = ordinary_usage(backend, assembly)?;
                        if let Some(reason) = row
                            .reserve_first(&mut state.progress, &mut self.ledger, usage)
                            .map_err(progress_error)?
                        {
                            record.outcome = CaptureOutcome::Skipped { reason };
                            return Ok(());
                        }
                        record.charged = record.charged.checked_add(usage)?;
                        state.data = Some(vec![0.0; assembly.logical_geometry().elements()]);
                        record.source_shape = Some(
                            assembly
                                .logical_geometry()
                                .source_shape()
                                .iter()
                                .map(|n| *n as u64)
                                .collect(),
                        );
                        record.selected_shape = Some(
                            assembly
                                .selected_shape()
                                .iter()
                                .map(|n| *n as u64)
                                .collect(),
                        );
                    }
                    let dtype = backend
                        .validate_capture_prefill_source(tensor, &fragment)
                        .ok_or_else(invalid)?
                        .map_err(CaptureExecutionError::Backend)?;
                    if !matches!(
                        dtype,
                        TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16
                    ) || state.dtype.as_ref().is_some_and(|old| old != &dtype)
                    {
                        return Err(invalid().into());
                    }
                    state.dtype = Some(dtype.clone());
                    record.source_dtype = Some(dtype);
                    if fragment.output_elements() != 0 {
                        let values = backend
                            .capture_prefill_fragment(tensor, &fragment)
                            .ok_or_else(invalid)?
                            .map_err(CaptureExecutionError::Backend)?;
                        if values.shape() != fragment.selected_shape() {
                            return Err(invalid().into());
                        }
                        let (_, values) = values.into_parts();
                        let TensorObservationData::F32(values) = values else {
                            return Err(invalid().into());
                        };
                        if values.len() != fragment.output_elements() {
                            return Err(invalid().into());
                        }
                        let destination = state.data.as_mut().ok_or_else(invalid)?;
                        for (value, mapping) in values.into_iter().zip(fragment.mappings()) {
                            *destination
                                .get_mut(mapping.destination_index())
                                .ok_or_else(invalid)? = value;
                            state.written =
                                state.written.checked_add(1).ok_or(CaptureError::Overflow)?;
                        }
                    }
                    row.finish_hook(&mut state.progress, &fragment)
                        .map_err(progress_error)?;
                    Ok(())
                })();
                if let Err(error) = attempt {
                    state.progress.fail_hook();
                    record.outcome = CaptureOutcome::Failed {
                        reason: match &error {
                            CaptureExecutionError::Admission(error) => {
                                policy::failure_reason(error)
                            }
                            CaptureExecutionError::Backend(_) => CaptureFailureReason::Native,
                        },
                        message: bounded_diagnostic(&error),
                    };
                    return Err(error);
                }
            }
            Ok(())
        })();
        self.capture_seconds += started.elapsed().as_secs_f64();
        if result.is_err() {
            p.failed = true;
        }
        self.ordinary_progress = Some(p);
        result
    }
    pub(crate) fn complete_ordinary_prefill(
        &mut self,
        epoch: DistributedCommitEpoch,
    ) -> Result<(), CaptureError> {
        let host = self.ordinary_error_custody.clone();
        let result = self.complete_ordinary_prefill_unretained(epoch);
        ordinary_error::retain_result(host, result)
    }

    fn complete_ordinary_prefill_unretained(
        &mut self,
        epoch: DistributedCommitEpoch,
    ) -> Result<(), CaptureError> {
        let mut p = self.ordinary_progress.take().ok_or_else(invalid)?;
        let result = (|| {
            if !transaction::can_complete(p.phase, p.failed, p.epoch, epoch) {
                return Err(invalid());
            }
            let policy = p.binding.policy()?;
            let chunk = p.chunk.input.start / p.binding.geometry.prefill_chunk_positions;
            for (index, state) in p.rows.iter().enumerate() {
                policy
                    .row(index)
                    .map_err(progress_error)?
                    .validate_chunk_end(&state.progress, chunk)
                    .map_err(progress_error)?;
            }
            if p.chunk.input.end == p.binding.geometry.input_positions {
                let host = self.owner.retained()?;
                for (index, state) in p.rows.iter_mut().enumerate() {
                    if state.aggregate.is_some() {
                        let row = policy.row(index).map_err(progress_error)?;
                        transforms::finish(
                            state,
                            &mut self.records.as_mut().ok_or_else(invalid)?[index],
                            &row,
                            &host,
                        )?;
                        continue;
                    }
                    let Some(values) = state.data.take() else {
                        continue;
                    };
                    let assembly = CapturePrefillRowAssembly::prepare(
                        p.binding.source().admission(),
                        index,
                        p.binding.geometry,
                    )
                    .map_err(|_| invalid())?;
                    if state.written != assembly.logical_geometry().elements()
                        || values.len() != state.written
                    {
                        return Err(invalid());
                    }
                    let tensor = TensorObservation::new(
                        assembly.selected_shape().to_vec(),
                        TensorObservationData::F32(values),
                    )
                    .map_err(|_| invalid())?;
                    let record = &mut self.records.as_mut().ok_or_else(invalid)?[index];
                    record.payload = Some(CapturePayload::SharedTensor(
                        SharedTensorObservation::retain(tensor, host.clone()),
                    ));
                    record.outcome = CaptureOutcome::Captured;
                    if !record_fits_encoding(record) {
                        return Err(CaptureError::Invalid(
                            "prepared-media capture encoding exceeded reservation".into(),
                        ));
                    }
                }
                self.complete_transaction(p.logical_epoch.ok_or_else(invalid)?)?;
            }
            for (index, state) in p.rows.iter_mut().enumerate() {
                policy
                    .row(index)
                    .map_err(progress_error)?
                    .advance_chunk(&mut state.progress, chunk)
                    .map_err(progress_error)?;
            }
            p.phase = Phase::Complete;
            Ok(())
        })();
        if result.is_err() {
            p.failed = true;
        }
        self.ordinary_progress = Some(p);
        result
    }
    pub(crate) fn finish_ordinary_chunk(&mut self, epoch: DistributedCommitEpoch, committed: bool) {
        let Some(p) = &mut self.ordinary_progress else {
            return;
        };
        transaction::finish(&mut p.phase, &mut p.failed, p.epoch, epoch, committed);
    }
    pub(crate) fn finish_ordinary_prefill(&mut self, committed: bool) {
        let Some(mut p) = self.ordinary_progress.take() else {
            return;
        };
        let accepted = committed
            && !p.failed
            && p.phase == Phase::Between
            && p.chunk.input.end == p.binding.geometry.input_positions;
        if let Some(records) = self.records.as_mut() {
            for (state, record) in p.rows.iter_mut().zip(records) {
                if let Some(outcome) = state.terminal.take() {
                    if accepted {
                        record.outcome = outcome;
                    } else {
                        record.payload = None;
                        record.outcome = CaptureOutcome::Missing;
                    }
                }
            }
        }
        if let Some(epoch) = p.logical_epoch {
            self.finish_transaction(epoch, accepted);
        }
        drop(p);
    }
}
fn ordinary_usage<B: CaptureBackend>(
    backend: &B,
    assembly: &CapturePrefillRowAssembly<'_>,
) -> Result<CaptureUsage, CaptureError> {
    let rank = assembly.selected_shape().len();
    let elements = assembly.logical_geometry().elements();
    let host = elements
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|n| n.checked_add(rank.checked_mul(3 * std::mem::size_of::<usize>())?))
        .ok_or(CaptureError::Overflow)?;
    let control = SharedTensorObservation::retained_control_bytes::<HostPreparationAuthority>()
        .ok_or(CaptureError::Overflow)?;
    let mut usage = CaptureUsage {
        captures: 1,
        retained_bytes: 0,
        host_bytes: u64::try_from(host)
            .map_err(|_| CaptureError::Overflow)?
            .checked_add(control)
            .ok_or(CaptureError::Overflow)?,
        encoded_bytes: u64::try_from(elements)
            .map_err(|_| CaptureError::Overflow)?
            .checked_mul(32)
            .and_then(|n| n.checked_add(128))
            .ok_or(CaptureError::Overflow)?,
    };
    for chunk in 0..assembly.chunk_count() {
        let fragment = assembly.fragment(chunk).map_err(|_| invalid())?;
        let mut physical = backend.estimate_capture_prefill_fragment(&fragment)?;
        physical.captures = 0;
        physical.encoded_bytes = 0;
        usage = usage.checked_add(physical)?;
    }
    Ok(usage)
}
