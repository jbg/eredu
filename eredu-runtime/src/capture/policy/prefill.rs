//! Logical row progression only. Geometry does not prove causal/readout support.
use super::*;
use crate::prefill::PrefillChunk;
use eredu_core::{InferenceGeometry, capture::*};

/// Allocation-free logical sequencing rejection; never a completion or grant.
#[derive(Debug, thiserror::Error)]
pub enum CapturePrefillProgressError {
    /// The actual ordinary source cannot derive the requested fragment.
    #[error(transparent)]
    Geometry(#[from] CapturePrefillGeometryError),
    /// Existing coordinate/path policy rejection.
    #[error(transparent)]
    Policy(#[from] CaptureProtocolError),
    /// Original logical quota error, without a new counter or refund.
    #[error(transparent)]
    Quota(#[from] CaptureError),
    /// Another actual source, selected row or schedule was supplied.
    #[error("capture row progression source differs")]
    Identity,
    /// Repeated, stale or skipped canonical chunk.
    #[error("capture row progression chunk is out of order")]
    Order,
    /// A previous hook is incomplete, failed, or already charged.
    #[error("capture row hook attempt is incomplete or already charged")]
    Attempt,
    /// An expected hook was not emitted, even if its contribution is zero.
    #[error("capture row expected hook was not completed")]
    Missing,
    /// The selection requires a hook the admitted point cannot emit.
    #[error("selected p0 hook is unavailable")]
    UnavailableHook,
}
/// A logical decision only; neither a host claim nor permission to run a factory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePrefillHookDecision {
    /// The path or row is inactive; do not validate or invoke a factory.
    Ignore,
    /// First logical attempt; reserve the full value before constructing it.
    First,
    /// Already charged logical value; handle this next physical fragment.
    Continue,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Logical {
    Uncharged,
    Charged,
    Skipped,
    Failed,
    Finished,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hook {
    Unseen,
    Attempting,
    Seen,
}

/// Private monotone states retain the actual immutable source without cloning
/// its strings. This is not Clone, resettable, or replaceable through an API.
/// The shared source's physical storage remains an original admission obligation.
#[derive(Debug)]
pub struct CapturePrefillRowProgress {
    source: SharedCapturePlan,
    inference: InferenceGeometry,
    index: usize,
    next: u64,
    logical: Logical,
    hook: Hook,
    terminal: bool,
}
impl CapturePrefillRowProgress {
    /// Pure diagnostic state; this proves no payload coverage or native settlement.
    pub fn finished(&self) -> bool {
        self.logical == Logical::Finished
    }
    /// Poison an attempted row without releasing payload or refunding quota.
    pub fn fail_hook(&mut self) {
        self.logical = Logical::Failed;
    }
    pub(crate) fn advance_preflighted(&mut self) {
        debug_assert!(
            matches!(self.logical, Logical::Charged | Logical::Skipped)
                || (self.terminal && self.logical == Logical::Uncharged)
        );
        debug_assert!(
            self.logical == Logical::Skipped
                || self.hook == Hook::Seen
                || (self.terminal && self.logical == Logical::Uncharged)
        );
        self.next += 1; // checked finite schedule, validated before whole-row commit
        self.hook = Hook::Unseen;
        if self.next
            == self
                .inference
                .input_positions
                .div_ceil(self.inference.prefill_chunk_positions)
        {
            self.logical = Logical::Finished;
        }
    }
}

/// Borrowed actual admitted source and fixed schedule. Creation is crate-private
/// until an architecture-bound causal/readout companion authorizes activation.
/// In particular a Sequence axis alone does not authorize this program to run.
pub struct CapturePrefillObservationPolicy<'a> {
    source: &'a SharedCapturePlan,
    inference: InferenceGeometry,
    transformed: bool,
}
impl<'a> CapturePrefillObservationPolicy<'a> {
    /// Activate logical sequencing only from the actual architecture companion.
    /// This still grants no native execution, source pin or host claim.
    pub fn from_bound(
        bound: crate::layered::BoundCaptureSelection<'a>,
    ) -> Result<Self, CapturePrefillProgressError> {
        for index in 0..bound
            .selection()
            .source()
            .admission()
            .plan()
            .selections
            .len()
        {
            bound
                .selection()
                .declaration(index)
                .map_err(|_| CapturePrefillProgressError::UnavailableHook)?;
        }
        Self::new_mode(
            bound.selection().source(),
            bound.geometry(),
            bound.selection().is_prepared_media(),
        )
    }
    pub(crate) fn new(
        source: &'a SharedCapturePlan,
        inference: InferenceGeometry,
    ) -> Result<Self, CapturePrefillProgressError> {
        Self::new_mode(source, inference, false)
    }
    fn new_mode(
        source: &'a SharedCapturePlan,
        inference: InferenceGeometry,
        transformed: bool,
    ) -> Result<Self, CapturePrefillProgressError> {
        let admission = source.admission();
        let request = admission.request();
        if admission.text_origin().map(|o| o.cached_positions) != Some(inference.cached_positions)
            || request.batch != inference.batch_size
            || request.prompt_tokens != inference.input_positions
            || inference.max_output_tokens > request.max_predictions
        {
            return Err(CapturePrefillProgressError::Identity);
        }
        if inference.batch_size == 0
            || inference.input_positions == 0
            || inference.max_output_tokens == 0
            || inference.prefill_chunk_positions == 0
            || inference.prefill_chunk_positions > inference.input_positions
        {
            return Err(CapturePrefillGeometryError::Schedule.into());
        }
        inference
            .cached_positions
            .checked_add(inference.input_positions)
            .and_then(|n| n.checked_add(inference.max_output_tokens))
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let result = Self {
            source,
            inference,
            transformed,
        };
        // Full selected-row validation precedes any progress/target construction.
        for i in 0..admission.plan().selections.len() {
            result.row(i)?;
        }
        Ok(result)
    }
    /// Borrow one actual selected row. Inactive rows require no tensor geometry.
    pub fn row(
        &self,
        index: usize,
    ) -> Result<CapturePrefillObservationRow<'a>, CapturePrefillProgressError> {
        let admission = self.source.admission();
        let selection = admission
            .plan()
            .selections
            .get(index)
            .ok_or(CapturePrefillProgressError::Identity)?;
        let active = selection.schedule.includes(CapturePhase::Prefill, 0);
        let candidate = if active
            && matches!(selection.transform, CaptureTransform::TopCandidates { .. })
        {
            Some(
                CaptureCandidateGeometry::prepare(admission, index, CapturePhase::Prefill, 0, None)
                    .map_err(CapturePrefillGeometryError::from)?,
            )
        } else {
            None
        };
        let token_scores =
            if active && matches!(selection.transform, CaptureTransform::TokenScores { .. }) {
                Some(
                    CaptureTokenScoreGeometry::prepare(
                        admission,
                        index,
                        CapturePhase::Prefill,
                        0,
                        None,
                    )
                    .map_err(CapturePrefillGeometryError::from)?,
                )
            } else {
                None
            };
        let transform = if active
            && (self.transformed
                || matches!(
                    selection.transform,
                    CaptureTransform::Summary | CaptureTransform::Histogram { .. }
                ))
            && matches!(
                selection.transform,
                CaptureTransform::Summary
                    | CaptureTransform::Histogram { .. }
                    | CaptureTransform::Preview { .. }
            ) {
            Some(CapturePrefillTransformPlan::prepare(
                admission,
                index,
                self.inference,
            )?)
        } else {
            None
        };
        let routed = if active && matches!(selection.transform, CaptureTransform::RoutedUnits) {
            Some(CaptureRoutedPrefillPlan::prepare(admission,index,self.inference)?)
        } else {None};
        let assembly =
            if active && candidate.is_none() && token_scores.is_none() && transform.is_none() && routed.is_none() {
                if !admission.points().get(index).is_some_and(|p| p.prefill) {
                    return Err(CapturePrefillProgressError::UnavailableHook);
                }
                Some(CapturePrefillRowAssembly::prepare(
                    admission,
                    index,
                    self.inference,
                )?)
            } else {
                None
            };
        Ok(CapturePrefillObservationRow {
            source: self.source,
            inference: self.inference,
            index,
            assembly,
            candidate,
            token_scores,
            transform,
            routed,
        })
    }
}
/// Full logical geometry and per-fragment hooks share one actual source/schedule.
/// A future prepared body-row contract must guarantee one hook in every chunk;
/// no absent-readout or output-count heuristic is encoded here.
pub struct CapturePrefillObservationRow<'a> {
    source: &'a SharedCapturePlan,
    inference: InferenceGeometry,
    index: usize,
    assembly: Option<CapturePrefillRowAssembly<'a>>,
    candidate: Option<CaptureCandidateGeometry<'a>>,
    token_scores: Option<CaptureTokenScoreGeometry<'a>>,
    transform: Option<CapturePrefillTransformPlan<'a>>,
    routed: Option<CaptureRoutedPrefillPlan<'a>>,
}
impl CapturePrefillObservationRow<'_> {
    /// Report a terminal quota skip from this exact immutable row and schedule.
    /// The result grants no source, payload or completion authority.
    pub fn is_skipped(&self, progress: &CapturePrefillRowProgress)
        -> Result<bool, CapturePrefillProgressError> {
        self.validate(progress, progress.next)?;
        Ok(progress.logical == Logical::Skipped)
    }
    /// Actual sparse source bound to the same canonical driver progression.
    pub fn routed_plan(&self)->Option<&CaptureRoutedPrefillPlan<'_>> {self.routed.as_ref()}
    /// Complete a sparse hook after its real provider ranges and rows were recorded.
    pub fn finish_routed_hook(&self,progress:&mut CapturePrefillRowProgress,
        fragment:&CaptureRoutedPrefillFragment<'_,'_>)->Result<(),CapturePrefillProgressError>{
        let plan=fragment.plan();
        if !std::ptr::eq(plan.geometry().admission(),self.source.admission())
            || plan.geometry().selection_index()!=self.index || plan.inference_geometry()!=self.inference {
            return Err(CapturePrefillProgressError::Identity);
        }
        self.validate(progress,fragment.chunk_index())?;
        if progress.logical!=Logical::Charged || progress.hook!=Hook::Attempting {
            return Err(CapturePrefillProgressError::Attempt);
        }
        progress.hook=Hook::Seen;
        Ok(())
    }
    /// Active ordinary transform plan. No original claim is provided.
    pub fn transform_plan(&self) -> Option<&CapturePrefillTransformPlan<'_>> {
        self.transform.as_ref()
    }
    /// Acknowledge one source-bound transform after validated host merge.
    pub fn finish_transform_hook(
        &self,
        progress: &mut CapturePrefillRowProgress,
        fragment: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<(), CapturePrefillProgressError> {
        let plan = fragment.plan();
        if !std::ptr::eq(plan.admission(), self.source.admission())
            || plan.selection_index() != self.index
            || plan.inference_geometry() != self.inference
        {
            return Err(CapturePrefillProgressError::Identity);
        }
        self.validate(progress, fragment.chunk_index())?;
        if progress.logical != Logical::Charged || progress.hook != Hook::Attempting {
            return Err(CapturePrefillProgressError::Attempt);
        }
        progress.hook = Hook::Seen;
        Ok(())
    }

    /// Only active rows have geometry. No execution/readout authority is implied.
    pub fn assembly(&self) -> Option<&CapturePrefillRowAssembly<'_>> {
        self.assembly.as_ref()
    }
    /// Terminal-only candidate geometry; this never converts earlier rows into
    /// independent prediction results or invokes the score head for StateOnly.
    pub fn candidate(&self) -> Option<&CaptureCandidateGeometry<'_>> {
        self.candidate.as_ref()
    }
    /// Resolve the canonical final physical readout, with no allocation/authority.
    pub fn candidate_for_chunk(
        &self,
        chunk: &PrefillChunk,
    ) -> Result<CaptureCandidateGeometry<'_>, CapturePrefillProgressError> {
        if self.candidate.is_none() || chunk.input.end != self.inference.input_positions {
            return Err(CapturePrefillProgressError::Order);
        }
        self.validate_terminal_chunk(
            chunk,
            chunk.input.start / self.inference.prefill_chunk_positions,
        )?;
        let rows = if chunk.output == eredu_core::OutputDemand::Sequence {
            chunk.input.end - chunk.input.start
        } else {
            1
        };
        CaptureCandidateGeometry::prepare(
            self.source.admission(),
            self.index,
            CapturePhase::Prefill,
            0,
            None,
        )
        .and_then(|g| g.terminal_readout(rows as usize))
        .map_err(|e| CapturePrefillGeometryError::from(e).into())
    }
    /// Terminal-only ordered token-score geometry; this never converts earlier rows into
    /// independent prediction results or invokes the score head for StateOnly.
    pub fn token_scores(&self) -> Option<&CaptureTokenScoreGeometry<'_>> {
        self.token_scores.as_ref()
    }
    /// Resolve the canonical final physical readout, with no allocation/authority.
    pub fn token_scores_for_chunk(
        &self,
        chunk: &PrefillChunk,
    ) -> Result<CaptureTokenScoreGeometry<'_>, CapturePrefillProgressError> {
        if self.token_scores.is_none() || chunk.input.end != self.inference.input_positions {
            return Err(CapturePrefillProgressError::Order);
        }
        self.validate_terminal_chunk(
            chunk,
            chunk.input.start / self.inference.prefill_chunk_positions,
        )?;
        let rows = if chunk.output == eredu_core::OutputDemand::Sequence {
            chunk.input.end - chunk.input.start
        } else {
            1
        };
        CaptureTokenScoreGeometry::prepare(
            self.source.admission(),
            self.index,
            CapturePhase::Prefill,
            0,
            None,
        )
        .and_then(|g| g.terminal_readout(rows as usize))
        .map_err(|e| CapturePrefillGeometryError::from(e).into())
    }
    fn validate_terminal_chunk(
        &self,
        chunk: &PrefillChunk,
        index: u64,
    ) -> Result<(), CapturePrefillProgressError> {
        let start = index
            .checked_mul(self.inference.prefill_chunk_positions)
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let end = start
            .checked_add(self.inference.prefill_chunk_positions)
            .ok_or(CapturePrefillGeometryError::Overflow)?
            .min(self.inference.input_positions);
        if chunk.input != (start..end)
            || self.inference.cached_positions.checked_add(start) != Some(chunk.position)
            || chunk.output
                != self
                    .inference
                    .output
                    .for_chunk(end == self.inference.input_positions)
        {
            return Err(CapturePrefillProgressError::Order);
        }
        Ok(())
    }
    /// Whether this selection reads only the actual canonical terminal logits.
    pub fn terminal(&self) -> bool {
        self.candidate.is_some() || self.token_scores.is_some()
    }
    /// Complete exactly the terminal candidate hook after its receipt is installed.
    pub fn finish_candidate_hook(
        &self,
        progress: &mut CapturePrefillRowProgress,
        chunk: &PrefillChunk,
    ) -> Result<(), CapturePrefillProgressError> {
        self.validate(progress, progress.next)?;
        self.candidate_for_chunk(chunk)?;
        if progress.logical != Logical::Charged || progress.hook != Hook::Attempting {
            return Err(CapturePrefillProgressError::Attempt);
        }
        progress.hook = Hook::Seen;
        Ok(())
    }
    /// Complete exactly the terminal token-score hook after its receipt is installed.
    pub fn finish_token_scores_hook(
        &self,
        progress: &mut CapturePrefillRowProgress,
        chunk: &PrefillChunk,
    ) -> Result<(), CapturePrefillProgressError> {
        self.validate(progress, progress.next)?;
        self.token_scores_for_chunk(chunk)?;
        if progress.logical != Logical::Charged || progress.hook != Hook::Attempting {
            return Err(CapturePrefillProgressError::Attempt);
        }
        progress.hook = Hook::Seen;
        Ok(())
    }
    /// Allocate no payload; retain one actual shared source handle.
    pub fn initial_progress(&self) -> CapturePrefillRowProgress {
        CapturePrefillRowProgress {
            source: self.source.clone(),
            inference: self.inference,
            index: self.index,
            next: 0,
            logical: if self.assembly.is_some() || self.terminal() || self.transform.is_some() || self.routed.is_some() {
                Logical::Uncharged
            } else {
                Logical::Skipped
            },
            hook: Hook::Unseen,
            terminal: self.terminal(),
        }
    }
    fn validate(
        &self,
        progress: &CapturePrefillRowProgress,
        chunk: u64,
    ) -> Result<(), CapturePrefillProgressError> {
        if !self.source.same_storage(&progress.source)
            || self.index != progress.index
            || self.inference != progress.inference
        {
            return Err(CapturePrefillProgressError::Identity);
        }
        if chunk != progress.next
            || chunk
                >= self
                    .inference
                    .input_positions
                    .div_ceil(self.inference.prefill_chunk_positions)
        {
            return Err(CapturePrefillProgressError::Order);
        }
        Ok(())
    }
    /// Enter Attempting before source validation/claim/factory. Duplicate zero
    /// hooks are duplicates; unrelated paths and disabled rows remain ignored.
    pub fn begin_hook(
        &self,
        progress: &mut CapturePrefillRowProgress,
        chunk: &PrefillChunk,
        path: &str,
    ) -> Result<CapturePrefillHookDecision, CapturePrefillProgressError> {
        if self.source.admission().plan().selections[self.index].path != path {
            return Ok(CapturePrefillHookDecision::Ignore);
        }
        self.validate(progress, progress.next)?;
        if progress.logical == Logical::Skipped {
            return Ok(CapturePrefillHookDecision::Ignore);
        }
        if self.terminal() {
            self.validate_terminal_chunk(chunk, progress.next)?;
            if chunk.input.end != self.inference.input_positions {
                return Ok(CapturePrefillHookDecision::Ignore);
            }
        } else if let Some(plan) = &self.routed {
            if !plan.fragment(progress.next)?.matches_chunk(&chunk.input,chunk.position,chunk.output) {
                return Err(CapturePrefillProgressError::Order);
            }
        } else if let Some(plan) = &self.transform {
            if !plan.fragment(progress.next)?.matches_chunk(
                &chunk.input,
                chunk.position,
                chunk.output,
            ) {
                return Err(CapturePrefillProgressError::Order);
            }
        } else {
            let fragment = self
                .assembly
                .as_ref()
                .ok_or(CapturePrefillProgressError::Identity)?
                .fragment(progress.next)?;
            if !fragment.matches_chunk(&chunk.input, chunk.position, chunk.output) {
                return Err(CapturePrefillProgressError::Order);
            }
        }
        if progress.logical == Logical::Failed || progress.hook == Hook::Attempting {
            return Err(CapturePrefillProgressError::Attempt);
        }
        if progress.hook == Hook::Seen {
            return Err(CaptureProtocolError::Duplicate.into());
        }
        let first = progress.logical == Logical::Uncharged;
        progress.hook = Hook::Attempting;
        Ok(if first {
            CapturePrefillHookDecision::First
        } else {
            CapturePrefillHookDecision::Continue
        })
    }
    /// Continue one routed provider invocation after its first batch charged
    /// the logical row. Only sparse hooks permit multiple physical batches.
    pub fn begin_routed_batch(
        &self, progress: &mut CapturePrefillRowProgress,
        chunk: &PrefillChunk, path: &str,
    ) -> Result<CapturePrefillHookDecision, CapturePrefillProgressError> {
        if self.source.admission().plan().selections[self.index].path != path {
            return Ok(CapturePrefillHookDecision::Ignore);
        }
        let plan = self.routed.as_ref().ok_or(CapturePrefillProgressError::Identity)?;
        self.validate(progress, progress.next)?;
        if !plan.fragment(progress.next)?.matches_chunk(&chunk.input, chunk.position, chunk.output) {
            return Err(CapturePrefillProgressError::Order);
        }
        if progress.logical == Logical::Charged && progress.hook == Hook::Attempting {
            return Ok(CapturePrefillHookDecision::Continue);
        }
        self.begin_hook(progress, chunk, path)
    }
    /// Charge the existing full-geometry estimator exactly once. Generated
    /// callers pass full_program_usage below, never a first-fragment estimate.
    pub fn reserve_first(
        &self,
        progress: &mut CapturePrefillRowProgress,
        ledger: &mut dyn eredu_core::capture::CaptureReservation,
        usage: CaptureUsage,
    ) -> Result<Option<CaptureSkipReason>, CapturePrefillProgressError> {
        self.validate(progress, progress.next)?;
        if progress.logical != Logical::Uncharged || progress.hook != Hook::Attempting {
            return Err(CapturePrefillProgressError::Attempt);
        }
        // No unwind/retry can turn a failed ledger attempt into an uncharged row.
        progress.logical = Logical::Failed;
        match ledger.reserve(usage)? {
            Some(reason) => {
                progress.logical = Logical::Skipped;
                progress.hook = Hook::Seen;
                Ok(Some(reason))
            }
            None => {
                progress.logical = Logical::Charged;
                Ok(None)
            }
        }
    }
    pub(crate) fn skip_partition_before_hook(&self, progress: &mut CapturePrefillRowProgress)
        -> Result<(), CapturePrefillProgressError> {
        self.validate(progress, 0)?;
        if progress.logical != Logical::Uncharged || progress.hook != Hook::Unseen {
            return Err(CapturePrefillProgressError::Attempt);
        }
        progress.logical = Logical::Skipped;
        Ok(())
    }
    /// Advance the same first-hook state using the existing global producer
    /// reservation. The receiver creates no second logical native charge.
    pub(crate) fn reserve_remote_first(
        &self, progress: &mut CapturePrefillRowProgress,
        charge: &crate::capture::partition::PreparedPartitionRemoteCharge<'_>,
    ) -> Result<(), CapturePrefillProgressError> {
        self.validate(progress, progress.next)?;
        if progress.logical != Logical::Uncharged || progress.hook != Hook::Attempting
            || !charge.validate(self.source, self.index)
        { return Err(CapturePrefillProgressError::Attempt); }
        progress.logical = Logical::Charged;
        Ok(())
    }
    /// Acknowledge the same already-paid global equation for local/remote
    /// fragment assembly. This source supplies no local native work credits.
    pub(crate) fn reserve_assembly_first(&self,progress:&mut CapturePrefillRowProgress,
        charge:&crate::capture::partition::PreparedPartitionAssemblyCharge<'_>)
        ->Result<(),CapturePrefillProgressError> {
        self.validate(progress,progress.next)?;
        if progress.logical!=Logical::Uncharged || progress.hook!=Hook::Attempting
            || !charge.validate(self.source,self.index) {return Err(CapturePrefillProgressError::Attempt);}
        progress.logical=Logical::Charged;Ok(())
    }
    /// Reuse the legacy generated creation charge from the checked FULL source.
    /// Actual chunk descriptors, source roots and physical traces remain separate.
    pub fn full_program_usage(
        &self,
        usage: CaptureUsage,
        program: eredu_nn::GeneratedTensorProgram<'_>,
    ) -> Result<CaptureUsage, CapturePrefillProgressError> {
        let eredu_nn::GeneratedTensorProgram::BlockFp8Input(plan) = program;
        let assembly = self
            .assembly
            .as_ref()
            .ok_or(CapturePrefillProgressError::Identity)?;
        if assembly.sequence_axis() + 1 == assembly.logical_geometry().source_shape().len() {
            return Err(CaptureProtocolError::PrefillAttribution.into());
        }
        let source = generated_capture_source(
            &plan
                .logical_capture_source()
                .map_err(|_| CaptureError::Overflow)?,
        );
        let step = CaptureObservationStep::new(self.source.admission(), CapturePhase::Prefill, 0)?;
        step.validate_generated(self.index, &source, program)?;
        Ok(step.generated_usage(usage, &source)?)
    }
    /// Derive the checked FULL reconstruction from this logical source. No
    /// chunk-sized creation estimate is substituted for the original quota.
    pub fn full_generated_usage(
        &self,
        usage: CaptureUsage,
    ) -> Result<CaptureUsage, CapturePrefillProgressError> {
        let assembly = self
            .assembly
            .as_ref()
            .ok_or(CapturePrefillProgressError::Identity)?;
        let shape = assembly.logical_geometry().source_shape();
        let mut signed = [0i32; 32];
        for (target, &axis) in signed.iter_mut().zip(shape) {
            *target = i32::try_from(axis).map_err(|_| CapturePrefillGeometryError::Overflow)?;
        }
        let plan = eredu_nn::BlockFp8InputReconstructionPlan::new(&signed[..shape.len()]).map_err(
            |error| match error {
                eredu_nn::ProjectionObservationError::Overflow => {
                    CapturePrefillProgressError::from(CaptureError::Overflow)
                }
                _ => CapturePrefillProgressError::from(CaptureProtocolError::GeneratedSource),
            },
        )?;
        self.full_program_usage(usage, eredu_nn::GeneratedTensorProgram::BlockFp8Input(plan))
    }
    /// Check the actual physical factory independently of the full logical quota.
    /// It must reconstruct this exact chunk geometry with its unchanged source
    /// descriptor. This method never invokes the factory or retains its inputs.
    pub fn validate_generated_fragment(
        &self,
        fragment: &CapturePrefillFragment<'_, '_>,
        source: &GeneratedCaptureSource,
        program: eredu_nn::GeneratedTensorProgram<'_>,
    ) -> Result<(), CapturePrefillProgressError> {
        self.validate_fragment(fragment)?;
        let eredu_nn::GeneratedTensorProgram::BlockFp8Input(plan) = program;
        if plan.shape().len() != fragment.source_shape().len()
            || plan
                .shape()
                .iter()
                .zip(fragment.source_shape())
                .any(|(&a, &b)| usize::try_from(a).ok() != Some(b))
            || source.source_dtype != Some(eredu_core::checkpoint::TensorDtype::F32)
            || plan.logical_capture_source().ok().map(|s| s.creation_bytes)
                != Some(source.creation_bytes)
        {
            return Err(CaptureProtocolError::GeneratedSource.into());
        }
        Ok(())
    }
    /// Preview(0) preserves the legacy nonempty-slice factory behavior. This is
    /// a logical decision, not authority to realize a generated native value.
    pub fn factory_required(
        &self,
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<bool, CapturePrefillProgressError> {
        self.validate_fragment(fragment)?;
        Ok(fragment.selected_elements() != 0)
    }
    fn validate_fragment(
        &self,
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<(), CapturePrefillProgressError> {
        let logical = fragment.assembly().logical_geometry();
        if !std::ptr::eq(logical.admission(), self.source.admission())
            || logical.selection_index() != self.index
            || fragment.assembly().inference_geometry() != self.inference
        {
            return Err(CapturePrefillProgressError::Identity);
        }
        Ok(())
    }
    /// Acknowledges semantic handling after actual host write/cold trace. Does
    /// not certify payload coverage, completion, source retirement or quota.
    pub fn finish_hook(
        &self,
        progress: &mut CapturePrefillRowProgress,
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<(), CapturePrefillProgressError> {
        self.validate_fragment(fragment)?;
        self.validate(progress, fragment.chunk_index())?;
        if progress.logical != Logical::Charged || progress.hook != Hook::Attempting {
            return Err(CapturePrefillProgressError::Attempt);
        }
        progress.hook = Hook::Seen;
        Ok(())
    }
    /// Checked single-row advancement for a metadata observer outside runtime.
    /// Multi-row callers first validate every row before invoking this method.
    /// This changes logical progress only, never host coverage or source custody.
    pub fn advance_chunk(
        &self,
        progress: &mut CapturePrefillRowProgress,
        chunk: u64,
    ) -> Result<(), CapturePrefillProgressError> {
        self.validate_chunk_end(progress, chunk)?;
        progress.advance_preflighted();
        Ok(())
    }
    /// All rows and independent host coverage must pass before any row advances.
    pub fn validate_chunk_end(
        &self,
        progress: &CapturePrefillRowProgress,
        chunk: u64,
    ) -> Result<(), CapturePrefillProgressError> {
        self.validate(progress, chunk)?;
        if self.terminal()
            && chunk + 1
                < self
                    .inference
                    .input_positions
                    .div_ceil(self.inference.prefill_chunk_positions)
        {
            return if progress.logical == Logical::Uncharged && progress.hook == Hook::Unseen {
                Ok(())
            } else {
                Err(CapturePrefillProgressError::Attempt)
            };
        }
        if progress.logical == Logical::Skipped {
            return Ok(());
        }
        if progress.logical == Logical::Failed || progress.hook == Hook::Attempting {
            return Err(CapturePrefillProgressError::Attempt);
        }
        if progress.logical != Logical::Charged || progress.hook != Hook::Seen {
            return Err(CapturePrefillProgressError::Missing);
        }
        Ok(())
    }
}
