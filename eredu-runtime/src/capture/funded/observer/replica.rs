//! Local lazy dependencies are distinct from producer-only capture transforms.
use super::*;
impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(super) fn complete_receiver_sources(&mut self, path: &str, value: &T)
        -> Result<(), FundedCaptureError<E>> {
        if self.backend.partition_capture().is_none() { return Ok(()); }
        if !self.active_transaction() { return Err(CaptureProtocolError::Transaction.into()); }
        if matches!(self.frame, Frame::Empty) { return Ok(()); }
        for index in 0..self.session.plan.plan().selections.len() {
            let selection = &self.session.plan.plan().selections[index];
            if selection.path != path || !selection.schedule.includes(self.session.phase, self.session.prediction) {
                continue;
            }
            let Frame::Active(frame) = &self.frame else { return Err(CaptureProtocolError::Transaction.into()); };
            if !policy::selected_record(selection, &frame.records()[index], path)? { continue; }
            if let Some(bound)=self.bound {
                let chunk=self.current_fragment_chunk()?;
                let k=chunk.input.start/bound.geometry().prefill_chunk_positions;
                let projected=self.backend.partition_capture().ok_or(CaptureProtocolError::Transaction)?
                    .take_projected_receiver_source(index,bound.geometry(),k)?;
                if let Some(source)=projected {
                    let actual=self.backend.validate_partition_prefill_source(value,&source)?;
                    if &actual!=source.dtype(){return Err(CaptureProtocolError::Geometry.into());}
                    let started=std::time::Instant::now();
                    let result=self.backend.complete_partition_source(value);
                    self.session.capture_seconds+=started.elapsed().as_secs_f64();
                    result?;continue;
                }
            }
            if self.bound.is_none(){
                let source=self.backend.partition_capture().ok_or(CaptureProtocolError::Transaction)?
                    .take_invocation_receiver_source(index)?;
                if let Some(source)=source {
                    let actual=self.backend.validate_partition_invocation_source(value,&source)?;
                    if &actual!=source.dtype(){return Err(CaptureProtocolError::Geometry.into());}
                    let started=std::time::Instant::now();
                    let result=self.backend.complete_partition_source(value);
                    self.session.capture_seconds+=started.elapsed().as_secs_f64();
                    result?;continue;
                }
            }
            let expected = self.backend.partition_capture().ok_or(CaptureProtocolError::Transaction)?
                .take_receiver_source(index)?;
            let Some(expected) = expected else { continue; };
            let actual = if let Some(bound) = self.bound {
                let chunk = self.current_fragment_chunk()?;
                let policy = crate::capture::CapturePrefillObservationPolicy::from_bound(bound)
                    .map_err(fragments::progress_error)?;
                let row = policy.row(index).map_err(fragments::progress_error)?;
                let k = chunk.input.start / bound.geometry().prefill_chunk_positions;
                if row.candidate().is_some() {
                    let geometry=row.candidate_for_chunk(&chunk).map_err(fragments::progress_error)?;
                    self.backend.validate_candidate_source(value,&geometry)?
                } else if row.token_scores().is_some() {
                    let geometry=row.token_scores_for_chunk(&chunk).map_err(fragments::progress_error)?;
                    self.backend.validate_token_score_source(value,&geometry)?
                } else if let Some(plan) = row.transform_plan()
                    .filter(|plan| matches!(plan.selection().transform, CaptureTransform::Summary)) {
                    let fragment = plan.fragment(k)?;
                    if fragment.selected_elements() == 0 { continue; }
                    let geometry = CaptureSummaryGeometry::prepare(plan.admission(), index,
                        CapturePhase::Prefill, 0, None)
                        .and_then(|geometry| geometry.fragment(&fragment))
                        .map_err(|cause| CaptureRunHostError::Step(cause.into()))?;
                    self.backend.validate_summary_source(value, &geometry)?
                } else if let Some(plan)=row.transform_plan()
                    .filter(|plan|matches!(plan.selection().transform,CaptureTransform::Histogram { .. })) {
                    let fragment=plan.fragment(k)?;
                    if fragment.selected_elements()==0 {continue;}
                    let geometry=CaptureHistogramGeometry::prepare(plan.admission(),index,CapturePhase::Prefill,0,None)
                        .and_then(|geometry|geometry.fragment(&fragment))
                        .map_err(|cause|CaptureRunHostError::Step(cause.into()))?;
                    self.backend.validate_histogram_source(value,&geometry)?
                } else {
                    let assembly = row.assembly().ok_or(CaptureProtocolError::Geometry)?;
                    let fragment = assembly.fragment(k).map_err(|cause|
                        fragments::progress_error(crate::capture::CapturePrefillProgressError::from(cause)))?;
                    if fragment.output_elements() == 0 { continue; }
                    self.backend.validate_prefill_source(value, &fragment)?
                }
            } else {
                let policy = CaptureObservationStep::with_invocation(&self.session.plan,
                    self.session.phase, self.session.prediction, self.invocation.map(|value| value.1))?
                    .with_window(self.window)?;
                if matches!(selection.transform,CaptureTransform::TopCandidates { .. }) {
                    let geometry=CaptureCandidateGeometry::prepare(&self.session.plan,index,self.session.phase,self.session.prediction,self.invocation.map(|value|value.1)).map_err(|cause|CaptureRunHostError::Step(cause.into()))?;
                    self.backend.validate_candidate_source(value,&geometry)?
                } else if matches!(selection.transform,CaptureTransform::TokenScores { .. }) {
                    let geometry=CaptureTokenScoreGeometry::prepare(&self.session.plan,index,self.session.phase,self.session.prediction,self.invocation.map(|value|value.1)).map_err(|cause|CaptureRunHostError::Step(cause.into()))?;
                    self.backend.validate_token_score_source(value,&geometry)?
                } else if matches!(selection.transform, CaptureTransform::Summary) {
                    let geometry = policy.summary_geometry(index)
                        .map_err(|cause| CaptureRunHostError::Step(cause.into()))?;
                    if policy.window().is_some() && geometry.elements() == 0 { continue; }
                    self.backend.validate_summary_source(value, &geometry)?
                } else if matches!(selection.transform,CaptureTransform::Histogram { .. }) {
                    let geometry=policy.histogram_geometry(index)
                        .map_err(|cause|CaptureRunHostError::Step(cause.into()))?;
                    if policy.window().is_some() && geometry.elements()==0 {continue;}
                    self.backend.validate_histogram_source(value,&geometry)?
                } else {
                    let geometry = policy.tensor_geometry(index)
                        .map_err(|cause| CaptureRunHostError::Step(cause.into()))?;
                    if policy.window().is_some() && geometry.elements() == 0 { continue; }
                    self.backend.validate_source(value, &geometry).map_err(FundedCaptureError::Backend)?
                }
            };
            if actual != expected { return Err(CaptureProtocolError::Geometry.into()); }
            let started = std::time::Instant::now();
            let result = self.backend.complete_partition_source(value);
            self.session.capture_seconds += started.elapsed().as_secs_f64();
            result?;
        }
        Ok(())
    }
}

// Callback state is borrowed. Concrete backend error/value controls stay with
// its existing native census; no generic payload size is replaced by a guess.
pub(in crate::capture::funded) fn control_bytes() -> usize {
    std::mem::size_of::<(
        &mut (), &str, &(), usize, TensorDtype, TensorDtype,
        Option<TensorDtype>, crate::prefill::PrefillChunk,
        Option<crate::capture::partition::PartitionInvocationReceiverSource>,
        Result<Option<crate::capture::partition::PartitionInvocationReceiverSource>,crate::capture::partition::PartitionCaptureProgramError>,
        crate::capture::partition::PartitionInvocationReceiverSource,
        Option<crate::capture::partition::PartitionPrefillReceiverSource>,
        Result<Option<crate::capture::partition::PartitionPrefillReceiverSource>,crate::capture::partition::PartitionCaptureProgramError>,
        crate::capture::partition::PartitionPrefillReceiverSource,u64,
        crate::capture::CapturePrefillObservationPolicy<'static>,
        crate::capture::CapturePrefillObservationRow<'static>,
        CaptureObservationStep<'static>, CaptureSummaryGeometry<'static>, CaptureHistogramGeometry<'static>,
        CaptureTensorGeometry<'static>,CaptureCandidateGeometry<'static>,CaptureTokenScoreGeometry<'static>,
        eredu_core::capture::CapturePrefillFragment<'static, 'static>,
        eredu_core::capture::CapturePrefillTransformFragment<'static, 'static>,
        std::time::Instant, Result<(), CaptureProtocolError>,
    )>()
}
