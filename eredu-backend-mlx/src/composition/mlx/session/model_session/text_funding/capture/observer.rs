//! Lexical scheduled capture from an already admitted native text operation.
use super::*;
mod fragments;
mod intervention;
mod generated;
mod routed_partition;
use crate::composition::mlx::session::{
    bounded_capture::estimate_tensor_geometry, model_session::SessionOperation,
};
use eredu_core::{
    capture::{CaptureError, CaptureUsage},
    checkpoint::TensorDtype,
};
use eredu_runtime::capture::{FundedCaptureError, FundedCaptureSession, ScheduledCaptureBackend};
use eredu_runtime::working_memory::WorkingMemoryError;

// Concrete borrowed controls only: this adapter owns no source/shape/data buffer,
// scope, submission lease, reservation, or independent native allocation grant.
pub(super) struct NativeScheduledCapture<'a> {
    work: &'a FundedWork,
    stream: &'a Stream,
    owner: Option<super::super::super::SubmissionResourcesOwner>,
    prefill: bool,
    partition: Option<&'a mut (dyn eredu_runtime::capture::partition::ScheduledPartitionCapture + 'a)>,
}
#[cfg(test)]
impl<'a> NativeScheduledCapture<'a> {
    pub(super) fn for_test(work: &'a FundedWork, stream: &'a Stream) -> Self {
        Self {
            work,
            stream,
            owner: None,
            prefill: false,
            partition: None,
        }
    }
}
impl NativeScheduledCapture<'_> {
    fn capture_observer(&self) -> Result<Option<safemlx::OriginalScopeObserver>, Error> {
        if self.work._controls.is_none() {
            return Ok(None);
        }
        // Existing standalone carrier fixtures exercise the ordinary numerical
        // leaf with paid metadata. They install no original native program.
        // Production always supplies an owner and cannot take this branch.
        #[cfg(test)]
        if self.owner.is_none() && self.work.native_storage.is_none() {
            return Ok(None);
        }
        let owner = self.owner.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
        if self.prefill {
            owner
                .prefill_scopes
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .as_ref()
                .ok_or(Error::PrefillScopeUnavailable)?
                .capture_observer()
                .map(Some)
        } else {
            owner
                .model_execution
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .as_ref()
                .ok_or(Error::PrefillScopeUnavailable)?
                .capture_observer()
                .map(Some)
        }
    }
}
impl ScheduledCaptureBackend for NativeScheduledCapture<'_> {
    type Tensor = Array;
    type Error = Error;

    fn validate_partition_intervention_source(&self,source:&Array,
        claim:&eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
        window:Option<eredu_runtime::intervention::InterventionPrefillWindow>)
        ->Result<(),FundedCaptureError<Error>> {self.partition_edit_validate(source,claim,window)}
    fn validate_partition_intervention_evidence_source(&self,source:&Array,
        claim:&eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
        side:eredu_core::capture::InterventionEvidenceSide,
        window:Option<eredu_runtime::intervention::InterventionPrefillWindow>)
        ->Result<(),FundedCaptureError<Error>> {self.partition_evidence_validate(source,claim,side,window)}
    fn apply_partition_intervention(&mut self,source:&Array,
        claim:&eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
        allowance:&mut eredu_runtime::capture::partition::PartitionInterventionLocalAllowance)
        ->Result<Option<Array>,FundedCaptureError<Error>> {self.partition_edit_apply(source,claim,allowance)}
    fn prefill_intervention_projection_usage(&self,source:&Array,
        claim:&eredu_runtime::working_memory::CaptureInterventionClaim<'_>,window:eredu_runtime::intervention::InterventionPrefillWindow)
        ->Result<[CaptureUsage;2],FundedCaptureError<Error>> {
        self.prefill_projection_usage(source,claim,window)
    }
    fn prefill_intervention_usage(&self,source:&Array,
        claim:&eredu_runtime::working_memory::CaptureInterventionClaim<'_>,window:eredu_runtime::intervention::InterventionPrefillWindow)
        ->Result<CaptureUsage,FundedCaptureError<Error>> {
        self.prefill_edit_usage(source,claim,window)
    }
    fn apply_prefill_intervention(&mut self,source:&Array,
        fragment:eredu_runtime::working_memory::InterventionPrefillFragment<'_,'_>,charged:CaptureUsage,projection:[CaptureUsage;2])
        ->Result<Option<Array>,FundedCaptureError<Error>> {
        self.prefill_edit(source,fragment,charged,projection)
    }
    fn intervention_usage(&self,source:&Array,claim:&eredu_runtime::working_memory::CaptureInterventionClaim<'_>)
        ->Result<CaptureUsage,FundedCaptureError<Error>>{self.scheduled_intervention_usage(source,claim)}
    fn intervention_projection_usage(&self,source:&Array,claim:&eredu_runtime::working_memory::CaptureInterventionClaim<'_>)
        ->Result<[CaptureUsage;2],FundedCaptureError<Error>>{self.scheduled_intervention_projection_usage(source,claim)}
    fn apply_intervention_projected(&mut self,source:&Array,claim:eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
        charged:CaptureUsage,projection:[CaptureUsage;2])
        ->Result<(Option<Array>,eredu_runtime::working_memory::ClaimedIntervention),FundedCaptureError<Error>>{
        self.scheduled_intervention(source,claim,charged,projection)
    }
    fn intervention_evidence_usage(&self,source:&Array,claim:&eredu_runtime::working_memory::CaptureInterventionEvidenceClaim<'_,'_>)
        ->Result<(TensorDtype,CaptureUsage),FundedCaptureError<Error>>{self.scheduled_evidence_usage(source,claim)}
    fn capture_intervention_evidence<'a>(&mut self,source:&Array,claim:eredu_runtime::working_memory::CaptureInterventionEvidenceClaim<'a,'_>)
        ->Result<eredu_runtime::working_memory::ClaimedInterventionEvidence<'a>,FundedCaptureError<Error>>{
        self.scheduled_evidence(source,claim)
    }

    fn routed_error(&self, cause: FundedCaptureError<Error>) -> eredu_nn::Error {
        eredu_nn::Error::backend_retained_source(self.work.capture_error(observer_error(cause)))
    }
    fn validate_routed_prefill_source(
        &self, source: &eredu_core::capture::RoutedUnitCaptureSource<'_, Array>,
        fragment: &eredu_core::capture::CaptureRoutedPrefillFragment<'_, '_>,
    ) -> Result<TensorDtype, FundedCaptureError<Error>> {
        crate::backend::array_copy::CompletedRoutedCaptureSource::validate_borrowed(
            source, fragment.plan().geometry().bank(), fragment.source_tokens())
            .map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(Error::Other(Box::new(cause)))))?;
        Ok(match source.values.dtype() {
            safemlx::Dtype::Float32 => TensorDtype::F32,
            safemlx::Dtype::Float16 => TensorDtype::F16,
            safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
            _ => unreachable!("validated floating source"),
        })
    }
    fn estimate_routed_prefill(
        &self, geometry: &eredu_core::capture::CaptureRoutedUnitsGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_routed_geometry(geometry)
    }
    fn transform_routed_prefill(
        &mut self, source: &eredu_core::capture::RoutedUnitCaptureSource<'_, Array>,
        writer: eredu_runtime::working_memory::CaptureRoutedPrefillWriter<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer.as_ref().map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);
            self.work.capture_prefill_routed(source, writer, self.stream, completion)
        })();
        result.map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(cause)))
    }

    fn validate_routed_invocation_source(
        &self, source: &eredu_core::capture::RoutedUnitCaptureSource<'_, Array>,
        geometry: &eredu_core::capture::CaptureRoutedUnitsGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Error>> {
        crate::backend::array_copy::CompletedRoutedCaptureSource::validate_borrowed(
            source, geometry.bank(), geometry.source_shape()[0] as u64)
            .map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(Error::Other(Box::new(cause)))))?;
        Ok(match source.values.dtype() {
            safemlx::Dtype::Float32 => TensorDtype::F32,
            safemlx::Dtype::Float16 => TensorDtype::F16,
            safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
            _ => unreachable!("validated floating source"),
        })
    }
    fn transform_routed_batch(
        &mut self, source: &eredu_core::capture::RoutedUnitCaptureSource<'_, Array>,
        writer: eredu_runtime::working_memory::CaptureRoutedBatchWriter<'_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer.as_ref().map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);
            self.work.capture_routed_batch(source, writer, self.stream, completion)
        })();
        result.map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(cause)))
    }

    fn validate_partition_routed_invocation(
        &self, input: &eredu_runtime::RoutedUnitInvocation<'_, Array>,
        layout: &eredu_core::capture::PartitionRoutedUnitCaptureLayout<'_>,
    ) -> Result<(u64, TensorDtype), FundedCaptureError<Error>> {
        routed_partition::validate_invocation(self, input, layout)
    }
    fn estimate_partition_routed(
        &self, request: &eredu_core::capture::PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_partition_routed_geometry(request)
    }
    fn validate_partition_routed_source(
        &self, source: &eredu_core::capture::PartitionRoutedUnitCaptureSource<'_, Array>,
        request: &eredu_core::capture::PartitionRoutedUnitCaptureRequest<'_>,
        invocation_source_tokens: u64, actual_native_rows: u64,
    ) -> Result<TensorDtype, FundedCaptureError<Error>> {
        routed_partition::validate_source(self, source, request, invocation_source_tokens, actual_native_rows)
    }
    fn validate_partition_routed_batch_source(
        &self, source: &eredu_core::capture::PartitionRoutedUnitCaptureSource<'_, Array>,
        layout: &eredu_core::capture::PartitionRoutedUnitCaptureLayout<'_>,
        actual_native_rows: u64,
    ) -> Result<(TensorDtype, u64), FundedCaptureError<Error>> {
        routed_partition::validate_batch(self, source, layout, actual_native_rows)
    }
    fn transform_partition_routed_batch(
        &mut self, source: &eredu_core::capture::PartitionRoutedUnitCaptureSource<'_, Array>,
        writer: eredu_runtime::working_memory::CapturePartitionRoutedWriter<'_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer.as_ref().map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);
            self.work.capture_partition_routed(source, writer, self.stream, completion, self.prefill)
        })();
        result.map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(cause)))
    }

    fn partition_capture(&mut self)
        -> Option<&mut (dyn eredu_runtime::capture::partition::ScheduledPartitionCapture + '_)> {
        match &mut self.partition { Some(program) => Some(&mut **program), None => None }
    }

    fn validate_partition_prefill_source(&self,source:&Array,
        geometry:&eredu_runtime::capture::partition::PartitionPrefillReceiverSource)
        ->Result<TensorDtype,FundedCaptureError<Error>> {
        validate_projected_source(source,geometry.source_shape())
    }
    fn validate_partition_invocation_source(&self,source:&Array,
        geometry:&eredu_runtime::capture::partition::PartitionInvocationReceiverSource)
        ->Result<TensorDtype,FundedCaptureError<Error>> {
        validate_projected_source(source,geometry.source_shape())
    }
    fn complete_partition_source(&mut self, source: &Array) -> Result<(), FundedCaptureError<Error>> {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer.as_ref()
                .map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);
            self.work.validate_capture_completion(completion)?;
            PreparedCaptureTensor::validate_stream(self.stream)
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            completion.reserve_roots(&self.work.roots, 1)
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            self.work.retain(source);
            if let Some(cause) = self.work.take_collection_failure() { return Err(cause); }
            drop(completion.settle(source, self.stream)
                .map_err(|cause| Error::Other(Box::new(cause)))?);
            Ok(())
        })();
        result.map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(cause)))
    }

    fn prepare_prefill_chunk_retention(
        &mut self,
        bootstrap: eredu_runtime::working_memory::CapturePrefillSourceBootstrap<'_>,
        context: &eredu_runtime::inspection::PrefillChunkRetentionContext<'_>,
    ) -> Result<
        Option<eredu_runtime::inspection::PreparedPrefillChunkRetention>,
        FundedCaptureError<Error>,
    > {
        self.work
            .prepare_capture_chunk(bootstrap, context)
            .map(Some)
            .map_err(FundedCaptureError::Backend)
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        ticket: eredu_runtime::inspection::SettledPrefillChunkRetention,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.work
            .retire_capture_chunk(ticket)
            .map_err(FundedCaptureError::Backend)
    }

    fn validate_source(
        &self,
        source: &Array,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Error> {
        PreparedCaptureTensor::validate_borrowed_source(source, geometry)
            .map(|dtype| match dtype {
                safemlx::Dtype::Float32 => TensorDtype::F32,
                safemlx::Dtype::Float16 => TensorDtype::F16,
                safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
                _ => unreachable!("closed source validator accepts only these precisions"),
            })
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn estimate(
        &self,
        _source: &Array,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate_tensor_geometry(geometry)
    }

    fn estimate_generated(
        &self,
        _prototype: &Array,
        _source: &eredu_core::capture::GeneratedCaptureSource,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate_tensor_geometry(geometry)
    }
    fn preflight_generated(
        &mut self,
        prototype: &Array,
        source: &eredu_core::capture::GeneratedCaptureSource,
        program: eredu_nn::GeneratedTensorProgram<'_>,
        construct: bool,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.prepare_generated(prototype, source, program, construct, claim)
    }
    fn retain_generated_source(
        &mut self,
        prototype: &Array,
        role: eredu_nn::GeneratedTensorSourceRole,
        value: &Array,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.retain_generated_operand(prototype, role, value, claim)
    }
    fn retain_generated_output(
        &mut self,
        value: &Array,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.retain_generated_root(value, claim)
    }

    fn validate_prefill_source(
        &self,
        source: &Array,
        fragment: &eredu_core::capture::CapturePrefillFragment<'_, '_>,
    ) -> Result<TensorDtype, FundedCaptureError<Error>> {
        crate::backend::array_copy::PreparedCaptureFragment::validate_borrowed_source(
            source, fragment,
        )
        .map(|dtype| match dtype {
            safemlx::Dtype::Float32 => TensorDtype::F32,
            safemlx::Dtype::Float16 => TensorDtype::F16,
            safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
            _ => unreachable!("closed floating dtype check"),
        })
        .map_err(|e| FundedCaptureError::Backend(Error::Other(Box::new(e))))
    }
    fn estimate_prefill(
        &self,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate_tensor_geometry(geometry)
    }
    fn transform_prefill_fragment(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer
                .as_ref()
                .map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);
            self.work.capture_prefill_fragment_with_completion(
                source,
                claim,
                self.stream,
                completion,
            )
        })();
        result.map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(cause)))
    }
    fn preflight_generated_fragment(
        &mut self,
        prototype: &Array,
        source: &eredu_core::capture::GeneratedCaptureSource,
        program: eredu_nn::GeneratedTensorProgram<'_>,
        construct: bool,
        claim: &eredu_runtime::working_memory::CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.prepare_generated_fragment(prototype, source, program, construct, claim)
    }
    fn retain_generated_fragment_source(
        &mut self,
        prototype: &Array,
        role: eredu_nn::GeneratedTensorSourceRole,
        value: &Array,
        claim: &eredu_runtime::working_memory::CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.retain_generated_fragment_operand(prototype, role, value, claim)
    }
    fn retain_generated_fragment_output(
        &mut self,
        value: &Array,
        claim: &eredu_runtime::working_memory::CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.work
            .retain_generated_capture_root(value, claim)
            .map_err(FundedCaptureError::Backend)
    }

    fn validate_candidate_source(
        &self,
        source: &Array,
        geometry: &eredu_core::capture::CaptureCandidateGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Error>> {
        let check = || {
            if !cfg!(all(
                feature = "metal",
                target_vendor = "apple",
                not(feature = "cuda")
            )) {
                return Err(
                    crate::backend::array_copy::CaptureTensorNativeError::UnsupportedMechanism,
                );
            }
            let program = crate::backend::array_copy::CandidateExtraction::from_geometry(geometry)?;
            program.validate_source(source)?;
            PreparedCaptureTensor::validate_stream(self.stream)?;
            Ok(match source.dtype() {
                safemlx::Dtype::Float32 => TensorDtype::F32,
                safemlx::Dtype::Float16 => TensorDtype::F16,
                safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
                _ => unreachable!("validated floating source"),
            })
        };
        check().map_err(|e| {
            FundedCaptureError::Backend(self.work.capture_error(Error::Other(Box::new(e))))
        })
    }
    fn validate_summary_source(
        &self,
        source: &Array,
        geometry: &eredu_core::capture::CaptureSummaryGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Error>> {
        let result = (|| {
            let program =
                crate::backend::array_copy::PreparedCaptureSummary::from_geometry(geometry)?;
            program.validate_source(source)
        })();
        result
            .map(|dtype| match dtype {
                safemlx::Dtype::Float32 => TensorDtype::F32,
                safemlx::Dtype::Float16 => TensorDtype::F16,
                safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
                _ => unreachable!("closed floating source"),
            })
            .map_err(|error| {
                FundedCaptureError::Backend(self.work.capture_error(Error::Other(Box::new(error))))
            })
    }
    fn estimate_summary(
        &self,
        geometry: &eredu_core::capture::CaptureSummaryGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_summary(geometry)
    }
    fn estimate_prefill_summary(
        &self,
        plan: &eredu_core::capture::CapturePrefillTransformPlan<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_prefill_summary(plan)
    }
    fn transform_summary(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureSummaryClaim<'_, '_>,
    ) -> Result<eredu_runtime::working_memory::ClaimedCaptureSummary, FundedCaptureError<Error>>
    {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer
                .as_ref()
                .map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);
            self.work
                .capture_summary_with_completion(source, claim, self.stream, completion)
        })();
        result.map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(cause)))
    }
    fn validate_histogram_source(
        &self,
        source: &Array,
        geometry: &eredu_core::capture::CaptureHistogramGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Error>> {
        let result = (|| {
            let program =
                crate::backend::array_copy::PreparedCaptureHistogram::from_geometry(geometry)?;
            program.validate_source(source)
        })();
        result
            .map(|dtype| match dtype {
                safemlx::Dtype::Float32 => TensorDtype::F32,
                safemlx::Dtype::Float16 => TensorDtype::F16,
                safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
                _ => unreachable!("closed floating source"),
            })
            .map_err(|error| {
                FundedCaptureError::Backend(self.work.capture_error(Error::Other(Box::new(error))))
            })
    }
    fn estimate_histogram(
        &self,
        geometry: &eredu_core::capture::CaptureHistogramGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_histogram(geometry)
    }
    fn estimate_prefill_histogram(
        &self,
        plan: &eredu_core::capture::CapturePrefillTransformPlan<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_prefill_histogram(plan)
    }
    fn transform_histogram(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureHistogramClaim<'_, '_>,
    ) -> Result<eredu_runtime::working_memory::ClaimedCaptureHistogram, FundedCaptureError<Error>>
    {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer
                .as_ref()
                .map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);
            self.work
                .capture_histogram_with_completion(source, claim, self.stream, completion)
        })();
        result.map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(cause)))
    }
    fn estimate_candidates(
        &self,
        geometry: &eredu_core::capture::CaptureCandidateGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_candidates(geometry)
    }
    fn transform_candidates(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureCandidateClaim<'_, '_>,
    ) -> Result<eredu_runtime::working_memory::ClaimedCaptureCandidates, FundedCaptureError<Error>>
    {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer
                .as_ref()
                .map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);
            self.work
                .capture_candidates_with_completion(source, claim, self.stream, completion)
        })();
        result.map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(cause)))
    }
    fn validate_token_score_source(
        &self,
        source: &Array,
        geometry: &eredu_core::capture::CaptureTokenScoreGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Error>> {
        let check = || {
            if !cfg!(all(
                feature = "metal",
                target_vendor = "apple",
                not(feature = "cuda")
            )) {
                return Err(
                    crate::backend::array_copy::CaptureTensorNativeError::UnsupportedMechanism,
                );
            }
            let program = crate::backend::array_copy::TokenScoreProgram::from_geometry(geometry)?;
            program.validate_source(source)?;
            PreparedCaptureTensor::validate_stream(self.stream)?;
            Ok(match source.dtype() {
                safemlx::Dtype::Float32 => TensorDtype::F32,
                safemlx::Dtype::Float16 => TensorDtype::F16,
                safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
                _ => unreachable!("validated floating source"),
            })
        };
        check().map_err(|e| {
            FundedCaptureError::Backend(self.work.capture_error(Error::Other(Box::new(e))))
        })
    }
    fn estimate_token_scores(
        &self,
        geometry: &eredu_core::capture::CaptureTokenScoreGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_token_scores(geometry)
    }
    fn transform_token_scores(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureTokenScoreClaim<'_, '_>,
    ) -> Result<eredu_runtime::working_memory::ClaimedCaptureTokenScores, FundedCaptureError<Error>>
    {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer
                .as_ref()
                .map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);

            self.work
                .capture_token_scores_with_completion(source, claim, self.stream, completion)
        })();
        result.map_err(|cause| FundedCaptureError::Backend(self.work.capture_error(cause)))
    }
    fn transform(
        &mut self,
        source: &Array,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Error> {
        let result = (|| {
            let observer = self.capture_observer()?;
            let completion = observer
                .as_ref()
                .map_or(CaptureCompletion::Ordinary, CaptureCompletion::Original);
            self.work
                .capture_tensor_with_completion(source, claim, self.stream, completion)
        })();
        result.map_err(|cause| self.work.capture_error(cause))
    }
}

/// One actual funded observer error Box and its move/result controls. Payload
/// arrays/host banks remain in their existing original owners.
pub(super) fn error_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        Stream::device_type_control_bytes()?,
        routed_partition::control_bytes()?,
        eredu_nn::Error::retained_source_control_bytes::<Error>()?,
        crate::backend::array_copy::capture_original_error_control_bytes()?.checked_mul(2)?,
        size_of::<NativeScheduledCapture<'static>>(),
        size_of::<Option<safemlx::OriginalScopeObserver>>(),
        size_of::<Result<Option<safemlx::OriginalScopeObserver>, Error>>(),
        size_of::<FundedCaptureError<Error>>(),
        size_of::<Box<FundedCaptureError<Error>>>(),
        size_of::<FundedCaptureError<Error>>(),
        size_of::<Error>(),
        size_of::<Result<(), FundedCaptureError<Error>>>(),
    ]
    .into_iter()
    .try_fold(intervention::control_bytes()?, usize::checked_add)
}

fn observer_error(error: FundedCaptureError<Error>) -> Error {
    Error::Other(Box::new(error))
}

impl<P: Probe> SessionOperation<'_, P> {
    /// Borrow one scheduled observer after the original text entry installed its
    /// work owner. This opens no submission/scope and cannot accept unrelated
    /// work. The existing model transaction remains the sole execution driver.
    ///
    /// The caller must have included the actual observer program/source and host
    /// bank in the original quote. This prerequisite does not open managed gates.
    /// Returned delivery is drained only after exact native completion, never
    /// merely because this lexical helper returned or its host frame was sealed.
    pub(in crate::composition::mlx::session::model_session) fn with_funded_capture<R>(
        &mut self,
        backend: &MlxBackend<'_>,
        capture: &mut FundedCaptureSession,
        prediction: u64,
        operation: impl FnOnce(
            &mut Executable,
            &mut dyn RuntimeActivationObserver<Array, Error>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.with_funded_capture_inner(backend, capture, prediction, None, None, false, operation)
    }

    /// Use the same work owner with the original installed prefill companion.
    pub(in crate::composition::mlx::session::model_session) fn with_funded_prefill_capture<R>(
        &mut self,
        backend: &MlxBackend<'_>,
        capture: &mut FundedCaptureSession,
        bound: &eredu_runtime::working_memory::AdmittedPrefillCapture<'_>,
        operation: impl FnOnce(
            &mut Executable,
            &mut dyn RuntimeActivationObserver<Array, Error>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.with_funded_capture_inner(backend, capture, 0, Some(bound), None, true, operation)
    }

    pub(in crate::composition::mlx::session::model_session) fn with_funded_continuation_capture<
        R,
    >(
        &mut self,
        backend: &MlxBackend<'_>,
        capture: &mut FundedCaptureSession,
        admitted: &eredu_runtime::working_memory::AdmittedCaptureContinuation<'_>,
        operation: impl FnOnce(
            &mut Executable,
            &mut dyn RuntimeActivationObserver<Array, Error>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.with_funded_capture_inner(
            backend,
            capture,
            admitted.first_prediction(),
            None,
            Some(admitted),
            true,
            operation,
        )
    }

    fn with_funded_capture_inner<R>(
        &mut self,
        backend: &MlxBackend<'_>,
        capture: &mut FundedCaptureSession,
        prediction: u64,
        bound: Option<&eredu_runtime::working_memory::AdmittedPrefillCapture<'_>>,
        continuation: Option<&eredu_runtime::working_memory::AdmittedCaptureContinuation<'_>>,
        physical_prefill: bool,
        operation: impl FnOnce(
            &mut Executable,
            &mut dyn RuntimeActivationObserver<Array, Error>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        if let Some(bound) = bound {
            bound
                .selection()
                .selection()
                .validate_sources(capture.source(), bound.selection().selection().paths())
                .map_err(|error| Error::Other(Box::new(error)))?;
        }

        self.session.validate_backend(backend)?;
        self.owner.ensure_healthy()?;
        // Release both RefCell guards before lending the actual model. The Rc
        // shares only the existing work owner, never the exclusive session payload.

        let work = self
            .owner
            .funding
            .try_borrow()
            .map_err(|error| Error::Other(Box::new(error)))?
            .as_ref()
            .cloned()
            .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)))?;
        {
            let scope = work
                .scope
                .try_borrow()
                .map_err(|error| Error::Other(Box::new(error)))?;
            let scope = scope
                .as_ref()
                .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::ExecutionFenced)))?;
            scope
                .validate_domain(backend.memory_pool())
                .and_then(|()| scope.validate_domain(&self.session.payload.memory_pool))
                .map_err(|error| Error::Other(Box::new(error)))?;
            capture
                .validate_native_scope(scope)
                .map_err(|error| Error::Other(Box::new(error)))?;
        }

        PreparedCaptureTensor::validate_stream(backend.stream())
            .map_err(|error| Error::Other(Box::new(error)))?;

        // Hold an independent owner loan while self.model() mutably borrows the
        // exclusive payload. The frame source retains only request/native control
        // authority; receipt/program destinations drop before this guard.
        let partition_owner = self.owner.clone();
        let mut partition_frame = partition_owner.partition_capture.try_borrow_mut()
            .map_err(|_| work.capture_error(Error::PrefillScopeReentrant))?;
        let mut partition_program = partition_frame.as_mut()
            .map(|frame| frame.program(capture, prediction)).transpose()
            .map_err(|cause| work.capture_error(cause))?.flatten();

        let mut native = NativeScheduledCapture {
            work: &work,
            stream: backend.stream(),
            owner: Some(self.owner.clone()),
            prefill: physical_prefill,
            partition: partition_program.as_mut().map(|program| program as
                &mut dyn eredu_runtime::capture::partition::ScheduledPartitionCapture),
        };
        let retain_observer_error = |error| work.capture_error(observer_error(error));
        let execute = |observer: &mut dyn RuntimeActivationObserver<Array, Error>| {
            let result = operation(self.model(), observer)?;

            observer.finish()?;
            Ok(result)
        };

        match (bound, continuation) {
            (Some(bound), None) => capture.with_admitted_prefill_observer(
                &mut native,
                bound,
                &retain_observer_error,
                execute,
            ),
            (None, Some(continuation)) => capture.with_admitted_continuation_observer(
                &mut native,
                continuation,
                &retain_observer_error,
                execute,
            ),
            (None, None) => {
                capture.with_observer(&mut native, prediction, &retain_observer_error, execute)
            }
            (Some(_), Some(_)) => return Err(Error::PrefillScopeUnavailable),
        }
        .map_err(|error| work.capture_error(Error::Other(Box::new(error))))?
    }
}

/// Actual replica readiness worker: one retained input and nested completion,
/// with no transform, source publication, host readout or receipt allocation.
pub(super) fn replica_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let values = [
        error_control_bytes()?,
        crate::backend::nn::tensor::TokenValidationScope::capture_observer_control_bytes()?,
        size_of::<(&NativeScheduledCapture<'static>,&Array,&eredu_runtime::capture::partition::PartitionPrefillReceiverSource)>(),
        size_of::<(&NativeScheduledCapture<'static>,&Array,&eredu_runtime::capture::partition::PartitionInvocationReceiverSource)>(),
        size_of::<(&Array,&[usize])>(),size_of::<Result<TensorDtype,FundedCaptureError<Error>>>(),
        safemlx::OperationEvent::nested_completion_control_bytes::<1>()?,
        safemlx::OriginalScopeObserver::control_bytes()?,
        safemlx::PreparedArrayClone::control_bytes()?,
        Array::inspection_clone_handle_bytes(),
        size_of::<(&mut NativeScheduledCapture<'static>, &Array)>(),
        size_of::<CaptureCompletion<'static>>(),
        size_of::<Option<safemlx::OriginalScopeObserver>>(),
        size_of::<safemlx::EvaluatedArray<'static>>(),
        size_of::<Result<safemlx::EvaluatedArray<'static>, crate::backend::array_copy::CaptureTensorNativeError>>(),
        size_of::<crate::backend::array_copy::CaptureTensorNativeError>(),
        size_of::<Box<crate::backend::array_copy::CaptureTensorNativeError>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<(), FundedCaptureError<Error>>>(),
        size_of::<std::cell::BorrowMutError>(),
        size_of::<std::collections::TryReserveError>(),
    ];
    values.into_iter().try_fold(size_of_val(&values), usize::checked_add)
}

fn validate_projected_source(source:&Array,shape:&[usize])->Result<TensorDtype,FundedCaptureError<Error>> {
        PreparedCaptureTensor::validate_borrowed_shape(source,shape)
            .map(|dtype|match dtype {
                safemlx::Dtype::Float32=>TensorDtype::F32,safemlx::Dtype::Float16=>TensorDtype::F16,
                safemlx::Dtype::Bfloat16=>TensorDtype::Bf16,_=>unreachable!("validated floating source"),
            }).map_err(|cause|FundedCaptureError::Backend(Error::Other(Box::new(cause))))
}
