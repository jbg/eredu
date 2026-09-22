//! Thin model-role adapter over the same selected native capture workers.
use crate::backend::array_copy::{
    CandidateExtraction, PreparedCaptureHistogram, PreparedCaptureSummary, PreparedCaptureTensor,
    TokenScoreProgram,
};
use crate::composition::mlx::session::intervention::PreparedModelInterventions;
use crate::{MlxTensor, backend::array_copy::CaptureTensorNativeError};
use eredu_core::{capture::*, checkpoint::TensorDtype};
use eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody;
use eredu_runtime::{
    capture::ScheduledCaptureBackend,
    working_memory::{CaptureTensorClaim, ClaimedCaptureTensor},
};
use safemlx::Array;
use std::cell::RefCell;

pub(in crate::composition::mlx) struct ScheduledNativeCapture<'a> {
    pub stream: &'a safemlx::Stream,
    pub roots: &'a RefCell<Vec<Array>>,
    pub edits: Option<&'a PreparedModelInterventions>,
    pub custody: &'a OriginalSpeculativeBudgetCustody,
    pub partition:
        Option<&'a mut (dyn eredu_runtime::capture::partition::ScheduledPartitionCapture + 'a)>,
    pub observer: &'a safemlx::OriginalScopeObserver,
}
impl ScheduledCaptureBackend for ScheduledNativeCapture<'_> {
    type Tensor = MlxTensor;
    type Error = CaptureTensorNativeError;
    fn partition_capture(
        &mut self,
    ) -> Option<&mut (dyn eredu_runtime::capture::partition::ScheduledPartitionCapture + '_)> {
        self.partition.as_mut().map(|program| {
            &mut **program as &mut dyn eredu_runtime::capture::partition::ScheduledPartitionCapture
        })
    }
    fn validate_partition_intervention_source(
        &self,
        value: &MlxTensor,
        claim: &eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
        window: Option<eredu_runtime::intervention::InterventionPrefillWindow>,
    ) -> Result<(), eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        use eredu_runtime::capture::FundedCaptureError;
        if window.is_some() {
            return Err(FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ));
        }
        self.partition_edit(claim)?
            .0
            .validate_with_custody(
                value.as_array(),
                claim,
                super::super::intervention::model::NativeCustody::Model(self.custody),
            )
            .map_err(partition_failure)
    }
    fn validate_partition_intervention_evidence_source(
        &self,
        value: &MlxTensor,
        claim: &eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
        side: InterventionEvidenceSide,
        window: Option<eredu_runtime::intervention::InterventionPrefillWindow>,
    ) -> Result<(), eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        use eredu_runtime::capture::FundedCaptureError;
        if window.is_some() {
            return Err(FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ));
        }
        let (source, rank) = self.partition_edit(claim)?;
        let evidence = source
            .evidence()
            .filter(|source| source.rank() == rank)
            .ok_or(FundedCaptureError::Backend(
                CaptureTensorNativeError::SourceChanged,
            ))?;
        evidence
            .validate_with_custody(
                value.as_array(),
                claim,
                side,
                None,
                super::super::intervention::model::NativeCustody::Model(self.custody),
            )
            .map_err(partition_failure)
    }
    fn apply_partition_intervention(
        &mut self,
        value: &MlxTensor,
        claim: &eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
        allowance: &mut eredu_runtime::capture::partition::PartitionInterventionLocalAllowance,
    ) -> Result<Option<MlxTensor>, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        use eredu_runtime::capture::FundedCaptureError;
        let (source, rank) = self.partition_edit(claim)?;
        if allowance.rank() != rank || allowance.window().is_some() {
            return Err(FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ));
        }
        source
            .execute_with_custody(
                value.as_array(),
                claim,
                allowance,
                super::super::intervention::model::NativeCustody::Model(self.custody),
                self.stream,
                self.observer,
                self.roots,
            )
            .map(|value| value.map(MlxTensor::from_array))
            .map_err(partition_failure)
    }
    fn validate_partition_prefill_source(
        &self,
        source: &MlxTensor,
        geometry: &eredu_runtime::capture::partition::PartitionPrefillReceiverSource,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        super::partition::native::validate_source(source.as_array(), geometry.source_shape())
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn validate_partition_invocation_source(
        &self,
        source: &MlxTensor,
        geometry: &eredu_runtime::capture::partition::PartitionInvocationReceiverSource,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        super::partition::native::validate_source(source.as_array(), geometry.source_shape())
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn complete_partition_source(
        &mut self,
        source: &MlxTensor,
    ) -> Result<(), eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        use crate::backend::array_copy::CaptureCompletion;
        let completion = CaptureCompletion::Original(self.observer);
        completion
            .validate()
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)?;
        super::partition::native::complete_source(
            &super::partition::native::ModelSourceRetainer {
                roots: self.roots,
                completion,
            },
            source.as_array(),
            self.stream,
            self.roots,
            completion,
        )
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }

    fn validate_routed_invocation_source(
        &self,
        source: &RoutedUnitCaptureSource<'_, MlxTensor>,
        geometry: &CaptureRoutedUnitsGeometry<'_>,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        crate::backend::array_copy::CompletedRoutedCaptureSource::validate_borrowed(
            &routed_source(source),
            geometry.bank(),
            geometry.source_shape()[0] as u64,
        )
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)?;
        floating_dtype(source.values.as_array())
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn estimate_routed_prefill(
        &self,
        geometry: &CaptureRoutedUnitsGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_routed_geometry(geometry)
    }
    fn transform_routed_batch(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, MlxTensor>,
        writer: eredu_runtime::working_memory::CaptureRoutedBatchWriter<'_, '_>,
    ) -> Result<(), eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        crate::backend::array_copy::execute_speculative_routed_capture(
            &routed_source(source),
            writer,
            self.stream,
            self.roots,
            self.custody,
            self.observer,
        )
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn routed_intervention_range(
        &self,
        source: &RoutedUnitCaptureSource<'_, MlxTensor>,
        claim: &eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<[u64; 2], eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        use eredu_runtime::capture::FundedCaptureError;
        self.edits
            .ok_or(FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?
            .routed_range(&routed_source(source), claim, self.custody)
            .map_err(FundedCaptureError::Backend)
    }
    fn routed_intervention_usage(
        &self,
        source: &RoutedUnitCaptureSource<'_, MlxTensor>,
        claim: &eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        use eredu_runtime::capture::FundedCaptureError;
        self.edits
            .ok_or(FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?
            .routed_usage(&routed_source(source), claim, self.custody)
            .map_err(FundedCaptureError::Backend)
    }
    fn apply_routed_intervention(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, MlxTensor>,
        batch: eredu_runtime::working_memory::RoutedInterventionBatch<'_, '_>,
    ) -> Result<Option<MlxTensor>, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        use eredu_runtime::capture::FundedCaptureError;
        self.edits
            .ok_or(FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?
            .execute_routed(
                &routed_source(source),
                batch,
                self.stream,
                self.roots,
                self.custody,
                self.observer,
            )
            .map(|value| value.map(MlxTensor::from_array))
    }
    fn intervention_usage(
        &self,
        source: &MlxTensor,
        claim: &eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        self.edits
            .ok_or(eredu_runtime::capture::FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?
            .usage(source.as_array(), claim, self.custody)
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn intervention_projection_usage(
        &self,
        source: &MlxTensor,
        claim: &eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<[CaptureUsage; 2], eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        self.edits
            .ok_or(eredu_runtime::capture::FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?
            .projection_usage(source.as_array(), claim, self.custody)
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn apply_intervention_projected(
        &mut self,
        source: &MlxTensor,
        claim: eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
        projection: [CaptureUsage; 2],
    ) -> Result<
        (
            Option<MlxTensor>,
            eredu_runtime::working_memory::ClaimedIntervention,
        ),
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        let (value, receipt) = self
            .edits
            .ok_or(eredu_runtime::capture::FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?
            .execute(
                source.as_array(),
                claim,
                charged,
                projection,
                self.stream,
                self.roots,
                self.custody,
                self.observer,
            )?;
        Ok((value.map(MlxTensor::from_array), receipt))
    }
    fn intervention_evidence_usage(
        &self,
        source: &MlxTensor,
        claim: &eredu_runtime::working_memory::CaptureInterventionEvidenceClaim<'_, '_>,
    ) -> Result<(TensorDtype, CaptureUsage), eredu_runtime::capture::FundedCaptureError<Self::Error>>
    {
        use eredu_runtime::{
            capture::FundedCaptureError, working_memory::CaptureInterventionEvidenceKind,
        };
        self.edits
            .ok_or(FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?
            .check_evidence(source.as_array(), claim, self.custody, false)
            .map_err(FundedCaptureError::Backend)?;
        Ok(match claim.kind() {
            CaptureInterventionEvidenceKind::Preview(claim) => (
                self.validate_source(source, claim.geometry())
                    .map_err(FundedCaptureError::Backend)?,
                self.estimate(source, claim.geometry())?,
            ),
            CaptureInterventionEvidenceKind::Summary(claim) => (
                self.validate_summary_source(source, claim.geometry())?,
                self.estimate_summary(claim.geometry())?,
            ),
        })
    }
    fn capture_intervention_evidence<'a>(
        &mut self,
        source: &MlxTensor,
        claim: eredu_runtime::working_memory::CaptureInterventionEvidenceClaim<'a, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedInterventionEvidence<'a>,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        use eredu_runtime::{
            capture::FundedCaptureError, working_memory::CaptureInterventionEvidenceKind,
        };
        self.edits
            .ok_or(FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?
            .check_evidence(source.as_array(), &claim, self.custody, true)
            .map_err(FundedCaptureError::Backend)?;
        let (receipt, kind) = claim.into_parts();
        // These shared model workers retain source and output aliases in Q;
        // selected-output/flatten completion settles lazy post-edit ancestors.
        Ok(match kind {
            CaptureInterventionEvidenceKind::Preview(claim) => receipt.finish_preview(
                self.transform(source, claim)
                    .map_err(FundedCaptureError::Backend)?,
            )?,
            CaptureInterventionEvidenceKind::Summary(claim) => {
                receipt.finish_summary(self.transform_summary(source, claim)?)?
            }
        })
    }
    fn validate_source(
        &self,
        source: &MlxTensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Self::Error> {
        Ok(
            match PreparedCaptureTensor::validate_borrowed_source(source.as_array(), geometry)? {
                safemlx::Dtype::Float32 => TensorDtype::F32,
                safemlx::Dtype::Float16 => TensorDtype::F16,
                safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
                _ => unreachable!("audited floating source"),
            },
        )
    }
    fn estimate(
        &self,
        _: &MlxTensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_tensor_geometry(geometry)
    }
    fn estimate_generated(
        &self,
        _: &MlxTensor,
        _: &GeneratedCaptureSource,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_tensor_geometry(geometry)
    }
    fn preflight_generated(
        &mut self,
        prototype: &MlxTensor,
        source: &GeneratedCaptureSource,
        program: eredu_nn::GeneratedTensorProgram<'_>,
        construct: bool,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        self.generated()
            .prepare(prototype.as_array(), source, program, construct, claim)
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn retain_generated_source(
        &mut self,
        prototype: &MlxTensor,
        role: eredu_nn::GeneratedTensorSourceRole,
        value: &MlxTensor,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        self.generated()
            .source(prototype.as_array(), role, value.as_array(), claim)
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn retain_generated_output(
        &mut self,
        value: &MlxTensor,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        self.generated()
            .retain(value.as_array(), claim)
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn validate_histogram_source(
        &self,
        source: &MlxTensor,
        geometry: &CaptureHistogramGeometry<'_>,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        let checked = || {
            PreparedCaptureHistogram::from_geometry(geometry)?
                .validate_source(source.as_array())?;
            floating_dtype(source.as_array())
        };
        checked().map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn estimate_histogram(
        &self,
        geometry: &CaptureHistogramGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_histogram(geometry)
    }
    fn transform_histogram(
        &mut self,
        source: &MlxTensor,
        claim: eredu_runtime::working_memory::CaptureHistogramClaim<'_, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedCaptureHistogram,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        crate::backend::array_copy::execute_speculative_histogram(
            source.as_array(),
            claim,
            self.stream,
            self.roots,
            self.custody,
            self.observer,
        )
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn validate_summary_source(
        &self,
        source: &MlxTensor,
        geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        let checked = || {
            PreparedCaptureSummary::from_geometry(geometry)?.validate_source(source.as_array())?;
            floating_dtype(source.as_array())
        };
        checked().map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn estimate_summary(
        &self,
        geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_summary(geometry)
    }
    fn transform_summary(
        &mut self,
        source: &MlxTensor,
        claim: eredu_runtime::working_memory::CaptureSummaryClaim<'_, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedCaptureSummary,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        crate::backend::array_copy::execute_speculative_summary(
            source.as_array(),
            claim,
            self.stream,
            self.roots,
            self.custody,
            self.observer,
        )
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn validate_candidate_source(
        &self,
        source: &MlxTensor,
        geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        let checked = || {
            CandidateExtraction::from_geometry(geometry)?.validate_source(source.as_array())?;
            floating_dtype(source.as_array())
        };
        checked().map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn estimate_candidates(
        &self,
        geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_candidates(geometry)
    }
    fn transform_candidates(
        &mut self,
        source: &MlxTensor,
        claim: eredu_runtime::working_memory::CaptureCandidateClaim<'_, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedCaptureCandidates,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        crate::backend::array_copy::execute_speculative_candidates(
            source.as_array(),
            claim,
            self.stream,
            self.roots,
            self.custody,
            self.observer,
            None,
        )
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn validate_token_score_source(
        &self,
        source: &MlxTensor,
        geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        let checked = || {
            TokenScoreProgram::from_geometry(geometry)?.validate_source(source.as_array())?;
            floating_dtype(source.as_array())
        };
        checked().map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn estimate_token_scores(
        &self,
        geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_token_scores(geometry)
    }
    fn transform_token_scores(
        &mut self,
        source: &MlxTensor,
        claim: eredu_runtime::working_memory::CaptureTokenScoreClaim<'_, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedCaptureTokenScores,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        crate::backend::array_copy::execute_speculative_token_scores(
            source.as_array(),
            claim,
            self.stream,
            self.roots,
            self.custody,
            self.observer,
            None,
        )
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn transform(
        &mut self,
        source: &MlxTensor,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Self::Error> {
        crate::backend::array_copy::execute_speculative_capture(
            source.as_array(),
            claim,
            self.stream,
            self.roots,
            self.custody,
            self.observer,
        )
    }
}
fn floating_dtype(source: &Array) -> Result<TensorDtype, CaptureTensorNativeError> {
    match source.dtype() {
        safemlx::Dtype::Float32 => Ok(TensorDtype::F32),
        safemlx::Dtype::Float16 => Ok(TensorDtype::F16),
        safemlx::Dtype::Bfloat16 => Ok(TensorDtype::Bf16),
        dtype => Err(CaptureTensorNativeError::UnsupportedDtype(dtype)),
    }
}

impl ScheduledNativeCapture<'_> {
    fn generated(&self) -> crate::backend::array_copy::GeneratedCaptureRetention<'_> {
        crate::backend::array_copy::GeneratedCaptureRetention::new(
            self.stream,
            self.roots,
            self.custody,
            self.observer,
        )
    }
}

fn routed_source<'a>(
    source: &RoutedUnitCaptureSource<'a, MlxTensor>,
) -> RoutedUnitCaptureSource<'a, Array> {
    RoutedUnitCaptureSource {
        values: source.values.as_array(),
        token_indices: source.token_indices.as_array(),
        selection_indices: source.selection_indices.as_array(),
        coefficients: source.coefficients.as_array(),
        source_groups: source.source_groups.as_array(),
        token_offset: source.token_offset,
        global_groups: source.global_groups,
    }
}

fn partition_failure(
    cause: super::super::intervention::partition::NativeFailure,
) -> eredu_runtime::capture::FundedCaptureError<CaptureTensorNativeError> {
    use super::super::intervention::partition::NativeFailure;
    eredu_runtime::capture::FundedCaptureError::Backend(match cause {
        NativeFailure::Native(cause) => cause,
        NativeFailure::Custody(cause) => CaptureTensorNativeError::Memory(cause),
        NativeFailure::Allowance(cause) => CaptureTensorNativeError::PartitionIntervention(cause),
    })
}
impl ScheduledNativeCapture<'_> {
    fn partition_edit(
        &self,
        claim: &eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<
        (
            &super::super::intervention::PreparedPartitionModelIntervention,
            usize,
        ),
        eredu_runtime::capture::FundedCaptureError<CaptureTensorNativeError>,
    > {
        use eredu_runtime::capture::FundedCaptureError;
        let rows = self.edits.ok_or(FundedCaptureError::Backend(
            CaptureTensorNativeError::SourceChanged,
        ))?;
        let (rank, _, _) = rows.partition_binding().ok_or(FundedCaptureError::Backend(
            CaptureTensorNativeError::SourceChanged,
        ))?;
        let source = rows
            .partition_source(claim.index())
            .map_err(FundedCaptureError::Backend)?
            .ok_or(FundedCaptureError::Backend(
                CaptureTensorNativeError::SourceChanged,
            ))?;
        Ok((source, rank))
    }
}
