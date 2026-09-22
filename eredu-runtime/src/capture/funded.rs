//! Closed original-account storage around the existing CaptureSession policy.
use super::*;
use crate::working_memory::{
    CaptureRunHostError, CaptureStepError, CaptureTensorClaim, ClaimedCaptureTensor,
    PendingCaptureDelivery, PreparedCaptureDelivery, PreparedCaptureRun, ScheduledCaptureStep,
    WorkingMemoryError,
};
use eredu_core::{DistributedCommitEpoch, checkpoint::TensorDtype};
use std::{fmt, mem::size_of};
mod embedded;
pub use embedded::{
    FundedAutoregressiveCaptureInvocation, FundedEmbeddedCaptureInvocation,
    FundedModelCaptureInvocation,
};
mod observer;
use observer::FundedCaptureObserver;
pub(crate) mod checkpoint;
pub(crate) mod speculative;
pub use checkpoint::{
    FundedCaptureCheckpoint, FundedCaptureCheckpointError, PreparedFundedCaptureCheckpoint,
};
pub use speculative::FundedSpeculativeCaptureInvocation;

/// Backend mechanism for the closed scheduled tensor destination. Implementations
/// must authenticate actual source geometry/dtype, original native admission and
/// completion. A host claim alone supplies none of those permissions. The source
/// check/estimate must allocate no shape/data payload or retain a native handle.
/// The transform consumes the exact coordinate claim and returns its sealed receipt.
pub trait ScheduledCaptureBackend {
    /// Logical controls and optional original-decision work for one authenticated
    /// selector invocation. Native allocation authority remains in its existing
    /// model or scheduled execution owner.
    fn routing_intervention_usage(
        &self,
        _rows: u64,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _window: Option<crate::intervention::InterventionPrefillWindow>,
    ) -> Result<CaptureUsage, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Copy the actual source control through its paid constructor after the
    /// shared logical ledger accepts usage. An empty overlap returns no edit.
    fn prepare_routing_intervention(
        &mut self,
        _rows: u64,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _window: Option<crate::intervention::InterventionPrefillWindow>,
    ) -> Result<
        Option<eredu_nn::routing_intervention::GroupSelectionControl>,
        FundedCaptureError<Self::Error>,
    > {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Authenticate the completed selector descriptors against the same source,
    /// row geometry and native owner before evidence or successful delivery.
    fn validate_routing_intervention_result(
        &self,
        _rows: u64,
        _original: Option<crate::RoutingDecision<'_, Self::Tensor>>,
        _effective: crate::RoutingDecision<'_, Self::Tensor>,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _window: Option<crate::intervention::InterventionPrefillWindow>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Explicit retained partition program for this same logical frame. The
    /// default keeps ordinary local capture behavior. A native implementation
    /// must supply its independently admitted transport and exact placement.
    fn partition_capture(
        &mut self,
    ) -> Option<&mut (dyn crate::capture::partition::ScheduledPartitionCapture + '_)> {
        None
    }
    /// Convert the routed callback's typed failure without formatting a new
    /// string. Concrete original owners additionally retain their native custody.
    fn routed_error(&self, cause: FundedCaptureError<Self::Error>) -> eredu_nn::Error {
        eredu_nn::Error::backend_retained_source(cause)
    }
    /// Validate the actual sparse provider source before quota or completion.
    fn validate_routed_prefill_source(
        &self,
        _source: &eredu_core::capture::RoutedUnitCaptureSource<'_, Self::Tensor>,
        _fragment: &eredu_core::capture::CaptureRoutedPrefillFragment<'_, '_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Existing full logical routed estimate, charged only by the shared ledger.
    fn estimate_routed_prefill(
        &self,
        _geometry: &eredu_core::capture::CaptureRoutedUnitsGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "routed prefill source is unavailable".into(),
        ))
    }
    /// Copy one actual provider batch through the existing stamped native carrier.
    fn transform_routed_prefill(
        &mut self,
        _source: &eredu_core::capture::RoutedUnitCaptureSource<'_, Self::Tensor>,
        _writer: crate::working_memory::CaptureRoutedPrefillWriter<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Validate a real ordinary/decode sparse batch against the admitted invocation.
    fn validate_routed_invocation_source(
        &self,
        _source: &eredu_core::capture::RoutedUnitCaptureSource<'_, Self::Tensor>,
        _geometry: &eredu_core::capture::CaptureRoutedUnitsGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Populate the existing original-account destination through a short source loan.
    fn transform_routed_batch(
        &mut self,
        _source: &eredu_core::capture::RoutedUnitCaptureSource<'_, Self::Tensor>,
        _writer: crate::working_memory::CaptureRoutedBatchWriter<'_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Validate the actual provider input independently of selected overlap.
    fn validate_partition_routed_invocation(
        &self,
        _invocation: &crate::RoutedUnitInvocation<'_, Self::Tensor>,
        _layout: &eredu_core::capture::PartitionRoutedUnitCaptureLayout<'_>,
    ) -> Result<(u64, TensorDtype), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Exact full fragment estimate, charged once by the retained shared ledger.
    fn estimate_partition_routed(
        &self,
        _request: &eredu_core::capture::PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "partition routed source is unavailable".into(),
        ))
    }
    /// Validate a real routed batch without requiring selected overlap. The
    /// returned extent is the actual coefficient token-row count, checked with
    /// all five descriptors. An idle invocation has no batch or result witness.
    fn validate_partition_routed_batch_source(
        &self,
        _source: &eredu_core::capture::PartitionRoutedUnitCaptureSource<'_, Self::Tensor>,
        _layout: &eredu_core::capture::PartitionRoutedUnitCaptureLayout<'_>,
        _actual_native_rows: u64,
    ) -> Result<(TensorDtype, u64), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Check a real five-source batch against both global selection and actual
    /// provider invocation extent; prefill input token IDs restart each chunk.
    fn validate_partition_routed_source(
        &self,
        _source: &eredu_core::capture::PartitionRoutedUnitCaptureSource<'_, Self::Tensor>,
        _request: &eredu_core::capture::PartitionRoutedUnitCaptureRequest<'_>,
        _invocation_source_tokens: u64,
        _actual_native_rows: u64,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Bind existing native source/completion loans to the short original Host
    /// writer. The callback cannot fabricate a destination or a transport vote.
    fn transform_partition_routed_batch(
        &mut self,
        _source: &eredu_core::capture::PartitionRoutedUnitCaptureSource<'_, Self::Tensor>,
        _writer: crate::working_memory::CapturePartitionRoutedWriter<'_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Actual borrowed native/portable tensor.
    type Tensor;
    /// Original typed mechanism failure, preserved by the enclosing observer.
    type Error: std::error::Error + Send + Sync + 'static;
    /// Validate the exact local chunk whose selected spatial overlap is empty.
    /// Completion remains the existing independent source-settlement method.
    fn validate_partition_prefill_source(
        &self,
        _source: &Self::Tensor,
        _geometry: &crate::capture::partition::PartitionPrefillReceiverSource,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Exact physical decode source for an empty selected overlap or replica.
    /// This supplies no new completion engine or native permission.
    fn validate_partition_invocation_source(
        &self,
        _source: &Self::Tensor,
        _geometry: &crate::capture::partition::PartitionInvocationReceiverSource,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Geometry.into())
    }
    /// Complete a selected receiver's actual local source before its peer enters
    /// a producer-only transform. The sealed partition program spends this
    /// selection/epoch first; the backend must retain the root in the existing
    /// model owner and authenticate its independently admitted native completion.
    /// This creates no capture record or logical capture charge.
    fn complete_partition_source(
        &mut self,
        _source: &Self::Tensor,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Check the exact borrowed source and already-spent original operation.
    /// This returns the existing logical activation cost only; native authority
    /// and all physical destinations remain in the numerical phase owner.
    /// Authenticate the actual local component/window program and original
    /// claim before the existing partition member preflight vote.
    fn validate_partition_intervention_source(
        &self,
        _source: &Self::Tensor,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _window: Option<crate::intervention::InterventionPrefillWindow>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Authenticate the actual before/after tensor against the retained original
    /// companion source. The runtime's existing receipt owns quota and delivery.
    fn validate_partition_intervention_evidence_source(
        &self,
        _source: &Self::Tensor,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _side: eredu_core::capture::InterventionEvidenceSide,
        _window: Option<crate::intervention::InterventionPrefillWindow>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Run the shared local edit worker under its prepaid original allowance.
    /// The runtime retains and finishes the frame claim after both member votes.
    fn apply_partition_intervention(
        &mut self,
        _source: &Self::Tensor,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _allowance: &mut crate::capture::partition::PartitionInterventionLocalAllowance,
    ) -> Result<Option<Self::Tensor>, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    fn intervention_usage(
        &self,
        _source: &Self::Tensor,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Exact window metadata followed by payload-projection metadata. Reserve
    /// each in order: a refused later stage never refunds an earlier charge.
    /// This host/encoding policy is separate from the native edit estimate.
    fn intervention_projection_usage(
        &self,
        _source: &Self::Tensor,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<[CaptureUsage; 2], FundedCaptureError<Self::Error>> {
        Ok([CaptureUsage::default(); 2])
    }
    /// The same scheduled operation over one authenticated physical prompt span.
    fn prefill_intervention_projection_usage(
        &self,
        _source: &Self::Tensor,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _window: crate::intervention::InterventionPrefillWindow,
    ) -> Result<[CaptureUsage; 2], FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    fn prefill_intervention_usage(
        &self,
        _source: &Self::Tensor,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _window: crate::intervention::InterventionPrefillWindow,
    ) -> Result<CaptureUsage, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    fn apply_prefill_intervention(
        &mut self,
        _source: &Self::Tensor,
        _fragment: crate::working_memory::InterventionPrefillFragment<'_, '_>,
        _charged: CaptureUsage,
        _projection: [CaptureUsage; 2],
    ) -> Result<Option<Self::Tensor>, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Exact native routed chunk range from the actual five source descriptors.
    /// Every invocation and source is checked against this same spent claim.
    fn routed_intervention_range(
        &self,
        _source: &RoutedUnitCaptureSource<'_, Self::Tensor>,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<[u64; 2], FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Existing full physical invocation edit cost, charged once before any
    /// source values are read. Numerical/Host admission is independent.
    fn routed_intervention_usage(
        &self,
        _source: &RoutedUnitCaptureSource<'_, Self::Tensor>,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Actual provider range local to this canonical prompt chunk. The shared
    /// cursor authenticates and maps it to the original logical prompt.
    fn prefill_routed_intervention_range(
        &self,
        _source: &RoutedUnitCaptureSource<'_, Self::Tensor>,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _window: crate::intervention::InterventionPrefillWindow,
    ) -> Result<[u64; 2], FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Full logical prompt edit cost, accepted once before the first native
    /// source read. This estimate is not physical allocation authority.
    fn prefill_routed_intervention_usage(
        &self,
        _source: &RoutedUnitCaptureSource<'_, Self::Tensor>,
        _claim: &crate::working_memory::CaptureInterventionClaim<'_>,
        _window: crate::intervention::InterventionPrefillWindow,
    ) -> Result<CaptureUsage, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Reuse the existing sparse lowerer and native selection/update worker.
    /// Successful implementations finish the exclusive batch with their exact
    /// paid lowering; leaving it unfinished poisons the operation cursor.
    fn apply_routed_intervention(
        &mut self,
        _source: &RoutedUnitCaptureSource<'_, Self::Tensor>,
        _batch: crate::working_memory::RoutedInterventionBatch<'_, '_>,
    ) -> Result<Option<Self::Tensor>, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// The same static worker with an explicit no-overlap acknowledgment.
    /// None creates no tensor and leaves the current input in place. Receipt
    /// publication still waits for the enclosing original completion.
    fn apply_intervention_projected(
        &mut self,
        source: &Self::Tensor,
        claim: crate::working_memory::CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
        projection: [CaptureUsage; 2],
    ) -> Result<
        (
            Option<Self::Tensor>,
            crate::working_memory::ClaimedIntervention,
        ),
        FundedCaptureError<Self::Error>,
    > {
        if projection != [CaptureUsage::default(); 2] {
            return Err(CaptureProtocolError::Transaction.into());
        }
        self.apply_intervention(source, claim, charged)
            .map(|(value, receipt)| (Some(value), receipt))
    }
    /// Apply the admitted action through the existing intervention worker. The
    /// receipt describes provisional successful construction. The enclosing
    /// numerical owner must settle every emitted root before successful delivery.
    fn apply_intervention(
        &mut self,
        _source: &Self::Tensor,
        _claim: crate::working_memory::CaptureInterventionClaim<'_>,
        _charged: CaptureUsage,
    ) -> Result<
        (Self::Tensor, crate::working_memory::ClaimedIntervention),
        FundedCaptureError<Self::Error>,
    > {
        Err(CaptureProtocolError::Transaction.into())
    }
    /// Optionally register a segment from this exact original bank before the
    /// first frame claim. Implementations use only their existing native scope.
    /// None keeps existing full-span retention and grants no per-chunk reuse.
    fn prepare_prefill_chunk_retention(
        &mut self,
        _bootstrap: crate::working_memory::CapturePrefillSourceBootstrap<'_>,
        _context: &crate::inspection::PrefillChunkRetentionContext<'_>,
    ) -> Result<
        Option<crate::inspection::PreparedPrefillChunkRetention>,
        FundedCaptureError<Self::Error>,
    > {
        Ok(None)
    }
    /// The final frame may already be sealed/provisional. This backend-only
    /// callback must not claim another frame or tensor. No communication, new
    /// native work, grant or implicit certification is permitted.
    fn retire_prefill_chunk_retention(
        &mut self,
        _settled: crate::inspection::SettledPrefillChunkRetention,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Ok(())
    }
    /// Validate the actual borrowed source against every admitted source dimension.
    /// Returns its actual supported floating dtype, without conversion or download.
    fn validate_source(
        &self,
        tensor: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Self::Error>;
    /// Logical capture usage for this selected program; independent of physical H.
    fn estimate(
        &self,
        tensor: &Self::Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError>;
    /// Logical generated-source transform usage before creation charge. The
    /// prototype supplies geometry; its actual precision may differ from the
    /// checked generated source. Defaults preserve typed unsupported behavior.
    fn estimate_generated(
        &self,
        _prototype: &Self::Tensor,
        _source: &GeneratedCaptureSource,
        _geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "retained generated capture is unavailable".into(),
        ))
    }
    /// Validate this already-spent claim against the same original native work
    /// and reserve fixed descriptor slots before visiting/calling the producer.
    /// No RefCell/scope guard may remain live when this returns. The checked
    /// program is not numerical authority; original admission remains required.
    fn preflight_generated(
        &mut self,
        _prototype: &Self::Tensor,
        _source: &GeneratedCaptureSource,
        _program: eredu_nn::GeneratedTensorProgram<'_>,
        _construct: bool,
        _claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Retain each actual compact operand in the original work before generation.
    fn retain_generated_source(
        &mut self,
        _prototype: &Self::Tensor,
        _role: eredu_nn::GeneratedTensorSourceRole,
        _value: &Self::Tensor,
        _claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Retain each newly created intermediate before further fallible work.
    /// Prior roots stay in original recovery on rejection/unwind. This does not
    /// publish storage, add native allowance or certify any work scope.
    fn retain_generated_output(
        &mut self,
        _value: &Self::Tensor,
        _claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Validate one actual physical source against the closed logical fragment.
    /// This read-only check precedes logical quota and any native evaluation.
    fn validate_prefill_source(
        &self,
        _tensor: &Self::Tensor,
        _fragment: &eredu_core::capture::CapturePrefillFragment<'_, '_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Full logical selection usage, independent of this chunk's shorter source.
    /// The existing ledger remains the only logical quota authority.
    fn estimate_prefill(
        &self,
        _geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "prefill fragments are unavailable".into(),
        ))
    }
    /// Consume the original target's short fragment claim in the same stamped
    /// native carrier. No new host target, native scope or completion is issued.
    fn transform_prefill_fragment(
        &mut self,
        _tensor: &Self::Tensor,
        _claim: crate::working_memory::CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Authenticate the spent original fragment claim and reserve all descriptor
    /// slots before the producer. All temporary scope/collector borrows must end.
    fn preflight_generated_fragment(
        &mut self,
        _prototype: &Self::Tensor,
        _source: &GeneratedCaptureSource,
        _program: eredu_nn::GeneratedTensorProgram<'_>,
        _construct: bool,
        _claim: &crate::working_memory::CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Retain each actual compact operand in this chunk's original carrier.
    fn retain_generated_fragment_source(
        &mut self,
        _prototype: &Self::Tensor,
        _role: eredu_nn::GeneratedTensorSourceRole,
        _value: &Self::Tensor,
        _claim: &crate::working_memory::CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Retain every generated intermediate before the next fallible operation.
    /// Prior hooks remain retained until the canonical settled chunk boundary.
    fn retain_generated_fragment_output(
        &mut self,
        _value: &Self::Tensor,
        _claim: &crate::working_memory::CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::GeneratedSource.into())
    }
    /// Exact selected source check before any original scalar-summary work.
    fn validate_summary_source(
        &self,
        _tensor: &Self::Tensor,
        _geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Validate one exact evidence source and estimate its ordinary logical cost.
    /// The same observer ledger reserves that cost before any payload work.
    fn intervention_evidence_usage(
        &self,
        _source: &Self::Tensor,
        _claim: &crate::working_memory::CaptureInterventionEvidenceClaim<'_, '_>,
    ) -> Result<(TensorDtype, CaptureUsage), FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Consume the existing Preview/Summary claim inside this actual operation.
    /// The returned evidence remains provisional until enclosing completion.
    fn capture_intervention_evidence<'a>(
        &mut self,
        _source: &Self::Tensor,
        _claim: crate::working_memory::CaptureInterventionEvidenceClaim<'a, '_>,
    ) -> Result<
        crate::working_memory::ClaimedInterventionEvidence<'a>,
        FundedCaptureError<Self::Error>,
    > {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Existing logical transform quota, independent of physical original admission.
    fn estimate_summary(
        &self,
        _geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "summary capture is unavailable".into(),
        ))
    }
    /// The same logical p0 quota sums actual physical fragment usage once.
    fn estimate_prefill_summary(
        &self,
        _plan: &CapturePrefillTransformPlan<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "prefill summary capture is unavailable".into(),
        ))
    }
    /// Consume the original fixed scalar claim after exact native completion.
    fn transform_summary(
        &mut self,
        _tensor: &Self::Tensor,
        _claim: crate::working_memory::CaptureSummaryClaim<'_, '_>,
    ) -> Result<crate::working_memory::ClaimedCaptureSummary, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Exact selected source check before any original scalar-histogram work.
    fn validate_histogram_source(
        &self,
        _tensor: &Self::Tensor,
        _geometry: &CaptureHistogramGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Existing logical transform quota, independent of physical original admission.
    fn estimate_histogram(
        &self,
        _geometry: &CaptureHistogramGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "histogram capture is unavailable".into(),
        ))
    }
    /// The same logical p0 quota sums actual physical fragment usage once.
    fn estimate_prefill_histogram(
        &self,
        _plan: &CapturePrefillTransformPlan<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "prefill histogram capture is unavailable".into(),
        ))
    }
    /// Consume the original fixed scalar claim after exact native completion.
    fn transform_histogram(
        &mut self,
        _tensor: &Self::Tensor,
        _claim: crate::working_memory::CaptureHistogramClaim<'_, '_>,
    ) -> Result<crate::working_memory::ClaimedCaptureHistogram, FundedCaptureError<Self::Error>>
    {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Validate the actual terminal candidate source before quota or native work.
    fn validate_candidate_source(
        &self,
        _tensor: &Self::Tensor,
        _geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Logical candidate usage is independent of its original host and native Q.
    fn estimate_candidates(
        &self,
        _geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "candidate extraction is unavailable".into(),
        ))
    }
    /// Consume one original terminal claim; all native roots remain in existing work.
    fn transform_candidates(
        &mut self,
        _tensor: &Self::Tensor,
        _claim: crate::working_memory::CaptureCandidateClaim<'_, '_>,
    ) -> Result<crate::working_memory::ClaimedCaptureCandidates, FundedCaptureError<Self::Error>>
    {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Validate the actual terminal token-score source before quota or native work.
    fn validate_token_score_source(
        &self,
        _tensor: &Self::Tensor,
        _geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Self::Error>> {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Logical token-score usage is independent of its original host and native Q.
    fn estimate_token_scores(
        &self,
        _geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "token-score extraction is unavailable".into(),
        ))
    }
    /// Consume one original terminal claim; all native roots remain in existing work.
    fn transform_token_scores(
        &mut self,
        _tensor: &Self::Tensor,
        _claim: crate::working_memory::CaptureTokenScoreClaim<'_, '_>,
    ) -> Result<crate::working_memory::ClaimedCaptureTokenScores, FundedCaptureError<Self::Error>>
    {
        Err(CaptureProtocolError::PrefillAttribution.into())
    }
    /// Execute the separately admitted native program and fill this host claim.
    /// Partial native roots and exact scope pins remain in the implementation's
    /// recovery owner on error/unwind. No completion is inferred by this observer.
    fn transform(
        &mut self,
        tensor: &Self::Tensor,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Self::Error>;
}

/// Original typed failure at a funded capture hook. No payload is detached into
/// this error: failed frame storage remains in its lexical/session owner.
#[derive(Debug, thiserror::Error)]
pub enum FundedCaptureError<E: std::error::Error + 'static> {
    /// Existing logical budget/admission policy.
    #[error(transparent)]
    Admission(#[from] CaptureError),
    /// Original physical host constructor/receipt rejection.
    #[error(transparent)]
    Host(#[from] CaptureRunHostError),
    /// Terminal empty-host completion failure retains its original H/cause.
    #[error(transparent)]
    HostFinish(#[from] crate::working_memory::ScheduledCaptureTensorFailure),
    /// Partial fixed candidate destination and exact original account failure.
    #[error(transparent)]
    CandidateFinish(#[from] crate::working_memory::CaptureCandidateFailure),
    /// Partial ordered token-score result retains its exact original host account.
    #[error(transparent)]
    TokenScoreFinish(#[from] crate::working_memory::CaptureTokenScoreFailure),
    /// Completed summary refusal retains the original scalar payload and account.
    #[error(transparent)]
    SummaryFinish(#[from] crate::working_memory::CaptureSummaryFailure),
    /// Rejected Histogram storage retains its exact paying account.
    #[error(transparent)]
    HistogramFinish(#[from] crate::working_memory::CaptureHistogramFailure),
    /// Allocation-free sequencing rejection.
    #[error(transparent)]
    Protocol(#[from] CaptureProtocolError),
    /// Original receipt/placement source, including independent transport custody.
    #[error(transparent)]
    Partition(#[from] crate::capture::partition::PartitionCaptureProgramError),
    /// Original native/backend source; not converted to a diagnostic String.
    #[error("{0}")]
    Backend(#[source] E),
}
/// A shared drain failed without consuming the owned pending delivery.
#[derive(Debug, thiserror::Error)]
pub enum FundedCaptureDrainError {
    /// The transaction has not received its terminal status.
    #[error("capture transaction has not finished")]
    PendingTransaction,
    /// Final host publication failed; the same aborted owner remains in the slot.
    #[error(transparent)]
    Delivery(#[from] CaptureStepError),
}

/// One ledger, one finite claim bank and one undrained delivery. Construct only by
/// consuming an unused PreparedCaptureRun; no mutable/raw legacy session escapes.
/// Native applicability, source registration, original request quotation and
/// exact completion remain the enclosing backend's obligations. This additive
/// collector supports the ordinary tensor geometry priced by CaptureRunHostPlan;
/// legacy partition/intervention/routed collectors retain their existing paths.
/// Retained generated sources use the same ledger/claims only through an
/// explicitly implementing original-work backend. Bare factories remain rejected.
/// Original observed admission and public managed integration remain separate.
pub struct FundedCaptureSession {
    // Ordinary speculative attribution is applied later by the shared driver.
    // This changes logical disposition only; every delivery still owns its H.
    untracked: bool,
    envelope_usage: Option<CaptureUsage>,
    prepared_prefix: Option<OriginalSpeculativeCapturePrefix>,
    pub(in crate::capture) partition_run: Option<partition::PreparedPartitionCaptureRunIdentity>,
    session: CaptureSession,
    delivery: Option<FundedDelivery>,
    lineage: crate::working_memory::CaptureRunLedger,
    // All session/frame payload and identity allocations retire before bank H.
    run: PreparedCaptureRun,
}
enum FundedDelivery {
    Ready(SharedCapturedStep),
    Aborted(PendingCaptureDelivery),
}
impl FundedCaptureSession {
    pub(crate) fn from_run(run: PreparedCaptureRun) -> Self {
        let lineage = run.new_ledger(CaptureUsage::default());
        Self::from_run_with_lineage(run, lineage)
    }
    fn from_run_with_lineage(
        run: PreparedCaptureRun,
        lineage: crate::working_memory::CaptureRunLedger,
    ) -> Self {
        Self {
            untracked: false,
            envelope_usage: None,
            prepared_prefix: None,
            partition_run: None,
            session: CaptureSession::new(run.source().clone()),
            delivery: None,
            lineage,
            run,
        }
    }
    pub(crate) fn validate_factory(&self) -> Result<(), WorkingMemoryError> {
        self.run.validate_session_account()
    }
    /// Recheck this bank's original account and the supplied active native scope
    /// before model work. This issues no claim and changes no accounting. It does
    /// not authenticate the semantic quote, selected plan, actual tensor source,
    /// or completion; those remain the enclosing original admission's duties.
    pub fn validate_native_scope(
        &self,
        native: &crate::working_memory::WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.run.validate_session_native_scope(native)
    }
    /// Exact retained physical admission source; no raw owning export.
    pub fn source(&self) -> &SharedCapturePlan {
        self.run.source()
    }
    /// Exact immutable intervention source retained by the admitted run.
    pub fn intervention_source(
        &self,
    ) -> Option<&crate::working_memory::OriginalInterventionSource> {
        self.run.intervention_source()
    }
    /// Cumulative logical quota, including failed/aborted attempts.
    pub fn usage(&self) -> CaptureUsage {
        self.session.ledger.total()
    }
    /// Whether this admitted schedule owns physical frame constructors, including
    /// intervention-only frames and already-spent constructors.
    pub fn has_frame_claims(&self) -> bool {
        self.run.has_frame_claims()
    }
    /// Spent constructor coordinates never become available again.
    pub fn spent_steps(&self) -> usize {
        self.run.spent_steps()
    }
    /// Includes aborted frames, failed starts without a frame, and transactions
    /// awaiting final status. A caller must not infer absence from a raw drain.
    pub fn has_pending_step(&self) -> bool {
        self.delivery.is_some() || self.session.transaction.is_some()
    }
    /// Borrow one lexical observer. The actual session and bank remain external
    /// to it, so the claim row never self-borrows its owning struct. The observer
    /// always drops before this function returns/unwinds. Backend payload/error
    /// mapper ownership is borrowed and must be priced by its own native caller.
    pub fn with_observer<T, E, N, R>(
        &mut self,
        backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        prediction: u64,
        map_error: &dyn Fn(FundedCaptureError<E>) -> N,
        operation: impl FnOnce(&mut dyn crate::ActivationObserver<T, N>) -> R,
    ) -> Result<R, CaptureProtocolError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.with_observer_inner(backend, prediction, None, None, None, map_error, operation)
    }
    /// Borrow the same observer/transaction with an architecture-bound ordinary
    /// row selection. The exact physical candidate must already be quoted and
    /// installed by the caller; this supplies no native or source authority.
    pub fn with_prefill_observer<T, E, N, R>(
        &mut self,
        backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        bound: crate::layered::BoundCaptureSelection<'_>,
        map_error: &dyn Fn(FundedCaptureError<E>) -> N,
        operation: impl FnOnce(&mut dyn crate::ActivationObserver<T, N>) -> R,
    ) -> Result<R, CaptureProtocolError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        if !bound.selection().source().same_storage(self.source()) {
            return Err(CaptureProtocolError::Geometry);
        }
        self.with_observer_inner(backend, 0, Some(bound), None, None, map_error, operation)
    }
    /// Borrow the same collector with the original accepted physical contract.
    /// The view stays borrowed through the lexical observer; no custody or
    /// original capture bank can be replaced by semantic binding alone.
    pub fn with_admitted_prefill_observer<T, E, N, R>(
        &mut self,
        backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        admitted: &crate::working_memory::AdmittedPrefillCapture<'_>,
        map_error: &dyn Fn(FundedCaptureError<E>) -> N,
        operation: impl FnOnce(&mut dyn crate::ActivationObserver<T, N>) -> R,
    ) -> Result<R, CaptureProtocolError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let bound = admitted.selection();
        if !bound.selection().source().same_storage(self.source()) {
            return Err(CaptureProtocolError::Geometry);
        }
        self.with_observer_inner(
            backend,
            0,
            Some(bound),
            Some(admitted),
            None,
            map_error,
            operation,
        )
    }
    /// The saved source's first native single-token prefill, observed at its
    /// absolute decode coordinate through the same transaction and ledger.
    pub fn with_admitted_continuation_observer<T, E, N, R>(
        &mut self,
        backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        admitted: &crate::working_memory::AdmittedCaptureContinuation<'_>,
        map_error: &dyn Fn(FundedCaptureError<E>) -> N,
        operation: impl FnOnce(&mut dyn crate::ActivationObserver<T, N>) -> R,
    ) -> Result<R, CaptureProtocolError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        if !admitted.selection().source().same_storage(self.source())
            || self.run.first_prediction() != admitted.first_prediction()
            || self.run.spent_steps() != 0
        {
            return Err(CaptureProtocolError::Geometry);
        }
        self.with_observer_inner(
            backend,
            admitted.first_prediction(),
            None,
            None,
            Some(admitted),
            map_error,
            operation,
        )
    }

    fn with_observer_inner<T, E, N, R>(
        &mut self,
        backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        prediction: u64,
        bound: Option<crate::layered::BoundCaptureSelection<'_>>,
        admitted: Option<&crate::working_memory::AdmittedPrefillCapture<'_>>,
        continuation: Option<&crate::working_memory::AdmittedCaptureContinuation<'_>>,
        map_error: &dyn Fn(FundedCaptureError<E>) -> N,
        operation: impl FnOnce(&mut dyn crate::ActivationObserver<T, N>) -> R,
    ) -> Result<R, CaptureProtocolError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        if self.has_pending_step() {
            return Err(CaptureProtocolError::Undrained);
        }
        if self.run.invocation().is_some()
            && (bound.is_some() || admitted.is_some() || continuation.is_some())
        {
            return Err(CaptureProtocolError::Geometry);
        }
        let invocation_readout = self.run.invocation_readout()?;
        let current = self.lineage.borrow()?;
        self.session.ledger = match &self.prepared_prefix {
            Some(prefix) => prefix.resume(self.run.source(), current.usage())?,
            None => CaptureLedger::with_inherited_usage(&self.session.plan, current.usage())
                .map_err(|_| CaptureProtocolError::Geometry)?,
        };
        let ledger = CumulativeUpdate {
            session: &mut self.session,
            current,
        };
        let mut observer = FundedCaptureObserver::new(
            &mut *ledger.session,
            &mut self.delivery,
            &mut self.run,
            backend,
            prediction,
            map_error,
            bound,
            admitted,
            continuation,
            self.untracked,
            self.envelope_usage,
            self.prepared_prefix.is_some(),
            invocation_readout,
        );
        Ok(operation(&mut observer))
    }
    /// Move one shared delivery only after the caller establishes exact native
    /// completion/recovery. This host operation does not certify native work.
    /// Failed aborted publication restores the same owner; no claim/quota refund
    /// or second allocation allowance is issued. Ready finalizers stay valid
    /// after account closure because all fallible sealing preceded agreement.
    pub fn take_shared_step(
        &mut self,
    ) -> Result<Option<SharedCapturedStep>, FundedCaptureDrainError> {
        self.drain_sealed_step()
    }
    // Numerical failure evidence is host-only. Its native producer separately
    // retains unresolved work in Recovery and never claims success here.
    pub(crate) fn take_failed_numerical_evidence(
        &mut self,
    ) -> Result<Option<SharedCapturedStep>, FundedCaptureDrainError> {
        self.drain_sealed_step()
    }
    fn drain_sealed_step(&mut self) -> Result<Option<SharedCapturedStep>, FundedCaptureDrainError> {
        if self
            .session
            .transaction
            .is_some_and(|(_, s)| s == CaptureTransactionStatus::Pending)
        {
            return Err(FundedCaptureDrainError::PendingTransaction);
        }
        let step = match self.delivery.take() {
            None => None,
            Some(FundedDelivery::Ready(step)) => Some(step),
            Some(FundedDelivery::Aborted(pending)) => match pending.finish() {
                Ok(step) => Some(step),
                Err(error) => {
                    let (pending, cause) = error.into_parts();
                    self.delivery = Some(FundedDelivery::Aborted(pending));
                    return Err(cause.into());
                }
            },
        };
        self.session.checkpoint_ready = match (&step, self.session.transaction) {
            (Some(step), _) => {
                step.as_ref().outcome == CaptureStepOutcome::Committed
                    && !step
                        .as_ref()
                        .records
                        .iter()
                        .any(|r| matches!(r.outcome, CaptureOutcome::Failed { .. }))
            }
            (None, Some((_, CaptureTransactionStatus::Committed))) => !self.run.has_frame_claims(),
            (None, None) => self.session.checkpoint_ready,
            _ => false,
        };
        self.session.transaction = None;
        Ok(step)
    }
}
// Declared before the observer, so its final terminal/drop processing occurs
// before totals are recorded. It also runs after every callback error or panic.
struct CumulativeUpdate<'a> {
    session: &'a mut CaptureSession,
    current: crate::working_memory::CaptureRunLedgerGuard<'a>,
}
impl Drop for CumulativeUpdate<'_> {
    fn drop(&mut self) {
        self.current.record(self.session.ledger.total());
    }
}

impl fmt::Debug for FundedCaptureSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FundedCaptureSession")
            .field("spent_steps", &self.spent_steps())
            .field("pending", &self.has_pending_step())
            .finish_non_exhaustive()
    }
}

// One fixed owner/identity plus its moves, one lexical observer/state machine and
// delivery/error controls, including the shared core delivery slot. This adds no
// arbitrary backend payload allowance.
// Backend/error mapper are borrowed fat pointers; their concrete representation
// never becomes a stored generic payload of the session/observer. The error
// wrapper's fixed additional control is included; the backend's original error
// value and caller-returned R remain caller-owned native/operation obligations.
// No permission to allocate them is supplied here. Identity Arc and
// account bookkeeping keep their established classification; the actual
// CaptureHostOwner payload is explicitly included here.
pub(crate) fn control_bytes() -> Result<u64, WorkingMemoryError> {
    // The native installed slot is optional; its actual collector control is
    // retained within H. Source-witness/identity handles remain accounting metadata.
    let n = size_of::<Option<FundedCaptureSession>>()
        .checked_add(observer::routed::control_bytes().ok_or(WorkingMemoryError::Overflow)?)
        .and_then(|n| n.checked_add(observer::routing::control_bytes()?))
        .ok_or(WorkingMemoryError::Overflow)?
        .checked_add(size_of::<CumulativeUpdate<'_>>())
        .and_then(|n| n.checked_add(size_of::<CaptureHostOwner>()))
        .and_then(|n| n.checked_add(size_of::<CaptureSession>()))
        .and_then(|n| {
            n.checked_add(size_of::<
                FundedCaptureObserver<'_, (), std::convert::Infallible, ()>,
            >())
        })
        .and_then(|n| {
            n.checked_add(size_of::<(
                Option<bool>,
                Result<Option<bool>, CaptureProtocolError>,
                CaptureObservationStep<'_>,
                Result<CaptureObservationStep<'_>, CaptureProtocolError>,
                Option<CaptureInvocationWindow>,
                CaptureUsage, // actual physical fragment metadata, alive beside value usage
                Result<Option<CaptureUsage>, CaptureError>,
                Result<Option<CaptureSkipReason>, CaptureError>,
                std::time::Instant,
                std::time::Duration,
                f64,
                &mut dyn crate::capture::partition::ScheduledPartitionCapture,
                &mut crate::working_memory::ScheduledCaptureStep<'_>,
                Result<(), crate::capture::partition::PartitionCaptureProgramError>,
            )>())
        })
        .and_then(|n| n.checked_add(size_of::<observer::generated::GeneratedState<()>>()))
        .and_then(|n| n.checked_add(observer::replica::control_bytes()))
        .and_then(|n| n.checked_add(observer::fragments::projection_control_bytes()))
        .and_then(|n| n.checked_add(observer::interventions::intervention_control_bytes()?))
        // Full logical FP8 reconstruction keeps this checked signed shape and
        // plan alive together while reserving the original one-time quota.
        .and_then(|n| n.checked_add(32 * size_of::<i32>()))
        .and_then(|n| n.checked_add(size_of::<eredu_nn::BlockFp8InputReconstructionPlan<'_>>()))
        .and_then(|n| n.checked_add(size_of::<FundedDelivery>()))
        .and_then(|n| n.checked_add(size_of::<FundedCaptureDrainError>()))
        .and_then(|n| n.checked_add(size_of::<FundedCaptureError<std::convert::Infallible>>()))
        .and_then(|n| n.checked_add(size_of::<Option<eredu_core::capture::SharedCapturedStep>>()))
        .and_then(|n| n.checked_mul(3))
        .ok_or(WorkingMemoryError::Overflow)?;
    u64::try_from(n)
        .ok()
        .and_then(|bytes| {
            bytes.checked_add(crate::working_memory::CaptureRunLedger::control_bytes().ok()?)
        })
        .ok_or(WorkingMemoryError::Overflow)
}

#[cfg(test)]
mod error_tests;
