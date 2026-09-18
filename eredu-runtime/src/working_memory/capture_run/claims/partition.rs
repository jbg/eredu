//! Direct canonical receipt decoding into one already spent scheduled tensor.
use super::*;
use eredu_core::capture::{PartitionCaptureContext, CaptureUsage};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use serde_json::bounded_events::{Event, Plan, PlanError, Sink};
use std::mem::{size_of, size_of_val};
mod reader;
mod vocabulary;
pub(in crate::working_memory::capture_run) use vocabulary::{VocabularyDestination,prepare_vocabulary_decoder,decode_vocabulary_receipt};
mod exchange;
pub use exchange::{PreparedPartitionTensorDelivery, PartitionCaptureTensorDeliveryError};
use reader::TensorReader;

/// Retained expected receipt coordinates, supplied by the shared partition
/// placement/admission. These scalar facts alone supply no native authority.
pub struct PartitionCaptureTensorReceipt<'a> {
    /// Exact retained execution and capture context.
    pub context: &'a PartitionCaptureContext,
    /// Exact immutable expected receipt-plan digest.
    pub identity: &'a str,
    /// Independently known sender in the selected transport.
    pub producer: usize,
    /// Actual admitted native source precision.
    pub dtype: TensorDtype,
    /// Exact prepaid native/host record charge expected from that producer.
    pub charged: CaptureUsage,
}
impl fmt::Debug for PartitionCaptureTensorReceipt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartitionCaptureTensorReceipt").field("producer", &self.producer).finish_non_exhaustive()
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("capture receipt source: {0}")]
    Source(&'static str),
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Host(#[from] CaptureRunHostError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("capture receipt destination is incomplete")]
    Incomplete(#[source] ScheduledCaptureTensorFailure),
    #[error("capture summary receipt is invalid")]
    Summary(#[source] CaptureSummaryFailure),
    #[error("capture histogram receipt is invalid")]
    Histogram(#[source] CaptureHistogramFailure),
    #[error("capture candidate receipt is invalid")]
    Candidates(#[source] CaptureCandidateFailure),
    #[error("capture token score receipt is invalid")]
    Scores(#[source] CaptureTokenScoreFailure),
}
/// Failed receipt retains the original spent claim's H and parser metadata.
/// Partial host values retire before this owner; no native completion or claim
/// retry can be inferred from a decoder error.
#[derive(Debug, thiserror::Error)]
#[error("partition capture receipt: {cause}")]
pub struct PartitionCaptureTensorDecodeError {
    #[source]
    cause: Cause,
    _histogram: Option<CaptureHistogram>,
    _vocabulary: Option<VocabularyFailure>,
    _custody: CaptureTensorCustody,
    _metadata: HostMetadataFunding,
}
impl<'a, 'c> CaptureTensorClaim<'a, 'c> {
    /// Fill this exact scheduled F32 destination from a completed canonical
    /// global-producer receipt. The existing serde parser validates JSON syntax;
    /// borrowed event checks validate every receipt field before publication.
    /// Native settlement and all-rank delivery agreement remain independent.
    pub fn decode_partition_receipt(
        self,
        bytes: &[u8],
        expected: PartitionCaptureTensorReceipt<'_>,
        funding: &HostMetadataFunding,
    ) -> Result<ClaimedCaptureTensor, PartitionCaptureTensorDecodeError> {
        self.decode_partition_receipt_combined(bytes,expected,funding,eredu_core::capture::PartitionCaptureCombination::Disjoint)
    }
    pub(in crate::working_memory::capture_run) fn decode_partition_receipt_combined(
        self,bytes:&[u8],expected:PartitionCaptureTensorReceipt<'_>,funding:&HostMetadataFunding,
        combination:eredu_core::capture::PartitionCaptureCombination,
    )->Result<ClaimedCaptureTensor,PartitionCaptureTensorDecodeError> {
        let custody = self.identity.custody.share_scheduled();
        let error = |cause|PartitionCaptureTensorDecodeError { cause, _histogram: None, _vocabulary: None, _custody: custody.share_scheduled(), _metadata: funding.clone() };
        let controls = [
            size_of::<Self>(), size_of::<PartitionCaptureTensorReceipt<'_>>(),
            size_of::<TensorReader<'_, 'a, 'c>>(), size_of::<ScheduledCaptureTensor<'a, 'c>>(),
            // Both source-derived arrays exist in this caller while their values
            // are moved/copied into the reader; do not count only the recipient.
            size_of::<[[usize; 32]; 2]>(),
            size_of::<(usize, usize, &CaptureTensorGeometry<'_>, &AdmittedCapturePlan)>(),
            size_of::<PartitionCaptureTensorDecodeError>(), size_of::<Cause>(),
            size_of::<Result<ClaimedCaptureTensor, PartitionCaptureTensorDecodeError>>(),
            size_of::<(&[u8], &HostMetadataFunding, bool)>(),
            size_of::<eredu_core::capture::PartitionCaptureCombination>(),
        ];
        let controls = controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(||error(Cause::Source("decoder controls overflow")))?;
        funding.reserve_metadata(controls).map_err(|cause|error(cause.into()))?;
        let geometry = self.geometry();
        if expected.context.capture_plan_identity != geometry.admission().identity()
            || expected.context.selection_index != geometry.selection_index()
            || expected.context.phase != geometry.phase()
            || expected.context.prediction != geometry.prediction()
            || expected.context.invocation.is_some()
            || expected.identity.is_empty()
            || !matches!(expected.dtype, TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16)
        { return Err(error(Cause::Source("receipt differs from its scheduled tensor claim"))); }
        let mut source_shape = [0usize; 32];
        let mut selected_shape = [0usize; 32];
        let source_rank = geometry.source_shape().len();
        source_shape[..source_rank].copy_from_slice(geometry.source_shape());
        for (axis, ((start, end), stride)) in geometry.starts().iter().zip(geometry.ends()).zip(geometry.strides()).enumerate() {
            selected_shape[axis] = usize::try_from((end-start).div_ceil(*stride))
                .map_err(|_|error(Cause::Source("selected shape overflow")))?;
        }
        let source = geometry.admission();
        let index = geometry.selection_index();
        funding.reserve_metadata(TensorReader::construction_control_bytes()
            .ok_or_else(||error(Cause::Source("receipt reader construction controls overflow")))?)
            .map_err(|cause|error(cause.into()))?;
        let plan = Plan::prepare(bytes).map_err(|cause|error(cause.into()))?;
        let parser = plan.requirements::<TensorReader<'_, 'a, 'c>>().map_err(|cause|error(cause.into()))?;
        funding.reserve_metadata(parser.required_bytes()).map_err(|cause|error(cause.into()))?;
        let mut destination = self.prepare().map_err(|cause|error(cause.into()))?;
        let mut reader = TensorReader::new(&mut destination, expected, source, index, source_shape, selected_shape, source_rank).combined(combination);
        let allocation=crate::working_memory::original_json_allocation::JsonAllocation::new(funding).map_err(|cause|error(cause.into()))?;
        let parsed = plan.parse(&mut reader,&allocation);
        if let Some(cause)=allocation.failure(){return Err(error(cause.into()));}
        let valid = reader.complete();
        let memory = reader.take_memory();
        drop(reader);
        parsed.map_err(|cause|error(cause.into()))?;
        if let Some(cause) = memory { return Err(error(cause.into())); }
        if !valid { return Err(error(Cause::Source("receipt identity, geometry, charge or payload differs"))); }
        destination.finish().map_err(|cause|error(Cause::Incomplete(cause.into_owned_error())))
    }
}

impl ScheduledCaptureStep<'_> {
    pub(crate) fn prepare_partition_evidence(&mut self, funding: &HostMetadataFunding)
        -> Result<(), CaptureRunHostError>
    { self.frame.prepare_partition_evidence(funding).map_err(Into::into) }
    pub(crate) fn record_partition_evidence(&mut self,
        evidence: crate::capture::partition::PreparedPartitionCaptureEvidence)
        -> Result<(), CaptureRunHostError>
    { self.frame.record_partition_evidence(evidence).map_err(Into::into) }
    #[cfg(test)]
    pub(crate) fn partition_evidence(&self) -> &[eredu_core::capture::PartitionCaptureEvidence] {
        self.frame.partition_evidence()
    }
}

/// Same canonical envelope/parser, with a closed fixed scalar destination.
/// The caller supplies a consumed original summary claim and performs its
/// existing reduction validation after every receipt field has matched.
pub(in crate::working_memory::capture_run) fn decode_summary_receipt(
    bytes: &[u8], expected: PartitionCaptureTensorReceipt<'_>,
    geometry: &CaptureSummaryGeometry<'_>, custody: &CaptureTensorCustody,
    funding: &HostMetadataFunding,
) -> Result<CaptureSummary, PartitionCaptureTensorDecodeError> {
    let error = |cause| PartitionCaptureTensorDecodeError { cause, _histogram: None, _vocabulary: None,
        _custody: custody.share_scheduled(), _metadata: funding.clone() };
    let controls = [size_of::<PartitionCaptureTensorReceipt<'_>>(),
        size_of::<TensorReader<'_, '_, '_>>(), size_of::<CaptureSummary>() * 2,
        size_of::<[[usize; 32]; 2]>(), size_of::<(usize, usize, &CaptureSummaryGeometry<'_>, &AdmittedCapturePlan)>(),
        size_of::<PartitionCaptureTensorDecodeError>(), size_of::<Cause>(),
        size_of::<Result<CaptureSummary, PartitionCaptureTensorDecodeError>>(),
        size_of::<(&[u8], &CaptureTensorCustody, &HostMetadataFunding, bool)>(),
    ];
    funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or_else(|| error(Cause::Source("summary decoder controls overflow")))?)
        .map_err(|cause| error(cause.into()))?;
    custody.validate().map_err(|cause| error(cause.into()))?;
    if expected.context.capture_plan_identity != geometry.admission().identity()
        || expected.context.selection_index != geometry.selection_index()
        || expected.context.phase != geometry.phase()
        || expected.context.prediction != geometry.prediction()
        || expected.context.invocation.is_some() || expected.identity.is_empty()
        || !matches!(expected.dtype, TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16)
    { return Err(error(Cause::Source("receipt differs from its scheduled summary claim"))); }
    let rank = geometry.source_shape().len();
    let mut source_shape = [0usize; 32];
    let mut selected_shape = [0usize; 32];
    source_shape[..rank].copy_from_slice(geometry.source_shape());
    selected_shape[..rank].copy_from_slice(geometry.shape());
    funding.reserve_metadata(TensorReader::construction_control_bytes()
        .ok_or_else(||error(Cause::Source("receipt reader construction controls overflow")))?)
        .map_err(|cause|error(cause.into()))?;
    let plan = Plan::prepare(bytes).map_err(|cause| error(cause.into()))?;
    let parser = plan.requirements::<TensorReader<'_, '_, '_>>().map_err(|cause| error(cause.into()))?;
    funding.reserve_metadata(parser.required_bytes()).map_err(|cause| error(cause.into()))?;
    let mut value = crate::capture::reduction::Summary::default().value();
    let mut reader = TensorReader::new_summary(&mut value, expected, geometry.admission(),
        geometry.selection_index(), source_shape, selected_shape, rank);
    let allocation=crate::working_memory::original_json_allocation::JsonAllocation::new(funding).map_err(|cause|error(cause.into()))?;
        let parsed = plan.parse(&mut reader,&allocation);
        if let Some(cause)=allocation.failure(){return Err(error(cause.into()));}
    let valid = reader.complete();
    let memory = reader.take_memory();
    drop(reader);
    parsed.map_err(|cause| error(cause.into()))?;
    if let Some(cause) = memory { return Err(error(cause.into())); }
    if !valid { return Err(error(Cause::Source("receipt identity, geometry, charge or summary differs"))); }
    Ok(value)
}
impl PartitionCaptureTensorDecodeError {
    pub(in crate::working_memory::capture_run) fn retaining_summary_failure(
        cause: CaptureSummaryFailure, custody: CaptureTensorCustody,
        funding: &HostMetadataFunding,
    ) -> Self { Self { cause: Cause::Summary(cause), _histogram: None, _vocabulary: None, _custody: custody, _metadata: funding.clone() } }
}

pub(in crate::working_memory::capture_run) fn decode_histogram_receipt(
    value: &mut CaptureHistogram, bytes: &[u8], expected: PartitionCaptureTensorReceipt<'_>,
    geometry: &CaptureHistogramGeometry<'_>, custody: &CaptureTensorCustody,
    funding: &HostMetadataFunding,
) -> Result<(), PartitionCaptureTensorDecodeError> {
    let error = |cause| PartitionCaptureTensorDecodeError { cause, _histogram: None, _vocabulary: None,
        _custody: custody.share_scheduled(), _metadata: funding.clone() };
    let controls = [size_of::<PartitionCaptureTensorReceipt<'_>>(),
        size_of::<TensorReader<'_, '_, '_>>(), size_of::<CaptureHistogram>() * 2,
        size_of::<[[usize; 32]; 2]>(), size_of::<(usize, usize, &CaptureHistogramGeometry<'_>, &AdmittedCapturePlan)>(),
        size_of::<PartitionCaptureTensorDecodeError>(), size_of::<Cause>(),
        size_of::<Result<(), PartitionCaptureTensorDecodeError>>(),
        size_of::<(&[u8], &CaptureTensorCustody, &HostMetadataFunding, bool)>(),
    ];
    funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or_else(|| error(Cause::Source("histogram decoder controls overflow")))?)
        .map_err(|cause| error(cause.into()))?;
    custody.validate().map_err(|cause| error(cause.into()))?;
    if expected.context.capture_plan_identity != geometry.admission().identity()
        || expected.context.selection_index != geometry.selection_index()
        || expected.context.phase != geometry.phase()
        || expected.context.prediction != geometry.prediction()
        || expected.context.invocation.is_some() || expected.identity.is_empty()
        || !matches!(expected.dtype, TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16)
    { return Err(error(Cause::Source("receipt differs from its scheduled histogram claim"))); }
    let rank = geometry.source_shape().len();
    let mut source_shape = [0usize; 32];
    let mut selected_shape = [0usize; 32];
    source_shape[..rank].copy_from_slice(geometry.source_shape());
    selected_shape[..rank].copy_from_slice(geometry.shape());
    funding.reserve_metadata(TensorReader::construction_control_bytes()
        .ok_or_else(||error(Cause::Source("receipt reader construction controls overflow")))?)
        .map_err(|cause|error(cause.into()))?;
    let plan = Plan::prepare(bytes).map_err(|cause| error(cause.into()))?;
    let parser = plan.requirements::<TensorReader<'_, '_, '_>>().map_err(|cause| error(cause.into()))?;
    funding.reserve_metadata(parser.required_bytes()).map_err(|cause| error(cause.into()))?;
    let mut reader = TensorReader::new_histogram(value, expected, geometry.admission(),
        geometry.selection_index(), source_shape, selected_shape, rank);
    let allocation=crate::working_memory::original_json_allocation::JsonAllocation::new(funding).map_err(|cause|error(cause.into()))?;
        let parsed = plan.parse(&mut reader,&allocation);
        if let Some(cause)=allocation.failure(){return Err(error(cause.into()));}
    let valid = reader.complete();
    let memory = reader.take_memory();
    drop(reader);
    parsed.map_err(|cause| error(cause.into()))?;
    if let Some(cause) = memory { return Err(error(cause.into())); }
    if !valid { return Err(error(Cause::Source("receipt identity, geometry, charge or histogram differs"))); }
    Ok(())
}
impl PartitionCaptureTensorDecodeError {
    pub(in crate::working_memory::capture_run) fn retaining_histogram_payload(mut self, value: CaptureHistogram) -> Self {
        self._histogram=Some(value);self
    }
    pub(in crate::working_memory::capture_run) fn retaining_histogram_failure(
        cause: CaptureHistogramFailure, custody: CaptureTensorCustody, funding: &HostMetadataFunding,
    ) -> Self { Self { cause: Cause::Histogram(cause), _histogram: None, _vocabulary: None, _custody: custody, _metadata: funding.clone() } }
}

pub(in crate::working_memory::capture_run) fn histogram_preparation_failure(
    cause: CaptureRunHostError, custody: &CaptureTensorCustody, funding: &HostMetadataFunding,
) -> PartitionCaptureTensorDecodeError {
    PartitionCaptureTensorDecodeError {cause:Cause::Host(cause),_histogram: None, _vocabulary: None,
        _custody:custody.share_scheduled(),_metadata:funding.clone()}
}

pub(in crate::working_memory::capture_run) fn prepare_histogram_decoder(
    custody: &CaptureTensorCustody, funding: &HostMetadataFunding,
) -> Result<(),PartitionCaptureTensorDecodeError> {
    let error=|cause|PartitionCaptureTensorDecodeError {cause,_histogram: None, _vocabulary: None,
        _custody:custody.share_scheduled(),_metadata:funding.clone()};
    let frames=[size_of::<CaptureHistogramClaim<'_, '_>>(),size_of::<ScheduledCaptureHistogram<'_, '_>>(),
        size_of::<PartitionCaptureTensorDecodeError>(),size_of::<Cause>(),
        size_of::<Result<ClaimedCaptureHistogram,PartitionCaptureTensorDecodeError>>(),
        size_of::<(&[u8],PartitionCaptureTensorReceipt<'_>,&HostMetadataFunding)>(),
        size_of::<CaptureTensorCustody>(),size_of::<(u64,u64,u64)>(),
        size_of::<(&CaptureTensorCustody,&HostMetadataFunding)>(),size_of::<Result<(),PartitionCaptureTensorDecodeError>>()];
    funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
        .ok_or_else(||error(Cause::Source("histogram claim decoder controls overflow")))?)
        .map_err(|cause|error(cause.into()))?;
    custody.validate().map_err(|cause|error(cause.into()))
}

#[derive(Debug)]
enum VocabularyFailure {Candidates(CaptureCandidateFailure),Scores(CaptureTokenScoreFailure)}
impl PartitionCaptureTensorDecodeError {
    pub(in crate::working_memory::capture_run) fn retaining_candidates(mut self,value:CaptureCandidateFailure)->Self {
        self._vocabulary=Some(VocabularyFailure::Candidates(value));self
    }
    pub(in crate::working_memory::capture_run) fn retaining_scores(mut self,value:CaptureTokenScoreFailure)->Self {
        self._vocabulary=Some(VocabularyFailure::Scores(value));self
    }
    pub(in crate::working_memory::capture_run) fn candidate_failure(value:CaptureCandidateFailure,custody:CaptureTensorCustody,funding:&HostMetadataFunding)->Self {
        Self {cause:Cause::Candidates(value),_histogram:None,_vocabulary:None,_custody:custody,_metadata:funding.clone()}
    }
    pub(in crate::working_memory::capture_run) fn score_failure(value:CaptureTokenScoreFailure,custody:CaptureTensorCustody,funding:&HostMetadataFunding)->Self {
        Self {cause:Cause::Scores(value),_histogram:None,_vocabulary:None,_custody:custody,_metadata:funding.clone()}
    }
}

mod fragments;
pub use fragments::{PartitionFragmentHostPlan, PreparedPartitionFragmentDestinations, PartitionFragmentDestination, NativePartitionFragmentDestination, PartitionFragmentValue, PartitionFragmentDestinationError, PreparedPartitionFragmentHostFunding, PartitionFragmentHostBindingError, PartitionFragmentHostPreparationError};
pub(in crate::working_memory) use fragments::FragmentHostPlan;

pub(crate) use fragments::{PartitionCaptureRankSource, PreparedPartitionFragmentDelivery, PartitionFragmentDelivered, PartitionFragmentDeliveryError};

pub(crate) use fragments::{PartitionLocalCaptureHook,PartitionCaptureHookContinuation,PartitionCaptureHookReturnError};

mod assembled;
pub(in crate::working_memory::capture_run) use assembled::invocation_assembly_control_bytes;
