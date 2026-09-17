//! The ordinary canonical reader fills existing fixed vocabulary destinations.
use super::*;

#[derive(Debug)]
pub(in crate::working_memory::capture_run) enum VocabularyDestination<'r,'a,'c> {
    Candidates(&'r mut ScheduledCaptureCandidates<'a,'c>),
    Scores(&'r mut ScheduledCaptureTokenScores<'a,'c>),
}
impl<'a> VocabularyDestination<'_,'a,'_> {
    fn source(&self)->(&'a AdmittedCapturePlan,usize,CapturePhase,u64,[usize;3]) {
        match self {Self::Candidates(value)=>value.partition_source(),Self::Scores(value)=>value.partition_source()}
    }
}
pub(in crate::working_memory::capture_run) fn prepare_vocabulary_decoder(custody:&CaptureTensorCustody,
    funding:&WorkspaceMetadataFunding)->Result<(),PartitionCaptureTensorDecodeError> {
    let error=|cause|PartitionCaptureTensorDecodeError {cause,_histogram:None,_vocabulary:None,
        _custody:custody.share_scheduled(),_metadata:funding.clone()};
    let parts=[
        crate::capture::partition::vocabulary_validation_control_bytes().ok_or_else(||error(Cause::Source("vocabulary validation controls overflow")))?,
        size_of::<CaptureCandidateClaim<'_,'_>>().max(size_of::<CaptureTokenScoreClaim<'_,'_>>()),
        size_of::<ScheduledCaptureCandidates<'_,'_>>().max(size_of::<ScheduledCaptureTokenScores<'_,'_>>()),
        size_of::<Result<ClaimedCaptureCandidates,PartitionCaptureTensorDecodeError>>().max(size_of::<Result<ClaimedCaptureTokenScores,PartitionCaptureTensorDecodeError>>()),
        size_of::<PartitionCaptureTensorDecodeError>()*2,size_of::<Cause>(),size_of::<VocabularyFailure>(),
        size_of::<VocabularyDestination<'_,'_,'_>>(),size_of::<CaptureTensorCustody>(),
        size_of::<(&[u8],PartitionCaptureTensorReceipt<'_>,&WorkspaceMetadataFunding)>(),
        size_of::<(Option<CandidateDomain>,f64,[usize;3],[u64;3],&AdmittedCapturePlan,usize)>(),
        size_of::<(&CaptureTensorCustody,&WorkspaceMetadataFunding)>(),
    ];
    funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or_else(||error(Cause::Source("vocabulary claim controls overflow")))?).map_err(|cause|error(cause.into()))?;
    custody.validate().map_err(|cause|error(cause.into()))
}
pub(in crate::working_memory::capture_run) fn decode_vocabulary_receipt<'a>(
    output:VocabularyDestination<'_,'a,'_>,bytes:&[u8],expected:PartitionCaptureTensorReceipt<'_>,
    custody:&CaptureTensorCustody,funding:&WorkspaceMetadataFunding,
)->Result<(Option<CandidateDomain>,f64),PartitionCaptureTensorDecodeError> {
    let error=|cause|PartitionCaptureTensorDecodeError {cause,_histogram:None,_vocabulary:None,
        _custody:custody.share_scheduled(),_metadata:funding.clone()};
    let controls=[TensorReader::construction_control_bytes()
        .ok_or_else(||error(Cause::Source("receipt reader construction controls overflow")))?,size_of::<TensorReader<'_,'_,'_>>(),size_of::<VocabularyDestination<'_,'_,'_>>(),
        size_of::<PartitionCaptureTensorReceipt<'_>>(),size_of::<Cause>(),size_of::<PartitionCaptureTensorDecodeError>(),
        size_of::<(&AdmittedCapturePlan,usize,CapturePhase,u64,[usize;3])>(),
        size_of::<[[usize;32];2]>(),size_of::<(Option<CandidateDomain>,f64)>(),
        size_of::<Result<(Option<CandidateDomain>,f64),PartitionCaptureTensorDecodeError>>(),
        size_of::<(&[u8],&CaptureTensorCustody,&WorkspaceMetadataFunding,bool)>(),
    ];
    funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
        .ok_or_else(||error(Cause::Source("vocabulary decoder controls overflow")))?).map_err(|cause|error(cause.into()))?;
    custody.validate().map_err(|cause|error(cause.into()))?;
    let (source,index,phase,prediction,shape)=output.source();
    if expected.context.capture_plan_identity!=source.identity()||expected.context.selection_index!=index
        ||expected.context.phase!=phase||expected.context.prediction!=prediction||expected.context.invocation.is_some()
        ||expected.identity.is_empty()||!matches!(expected.dtype,TensorDtype::F32|TensorDtype::F16|TensorDtype::Bf16) {
        return Err(error(Cause::Source("receipt differs from its scheduled vocabulary claim")));
    }
    let plan=Plan::prepare(bytes).map_err(|cause|error(cause.into()))?;
    let parser=plan.requirements::<TensorReader<'_,'_,'_>>().map_err(|cause|error(cause.into()))?;
    funding.reserve_metadata(parser.required_bytes()).map_err(|cause|error(cause.into()))?;
    let mut reader=match output {
        VocabularyDestination::Candidates(output)=>TensorReader::new_candidates(output,expected,source,index,shape),
        VocabularyDestination::Scores(output)=>TensorReader::new_scores(output,expected,source,index,shape),
    };
    let parsed=plan.parse(&mut reader);let valid=reader.complete();let memory=reader.take_memory();
    let metadata=reader.vocabulary_metadata();drop(reader);
    parsed.map_err(|cause|error(cause.into()))?;
    if let Some(cause)=memory {return Err(error(cause.into()));}
    if !valid {return Err(error(Cause::Source("receipt identity, shape, charge or vocabulary payload differs")));}
    Ok(metadata)
}

/// Branch-local caller frames before the original claim is issued. The typed
/// decoder separately prices its destination and reader, before allocation.
pub(super) fn exchange_control_bytes(transform: &CaptureTransform) -> Option<usize> {
    let parts = match transform {
        CaptureTransform::TopCandidates { .. } => [
            size_of::<CaptureCandidateClaim<'_, '_>>(),
            size_of::<Result<CaptureCandidateClaim<'_, '_>, CaptureRunHostError>>(),
            size_of::<CaptureCandidateHostPlan<'_>>(),
            size_of::<Result<CaptureCandidateHostPlan<'_>, WorkingMemoryError>>(),
            size_of::<CaptureCandidateGeometry<'_>>(),
            size_of::<Result<CaptureCandidateGeometry<'_>, CaptureTensorGeometryError>>(),
            size_of::<ClaimedCaptureCandidates>(),
            size_of::<Result<ClaimedCaptureCandidates, PartitionCaptureTensorDecodeError>>(),
        ],
        CaptureTransform::TokenScores { .. } => [
            size_of::<CaptureTokenScoreClaim<'_, '_>>(),
            size_of::<Result<CaptureTokenScoreClaim<'_, '_>, CaptureRunHostError>>(),
            size_of::<CaptureTokenScoreHostPlan<'_>>(),
            size_of::<Result<CaptureTokenScoreHostPlan<'_>, WorkingMemoryError>>(),
            size_of::<CaptureTokenScoreGeometry<'_>>(),
            size_of::<Result<CaptureTokenScoreGeometry<'_>, CaptureTensorGeometryError>>(),
            size_of::<ClaimedCaptureTokenScores>(),
            size_of::<Result<ClaimedCaptureTokenScores, PartitionCaptureTensorDecodeError>>(),
        ],
        _ => return Some(0),
    };
    parts.into_iter().try_fold(size_of_val(&parts).checked_add(size_of::<(
        &mut ScheduledCaptureStep<'_>, usize, bool, Option<usize>,
        Result<Option<usize>, CaptureRunHostError>, eredu_core::InferenceGeometry,
    )>())?, usize::checked_add)
}
