//! Completed token inputs leave the same numerical compiler as owned tensors.
mod registered;
pub(crate) use registered::RegisteredTensorSource;
use super::*;
use crate::backend::{OriginalCopyEnvironment, runtime::cache::state::CompletedResidentSource};
use crate::composition::mlx::replicated_text::OriginalEmbeddedCachePreparation;
use eredu_architectures::speculative_execution::EmbeddedPredictionTensor;
use eredu_runtime::working_memory::{WorkingMemoryError, CompletedWorkspaceSourceAccount};
use eredu_architectures::speculative_execution::PreparedEmbeddedEvidence;
use eredu_nn::{Index, Tensor, workspace::{WorkspaceContext, WorkspaceMetadataError, WorkspaceTensor}};
use eredu_core::{HostPreparationAuthority, BackendFailure};
/// Native-local, array-free evidence keeps the view's own native operation
/// budget separate from the account-only HostPreparationAuthority.
pub(crate) struct CompletedTensorSource {
    pub(crate) operation: Option<(safemlx::OriginalBufferBudget, OriginalSpeculativeNumericalBudgetCustody)>,
    pub(crate) source: CompletedResidentSource,
}
struct PhaseCustody {
    _account: OriginalSpeculativeNumericalBudgetCustody,
    _funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct CloneFailure {
    #[source]
    cause: safemlx::PreparedArrayCloneCause,
    _funding: HostMetadataFunding,
}

fn invalid() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn packet(
    source: OriginalNumericalValue,
    sources: &OriginalSpeculativeNumericalSources,
    environment: &OriginalCopyEnvironment<'_>,
    preparation: &OriginalEmbeddedCachePreparation,
    prior: &[&CompletedResidentSource],
    parent: Option<&EmbeddedPredictionTensor<MlxTensor>>,
    registered: Option<&RegisteredTensorSource>,
) -> Result<EmbeddedPredictionTensor<MlxTensor>, Error> {
    packet_with_parents(source,sources,environment,environment.stream(),preparation,prior,[parent,None],registered)
}
fn packet_with_parents(
    mut source:OriginalNumericalValue,sources:&OriginalSpeculativeNumericalSources,
    environment:&OriginalCopyEnvironment<'_>,registered_stream:&Stream,preparation:&OriginalEmbeddedCachePreparation,
    prior:&[&CompletedResidentSource],parents:[Option<&EmbeddedPredictionTensor<MlxTensor>>;2],
    registered:Option<&RegisteredTensorSource>,
)->Result<EmbeddedPredictionTensor<MlxTensor>,Error>{
    sources.validate_environment(environment)?;
    let funding = sources.metadata_funding();
    let parts = [
        OriginalEmbeddedCachePreparation::tensor_handoff_control_bytes()
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
        size_of::<&Stream>(),
        size_of::<Value>(),
        size_of::<OriginalNumericalValue>(),
        size_of::<CompletedResidentSource>(),
        size_of::<CompletedTensorSource>(),
        size_of::<MlxTensor>(),
        size_of::<Result<EmbeddedPredictionTensor<MlxTensor>, Error>>(),
        size_of::<Result<EmbeddedPredictionTensor<MlxTensor>, eredu_core::BackendFailure>>(),
        size_of::<[&CompletedResidentSource; 0]>(),
        size_of::<Option<Value>>(),
        size_of::<PhaseCustody>(),
        safemlx::OriginalBufferBudget::inspection_control_bytes().ok_or_else(model::overflow)?,
        size_of::<RegisteredTensorSource>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<[Option<EmbeddedPredictionTensor<MlxTensor>>;2]>(),
        HostPreparationAuthority::retention_bytes::<PhaseCustody>()
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
    ];
    funding
        .reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(
                    HostMetadataFundingError::Overflow,
                ))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let value = source.value();
    let dtype_matches = match (value.meaning, value.array.dtype()) {
        (Meaning::TokenIds, safemlx::Dtype::Uint32 | safemlx::Dtype::Int32) => true,
        (Meaning::Capture, safemlx::Dtype::Float32 | safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16) => true,
        _ => false,
    };
    if !dtype_matches
        || !value
            .provenance
            .source()
            .belongs_to_request(sources.request())
        || source.1.is_some()
        || Rc::strong_count(source.0.as_ref().expect("live tensor")) != 1
    {
        return Err(invalid());
    }
    let Provenance::Numerical(custody) = &value.provenance else {
        return Err(invalid());
    };
    let budget=value.original_budget.as_ref().ok_or_else(invalid)?;
    let registered=match registered {
        Some(source) if match budget.inspect_array(&value.array) {
            Ok(None) | Err(safemlx::OriginalBufferCause::ForeignDomain)=>true,
            Ok(Some(_))=>false,
            Err(cause)=>return Err(sources.retain_startup_error(cause)),
        }=>{
            // The new view completed on environment; the retained copy proof
            // still names the actual source publication on registered_stream.
            source.validate(sources.request(),registered_stream,funding)?;
            source.validate_array(&value.array,funding)?;
            if parents[1].is_some(){return Err(invalid());}
            Some(source.with_view_budget(budget.clone(),custody.clone()))
        }
        _=>None,
    };
    let completed=if registered.is_none(){
        Some(CompletedResidentSource::capture_numerical_array_sources(
            |visit|visit(&value.array),prior,budget,custody,funding)?
            .with_completed_stream(environment.stream())?)
    } else {None};
    // Source entries name the full backing's account. The new view also owns
    // its particular numerical Graph/Q even when it creates no new backing.
    let transport = HostPreparationAuthority::retain(PhaseCustody {
        _account: custody.clone(), _funding: funding.clone(),
    });
    // No clone/adoption: the unique completed numerical owner transfers its
    // exact Array after source capture, then the source owns its original Q/H.
    let operation=(budget.clone(),custody.clone());
    let value = Rc::into_inner(source.0.take().expect("live tensor"))
        .expect("checked unique completed numerical tensor");
    let result=match registered {
        Some(registered)=>preparation.prepare_registered_tensor(MlxTensor::from_array(value.array),
            registered,sources,transport,parents[0].cloned()),
        None=>preparation.prepare_tensor_with_parents(MlxTensor::from_array(value.array),
            CompletedTensorSource {source:completed.expect("completed backing source"),operation:Some(operation)},
            sources,transport,parents.map(|parent|parent.cloned())),
    };
    result.map_err(Error::StorageSource)
}
pub(crate) fn token_ids(
    tokens: &[u32],
    sources: &OriginalSpeculativeNumericalSources,
    environment: &OriginalCopyEnvironment<'_>,
    preparation: &OriginalEmbeddedCachePreparation,
) -> Result<EmbeddedPredictionTensor<MlxTensor>, Error> {
    let result = (|| {
        preparation
            .validate_sources(sources)
            .map_err(Error::StorageSource)?;
        let (roots, mechanisms) = sources.numerical_prerequisites();
        match NumericalProducer::execute_token_ids(sources, environment, roots, mechanisms, tokens)?
        {
            NumericalOutput::Tensor(value) => packet(value, sources, environment, preparation, &[], None, None),
            _ => Err(invalid()),
        }
    })();
    result.map_err(|cause| sources.retain_error(cause))
}

fn borrow_tensor(
    source: &MlxTensor, completed: Option<&CompletedResidentSource>,
    registered: Option<&RegisteredTensorSource>, meaning: Meaning,
    sources: &OriginalSpeculativeNumericalSources, environment: &OriginalCopyEnvironment<'_>,
) -> Result<OriginalNumericalValue, Error> {
    sources.validate_environment(environment)?;
    let funding = sources.metadata_funding();
    let controls = [value_control_bytes().ok_or_else(model::overflow)?,
        safemlx::PreparedArrayClone::control_bytes().ok_or_else(model::overflow)?,
        Array::inspection_clone_handle_bytes(), size_of::<CloneFailure>(),
        BackendFailure::source_retention_peak_bytes::<CloneFailure>().ok_or_else(model::overflow)?,
        size_of::<Result<Array, safemlx::PreparedArrayCloneCause>>(),
        size_of::<CompletedWorkspaceSourceAccount>(), size_of::<Provenance>(),
        if completed.is_some(){CompletedResidentSource::array_source_control_bytes()
            .ok_or_else(model::overflow)?}else{0},
        size_of::<Result<OriginalNumericalValue, Error>>(),
    ];
    funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or_else(model::overflow)?).map_err(Error::WorkspacePlanning)?;
    let valid = match (meaning, source.as_array().dtype()) {
        (Meaning::TokenIds, safemlx::Dtype::Uint32 | safemlx::Dtype::Int32) => true,
        (Meaning::Logits, safemlx::Dtype::Float32) => true,
        (Meaning::Capture, safemlx::Dtype::Float32 | safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16) => true,
        _ => false,
    };
    if !valid { return Err(invalid()); }
    let (stream,provenance,budget)=match (completed,registered) {
        (Some(completed),None)=>{
            completed.validate_request_source(sources.request(),environment.stream(),funding)?;
            let (budget,account)=completed.array_source_account(source.as_array(),funding)?;
            let stream=model::prepare_account_stream(account.clone(),environment.stream(),funding)?;
            let provenance=match account {
                CompletedWorkspaceSourceAccount::Model(account)=>Provenance::Model(account.clone()),
                CompletedWorkspaceSourceAccount::Numerical(account)=>Provenance::Numerical(account.clone()),
                CompletedWorkspaceSourceAccount::Standalone(_)=>return Err(invalid()),
            };
            (stream,provenance,Some(budget.clone()))
        }
        (None,Some(registered))=>{
            registered.validate(sources.request(),environment.stream(),funding)?;
            registered.validate_array(source.as_array(),funding)?;
            let provenance=registered.provenance(sources)?;
            let stream=model::prepare_registered_stream(provenance.clone(),environment.stream(),funding)?;
            (stream,Provenance::Registered(provenance),None)
        }
        _=>return Err(invalid()),
    };
    let fail = |cause| Error::StorageSource(BackendFailure::from_error(CloneFailure {
        cause, _funding: funding.clone(),
    }));
    let mut slot = safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(&fail)?;
    let array = slot.fill_for_inspection(source.as_array()).map_err(fail)?;
    Ok(OriginalNumericalValue(Some(Rc::new(Value {
        array, original_budget: budget, stream: ValueStream::Embedded(stream),
        meaning, provenance, funding: funding.clone(), _copy: None, _snapshot_host: None,
        _readout_source: None,
    })), None))
}
/// Actual sequence-preserving Slice. The source packet, when supplied, remains
/// immutable and owns every earlier view's numerical custody through this call.
pub(crate) fn tensor_range(
    value: &MlxTensor, start: usize, end: usize, tokens: bool,
    evidence: Option<&PreparedEmbeddedEvidence>, parent: Option<&EmbeddedPredictionTensor<MlxTensor>>,
    sources: &OriginalSpeculativeNumericalSources, environment: &OriginalCopyEnvironment<'_>,
    preparation: &OriginalEmbeddedCachePreparation,
) -> Result<EmbeddedPredictionTensor<MlxTensor>, Error> {
    tensor_axis_range(value, 1, start, end, tokens, evidence, parent, sources, environment, preparation)
}
/// The axis is an explicit part of the same static numerical view operation.
pub(crate) fn tensor_axis_range(
    value: &MlxTensor, axis: u8, start: usize, end: usize, tokens: bool,
    evidence: Option<&PreparedEmbeddedEvidence>, parent: Option<&EmbeddedPredictionTensor<MlxTensor>>,
    sources: &OriginalSpeculativeNumericalSources, environment: &OriginalCopyEnvironment<'_>,
    preparation: &OriginalEmbeddedCachePreparation,
) -> Result<EmbeddedPredictionTensor<MlxTensor>, Error> {
    tensor_axis_range_in(value,axis,start,end,tokens,evidence,parent,sources,environment,preparation,None)
}
/// Selected-side view through the same static range and numerical compiler.
pub(crate) fn tensor_axis_range_at(value:&MlxTensor,axis:u8,start:usize,end:usize,tokens:bool,
    evidence:&PreparedEmbeddedEvidence,
    context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
    placement:eredu_core::speculative::SamplingPlacement,
)->Result<EmbeddedPredictionTensor<MlxTensor>,Error>{
    let (sources,environment)=context.original_numerical_for(placement).ok_or_else(invalid)?;
    let preparation=context.original_cache_preparation().ok_or_else(invalid)?;
    tensor_axis_range_in(value,axis,start,end,tokens,Some(evidence),None,sources,environment,preparation,
        Some((context,placement)))
}
fn tensor_axis_range_in(value:&MlxTensor,axis:u8,start:usize,end:usize,tokens:bool,
    evidence:Option<&PreparedEmbeddedEvidence>,parent:Option<&EmbeddedPredictionTensor<MlxTensor>>,
    sources:&OriginalSpeculativeNumericalSources,environment:&OriginalCopyEnvironment<'_>,
    preparation:&OriginalEmbeddedCachePreparation,
    inputs:Option<(crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
        eredu_core::speculative::SamplingPlacement)>,
)->Result<EmbeddedPredictionTensor<MlxTensor>,Error>{
    let result = (|| {
        let frames=[size_of_val(&inputs),
            size_of::<(&MlxTensor,u8,usize,usize,bool,Option<&PreparedEmbeddedEvidence>,
                Option<&EmbeddedPredictionTensor<MlxTensor>>,&OriginalSpeculativeNumericalSources,
                &OriginalCopyEnvironment<'_>,&OriginalEmbeddedCachePreparation)>(),
            size_of::<(&MlxTensor,u8,usize,usize,bool,&PreparedEmbeddedEvidence,
                crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
                eredu_core::speculative::SamplingPlacement)>(),
            size_of::<Result<EmbeddedPredictionTensor<MlxTensor>,Error>>(),
            size_of::<Result<&OriginalCopyEnvironment<'_>,Error>>()];
        sources.metadata_funding().reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(model::overflow)?).map_err(Error::WorkspacePlanning)?;
        preparation.validate_sources(sources).map_err(Error::StorageSource)?;
        if parent.is_some_and(|p| !std::ptr::eq(&**p, value)
            || p.evidence().zip(evidence).is_none_or(|(a,b)| !std::ptr::eq(a,b))) {
            return Err(invalid());
        }
        let completed=evidence.and_then(crate::composition::mlx::speculative::completed_tensor_source);
        let registered=evidence.and_then(|e|e.get::<RegisteredTensorSource>());
        if completed.is_none() && registered.is_none(){return Err(invalid());}
        let start = u32::try_from(start).map_err(|_| invalid())?;
        let end = u32::try_from(end).map_err(|_| invalid())?;
        if tokens && axis != 1 { return Err(invalid()); }
        let kind = if tokens { program::SpeculativeNumericalKind::TokenRange { start, end } }
            else if axis == 1 { program::SpeculativeNumericalKind::TensorRange { start, end } }
            else { program::SpeculativeNumericalKind::TensorAxisRange { axis, start, end } };
        let meaning = if tokens { Meaning::TokenIds } else { Meaning::Capture };
        let source_environment=match inputs {
            Some((context,placement))=>crate::composition::mlx::speculative::tensor_sources::input_array_environment(
                evidence.ok_or_else(invalid)?,value.as_array(),context,placement,sources.metadata_funding())?,
            None=>environment,
        };
        let source = borrow_tensor(value, completed, registered, meaning, sources, source_environment)?;
        let output=match inputs {
            Some((context,placement))=>NumericalProducer::execute_at(context,placement,kind,&source,None)?,
            None=>{
                let (roots,mechanisms)=sources.numerical_prerequisites();
                NumericalProducer::execute(sources,environment,roots,mechanisms,kind,&source,None)?
            },
        };
        match output {
            NumericalOutput::Tensor(value) => {
                let prior=completed.map(|source|[source]);
                packet_with_parents(value,sources,environment,source_environment.stream(),preparation,
                    prior.as_ref().map_or(&[],|sources|sources.as_slice()),[parent,None],registered)
            },
            _ => Err(invalid()),
        }
    })();
    result.map_err(|cause| sources.retain_error(cause))
}
/// First static range from an actual imported whole prepared-input copy.
/// Its signed or unsigned dtype and independent source-copy account survive.
pub(crate) fn registered_range(
    source:&OriginalNumericalValue,start:usize,end:usize,
    sources:&OriginalSpeculativeNumericalSources,environment:&OriginalCopyEnvironment<'_>,
    preparation:&OriginalEmbeddedCachePreparation,
)->Result<EmbeddedPredictionTensor<MlxTensor>,Error>{
    let result=(||{
        preparation.validate_sources(sources).map_err(Error::StorageSource)?;
        let registered = if matches!(&source.value().provenance, Provenance::Registered(_)) {
            Some(RegisteredTensorSource::from_value(source, sources, environment)?)
        } else { None };
        let completed = match &source.value().provenance {
            Provenance::Numerical(custody) => Some(CompletedResidentSource::capture_numerical_array_sources(
                |visit| visit(&source.value().array), &[],
                source.value().original_budget.as_ref().ok_or_else(invalid)?, custody, sources.metadata_funding())?
                .with_completed_stream(environment.stream())?),
            Provenance::Registered(_) => None,
            _ => return Err(invalid()),
        };
        let prior = completed.as_ref().map(|source| [source]);
        let start=u32::try_from(start).map_err(|_|invalid())?;
        let end=u32::try_from(end).map_err(|_|invalid())?;
        let (roots,mechanisms)=sources.numerical_prerequisites();
        let kind=program::SpeculativeNumericalKind::TokenRange{start,end};
        match NumericalProducer::execute(sources,environment,roots,mechanisms,kind,source,None)?{
            NumericalOutput::Tensor(value)=>packet(value,sources,environment,preparation,
                prior.as_ref().map_or(&[], |p| p.as_slice()),None,registered.as_ref()),
            _=>Err(invalid()),
        }
    })();
    result.map_err(|cause|sources.retain_error(cause))
}
pub(super) fn metadata_range(value: &WorkspaceTensor, start: u32, end: u32, context: &WorkspaceContext)
    -> Result<WorkspaceTensor, eredu_nn::Error> {
    metadata_axis_range(value, 1, start, end, context)
}
pub(super) fn metadata_axis_range(value: &WorkspaceTensor, axis: u8, start: u32, end: u32, context: &WorkspaceContext)
    -> Result<WorkspaceTensor, eredu_nn::Error> {
    let parts = [size_of::<(&WorkspaceTensor,u8,u32,u32,&WorkspaceContext)>(),
        size_of::<[Index; 4]>(), size_of::<Result<WorkspaceTensor,eredu_nn::Error>>()];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(WorkspaceMetadataError::Overflow)?)?;
    let start=i32::try_from(start).map_err(|_|WorkspaceMetadataError::Overflow)?;
    let end=i32::try_from(end).map_err(|_|WorkspaceMetadataError::Overflow)?;
    let rank=value.shape().len(); let axis=usize::from(axis);
    if !(2..=4).contains(&rank) || axis == 0 || axis >= rank {
        return Err(WorkspaceMetadataError::Overflow.into());
    }
    let mut indices=[Index::Full, Index::Full, Index::Full, Index::Full];
    indices[axis]=Index::Range(start,end);
    value.index(&indices[..rank],context)
}
pub(super) fn native_axis_range_control_bytes() -> Option<usize> {
    let parts=[size_of::<(&Array,u8,u32,u32,&safemlx::Stream)>(),
        size_of::<(i32,i32)>(), size_of::<(usize,u8)>(),
        size_of::<(std::ops::RangeFull,std::ops::Range<i32>,std::ops::RangeFull,std::ops::RangeFull)>(),
        size_of::<Result<Array,Error>>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
/// Same fixed static Slice worker for sequence and K/V views. All six possible
/// non-batch axes use native fixed-index tuples, with no dynamic index table.
pub(in crate::composition::mlx::speculative) fn native_axis_range(value: &Array, axis: u8, start: u32, end: u32, stream: &safemlx::Stream)
    -> Result<Array, Error> {
    use safemlx::ops::indexing::TryIndexOp;
    let start=i32::try_from(start).map_err(|_|invalid())?;
    let end=i32::try_from(end).map_err(|_|invalid())?;
    match (value.shape().len(),axis) {
        (2,1)=>value.try_index_device((..,start..end),stream).map_err(Error::from),
        (3,1)=>value.try_index_device((..,start..end,..),stream).map_err(Error::from),
        (3,2)=>value.try_index_device((..,..,start..end),stream).map_err(Error::from),
        (4,1)=>value.try_index_device((..,start..end,..,..),stream).map_err(Error::from),
        (4,2)=>value.try_index_device((..,..,start..end,..),stream).map_err(Error::from),
        (4,3)=>value.try_index_device((..,..,..,start..end),stream).map_err(Error::from),
        _=>Err(invalid()),
    }
}

/// Joins two authenticated completed capture packets through the same numerical
/// compiler. Neither input is copied or relabelled as a model-owned allocation.
pub(crate) fn tensor_concatenate(
    left:&EmbeddedPredictionTensor<MlxTensor>,right:&EmbeddedPredictionTensor<MlxTensor>,
    sources:&OriginalSpeculativeNumericalSources,environment:&OriginalCopyEnvironment<'_>,
    preparation:&OriginalEmbeddedCachePreparation,
)->Result<EmbeddedPredictionTensor<MlxTensor>,Error>{
    let result=(||{
        preparation.validate_sources(sources).map_err(Error::StorageSource)?;
        let funding=sources.metadata_funding();
        let controls=[size_of::<[Option<&CompletedResidentSource>;2]>(),
            size_of::<[&CompletedResidentSource;2]>(),size_of::<[Option<&RegisteredTensorSource>;2]>(),
            size_of::<[OriginalNumericalValue;2]>(),size_of::<[Option<&EmbeddedPredictionTensor<MlxTensor>>;2]>(),
            size_of::<Result<EmbeddedPredictionTensor<MlxTensor>,Error>>()];
        funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or_else(model::overflow)?).map_err(Error::WorkspacePlanning)?;
        let completed=[left.evidence(),right.evidence()].map(|evidence|
            evidence.and_then(crate::composition::mlx::speculative::completed_tensor_source));
        let registered=[left.evidence(),right.evidence()].map(|evidence|
            evidence.and_then(|evidence|evidence.get::<RegisteredTensorSource>()));
        let values=[borrow_tensor(left,completed[0],registered[0],Meaning::Capture,sources,environment)?,
            borrow_tensor(right,completed[1],registered[1],Meaning::Capture,sources,environment)?];
        let right_positions=right.as_array().shape().get(1).copied()
            .and_then(|value|u32::try_from(value).ok()).ok_or_else(invalid)?;
        let (roots,mechanisms)=sources.numerical_prerequisites();
        let kind=program::SpeculativeNumericalKind::TensorConcatenate{right_positions};
        let output=NumericalProducer::execute(sources,environment,roots,mechanisms,kind,&values[0],Some(&values[1]))?;
        let mut priors=[None,None];
        let mut count=0;
        for source in completed.into_iter().flatten(){priors[count]=Some(source);count+=1;}
        let prior=match priors { [Some(left),Some(right)]=>[left,right],
            [Some(left),None]=>[left,left],_=>{
                let NumericalOutput::Tensor(value)=output else{return Err(invalid());};
                return packet_with_parents(value,sources,environment,environment.stream(),preparation,&[],[Some(left),Some(right)],None);
            }};
        let NumericalOutput::Tensor(value)=output else{return Err(invalid());};
        packet_with_parents(value,sources,environment,environment.stream(),preparation,&prior[..count],[Some(left),Some(right)],None)
    })();
    result.map_err(|cause|sources.retain_error(cause))
}
pub(super) fn metadata_concatenate(left:&WorkspaceTensor,right:&WorkspaceTensor,context:&WorkspaceContext)
    ->Result<WorkspaceTensor,eredu_nn::Error>{
    let parts=[size_of::<[WorkspaceTensor;2]>(),size_of::<[&Array;2]>(),
        size_of::<Result<WorkspaceTensor,eredu_nn::Error>>()];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(WorkspaceMetadataError::Overflow)?)?;
    WorkspaceTensor::concatenate(&[left.clone(),right.clone()],1,context)
}

#[cfg(test)]
mod axis_range_tests {
    use super::*;
    #[test]
    fn key_value_range_selects_positions_independently_for_each_head() {
        let stream=safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
        let values=(0..56).map(|i| i as f32 * 0.25 + 0.125).collect::<Vec<_>>();
        let source=Array::from_slice(&values,&[1,2,7,4]);
        let range=native_axis_range(&source,2,1,6,&stream).unwrap();
        assert_eq!(range.shape(),&[1,2,5,4]);
        let expected=values[4..24].iter().chain(&values[32..52]).copied().collect::<Vec<_>>();
        let contiguous=range.contiguous(false,&stream).unwrap();
        assert_eq!(contiguous.evaluated().unwrap().as_slice::<f32>(),expected);
        assert!(native_axis_range(&source,0,0,1,&stream).is_err());
    }
}

/// Borrow an actual completed logits tensor without reclassifying a registered
/// copy as a model allocation. The proof keeps its native copy/view ownership.
pub(super) fn borrow_logits(value:&MlxTensor,evidence:&PreparedEmbeddedEvidence,
    context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
    placement:eredu_core::speculative::SamplingPlacement,
)->Result<OriginalNumericalValue,Error>{
    let (sources,_)=context.original_numerical_for(placement).ok_or_else(invalid)?;
    let funding=sources.metadata_funding();
    let frames=[size_of::<(&MlxTensor,&PreparedEmbeddedEvidence,
            crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
            eredu_core::speculative::SamplingPlacement)>(),
        size_of::<OriginalNumericalValue>(),size_of::<Result<OriginalNumericalValue,Error>>(),
        size_of::<PreparedEmbeddedEvidence>(),size_of::<Option<&RegisteredTensorSource>>(),
        size_of::<Option<&CompletedResidentSource>>(),size_of::<RetainedValues>(),
        size_of::<Result<&OriginalCopyEnvironment<'_>,Error>>()];
    funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
        .ok_or_else(model::overflow)?).map_err(Error::WorkspacePlanning)?;
    let array=value.as_array();
    if !(2..=3).contains(&array.ndim()) || array.dim(0)!=1
        || array.shape().iter().any(|&extent|extent<=0) || array.dtype()!=safemlx::Dtype::Float32 {
        return Err(invalid());
    }
    let origin=crate::composition::mlx::speculative::tensor_sources::input_array_environment(
        evidence,array,context,placement,funding)?;
    let registered=crate::composition::mlx::speculative::tensor_sources::registered_source_for_array(evidence,array,funding)?;
    let completed=if registered.is_none(){
        crate::composition::mlx::speculative::tensor_sources::completed_tensor_source(evidence)
    }else{None};
    let mut result=borrow_tensor(value,completed,registered,Meaning::Logits,sources,origin)?;
    let owner=Rc::get_mut(result.0.as_mut().expect("new completed logits owner"))
        .expect("unpublished completed logits owner");
    debug_assert!(owner._readout_source.is_none());
    owner._readout_source=Some(RetainedValues::Evidence(evidence.clone()));
    Ok(result)
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod cpu_concat_tests;
