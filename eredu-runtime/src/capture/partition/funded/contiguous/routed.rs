//! Original sparse source descriptors feed the shared pending receipt owner.
use super::*;
use crate::working_memory::PreparedPartitionFragmentHostFunding;
use eredu_core::capture::{PartitionRoutedUnitCaptureLayout,PartitionRoutedUnitCaptureRequest};

/// Actual local source, independently of whether this rank exports a fragment.
#[derive(Debug)]
pub struct PartitionCaptureRoutedLocalSource<'a> {
    pub producer:usize,
    /// Whole-frame source facts; canonical prefill invocation windows are
    /// derived later from the same retained inference geometry.
    pub layout:PartitionRoutedUnitCaptureLayout<'a>,
    /// Actual cold grouped-result scalar, never inferred from its input tensor.
    pub dtype:Option<TensorDtype>,
}
#[derive(Debug)]
pub(crate) struct LocalSource {
    pub(crate) producer:usize,pub(crate) ownership:RoutedUnitCaptureOwnership,
    pub(crate) source_tokens:u64,pub(crate) dtype:Option<TensorDtype>,
}
#[derive(Debug)]
struct Producer {rank:usize,ownership:RoutedUnitCaptureOwnership}
#[derive(Debug)]
struct Native {producer:usize,fragment:usize,slice:ResolvedCaptureSlice,estimate:PartitionCaptureNativeEstimate}
/// Paid exact ownership and native fragment estimates, before the ordinary
/// source vote supplies peer scalar witnesses and original Host binding.
#[derive(Debug)]
pub struct PreparedPartitionRoutedSource {
    index:usize,coordinate:Coordinate,bank:RoutedUnitGeometry,source_tokens:u64,
    producers:Vec<Producer>,native:Vec<Native>,local:Option<LocalSource>,
    source:SharedCapturePlan,metadata:HostMetadataFunding,
}
impl PreparedPartitionRoutedSource {
    #[allow(clippy::too_many_arguments)]
    pub fn new_local_prefill(source:&SharedCapturePlan,index:usize,
        producers:&[PartitionCaptureRoutedProducerSource<'_>],native:&[PartitionCaptureRoutedFragmentGeometry<'_>],
        local:Option<PartitionCaptureRoutedLocalSource<'_>>,inference:InferenceGeometry,metadata:&HostMetadataFunding)
        ->Result<Self,PartitionCaptureProgramError> {
        Self::new(source,index,producers,native,local,Coordinate::Prefill(inference),metadata)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new_local_decode(source:&SharedCapturePlan,index:usize,
        producers:&[PartitionCaptureRoutedProducerSource<'_>],native:&[PartitionCaptureRoutedFragmentGeometry<'_>],
        local:Option<PartitionCaptureRoutedLocalSource<'_>>,prediction:u64,metadata:&HostMetadataFunding)
        ->Result<Self,PartitionCaptureProgramError> {
        Self::new(source,index,producers,native,local,Coordinate::Decode(prediction),metadata)
    }
    fn new(source:&SharedCapturePlan,index:usize,
        producers:&[PartitionCaptureRoutedProducerSource<'_>],native:&[PartitionCaptureRoutedFragmentGeometry<'_>],
        local:Option<PartitionCaptureRoutedLocalSource<'_>>,coordinate:Coordinate,metadata:&HostMetadataFunding)
        ->Result<Self,PartitionCaptureProgramError> {
        let error=|cause|PartitionCaptureProgramError{cause,_source:source.clone(),_metadata:metadata.clone()};
        metadata.reserve_metadata(Self::control_bytes().ok_or_else(||error(Cause::Source("sparse source controls overflow")))?)
            .map_err(|e|error(e.into()))?;
        let selection=source.admission().plan().selections.get(index).ok_or_else(||error(Cause::Source("sparse selection is absent")))?;
        if producers.is_empty() || !coordinate.accepts(selection) || !matches!(selection.transform,CaptureTransform::RoutedUnits) {
            return Err(error(Cause::Source("sparse selection or coordinate differs")));
        }
        let (phase,prediction)=match coordinate {Coordinate::Prefill(_)=>(CapturePhase::Prefill,0),Coordinate::Decode(n)=>(CapturePhase::Decode,n),Coordinate::Invocation(phase,prediction)=>(phase,prediction)};
        let geometry=eredu_core::capture::CaptureRoutedUnitsGeometry::prepare(source.admission(),index,phase,prediction,None)
            .map_err(|e|PartitionCaptureProgramError::local(e,source.clone(),metadata.clone()))?;
        let bank=geometry.bank();let source_tokens=geometry.source_shape()[0] as u64;
        if let Coordinate::Prefill(inference)=coordinate {
            eredu_core::capture::CaptureRoutedPrefillPlan::prepare(source.admission(),index,inference)
                .map_err(|e|PartitionCaptureProgramError::local(e,source.clone(),metadata.clone()))?;
        }
        let mut owned=metadata.metadata_vec(producers.len()).map_err(|e|error(e.into()))?;
        for producer in producers {
            producer.ownership.validate(bank).map_err(|e|error(e.into()))?;
            if owned.last().is_some_and(|previous:&Producer|previous.rank>=producer.rank) {
                return Err(error(Cause::Source("sparse producers are not canonical")));
            }
            owned.push(Producer{rank:producer.rank,ownership:copy_ownership(producer.ownership,metadata).map_err(error)?});
        }
        let mut rows=metadata.metadata_vec(native.len()).map_err(|e|error(e.into()))?;
        for row in native {
            let owner=owned.iter().find(|p|p.rank==row.producer).ok_or_else(||error(Cause::Source("sparse native producer is absent")))?;
            if row.request.geometry!=bank || row.request.source_tokens!=source_tokens || row.request.ownership!=&owner.ownership
                || row.estimate.generated_creation_bytes!=0
                || rows.last().is_some_and(|previous:&Native|(previous.producer,previous.fragment)>=(row.producer,row.fragment)) {
                return Err(error(Cause::Source("sparse native source differs")));
            }
            row.request.validate().map_err(|e|error(e.into()))?;
            let copy=|values:&[u64]|->Result<Vec<u64>,eredu_nn::Error>{let mut out=metadata.metadata_vec(values.len())?;out.extend_from_slice(values);Ok(out)};
            rows.push(Native{producer:row.producer,fragment:row.fragment,estimate:row.estimate,slice:ResolvedCaptureSlice{
                starts:copy(&row.request.slice.starts).map_err(|e|error(e.into()))?,ends:copy(&row.request.slice.ends).map_err(|e|error(e.into()))?,
                strides:copy(&row.request.slice.strides).map_err(|e|error(e.into()))?,shape:copy(&row.request.slice.shape).map_err(|e|error(e.into()))?}});
        }
        let local=local.map(|local|{
            local.layout.validate().map_err(|e|error(e.into()))?;
            if local.layout.geometry!=bank || local.layout.source_tokens!=source_tokens
                || !matches!(local.dtype,None|Some(TensorDtype::F16|TensorDtype::Bf16|TensorDtype::F32))
                || owned.iter().find(|p|p.rank==local.producer).is_some_and(|p|p.ownership!=*local.layout.ownership) {
                return Err(error(Cause::Source("actual sparse local source differs")));
            }
            Ok(LocalSource{producer:local.producer,source_tokens,dtype:local.dtype,
                ownership:copy_ownership(local.layout.ownership,metadata).map_err(error)?})
        }).transpose()?;
        Ok(Self{index,coordinate,bank,source_tokens,producers:owned,native:rows,local,source:source.clone(),metadata:metadata.clone()})
    }
    pub(in super::super) fn matches_program(&self,source:&SharedCapturePlan,index:usize)->bool {self.source.same_storage(source)&&self.index==index}
    pub(in super::super) fn matches_context(&self,context:&PartitionCaptureContext)->bool {self.coordinate.matches(context)&&context.invocation.is_none()}
    fn error(&self,cause:Cause)->PartitionCaptureProgramError {PartitionCaptureProgramError{cause,_source:self.source.clone(),_metadata:self.metadata.clone()}}
    pub(in super::super) fn prepare_selected_with_host<'t,T:PartitionCaptureTransport>(self,transport:&'t T,
        context:&PartitionCaptureContext,epoch:DistributedCommitEpoch,host:PreparedPartitionFragmentHostFunding,
        limits:PartitionCaptureReceiptLimits,ledger:&mut CaptureLedger)
        ->Result<PreparedPartitionContiguousRow<'t,T>,PreparationFailure>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        let source=self.source.clone();let metadata=self.metadata.clone();let mut host=Some(host);let mut descriptor=None;
        let result=(||{
            metadata.reserve_metadata(PreparedPartitionContiguousRow::<T>::control_bytes()
                .and_then(|n|n.checked_add(Self::control_bytes()?)).ok_or_else(||self.error(Cause::Source("sparse issuance controls overflow")))?)
                .map_err(|e|self.error(e.into()))?;
            let usage=self.native.iter().try_fold(CaptureUsage::default(),|n,row|n.checked_add(row.estimate.capture)).map_err(|e|self.error(e.into()))?;
            let first=self.producers.first().ok_or_else(||self.error(Cause::Source("sparse producers are absent")))?.rank;
            descriptor=Some((first,usage));
            self.prepare_inner(transport,context,epoch,&mut host,limits,ledger,usage)
        })();
        result.map_err(|cause|{
            let reason=match &cause.cause {Cause::Receipt(e)=>e.limit_skip(),Cause::FragmentAllowance(e)=>e.limit_skip(),_=>None};
            let cause=if let Some(host)=host.take(){PartitionCaptureProgramError::local(HeldPreparation{cause,_host:host},source,metadata)}else{cause};
            PreparationFailure{reason,descriptor,cause}
        })
    }
    fn prepare_inner<'t,T:PartitionCaptureTransport>(self,transport:&'t T,context:&PartitionCaptureContext,
        epoch:DistributedCommitEpoch,host:&mut Option<PreparedPartitionFragmentHostFunding>,limits:PartitionCaptureReceiptLimits,
        ledger:&mut CaptureLedger,usage:CaptureUsage)->Result<PreparedPartitionContiguousRow<'t,T>,PartitionCaptureProgramError>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        if !self.matches_context(context) || context.selection_index!=self.index || context.forward_epoch!=epoch.value()
            || context.capture_plan_identity!=self.source.admission().identity()
            || self.local.as_ref().is_some_and(|local|local.producer!=transport.capture_rank()) {
            return Err(self.error(Cause::Source("sparse forward source binding differs")));
        }
        let mut producers=self.metadata.metadata_vec(self.producers.len()).map_err(|e|self.error(e.into()))?;
        for row in &self.producers {producers.push(PartitionCaptureRoutedProducerSource{rank:row.rank,ownership:&row.ownership});}
        let mut receipt=PartitionCaptureReceiptPlan::new_routed_shared_funded(&self.source,context,&producers,
            transport.participant_count(),limits,&self.metadata,ledger).map_err(|e|self.error(e.into()))?;
        let mut rows=self.metadata.metadata_vec(self.native.len()).map_err(|e|self.error(e.into()))?;
        for row in &self.native {
            let ownership=&self.producers.iter().find(|p|p.rank==row.producer).expect("retained sparse producer").ownership;
            rows.push(PartitionCaptureRoutedFragmentGeometry{producer:row.producer,fragment:row.fragment,estimate:row.estimate,
                request:PartitionRoutedUnitCaptureRequest{geometry:self.bank,source_tokens:self.source_tokens,ownership,slice:&row.slice}});
        }
        let allowance=PreparedPartitionFragmentSourceAllowance::prepare_routed(transport,&mut receipt,&rows,&self.metadata,ledger)
            .map_err(|e|self.error(e.into()))?;
        let local=transport.capture_rank();let active=receipt.producer(local).is_some_and(|p|!p.fragments().is_empty());
        if active&&self.local.is_none(){return Err(self.error(Cause::Source("sparse producer has no actual local source")));}
        let present=self.local.is_some();let local_dtype=self.local.as_ref().and_then(|local|local.dtype.clone());
        drop(rows);drop(producers);
        let host=host.take().ok_or_else(||self.error(Cause::Source("sparse source has no original Host")))?;
        Ok(PreparedPartitionContiguousRow{transport,delivery:None,continuation:None,
            pending:Some(voted::Pending{receipt,allowance,host}),local_source:None,routed:true,routed_source:self.local,
            active,present,local,local_dtype,next:0,last_epoch:None,coordinate:self.coordinate,usage,source:self.source,metadata:self.metadata})
    }
    /// Exact descriptive constructor/copy frames; owned vectors charge their
    /// actual requested storage separately through the original metadata owner.
    pub fn control_bytes()->Option<usize> {
        let parts=[size_of::<Self>()*2,size_of::<Producer>()*2,size_of::<Native>()*2,size_of::<LocalSource>()*2,
            size_of::<Option<LocalSource>>(),size_of::<PartitionCaptureRoutedLocalSource<'_>>(),
            size_of::<PartitionCaptureRoutedProducerSource<'_>>(),size_of::<PartitionCaptureRoutedFragmentGeometry<'_>>(),
            size_of::<PartitionRoutedUnitCaptureRequest<'_>>(),size_of::<PartitionRoutedUnitCaptureLayout<'_>>(),
            size_of::<Result<Self,PartitionCaptureProgramError>>(),size_of::<PartitionCaptureProgramError>(),
            size_of::<Vec<Producer>>(),size_of::<Vec<Native>>(),size_of::<Vec<PartitionCaptureRoutedProducerSource<'_>>>(),
            size_of::<Vec<PartitionCaptureRoutedFragmentGeometry<'_>>>(),size_of::<ResolvedCaptureSlice>(),
            size_of::<RoutedUnitCaptureOwnership>(),size_of::<Coordinate>(),size_of::<CaptureUsage>(),
            size_of::<Option<(usize,CaptureUsage)>>(),size_of::<Option<PreparedPartitionFragmentHostFunding>>(),
            size_of::<std::slice::Iter<'_,Producer>>(),size_of::<std::slice::Iter<'_,Native>>(),
            size_of::<(&SharedCapturePlan,usize,&[PartitionCaptureRoutedProducerSource<'_>],&[PartitionCaptureRoutedFragmentGeometry<'_>],
                Option<PartitionCaptureRoutedLocalSource<'_>>,Coordinate,&HostMetadataFunding)>(),
            eredu_core::capture::CaptureRoutedUnitsGeometry::partition_control_bytes()?,
            eredu_core::component::ComponentCoordinateMap::copy_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
fn copy_ownership(source:&RoutedUnitCaptureOwnership,metadata:&HostMetadataFunding)->Result<RoutedUnitCaptureOwnership,Cause> {
    fn copy(source:&eredu_core::component::ComponentCoordinateMap,metadata:&HostMetadataFunding)
        ->Result<eredu_core::component::ComponentCoordinateMap,Cause> {
        source.try_clone_with_funding(metadata).map_err(Into::into)
    }
    Ok(RoutedUnitCaptureOwnership{coordinates:eredu_core::component::RoutedComponentCoordinateMap::new(
        copy(source.coordinates.experts(),metadata)?,copy(source.coordinates.units(),metadata)?),
        source_peer:source.source_peer,source_peers:source.source_peers})
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
