//! Original Host held while the shared Source vote supplies remote precision.
use super::*;
use crate::working_memory::PreparedPartitionFragmentHostFunding;

/// An actual physical local hook source, including a nonexporting replica.
/// Absence is represented by None at construction, not a fabricated scalar.
#[derive(Debug)]
pub struct PartitionCaptureLocalSource<'a> {
    /// Actual world rank that observed this source.
    pub producer:usize,
    /// Full local prompt shape, before canonical chunk substitution.
    pub shape:&'a [u64],
    /// Actual native source scalar from the cold observation.
    pub dtype:TensorDtype,
}
#[derive(Debug)]
pub(super) struct LocalSource {pub(super) producer:usize,pub(super) shape:Vec<u64>,pub(super) dtype:TensorDtype}
impl LocalSource {
    pub(super) fn copy(source:PartitionCaptureLocalSource<'_>,
        metadata:&HostMetadataFunding)->Result<Self,Cause> {
        if source.shape.is_empty()||source.shape.len()>32||!matches!(source.dtype,TensorDtype::F16|TensorDtype::F32|TensorDtype::Bf16) {
            return Err(Cause::Source("local physical source differs"));
        }
        let mut shape=metadata.metadata_vec(source.shape.len())?;shape.extend_from_slice(source.shape);
        Ok(Self{producer:source.producer,shape,dtype:source.dtype})
    }
}
pub(super) enum Sources<'s,'v> {
    Typed(&'s [PartitionCaptureFragmentSource<'v>]),Geometry(&'s [PartitionCaptureFragmentGeometry<'v>]),
}
pub(super) struct Source<'a>{pub(super) producer:usize,pub(super) fragment:usize,
    pub(super) local_shape:&'a [u64],pub(super) local_slice:&'a ResolvedCaptureSlice,
    pub(super) transform:&'a CaptureTransform,pub(super) dtype:Option<&'a TensorDtype>,pub(super) estimate:PartitionCaptureNativeEstimate}
impl Sources<'_,'_> {
    pub(super) fn typed(&self)->bool{matches!(self,Self::Typed(_))}
    pub(super) fn len(&self)->usize{match self{Self::Typed(rows)=>rows.len(),Self::Geometry(rows)=>rows.len()}}
    pub(super) fn get(&self,index:usize)->Option<Source<'_>>{match self{
        Self::Typed(rows)=>rows.get(index).map(|row|Source{producer:row.producer,fragment:row.fragment,local_shape:row.local_shape,
            local_slice:row.local_slice,transform:row.transform,dtype:Some(&row.dtype),estimate:row.estimate}),
        Self::Geometry(rows)=>rows.get(index).map(|row|Source{producer:row.producer,fragment:row.fragment,local_shape:row.local_shape,
            local_slice:row.local_slice,transform:row.transform,dtype:None,estimate:row.estimate}),
    }}
}
#[derive(Debug)]
pub(super) struct Pending {
    pub(super) receipt:PartitionCaptureReceiptPlan,
    pub(super) allowance:PreparedPartitionFragmentSourceAllowance,
    pub(super) host:PreparedPartitionFragmentHostFunding,
}
#[derive(Debug,thiserror::Error)]
#[error("unbound partition source: {cause}")]
struct HeldHostError<E:std::error::Error+'static>{#[source] cause:E,_host:PreparedPartitionFragmentHostFunding}
#[derive(Debug,thiserror::Error)]
#[error("partition source vote: {cause}")]
struct HeldPendingError<E:std::error::Error+'static>{#[source] cause:E,_pending:Option<Pending>}

impl PreparedPartitionContiguousSource {
    pub(super) fn prepare_unvoted<'t,T:PartitionCaptureTransport>(self,transport:&'t T,
        mut receipt:PartitionCaptureReceiptPlan,funding:&mut Option<RowFunding<'_>>,ledger:&mut CaptureLedger,usage:CaptureUsage)
        ->Result<PreparedPartitionContiguousRow<'t,T>,PartitionCaptureProgramError>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        let result=(||->Result<(PreparedPartitionFragmentSourceAllowance,CaptureUsage),PartitionCaptureProgramError>{
            if self.local.as_ref().is_some_and(|local|local.producer!=transport.capture_rank()) {
                return Err(self.error(Cause::Source("physical source belongs to another rank")));
            }
            if let Some(local)=&self.local {
                let global=receipt.producers().next().ok_or_else(||self.error(Cause::Source("missing global source")))?.1.global_shape();
                if local.shape.len()!=global.len()||local.shape.iter().zip(global).enumerate()
                    .any(|(axis,(&n,&g))|if axis==self.axis{n>g}else{n!=g})
                    ||receipt.producer(local.producer).is_some_and(|p|p.local_shape()!=local.shape) {
                    return Err(self.error(Cause::Source("local physical source shape differs from its retained projection")));
                }
            }
            let selected=&self.source.admission().plan().selections[self.index].transform;
            let raw=super::super::super::sum::native_transform(selected);
            let transform=if self.combination==PartitionCaptureCombination::SumF64ToF32{&raw}else{selected};
            let mut rows=self.metadata.metadata_vec(self.native.len()).map_err(|e|self.error(e.into()))?;
            for row in &self.native {rows.push(PartitionCaptureFragmentGeometry{producer:row.producer,fragment:row.fragment,
                local_shape:&row.shape,local_slice:&row.slice,transform,estimate:row.estimate});}
            let allowance=PreparedPartitionFragmentSourceAllowance::prepare(transport,&mut receipt,&rows,&self.metadata,ledger)
                .map_err(|e|self.error(e.into()))?;
            Ok((allowance,usage))
        })();
        let (allowance,usage)=result?;
        let Some(RowFunding::Prepared(host))=funding.take() else {
            return Err(self.error(Cause::Source("unvoted source has no original Host")));
        };
        let local=transport.capture_rank();let active=receipt.producer(local).is_some_and(|p|!p.fragments().is_empty());
        let present=self.local.is_some();let local_dtype=self.local.as_ref().map(|local|local.dtype.clone());
        Ok(PreparedPartitionContiguousRow{transport,delivery:None,continuation:None,
            pending:Some(Pending{receipt,allowance,host}),local_source:self.local,routed:false,routed_source:None,active,present,local,local_dtype,
            next:0,last_epoch:None,coordinate:self.coordinate,usage,source:self.source,metadata:self.metadata})
    }
}
impl<'t,T:PartitionCaptureTransport> PreparedPartitionContiguousRow<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    pub(in super::super) fn retain_vote_failure(&mut self,cause:crate::capture::partition::PartitionCaptureCoordinationError)->PartitionCaptureProgramError {
        let pending=self.pending.take();self.retained(HeldPendingError{cause,_pending:pending})
    }
    pub(in super::super) fn bind_voted_scalar(&mut self,dtype:TensorDtype)->Result<(),PartitionCaptureProgramError>{
        if self.routed{return Err(self.error("sparse source requires its actual per-rank witnesses"));}
        self.bind_source_vote(dtype,None)
    }
    pub(in super::super) fn bind_routed_vote(&mut self,dtype:TensorDtype,ranks:Vec<PartitionCaptureRankSource>)
        ->Result<(),PartitionCaptureProgramError> {
        if !self.routed{return Err(self.error("dense source cannot consume a sparse witness table"));}
        self.bind_source_vote(dtype,Some(ranks))
    }
    fn bind_source_vote(&mut self,dtype:TensorDtype,actual:Option<Vec<PartitionCaptureRankSource>>)
        ->Result<(),PartitionCaptureProgramError> {
        let Some(pending)=self.pending.take() else{return Ok(())};
        let result=(||->Result<Vec<PartitionCaptureRankSource>,PartitionCaptureProgramError>{
            if self.local_dtype.as_ref().is_some_and(|local|local!=&dtype){return Err(self.error("local scalar differs from its source vote"));}
            if let Some(ranks)=actual {
                if ranks.len()!=pending.receipt.producers().count() || pending.receipt.producers().zip(&ranks)
                    .any(|((producer,_),row)|row.producer!=producer||row.dtype.as_ref().is_some_and(|value|value!=&dtype))
                    || ranks.iter().find(|row|row.producer==self.local).is_some_and(|row|row.dtype!=self.local_dtype) {
                    return Err(self.error("sparse vote table differs from retained source"));
                }
                return Ok(ranks);
            }
            let mut ranks=self.metadata.metadata_vec(pending.receipt.producers().count()).map_err(|e|self.retained(e))?;
            for (producer,_) in pending.receipt.producers(){ranks.push(PartitionCaptureRankSource{producer,dtype:Some(dtype.clone())});}
            Ok(ranks)
        })();
        let ranks=match result{Ok(value)=>value,Err(cause)=>return Err(self.retained(HeldPendingError{cause,_pending:Some(pending)}))};
        let Pending{receipt,allowance,host}=pending;
        let allowance=match allowance.bind(&receipt,dtype){Ok(value)=>value,
            Err(cause)=>return Err(self.retained(HeldHostError{cause,_host:host}))};
        let bank=host.bind(&receipt,allowance).map_err(|e|self.retained(e))?;
        let delivery=PreparedPartitionFragmentDelivery::prepare(self.transport,receipt,bank,&ranks,&self.metadata)
            .map_err(|e|self.retained(e))?;
        self.delivery=Some(delivery);Ok(())
    }
}

pub(super) fn construction_controls()->Option<usize>{
    let parts=[size_of::<Coordinate>()*2,size_of::<Option<PartitionCaptureLocalSource<'_>>>(),size_of::<LocalSource>()*2,size_of::<Option<LocalSource>>(),
        size_of::<Vec<Option<TensorDtype>>>(),size_of::<Option<TensorDtype>>(),size_of::<Sources<'_,'_>>(),size_of::<Source<'_>>(),
        size_of::<(&SharedCapturePlan,usize,usize,&[PartitionCaptureContiguousProducer],&[PartitionCaptureFragmentGeometry<'_>],
            Option<PartitionCaptureLocalSource<'_>>,PartitionCaptureCombination,InferenceGeometry,&HostMetadataFunding)>(),
        size_of::<(&SharedCapturePlan,usize,usize,&[PartitionCaptureContiguousProducer],&[PartitionCaptureFragmentGeometry<'_>],
            Option<PartitionCaptureLocalSource<'_>>,PartitionCaptureCombination,u64,&HostMetadataFunding)>(),
        size_of::<(&SharedCapturePlan,usize,usize,&[PartitionCaptureContiguousProducer],&[PartitionCaptureFragmentGeometry<'_>],
            Option<PartitionCaptureLocalSource<'_>>,PartitionCaptureCombination,Coordinate,&HostMetadataFunding)>(),
        size_of::<(PartitionCaptureLocalSource<'_>,&HostMetadataFunding)>(),size_of::<Result<LocalSource,Cause>>(),
        size_of::<std::slice::Iter<'_,PartitionCaptureContiguousProducer>>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
pub(super) fn pending_controls()->Option<usize>{
    let parts=[size_of::<Pending>()*2,size_of::<Option<Pending>>(),size_of::<Vec<PartitionCaptureFragmentGeometry<'_>>>(),
        size_of::<PartitionCaptureFragmentGeometry<'_>>(),size_of::<PreparedPartitionFragmentSourceAllowance>()*2,
        PreparedPartitionFragmentSourceAllowance::control_bytes()?,size_of::<Result<PreparedPartitionFragmentSourceAllowance,PartitionCaptureFragmentAllowanceError>>(),
        size_of::<Result<(PreparedPartitionFragmentSourceAllowance,CaptureUsage),PartitionCaptureProgramError>>(),
        size_of::<Vec<PartitionCaptureRankSource>>(),size_of::<Result<Vec<PartitionCaptureRankSource>,PartitionCaptureProgramError>>(),
        size_of::<HeldPendingError<crate::capture::partition::PartitionCaptureCoordinationError>>(),
        size_of::<HeldPendingError<PartitionCaptureProgramError>>(),size_of::<HeldHostError<PartitionCaptureProgramError>>(),
        size_of::<HeldHostError<SourceBindingError>>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<HeldHostError<SourceBindingError>>()?,
        size_of::<std::iter::Enumerate<std::iter::Zip<std::slice::Iter<'_,u64>,std::slice::Iter<'_,u64>>>>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<HeldPendingError<crate::capture::partition::PartitionCaptureCoordinationError>>()?,
        eredu_core::BackendFailure::source_retention_peak_bytes::<HeldPendingError<PartitionCaptureProgramError>>()?,
        eredu_core::BackendFailure::source_retention_peak_bytes::<HeldHostError<PartitionCaptureProgramError>>()?,
        size_of::<(PreparedPartitionContiguousSource,PartitionCaptureReceiptPlan,&mut Option<RowFunding<'_>>,&mut CaptureLedger)>(),
        size_of::<(TensorDtype,&PartitionCaptureReceiptPlan)>(),size_of::<std::slice::Iter<'_,NativeSource>>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
