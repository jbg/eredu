//! Retained source-to-host ownership for one original contiguous prefill row.
use super::*;
use crate::working_memory::{WorkingMemoryFundingRun,WorkingMemoryReservation,
    PartitionFragmentHostPlan,PreparedPartitionFragmentDelivery,PartitionCaptureRankSource,
    PartitionCaptureHookContinuation,PartitionCaptureHookReturnError,PartitionFragmentDestinationError};
use eredu_core::InferenceGeometry;

#[derive(Debug)]
struct NativeSource {producer:usize,fragment:usize,shape:Vec<u64>,slice:ResolvedCaptureSlice,
    dtype:Option<TensorDtype>,estimate:PartitionCaptureNativeEstimate}
mod voted;
mod routed;
pub use routed::{PreparedPartitionRoutedSource,PartitionCaptureRoutedLocalSource};
pub(crate) use routed::LocalSource as PreparedPartitionRoutedLocalSource;
mod coordinate;
mod coordinates;
use coordinates::{Sources as ProducerSources,Owned as OwnedProducers};
use coordinate::Coordinate;
pub use voted::PartitionCaptureLocalSource;
use voted::LocalSource;
/// Paid descriptive native sources for one selected global row. This retains
/// actual local geometry and precision, not a rewritten admission or work grant.
/// The sealed program must bind its real forward context and original account
/// before this source can construct any fragment destination.
#[derive(Debug)]
pub struct PreparedPartitionContiguousSource {
    index:usize,axis:usize,combination:PartitionCaptureCombination,coordinate:Coordinate,
    producers:OwnedProducers,ranks:Vec<PartitionCaptureRankSource>,native:Vec<NativeSource>,
    local:Option<LocalSource>,await_vote:bool,
    pub(in crate::capture::partition::funded) evidence_budget:Option<super::super::receipt::EvidenceBudgetSource>,
    source:SharedCapturePlan,metadata:WorkspaceMetadataFunding,
}
/// A typed pre-native refusal. The original Host remains in the cause, even
/// when the shared policy elects to coordinate a skipped observation.
#[derive(Debug)]
pub(super) struct PreparationFailure {
    pub(super) reason:Option<CaptureSkipReason>,
    pub(super) descriptor:Option<(usize,CaptureUsage)>,
    pub(super) cause:PartitionCaptureProgramError,
}
#[derive(Debug,thiserror::Error)]
#[error("partition fragment preparation: {cause}")]
struct HeldPreparation {
    #[source] cause:PartitionCaptureProgramError,
    _host:crate::working_memory::PreparedPartitionFragmentHostFunding,
}
enum RowFunding<'a> {
    Immediate{run:&'a WorkingMemoryFundingRun,reservation:&'a WorkingMemoryReservation},
    Prepared(crate::working_memory::PreparedPartitionFragmentHostFunding),
}
impl PreparedPartitionContiguousSource {
    /// Retain the exact per-rank cold sources before the actual forward epoch.
    /// All fragments are revalidated against the ordinary receipt at issuance.
    #[allow(clippy::too_many_arguments)]
    pub fn new(source:&SharedCapturePlan,index:usize,axis:usize,
        producers:&[PartitionCaptureContiguousProducer],dtypes:&[Option<TensorDtype>],
        native:&[PartitionCaptureFragmentSource<'_>],combination:PartitionCaptureCombination,
        inference:InferenceGeometry,metadata:&WorkspaceMetadataFunding)->Result<Self,PartitionCaptureProgramError> {
        Self::new_sources(source,index,axis,ProducerSources::Contiguous(producers),dtypes,voted::Sources::Typed(native),combination,
            Coordinate::Prefill(inference),None,metadata)
    }
    fn new_sources(source:&SharedCapturePlan,index:usize,axis:usize,
        producers:ProducerSources<'_>,dtypes:&[Option<TensorDtype>],
        native:voted::Sources<'_,'_>,combination:PartitionCaptureCombination,coordinate:Coordinate,
        local:Option<PartitionCaptureLocalSource<'_>>,metadata:&WorkspaceMetadataFunding)->Result<Self,PartitionCaptureProgramError> {
        let error=|cause|PartitionCaptureProgramError{cause,_source:source.clone(),_metadata:metadata.clone()};
        let frames=[coordinates::control_bytes().ok_or_else(||error(Cause::Source("component controls overflow")))?,Coordinate::control_bytes().ok_or_else(||error(Cause::Source("coordinate controls overflow")))?,size_of::<voted::Sources<'_,'_>>(),size_of::<voted::Source<'_>>(),size_of::<Option<PartitionCaptureLocalSource<'_>>>(),
            size_of::<Option<LocalSource>>(),size_of::<std::ops::Range<usize>>(),size_of::<Self>()*2,size_of::<NativeSource>()*2,size_of::<ResolvedCaptureSlice>()*2,
            size_of::<Result<Self,PartitionCaptureProgramError>>(),size_of::<PartitionCaptureProgramError>(),
            size_of::<PartitionCaptureContiguousProducer>(),size_of::<PartitionCaptureRankSource>(),
            size_of::<CaptureTransform>(),size_of::<Option<TensorDtype>>(),size_of::<TensorDtype>(),
            size_of::<Vec<NativeSource>>(),size_of::<Vec<PartitionCaptureContiguousProducer>>(),size_of::<Vec<PartitionCaptureRankSource>>(),
            size_of::<(&SharedCapturePlan,usize,usize,&[PartitionCaptureContiguousProducer],&[Option<TensorDtype>],
                &[PartitionCaptureFragmentSource<'_>],PartitionCaptureCombination,InferenceGeometry,&WorkspaceMetadataFunding)>(),
            size_of::<(&SharedCapturePlan,usize,usize,&[PartitionCaptureContiguousProducer],&[Option<TensorDtype>],
                voted::Sources<'_,'_>,PartitionCaptureCombination,Coordinate,Option<PartitionCaptureLocalSource<'_>>,&WorkspaceMetadataFunding)>(),
            size_of::<(&WorkspaceMetadataFunding,&[u64])>(),size_of::<Result<Vec<u64>,eredu_nn::Error>>(),
            size_of::<std::slice::Iter<'_,PartitionCaptureFragmentSource<'_>>>(),
            size_of::<std::iter::Zip<std::slice::Iter<'_,PartitionCaptureContiguousProducer>,std::slice::Iter<'_,Option<TensorDtype>>>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_,PartitionCaptureContiguousProducer>>>(),
            size_of::<std::slice::Iter<'_,PartitionCaptureContiguousProducer>>(),
            size_of::<std::slice::Iter<'_,u64>>(),size_of::<[usize;4]>(),
            size_of::<std::array::IntoIter<usize,4>>(),size_of::<(&WorkspaceMetadataFunding,)>(),
            size_of::<usize>()*4,size_of::<bool>()*3];
        metadata.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||error(Cause::Source("contiguous source controls overflow")))?).map_err(|e|error(e.into()))?;
        let selected=source.admission().plan().selections.get(index)
            .ok_or_else(||error(Cause::Source("contiguous selection is absent")))?;
        if !coordinate.accepts(selected) || producers.is_empty() || producers.len()!=dtypes.len()
            || !matches!(selected.transform,CaptureTransform::FullTensor|CaptureTransform::Slice|CaptureTransform::Preview{..}
                |CaptureTransform::Summary|CaptureTransform::Histogram{..}) {
            return Err(error(Cause::Source("contiguous prefill source differs")));
        }
        let raw=super::super::sum::native_transform(&selected.transform);
        let transform=if combination==PartitionCaptureCombination::SumF64ToF32{&raw}else{&selected.transform};
        let owned=producers.copy(metadata).map_err(|cause|error(cause))?;
        let mut ranks=metadata.metadata_vec(producers.len()).map_err(|cause|error(cause.into()))?;
        for (index,dtype) in dtypes.iter().enumerate() {
            if !matches!(dtype,None|Some(TensorDtype::F16|TensorDtype::F32|TensorDtype::Bf16)) {
                return Err(error(Cause::Source("component rank scalar differs")));
            }
            let rank=producers.rank(index);
            if ranks.iter().any(|row:&PartitionCaptureRankSource|row.producer==rank) {
                return Err(error(Cause::Source("duplicate component rank")));
            }
            ranks.push(PartitionCaptureRankSource{producer:rank,dtype:dtype.clone()});
        }
        let mut sources=metadata.metadata_vec(native.len()).map_err(|e|error(e.into()))?;
        for n in 0..native.len() {let row=native.get(n).expect("native source range");
            let rank=row.local_shape.len();
            if rank==0 || rank>32 || axis>=rank || (producers.contiguous() && row.fragment!=0) || row.transform!=transform
                || row.estimate.generated_creation_bytes!=0 || row.local_shape.iter().any(|&n|n==0)
                || [row.local_slice.starts.len(),row.local_slice.ends.len(),row.local_slice.strides.len(),row.local_slice.shape.len()].into_iter().any(|n|n!=rank)
                || !ranks.iter().any(|rank|rank.producer==row.producer
                    && (!native.typed()||rank.dtype.as_ref()==row.dtype))
                || sources.last().is_some_and(|previous:&NativeSource|(previous.producer,previous.fragment)>=(row.producer,row.fragment)) {
                return Err(error(Cause::Source("native contiguous fragment source differs")));
            }
            let copy=|values:&[u64]|->Result<Vec<u64>,eredu_nn::Error>{let mut out=metadata.metadata_vec(values.len())?;out.extend_from_slice(values);Ok(out)};
            sources.push(NativeSource{producer:row.producer,fragment:row.fragment,
                shape:copy(row.local_shape).map_err(|e|error(e.into()))?,slice:ResolvedCaptureSlice{
                    starts:copy(&row.local_slice.starts).map_err(|e|error(e.into()))?,
                    ends:copy(&row.local_slice.ends).map_err(|e|error(e.into()))?,
                    strides:copy(&row.local_slice.strides).map_err(|e|error(e.into()))?,
                    shape:copy(&row.local_slice.shape).map_err(|e|error(e.into()))?},dtype:row.dtype.cloned(),estimate:row.estimate});
        }
        let local=local.map(|local|LocalSource::copy(local,metadata)).transpose()
            .map_err(|cause|error(cause))?;
        Ok(Self{index,axis,combination,coordinate,producers:owned,ranks,native:sources,local,await_vote:!native.typed(),
            source:source.clone(),metadata:metadata.clone(),evidence_budget:None})
    }
    /// Retain the actual local scalar while peer precision is still unknown.
    /// All projected estimates remain geometry-only until the shared Source vote.
    #[allow(clippy::too_many_arguments)]
    pub fn new_local(source:&SharedCapturePlan,index:usize,axis:usize,
        producers:&[PartitionCaptureContiguousProducer],native:&[PartitionCaptureFragmentGeometry<'_>],
        local:Option<PartitionCaptureLocalSource<'_>>,combination:PartitionCaptureCombination,
        inference:InferenceGeometry,metadata:&WorkspaceMetadataFunding)->Result<Self,PartitionCaptureProgramError> {
        Self::new_local_coordinate(source,index,axis,producers,native,local,combination,Coordinate::Prefill(inference),metadata)
    }
    /// Retain one real decode invocation, with no fabricated prefill geometry.
    /// The original Host and actual scalar vote remain required at issuance.
    #[allow(clippy::too_many_arguments)]
    pub fn new_local_decode(source:&SharedCapturePlan,index:usize,axis:usize,
        producers:&[PartitionCaptureContiguousProducer],native:&[PartitionCaptureFragmentGeometry<'_>],
        local:Option<PartitionCaptureLocalSource<'_>>,combination:PartitionCaptureCombination,
        prediction:u64,metadata:&WorkspaceMetadataFunding)->Result<Self,PartitionCaptureProgramError> {
        Self::new_local_coordinate(source,index,axis,producers,native,local,combination,Coordinate::Decode(prediction),metadata)
    }
    fn new_local_coordinate(source:&SharedCapturePlan,index:usize,axis:usize,
        producers:&[PartitionCaptureContiguousProducer],native:&[PartitionCaptureFragmentGeometry<'_>],
        local:Option<PartitionCaptureLocalSource<'_>>,combination:PartitionCaptureCombination,
        coordinate:Coordinate,metadata:&WorkspaceMetadataFunding)->Result<Self,PartitionCaptureProgramError> {
        let error=|cause|PartitionCaptureProgramError{cause,_source:source.clone(),_metadata:metadata.clone()};
        metadata.reserve_metadata(voted::construction_controls().ok_or_else(||error(Cause::Source("local source controls overflow")))?)
            .map_err(|e|error(e.into()))?;
        let mut dtypes=metadata.metadata_vec(producers.len()).map_err(|e|error(e.into()))?;
        for producer in producers {dtypes.push(local.as_ref().filter(|local|local.producer==producer.rank).map(|local|local.dtype.clone()));}
        Self::new_sources(source,index,axis,ProducerSources::Contiguous(producers),&dtypes,voted::Sources::Geometry(native),combination,coordinate,local,metadata)
    }
    pub(super) fn matches_program(&self,source:&SharedCapturePlan,index:usize)->bool {
        self.source.same_storage(source)&&self.index==index
    }
    pub(super) fn matches_context(&self,context:&PartitionCaptureContext)->bool{self.coordinate.matches(context)}
    fn error(&self,cause:Cause)->PartitionCaptureProgramError {
        PartitionCaptureProgramError{cause,_source:self.source.clone(),_metadata:self.metadata.clone()}
    }
    fn retained<E:std::error::Error+Send+Sync+'static>(&self,cause:E)->PartitionCaptureProgramError {
        PartitionCaptureProgramError::local(cause,self.source.clone(),self.metadata.clone())
    }
    /// Called only by the sealed program after its shared first-epoch binding.
    /// Host holds are issued from this exact run/reservation at the real epoch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare<'t,T:PartitionCaptureTransport>(self,transport:&'t T,
        context:&PartitionCaptureContext,epoch:DistributedCommitEpoch,run:&WorkingMemoryFundingRun,
        reservation:&WorkingMemoryReservation,limits:PartitionCaptureReceiptLimits,ledger:&mut CaptureLedger)
        ->Result<PreparedPartitionContiguousRow<'t,T>,PartitionCaptureProgramError>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        self.prepare_funded(transport,context,epoch,RowFunding::Immediate{run,reservation},limits,ledger)
            .map_err(|failure|failure.cause)
    }
    /// Consume already protected original Host destinations at the actual
    /// first forward. This grants no new run, allocation or native authority.
    pub(crate) fn prepare_with_host<'t,T:PartitionCaptureTransport>(self,transport:&'t T,
        context:&PartitionCaptureContext,epoch:DistributedCommitEpoch,
        host:crate::working_memory::PreparedPartitionFragmentHostFunding,
        limits:PartitionCaptureReceiptLimits,ledger:&mut CaptureLedger)
        ->Result<PreparedPartitionContiguousRow<'t,T>,PartitionCaptureProgramError>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        self.prepare_funded(transport,context,epoch,RowFunding::Prepared(host),limits,ledger)
            .map_err(|failure|failure.cause)
    }
    pub(super) fn prepare_selected_with_host<'t,T:PartitionCaptureTransport>(self,transport:&'t T,
        context:&PartitionCaptureContext,epoch:DistributedCommitEpoch,
        host:crate::working_memory::PreparedPartitionFragmentHostFunding,
        limits:PartitionCaptureReceiptLimits,ledger:&mut CaptureLedger)
        ->Result<PreparedPartitionContiguousRow<'t,T>,PreparationFailure>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        self.prepare_funded(transport,context,epoch,RowFunding::Prepared(host),limits,ledger)
    }
    pub(super) fn skip_source(&self)->Result<(usize,CaptureUsage),PartitionCaptureProgramError> {
        let producer=self.producers.minimum_rank()
            .ok_or_else(||self.error(Cause::Source("contiguous source has no actual rank")))?;
        Ok((producer,self.source_usage()?))
    }
    fn source_usage(&self)->Result<CaptureUsage,PartitionCaptureProgramError> {
        self.native.iter().try_fold(CaptureUsage::default(),|sum,row|sum.checked_add(row.estimate.capture))
            .map_err(|e|self.error(e.into()))
    }
    fn prepare_funded<'t,T:PartitionCaptureTransport>(self,transport:&'t T,
        context:&PartitionCaptureContext,epoch:DistributedCommitEpoch,funding:RowFunding<'_>,
        limits:PartitionCaptureReceiptLimits,ledger:&mut CaptureLedger)
        ->Result<PreparedPartitionContiguousRow<'t,T>,PreparationFailure>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        let source=self.source.clone();let metadata=self.metadata.clone();
        let mut funding=Some(funding);
        let mut descriptor=None;
        let result=(||{
        self.metadata.reserve_metadata(PreparedPartitionContiguousRow::<T>::control_bytes()
            .ok_or_else(||self.error(Cause::Source("contiguous owner controls overflow")))?).map_err(|e|self.error(e.into()))?;
            let value=self.skip_source()?;descriptor=Some(value);
            self.prepare_funded_inner(transport,context,epoch,&mut funding,limits,ledger,value.1)
        })();
        result.map_err(|cause|{
            let reason=match &cause.cause {
                Cause::Receipt(error)=>error.limit_skip(),
                Cause::FragmentAllowance(error)=>error.limit_skip(),
                _=>None,
            };
            let cause=match funding.take() {
                Some(RowFunding::Prepared(host))=>PartitionCaptureProgramError::local(
                    HeldPreparation{cause,_host:host},source,metadata),
                _=>cause,
            };
            PreparationFailure{reason,descriptor,cause}
        })
    }
    fn prepare_funded_inner<'t,T:PartitionCaptureTransport>(mut self,transport:&'t T,
        context:&PartitionCaptureContext,epoch:DistributedCommitEpoch,funding:&mut Option<RowFunding<'_>>,
        limits:PartitionCaptureReceiptLimits,ledger:&mut CaptureLedger,usage:CaptureUsage)
        ->Result<PreparedPartitionContiguousRow<'t,T>,PartitionCaptureProgramError>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        if context.selection_index!=self.index || !self.coordinate.matches(context)
            || context.invocation.is_some() || context.forward_epoch!=epoch.value()
            || context.capture_plan_identity!=self.source.admission().identity() {
            return Err(self.error(Cause::Source("contiguous forward binding differs")));
        }
        let mut receipt=self.producers.receipt(&self.source,context,self.axis,
            self.combination,transport.participant_count(),limits,&self.metadata,ledger).map_err(|e|self.error(e.into()))?;
        if let Some(source)=self.evidence_budget.take() {
            receipt.bind_evidence_budget(source).map_err(|cause|self.error(cause.into()))?;
        }
        if self.await_vote {
            if !matches!(funding,Some(RowFunding::Prepared(_))) {
                return Err(self.error(Cause::Source("unvoted source requires retained original Host")));
            }
            return self.prepare_unvoted(transport,receipt,funding,ledger,usage);
        }
        let selected=&self.source.admission().plan().selections[self.index].transform;
        let raw=super::super::sum::native_transform(selected);
        let transform=if self.combination==PartitionCaptureCombination::SumF64ToF32{&raw}else{selected};
        let mut sources=self.metadata.metadata_vec(self.native.len()).map_err(|e|self.error(e.into()))?;
        for row in &self.native {sources.push(PartitionCaptureFragmentSource{producer:row.producer,fragment:row.fragment,
            local_shape:&row.shape,local_slice:&row.slice,transform,dtype:row.dtype.clone().ok_or_else(||self.error(Cause::Source("typed source scalar is absent")))?,estimate:row.estimate});}
        let allowance=PreparedPartitionFragmentAllowance::prepare(transport,&mut receipt,&sources,&self.metadata,ledger)
            .map_err(|e|self.error(e.into()))?;
        let local=transport.capture_rank();let local_source=self.ranks.iter().find(|row|row.producer==local);
        let local_dtype=local_source.and_then(|row|row.dtype.clone());
        let active=receipt.producer(local).is_some_and(|p|!p.fragments().is_empty());
        let present=receipt.producer(local).is_some();
        let plan=self.coordinate.host(&receipt).map_err(|e|self.error(e.into()))?;
        // The existing receipt constructor owns canonical producer ordering.
        // Copy only its rank order; no second sorting worker or identity writer.
        let mut ranks=self.metadata.metadata_vec(self.ranks.len()).map_err(|e|self.error(e.into()))?;
        for (producer,_) in receipt.producers() {
            let row=self.ranks.iter().find(|row|row.producer==producer)
                .ok_or_else(||self.error(Cause::Source("canonical producer has no retained native source")))?;
            ranks.push(PartitionCaptureRankSource{producer,dtype:row.dtype.clone()});
        }
        let bank=match funding.take().expect("unconsumed original funding") {
            RowFunding::Immediate{run,reservation}=>run.prepare_partition_fragments(reservation,plan,allowance)
                .map_err(|e|self.retained(e))?,
            RowFunding::Prepared(host)=>host.bind(&receipt,allowance).map_err(|e|self.retained(e))?,
        };
        let delivery=PreparedPartitionFragmentDelivery::prepare(transport,receipt,bank,&ranks,&self.metadata).map_err(|e|self.retained(e))?;
        Ok(PreparedPartitionContiguousRow{transport,delivery:Some(delivery),continuation:None,pending:None,local_source:None,routed:false,routed_source:None,active,present,local,local_dtype,
            next:0,last_epoch:None,coordinate:self.coordinate,usage,source:self.source.clone(),metadata:self.metadata.clone()})
    }
}

#[derive(Debug,thiserror::Error)]
enum Refusal {
    #[error("contiguous local source coverage is incomplete")] Incomplete,
    #[error(transparent)] Destination(PartitionFragmentDestinationError),
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct RejectedRow { #[source] cause:Refusal,_hook:crate::working_memory::PartitionLocalCaptureHook }

/// One actual paid row. Taking its hook suspends every transport/receipt method
/// until the exact original owner returns; no clone creates an issuance cursor.
pub(crate) struct PreparedPartitionContiguousRow<'t,T:PartitionCaptureTransport> {
    transport:&'t T,
    delivery:Option<PreparedPartitionFragmentDelivery<'t,T>>,continuation:Option<PartitionCaptureHookContinuation<'t,T>>,
    pending:Option<voted::Pending>,local_source:Option<LocalSource>,
    routed:bool,routed_source:Option<routed::LocalSource>,
    active:bool,present:bool,local:usize,local_dtype:Option<TensorDtype>,next:u64,last_epoch:Option<DistributedCommitEpoch>,
    coordinate:Coordinate,usage:CaptureUsage,source:SharedCapturePlan,metadata:WorkspaceMetadataFunding,
}
impl<'t,T:PartitionCaptureTransport> PreparedPartitionContiguousRow<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    fn error(&self,message:&'static str)->PartitionCaptureProgramError {
        PartitionCaptureProgramError{cause:Cause::Source(message),_source:self.source.clone(),_metadata:self.metadata.clone()}
    }
    fn retained<E:std::error::Error+Send+Sync+'static>(&self,cause:E)->PartitionCaptureProgramError {
        PartitionCaptureProgramError::local(cause,self.source.clone(),self.metadata.clone())
    }
    fn control_bytes()->Option<usize> {
        let parts=[size_of::<Option<routed::LocalSource>>(),size_of::<bool>(),preparation_controls::<T>()?,voted::pending_controls()?,size_of::<Option<voted::Pending>>(),size_of::<Option<LocalSource>>(),size_of::<Self>()*2,size_of::<PreparedPartitionContiguousSource>(),
            size_of::<Result<Self,PartitionCaptureProgramError>>(),size_of::<Option<PreparedPartitionFragmentDelivery<'t,T>>>(),
            size_of::<Option<PartitionCaptureHookContinuation<'t,T>>>(),size_of::<PartitionCaptureContext>(),
            size_of::<PartitionCaptureReceiptPlan>(),size_of::<CaptureTransform>(),size_of::<PartitionFragmentHostPlan<'_>>(),
            size_of::<RowFunding<'_>>(),size_of::<crate::working_memory::PreparedPartitionFragmentHostFunding>(),
            size_of::<(&Self,&PartitionCaptureContext,DistributedCommitEpoch,RowFunding<'_>,PartitionCaptureReceiptLimits,&mut CaptureLedger)>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<crate::working_memory::PartitionFragmentHostBindingError>()?,
            size_of::<Vec<PartitionCaptureFragmentSource<'_>>>(),size_of::<PartitionCaptureFragmentSource<'_>>(),
            size_of::<(&Self,&PartitionCaptureContext,DistributedCommitEpoch,&WorkingMemoryFundingRun,&WorkingMemoryReservation,
                PartitionCaptureReceiptLimits,&mut CaptureLedger)>(),
            size_of::<(&mut Self,u64,DistributedCommitEpoch)>(),size_of::<(&mut Self,PartitionCaptureLocalHook)>(),
            size_of::<Result<Option<PartitionCaptureLocalHook>,PartitionCaptureProgramError>>(),
            size_of::<Result<Option<PartitionPrefillReceiverSource>,PartitionCaptureProgramError>>(),
            size_of::<Result<PreparedPartitionFragmentDelivery<'t,T>,PartitionCaptureProgramError>>(),
            size_of::<Result<PreparedPartitionFragmentDelivery<'t,T>,PartitionCaptureHookReturnError>>(),
            size_of::<std::slice::Iter<'_,NativeSource>>(),size_of::<Option<&PartitionCaptureRankSource>>(),
            size_of::<Option<TensorDtype>>(),size_of::<Option<DistributedCommitEpoch>>(),size_of::<u64>()*2,
            size_of::<CaptureUsage>()*2,size_of::<Result<CaptureUsage,CaptureError>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<PartitionCaptureFragmentAllowanceError>()?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<PartitionFragmentDestinationError>()?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<PartitionCaptureHookReturnError>()?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<RejectedRow>()?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<PartitionPrefillCaptureSourceError>()?,
            size_of::<RejectedRow>(),size_of::<Refusal>(),size_of::<Vec<PartitionCaptureRankSource>>(),
            size_of::<std::slice::Iter<'_,PartitionCaptureRankSource>>(),
            PartitionCaptureLocalHook::control_bytes()?,PartitionPrefillReceiverSource::control_bytes()?,
            super::super::PartitionInvocationReceiverSource::control_bytes()?,size_of::<Coordinate>()*2];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(super) fn is_routed(&self)->bool {self.routed}
    pub(super) fn source_present(&self)->bool {self.present}
    pub(super) fn local_dtype(&self)->Option<&TensorDtype>{self.local_dtype.as_ref()}
    pub(super) fn source_usage(&self)->CaptureUsage{self.usage}
    pub(super) fn next_chunk(&self)->u64{self.next}
    pub(super) fn geometry(&self)->Option<InferenceGeometry>{self.coordinate.prefill()}
    pub(crate) fn receipt(&self)->Result<&PartitionCaptureReceiptPlan,PartitionCaptureProgramError> {
        self.pending.as_ref().map(|pending|&pending.receipt).or_else(||self.delivery.as_ref().map(|d|d.receipt_plan()))
            .ok_or_else(||self.error("contiguous receipt is lent or spent"))
    }
    pub(crate) fn produces(&self)->bool {self.active}
    pub(crate) fn reserve_target(&mut self,frame:&mut ScheduledCaptureStep<'_>)->Result<(),PartitionCaptureProgramError> {
        let Some(delivery)=self.delivery.as_mut() else{return Err(self.error("contiguous target owner is lent or spent"));};
        delivery.reserve_prefill_target(frame).map_err(|e|self.retained(e))
    }
    fn check_attempt(&self,chunk:u64,epoch:DistributedCommitEpoch)->Result<(),PartitionCaptureProgramError> {
        if chunk!=self.next || chunk>=self.coordinate.attempts()
            || self.last_epoch.is_some_and(|last|last>=epoch) || epoch.value()<self.receipt()?.context().forward_epoch {
            return Err(self.error("contiguous chunk or epoch attempt differs"));
        }Ok(())
    }
    pub(crate) fn take_local(&mut self,chunk:u64,epoch:DistributedCommitEpoch)
        ->Result<Option<PartitionCaptureLocalHook>,PartitionCaptureProgramError> {
        if !self.active&&!(self.routed&&self.present){return Ok(None);}self.check_attempt(chunk,epoch)?;
        if self.routed&&self.routed_source.is_none(){return Err(self.error("sparse source owner is already lent"));}
        self.last_epoch=Some(epoch);
        let delivery=self.delivery.take().ok_or_else(||self.error("contiguous hook is already lent"))?;
        let (continuation,mut hook)=delivery.into_local_hook();self.continuation=Some(continuation);
        if self.routed {hook.bind_routed_source(self.routed_source.take().expect("checked original sparse source"));}
        Ok(Some(PartitionCaptureLocalHook::new(hook)))
    }
    pub(crate) fn return_local(&mut self,hook:PartitionCaptureLocalHook)->Result<(),PartitionCaptureProgramError> {
        let Some(continuation)=self.continuation.take() else{return Err(hook.reject());};
        if self.routed {
            let (delivery,source)=continuation.resume_routed(hook.into_inner()).map_err(|e|self.retained(e))?;
            self.delivery=Some(delivery);self.routed_source=source;
            if self.routed_source.is_none(){return Err(self.reject_missing_routed_source());}
        }else{self.delivery=Some(continuation.resume(hook.into_inner()).map_err(|e|self.retained(e))?);}
        self.next+=1;Ok(())
    }
    fn reject_missing_routed_source(&mut self)->PartitionCaptureProgramError {
        let Some(delivery)=self.delivery.take() else{return self.error("sparse source return lost its original Host");};
        let (_,hook)=delivery.into_local_hook();
        PartitionCaptureProgramError::local(RejectedRow{cause:Refusal::Incomplete,_hook:hook},self.source.clone(),self.metadata.clone())
    }
    pub(crate) fn take_receiver(&mut self,chunk:u64,epoch:DistributedCommitEpoch)
        ->Result<Option<PartitionPrefillReceiverSource>,PartitionCaptureProgramError> {
        if self.active||!self.present{return Ok(None);}
        let inference=self.coordinate.prefill().ok_or_else(||self.error("decode source is not a prefill receiver"))?;
        self.check_attempt(chunk,epoch)?;
        self.last_epoch=Some(epoch);self.next+=1;
        let dtype=self.local_dtype.clone().ok_or_else(||self.error("empty local source has no actual scalar"))?;
        let receiver=if let Some(local)=&self.local_source {
            PartitionPrefillReceiverSource::prepare_local(self.receipt()?,self.local,dtype,inference,chunk,&local.shape)
        }else{PartitionPrefillReceiverSource::prepare(self.receipt()?,self.local,dtype,inference,chunk)};
        receiver.map(Some)
            .map_err(|e|self.retained(e))
    }
    pub(crate) fn take_invocation_receiver(&mut self,epoch:DistributedCommitEpoch)
        ->Result<Option<super::super::PartitionInvocationReceiverSource>,PartitionCaptureProgramError> {
        if self.active||!self.present{return Ok(None);}
        if self.coordinate.prefill().is_some(){return Err(self.error("prefill source is not a decode receiver"));}
        self.check_attempt(0,epoch)?;self.last_epoch=Some(epoch);self.next+=1;
        let local=self.local_source.as_ref().ok_or_else(||self.error("decode replica has no retained physical source"))?;
        super::super::PartitionInvocationReceiverSource::prepare_local(self.receipt()?,self.local,local.dtype.clone(),&local.shape)
            .map(Some).map_err(|cause|self.retained(cause))
    }
    fn reject_ready(mut self,cause:Refusal)->PartitionCaptureProgramError {
        let Some(delivery)=self.delivery.take() else{return self.error("contiguous hook has not returned");};
        let (_continuation,hook)=delivery.into_local_hook();
        PartitionCaptureProgramError::local(RejectedRow{cause,_hook:hook},self.source.clone(),self.metadata.clone())
    }
    pub(crate) fn finish(mut self)->Result<PreparedPartitionFragmentDelivery<'t,T>,PartitionCaptureProgramError> {
        if self.continuation.is_some() {return Err(self.error("contiguous hook has not returned"));}
        if self.present&&self.next!=self.coordinate.attempts() {
            return Err(self.reject_ready(Refusal::Incomplete));
        }
        if self.active && !self.routed {
            let Some(delivery)=self.delivery.as_mut() else{return Err(self.error("contiguous delivery was spent"));};
            let result=if self.coordinate.prefill().is_some(){delivery.finish_local_prefill(0)}else{delivery.finish_local_invocation(0)};
            if let Err(cause)=result{return Err(self.reject_ready(Refusal::Destination(cause)));}
        }
        self.delivery.take().ok_or_else(||self.error("contiguous delivery was spent"))
    }
}

fn preparation_controls<T:PartitionCaptureTransport>()->Option<usize> {
    let parts=[size_of::<PreparationFailure>()*2,size_of::<HeldPreparation>(),
        size_of::<Option<CaptureSkipReason>>(),size_of::<CaptureSkipReason>(),
        size_of::<Option<RowFunding<'_>>>(),size_of::<&mut Option<RowFunding<'_>>>(),
        size_of::<Result<PreparedPartitionContiguousRow<'_,T>,PreparationFailure>>(),
        size_of::<Result<(usize,CaptureUsage),PartitionCaptureProgramError>>(),
        size_of::<(PreparedPartitionContiguousSource,&T,&PartitionCaptureContext,DistributedCommitEpoch,
            RowFunding<'_>,PartitionCaptureReceiptLimits,&mut CaptureLedger)>(),
        size_of::<(PreparedPartitionContiguousSource,&T,&PartitionCaptureContext,DistributedCommitEpoch,
            &mut Option<RowFunding<'_>>,PartitionCaptureReceiptLimits,&mut CaptureLedger,CaptureUsage)>(),
        size_of::<(usize,CaptureUsage)>(),size_of::<Option<(usize,CaptureUsage)>>(),size_of::<SharedCapturePlan>(),size_of::<WorkspaceMetadataFunding>(),
        size_of::<std::slice::Iter<'_,PartitionCaptureContiguousProducer>>(),
        size_of::<std::slice::Iter<'_,NativeSource>>(),size_of::<Option<usize>>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<HeldPreparation>()?];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
