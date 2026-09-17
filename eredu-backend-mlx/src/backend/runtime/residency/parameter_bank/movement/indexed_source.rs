//! Exact completed-ID and immutable remap sources for one addressable chunk.
use super::*;
use eredu_core::{SharedStorageOwner, SharedStorageRetirement};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::expert::{AddressableChunkCensus, AddressableChunkPlan, IndexedDemandSource};
use safemlx::{OriginalScopeObserver, PreparedInputRuntime, OwnedHostCopyPlan,
    OwnedHostCopyFacts, PreparedSubmissionGraphQuota, StreamCopyPlan};
use std::{alloc::Layout, cell::Cell, mem::{size_of, size_of_val}, rc::Rc};

/// Native binding source supplied explicitly by the actual model constructor.
/// This owns no invocation, native allocation or session-budget authority.
#[derive(Clone)]
pub(crate) struct IndexedBankSource {
    pub(super) bank:SharedAddressableParameterBank,
    pub(super) options:eredu_runtime::ParameterBankLoadOptions,
    request:Rc<request::Channel>,
}
#[path = "indexed_source/binding_layouts.rs"]
mod binding_layouts;
pub(crate) use binding_layouts::{IndexedBindingLayout,IndexedBindingStorage,IndexedBindingIdentity};
#[path = "indexed_source/request.rs"]
mod request;
pub(crate) use request::{IndexedRequestSource,IndexedRequestInstallation};
pub(super) type IndexedBankBinding=IndexedBankSource;
impl std::ops::Deref for IndexedBankSource {
    type Target=SharedAddressableParameterBank;
    fn deref(&self)->&Self::Target{&self.bank}
}
impl IndexedBankSource {
    pub(crate) fn storage(&self)->&SharedAddressableParameterBank{&self.bank}

    pub(crate) fn new(bank:SharedAddressableParameterBank,options:eredu_runtime::ParameterBankLoadOptions)->Self {
        Self{bank,options,request:Rc::new(request::Channel::default())}
    }
    pub(crate) fn with_workspace_source<T,E,F>(&self,funding:&WorkspaceMetadataFunding,inspect:F)
        ->Result<Result<T,E>,AddressableSourceFailure>
    where F:for<'a> FnOnce(AddressableBankSourceLoan<'a>)->Result<T,E> {
        self.bank.with_workspace_source(funding,inspect)
    }
    pub(crate) fn control_bytes()->Option<usize> {
        size_of::<Self>().checked_add(size_of::<Option<Self>>())?
            .checked_add(size_of::<eredu_runtime::ParameterBankLoadOptions>())?
            .checked_add(request::Channel::control_bytes()?)
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("addressable indexed source, bank, chunk or descriptor identity differs")]
    Identity,
    #[error("addressable indexed source differs at {0}")]
    IdentityAt(&'static str),
    #[error("addressable indexed source has invalid integer geometry or mapping")]
    Geometry,
    #[error("addressable indexed source attempt is spent")]
    Spent,
    #[error("addressable indexed source extent overflowed")]
    Overflow,
    #[error("addressable indexed source funding: {0}")]
    Funding(#[source] WorkspaceMetadataFundingError),
    #[error("addressable indexed destination allocation: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("addressable indexed source bank: {0}")]
    Bank(#[source] AddressableSourceFailure),
    #[error("addressable indexed source native completion: {0}")]
    Native(#[source] safemlx::error::Exception),
    #[error("addressable indexed {stage} completion: {cause}")]
    NativeCompletion { stage: &'static str, #[source] cause: safemlx::error::Exception },
    #[error("addressable indexed source descriptor: {0}")]
    Descriptor(#[source] safemlx::ArrayDescriptorError),
    #[error("addressable indexed source clone: {0}")]
    Clone(#[source] safemlx::PreparedArrayCloneCause),
    #[error("addressable indexed source read: {0}")]
    Read(#[source] safemlx::error::AsSliceError),
    #[error("addressable remap source copy: {0}")]
    Copy(#[source] safemlx::OwnedHostCopyCause),
    #[error("addressable remap source arena: {0}")]
    Graph(#[source] safemlx::SubmissionGraphQuotaCause),
    #[error("addressable selected cache acquisition: {0}")]
    Residency(#[source] ResidencyError),
    #[error("addressable counter publication lock unavailable (busy={busy})")]
    CounterLock { busy: bool },
    #[error("addressable indexed telemetry: {0}")]
    Telemetry(#[source] AddressableParameterBankError),
    #[error("addressable retained host source: {0}")]
    Host(#[source] crate::backend::runtime::residency::manager::HostCopyWorkspaceError),
    #[error("addressable selected source population: {0}")]
    Source(#[source] crate::backend::runtime::residency::manager::OperationSourceFailure),
    #[error("addressable selected source bank: {0}")]
    Budget(#[source] eredu_runtime::working_memory::HostDestinationCause),
    #[error("addressable accepted source memory: {0}")]
    Memory(#[source] eredu_runtime::working_memory::WorkingMemoryError),
    #[error("addressable indexed consumer: {0}")]
    Consumer(#[source] Error),
}
#[derive(thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source] cause: Cause,
    source: Option<SharedStorageOwner<SourceIdentity>>,
    bank: SharedAddressableParameterBank,
    funding: WorkspaceMetadataFunding,
}
impl std::fmt::Debug for Failure {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("AddressableIndexedFailure").field("cause",&self.cause).finish_non_exhaustive()
    }
}
fn failed(cause:Cause, bank:&SharedAddressableParameterBank, funding:&WorkspaceMetadataFunding,
    source:Option<&SharedStorageOwner<SourceIdentity>>)->Error {
    Error::Neural(eredu_nn::Error::backend_retained_source(Failure {
        cause, source:source.cloned(), bank:bank.clone(), funding:funding.clone(),
    }))
}
pub(super) fn require_ordinary()->Result<(),Error> {
    if OriginalScopeObserver::try_current()?.is_some() {
        return Err(Error::OriginalSourceContract {
            stage: "ordinary indexed operation inside an original native scope",
            cause: eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        });
    }
    Ok(())
}
// A remap source arena carries account custody only: no array, bank/manager,
// observer or source object can form a native allocation backedge here.
#[derive(Clone)]
struct CopyCustody { funding: WorkspaceMetadataFunding }
struct SourceIdentity {
    input:Array,
    bank:SharedAddressableParameterBank,
    census:AddressableChunkCensus,
    bulk_target_bytes:u64,
    parameter_revision:u64,
    funding:WorkspaceMetadataFunding,
}
impl SharedStorageRetirement for SourceIdentity {
    fn retire(self:Arc<Self>){drop(Arc::into_inner(self));}
}
struct CompletedIds {
    ids:Vec<usize>,
    source:SharedStorageOwner<SourceIdentity>,
}
impl SharedStorageRetirement for CompletedIds {
    fn retire(self:Arc<Self>){drop(Arc::into_inner(self));}
}

/// Descriptive counts for this exact existing worker, including its source copy.
/// This does not reserve a native scope, parent completion or cache acquisition.
#[derive(Clone,Copy,Debug)]
pub(crate) struct IndexedChunkLayout {
    census:AddressableChunkCensus,
    copy:OwnedHostCopyFacts,
    host_bytes:usize,
}
fn shared_bytes<T>()->Option<usize>{
    Some(Layout::new::<[std::sync::atomic::AtomicUsize;2]>()
        .extend(Layout::new::<T>()).ok()?.0.pad_to_align().size())
}
impl IndexedChunkLayout {
    pub(crate) fn inspect(runtime:&PreparedInputRuntime,census:AddressableChunkCensus)->Option<Self>{
        let count=census.values()?;
        let shape=[i32::try_from(census.rows()).ok()?,i32::try_from(census.routes()).ok()?];
        if count==0 || census.members()==0 || i32::try_from(census.members()).is_err(){return None;}
        let copy=OwnedHostCopyPlan::<i32>::new(runtime,&shape,count).ok()?;
        let frames=[size_of::<Self>(),size_of::<Body>(),size_of::<SourceIdentity>(),
            size_of::<OriginalIndexedChunkSource>(),size_of::<Result<OriginalIndexedChunkSource,Error>>(),
            size_of::<CompletedIds>(),size_of::<CopyCustody>(),size_of::<Failure>(),size_of::<Cause>(),
            size_of::<Result<IndexedDemandSource,Error>>(),size_of::<Result<MlxTensor,Error>>(),
            size_of::<(Vec<usize>,Vec<u64>,Vec<(usize,u64)>,Vec<i32>)>(),
            size_of::<[i32;2]>(),size_of::<[usize;8]>(),
            size_of::<safemlx::EvaluatedArray<'_>>(),size_of::<Result<safemlx::EvaluatedArray<'_>,safemlx::error::Exception>>(),
            safemlx::EvaluatedArray::iteration_control_bytes::<i32>()?
                .max(safemlx::EvaluatedArray::iteration_control_bytes::<u32>()?)
                .max(safemlx::EvaluatedArray::iteration_control_bytes::<i64>()?)
                .max(safemlx::EvaluatedArray::iteration_control_bytes::<u64>()?),
            IndexedDemandSource::control_bytes()?,
            eredu_nn::Error::retained_source_control_bytes::<Failure>()?,
            OriginalScopeObserver::control_bytes()?,
            Array::descriptor_comparison_control_bytes()?.checked_mul(3)?,
            Array::inspection_clone_handle_bytes(),safemlx::PreparedArrayClone::control_bytes()?,
            crate::backend::runtime::cache::value_completion_control_bytes(1)?,
            usize::try_from(ResidentTransfer::original_retirement_control_bytes()?).ok()?,
            // Three source handoffs use the same non-consuming traversal and
            // nested-bank validators already bounded by these shared workers.
            crate::backend::runtime::cache::value_completion_control_bytes(1)?
                .checked_add(safemlx::OperationEvent::traversal_leaf_control_bytes()?)?
                .checked_add(size_of::<(&OriginalIndexedChunkSource,&'static str,&'static str)>())?
                .checked_mul(3)?,
            safemlx::OperationEvent::traversal_leaf_control_bytes()?.checked_mul(2)?,
            shared_bytes::<SourceIdentity>()?,shared_bytes::<CompletedIds>()?,
            Layout::new::<[usize;2]>().extend(Layout::new::<Body>()).ok()?.0.pad_to_align().size(),
            copy.control_bytes::<CopyCustody>()?,copy.facts().backing_bytes(),
            PreparedInputRuntime::inspection_alias_control_bytes(),
            Layout::array::<usize>(count).ok()?.size(),
            Layout::array::<u64>(census.members()).ok()?.size(),
            Layout::array::<(usize,u64)>(census.maximum_members()).ok()?.size(),
            Layout::array::<i32>(count).ok()?.size(),
        ];
        let host_bytes=frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)?;
        Some(Self{census,copy:copy.facts(),host_bytes})
    }
    pub(crate) fn host_bytes(self)->usize{self.host_bytes}
    pub(crate) fn parent_completions(self)->usize{1}
    /// Exact independent copy arena facts; the remap does not consume parent Graph.
    pub(crate) fn copy_facts(self)->OwnedHostCopyFacts{self.copy}
}
struct Body {
    identity:SharedStorageOwner<SourceIdentity>,
    runtime:PreparedInputRuntime,
    observer:OriginalScopeObserver,
    stream:StreamCopyPlan<CopyCustody>,
    layout:IndexedChunkLayout,
    discover_attempted:Cell<bool>,
    discovered:Cell<bool>,
    remap_attempted:Cell<bool>,
    remapped:Cell<bool>,
    closed:Cell<bool>,
    acquire_attempted:Cell<bool>,
    acquired:Cell<bool>,
    binding_attempted:Cell<bool>,
    bound:Cell<bool>,
    completion_attempted:Cell<bool>,
    completed:Cell<bool>,
    residency:std::cell::RefCell<Option<Box<dyn slots::IndexedResidency>>>,
    route_copies:Cell<usize>,
    funding:WorkspaceMetadataFunding,
}
#[derive(Clone)]
pub(crate) struct OriginalIndexedChunkSource(Option<Rc<Body>>);
impl Drop for OriginalIndexedChunkSource {
    fn drop(&mut self){if let Some(body)=self.0.take(){drop(Rc::into_inner(body));}}
}
impl OriginalIndexedChunkSource {
    fn body(&self)->&Body{self.0.as_deref().expect("live indexed source")}
    fn failure(&self,cause:Cause)->Error {
        let b=self.body();failed(cause,&b.identity.bank,&b.funding,Some(&b.identity))
    }
    fn validate_completion_source(&self,context_stage:&'static str,bank_stage:&'static str)->Result<(),Error> {
        safemlx::OperationEvent::validate_traversal_context(&self.body().observer)
            .map_err(|cause|self.failure(Cause::NativeCompletion{stage:context_stage,cause}))?;
        safemlx::OperationEvent::validate_nested_completion(1)
            .map_err(|cause|self.failure(Cause::NativeCompletion{stage:bank_stage,cause}))?;
        Ok(())
    }
    fn validate_parent(&self, stream: &Stream) -> Result<(), Error> {
        let b = self.body();
        let actual = OriginalScopeObserver::require_current()
            .map_err(|e| self.failure(Cause::Native(e)))?;
        if b.closed.get() || !b.stream.matches_source(stream) || !b.observer.same_scope(&actual) {
            return Err(self.failure(Cause::Identity));
        }
        Ok(())
    }
    fn validate(&self,binding:Option<&IndexedBankBinding>,input:&Array,census:AddressableChunkCensus,
        stream:&Stream)->Result<(),Error>{
        let b=self.body();let binding=binding.ok_or_else(||self.failure(Cause::Identity))?;
        if b.closed.get() || b.identity.census!=census || !b.stream.matches_source(stream)
            || !Arc::ptr_eq(&binding.bank.inner,&b.identity.bank.inner)
            || binding.bank.scope!=b.identity.bank.scope {
            return Err(self.failure(Cause::Identity));
        }
        self.validate_parent(stream)?;
        let loan=b.identity.input.try_descriptor().map_err(|e|self.failure(Cause::Descriptor(e)))?;
        if !loan.same_descriptor(input).map_err(|e|self.failure(Cause::Descriptor(e)))? {
            return Err(self.failure(Cause::Identity));
        }
        Ok(())
    }
    pub(super) fn discover(&self,binding:Option<&IndexedBankBinding>,indices:&MlxTensor,
        census:AddressableChunkCensus,stream:&Stream)->Result<IndexedDemandSource,Error>{
        self.validate(binding,indices.as_array(),census,stream)?;
        let b=self.body();
        if b.discover_attempted.replace(true){return Err(self.failure(Cause::Spent));}
        // The caller supplied this exact model observer; the worker consumes one
        // already-admitted parent traversal, never an ordinary Eval fallback.
        crate::backend::runtime::cache::complete_values([indices.as_array()],stream)
            .map_err(|cause|self.failure(Cause::NativeCompletion { stage:"route discovery",cause }))?;
        safemlx::OperationEvent::validate_traversal_leaf(indices.as_array(),&b.observer)
            .map_err(|e|self.failure(Cause::Native(e)))?;
        let evaluated=indices.as_array().completed_in_original_scope(&b.observer)
            .map_err(|e|self.failure(Cause::Native(e)))?;
        let mut counts=Vec::new();counts.try_reserve_exact(census.members())
            .map_err(|e|self.failure(Cause::Allocation(e)))?;
        counts.resize(census.members(),0u64);
        let mut ids=Vec::new();ids.try_reserve_exact(census.values().ok_or_else(||self.failure(Cause::Overflow))?)
            .map_err(|e|self.failure(Cause::Allocation(e)))?;
        let mut accept=|identity:Option<usize>|->Result<(),Error>{
            let identity=identity.filter(|&id|id<census.members()).ok_or_else(||self.failure(Cause::Geometry))?;
            let count=counts.get_mut(identity).ok_or_else(||self.failure(Cause::Geometry))?;
            *count=count.checked_add(1).ok_or_else(||self.failure(Cause::Overflow))?;
            ids.push(identity);Ok(())
        };
        macro_rules! read {($ty:ty)=>{{
            let values=evaluated.try_iter::<$ty>().map_err(|e|self.failure(Cause::Read(e)))?;
            if Some(values.len())!=census.values(){return Err(self.failure(Cause::Geometry));}
            for value in values {accept(usize::try_from(value).ok())?;}
        }};}
        match indices.as_array().dtype(){Dtype::Int32=>read!(i32),Dtype::Uint32=>read!(u32),
            Dtype::Int64=>read!(i64),Dtype::Uint64=>read!(u64),_=>return Err(self.failure(Cause::Geometry))}
        let mut demands=Vec::new();demands.try_reserve_exact(census.maximum_members())
            .map_err(|e|self.failure(Cause::Allocation(e)))?;
        for (id,count) in counts.into_iter().enumerate(){if count!=0{demands.push((id,count));}}
        if demands.len()>census.maximum_members(){return Err(self.failure(Cause::Geometry));}
        let completed=SharedStorageOwner::new(CompletedIds{ids,source:b.identity.clone()}).erase();
        b.discovered.set(true);
        Ok(IndexedDemandSource::from_prepared(demands,completed,b.funding.clone()))
    }
    pub(super) fn remap(&self,binding:Option<&IndexedBankBinding>,indices:&MlxTensor,
        mapping:&[(usize,usize)],source:&IndexedDemandSource,stream:&Stream)->Result<MlxTensor,Error>{
        let b=self.body();let census=b.identity.census;
        self.validate(binding,indices.as_array(),census,stream)?;
        let completed=source.source().and_then(|s|s.downcast_ref::<CompletedIds>())
            .ok_or_else(||self.failure(Cause::Identity))?;
        if !completed.source.same_owner(&b.identity) || !source.funding().is_some_and(|f|f.same_account(&b.funding))
            || !b.discovered.get() || b.remap_attempted.replace(true) {return Err(self.failure(Cause::Spent));}
        if mapping.len()!=source.demands().len() || mapping.iter().zip(source.demands()).enumerate()
            .any(|(compact,(&(id,actual),&(expected,_)))|id!=expected||actual!=compact){return Err(self.failure(Cause::Geometry));}
        let mut values=Vec::new();values.try_reserve_exact(completed.ids.len())
            .map_err(|e|self.failure(Cause::Allocation(e)))?;
        for &id in &completed.ids {
            let position=mapping.binary_search_by_key(&id,|entry|entry.0).map_err(|_|self.failure(Cause::Geometry))?;
            values.push(i32::try_from(mapping[position].1).map_err(|_|self.failure(Cause::Geometry))?);
        }
        let shape=[i32::try_from(census.rows()).map_err(|_|self.failure(Cause::Geometry))?,
            i32::try_from(census.routes()).map_err(|_|self.failure(Cause::Geometry))?];
        let plan=OwnedHostCopyPlan::<i32>::new(&b.runtime,&shape,completed.ids.len())
            .map_err(|e|self.failure(Cause::Copy(e)))?;
        if plan.facts().metadata_bytes()!=b.layout.copy.metadata_bytes()
            || plan.facts().backing_bytes()!=b.layout.copy.backing_bytes(){return Err(self.failure(Cause::Identity));}
        let quota=PreparedSubmissionGraphQuota::try_new(plan.facts().metadata_bytes(),CopyCustody{funding:b.funding.clone()})
            .map_err(|e|{let(cause,owner)=e.into_parts();let error=self.failure(Cause::Graph(cause));drop(owner);error})?;
        let slot=plan.prepare(quota).map_err(|e|{let(cause,owner)=e.into_parts();
            let error=self.failure(Cause::Copy(cause));drop(owner);error})?;
        let output=slot.try_fill(values,&b.observer).map_err(|mut e|{
            let error=match e.take_native_source(){Some(cause)=>self.failure(Cause::Native(cause)),
                None=>self.failure(Cause::Copy(e.cause()))};drop(e);error})?;
        safemlx::OperationEvent::validate_traversal_leaf(&output,&b.observer)
            .map_err(|e|self.failure(Cause::Native(e)))?;
        self.validate_completion_source("remapped route context","remapped route bank")?;
        b.remapped.set(true);
        Ok(MlxTensor::from_array(output))
    }
}
impl MlxIndexedMovement {
    /// Prepares one source under an explicitly supplied original invocation.
    /// The parent reserves the one completion reported by IndexedChunkLayout;
    /// this constructor funds only its exact host/source population.
    pub(crate) fn prepare_original_chunk(&self,census:AddressableChunkCensus,input:&MlxTensor,
        runtime:&PreparedInputRuntime,observer:&OriginalScopeObserver,stream:&Stream,
        funding:&WorkspaceMetadataFunding)->Result<OriginalIndexedChunkSource,Error>{
        let binding=self.binding.as_ref().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?;
        let fail=|cause|failed(cause,&binding.bank,funding,None);
        let fixed=[size_of::<Result<IndexedChunkLayout,Error>>(),size_of::<IndexedChunkLayout>(),
            size_of::<(&Self,AddressableChunkCensus,&MlxTensor,&PreparedInputRuntime,&OriginalScopeObserver,&Stream)>(),
            eredu_nn::Error::retained_source_control_bytes::<Failure>().ok_or_else(||fail(Cause::Overflow))?,
            OriginalScopeObserver::control_bytes().ok_or_else(||fail(Cause::Overflow))?];
        funding.reserve_metadata(fixed.into_iter().try_fold(size_of_val(&fixed),usize::checked_add)
            .ok_or_else(||fail(Cause::Overflow))?).map_err(|e|fail(Cause::Funding(e)))?;
        let actual=OriginalScopeObserver::require_current().map_err(|e|fail(Cause::Native(e)))?;
        if self.original.is_some() || !observer.same_scope(&actual) || input.as_array().shape()!=[
            i32::try_from(census.rows()).map_err(|_|fail(Cause::Geometry))?,
            i32::try_from(census.routes()).map_err(|_|fail(Cause::Geometry))?]
            || !matches!(input.as_array().dtype(),Dtype::Int32|Dtype::Uint32|Dtype::Int64|Dtype::Uint64) {
            return Err(fail(Cause::Identity));
        }
        let parameter_revision=binding.bank.with_workspace_source(funding,|source|{
            let (members,maximum)=source.unit_population(census.bank(),census.unit()).ok_or_else(||fail(Cause::Geometry))?;
            let expected=AddressableChunkPlan::new(census.total_rows(),census.routes(),members,census.access(),
                Some(maximum),binding.options.prefill_compact_bank_target_bytes()).map_err(|_|fail(Cause::Geometry))?;
            if !source.same_source(&binding.bank)||expected!=census.plan(){return Err(fail(Cause::Identity));}
            Ok::<_,Error>(source.parameter_revision())
        }).map_err(|e|fail(Cause::Bank(e)))??;
        let layout=IndexedChunkLayout::inspect(runtime,census).ok_or_else(||fail(Cause::Geometry))?;
        let stream=StreamCopyPlan::<CopyCustody>::capture(stream).map_err(|_|fail(Cause::Identity))?;
        let bytes=layout.host_bytes().checked_add(stream.control_bytes().ok_or_else(||fail(Cause::Overflow))?)
            .ok_or_else(||fail(Cause::Overflow))?;
        funding.reserve_metadata(bytes).map_err(|e|fail(Cause::Funding(e)))?;
        // This final handle is independently paid here. It does not consume a
        // parent graph shell or invoke the ordinary native setter.
        let mut slot=safemlx::PreparedArrayClone::try_prepare_for_inspection()
            .map_err(|e|fail(Cause::Clone(e)))?;
        let input=slot.fill_for_inspection(input.as_array()).map_err(|e|fail(Cause::Clone(e)))?;
        let identity=SharedStorageOwner::new(SourceIdentity{input,bank:binding.bank.clone(),census,bulk_target_bytes:binding.options.prefill_compact_bank_target_bytes(),parameter_revision,funding:funding.clone()});
        Ok(OriginalIndexedChunkSource(Some(Rc::new(Body {identity,runtime:runtime.inspection_alias(),
            observer:observer.clone(),stream,layout,discover_attempted:Cell::new(false),discovered:Cell::new(false),remap_attempted:Cell::new(false),remapped:Cell::new(false),
            closed:Cell::new(false),acquire_attempted:Cell::new(false),acquired:Cell::new(false),binding_attempted:Cell::new(false),bound:Cell::new(false),completion_attempted:Cell::new(false),completed:Cell::new(false),residency:std::cell::RefCell::new(None),route_copies:Cell::new(0),funding:funding.clone()}))))
    }
    /// Lends the exact source to the existing discovery/remap methods. No source
    /// is found through TLS, and failed or duplicate chunks remain spent.
    pub(crate) fn with_original_chunk<T,F>(&mut self,source:OriginalIndexedChunkSource,
        run:F)->Result<T,Error> where F:FnOnce(&mut Self)->Result<T,Error>{
        let frames=[size_of::<T>(),size_of::<F>(),size_of::<Result<T,Error>>(),
            size_of::<OriginalIndexedChunkSource>(),size_of::<Option<OriginalIndexedChunkSource>>(),
            size_of::<&mut Self>(),eredu_nn::Error::retained_source_control_bytes::<Failure>()
                .ok_or_else(||source.failure(Cause::Overflow))?];
        source.body().funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||source.failure(Cause::Overflow))?).map_err(|e|source.failure(Cause::Funding(e)))?;
        if self.original.is_some() || source.body().closed.get(){return Err(source.failure(Cause::Spent));}
        struct Restore<'a>(&'a mut MlxIndexedMovement);
        impl Drop for Restore<'_>{fn drop(&mut self){if let Some(source)=self.0.original.take(){source.body().closed.set(true);}}}
        self.original=Some(source.clone());
        let guard=Restore(self);
        let result=run(&mut *guard.0);
        let complete=source.body().discovered.get()&&source.body().remapped.get()
            && (!source.body().acquire_attempted.get()||source.body().completed.get());
        drop(guard);
        match result {
            Ok(value) if complete=>Ok(value),
            Ok(_)=>Err(source.failure(Cause::Identity)),
            Err(cause)=>Err(source.failure(Cause::Consumer(cause))),
        }
    }
}

#[path = "indexed_source/acquisition.rs"]
mod acquisition;

#[path = "indexed_source/compact.rs"]
mod compact;

#[path = "indexed_source/construction.rs"]
mod construction;

#[path = "indexed_source/slots.rs"]
mod slots;

#[path = "indexed_source/invocation.rs"]
mod invocation;
pub(crate) use invocation::{IndexedResidencyPlan,IndexedConstructorPartitions,OriginalIndexedResidencyInvocation,OriginalIndexedResidencyFactory};
