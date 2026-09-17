//! Finite word/header destinations for the existing completed-result resolver.
use super::*;
use super::prepared::{Cause,ResourceCustody,ReadyCompletionResources,error};
use crate::backend::runtime::distributed::topology::OriginalCommunicationSource;
use eredu_nn::workspace::WorkspaceMetadataFundingError;
use std::{alloc::Layout,mem::{size_of,size_of_val},ops::{Deref,DerefMut}};

#[derive(Debug)]
struct WordState<T> { ready:Option<Vec<T>>, pending:Option<Vec<T>> }
#[derive(Clone,Debug)]
pub(crate) struct WordsResult<T=i32> {
    value:Rc<RefCell<WordState<T>>>,
    custody:Option<ResourceCustody>,
}
impl<T:Copy> WordsResult<T> {
    pub(super) fn ordinary()->Self {
        Self { value:Rc::new(RefCell::new(WordState { ready:None,pending:None })),custody:None }
    }
    fn original(values:Vec<T>,custody:ResourceCustody)->Self {
        Self { value:Rc::new(RefCell::new(WordState { ready:None,pending:Some(values) })),custody:Some(custody) }
    }
    fn custody(&self)->&ResourceCustody { self.custody.as_ref().expect("original result source") }
    pub(super) fn take(&self)->Option<Vec<T>> { self.value.borrow_mut().ready.take() }
    pub(super) fn store(&self,values:&[T])->bool {
        let mut state=self.value.borrow_mut();
        if self.custody.is_none() { state.ready=Some(values.to_vec());return true; }
        if let Some(target)=state.ready.as_mut() {
            if target.len()!=values.len() { return false; }
            target.copy_from_slice(values);return true;
        }
        if let Some(mut target)=state.pending.take() {
            if target.len()!=values.len() { state.pending=Some(target);return false; }
            target.copy_from_slice(values);state.ready=Some(target);
        }
        // A previously consumed original receipt is not repopulated by polling.
        true
    }
}
/// Shared completed view decoder used by event resolution and a final
/// already-completed numerical child. Neither caller evaluates or waits here.
pub(super) fn store_words<T: CommunicationWord>(evaluated: &safemlx::EvaluatedArray<'_>,
    result: &WordsResult<T>) -> Result<bool, safemlx::error::AsSliceError> {
    Ok(result.store(evaluated.try_as_slice::<T>()?))
}
/// Outer custody outlives both the row elements and the Vec allocation itself.
#[derive(Debug,Default)]
pub(super) struct BoundaryHeaders {
    rows:Vec<communication::BoundaryHeaderResolution>,
    custody:Option<ResourceCustody>,
}
impl Deref for BoundaryHeaders {
    type Target=Vec<communication::BoundaryHeaderResolution>;
    fn deref(&self)->&Self::Target { &self.rows }
}
impl DerefMut for BoundaryHeaders { fn deref_mut(&mut self)->&mut Self::Target { &mut self.rows } }
impl<'a> IntoIterator for &'a BoundaryHeaders {
    type Item=&'a communication::BoundaryHeaderResolution;
    type IntoIter=std::slice::Iter<'a,communication::BoundaryHeaderResolution>;
    fn into_iter(self)->Self::IntoIter { self.rows.iter() }
}

// Only the two existing wire representations use this shared destination.
mod sealed {
    pub trait Word {}
    impl Word for i32 {}
    impl Word for u32 {}
}
pub(crate) trait CommunicationWord: sealed::Word + safemlx::ArrayElement + Copy + Default + std::fmt::Debug {
    fn install(completion:&mut MlxCommunicationCompletion,output:Array,result:WordsResult<Self>);
}
impl CommunicationWord for i32 {
    fn install(completion:&mut MlxCommunicationCompletion,output:Array,result:WordsResult<Self>) {
        completion.install_original_words(output,result);
    }
}
impl CommunicationWord for u32 {
    fn install(completion:&mut MlxCommunicationCompletion,output:Array,result:WordsResult<Self>) {
        completion.install_original_u32_words(output,result);
    }
}
/// The optional existing output preserves the ordinary signed-word adapter;
/// source-derived operation destinations are prepared before construction.
pub(crate) struct PreparedWordDestination<T> { output:Option<Array>,result:WordsResult<T> }
#[derive(Debug)]
pub(crate) struct OriginalWordDestination<T> { result:WordsResult<T> }
/// Host data stays paid after extraction from the shared completion cell.
#[derive(Debug)]
pub(crate) struct CompletedWordDestination<T> { values:Vec<T>,_custody:ResourceCustody }
pub(crate) type PreparedCommunicationWords=PreparedWordDestination<i32>;
pub(crate) type OriginalCommunicationWords=OriginalWordDestination<i32>;
pub(crate) type CompletedCommunicationWords=CompletedWordDestination<i32>;
pub(crate) type PreparedCommunicationU32Words=PreparedWordDestination<u32>;
pub(crate) type OriginalCommunicationU32Words=OriginalWordDestination<u32>;
pub(crate) type CompletedCommunicationU32Words=CompletedWordDestination<u32>;
impl<T> CompletedWordDestination<T> { pub(crate) fn as_slice(&self)->&[T] { &self.values } }
impl<T:Copy> OriginalWordDestination<T> {
    pub(crate) fn resolve(self)->Result<CompletedWordDestination<T>,Error> {
        let values=self.result.take().ok_or_else(||error(Cause::Identity,self.result.custody()))?;
        Ok(CompletedWordDestination { values,_custody:self.result.custody().clone() })
    }
}
impl<T:CommunicationWord> PreparedWordDestination<T> {
    /// The actual owned output supplies exact dtype and word population. Only
    /// its host destination is born here; its native producer is already paid.
    pub(crate) fn prepare(source:&OriginalCommunicationSource<'_>,output:Array)->Result<Self,Error> {
        let custody=prepare_custody::<T>(source)?;
        if output.dtype()!=T::DTYPE { return Err(error(Cause::Identity,&custody)); }
        let result=Self::destination(output.size(),custody)?;
        Ok(Self { output:Some(output),result })
    }
    /// Quotes the host destination from the same retained native constructor,
    /// before its output exists. No caller word count or capacity is accepted.
    pub(crate) fn prepare_operation(source:&OriginalCommunicationSource<'_>,
        operation:&crate::backend::runtime::distributed::topology::OriginalCommunicationCompletedOperation<'_>,
    )->Result<Self,Error> {
        let custody=prepare_custody::<T>(source)?;
        if !operation.source().same_source(source.source()) {
            return Err(error(Cause::Identity,&custody));
        }
        let constructor=operation.native().operation().constructor();
        if constructor.output_dtype()!=T::DTYPE {
            return Err(error(Cause::Identity,&custody));
        }
        let result=Self::destination(constructor.output_geometry().1,custody)?;
        Ok(Self { output:None,result })
    }
    fn destination(length:usize,custody:ResourceCustody)->Result<WordsResult<T>,Error> {
        let mut values=vector(length,&custody)?;values.resize(length,T::default());
        Ok(WordsResult::original(values,custody))
    }
    pub(crate) fn submit(self,ready:ReadyCompletionResources,source:&OriginalCommunicationSource<'_>,
        observer:&safemlx::OriginalScopeObserver,stream:&Stream)
        ->Result<(OriginalWordDestination<T>,OriginalCommunicationCompletion),Error> {
        if !self.result.custody().source.same_source(source.source()) { return Err(error(Cause::Identity,self.result.custody())); }
        let output=self.output.ok_or_else(||error(Cause::Identity,self.result.custody()))?;
        let mut completion=ready.submit_original(source,observer,stream,std::slice::from_ref(&output))?;
        T::install(&mut completion.0,output,self.result.clone());
        Ok((OriginalWordDestination { result:self.result },completion))
    }
    /// Resolve the actual completed output inside its original final child.
    /// The exact observer checks availability and ownership; the existing paid
    /// word decoder fills its one destination without a second Eval/event.
    pub(crate) fn resolve_completed(self, observer: &safemlx::OriginalScopeObserver)
        -> Result<CompletedWordDestination<T>, Error> {
        let output = self.output.as_ref().ok_or_else(|| error(Cause::Identity, self.result.custody()))?;
        Self::resolve_view(output, self.result, observer)
    }
    /// The exact completed source may lend its native output instead of making
    /// another Array handle solely for readout. The same destination and decoder
    /// are used, and its returned host custody is independent of that borrow.
    pub(crate) fn read_completed(source: &OriginalCommunicationSource<'_>, output: &Array,
        observer: &safemlx::OriginalScopeObserver) -> Result<CompletedWordDestination<T>, Error> {
        let custody = prepare_custody::<T>(source)?;
        if output.dtype() != T::DTYPE { return Err(error(Cause::Identity, &custody)); }
        let result = Self::destination(output.size(), custody)?;
        Self::resolve_view(output, result, observer)
    }
    fn resolve_view(output: &Array, result: WordsResult<T>, observer: &safemlx::OriginalScopeObserver)
        -> Result<CompletedWordDestination<T>, Error> {
        let custody = result.custody();
        let evaluated = output.completed_in_original_scope(observer)
            .map_err(|cause| error(Cause::Native(cause), custody))?;
        if !store_words(&evaluated, &result).map_err(|_| error(Cause::Identity, custody))? {
            return Err(error(Cause::Identity, custody));
        }
        drop(evaluated);
        OriginalWordDestination { result }.resolve()
    }
    /// Joins one constructor-minted root to its prepaid host destination and
    /// matching event. Accepted work requires no fresh availability query.
    pub(crate) fn submit_accepted(self,
        accepted:crate::backend::runtime::distributed::topology::AcceptedCommunicationSource<'_>,
        ready:ReadyCompletionResources,
    )->Result<(OriginalWordDestination<T>,OriginalCommunicationCompletion),Error> {
        if self.output.is_some() || !self.result.custody().source.same_source(accepted.source())
            || accepted.outputs().len()!=1 || accepted.outputs()[0].dtype()!=T::DTYPE
            || self.result.value.borrow().pending.as_ref().map(Vec::len)!=Some(accepted.outputs()[0].size()) {
            return Err(error(Cause::Identity,self.result.custody()));
        }
        let (output,mut completion)=accepted.submit(ready)?;
        let (value,source,funding)=output.into_parts();
        T::install(&mut completion.0,value,self.result.clone());
        let result=OriginalWordDestination{result:self.result};
        drop((source,funding));
        Ok((result,completion))
    }
}

/// One in-band header belongs to one actual boundary transfer/event. Its exact
/// immutable expectation is copied before submission and retained through read.
pub(crate) struct PreparedCommunicationHeader { headers:BoundaryHeaders }
impl PreparedCommunicationHeader {
    pub(crate) fn prepare(source:&OriginalCommunicationSource<'_>,received:Array,expected:&[u8])->Result<Self,Error> {
        let custody=prepare_custody::<i32>(source)?;
        if received.dtype()!=safemlx::Dtype::Uint8 || received.size()!=expected.len() {
            return Err(error(Cause::Identity,&custody));
        }
        let mut bytes=vector(expected.len(),&custody)?;bytes.extend_from_slice(expected);
        let mut rows=vector(1,&custody)?;
        rows.push(communication::BoundaryHeaderResolution { received,expected:bytes });
        Ok(Self { headers:BoundaryHeaders { rows,custody:Some(custody) } })
    }
    /// The selected payload and its actual received header share one exact
    /// completion. The existing resolver still compares in-band bytes before
    /// this owner can report success.
    pub(crate) fn submit_with_payload(self,ready:ReadyCompletionResources,source:&OriginalCommunicationSource<'_>,
        observer:&safemlx::OriginalScopeObserver,stream:&Stream,payload:&Array)->Result<OriginalCommunicationCompletion,Error>{
        let custody=self.headers.custody.as_ref().expect("original header source");
        custody.funding.reserve_metadata(size_of::<[&Array;2]>() + size_of::<Option<&Array>>())
            .map_err(|cause|error(Cause::Funding(cause),custody))?;
        if !custody.source.same_source(source.source()){return Err(error(Cause::Identity,custody));}
        let roots=[payload,&self.headers.rows[0].received];
        let mut completion=ready.submit_original(source,observer,stream,roots)?;
        completion.0.install_original_header(self.headers);Ok(completion)
    }
    pub(crate) fn submit(self,ready:ReadyCompletionResources,source:&OriginalCommunicationSource<'_>,
        observer:&safemlx::OriginalScopeObserver,stream:&Stream)->Result<OriginalCommunicationCompletion,Error> {
        let custody=self.headers.custody.as_ref().expect("original header source");
        if !custody.source.same_source(source.source()) { return Err(error(Cause::Identity,custody)); }
        let mut completion=ready.submit_original(source,observer,stream,std::slice::from_ref(&self.headers.rows[0].received))?;
        completion.0.install_original_header(self.headers);Ok(completion)
    }
}
fn prepare_custody<T:CommunicationWord>(source:&OriginalCommunicationSource<'_>)->Result<ResourceCustody,Error> {
    source.validate()?;
    source.funding().reserve_metadata(control_bytes::<T>().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    Ok(ResourceCustody { source:source.source().clone(),funding:source.funding().clone() })
}
fn vector<T>(length:usize,custody:&ResourceCustody)->Result<Vec<T>,Error> {
    let bytes=Layout::array::<T>(length).map_err(|_|error(Cause::Identity,custody))?.size();
    custody.funding.reserve_metadata(bytes).map_err(|cause|error(Cause::Funding(cause),custody))?;
    let mut values=Vec::new();values.try_reserve_exact(length).map_err(|cause|error(Cause::Capacity(cause),custody))?;Ok(values)
}
fn control_bytes<T:CommunicationWord>()->Option<usize> {
    let allocation=Layout::new::<[usize;2]>().extend(Layout::new::<RefCell<WordState<T>>>()).ok()?.0.pad_to_align().size();
    let frames=[allocation,size_of::<WordState<T>>(),size_of::<WordsResult<T>>(),size_of::<BoundaryHeaders>(),
        size_of::<PreparedWordDestination<T>>(),size_of::<OriginalWordDestination<T>>(),size_of::<CompletedWordDestination<T>>(),
        size_of::<PreparedCommunicationHeader>(),size_of::<communication::BoundaryHeaderResolution>(),
        size_of::<ResourceCustody>(),size_of::<Result<ResourceCustody,Error>>(),
        size_of::<crate::backend::runtime::distributed::topology::OriginalCommunicationConstructed>(),
        size_of::<crate::backend::runtime::distributed::topology::AcceptedCommunicationSource<'_>>(),
        size_of::<ReadyCompletionResources>(),size_of::<Option<Array>>(),
        size_of::<Result<WordsResult<T>,Error>>(),size_of::<(usize,ResourceCustody)>(),
        size_of::<(&OriginalCommunicationSource<'_>,&crate::backend::runtime::distributed::topology::OriginalCommunicationCompletedOperation<'_>)>(),
        size_of::<Result<(crate::backend::runtime::distributed::topology::OriginalCommunicationConstructed,OriginalCommunicationCompletion),Error>>(),
        size_of::<(Array,eredu_runtime::RetainedCommunicationSource,eredu_nn::workspace::WorkspaceMetadataFunding)>(),
        size_of::<Result<PreparedWordDestination<T>,Error>>(),size_of::<Result<PreparedCommunicationHeader,Error>>(),
        size_of::<Result<(OriginalWordDestination<T>,OriginalCommunicationCompletion),Error>>(),
        size_of::<Result<CompletedWordDestination<T>,Error>>(),size_of::<Result<OriginalCommunicationCompletion,Error>>(),
        size_of::<Vec<T>>(),size_of::<Vec<u8>>(),size_of::<Vec<communication::BoundaryHeaderResolution>>(),
        size_of::<Result<Vec<T>,Error>>(),size_of::<Result<Vec<u8>,Error>>(),
        size_of::<Result<Vec<communication::BoundaryHeaderResolution>,Error>>(),
        size_of::<std::cell::RefMut<'_,WordState<T>>>(),size_of::<Option<Vec<T>>>(),
        size_of::<Result<(),std::collections::TryReserveError>>(),size_of::<Layout>(),
        size_of::<Result<Layout,std::alloc::LayoutError>>(),size_of::<(&ResourceCustody,usize)>(),
        size_of::<(&OriginalCommunicationSource<'_>,&Array,&[u8])>(),
        size_of::<(&safemlx::OriginalScopeObserver,&Stream)>(),size_of::<std::fmt::Arguments<'_>>(),
        size_of::<Result<safemlx::EvaluatedArray<'_>,safemlx::error::Exception>>(),
        size_of::<Result<&[T],safemlx::error::AsSliceError>>(),size_of::<Result<&[u8],safemlx::error::AsSliceError>>(),
        safemlx::OriginalScopeObserver::control_bytes()?,
        safemlx::EvaluatedArray::iteration_control_bytes::<T>()?,
        safemlx::EvaluatedArray::iteration_control_bytes::<u8>()?,
        prepared::error_control_bytes()?,
    ];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}

#[cfg(test)]
mod tests;
