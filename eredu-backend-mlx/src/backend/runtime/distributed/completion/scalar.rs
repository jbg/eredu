//! Source-paid scalar result cells consumed by the shared completion resolver.
use super::*;
use super::prepared::{Cause, ResourceCustody, ReadyCompletionResources, error};
use crate::backend::runtime::distributed::topology::OriginalCommunicationSource;
use eredu_nn::workspace::HostMetadataFundingError;
use std::{alloc::Layout,mem::{size_of,size_of_val},ops::Deref};

/// Every escaped alias retains account custody after the actual Rc allocation.
/// There are no Weak aliases, native roots or observer backedges in this cell.
#[derive(Clone,Debug)]
pub(super) struct BoolResult {
    value:Rc<Cell<Option<bool>>>,
    custody:Option<ResourceCustody>,
}
impl BoolResult {
    pub(super) fn ordinary()->Self { Self { value:Rc::new(Cell::new(None)),custody:None } }
    fn original(custody:ResourceCustody)->Self { Self { value:Rc::new(Cell::new(None)),custody:Some(custody) } }
    fn custody(&self)->&ResourceCustody { self.custody.as_ref().expect("prepared original result") }
}
impl BoolResult {
    pub(super) fn original_custody(&self)->Option<&ResourceCustody> {self.custody.as_ref()}
}
impl Deref for BoolResult {
    type Target=Cell<Option<bool>>;
    fn deref(&self)->&Self::Target { &self.value }
}
#[derive(Clone,Copy,Debug)]
pub(super) enum ScalarKind { Flag, Agreement(i32) }
#[derive(Clone,Copy)]
enum Selector { Flag, Agreement(usize) }

/// Owns the finite scalar destination before the native role starts. The exact
/// scalar Array is supplied later by its actual communication producer.
pub(crate) struct PreparedCommunicationScalar { kind:ScalarKind, result:BoolResult }
/// A completed boolean can outlive the native completion, retaining only its
/// source/account and the paid result cell until extraction.
#[derive(Debug)]
pub(crate) struct OriginalCommunicationBool { result:BoolResult }
impl OriginalCommunicationBool {
    pub(crate) fn resolve(self)->Result<bool,Error> {
        self.result.get().ok_or_else(||error(Cause::Identity,self.result.custody()))
    }
    pub(crate) fn into_agreement(self)->MlxFailureAgreement {
        MlxFailureAgreement::original(self.result)
    }
    #[cfg(test)]
    fn pending(&self)->bool { self.result.get().is_none() }
}
impl PreparedCommunicationScalar {
    pub(crate) fn prepare_flag(source:&OriginalCommunicationSource<'_>)->Result<Self,Error> {
        Self::prepare(source,Selector::Flag)
    }
    /// The expected count comes from the same retained native group's declared
    /// membership, rather than a caller-supplied scalar or population estimate.
    pub(crate) fn prepare_agreement(source:&OriginalCommunicationSource<'_>,group:usize)->Result<Self,Error> {
        Self::prepare(source,Selector::Agreement(group))
    }
    fn prepare(source:&OriginalCommunicationSource<'_>,selector:Selector)->Result<Self,Error> {
        source.validate()?;
        source.funding().reserve_metadata(Self::control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let custody=ResourceCustody { source:source.source().clone(),funding:source.funding().clone() };
        let kind=match selector {
            Selector::Flag=>ScalarKind::Flag,
            Selector::Agreement(group)=>ScalarKind::Agreement(source.group(group)
                .and_then(|(_,descriptor,_)|i32::try_from(descriptor.members().len()).ok())
                .ok_or_else(||error(Cause::Identity,&custody))?),
        };
        Ok(Self { kind,result:BoolResult::original(custody) })
    }
    /// The owned scalar is the exact sole submitted root. No handle clone or
    /// late allocation occurs when it joins the shared result resolver.
    pub(crate) fn submit(self,ready:ReadyCompletionResources,source:&OriginalCommunicationSource<'_>,
        observer:&safemlx::OriginalScopeObserver,stream:&Stream,output:Array)
        ->Result<(OriginalCommunicationBool,OriginalCommunicationCompletion),Error> {
        let expected=match self.kind { ScalarKind::Flag=>safemlx::Dtype::Float32,ScalarKind::Agreement(_)=>safemlx::Dtype::Int32 };
        if !self.result.custody().source.same_source(source.source()) || output.dtype()!=expected || output.size()!=1 {
            return Err(error(Cause::Identity,self.result.custody()));
        }
        let mut completion=ready.submit_original(source,observer,stream,std::slice::from_ref(&output))?;
        completion.0.install_original_scalar(output,self.result.clone(),self.kind);
        Ok((OriginalCommunicationBool { result:self.result },completion))
    }
    /// Consume one constructor-minted root and its matching ready completion.
    /// Submission and scalar installation use the same accepted output; no raw
    /// output or independently selected event can enter this join.
    pub(crate) fn submit_accepted(self,
        accepted:crate::backend::runtime::distributed::topology::AcceptedCommunicationSource<'_>,
        ready:ReadyCompletionResources,
    )->Result<(OriginalCommunicationBool,OriginalCommunicationCompletion),Error> {
        if !self.result.custody().source.same_source(accepted.source())
            || accepted.outputs().len()!=1
            || accepted.outputs()[0].dtype()!=safemlx::Dtype::Int32
            || accepted.outputs()[0].size()!=1
            || !matches!(self.kind,ScalarKind::Agreement(_)) {
            return Err(error(Cause::Identity,self.result.custody()));
        }
        let (output,mut completion)=accepted.submit(ready)?;
        let (value,source,funding)=output.into_parts();
        completion.0.install_original_scalar(value,self.result.clone(),self.kind);
        let result=OriginalCommunicationBool{result:self.result};
        drop((source,funding));
        Ok((result,completion))
    }

    pub(crate) fn control_bytes()->Option<usize> {
        let allocation=Layout::new::<[usize;2]>().extend(Layout::new::<Cell<Option<bool>>>()).ok()?.0.pad_to_align().size();
        let frames=[allocation,size_of::<BoolResult>(),size_of::<Self>(),size_of::<OriginalCommunicationBool>(),
            size_of::<ScalarKind>(),size_of::<Selector>(),size_of::<ResourceCustody>(),
            size_of::<crate::backend::runtime::distributed::topology::OriginalCommunicationConstructed>(),
            size_of::<crate::backend::runtime::distributed::topology::AcceptedCommunicationSource<'_>>(),
            size_of::<ReadyCompletionResources>(),
            size_of::<Result<(crate::backend::runtime::distributed::topology::OriginalCommunicationConstructed,OriginalCommunicationCompletion),Error>>(),
            size_of::<(Array,eredu_runtime::RetainedCommunicationSource,eredu_nn::workspace::HostMetadataFunding)>(),
            size_of::<Result<Self,Error>>(),size_of::<Result<bool,Error>>(),
            size_of::<Result<(OriginalCommunicationBool,OriginalCommunicationCompletion),Error>>(),
            size_of::<(OriginalCommunicationBool,OriginalCommunicationCompletion)>(),size_of::<Option<bool>>(),
            size_of::<(&OriginalCommunicationSource<'_>,usize)>(),size_of::<safemlx::Dtype>(),
            size_of::<(&safemlx::OriginalScopeObserver,&Stream,Array)>(),
            size_of::<(&Array,&MlxCommunicationCompletion)>(),
            size_of::<safemlx::EvaluatedArray<'_>>(),
            size_of::<Result<safemlx::EvaluatedArray<'_>,safemlx::error::Exception>>(),
            size_of::<Result<&[i32],safemlx::error::AsSliceError>>(),size_of::<Result<&[f32],safemlx::error::AsSliceError>>(),
            size_of::<std::fmt::Arguments<'_>>(),size_of::<safemlx::error::AsSliceError>(),
            safemlx::OriginalScopeObserver::control_bytes()?,
            safemlx::EvaluatedArray::iteration_control_bytes::<i32>()?,
            safemlx::EvaluatedArray::iteration_control_bytes::<f32>()?,
            // Shared fixed source-failure constructor, including its retained H.
            super::prepared::error_control_bytes()?,
        ];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}

#[cfg(test)]
mod tests;
