//! Per-equation source programs consumed after the existing speculative claim.
//! Descriptions and accepted roots remain separate; every actual prefill span
//! gets its own one-use constructor/read program and its exact target child.
use super::{AddressableRequestOwner,AddressableExecutionRow};
use super::request_sources::{AddressableRequestSourcePlan,AddressableRequestSources};
use crate::backend::{error::Error,nn::workspace::{AddressableInvocation,ResidentSpanRecipe},
    runtime::execution::generic::OriginalSelectedResidencyAccess};
use eredu_nn::workspace::{WorkspaceMetadataFunding,WorkspaceMetadataFundingError};
use eredu_runtime::working_memory::{HostSourceConstructionFacts,OriginalHostSourceBank,WorkingMemoryError};
use safemlx::{OriginalScopeObserver,OriginalBufferBudget};
use std::{cell::{Cell,RefCell},mem::{size_of,size_of_val}};
fn overflow()->Error{Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow)}
fn identity()->Error{Error::PrefillControl(WorkingMemoryError::IdentityMismatch)}
fn sum(values:&[usize])->Option<usize>{values.iter().copied().try_fold(size_of_val(values),usize::checked_add)}
struct Row {taken:Cell<bool>,value:RefCell<Option<SpeculativeAddressableSpan>>}
pub(crate) struct SpeculativeAddressableSources {
    rows:Vec<Row>,facts:Option<Vec<Option<HostSourceConstructionFacts>>>,controls:u64,
    _funding:WorkspaceMetadataFunding,
}
pub(crate) struct SpeculativeAddressableSpan {
    plan:Option<AddressableRequestSourcePlan>,
    invocation:AddressableInvocation,
    accepted:Option<AddressableRequestSources>,
    funding:WorkspaceMetadataFunding,
}
impl SpeculativeAddressableSources {
    pub(crate) fn prepare(records:&[ResidentSpanRecipe],target:Option<HostSourceConstructionFacts>,
        funding:&WorkspaceMetadataFunding)->Result<Option<Self>,Error>{
        if !records.iter().any(|row|row.addressable().is_some()){return Ok(None);}
        let fixed=sum(&[size_of::<Self>(),size_of::<Result<Option<Self>,Error>>(),
            size_of::<(&[ResidentSpanRecipe],Option<HostSourceConstructionFacts>,&WorkspaceMetadataFunding)>(),
            size_of::<Vec<Row>>(),size_of::<Vec<Option<HostSourceConstructionFacts>>>(),
            size_of::<AddressableRequestSourcePlan>(),size_of::<SpeculativeAddressableSpan>(),
            size_of::<AddressableInvocation>(),size_of::<[(&AddressableInvocation,usize);1]>(),
            size_of::<Option<HostSourceConstructionFacts>>(),size_of::<usize>()*4,size_of::<u64>()*3,
            size_of::<std::slice::Iter<'_,ResidentSpanRecipe>>()]).ok_or_else(overflow)?;
        funding.reserve_metadata(fixed).map_err(Error::WorkspacePlanning)?;
        let mut rows=funding.metadata_vec(records.len()).map_err(Error::Neural)?;
        let mut facts=funding.metadata_vec(records.len()).map_err(Error::Neural)?;
        let mut controls=0u64;
        for record in records {
            let value=if let Some(invocation)=record.addressable(){
                let selected=[(invocation,1usize)];
                let plan=AddressableRequestSourcePlan::prepare(&selected,target,None,funding)?;
                facts.push(Some(plan.facts()));
                let wrappers=AddressableRequestOwner::prepared_runtime_control_bytes(&selected)
                    .and_then(|n|n.checked_add(SpeculativeAddressableSpan::control_bytes()?)).ok_or_else(overflow)?;
                let wrappers=wrappers.checked_add(plan.runtime_control_bytes()?).ok_or_else(overflow)?;
                controls=controls.checked_add(u64::try_from(wrappers).map_err(|_|overflow())?).ok_or_else(overflow)?;
                // Graph/Record owners are allocated by each dedicated region
                // outside the model graph; Buffer births are already in its
                // numerical storage population and must not be added twice.
                for (_,quote) in invocation.occurrences(){
                    let arenas=quote.capacity.graph.checked_add(quote.capacity.records).ok_or_else(overflow)?;
                    controls=controls.checked_add(u64::try_from(arenas).map_err(|_|overflow())?).ok_or_else(overflow)?;
                }
                Some(SpeculativeAddressableSpan{plan:Some(plan),
                    invocation:invocation.try_clone_for_retention().map_err(Error::Neural)?,accepted:None,funding:funding.clone()})
            }else{facts.push(target);None};
            rows.push(Row{taken:Cell::new(false),value:RefCell::new(value)});
        }
        Ok(Some(Self{rows,facts:Some(facts),controls,_funding:funding.clone()}))
    }
    pub(crate) fn controls(&self)->u64{self.controls}
    pub(crate) fn take_facts(&mut self)->Result<Vec<Option<HostSourceConstructionFacts>>,Error>{
        self.facts.take().ok_or_else(identity)
    }
    pub(crate) fn take(&self,index:usize)->Result<Option<SpeculativeAddressableSpan>,Error>{
        let row=self.rows.get(index).ok_or_else(identity)?;
        if row.taken.replace(true){return Err(identity());}
        row.value.try_borrow_mut().map_err(|_|identity()).map(|mut value|value.take())
    }
}
impl SpeculativeAddressableSpan {
    fn control_bytes()->Option<usize>{
        sum(&[size_of::<Self>(),size_of::<Option<Self>>(),
            size_of::<Option<AddressableRequestSources>>(),size_of::<AddressableRequestSources>(),
            size_of::<Option<AddressableRequestSourcePlan>>(),
            size_of::<Option<OriginalHostSourceBank>>(),size_of::<Result<Option<OriginalHostSourceBank>,Error>>(),
            size_of::<(&mut Self,Option<OriginalHostSourceBank>)>(),
            size_of::<(Self,OriginalSelectedResidencyAccess,OriginalBufferBudget,&OriginalScopeObserver,Option<std::time::Duration>,&WorkspaceMetadataFunding)>(),
            size_of::<AddressableRequestOwner>(),size_of::<AddressableExecutionRow>(),
            size_of::<Result<AddressableExecutionRow,Error>>(),size_of::<&mut Self>()])
    }
    /// Called only by the selected operation bank after its cumulative claim.
    pub(crate) fn accept(&mut self,root:Option<OriginalHostSourceBank>)->Result<Option<OriginalHostSourceBank>,Error>{
        self.funding.reserve_metadata(Self::control_bytes().ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)?;
        if self.accepted.is_some(){return Err(identity());}
        let source=self.plan.take().ok_or_else(identity)?.accept(root.ok_or_else(identity)?)?;
        let target=source.take_target()?;
        self.accepted=Some(source);
        Ok(target)
    }
    pub(crate) fn activate(self,access:OriginalSelectedResidencyAccess,budget:OriginalBufferBudget,
        observer:&OriginalScopeObserver,timeout:Option<std::time::Duration>,funding:&WorkspaceMetadataFunding)
        ->Result<AddressableExecutionRow,Error>{
        if self.plan.is_some()||!self.funding.same_account(funding){return Err(identity());}
        let owner=AddressableRequestOwner::new_prepared(self.accepted.ok_or_else(identity)?,
            access,budget,observer,timeout,funding)?;
        owner.row(0,0,self.invocation)
    }
}

