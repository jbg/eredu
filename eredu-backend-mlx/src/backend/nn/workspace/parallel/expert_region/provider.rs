//! The architecture-declared provider votes use the same selected group sources.
use super::*;
use crate::backend::nn::workspace::LogicalCollectiveQuote;
use crate::backend::runtime::distributed::topology::original_source::parallel::RetainedLogicalCollective;
pub(crate) struct ExpertProviderVote {
    pub(crate) group:eredu_core::CollectiveGroupId,
    pub(crate) order:usize,
    pub(crate) peers:usize,
    pub(crate) backing:usize,
    pub(crate) logical:Option<RetainedLogicalCollective>,
}
pub(crate) struct ExpertProviderQuote {
    votes:Vec<ExpertProviderVote>,
    pub(crate) backing:u64,
    pub(crate) births:usize,
}
impl ExpertProviderQuote {
    pub(super) fn prepare(local:&ExpertLocalQuote)->Result<Self,Error>{
        let view=local.declaration.as_view();
        Self::prepare_for(&local.source,local.mechanism,[view.provider_tensor_group,Some(view.group),view.provider_wave_group])
    }
    pub(super) fn prepare_for(source:&OriginalParallelSource,mechanism:ResidentExecutionMechanisms,
        groups:[Option<eredu_core::CollectiveGroupId>;3])->Result<Self,Error>{
        let context=WorkspaceContext::new_with_metadata_funding(mechanism,source.funding().clone())?;
        context.charge_metadata(size_of::<(Self,ExpertProviderVote,Result<Self,Error>,WorkspaceContext,[i32;1],
            [Option<eredu_core::CollectiveGroupId>;3])>())?;
        let invalid=||context.metadata_error(format_args!("expert provider vote lacks its selected source"));
        let actual=source.communication_source().map_err(|cause|source.neural_error(cause))?;
        let mut votes=context.metadata_vec(groups.into_iter().flatten().count())?;
        let mut backing=0u64;let mut births=0usize;
        for id in groups.into_iter().flatten(){
            let selected=actual.source().manifest().select_group_operation(id,
                eredu_runtime::CommunicationOperation::FailureAgreement).map_err(|cause|context.metadata_source(cause))?;
            let order=selected.order();let (group,descriptor,_)=actual.group(order).ok_or_else(invalid)?;
            if !selected.requirement().exact_completion()||descriptor.local_index()!=Some(group.rank()){
                return Err(invalid());
            }
            let (bytes,created,logical)=if group.is_logical(){
                let layout=context.layout(&[1],WorkspaceDtype::Int32)?;
                let quote=LogicalCollectiveQuote::prepare_agreement(source,order,layout.as_view(),mechanism)?;
                let bytes=quote.output.checked_add(quote.scratch).and_then(|n|usize::try_from(n).ok()).ok_or_else(invalid)?;
                let created=quote.maximum_backing_births().ok_or_else(invalid)?;
                let quote=RetainedLogicalCollective::retain_prepared(source,quote).map_err(|cause|source.neural_error(cause))?;
                (bytes,created,Some(quote))
            }else{
                let persistent=actual.group_persistent(order).map_err(|cause|source.neural_error(cause))?;
                let native=group.native_group();
                if persistent.native().has_unqualified_storage()||!persistent.native().is_for(native)
                    ||!persistent.source().same_source(actual.source()){return Err(invalid());}
                context.charge_metadata(native.cpu_layout_storage_control_bytes().ok_or_else(invalid)?)?;
                let layout=native.cpu_layout_storage(&[1],safemlx::Dtype::Int32,
                    safemlx::distributed::GroupWorkerOperation::Sum).map_err(|cause|context.metadata_source(cause))?;
                context.charge_metadata(layout.backing_control_bytes().ok_or_else(invalid)?)?;
                let runtime=source.agreement_inputs().ok_or_else(invalid)?.runtime();
                (layout.backing_capacity(runtime).map_err(|cause|context.metadata_source(cause))?,
                    layout.evaluation().logical_backing_population().0,None)
            };
            backing=backing.checked_add(u64::try_from(bytes).map_err(|_|invalid())?).ok_or_else(invalid)?;
            births=births.checked_add(created).ok_or_else(invalid)?;
            votes.push(ExpertProviderVote{group:id,order,peers:group.size(),backing:bytes,logical});
        }
        Ok(Self{votes,backing,births})
    }
    pub(crate) fn len(&self)->usize{self.votes.len()}
    pub(crate) fn vote(&self,index:usize)->Option<&ExpertProviderVote>{self.votes.get(index)}
}
