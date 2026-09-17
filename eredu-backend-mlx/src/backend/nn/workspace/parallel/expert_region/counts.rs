//! Retained count itinerary, shared with the completed source reader.
use super::*;
use crate::backend::runtime::distributed::topology::original_source::parallel::RetainedLogicalCollective;
use crate::backend::nn::workspace::{LogicalCollectiveKind,LogicalCollectiveQuote};

struct CountStep { value:usize, step:usize, source:RetainedLogicalCollective }
enum CountItinerary {
    Physical,
    Logical(RetainedLogicalCollective),
    Routed(Vec<CountStep>),
}
pub(crate) struct ExpertCountQuote {
    order:usize,
    rank:usize,
    peers:usize,
    has_world:bool,
    itinerary:CountItinerary,
    pub(crate) backing:u64,
    pub(crate) births:usize,
}
impl ExpertCountQuote {
    pub(super) fn prepare(local:&ExpertLocalQuote)->Result<Self,Error> {
        let view=local.declaration.as_view();
        Self::prepare_for(&local.source,local.mechanism,view.group,view.rank,view.peers)
    }
    pub(super) fn prepare_for(source:&OriginalParallelSource,mechanism:ResidentExecutionMechanisms,
        group_id:eredu_core::CollectiveGroupId,rank:usize,peers:usize)->Result<Self,Error>{
        let context=WorkspaceContext::new_with_metadata_funding(mechanism,source.funding().clone())?;
        context.charge_metadata(size_of::<(Self,Result<Self,Error>,CountItinerary,CountStep,
            WorkspaceContext,[i32;1],usize,u64)>())?;
        let invalid=||context.metadata_error(format_args!("expert count itinerary differs from its retained group"));
        let actual=source.communication_source().map_err(|cause|source.neural_error(cause))?;
        let selected=actual.source().manifest().select_group_operation(group_id,
            eredu_runtime::CommunicationOperation::AllGatherEven).map_err(|cause|context.metadata_source(cause))?;
        let order=selected.order();
        let (group,descriptor,wave)=actual.group(order).ok_or_else(invalid)?;
        if descriptor.members().len()!=peers||descriptor.local_index()!=Some(rank)
            ||group.rank()!=rank {return Err(invalid());}
        let mut backing=0u64;let mut births=0usize;
        let mut add_native=|native:&safemlx::distributed::Group,words:usize|->Result<(),Error>{
            context.charge_metadata(native.cpu_layout_storage_control_bytes().ok_or_else(invalid)?)?;
            let shape=[i32::try_from(words).map_err(|_|invalid())?];
            let layout=native.cpu_layout_storage(&shape,safemlx::Dtype::Int32,
                safemlx::distributed::GroupWorkerOperation::Gather).map_err(|cause|context.metadata_source(cause))?;
            context.charge_metadata(layout.backing_control_bytes().ok_or_else(invalid)?)?;
            let runtime=source.agreement_inputs().ok_or_else(invalid)?.runtime();
            let bytes=layout.backing_capacity(runtime).map_err(|cause|context.metadata_source(cause))?;
            backing=backing.checked_add(u64::try_from(bytes).map_err(|_|invalid())?).ok_or_else(invalid)?;
            births=births.checked_add(layout.evaluation().logical_backing_population().0).ok_or_else(invalid)?;
            Ok(())
        };
        context.charge_metadata(std::mem::size_of_val(&add_native))?;
        let (itinerary,has_world)=if !group.is_logical() {
            let persistent=actual.group_persistent(order).map_err(|cause|source.neural_error(cause))?;
            if persistent.native().has_unqualified_storage()||!persistent.native().is_for(group.native_group())
                ||!persistent.source().same_source(actual.source()){return Err(invalid());}
            add_native(group.native_group(),peers)?;
            (CountItinerary::Physical,false)
        }else{
            let world=actual.source().manifest().world_size();
            let words=OriginalParallelSource::peer_count_wire_words(world).ok_or_else(invalid)?;
            let layout=context.layout(&[i32::try_from(words).map_err(|_|invalid())?],WorkspaceDtype::Int32)?;
            let plan=group.logical_routed_plan().map_err(|_|invalid())?;
            let (itinerary,packed)=if let Some(plan)=plan {
                if plan.len()!=peers{return Err(invalid());}
                let count=plan.values().try_fold(0usize,|count,route|
                    count.checked_add((0..route.steps()).filter(|step|route.exchange(*step).is_some()).count()))
                    .ok_or_else(invalid)?;
                let mut steps=context.metadata_vec(count)?;
                for (value,route) in plan.values().enumerate() {
                    for step in 0..route.steps() {
                        if route.exchange(step).is_none(){continue;}
                        let quote=LogicalCollectiveQuote::prepare_routed_peer(source,order,value,step,
                            layout.as_view(),safemlx::Dtype::Int32,mechanism)?;
                        steps.push(CountStep{value,step,source:RetainedLogicalCollective::retain_prepared(source,quote)
                            .map_err(|cause|source.neural_error(cause))?});
                    }
                }
                (CountItinerary::Routed(steps),false)
            }else{
                let quote=LogicalCollectiveQuote::prepare_group(source,order,LogicalCollectiveKind::Gather,
                    layout.as_view(),safemlx::Dtype::Int32,mechanism)?;
                let packed=quote.packed().is_some();
                (CountItinerary::Logical(RetainedLogicalCollective::retain_prepared(source,quote)
                    .map_err(|cause|source.neural_error(cause))?),packed)
            };
            let extra_world=group.logical_variable_world_plan().is_some()&&!packed;
            if extra_world {
                let plan=group.logical_variable_world_plan().ok_or_else(invalid)?;
                if !wave||plan.members()!=descriptor.members()
                    ||!group.native_group().shares_native_handle(actual.world().native_group()){return Err(invalid());}
                let persistent=actual.world_persistent().map_err(|cause|source.neural_error(cause))?;
                if persistent.native().has_unqualified_storage()||!persistent.native().is_for(actual.world().native_group())
                    ||!persistent.source().same_source(actual.source()){return Err(invalid());}
                add_native(actual.world().native_group(),words)?;
            }
            (itinerary,packed||extra_world)
        };
        // End the temporary mutable borrow before accumulating retained stages.
        drop(add_native);
        let mut add=|source:&RetainedLogicalCollective|->Result<(),Error>{
            let quote=source.value();
            backing=backing.checked_add(quote.output).and_then(|n|n.checked_add(quote.scratch)).ok_or_else(invalid)?;
            births=births.checked_add(quote.maximum_backing_births().ok_or_else(invalid)?).ok_or_else(invalid)?;
            Ok(())
        };
        context.charge_metadata(std::mem::size_of_val(&add))?;
        match &itinerary {
            CountItinerary::Physical=>{},
            CountItinerary::Logical(source)=>add(source)?,
            CountItinerary::Routed(steps)=>for step in steps{add(&step.source)?;},
        }
        drop(add);
        Ok(Self{order,rank:rank,peers:peers,has_world,itinerary,backing,births})
    }
    pub(crate) fn matches(&self,order:usize,peers:usize,rank:usize)->bool {
        self.order==order&&self.peers==peers&&self.rank==rank
    }
    pub(crate) fn has_world(&self)->bool{self.has_world}
    pub(crate) fn logical(&self,selected:Option<(usize,usize)>)->Option<RetainedLogicalCollective>{
        match (&self.itinerary,selected) {
            (CountItinerary::Logical(source),None)=>Some(source.clone()),
            (CountItinerary::Routed(steps),Some((value,step)))=>steps.iter()
                .find(|row|row.value==value&&row.step==step).map(|row|row.source.clone()),
            _=>None,
        }
    }
}
