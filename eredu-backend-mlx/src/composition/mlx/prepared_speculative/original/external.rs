//! Exact external selection and source startup above the existing shared driver.
use super::*;
use super::batch::{buffer,inspect};
use crate::backend::{OriginalCopyEnvironment,PreparedOriginalCopyEnvironment,PreparedOriginalCopyEnvironmentError};
use crate::composition::mlx::{
    model::Executable,
    replicated_text::{MlxPredictionTargetState,OriginalEmbeddedCachePreparation},
    session::{MlxModelSession,OriginalInterventionDeclaration},
    speculative::{ExternalInvocationSource,OriginalExternalSources,OriginalSpeculativeNumericalPreparation,
        MlxExternalAssistant,external::{MlxExternalAssistantMechanisms,MlxExternalPredictionCache}},
};
use eredu_architectures::{ExternalAssistantArchitecture,ExternalAssistantExecutorVisitor,
    external_assistant::{ExternalSelectionSource,SelectedExternalAssistantVisitor}};
use eredu_core::{HostPreparationAuthority,SpeculativeBuffer};
use eredu_runtime::{prefill::PrefillControlPlan,
    speculative::external_occurrence::ExternalSchedulePlan,
    working_memory::{OriginalExternalSpeculativeSource,OriginalExternalSpeculativeStartup,
        OriginalSpeculativeSemanticPreparation}};

struct Lane<'lane,'world,C:SpeculativeTokenFilterController> {
    lane:SpeculativeGenerationLane<'lane,MlxBackend<'world>,C>,
    input:NonZeroU64, frontier:u64,output:u64,context:u64,chunk:u64,
    source:OriginalSpeculativeNumericalPreparation,
    declaration:Option<OriginalInterventionDeclaration>,
    preparation:OriginalSpeculativeSemanticPreparation,
}
struct Bound<'selected> {
    cache:OriginalEmbeddedCachePreparation,
    sources:OriginalExternalSources<'selected>,
    preparation:OriginalSpeculativeSemanticPreparation,
}
struct CacheHost {
    _startup:OriginalExternalSpeculativeStartup,
    _funding:WorkspaceMetadataFunding,
}
fn pay<const N:usize>(funding:&WorkspaceMetadataFunding,parts:[usize;N])->Result<(),Error>{
    let bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?;
    funding.reserve_metadata(bytes).map_err(Error::WorkspacePlanning)
}
fn retained(error:Error,funding:&WorkspaceMetadataFunding)->Error{
    super::super::super::model::retain_planning_error(error,funding.clone())
}

pub(super) fn run_batch<'lane,'world,C,V>(backend:&MlxBackend<'_>,session:&mut MlxModelSession,
    drafter:&mut MlxDrafter,lanes:SpeculativeBuffer<SpeculativeGenerationLane<'lane,MlxBackend<'world>,C>>,
    options:eredu_core::SpeculativeSchedulerOptions,visitor:V)->Result<SpeculativeGenerationBatchOutput,Error>
where C:SpeculativeTokenFilterController,V:SpeculativeGenerationVisitor {

    let first=lanes.first().ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let (first,_)=inspect(backend,session,first)?;
    let funding=first.metadata_funding().clone();
    pay(&funding,[size_of::<SpeculativeBuffer<SpeculativeGenerationLane<'lane,MlxBackend<'world>,C>>>(),
        size_of::<eredu_core::SpeculativeBufferIntoIter<SpeculativeGenerationLane<'lane,MlxBackend<'world>,C>>>(),
        size_of::<SpeculativeBuffer<Lane<'lane,'world,C>>>(),
        size_of::<Select<'_,'_,'lane,'world,C,V>>(),
        size_of::<OriginalSpeculativeSemanticPreparation>(),size_of::<WorkspaceMetadataFunding>(),
        size_of::<(eredu_core::SpeculativeSchedulerOptions,usize)>(),
        size_of::<(&MlxBackend<'_>,&mut MlxModelSession,&mut MlxDrafter)>(),
        size_of::<Result<SpeculativeGenerationBatchOutput,Error>>(),
        OriginalCopyEnvironment::control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?])?;
    let run=||{
        for lane in &lanes {inspect(backend,session,lane)?;}
        let topology=drafter.topology();
        // Topology selects this shared external driver; the mandatory
        // bind_original_external below authenticates the exact retained pair
        // before any invocation. CrossDeviceSplit requires the actual CPU side's
        // selected Float32Tiles mechanism and the same pool in either direction.
        // This dispatch predicate grants no model, numerical or copy source.
        if drafter.is_autoregressive() || !matches!(topology,
            eredu_core::SpeculativeExecutionTopology::Single
                |eredu_core::SpeculativeExecutionTopology::SameDeviceSplit
                |eredu_core::SpeculativeExecutionTopology::CrossDeviceSplit) {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        let capacity=drafter.selected().requirements().strategy().proposal_capacity();
        let environment=backend.original_copy_environment().map_err(|cause|
            super::super::super::model::retain_planning_error(cause,funding.clone()))?;
        pay(&funding,[size_of::<Option<PreparedOriginalCopyEnvironment>>(),
            size_of::<Result<Option<OriginalCopyEnvironment<'_>>,crate::backend::OriginalCopyEnvironmentError>>(),
            size_of::<Result<PreparedOriginalCopyEnvironment,PreparedOriginalCopyEnvironmentError>>(),
            size_of::<Option<OriginalCopyEnvironment<'_>>>(),
            size_of::<Result<OriginalCopyEnvironment<'_>,PreparedOriginalCopyEnvironmentError>>(),
            size_of::<HostPreparationAuthority>(),size_of::<eredu_core::SpeculativeExecutionTopology>(),
            HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<PreparedOriginalCopyEnvironmentError>()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?])?;
        let host=HostPreparationAuthority::retain(funding.clone());
        // The actual selected backend is borrowed only while constructing this
        // independent paid stream/prerequisite owner, before mutable visitation.
        let draft_owner=match drafter.placement_environment().map_err(|cause|
            super::super::super::model::retain_planning_error(cause,funding.clone()))? {
            Some(selected)=>Some(PreparedOriginalCopyEnvironment::prepare(&selected,&host,&funding)
                .map_err(|cause|super::super::super::model::retain_planning_error(cause,funding.clone()))?),
            None=>None,
        };
        let draft_environment=match &draft_owner {
            Some(owner)=>Some(owner.loan(backend.memory_pool())
                .map_err(|cause|super::super::super::model::retain_planning_error(cause,funding.clone()))?),
            None=>None,
        };
        let draft=draft_environment.as_ref().unwrap_or(&environment);
        if drafter.stream()!=draft.stream() || !environment.pool().same_domain(draft.pool()) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let mut prepared=buffer(lanes.len(),&funding)?;
        for lane in lanes {
            let (preparation,input)=inspect(backend,session,&lane)?;
            let local=preparation.metadata_funding();
            pay(local,[size_of::<Lane<'lane,'world,C>>(),size_of::<Result<Lane<'lane,'world,C>,Error>>(),
                size_of::<eredu_core::speculative::SpeculativeRequestGeometry>(),
                size_of::<OriginalSpeculativeNumericalPreparation>(),
                size_of::<Option<OriginalInterventionDeclaration>>(),
                size_of::<Result<Option<OriginalInterventionDeclaration>,Error>>(),
                size_of::<(u64,u64,u64,u64,NonZeroU64)>(),
                size_of::<Result<Option<u64>,Error>>()])?;
            let model=session.original_model_source().map_err(Error::PrefillControl)?;
            let frontier=model.erased().original_text_frontier().map_err(|cause|
                super::super::super::model::retain_planning_error(cause,local.clone()))?
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;

            let source=OriginalSpeculativeNumericalPreparation::prepare(model,backend.memory_pool(),local.clone())?;
            // Retain the loaded session declaration before lending either model.
            // Each lane pays its own identity text and keeps the exact origin.
            let declaration=session.original_intervention_declaration(local)?;
            let geometry=eredu_core::speculative::SpeculativeRequestGeometry::new(lane.config(),capacity.get());
            let output=u64::try_from(geometry.output_positions()).map_err(|_|Error::PrefillControl(WorkingMemoryError::Overflow))?;
            let context=frontier.checked_add(input.get()).and_then(|n|n.checked_add(output))
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
            let chunk=lane.prompt().with_borrowed(|input|input.prefill_chunk_positions()).map(NonZeroU64::get)
                .unwrap_or(eredu_runtime::prefill::DEFAULT_PREFILL_CHUNK_POSITIONS).min(input.get());
            prepared.try_push(Lane{lane,input,frontier,output,context,chunk,source,declaration,preparation})
                .map_err(|cause|super::super::super::model::retain_planning_error(cause,funding.clone()))?;
        }
        let with_target=|target:&mut Executable|{

            drafter.visit_selected_funded(funding.clone(),Select{target,lanes:prepared,options,
                environment:&environment,draft,topology,visitor,funding:funding.clone()})
        };
        pay(&funding,[size_of_val(&with_target)])?;
        session.with_model_operation_funded(funding.clone(),with_target)
    };
    pay(&funding,[size_of_val(&run)])?;
    run().map_err(|cause|retained(cause,&funding))
}

struct Select<'target,'environment,'lane,'world,C:SpeculativeTokenFilterController,V>{
    target:&'target mut Executable,
    lanes:SpeculativeBuffer<Lane<'lane,'world,C>>,
    options:eredu_core::SpeculativeSchedulerOptions,
    environment:&'environment OriginalCopyEnvironment<'environment>,
    draft:&'environment OriginalCopyEnvironment<'environment>,
    topology:eredu_core::SpeculativeExecutionTopology,
    visitor:V,
    funding:WorkspaceMetadataFunding,
}
impl<C,V> SelectedExternalAssistantVisitor<MlxAssistantPreparationVisitor> for Select<'_,'_,'_,'_,C,V>
where C:SpeculativeTokenFilterController,V:SpeculativeGenerationVisitor {
    type Output=Result<SpeculativeGenerationBatchOutput,Error>;
    fn visit<A:ExternalAssistantArchitecture>(self,_assistant:&mut MlxExternalAssistant<A>,
        _selected:&eredu_runtime::SelectedSpeculativeRealization,
        _capture:&eredu_architectures::composite_execution::ExternalPredictionCaptureRequest)->Self::Output {
        // An anonymous borrowed contract cannot replace the actual retained owner.
        Err(retained(Error::PrefillControl(WorkingMemoryError::IdentityMismatch),&self.funding))
    }
    fn visit_source<A:ExternalAssistantArchitecture>(self,assistant:&mut MlxExternalAssistant<A>,
        selected:&ExternalSelectionSource)->Self::Output {

        let count=self.lanes.len();
        pay(&self.funding,[size_of::<A::Executor<'_,MlxExternalAssistantMechanisms>>(),
            size_of::<Bound<'_>>(),size_of::<ExternalSchedulePlan<'_>>(),size_of::<PrefillControlPlan>(),
            size_of::<SpeculativeBuffer<Bound<'_>>>(),size_of::<SpeculativeBuffer<MlxExternalPredictionCache>>(),
            size_of::<SpeculativeExecutionStreams<'_>>(),size_of::<eredu_core::InferenceGeometry>(),
            size_of::<Result<SpeculativeGenerationBatchOutput,Error>>(),
            size_of::<eredu_core::SpeculativeBufferIntoIter<Lane<'_,'_,C>>>()])?;
        let mut sources=buffer(count,&self.funding)?;
        let mut lanes=buffer(count,&self.funding)?;
        let mut caches=buffer(count,&self.funding)?;
        for value in self.lanes {
            let local=value.preparation.metadata_funding();
            let prefill=PrefillControlPlan::new(eredu_core::InferenceGeometry{batch_size:1,
                cached_positions:value.frontier,input_positions:value.input.get(),max_output_tokens:value.output,
                prefill_chunk_positions:value.chunk,output:eredu_core::OutputDemand::LastPosition},false)
                .map_err(Error::PrefillControl)?;
            let schedule=ExternalSchedulePlan::new(selected.selected(),A::invocation_shape(),prefill,
                value.context,value.lane.config(),self.options)
                .map_err(|cause|super::super::super::model::retain_planning_error(cause,local.clone()))?;

            let source=OriginalExternalSources::prepare_from_source(value.source,schedule,value.preparation.capacity_bytes(),value.declaration)?;
            let numerical=source.numerical_sources();
            // This token pays only issuance controls. The shared provider pays
            // its exact stream/host constructor separately before each birth.
            let assistant_startup=numerical.request().reserve_external_startup(OriginalExternalSpeculativeSource::Assistant,0)
                .map_err(|cause|numerical.retain_startup_error(cause))?;

            let service=OriginalEmbeddedCachePreparation::new_external(assistant_startup,&value.preparation,
                numerical,self.environment)?;

            let cache=prepare_cache(self.target,selected,&source,&value.preparation,self.environment,value.frontier)?;
            sources.try_push(Bound{cache:service,sources:source,preparation:value.preparation})
                .map_err(|cause|super::super::super::model::retain_planning_error(cause,self.funding.clone()))?;
            caches.try_push(cache).map_err(|cause|super::super::super::model::retain_planning_error(cause,self.funding.clone()))?;
            lanes.try_push(value.lane).map_err(|cause|super::super::super::model::retain_planning_error(cause,self.funding.clone()))?;
        }
        let run=||{

            let mut assignments=buffer(count,&self.funding)?;
            for source in &sources {
                let context=SpeculativeExecutionStreams::bind_original_external(&source.sources,self.environment,self.draft,self.topology)?
                    .with_original_cache_preparation(&source.cache).map_err(Error::StorageSource)?;
                assignments.try_push(context).map_err(|cause|source.sources.numerical_sources().retain_startup_error(cause))?;
            }
            let context=assignments[0].with_request_assignments(&assignments)?;
            // Keep mutable cache storage outside the executor's typed loan.
            // The shared driver borrows these exact prepaid destinations.
            let mut runtime_lanes=buffer(count,&self.funding)?;
            for (index,(lane,source)) in lanes.into_iter().zip(sources.iter()).enumerate(){
                let context=context.request_context(eredu_core::SpeculativeRequestId::new(index))?;
                pay(source.preparation.metadata_funding(),[
                    size_of::<MlxSpeculativeLaneRuntime<'_,C>>(),
                    size_of::<Result<MlxSpeculativeLaneRuntime<'_,C>,Error>>(),
                    size_of::<Result<(),eredu_core::GenerationError>>()])?;

                let prepared=prepare_lane(lane,&source.preparation,context)?;
                runtime_lanes.try_push(prepared).map_err(|cause|source.sources.numerical_sources().retain_startup_error(cause))?;
            }
            let target=self.target.erased_mut().original_external_prediction_mut()
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            let runner=Runner{lanes:runtime_lanes,caches:&mut caches,context,visitor:self.visitor};
            pay(&self.funding,[size_of_val(&runner)])?;

            A::visit_executor_borrowed::<MlxExternalAssistantMechanisms,_>(target,assistant,selected.capture(),runner)
        };
        pay(&self.funding,[size_of_val(&run)])?;
        let result=run();
        // Closing issuance does not retire any escaped native/copy/source owner.
        let mut closed=Ok(());
        for source in &sources {
            if let Err(cause)=source.sources.numerical_sources().request().close(){
                if closed.is_ok(){closed=Err(source.sources.numerical_sources().retain_startup_error(cause));}
            }
        }
        match result{Err(cause)=>Err(retained(cause,&self.funding)),Ok(value)=>closed.map(|()|value)}
    }
}

fn prepare_cache(target:&Executable,selected:&ExternalSelectionSource,source:&OriginalExternalSources<'_>,
    preparation:&OriginalSpeculativeSemanticPreparation,environment:&OriginalCopyEnvironment<'_>,frontier:u64)
    ->Result<MlxExternalPredictionCache,Error>{
    use crate::backend::runtime::cache::state::PreparedResidentDecoderCopy;
    use eredu_runtime::replicated_session::ReplicatedTextControlOrigin;
    let numerical=source.numerical_sources();let funding=numerical.metadata_funding();
    pay(funding,[size_of::<MlxExternalPredictionCache>(),size_of::<Result<MlxExternalPredictionCache,Error>>(),
        size_of::<CacheHost>(),size_of::<HostPreparationAuthority>(),size_of::<ExternalSelectionSource>(),
        size_of::<(PreparedResidentDecoderCopy<'_>,ReplicatedTextControlOrigin)>(),
        size_of::<Result<(PreparedResidentDecoderCopy<'_>,ReplicatedTextControlOrigin),crate::composition::mlx::replicated_text::StartupCause>>(),
        size_of::<(MlxPredictionTargetState,u64)>(),size_of::<Result<(),String>>(),
        HostPreparationAuthority::retention_bytes::<CacheHost>().ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?])?;
    let startup=numerical.request().reserve_external_startup(OriginalExternalSpeculativeSource::Target,
        u64::try_from(size_of::<MlxExternalPredictionCache>()).map_err(|_|Error::PrefillControl(WorkingMemoryError::Overflow))?)
        .map_err(|cause|numerical.retain_startup_error(cause))?;
    let host=HostPreparationAuthority::retain(CacheHost{_startup:startup,_funding:funding.clone()});

    let (state,origin)=target.erased().prepare_original_external_target_source()
        .map_err(|cause|numerical.retain_startup_error(cause))?;

    if !origin.same_origin(numerical.target_origin()){
        return Err(numerical.retain_startup_error(WorkingMemoryError::IdentityMismatch));
    }
    let (roots,mechanisms)=numerical.numerical_prerequisites();

    let native=MlxPredictionTargetState::copy_original_source(state,environment,roots,mechanisms,funding,preparation.capacity_bytes())?;

    if native.generation_fixed()!=Some(frontier){
        return Err(numerical.retain_startup_error(WorkingMemoryError::IdentityMismatch));
    }
    let frontier=i32::try_from(frontier).map_err(|_|numerical.retain_startup_error(WorkingMemoryError::Overflow))?;
    let mut cache=MlxExternalPredictionCache::from_selected_source(native,selected.clone(),host);
    cache.advance_frontier(frontier).map_err(|cause|numerical.retain_error(Error::ArchitectureModel(cause)))?;
    Ok(cache)
}

struct Runner<'cache,'lane,'context,C:SpeculativeTokenFilterController,V>{
    lanes:SpeculativeBuffer<MlxSpeculativeLaneRuntime<'lane,C>>,
    caches:&'cache mut [MlxExternalPredictionCache],
    context:SpeculativeExecutionStreams<'context>,visitor:V,
}
impl<A,C,V> ExternalAssistantExecutorVisitor<A,MlxExternalAssistantMechanisms> for Runner<'_,'_,'_,C,V>
where A:ExternalAssistantArchitecture,C:SpeculativeTokenFilterController,V:SpeculativeGenerationVisitor {
    type Output=Result<SpeculativeGenerationBatchOutput,Error>;
    fn execute<'run,E>(self,executor:&'run mut E)->Self::Output
    where Self:'run,E:SpeculativeExecutor<Input=MlxModelInput,Cache=MlxExternalPredictionCache,
        Logits=IndependentLogits,Context<'run>=SpeculativeExecutionStreams<'run>,
        Completion=crate::composition::mlx::speculative::TypedSpeculativeCompletion,
        Telemetry=crate::composition::mlx::speculative::scheduler::SpeculativeComponentTimings,Error=Error>+'run {

        run_speculative_batch(executor,self.lanes,self.caches,Ok,self.context,self.visitor)
    }
}
