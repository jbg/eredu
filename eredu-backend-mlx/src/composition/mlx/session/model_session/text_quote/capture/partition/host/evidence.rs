//! Original evidence destinations reuse the ordinary fragment Host and writer.
use super::*;
use std::sync::Arc;
use crate::composition::mlx::session::model_session::partition_capture::LoadedPartitionCapture;
use eredu_architectures::component_partition::ComponentPartitionCaptureSource;
use eredu_core::capture::{CaptureCoordinateProjectionPlan, PartitionCaptureContext};
use eredu_core::intervention::InterventionEvidence;
use eredu_runtime::{layered::PreparedCaptureSelection, working_memory::OriginalInterventionSource};
use eredu_runtime::capture::partition::PartitionCaptureCoordinateProducer;

pub(super) struct RowSource {
    index: usize,
    axis: usize,
    prefill: bool,
    combination: PartitionCaptureCombination,
    local_shape: Option<Vec<u64>>,
    native: Vec<NativeRow>,
    loaded: Arc<LoadedPartitionCapture>,
}
impl std::fmt::Debug for RowSource {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("EvidenceHostSource").field("index",&self.index)
            .field("prefill",&self.prefill).field("native",&self.native).finish_non_exhaustive()
    }
}
#[derive(Debug)]
struct Entry<T> { selection:Option<PreparedCaptureSelection>, hosts:T }
#[derive(Debug)]
pub(in crate::composition::mlx::session::model_session::text_quote::capture) struct Admission {
    entries:Vec<Option<Entry<HostAdmission>>>,
    bytes:u64,
    original:OriginalInterventionSource,
    metadata:WorkspaceMetadataFunding,
}
#[derive(Debug)]
pub(in crate::composition::mlx::session::model_session::text_quote::capture) struct Hosts {
    entries:Vec<Option<Entry<HostRows>>>,
    original:OriginalInterventionSource,
    _metadata:WorkspaceMetadataFunding,
}
/// One actual operation/prediction, retaining the same two original Host rows.
#[derive(Debug)]
pub(in crate::composition::mlx::session::model_session::text_quote::capture) struct Operation {
    pub(in crate::composition::mlx::session::model_session::text_quote::capture) source:SharedCapturePlan,
    pub(in crate::composition::mlx::session::model_session::text_quote::capture) selection:Option<PreparedCaptureSelection>,
    pub(in crate::composition::mlx::session::model_session::text_quote::capture) hosts:Vec<Option<HostRow>>,
}
fn controls()->Option<usize> {
    let parts=[super::controls()?,size_of::<RowSource>()*2,size_of::<Admission>()*2,size_of::<Hosts>()*2,
        size_of::<Operation>()*2,size_of::<Entry<HostAdmission>>()*2,size_of::<Entry<HostRows>>()*2,
        size_of::<Vec<Option<Entry<HostAdmission>>>>(),size_of::<Vec<Option<Entry<HostRows>>>>(),
        size_of::<Vec<Option<Operation>>>(),size_of::<Option<PreparedCaptureSelection>>()*2,
        size_of::<PartitionCaptureCoordinateProducer<'_>>()*2,size_of::<Vec<PartitionCaptureCoordinateProducer<'_>>>(),
        size_of::<CaptureCoordinateProjectionPlan<'_>>()*2,size_of::<CaptureTensorGeometry<'_>>()*2,
        size_of::<CaptureSummaryGeometry<'_>>()*2,size_of::<Result<ResolvedCaptureSlice,Error>>(),
        size_of::<Result<Option<Admission>,Error>>(),size_of::<Result<Hosts,Error>>(),
        size_of::<Result<Operation,Error>>(),size_of::<Result<Prototype,Error>>(),
        size_of::<Result<Option<HostAdmission>,Error>>(),size_of::<Arc<LoadedPartitionCapture>>(),
        size_of::<(&MlxModelSession,&SharedCapturePlan,Option<&OriginalInterventionSource>,&PreparedCaptureSelection,InferenceGeometry,u64,&WorkspaceMetadataFunding)>(),
        size_of::<(&Arc<LoadedPartitionCapture>,&SharedCapturePlan,&OriginalInterventionSource,usize,&SharedCapturePlan,&PartitionCaptureContext,InferenceGeometry,usize,&WorkspaceMetadataFunding)>(),
        size_of::<(&mut Hosts,&OriginalInterventionSource,InferenceGeometry,u64,&WorkspaceMetadataFunding)>(),
        size_of::<std::vec::IntoIter<Option<Entry<HostAdmission>>>>(),
        size_of::<std::slice::IterMut<'_,Option<Entry<HostRows>>>>(),
        size_of::<std::ops::Range<u64>>()*2,size_of::<(usize,u64,bool)>(),
        ComponentPartitionCaptureSource::control_bytes()?,
        usize::try_from(PreparedCaptureSelection::control_peak_bytes()?).ok()?];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
impl Admission {
    pub(in crate::composition::mlx::session::model_session::text_quote::capture) fn prepare(
        session:&MlxModelSession,parent:&SharedCapturePlan,original:Option<&OriginalInterventionSource>,
        selected:&PreparedCaptureSelection,geometry:InferenceGeometry,first:u64,metadata:&WorkspaceMetadataFunding,
    )->Result<Option<Self>,Error> {
        let Some(original)=original else{return Ok(None)};
        if session.payload.distributed.is_none() {return Ok(None)};
        let plan=original.plan().admission();
        if plan.plan().operations.iter().all(|operation|operation.evidence==InterventionEvidence::None) {return Ok(None)};
        metadata.reserve_metadata(controls().ok_or_else(overflow)?)?;
        if plan.request()!=parent.admission().request()||plan.text_origin()!=parent.admission().text_origin()
            ||plan.invocation_bounds()!=parent.admission().invocation_bounds()
            ||!selected.source().same_storage(parent) {return Err(memory(WorkingMemoryError::IdentityMismatch));}
        let mut entries=metadata.metadata_vec(plan.plan().operations.len())?;let mut bytes=0u64;
        for (operation,entry) in plan.plan().operations.iter().enumerate() {
            if entry.evidence==InterventionEvidence::None {entries.push(None);continue;}
            let companion=original.plan().evidence(operation).ok_or_else(unknown)?.shared_geometry_source();
            let Some(hosts)=HostAdmission::prepare_evidence(session,parent,original,operation,companion,geometry,first,metadata)?
                else{entries.push(None);continue};
            let prefill=first==0&&entry.schedule.includes(CapturePhase::Prefill,0)
                &&eredu_runtime::intervention::InterventionPrefillWindow::row_axis(&plan.points()[operation]);
            let selection=if prefill {
                Some(if selected.is_prepared_media(){selected.paths().prepare_media_capture_selection(companion)}
                    else{selected.paths().prepare_capture_selection(companion)}
                    .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?)
            }else{None};
            bytes=bytes.checked_add(hosts.bytes()).ok_or_else(overflow)?;
            entries.push(Some(Entry{selection,hosts}));
        }
        Ok(Some(Self{entries,bytes,original:original.clone(),metadata:metadata.clone()}))
    }
    pub(in crate::composition::mlx::session::model_session::text_quote::capture) const fn bytes(&self)->u64 {self.bytes}
    pub(in crate::composition::mlx::session::model_session::text_quote::capture) fn protect(self,
        funding:&WorkingMemoryFundingRun,reservation:&WorkingMemoryReservation)->Result<Hosts,Error> {
        self.metadata.reserve_metadata(controls().ok_or_else(overflow)?)?;
        let mut entries=self.metadata.metadata_vec(self.entries.len())?;
        for entry in self.entries {entries.push(entry.map(|entry|Ok::<_,Error>(Entry{
            selection:entry.selection,hosts:entry.hosts.protect(funding,reservation)?})).transpose()?);}
        Ok(Hosts{entries,original:self.original,_metadata:self.metadata})
    }
}
impl Hosts {
    pub(in crate::composition::mlx::session::model_session::text_quote::capture) fn take(&mut self,
        original:&OriginalInterventionSource,geometry:InferenceGeometry,prediction:u64,metadata:&WorkspaceMetadataFunding)->Result<Vec<Option<Operation>>,Error> {
        metadata.reserve_metadata(controls().ok_or_else(overflow)?)?;
        if !self.original.same_source(original) {return Err(memory(WorkingMemoryError::IdentityMismatch));}
        let plan=original.plan().admission();let phase=if prediction==0{CapturePhase::Prefill}else{CapturePhase::Decode};
        let mut output=metadata.metadata_vec(self.entries.len())?;
        for (index,entry) in self.entries.iter_mut().enumerate() {
            if !plan.plan().operations[index].schedule.includes(phase,prediction)
                ||plan.plan().operations[index].evidence==InterventionEvidence::None {output.push(None);continue;}
            let entry=entry.as_mut().ok_or_else(unknown)?;
            let source=original.plan().evidence(index).ok_or_else(unknown)?.shared_geometry_source();
            let hosts=entry.hosts.take(source,geometry,prediction)?;
            if hosts.len()!=2||hosts.iter().any(Option::is_none){return Err(memory(WorkingMemoryError::IdentityMismatch));}
            output.push(Some(Operation{source:source.clone(),selection:entry.selection.clone(),hosts}));
        }
        Ok(output)
    }
}
impl HostAdmission {
    fn prepare_evidence(session:&MlxModelSession,parent:&SharedCapturePlan,original:&OriginalInterventionSource,
        operation:usize,source:&SharedCapturePlan,geometry:InferenceGeometry,first:u64,metadata:&WorkspaceMetadataFunding)
        ->Result<Option<Self>,Error> {
        let distributed=session.payload.distributed.as_ref().ok_or_else(unknown)?;
        let loaded=session.partition_capture.get().and_then(|v|v.as_ref().ok()).ok_or_else(unknown)?;
        metadata.reserve_metadata(controls().ok_or_else(overflow)?)?;
        let plan=original.plan().admission();let entry=plan.plan().operations.get(operation).ok_or_else(unknown)?;
        let (artifact,execution,setup)=loaded.source_labels();
        if setup!=distributed.session_identity()||first>=plan.request().max_predictions
            ||source.admission().plan().selections.len()!=2 {return Err(memory(WorkingMemoryError::IdentityMismatch));}
        let count=(first..plan.request().max_predictions).filter(|&prediction|
            entry.schedule.includes(if prediction==0{CapturePhase::Prefill}else{CapturePhase::Decode},prediction)).count();
        if count==0{return Ok(None)};
        let mut frames=metadata.metadata_vec(count)?;let mut bytes=0u64;
        for prediction in first..plan.request().max_predictions {
            let phase=if prediction==0{CapturePhase::Prefill}else{CapturePhase::Decode};
            if !entry.schedule.includes(phase,prediction){continue;}
            let mut context=PreparedPartitionCaptureRunIdentity::prepare_host_context(source,artifact,execution,setup,
                session.payload.parameter_state.active.as_deref(),phase,prediction,metadata)
                .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
            let mut rows=metadata.metadata_vec(2)?;
            for index in 0..2 {
                context.selection_index=index;
                let row=prepare(loaded,parent,original,operation,source,&context,geometry,distributed.native_world().rank(),metadata)?;
                let super::RowSource::Evidence(ref owner)=row.source else{unreachable!("evidence source")};
                let host=if owner.prefill{PartitionFragmentHostPlan::prepare_prefill(&row.receipt,geometry)}
                    else{PartitionFragmentHostPlan::prepare(&row.receipt)}
                    .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                bytes=bytes.checked_add(host.initialization_peak_bytes()).ok_or_else(overflow)?;
                rows.push(Some(row));
            }
            frames.push(Frame{prediction,rows:Some(rows)});
        }
        Ok(Some(Self{frames,geometry,bytes,source:source.clone(),metadata:metadata.clone()}))
    }
}
fn prepare(loaded:&Arc<LoadedPartitionCapture>,parent:&SharedCapturePlan,original:&OriginalInterventionSource,
    operation:usize,source:&SharedCapturePlan,context:&PartitionCaptureContext,inference:InferenceGeometry,
    rank:usize,metadata:&WorkspaceMetadataFunding)->Result<Prototype,Error> {
    let admission=source.admission();let index=context.selection_index;let selection=&admission.plan().selections[index];
    let selected=loaded.layouts().component_capture_source(&selection.path)
        .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
    let point=&original.plan().admission().points()[operation];
    let prefill=context.phase==CapturePhase::Prefill&&eredu_runtime::intervention::InterventionPrefillWindow::row_axis(point);
    let axis=admission.points()[index].axes.as_ref().and_then(|axes|axes.iter().position(|axis|axis.name==selected.axis())).ok_or_else(unknown)?;
    // This structural ceiling is the exact sum of retained coordinate-map
    // capacities. Actual Host slots/estimates below use only constructed fragments.
    let mut producers=metadata.metadata_vec(selected.producer_count())?;let mut maximum=0usize;
    for rank in 0..selected.world_size(){if let Some(local)=selected.rank(rank).filter(|local|local.produces) {
        maximum=maximum.checked_add(local.coordinates.local_count().max(1)).ok_or_else(overflow)?;
        producers.push(PartitionCaptureCoordinateProducer{rank,coordinates:local.coordinates});
    }}
    let limits=PartitionCaptureReceiptLimits{max_producers:selected.world_size(),max_fragments:maximum,
        max_record_bytes:parent.admission().plan().limits.per_step.encoded_bytes};
    // Prospective geometry only: original parent limits and the unchanged
    // companion source. Actual reservation still belongs to the parent ledger.
    let mut quotation=CaptureLedger::new(parent.admission());quotation.begin_step();
    let receipt=PartitionCaptureReceiptPlan::new_coordinates_shared_funded(source,context,axis,&producers,
        selected.combination(),selected.world_size(),limits,metadata,&mut quotation)
        .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
    let global=receipt.producers().next().ok_or_else(unknown)?.1.global_shape();
    if global.get(axis).copied()!=u64::try_from(selected.width()).ok(){return Err(memory(WorkingMemoryError::IdentityMismatch));}
    let member=loaded.layouts().rank(rank).ok_or_else(unknown)?
        .intervention_source(original.plan().admission(),operation,context.phase,context.prediction)
        .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
    let local_shape=member.map(|member|{
        if member.axis()!=axis||selected.rank(rank).is_none_or(|row|row.coordinates!=member.coordinates()) {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let mut shape=copy(global,metadata)?;shape[axis]=u64::try_from(member.coordinates().local_count()).map_err(|_|overflow())?;
        Ok::<_,Error>(shape)
    }).transpose()?;
    let count=receipt.producers().try_fold(0usize,|count,(_,p)|count.checked_add(p.fragments().len())).ok_or_else(overflow)?;
    let mut native=metadata.metadata_vec(count)?;
    for (producer,projection) in receipt.producers(){for fragment in 0..projection.fragments().len(){
        let estimate=if prefill {
            let plan=PartitionPrefillCaptureGeometry::prepare(admission,index,projection,fragment,selected.combination(),inference)
                .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
            crate::composition::mlx::session::capture_workspace::estimate_partition_prefill(&plan)
        }else{
            let plan=PartitionInvocationCaptureGeometry::prepare(admission,index,context.phase,context.prediction,projection,fragment,selected.combination())
                .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
            crate::composition::mlx::session::capture_workspace::estimate_partition_invocation(&plan)
        }.map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
        let slice=projection.fragments()[fragment].local();
        native.push(NativeRow{producer,fragment,shape:copy(projection.local_shape(),metadata)?,slice:ResolvedCaptureSlice{
            starts:copy(&slice.starts,metadata)?,ends:copy(&slice.ends,metadata)?,strides:copy(&slice.strides,metadata)?,shape:copy(&slice.shape,metadata)?},estimate});
    }}
    Ok(Prototype{source:super::RowSource::Evidence(RowSource{index,axis,prefill,combination:selected.combination(),local_shape,native,loaded:loaded.clone()}),receipt})
}
impl RowSource {
    pub(super) fn prefill(&self)->bool {self.prefill}
    pub(super) fn bind(self,source:&SharedCapturePlan,rank:usize,scalar:Option<WorkspaceFloatingType>,
        inference:InferenceGeometry,prediction:u64,metadata:&WorkspaceMetadataFunding)->Result<PreparedPartitionContiguousSource,Error> {
        metadata.reserve_metadata(controls().ok_or_else(overflow)?)?;
        let selected=self.loaded.layouts().component_capture_source(&source.admission().plan().selections[self.index].path)
            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
        if selected.combination()!=self.combination {return Err(memory(WorkingMemoryError::IdentityMismatch));}
        let mut producers=metadata.metadata_vec(selected.producer_count())?;
        for rank in 0..selected.world_size(){if let Some(local)=selected.rank(rank).filter(|local|local.produces){
            producers.push(PartitionCaptureCoordinateProducer{rank,coordinates:local.coordinates});
        }}
        let transform=&source.admission().plan().selections[self.index].transform;
        const RAW:CaptureTransform=CaptureTransform::Slice;
        let transform=if self.combination==PartitionCaptureCombination::SumF64ToF32&&!matches!(transform,CaptureTransform::Preview{..}){&RAW}else{transform};
        let local=match (self.local_shape.as_ref(),scalar) {
            (None,None)=>None,
            (Some(shape),Some(dtype))=>Some(PartitionCaptureLocalSource{producer:rank,shape,dtype:match dtype {
                WorkspaceFloatingType::Float32=>TensorDtype::F32,WorkspaceFloatingType::Float16=>TensorDtype::F16,WorkspaceFloatingType::Bfloat16=>TensorDtype::Bf16}}),
            _=>return Err(memory(WorkingMemoryError::IdentityMismatch)),
        };
        let mut native=metadata.metadata_vec(self.native.len())?;
        for row in &self.native {native.push(PartitionCaptureFragmentGeometry{producer:row.producer,fragment:row.fragment,
            local_shape:&row.shape,local_slice:&row.slice,transform,estimate:row.estimate});}
        let phase=if prediction==0{CapturePhase::Prefill}else{CapturePhase::Decode};
        if self.prefill {
            PreparedPartitionContiguousSource::new_local_coordinates(source,self.index,self.axis,&producers,&native,local,self.combination,inference,metadata)
        }else{
            PreparedPartitionContiguousSource::new_local_coordinates_invocation(source,self.index,self.axis,&producers,&native,local,self.combination,phase,prediction,metadata)
        }.map_err(|cause|Error::Neural(metadata.metadata_source(cause)))
    }
}
