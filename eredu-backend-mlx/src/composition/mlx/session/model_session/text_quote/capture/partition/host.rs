//! Original fragment Host protected before the request funding run moves.
mod routed;
pub(in super::super) mod evidence;
use super::*;
use eredu_core::capture::{CaptureLedger, PartitionCaptureCombination, ResolvedCaptureSlice};
use eredu_runtime::capture::partition::{PartitionCaptureContiguousProducer,
    PartitionCaptureFragmentGeometry, PartitionCaptureLocalSource, PartitionCaptureReceiptPlan,
    PartitionPrefillCaptureGeometry, PartitionInvocationCaptureGeometry, PreparedPartitionContiguousSource};
use eredu_runtime::working_memory::{PartitionFragmentHostPlan, PreparedPartitionFragmentHostFunding,
    WorkingMemoryReservation};
use eredu_nn::workspace::WorkspaceMetadataFunding;

#[derive(Debug)]
struct NativeRow {
    producer: usize,
    fragment: usize,
    shape: Vec<u64>,
    slice: ResolvedCaptureSlice,
    estimate: PartitionCaptureNativeEstimate,
}
#[derive(Debug)]
struct ContiguousRowSource {
    index: usize,
    axis: usize,
    producers: Vec<PartitionCaptureContiguousProducer>,
    combination: PartitionCaptureCombination,
    local_shape: Option<Vec<u64>>,
    native: Vec<NativeRow>,
}
#[derive(Debug)]
enum RowSource { Contiguous(ContiguousRowSource), Routed(routed::RowSource), Evidence(evidence::RowSource) }
/// Source kind stays explicit through the original Host binding.
#[derive(Debug)]
pub(super) enum BoundSource {
    Contiguous(PreparedPartitionContiguousSource),
    Routed(eredu_runtime::capture::partition::PreparedPartitionRoutedSource),
}
#[derive(Debug)]
struct Prototype { source: RowSource, receipt: PartitionCaptureReceiptPlan }

/// A retained slot for an admitted prediction. Taking its rows consumes this
/// coordinate; an empty slot cannot mint another Host or native allowance.
#[derive(Debug)]
struct Frame<T> { prediction:u64, rows:Option<Vec<Option<T>>> }

/// Candidate-specific descriptive sources; no Host or native grant exists yet.
#[derive(Debug)]
pub(in super::super) struct HostAdmission {
    frames: Vec<Frame<Prototype>>,
    geometry: InferenceGeometry,
    bytes: u64,
    source: SharedCapturePlan,
    metadata: WorkspaceMetadataFunding,
}
/// One original fragment Host owner and its exact architecture-derived source.
#[derive(Debug)]
pub(in super::super) struct HostRow {
    source: RowSource,
    geometry: InferenceGeometry,
    prediction: u64,
    host: PreparedPartitionFragmentHostFunding,
    fragments: usize,
}
#[derive(Debug)]
pub(in super::super) struct HostRows {
    frames: Vec<Frame<HostRow>>,
    source: SharedCapturePlan,
    geometry: InferenceGeometry,
    _metadata: WorkspaceMetadataFunding,
}

fn overflow() -> Error { memory(WorkingMemoryError::Overflow) }
fn copy(values: &[u64], metadata: &WorkspaceMetadataFunding) -> Result<Vec<u64>, Error> {
    let mut output = metadata.metadata_vec(values.len())?;
    output.extend_from_slice(values);
    Ok(output)
}
fn controls() -> Option<usize> {
    let parts = [size_of::<HostAdmission>()*2, size_of::<HostRows>()*2,
        size_of::<Prototype>()*2, size_of::<HostRow>()*2, size_of::<RowSource>()*2,
        size_of::<Frame<Prototype>>()*2,size_of::<Frame<HostRow>>()*2,
        size_of::<Vec<Frame<Prototype>>>(),size_of::<Vec<Frame<HostRow>>>(),
        size_of::<std::vec::IntoIter<Frame<Prototype>>>(),size_of::<std::slice::IterMut<'_,Frame<HostRow>>>(),
        size_of::<Option<Vec<Option<Prototype>>>>(),size_of::<Option<Vec<Option<HostRow>>>>(),
        size_of::<CapturePhase>(),size_of::<std::ops::Range<u64>>(),
        size_of::<PartitionCaptureReceiptLimits>()*3,
        size_of::<PartitionInvocationCaptureGeometry<'_>>()*2,
        size_of::<Result<PartitionCaptureNativeEstimate,eredu_core::capture::CaptureError>>(),
        size_of::<NativeRow>()*2, size_of::<PartitionCaptureReceiptPlan>()*2,
        size_of::<PartitionPrefillCaptureGeometry<'_>>()*2,
        size_of::<PartitionCaptureFragmentGeometry<'_>>()*2,
        size_of::<PartitionCaptureLocalSource<'_>>(), size_of::<PartitionFragmentHostPlan<'_>>(),
        size_of::<PartitionCaptureContiguousProducer>()*2,
        size_of::<eredu_architectures::component_partition::ContiguousPartitionCaptureSource<'_>>(),
        size_of::<Option<eredu_architectures::component_partition::ContiguousPartitionCaptureRank>>(),
        size_of::<Result<Option<HostAdmission>, Error>>(), size_of::<Result<HostRows, Error>>(),
        size_of::<Result<PreparedPartitionContiguousSource, Error>>(),
        size_of::<ContiguousRowSource>()*2, size_of::<BoundSource>()*2, routed::controls()?,
        size_of::<Result<(BoundSource, PreparedPartitionFragmentHostFunding), Error>>(),
        size_of::<CaptureLedger>(), size_of::<ResolvedCaptureSlice>()*2,
        size_of::<Vec<Option<Prototype>>>(), size_of::<Vec<Option<HostRow>>>(),
        size_of::<Vec<NativeRow>>(), size_of::<Vec<PartitionCaptureContiguousProducer>>(),
        size_of::<Vec<PartitionCaptureFragmentGeometry<'_>>>(), size_of::<Option<Vec<u64>>>(),
        size_of::<(&MlxModelSession,&SharedCapturePlan,InferenceGeometry,u64,&WorkspaceMetadataFunding)>(),
        size_of::<(&[u64],&WorkspaceMetadataFunding)>(),
        size_of::<(HostAdmission,&WorkingMemoryFundingRun,&WorkingMemoryReservation)>(),
        size_of::<(HostRow,&SharedCapturePlan,usize,Option<WorkspaceFloatingType>,&WorkspaceMetadataFunding)>(),
        size_of::<(&mut HostRows,&SharedCapturePlan,InferenceGeometry,u64)>(),
        size_of::<std::slice::Iter<'_,eredu_core::capture::CaptureSelection>>(),
        size_of::<std::slice::Iter<'_,NativeRow>>(),size_of::<std::vec::IntoIter<Option<Prototype>>>(),
        size_of::<[usize;4]>(),size_of::<[u64;4]>(),
        eredu_architectures::component_partition::ComponentPartitionLayouts::complete_capture_source_control_bytes()?,
        eredu_architectures::component_partition::ContiguousPartitionCaptureSource::control_bytes()?,
        PartitionPrefillCaptureGeometry::control_bytes()?,
        PartitionInvocationCaptureGeometry::control_bytes()?,
    ];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}

impl HostAdmission {
    pub(in super::super) fn prepare(session:&MlxModelSession,source:&SharedCapturePlan,
        geometry:InferenceGeometry,first_prediction:u64,metadata:&WorkspaceMetadataFunding)
        ->Result<Option<Self>,Error> {
        let Some(distributed)=session.payload.distributed.as_ref() else{return Ok(None)};
        let loaded=session.partition_capture.get().and_then(|v|v.as_ref().ok()).ok_or_else(unknown)?;
        metadata.reserve_metadata(controls().ok_or_else(overflow)?)?;
        let admission=source.admission();
        if first_prediction>=admission.request().max_predictions{return Err(memory(WorkingMemoryError::IdentityMismatch));}
        let scheduled=|prediction| {
            let phase=if prediction==0{CapturePhase::Prefill}else{CapturePhase::Decode};
            admission.plan().selections.iter().any(|selection|selection.schedule.includes(phase,prediction)
                && loaded.layouts().complete_capture_source(&selection.path).is_none())
        };
        let frame_count=(first_prediction..admission.request().max_predictions).try_fold(0usize,|n,prediction|
            if scheduled(prediction){n.checked_add(1)}else{Some(n)}).ok_or_else(overflow)?;
        if frame_count==0{return Ok(None);}
        let (artifact,execution,setup)=loaded.source_labels();
        if setup!=distributed.session_identity(){return Err(memory(WorkingMemoryError::IdentityMismatch));}
        let rank=distributed.native_world().rank();
        let mut frames=metadata.metadata_vec(frame_count)?;
        let mut bytes=0u64;
        for prediction in first_prediction..admission.request().max_predictions {
            if !scheduled(prediction){continue;}
            let phase=if prediction==0{CapturePhase::Prefill}else{CapturePhase::Decode};
            let mut context=PreparedPartitionCaptureRunIdentity::prepare_host_context(source,artifact,execution,
                setup,session.payload.parameter_state.active.as_deref(),phase,prediction,metadata)
                .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
            let mut rows=metadata.metadata_vec(admission.plan().selections.len())?;
        for (index,selection) in admission.plan().selections.iter().enumerate(){
            if !selection.schedule.includes(phase,prediction)
                || loaded.layouts().complete_capture_source(&selection.path).is_some(){rows.push(None);continue}
            if matches!(selection.transform,CaptureTransform::RoutedUnits) {
                context.selection_index=index;
                let prototype=routed::prepare(loaded,source,&context,index,geometry,metadata)?;
                let host=if prediction==0 {PartitionFragmentHostPlan::prepare_prefill(&prototype.receipt,geometry)}
                    else{PartitionFragmentHostPlan::prepare(&prototype.receipt)}
                    .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                bytes=bytes.checked_add(host.initialization_peak_bytes()).ok_or_else(overflow)?;
                rows.push(Some(prototype));continue;
            }
            let selected=loaded.layouts().contiguous_capture_source(&selection.path)
                .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
            if selected.world_size()!=setup.participant_count(){return Err(memory(WorkingMemoryError::IdentityMismatch));}
            let axis=admission.points()[index].axes.as_ref().and_then(|axes|
                axes.iter().position(|axis|axis.name==selected.axis())).ok_or_else(unknown)?;
            let mut producers=metadata.metadata_vec(selected.producer_count())?;
            for rank in 0..selected.world_size(){if let Some(local)=selected.rank(rank).filter(|local|local.produces){
                producers.push(PartitionCaptureContiguousProducer{rank,coordinates:
                    u64::try_from(local.coordinates.start).map_err(|_|overflow())?..
                    u64::try_from(local.coordinates.end).map_err(|_|overflow())?});
            }}
            context.selection_index=index;
            // Separate prospective ledger: no request usage, receipt vote or
            // native allowance is issued by this descriptive Host source.
            let mut quotation=CaptureLedger::new(admission);quotation.begin_step();
            let limits=PartitionCaptureReceiptLimits{max_producers:setup.participant_count(),
                max_fragments:setup.participant_count(),max_record_bytes:admission.plan().limits.per_step.encoded_bytes};
            let receipt=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(source,&context,axis,&producers,
                selected.combination(),selected.world_size(),limits,metadata,&mut quotation)
                .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
            let global=receipt.producers().next().ok_or_else(unknown)?.1.global_shape();
            if global.get(axis).copied()!=u64::try_from(selected.width()).ok(){return Err(memory(WorkingMemoryError::IdentityMismatch));}
            let local_shape=selected.rank(rank).map(|local|{
                let mut shape=copy(global,metadata)?;
                shape[axis]=u64::try_from(local.coordinates.end-local.coordinates.start).map_err(|_|overflow())?;
                Ok::<_,Error>(shape)
            }).transpose()?;
            let count=receipt.producers().try_fold(0usize,|n,(_,projection)|n.checked_add(projection.fragments().len())).ok_or_else(overflow)?;
            let mut native=metadata.metadata_vec(count)?;
            for (producer,projection) in receipt.producers(){for fragment in 0..projection.fragments().len(){
                let estimate=if prediction==0 {
                    let plan=PartitionPrefillCaptureGeometry::prepare(admission,index,projection,fragment,
                        selected.combination(),geometry).map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                    crate::composition::mlx::session::capture_workspace::estimate_partition_prefill(&plan)
                }else{
                    let plan=PartitionInvocationCaptureGeometry::prepare(admission,index,phase,prediction,projection,fragment,
                        selected.combination()).map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                    crate::composition::mlx::session::capture_workspace::estimate_partition_invocation(&plan)
                }.map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                let slice=projection.fragments()[fragment].local();
                native.push(NativeRow{producer,fragment,shape:copy(projection.local_shape(),metadata)?,slice:ResolvedCaptureSlice{
                    starts:copy(&slice.starts,metadata)?,ends:copy(&slice.ends,metadata)?,
                    strides:copy(&slice.strides,metadata)?,shape:copy(&slice.shape,metadata)?},estimate});
            }}
            let host=if prediction==0 {PartitionFragmentHostPlan::prepare_prefill(&receipt,geometry)}
                else{PartitionFragmentHostPlan::prepare(&receipt)}.map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
            bytes=bytes.checked_add(host.initialization_peak_bytes()).ok_or_else(overflow)?;
            rows.push(Some(Prototype{source:RowSource::Contiguous(ContiguousRowSource{index,axis,producers,combination:selected.combination(),local_shape,native}),receipt}));
        }
            frames.push(Frame{prediction,rows:Some(rows)});
        }
        Ok(Some(Self{frames,geometry,bytes,source:source.clone(),metadata:metadata.clone()}))
    }
    pub(in super::super) const fn bytes(&self)->u64{self.bytes}
    pub(in super::super) fn protect(self,funding:&WorkingMemoryFundingRun,reservation:&WorkingMemoryReservation)
        ->Result<HostRows,Error>{
        let mut frames=self.metadata.metadata_vec(self.frames.len())?;
        for frame in self.frames {
            let source=frame.rows.ok_or_else(||memory(WorkingMemoryError::IdentityMismatch))?;
            let mut rows=self.metadata.metadata_vec(source.len())?;
            for row in source {rows.push(match row {
                None=>None,
                Some(row)=>{
                    let prefill=match &row.source {RowSource::Evidence(source)=>source.prefill(),_=>frame.prediction==0};
                    let host=funding.prepare_partition_fragment_host(reservation,row.receipt,
                        prefill.then_some(self.geometry))
                        .map_err(|cause|Error::Neural(self.metadata.metadata_source(cause)))?;
                    let fragments=host.original_receipt_limits().max_fragments;
                    Some(HostRow{source:row.source,geometry:self.geometry,prediction:frame.prediction,fragments,host})
                },
            });}
            frames.push(Frame{prediction:frame.prediction,rows:Some(rows)});
        }
        Ok(HostRows{frames,source:self.source,geometry:self.geometry,_metadata:self.metadata})
    }
}
impl HostRows {
    pub(in super::super) fn take(&mut self,source:&SharedCapturePlan,geometry:InferenceGeometry,prediction:u64)
        ->Result<Vec<Option<HostRow>>,Error>{
        if !self.source.same_storage(source)||self.geometry!=geometry||prediction>=source.admission().request().max_predictions {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let Some(frame)=self.frames.iter_mut().find(|frame|frame.prediction==prediction) else{return Ok(Vec::new())};
        frame.rows.take().ok_or_else(||memory(WorkingMemoryError::IdentityMismatch))
    }
}

impl HostRow {
    /// Original receipt limit, retained separately from its actual fragment population.
    pub(super) const fn fragment_limit(&self)->usize {self.fragments}
    pub(super) fn bind(self,source:&SharedCapturePlan,rank:usize,scalar:Option<WorkspaceFloatingType>,
        metadata:&WorkspaceMetadataFunding)->Result<(BoundSource,PreparedPartitionFragmentHostFunding),Error>{
        let source_row=match self.source {
            RowSource::Contiguous(row)=>row,
            RowSource::Evidence(row)=>return row.bind(source,rank,scalar,self.geometry,self.prediction,metadata)
                .map(|source|(BoundSource::Contiguous(source),self.host)),
            RowSource::Routed(row)=>return row.bind(source,rank,scalar,self.geometry,self.prediction,metadata)
                .map(|source|(BoundSource::Routed(source),self.host)),
        };
        const RAW:CaptureTransform=CaptureTransform::Slice;
        let selected=&source.admission().plan().selections[source_row.index].transform;
        let transform=if source_row.combination==PartitionCaptureCombination::SumF64ToF32&&!matches!(selected,CaptureTransform::Preview{..}){&RAW}else{selected};
        let local=match (source_row.local_shape.as_ref(),scalar){
            (None,None)=>None,
            (Some(shape),Some(dtype))=>Some(PartitionCaptureLocalSource{producer:rank,shape,dtype:match dtype{
                WorkspaceFloatingType::Float32=>TensorDtype::F32,WorkspaceFloatingType::Float16=>TensorDtype::F16,
                WorkspaceFloatingType::Bfloat16=>TensorDtype::Bf16}}),
            _=>return Err(memory(WorkingMemoryError::IdentityMismatch)),
        };
        let mut rows=metadata.metadata_vec(source_row.native.len())?;
        for row in &source_row.native{rows.push(PartitionCaptureFragmentGeometry{producer:row.producer,fragment:row.fragment,
            local_shape:&row.shape,local_slice:&row.slice,transform,estimate:row.estimate});}
        let prepared=if self.prediction==0 {
            PreparedPartitionContiguousSource::new_local(source,source_row.index,source_row.axis,
                &source_row.producers,&rows,local,source_row.combination,self.geometry,metadata)
        }else{
            PreparedPartitionContiguousSource::new_local_decode(source,source_row.index,source_row.axis,
                &source_row.producers,&rows,local,source_row.combination,self.prediction,metadata)
        }.map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
        Ok((BoundSource::Contiguous(prepared),self.host))
    }
}
