//! Actual contiguous local source attached to the shared cold capture observer.
use super::*;
use eredu_architectures::component_partition::{ComponentPartitionLayouts,ContiguousPartitionCaptureSource};
use eredu_runtime::capture::{CapturePrefillObservationRow,CapturePrefillRowProgress,CapturePrefillHookDecision};
use eredu_runtime::prefill::PrefillChunk;
use std::ops::Range;

/// Geometry storage is paid before the ordinary projection worker fills it.
/// No receipt identity or source precision is synthesized by this adapter.
pub(super) fn projection(admission:&AdmittedCapturePlan,index:usize,axis:usize,
    range:Range<u64>,context:&WorkspaceContext)->Result<CaptureSlicePartition> {
    projection_at(admission,index,CapturePhase::Prefill,0,axis,range,context)
}

pub(super) fn projection_at(admission:&AdmittedCapturePlan,index:usize,phase:CapturePhase,prediction:u64,
    axis:usize,range:Range<u64>,context:&WorkspaceContext)->Result<CaptureSlicePartition> {
    let metadata=Metadata::new(context)?;
    let parts=[size_of::<(CaptureSlicePartition,ResolvedCaptureSlice)>(),size_of::<[u64;32]>(),
        size_of::<Option<CaptureTensorGeometry<'_>>>(),size_of::<Option<CaptureSummaryGeometry<'_>>>(),
        size_of::<Option<CaptureHistogramGeometry<'_>>>(),size_of::<(&[usize],&[u64],&[u64],&[u64])>(),
        size_of::<(&AdmittedCapturePlan,usize,usize,Range<u64>,&WorkspaceContext)>(),
        size_of::<(&AdmittedCapturePlan,usize,CapturePhase,u64,usize,Range<u64>,&WorkspaceContext)>(),
        size_of::<Result<CaptureSlicePartition>>(),size_of::<[Vec<u64>;4]>(),
        size_of::<std::iter::Zip<std::iter::Zip<std::slice::Iter<'_,u64>,std::slice::Iter<'_,u64>>,std::slice::Iter<'_,u64>>>(),
        size_of::<std::iter::Zip<std::slice::IterMut<'_,u64>,std::slice::Iter<'_,usize>>>(),
        CaptureTensorGeometry::preparation_control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        CaptureSummaryGeometry::preparation_control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        CaptureHistogramGeometry::preparation_control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
    let selection=admission.plan().selections.get(index).ok_or_else(||metadata.coordinate())?;
    let tensor=if matches!(selection.transform,CaptureTransform::FullTensor|CaptureTransform::Slice|CaptureTransform::Preview{..}) {
        Some(CaptureTensorGeometry::prepare(admission,index,phase,prediction,None).map_err(|e|metadata.error(e))?)
    }else{None};
    let summary=if matches!(selection.transform,CaptureTransform::Summary) {
        Some(CaptureSummaryGeometry::prepare(admission,index,phase,prediction,None).map_err(|e|metadata.error(e))?)
    }else{None};
    let histogram=if matches!(selection.transform,CaptureTransform::Histogram{..}) {
        Some(CaptureHistogramGeometry::prepare(admission,index,phase,prediction,None).map_err(|e|metadata.error(e))?)
    }else{None};
    let (shape,starts,ends,strides)=match (&tensor,&summary,&histogram) {
        (Some(g),None,None)=>(g.source_shape(),g.starts(),g.ends(),g.strides()),
        (None,Some(g),None)=>(g.source_shape(),g.starts(),g.ends(),g.strides()),
        (None,None,Some(g))=>(g.source_shape(),g.starts(),g.ends(),g.strides()),
        _=>return Err(metadata.coordinate()),
    };
    if shape.is_empty()||shape.len()>32||axis>=shape.len(){return Err(metadata.coordinate());}
    let mut global=[0u64;32];
    for (out,&n) in global.iter_mut().zip(shape){*out=u64::try_from(n).map_err(|e|metadata.error(e))?;}
    let mut slice=ResolvedCaptureSlice{starts:context.metadata_vec(shape.len())?,ends:context.metadata_vec(shape.len())?,
        strides:context.metadata_vec(shape.len())?,shape:context.metadata_vec(shape.len())?};
    slice.starts.extend_from_slice(starts);slice.ends.extend_from_slice(ends);slice.strides.extend_from_slice(strides);
    for ((start,end),stride) in starts.iter().zip(ends).zip(strides){slice.shape.push((end-start).div_ceil(*stride));}
    let plan=CaptureContiguousProjectionPlan::prepare(&global[..shape.len()],&slice,axis,range,1).map_err(|e|metadata.error(e))?;
    context.charge_metadata(plan.requested_bytes())?;
    Ok(plan.construct())
}

/// Trace an actual local projected source, or settle its actual replica/empty
/// source. Global logical row progression remains in the existing policy worker.
#[allow(clippy::too_many_arguments)]
pub(super) fn trace(source:&SharedCapturePlan,index:usize,projection:&CaptureSlicePartition,
    combination:PartitionCaptureCombination,producer:usize,produces:bool,inference:InferenceGeometry,
    chunk:u64,sequence_axis:usize,value:&WorkspaceTensor,context:&WorkspaceContext,
    roots:&mut Vec<WorkspaceTensor>,transfers:Option<&Cell<CaptureNativePopulation>>,
    scalars:Option<&[Cell<Option<WorkspaceFloatingType>>]>)->Result<()> {
    let metadata=Metadata::new(context)?;
    let parts=[size_of::<(&SharedCapturePlan,usize,&CaptureSlicePartition,PartitionCaptureCombination,usize,bool,
        InferenceGeometry,u64,usize,&WorkspaceTensor,&WorkspaceContext,&mut Vec<WorkspaceTensor>,
        Option<&Cell<CaptureNativePopulation>>,Option<&[Cell<Option<WorkspaceFloatingType>>]>)>(),
        size_of::<[i32;32]>(),size_of::<Result<()>>(),size_of::<(usize,u64,u64)>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_,u64>>>(),
        size_of::<PartitionPrefillCaptureGeometry<'_>>(),size_of::<std::result::Result<PartitionPrefillCaptureGeometry<'_>,
            eredu_runtime::capture::partition::PartitionPrefillCaptureSourceError>>(),
        PartitionPrefillCaptureGeometry::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
    context.validate_values([value])?;
    let rank=projection.local_shape().len();
    if rank==0||rank>32||sequence_axis>=rank||projection.local_shape()[sequence_axis]!=inference.input_positions
        ||inference.prefill_chunk_positions==0||chunk>=inference.input_positions.div_ceil(inference.prefill_chunk_positions) {
        return Err(metadata.coordinate());
    }
    let start=chunk.checked_mul(inference.prefill_chunk_positions).ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
    let mut shape=[0i32;32];
    for (axis,&n) in projection.local_shape().iter().enumerate(){shape[axis]=i32::try_from(n).map_err(|e|metadata.error(e))?;}
    shape[sequence_axis]=i32::try_from(inference.prefill_chunk_positions.min(inference.input_positions-start))
        .map_err(|e|metadata.error(e))?;
    if value.shape()!=&shape[..rank]||value.layout().representation().is_none(){return Err(metadata.coordinate());}
    record_source(scalars,index,context,value)?;
    if !produces||projection.fragments().is_empty(){
        return record_replica(transfers,context,roots,value);
    }
    let geometry=PartitionPrefillCaptureGeometry::prepare(source.admission(),index,projection,0,combination,inference)
        .map_err(|e|metadata.error(e))?;
    let equation=PartitionPrefillEquation::from_geometry(geometry,producer,context)?;
    let (_,population)=equation.trace_fragment(chunk,value,context,roots)?;
    record_population(transfers,context,population)
}

/// Called only for a selected non-complete row. Source placement comes from
/// the retained architecture table, never from the local tensor's shape.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn observe(source:&SharedCapturePlan,index:usize,row:&CapturePrefillObservationRow<'_>,
    progress:&mut CapturePrefillRowProgress,decision:CapturePrefillHookDecision,
    placement:(&ComponentPartitionLayouts,usize),chunk:&PrefillChunk,inference:InferenceGeometry,
    source_present:bool,value:&WorkspaceTensor,context:&WorkspaceContext,roots:&mut Vec<WorkspaceTensor>,
    ledger:&mut CaptureLedger,transfers:Option<&Cell<CaptureNativePopulation>>,
    scalars:Option<&[Cell<Option<WorkspaceFloatingType>>]>)->Result<()> {
    let metadata=Metadata::new(context)?;
    let parts=[size_of::<(&SharedCapturePlan,usize,&CapturePrefillObservationRow<'_>,&mut CapturePrefillRowProgress,
        CapturePrefillHookDecision,(&ComponentPartitionLayouts,usize),&PrefillChunk,InferenceGeometry,bool,
        &WorkspaceTensor,&WorkspaceContext,&mut Vec<WorkspaceTensor>,&mut CaptureLedger,
        Option<&Cell<CaptureNativePopulation>>,Option<&[Cell<Option<WorkspaceFloatingType>>]>)>(),
        size_of::<ContiguousPartitionCaptureSource<'_>>(),size_of::<CaptureSlicePartition>(),
        size_of::<std::result::Result<ContiguousPartitionCaptureSource<'_>,eredu_architectures::component_partition::PartitionCaptureSourceError>>(),
        size_of::<Option<eredu_architectures::component_partition::ContiguousPartitionCaptureRank>>(),
        size_of::<std::slice::Iter<'_,eredu_core::TensorAxis>>(),size_of::<Range<u64>>(),
        size_of::<Option<&Vec<eredu_core::TensorAxis>>>(),size_of::<Result<()>>(),
        size_of::<eredu_architectures::component_partition::ContiguousPartitionCaptureRank>(),
        size_of::<CaptureUsage>(),size_of::<Option<CaptureSkipReason>>(),size_of::<usize>()*3,
        size_of::<CapturePrefillFragment<'_,'_>>(),size_of::<CapturePrefillTransformFragment<'_,'_>>(),
        ContiguousPartitionCaptureSource::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
    if !source_present{return Err(metadata.coordinate());}
    let selection=&source.admission().plan().selections[index];
    let selected=placement.0.contiguous_capture_source(&selection.path).map_err(|e|metadata.error(e))?;
    let local=selected.rank(placement.1).ok_or_else(||metadata.coordinate())?;
    let axes=source.admission().points()[index].axes.as_ref().ok_or_else(||metadata.coordinate())?;
    let axis=axes.iter().position(|axis|axis.name==selected.axis()).ok_or_else(||metadata.coordinate())?;
    let projection=projection(source.admission(),index,axis,
        u64::try_from(local.coordinates.start).map_err(|e|metadata.error(e))?..
        u64::try_from(local.coordinates.end).map_err(|e|metadata.error(e))?,context)?;
    if projection.global_shape().get(axis).copied()!=u64::try_from(selected.width()).ok(){return Err(metadata.coordinate());}
    let sequence_axis=if let Some(plan)=row.transform_plan(){plan.window().axis()}
        else{row.assembly().ok_or_else(||metadata.coordinate())?.sequence_axis()};
    if decision==CapturePrefillHookDecision::First {
        let usage=if let Some(plan)=row.transform_plan(){match selection.transform {
            CaptureTransform::Summary=>super::super::super::bounded_capture::estimate_prefill_summary(plan),
            CaptureTransform::Histogram{..}=>super::super::super::bounded_capture::estimate_prefill_histogram(plan),
            _=>return Err(metadata.coordinate()),
        }}else{super::super::super::bounded_capture::estimate_tensor_geometry(row.assembly().ok_or_else(||metadata.coordinate())?.logical_geometry())}
            .map_err(|e|metadata.error(e))?;
        if row.reserve_first(progress,ledger,usage).map_err(|e|metadata.error(e))?.is_some(){return Ok(());}
    }
    let k=chunk.input.start/inference.prefill_chunk_positions;
    trace(source,index,&projection,selected.combination(),local.rank,local.produces,inference,k,sequence_axis,
        value,context,roots,transfers,scalars)?;
    if let Some(plan)=row.transform_plan(){
        row.finish_transform_hook(progress,&plan.fragment(k).map_err(|e|metadata.error(e))?).map_err(|e|metadata.error(e))?;
    }else{
        row.finish_hook(progress,&row.assembly().ok_or_else(||metadata.coordinate())?.fragment(k).map_err(|e|metadata.error(e))?)
            .map_err(|e|metadata.error(e))?;
    }
    Ok(())
}

/// The global result is charged once on every rank; only actual local sources
/// add native transformation or replica-completion populations.
fn invocation_usage(source:&SharedCapturePlan,index:usize,phase:CapturePhase,prediction:u64,
    context:&WorkspaceContext)->Result<CaptureUsage> {
    let metadata=Metadata::new(context)?;
    let parts=[size_of::<(&SharedCapturePlan,usize,CapturePhase,u64,&WorkspaceContext)>(),
        size_of::<CaptureTensorGeometry<'_>>(),size_of::<CaptureSummaryGeometry<'_>>(),
        size_of::<CaptureHistogramGeometry<'_>>(),size_of::<Result<CaptureUsage>>(),
        CaptureTensorGeometry::preparation_control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        CaptureSummaryGeometry::preparation_control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        CaptureHistogramGeometry::preparation_control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
    if phase!=CapturePhase::Decode||prediction==0{return Err(metadata.coordinate());}
    let admission=source.admission();
    let selection=admission.plan().selections.get(index).ok_or_else(||metadata.coordinate())?;
    match selection.transform {
        CaptureTransform::FullTensor|CaptureTransform::Slice|CaptureTransform::Preview{..}=>{
            let geometry=CaptureTensorGeometry::prepare(admission,index,phase,prediction,None).map_err(|cause|metadata.error(cause))?;
            super::super::super::bounded_capture::estimate_tensor_geometry(&geometry)
        },
        CaptureTransform::Summary=>{
            let geometry=CaptureSummaryGeometry::prepare(admission,index,phase,prediction,None).map_err(|cause|metadata.error(cause))?;
            super::super::super::bounded_capture::estimate_summary(&geometry)
        },
        CaptureTransform::Histogram{..}=>{
            let geometry=CaptureHistogramGeometry::prepare(admission,index,phase,prediction,None).map_err(|cause|metadata.error(cause))?;
            super::super::super::bounded_capture::estimate_histogram(&geometry)
        },
        _=>return Err(metadata.coordinate()),
    }.map_err(|cause|metadata.error(cause))
}

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn observe_invocation(source:&SharedCapturePlan,index:usize,phase:CapturePhase,prediction:u64,
    placement:(&ComponentPartitionLayouts,usize),source_present:bool,value:&WorkspaceTensor,context:&WorkspaceContext,
    roots:&mut Vec<WorkspaceTensor>,ledger:&mut CaptureLedger,transfers:Option<&Cell<CaptureNativePopulation>>,
    scalars:Option<&[Cell<Option<WorkspaceFloatingType>>]>)->Result<bool> {
    let metadata=Metadata::new(context)?;
    let parts=[size_of::<(&SharedCapturePlan,usize,CapturePhase,u64,(&ComponentPartitionLayouts,usize),bool,
        &WorkspaceTensor,&WorkspaceContext,&mut Vec<WorkspaceTensor>,&mut CaptureLedger,
        Option<&Cell<CaptureNativePopulation>>,Option<&[Cell<Option<WorkspaceFloatingType>>]>)>(),
        size_of::<CaptureObservationStep<'_>>(),size_of::<CaptureUsage>(),size_of::<Result<bool>>(),
        size_of::<ContiguousPartitionCaptureSource<'_>>(),size_of::<CaptureSlicePartition>(),
        size_of::<eredu_architectures::component_partition::ContiguousPartitionCaptureRank>(),
        size_of::<Option<eredu_architectures::component_partition::ContiguousPartitionCaptureRank>>(),
        size_of::<Range<u64>>(),size_of::<std::slice::Iter<'_,eredu_core::TensorAxis>>(),
        ContiguousPartitionCaptureSource::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
    let selection=source.admission().plan().selections.get(index).ok_or_else(||metadata.coordinate())?;
    let selected=placement.0.contiguous_capture_source(&selection.path).map_err(|cause|metadata.error(cause))?;
    let policy=CaptureObservationStep::with_invocation(source.admission(),phase,prediction,None).map_err(|cause|metadata.error(cause))?;
    let usage=invocation_usage(source,index,phase,prediction,context)?;
    if policy.reserve_value(ledger,usage).map_err(|cause|metadata.error(cause))?.is_some(){return Ok(true);}
    let local=selected.rank(placement.1);
    if source_present!=local.is_some(){return Err(metadata.coordinate());}
    let Some(local)=local else{return Ok(false)};
    let axis=source.admission().points()[index].axes.as_ref().and_then(|axes|
        axes.iter().position(|axis|axis.name==selected.axis())).ok_or_else(||metadata.coordinate())?;
    let projection=projection_at(source.admission(),index,phase,prediction,axis,
        u64::try_from(local.coordinates.start).map_err(|cause|metadata.error(cause))?..
        u64::try_from(local.coordinates.end).map_err(|cause|metadata.error(cause))?,context)?;
    if projection.global_shape().get(axis).copied()!=u64::try_from(selected.width()).ok(){return Err(metadata.coordinate());}
    trace_invocation(source,index,phase,prediction,&projection,selected.combination(),local.rank,local.produces,
        value,context,roots,transfers,scalars)?;
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn trace_invocation(source:&SharedCapturePlan,index:usize,phase:CapturePhase,prediction:u64,
    projection:&CaptureSlicePartition,combination:PartitionCaptureCombination,producer:usize,produces:bool,
    value:&WorkspaceTensor,context:&WorkspaceContext,roots:&mut Vec<WorkspaceTensor>,
    transfers:Option<&Cell<CaptureNativePopulation>>,scalars:Option<&[Cell<Option<WorkspaceFloatingType>>]>)->Result<()> {
    use eredu_runtime::capture::partition::PartitionInvocationCaptureGeometry;
    let metadata=Metadata::new(context)?;
    let parts=[size_of::<(&SharedCapturePlan,usize,CapturePhase,u64,&CaptureSlicePartition,
        PartitionCaptureCombination,usize,bool,&WorkspaceTensor,&WorkspaceContext,&mut Vec<WorkspaceTensor>,
        Option<&Cell<CaptureNativePopulation>>,Option<&[Cell<Option<WorkspaceFloatingType>>]>)>(),
        size_of::<[i32;32]>(),size_of::<Result<()>>(),size_of::<std::iter::Enumerate<std::slice::Iter<'_,u64>>>(),
        size_of::<PartitionInvocationCaptureGeometry<'_>>(),
        size_of::<std::result::Result<PartitionInvocationCaptureGeometry<'_>,eredu_runtime::capture::partition::PartitionInvocationCaptureSourceError>>(),
        PartitionInvocationCaptureGeometry::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
    if phase!=CapturePhase::Decode||prediction==0{return Err(metadata.coordinate());}
    context.validate_values([value])?;
    let rank=projection.local_shape().len();
    if rank==0||rank>32||value.layout().representation().is_none(){return Err(metadata.coordinate());}
    let mut shape=[0i32;32];
    for (axis,&n) in projection.local_shape().iter().enumerate(){shape[axis]=i32::try_from(n).map_err(|cause|metadata.error(cause))?;}
    if value.shape()!=&shape[..rank]{return Err(metadata.coordinate());}
    record_source(scalars,index,context,value)?;
    if !produces||projection.fragments().is_empty(){return record_replica(transfers,context,roots,value);}
    let geometry=PartitionInvocationCaptureGeometry::prepare(source.admission(),index,phase,prediction,projection,0,combination)
        .map_err(|cause|metadata.error(cause))?;
    let equation=super::invocation::PartitionInvocationEquation::from_geometry(geometry,producer,context)?;
    let (_,population)=equation.trace(value,context,roots)?;
    record_population(transfers,context,population)
}

impl CaptureWorkspaceObserver<'_> {
    /// An absent pipeline rank follows global accounting without manufacturing a
    /// tensor, source precision or native completion. Local hooks remain required.
    pub(in super::super) fn finish_absent_invocation_projections(&mut self)->Result<()> {
        let Some((layouts,rank))=self.placement else{return Ok(())};
        let metadata=Metadata::new(&self.context)?;
        let prediction=self.active.ok_or_else(||metadata.coordinate())?;
        let phase=self.phase(prediction);
        self.context.charge_metadata(size_of::<(&mut Self,usize,u64,CapturePhase,CaptureUsage,
            CaptureRecordStatus,Result<()>,ContiguousPartitionCaptureSource<'_>)>()
            .checked_add(ContiguousPartitionCaptureSource::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
        let policy=self.policy()?;
        for index in 0..self.statuses.len(){
            if self.statuses[index]!=CaptureRecordStatus::Missing{continue;}
            let selection=&self.source.admission().plan().selections[index];
            if layouts.complete_capture_source(&selection.path).is_some(){continue;}
            if matches!(self.source.admission().points()[index].value_type,
                eredu_core::ObservationValueType::RoutedUnits { .. }) {
                if super::super::routed::partition::local(self.source,(layouts,rank),index,&self.context)?.is_some(){
                    return Err(metadata.coordinate());
                }
                let geometry=policy.routed_geometry(index).map_err(|e|metadata.error(e))?;
                let usage=super::super::super::bounded_capture::estimate_routed_geometry(&geometry).map_err(|e|metadata.error(e))?;
                self.statuses[index]=CaptureRecordStatus::Consumed;
                if policy.reserve_value(&mut self.ledger,usage).map_err(|e|metadata.error(e))?.is_some(){
                    self.statuses[index]=CaptureRecordStatus::Skipped;
                }
                continue;
            }
            let selected=layouts.contiguous_capture_source(&selection.path).map_err(|cause|metadata.error(cause))?;
            if selected.rank(rank).is_some(){return Err(metadata.coordinate());}
            let usage=invocation_usage(self.source,index,phase,prediction,&self.context)?;
            self.statuses[index]=CaptureRecordStatus::Consumed;
            if policy.reserve_value(&mut self.ledger,usage).map_err(|cause|metadata.error(cause))?.is_some(){
                self.statuses[index]=CaptureRecordStatus::Skipped;
            }
        }
        Ok(())
    }
}
