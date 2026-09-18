//! Original Host source from the retained architecture's sparse ownership table.
use super::*;
use std::sync::Arc;
use crate::composition::mlx::session::model_session::partition_capture::LoadedPartitionCapture;
use eredu_architectures::component_partition::{RoutedPartitionCaptureSource,RoutedPartitionCaptureRank};
use eredu_core::capture::{CaptureCoordinateProjectionPlan,CaptureRoutedUnitsGeometry,
    PartitionRoutedUnitCaptureLayout,PartitionRoutedUnitCaptureRequest,RoutedUnitGeometry};
use eredu_core::capture::PartitionCaptureContext;
use eredu_runtime::capture::partition::{PartitionCaptureRoutedProducerSource,
    PartitionCaptureRoutedFragmentGeometry,PartitionCaptureRoutedLocalSource,PreparedPartitionRoutedSource};

#[derive(Debug)]
struct Native { producer:usize, fragment:usize, slice:ResolvedCaptureSlice, estimate:PartitionCaptureNativeEstimate }
pub(super) struct RowSource {
    index:usize, bank:RoutedUnitGeometry, source_tokens:u64, native:Vec<Native>,
    // Borrowed native source labels and immutable architecture maps remain the
    // same loaded owner. The issued runtime source copies them under its payer.
    loaded:LoadedPartitionCapture,
}
impl std::fmt::Debug for RowSource {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("RoutedHostSource").field("index",&self.index).field("bank",&self.bank)
            .field("source_tokens",&self.source_tokens).field("native",&self.native).finish_non_exhaustive()
    }
}
fn slice(axes:[[u64;3];4],metadata:&HostMetadataFunding)->Result<ResolvedCaptureSlice,Error> {
    Ok(ResolvedCaptureSlice{starts:copy(&axes[0],metadata)?,ends:copy(&axes[1],metadata)?,
        strides:copy(&axes[2],metadata)?,shape:copy(&axes[3],metadata)?})
}
/// Exact ordinary projection count, including permuted and strided unit maps.
/// No projection storage or authority is issued by this borrowed traversal.
fn fragment_limit(selected:&RoutedPartitionCaptureSource<'_>,geometry:&CaptureRoutedUnitsGeometry<'_>,
    metadata:&HostMetadataFunding)->Result<usize,Error> {
    let mut global=[0u64;3];
    for (out,&n) in global.iter_mut().zip(geometry.source_shape()) {*out=u64::try_from(n).map_err(|_|overflow())?;}
    let mut shape=[0u64;3];
    for (out,&n) in shape.iter_mut().zip(geometry.shape()) {*out=u64::try_from(n).map_err(|_|overflow())?;}
    let axes=[geometry.starts().try_into().map_err(|_|unknown())?,geometry.ends().try_into().map_err(|_|unknown())?,
        geometry.strides().try_into().map_err(|_|unknown())?,shape];
    let selected_slice=slice(axes,metadata)?;
    let mut total=0usize;
    for rank in 0..selected.world_size() {
        let Some(local)=selected.rank(rank).filter(|local|local.produces) else {continue};
        let plan=CaptureCoordinateProjectionPlan::prepare(&global,&selected_slice,2,
            local.ownership.coordinates.units(),local.ownership.coordinates.units().local_count().max(1))
            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
        metadata.reserve_metadata(plan.control_bytes())?;
        total=total.checked_add(plan.fragments()).ok_or_else(overflow)?;
    }
    Ok(total.max(1))
}
pub(super) fn prepare(loaded:&LoadedPartitionCapture,source:&SharedCapturePlan,
    context:&PartitionCaptureContext,index:usize,inference:InferenceGeometry,metadata:&HostMetadataFunding)
    ->Result<Prototype,Error> {
    let selection=&source.admission().plan().selections[index];
    let selected=loaded.layouts().routed_capture_source(&selection.path)
        .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
    let geometry=CaptureRoutedUnitsGeometry::prepare(source.admission(),index,context.phase,context.prediction,context.invocation)
        .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
    let eredu_core::ObservationValueType::RoutedUnits{geometry:declared,routing,..}=&source.admission().points()[index].value_type
        else{return Err(unknown())};
    if geometry.bank()!=selected.geometry() || *declared!=selected.geometry() || routing!=selected.routing()
        || selected.effective()!=(source.admission().points()[index].position==eredu_core::ObservationPosition::AfterIntervention)
        || selected.world_size()!=loaded.source_labels().2.participant_count() {
        return Err(memory(WorkingMemoryError::IdentityMismatch));
    }
    let limits=PartitionCaptureReceiptLimits{max_producers:selected.world_size(),
        max_fragments:fragment_limit(&selected,&geometry,metadata)?,
        max_record_bytes:source.admission().plan().limits.per_step.encoded_bytes};
    let mut producers=metadata.metadata_vec(selected.producer_count())?;
    for rank in 0..selected.world_size(){if let Some(local)=selected.rank(rank).filter(|local|local.produces) {
        producers.push(PartitionCaptureRoutedProducerSource{rank,ownership:local.ownership});
    }}
    let mut quotation=CaptureLedger::new(source.admission());quotation.begin_step();
    let receipt=PartitionCaptureReceiptPlan::new_routed_shared_funded(source,context,&producers,
        selected.world_size(),limits,metadata,&mut quotation)
        .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
    let count=receipt.producers().try_fold(0usize,|n,(_,p)|n.checked_add(p.fragments().len())).ok_or_else(overflow)?;
    let bank=geometry.bank();let source_tokens=u64::try_from(geometry.source_shape()[0]).map_err(|_|overflow())?;
    let mut native=metadata.metadata_vec(count)?;
    for (producer,projection) in receipt.producers(){
        let ownership=receipt.routed_producer(producer).ok_or_else(unknown)?;
        for fragment in 0..projection.fragments().len(){
            let axes=CaptureRoutedUnitsGeometry::fragment_axes(projection,fragment)
                .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
            let slice=slice(axes,metadata)?;
            let request=PartitionRoutedUnitCaptureRequest{geometry:bank,source_tokens,ownership,slice:&slice};
            // Same existing collector source estimate as ordinary partition
            // capture; the five evaluated loans create no deferred input tensor.
            let capture=crate::composition::mlx::session::bounded_capture::estimate_partition_routed_geometry(&request)
                .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
            native.push(Native{producer,fragment,slice,estimate:PartitionCaptureNativeEstimate{
                capture,generated_creation_bytes:0}});
        }
    }
    if context.prediction==0 {
        eredu_core::capture::CaptureRoutedPrefillPlan::prepare(source.admission(),index,inference)
            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
    }
    Ok(Prototype{source:super::RowSource::Routed(RowSource{index,bank,source_tokens,native,loaded:loaded.clone()}),receipt})
}
impl RowSource {
    pub(super) fn bind(self,source:&SharedCapturePlan,rank:usize,scalar:Option<WorkspaceFloatingType>,
        inference:InferenceGeometry,prediction:u64,metadata:&HostMetadataFunding)
        ->Result<PreparedPartitionRoutedSource,Error> {
        let selection=source.admission().plan().selections.get(self.index).ok_or_else(unknown)?;
        let selected=self.loaded.layouts().routed_capture_source(&selection.path)
            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
        if selected.geometry()!=self.bank || rank>=selected.world_size(){return Err(memory(WorkingMemoryError::IdentityMismatch));}
        let dtype=scalar.map(|value|match value {WorkspaceFloatingType::Float32=>TensorDtype::F32,
            WorkspaceFloatingType::Float16=>TensorDtype::F16,WorkspaceFloatingType::Bfloat16=>TensorDtype::Bf16});
        let local=match selected.rank(rank) {
            Some(local)=>Some(PartitionCaptureRoutedLocalSource{producer:rank,
                layout:PartitionRoutedUnitCaptureLayout{geometry:self.bank,source_tokens:self.source_tokens,ownership:local.ownership},dtype}),
            None if dtype.is_none()=>None,
            None=>return Err(memory(WorkingMemoryError::IdentityMismatch)),
        };
        let mut producers=metadata.metadata_vec(selected.producer_count())?;
        for rank in 0..selected.world_size(){if let Some(local)=selected.rank(rank).filter(|local|local.produces){
            producers.push(PartitionCaptureRoutedProducerSource{rank,ownership:local.ownership});
        }}
        let mut native=metadata.metadata_vec(self.native.len())?;
        for row in &self.native {
            let owner=selected.rank(row.producer).filter(|local|local.produces).ok_or_else(unknown)?;
            native.push(PartitionCaptureRoutedFragmentGeometry{producer:row.producer,fragment:row.fragment,estimate:row.estimate,
                request:PartitionRoutedUnitCaptureRequest{geometry:self.bank,source_tokens:self.source_tokens,
                    ownership:owner.ownership,slice:&row.slice}});
        }
        let prepared=if prediction==0 {PreparedPartitionRoutedSource::new_local_prefill(source,self.index,
            &producers,&native,local,inference,metadata)}else{PreparedPartitionRoutedSource::new_local_decode(source,self.index,
            &producers,&native,local,prediction,metadata)};
        prepared.map_err(|cause|Error::Neural(metadata.metadata_source(cause)))
    }
}
pub(super) fn controls()->Option<usize> {
    let parts=[size_of::<RowSource>()*2,size_of::<Native>()*2,size_of::<LoadedPartitionCapture>() *2,
        size_of::<Vec<Native>>(),size_of::<RoutedPartitionCaptureSource<'_>>() *2,
        size_of::<Option<RoutedPartitionCaptureRank<'_>>>() *2,size_of::<CaptureRoutedUnitsGeometry<'_>>() *2,
        size_of::<CaptureCoordinateProjectionPlan<'_>>() *2,size_of::<[[u64;3];4]>() *2,size_of::<[u64;3]>()*2,
        size_of::<ResolvedCaptureSlice>()*2,size_of::<PartitionRoutedUnitCaptureRequest<'_>>() *2,
        size_of::<PartitionRoutedUnitCaptureLayout<'_>>(),size_of::<PartitionCaptureRoutedLocalSource<'_>>(),
        size_of::<Vec<PartitionCaptureRoutedProducerSource<'_>>>()*2,size_of::<Vec<PartitionCaptureRoutedFragmentGeometry<'_>>>(),
        size_of::<PartitionCaptureRoutedProducerSource<'_>>(),size_of::<PartitionCaptureRoutedFragmentGeometry<'_>>(),
        size_of::<Result<Prototype,Error>>(),size_of::<Result<PreparedPartitionRoutedSource,Error>>(),
        size_of::<Result<ResolvedCaptureSlice,Error>>(),size_of::<Result<usize,Error>>(),
        size_of::<Result<eredu_core::capture::CaptureUsage,eredu_core::capture::CaptureError>>(),
        size_of::<Result<[[u64;3];4],eredu_core::capture::RoutedUnitValidationError>>(),
        size_of::<(&LoadedPartitionCapture,&SharedCapturePlan,&PartitionCaptureContext,usize,InferenceGeometry,&HostMetadataFunding)>(),
        size_of::<(RowSource,&SharedCapturePlan,usize,Option<WorkspaceFloatingType>,InferenceGeometry,u64,&HostMetadataFunding)>(),
        size_of::<(&RoutedPartitionCaptureSource<'_>,&CaptureRoutedUnitsGeometry<'_>,&HostMetadataFunding)>(),
        size_of::<([[u64;3];4],&HostMetadataFunding)>(),size_of::<Option<TensorDtype>>(),
        size_of::<std::ops::Range<usize>>()*3,size_of::<std::slice::Iter<'_,Native>>(),
        size_of::<std::iter::Zip<std::slice::IterMut<'_,u64>,std::slice::Iter<'_,usize>>>(),
        RoutedPartitionCaptureSource::control_bytes()?,CaptureRoutedUnitsGeometry::partition_control_bytes()?,
        size_of::<eredu_core::capture::CaptureRoutedPrefillPlan<'_>>()*2,
        size_of::<Result<eredu_core::capture::CaptureRoutedPrefillPlan<'_>,eredu_core::capture::CapturePrefillGeometryError>>(),
        size_of::<(&eredu_core::capture::AdmittedCapturePlan,usize,InferenceGeometry)>(),
        size_of::<&eredu_core::ObservationPoint>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
