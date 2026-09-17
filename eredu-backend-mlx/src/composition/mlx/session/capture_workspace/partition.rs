//! Native equations selected by the actual original projected receipt source.
use super::*;
use crate::backend::array_copy::{PreparedCaptureSummary,PreparedCaptureHistogram};
use eredu_runtime::capture::partition::{PartitionCaptureReceiptPlan,PartitionCaptureNativeEstimate,
    PartitionCaptureFragmentSource,PartitionPrefillCapturePlan,PartitionPrefillCaptureGeometry,PartitionPrefillCaptureKind};
use eredu_core::checkpoint::TensorDtype;
use std::mem::{size_of,size_of_val};

pub(in crate::composition::mlx::session) fn estimate_geometry(source:&PartitionPrefillCaptureGeometry<'_>)
    ->std::result::Result<PartitionCaptureNativeEstimate,CaptureError> {
    let capture=match source.kind(){
        PartitionPrefillCaptureKind::Tensor(rows)=>super::super::bounded_capture::estimate_tensor_geometry(rows.logical_geometry()),
        PartitionPrefillCaptureKind::Summary(plan)=>super::super::bounded_capture::estimate_prefill_summary(plan),
        PartitionPrefillCaptureKind::Histogram(plan)=>super::super::bounded_capture::estimate_prefill_histogram(plan),
    }?;
    Ok(PartitionCaptureNativeEstimate{capture,generated_creation_bytes:0})
}

pub(super) struct PartitionPrefillEquation<'a> {
    source:PartitionPrefillCaptureGeometry<'a>,
    producer:usize,
    estimate:PartitionCaptureNativeEstimate,
}
impl<'a> PartitionPrefillEquation<'a> {
    pub(super) fn prepare(receipt:&'a PartitionCaptureReceiptPlan,producer:usize,fragment:usize,
        inference:InferenceGeometry,context:&WorkspaceContext)->Result<Self> {
        let metadata=Metadata::new(context)?;
        let controls=PartitionPrefillCapturePlan::control_bytes().and_then(|n|n.checked_add(Self::control_bytes()?))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(controls)?;
        let source=PartitionPrefillCapturePlan::prepare(receipt,producer,fragment,inference)
            .map_err(|cause|metadata.error(cause))?;
        Self::from_geometry(source.into_geometry(),producer,context)
    }
    pub(super) fn from_geometry(source:PartitionPrefillCaptureGeometry<'a>,producer:usize,context:&WorkspaceContext)->Result<Self>{
        let metadata=Metadata::new(context)?;
        context.charge_metadata(Self::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
        let estimate=estimate_geometry(&source).map_err(|cause|metadata.error(cause))?;
        Ok(Self{source,producer,estimate})
    }
    /// Actual native estimate supplied to the shared fragment allowance.
    /// The scalar witness must come from the recorded local value/source vote.
    pub(super) fn native_source(&self,dtype:TensorDtype)->PartitionCaptureFragmentSource<'_> {
        const RAW:CaptureTransform=CaptureTransform::Slice;
        let projection=self.source.projection();
        let selection=&self.source.admission().plan().selections[self.source.selection_index()];
        let transform=if matches!(selection.transform,CaptureTransform::Summary|CaptureTransform::Histogram{..})
            && matches!(self.source.kind(),PartitionPrefillCaptureKind::Tensor(_)){&RAW}else{&selection.transform};
        PartitionCaptureFragmentSource{producer:self.producer,fragment:self.source.fragment_index(),
            local_shape:projection.local_shape(),local_slice:projection.fragments()[self.source.fragment_index()].local(),
            transform,dtype,estimate:self.estimate}
    }
    /// Trace the same selected native primitives and completed scalar frontiers
    /// for one actual local tensor. No family, quota or operation admission is
    /// inferred from this descriptor; normal context/recipe source checks apply.
    pub(super) fn trace_fragment(&self,chunk:u64,value:&WorkspaceTensor,context:&WorkspaceContext,
        roots:&mut Vec<WorkspaceTensor>)->Result<(WorkspaceFloatingType,CaptureNativePopulation)> {
        let metadata=Metadata::new(context)?;
        context.charge_metadata(Self::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
        context.validate_values([value])?;
        let dtype=value.layout().representation().map(|r|r.dtype()).ok_or_else(||metadata.coordinate())?;
        let population=match self.source.kind() {
            PartitionPrefillCaptureKind::Tensor(rows)=>{
                let fragment=rows.fragment(chunk).map_err(|cause|metadata.error(cause))?;
                let program=CaptureTensorSelection::from_fragment(&fragment).map_err(|cause|metadata.error(cause))?;
                program.validate_workspace_source(value,context).map_err(|cause|metadata.error(cause))?;
                if fragment.output_elements()==0 {CaptureNativePopulation::default()}else{
                    metadata.reserve(roots,1)?;roots.push(value.clone());
                    program.trace_retained_within(value,context,roots).map_err(|cause|metadata.error(cause))?;
                    CaptureNativePopulation::raw(1).ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?
                }
            },
            PartitionPrefillCaptureKind::Summary(plan)=>{
                let fragment=plan.fragment(chunk).map_err(|cause|metadata.error(cause))?;
                let geometry=CaptureSummaryGeometry::prepare_partition(plan.admission(),plan.selection_index(),CapturePhase::Prefill,0,None,
                    self.source.projection(),self.source.fragment_index())
                    .and_then(|g|g.fragment(&fragment)).map_err(|cause|metadata.error(cause))?;
                let program=PreparedCaptureSummary::from_geometry(&geometry).map_err(|cause|metadata.error(cause))?;
                program.validate_workspace_source(value,context).map_err(|cause|metadata.error(cause))?;
                if fragment.selected_elements()==0 {CaptureNativePopulation::default()}else{
                    metadata.reserve(roots,1)?;roots.push(value.clone());program.trace(value,context,roots).map_err(|cause|metadata.error(cause))?;
                    program.population().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?
                }
            },
            PartitionPrefillCaptureKind::Histogram(plan)=>{
                let fragment=plan.fragment(chunk).map_err(|cause|metadata.error(cause))?;
                let geometry=CaptureHistogramGeometry::prepare_partition(plan.admission(),plan.selection_index(),CapturePhase::Prefill,0,None,
                    self.source.projection(),self.source.fragment_index())
                    .and_then(|g|g.fragment(&fragment)).map_err(|cause|metadata.error(cause))?;
                let program=PreparedCaptureHistogram::from_geometry(&geometry).map_err(|cause|metadata.error(cause))?;
                program.validate_workspace_source(value,context).map_err(|cause|metadata.error(cause))?;
                if fragment.selected_elements()==0 {CaptureNativePopulation::default()}else{
                    metadata.reserve(roots,1)?;roots.push(value.clone());program.trace(value,context,roots).map_err(|cause|metadata.error(cause))?;
                    program.population().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?
                }
            },
        };Ok((dtype,population))
    }
    fn control_bytes()->Option<usize> {
        let parts=[size_of::<Self>()*2,size_of::<Result<Self>>(),size_of::<PartitionCaptureNativeEstimate>()*2,size_of::<(PartitionPrefillCaptureGeometry<'_>,usize,&WorkspaceContext)>(),
            size_of::<(&PartitionCaptureReceiptPlan,usize,usize,InferenceGeometry,&WorkspaceContext)>(),
            size_of::<(&Self,u64,&WorkspaceTensor,&WorkspaceContext,&mut Vec<WorkspaceTensor>)>(),
            size_of::<PartitionCaptureFragmentSource<'_>>(),size_of::<TensorDtype>(),
            size_of::<Result<(WorkspaceFloatingType,CaptureNativePopulation)>>(),size_of::<WorkspaceFloatingType>(),
            size_of::<CaptureNativePopulation>(),size_of::<CaptureTensorSelection>(),size_of::<PreparedCaptureSummary>(),
            size_of::<PreparedCaptureHistogram<'_>>(),size_of::<CapturePrefillFragment<'_,'_>>(),
            size_of::<CapturePrefillTransformFragment<'_,'_>>(),size_of::<CaptureSummaryGeometry<'_>>(),size_of::<CaptureHistogramGeometry<'_>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}

pub(super) mod observer;

#[cfg(test)]
mod tests;

pub(in crate::composition::mlx) mod source;

pub(super) mod invocation;

/// Borrow the existing exact projected equation for another original source
/// owner, including an intervention's before/after companion. The geometry and
/// actual tensor remain distinct; this records no producer decision or grant.
pub(in crate::composition::mlx::session) fn trace_geometry(
    source:PartitionPrefillCaptureGeometry<'_>,producer:usize,chunk:u64,
    value:&WorkspaceTensor,context:&WorkspaceContext,roots:&mut Vec<WorkspaceTensor>,
)->Result<CaptureNativePopulation> {
    let parts=[
        size_of::<(PartitionPrefillCaptureGeometry<'_>,usize,u64,&WorkspaceTensor,&WorkspaceContext,&mut Vec<WorkspaceTensor>)>(),
        size_of::<PartitionPrefillEquation<'_>>(),
        size_of::<Result<PartitionPrefillEquation<'_>>>(),
        size_of::<Result<(WorkspaceFloatingType,CaptureNativePopulation)>>(),
        size_of::<Result<CaptureNativePopulation>>(),
    ];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
    let equation=PartitionPrefillEquation::from_geometry(source,producer,context)?;
    let (_,population)=equation.trace_fragment(chunk,value,context,roots)?;
    Ok(population)
}
