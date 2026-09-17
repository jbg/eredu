//! One exact projected prefill mechanism shared by quotation and native hooks.
use super::*;
use eredu_core::InferenceGeometry;
use std::mem::{size_of,size_of_val};

/// Source failures retain the original geometry error without formatted text.
#[derive(Debug,thiserror::Error)]
pub enum PartitionPrefillCaptureSourceError {
    /// This receipt does not describe an applicable scheduled dense fragment.
    #[error("partition prefill source differs from its original receipt")]
    Source,
    /// The existing temporal/spatial worker rejected the exact projection.
    #[error(transparent)] Geometry(#[from] CapturePrefillPartitionError),
}
/// The actual ordinary worker, including raw additive terms before reduction.
#[derive(Debug)]
pub enum PartitionPrefillCaptureKind<'a> {
    /// Selected F32 values collected by the same fixed scatter/preview worker.
    Tensor(CapturePrefillRowAssembly<'a>),
    /// Disjoint partial statistics from each actual selected chunk.
    Summary(CapturePrefillTransformPlan<'a>),
    /// Disjoint fixed-edge partial bins from each actual selected chunk.
    Histogram(CapturePrefillTransformPlan<'a>),
}
/// Source geometry shared by cold quotation and the actual receipt consumer.
/// This borrows the original admission and projection; it carries no receipt,
/// scalar witness, account, native authority, or submission identity.
#[derive(Debug)]
pub struct PartitionPrefillCaptureGeometry<'a> {
    admission:&'a AdmittedCapturePlan,
    selection_index:usize,
    projection:&'a CaptureSlicePartition,
    fragment:usize,
    kind:PartitionPrefillCaptureKind<'a>,
}
impl<'a> PartitionPrefillCaptureGeometry<'a> {
    /// Select the existing temporal/spatial worker from exact retained geometry.
    /// Raw additive terms remain raw until the global F64-to-F32 assembly.
    pub fn prepare(admission:&'a AdmittedCapturePlan,selection_index:usize,
        projection:&'a CaptureSlicePartition,fragment:usize,combination:PartitionCaptureCombination,
        inference:InferenceGeometry)->Result<Self,PartitionPrefillCaptureSourceError> {
        use PartitionPrefillCaptureSourceError as E;
        if projection.fragments().get(fragment).is_none(){return Err(E::Source);}
        let selection=admission.plan().selections.get(selection_index).ok_or(E::Source)?;
        let kind=match (&selection.transform,combination) {
            (CaptureTransform::Summary,PartitionCaptureCombination::Disjoint)=>PartitionPrefillCaptureKind::Summary(
                CapturePrefillTransformPlan::prepare_partition(admission,selection_index,inference,projection,fragment,combination)?),
            (CaptureTransform::Histogram{..},PartitionCaptureCombination::Disjoint)=>PartitionPrefillCaptureKind::Histogram(
                CapturePrefillTransformPlan::prepare_partition(admission,selection_index,inference,projection,fragment,combination)?),
            (CaptureTransform::Summary|CaptureTransform::Histogram{..},PartitionCaptureCombination::SumF64ToF32)=>PartitionPrefillCaptureKind::Tensor(
                CapturePrefillRowAssembly::prepare_additive_transform_partition(admission,selection_index,inference,projection,fragment)?),
            (CaptureTransform::FullTensor|CaptureTransform::Slice|CaptureTransform::Preview{..},_)=>PartitionPrefillCaptureKind::Tensor(
                CapturePrefillRowAssembly::prepare_partition(admission,selection_index,inference,projection,fragment,combination)
                    .map_err(CapturePrefillPartitionError::from)?),
            _=>return Err(E::Source),
        };
        Ok(Self{admission,selection_index,projection,fragment,kind})
    }
    /// Actual ordinary local transform or raw contribution worker.
    pub fn kind(&self)->&PartitionPrefillCaptureKind<'a>{&self.kind}
    /// Borrowed original admission, independent of any generated receipt label.
    pub const fn admission(&self)->&'a AdmittedCapturePlan{self.admission}
    /// Original selection ordinal.
    pub const fn selection_index(&self)->usize{self.selection_index}
    /// Retained exact spatial projection.
    pub const fn projection(&self)->&'a CaptureSlicePartition{self.projection}
    /// Fragment ordinal in that projection.
    pub const fn fragment_index(&self)->usize{self.fragment}
    /// Fixed geometry construction frames, with no backing allocation.
    pub fn control_bytes()->Option<usize> {
        let parts=[size_of::<Self>()*2,size_of::<PartitionPrefillCaptureKind<'_>>(),
            size_of::<Result<Self,PartitionPrefillCaptureSourceError>>(),size_of::<PartitionPrefillCaptureSourceError>(),
            size_of::<(&AdmittedCapturePlan,usize,&CaptureSlicePartition,usize,PartitionCaptureCombination,InferenceGeometry)>(),
            CapturePrefillRowAssembly::partition_preparation_control_bytes()?,
            CapturePrefillRowAssembly::additive_partition_preparation_control_bytes()?,
            CapturePrefillTransformPlan::partition_preparation_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
/// Payload-free borrowed receipt source. The runtime keeps the actual receipt
/// while the cold equation can consume the same source geometry independently.
#[derive(Debug)]
pub struct PartitionPrefillCapturePlan<'a> {
    receipt:&'a PartitionCaptureReceiptPlan,
    producer:usize,
    geometry:PartitionPrefillCaptureGeometry<'a>,
}
impl<'a> PartitionPrefillCapturePlan<'a> {
    /// Authenticate original receipt identity before selecting its shared worker.
    pub fn prepare(receipt:&'a PartitionCaptureReceiptPlan,producer:usize,fragment:usize,inference:InferenceGeometry)
        ->Result<Self,PartitionPrefillCaptureSourceError> {
        use PartitionPrefillCaptureSourceError as E;
        let context=receipt.context();let source=receipt.shared_plan_source().ok_or(E::Source)?;
        let projection=receipt.producer(producer).ok_or(E::Source)?;
        if context.phase!=CapturePhase::Prefill || context.prediction!=0 || context.invocation.is_some()
            || context.capture_plan_identity!=source.admission().identity() || receipt.routed_producer(producer).is_some()
            {return Err(E::Source);}
        let geometry=PartitionPrefillCaptureGeometry::prepare(source.admission(),context.selection_index,
            projection,fragment,receipt.combination(),inference)?;
        Ok(Self{receipt,producer,geometry})
    }
    /// Actual source-bound row or nonlinear worker.
    pub fn kind(&self)->&PartitionPrefillCaptureKind<'a>{self.geometry.kind()}
    /// Move the same borrowed descriptive geometry into a cold native equation.
    /// This does not carry or grant the enclosing receipt's delivery authority.
    pub fn into_geometry(self)->PartitionPrefillCaptureGeometry<'a>{self.geometry}
    /// Original retained receipt; its identity is not recreated from shapes.
    pub fn receipt(&self)->&'a PartitionCaptureReceiptPlan{self.receipt}
    /// Authoritative world rank.
    pub const fn producer(&self)->usize{self.producer}
    /// Actual local fragment ordinal.
    pub const fn fragment_index(&self)->usize{self.geometry.fragment_index()}
    /// Exact fixed constructor/control frames; no source storage is allocated.
    pub fn control_bytes()->Option<usize> {
        let parts=[size_of::<Self>()*2,size_of::<PartitionPrefillCaptureGeometry<'_>>(),
            size_of::<Result<Self,PartitionPrefillCaptureSourceError>>(),size_of::<PartitionPrefillCaptureSourceError>(),
            size_of::<(&PartitionCaptureReceiptPlan,usize,usize,InferenceGeometry)>(),
            PartitionPrefillCaptureGeometry::control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
