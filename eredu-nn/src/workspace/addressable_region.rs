//! Whole independently addressable provider source; no exchange fields.
use super::*;
use crate::{GroupSelection, Tensor, TensorParallelGroupedOutput};
use std::{mem::{size_of, size_of_val}, ops::Range};

pub(super) mod observation;
pub use observation::{WorkspaceAddressableObservationSource, WorkspaceAddressableObservationView, WorkspaceAddressableObservationLayout};

/// Exact scalar projection of the runtime's retained compact iteration plan.
/// It grants no cache lease, selected identity, memory or execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceAddressableChunkPlan {
    pub rows: usize,
    pub chunk_rows: usize,
    pub routes: usize,
    pub members: usize,
}
impl WorkspaceAddressableChunkPlan {
    pub fn validate(self) -> Result<(), WorkspaceMetadataError> {
        if self.chunk_rows == 0 || self.routes == 0 || self.members == 0 {
            return Err(WorkspaceMetadataError::Unqualified);
        }
        self.rows.checked_mul(self.routes).ok_or(WorkspaceMetadataError::Overflow)?;
        Ok(())
    }
    pub fn len(self) -> Option<usize> {
        if self.chunk_rows==0 {None}else{Some(self.rows.div_ceil(self.chunk_rows))}
    }
    pub fn is_empty(self) -> bool { self.rows == 0 }
    pub fn range(self, index: usize) -> Option<Range<usize>> {
        if index >= self.len()? { return None; }
        let start = index.checked_mul(self.chunk_rows)?;
        Some(start..start.saturating_add(self.chunk_rows).min(self.rows))
    }
    pub fn maximum_members(self, index: usize) -> Option<usize> {
        self.range(index)?.len().checked_mul(self.routes).map(|n|n.min(self.members))
    }
}

/// Borrowed architecture source of one existing addressable execution callback.
/// `local_members=None` is the ordinary replicated 0..members identity mapping.
/// A present map is the exact owner-local to global member table, not route IDs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkspaceAddressableRegionView<'a> {
    pub owner_group: &'a str,
    pub bank: u32,
    pub unit: usize,
    pub prefill: bool,
    pub chunks: WorkspaceAddressableChunkPlan,
    pub local_members: Option<&'a [usize]>,
    pub kernel: WorkspaceExpertKernel<'a>,
    pub tensor_partitions: Option<usize>,
    pub compact_scratch_bytes: u64,
    pub bulk_target_bytes: u64,
    /// Actual caller-owned callback/error storage, derived from that shared
    /// worker's Rust layouts. Descriptive only; it grants no native authority.
    pub callback_control_bytes: usize,
}
impl WorkspaceAddressableRegionView<'_> {
    pub fn validate(self) -> Result<(), WorkspaceMetadataError> {
        self.chunks.validate()?;
        let groups = match self.kernel {
            WorkspaceExpertKernel::Gated(spec) => spec.group_count(),
            WorkspaceExpertKernel::Linear(spec) => spec.group_count(),
            WorkspaceExpertKernel::Relu2(spec) => spec.group_count(),
        };
        let (input,output) = self.kernel.dimensions();
        if self.owner_group.is_empty() || input <= 0 || output <= 0
            || usize::try_from(groups).ok() != Some(self.chunks.members)
            || self.tensor_partitions == Some(0)
            || self.local_members.is_some_and(|members| members.len() != self.chunks.members
                || members.windows(2).any(|pair|pair[0] >= pair[1])) {
            return Err(WorkspaceMetadataError::Unqualified);
        }
        i32::try_from(self.chunks.rows).map_err(|_|WorkspaceMetadataError::Overflow)?;
        i32::try_from(self.chunks.routes).map_err(|_|WorkspaceMetadataError::Overflow)?;
        Ok(())
    }
    pub fn retain(self, context:&WorkspaceContext) -> Result<WorkspaceAddressableRegion,Error> {
        context.charge_metadata(size_of::<(Self,WorkspaceAddressableRegion,
            Result<WorkspaceAddressableRegion,Error>,Option<Vec<usize>>,WorkspaceGroupedBank)>())?;
        self.validate()?;
        let local_members=self.local_members.map(|members| {
            let mut values=context.metadata_vec(members.len())?;
            values.extend_from_slice(members);
            Ok::<_,Error>(values)
        }).transpose()?;
        Ok(WorkspaceAddressableRegion {
            owner_group:context.metadata_string(format_args!("{}",self.owner_group))?,
            bank:self.bank,unit:self.unit,prefill:self.prefill,chunks:self.chunks,
            local_members,kernel:self.kernel.retain(context)?,
            tensor_partitions:self.tensor_partitions,
            compact_scratch_bytes:self.compact_scratch_bytes,bulk_target_bytes:self.bulk_target_bytes,
            callback_control_bytes:self.callback_control_bytes,observation:None,
        })
    }
    /// Fixed outer declaration controls, before any spec clone or tensor work.
    pub fn control_bytes() -> Option<usize> {
        let values=[size_of::<Self>(),size_of::<WorkspaceAddressableChunkPlan>(),
            size_of::<ExpertRegionInputShape>(),size_of::<Result<(),WorkspaceMetadataError>>(),
            size_of::<[usize;4]>(),size_of::<[&[i32];4]>()];
        values.into_iter().try_fold(size_of_val(&values),usize::checked_add)
    }
}

#[derive(Clone, Debug)]
pub struct WorkspaceAddressableRegion {
    owner_group:String,
    bank:u32,
    unit:usize,
    prefill:bool,
    chunks:WorkspaceAddressableChunkPlan,
    local_members:Option<Vec<usize>>,
    kernel:WorkspaceGroupedBank,
    tensor_partitions:Option<usize>,
    compact_scratch_bytes:u64,
    bulk_target_bytes:u64,
    callback_control_bytes:usize,
    observation:Option<WorkspaceAddressableObservationSource>,
}
impl WorkspaceAddressableRegion {
    pub fn observation(&self)->Option<WorkspaceAddressableObservationSource>{self.observation}
    pub fn retain(&self,context:&WorkspaceContext)->Result<Self,Error>{
        let mut result=self.as_view().retain(context)?;result.observation=self.observation;Ok(result)
    }

    pub fn as_view(&self)->WorkspaceAddressableRegionView<'_> {
        WorkspaceAddressableRegionView {
            owner_group:&self.owner_group,bank:self.bank,unit:self.unit,prefill:self.prefill,
            chunks:self.chunks,local_members:self.local_members.as_deref(),
            kernel:match &self.kernel {
                WorkspaceGroupedBank::GatedProduct(spec)=>WorkspaceExpertKernel::Gated(spec),
                WorkspaceGroupedBank::Linear(spec)=>WorkspaceExpertKernel::Linear(spec),
                WorkspaceGroupedBank::Relu2(spec)=>WorkspaceExpertKernel::Relu2(spec),
            },
            tensor_partitions:self.tensor_partitions,
            compact_scratch_bytes:self.compact_scratch_bytes,bulk_target_bytes:self.bulk_target_bytes,
            callback_control_bytes:self.callback_control_bytes,
        }
    }
}

/// Records the whole ordinary provider as one distinct source. The original
/// four tensors are operands; normalization, chunk selections and final joins
/// remain inside its exact native source and are not charged as a second graph.
pub fn record_addressable_region(source:WorkspaceAddressableRegionView<'_>,
    input:&WorkspaceTensor,routes:&GroupSelection<WorkspaceTensor>,context:&WorkspaceContext)
    ->Result<TensorParallelGroupedOutput<WorkspaceTensor>,Error> {
    record_addressable_region_with_observation(source,input,routes,context,None)
}

/// Records a borrowed prospective observer on the same whole provider source.
/// Native callbacks retain actual member/route identity and do not execute here.
pub fn record_addressable_region_with_observation(source:WorkspaceAddressableRegionView<'_>,
    input:&WorkspaceTensor,routes:&GroupSelection<WorkspaceTensor>,context:&WorkspaceContext,
    observe:Option<&mut dyn FnMut(WorkspaceAddressableObservationView<'_>)->Result<WorkspaceAddressableObservationSource,Error>>)
    ->Result<TensorParallelGroupedOutput<WorkspaceTensor>,Error> {
    source.validate()?;
    context.charge_metadata(WorkspaceAddressableRegionView::control_bytes()
        .and_then(|n| n.checked_add(size_of::<(WorkspaceOperationKind,
            Vec<WorkspaceLayout>,Vec<WorkspaceTensor>,TensorParallelGroupedOutput<WorkspaceTensor>,
            Result<TensorParallelGroupedOutput<WorkspaceTensor>,Error>,[&WorkspaceTensor;4])>()))
        .ok_or(WorkspaceMetadataError::Overflow)?)?;
    let shape=ExpertRegionInputShape::inspect(input.shape(),routes.group_indices().shape())?;
    let (width,output_width)=source.kernel.dimensions();
    if usize::try_from(shape.rows).ok()!=Some(source.chunks.rows)
        || usize::try_from(shape.routes).ok()!=Some(source.chunks.routes)
        || shape.width!=width || input.shape().len()<2
        || routes.group_indices().shape()!=routes.selected_scores().shape()
        || routes.group_indices().shape()!=routes.coefficients().shape()
        || !matches!(routes.group_indices().layout().dtype(),WorkspaceDtype::Int32|WorkspaceDtype::Uint32)
        || input.layout().dtype()!=WorkspaceDtype::Float32
        || routes.selected_scores().layout().dtype()!=WorkspaceDtype::Float32
        || routes.coefficients().layout().dtype()!=WorkspaceDtype::Float32 {
        return Err(WorkspaceMetadataError::Unqualified.into());
    }
    let mut output_shape=context.metadata_vec(input.shape().len())?;
    output_shape.extend_from_slice(input.shape());
    *output_shape.last_mut().ok_or(WorkspaceMetadataError::Unqualified)?=output_width;
    let bias=source.kernel.separate_bias(source.tensor_partitions);
    let mut layouts=context.metadata_vec(1+usize::from(bias))?;
    layouts.push(context.layout(&output_shape,WorkspaceDtype::Float32)?);
    if bias { layouts.push(context.layout(&output_shape,WorkspaceDtype::Float32)?); }
    let mut region=source.retain(context)?;
    if let Some(observe)=observe {
        region.observation=Some(observation::inspect(source,
            &[input,routes.group_indices(),routes.selected_scores(),routes.coefficients()],context,observe)?);
    }
    let region=context.box_metadata(region)?;
    let mut output=context.execute(WorkspaceOperationKind::AddressableRegion(region),
        &[input,routes.group_indices(),routes.selected_scores(),routes.coefficients()],layouts)?;
    let bias=if bias {Some(output.pop().ok_or(WorkspaceMetadataError::Unqualified)?)}else{None};
    Ok(TensorParallelGroupedOutput::new(output.pop().ok_or(WorkspaceMetadataError::Unqualified)?,bias))
}
