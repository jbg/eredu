//! Source-derived dynamic expert regions. Row limits are not allocation grants.
use super::*;
use crate::Tensor;
use crate::{GroupedGatedProductSpec, GroupedLinearSpec, GroupedRelu2Spec};
use std::mem::{size_of, size_of_val};
pub(super) mod observation;

/// The same four parent reshape equations used by cold recording and actual
/// route preparation. Only semantic rows/axes occur here, never selected IDs.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct ExpertRegionInputShape { pub rows:i32, pub width:i32, pub routes:i32 }
impl ExpertRegionInputShape {
    pub fn inspect(input:&[i32],selection:&[i32])->Result<Self,crate::Error> {
        let invalid=||crate::Error::backend("expert region input and selection axes differ");
        let (&width,prefix)=input.split_last().ok_or_else(invalid)?;
        let (&routes,selected)=selection.split_last().ok_or_else(invalid)?;
        let product=|shape:&[i32]|shape.iter().try_fold(1i32,|n,v|if *v<0{None}else{n.checked_mul(*v)});
        let rows=product(prefix).ok_or_else(invalid)?;
        if width<=0||routes<=0||product(selected)!=Some(rows){return Err(invalid());}
        Ok(Self{rows,width,routes})
    }
    pub fn normalize<T:crate::Tensor>(self,input:&T,indices:&T,scores:&T,coefficients:&T,
        context:&T::Context)->Result<[T;4],crate::Error> {
        Ok([input.reshape(&[self.rows,self.width],context)?,
            indices.reshape(&[self.rows,self.routes],context)?,
            scores.reshape(&[self.rows,self.routes],context)?,
            coefficients.reshape(&[self.rows,self.routes],context)?])
    }
    pub fn control_bytes<T>()->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<Result<Self,crate::Error>>(),
            size_of::<[T;4]>(),size_of::<Result<[T;4],crate::Error>>(),size_of::<[i32;2]>(),
            size_of::<(&[i32],&[i32])>(),size_of::<(&T,&T,&T,&T)>()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}

/// Borrows the exact constructor-selected local numerical equation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorkspaceExpertKernel<'a> {
    Gated(&'a GroupedGatedProductSpec),
    Linear(&'a GroupedLinearSpec),
    Relu2(&'a GroupedRelu2Spec),
}
impl WorkspaceExpertKernel<'_> {
    pub fn reduction(self) -> crate::GroupReduction {
        match self { Self::Gated(v) => v.reduction(), Self::Linear(v) => v.reduction(),
            Self::Relu2(_) => crate::GroupReduction::Sum }
    }
    pub fn separate_bias(self, partitions: Option<usize>) -> bool {
        if partitions.is_none() { return false; }
        match self {
            Self::Gated(v) => match v.layout() {
                crate::GatedProductGroupLayout::Packed { down, .. } => down.bias().is_some(),
                crate::GatedProductGroupLayout::Independent(groups) => groups.iter().any(|v| v.down().bias().is_some()),
                _ => false,
            },
            Self::Relu2(v) => v.down().bias().is_some(),
            Self::Linear(_) => false,
        }
    }
    pub fn dimensions(self) -> (i32, i32) {
        match self {
            Self::Gated(v) => (v.input_dimensions(), v.output_dimensions()),
            Self::Linear(v) => (v.input_dimensions(), v.output_dimensions()),
            Self::Relu2(v) => (v.hidden_dimensions(), v.hidden_dimensions()),
        }
    }
    /// The same compact operator constructor used after selected member
    /// acquisition. Only the group axis changes; equation policy is retained.
    pub fn compact(self, groups: i32, context: &WorkspaceContext)
        -> Result<WorkspaceGroupedBank, Error> {
        if groups <= 0 { return Err(WorkspaceMetadataError::Unqualified.into()); }
        context.charge_metadata(size_of::<(Self, i32, WorkspaceGroupedBank,
            Result<WorkspaceGroupedBank, Error>)>())?;
        Ok(match self {
            Self::Gated(spec) => WorkspaceGroupedBank::GatedProduct(
                context.clone_metadata(spec)?.with_group_geometry(groups, spec.intermediate_dimensions())?),
            Self::Linear(spec) => WorkspaceGroupedBank::Linear(
                context.clone_metadata(spec)?.with_group_count(groups)?),
            Self::Relu2(spec) => WorkspaceGroupedBank::Relu2(
                context.clone_metadata(spec)?.with_group_count(groups)?),
        })
    }

    pub(super) fn retain(self, context: &WorkspaceContext) -> Result<WorkspaceGroupedBank, Error> {
        Ok(match self {
            Self::Gated(v) => WorkspaceGroupedBank::GatedProduct(context.clone_metadata(v)?),
            Self::Linear(v) => WorkspaceGroupedBank::Linear(context.clone_metadata(v)?),
            Self::Relu2(v) => WorkspaceGroupedBank::Relu2(context.clone_metadata(v)?),
        })
    }
}

/// Finite invocation counts supplied by the shared architecture route driver.
/// Counts describe children and completion roots; they grant no native storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceExpertMovementPopulation {
    pub row_gathers: usize,
    pub scalar_gathers: usize,
    pub zeros: usize,
    pub indexed_adds: usize,
}
impl WorkspaceExpertMovementPopulation {
    pub fn counts(self) -> [usize; 4] {
        [self.row_gathers, self.scalar_gathers, self.zeros, self.indexed_adds]
    }
    pub fn children(self) -> Option<usize> {
        self.counts().into_iter().try_fold(0usize, usize::checked_add)
    }
    /// Each child lends its actual completed operands from the parent collector.
    pub fn parent_completions(self) -> Option<usize> {
        self.row_gathers.checked_add(self.scalar_gathers)?.checked_mul(2)?
            .checked_add(self.zeros)?.checked_add(self.indexed_adds.checked_mul(3)?)
    }
}

/// Semantic payloads in the architecture's shared expert route itinerary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceExpertTransfer { ForwardIndex, ForwardInput, ForwardScores,
    ForwardCoefficients, ReverseOutput, ReverseBias, ReverseIndex }
impl WorkspaceExpertTransfer {
    pub fn reverse(self)->bool {
        matches!(self,Self::ReverseOutput|Self::ReverseBias|Self::ReverseIndex)
    }
}
/// Finite paid-by-value transfer directory; no count values or native authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceExpertTransfers(pub [Option<WorkspaceExpertTransfer>;9]);
impl WorkspaceExpertTransfers {
    pub fn iter(&self)->impl Iterator<Item=WorkspaceExpertTransfer>+'_ {self.0.iter().copied().flatten()}
    pub fn len(self)->usize {self.iter().count()}
    pub fn get(self,index:usize)->Option<WorkspaceExpertTransfer> {self.iter().nth(index)}
}

/// Architecture construction and the current invocation provide this view.
/// No selected IDs, peer counts, native resource, or allocation authority occurs
/// here. A native producer must bind the completed source before executing it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkspaceExpertRegionView<'a> {
    pub bank: u32,
    pub unit: usize,
    pub prefill: bool,
    pub group: eredu_core::CollectiveGroupId,
    pub rank: usize,
    pub peers: usize,
    pub source_rows: usize,
    pub routes_per_row: usize,
    pub owners: &'a [usize],
    pub owner_local: &'a [usize],
    pub kernel: WorkspaceExpertKernel<'a>,
    pub tensor_partitions: Option<usize>,
    /// Exact provider votes, in tensor/expert/pipeline-wave order.
    pub provider_tensor_group: Option<eredu_core::CollectiveGroupId>,
    pub provider_wave_group: Option<eredu_core::CollectiveGroupId>,
    pub movement: WorkspaceExpertMovementPopulation,
    pub transfers: WorkspaceExpertTransfers,
}
impl WorkspaceExpertRegionView<'_> {
    pub fn provider_groups(self) -> impl Iterator<Item=eredu_core::CollectiveGroupId> {
        [self.provider_tensor_group,Some(self.group),self.provider_wave_group].into_iter().flatten()
    }
    pub fn selected_rows(self) -> Option<usize> {
        self.source_rows.checked_mul(self.routes_per_row)
    }
    pub fn local_experts(self) -> usize {
        self.owners.iter().filter(|owner| **owner == self.rank).count()
    }
    pub fn maximum_received_rows(self) -> Option<usize> {
        if self.local_experts() == 0 { Some(0) }
        else { self.selected_rows()?.checked_mul(self.peers) }
    }
    pub fn validate(self) -> Result<(), WorkspaceMetadataError> {
        let (input, output) = self.kernel.dimensions();
        if self.peers == 0 || self.rank >= self.peers || self.owners.is_empty()
            || self.owners.len() != self.owner_local.len() || self.routes_per_row == 0
            || self.owners.iter().any(|owner| *owner >= self.peers)
            || input <= 0 || output <= 0 || self.tensor_partitions == Some(0) {
            return Err(WorkspaceMetadataError::Unqualified);
        }
        self.movement.children().ok_or(WorkspaceMetadataError::Overflow)?;
        self.movement.parent_completions().ok_or(WorkspaceMetadataError::Overflow)?;
        let selected = self.selected_rows().ok_or(WorkspaceMetadataError::Overflow)?;
        let received = self.maximum_received_rows().ok_or(WorkspaceMetadataError::Overflow)?;
        i32::try_from(selected.max(received)).map_err(|_| WorkspaceMetadataError::Overflow)?;
        Ok(())
    }
    pub fn retain(self, context: &WorkspaceContext) -> Result<WorkspaceExpertRegion, Error> {
        let frames = [size_of::<Self>(), size_of::<WorkspaceExpertRegion>(),
            size_of::<Result<WorkspaceExpertRegion, Error>>(), size_of::<[Vec<usize>; 2]>(),
            size_of::<WorkspaceGroupedBank>(), size_of::<(usize, usize, Option<usize>)>()];
        context.charge_metadata(frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?)?;
        self.validate()?;
        let mut owners = context.metadata_vec(self.owners.len())?;
        owners.extend_from_slice(self.owners);
        let mut owner_local = context.metadata_vec(self.owner_local.len())?;
        owner_local.extend_from_slice(self.owner_local);
        Ok(WorkspaceExpertRegion {
            bank: self.bank, unit: self.unit, prefill: self.prefill, group: self.group,
            rank: self.rank, peers: self.peers, source_rows: self.source_rows,
            routes_per_row: self.routes_per_row, owners, owner_local,
            observation: None, kernel: self.kernel.retain(context)?, tensor_partitions: self.tensor_partitions, provider_tensor_group:self.provider_tensor_group, provider_wave_group:self.provider_wave_group, movement: self.movement, transfers: self.transfers,
        })
    }
}

/// Paid immutable region declaration retained by the ordinary equation report.
#[derive(Clone, Debug)]
pub struct WorkspaceExpertRegion {
    bank: u32,
    unit: usize,
    prefill: bool,
    group: eredu_core::CollectiveGroupId,
    rank: usize,
    peers: usize,
    source_rows: usize,
    routes_per_row: usize,
    owners: Vec<usize>,
    owner_local: Vec<usize>,
    kernel: WorkspaceGroupedBank,
    observation: Option<WorkspaceExpertObservationSource>,
    tensor_partitions: Option<usize>,
    provider_tensor_group: Option<eredu_core::CollectiveGroupId>,
    provider_wave_group: Option<eredu_core::CollectiveGroupId>,
    movement: WorkspaceExpertMovementPopulation,
    transfers: WorkspaceExpertTransfers,
}
impl WorkspaceExpertRegion {
    pub fn as_view(&self) -> WorkspaceExpertRegionView<'_> {
        WorkspaceExpertRegionView {
            bank: self.bank, unit: self.unit, prefill: self.prefill, group: self.group,
            rank: self.rank, peers: self.peers, source_rows: self.source_rows,
            routes_per_row: self.routes_per_row, owners: &self.owners, owner_local: &self.owner_local,
            kernel: match &self.kernel {
                WorkspaceGroupedBank::GatedProduct(v) => WorkspaceExpertKernel::Gated(v),
                WorkspaceGroupedBank::Linear(v) => WorkspaceExpertKernel::Linear(v),
                WorkspaceGroupedBank::Relu2(v) => WorkspaceExpertKernel::Relu2(v),
            }, tensor_partitions: self.tensor_partitions, provider_tensor_group:self.provider_tensor_group, provider_wave_group:self.provider_wave_group, movement: self.movement, transfers: self.transfers,
        }
    }
    pub fn kernel(&self) -> &WorkspaceGroupedBank { &self.kernel }
    pub fn observation(&self) -> Option<WorkspaceExpertObservationSource> { self.observation }
}

/// Records the explicit child boundary without reading routing values. Native
/// facts must supply the complete dynamic-region bound for this operation;
/// otherwise the existing report remains incomplete and admission rejects it.
pub fn record_expert_region<P: crate::Parameterized<WorkspaceTensor>>(
    source: WorkspaceExpertRegionView<'_>, bank: &P, input: &WorkspaceTensor,
    routes: &crate::GroupSelection<WorkspaceTensor>, context: &WorkspaceContext,
) -> Result<crate::TensorParallelGroupedOutput<WorkspaceTensor>, Error> {
    record_expert_region_with_observation(source,bank,input,routes,context,None)
}

/// Records the same region with an explicit prospective observer source. The
/// callback never receives a concrete batch or fabricated exchanged origins.
pub fn record_expert_region_with_observation<P: crate::Parameterized<WorkspaceTensor>>(
    source: WorkspaceExpertRegionView<'_>, bank: &P, input: &WorkspaceTensor,
    routes: &crate::GroupSelection<WorkspaceTensor>, context: &WorkspaceContext,
    observation: Option<&mut dyn FnMut(WorkspaceExpertObservationView<'_>) -> Result<WorkspaceExpertObservationSource,Error>>,
) -> Result<crate::TensorParallelGroupedOutput<WorkspaceTensor>, Error> {
    source.validate()?;
    context.charge_metadata(ExpertRegionInputShape::control_bytes::<WorkspaceTensor>()
        .ok_or(WorkspaceMetadataError::Overflow)?)?;
    let original_shape=input.shape();
    let geometry=ExpertRegionInputShape::inspect(original_shape,routes.group_indices().shape())?;
    let [input,ids,scores,coefficients]=geometry.normalize(input,routes.group_indices(),
        routes.selected_scores(),routes.coefficients(),context)?;
    let slots = bank.retained_value_slot_bound().ok_or(WorkspaceMetadataError::Unqualified)?;
    let mut values = context.metadata_vec(slots)?;
    let mut excess = false;
    if !bank.visit_retained_values(&mut |value| {
        if values.len() == slots { excess = true; } else { values.push(value.clone()); }
    }) || excess { return Err(WorkspaceMetadataError::Unqualified.into()); }
    let (width, output_width) = source.kernel.dimensions();

    let rows = input.shape().split_last().filter(|(last, _)| **last == width)
        .and_then(|(_, prefix)| prefix.iter().try_fold(1usize, |rows, value|
            rows.checked_mul(usize::try_from(*value).ok()?)))
        .ok_or(WorkspaceMetadataError::Unqualified)?;
    if rows != source.source_rows || ids.shape().last().copied()
        != i32::try_from(source.routes_per_row).ok()
        || !matches!(ids.layout().dtype, WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
        || ids.layout().as_view().elements().ok() != source.selected_rows().and_then(|n| u64::try_from(n).ok())
        || ids.shape() != scores.shape()
        || ids.shape() != coefficients.shape() {
        return Err(WorkspaceMetadataError::Unqualified.into());
    }
    let mut inputs = context.metadata_vec(values.len().checked_add(4).ok_or(WorkspaceMetadataError::Overflow)?)?;
    inputs.extend([&input, &ids, &scores, &coefficients]);
    inputs.extend(values.iter());
    let mut shape = context.metadata_vec(original_shape.len())?;
    shape.extend_from_slice(original_shape);
    *shape.last_mut().ok_or(WorkspaceMetadataError::Unqualified)? = output_width;
    let separate_bias = source.kernel.separate_bias(source.tensor_partitions);
    let mut outputs = context.metadata_vec(1 + usize::from(separate_bias))?;
    outputs.push(context.layout(&[geometry.rows,output_width], WorkspaceDtype::Float32)?);
    if separate_bias { outputs.push(context.layout(&[geometry.rows,output_width], WorkspaceDtype::Float32)?); }
    let mut region = source.retain(context)?;
    if let Some(observation)=observation {
        let selected=observation::inspect(&region,&inputs,context,observation)?;
        region.observation=Some(selected);
    }
    let region = context.box_metadata(region)?;
    let mut output = context.execute(WorkspaceOperationKind::ExpertRegion(region), &inputs, outputs)?;
    let bias = if separate_bias { Some(output.pop().ok_or(WorkspaceMetadataError::Unqualified)?.reshape(&shape,context)?) } else { None };
    let output=output.pop().ok_or(WorkspaceMetadataError::Unqualified)?.reshape(&shape,context)?;
    Ok(crate::TensorParallelGroupedOutput::new(output,bias))
}

/// An inactive replicated provider contributes only the exact ordered success
/// votes. No local numerical bank, selected IDs or row population is invented.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct WorkspaceExpertProviderWave {
    pub bank:u32,
    pub wave:Option<usize>,
    pub unit:usize,
    pub tensor_group:Option<eredu_core::CollectiveGroupId>,
    pub expert_group:eredu_core::CollectiveGroupId,
    pub wave_group:Option<eredu_core::CollectiveGroupId>,
}
impl WorkspaceExpertProviderWave {
    pub fn groups(self)->[Option<eredu_core::CollectiveGroupId>;3]{
        [self.tensor_group,Some(self.expert_group),self.wave_group]
    }
}
pub fn record_expert_provider_wave(source:WorkspaceExpertProviderWave,context:&WorkspaceContext)->Result<(),Error>{
    context.charge_metadata(size_of::<(WorkspaceExpertProviderWave,WorkspaceOperationKind,
        Vec<WorkspaceLayout>,Vec<WorkspaceTensor>,Result<(),Error>)>())?;
    context.execute(WorkspaceOperationKind::ExpertProviderWave(source),&[],Vec::new())?;
    Ok(())
}

/// A zero-row contribution to one retained dynamic pipeline expert wave.
/// The selected row ceiling describes active peers; it is not a local count.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct WorkspaceExpertInactiveWave {
    pub provider:WorkspaceExpertProviderWave,
    pub rank:usize,
    pub peers:usize,
    pub selected_rows:usize,
    pub input_width:i32,
    pub output_width:i32,
    pub dtype:WorkspaceFloatingType,
    pub transfers:WorkspaceExpertTransfers,
}
impl WorkspaceExpertInactiveWave {
    pub fn validate(self)->Result<(),WorkspaceMetadataError>{
        if self.provider.wave.is_none()||self.provider.wave_group.is_none()||self.peers==0||self.rank>=self.peers||self.input_width<=0||self.output_width<=0
            ||self.transfers.len()==0{return Err(WorkspaceMetadataError::Unqualified);}
        use WorkspaceExpertTransfer as T;
        let prefix=[T::ForwardIndex,T::ForwardIndex,T::ForwardIndex,T::ForwardInput,T::ForwardScores,T::ForwardCoefficients,T::ReverseOutput];
        if prefix.into_iter().enumerate().any(|(index,value)|self.transfers.get(index)!=Some(value))
            || !matches!((self.transfers.get(7),self.transfers.get(8),self.transfers.len()),
                (Some(T::ReverseIndex),None,8)|(Some(T::ReverseBias),Some(T::ReverseIndex),9)){
            return Err(WorkspaceMetadataError::Unqualified);
        }
        i32::try_from(self.selected_rows.checked_mul(self.peers).ok_or(WorkspaceMetadataError::Overflow)?)
            .map_err(|_|WorkspaceMetadataError::Overflow)?;
        Ok(())
    }
}

pub fn record_expert_inactive_wave(source:WorkspaceExpertInactiveWave,context:&WorkspaceContext)->Result<(),Error>{
    source.validate()?;
    context.charge_metadata(size_of::<(WorkspaceExpertInactiveWave,WorkspaceOperationKind,
        Vec<WorkspaceLayout>,Vec<WorkspaceTensor>,Result<(),Error>)>())?;
    let source=context.box_metadata(source)?;
    context.execute(WorkspaceOperationKind::ExpertInactiveWave(source),&[],Vec::new())?;Ok(())
}
