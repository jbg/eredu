//! Cold row ceilings and exact completed-count binding share the retained owner map.
use super::{ExpertRouteCountPlan, ExpertRoutePackingCause, ExpertRoutePackingGeometry};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// Finite semantic row population; none of these counts grants native bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRouteRegionPopulation {
    packing: ExpertRoutePackingGeometry,
    rank: usize,
    local_experts: usize,
    maximum_received: usize,
}
impl ExpertRouteRegionPopulation {
    pub const fn source_rows(self) -> usize { self.packing.source_tokens() }
    pub const fn selected_routes(self) -> usize { self.packing.routes() }
    pub const fn routes_per_row(self) -> usize { self.packing.routes_per_token() }
    pub const fn peers(self) -> usize { self.packing.peers() }
    pub const fn local_rank(self) -> usize { self.rank }
    pub const fn local_experts(self) -> usize { self.local_experts }
    /// Every member can send at most its full selected route population here.
    pub const fn maximum_received_rows(self) -> usize { self.maximum_received }
    /// Dispatch and return use this many rows at the original source rank.
    pub const fn maximum_returned_rows(self) -> usize { self.packing.routes() }
    pub const fn packing_payload_bytes(self) -> usize { self.packing.payload_bytes() }
}

/// Constructor-owned expert mapping plus request-derived row geometry.
#[derive(Clone, Copy, Debug)]
pub struct ExpertRouteRegionSource<'a> {
    owners: &'a [usize],
    local: &'a [usize],
    population: ExpertRouteRegionPopulation,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExpertRouteRegionCause {
    #[error(transparent)] Packing(#[from] ExpertRoutePackingCause),
    #[error("expert region differs from its retained owner/count geometry")] Geometry,
    #[error("expert region has no unspent completed count source")] Source,
    #[error("expert region row population overflowed")] Overflow,
    #[error("expert region metadata was refused: {0}")] Funding(#[source] WorkspaceMetadataFundingError),
}
pub(super) fn source<'a>(owners: &'a [usize], local: &'a [usize],
    packing: ExpertRoutePackingGeometry, rank: usize) -> Result<ExpertRouteRegionSource<'a>, ExpertRouteRegionCause> {
    if rank >= packing.peers() || owners.len() != local.len() || owners.is_empty()
        || owners.iter().any(|owner| *owner >= packing.peers()) { return Err(ExpertRouteRegionCause::Geometry); }
    let local_experts = owners.iter().filter(|owner| **owner == rank).count();
    let maximum_received = if local_experts == 0 { 0 } else {
        packing.routes().checked_mul(packing.peers()).ok_or(ExpertRouteRegionCause::Overflow)?
    };
    Ok(ExpertRouteRegionSource { owners, local, population: ExpertRouteRegionPopulation {
        packing, rank, local_experts, maximum_received,
    } })
}
impl ExpertRouteRegionSource<'_> {
    /// Borrows the same constructor mapping for the shared cold/native region
    /// callback. The descriptor preserves semantic ceilings, not an ID guess.
    pub fn workspace_view<'a>(self, bank: u32, unit: usize, prefill: bool,
        group: eredu_core::CollectiveGroupId,
        kernel: eredu_nn::workspace::WorkspaceExpertKernel<'a>, tensor_partitions: Option<usize>)
        -> Result<eredu_nn::workspace::WorkspaceExpertRegionView<'a>, ExpertRouteRegionCause>
    where Self: 'a {
        let movement = movement_population(self.population.source_rows(), self.population.routes_per_row(),
            kernel.reduction(), kernel.separate_bias(tensor_partitions)).ok_or(ExpertRouteRegionCause::Overflow)?;
        Ok(eredu_nn::workspace::WorkspaceExpertRegionView {
            bank, unit, prefill, group, rank: self.population.rank,
            peers: self.population.peers(), source_rows: self.population.source_rows(),
            routes_per_row: self.population.routes_per_row(), owners: self.owners,
            owner_local: self.local, kernel, tensor_partitions, movement,
            transfers: transfer_itinerary(kernel.separate_bias(tensor_partitions)),
            provider_tensor_group:None,provider_wave_group:None,
        })
    }
    pub const fn population(self) -> ExpertRouteRegionPopulation { self.population }
    pub fn global_experts(self) -> usize { self.owners.len() }
    /// A foreign expert contributes no local numerical rows. The bound allows
    /// repeated selected IDs, preserving interventions that duplicate routes.
    pub fn maximum_expert_rows(self, expert: usize) -> Option<usize> {
        self.owners.get(expert).map(|owner|
            if *owner == self.population.rank { self.population.maximum_received } else { 0 })
    }
    pub fn owner_local_expert(self, expert: usize) -> Option<(usize, usize)> {
        Some((*self.owners.get(expert)?, *self.local.get(expert)?))
    }
    pub fn binding_control_bytes() -> Option<usize> {
        let frames = [size_of::<Self>(), size_of::<ExpertRouteRegionRows>(),
            size_of::<ExpertRouteRegionPopulation>(), size_of::<ExpertRoutePackingGeometry>(),
            size_of::<Result<Self, ExpertRouteRegionCause>>(),
            size_of::<(&[usize], &[usize], ExpertRoutePackingGeometry, usize)>(),
            size_of::<ExpertRouteRegionCause>(), size_of::<FundedExpertRouteRegionFailure>(),
            size_of::<Result<ExpertRouteRegionRows, ExpertRouteRegionCause>>(),
            size_of::<Result<(), WorkspaceMetadataFundingError>>(), size_of::<[usize; 7]>(),
            size_of::<std::slice::ChunksExact<'_, usize>>(), size_of::<std::slice::Iter<'_, usize>>(),
            eredu_nn::Error::retained_source_control_bytes::<FundedExpertRouteRegionFailure>()?];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Binds exact source-major counts; no selected expert IDs are invented.
    pub fn bind_counts(self, counts: &ExpertRouteCountPlan) -> Result<ExpertRouteRegionRows, ExpertRouteRegionCause> {
        let p = self.population;
        if counts.local_rank() != p.rank || counts.group_size() != p.peers()
            || counts.count_matrix().len() != p.peers().checked_mul(p.peers()).ok_or(ExpertRouteRegionCause::Overflow)? {
            return Err(ExpertRouteRegionCause::Geometry);
        }
        for row in counts.count_matrix().chunks_exact(p.peers()) {
            let sent = row.iter().copied().try_fold(0usize, usize::checked_add).ok_or(ExpertRouteRegionCause::Overflow)?;
            if sent > p.selected_routes() { return Err(ExpertRouteRegionCause::Geometry); }
        }
        let sent = counts.forward().send().iter().copied().try_fold(0usize, usize::checked_add)
            .ok_or(ExpertRouteRegionCause::Overflow)?;
        let received = counts.forward().receive().iter().copied().try_fold(0usize, usize::checked_add)
            .ok_or(ExpertRouteRegionCause::Overflow)?;
        if sent != p.selected_routes() || received > p.maximum_received {
            return Err(ExpertRouteRegionCause::Geometry);
        }
        Ok(ExpertRouteRegionRows { population: p, received, returned: sent })
    }
}
/// Exact row cardinalities kept inside the original completed count owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRouteRegionRows {
    population: ExpertRouteRegionPopulation,
    received: usize,
    returned: usize,
}
impl ExpertRouteRegionRows {
    pub const fn population(self) -> ExpertRouteRegionPopulation { self.population }
    pub const fn received_rows(self) -> usize { self.received }
    pub const fn returned_rows(self) -> usize { self.returned }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct FundedExpertRouteRegionFailure {
    #[source] cause: ExpertRouteRegionCause,
    completed_source: Option<eredu_core::ErasedSharedStorageOwner>,
    funding: WorkspaceMetadataFunding,
}
impl FundedExpertRouteRegionFailure {
    pub(super) fn new(cause: ExpertRouteRegionCause, funding: WorkspaceMetadataFunding) -> Self {
        Self { cause, completed_source: None, funding }
    }
    pub(super) fn with_completed_source(mut self, source: Option<eredu_core::ErasedSharedStorageOwner>) -> Self {
        self.completed_source = source; self
    }
    pub fn funding(&self) -> &WorkspaceMetadataFunding { &self.funding }
}

/// Actual received routing values, kept with the completed issuing count
/// owner. The two finite destinations retire before source and funding.
#[derive(Debug)]
pub struct FundedExpertRegionSelection {
    local_rows: Vec<usize>,
    local_indices: Vec<i32>,
    rows: ExpertRouteRegionRows,
    completed_source: eredu_core::ErasedSharedStorageOwner,
    funding: WorkspaceMetadataFunding,
}
impl FundedExpertRegionSelection {
    pub fn local_expert_rows(&self) -> &[usize] { &self.local_rows }
    pub fn local_indices(&self) -> &[i32] { &self.local_indices }
    pub fn rows(&self) -> ExpertRouteRegionRows { self.rows }
    pub fn completed_source(&self) -> &eredu_core::ErasedSharedStorageOwner { &self.completed_source }
    pub fn funding(&self) -> &WorkspaceMetadataFunding { &self.funding }
}
pub(super) fn bind_local(source: ExpertRouteRegionSource<'_>, rows: ExpertRouteRegionRows,
    global: &[usize], local: &[usize], completed_source: eredu_core::ErasedSharedStorageOwner,
    funding: WorkspaceMetadataFunding) -> Result<FundedExpertRegionSelection, FundedExpertRouteRegionFailure> {
    let result = (|| {
        let frames = [size_of::<FundedExpertRegionSelection>(), size_of::<[Vec<usize>; 1]>(),
            size_of::<Vec<i32>>(), size_of::<ExpertRouteRegionRows>(),
            size_of::<eredu_core::ErasedSharedStorageOwner>(),
            size_of::<(&[usize], &[usize], ExpertRouteRegionSource<'_>)>(),
            size_of::<Result<FundedExpertRegionSelection, FundedExpertRouteRegionFailure>>(),
            size_of::<std::slice::Iter<'_, usize>>(),
            size_of::<std::iter::Zip<std::slice::Iter<'_, usize>, std::slice::Iter<'_, usize>>>(),
            size_of::<Result<(), WorkspaceMetadataFundingError>>()];
        let controls = frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
            .and_then(|bytes| bytes.checked_add(ExpertRouteRegionSource::binding_control_bytes()?))
            .ok_or(ExpertRouteRegionCause::Overflow)?;
        funding.reserve_metadata(controls).map_err(ExpertRouteRegionCause::Funding)?;
        if rows.population != source.population || global.len() != rows.received || local.len() != rows.received {
            return Err(ExpertRouteRegionCause::Geometry);
        }
        for (&global, &local) in global.iter().zip(local) {
            if source.owner_local_expert(global) != Some((rows.population.rank, local))
                || local >= rows.population.local_experts || i32::try_from(local).is_err() {
                return Err(ExpertRouteRegionCause::Geometry);
            }
        }
        let counts_bytes = rows.population.local_experts.checked_mul(size_of::<usize>())
            .filter(|bytes| *bytes <= isize::MAX as usize).ok_or(ExpertRouteRegionCause::Overflow)?;
        let ids_bytes = local.len().checked_mul(size_of::<i32>())
            .filter(|bytes| *bytes <= isize::MAX as usize).ok_or(ExpertRouteRegionCause::Overflow)?;
        funding.reserve_metadata(counts_bytes.checked_add(ids_bytes).ok_or(ExpertRouteRegionCause::Overflow)?)
            .map_err(ExpertRouteRegionCause::Funding)?;
        let mut local_rows = vec![0usize; rows.population.local_experts];
        let mut local_indices = Vec::with_capacity(local.len());
        for &local in local {
            local_rows[local] = local_rows[local].checked_add(1).ok_or(ExpertRouteRegionCause::Overflow)?;
            local_indices.push(local as i32);
        }
        Ok((local_rows, local_indices))
    })();
    match result {
        Ok((local_rows, local_indices)) => Ok(FundedExpertRegionSelection {
            local_rows, local_indices, rows, completed_source, funding,
        }),
        Err(cause) => Err(FundedExpertRouteRegionFailure::new(cause, funding)
            .with_completed_source(Some(completed_source))),
    }
}

/// Shared by the retained cold declaration and the actual route driver after
/// the provider declares its optional post-reduction output. Exact returned
/// route permutation gives every source token precisely routes_per_row entries.
pub(super) fn movement_population(source_rows: usize, routes_per_row: usize,
    reduction: eredu_nn::GroupReduction, post_reduce: bool)
    -> Option<eredu_nn::workspace::WorkspaceExpertMovementPopulation> {
    let outputs = 1usize.checked_add(usize::from(post_reduce))?;
    let sequential = reduction == eredu_nn::GroupReduction::SequentialGroupOrder;
    let waves = if source_rows == 0 { 0 } else { routes_per_row };
    Some(eredu_nn::workspace::WorkspaceExpertMovementPopulation {
        row_gathers: if sequential { outputs.checked_mul(waves.checked_add(1)?)?.checked_add(1)? } else { 1 },
        scalar_gathers: 2,
        zeros: outputs,
        indexed_adds: outputs.checked_mul(if sequential { waves } else { 1 })?,
    })
}

/// Mirrors the shared dispatch's named payload sequence, including the inverse
/// tag vote. Native realizations consume this declaration, never family names.
pub(super) fn transfer_itinerary(post_reduce:bool)->eredu_nn::workspace::WorkspaceExpertTransfers {
    use eredu_nn::workspace::WorkspaceExpertTransfer::*;
    eredu_nn::workspace::WorkspaceExpertTransfers([Some(ForwardIndex),Some(ForwardIndex),Some(ForwardIndex),
        Some(ForwardInput),Some(ForwardScores),Some(ForwardCoefficients),Some(ReverseOutput),
        post_reduce.then_some(ReverseBias),Some(ReverseIndex)])
}
