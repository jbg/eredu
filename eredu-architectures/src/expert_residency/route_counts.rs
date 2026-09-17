//! Count geometry from the selected opaque group and actual local route row.
use super::{CollectiveGroupId, CommunicationPeerCounts, ExpertRouteCountPlan};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::CommunicationGroupDescriptor;
use std::mem::{size_of, size_of_val};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ExpertRouteCountCause {
    #[error("expert count consensus differs from opaque group membership")]
    Group,
    #[error("peer count gather has no matching original source account")]
    Source,
    #[error("expert route count matrix geometry is invalid")]
    Geometry,
    #[error("expert route count consensus changed the local send row")]
    LocalRow,
    #[error("expert route count at position {position} is negative: {count}")]
    Negative { position: usize, count: i32 },
    #[error("expert route count destination geometry overflowed")]
    Overflow,
    #[error("expert route count exceeds i32 consensus geometry")]
    Encoding,
    #[error("expert route count metadata was refused: {0}")]
    Funding(#[source] WorkspaceMetadataFundingError),
}

/// Geometry only: this loan grants neither communication nor native readout.
#[derive(Clone, Copy)]
pub struct ExpertRouteCountSource<'a> {
    group: &'a CommunicationGroupDescriptor,
    local: &'a [usize],
    rank: usize,
}
impl<'a> ExpertRouteCountSource<'a> {
    pub fn new(group: &'a CommunicationGroupDescriptor, local: &'a [usize])
        -> Result<Self, ExpertRouteCountCause>
    {
        let rank = group.local_index().ok_or(ExpertRouteCountCause::Group)?;
        if local.is_empty() || group.members().len()!=local.len() || rank>=local.len() {
            return Err(ExpertRouteCountCause::Group);
        }
        Ok(Self { group, local, rank })
    }
    pub fn group(&self) -> CollectiveGroupId { self.group.id() }
    pub fn local_rank(&self) -> usize { self.rank }
    pub fn peers(&self) -> usize { self.local.len() }
    pub fn local_counts(&self) -> &[usize] { self.local }
    /// The one matrix plus four peer vectors retained by the shared count plan.
    pub fn result_payload_bytes(&self) -> Result<usize, ExpertRouteCountCause> {
        let peers=self.peers();
        peers.checked_mul(peers).and_then(|matrix|peers.checked_mul(4).and_then(|peer|matrix.checked_add(peer)))
            .and_then(|count|count.checked_mul(size_of::<usize>()))
            .filter(|bytes|*bytes<=isize::MAX as usize).ok_or(ExpertRouteCountCause::Overflow)
    }
    pub fn preparation_control_bytes() -> Option<usize> {
        let parts=[size_of::<eredu_runtime::PreparedPeerCountLoan<'_>>(),size_of::<Option<eredu_core::ErasedSharedStorageOwner>>(),size_of::<Self>(),size_of::<ExpertRouteCountPlan>(),
            size_of::<FundedExpertRouteCounts>(),size_of::<FundedExpertRouteCountFailure>(),
            size_of::<FundedCountRow>(),size_of::<Result<FundedCountRow,FundedExpertRouteCountFailure>>(),
            size_of::<ExpertRouteCountCause>(),size_of::<[Vec<usize>;5]>(),
            size_of::<WorkspaceMetadataFunding>(),size_of::<Result<(),WorkspaceMetadataFundingError>>(),
            size_of::<Result<FundedExpertRouteCounts,FundedExpertRouteCountFailure>>(),
            size_of::<(usize,usize,usize,usize)>(),size_of::<(&[usize],&[i32])>(),
            size_of::<(CollectiveGroupId,usize,Vec<usize>,Vec<usize>)>(),
            size_of::<[CommunicationPeerCounts;2]>(),
            size_of::<Result<CommunicationPeerCounts,eredu_runtime::CommunicationManifestError>>(),
            size_of::<Result<Vec<i32>,ExpertRouteCountCause>>(),
            size_of::<std::slice::Iter<'_,usize>>(),size_of::<std::slice::Iter<'_,i32>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_,i32>>>(),
            size_of::<std::iter::Zip<std::slice::Iter<'_,usize>,std::slice::Iter<'_,i32>>>(),
            size_of::<Result<ExpertRouteCountPlan,ExpertRouteCountCause>>(),
            eredu_nn::Error::retained_source_control_bytes::<FundedExpertRouteCountFailure>()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(super) fn prepare_row(self, funding: WorkspaceMetadataFunding)
        -> Result<FundedCountRow, FundedExpertRouteCountFailure>
    {
        let result=(||{
            let controls=Self::preparation_control_bytes().ok_or(ExpertRouteCountCause::Overflow)?;
            funding.reserve_metadata(controls).map_err(ExpertRouteCountCause::Funding)?;
            for &count in self.local {
                i32::try_from(count).map_err(|_|ExpertRouteCountCause::Encoding)?;
            }
            let bytes=self.peers().checked_mul(size_of::<i32>())
                .filter(|bytes|*bytes<=isize::MAX as usize).ok_or(ExpertRouteCountCause::Overflow)?;
            funding.reserve_metadata(bytes).map_err(ExpertRouteCountCause::Funding)?;
            let mut values=Vec::with_capacity(self.peers());
            values.extend(self.local.iter().map(|&count|count as i32));
            Ok(values)
        })();
        match result {
            Ok(values)=>Ok(FundedCountRow { values,_funding:funding }),
            Err(cause)=>Err(FundedExpertRouteCountFailure { cause,funding }),
        }
    }
    /// Borrows the actual completed matrix. No cast/evaluation or byte authority
    /// is accepted here; those remain responsibilities of the selected source.
    pub fn prepare_i32(self, matrix: &[i32], funding: WorkspaceMetadataFunding)
        -> Result<FundedExpertRouteCounts, FundedExpertRouteCountFailure>
    {
        let result=(||{
            let controls=Self::preparation_control_bytes().ok_or(ExpertRouteCountCause::Overflow)?;
            funding.reserve_metadata(controls).map_err(ExpertRouteCountCause::Funding)?;
            let peers=self.peers();
            if matrix.len()!=peers.checked_mul(peers).ok_or(ExpertRouteCountCause::Overflow)? {
                return Err(ExpertRouteCountCause::Geometry);
            }
            for (position,&count) in matrix.iter().enumerate() {
                if count<0 { return Err(ExpertRouteCountCause::Negative { position,count }); }
            }
            let local_start=self.rank*peers;
            if !self.local.iter().zip(&matrix[local_start..local_start+peers])
                .all(|(&local,&actual)|local==actual as usize) {
                return Err(ExpertRouteCountCause::LocalRow);
            }
            funding.reserve_metadata(self.result_payload_bytes()?).map_err(ExpertRouteCountCause::Funding)?;
            let mut counts=Vec::with_capacity(matrix.len());
            counts.extend(matrix.iter().map(|&count|count as usize));
            let mut local=Vec::with_capacity(peers);local.extend_from_slice(self.local);
            build(self.group(),self.rank,local,counts)
        })();
        match result {
            Ok(plan)=>Ok(FundedExpertRouteCounts { plan,completed_source:None,funding,region:None,
                local_binding_attempted: std::cell::Cell::new(false) }),
            Err(cause)=>Err(FundedExpertRouteCountFailure { cause,funding }),
        }
    }
}

pub(super) struct FundedCountRow { values: Vec<i32>, _funding: WorkspaceMetadataFunding }
impl FundedCountRow { pub(super) fn values(&self)->&[i32] { &self.values } }
pub(super) fn retain(cause:ExpertRouteCountCause,funding:&WorkspaceMetadataFunding)->FundedExpertRouteCountFailure {
    FundedExpertRouteCountFailure { cause,funding:funding.clone() }
}

/// The five concrete count destinations retire before their cumulative account.
#[derive(Debug)]
pub struct FundedExpertRouteCounts {
    plan: ExpertRouteCountPlan,
    completed_source: Option<eredu_core::ErasedSharedStorageOwner>,
    funding: WorkspaceMetadataFunding,
    region: Option<super::ExpertRouteRegionRows>,
    local_binding_attempted: std::cell::Cell<bool>,
}
impl FundedExpertRouteCounts {
    pub(crate) fn bind_local(&self, source: super::ExpertRouteRegionSource<'_>,
        global: &[usize], local: &[usize])
        -> Result<super::region::FundedExpertRegionSelection, super::region::FundedExpertRouteRegionFailure> {
        let failed = |cause| super::region::FundedExpertRouteRegionFailure::new(cause, self.funding.clone())
            .with_completed_source(self.completed_source.clone());
        // Spend before validation or reservation. Refusal cannot replay the
        // received row source under this completed count occurrence.
        let attempted = self.local_binding_attempted.replace(true);
        let controls = super::ExpertRouteRegionSource::binding_control_bytes()
            .ok_or_else(|| failed(super::ExpertRouteRegionCause::Overflow))?;
        self.funding.reserve_metadata(controls)
            .map_err(|cause| failed(super::ExpertRouteRegionCause::Funding(cause)))?;
        if attempted { return Err(failed(super::ExpertRouteRegionCause::Source)); }
        let rows = self.region.ok_or_else(|| failed(super::ExpertRouteRegionCause::Source))?;
        let actual = source.bind_counts(&self.plan).map_err(failed)?;
        if actual != rows { return Err(failed(super::ExpertRouteRegionCause::Geometry)); }
        let completed = self.completed_source.clone().ok_or_else(|| failed(super::ExpertRouteRegionCause::Source))?;
        super::region::bind_local(source, rows, global, local, completed, self.funding.clone())
    }
    pub(crate) fn bind_region(&mut self, source: super::ExpertRouteRegionSource<'_>)
        -> Result<(), super::region::FundedExpertRouteRegionFailure> {
        let build = || {
            let controls = super::ExpertRouteRegionSource::binding_control_bytes()
                .ok_or(super::ExpertRouteRegionCause::Overflow)?;
            self.funding.reserve_metadata(controls)
                .map_err(super::ExpertRouteRegionCause::Funding)?;
            if self.completed_source.is_none() || self.region.is_some() {
                return Err(super::ExpertRouteRegionCause::Source);
            }
            source.bind_counts(&self.plan)
        };
        match build() {
            Ok(region) => { self.region = Some(region); Ok(()) }
            Err(cause) => Err(super::region::FundedExpertRouteRegionFailure::new(cause, self.funding.clone())),
        }
    }
    pub(crate) fn region_rows(&self) -> Option<&super::ExpertRouteRegionRows> { self.region.as_ref() }

    pub(crate) fn with_completed_source(mut self, source: Option<eredu_core::ErasedSharedStorageOwner>) -> Self {
        self.completed_source = source; self
    }
    pub(crate) fn completed_source(&self) -> Option<&eredu_core::ErasedSharedStorageOwner> {
        self.completed_source.as_ref()
    }
    pub(crate) fn plan(&self) -> &ExpertRouteCountPlan { &self.plan }
    pub fn forward(&self) -> &CommunicationPeerCounts { self.plan.forward() }
    pub fn reverse(&self) -> &CommunicationPeerCounts { self.plan.reverse() }
    pub fn count_matrix(&self) -> &[usize] { self.plan.count_matrix() }
    pub fn funding(&self) -> &WorkspaceMetadataFunding { &self.funding }
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
pub struct FundedExpertRouteCountFailure {
    #[source]
    cause: ExpertRouteCountCause,
    funding: WorkspaceMetadataFunding,
}
impl FundedExpertRouteCountFailure {
    pub fn cause(&self) -> ExpertRouteCountCause { self.cause }
    pub fn funding(&self) -> &WorkspaceMetadataFunding { &self.funding }
}

pub(super) fn build(group:CollectiveGroupId, local_rank:usize,
    local_send_counts:Vec<usize>,count_matrix:Vec<usize>)
    -> Result<ExpertRouteCountPlan,ExpertRouteCountCause>
{
    let group_size=local_send_counts.len();
    if group_size==0 || local_rank>=group_size { return Err(ExpertRouteCountCause::Group); }
    let expected=group_size.checked_mul(group_size).ok_or(ExpertRouteCountCause::Overflow)?;
    if count_matrix.len()!=expected { return Err(ExpertRouteCountCause::Geometry); }
    let local_start=local_rank*group_size;
    if count_matrix[local_start..local_start+group_size]!=local_send_counts {
        return Err(ExpertRouteCountCause::LocalRow);
    }
    let mut total=0usize;
    for source in 0..group_size {
        total=total.checked_add(local_send_counts[source])
            .and_then(|sum|sum.checked_add(count_matrix[source*group_size+local_rank]))
            .ok_or(ExpertRouteCountCause::Overflow)?;
    }
    let mut receive=Vec::with_capacity(group_size);
    for source in 0..group_size { receive.push(count_matrix[source*group_size+local_rank]); }
    let mut forward_send=Vec::with_capacity(group_size);forward_send.extend_from_slice(&local_send_counts);
    let mut forward_receive=Vec::with_capacity(group_size);forward_receive.extend_from_slice(&receive);
    let forward=CommunicationPeerCounts::new(forward_send,forward_receive,group_size)
        .map_err(|_|ExpertRouteCountCause::Geometry)?;
    let reverse=CommunicationPeerCounts::new(receive,local_send_counts,group_size)
        .map_err(|_|ExpertRouteCountCause::Geometry)?;
    Ok(ExpertRouteCountPlan { group,group_size,local_rank,count_matrix,forward,reverse })
}
