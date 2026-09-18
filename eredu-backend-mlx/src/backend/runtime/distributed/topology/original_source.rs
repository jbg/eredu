//! Funded finite loans from the actual retained native communication source.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::{PartitionCommunicationAuthority, RetainedCommunicationSource};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(in crate::backend::runtime::distributed) enum Cause {
    #[error("prepared boundary root destination failed")]
    BoundaryRoots(#[source] safemlx::PreparedNestedRootsCause),
    #[error("synchronized sampling differs from the selected request source or sequence")]
    SamplingIdentity,
    #[error("completed synchronized sampling word is invalid")]
    SamplingWord,
    #[error(transparent)]
    SamplingPlan(eredu_runtime::generation::SamplingSynchronizationError),
    #[error(transparent)]
    SamplingCommunication(eredu_runtime::PartitionExecutionError),
    #[error(transparent)]
    SamplingData(safemlx::error::AsSliceError),
    #[error(transparent)]
    BoundaryCommunication(eredu_runtime::PartitionExecutionError),
    #[error(transparent)]
    BoundaryFrame(eredu_runtime::PreparedBoundaryFrameError),
    #[error("native canonical boundary header copy failed: {cause}")]
    BoundaryHeader { cause:safemlx::OwnedHostCopyCause, #[source] native:Option<safemlx::error::Exception> },
    #[error("prepaid native boundary header destination allocation failed")]
    BoundaryDestination(#[source] std::collections::TryReserveError),
    #[error(transparent)]
    Control(control::ControlCause),
    #[error(transparent)]
    Invocation(#[from] parallel::ParallelInvocationCause),
    #[error("distributed source differs from its selected manifest, setup or native world")]
    Identity,
    #[error(transparent)]
    LogicalExchange(#[from] crate::backend::runtime::distributed::group::LogicalExchangeCause),
    #[error("selected route requires its packed-world transport producer")]
    LogicalWorldTransport,
    #[error("selected route exchange round differs from its retained endpoint plan")]
    LogicalRound,
    #[error("distributed source owner is terminal or quarantined")]
    Unavailable,
    #[error("distributed source does not contain the exact selected native resource")]
    Resource,
    #[error("actual collective binding failed at occurrence {index:?}, equation {ordinal:?}: {cause}")]
    NativeBinding {
        index: Option<usize>,
        ordinal: Option<usize>,
        #[source]
        cause: safemlx::distributed::GroupCpuBindingError,
    },
    #[error("member-only status chain exceeded its original deadline")]
    StatusDeadline,
    #[error("member-only status numerical source failed")]
    StatusPlanning(#[source] eredu_nn::Error),
    #[error("member-only status source alias failed")]
    StatusClone(#[source] safemlx::PreparedArrayCloneCause),
    #[error("distributed source lacks its exact selected native resource at {0}")]
    ResourceAt(&'static std::panic::Location<'static>),
    #[error("parallel workspace {operation:?} on {group:?}, local rank {rank}/{partitions}, lacks floating scalar evidence: input rank {input_rank}, width {input_width:?}, elements {input_elements:?}")]
    WorkspaceRepresentation {
        operation: CommunicationOperation,
        group: CollectiveGroupId,
        rank: usize,
        partitions: usize,
        input_rank: usize,
        input_width: Option<i32>,
        input_elements: Option<u64>,
    },
    #[error("actual native group refused the selected CPU layout source")]
    NativeLayout,
    #[error("native parallel backing population is not the selected Sum/Gather producer: {0}")]
    BackingPopulation(usize),
    #[error(transparent)]
    Gather(#[from] eredu_nn::ParallelGatherError),
    #[error(transparent)]
    Vocabulary(eredu_nn::VocabularyRangeError),
    #[error(transparent)]
    Embedding(eredu_nn::EmbeddingValidationError),
    #[error("native distributed graph construction failed")]
    Native(#[source] safemlx::error::Exception),
    #[error("selected rank operation is unavailable")]
    Rank(#[source] eredu_runtime::CommunicationGroupOperationError),
    #[error("actual rank tensor differs from its selected contract")]
    Tensor(#[source] eredu_runtime::CommunicationTensorContractError),
    #[error("actual rank collective output differs from the selected equation")]
    Output,
    #[error("prepared parallel model context was rejected")]
    Context(#[source] eredu_runtime::replicated_session::PreparedParallelContextCause),
    #[error("distributed status source input failed")]
    Input(#[source] safemlx::PreparedInputCause),
    #[error("distributed status source arena failed")]
    SourceGraph(#[source] safemlx::SubmissionGraphQuotaCause),
    #[error("distributed source allocator domain is unavailable")]
    Allocator(#[source] eredu_runtime::working_memory::WorkingMemoryError),
    #[error("native distributed backing source is unavailable")]
    Buffer(#[source] safemlx::OriginalBufferCause),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    _source: RetainedCommunicationSource,
    funding: HostMetadataFunding,
}
fn failure_control_bytes() -> Option<usize> {
    let controls = [size_of::<Failure>(), size_of::<Cause>(), size_of::<Error>(),
        size_of::<eredu_core::BackendFailure>(), size_of::<Result<(), Error>>(),
        size_of::<(&RetainedCommunicationSource, &HostMetadataFunding, Cause)>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?];
    controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
}
#[track_caller]
fn failure(cause: Cause, source: &RetainedCommunicationSource, funding: &HostMetadataFunding) -> Error {
    let cause = match cause { Cause::Resource => Cause::ResourceAt(std::panic::Location::caller()), other => other };
    Error::with_original_control_source(eredu_core::BackendFailure::new(
        eredu_core::BackendFailureKind::Other,
        Failure { cause, _source: source.clone(), funding: funding.clone() }), false)
}

/// Borrowed actual resources plus their paid source/transport controls. No
/// caller can construct this from a manifest or raw native handle alone.
pub(crate) struct OriginalCommunicationSource<'a> {
    actual: &'a ParallelCommunicators,
    authority: &'a PartitionCommunicationAuthority,
    source: RetainedCommunicationSource,
    registered_buffers: Option<registered_buffers::RegisteredBuffers>,
    // Retires after source and every loan/control field.
    funding: HostMetadataFunding,
}
impl ParallelCommunicators {
    pub(crate) fn bind_original_source<'a>(
        &'a self, selected: &CommunicationManifest, world: &NativeGroup,
        authority: &'a PartitionCommunicationAuthority, funding: &HostMetadataFunding,
    ) -> Result<OriginalCommunicationSource<'a>, Error> {
        self.bind_original_source_retaining(selected,world,authority,funding,&self.source)
    }
    fn bind_original_source_retaining<'a>(
        &'a self, selected:&CommunicationManifest, world:&NativeGroup,
        authority:&'a PartitionCommunicationAuthority, funding:&HostMetadataFunding,
        retained:&RetainedCommunicationSource,
    )->Result<OriginalCommunicationSource<'a>,Error> {
        let controls = [size_of::<OriginalCommunicationSource<'a>>(),
            size_of::<Result<OriginalCommunicationSource<'a>, Error>>(),
            size_of::<(&Self, &CommunicationManifest, &NativeGroup, &PartitionCommunicationAuthority, &HostMetadataFunding)>(),
            size_of::<(&Self,&CommunicationManifest,&NativeGroup,&PartitionCommunicationAuthority,
                &HostMetadataFunding,&RetainedCommunicationSource)>(),
            size_of::<Cause>(), size_of::<Failure>(), size_of::<eredu_core::BackendFailure>(),
            size_of::<Result<(), eredu_runtime::PartitionExecutionError>>(),
            crate::backend::runtime::distributed::completion::group_source_controls()
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            size_of::<(usize, &CommunicationGroupDescriptor, bool)>(),
            size_of::<(usize, &CommunicationRouteDescriptor, bool)>(),
            size_of::<Option<(&Group, &CommunicationGroupDescriptor, bool)>>(),
            size_of::<Option<(&CommunicationRouteRealization, &CommunicationRouteDescriptor, bool)>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
        ];
        funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let fail = |cause| failure(cause, retained, funding);
        if !retained.same_source(&self.source) || self.source.manifest() != selected || self.session_identity != self.source.session_identity()
            || self.world_size != selected.world_size() || self.global_rank != selected.rank()
            || !self.control_world.native_group().shares_native_handle(world)
            || !self.control_world.retained_source().is_some_and(|source| source.same_source(&self.source))
            || self.groups.len() != selected.groups().len() || self.routes.len() != selected.routes().len() {
            return Err(fail(Cause::Identity));
        }
        if authority.ensure_active().is_err() || self.control_world.native_group().terminal_submission()
            || !crate::backend::runtime::distributed::completion::group_source_available(&self.control_world) {
            return Err(fail(Cause::Unavailable));
        }
        for (order, descriptor) in selected.groups().iter().enumerate() {
            let (_, wave) = self.source.group(order).ok_or_else(|| fail(Cause::Resource))?;
            let actual = self.groups.get(&descriptor.id()).ok_or_else(|| fail(Cause::Resource))?;
            let group = actual.native.as_ref().ok_or_else(|| fail(Cause::Resource))?;
            if &actual.descriptor != descriptor || !group.shares_native_world(&self.control_world)
                || !group.matches_retained_group(&self.source, descriptor, wave) {
                return Err(fail(Cause::Resource));
            }
        }
        for (order, descriptor) in selected.routes().iter().enumerate() {
            let (_, wave) = self.source.route(order).ok_or_else(|| fail(Cause::Resource))?;
            let route = self.routes.get(&descriptor.id()).ok_or_else(|| fail(Cause::Resource))?;
            let endpoint = if descriptor.source() == selected.rank() { Some(CommunicationRouteEndpoint::Source) }
                else if descriptor.destination() == selected.rank() { Some(CommunicationRouteEndpoint::Destination) } else { None };
            if &route.descriptor != descriptor || route.endpoint != endpoint
                || !route.source.as_ref().is_some_and(|source| source.same_source(&self.source))
                || match (&route.group, endpoint) {
                    (None, None) => route.peer_rank.is_some(),
                    (Some(group), Some(endpoint)) => !group.shares_native_world(&self.control_world)
                        || !group.retained_source().is_some_and(|source| source.same_source(&self.source))
                        || !group.matches_retained_route(&self.source, descriptor, wave)
                        || route.peer_rank != Some(if endpoint == CommunicationRouteEndpoint::Source { 1 } else { 0 }),
                    _ => true,
                } {
                return Err(fail(Cause::Resource));
            }
        }
        Ok(OriginalCommunicationSource { actual: self, authority, source: retained.clone(), registered_buffers: None, funding: funding.clone() })
    }
}
impl<'a> OriginalCommunicationSource<'a> {
    /// Revalidate the same borrowed authority immediately before a later
    /// qualified consumer; this does not reap or certify pending native work.
    pub(crate) fn validate(&self) -> Result<(), Error> {
        // Every attempt can return an independently escaped retained failure.
        // The initial source-binding allowance cannot be reused by later calls.
        self.funding.reserve_metadata(failure_control_bytes()
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        if self.authority.ensure_active().is_err() || self.actual.control_world.native_group().terminal_submission()
            || !crate::backend::runtime::distributed::completion::group_source_available(&self.actual.control_world) {
            return Err(failure(Cause::Unavailable, &self.source, &self.funding));
        }
        Ok(())
    }

    pub(crate) fn matches_world(&self, value: &Group) -> bool {
        value.matches_retained_world(&self.source)
            && value.native_group().shares_native_handle(self.world().native_group())
    }
    pub(crate) fn matches_group(&self, order: usize, value: &Group) -> bool {
        self.group(order).is_some_and(|(actual, descriptor, wave)|
            value.native_group().shares_native_handle(actual.native_group())
                && value.matches_retained_group(&self.source, descriptor, wave))
    }
    pub(crate) fn matches_route(&self, order: usize, value: &CommunicationRouteRealization) -> bool {
        self.route(order).is_some_and(|(actual, descriptor, wave)| {
            &value.descriptor == descriptor && value.endpoint == actual.endpoint
                && value.peer_rank == actual.peer_rank
                && value.source.as_ref().is_some_and(|source| source.same_source(&self.source))
                && match (&value.group, &actual.group) {
                    (None, None) => true,
                    (Some(value), Some(actual)) => value.native_group().shares_native_handle(actual.native_group())
                        && value.matches_retained_route(&self.source, descriptor, wave),
                    _ => false,
                }
        })
    }

    pub(crate) fn source(&self) -> &RetainedCommunicationSource { &self.source }
    pub(crate) fn funding(&self) -> &HostMetadataFunding { &self.funding }
    pub(crate) fn world(&self) -> &'a Group { &self.actual.control_world }
    pub(crate) fn group(&self, order: usize) -> Option<(&'a Group, &CommunicationGroupDescriptor, bool)> {
        let (descriptor, wave) = self.source.group(order)?;
        Some((self.actual.groups.get(&descriptor.id())?.native.as_ref()?, descriptor, wave))
    }
    pub(crate) fn route(&self, order: usize) -> Option<(&'a CommunicationRouteRealization, &CommunicationRouteDescriptor, bool)> {
        let (descriptor, wave) = self.source.route(order)?;
        Some((self.actual.routes.get(&descriptor.id())?, descriptor, wave))
    }
}

#[cfg(test)]
mod tests;

mod inventory;
pub(crate) use inventory::OriginalCommunicatorInventory;

mod worker_storage;
pub(crate) use worker_storage::OriginalCommunicationWorkers;

mod dispatch_storage;
pub(crate) use dispatch_storage::OriginalCommunicationDispatch;

mod constructor_storage;
pub(crate) use constructor_storage::{OriginalCommunicationConstructor,OriginalCommunicationConstructed,OriginalCommunicationEvaluation};

mod persistent;
pub(crate) use persistent::OriginalCommunicatorPersistent;

mod operation_storage;
pub(crate) use operation_storage::{OriginalCommunicationOperation,OriginalCommunicationCompletedOperation,AcceptedCommunicationSource};

mod rank_collective;
pub(crate) use rank_collective::{OriginalRankCollective, PreparedRankCollective, AcceptedRankCollective};

mod workspace_sum;
pub(crate) use workspace_sum::OriginalWorkspaceSum;

pub(crate) mod parallel;

pub(crate) mod agreement;

mod owner;
pub(crate) use owner::OriginalCommunicationOwner;

pub(crate) mod control;

mod registered_buffers;

mod inputs;
mod preparation;
pub(crate) use preparation::{OriginalPreparationFrame,PreparedOriginalPreparationGather,PreparationCompletion};

mod route_exchange;
pub(crate) use route_exchange::{OriginalGroupExchange,OwnedOriginalExchangeLayoutRound};
pub(crate) use route_exchange::{OwnedOriginalRouteLayoutRound,OriginalRouteExchange,OriginalRouteRound,PreparedRouteRound,
    AcceptedRouteSource,AcceptedRouteExchange,ConstructedRouteExchange};

mod boundary_headers;
pub(crate) use boundary_headers::{OriginalBoundaryHeaders, OriginalBoundaryHeader};

mod packed_world;
pub(crate) use packed_world::OwnedPackedWorldCompletion;
