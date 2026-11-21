//! Opaque communication manifests projected from semantic parallel topology.
//!
//! This module is the ownership seam between neutral topology planning and a
//! concrete communication implementation. Projection may inspect semantic
//! axes, but the descriptors handed to a backend contain only ordered world
//! ranks, opaque identities, operations, and mechanical limits.

use eredu_core::communication::{
    validate_compatible_communication_manifests, CommunicationBindingError,
    CommunicationBindingGeometryError, CommunicationBindings, CommunicationCapabilities,
    CommunicationCapabilityError, CommunicationCompletionPolicy, CommunicationGroupDescriptor,
    CommunicationGroupRequirements, CommunicationManifest, CommunicationManifestError,
    CommunicationOperation, CommunicationOperationRequirement, CommunicationRouteDescriptor,
    CommunicationRouteId, CommunicationSetupTransport, CommunicationTopologyCapabilities,
    WorldCommunicationBinding,
};
#[cfg(test)]
use eredu_core::communication::{
    BoundaryDimensionContract, BoundaryFramingProtocol, BoundaryRoleContract,
    CommunicationCompletionCapabilities, CommunicationTensorLimits, RoleExactBoundaryContract,
};
use std::{collections::BTreeSet, ops::Range};

#[cfg(test)]
use eredu_core::{checkpoint::TensorDtype, CollectiveGroupDescriptor, CompletionCancellationMode};
use eredu_core::{
    consensus::ConsensusTransport, CollectiveGroupId, ParallelAxis, ParallelRankTopology,
    ParallelTopology,
};
use serde::{Deserialize, Serialize};

mod session_identity;
pub use session_identity::{establish_communication_session, AgreedCommunicationSession};

/// Failure projecting semantic parallel axes into opaque descriptors.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CommunicationProjectionError {
    /// Invalid opaque descriptor or incompatible rank manifests.
    #[error(transparent)]
    Manifest(#[from] CommunicationManifestError),
    /// Rank topology and complete topology disagree.
    #[error("rank topology does not match the topology being projected")]
    TopologyMismatch,
    /// A validated core topology could not be projected consistently.
    #[error("parallel topology could not be projected into communication descriptors")]
    InvalidTopologyProjection,
    /// The projected descriptor count cannot be represented by the stable ID.
    #[error("communication descriptor count exceeds its stable ID representation")]
    DescriptorCountOverflow,
}

/// Semantic inputs used only while projecting opaque communication descriptors.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct TopologyCommunicationPlan {
    session_group: Option<CommunicationGroupRequirements>,
    tensor_groups: Option<CommunicationGroupRequirements>,
    pipeline_groups: Option<CommunicationGroupRequirements>,
    expert_groups: Option<CommunicationGroupRequirements>,
    data_groups: Option<CommunicationGroupRequirements>,
    pipeline_stage_groups: Option<CommunicationGroupRequirements>,
    pipeline_routes: Option<CommunicationOperationRequirement>,
    completion: Option<CommunicationCompletionPolicy>,
}

impl TopologyCommunicationPlan {
    /// Creates an empty projection plan.
    pub const fn new() -> Self {
        Self {
            session_group: None,
            tensor_groups: None,
            pipeline_groups: None,
            expert_groups: None,
            data_groups: None,
            pipeline_stage_groups: None,
            pipeline_routes: None,
            completion: None,
        }
    }

    /// Selects one bounded completion policy for every operation in the manifest.
    pub fn with_completion_policy(mut self, policy: CommunicationCompletionPolicy) -> Self {
        self.completion = Some(policy);
        self
    }

    /// Requests one group containing every participant in world-rank order.
    ///
    /// Session groups are projected before axis groups, so their opaque ID is
    /// stable across the set of active Cartesian axes.
    pub fn with_session_group(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.session_group = Some(requirements);
        self
    }

    /// Exact opaque ID assigned to the requested session group.
    pub const fn session_group_id(&self) -> Option<CollectiveGroupId> {
        if self.session_group.is_some() {
            Some(CollectiveGroupId::new(1))
        } else {
            None
        }
    }

    /// Requests groups varying the tensor coordinate.
    pub fn with_tensor_groups(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.tensor_groups = Some(requirements);
        self
    }

    /// Exact opaque tensor-group ID assigned to one rank by this projection plan.
    pub fn tensor_group_id(
        &self,
        topology: ParallelTopology,
        rank: ParallelRankTopology,
    ) -> Result<Option<CollectiveGroupId>, CommunicationProjectionError> {
        if self.tensor_groups.is_none() {
            return Ok(None);
        }
        if rank.topology() != topology {
            return Err(CommunicationProjectionError::TopologyMismatch);
        }
        let groups = unique_axis_groups(topology, ParallelAxis::Tensor)?;
        let subgroup = groups
            .iter()
            .position(|members| members.contains(&rank.global_rank()))
            .ok_or(CommunicationProjectionError::InvalidTopologyProjection)?;
        let first = 1usize + usize::from(self.session_group.is_some());
        let numeric = first
            .checked_add(subgroup)
            .ok_or(CommunicationProjectionError::DescriptorCountOverflow)?;
        Ok(Some(CollectiveGroupId::new(
            u32::try_from(numeric)
                .map_err(|_| CommunicationProjectionError::DescriptorCountOverflow)?,
        )))
    }

    /// Requests groups varying the pipeline coordinate.
    pub fn with_pipeline_groups(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.pipeline_groups = Some(requirements);
        self
    }

    /// Requests groups varying the expert coordinate.
    pub fn with_expert_groups(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.expert_groups = Some(requirements);
        self
    }

    /// Represents future groups varying the data coordinate.
    ///
    /// This is a topology projection facility, not product admission of data
    /// parallel execution. Product selection must continue to reject it until
    /// its semantics are implemented.
    pub fn with_data_groups(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.data_groups = Some(requirements);
        self
    }

    /// Requests all tensor and expert ranks at each fixed pipeline/data coordinate.
    /// These groups follow the axis groups, preserving existing opaque axis IDs.
    pub fn with_pipeline_stage_groups(
        mut self,
        requirements: CommunicationGroupRequirements,
    ) -> Self {
        self.pipeline_stage_groups = Some(requirements);
        self
    }

    /// Requests directed routes between adjacent pipeline coordinates.
    pub fn with_pipeline_routes(
        mut self,
        requirement: CommunicationOperationRequirement,
    ) -> Result<Self, CommunicationManifestError> {
        if requirement.operation() != CommunicationOperation::SendReceive {
            return Err(CommunicationManifestError::InvalidRouteOperation {
                id: CommunicationRouteId::new(0),
            });
        }
        self.pipeline_routes = Some(requirement);
        Ok(self)
    }
}

/// Projects one rank's semantic Cartesian topology into opaque descriptors.
pub fn project_communication_manifest(
    topology: ParallelTopology,
    rank: ParallelRankTopology,
    plan: &TopologyCommunicationPlan,
) -> Result<CommunicationManifest, CommunicationProjectionError> {
    if rank.topology() != topology {
        return Err(CommunicationProjectionError::TopologyMismatch);
    }

    let mut groups = Vec::new();
    let mut next_group_id = 1usize;
    if let Some(requirements) = &plan.session_group {
        let id = CollectiveGroupId::new(1);
        groups.push(CommunicationGroupDescriptor::new(
            id,
            0,
            (0..topology.world_size()).collect(),
            Some(rank.global_rank()),
            requirements.clone(),
        )?);
        next_group_id = 2;
    }
    for (axis, requirements) in [
        (ParallelAxis::Tensor, plan.tensor_groups.as_ref()),
        (ParallelAxis::Pipeline, plan.pipeline_groups.as_ref()),
        (ParallelAxis::Expert, plan.expert_groups.as_ref()),
        (ParallelAxis::Data, plan.data_groups.as_ref()),
    ] {
        let Some(requirements) = requirements else {
            continue;
        };
        let axis_groups = unique_axis_groups(topology, axis)?;
        let (subgroup_index, members) = axis_groups
            .iter()
            .enumerate()
            .find(|(_, members)| members.contains(&rank.global_rank()))
            .ok_or(CommunicationProjectionError::InvalidTopologyProjection)?;
        let numeric_id = next_group_id
            .checked_add(subgroup_index)
            .ok_or(CommunicationProjectionError::DescriptorCountOverflow)?;
        let id = CollectiveGroupId::new(
            u32::try_from(numeric_id)
                .map_err(|_| CommunicationProjectionError::DescriptorCountOverflow)?,
        );
        let local_index = members
            .iter()
            .position(|member| *member == rank.global_rank());
        groups.push(CommunicationGroupDescriptor::new(
            id,
            groups.len(),
            members.clone(),
            local_index,
            requirements.clone(),
        )?);
        next_group_id = next_group_id
            .checked_add(axis_groups.len())
            .ok_or(CommunicationProjectionError::DescriptorCountOverflow)?;
    }

    if let Some(requirements) = &plan.pipeline_stage_groups {
        let width = topology
            .tensor()
            .checked_mul(topology.expert())
            .ok_or(CommunicationProjectionError::DescriptorCountOverflow)?;
        let stage = rank.global_rank() / width;
        let start = stage * width;
        let numeric = next_group_id
            .checked_add(stage)
            .ok_or(CommunicationProjectionError::DescriptorCountOverflow)?;
        groups.push(CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(
                u32::try_from(numeric)
                    .map_err(|_| CommunicationProjectionError::DescriptorCountOverflow)?,
            ),
            groups.len(),
            (start..start + width).collect(),
            Some(rank.global_rank() - start),
            requirements.clone(),
        )?);
    }

    let mut routes = Vec::new();
    if let Some(requirement) = &plan.pipeline_routes {
        for source in 0..topology.world_size() {
            let coordinates = topology
                .coordinates(source)
                .map_err(|_| CommunicationProjectionError::InvalidTopologyProjection)?;
            if coordinates.pipeline() + 1 == topology.pipeline() {
                continue;
            }
            let destination = topology
                .rank_for(coordinates.with_pipeline(coordinates.pipeline() + 1))
                .map_err(|_| CommunicationProjectionError::InvalidTopologyProjection)?;
            let id = CommunicationRouteId::new(routes.len() as u64);
            routes.push(CommunicationRouteDescriptor::new(
                id,
                routes.len(),
                source,
                destination,
                requirement.clone(),
            )?);
        }
    }

    let manifest =
        CommunicationManifest::new(topology.world_size(), rank.global_rank(), groups, routes)?;
    Ok(match plan.completion {
        Some(policy) => manifest.with_completion_policy(policy),
        None => manifest,
    })
}

/// Projects and cross-validates compatible opaque descriptors for every rank.
pub fn project_all_communication_manifests(
    topology: ParallelTopology,
    plan: &TopologyCommunicationPlan,
) -> Result<Vec<CommunicationManifest>, CommunicationProjectionError> {
    let manifests = (0..topology.world_size())
        .map(|global_rank| {
            let rank = ParallelRankTopology::new(topology, global_rank)
                .map_err(|_| CommunicationProjectionError::InvalidTopologyProjection)?;
            project_communication_manifest(topology, rank, plan)
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_compatible_communication_manifests(&manifests)?;
    Ok(manifests)
}

/// Gathers and cross-validates one complete opaque manifest from every world rank.
///
/// The transport only needs an equal-word all-gather. A fixed two-word length
/// gather determines the largest encoded manifest, then every rank pads its
/// complete payload to that size for a second equal-word gather. Validation is
/// deliberately deferred until both collectives complete, so a rank-local
/// world, rank, descriptor, or limit mismatch cannot strand its peers before
/// the shared compatibility result is available.
pub fn validate_communication_manifest_consensus<T: ConsensusTransport>(
    transport: &T,
    local: &CommunicationManifest,
) -> Result<Vec<CommunicationManifest>, CommunicationManifestConsensusError<T::Error>> {
    let encoded = serde_json::to_vec(local)
        .map_err(|error| CommunicationManifestConsensusError::Encoding(error.to_string()))?;
    let manifests = gather_communication_payloads(transport, &encoded, false)?
        .into_iter()
        .enumerate()
        .map(|(rank, bytes)| {
            serde_json::from_slice(&bytes).map_err(|error| {
                CommunicationManifestConsensusError::InvalidEncoding {
                    rank,
                    message: error.to_string(),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_compatible_communication_manifests(&manifests)?;
    Ok(manifests)
}

fn gather_communication_payloads<T: ConsensusTransport>(
    transport: &T,
    encoded: &[u8],
    bounded: bool,
) -> Result<Vec<Vec<u8>>, CommunicationManifestConsensusError<T::Error>> {
    let participants = transport.participant_count();
    if participants == 0 {
        return Err(CommunicationManifestConsensusError::EmptyTopology);
    }

    let encoded_len = u64::try_from(encoded.len()).map_err(|_| {
        CommunicationManifestConsensusError::MetadataOverflow("encoded manifest length")
    })?;
    let length_words = [encoded_len as u32, (encoded_len >> 32) as u32];
    let gathered_lengths =
        gather_manifest_words(transport, &length_words, participants, "manifest lengths")?;

    let lengths = gathered_lengths
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
        .map(|(rank, words)| {
            let length = u64::from(words[0]) | (u64::from(words[1]) << 32);
            usize::try_from(length).map_err(|_| {
                CommunicationManifestConsensusError::PayloadLengthOverflow { rank, length }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if bounded {
        const PER_RANK: usize = 16 * 1024 * 1024;
        const TOTAL: usize = 128 * 1024 * 1024;
        let largest = lengths.iter().copied().max().unwrap_or(0);
        if largest > PER_RANK
            || words_for_bytes(largest)
                .checked_mul(4)
                .and_then(|padded| padded.checked_mul(participants))
                .is_none_or(|n| n > TOTAL)
        {
            return Err(CommunicationManifestConsensusError::MetadataLimit {
                per_rank: PER_RANK,
                total: TOTAL,
            });
        }
    }
    let payload_words = lengths
        .iter()
        .copied()
        .map(words_for_bytes)
        .max()
        .unwrap_or(0);
    let local_words = encode_manifest_words(encoded, payload_words);
    let gathered_payloads =
        gather_manifest_words(transport, &local_words, participants, "manifest payloads")?;

    let mut payloads = Vec::with_capacity(participants);
    for (rank, &length) in lengths.iter().enumerate() {
        let available = payload_words.checked_mul(4).ok_or(
            CommunicationManifestConsensusError::MetadataOverflow("manifest payload bytes"),
        )?;
        if length > available {
            return Err(CommunicationManifestConsensusError::InvalidPayloadLength {
                rank,
                length,
                available,
            });
        }
        let start = rank.checked_mul(payload_words).ok_or(
            CommunicationManifestConsensusError::MetadataOverflow("manifest payload offset"),
        )?;
        let end = start.checked_add(payload_words).ok_or(
            CommunicationManifestConsensusError::MetadataOverflow("manifest payload end"),
        )?;
        let mut bytes = Vec::with_capacity(available);
        for word in &gathered_payloads[start..end] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.truncate(length);
        payloads.push(bytes);
    }

    Ok(payloads)
}

fn words_for_bytes(bytes: usize) -> usize {
    bytes / 4 + usize::from(!bytes.is_multiple_of(4))
}

fn encode_manifest_words(encoded: &[u8], padded_words: usize) -> Vec<u32> {
    let mut words = Vec::with_capacity(padded_words);
    for chunk in encoded.chunks(4) {
        let mut word = [0; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(word));
    }
    words.resize(padded_words, 0);
    words
}

fn gather_manifest_words<T: ConsensusTransport>(
    transport: &T,
    local: &[u32],
    participants: usize,
    stage: &'static str,
) -> Result<Vec<u32>, CommunicationManifestConsensusError<T::Error>> {
    let expected = local.len().checked_mul(participants).ok_or(
        CommunicationManifestConsensusError::MetadataOverflow("gathered manifest word count"),
    )?;
    let gathered = transport
        .all_gather_words(local)
        .map_err(CommunicationManifestConsensusError::Transport)?;
    if gathered.len() != expected {
        return Err(CommunicationManifestConsensusError::MalformedGather {
            stage,
            expected,
            actual: gathered.len(),
            participants,
        });
    }
    Ok(gathered)
}

/// Failure while converting compatible manifests into a native-resource-free
/// communication realization.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CommunicationRealizationError {
    /// Rank-local manifests are not one compatible complete-world projection.
    #[error(transparent)]
    Manifest(#[from] CommunicationManifestError),
    /// Required operation, completion, or reachability support is unavailable.
    #[error(transparent)]
    Capability(#[from] CommunicationCapabilityError),
    /// The selected choices could not be bound to the native descriptors.
    #[error(transparent)]
    Binding(#[from] CommunicationBindingGeometryError),
    /// The separately supplied local manifest differs from its agreed rank entry.
    #[error("local communication manifest differs from the agreed rank projection")]
    LocalManifestMismatch,
}

/// Validates capability and reachability once, before any native subgroup or
/// directed-route resource is created.
pub fn prepare_communication_realization(
    manifest: &CommunicationManifest,
    compatible_manifests: &[CommunicationManifest],
    capabilities: &CommunicationCapabilities,
    topology: CommunicationTopologyCapabilities,
) -> Result<CommunicationBindings, CommunicationRealizationError> {
    validate_compatible_communication_manifests(compatible_manifests)?;
    let agreed = compatible_manifests
        .iter()
        .find(|candidate| candidate.rank() == manifest.rank())
        .ok_or(CommunicationRealizationError::LocalManifestMismatch)?;
    if agreed != manifest {
        return Err(CommunicationRealizationError::LocalManifestMismatch);
    }
    capabilities.validate_manifest(manifest)?;

    let group_wave_proofs = compatible_group_world_wave_proofs(compatible_manifests);
    let route_wave_proofs = route_world_wave_proofs(manifest);
    let mut selected_group_waves = vec![false; manifest.groups().len()];
    let mut selected_route_waves = vec![false; manifest.routes().len()];

    if topology == CommunicationTopologyCapabilities::RingWithWorldWaves {
        for (order, group) in manifest.groups().iter().enumerate() {
            let directly_reachable = group.members().len() <= 1
                || group.members().len() == manifest.world_size()
                || (group.members().len() == 2
                    && ring_neighbors(
                        group.members()[0],
                        group.members()[1],
                        manifest.world_size(),
                    ));
            if !directly_reachable {
                if !group_wave_proofs.get(order).copied().unwrap_or(false) {
                    return Err(
                        CommunicationCapabilityError::UnreachableGroup { id: group.id() }.into(),
                    );
                }
                selected_group_waves[order] = true;
            }
        }
        for (order, route) in manifest.routes().iter().enumerate() {
            if !route.world_broadcast()
                && !ring_neighbors(route.source(), route.destination(), manifest.world_size())
            {
                if !route_wave_proofs.get(order).copied().unwrap_or(false) {
                    return Err(CommunicationCapabilityError::UnreachableRoute {
                        id: route.id(),
                        source_rank: route.source(),
                        destination_rank: route.destination(),
                    }
                    .into());
                }
                selected_route_waves[order] = true;
            }
        }
    }

    CommunicationBindings::new(manifest.clone(), selected_group_waves, selected_route_waves)
        .map_err(Into::into)
}

/// Failure agreeing the world setup or preparing its rank-local realization.
#[derive(Debug, thiserror::Error)]
pub enum CommunicationSetupError<E: std::error::Error> {
    /// Ranks failed to agree on a compatible world.
    #[error(transparent)]
    Consensus(#[from] CommunicationManifestConsensusError<E>),
    /// Agreed topology exceeds the native communication capabilities.
    #[error(transparent)]
    Realization(#[from] CommunicationRealizationError),
    /// An agreed descriptor cannot be represented from this rank.
    #[error(transparent)]
    Manifest(#[from] CommunicationManifestError),
    /// Could not establish the final native binding.
    #[error(transparent)]
    Binding(#[from] CommunicationBindingError),
}

/// Agrees exact opaque manifests and plans native handles before allocation.
pub fn prepare_world_communication<T: CommunicationSetupTransport>(
    transport: T,
    manifest: &CommunicationManifest,
) -> Result<WorldCommunicationBinding<T>, CommunicationSetupError<T::Error>> {
    let agreed = establish_communication_session(&transport, manifest, transport.setup_nonce())?;
    let (identity, manifests) = agreed.into_parts();
    let realization = prepare_communication_realization(
        manifest,
        &manifests,
        &transport.capabilities(),
        transport.topology_capabilities(),
    )?;
    let mut descriptors = std::collections::BTreeMap::new();
    for group in manifests.iter().flat_map(CommunicationManifest::groups) {
        if let std::collections::btree_map::Entry::Vacant(entry) = descriptors.entry(group.id()) {
            entry.insert(CommunicationGroupDescriptor::new(
                group.id(),
                group.creation_order(),
                group.members().to_vec(),
                group
                    .members()
                    .iter()
                    .position(|rank| *rank == manifest.rank()),
                group.requirements().clone(),
            )?);
        }
    }
    WorldCommunicationBinding::new(
        transport,
        realization,
        descriptors.into_values().collect(),
        identity,
    )
    .map_err(Into::into)
}

/// Returns complete-world collective-wave proofs for each compatible group order.
///
/// Callers must validate the complete manifest set first. The prepared
/// realization constructor performs both steps atomically for production use.
pub fn compatible_group_world_wave_proofs(manifests: &[CommunicationManifest]) -> Vec<bool> {
    let group_count = manifests
        .first()
        .map_or(0, |manifest| manifest.groups().len());
    (0..group_count)
        .map(|order| {
            let Some(requirements) = manifests
                .first()
                .and_then(|manifest| manifest.groups().get(order))
                .map(CommunicationGroupDescriptor::requirements)
            else {
                return false;
            };
            manifests.iter().all(|manifest| {
                manifest
                    .groups()
                    .get(order)
                    .is_some_and(|group| group.requirements() == requirements)
            })
        })
        .collect()
}

/// Returns the manifest-proven route submission waves in canonical order.
///
/// A multi-route wave is selected only when one contiguous batch has an
/// identical mechanical contract, disjoint endpoints, and covers every
/// world rank exactly once. Such logical endpoint groups may share one
/// world-backed native collective ordinal. Routes without that proof stay
/// in singleton waves and retain ordinary sequential point-to-point order.
pub(crate) fn route_submission_waves(manifest: &CommunicationManifest) -> Vec<Range<usize>> {
    let mut waves = Vec::new();
    let mut start = 0;
    while start < manifest.routes().len() {
        let reference = &manifest.routes()[start];
        if reference.world_broadcast() {
            waves.push(start..start + 1);
            start += 1;
            continue;
        }
        let mut endpoints = vec![false; manifest.world_size()];
        let mut end = start;
        while end < manifest.routes().len() {
            let route = &manifest.routes()[end];
            if route.world_broadcast()
                || route.requirement() != reference.requirement()
                || route.boundary_contract() != reference.boundary_contract()
                || endpoints[route.source()]
                || endpoints[route.destination()]
            {
                break;
            }
            endpoints[route.source()] = true;
            endpoints[route.destination()] = true;
            end += 1;
            if endpoints.iter().all(|endpoint| *endpoint) {
                break;
            }
        }
        if !endpoints.iter().all(|endpoint| *endpoint) {
            end = start + 1;
        }
        waves.push(start..end);
        start = end;
    }
    waves
}

/// Returns complete-world collective-wave proofs for each route submission order.
///
/// Production backends should normally consume these through
/// [`prepare_communication_realization`].
pub fn route_world_wave_proofs(manifest: &CommunicationManifest) -> Vec<bool> {
    let mut proofs = vec![false; manifest.routes().len()];
    for wave in route_submission_waves(&manifest) {
        let mut members = vec![false; manifest.world_size()];
        for route in &manifest.routes()[wave.clone()] {
            members[route.source()] = true;
            members[route.destination()] = true;
        }
        if members.iter().all(|member| *member) {
            proofs[wave].fill(true);
        }
    }
    proofs
}

fn ring_neighbors(left: usize, right: usize, world_size: usize) -> bool {
    world_size > 1 && ((left + 1) % world_size == right || (right + 1) % world_size == left)
}

fn unique_axis_groups(
    topology: ParallelTopology,
    axis: ParallelAxis,
) -> Result<Vec<Vec<usize>>, CommunicationProjectionError> {
    let mut seen = BTreeSet::new();
    let mut groups = Vec::new();
    for rank in 0..topology.world_size() {
        let members = topology
            .axis_members(rank, axis)
            .map_err(|_| CommunicationProjectionError::InvalidTopologyProjection)?;
        if seen.insert(members.clone()) {
            groups.push(members);
        }
    }
    Ok(groups)
}

/// Failure while exchanging or validating complete rank-local manifests.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CommunicationManifestConsensusError<E = std::convert::Infallible> {
    /// A consensus topology cannot be empty.
    #[error("communication manifest consensus topology has no participants")]
    EmptyTopology,
    /// Portable metadata exceeded its wire or host representation.
    #[error("communication manifest consensus {0} overflowed")]
    MetadataOverflow(&'static str),
    /// The local manifest could not be encoded.
    #[error("communication manifest encoding failed: {0}")]
    Encoding(String),
    /// Agreed frame sizes exceed the fixed session-setup admission bounds.
    #[error(
        "communication session metadata exceeds {per_rank} bytes per rank or {total} gathered bytes"
    )]
    MetadataLimit {
        /// Maximum serialized bytes in one proposal.
        per_rank: usize,
        /// Maximum padded rank-major serialized payload.
        total: usize,
    },
    /// Session metadata could not establish a fresh common setup identity.
    #[error("communication session proposal at rank {rank} is invalid: {reason}")]
    InvalidSessionProposal {
        /// First invalid proposal in world-rank order.
        rank: usize,
        /// Version or nonce validation failure.
        reason: &'static str,
    },
    /// The control-plane collective failed.
    #[error("communication manifest consensus transport failed: {0}")]
    Transport(#[source] E),
    /// An equal-word gather returned a malformed rank-major frame.
    #[error(
        "communication manifest {stage} gather returned {actual} words; expected {expected} for {participants} ranks"
    )]
    MalformedGather {
        /// Protocol stage being gathered.
        stage: &'static str,
        /// Expected gathered word count.
        expected: usize,
        /// Actual gathered word count.
        actual: usize,
        /// Expected participant count.
        participants: usize,
    },
    /// A peer encoded a payload length that this host cannot represent.
    #[error("communication manifest payload length {length} from rank {rank} exceeds usize")]
    PayloadLengthOverflow {
        /// Rank that advertised the length.
        rank: usize,
        /// Advertised encoded byte length.
        length: u64,
    },
    /// A peer advertised more bytes than fit in the gathered padded frame.
    #[error(
        "communication manifest payload from rank {rank} has length {length}, but only {available} bytes were gathered"
    )]
    InvalidPayloadLength {
        /// Rank that advertised the invalid length.
        rank: usize,
        /// Advertised encoded byte length.
        length: usize,
        /// Bytes available in one gathered rank frame.
        available: usize,
    },
    /// A peer payload was not a valid checked opaque manifest.
    #[error("communication manifest payload from rank {rank} is invalid: {message}")]
    InvalidEncoding {
        /// Rank whose payload failed decoding or checked construction.
        rank: usize,
        /// Serialization or manifest validation detail.
        message: String,
    },
    /// The reconstructed rank manifests are not mutually compatible.
    #[error(transparent)]
    Manifest(#[from] CommunicationManifestError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_completion_policy() -> CommunicationCompletionPolicy {
        CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap()
    }

    fn test_completion_capabilities() -> CommunicationCompletionCapabilities {
        CommunicationCompletionCapabilities::new([
            CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .unwrap()
    }

    struct ScriptedManifestTransport {
        encoded: Vec<Vec<u8>>,
        calls: std::cell::Cell<usize>,
    }

    impl ScriptedManifestTransport {
        fn new(manifests: &[CommunicationManifest]) -> Self {
            Self {
                encoded: manifests
                    .iter()
                    .map(|manifest| serde_json::to_vec(manifest).unwrap())
                    .collect(),
                calls: std::cell::Cell::new(0),
            }
        }
    }

    impl ConsensusTransport for ScriptedManifestTransport {
        type Error = std::convert::Infallible;

        fn participant_count(&self) -> usize {
            self.encoded.len()
        }

        fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            match call {
                0 => {
                    assert_eq!(local.len(), 2);
                    Ok(self
                        .encoded
                        .iter()
                        .flat_map(|payload| {
                            let length = u64::try_from(payload.len()).unwrap();
                            [length as u32, (length >> 32) as u32]
                        })
                        .collect())
                }
                1 => {
                    let padded_words = self
                        .encoded
                        .iter()
                        .map(|payload| words_for_bytes(payload.len()))
                        .max()
                        .unwrap();
                    assert_eq!(local.len(), padded_words);
                    Ok(self
                        .encoded
                        .iter()
                        .flat_map(|payload| encode_manifest_words(payload, padded_words))
                        .collect())
                }
                _ => panic!("manifest consensus performs exactly two gathers"),
            }
        }
    }

    fn limits() -> CommunicationTensorLimits {
        CommunicationTensorLimits::new(2, 3, 4096, None).unwrap()
    }

    fn requirement(operation: CommunicationOperation) -> CommunicationOperationRequirement {
        CommunicationOperationRequirement::tensors(
            operation,
            [TensorDtype::F32, TensorDtype::Bf16],
            if operation == CommunicationOperation::VariableAllToAll {
                CommunicationTensorLimits::new(2, 3, 4096, Some(2048)).unwrap()
            } else {
                limits()
            },
            true,
        )
        .unwrap()
    }

    fn group_requirements(operation: CommunicationOperation) -> CommunicationGroupRequirements {
        CommunicationGroupRequirements::new([requirement(operation)]).unwrap()
    }

    #[test]
    fn retained_group_rejects_invocations_outside_selected_communication_bounds() {
        let operation = CommunicationOperation::AllReduceSum;
        let group = group_requirements(operation);
        let request = |dtype, tensors, rank, input, output, exact| {
            CommunicationOperationRequirement::tensors(
                operation,
                [dtype],
                CommunicationTensorLimits::new(tensors, rank, input, None)
                    .unwrap()
                    .with_output_tensor_elements(output)
                    .unwrap(),
                exact,
            )
            .unwrap()
        };
        // An admitted target block of 32x128 also admits a shorter assistant
        // block, but never a larger result, tensor batch or logical rank.
        group
            .validate_requirement(&request(TensorDtype::F32, 1, 3, 896, 896, true))
            .unwrap();
        for (tensors, rank, input, output) in [
            (3, 3, 4096, 4096),
            (2, 4, 4096, 4096),
            (2, 3, 4097, 4096),
            (2, 3, 4096, 4097),
        ] {
            assert_eq!(
                group.validate_requirement(&request(
                    TensorDtype::F32,
                    tensors,
                    rank,
                    input,
                    output,
                    true,
                )),
                Err(CommunicationCapabilityError::InsufficientLimits { operation })
            );
        }
        assert_eq!(
            group.validate_requirement(&request(TensorDtype::F16, 1, 3, 896, 896, true)),
            Err(CommunicationCapabilityError::UnsupportedDtype {
                operation,
                dtype: TensorDtype::F16
            })
        );
        assert_eq!(
            group.validate_requirement(&CommunicationOperationRequirement::barrier(true)),
            Err(CommunicationCapabilityError::MissingOperation {
                operation: CommunicationOperation::Barrier
            })
        );
        let inexact = CommunicationGroupRequirements::new([request(
            TensorDtype::F32,
            2,
            3,
            4096,
            4096,
            false,
        )])
        .unwrap();
        assert_eq!(
            inexact.validate_requirement(&request(TensorDtype::F32, 1, 3, 896, 896, true)),
            Err(CommunicationCapabilityError::MissingExactCompletion { operation })
        );
    }

    fn route_requirement() -> CommunicationOperationRequirement {
        requirement(CommunicationOperation::SendReceive)
    }

    fn projection_plan() -> TopologyCommunicationPlan {
        TopologyCommunicationPlan::new()
            .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum))
            .with_expert_groups(group_requirements(CommunicationOperation::VariableAllToAll))
            .with_data_groups(group_requirements(CommunicationOperation::AllGatherEven))
            .with_pipeline_routes(route_requirement())
            .unwrap()
    }

    #[test]
    fn pipeline_stage_groups_join_tensor_expert_replicas_and_preserve_axis_authority() {
        let requirements = CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::failure_agreement(true),
        ])
        .unwrap();
        for (tensor, pipeline, expert, data) in [(1, 2, 2, 1), (2, 2, 2, 1), (3, 3, 2, 2)] {
            let topology = ParallelTopology::new(tensor, pipeline, expert, data).unwrap();
            let original = projection_plan();
            let baseline = project_all_communication_manifests(topology, &original).unwrap();
            let plan = original.with_pipeline_stage_groups(requirements.clone());
            let manifests = project_all_communication_manifests(topology, &plan).unwrap();
            for (rank, (manifest, before)) in manifests.iter().zip(&baseline).enumerate() {
                assert_eq!(&manifest.groups()[..before.groups().len()], before.groups());
                let group = manifest.groups().last().unwrap();
                let owner = topology.coordinates(rank).unwrap();
                let expected = (0..topology.world_size())
                    .filter(|other| {
                        let other = topology.coordinates(*other).unwrap();
                        other.pipeline() == owner.pipeline() && other.data() == owner.data()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(group.members(), expected);
                assert_eq!(
                    group.local_index(),
                    expected.iter().position(|member| *member == rank)
                );
                assert_eq!(group.requirements(), &requirements);
                assert_eq!(
                    plan.tensor_group_id(
                        topology,
                        ParallelRankTopology::new(topology, rank).unwrap()
                    )
                    .unwrap(),
                    manifest
                        .groups()
                        .first()
                        .map(CommunicationGroupDescriptor::id)
                );
            }
        }
    }

    fn publication_requirements() -> CommunicationGroupRequirements {
        CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 8192, None).unwrap(),
                true,
            )
            .unwrap(),
            CommunicationOperationRequirement::barrier(true),
        ])
        .unwrap()
    }

    #[test]
    fn session_group_is_first_stable_world_group_with_exact_publication_requirements() {
        let topology = ParallelTopology::new(2, 2, 1, 1).unwrap();
        let plan = TopologyCommunicationPlan::new()
            .with_completion_policy(test_completion_policy())
            .with_session_group(publication_requirements())
            .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum));
        let selected_id = plan.session_group_id().unwrap();
        let manifests = project_all_communication_manifests(topology, &plan).unwrap();

        assert_eq!(selected_id, CollectiveGroupId::new(1));
        for (rank, manifest) in manifests.iter().enumerate() {
            let session = &manifest.groups()[0];
            assert_eq!(session.id(), selected_id);
            assert_eq!(session.creation_order(), 0);
            assert_eq!(session.members(), [0, 1, 2, 3]);
            assert_eq!(session.local_index(), Some(rank));
            assert_eq!(session.requirements(), &publication_requirements());
            assert_ne!(manifest.groups()[1].id(), selected_id);
        }

        let capabilities = CommunicationCapabilities::new([
            requirement(CommunicationOperation::AllReduceSum),
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 8192, None).unwrap(),
                true,
            )
            .unwrap(),
            CommunicationOperationRequirement::barrier(true),
        ])
        .unwrap()
        .with_completion_capabilities(test_completion_capabilities());
        capabilities.validate_manifest(&manifests[0]).unwrap();

        let graph =
            crate::ExecutionGraph::new(vec![crate::ExecutionGroupSpec::root("decoder")], "decoder")
                .unwrap();
        let execution = crate::PartitionedExecutionPlan::new(
            graph,
            vec![(crate::ArchitectureGroupKind::Decoder, false)],
            vec![None],
            Vec::new(),
            Some(crate::PartitionOutputPublication {
                group: selected_id,
                owner_rank: 3,
            }),
            Some(selected_id),
            crate::PipelineWireContract::new(
                eredu_core::communication::PipelineActivationDtype::Float32,
            ),
        )
        .unwrap();
        assert_eq!(execution.publication().unwrap().group, selected_id);
        assert_eq!(execution.commit_barrier(), Some(selected_id));
    }

    #[test]
    fn group_and_manifest_validation_reject_membership_and_local_index_corruption() {
        let requirements = group_requirements(CommunicationOperation::AllReduceSum);
        assert_eq!(
            CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(4),
                0,
                vec![0, 0],
                Some(0),
                requirements.clone(),
            ),
            Err(CommunicationManifestError::DuplicateGroupMember {
                id: CollectiveGroupId::new(4)
            })
        );

        let out_of_range = CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(4),
            0,
            vec![0, 3],
            Some(0),
            requirements.clone(),
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![out_of_range], vec![]),
            Err(CommunicationManifestError::GroupMemberOutOfRange {
                id: CollectiveGroupId::new(4),
                world_size: 2,
            })
        );

        let wrong_local = CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(4),
            0,
            vec![0, 1],
            Some(1),
            requirements,
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![wrong_local], vec![]),
            Err(CommunicationManifestError::WrongLocalIndex {
                id: CollectiveGroupId::new(4)
            })
        );
    }

    #[test]
    fn route_and_order_validation_rejects_wrong_endpoints_and_sequences() {
        let route = CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(8),
            0,
            0,
            3,
            route_requirement(),
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![], vec![route]),
            Err(CommunicationManifestError::RouteEndpointOutOfRange {
                id: CommunicationRouteId::new(8),
                world_size: 2,
            })
        );

        let group = CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(2),
            1,
            vec![0, 1],
            Some(0),
            group_requirements(CommunicationOperation::AllReduceSum),
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![group], vec![]),
            Err(CommunicationManifestError::WrongGroupOrder {
                id: CollectiveGroupId::new(2),
                expected: 0,
                actual: 1,
            })
        );

        let route = CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(8),
            1,
            0,
            1,
            route_requirement(),
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![], vec![route]),
            Err(CommunicationManifestError::WrongRouteOrder {
                id: CommunicationRouteId::new(8),
                expected: 0,
                actual: 1,
            })
        );
    }

    #[test]
    fn cartesian_projection_is_deterministic_and_compatible_for_every_rank() {
        let topology = ParallelTopology::new(2, 3, 2, 2).unwrap();
        let manifests = project_all_communication_manifests(topology, &projection_plan()).unwrap();
        assert_eq!(manifests.len(), topology.world_size());
        assert_eq!(manifests[0].groups().len(), 3);
        assert_eq!(manifests[0].routes().len(), 16);
        assert_eq!(manifests[0].groups()[0].members(), [0, 2]);
        assert_eq!(manifests[0].groups()[0].local_index(), Some(0));
        assert_eq!(manifests[1].groups()[0].members(), [1, 3]);
        assert_eq!(manifests[1].groups()[0].local_index(), Some(0));
        assert_eq!(manifests[0].routes()[0].source(), 0);
        assert_eq!(manifests[0].routes()[0].destination(), 4);

        let again = project_all_communication_manifests(topology, &projection_plan()).unwrap();
        assert_eq!(manifests, again);
        validate_compatible_communication_manifests(&manifests).unwrap();
    }

    #[test]
    fn cross_rank_validation_detects_descriptor_disagreement() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let mut manifests = project_all_communication_manifests(
            topology,
            &TopologyCommunicationPlan::new()
                .with_completion_policy(test_completion_policy())
                .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum)),
        )
        .unwrap();
        let original = &manifests[1].groups()[0];
        let mut members = original.members().to_vec();
        members.swap(0, 1);
        let group = CommunicationGroupDescriptor::new(
            original.id(),
            original.creation_order(),
            members,
            Some(0),
            original.requirements().clone(),
        )
        .unwrap();
        manifests[1] = CommunicationManifest::new(2, 1, vec![group], vec![])
            .unwrap()
            .with_completion_policy(test_completion_policy());
        assert_eq!(
            validate_compatible_communication_manifests(&manifests),
            Err(CommunicationManifestError::IncompatibleRankManifest { rank: 1 })
        );
    }

    #[test]
    fn manifest_consensus_gathers_variable_payloads_and_rejects_limit_disagreement() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let mut manifests = project_all_communication_manifests(
            topology,
            &TopologyCommunicationPlan::new()
                .with_completion_policy(test_completion_policy())
                .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum)),
        )
        .unwrap();
        let rank_one = &manifests[1].groups()[0];
        let group = CommunicationGroupDescriptor::new(
            rank_one.id(),
            rank_one.creation_order(),
            rank_one.members().to_vec(),
            rank_one.local_index(),
            CommunicationGroupRequirements::new([CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllReduceSum,
                [TensorDtype::F32, TensorDtype::Bf16],
                CommunicationTensorLimits::new(2, 3, 32, None).unwrap(),
                true,
            )
            .unwrap()])
            .unwrap(),
        )
        .unwrap();
        manifests[1] = CommunicationManifest::new(2, 1, vec![group], vec![])
            .unwrap()
            .with_completion_policy(test_completion_policy());
        let transport = ScriptedManifestTransport::new(&manifests);

        assert_eq!(
            validate_communication_manifest_consensus(&transport, &manifests[0]),
            Err(CommunicationManifestConsensusError::Manifest(
                CommunicationManifestError::IncompatibleRankManifest { rank: 1 }
            ))
        );
        assert_eq!(transport.calls.get(), 2);
        assert_ne!(transport.encoded[0].len(), transport.encoded[1].len());
    }

    #[test]
    fn manifest_consensus_accepts_complete_compatible_rank_artifacts() {
        let manifests = project_all_communication_manifests(
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            &TopologyCommunicationPlan::new()
                .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum)),
        )
        .unwrap();
        let transport = ScriptedManifestTransport::new(&manifests);

        validate_communication_manifest_consensus(&transport, &manifests[0]).unwrap();
        assert_eq!(transport.calls.get(), 2);
    }

    #[test]
    fn manifest_consensus_defers_world_validation_until_after_both_gathers() {
        let manifests = vec![
            CommunicationManifest::new(2, 0, Vec::new(), Vec::new()).unwrap(),
            CommunicationManifest::new(3, 1, Vec::new(), Vec::new()).unwrap(),
        ];
        let transport = ScriptedManifestTransport::new(&manifests);

        assert_eq!(
            validate_communication_manifest_consensus(&transport, &manifests[0]),
            Err(CommunicationManifestConsensusError::Manifest(
                CommunicationManifestError::IncompatibleRankManifest { rank: 1 }
            ))
        );
        assert_eq!(transport.calls.get(), 2);
    }

    #[test]
    fn capability_validation_is_fine_grained_and_fail_closed() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let manifest = project_all_communication_manifests(
            topology,
            &TopologyCommunicationPlan::new()
                .with_completion_policy(test_completion_policy())
                .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum)),
        )
        .unwrap()
        .remove(0);

        let missing = CommunicationCapabilities::new([])
            .unwrap()
            .with_completion_capabilities(test_completion_capabilities());
        assert_eq!(
            missing.validate_manifest(&manifest),
            Err(CommunicationCapabilityError::MissingOperation {
                operation: CommunicationOperation::AllReduceSum,
            })
        );

        let narrow = CommunicationCapabilities::new([CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 2, 32, None).unwrap(),
            false,
        )
        .unwrap()])
        .unwrap()
        .with_completion_capabilities(test_completion_capabilities());
        assert_eq!(
            narrow.validate_manifest(&manifest),
            Err(CommunicationCapabilityError::UnsupportedDtype {
                operation: CommunicationOperation::AllReduceSum,
                dtype: TensorDtype::Bf16,
            })
        );
    }

    #[test]
    fn capability_validation_covers_gather_result_size_separately() {
        let required = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllGatherUneven,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 32, None)
                .unwrap()
                .with_output_tensor_elements(64)
                .unwrap(),
            true,
        )
        .unwrap();
        let available =
            CommunicationCapabilities::new([CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllGatherUneven,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 64, None).unwrap(),
                true,
            )
            .unwrap()])
            .unwrap();
        assert!(available.validate_requirement(&required).is_ok());

        let insufficient =
            CommunicationCapabilities::new([CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllGatherUneven,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 32, None).unwrap(),
                true,
            )
            .unwrap()])
            .unwrap();
        assert_eq!(
            insufficient.validate_requirement(&required),
            Err(CommunicationCapabilityError::InsufficientLimits {
                operation: CommunicationOperation::AllGatherUneven,
            })
        );
    }

    #[test]
    fn variable_peer_counts_preserve_zeroes_and_exact_member_order() {
        let counts = eredu_core::communication::CommunicationPeerCounts::new(
            vec![0, 3, 1],
            vec![2, 0, 2],
            3,
        )
        .unwrap();
        assert_eq!(counts.send(), [0, 3, 1]);
        assert_eq!(counts.receive(), [2, 0, 2]);
        assert_eq!(counts.group_size(), 3);
        assert_eq!(
            eredu_core::communication::CommunicationPeerCounts::new(vec![1], vec![1, 0], 2),
            Err(eredu_core::communication::CommunicationPeerCountsError {
                group_size: 2,
                send: 1,
                receive: 2,
            })
        );
    }

    #[test]
    fn failure_agreement_is_payload_free_and_not_covered_by_a_barrier() {
        let required = CommunicationOperationRequirement::failure_agreement(true);
        assert_eq!(
            required.operation(),
            CommunicationOperation::FailureAgreement
        );
        assert!(required.dtypes().is_empty());
        assert_eq!(required.limits(), None);

        let barrier_only =
            CommunicationCapabilities::new([CommunicationOperationRequirement::barrier(true)])
                .unwrap();
        assert_eq!(
            barrier_only.validate_requirement(&required),
            Err(CommunicationCapabilityError::MissingOperation {
                operation: CommunicationOperation::FailureAgreement,
            })
        );
    }

    #[test]
    fn completion_policy_deserialization_revalidates_positive_deadline() {
        assert!(
            serde_json::from_value::<CommunicationCompletionPolicy>(serde_json::json!({
                "timeout_millis": 0,
                "cancellation": "quarantine_until_complete"
            }))
            .is_err()
        );
    }

    #[test]
    fn communication_resources_require_selected_supported_completion() {
        let group = CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(1),
            0,
            vec![0],
            Some(0),
            CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)])
                .unwrap(),
        )
        .unwrap();
        let manifest = CommunicationManifest::new(1, 0, vec![group], Vec::new()).unwrap();
        let capabilities =
            CommunicationCapabilities::new([CommunicationOperationRequirement::barrier(true)])
                .unwrap()
                .with_completion_capabilities(
                    CommunicationCompletionCapabilities::new([
                        CompletionCancellationMode::QuarantineUntilComplete,
                    ])
                    .unwrap(),
                );
        assert_eq!(
            capabilities.validate_manifest(&manifest),
            Err(CommunicationCapabilityError::MissingSelectedCompletionPolicy)
        );
        let manifest = manifest.with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(1),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
        capabilities.validate_manifest(&manifest).unwrap();
    }

    #[test]
    fn role_exact_frames_use_actual_variable_dimensions_and_reject_role_or_fixed_drift() {
        let admitted = RoleExactBoundaryContract::new(
            "nemotron_h.target",
            [
                BoundaryRoleContract::symbolic(
                    "hidden",
                    TensorDtype::F32,
                    vec![
                        BoundaryDimensionContract::Variable { maximum: 4 },
                        BoundaryDimensionContract::Variable { maximum: 16 },
                        BoundaryDimensionContract::Fixed(12),
                    ],
                )
                .unwrap(),
                BoundaryRoleContract::symbolic(
                    "embedded",
                    TensorDtype::F32,
                    vec![
                        BoundaryDimensionContract::Variable { maximum: 4 },
                        BoundaryDimensionContract::Variable { maximum: 16 },
                        BoundaryDimensionContract::Fixed(12),
                    ],
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let actual = vec![
            BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1, 3, 12]).unwrap(),
            BoundaryRoleContract::new("embedded", TensorDtype::F32, vec![1, 3, 12]).unwrap(),
        ];
        let framed = admitted
            .frame_values(CommunicationRouteId::new(7), &actual, vec![1_u8, 2])
            .unwrap();
        assert_ne!(framed[0].header(), framed[1].header());

        let swapped = vec![actual[1].clone(), actual[0].clone()];
        assert_eq!(
            admitted.frame_values(CommunicationRouteId::new(7), &swapped, vec![1_u8, 2]),
            Err(CommunicationManifestError::InvalidBoundaryContract)
        );
        let fixed_shrink = vec![
            BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1, 3, 11]).unwrap(),
            actual[1].clone(),
        ];
        assert_eq!(
            admitted.frame_values(CommunicationRouteId::new(7), &fixed_shrink, vec![1_u8, 2],),
            Err(CommunicationManifestError::InvalidBoundaryContract)
        );
    }

    #[test]
    fn role_exact_framing_capability_and_byte_overflow_fail_before_submission() {
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 4096, None).unwrap(),
            true,
        )
        .unwrap();
        let role = BoundaryRoleContract::symbolic(
            "hidden",
            TensorDtype::F32,
            vec![BoundaryDimensionContract::Fixed(1)],
        )
        .unwrap();
        let route = CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(3),
            0,
            0,
            1,
            requirement.clone(),
        )
        .unwrap()
        .with_boundary_contract(RoleExactBoundaryContract::new("none", [role]).unwrap())
        .unwrap();
        let manifest = CommunicationManifest::new(2, 0, Vec::new(), vec![route])
            .unwrap()
            .with_completion_policy(
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        let capability = CommunicationCapabilities::new([requirement.clone()])
            .unwrap()
            .with_completion_capabilities(
                CommunicationCompletionCapabilities::new([
                    CompletionCancellationMode::QuarantineUntilComplete,
                ])
                .unwrap(),
            );
        assert!(matches!(
            capability.validate_manifest(&manifest),
            Err(CommunicationCapabilityError::MissingBoundaryFraming {
                protocol: BoundaryFramingProtocol::RoleExactV1,
            })
        ));

        for invalid in [
            BoundaryRoleContract::new("hidden", TensorDtype::I32, vec![1]).unwrap(),
            BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1, 1, 1, 1]).unwrap(),
            BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![4097]).unwrap(),
        ] {
            assert_eq!(
                CommunicationRouteDescriptor::new(
                    CommunicationRouteId::new(30),
                    0,
                    0,
                    1,
                    requirement.clone(),
                )
                .unwrap()
                .with_boundary_contract(
                    RoleExactBoundaryContract::new("invalid", [invalid]).unwrap(),
                ),
                Err(CommunicationManifestError::InvalidBoundaryContract),
            );
        }
        let inexact = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 4096, None).unwrap(),
            false,
        )
        .unwrap();
        assert_eq!(
            CommunicationRouteDescriptor::new(CommunicationRouteId::new(31), 0, 0, 1, inexact,)
                .unwrap()
                .with_boundary_contract(
                    RoleExactBoundaryContract::new(
                        "inexact",
                        [BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1]).unwrap()],
                    )
                    .unwrap(),
                ),
            Err(CommunicationManifestError::InvalidBoundaryContract),
        );

        let exact = RoleExactBoundaryContract::new(
            "overflow",
            [BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![usize::MAX, 2]).unwrap()],
        )
        .unwrap();
        assert_eq!(
            exact.frame_values(CommunicationRouteId::new(4), exact.roles(), vec![0_u8],),
            Err(CommunicationManifestError::InvalidBoundaryContract)
        );
    }

    #[test]
    fn role_exact_contract_deserialization_revalidates_roles_dimensions_and_dtypes() {
        let contract = RoleExactBoundaryContract::new(
            "fixture",
            [BoundaryRoleContract::symbolic(
                "hidden",
                TensorDtype::F32,
                vec![BoundaryDimensionContract::Fixed(4)],
            )
            .unwrap()],
        )
        .unwrap();
        let valid = serde_json::to_value(&contract).unwrap();

        let mut empty_role = valid.clone();
        empty_role["roles"][0]["role"] = serde_json::json!("");
        assert!(serde_json::from_value::<RoleExactBoundaryContract>(empty_role).is_err());

        let mut duplicate_role = valid.clone();
        let repeated = duplicate_role["roles"][0].clone();
        duplicate_role["roles"]
            .as_array_mut()
            .unwrap()
            .push(repeated);
        assert!(serde_json::from_value::<RoleExactBoundaryContract>(duplicate_role).is_err());

        let mut zero_dimension = valid.clone();
        zero_dimension["roles"][0]["shape"][0] = serde_json::json!({ "fixed": 0 });
        assert!(serde_json::from_value::<RoleExactBoundaryContract>(zero_dimension).is_err());

        let mut encoded_dtype = valid;
        encoded_dtype["roles"][0]["dtype"] = serde_json::json!({ "encoded": "q4" });
        assert!(serde_json::from_value::<RoleExactBoundaryContract>(encoded_dtype).is_err());
    }

    #[test]
    fn route_submission_waves_require_complete_disjoint_world_batches() {
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 64, None).unwrap(),
            true,
        )
        .unwrap();
        let boundary = RoleExactBoundaryContract::new(
            "hidden-v1",
            [BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1, 2, 4]).unwrap()],
        )
        .unwrap();
        let route = |order: usize, source: usize, destination: usize| {
            CommunicationRouteDescriptor::new(
                CommunicationRouteId::new(order as u64),
                order,
                source,
                destination,
                requirement.clone(),
            )
            .unwrap()
            .with_boundary_contract(boundary.clone())
            .unwrap()
        };

        let complete = CommunicationManifest::new(
            8,
            0,
            Vec::new(),
            (0..4).map(|rank| route(rank, rank + 4, rank)).collect(),
        )
        .unwrap();
        assert_eq!(route_submission_waves(&complete).len(), 1);
        assert_eq!(route_submission_waves(&complete)[0], 0..4);

        let omitted = CommunicationManifest::new(
            8,
            0,
            Vec::new(),
            (0..3).map(|rank| route(rank, rank + 4, rank)).collect(),
        )
        .unwrap();
        assert_eq!(route_submission_waves(&omitted), [0..1, 1..2, 2..3]);

        let overlapping = CommunicationManifest::new(
            8,
            0,
            Vec::new(),
            vec![
                route(0, 4, 0),
                route(1, 4, 1),
                route(2, 6, 2),
                route(3, 7, 3),
            ],
        )
        .unwrap();
        assert_eq!(
            route_submission_waves(&overlapping),
            [0..1, 1..2, 2..3, 3..4]
        );
    }

    fn route_only_capabilities(
        requirement: CommunicationOperationRequirement,
    ) -> CommunicationCapabilities {
        CommunicationCapabilities::new([requirement])
            .unwrap()
            .with_completion_capabilities(test_completion_capabilities())
    }

    #[test]
    fn prepared_realization_rejects_unreachable_ring_routes_before_callbacks() {
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 64, None).unwrap(),
            true,
        )
        .unwrap();
        let plan = TopologyCommunicationPlan::new()
            .with_completion_policy(test_completion_policy())
            .with_pipeline_routes(requirement.clone())
            .unwrap();
        let topology = ParallelTopology::new(2, 3, 1, 1).unwrap();
        let manifests = project_all_communication_manifests(topology, &plan).unwrap();
        let error = prepare_communication_realization(
            &manifests[0],
            &manifests,
            &route_only_capabilities(requirement),
            CommunicationTopologyCapabilities::RingWithWorldWaves,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CommunicationRealizationError::Capability(
                CommunicationCapabilityError::UnreachableRoute { .. }
            )
        ));
    }

    #[test]
    fn world_boundary_broadcast_requires_advertised_participation_before_realization() {
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 64, None).unwrap(),
            true,
        )
        .unwrap();
        let plan = TopologyCommunicationPlan::new()
            .with_completion_policy(test_completion_policy())
            .with_pipeline_routes(requirement.clone())
            .unwrap();
        let manifests =
            project_all_communication_manifests(ParallelTopology::new(2, 3, 1, 1).unwrap(), &plan)
                .unwrap()
                .into_iter()
                .map(|manifest| {
                    CommunicationManifest::new(
                        manifest.world_size(),
                        manifest.rank(),
                        manifest.groups().to_vec(),
                        manifest
                            .routes()
                            .iter()
                            .cloned()
                            .map(CommunicationRouteDescriptor::with_world_broadcast)
                            .collect(),
                    )
                    .unwrap()
                    .with_completion_policy(test_completion_policy())
                })
                .collect::<Vec<_>>();
        let ordinary = route_only_capabilities(requirement);
        assert!(matches!(
            prepare_communication_realization(
                &manifests[0],
                &manifests,
                &ordinary,
                CommunicationTopologyCapabilities::RingWithWorldWaves
            ),
            Err(CommunicationRealizationError::Capability(
                CommunicationCapabilityError::MissingWorldBoundaryBroadcast
            ))
        ));
        let prepared = prepare_communication_realization(
            &manifests[0],
            &manifests,
            &ordinary.with_world_boundary_broadcast(),
            CommunicationTopologyCapabilities::RingWithWorldWaves,
        )
        .unwrap();
        let route = &prepared.manifest().routes()[0];
        assert_eq!((route.source(), route.destination()), (0, 2));
        assert!(
            route.participates(5),
            "idle rank submits bounded transport without owning its destination"
        );
        let decoded: CommunicationManifest =
            serde_json::from_slice(&serde_json::to_vec(prepared.manifest()).unwrap()).unwrap();
        assert_eq!(&decoded, prepared.manifest());
        assert!(route_submission_waves(&decoded)
            .iter()
            .all(|wave| wave.len() == 1));
    }

    #[test]
    fn prepared_realization_proves_world_waves_before_mechanism_callbacks() {
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 64, None).unwrap(),
            true,
        )
        .unwrap();
        let plan = TopologyCommunicationPlan::new()
            .with_completion_policy(test_completion_policy())
            .with_pipeline_routes(requirement.clone())
            .unwrap();
        let topology = ParallelTopology::new(2, 2, 1, 1).unwrap();
        let manifests = project_all_communication_manifests(topology, &plan).unwrap();
        let prepared = prepare_communication_realization(
            &manifests[0],
            &manifests,
            &route_only_capabilities(requirement),
            CommunicationTopologyCapabilities::RingWithWorldWaves,
        )
        .unwrap();
        assert!(prepared
            .manifest()
            .routes()
            .iter()
            .enumerate()
            .all(|(order, _)| prepared.route_world_wave(order) == Some(true)));

        let routes = prepared
            .routes()
            .map(|(descriptor, world_wave)| {
                assert!(world_wave);
                descriptor.id()
            })
            .collect::<Vec<_>>();
        assert_eq!(routes.len(), prepared.manifest().routes().len());
    }
}
