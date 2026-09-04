//! MLX checkpoint placement and selective materialization.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    path::{Path, PathBuf},
};

use eredu_checkpoint::store::{
    CheckpointSource, ReadPolicy, SafetensorsWeightStore, TensorReadRequest, TensorSelection,
};
use eredu_core::{
    checkpoint::TensorDtype, BoundedSubmissionOutcome, CollectiveGroupDescriptor,
    CollectiveGroupId, Submission,
};
use eredu_runtime::{
    CommunicationCapabilities, CommunicationCompletionCapabilities, CommunicationGroupDescriptor,
    CommunicationManifest, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationRouteDescriptor, CommunicationRouteId, CommunicationTensorLimits, TensorPlacement,
};
use safemlx::{distributed::Group as NativeGroup, Array, Stream};

use crate::{
    backend::error::Error, backend::runtime::checkpoint::store::MlxParameterMaterializationContext,
};

use crate::backend::runtime::distributed::Group;
use crate::backend::topology::MlxRankContext;
#[cfg(test)]
use crate::backend::DeviceAssignment;
#[cfg(test)]
use safemlx::{Device, DeviceType};

#[cfg(test)]
static MANIFEST_GROUP_REALIZATIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(super) fn reset_manifest_group_realizations() {
    MANIFEST_GROUP_REALIZATIONS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn manifest_group_realizations() -> usize {
    MANIFEST_GROUP_REALIZATIONS.load(Ordering::Relaxed)
}

type LogicalRoutePlan = Vec<(usize, Vec<Option<usize>>)>;

/// Backend-only realization of one opaque collective group.
#[derive(Debug, Clone)]
pub(crate) struct CollectiveGroupRealization {
    id: CollectiveGroupId,
    members: Vec<usize>,
    local_rank: usize,
    split_color: usize,
    logical_routes: Option<LogicalRoutePlan>,
    manifest_descriptor: Option<CommunicationGroupDescriptor>,
}

impl CollectiveGroupRealization {
    pub(crate) fn new(
        id: CollectiveGroupId,
        members: Vec<usize>,
        local_rank: usize,
        split_color: usize,
        logical_routes: Option<LogicalRoutePlan>,
    ) -> Result<Self, Error> {
        CollectiveGroupDescriptor::new(id, members.clone(), local_rank)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(Self {
            id,
            members,
            local_rank,
            split_color,
            logical_routes,
            manifest_descriptor: None,
        })
    }

    pub(crate) fn descriptor(&self) -> CollectiveGroupDescriptor {
        CollectiveGroupDescriptor::new(self.id, self.members.clone(), self.local_rank)
            .expect("collective realization is validated at construction")
    }

    fn from_manifest_descriptor(descriptor: &CommunicationGroupDescriptor) -> Result<Self, Error> {
        let local_rank = descriptor.local_index().ok_or_else(|| {
            Error::Parallel(format!(
                "MLX communication group {} does not contain manifest rank",
                descriptor.id().value()
            ))
        })?;
        let split_color = usize::try_from(descriptor.id().value())
            .map_err(|_| Error::Parallel("opaque communication group ID exceeds usize".into()))?;
        if split_color == 0 {
            return Err(Error::Parallel(
                "opaque communication group ID zero is reserved".into(),
            ));
        }
        i32::try_from(split_color)
            .map_err(|_| Error::Parallel("opaque communication group ID exceeds i32".into()))?;
        i32::try_from(local_rank).map_err(|_| {
            Error::Parallel("opaque communication group local index exceeds i32".into())
        })?;
        let mut realization = Self::new(
            descriptor.id(),
            descriptor.members().to_vec(),
            local_rank,
            split_color,
            None,
        )?;
        realization.manifest_descriptor = Some(descriptor.clone());
        Ok(realization)
    }
}

/// Backend-only communication realization selected by architecture composition.
#[derive(Debug, Clone)]
pub(crate) struct CollectiveRealization {
    world_size: usize,
    global_rank: usize,
    groups: Vec<CollectiveGroupRealization>,
}

impl CollectiveRealization {
    pub(crate) fn new(
        world_size: usize,
        global_rank: usize,
        groups: Vec<CollectiveGroupRealization>,
    ) -> Result<Self, Error> {
        if world_size == 0 || global_rank >= world_size {
            return Err(Error::Parallel(
                "collective realization has an invalid world rank".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        if groups.iter().any(|group| {
            !ids.insert(group.id)
                || group.members.iter().any(|member| *member >= world_size)
                || group.members[group.local_rank] != global_rank
        }) {
            return Err(Error::Parallel(
                "collective realization has invalid or duplicate opaque groups".into(),
            ));
        }
        Ok(Self {
            world_size,
            global_rank,
            groups,
        })
    }

    pub(crate) const fn world_size(&self) -> usize {
        self.world_size
    }

    pub(crate) const fn global_rank(&self) -> usize {
        self.global_rank
    }

    fn from_manifest(manifest: &CommunicationManifest) -> Result<Self, Error> {
        let groups =
            manifest.try_create_groups(CollectiveGroupRealization::from_manifest_descriptor)?;
        Self::new(manifest.world_size(), manifest.rank(), groups)
    }
}

/// Backend-owned handle for one opaque directed communication route.
#[derive(Debug, Clone)]
pub struct CommunicationRouteRealization {
    descriptor: CommunicationRouteDescriptor,
    group: Option<Group>,
    endpoint: Option<CommunicationRouteEndpoint>,
    peer_rank: Option<usize>,
}

/// This rank's role in one realized directed route.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CommunicationRouteEndpoint {
    /// Supplies the boundary tensor bundle.
    Source,
    /// Receives the boundary tensor bundle.
    Destination,
}

impl CommunicationRouteRealization {
    fn from_descriptor(
        descriptor: &CommunicationRouteDescriptor,
        world: &Group,
        world_collective_wave: bool,
    ) -> Result<Self, Error> {
        if descriptor.source() >= world.size() || descriptor.destination() >= world.size() {
            return Err(Error::Parallel(format!(
                "communication route {} endpoints {} -> {} exceed owned world size {}",
                descriptor.id().value(),
                descriptor.source(),
                descriptor.destination(),
                world.size()
            )));
        }
        let endpoint = match world.rank() {
            rank if rank == descriptor.source() => Some(CommunicationRouteEndpoint::Source),
            rank if rank == descriptor.destination() => {
                Some(CommunicationRouteEndpoint::Destination)
            }
            _ => None,
        };
        let group = endpoint
            .map(|_| world.logical_subgroup(&[descriptor.source(), descriptor.destination()]))
            .transpose()
            .map(|group| group.map(|group| group.with_world_collective_wave(world_collective_wave)))
            .map_err(|error| {
                Error::Parallel(format!(
                    "failed to realize communication route {} as endpoint subgroup [{}, {}]: {error}",
                    descriptor.id().value(),
                    descriptor.source(),
                    descriptor.destination()
                ))
            })?;
        let peer_rank = endpoint.map(|endpoint| match endpoint {
            CommunicationRouteEndpoint::Source => 1,
            CommunicationRouteEndpoint::Destination => 0,
        });
        Ok(Self {
            descriptor: descriptor.clone(),
            group,
            endpoint,
            peer_rank,
        })
    }

    /// Returns the exact neutral descriptor retained by this route handle.
    pub const fn descriptor(&self) -> &CommunicationRouteDescriptor {
        &self.descriptor
    }

    /// Returns this rank's retained endpoint role, if it participates.
    pub const fn endpoint(&self) -> Option<CommunicationRouteEndpoint> {
        self.endpoint
    }

    /// Returns the endpoint-local peer index in the ordered `[source, destination]` group.
    pub const fn peer_rank(&self) -> Option<usize> {
        self.peer_rank
    }

    /// Returns the exact two-member endpoint subgroup on a participating rank.
    pub(crate) const fn group(&self) -> Option<&Group> {
        self.group.as_ref()
    }
}

#[derive(Debug)]
struct PreparedCommunicationManifest {
    groups: CollectiveRealization,
    routes: Vec<CommunicationRouteDescriptor>,
}

fn world_collective_wave_proofs(manifests: &[CommunicationManifest]) -> Result<Vec<bool>, Error> {
    let Some(reference) = manifests.first() else {
        return Err(Error::Parallel(
            "communication consensus returned no rank manifests".into(),
        ));
    };
    let world_size = reference.world_size();
    let mut by_rank = vec![None; world_size];
    for manifest in manifests {
        if manifest.world_size() != world_size
            || manifest.groups().len() != reference.groups().len()
            || manifest.rank() >= world_size
            || by_rank[manifest.rank()].replace(manifest).is_some()
        {
            return Err(Error::Parallel(
                "communication consensus returned invalid or duplicate rank manifests".into(),
            ));
        }
    }
    if by_rank.iter().any(Option::is_none) {
        return Err(Error::Parallel(
            "communication consensus omitted one or more rank manifests".into(),
        ));
    }
    Ok((0..reference.groups().len())
        .map(|order| {
            let requirements = reference.groups()[order].requirements();
            manifests.iter().all(|manifest| {
                let rank = manifest.rank();
                let Some(group) = manifest.groups().get(order) else {
                    return false;
                };
                if group.requirements() != requirements
                    || group.creation_order() != order
                    || group
                        .members()
                        .get(group.local_index().unwrap_or(usize::MAX))
                        != Some(&rank)
                {
                    return false;
                }
                group.members().iter().all(|member| {
                    by_rank
                        .get(*member)
                        .and_then(|peer| *peer)
                        .and_then(|peer| peer.groups().get(order))
                        .is_some_and(|peer| {
                            peer.id() == group.id()
                                && peer.members() == group.members()
                                && peer.requirements() == group.requirements()
                                && peer.creation_order() == group.creation_order()
                        })
                })
            })
        })
        .collect())
}

fn validate_logical_variable_all_to_all_waves(
    manifest: &CommunicationManifest,
    proofs: &[bool],
) -> Result<(), Error> {
    for (order, group) in manifest.groups().iter().enumerate() {
        let requires_wave = group.members().len() < manifest.world_size()
            && group.requirements().operations().iter().any(|requirement| {
                requirement.operation() == CommunicationOperation::VariableAllToAll
            });
        if requires_wave && proofs.get(order) != Some(&true) {
            return Err(Error::Parallel(format!(
                "opaque logical group {} selects VariableAllToAll without a consensus-proven world participation wave",
                group.id().value()
            )));
        }
    }
    Ok(())
}

fn route_world_collective_wave_proofs(
    world_size: usize,
    routes: &[CommunicationRouteDescriptor],
) -> Vec<bool> {
    let mut proofs = vec![false; routes.len()];
    let mut batch_start = 0;
    while batch_start < routes.len() {
        let reference = &routes[batch_start];
        let mut members = vec![false; world_size];
        let mut batch_end = batch_start;
        while batch_end < routes.len() {
            let route = &routes[batch_end];
            if route.requirement() != reference.requirement()
                || route.boundary_contract() != reference.boundary_contract()
                || members[route.source()]
                || members[route.destination()]
            {
                break;
            }
            members[route.source()] = true;
            members[route.destination()] = true;
            batch_end += 1;
            if members.iter().all(|member| *member) {
                proofs[batch_start..batch_end].fill(true);
                break;
            }
        }
        batch_start = batch_end.max(batch_start + 1);
    }
    proofs
}

struct NativeWorldManifestTransport<'a> {
    world: &'a NativeGroup,
    stream: &'a Stream,
    completion: eredu_runtime::CommunicationCompletionPolicy,
}

fn manifest_consensus_completion_policy() -> eredu_runtime::CommunicationCompletionPolicy {
    // The manifest's own policy is untrusted input until all ranks have
    // exchanged it. Use one backend setup bound solely for that control-plane
    // exchange, then validate and install the agreed serialized policy.
    eredu_runtime::CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(30),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .expect("static manifest-consensus completion policy is valid")
}

impl eredu_core::consensus::ConsensusTransport for NativeWorldManifestTransport<'_> {
    type Error = Error;

    fn participant_count(&self) -> usize {
        self.world.size()
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
        let deadline = safemlx::RuntimeCallDeadline::new(self.completion.timeout())?;
        let _setup = deadline.enter()?;
        let length = i32::try_from(local.len())
            .map_err(|_| Error::Parallel("communication manifest frame exceeds i32".into()))?;
        let signed = local.iter().map(|word| *word as i32).collect::<Vec<_>>();
        let local = Array::from_slice(&signed, &[length]);
        _setup.check()?;
        let gathered = safemlx::distributed::all_gather(&local, self.world, self.stream)?;
        let completion =
            crate::backend::runtime::distributed::completion::MlxCommunicationCompletion::submit(
                [&gathered],
                vec![local, gathered.clone()],
                Vec::new(),
                vec![Group::uncontracted(self.world)],
                Vec::new(),
                vec![self.stream.clone()],
            )?;
        let (words, completion) = completion.with_i32_words(gathered);
        match (Submission {
            output: words,
            completion,
        })
        .wait_bounded(self.completion.bounded_wait())?
        {
            BoundedSubmissionOutcome::Completed(words) => {
                Ok(words
                    .resolve()?
                    .iter()
                    .map(|word| *word as u32)
                    .collect())
            }
            BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
                Err(Error::Parallel(format!(
                    "communication manifest consensus exceeded its selected deadline ({cancellation:?})"
                )))
            }
        }
    }
}

impl PreparedCommunicationManifest {
    fn new(manifest: &CommunicationManifest) -> Result<Self, Error> {
        let groups = CollectiveRealization::from_manifest(manifest)?;
        let routes = manifest.routes().to_vec();
        Ok(Self { groups, routes })
    }
}

fn mlx_collective_dtypes() -> [TensorDtype; 3] {
    [TensorDtype::F32, TensorDtype::F16, TensorDtype::Bf16]
}

fn mlx_even_gather_dtypes() -> [TensorDtype; 4] {
    [
        TensorDtype::F32,
        TensorDtype::F16,
        TensorDtype::Bf16,
        TensorDtype::I32,
    ]
}

fn mlx_variable_all_to_all_dtypes() -> [TensorDtype; 4] {
    [
        TensorDtype::F32,
        TensorDtype::F16,
        TensorDtype::Bf16,
        TensorDtype::I32,
    ]
}

fn mlx_point_to_point_dtypes() -> [TensorDtype; 5] {
    [
        TensorDtype::F32,
        TensorDtype::F16,
        TensorDtype::Bf16,
        TensorDtype::I32,
        TensorDtype::U32,
    ]
}

/// Conservative operation surface implemented by reusable MLX mechanisms.
pub(crate) fn mlx_communication_capabilities() -> CommunicationCapabilities {
    let tensor_limits =
        CommunicationTensorLimits::new(1, i32::MAX as usize, i32::MAX as usize, None)
            .expect("static MLX tensor limits are valid");
    let collective_requirement = |operation| {
        CommunicationOperationRequirement::tensors(
            operation,
            mlx_collective_dtypes(),
            tensor_limits,
            true,
        )
        .expect("static MLX communication capability is valid")
    };
    let variable = CommunicationOperationRequirement::tensors(
        CommunicationOperation::VariableAllToAll,
        mlx_variable_all_to_all_dtypes(),
        CommunicationTensorLimits::new(
            1,
            i32::MAX as usize,
            i32::MAX as usize,
            Some(i32::MAX as usize),
        )
        .expect("static MLX variable all-to-all limits are valid"),
        true,
    )
    .expect("static MLX variable all-to-all capability is valid");
    let point_to_point = CommunicationOperationRequirement::tensors(
        CommunicationOperation::SendReceive,
        mlx_point_to_point_dtypes(),
        CommunicationTensorLimits::new(
            i32::MAX as usize,
            i32::MAX as usize,
            i32::MAX as usize,
            None,
        )
        .expect("static MLX point-to-point limits are valid"),
        true,
    )
    .expect("static MLX point-to-point capability is valid");
    CommunicationCapabilities::new([
        collective_requirement(CommunicationOperation::AllReduceSum),
        CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllGatherEven,
            mlx_even_gather_dtypes(),
            tensor_limits,
            true,
        )
        .expect("static MLX even all-gather capability is valid"),
        collective_requirement(CommunicationOperation::AllGatherUneven),
        variable,
        point_to_point,
        collective_requirement(CommunicationOperation::Broadcast),
        CommunicationOperationRequirement::barrier(true),
        CommunicationOperationRequirement::failure_agreement(true),
    ])
    .expect("static MLX communication capabilities are valid")
    .with_boundary_framing([eredu_runtime::BoundaryFramingProtocol::RoleExactV1])
    .expect("static MLX boundary framing capability is valid")
    .with_completion_capabilities(
        CommunicationCompletionCapabilities::new([
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .expect("static MLX completion capabilities are valid"),
    )
}

fn validate_mlx_communication_manifest(manifest: &CommunicationManifest) -> Result<(), Error> {
    mlx_communication_capabilities()
        .validate_manifest(manifest)
        .map_err(|error| {
            Error::Parallel(format!(
                "communication manifest exceeds MLX mechanism capabilities: {error}"
            ))
        })
}

/// Backend communication contexts materialized from opaque group realizations.
///
/// Uncontracted construction may enter native subgroup splits. Opaque manifest
/// construction creates exact logical membership views and permits a
/// world-collective implementation only when consensus proves a complete,
/// same-requirement subgroup wave at that creation order.
pub struct ParallelCommunicators {
    realization: CollectiveRealization,
    control_world: Group,
    groups: HashMap<CollectiveGroupId, GroupCommunicator>,
    routes: HashMap<CommunicationRouteId, CommunicationRouteRealization>,
}

struct GroupCommunicator {
    realization: CollectiveGroupRealization,
    native: Option<Group>,
}

impl std::fmt::Debug for ParallelCommunicators {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ParallelCommunicators")
            .field("realization", &self.realization)
            .finish()
    }
}

impl ParallelCommunicators {
    /// Realizes an already selected opaque manifest without semantic topology input.
    ///
    /// Capability and descriptor conversion complete before group realization.
    /// Selected subgroups never enter MLX's non-cancellable native split.
    pub fn from_manifest(
        manifest: &CommunicationManifest,
        world: &NativeGroup,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let agreed_manifests = eredu_runtime::validate_communication_manifest_consensus(
            &NativeWorldManifestTransport {
                world,
                stream,
                completion: manifest_consensus_completion_policy(),
            },
            manifest,
        )
        .map_err(|error| {
            Error::Parallel(format!("communication manifest consensus failed: {error}"))
        })?;
        // Every rank must finish the complete opaque-manifest exchange before
        // any rank-local backend capability, quarantine, or world-identity
        // check can diverge. A corrupt projection therefore has one shared
        // fail-closed result and cannot strand peers in setup or payload work.
        validate_mlx_communication_manifest(manifest)?;
        let completion = manifest.completion_policy().ok_or_else(|| {
            Error::Parallel(
                "communication manifest requires an explicit bounded completion policy".into(),
            )
        })?;
        let control = Group::uncontracted(world);
        crate::backend::runtime::distributed::completion::ensure_group_available(&control)?;
        let world_collective_waves = world_collective_wave_proofs(&agreed_manifests)?;
        validate_logical_variable_all_to_all_waves(manifest, &world_collective_waves)?;
        let prepared = PreparedCommunicationManifest::new(manifest)?;
        let route_world_collective_waves =
            route_world_collective_wave_proofs(manifest.world_size(), &prepared.routes);
        Self::new_with_routes(
            prepared.groups,
            prepared.routes,
            world,
            Some(completion),
            Some(world_collective_waves),
            Some(route_world_collective_waves),
        )
    }

    fn new_with_routes(
        realization: CollectiveRealization,
        routes: Vec<CommunicationRouteDescriptor>,
        world: &NativeGroup,
        completion: Option<eredu_runtime::CommunicationCompletionPolicy>,
        world_collective_waves: Option<Vec<bool>>,
        route_world_collective_waves: Option<Vec<bool>>,
    ) -> Result<Self, Error> {
        // Fence both manifest and uncontracted construction while any timed-out
        // work on this exact native communicator remains quarantined.
        let owned_world = completion.map_or_else(
            || Group::uncontracted(world),
            |policy| Group::uncontracted(world).with_completion_policy(policy),
        );
        crate::backend::runtime::distributed::completion::ensure_group_available(&owned_world)?;
        let _setup = owned_world.begin_bounded_setup()?;
        if world.rank() != realization.global_rank || world.size() != realization.world_size {
            return Err(Error::Parallel(format!(
                "collective realization expects world rank {}/{} but received {}/{}",
                realization.global_rank,
                realization.world_size,
                world.rank(),
                world.size()
            )));
        }
        // This unsplit handle is intentionally uncontracted: it is the control
        // plane from which exact manifest handles are realized.
        let world = owned_world;
        let mut groups = HashMap::new();
        for (order, group) in realization.groups.iter().cloned().enumerate() {
            let id = group.id;
            groups.insert(
                id,
                Self::materialize(
                    group,
                    realization.world_size,
                    &world,
                    world_collective_waves
                        .as_ref()
                        .and_then(|proofs| proofs.get(order))
                        .copied()
                        .unwrap_or(false),
                )?,
            );
        }
        let routes = routes
            .into_iter()
            .enumerate()
            .map(|(order, descriptor)| {
                let route = CommunicationRouteRealization::from_descriptor(
                    &descriptor,
                    &world,
                    route_world_collective_waves
                        .as_ref()
                        .and_then(|proofs| proofs.get(order))
                        .copied()
                        .unwrap_or(false),
                )?;
                Ok((descriptor.id(), route))
            })
            .collect::<Result<HashMap<_, _>, Error>>()?;
        Ok(Self {
            realization,
            control_world: world,
            groups,
            routes,
        })
    }

    fn materialize(
        realization: CollectiveGroupRealization,
        world_size: usize,
        world: &Group,
        world_collective_wave: bool,
    ) -> Result<GroupCommunicator, Error> {
        #[cfg(test)]
        MANIFEST_GROUP_REALIZATIONS.fetch_add(1, Ordering::Relaxed);
        let size = realization.members.len();
        let native = if size == world_size {
            world.clone()
        } else if realization.manifest_descriptor.is_some() {
            match realization.logical_routes.clone() {
                Some(routes) => world
                    .logical_subgroup_with_routes(&realization.members, routes)
                    .map(|group| group.with_world_collective_wave(world_collective_wave))
                    .map_err(|error| {
                        Error::Parallel(format!(
                            "failed to materialize routed logical group {} with members {:?}: {error}",
                            realization.id.value(), realization.members
                        ))
                    })?,
                None => world
                    .logical_subgroup(&realization.members)
                    .map(|group| group.with_world_collective_wave(world_collective_wave))
                    .map_err(|error| {
                        Error::Parallel(format!(
                            "failed to materialize logical group {} with members {:?}: {error}",
                            realization.id.value(), realization.members
                        ))
                    })?,
            }
        } else {
            let color = i32::try_from(realization.split_color)
                .map_err(|_| Error::Parallel("collective group split color exceeds i32".into()))?;
            let key = i32::try_from(realization.local_rank)
                .map_err(|_| Error::Parallel("collective group local rank exceeds i32".into()))?;
            let group = match world.split(color, Some(key)) {
                Ok(group) => group,
                Err(_) => match realization.logical_routes.clone() {
                    Some(routes) => world
                        .logical_subgroup_with_routes(&realization.members, routes)
                        .map_err(|error| {
                            Error::Parallel(format!(
                                "failed to materialize routed logical group {} with members {:?}: {error}",
                                realization.id.value(), realization.members
                            ))
                        })?,
                    None => world
                        .logical_subgroup(&realization.members)
                        .map_err(|error| {
                            Error::Parallel(format!(
                                "failed to materialize native or logical group {} with members {:?}: {error}",
                                realization.id.value(), realization.members
                            ))
                        })?,
                },
            };
            if group.rank() != realization.local_rank || group.size() != size {
                return Err(Error::Parallel(format!(
                    "collective group {} expected rank {}/{} but backend produced {}/{}",
                    realization.id.value(),
                    realization.local_rank,
                    size,
                    group.rank(),
                    group.size()
                )));
            }
            group
        };
        let native = match &realization.manifest_descriptor {
            Some(descriptor) => native
                .with_manifest_contract(
                    descriptor,
                    world.completion_policy().ok_or_else(|| {
                        Error::Parallel("manifest group has no selected completion policy".into())
                    })?,
                )
                .map_err(|error| Error::Parallel(error.to_string()))?,
            None => native,
        };
        Ok(GroupCommunicator {
            realization,
            native: Some(native),
        })
    }

    /// Returns the unsplit world handle retained only for portable metadata consensus.
    pub(crate) const fn control_world(&self) -> &Group {
        &self.control_world
    }

    /// Returns the group for an active opaque collective identity.
    pub fn group(&self, id: CollectiveGroupId) -> Option<&Group> {
        let communicator = self.groups.get(&id)?;
        if communicator.realization.members.len() == 1 {
            return None;
        }
        communicator.native.as_ref()
    }

    /// Returns the opaque mechanism handle, including singleton identities.
    pub fn communication_group(&self, id: CollectiveGroupId) -> Option<&Group> {
        let communicator = self.groups.get(&id)?;
        communicator.native.as_ref()
    }

    /// Consumes manifest-realized communicators into the neutral runtime's exact resource order.
    pub(crate) fn into_partition_resources(
        self,
        manifest: &CommunicationManifest,
    ) -> Result<
        (
            Vec<eredu_runtime::RealizedCommunicationGroup<Group>>,
            Vec<eredu_runtime::RealizedCommunicationRoute<CommunicationRouteRealization>>,
        ),
        Error,
    > {
        let Self {
            realization: _,
            control_world: _,
            mut groups,
            mut routes,
        } = self;
        let realized_groups = manifest
            .groups()
            .iter()
            .map(|descriptor| {
                let communicator = groups.remove(&descriptor.id()).ok_or_else(|| {
                    Error::Parallel(format!(
                        "manifest group {} was not realized",
                        descriptor.id().value()
                    ))
                })?;
                let group = communicator.native.ok_or_else(|| {
                    Error::Parallel(format!(
                        "manifest group {} has no native or logical resource",
                        descriptor.id().value()
                    ))
                })?;
                Ok(eredu_runtime::RealizedCommunicationGroup::new(
                    descriptor.id(),
                    group,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let realized_routes = manifest
            .routes()
            .iter()
            .map(|descriptor| {
                routes
                    .remove(&descriptor.id())
                    .map(|route| {
                        eredu_runtime::RealizedCommunicationRoute::new(descriptor.id(), route)
                    })
                    .ok_or_else(|| {
                        Error::Parallel(format!(
                            "manifest route {} was not realized",
                            descriptor.id().value()
                        ))
                    })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        if !groups.is_empty() || !routes.is_empty() {
            return Err(Error::Parallel(
                "realized communication contains resources outside the selected manifest".into(),
            ));
        }
        Ok((realized_groups, realized_routes))
    }

    /// Returns one opaque point-to-point route selected by neutral composition.
    pub fn route(&self, id: CommunicationRouteId) -> Option<&CommunicationRouteRealization> {
        self.routes.get(&id)
    }

    pub(crate) fn descriptors(&self) -> Vec<CollectiveGroupDescriptor> {
        self.realization
            .groups
            .iter()
            .filter(|group| group.members.len() > 1)
            .map(CollectiveGroupRealization::descriptor)
            .collect()
    }

    pub(crate) fn group_ids(&self) -> Vec<CollectiveGroupId> {
        self.realization
            .groups
            .iter()
            .filter(|group| group.members.len() > 1)
            .map(|group| group.id)
            .collect()
    }

    pub(crate) const fn realization(&self) -> &CollectiveRealization {
        &self.realization
    }
}

/// A validated contiguous slice of a source tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TensorSlice {
    /// Source tensor axis being divided.
    axis: usize,
    /// Inclusive element offset on `axis`.
    start: usize,
    /// Exclusive element offset on `axis`.
    end: usize,
    /// Shard index.
    index: usize,
    /// Total number of equal shards.
    parts: usize,
}

impl TensorSlice {
    /// Returns the source tensor axis being divided.
    pub const fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the inclusive element offset on the divided axis.
    pub const fn start(&self) -> usize {
        self.start
    }

    /// Returns the exclusive element offset on the divided axis.
    pub const fn end(&self) -> usize {
        self.end
    }

    /// Returns the zero-based shard index.
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Returns the total number of equal shards.
    pub const fn parts(&self) -> usize {
        self.parts
    }

    /// Validates and calculates an equal contiguous tensor slice.
    pub fn for_shape(
        shape: &[usize],
        axis: usize,
        index: usize,
        parts: usize,
    ) -> Result<Self, Error> {
        if axis >= shape.len() {
            return Err(Error::Parallel(format!(
                "tensor axis {axis} is outside rank {} shape {shape:?}",
                shape.len()
            )));
        }
        if parts == 0 {
            return Err(Error::Parallel("tensor shard count must be nonzero".into()));
        }
        if index >= parts {
            return Err(Error::Parallel(format!(
                "tensor shard index {index} is outside {parts} parts"
            )));
        }
        let dimension = shape[axis];
        if dimension == 0 || !dimension.is_multiple_of(parts) {
            return Err(Error::Parallel(format!(
                "tensor dimension {dimension} on axis {axis} is not nonzero and divisible by {parts}"
            )));
        }
        let width = dimension / parts;
        let start = index
            .checked_mul(width)
            .ok_or_else(|| Error::Parallel("tensor slice offset overflowed usize".into()))?;
        Ok(Self {
            axis,
            start,
            end: start + width,
            index,
            parts,
        })
    }

    /// Returns the local tensor shape produced by this slice.
    pub fn local_shape(&self, source_shape: &[usize]) -> Vec<usize> {
        let mut shape = source_shape.to_vec();
        shape[self.axis] = self.end - self.start;
        shape
    }
}

#[derive(Debug, Clone)]
struct TensorPlan {
    placement: TensorPlacement,
    expected_source_shape: Option<Vec<usize>>,
}

/// Inspectable mapping from exact checkpoint names to typed placement decisions.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    topology: MlxRankContext,
    tensors: HashMap<String, TensorPlan>,
    default: Option<TensorPlacement>,
}

impl PlacementPlan {
    /// Creates a strict plan in which every checkpoint tensor must be named.
    pub fn new(topology: MlxRankContext) -> Self {
        Self {
            topology,
            tensors: HashMap::new(),
            default: None,
        }
    }

    /// Creates a plan that replicates every checkpoint tensor.
    pub fn replicated(topology: MlxRankContext) -> Self {
        Self::new(topology).with_default(TensorPlacement::Replicated)
    }

    /// Sets the placement used for checkpoint keys without an explicit entry.
    pub fn with_default(mut self, placement: TensorPlacement) -> Self {
        self.default = Some(placement);
        self
    }

    /// Returns the topology captured by this plan.
    pub const fn topology(&self) -> MlxRankContext {
        self.topology
    }

    /// Adds or replaces a checkpoint-tensor placement.
    pub fn insert(&mut self, source: impl Into<String>, placement: TensorPlacement) {
        self.tensors.insert(
            source.into(),
            TensorPlan {
                placement,
                expected_source_shape: None,
            },
        );
    }

    /// Adds a placement with a required pre-slice checkpoint shape.
    pub fn insert_expected(
        &mut self,
        source: impl Into<String>,
        expected_source_shape: impl Into<Vec<usize>>,
        placement: TensorPlacement,
    ) -> Result<(), Error> {
        let expected_source_shape = expected_source_shape.into();
        validate_placement(&placement, &expected_source_shape, self.topology)?;
        self.tensors.insert(
            source.into(),
            TensorPlan {
                placement,
                expected_source_shape: Some(expected_source_shape),
            },
        );
        Ok(())
    }

    /// Adds weight, scales, and optional biases using one logical placement.
    ///
    /// Keeping companions in one call prevents a quantized module's metadata
    /// from being accidentally placed differently from its packed weight.
    pub fn insert_quantized_companions(
        &mut self,
        prefix: &str,
        placement: TensorPlacement,
        has_biases: bool,
    ) {
        self.insert(format!("{prefix}.weight"), placement.clone());
        self.insert(format!("{prefix}.scales"), placement.clone());
        if has_biases {
            self.insert(format!("{prefix}.biases"), placement);
        }
    }

    /// Returns an explicit tensor placement by exact checkpoint name.
    pub fn placement(&self, source: &str) -> Option<&TensorPlacement> {
        self.tensors.get(source).map(|plan| &plan.placement)
    }

    /// Validates every placement whose constraints are known before loading.
    ///
    /// Axis bounds and divisibility require `insert_expected`; ownership and
    /// shard-coordinate bounds are validated for all entries.
    pub fn validate(&self) -> Result<(), Error> {
        for (source, tensor) in &self.tensors {
            validate_plan_entry(tensor, self.topology).map_err(|error| {
                Error::Parallel(format!("placement for tensor {source}: {error}"))
            })?;
        }
        if let Some(default) = &self.default {
            validate_plan_entry(
                &TensorPlan {
                    placement: default.clone(),
                    expected_source_shape: None,
                },
                self.topology,
            )?;
        }
        Ok(())
    }

    fn source_plan(&self, source: &str) -> SourcePlan {
        if let Some(plan) = self.tensors.get(source) {
            return SourcePlan::Known(plan.clone());
        }
        if let Some(placement) = &self.default {
            SourcePlan::Known(TensorPlan {
                placement: placement.clone(),
                expected_source_shape: None,
            })
        } else {
            SourcePlan::Unexpected
        }
    }
}

fn validate_plan_entry(plan: &TensorPlan, topology: MlxRankContext) -> Result<(), Error> {
    match &plan.placement {
        TensorPlacement::Rank { rank } if *rank >= topology.world_size() => {
            Err(Error::Parallel(format!(
                "owner rank {rank} is outside world size {}",
                topology.world_size()
            )))
        }
        TensorPlacement::Shard { index, parts, .. } if *parts == 0 || *index >= *parts => {
            Err(Error::Parallel(format!(
                "tensor shard index {index} is invalid for {parts} parts"
            )))
        }
        TensorPlacement::Range { start, end, .. } if start >= end => Err(Error::Parallel(format!(
            "tensor range {start}..{end} must be non-empty"
        ))),
        TensorPlacement::Indices { indices, .. } if indices.is_empty() => Err(Error::Parallel(
            "tensor index selection must be non-empty".into(),
        )),
        TensorPlacement::Indices { indices, .. }
            if indices.iter().collect::<HashSet<_>>().len() != indices.len() =>
        {
            Err(Error::Parallel(
                "tensor index selection must not contain duplicates".into(),
            ))
        }
        placement => {
            if let Some(shape) = &plan.expected_source_shape {
                validate_placement(placement, shape, topology)?;
            }
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
enum SourcePlan {
    Known(TensorPlan),
    Unexpected,
}

#[derive(Debug)]
enum ResolvedPlacement {
    Materialize,
    Omit,
    Shard(TensorSlice),
    Indices { axis: usize, indices: Vec<usize> },
}

fn validate_placement(
    placement: &TensorPlacement,
    shape: &[usize],
    topology: MlxRankContext,
) -> Result<(), Error> {
    match placement {
        TensorPlacement::Rank { rank } if *rank >= topology.world_size() => {
            Err(Error::Parallel(format!(
                "owner rank {rank} is outside world size {}",
                topology.world_size()
            )))
        }
        TensorPlacement::Shard { axis, index, parts } => {
            TensorSlice::for_shape(shape, *axis, *index, *parts).map(|_| ())
        }
        TensorPlacement::Range { axis, start, end } => {
            if *axis >= shape.len() {
                return Err(Error::Parallel(format!(
                    "tensor range axis {axis} is outside rank {} shape {shape:?}",
                    shape.len()
                )));
            }
            if start >= end || *end > shape[*axis] {
                return Err(Error::Parallel(format!(
                    "tensor range {start}..{end} is invalid for dimension {} on axis {axis}",
                    shape[*axis]
                )));
            }
            Ok(())
        }
        TensorPlacement::Indices { axis, indices } => {
            if *axis >= shape.len() {
                return Err(Error::Parallel(format!(
                    "tensor index axis {axis} is outside rank {} shape {shape:?}",
                    shape.len()
                )));
            }
            if indices.is_empty() {
                return Err(Error::Parallel(
                    "tensor index selection must be non-empty".into(),
                ));
            }
            if indices.iter().collect::<HashSet<_>>().len() != indices.len() {
                return Err(Error::Parallel(
                    "tensor index selection must not contain duplicates".into(),
                ));
            }
            if let Some(index) = indices.iter().copied().find(|index| *index >= shape[*axis]) {
                return Err(Error::Parallel(format!(
                    "tensor index {index} is outside dimension {} on axis {axis}",
                    shape[*axis]
                )));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn resolve_placement(
    plan: &TensorPlan,
    shape: &[usize],
    topology: MlxRankContext,
) -> Result<ResolvedPlacement, Error> {
    if let Some(expected) = &plan.expected_source_shape {
        if expected != shape {
            return Err(Error::Parallel(format!(
                "expected checkpoint shape {expected:?}, got {shape:?}"
            )));
        }
    }
    validate_placement(&plan.placement, shape, topology)?;
    Ok(match &plan.placement {
        TensorPlacement::Replicated | TensorPlacement::Local => ResolvedPlacement::Materialize,
        TensorPlacement::Omit => ResolvedPlacement::Omit,
        TensorPlacement::Rank { rank } => {
            if *rank == topology.global_rank() {
                ResolvedPlacement::Materialize
            } else {
                ResolvedPlacement::Omit
            }
        }
        TensorPlacement::Shard { axis, index, parts } => {
            ResolvedPlacement::Shard(TensorSlice::for_shape(shape, *axis, *index, *parts)?)
        }
        TensorPlacement::Range { axis, start, end } => ResolvedPlacement::Shard(TensorSlice {
            axis: *axis,
            start: *start,
            end: *end,
            index: 0,
            parts: 1,
        }),
        TensorPlacement::Indices { axis, indices } => ResolvedPlacement::Indices {
            axis: *axis,
            indices: indices.clone(),
        },
    })
}

/// Locally materialized checkpoint partition.
///
/// This is intentionally not an executable model. Later distributed execution
/// phases can consume it together with a communication group without storing a
/// borrowed group inside long-lived model state.
#[derive(Debug)]
pub struct RankPartition {
    topology: MlxRankContext,
    tensors: HashMap<String, Array>,
    opened_shards: Vec<PathBuf>,
}

impl RankPartition {
    /// Returns the validated topology used for this partition.
    pub const fn topology(&self) -> MlxRankContext {
        self.topology
    }

    /// Returns a locally materialized tensor by exact checkpoint name.
    pub fn get(&self, source: &str) -> Option<&Array> {
        self.tensors.get(source)
    }

    /// Iterates over locally materialized tensors.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Array)> {
        self.tensors
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    /// Returns the number of locally materialized tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns whether this partition contains no local tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns checkpoint payload shards that were actually opened.
    pub fn opened_shards(&self) -> &[PathBuf] {
        &self.opened_shards
    }
}

#[derive(Default)]
struct PartitionReport {
    loaded: HashSet<String>,
    unexpected: Vec<String>,
}

impl PartitionReport {
    fn finish(self, plan: &PlacementPlan) -> Result<(), Error> {
        let mut missing = Vec::new();
        for (source, tensor) in &plan.tensors {
            let locally_required = match tensor.placement {
                TensorPlacement::Replicated
                | TensorPlacement::Local
                | TensorPlacement::Shard { .. }
                | TensorPlacement::Range { .. }
                | TensorPlacement::Indices { .. } => true,
                TensorPlacement::Omit => false,
                TensorPlacement::Rank { rank } => rank == plan.topology.global_rank(),
            };
            if locally_required && !self.loaded.contains(source) {
                missing.push(source.clone());
            }
        }
        missing.sort();
        let mut unexpected = self.unexpected;
        unexpected.sort();
        unexpected.dedup();
        if missing.is_empty() && unexpected.is_empty() {
            Ok(())
        } else {
            Err(Error::StrictLoadValidation {
                missing,
                unused: unexpected,
            })
        }
    }
}

/// Selectively loads a safetensors checkpoint directory according to `plan`.
///
/// For indexed checkpoints, exact-name placement is resolved from the
/// index before any payload shard is opened. A shard containing no local
/// tensors is therefore skipped completely. Every opened shard header must
/// exactly match all index entries assigned to that shard, while omitted
/// tensors never become MLX arrays. Selected source views are sliced before
/// their final stream copy, then explicitly evaluated while the mmap is alive.
/// Peak temporary memory is bounded by the accumulated local partition plus at
/// most the selected source tensor currently being transformed.
pub fn load_safetensors_partition(
    model_dir: impl AsRef<Path>,
    plan: &PlacementPlan,
    stream: &Stream,
) -> Result<RankPartition, Error> {
    load_safetensors_partition_on_streams(model_dir, plan, stream, stream)
}

/// Selectively loads on a source/weights stream, then places only local results
/// on `execution_stream`.
///
/// Use a CPU `source_stream` with a GPU `execution_stream` to ensure a full
/// source tensor is never copied to the GPU merely to discard other ranks'
/// slices. The source device holds at most the tensor currently being
/// transformed in addition to the accumulated local partition.
pub fn load_safetensors_partition_on_streams(
    model_dir: impl AsRef<Path>,
    plan: &PlacementPlan,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<RankPartition, Error> {
    let store = SafetensorsWeightStore::open(model_dir)?;
    load_partition_from_store_on_streams(&store, plan, source_stream, execution_stream)
}

/// Selectively loads a rank partition from a reusable checkpoint store using
/// explicit source and execution streams.
///
/// Placement is resolved from catalog metadata before a lease materializes an
/// array. Remote-only indexed shards are therefore never acquired or buffered.
pub fn load_partition_from_store_on_streams(
    store: &(impl CheckpointSource + ?Sized),
    plan: &PlacementPlan,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<RankPartition, Error> {
    plan.validate()?;
    plan.topology.validate_execution_stream(execution_stream)?;
    let mut report = PartitionReport::default();
    let mut tensors = HashMap::new();
    let mut opened_shards = BTreeSet::new();
    let context = MlxParameterMaterializationContext::new(source_stream, execution_stream);

    for source in store.source_keys() {
        let SourcePlan::Known(tensor) = plan.source_plan(&source) else {
            report.unexpected.push(source);
            continue;
        };
        let potentially_local = !matches!(tensor.placement, TensorPlacement::Omit)
            && !matches!(tensor.placement, TensorPlacement::Rank { rank } if rank != plan.topology.global_rank());
        if !potentially_local {
            continue;
        }

        let metadata = store.source_metadata(&source)?;
        let resolved = resolve_placement(&tensor, &metadata.logical_shape, plan.topology)
            .map_err(|error| Error::Parallel(format!("checkpoint tensor {source}: {error}")))?;
        let selection = match resolved {
            ResolvedPlacement::Omit => continue,
            ResolvedPlacement::Materialize => TensorSelection::Full,
            ResolvedPlacement::Shard(slice) => TensorSelection::Range {
                axis: slice.axis,
                start: slice.start,
                end: slice.end,
            },
            ResolvedPlacement::Indices { axis, indices } => {
                TensorSelection::Indices { axis, indices }
            }
        };
        let lease = store.acquire_lease(TensorReadRequest {
            key: source.clone(),
            selection,
            policy: ReadPolicy::RequireBounded,
        })?;
        if let Some(path) = eredu_checkpoint::store::EncodedTensorLease::backing_path(&lease) {
            opened_shards.insert(path.to_path_buf());
        }
        let value = context
            .weight_lease(lease)?
            .materialize(source_stream, execution_stream)?
            .synchronize()?;
        report.loaded.insert(source.clone());
        if tensors.insert(source.clone(), value).is_some() {
            return Err(Error::Parallel(format!(
                "checkpoint tensor {source} was materialized more than once"
            )));
        }
    }

    report.finish(plan)?;
    Ok(RankPartition {
        topology: plan.topology,
        tensors,
        opened_shards: opened_shards.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::ParallelTopology;
    use eredu_runtime::{
        project_all_communication_manifests, CommunicationGroupRequirements,
        TopologyCommunicationPlan,
    };
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    fn completion_policy() -> eredu_runtime::CommunicationCompletionPolicy {
        eredu_runtime::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(30),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap()
    }

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn write_index(dir: &Path, mappings: &[(&str, &str)]) {
        let weight_map = mappings
            .iter()
            .map(|(key, file)| ((*key).to_string(), serde_json::json!(file)))
            .collect::<serde_json::Map<_, _>>();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {},
                "weight_map": weight_map,
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn write_i32_tensor(path: &Path, name: &str, values: &[i32], shape: Vec<usize>) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::I32, shape, &bytes).unwrap();
        serialize_to_file([(name, view)], None, path).unwrap();
    }

    fn topology(rank: usize, tp: usize, pp: usize, ep: usize) -> MlxRankContext {
        let topology = crate::test_parallel_rank(rank, tp, pp, ep);
        MlxRankContext::new(
            topology.world_size(),
            topology.global_rank(),
            DeviceAssignment::new(DeviceType::Cpu, 0),
        )
        .unwrap()
    }

    fn communication_requirement(
        operation: CommunicationOperation,
    ) -> CommunicationOperationRequirement {
        CommunicationOperationRequirement::tensors(
            operation,
            [TensorDtype::F32, TensorDtype::Bf16],
            CommunicationTensorLimits::new(
                1,
                4,
                16_384,
                (operation == CommunicationOperation::VariableAllToAll).then_some(4096),
            )
            .unwrap(),
            true,
        )
        .unwrap()
    }

    fn communication_group_requirements(
        operation: CommunicationOperation,
    ) -> CommunicationGroupRequirements {
        CommunicationGroupRequirements::new([communication_requirement(operation)]).unwrap()
    }

    #[test]
    fn complete_subgroup_batches_are_the_only_world_collective_waves() {
        let topology = ParallelTopology::new(4, 2, 2, 1).unwrap();
        let plan = TopologyCommunicationPlan::new()
            .with_tensor_groups(communication_group_requirements(
                CommunicationOperation::AllReduceSum,
            ))
            .with_pipeline_groups(communication_group_requirements(
                CommunicationOperation::AllGatherEven,
            ));
        let manifests = project_all_communication_manifests(topology, &plan).unwrap();
        assert_eq!(
            world_collective_wave_proofs(&manifests).unwrap(),
            [true, true]
        );

        let asymmetric = [
            CommunicationManifest::new(
                2,
                0,
                vec![CommunicationGroupDescriptor::new(
                    CollectiveGroupId::new(1),
                    0,
                    vec![0, 1],
                    Some(0),
                    communication_group_requirements(CommunicationOperation::AllReduceSum),
                )
                .unwrap()],
                vec![],
            )
            .unwrap(),
            CommunicationManifest::new(
                2,
                1,
                vec![CommunicationGroupDescriptor::new(
                    CollectiveGroupId::new(1),
                    0,
                    vec![0, 1],
                    Some(1),
                    communication_group_requirements(CommunicationOperation::AllGatherEven),
                )
                .unwrap()],
                vec![],
            )
            .unwrap(),
        ];
        assert_eq!(world_collective_wave_proofs(&asymmetric).unwrap(), [false]);

        let inconsistent_overlap = (0..4)
            .map(|rank| {
                let (id, members) = match rank {
                    0 | 1 => (CollectiveGroupId::new(1), vec![0, 1]),
                    2 => (CollectiveGroupId::new(2), vec![1, 2]),
                    3 => (CollectiveGroupId::new(3), vec![2, 3]),
                    _ => unreachable!(),
                };
                let local = members.iter().position(|member| *member == rank).unwrap();
                CommunicationManifest::new(
                    4,
                    rank,
                    vec![CommunicationGroupDescriptor::new(
                        id,
                        0,
                        members,
                        Some(local),
                        communication_group_requirements(CommunicationOperation::VariableAllToAll),
                    )
                    .unwrap()],
                    vec![],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            world_collective_wave_proofs(&inconsistent_overlap).unwrap(),
            [false],
            "same-contract descriptors that do not form consistent disjoint classes cannot share a native world wave"
        );
        super::super::group::reset_native_collective_submissions();
        let error = validate_logical_variable_all_to_all_waves(
            &inconsistent_overlap[0],
            &world_collective_wave_proofs(&inconsistent_overlap).unwrap(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("without a consensus-proven world participation wave"));
        assert_eq!(
            super::super::group::native_collective_submissions(),
            0,
            "an unproven logical exchange must fail before native payload submission"
        );

        let topology = ParallelTopology::new(2, 1, 2, 1).unwrap();
        let plan = TopologyCommunicationPlan::new()
            .with_tensor_groups(communication_group_requirements(
                CommunicationOperation::AllReduceSum,
            ))
            .with_expert_groups(communication_group_requirements(
                CommunicationOperation::VariableAllToAll,
            ));
        let compound = project_all_communication_manifests(topology, &plan).unwrap();
        let proofs = world_collective_wave_proofs(&compound).unwrap();
        assert_eq!(proofs, [true, true]);
        for manifest in &compound {
            validate_logical_variable_all_to_all_waves(manifest, &proofs).unwrap();
        }
    }

    #[test]
    fn route_wave_requires_identical_disjoint_pairs_covering_the_world() {
        let route = |id, source, destination, operation| {
            CommunicationRouteDescriptor::new(
                CommunicationRouteId::new(id),
                id as usize,
                source,
                destination,
                communication_requirement(operation),
            )
            .unwrap()
        };
        let complete = [
            route(1, 0, 2, CommunicationOperation::SendReceive),
            route(2, 1, 3, CommunicationOperation::SendReceive),
        ];
        assert_eq!(
            route_world_collective_wave_proofs(4, &complete),
            [true, true]
        );

        let partial = [route(1, 0, 2, CommunicationOperation::SendReceive)];
        assert_eq!(route_world_collective_wave_proofs(4, &partial), [false]);

        let overlapping = [
            route(1, 0, 2, CommunicationOperation::SendReceive),
            route(2, 0, 3, CommunicationOperation::SendReceive),
        ];
        assert_eq!(
            route_world_collective_wave_proofs(4, &overlapping),
            [false, false]
        );

        let incompatible = [
            route(1, 0, 2, CommunicationOperation::SendReceive),
            CommunicationRouteDescriptor::new(
                CommunicationRouteId::new(2),
                2,
                1,
                3,
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::SendReceive,
                    [TensorDtype::I32],
                    CommunicationTensorLimits::new(1, 1, 8, None).unwrap(),
                    true,
                )
                .unwrap(),
            )
            .unwrap(),
        ];
        assert_eq!(
            route_world_collective_wave_proofs(4, &incompatible),
            [false, false]
        );
    }

    #[test]
    fn manifest_preparation_preserves_opaque_groups_and_routes() {
        let manifest = CommunicationManifest::new(
            6,
            2,
            vec![
                CommunicationGroupDescriptor::new(
                    CollectiveGroupId::new(7),
                    0,
                    vec![0, 2, 4],
                    Some(1),
                    communication_group_requirements(CommunicationOperation::AllReduceSum),
                )
                .unwrap(),
                CommunicationGroupDescriptor::new(
                    CollectiveGroupId::new(11),
                    1,
                    vec![2, 3],
                    Some(0),
                    communication_group_requirements(CommunicationOperation::AllGatherEven),
                )
                .unwrap(),
            ],
            vec![CommunicationRouteDescriptor::new(
                CommunicationRouteId::new(19),
                0,
                2,
                5,
                communication_requirement(CommunicationOperation::SendReceive),
            )
            .unwrap()],
        )
        .unwrap();

        let prepared = PreparedCommunicationManifest::new(&manifest).unwrap();
        assert_eq!(prepared.groups.world_size, 6);
        assert_eq!(prepared.groups.global_rank, 2);
        assert_eq!(prepared.groups.groups.len(), 2);
        assert_eq!(prepared.groups.groups[0].id, CollectiveGroupId::new(7));
        assert_eq!(prepared.groups.groups[0].members, [0, 2, 4]);
        assert_eq!(prepared.groups.groups[0].local_rank, 1);
        assert_eq!(prepared.groups.groups[0].split_color, 7);
        assert!(prepared.groups.groups[0].logical_routes.is_none());
        assert_eq!(prepared.groups.groups[1].id, CollectiveGroupId::new(11));
        assert_eq!(prepared.groups.groups[1].split_color, 11);
        assert_eq!(prepared.routes.len(), 1);
        assert_eq!(prepared.routes[0].id(), CommunicationRouteId::new(19));
        assert_eq!(prepared.routes[0].source(), 2);
        assert_eq!(prepared.routes[0].destination(), 5);
    }

    #[test]
    fn manifest_constructor_preserves_singleton_group_mechanics() {
        let all_reduce_requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 1, 4, None).unwrap(),
            true,
        )
        .unwrap();
        let gather_requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllGatherEven,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 1, 4, None).unwrap(),
            true,
        )
        .unwrap();
        let variable_requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::VariableAllToAll,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 1, 4, Some(1)).unwrap(),
            true,
        )
        .unwrap();
        let inexact_requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 1, 4, None).unwrap(),
            false,
        )
        .unwrap();
        let manifest = CommunicationManifest::new(
            1,
            0,
            vec![
                CommunicationGroupDescriptor::new(
                    CollectiveGroupId::new(23),
                    0,
                    vec![0],
                    Some(0),
                    CommunicationGroupRequirements::new([all_reduce_requirement]).unwrap(),
                )
                .unwrap(),
                CommunicationGroupDescriptor::new(
                    CollectiveGroupId::new(29),
                    1,
                    vec![0],
                    Some(0),
                    CommunicationGroupRequirements::new([gather_requirement]).unwrap(),
                )
                .unwrap(),
                CommunicationGroupDescriptor::new(
                    CollectiveGroupId::new(31),
                    2,
                    vec![0],
                    Some(0),
                    CommunicationGroupRequirements::new([variable_requirement]).unwrap(),
                )
                .unwrap(),
                CommunicationGroupDescriptor::new(
                    CollectiveGroupId::new(37),
                    3,
                    vec![0],
                    Some(0),
                    CommunicationGroupRequirements::new([inexact_requirement]).unwrap(),
                )
                .unwrap(),
            ],
            vec![],
        )
        .unwrap()
        .with_completion_policy(completion_policy());
        let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();

        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let communicators =
            ParallelCommunicators::from_manifest(&manifest, &world, &stream).unwrap();
        assert!(communicators.group(CollectiveGroupId::new(23)).is_none());
        let all_reduce = communicators
            .communication_group(CollectiveGroupId::new(23))
            .unwrap();
        let gather = communicators
            .communication_group(CollectiveGroupId::new(29))
            .unwrap();
        let variable = communicators
            .communication_group(CollectiveGroupId::new(31))
            .unwrap();
        let inexact = communicators
            .communication_group(CollectiveGroupId::new(37))
            .unwrap();
        assert_eq!(all_reduce.opaque_id(), Some(CollectiveGroupId::new(23)));
        assert_eq!(gather.opaque_id(), Some(CollectiveGroupId::new(29)));
        assert_eq!(
            all_reduce.native_group().size(),
            gather.native_group().size()
        );

        super::super::group::reset_native_collective_submissions();
        let bf16 = Array::from_slice(&[1.0_f32], &[1])
            .as_dtype(safemlx::Dtype::Bfloat16, &stream)
            .unwrap();
        let error = super::super::group::all_sum(&bf16, all_reduce, &stream)
            .expect_err("F32-only manifest must reject BF16 before native submission");
        assert!(error.what().contains("does not admit dtype"));
        assert_eq!(super::super::group::native_collective_submissions(), 0);

        let input = Array::from_slice(&[1.0_f32], &[1]);
        let error = super::super::group::all_sum(&input, gather, &stream)
            .expect_err("one opaque ID must not borrow another ID's operation contract");
        assert!(error.what().contains("does not select operation"));
        assert_eq!(super::super::group::native_collective_submissions(), 0);

        let oversized = Array::from_slice(&[0.0_f32; 5], &[5]);
        let error = super::super::group::all_sum(&oversized, all_reduce, &stream)
            .expect_err("manifest tensor limit must reject before native submission");
        assert!(error.what().contains("rejects tensor shape"));
        assert_eq!(super::super::group::native_collective_submissions(), 0);

        let variable_input = Array::from_slice(&[1.0_f32, 2.0], &[2]);
        let error =
            super::super::group::all_to_all_v(&variable_input, &[2], &[2], variable, &stream)
                .expect_err("per-peer contract must reject before native submission");
        assert!(error.what().contains("peer counts exceed"));
        assert_eq!(super::super::group::native_collective_submissions(), 0);

        let error = super::super::group::all_sum(&input, inexact, &stream)
            .expect_err("inexact manifest operation must not reach exact native entry point");
        assert!(error.what().contains("selects inexact operation"));
        assert_eq!(super::super::group::native_collective_submissions(), 0);

        let output = super::super::group::all_sum(&input, all_reduce, &stream).unwrap();
        assert_eq!(super::super::group::native_collective_submissions(), 1);
        assert_eq!(output.evaluated().unwrap().as_slice::<f32>(), &[1.0]);
        assert!(communicators.route(CommunicationRouteId::new(1)).is_none());
    }

    #[test]
    fn route_handle_rejects_endpoints_outside_its_owned_world() {
        let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
        assert_eq!(world.size(), 1);
        let world = Group::uncontracted(&world);
        let descriptor = CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(29),
            0,
            0,
            1,
            communication_requirement(CommunicationOperation::SendReceive),
        )
        .unwrap();
        let error = CommunicationRouteRealization::from_descriptor(&descriptor, &world, false)
            .expect_err("route must be bound to a world containing both endpoints");
        assert!(error.to_string().contains("owned world size 1"));
    }

    #[test]
    fn projected_manifests_prepare_one_local_group_per_creation_batch_on_every_rank() {
        let topology = ParallelTopology::new(2, 2, 2, 1).unwrap();
        let plan = TopologyCommunicationPlan::new()
            .with_tensor_groups(communication_group_requirements(
                CommunicationOperation::AllReduceSum,
            ))
            .with_pipeline_groups(communication_group_requirements(
                CommunicationOperation::AllGatherEven,
            ))
            .with_expert_groups(communication_group_requirements(
                CommunicationOperation::VariableAllToAll,
            ));
        let manifests = project_all_communication_manifests(topology, &plan).unwrap();

        for manifest in &manifests {
            let prepared = PreparedCommunicationManifest::new(manifest).unwrap();
            assert_eq!(prepared.groups.groups.len(), 3);
            for (creation_order, group) in prepared.groups.groups.iter().enumerate() {
                assert_eq!(
                    manifest.groups()[creation_order].creation_order(),
                    creation_order
                );
                assert_eq!(group.id, manifest.groups()[creation_order].id());
                assert_eq!(group.split_color, group.id.value() as usize);
                assert_eq!(group.members[group.local_rank], manifest.rank());
                assert_eq!(group.members.len(), 2);
            }
        }
    }

    #[test]
    fn mlx_manifest_capabilities_admit_publication_but_fail_closed_on_unsupported_dtype() {
        let publication = CommunicationManifest::new(
            2,
            0,
            vec![CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(1),
                0,
                vec![0, 1],
                Some(0),
                CommunicationGroupRequirements::new([
                    communication_requirement(CommunicationOperation::Broadcast),
                    CommunicationOperationRequirement::barrier(true),
                    CommunicationOperationRequirement::failure_agreement(true),
                ])
                .unwrap(),
            )
            .unwrap()],
            vec![],
        )
        .unwrap()
        .with_completion_policy(completion_policy());
        validate_mlx_communication_manifest(&publication).unwrap();

        let unsupported = CommunicationManifest::new(
            1,
            0,
            vec![CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(1),
                0,
                vec![0],
                Some(0),
                CommunicationGroupRequirements::new([CommunicationOperationRequirement::tensors(
                    CommunicationOperation::Broadcast,
                    [TensorDtype::I32],
                    CommunicationTensorLimits::new(1, 1, 1, None).unwrap(),
                    true,
                )
                .unwrap()])
                .unwrap(),
            )
            .unwrap()],
            vec![],
        )
        .unwrap()
        .with_completion_policy(completion_policy());
        let error = validate_mlx_communication_manifest(&unsupported)
            .expect_err("integer broadcast must be rejected");
        assert!(error.to_string().contains("I32"));

        let zero_id = CommunicationManifest::new(
            2,
            0,
            vec![CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(0),
                0,
                vec![0, 1],
                Some(0),
                communication_group_requirements(CommunicationOperation::AllReduceSum),
            )
            .unwrap()],
            vec![],
        )
        .unwrap();
        let error = PreparedCommunicationManifest::new(&zero_id)
            .expect_err("zero group ID must be rejected");
        assert!(error.to_string().contains("ID zero is reserved"));
    }

    #[test]
    fn mlx_failure_agreement_capability_is_exact_and_distinct_from_barrier() {
        let failure_agreement = CommunicationManifest::new(
            1,
            0,
            vec![CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(3),
                0,
                vec![0],
                Some(0),
                CommunicationGroupRequirements::new([
                    CommunicationOperationRequirement::failure_agreement(true),
                ])
                .unwrap(),
            )
            .unwrap()],
            vec![],
        )
        .unwrap()
        .with_completion_policy(completion_policy());
        validate_mlx_communication_manifest(&failure_agreement).unwrap();

        let capabilities_without_agreement =
            CommunicationCapabilities::new([CommunicationOperationRequirement::barrier(true)])
                .unwrap()
                .with_completion_capabilities(
                    CommunicationCompletionCapabilities::new([
                        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                    ])
                    .unwrap(),
                );
        assert!(capabilities_without_agreement
            .validate_manifest(&failure_agreement)
            .is_err());

        let inexact_agreement =
            CommunicationCapabilities::new([CommunicationOperationRequirement::failure_agreement(
                false,
            )])
            .unwrap()
            .with_completion_capabilities(
                CommunicationCompletionCapabilities::new([
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                ])
                .unwrap(),
            );
        assert!(inexact_agreement
            .validate_manifest(&failure_agreement)
            .is_err());
    }

    #[test]
    fn mlx_capabilities_are_operation_specific_and_exclude_encoded_payloads() {
        let capabilities = mlx_communication_capabilities();
        let group_manifest = |operation, dtype| {
            let max_peer_count =
                (operation == CommunicationOperation::VariableAllToAll).then_some(8);
            CommunicationManifest::new(
                1,
                0,
                vec![CommunicationGroupDescriptor::new(
                    CollectiveGroupId::new(1),
                    0,
                    vec![0],
                    Some(0),
                    CommunicationGroupRequirements::new([
                        CommunicationOperationRequirement::tensors(
                            operation,
                            [dtype],
                            CommunicationTensorLimits::new(1, 2, 8, max_peer_count).unwrap(),
                            true,
                        )
                        .unwrap(),
                    ])
                    .unwrap(),
                )
                .unwrap()],
                vec![],
            )
            .unwrap()
            .with_completion_policy(completion_policy())
        };

        capabilities
            .validate_manifest(&group_manifest(
                CommunicationOperation::AllGatherEven,
                TensorDtype::I32,
            ))
            .unwrap();
        assert!(capabilities
            .validate_manifest(&group_manifest(
                CommunicationOperation::AllReduceSum,
                TensorDtype::I32,
            ))
            .is_err());
        assert!(capabilities
            .validate_manifest(&group_manifest(
                CommunicationOperation::AllGatherUneven,
                TensorDtype::I32,
            ))
            .is_err());
        capabilities
            .validate_manifest(&group_manifest(
                CommunicationOperation::VariableAllToAll,
                TensorDtype::I32,
            ))
            .unwrap();
        assert!(capabilities
            .validate_manifest(&group_manifest(
                CommunicationOperation::VariableAllToAll,
                TensorDtype::U32,
            ))
            .is_err());
        assert!(capabilities
            .validate_manifest(&group_manifest(
                CommunicationOperation::AllGatherEven,
                TensorDtype::U32,
            ))
            .is_err());

        let route_manifest = |dtype| {
            CommunicationManifest::new(
                2,
                0,
                vec![],
                vec![CommunicationRouteDescriptor::new(
                    CommunicationRouteId::new(1),
                    0,
                    0,
                    1,
                    CommunicationOperationRequirement::tensors(
                        CommunicationOperation::SendReceive,
                        [dtype],
                        CommunicationTensorLimits::new(1, 2, 8, None).unwrap(),
                        true,
                    )
                    .unwrap(),
                )
                .unwrap()],
            )
            .unwrap()
            .with_completion_policy(completion_policy())
        };
        for dtype in [TensorDtype::I32, TensorDtype::U32] {
            capabilities
                .validate_manifest(&route_manifest(dtype))
                .unwrap();
        }
        assert!(capabilities
            .validate_manifest(&route_manifest(TensorDtype::I64))
            .is_err());

        let requirements =
            CommunicationGroupRequirements::new([CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllReduceSum,
                [TensorDtype::Encoded("packed".into())],
                CommunicationTensorLimits::new(1, 1, 8, None).unwrap(),
                true,
            )
            .unwrap()])
            .unwrap();
        let manifest = CommunicationManifest::new(
            1,
            0,
            vec![CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(1),
                0,
                vec![0],
                Some(0),
                requirements,
            )
            .unwrap()],
            vec![],
        )
        .unwrap()
        .with_completion_policy(completion_policy());
        assert!(capabilities.validate_manifest(&manifest).is_err());

        let route_manifest = CommunicationManifest::new(
            2,
            0,
            vec![],
            vec![CommunicationRouteDescriptor::new(
                CommunicationRouteId::new(1),
                0,
                0,
                1,
                communication_requirement(CommunicationOperation::SendReceive),
            )
            .unwrap()],
        )
        .unwrap()
        .with_completion_policy(completion_policy());
        validate_mlx_communication_manifest(&route_manifest).unwrap();

        let exact_manifest = CommunicationManifest::new(
            1,
            0,
            vec![CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(1),
                0,
                vec![0],
                Some(0),
                communication_group_requirements(CommunicationOperation::AllReduceSum),
            )
            .unwrap()],
            vec![],
        )
        .unwrap()
        .with_completion_policy(completion_policy());
        validate_mlx_communication_manifest(&exact_manifest).unwrap();

        let barrier = CommunicationManifest::new(
            1,
            0,
            vec![CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(2),
                0,
                vec![0],
                Some(0),
                CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(
                    true,
                )])
                .unwrap(),
            )
            .unwrap()],
            vec![],
        )
        .unwrap()
        .with_completion_policy(completion_policy());
        validate_mlx_communication_manifest(&barrier).unwrap();
    }

    #[test]
    fn validates_tensor_slices() {
        let slice = TensorSlice::for_shape(&[4, 12], 1, 2, 3).unwrap();
        assert_eq!(slice.start(), 8);
        assert_eq!(slice.end(), 12);
        assert_eq!(slice.local_shape(&[4, 12]), [4, 4]);
        assert!(TensorSlice::for_shape(&[4, 11], 1, 0, 3).is_err());
        assert!(TensorSlice::for_shape(&[4, 12], 2, 0, 3).is_err());
        assert!(TensorSlice::for_shape(&[4, 12], 1, 3, 3).is_err());
    }

    #[test]
    fn validates_explicit_execution_stream_device() {
        let stream = stream();
        topology(0, 1, 1, 1)
            .validate_execution_stream(&stream)
            .unwrap();
        let other_assignment =
            MlxRankContext::new(1, 0, DeviceAssignment::new(DeviceType::Cpu, 1)).unwrap();
        assert!(other_assignment.validate_execution_stream(&stream).is_err());
    }

    #[test]
    fn plan_exposes_replicated_omitted_and_quantized_companions() {
        let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
        plan.insert("replicated", TensorPlacement::Replicated);
        plan.insert("remote", TensorPlacement::Omit);
        plan.insert_quantized_companions("projection", TensorPlacement::Local, true);
        assert_eq!(
            plan.placement("replicated"),
            Some(&TensorPlacement::Replicated)
        );
        assert_eq!(plan.placement("remote"), Some(&TensorPlacement::Omit));
        assert_eq!(
            plan.placement("projection.weight"),
            Some(&TensorPlacement::Local)
        );
        assert_eq!(
            plan.placement("projection.scales"),
            Some(&TensorPlacement::Local)
        );
        assert_eq!(
            plan.placement("projection.biases"),
            Some(&TensorPlacement::Local)
        );

        let mut invalid = PlacementPlan::new(topology(0, 1, 1, 1));
        invalid.insert("bad_owner", TensorPlacement::Rank { rank: 1 });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn plan_supports_balanced_uneven_ranges() {
        let mut plan = PlacementPlan::new(topology(2, 3, 1, 1));
        let range = eredu_core::balanced_contiguous_range(11, 3, 2, false).unwrap();
        plan.insert(
            "embedding.weight",
            TensorPlacement::Range {
                axis: 0,
                start: range.start,
                end: range.end,
            },
        );
        assert_eq!(range, 8..11);
        assert_eq!(
            plan.placement("embedding.weight"),
            Some(&TensorPlacement::Range {
                axis: 0,
                start: 8,
                end: 11,
            })
        );
        plan.insert_expected(
            "head.weight",
            vec![11, 4],
            TensorPlacement::Range {
                axis: 0,
                start: 8,
                end: 11,
            },
        )
        .unwrap();
        plan.validate().unwrap();
    }

    #[test]
    fn typed_rank_ownership_resolves_locally() {
        let rank_zero = topology(0, 2, 2, 1);
        let rank_three = topology(3, 2, 2, 1);
        let rank_owned = TensorPlan {
            placement: TensorPlacement::Rank { rank: 3 },
            expected_source_shape: None,
        };
        assert!(matches!(
            resolve_placement(&rank_owned, &[2], rank_zero).unwrap(),
            ResolvedPlacement::Omit
        ));
        assert!(matches!(
            resolve_placement(&rank_owned, &[2], rank_three).unwrap(),
            ResolvedPlacement::Materialize
        ));
    }

    #[test]
    fn selective_loader_skips_remote_shards_and_reconstructs_tp_slices() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("local.safetensors"),
            "model.projection.weight",
            &[0, 1, 2, 3, 10, 11, 12, 13],
            vec![2, 4],
        );
        // This is deliberately not a safetensors file. Correct index-level
        // selection must never open it for either rank.
        std::fs::write(dir.path().join("remote.safetensors"), b"must not be opened").unwrap();
        write_index(
            dir.path(),
            &[
                ("model.projection.weight", "local.safetensors"),
                ("model.remote.weight", "remote.safetensors"),
            ],
        );
        let local_shard = dir.path().join("local.safetensors").canonicalize().unwrap();

        let mut reconstructed = Vec::new();
        for rank in 0..2 {
            let topology = topology(rank, 2, 1, 1);
            let mut plan = PlacementPlan::new(topology);
            plan.insert_expected(
                "model.projection.weight",
                vec![2, 4],
                TensorPlacement::Shard {
                    axis: 1,
                    index: rank,
                    parts: 2,
                },
            )
            .unwrap();
            plan.insert("model.remote.weight", TensorPlacement::Omit);
            let partition = load_safetensors_partition(dir.path(), &plan, &stream).unwrap();
            assert_eq!(partition.len(), 1);
            assert_eq!(
                partition.opened_shards(),
                std::slice::from_ref(&local_shard)
            );
            assert!(partition.get("model.remote.weight").is_none());
            let local = partition
                .get("model.projection.weight")
                .unwrap()
                .evaluated()
                .unwrap();
            assert_eq!(local.as_array().shape(), &[2, 2]);
            reconstructed.push(local.as_slice::<i32>().to_vec());
        }
        // Slices are axis-1 contiguous views, so reconstruct each row from
        // the corresponding rows of both rank-local tensors.
        assert_eq!(reconstructed[0], [0, 1, 10, 11]);
        assert_eq!(reconstructed[1], [2, 3, 12, 13]);
        let union = [
            reconstructed[0][0],
            reconstructed[0][1],
            reconstructed[1][0],
            reconstructed[1][1],
            reconstructed[0][2],
            reconstructed[0][3],
            reconstructed[1][2],
            reconstructed[1][3],
        ];
        assert_eq!(union, [0, 1, 2, 3, 10, 11, 12, 13]);
    }

    #[test]
    fn selective_loader_materializes_only_ordered_noncontiguous_indices() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("model.safetensors"),
            "experts",
            &[0, 1, 10, 11, 20, 21, 30, 31, 40, 41],
            vec![5, 2],
        );
        let mut plan = PlacementPlan::new(topology(1, 1, 1, 2));
        plan.insert_expected(
            "experts",
            vec![5, 2],
            TensorPlacement::Indices {
                axis: 0,
                indices: vec![3, 1],
            },
        )
        .unwrap();
        let partition = load_safetensors_partition(dir.path(), &plan, &stream).unwrap();
        let local = partition.get("experts").unwrap().evaluated().unwrap();
        assert_eq!(local.as_array().shape(), &[2, 2]);
        assert_eq!(local.as_slice::<i32>(), &[30, 31, 10, 11]);

        for indices in [vec![], vec![1, 1], vec![1, 5]] {
            let mut invalid = PlacementPlan::new(topology(1, 1, 1, 2));
            assert!(invalid
                .insert_expected(
                    "experts",
                    vec![5, 2],
                    TensorPlacement::Indices { axis: 0, indices },
                )
                .is_err());
        }
    }

    #[test]
    fn replicated_default_loads_the_original_full_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("model.safetensors"),
            "weight",
            &[3, 5, 7, 9],
            vec![2, 2],
        );
        let plan = PlacementPlan::replicated(topology(0, 1, 1, 1));
        let partition = load_safetensors_partition(dir.path(), &plan, &stream).unwrap();
        let loaded = partition.get("weight").unwrap().evaluated().unwrap();
        assert_eq!(loaded.as_slice::<i32>(), &[3, 5, 7, 9]);
    }

    #[test]
    fn omitted_unsupported_tensor_is_never_materialized() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = [0u8; 4];
        let unsupported = TensorView::new(Dtype::F8_E5M2, vec![4], &bytes).unwrap();
        serialize_to_file(
            [("remote", unsupported)],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
        plan.insert("remote", TensorPlacement::Omit);
        let partition = load_safetensors_partition(dir.path(), &plan, &stream()).unwrap();
        assert!(partition.is_empty());
    }

    #[test]
    fn remote_only_index_shard_is_never_opened() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("remote.safetensors"), b"not safetensors").unwrap();
        write_index(dir.path(), &[("remote.weight", "remote.safetensors")]);
        let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
        plan.insert("remote.weight", TensorPlacement::Omit);
        let partition = load_safetensors_partition(dir.path(), &plan, &stream()).unwrap();
        assert!(partition.is_empty());
        assert!(partition.opened_shards().is_empty());
    }

    #[test]
    fn selective_loader_rejects_an_incomplete_opened_shard() {
        let dir = tempfile::tempdir().unwrap();
        write_i32_tensor(
            &dir.path().join("local.safetensors"),
            "requested.weight",
            &[1],
            vec![1],
        );
        std::fs::write(dir.path().join("remote.safetensors"), b"not safetensors").unwrap();
        write_index(
            dir.path(),
            &[
                ("requested.weight", "local.safetensors"),
                ("missing.weight", "local.safetensors"),
                ("remote.weight", "remote.safetensors"),
            ],
        );
        let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
        plan.insert("requested.weight", TensorPlacement::Local);
        plan.insert("missing.weight", TensorPlacement::Omit);
        plan.insert("remote.weight", TensorPlacement::Omit);

        assert!(matches!(
            load_safetensors_partition(dir.path(), &plan, &stream()),
            Err(Error::CheckpointStore(
                eredu_checkpoint::store::StoreError::ContradictoryIndexMapping { key, .. }
            )) if key == "missing.weight"
        ));
    }

    #[test]
    fn strict_partition_rejects_missing_malformed_and_unexpected_local_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("model.safetensors"),
            "present",
            &[1, 2, 3, 4],
            vec![2, 2],
        );
        let topology = topology(0, 1, 1, 1);

        let mut malformed = PlacementPlan::new(topology);
        malformed
            .insert_expected("present", vec![4, 2], TensorPlacement::Local)
            .unwrap();
        assert!(matches!(
            load_safetensors_partition(dir.path(), &malformed, &stream),
            Err(Error::Parallel(_))
        ));

        let mut missing = PlacementPlan::new(topology);
        missing.insert("present", TensorPlacement::Omit);
        missing.insert("required", TensorPlacement::Local);
        let error = load_safetensors_partition(dir.path(), &missing, &stream).unwrap_err();
        match error {
            Error::StrictLoadValidation { missing, unused } => {
                assert_eq!(missing, ["required"]);
                assert!(unused.is_empty());
            }
            other => panic!("unexpected error: {other}"),
        }

        let strict_empty = PlacementPlan::new(topology);
        let error = load_safetensors_partition(dir.path(), &strict_empty, &stream).unwrap_err();
        match error {
            Error::StrictLoadValidation { missing, unused } => {
                assert!(missing.is_empty());
                assert_eq!(unused, ["present"]);
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
