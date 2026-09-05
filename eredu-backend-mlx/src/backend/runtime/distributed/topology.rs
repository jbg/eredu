//! MLX checkpoint placement and selective materialization.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::{BTreeSet, HashMap},
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
static PARTITION_NATIVE_MATERIALIZATION_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(super) fn reset_manifest_group_realizations() {
    MANIFEST_GROUP_REALIZATIONS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn manifest_group_realizations() -> usize {
    MANIFEST_GROUP_REALIZATIONS.load(Ordering::Relaxed)
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

/// Backend communication contexts materialized from opaque group realizations.
///
/// Uncontracted construction may enter native subgroup splits. Opaque manifest
/// construction creates exact logical membership views and permits a
/// world-collective implementation only when consensus proves a complete,
/// same-requirement subgroup wave at that creation order.
#[derive(Clone)]
pub struct ParallelCommunicators {
    world_size: usize,
    global_rank: usize,
    descriptors: Vec<CommunicationGroupDescriptor>,
    control_world: Group,
    groups: HashMap<CollectiveGroupId, GroupCommunicator>,
    routes: HashMap<CommunicationRouteId, CommunicationRouteRealization>,
}

#[derive(Clone)]
struct GroupCommunicator {
    descriptor: CommunicationGroupDescriptor,
    native: Option<Group>,
}

impl std::fmt::Debug for ParallelCommunicators {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ParallelCommunicators")
            .field("world_size", &self.world_size)
            .field("global_rank", &self.global_rank)
            .field("descriptors", &self.descriptors)
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
        let prepared = eredu_runtime::prepare_communication_realization(
            manifest,
            &agreed_manifests,
            &mlx_communication_capabilities(),
            eredu_runtime::CommunicationTopologyCapabilities::RingWithWorldWaves,
        )
        .map_err(|error| {
            Error::Parallel(format!(
                "communication manifest cannot be realized by MLX mechanisms: {error}"
            ))
        })?;
        let completion = manifest.completion_policy().ok_or_else(|| {
            Error::Parallel(
                "communication manifest requires an explicit bounded completion policy".into(),
            )
        })?;
        let control = Group::uncontracted(world);
        crate::backend::runtime::distributed::completion::ensure_group_available(&control)?;
        Self::new_with_routes(prepared, world, completion)
    }

    fn new_with_routes(
        prepared: eredu_runtime::PreparedCommunicationRealization,
        world: &NativeGroup,
        completion: eredu_runtime::CommunicationCompletionPolicy,
    ) -> Result<Self, Error> {
        // Fence both manifest and uncontracted construction while any timed-out
        // work on this exact native communicator remains quarantined.
        let owned_world = Group::uncontracted(world).with_completion_policy(completion);
        crate::backend::runtime::distributed::completion::ensure_group_available(&owned_world)?;
        let _setup = owned_world.begin_bounded_setup()?;
        let manifest = prepared.manifest();
        if world.rank() != manifest.rank() || world.size() != manifest.world_size() {
            return Err(Error::Parallel(format!(
                "collective realization expects world rank {}/{} but received {}/{}",
                manifest.rank(),
                manifest.world_size(),
                world.rank(),
                world.size()
            )));
        }
        // This unsplit handle is intentionally uncontracted: it is the control
        // plane from which exact manifest handles are realized.
        let world = owned_world;
        let groups = prepared
            .try_create_groups(|descriptor, world_wave| {
                let group =
                    Self::materialize(descriptor, manifest.world_size(), &world, world_wave)?;
                Ok::<_, Error>((descriptor.id(), group))
            })?
            .into_iter()
            .collect::<HashMap<_, _>>();
        let routes = prepared
            .try_create_routes(|descriptor, world_wave| {
                let route =
                    CommunicationRouteRealization::from_descriptor(descriptor, &world, world_wave)?;
                Ok::<_, Error>((descriptor.id(), route))
            })?
            .into_iter()
            .collect::<HashMap<_, _>>();
        Ok(Self {
            world_size: manifest.world_size(),
            global_rank: manifest.rank(),
            descriptors: manifest.groups().to_vec(),
            control_world: world,
            groups,
            routes,
        })
    }

    fn materialize(
        descriptor: &CommunicationGroupDescriptor,
        world_size: usize,
        world: &Group,
        world_collective_wave: bool,
    ) -> Result<GroupCommunicator, Error> {
        #[cfg(test)]
        MANIFEST_GROUP_REALIZATIONS.fetch_add(1, Ordering::Relaxed);
        let size = descriptor.members().len();
        let native = if size == world_size {
            world.clone()
        } else {
            world
                .logical_subgroup(descriptor.members())
                .map(|group| group.with_world_collective_wave(world_collective_wave))
                .map_err(|error| {
                    Error::Parallel(format!(
                        "failed to materialize logical group {} with members {:?}: {error}",
                        descriptor.id().value(),
                        descriptor.members()
                    ))
                })?
        };
        let native = native
            .with_manifest_contract(
                descriptor,
                world.completion_policy().ok_or_else(|| {
                    Error::Parallel("manifest group has no selected completion policy".into())
                })?,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(GroupCommunicator {
            descriptor: descriptor.clone(),
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
        if communicator.descriptor.members().len() == 1 {
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
            world_size: _,
            global_rank: _,
            descriptors: _,
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
        self.descriptors
            .iter()
            .filter(|group| group.members().len() > 1)
            .filter_map(CommunicationGroupDescriptor::collective_descriptor)
            .collect()
    }

    pub(crate) fn group_ids(&self) -> Vec<CollectiveGroupId> {
        self.descriptors
            .iter()
            .filter(|group| group.members().len() > 1)
            .map(CommunicationGroupDescriptor::id)
            .collect()
    }

    pub(crate) const fn world_size(&self) -> usize {
        self.world_size
    }

    pub(crate) const fn global_rank(&self) -> usize {
        self.global_rank
    }
}

mod placement;
pub use placement::{
    load_partition_from_store_on_streams, load_safetensors_partition,
    load_safetensors_partition_on_streams, PlacementPlan, RankPartition,
};

#[cfg(test)]
#[path = "topology/tests.rs"]
mod tests;
