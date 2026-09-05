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

#[cfg(test)]
use eredu_runtime::TensorSlice;

/// MLX device binding around the authoritative neutral placement plan.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    topology: MlxRankContext,
    logical: eredu_runtime::PlacementPlan,
}

impl PlacementPlan {
    /// Creates a strict plan in which every checkpoint tensor must be named.
    pub fn new(topology: MlxRankContext) -> Self {
        let rank = eredu_runtime::PlacementRank::new(topology.world_size(), topology.global_rank())
            .expect("MLX rank context is already validated");
        Self {
            topology,
            logical: eredu_runtime::PlacementPlan::new(rank),
        }
    }

    /// Creates a plan that replicates every checkpoint tensor.
    pub fn replicated(topology: MlxRankContext) -> Self {
        Self::new(topology).with_default(TensorPlacement::Replicated)
    }

    /// Sets the placement used for checkpoint keys without an explicit entry.
    pub fn with_default(mut self, placement: TensorPlacement) -> Self {
        self.logical = self.logical.with_default(placement);
        self
    }

    /// Returns the mechanism-bound topology.
    pub const fn topology(&self) -> MlxRankContext {
        self.topology
    }

    /// Adds or replaces one exact source placement.
    pub fn insert(&mut self, source: impl Into<String>, placement: TensorPlacement) {
        self.logical.insert(source, placement);
    }

    /// Adds one placement with an exact admitted source shape.
    pub fn insert_expected(
        &mut self,
        source: impl Into<String>,
        expected_source_shape: impl Into<Vec<usize>>,
        placement: TensorPlacement,
    ) -> Result<(), Error> {
        self.logical
            .insert_expected(source, expected_source_shape, placement)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Adds a packed weight and its companions with one logical placement.
    pub fn insert_quantized_companions(
        &mut self,
        prefix: &str,
        placement: TensorPlacement,
        has_biases: bool,
    ) {
        self.logical
            .insert_quantized_companions(prefix, placement, has_biases);
    }

    /// Returns an explicit placement by exact checkpoint name.
    pub fn placement(&self, source: &str) -> Option<&TensorPlacement> {
        self.logical.placement(source)
    }

    /// Validates all cold logical geometry.
    pub fn validate(&self) -> Result<(), Error> {
        self.logical
            .validate()
            .map_err(|error| Error::Parallel(error.to_string()))
    }
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
    loaded: BTreeSet<String>,
    unexpected: Vec<String>,
}

impl PartitionReport {
    fn finish(self, plan: &PlacementPlan) -> Result<(), Error> {
        match plan
            .logical
            .validate_loaded_sources(&self.loaded, self.unexpected)
        {
            Ok(()) => Ok(()),
            Err(eredu_runtime::PlacementPlanError::Coverage {
                missing,
                unexpected,
            }) => Err(Error::StrictLoadValidation {
                missing,
                unused: unexpected,
            }),
            Err(error) => Err(Error::Parallel(error.to_string())),
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
    store: &dyn CheckpointSource,
    plan: &PlacementPlan,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<RankPartition, Error> {
    let prepared = prepare_partition_bindings(store, plan)?;
    plan.topology.validate_execution_stream(execution_stream)?;
    let mut tensors = HashMap::new();
    let mut opened_shards = BTreeSet::new();
    let context = MlxParameterMaterializationContext::new(source_stream, execution_stream);

    for prepared in prepared {
        let lease = store.acquire_lease(TensorReadRequest {
            key: prepared.source.clone(),
            selection: prepared.selection,
            policy: ReadPolicy::RequireBounded,
        })?;
        if let Some(path) = eredu_checkpoint::store::EncodedTensorLease::backing_path(&lease) {
            opened_shards.insert(path.to_path_buf());
        }
        #[cfg(test)]
        PARTITION_NATIVE_MATERIALIZATION_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
        let value = context
            .weight_lease(lease)?
            .materialize(source_stream, execution_stream)?
            .synchronize()?;
        if tensors.insert(prepared.source.clone(), value).is_some() {
            return Err(Error::Parallel(format!(
                "checkpoint tensor {:?} was materialized more than once",
                prepared.source
            )));
        }
    }

    Ok(RankPartition {
        topology: plan.topology,
        tensors,
        opened_shards: opened_shards.into_iter().collect(),
    })
}

struct PreparedPartitionBinding {
    source: String,
    selection: TensorSelection,
}

fn prepare_partition_bindings(
    store: &dyn CheckpointSource,
    plan: &PlacementPlan,
) -> Result<Vec<PreparedPartitionBinding>, Error> {
    plan.validate()?;
    let mut report = PartitionReport::default();
    let mut prepared = Vec::new();
    let mut bindings = Vec::new();

    for source in store.source_keys() {
        match plan.logical.potentially_local(&source) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(eredu_runtime::PlacementPlanError::UnexpectedSource { .. }) => {
                report.unexpected.push(source);
                continue;
            }
            Err(error) => return Err(Error::Parallel(error.to_string())),
        }

        let metadata = store.source_metadata(&source)?;
        let resolved = plan
            .logical
            .resolve(&source, &metadata.logical_shape)
            .map_err(|error| Error::Parallel(format!("checkpoint tensor {source}: {error}")))?;
        let selection = match resolved {
            eredu_runtime::ResolvedTensorPlacement::Omit => continue,
            eredu_runtime::ResolvedTensorPlacement::Materialize => TensorSelection::Full,
            eredu_runtime::ResolvedTensorPlacement::Selection(selection) => selection,
        };
        if !report.loaded.insert(source.clone()) {
            return Err(Error::Parallel(format!(
                "checkpoint tensor {source} was selected more than once"
            )));
        }
        let recipe = eredu_checkpoint::recipe::DerivedWeightRecipe::source(
            source.clone(),
            selection.clone(),
        );
        let expected = recipe.infer(store)?.byte_len();
        bindings.push(eredu_runtime::WeightBinding::from_recipe(
            source.clone(),
            recipe,
            expected,
        )?);
        prepared.push(PreparedPartitionBinding { source, selection });
    }

    report.finish(plan)?;
    eredu_runtime::preflight_bindings::<crate::backend::nn::shared::MlxNeuralBackend>(
        store, &bindings,
    )
    .map_err(|error| Error::Parallel(error.to_string()))?;
    Ok(prepared)
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

        assert_eq!(manifest.world_size(), 6);
        assert_eq!(manifest.rank(), 2);
        assert_eq!(manifest.groups().len(), 2);
        assert_eq!(manifest.groups()[0].id(), CollectiveGroupId::new(7));
        assert_eq!(manifest.groups()[0].members(), [0, 2, 4]);
        assert_eq!(manifest.groups()[0].local_index(), Some(1));
        assert_eq!(manifest.groups()[1].id(), CollectiveGroupId::new(11));
        assert_eq!(manifest.routes().len(), 1);
        assert_eq!(manifest.routes()[0].id(), CommunicationRouteId::new(19));
        assert_eq!(manifest.routes()[0].source(), 2);
        assert_eq!(manifest.routes()[0].destination(), 5);
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
        eredu_runtime::validate_compatible_communication_manifests(&manifests).unwrap();

        for manifest in &manifests {
            assert_eq!(manifest.groups().len(), 3);
            for (creation_order, group) in manifest.groups().iter().enumerate() {
                assert_eq!(
                    manifest.groups()[creation_order].creation_order(),
                    creation_order
                );
                assert_eq!(group.id(), manifest.groups()[creation_order].id());
                assert_eq!(
                    group.members()[group.local_index().unwrap()],
                    manifest.rank()
                );
                assert_eq!(group.members().len(), 2);
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
        mlx_communication_capabilities()
            .validate_manifest(&publication)
            .unwrap();

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
        let error = mlx_communication_capabilities()
            .validate_manifest(&unsupported)
            .expect_err("integer broadcast must be rejected");
        assert!(error.to_string().contains("I32"));
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
        mlx_communication_capabilities()
            .validate_manifest(&failure_agreement)
            .unwrap();

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
        mlx_communication_capabilities()
            .validate_manifest(&route_manifest)
            .unwrap();

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
        mlx_communication_capabilities()
            .validate_manifest(&exact_manifest)
            .unwrap();

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
        mlx_communication_capabilities()
            .validate_manifest(&barrier)
            .unwrap();
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
        let mut rank_zero = PlacementPlan::new(topology(0, 2, 2, 1));
        rank_zero.insert("owned", TensorPlacement::Rank { rank: 3 });
        let mut rank_three = PlacementPlan::new(topology(3, 2, 2, 1));
        rank_three.insert("owned", TensorPlacement::Rank { rank: 3 });
        assert!(matches!(
            rank_zero.logical.resolve("owned", &[2]).unwrap(),
            eredu_runtime::ResolvedTensorPlacement::Omit
        ));
        assert!(matches!(
            rank_three.logical.resolve("owned", &[2]).unwrap(),
            eredu_runtime::ResolvedTensorPlacement::Materialize
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

    #[test]
    fn late_partition_preflight_failure_performs_zero_payload_reads_or_native_work() {
        let dir = tempfile::tempdir().unwrap();
        let early_bytes = [1_i32, 2]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let late_bytes = [3_i32, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let early = TensorView::new(Dtype::I32, vec![2], &early_bytes).unwrap();
        let late = TensorView::new(Dtype::I32, vec![2], &late_bytes).unwrap();
        serialize_to_file(
            [("a.valid", early), ("z.invalid", late)],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();

        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
        plan.insert_expected("a.valid", vec![2], TensorPlacement::Local)
            .unwrap();
        plan.insert_expected("z.invalid", vec![3], TensorPlacement::Local)
            .unwrap();
        PARTITION_NATIVE_MATERIALIZATION_ATTEMPTS.store(0, Ordering::Relaxed);
        let source_stream = stream();
        let execution_stream = stream();

        assert!(matches!(
            load_partition_from_store_on_streams(&store, &plan, &source_stream, &execution_stream,),
            Err(Error::Parallel(_))
        ));
        assert_eq!(store.source_diagnostics().unwrap().physical_reads, 0);
        assert_eq!(
            PARTITION_NATIVE_MATERIALIZATION_ATTEMPTS.load(Ordering::Relaxed),
            0
        );
    }
}
