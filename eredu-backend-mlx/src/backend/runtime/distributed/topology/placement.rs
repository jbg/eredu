//! Checkpoint placement validation and rank-local materialization.

use super::*;

/// MLX device binding around the authoritative neutral placement plan.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    topology: MlxRankContext,
    pub(super) logical: eredu_runtime::PlacementPlan,
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
        PARTITION_NATIVE_MATERIALIZATION_ATTEMPTS.with(|count| count.set(count.get() + 1));
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
