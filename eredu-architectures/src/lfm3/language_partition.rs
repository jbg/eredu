//! Shared language pipeline placement for LFM3 conditional decoders.

use std::sync::Arc;

use eredu_core::{BackendFailure, ParallelRankTopology};
use eredu_nn::{AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, Tensor};
use eredu_runtime::{
    ComponentDefinition, LayerRuntimeState, PreparedComponent, RuntimeStateComponents,
};

use super::Declaration;
type Context<B> = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context;

/// Source-independent ownership and state geometry for one language rank.
#[derive(Debug, Clone)]
pub struct SelectedLanguagePartition {
    rank: ParallelRankTopology,
    vocabulary_prefix: Option<std::num::NonZeroUsize>,
    declaration: Arc<Declaration>,
    layout: eredu_runtime::LocalModelLayout,
    partition: eredu_runtime::ArchitecturePartition<
        crate::lfm3::PartitionLocalGeometry,
        eredu_runtime::NoAuxiliaryBoundarySchema,
    >,
}

/// The retained source and exact resource owners for one language pipeline stage.
pub struct PreparedLanguagePartition {
    selected: SelectedLanguagePartition,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    units: Vec<Vec<eredu_runtime::WeightBinding>>,
    pinned: Vec<eredu_runtime::WeightBinding>,
}

impl SelectedLanguagePartition {
    pub(super) fn select(
        declaration: Arc<Declaration>,
        rank: ParallelRankTopology,
        vocabulary_prefix: Option<std::num::NonZeroUsize>,
    ) -> Result<Self, BackendFailure> {
        use eredu_runtime::{ArchitecturePartition, NoAuxiliaryBoundarySchema, PartitionOwnership};

        if rank.expert_parallel_size() != 1 || rank.data_parallel_size() != 1 {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "LFM3 language partitioning requires tensor and pipeline axes only",
            )));
        }
        let definition = declaration.definition();
        let range = rank
            .layer_range(definition.state_layout().len())
            .map_err(BackendFailure::from_error)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(definition.parameters(), rank)
            .map_err(|reason| BackendFailure::from_error(eredu_nn::Error::backend(reason)))?;
        let geometry = crate::lfm3::partition_local_geometry(&declaration, &layout, range.clone())
            .map_err(BackendFailure::from_error)?;
        let transport = crate::transport::decoder();
        let mut static_roles = Vec::new();
        if rank.owns_embedding() {
            static_roles.extend(transport.first_owner_static_roles);
        }
        if rank.owns_output_head() {
            static_roles.extend(transport.last_owner_static_roles);
        }
        let ownership =
            PartitionOwnership::new(rank.owns_embedding(), rank.owns_output_head(), static_roles)
                .map_err(BackendFailure::from_error)?;
        let state = geometry.complete_state_layout().clone();
        let partition = ArchitecturePartition::from_description(
            definition.parameters(),
            [(crate::decoder::TARGET_EXECUTION_GROUP, range)],
            ownership,
            &state,
            &crate::transport::pipeline_state(0, &state),
            geometry,
            NoAuxiliaryBoundarySchema::new(declaration.args().hidden_size),
        )
        .map_err(BackendFailure::from_error)?;
        Ok(Self {
            rank,
            vocabulary_prefix,
            declaration,
            layout,
            partition,
        })
    }

    pub(super) fn bind(
        self,
        parameters: PreparedComponent<Arc<ComponentDefinition>>,
    ) -> Result<PreparedLanguagePartition, BackendFailure> {
        if parameters.declaration() != self.declaration.definition() {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "prepared LFM3 language parameters differ from the selected declaration",
            )));
        }
        let partition = &self.partition;
        let parameters = parameters
            .select(&self.layout)
            .map_err(BackendFailure::from_error)?
            .select_units(partition.units())
            .map_err(BackendFailure::from_error)?;
        let (_, invocations, source, units, pinned) = parameters.into_parts();
        if &invocations != partition.invocations() {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "retained LFM3 language invocations differ from the selected pipeline stage",
            )));
        }
        let targets = partition
            .parameter_bindings()
            .iter()
            .flat_map(|group| group.members())
            .map(|member| member.target())
            .collect::<std::collections::BTreeSet<_>>();
        let pinned = pinned
            .into_values()
            .flatten()
            .filter(|binding| targets.contains(binding.logical_target().unwrap_or(binding.name())))
            .collect::<Vec<_>>();
        let retained_targets = units
            .iter()
            .flatten()
            .chain(&pinned)
            .map(|binding| binding.logical_target().unwrap_or(binding.name()))
            .collect::<std::collections::BTreeSet<_>>();
        if retained_targets != targets {
            return Err(BackendFailure::from_error(eredu_nn::Error::backend(
                "retained LFM3 language parameters differ from the selected pipeline stage",
            )));
        }
        let keys = units
            .iter()
            .flatten()
            .chain(&pinned)
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned))
            .collect();
        let source = Arc::new(
            eredu_checkpoint::store::RestrictedCheckpointSource::including(
                source,
                "lfm3/language/pipeline",
                keys,
            )
            .map_err(BackendFailure::from_error)?,
        );
        Ok(PreparedLanguagePartition {
            selected: self,
            source,
            units,
            pinned,
        })
    }

    /// Rank identity of the retained stage.
    pub fn rank(&self) -> ParallelRankTopology {
        self.rank
    }

    /// Number of language blocks across all pipeline stages.
    pub fn complete_layer_count(&self) -> usize {
        self.declaration.definition().state_layout().len()
    }

    pub(super) fn args(&self) -> &crate::lfm3::ModelArgs {
        self.declaration.args()
    }

    /// Maximum decoder prefix whose vocabulary rows may be published together.
    pub fn vocabulary_prefix(&self) -> Option<std::num::NonZeroUsize> {
        self.vocabulary_prefix
    }

    /// The retained semantic and state ownership for the local language stage.
    pub fn partition(
        &self,
    ) -> &eredu_runtime::ArchitecturePartition<
        crate::lfm3::PartitionLocalGeometry,
        eredu_runtime::NoAuxiliaryBoundarySchema,
    > {
        &self.partition
    }
}

impl PreparedLanguagePartition {
    pub(super) fn select(
        declaration: Arc<Declaration>,
        parameters: PreparedComponent<Arc<ComponentDefinition>>,
        rank: ParallelRankTopology,
        vocabulary_prefix: Option<std::num::NonZeroUsize>,
    ) -> Result<Self, BackendFailure> {
        SelectedLanguagePartition::select(declaration, rank, vocabulary_prefix)?.bind(parameters)
    }

    /// Exact source-independent rank selection used to bind this stage.
    pub fn selected(&self) -> &SelectedLanguagePartition {
        &self.selected
    }

    /// Rank identity of the retained stage.
    pub fn rank(&self) -> ParallelRankTopology {
        self.selected.rank()
    }

    /// Number of language blocks across all pipeline stages.
    pub fn complete_layer_count(&self) -> usize {
        self.selected.complete_layer_count()
    }

    /// Maximum decoder prefix whose vocabulary rows may be published together.
    pub fn vocabulary_prefix(&self) -> Option<std::num::NonZeroUsize> {
        self.selected.vocabulary_prefix()
    }

    /// The retained semantic and state ownership for the local language stage.
    pub fn partition(
        &self,
    ) -> &eredu_runtime::ArchitecturePartition<
        crate::lfm3::PartitionLocalGeometry,
        eredu_runtime::NoAuxiliaryBoundarySchema,
    > {
        self.selected.partition()
    }

    /// Stage-local unit and pinned bindings used for residency sizing.
    pub fn bindings(
        &self,
    ) -> (
        &[Vec<eredu_runtime::WeightBinding>],
        &[eredu_runtime::WeightBinding],
    ) {
        (&self.units, &self.pinned)
    }

    /// Builds the stage-local modules and applies their physical storage preferences.
    pub fn into_layerwise<B, S>(
        self,
        context: &Context<B>,
    ) -> Result<
        (
            crate::lfm3::PartitionedLayeredModel<B>,
            eredu_checkpoint::store::SharedCheckpointSource,
            eredu_runtime::ExecutionUnitAddressMap,
            Vec<eredu_runtime::WeightBinding>,
            Vec<Vec<eredu_runtime::WeightBinding>>,
        ),
        BackendFailure,
    >
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        use eredu_runtime::LayeredArchitecture;

        let definition = self.selected.declaration.definition().clone();
        let model = crate::lfm3::Lfm3Kernel::<B>::from_partition(
            self.selected.declaration,
            definition,
            &self.selected.partition,
            context,
        )
        .map_err(BackendFailure::from_error)?;
        let pinned = eredu_runtime::apply_parameter_storage_layouts(
            <_ as LayeredArchitecture<B, S>>::static_modules(&model),
            self.pinned,
        );
        let invocations = self.selected.partition.invocations().clone();
        let units = self
            .units
            .into_iter()
            .enumerate()
            .map(|(ordinal, bindings)| {
                let location = invocations
                    .storage_location(ordinal)
                    .expect("retained owner");
                let module =
                    <_ as LayeredArchitecture<B, S>>::build_storage_unit(&model, location, context)
                        .map_err(BackendFailure::from_error)?;
                Ok(eredu_runtime::apply_parameter_storage_layouts(
                    &module, bindings,
                ))
            })
            .collect::<Result<_, BackendFailure>>()?;
        Ok((model, self.source, invocations, pinned, units))
    }
}
