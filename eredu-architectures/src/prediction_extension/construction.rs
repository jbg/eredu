//! Inputs retained by successful ordinary construction, reused by its workspace loan.
use super::*;
use crate::prepared_sources::PreparedModelSources;
use eredu_nn::{
    workspace::{WorkspaceContext, WorkspaceMetadataError},
    Error,
};
use std::{
    mem::{size_of, size_of_val},
    sync::Arc,
};

/// Closed immutable construction inputs from the exact prepared source graph.
/// This owner conveys neither native state provenance nor invocation permission.
#[derive(Debug)]
pub struct PreparedPredictionConstruction {
    selected_tasks: Vec<ReplicatedTextMaterializationTask>,
    rows: Vec<Arc<Vec<ReplicatedTextMaterializationTask>>>,
    source_layout: Option<Arc<LocalModelLayout>>,
    coordinates: Vec<Option<Arc<eredu_core::component::ComponentCoordinateMap>>>,
    partitioned: bool,
    factory: Factory,
}
#[derive(Debug)]
enum Factory {
    V4 {
        source: Vec<crate::deepseek::v4::V4PredictionUnitSpec>,
        local: Vec<crate::deepseek::v4::V4PredictionUnitSpec>,
        state: Vec<(usize, LayerCachePolicy)>,
        dspark: Option<DsparkConstruction>,
    },
    Nemotron {
        source: Vec<crate::nemotron_h::PredictionUnitSpec>,
        local: Vec<crate::nemotron_h::PredictionUnitSpec>,
        pattern: usize,
        state: StateLayout,
    },
    V3 {
        source: Vec<crate::deepseek::mtp::V3PredictionLayerSpec>,
        local: Vec<crate::deepseek::mtp::V3PredictionLayerSpec>,
        state: Vec<(usize, LayerCachePolicy)>,
    },
    Qwen {
        source: Vec<crate::qwen::hybrid::PredictionUnitSpec>,
        local: Vec<crate::qwen::hybrid::PredictionUnitSpec>,
        shared: crate::qwen::hybrid::PredictionSharedSpec,
        shared_tasks: Arc<Vec<ReplicatedTextMaterializationTask>>,
        state: StateLayout,
    },
    Inkling {
        source: crate::inkling::MtpModelSpec,
        local: crate::inkling::MtpModelSpec,
        shared_tasks: Option<Arc<Vec<ReplicatedTextMaterializationTask>>>,
        state: StateLayout,
    },
}
#[derive(Debug)]
pub(super) struct DsparkConstruction {
    pub(super) source: crate::deepseek::v4::DsparkStaticSpec,
    pub(super) local: crate::deepseek::v4::DsparkStaticSpec,
    pub(super) strategy: DsparkPredictionStrategy,
    pub(super) tasks: Arc<Vec<ReplicatedTextMaterializationTask>>,
}
impl PreparedPredictionConstruction {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        source: Vec<crate::deepseek::mtp::V3PredictionLayerSpec>,
        local: Vec<crate::deepseek::mtp::V3PredictionLayerSpec>,
        selected_tasks: Vec<ReplicatedTextMaterializationTask>,
        rows: Vec<Arc<Vec<ReplicatedTextMaterializationTask>>>,
        source_layout: Option<Arc<LocalModelLayout>>,
        coordinates: Vec<Option<Arc<eredu_core::component::ComponentCoordinateMap>>>,
        state: Vec<(usize, LayerCachePolicy)>,
        partitioned: bool,
    ) -> Self {
        Self {
            selected_tasks,
            rows,
            source_layout,
            coordinates,
            partitioned,
            factory: Factory::V3 {
                source,
                local,
                state,
            },
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen(
        source: Vec<crate::qwen::hybrid::PredictionUnitSpec>,
        local: Vec<crate::qwen::hybrid::PredictionUnitSpec>,
        shared: crate::qwen::hybrid::PredictionSharedSpec,
        selected_tasks: Vec<ReplicatedTextMaterializationTask>,
        rows: Vec<Arc<Vec<ReplicatedTextMaterializationTask>>>,
        shared_tasks: Arc<Vec<ReplicatedTextMaterializationTask>>,
        coordinates: Vec<Option<Arc<eredu_core::component::ComponentCoordinateMap>>>,
        state: StateLayout,
        partitioned: bool,
    ) -> Self {
        Self {
            selected_tasks,
            rows,
            source_layout: None,
            coordinates,
            partitioned,
            factory: Factory::Qwen {
                source,
                local,
                shared,
                shared_tasks,
                state,
            },
        }
    }
    pub(super) fn inkling(
        source: crate::inkling::MtpModelSpec,
        local: crate::inkling::MtpModelSpec,
        selected_tasks: Vec<ReplicatedTextMaterializationTask>,
        rows: Vec<Arc<Vec<ReplicatedTextMaterializationTask>>>,
        shared_tasks: Option<Arc<Vec<ReplicatedTextMaterializationTask>>>,
        state: StateLayout,
    ) -> Self {
        let coordinates = vec![None; rows.len()];
        Self {
            selected_tasks,
            rows,
            source_layout: None,
            coordinates,
            partitioned: false,
            factory: Factory::Inkling {
                source,
                local,
                shared_tasks,
                state,
            },
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn nemotron(
        source: Vec<crate::nemotron_h::PredictionUnitSpec>,
        local: Vec<crate::nemotron_h::PredictionUnitSpec>,
        selected_tasks: Vec<ReplicatedTextMaterializationTask>,
        rows: Vec<Arc<Vec<ReplicatedTextMaterializationTask>>>,
        source_layout: Option<Arc<LocalModelLayout>>,
        coordinates: Vec<Option<Arc<eredu_core::component::ComponentCoordinateMap>>>,
        pattern: usize,
        state: StateLayout,
        partitioned: bool,
    ) -> Self {
        Self {
            selected_tasks,
            rows,
            source_layout,
            coordinates,
            partitioned,
            factory: Factory::Nemotron {
                source,
                local,
                pattern,
                state,
            },
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn v4(
        source: Vec<crate::deepseek::v4::V4PredictionUnitSpec>,
        local: Vec<crate::deepseek::v4::V4PredictionUnitSpec>,
        selected_tasks: Vec<ReplicatedTextMaterializationTask>,
        rows: Vec<Arc<Vec<ReplicatedTextMaterializationTask>>>,
        source_layout: Option<Arc<LocalModelLayout>>,
        coordinates: Vec<Option<Arc<eredu_core::component::ComponentCoordinateMap>>>,
        state: Vec<(usize, LayerCachePolicy)>,
        dspark: Option<DsparkConstruction>,
        partitioned: bool,
    ) -> Self {
        Self {
            selected_tasks,
            rows,
            source_layout,
            coordinates,
            partitioned,
            factory: Factory::V4 {
                source,
                local,
                state,
                dspark,
            },
        }
    }
    fn state_source_layout(&self) -> PredictionStateSourceLayout<'_> {
        match &self.factory {
            Factory::V3 { state, .. } => PredictionStateSourceLayout::Sequential(state),
            Factory::V4 { state, .. } => PredictionStateSourceLayout::Pooling(state),
            Factory::Nemotron { state, .. }
            | Factory::Qwen { state, .. }
            | Factory::Inkling { state, .. } => PredictionStateSourceLayout::Model(state),
        }
    }
    fn matches(&self, model: &SafetensorsModelConfig) -> bool {
        let count = self.rows.len();
        match (&self.factory, model) {
            (
                Factory::V4 {
                    source,
                    local,
                    state,
                    dspark,
                },
                SafetensorsModelConfig::DeepSeekV4(args),
            ) => {
                source.len() == count
                    && local.len() == count
                    && state.len() == count
                    && dspark.is_some() == args.dspark.is_some()
                    && source
                        .iter()
                        .chain(local)
                        .all(|spec| spec.fused() == dspark.is_some())
            }
            (
                Factory::Nemotron {
                    source,
                    local,
                    pattern,
                    state,
                },
                SafetensorsModelConfig::NemotronH(_),
            ) => {
                *pattern > 0
                    && count.is_multiple_of(*pattern)
                    && source.len() == count
                    && local.len() == count
                    && state.len() == count
            }
            (
                Factory::V3 {
                    source,
                    local,
                    state,
                },
                SafetensorsModelConfig::DeepSeekV3(_),
            ) => source.len() == count && local.len() == count && state.len() == count,
            (
                Factory::Qwen {
                    source,
                    local,
                    state,
                    ..
                },
                SafetensorsModelConfig::QwenHybrid(_),
            ) => source.len() == count && local.len() == count && state.len() == count,
            (
                Factory::Inkling {
                    source,
                    local,
                    shared_tasks,
                    state,
                },
                SafetensorsModelConfig::Inkling(_),
            ) => {
                source.len() == count
                    && local.len() == count
                    && state.len() == count
                    && source.has_shared() == local.has_shared()
                    && source.has_shared() == shared_tasks.is_some()
            }
            _ => false,
        }
    }
}
#[derive(Debug, thiserror::Error)]
enum ConstructionError {
    #[error("prediction construction has no completed retained source")]
    Missing,
    #[error("prediction construction differs from its retained selection")]
    Selection,
    #[error("prediction construction has no retained unit factory for this profile")]
    Factory,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct ConstructionFailure {
    #[source]
    cause: Error,
    // Both original metadata accounts outlive the failure and its source buffers.
    _source: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
    _local: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
}
fn controls<T>(context: &WorkspaceContext) -> Result<(), Error> {
    let parts = [
        size_of::<T>(),
        size_of::<Result<T, Error>>(),
        size_of::<&WorkspaceContext>(),
    ];
    context.charge_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    Ok(())
}

/// Reconstructs the same typed units from the completed load's immutable inputs.
/// The native binder separately authenticates actual parameter and state sources.
/// No cold source is manufactured when this exact source graph was not loaded.
pub(crate) fn prepare_retained<B, P, F>(
    sources: &PreparedModelSources,
    source_context: &<B::Tensor as Tensor>::Context,
    execution_context: &<B::Tensor as Tensor>::Context,
    project: F,
) -> Result<(PreparedPredictionExtension<B>, P), Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    F: for<'state> FnOnce(PredictionStateSourceLayout<'state>) -> Result<P, Error>,
{
    let source =
        B::construction_metadata(source_context).ok_or(WorkspaceMetadataError::Unqualified)?;
    let local =
        B::construction_metadata(execution_context).ok_or(WorkspaceMetadataError::Unqualified)?;
    controls::<(
        ConstructionFailure,
        &PreparedModelSources,
        PreparedPredictionExtension<B>,
        F,
        P,
        Result<P, Error>,
    )>(local)?;
    local.charge_metadata(
        WorkspaceContext::metadata_source_bytes::<ConstructionFailure>()
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    let source_funding = source.metadata_funding();
    let local_funding = local.metadata_funding();
    prepare_inner::<B, P, F>(
        sources,
        source_context,
        execution_context,
        source,
        local,
        project,
    )
    .map_err(|cause| {
        Error::backend_retained_source(ConstructionFailure {
            cause,
            _source: source_funding,
            _local: local_funding,
        })
    })
}
fn prepare_inner<B, P, F>(
    sources: &PreparedModelSources,
    source_context: &<B::Tensor as Tensor>::Context,
    execution_context: &<B::Tensor as Tensor>::Context,
    source: &WorkspaceContext,
    local: &WorkspaceContext,
    project: F,
) -> Result<(PreparedPredictionExtension<B>, P), Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    F: for<'state> FnOnce(PredictionStateSourceLayout<'state>) -> Result<P, Error>,
{
    let selected = sources.selected();
    let fail = |cause| local.metadata_source(cause);
    let extension = sources
        .prediction_extension()
        .ok_or_else(|| fail(ConstructionError::Missing))?;
    let placement = sources
        .retained_prediction_placement()
        .ok_or_else(|| fail(ConstructionError::Missing))?;
    let construction = placement
        .construction()
        .ok_or_else(|| fail(ConstructionError::Factory))?;
    let topology = selected.execution().parallel_topology().unwrap_or(
        ParallelTopology::new(1, 1, 1, 1)
            .and_then(|t| ParallelRankTopology::new(t, 0))
            .map_err(|cause| local.metadata_source(cause))?,
    );
    if topology != placement.topology()
        || selected
            .prediction_extension()
            .is_none_or(|other| !extension.same_admission(other))
        || selected
            .text_realization()
            .auxiliary_materialization_tasks()
            != construction.selected_tasks.as_slice()
        || extension.depth() != construction.rows.len()
        || construction.rows.len() != construction.coordinates.len()
        || !construction.matches(extension.complete_architecture().model())
    {
        return Err(fail(ConstructionError::Selection));
    }
    let layout = placement
        .retained_layout()
        .ok_or_else(|| fail(ConstructionError::Missing))?;
    let parameters = placement
        .retained_parameters()
        .ok_or_else(|| fail(ConstructionError::Missing))?;
    controls::<(
        Arc<PreparedPredictionConstruction>,
        Arc<LocalModelLayout>,
        Arc<eredu_runtime::ArchitectureParameterDescription>,
        ParallelRankTopology,
    )>(local)?;
    // Each cold family constructor has its own stack frame. Keeping all model
    // temporaries in this dispatcher made unselected families consume stack too.
    controls::<RetainedFactory<'_, B>>(local)?;
    let factory = RetainedFactory {
        source_context,
        execution_context,
        source,
        local,
        selected,
        construction,
        layout,
        parameters,
    };
    // The immutable retained state declaration is available before constructing
    // any workspace tensors. Bind actual native source loans at this boundary,
    // after selection validation and before constructors can start tracing.
    let projected = project(construction.state_source_layout())?;
    let prepared = match &construction.factory {
        Factory::V4 {
            source: source_specs,
            local: local_specs,
            state: source_state,
            dspark,
        } => factory.v4(source_specs, local_specs, source_state, dspark),
        Factory::Nemotron {
            source: source_specs,
            local: local_specs,
            pattern,
            state,
        } => factory.nemotron(source_specs, local_specs, pattern, state),
        Factory::V3 {
            source: source_specs,
            local: local_specs,
            state: source_state,
        } => factory.v3(source_specs, local_specs, source_state),
        Factory::Qwen {
            source: source_specs,
            local: local_specs,
            shared,
            shared_tasks,
            state,
        } => factory.qwen(source_specs, local_specs, shared, shared_tasks, state),
        Factory::Inkling {
            source: source_specs,
            local: local_specs,
            shared_tasks,
            state,
        } => factory.inkling(source_specs, local_specs, shared_tasks, state),
    }?;
    Ok((prepared, projected))
}

// Borrowed exact inputs plus the two already retained placement owners. Moving
// this frame selects a constructor; it creates no new source or allocation.
struct RetainedFactory<'a, B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    source_context: &'a <B::Tensor as Tensor>::Context,
    execution_context: &'a <B::Tensor as Tensor>::Context,
    source: &'a WorkspaceContext,
    local: &'a WorkspaceContext,
    selected: &'a crate::selected_execution::SelectedPreparation,
    construction: &'a Arc<PreparedPredictionConstruction>,
    layout: Arc<LocalModelLayout>,
    parameters: Arc<eredu_runtime::ArchitectureParameterDescription>,
}
impl<B> RetainedFactory<'_, B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    // Preserve a separate cold construction frame in unoptimized builds too.
    #[inline(never)]
    fn v4(
        self,
        source_specs: &[crate::deepseek::v4::V4PredictionUnitSpec],
        local_specs: &[crate::deepseek::v4::V4PredictionUnitSpec],
        source_state: &[(usize, LayerCachePolicy)],
        dspark: &Option<DsparkConstruction>,
    ) -> Result<PreparedPredictionExtension<B>, Error> {
        controls::<(
            Self,
            &[crate::deepseek::v4::V4PredictionUnitSpec],
            &[crate::deepseek::v4::V4PredictionUnitSpec],
            &[(usize, LayerCachePolicy)],
            &Option<DsparkConstruction>,
            PreparedPredictionExtension<B>,
        )>(self.local)?;
        let Self {
            source_context,
            execution_context,
            source,
            local,
            selected,
            construction,
            layout,
            parameters,
        } = self;
        let (units, state) = Self::v4_units(
            source_context, execution_context, source, local, selected,
            construction, source_specs, local_specs, source_state,
        )?;
        match dspark {
            None => Ok(PreparedPredictionExtension::DeepSeekV4 {
                layout,
                parameters,
                units,
                state,
                construction: construction.clone(),
            }),
            Some(shared) => {
                controls::<(
                    PreparedPredictionUnit<crate::deepseek::v4::DsparkStatic<B>>,
                    PreparedDsparkPredictionExtension<B>,
                )>(local)?;
                source.charge_metadata(size_of::<crate::deepseek::v4::DsparkStatic<B>>())?;
                let static_modules = PreparedPredictionUnit {
                    source: shared.source.instantiate::<B>(source_context)?,
                    local: shared.local.instantiate::<B>(execution_context)?,
                    tasks: shared.tasks.clone(),
                    source_layout: construction.source_layout.clone(),
                    residency: selected.text_realization().residency(),
                    role: PredictionModuleRole::Shared,
                };
                let strategy = shared.strategy.clone_workspace(local)?;
                Ok(PreparedPredictionExtension::DeepSeekV4Dspark {
                    layout,
                    parameters,
                    units,
                    state,
                    extension: PreparedDsparkPredictionExtension {
                        strategy,
                        static_modules,
                    },
                    construction: construction.clone(),
                })
            }
        }
    }
    // Unit construction finishes before the family envelope and optional shared
    // modules are assembled. Their by-value temporaries need not coexist with
    // the nested block/attention constructors in unoptimized execution.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn v4_units(
        source_context: &<B::Tensor as Tensor>::Context,
        execution_context: &<B::Tensor as Tensor>::Context,
        source: &WorkspaceContext,
        local: &WorkspaceContext,
        selected: &crate::selected_execution::SelectedPreparation,
        construction: &Arc<PreparedPredictionConstruction>,
        source_specs: &[crate::deepseek::v4::V4PredictionUnitSpec],
        local_specs: &[crate::deepseek::v4::V4PredictionUnitSpec],
        source_state: &[(usize, LayerCachePolicy)],
    ) -> Result<(
        Vec<PreparedPredictionUnit<crate::deepseek::v4::Unit<B>>>,
        Vec<(usize, LayerCachePolicy)>,
    ), Error> {
        controls::<(
            &<B::Tensor as Tensor>::Context,
            &<B::Tensor as Tensor>::Context,
            &WorkspaceContext, &WorkspaceContext,
            &crate::selected_execution::SelectedPreparation,
            &Arc<PreparedPredictionConstruction>,
            &[crate::deepseek::v4::V4PredictionUnitSpec],
            &[crate::deepseek::v4::V4PredictionUnitSpec],
            &[(usize, LayerCachePolicy)],
            (Vec<PreparedPredictionUnit<crate::deepseek::v4::Unit<B>>>, Vec<(usize, LayerCachePolicy)>),
            Result<(Vec<PreparedPredictionUnit<crate::deepseek::v4::Unit<B>>>, Vec<(usize, LayerCachePolicy)>), Error>,
        )>(local)?;
        let fail = |cause| local.metadata_source(cause);
        let mut units = local.metadata_vec(construction.rows.len())?;
        let mut state = local.metadata_vec(source_state.len())?;
        for (depth, tasks) in construction.rows.iter().enumerate() {
            controls::<(
                PreparedPredictionUnit<crate::deepseek::v4::Unit<B>>,
                usize,
                Arc<Vec<ReplicatedTextMaterializationTask>>,
                Option<Arc<LocalModelLayout>>,
            )>(local)?;
            source.charge_metadata(size_of::<crate::deepseek::v4::Unit<B>>())?;
            let source_unit = source_specs[depth].instantiate::<B>(source_context)?;
            let mut local_unit = local_specs[depth].instantiate::<B>(execution_context)?;
            let routed = match &mut local_unit {
                crate::deepseek::v4::Unit::Prediction(p) => &mut p.decoder.feed_forward,
                crate::deepseek::v4::Unit::Dspark(b) => &mut b.feed_forward,
                crate::deepseek::v4::Unit::Target(_) => {
                    return Err(fail(ConstructionError::Selection));
                }
            };
            let coordinates = construction.coordinates[depth]
                .as_ref()
                .ok_or_else(|| fail(ConstructionError::Selection))?;
            routed.bind_shared_resident_unit_coordinates(
                coordinates.clone(),
                construction.partitioned,
            );
            let (ordinal, policy) = &source_state[depth];
            state.push((
                *ordinal,
                StateLayout::clone_policy_workspace(policy, local)?,
            ));
            units.push(PreparedPredictionUnit {
                source: source_unit,
                local: local_unit,
                tasks: tasks.clone(),
                source_layout: construction.source_layout.clone(),
                residency: selected.text_realization().residency(),
                role: PredictionModuleRole::Unit,
            });
        }
        Ok((units, state))
    }
    // Preserve a separate cold construction frame in unoptimized builds too.
    #[inline(never)]
    fn nemotron(
        self,
        source_specs: &[crate::nemotron_h::PredictionUnitSpec],
        local_specs: &[crate::nemotron_h::PredictionUnitSpec],
        pattern: &usize,
        state: &StateLayout,
    ) -> Result<PreparedPredictionExtension<B>, Error> {
        controls::<(
            Self,
            &[crate::nemotron_h::PredictionUnitSpec],
            &[crate::nemotron_h::PredictionUnitSpec],
            &usize,
            &StateLayout,
            PreparedPredictionExtension<B>,
        )>(self.local)?;
        let Self {
            source_context,
            execution_context,
            source,
            local,
            selected,
            construction,
            layout,
            parameters,
        } = self;
        let fail = |cause| local.metadata_source(cause);
        let mut groups = local.metadata_vec(construction.rows.len() / pattern)?;
        for first in (0..construction.rows.len()).step_by(*pattern) {
            controls::<Vec<PreparedPredictionUnit<crate::nemotron_h::PredictionUnit<B>>>>(local)?;
            let mut units = local.metadata_vec(*pattern)?;
            for physical in first..first + pattern {
                controls::<(
                    PreparedPredictionUnit<crate::nemotron_h::PredictionUnit<B>>,
                    usize,
                    Arc<Vec<ReplicatedTextMaterializationTask>>,
                )>(local)?;
                source.charge_metadata(size_of::<crate::nemotron_h::PredictionUnit<B>>())?;
                let source_unit = source_specs[physical].instantiate::<B>(source_context)?;
                let mut local_unit = local_specs[physical].instantiate::<B>(execution_context)?;
                match (
                    &mut local_unit.block.operator,
                    &construction.coordinates[physical],
                ) {
                    (crate::nemotron_h::Operator::Sparse(moe), Some(coordinates)) => moe
                        .bind_shared_resident_unit_coordinates(
                            coordinates.clone(),
                            construction.partitioned,
                        ),
                    (crate::nemotron_h::Operator::Attention(_), None) => {}
                    _ => return Err(fail(ConstructionError::Selection)),
                }
                units.push(PreparedPredictionUnit {
                    source: source_unit,
                    local: local_unit,
                    tasks: construction.rows[physical].clone(),
                    source_layout: construction.source_layout.clone(),
                    residency: selected.text_realization().residency(),
                    role: PredictionModuleRole::Unit,
                });
            }
            groups.push(units);
        }
        let state = state.clone_workspace(local)?;
        Ok(PreparedPredictionExtension::NemotronH {
            layout,
            parameters,
            groups,
            state,
            construction: construction.clone(),
        })
    }
    // Preserve a separate cold construction frame in unoptimized builds too.
    #[inline(never)]
    fn v3(
        self,
        source_specs: &[crate::deepseek::mtp::V3PredictionLayerSpec],
        local_specs: &[crate::deepseek::mtp::V3PredictionLayerSpec],
        source_state: &[(usize, LayerCachePolicy)],
    ) -> Result<PreparedPredictionExtension<B>, Error> {
        controls::<(
            Self,
            &[crate::deepseek::mtp::V3PredictionLayerSpec],
            &[crate::deepseek::mtp::V3PredictionLayerSpec],
            &[(usize, LayerCachePolicy)],
            PreparedPredictionExtension<B>,
        )>(self.local)?;
        let Self {
            source_context,
            execution_context,
            source,
            local,
            selected,
            construction,
            layout,
            parameters,
        } = self;
        let fail = |cause| local.metadata_source(cause);
        let mut units = local.metadata_vec(construction.rows.len())?;
        let mut state = local.metadata_vec(source_state.len())?;
        for (depth, tasks) in construction.rows.iter().enumerate() {
            controls::<(
                PreparedPredictionUnit<crate::deepseek::v3::Unit<B>>,
                usize,
                Arc<Vec<ReplicatedTextMaterializationTask>>,
                Option<Arc<LocalModelLayout>>,
            )>(local)?;
            source.charge_metadata(size_of::<crate::deepseek::v3::Unit<B>>())?;
            let source_unit = crate::deepseek::v3::Unit::Prediction(
                source_specs[depth].instantiate::<B>(source_context)?,
            );
            let mut local_unit = local_specs[depth].instantiate::<B>(execution_context)?;
            match (
                &mut local_unit.decoder.feed_forward,
                &construction.coordinates[depth],
            ) {
                (crate::deepseek::block::V3FeedForward::Routed(moe), Some(coordinates)) => moe
                    .bind_shared_resident_unit_coordinates(
                        coordinates.clone(),
                        construction.partitioned,
                    ),
                (crate::deepseek::block::V3FeedForward::Dense(_), None) => {}
                _ => return Err(fail(ConstructionError::Selection)),
            }
            let (ordinal, policy) = &source_state[depth];
            if !matches!(policy, LayerCachePolicy::CompressedLatentRotary { .. }) {
                return Err(fail(ConstructionError::Selection));
            }
            state.push((*ordinal, policy.clone()));
            units.push(PreparedPredictionUnit {
                source: source_unit,
                local: crate::deepseek::v3::Unit::Prediction(local_unit),
                tasks: tasks.clone(),
                source_layout: construction.source_layout.clone(),
                residency: selected.text_realization().residency(),
                role: PredictionModuleRole::Unit,
            });
        }
        Ok(PreparedPredictionExtension::DeepSeekV3 {
            layout,
            parameters,
            units,
            state,
            construction: construction.clone(),
        })
    }
    // Preserve a separate cold construction frame in unoptimized builds too.
    #[inline(never)]
    fn qwen(
        self,
        source_specs: &[crate::qwen::hybrid::PredictionUnitSpec],
        local_specs: &[crate::qwen::hybrid::PredictionUnitSpec],
        shared: &crate::qwen::hybrid::PredictionSharedSpec,
        shared_tasks: &Arc<Vec<ReplicatedTextMaterializationTask>>,
        state: &StateLayout,
    ) -> Result<PreparedPredictionExtension<B>, Error> {
        controls::<(
            Self,
            &[crate::qwen::hybrid::PredictionUnitSpec],
            &[crate::qwen::hybrid::PredictionUnitSpec],
            &crate::qwen::hybrid::PredictionSharedSpec,
            &Arc<Vec<ReplicatedTextMaterializationTask>>,
            &StateLayout,
            PreparedPredictionExtension<B>,
        )>(self.local)?;
        let Self {
            source_context,
            execution_context,
            source,
            local,
            selected,
            construction,
            layout,
            parameters,
        } = self;
        let fail = |cause| local.metadata_source(cause);
        let mut units = local.metadata_vec(construction.rows.len())?;
        for (depth, tasks) in construction.rows.iter().enumerate() {
            controls::<(
                PreparedPredictionUnit<crate::qwen::hybrid::PredictionUnit<B>>,
                usize,
                Arc<Vec<ReplicatedTextMaterializationTask>>,
            )>(local)?;
            source.charge_metadata(size_of::<crate::qwen::hybrid::PredictionUnit<B>>())?;
            let source_unit = source_specs[depth].instantiate::<B>(source_context)?;
            let mut local_unit = local_specs[depth].instantiate::<B>(execution_context)?;
            match (
                &mut local_unit.block.feed_forward,
                &construction.coordinates[depth],
            ) {
                (crate::qwen::hybrid::FeedForward::Routed(moe), Some(coordinates)) => moe
                    .bind_shared_resident_unit_coordinates(
                        coordinates.clone(),
                        construction.partitioned,
                    ),
                (crate::qwen::hybrid::FeedForward::Dense(_), None) => {}
                _ => return Err(fail(ConstructionError::Selection)),
            }
            units.push(PreparedPredictionUnit {
                source: source_unit,
                local: local_unit,
                tasks: tasks.clone(),
                source_layout: None,
                residency: selected.text_realization().residency(),
                role: PredictionModuleRole::Unit,
            });
        }
        let state = state.clone_workspace(local)?;
        controls::<PreparedPredictionUnit<crate::qwen::hybrid::PredictionShared<B>>>(local)?;
        source.charge_metadata(size_of::<crate::qwen::hybrid::PredictionShared<B>>())?;
        let shared = PreparedPredictionUnit {
            source: shared.instantiate::<B>(source_context)?,
            local: shared.instantiate::<B>(execution_context)?,
            tasks: shared_tasks.clone(),
            source_layout: None,
            residency: selected.text_realization().residency(),
            role: PredictionModuleRole::Shared,
        };
        Ok(PreparedPredictionExtension::QwenHybrid {
            layout,
            parameters,
            units,
            shared,
            state,
            construction: construction.clone(),
        })
    }
    // Preserve a separate cold construction frame in unoptimized builds too.
    #[inline(never)]
    fn inkling(
        self,
        source_specs: &crate::inkling::MtpModelSpec,
        local_specs: &crate::inkling::MtpModelSpec,
        shared_tasks: &Option<Arc<Vec<ReplicatedTextMaterializationTask>>>,
        state: &StateLayout,
    ) -> Result<PreparedPredictionExtension<B>, Error> {
        controls::<(
            Self,
            &crate::inkling::MtpModelSpec,
            &crate::inkling::MtpModelSpec,
            &Option<Arc<Vec<ReplicatedTextMaterializationTask>>>,
            &StateLayout,
            PreparedPredictionExtension<B>,
        )>(self.local)?;
        let Self {
            source_context,
            execution_context,
            source,
            local,
            selected,
            construction,
            layout,
            parameters,
        } = self;
        let fail = |cause| local.metadata_source(cause);
        let mut units = local.metadata_vec(construction.rows.len())?;
        for (depth, tasks) in construction.rows.iter().enumerate() {
            controls::<(
                PreparedPredictionUnit<crate::inkling::MtpDepth<B>>,
                usize,
                Arc<Vec<ReplicatedTextMaterializationTask>>,
            )>(local)?;
            source.charge_metadata(size_of::<crate::inkling::MtpDepth<B>>())?;
            let source_unit = source_specs
                .depth(depth)
                .ok_or_else(|| fail(ConstructionError::Selection))?
                .instantiate::<B>(source_context)?;
            let local_unit = local_specs
                .depth(depth)
                .ok_or_else(|| fail(ConstructionError::Selection))?
                .instantiate::<B>(execution_context)?;
            units.push(PreparedPredictionUnit {
                source: source_unit,
                local: local_unit,
                tasks: tasks.clone(),
                source_layout: None,
                residency: selected.text_realization().residency(),
                role: PredictionModuleRole::Unit,
            });
        }
        controls::<Option<PreparedPredictionUnit<crate::inkling::MtpShared<B>>>>(local)?;
        let shared = match shared_tasks {
            Some(tasks) => {
                source.charge_metadata(size_of::<crate::inkling::MtpShared<B>>())?;
                let source = source_specs
                    .shared::<B>(source_context)?
                    .ok_or_else(|| fail(ConstructionError::Selection))?;
                let local = local_specs
                    .shared::<B>(execution_context)?
                    .ok_or_else(|| fail(ConstructionError::Selection))?;
                Some(PreparedPredictionUnit {
                    source,
                    local,
                    tasks: tasks.clone(),
                    source_layout: None,
                    residency: selected.text_realization().residency(),
                    role: PredictionModuleRole::Shared,
                })
            }
            None => None,
        };
        let state = state.clone_workspace(local)?;
        Ok(PreparedPredictionExtension::Inkling {
            layout,
            parameters,
            units,
            shared,
            state,
            construction: construction.clone(),
        })
    }
}

impl DsparkPredictionStrategy {
    fn clone_workspace(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        controls::<(Self, &Self, crate::deepseek::DsparkConfig)>(context)?;
        Ok(Self {
            config: self.config.clone(),
            capture_policy: self.capture_policy.clone_workspace(context)?,
            hidden_size: self.hidden_size,
        })
    }
}
