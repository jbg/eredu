use super::*;
mod checkpoint;
pub(super) use checkpoint::StateCheckpoint;

// Both ordinary and original inspection read the same selected layout/offset.
// The conversion error itself is fixed; only the legacy adapter boxes it.
fn checked_prefill_frontier<S: MlxStateMechanisms>(
    state: &S,
) -> Result<Option<u64>, std::num::TryFromIntError> {
    if state.optional_layout().is_none() {
        return Ok(None);
    }
    u64::try_from(state.offset()).map(Some)
}

mod opening_sources;
mod prefill_controls;
mod prefill_entry;
use opening_sources::{NativeOpeningSourceBinding, NativeOpeningSourceGuard};

impl<A, S> eredu_runtime::replicated_session::ReplicatedTextSnapshotMechanisms<A, MlxNeuralBackend>
    for MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    fn estimate_snapshot_state(
        &self,
        state: &S,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        state.isolated_snapshot_estimate()
    }

    fn estimate_original_snapshot_state(
        &self,
        state: &S,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        state.original_isolated_snapshot_estimate()
    }

    fn estimate_reset_state(
        &self,
        state: &S,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        // The snapshot metadata bound covers empty role maps and the selected
        // layout. Growth at zero includes every declared fixed component and
        // cache capacity, even when its tensor is currently absent. Fresh MLX
        // realization uses that same geometry and starts with no token history.
        let metadata = state.isolated_snapshot_estimate()?;
        let initial = state.isolated_snapshot_growth(0)?;
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: metadata.retained_bytes.checked_add(initial)?,
            copy_bytes: metadata.copy_bytes.checked_add(initial)?,
        })
    }

    fn estimate_snapshot_growth(&self, state: &S, additional: u64) -> Option<u64> {
        state.isolated_snapshot_growth(additional)
    }

    fn copy_snapshot_state(&mut self, state: &S, context: &Stream) -> Result<S, Error> {
        let copied = state.isolated_snapshot(context)?;
        async_eval_with_event(copied.retained_arrays())?.synchronize()?;
        Ok(copied)
    }
}

pub(in crate::composition::mlx::replicated_text) struct MlxExecutionReport {
    pub(super) residency: ResidencyReport,
    pub(super) dense: Option<DenseDiskStreamReport>,
}

pub(in crate::composition::mlx::replicated_text) struct MlxStateReport {
    pub(super) residency: Option<CacheResidencyReport>,
    #[cfg(test)]
    pub(super) presence: StatePresenceSnapshot,
    #[cfg(test)]
    pub(super) fixed_numeric: FixedNumericStateSnapshot,
    #[cfg(test)]
    pub(super) retained_numeric: RetainedNumericStateSnapshot,
}

pub(in crate::composition::mlx::replicated_text) struct MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    prepared_bindings: Option<PreparedExactBindings>,
    prepared_layerwise_manager:
        Option<crate::backend::runtime::execution::generic::PreparedLayerwiseManager>,
    prediction_residency: super::super::prediction::parameters::PredictionResidency,
    prepared_parameters: Vec<eredu_runtime::parameter_operations::PreparedParameterSlot>,
    parameter_declarations: Vec<eredu_nn::ParameterMetadata>,
    resident_report: Option<ResidencyReport>,
    layerwise_workspace: Option<
        Box<
            dyn Fn(
                crate::backend::nn::workspace::NativeAllocationFacts,
            ) -> Result<
                crate::backend::runtime::execution::generic::LayerwiseWorkspace,
                Error,
            >,
        >,
    >,
    residency_manager: Option<crate::backend::runtime::residency::manager::ResidencyManager>,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    stream: Stream,
    weights_stream: Stream,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    source_parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    ignored_checkpoint_sources: std::collections::BTreeSet<String>,
    state_rank: Option<eredu_core::cache::CacheRankIdentity>,
    state_global_layer_start: usize,
    opening_sources: NativeOpeningSourceBinding,
    // Initialized at ordinary model construction, before any original request.
    prefill_roots_runtime: safemlx::PrefillRootsRuntime,
    // Ordinary model construction, before requests. Keep actual failure for
    // original projection without making an ordinary load fail or retry here.
    gguf_host_runtime:
        Result<std::rc::Rc<safemlx::PreparedInputRuntime>, eredu_core::SharedBackendFailure>,
    // Constructor coverage is partial: Device/TLS/context ownership remains
    // separate even when the allocator was genuinely admitted earlier.
    gguf_host_allocator_coverage:
        crate::backend::managed_memory::input_allocator::InputAllocatorCoverage,
    native_storage_selection: eredu_runtime::working_memory::NativeStorageSelection,
    prefill_controls: std::cell::RefCell<
        Option<crate::backend::submission_recovery::prefill::PrefillControlProjection>,
    >,
    prefill_roots: Option<crate::backend::submission_recovery::prefill::RootsProjection>,
    parallel_control: std::cell::RefCell<Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlProjection>>,
    nested_completion: std::cell::RefCell<Option<crate::backend::submission_recovery::prefill::nested::NestedCompletionProjection>>,
    // Weak host-only installation: the original quote/work owns all row payloads.
    opening_rows: std::cell::RefCell<Option<opening_sources::InstalledOpeningRows>>,
    state: PhantomData<fn() -> (A, S)>,
}

pub(in crate::composition::mlx::replicated_text) struct PreparedExactBindings {
    layout: eredu_runtime::ExecutionUnitLayout,
    addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
    static_bindings: Vec<WeightBinding>,
    unit_bindings: Vec<Vec<WeightBinding>>,
    excluded_parameters: std::collections::BTreeSet<String>,
    local_parameters: Option<std::collections::BTreeSet<String>>,
}

pub(in crate::composition::mlx::replicated_text) struct MlxPromptCacheSaveTransaction {
    publication: eredu_runtime::ReversiblePromptCachePublication,
    manifest: PromptCacheManifest,
}

impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    pub(super) fn new(
        store: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        let store = store.into();
        let prefill_roots_runtime =
            safemlx::PrefillRootsRuntime::prepare_for_stream(stream, weights_stream).map_err(
                |error| match error {
                    safemlx::PrefillRuntimePreparationError::Native(error) => Error::from(error),
                    safemlx::PrefillRuntimePreparationError::ActiveOriginalScope => {
                        Error::from(safemlx::OriginalNativeControlError::InvalidScope)
                    }
                    safemlx::PrefillRuntimePreparationError::InvalidInput => {
                        Error::from(safemlx::OriginalNativeControlError::ForeignDomain)
                    }
                    safemlx::PrefillRuntimePreparationError::InvalidStatus(status) => {
                        Error::from(safemlx::OriginalNativeControlError::InvalidStatus(status))
                    }
                },
            )?;
        let (gguf_host_runtime, gguf_host_allocator_coverage) =
            match crate::backend::managed_memory::input_allocator::prepare_ordinary() {
                Ok((runtime, coverage)) => (Ok(runtime), coverage),
                Err(cause) => (Err(cause), crate::backend::managed_memory::input_allocator::InputAllocatorCoverage::Ordinary),
            };
        Ok(Self {
            store,
            prepared_bindings: None,
            prepared_layerwise_manager: None,
            prediction_residency: Default::default(),
            prepared_parameters: Vec::new(),
            parameter_declarations: Vec::new(),
            resident_report: None,
            layerwise_workspace: None,
            residency_manager: None,
            materialization: None,
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            parallel_layout: None,
            source_parallel_layout: None,
            ignored_checkpoint_sources: std::collections::BTreeSet::new(),
            state_rank: None,
            state_global_layer_start: 0,
            opening_sources: NativeOpeningSourceBinding::default(),
            prefill_roots_runtime,
            gguf_host_runtime: gguf_host_runtime.map(std::rc::Rc::new).map_err(|cause| {
                // One ordinary cold allocation owns the actual Exception.
                // Later request inspections retain this same source only.
                eredu_core::SharedBackendFailure::new(eredu_core::BackendFailureKind::Other, cause)
            }),
            gguf_host_allocator_coverage,
            native_storage_selection: Default::default(),
            prefill_controls: std::cell::RefCell::new(None),
            prefill_roots: None,
            parallel_control: std::cell::RefCell::new(None),
            nested_completion: std::cell::RefCell::new(None),
            opening_rows: std::cell::RefCell::new(None),
            state: PhantomData,
        })
    }

    /// Prepares only the private source handoff. No production caller enables it
    /// until original collector/control pricing and recovery ownership are bound.
    pub(in crate::composition::mlx::replicated_text) fn bind_opening_sources(
        &mut self,
        request: &eredu_runtime::working_memory::InferenceRequest,
    ) -> Result<NativeOpeningSourceGuard, Error> {
        self.opening_sources.bind(request)
    }

    pub(super) fn set_prediction_residency(
        &mut self,
        residency: super::super::prediction::parameters::PredictionResidency,
    ) {
        self.prediction_residency = residency;
    }

    pub(super) fn retained_storage(
        &self,
        state: &S,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(state, &mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    pub(super) fn collect_retained_storage(
        &self,
        state: &S,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        self.collect_retained_parameter_storage(storage)?;
        state.collect_retained_storage(storage)
    }

    pub(super) fn retained_parameter_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_parameter_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    pub(super) fn collect_retained_parameter_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        let manager = self.residency_manager.as_ref().ok_or_else(|| {
            Error::ArchitectureModel("target residency manager was not retained".into())
        })?;
        manager.collect_retained_storage(storage)?;
        storage.include_checkpoint_source(self.store.as_ref())?;
        Ok(())
    }

    pub(super) fn layerwise_workspace(
        &self,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
    ) -> Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error> {
        self.layerwise_workspace.as_ref().ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))
        })?(allocation)
    }

    pub(super) fn set_prepared_layerwise_manager(
        &mut self,
        manager: Option<crate::backend::runtime::execution::generic::PreparedLayerwiseManager>,
    ) {
        self.prepared_layerwise_manager = manager;
    }

    pub(super) fn set_parallel_layout(&mut self, layout: eredu_runtime::LocalModelLayout) {
        self.parallel_layout = Some(layout);
    }

    pub(super) fn set_source_parallel_layout(
        &mut self,
        layout: Option<eredu_runtime::LocalModelLayout>,
    ) {
        self.source_parallel_layout = layout;
    }

    pub(super) fn set_ignored_checkpoint_sources(
        &mut self,
        sources: std::collections::BTreeSet<String>,
    ) {
        self.ignored_checkpoint_sources = sources;
    }

    pub(super) fn set_state_partition(
        &mut self,
        rank: eredu_core::cache::CacheRankIdentity,
        global_layer_start: usize,
    ) {
        self.state_rank = Some(rank);
        self.state_global_layer_start = global_layer_start;
    }

    pub(super) fn apply_selected_transforms(
        &mut self,
        target_architecture: &A,
        target_units: &[A::Unit],
        source_architecture: Option<&A>,
        source_units: Option<&[A::Unit]>,
        source_layout: Option<&eredu_runtime::LocalModelLayout>,
        tasks: &[ReplicatedTextMaterializationTask],
    ) -> Result<(), Error> {
        let task_groups = eredu_runtime::group_replicated_text_transform_tasks(tasks)
            .map_err(|error| Error::Quantization(error.to_string()))?;
        if task_groups.is_empty() {
            if source_architecture.is_some() || source_units.is_some() {
                return Err(Error::Quantization(
                    "selected source architecture has no materialization tasks".into(),
                ));
            }
            return Ok(());
        }
        let source = source_architecture.ok_or_else(|| {
            Error::Quantization("selected transform tasks have no source architecture".into())
        })?;
        let source_units = source_units.ok_or_else(|| {
            Error::ArchitectureModel(
                "selected source architecture has no neutral materialization units".into(),
            )
        })?;
        if source_units.len() != target_units.len() {
            return Err(Error::Quantization(
                "selected materialization tasks changed the local execution-unit cardinality"
                    .into(),
            ));
        }
        let source_static = source.static_modules();
        let target_static = target_architecture.static_modules();
        let mut combined = eredu_runtime::WeightMaterializationReport::default();
        for group in task_groups {
            let exact_tasks = group
                .tasks(tasks)
                .map_err(|error| Error::Quantization(error.to_string()))?;
            #[cfg(test)]
            crate::tests::support::path_instrumentation::materialization();
            let (store, report) = if let Some(manager) = &mut self.prepared_layerwise_manager {
                crate::backend::runtime::execution::layerwise::adopt_exact_replicated_text_quantization(
                    manager.take_conversion()?, self.store.clone(), source_static, target_static,
                    source_units, target_units, source_layout, group.quantization(), &exact_tasks,
                )?
            } else {
                quantize_exact_replicated_text_tasks(
                    self.store.clone(),
                    source_static,
                    target_static,
                    source_units,
                    target_units,
                    source_layout,
                    group.quantization(),
                    &exact_tasks,
                    &self.stream,
                )?
            };
            self.store = store;
            combined.merge(report);
        }
        self.materialization = Some(combined);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_local_partition_materialization_with_addressable_parameters(
        &mut self,
        architecture: &A,
        source_architecture: Option<&A>,
        global_layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
        task_plan: &eredu_runtime::ReplicatedTextMaterializationPartitionPlan,
        units: &[A::Unit],
        source_units: Option<&[A::Unit]>,
        source_layout: Option<&eredu_runtime::LocalModelLayout>,
        tasks: &[ReplicatedTextMaterializationTask],
        addressable_parameters: &std::collections::BTreeSet<String>,
    ) -> Result<(), Error> {
        if addresses.len() != units.len() || addresses.is_empty() {
            return Err(Error::ArchitectureModel(
                "local partition addresses and constructed units differ".into(),
            ));
        }
        let _ = global_layout;
        let static_tasks = task_plan
            .static_tasks(tasks)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let unit_tasks = task_plan
            .unit_tasks(tasks)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        self.apply_selected_transforms(
            architecture,
            units,
            source_architecture,
            source_units,
            source_layout,
            tasks,
        )?;
        let selected_static_parameters = static_tasks
            .iter()
            .flat_map(|task| {
                std::iter::once(task.name().to_owned()).chain(
                    task.output_companions()
                        .iter()
                        .map(|companion| companion.name().to_owned()),
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        // Composite family modules may retain lazy handles for pinned roles
        // owned by another pipeline rank. The architecture-selected tasks are
        // the sole ownership authority: expose precisely those targets to the
        // exact binder and keep every selected target mandatory.
        let mut excluded_parameters = neutral_parameter_refs(architecture.static_modules(), false)
            .flatten()
            .into_keys()
            .map(|name| name.as_ref().to_owned())
            .filter(|name| !selected_static_parameters.contains(name))
            .collect::<std::collections::BTreeSet<_>>();
        #[cfg(test)]
        let excluded_static_parameters = excluded_parameters.len();
        excluded_parameters.extend(addressable_parameters.iter().cloned());
        let static_bindings = build_mlx_exact_replicated_text_bindings(
            architecture.static_modules(),
            self.store.as_ref(),
            &static_tasks,
            &excluded_parameters,
            self.parallel_layout.as_ref(),
        )?;
        #[cfg(test)]
        crate::tests::support::path_instrumentation::local_static_materialization(
            static_bindings.len(),
            excluded_static_parameters,
        );
        let unit_bindings = units
            .iter()
            .zip(&unit_tasks)
            .map(|(unit, tasks)| {
                #[cfg(test)]
                crate::tests::support::path_instrumentation::unit_construction();
                build_mlx_exact_replicated_text_bindings(
                    unit,
                    self.store.as_ref(),
                    tasks,
                    addressable_parameters,
                    self.parallel_layout.as_ref(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let graph = architecture
            .execution_graph()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?.into_owned();
        let layout = eredu_runtime::partitioned_materialization_unit_layout(&graph, addresses)
            .map_err(Error::ArchitectureModel)?;
        (self.prepared_parameters, self.parameter_declarations) =
            super::prepared_parameters::collect::<A, S>(
                architecture,
                units,
                addresses,
                &static_bindings,
                &unit_bindings,
                self.store.as_ref(),
            )?;
        self.prepared_bindings = Some(PreparedExactBindings {
            layout,
            addresses: addresses.to_vec(),
            static_bindings,
            unit_bindings,
            excluded_parameters,
            local_parameters: Some(
                tasks
                    .iter()
                    .flat_map(|task| {
                        std::iter::once(task.name().to_owned()).chain(
                            task.output_companions()
                                .iter()
                                .map(|companion| companion.name().to_owned()),
                        )
                    })
                    .collect(),
            ),
        });
        Ok(())
    }

    pub(super) fn take_prepared_policy(
        &mut self,
        architecture: &mut A,
        selected: &SelectedReplicatedTextRealization,
    ) -> Result<
        (
            MlxLayerwisePolicy<A::Unit, MlxSelectiveUnitPopulator>,
            eredu_runtime::ExecutionUnitLayout,
            Vec<eredu_runtime::ExecutionUnitAddress>,
        ),
        Error,
    >
    where
        A::Unit: 'static,
    {
        let prepared = self.prepared_bindings.take().ok_or_else(|| {
            Error::ArchitectureModel("execution policy requested before materialization".into())
        })?;
        let layout = prepared.layout.clone();
        let mut ignored_sources = self.ignored_checkpoint_sources.clone();
        ignored_sources.extend(
            selected
                .requirements()
                .parameters()
                .iter()
                .flat_map(|parameter| parameter.admitted_redundant_sources().iter().cloned()),
        );
        for parameter in selected
            .requirements()
            .parameters()
            .iter()
            .filter(|parameter| {
                prepared.excluded_parameters.contains(parameter.name())
                    || prepared
                        .local_parameters
                        .as_ref()
                        .is_some_and(|local| !local.contains(parameter.name()))
            })
        {
            ignored_sources.extend(parameter.sources().iter().cloned());
            if let Some(recipe) = selected
                .requirements()
                .derived_recipes()
                .get(parameter.name())
            {
                ignored_sources.extend(recipe.source_keys().into_iter().map(str::to_owned));
            }
        }
        let populator = match self.prepared_layerwise_manager.as_ref() {
            Some(manager) => MlxSelectiveUnitPopulator::from_prepared(
                manager.parameter_exclusions(&prepared.excluded_parameters)?),
            None => MlxSelectiveUnitPopulator::new(prepared.excluded_parameters.clone()),
        };
        let (policy, _) = crate::backend::runtime::execution::generic::prepare_layerwise_policy_with_prepared_manager(
            self.store.clone(),
            architecture,
            populator,
            PhantomData::<S>,
            selected.residency(),
            &self.stream,
            &self.weights_stream,
            move |key| ignored_sources.contains(key),
            prepared.layout,
            prepared.static_bindings,
            prepared.unit_bindings,
            std::mem::take(&mut self.prediction_residency.units),
            self.prepared_layerwise_manager.take(),
        )?;
        self.prediction_residency
            .install(policy.residency_manager())?;
        self.residency_manager = Some(policy.residency_manager().clone());
        Ok((policy, layout, prepared.addresses))
    }
}

impl<A, S> ReplicatedTextSessionMechanisms<A, MlxNeuralBackend>
    for MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Error: std::fmt::Display,
    A::Unit: 'static,
{
    fn requires_prefill_opening_sources(&self) -> bool {
        self.opening_sources.is_active()
            || self
                .opening_rows
                .borrow()
                .as_ref()
                .is_some_and(|r| r.is_live())
    }

    fn prepare_prefill_opening_sources(
        &mut self,
        state: &Self::State,
        context: &eredu_runtime::inspection::PrefillChunkRetentionContext<'_>,
        execution: &eredu_runtime::inspection::PrefillOpeningExecution<'_, MlxTensor>,
    ) -> Result<(), Self::Error> {
        let rows = {
            let slot = self
                .opening_rows
                .try_borrow()
                .map_err(|e| Error::Other(Box::new(e)))?;
            slot.as_ref().and_then(|rows| rows.upgrade())
        };
        if let Some(rows) = rows {
            let manager = self.residency_manager.as_ref().ok_or_else(|| {
                Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                ))
            })?;
            return rows.collect(state, execution, context, self.store.as_ref(), manager);
        }
        self.opening_sources.collect(
            state,
            context,
            || self.retained_parameter_storage(),
            |visitor| execution.visit(visitor),
        )
    }

    fn requires_prepared_prefill_sources(&self) -> bool {
        self.opening_rows
            .borrow()
            .as_ref()
            .is_some_and(|r| r.is_live())
    }

    fn prepare_prefill_retirement_sources(
        &mut self,
        state: &S,
        ticket: &eredu_runtime::inspection::SettledPrefillChunkRetention,
        execution: &eredu_runtime::inspection::PrefillOpeningExecution<'_, MlxTensor>,
    ) -> Result<(), Self::Error> {
        let rows = {
            let slot = self
                .opening_rows
                .try_borrow()
                .map_err(|e| Error::Other(Box::new(e)))?;
            slot.as_ref().and_then(|rows| rows.upgrade())
        };
        if let Some(rows) = rows {
            let manager = self.residency_manager.as_ref().ok_or_else(|| {
                Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                ))
            })?;
            return rows.collect_completed(state, execution, ticket, self.store.as_ref(), manager);
        }
        Ok(())
    }

    fn with_execution_parallel_control<T,E,F>(
        &self,event:eredu_runtime::replicated_session::ParallelControlEvent,context:&Stream,run:F,
    )->Result<Result<T,E>,eredu_core::BackendFailure>
    where F:FnOnce(Option<(&<MlxNeuralBackend as NeuralBackend>::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)->Result<T,E>, {
        let projection={
            let slot=self.parallel_control.try_borrow()
                .map_err(|_|eredu_core::PreparedRequestRejection::Busy.into_backend_failure())?;
            slot.as_ref().map(|value|value.retained()).transpose().map_err(Error::into_backend_failure)?
        };
        match projection {Some(projection)=>projection.run(event,context,run),None=>Ok(run(None))}
    }

    fn with_execution_parallel_control_context<T,E,F>(
        &self,_context:&Stream,run:F,
    )->Result<Result<T,E>,Self::Error>
    where F:FnOnce(Option<(&mut Option<Box<<MlxNeuralBackend as NeuralBackend>::ParallelContext>>,&eredu_nn::workspace::HostMetadataFunding)>)->Result<T,E>,
    {
        let control={
            let slot=self.parallel_control.try_borrow().map_err(|_|
                Error::with_original_control_source(eredu_core::PreparedRequestRejection::Busy.into_backend_failure(),false))?;
            slot.as_ref().map(|control|control.retained()).transpose()?
        };
        match control {
            Some(control)=>{
                let roots=self.prefill_roots.as_ref().map(|roots|roots.transient_roots()).transpose()?;
                let binding=match &self.prefill_roots {
                    Some(roots)=>roots.with_parallel(|parallel|parallel
                        .map(|(parallel,_)|parallel.boundary_binding()).transpose())?,
                    None=>None,
                };
                control.with_request_context(roots,binding,run)
            },
            None=>Ok(run(None)),
        }
    }

    fn with_execution_parallel<T,E,F>(
        &self,context:&Stream,run:F,
    )->Result<Result<T,E>,Self::Error>
    where F:FnOnce(Option<(&mut <MlxNeuralBackend as NeuralBackend>::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)->Result<T,E>,
    {
        // Release the mechanism slot loan before model work or nested policy
        // calls. The copied view is weak and shares the submission activation.
        let control={
            let slot=self.parallel_control.try_borrow().map_err(|_|
                Error::with_original_control_source(eredu_core::PreparedRequestRejection::Busy.into_backend_failure(),false))?;
            slot.as_ref().map(|control|control.retained()).transpose()?
        };
        match &self.prefill_roots {
            Some(roots)=>roots.with_addressable(|| roots.with_parallel(|parallel|match parallel {
                Some((parallel,observer))=>{
                    let neural=parallel.has_neural_context();
                    parallel.with_model_context(observer,context,control.as_ref(),roots.transient_roots()?,
                        |parallel,funding|run(neural.then_some((parallel,funding))))
                },
                None=>Ok(run(None)),
            })),
            None=>Ok(run(None)),
        }
    }

    fn with_execution_parallel_publication<T,E,F>(&self,context:&Stream,run:F)
        ->Result<Result<T,E>,Self::Error>
    where F:FnOnce(Option<(&<MlxNeuralBackend as NeuralBackend>::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)->Result<T,E> {
        match &self.prefill_roots {
            Some(roots)=>roots.with_parallel_publication(context,run),
            None=>Ok(run(None)),
        }
    }

    fn prefill_state_frontier(&self, state: &S) -> Result<Option<u64>, Self::Error> {
        checked_prefill_frontier(state).map_err(|error| Error::Other(Box::new(error)))
    }

    fn original_prefill_state_frontier(
        &self,
        state: &S,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
        checked_prefill_frontier(state)
            .map_err(|_| eredu_runtime::working_memory::WorkingMemoryError::Overflow)
    }

    type PrefillReservationGuard = crate::backend::submission_recovery::prefill::ReservationGuard;

    fn begin_prefill_reservation(
        &mut self,
        reservation: eredu_runtime::working_memory::InferenceRequest,
    ) -> Result<Self::PrefillReservationGuard, Self::Error> {
        self.enter_prefill_retention(reservation, None, false)
    }

    fn begin_prefill_control(
        &mut self,
        reservation: eredu_runtime::working_memory::InferenceRequest,
        role: eredu_runtime::prefill::PrefillControlRole,
    ) -> Result<Self::PrefillReservationGuard, Self::Error> {
        self.enter_prefill_retention(reservation, Some(role), false)
    }

    fn coordinate_prefill_entry(
        &mut self,
        reservation: eredu_runtime::working_memory::InferenceRequest,
        role: Option<eredu_runtime::prefill::PrefillControlRole>,
    ) -> Result<Self::PrefillReservationGuard, Self::Error> {
        self.enter_prefill_retention(reservation, role, true)
    }

    fn finish_prefill_reservation(
        &mut self,
        guard: Self::PrefillReservationGuard,
    ) -> Result<(), Self::Error> {
        let guard = match guard.into_native() {
            crate::backend::submission_recovery::prefill::NativeReservationGuard::Model(guard) => {
                let result = guard.finish();
                drop(self.prefill_roots.take());
                return result;
            }
            crate::backend::submission_recovery::prefill::NativeReservationGuard::Prefill(guard) => guard,
        };
        let status = crate::backend::submission_recovery::prefill::finish(guard);
        // End the mechanism's temporary projection after its actual guard. A
        // quarantined Recovery keeps the strong payload on failed settlement.
        let expired = self.prefill_roots.take();
        drop(expired);
        let status = status?;
        if status.failed || status.blocked || !status.settled {
            return Err(Error::ArchitectureModel(
                "reserved inference work failed or remains unobservable".into(),
            ));
        }
        Ok(())
    }

    type State = S;
    type PolicyError = Error;
    type ResidentPolicy = MlxArchitectureLayerwisePolicy<A, S>;
    type BoundedPolicy = MlxArchitectureLayerwisePolicy<A, S>;
    type StateCheckpoint = StateCheckpoint<S>;
    type StateReport = MlxStateReport;
    type ExecutionReport = MlxExecutionReport;
    type Error = Error;

    fn parameter_declarations(&self) -> &[eredu_nn::ParameterMetadata] {
        &self.parameter_declarations
    }

    fn prepared_parameter_slots(
        &self,
    ) -> &[eredu_runtime::parameter_operations::PreparedParameterSlot] {
        &self.prepared_parameters
    }

    fn take_materialization_report(
        &mut self,
    ) -> Result<Option<eredu_runtime::WeightMaterializationReport>, Self::Error> {
        Ok(self.materialization.take())
    }

    fn configure_partition(
        &mut self,
        target_layout: eredu_runtime::LocalModelLayout,
        source_layout: Option<eredu_runtime::LocalModelLayout>,
        rank: eredu_core::cache::CacheRankIdentity,
        global_layer_start: usize,
    ) {
        self.set_parallel_layout(target_layout);
        self.set_source_parallel_layout(source_layout);
        self.set_state_partition(rank, global_layer_start);
    }

    fn prepare_partition_materialization(
        &mut self,
        architecture: &mut A,
        global_layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
        task_partition: &eredu_runtime::ReplicatedTextMaterializationPartitionPlan,
        units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[ReplicatedTextMaterializationTask],
        addressable_parameters: &[String],
        _context: &Stream,
    ) -> Result<(), Self::Error> {
        let addressable_parameters = addressable_parameters
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let source_layout = self.source_parallel_layout.clone();
        self.prepare_local_partition_materialization_with_addressable_parameters(
            architecture,
            source_architecture.as_deref(),
            global_layout,
            addresses,
            task_partition,
            units,
            source_units.as_deref(),
            source_layout.as_ref(),
            tasks,
            &addressable_parameters,
        )
    }

    fn prepare_materialization(
        &mut self,
        architecture: &mut A,
        target_layout: &eredu_runtime::ExecutionUnitLayout,
        target_units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[ReplicatedTextMaterializationTask],
        addressable_parameters: &[String],
        _context: &Stream,
    ) -> Result<(), Self::Error> {
        if target_units.len() != target_layout.len() {
            return Err(Error::ArchitectureModel(
                "neutral target unit set differs from the selected execution layout".into(),
            ));
        }
        let source_layout = self.source_parallel_layout.clone();
        self.apply_selected_transforms(
            architecture,
            target_units,
            source_architecture.as_deref(),
            source_units.as_deref(),
            source_layout.as_ref(),
            tasks,
        )?;

        let task_plan =
            eredu_runtime::plan_replicated_text_materialization_tasks(tasks, target_layout)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let static_tasks = task_plan
            .static_tasks(tasks)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let unit_tasks = task_plan
            .unit_tasks(tasks)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let addressable_parameters = addressable_parameters
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let static_bindings = build_mlx_exact_replicated_text_bindings(
            architecture.static_modules(),
            self.store.as_ref(),
            &static_tasks,
            &addressable_parameters,
            self.parallel_layout.as_ref(),
        )?;
        let unit_bindings = target_units
            .iter()
            .zip(&unit_tasks)
            .enumerate()
            .map(|(ordinal, (unit, tasks))| {
                #[cfg(test)]
                crate::tests::support::path_instrumentation::unit_construction();
                build_mlx_exact_replicated_text_bindings(
                    unit,
                    self.store.as_ref(),
                    tasks,
                    &addressable_parameters,
                    self.parallel_layout.as_ref(),
                )
                .map_err(|error| {
                    Error::ArchitectureModel(format!(
                        "execution unit {ordinal} exact bindings failed: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let addresses = (0..target_layout.len())
            .map(|ordinal| {
                target_layout
                    .address(ordinal)
                    .expect("validated selected layout")
            })
            .collect::<Vec<_>>();
        (self.prepared_parameters, self.parameter_declarations) =
            super::prepared_parameters::collect::<A, S>(
                architecture,
                target_units,
                &addresses,
                &static_bindings,
                &unit_bindings,
                self.store.as_ref(),
            )?;
        self.prepared_bindings = Some(PreparedExactBindings {
            layout: target_layout.clone(),
            addresses,
            static_bindings,
            unit_bindings,
            excluded_parameters: addressable_parameters,
            local_parameters: None,
        });
        Ok(())
    }

    fn realize_state(
        &mut self,
        selected: &SelectedStateRealization,
        _context: &Stream,
    ) -> Result<S, Error> {
        S::realize(selected, self.state_rank, self.state_global_layer_start)
    }

    fn resident_policy(
        &mut self,
        architecture: &mut A,
        units: Vec<A::Unit>,
        selected: &SelectedReplicatedTextRealization,
        context: &Stream,
    ) -> Result<Self::ResidentPolicy, Self::Error> {
        let (policy, layout, addresses) = self.take_prepared_policy(architecture, selected)?;
        let resident = policy.into_resident_units(units, context)?;
        self.resident_report = Some(resident.residency_report()?);
        MlxSelectedLayerwisePolicy::resident(resident, &layout, &addresses)
    }

    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        selected: &SelectedReplicatedTextRealization,
        _context: &Stream,
    ) -> Result<Self::BoundedPolicy, Self::Error> {
        let (policy, layout, addresses) = self.take_prepared_policy(architecture, selected)?;
        let policy = MlxSelectedLayerwisePolicy::bounded(policy, &layout, &addresses)?;
        if matches!(
            selected.residency(),
            eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
        ) {
            let retained = policy.clone();
            self.layerwise_workspace = Some(Box::new(move |allocation| {
                retained.layerwise_workspace(allocation)
            }));
        }
        Ok(policy)
    }

    fn index_text_output(
        &mut self,
        output: MlxTensor,
        sequence_index: i32,
        context: &Stream,
    ) -> Result<MlxTensor, Error> {
        use eredu_nn::Tensor;
        let [_, positions, _] = output.shape() else {
            return Err(Error::ArchitectureModel("text output must have three axes".into()));
        };
        let start = if sequence_index < 0 {
            positions.checked_add(sequence_index)
        } else {
            Some(sequence_index)
        }.ok_or_else(|| Error::ArchitectureModel("text output index overflow".into()))?;
        let end = start.checked_add(1)
            .ok_or_else(|| Error::ArchitectureModel("text output index overflow".into()))?;
        let indexed = output.narrow_axis(1, start, end, context)?.squeeze_axes(&[1], context)?;
        // The model root is complete, but selecting the row introduces lazy views.
        // Finish that operation while the enclosing prefill reservation still
        // owns it, before publishing the output and its physical allocation.
        indexed.as_array().evaluated()?;
        Ok(indexed)
    }

    fn copy_checkpoint_state(
        &mut self,
        state: &S,
        _context: &Stream,
    ) -> Result<StateCheckpoint<S>, Error> {
        state
            .deep_checkpoint()
            .map(StateCheckpoint::ordinary)
            .map_err(Into::into)
    }

    fn restore_checkpoint_state(
        &mut self,
        state: &mut S,
        checkpoint: StateCheckpoint<S>,
        context: &Stream,
    ) -> Result<(), Error> {
        checkpoint.restore(state, context)
    }

    fn fork_prediction_target_state(
        &mut self,
        state: &S,
        _selected: &SelectedStateRealization,
        context: &Stream,
    ) -> Result<S, Error> {
        fork_mlx_prediction_target_state(state, context)
    }

    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        selected: &SelectedStateRealization,
        context: &Stream,
    ) -> Result<(S, PromptCacheManifest), Error> {
        let directory = eredu_runtime::prompt_cache_rank_path(directory, expected.topology());
        S::load_prompt_cache(
            selected,
            &directory,
            expected,
            identity,
            prefix_token_ids,
            context,
        )
    }

    fn save_prompt_cache(
        &mut self,
        state: &mut S,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _context: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let destination = eredu_runtime::prompt_cache_rank_path(destination, descriptor.topology());
        S::save_prompt_cache(state, &destination, descriptor, prefix_token_ids, options)
    }

    fn state_report(&self, state: &S) -> Result<Self::StateReport, Error> {
        Ok(MlxStateReport {
            residency: state.residency_report()?,
            #[cfg(test)]
            presence: S::state_snapshot(state),
            #[cfg(test)]
            fixed_numeric: S::fixed_numeric_snapshot(state)?,
            #[cfg(test)]
            retained_numeric: S::retained_numeric_snapshot(state)?,
        })
    }

    fn execution_report(
        &self,
        _residency: eredu_runtime::LayerWeightResidency,
        bounded: Option<&Self::BoundedPolicy>,
    ) -> Result<Self::ExecutionReport, Error> {
        match bounded {
            Some(policy) => Ok(MlxExecutionReport {
                residency: policy.residency_report()?,
                dense: policy.dense_stream_report()?,
            }),
            None => Ok(MlxExecutionReport {
                residency: self.resident_report.clone().ok_or_else(|| {
                    Error::ArchitectureModel("resident report was not captured".into())
                })?,
                dense: None,
            }),
        }
    }

    fn supports_media_ingress_completion(&self) -> bool {
        true
    }

    fn complete_media_ingress(
        &mut self,
        output: Option<&MlxTensor>,
        state: &S,
        roots: &eredu_runtime::media_prefill::RetainedMediaRoots<'_, MlxTensor>,
        _context: &Stream,
    ) -> Option<Result<(), Error>> {
        match self.active_nested_completion() {
            Ok(Some(nested)) => return Some(self.complete_nested_roots(&nested, output, state, Some(roots), _context)),
            Err(cause) => return Some(Err(cause)),
            Ok(None) => {}
        }
        if self.prefill_roots.is_some() {
            #[cfg(test)]
            crate::tests::support::media_completion::record(roots, false);
            let result = self.complete_prefill_roots(output, state, Some(roots), _context);
            #[cfg(test)]
            if result.is_ok() {
                crate::tests::support::media_completion::record(roots, true);
            }
            return Some(result);
        }
        // Callback-local root borrows cannot escape. These ordinary native
        // handle clones keep every future root alive through the same event.
        let token_validations = active_token_validation_arrays();
        let mut retained = output
            .into_iter()
            .map(MlxTensor::as_array)
            .chain(state.retained_arrays())
            .chain(token_validations.iter())
            .cloned()
            .collect::<Vec<_>>();
        roots.visit(&mut |root| retained.push(root.as_array().clone()));
        #[cfg(test)]
        crate::tests::support::media_completion::record(roots, false);
        let result = complete_root_arrays(retained.iter());
        #[cfg(test)]
        if result.is_ok() {
            crate::tests::support::media_completion::record(roots, true);
        }
        Some(result)
    }

    fn complete(
        &mut self,
        output: Option<&MlxTensor>,
        state: &S,
        _context: &Stream,
    ) -> Result<(), Error> {
        if let Some(roots) = self.active_nested_completion()? {
            return self.complete_nested_roots(&roots, output, state, None, _context);
        }
        if self.prefill_roots.is_some() {
            return self.complete_prefill_roots(output, state, None, _context);
        }
        let token_validations = active_token_validation_arrays();
        let retained = output
            .into_iter()
            .map(MlxTensor::as_array)
            .chain(state.retained_arrays())
            .chain(token_validations.iter())
            .collect::<Vec<_>>();
        complete_root_arrays(retained.iter().copied())
    }
}

// Both ordinary and retained-media completion have one native boundary. The
// iterator only borrows already-owned roots; native VectorArray/Event creation
// remains ordinary work under the surrounding operation's actual owner.
fn complete_root_arrays<'a>(arrays: impl Iterator<Item = &'a Array> + Clone) -> Result<(), Error> {
    #[cfg(test)]
    crate::tests::support::path_instrumentation::completion();
    async_eval_with_event(arrays.clone())?.synchronize()?;
    for array in arrays {
        array.evaluated()?;
    }
    validate_active_token_validations().map_err(Into::into)
}

impl<A, S> TransactionalPromptCacheMechanisms<A, MlxNeuralBackend>
    for MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Error: std::fmt::Display,
    A::Unit: 'static,
{
    type PromptCacheSaveTransaction = MlxPromptCacheSaveTransaction;

    fn prepare_prompt_cache_save(
        &mut self,
        state: &mut S,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _context: &Stream,
    ) -> Result<Self::PromptCacheSaveTransaction, Error> {
        let destination = eredu_runtime::prompt_cache_rank_path(destination, descriptor.topology());
        let publication = eredu_runtime::ReversiblePromptCachePublication::begin(
            &destination,
            options.replace_existing(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let staging_options =
            PromptCacheOptions::new(options.application_namespace().map(str::to_owned), false)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let manifest = S::save_prompt_cache(
            state,
            publication.staging_destination(),
            descriptor,
            prefix_token_ids,
            &staging_options,
        )?;
        Ok(MlxPromptCacheSaveTransaction {
            publication,
            manifest,
        })
    }

    fn prepared_prompt_cache_manifest(
        transaction: &Self::PromptCacheSaveTransaction,
    ) -> &PromptCacheManifest {
        &transaction.manifest
    }

    fn publish_prompt_cache_save(
        &mut self,
        transaction: &mut Self::PromptCacheSaveTransaction,
    ) -> Result<(), Error> {
        transaction
            .publication
            .publish()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn commit_prompt_cache_save(&mut self, transaction: Self::PromptCacheSaveTransaction) {
        transaction.publication.commit().unwrap_or_else(|error| {
            panic!("committed prompt-cache publication cleanup failed: {error}")
        });
    }

    fn rollback_prompt_cache_save(&mut self, transaction: Self::PromptCacheSaveTransaction) {
        transaction
            .publication
            .rollback()
            .unwrap_or_else(|error| panic!("prompt-cache publication rollback failed: {error}"));
    }
}

/// Native recovery owns this charge even when the session or driver is dropped.
#[cfg(test)]
pub(crate) struct PrefillReservationRetention(
    #[allow(dead_code)] eredu_runtime::working_memory::InferenceRequest,
);
#[cfg(test)]
impl crate::backend::submission_recovery::Retention for PrefillReservationRetention {
    fn observe(&self, _: crate::backend::submission_recovery::Status) {}
}

#[cfg(test)]
mod prefill_reservation_tests {
    use super::PrefillReservationRetention;
    use crate::backend::submission_recovery::{self, Probe, Recovery, Status};
    use eredu_core::*;
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
    use std::{cell::Cell, rc::Rc};

    struct Deferred(Rc<Cell<Status>>);
    impl Probe for Deferred {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            self.0.get()
        }
    }

    fn retention_admission() -> Admission {
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 3,
            max_output_tokens: 1,
            prefill_chunk_positions: 2,
            output: OutputDemand::LastPosition,
        };
        let request = AdmissionRequest {
            input: InputTokenCount::text(3),
            max_output_tokens: 1,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        };
        let capabilities = ModelCapabilities {
            effective_model_type: "native retention fixture".into(),
            native_max_context: Observed::exact(8, "fixture"),
            effective_max_context: Observed::exact(8, "fixture"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::PersistentStateOnly,
        };
        let layout = StateMemoryLayout::new(
            LayerSchedule::empty(),
            vec![],
            1,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        let bound = || WorkspaceBound::bounded(16, "retention fixture charge");
        let state = estimate_runtime_state(
            &layout,
            request.input,
            1,
            1,
            std::num::NonZeroU8::new(4).unwrap(),
        )
        .unwrap()
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(),
            attention: bound(),
            vocabulary: bound(),
            state_update: bound(),
            materialization: bound(),
            retained: bound(),
        })
        .unwrap();
        let AdmissionResult::Admitted(admission) =
            apply_admission_policy(&capabilities, request, state, None).unwrap()
        else {
            panic!("fixture admission")
        };
        admission
    }

    #[test]
    fn reservation_remains_charged_after_unobservable_native_scope_is_dropped() {
        let admission = retention_admission();
        for blocked in [false, true] {
            let pool = WorkingMemoryPool::new(admission.incremental_required_bytes, 0).unwrap();
            let reservation = pool
                .reserve(&InferenceExecutionIdentity::default(), &admission)
                .unwrap();
            let charge = reservation.bytes();
            let status = Rc::new(Cell::new(Status {
                settled: false,
                failed: !blocked,
                blocked,
            }));
            let recovery = Recovery::with_probe(
                PrefillReservationRetention(reservation.into()),
                Deferred(status.clone()),
            );
            let observed = recovery.finish().unwrap();
            assert!(!observed.settled);
            assert_eq!(
                pool.used_bytes().unwrap(),
                charge,
                "scope failure cannot refund native authority"
            );
            assert!(matches!(
                pool.reserve(&InferenceExecutionIdentity::default(), &admission),
                Err(eredu_runtime::working_memory::WorkingMemoryError::BudgetExceeded { .. })
            ));
            status.set(Status {
                settled: true,
                failed: !blocked,
                blocked: false,
            });
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while pool.used_bytes().unwrap() != 0 && std::time::Instant::now() < deadline {
                submission_recovery::reap();
                std::thread::yield_now();
            }
            assert_eq!(
                pool.used_bytes().unwrap(),
                0,
                "independent settlement releases the retained charge"
            );
            assert!(pool.peak_bytes().unwrap() >= charge);
        }
    }

    #[test]
    #[ignore = "requires native CPU execution"]
    fn native_state_copies_keep_request_charges_after_parent_and_driver_drop() {
        use super::*;
        use crate::backend::runtime::cache::kv::KeyValueCache;
        use eredu_core::cache::{
            LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
            StateTensorPolicy, StateTensorRole,
        };
        use eredu_nn::PoolingAttentionCache;
        use eredu_runtime::working_memory::InferenceRequest;
        use eredu_runtime::{RuntimeStateComponents, StateLayout};

        fn values<S: MlxStateMechanisms>(state: &S, stream: &Stream) -> Vec<(Vec<i32>, Vec<f32>)> {
            state
                .retained_arrays()
                .into_iter()
                .map(|array| {
                    let contiguous = array.contiguous(false, stream).unwrap();
                    let evaluated = contiguous.evaluated().unwrap();
                    (array.shape().to_vec(), evaluated.as_slice::<f32>().to_vec())
                })
                .collect()
        }

        fn check<S: MlxStateMechanisms>(mut state: S, stream: &Stream) {
            // A synthetic accounting witness, not an allocation estimate for
            // these native arrays. Numerical/native capacity bounds are tested
            // separately; this test follows exact charge ownership only.
            let admission = retention_admission();
            let charge = admission.incremental_required_bytes;
            let pool = WorkingMemoryPool::new(charge * 2, 0).unwrap();
            let execution = InferenceExecutionIdentity::default();
            let first: InferenceRequest = pool.reserve(&execution, &admission).unwrap().into();
            state.retain_inference(&first);
            drop(first);
            let before = values(&state, stream);
            assert!(!before.is_empty());
            let checkpoint = state.deep_checkpoint().unwrap();
            let saved = state.isolated_snapshot(stream).unwrap();
            let fork = state.fork_prediction_target_state(stream).unwrap();
            for descendant in [&checkpoint, &saved, &fork] {
                assert_eq!(descendant.inference_retention().requests().len(), 1);
                assert_eq!(values(descendant, stream), before);
            }
            let second: InferenceRequest = pool.reserve(&execution, &admission).unwrap().into();
            state.retain_inference(&second);
            drop(second);
            state.restore_checkpoint(&checkpoint, stream).unwrap();
            assert_eq!(state.inference_retention().requests().len(), 2);
            assert_eq!(values(&state, stream), before);
            assert_eq!(pool.used_bytes().unwrap(), charge * 2);
            drop(state);
            assert_eq!(pool.used_bytes().unwrap(), charge);
            drop(checkpoint);
            drop(saved);
            assert_eq!(pool.used_bytes().unwrap(), charge);
            drop(fork);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            assert_eq!(pool.peak_bytes().unwrap(), charge * 2);
        }

        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let layout =
            |policy| StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap();
        let input = Array::from_slice(&[1.0_f32, 2., 3., 5., 7., 11.], &[1, 1, 3, 2]);
        let mut kv = MlxKeyValueState::device(layout(
            LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap(),
        ))
        .unwrap();
        kv.layer(0)
            .unwrap()
            .update_and_fetch(input.clone(), input.clone(), &stream)
            .unwrap();
        check(kv, &stream);

        let role = StateTensorRole::Recurrent;
        let policy = LayerCachePolicy::fixed_only(vec![
            StateTensorPolicy::new(
                role,
                vec![
                    StateTensorDimension::fixed(2).unwrap(),
                    StateTensorDimension::fixed(2).unwrap(),
                ],
                StateTensorDtype::Float32,
                MutableStateResidency::LayerScopedOffloadable,
            )
            .unwrap(),
        ])
        .unwrap();
        let mut hybrid = MlxHybridState::device(layout(policy)).unwrap();
        let layer = hybrid.layer(0).unwrap();
        *layer.fixed_component(role).unwrap() = Some(MlxTensor::from_array(
            Array::from_slice(&[2.0_f32, 3., 5., 7.], &[2, 2])
                .transpose(&stream)
                .unwrap(),
        ));
        layer.advance_fixed(3).unwrap();
        check(hybrid, &stream);

        let mut pooling = MlxPoolingAttentionStateFactory::device(layout(
            LayerCachePolicy::key_only(AttentionPolicy::sliding(3).unwrap(), 1, 2).unwrap(),
        ))
        .unwrap();
        pooling
            .layer(0)
            .unwrap()
            .append_local(
                MlxTensor::from_array(input.reshape(&[1, 3, 2], &stream).unwrap()),
                &stream,
            )
            .unwrap();
        check(pooling, &stream);
    }
}

#[cfg(test)]
#[path = "mechanisms/terminal_tests.rs"]
mod terminal_tests;

#[cfg(test)]
pub(in crate::composition::mlx::replicated_text) use opening_sources::{
    OpeningPinFailure, PreparedOpeningPins,
};

#[cfg(test)]
pub(in crate::composition::mlx::replicated_text) use opening_sources::OpeningPinSetup;

pub(crate) use opening_sources::{
    NativeOpeningRows, NativeOpeningRowsOwner, NativeOpeningRowsPlan, RetiredOpeningRow,
    SealedOpeningRows,
};
