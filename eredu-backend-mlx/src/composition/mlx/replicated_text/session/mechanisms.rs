use super::*;

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
    store: Arc<dyn CheckpointSource>,
    prepared_bindings: Option<PreparedExactBindings>,
    resident_report: Option<ResidencyReport>,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    stream: Stream,
    weights_stream: Stream,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    source_parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    ignored_checkpoint_sources: std::collections::BTreeSet<String>,
    state_rank: Option<eredu_core::cache::CacheRankIdentity>,
    state_global_layer_start: usize,
    state: PhantomData<fn() -> (A, S)>,
}

pub(in crate::composition::mlx::replicated_text) struct PreparedExactBindings {
    layout: eredu_runtime::ExecutionUnitLayout,
    static_bindings: Vec<WeightBinding>,
    unit_bindings: Vec<Vec<WeightBinding>>,
    excluded_parameters: std::collections::BTreeSet<String>,
    local_parameters: Option<std::collections::BTreeSet<String>>,
}

pub(in crate::composition::mlx::replicated_text) struct MlxPromptCacheSaveTransaction {
    publication: eredu_runtime::ReversiblePromptCachePublication,
    manifest: PromptCacheManifest,
}

pub(in crate::composition::mlx::replicated_text) fn shard_unmaterialized_bindings(
    bindings: Vec<WeightBinding>,
    store: &dyn CheckpointSource,
    layout: &eredu_runtime::LocalModelLayout,
    locally_materialized: &std::collections::BTreeSet<String>,
) -> Result<Vec<WeightBinding>, Error> {
    let mut output = Vec::with_capacity(bindings.len());
    for binding in bindings {
        if locally_materialized.contains(binding.name()) {
            output.push(binding);
        } else {
            output.extend(shard_layer_bindings(vec![binding], store, layout)?);
        }
    }
    Ok(output)
}

impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    pub(super) fn new(
        store: Arc<dyn CheckpointSource>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Self {
        Self {
            store,
            prepared_bindings: None,
            resident_report: None,
            materialization: None,
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            parallel_layout: None,
            source_parallel_layout: None,
            ignored_checkpoint_sources: std::collections::BTreeSet::new(),
            state_rank: None,
            state_global_layer_start: 0,
            state: PhantomData,
        }
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
            let (store, report) = quantize_exact_replicated_text_tasks(
                Arc::clone(&self.store),
                source_static,
                target_static,
                source_units,
                target_units,
                source_layout,
                group.quantization(),
                &exact_tasks,
                &self.stream,
            )?;
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
            excluded_parameters.len(),
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
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let counts = (0..graph.groups().len())
            .map(|group| {
                addresses
                    .iter()
                    .filter(|address| address.group() == group)
                    .count()
            })
            .collect::<Vec<_>>();
        let layout = eredu_runtime::ExecutionUnitLayout::new(&graph, counts)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        self.prepared_bindings = Some(PreparedExactBindings {
            layout,
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
        let (policy, _) = prepare_layerwise_policy_from_bindings(
            Arc::clone(&self.store),
            architecture,
            MlxSelectiveUnitPopulator::new(prepared.excluded_parameters.clone()),
            PhantomData::<S>,
            selected.residency(),
            &self.stream,
            &self.weights_stream,
            move |key| ignored_sources.contains(key),
            prepared.layout,
            prepared.static_bindings,
            prepared.unit_bindings,
        )?;
        Ok((policy, layout))
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
    type State = S;
    type PolicyError = Error;
    type ResidentPolicy = MlxArchitectureLayerwisePolicy<A, S>;
    type BoundedPolicy = MlxArchitectureLayerwisePolicy<A, S>;
    type StateCheckpoint = S;
    type StateReport = MlxStateReport;
    type ExecutionReport = MlxExecutionReport;
    type Error = Error;

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

        self.prepared_bindings = Some(PreparedExactBindings {
            layout: target_layout.clone(),
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
        let (policy, _) = self.take_prepared_policy(architecture, selected)?;
        let resident = policy.into_resident_units(units, context)?;
        self.resident_report = Some(resident.residency_report()?);
        Ok(MlxSelectedLayerwisePolicy::resident(resident))
    }

    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        selected: &SelectedReplicatedTextRealization,
        _context: &Stream,
    ) -> Result<Self::BoundedPolicy, Self::Error> {
        self.take_prepared_policy(architecture, selected)
            .map(|(policy, layout)| MlxSelectedLayerwisePolicy::bounded(policy, &layout))
    }

    fn index_text_output(
        &mut self,
        output: MlxTensor,
        sequence_index: i32,
        context: &Stream,
    ) -> Result<MlxTensor, Error> {
        output
            .as_array()
            .try_index_device((.., sequence_index, ..), context)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }

    fn checkpoint_state(&mut self, state: &S, _context: &Stream) -> Result<S, Error> {
        state.deep_checkpoint().map_err(Into::into)
    }

    fn restore_state(
        &mut self,
        state: &mut S,
        checkpoint: S,
        context: &Stream,
    ) -> Result<(), Error> {
        state
            .restore_checkpoint(&checkpoint, context)
            .map_err(Into::into)
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

    fn complete(&mut self, output: &MlxTensor, state: &S, _context: &Stream) -> Result<(), Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::completion();
        let token_validations = active_token_validation_arrays();
        async_eval_with_event(
            std::iter::once(output.as_array())
                .chain(state.retained_arrays())
                .chain(token_validations.iter()),
        )?
        .synchronize()?;
        validate_active_token_validations().map_err(Into::into)
    }
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
