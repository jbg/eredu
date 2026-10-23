//! Cold preparation of bounded and resident execution policies.

use super::*;

/// Builds a generic MLX layerwise policy from neutral parameter topologies.
#[allow(clippy::too_many_arguments)]
pub fn prepare_layerwise_policy<A, S, P, I>(
    store: SharedCheckpointSource,
    architecture: &mut A,
    populator: P,
    state: std::marker::PhantomData<S>,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    ignored: I,
) -> Result<(MlxLayerwisePolicy<A::Unit, P>, LayerwiseModelMetadata), Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    S: RuntimeState<MlxNeuralBackend>,
    A::StaticModules: Clone,
    A::Unit: 'static,
    A::Error: std::fmt::Display,
    P: MlxUnitPopulator<A::Unit>,
    I: Fn(&str) -> bool,
{
    prepare_layerwise_policy_with_bindings(
        store,
        architecture,
        populator,
        state,
        options,
        stream,
        weights_stream,
        ignored,
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        |_ordinal, _address, _path, unit, store, _stream| {
            build_module_bindings(&MlxModule::new(unit), "", store).map_err(Into::into)
        },
    )
}

/// Builds the generic policy from caller-realized local binding selections.
///
/// Parallel composition uses this entry point to provide rank-local checkpoint
/// selections while retaining the same residency, overlap, and completion
/// algorithm used by replicated execution. The unit-binding callback receives
/// the architecture's validated flat ordinal, group-local address, and
/// canonical unit path; callers must not reconstruct those identities from
/// residency order.
#[allow(clippy::too_many_arguments)]
pub fn prepare_layerwise_policy_with_bindings<A, S, P, I, SB, UB>(
    store: SharedCheckpointSource,
    architecture: &mut A,
    populator: P,
    _state: std::marker::PhantomData<S>,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    ignored: I,
    static_bindings: SB,
    mut unit_bindings: UB,
) -> Result<(MlxLayerwisePolicy<A::Unit, P>, LayerwiseModelMetadata), Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Unit: 'static,
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
    P: MlxUnitPopulator<A::Unit>,
    I: Fn(&str) -> bool,
    SB: FnOnce(
        &A::StaticModules,
        &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>,
    UB: FnMut(
        usize,
        ExecutionUnitAddress,
        &str,
        A::Unit,
        &dyn eredu_checkpoint::store::CheckpointSource,
        &Stream,
    ) -> Result<Vec<WeightBinding>, Error>,
{
    let layout = architecture_execution_layout::<A, S>(architecture)?;
    let unit_count = layout.len();
    if unit_count == 0 {
        return Err(Error::Parallel(
            "generic MLX architecture declared no execution units".into(),
        ));
    }
    let offload = options.offload()?;
    let depth = options.device_depth(unit_count);
    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();

    let static_bindings = static_bindings(architecture.static_modules(), store.as_ref())?;
    let pinned = eredu_runtime::StaticUnitBindings::new("model.static", static_bindings)?
        .prepare::<Error>()?;
    let static_bytes = pinned.bytes;
    let static_parameters = pinned.parameters;
    consumed.extend(pinned.source_keys);
    let static_id = pinned.resource.map(|(definition, spec)| {
        let id = definition.id().clone();
        definitions.push(definition);
        specs.push(spec);
        id
    });

    let mut execution_units =
        eredu_runtime::ExecutionUnitResourcePreparation::new("model", &layout, options);
    for index in 0..unit_count {
        let address = layout
            .address(index)
            .expect("validated layout covers every flat unit");
        let path = architecture
            .unit_path(address.group(), address.index())
            .map_err(|error| Error::Other(Box::new(error)))?;
        let unit = architecture
            .build_unit(address.group(), address.index(), stream)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let bindings = unit_bindings(index, address, &path, unit, store.as_ref(), stream)?;
        execution_units = execution_units.push(bindings, |bindings| {
            host_capacity_upper_bound_for_bindings(bindings).map_err(Error::from)
        })?;
    }
    let units = execution_units.finish()?;
    definitions.extend(units.definitions);
    specs.extend(units.specs);
    consumed.extend(units.source_keys);
    consumed.extend(store.materialized_source_keys());
    validate_unused(store.as_ref(), &consumed, ignored)?;
    let requirements = eredu_runtime::execution_unit_resource_requirements(
        &layout,
        &units.unit_bytes,
        units.total_host_bytes,
        units.maximum_host_bytes,
        options,
        Default::default(),
    )
    .map_err(Error::from)?;
    validate_host_budget(offload, requirements.required_host_bytes)?;
    validate_device_budget(
        offload,
        static_bytes,
        requirements.required_device_bytes,
        depth,
    )?;

    let plan = OffloadPlan::new(offload, specs)?;
    let metadata = LayerwiseModelMetadata::new(
        "generic",
        None,
        unit_count,
        static_bytes,
        options.execution_residency(),
        units.parameter_bytes,
        requirements.device_window_bytes,
        units.maximum_host_bytes,
        depth,
    );
    materialize_parameter_unit_policy::<A::Unit, _, _>(
        store,
        architecture.static_modules_mut(),
        populator,
        options,
        stream,
        weights_stream,
        plan,
        definitions,
        std::collections::BTreeMap::new(),
        layout,
        units.unit_ids,
        static_id,
        static_parameters,
        metadata,
    )
}

/// Builds an MLX residency policy from exact binding definitions projected by
/// a neutral constructor over its already-constructed execution units.
#[allow(clippy::too_many_arguments)]
pub fn prepare_layerwise_policy_from_bindings<A, S, P, I>(
    store: SharedCheckpointSource,
    architecture: &mut A,
    populator: P,
    state: std::marker::PhantomData<S>,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    ignored: I,
    layout: ExecutionUnitLayout,
    static_bindings: Vec<WeightBinding>,
    unit_bindings: Vec<Vec<WeightBinding>>,
) -> Result<(MlxLayerwisePolicy<A::Unit, P>, LayerwiseModelMetadata), Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Unit: 'static,
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
    P: MlxUnitPopulator<A::Unit>,
    I: Fn(&str) -> bool,
{
    prepare_layerwise_policy_with_supplementary_bindings(
        store,
        architecture,
        populator,
        state,
        options,
        stream,
        weights_stream,
        ignored,
        layout,
        static_bindings,
        unit_bindings,
        Vec::new(),
    )
}

/// Builds a policy over exact selected canonical addresses and their prepared bindings.
/// Local storage ordinals remain compact while architecture unit indices and
/// group identities retain their original meaning across pipeline cuts.
#[allow(clippy::too_many_arguments)]
pub fn prepare_layerwise_policy_from_addressed_bindings<A, S, P, I>(
    store: SharedCheckpointSource,
    architecture: &mut A,
    populator: P,
    state: std::marker::PhantomData<S>,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    ignored: I,
    static_bindings: Vec<WeightBinding>,
    units: Vec<(ExecutionUnitAddress, Vec<WeightBinding>)>,
) -> Result<(MlxLayerwisePolicy<A::Unit, P>, LayerwiseModelMetadata), Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Unit: 'static,
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
    P: MlxUnitPopulator<A::Unit>,
    I: Fn(&str) -> bool,
{
    let layout = architecture_execution_layout::<A, S>(architecture)?
        .select(units.iter().map(|(address, _)| *address))
        .map_err(|error| Error::Other(Box::new(error)))?;
    prepare_layerwise_policy_from_bindings(
        store,
        architecture,
        populator,
        state,
        options,
        stream,
        weights_stream,
        ignored,
        layout,
        static_bindings,
        units.into_iter().map(|(_, bindings)| bindings).collect(),
    )
}

/// Builds one shared residency policy over ordinary and supplementary modules,
/// retaining the exact prepared source associated with each supplementary owner.
#[allow(clippy::too_many_arguments)]
pub fn prepare_layerwise_policy_with_supplementary_bindings<A, S, P, I>(
    store: SharedCheckpointSource,
    architecture: &mut A,
    populator: P,
    _state: std::marker::PhantomData<S>,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    ignored: I,
    layout: ExecutionUnitLayout,
    static_bindings: Vec<WeightBinding>,
    unit_bindings: Vec<Vec<WeightBinding>>,
    supplementary: Vec<SupplementaryResidencyUnit>,
) -> Result<(MlxLayerwisePolicy<A::Unit, P>, LayerwiseModelMetadata), Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Unit: 'static,
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
    P: MlxUnitPopulator<A::Unit>,
    I: Fn(&str) -> bool,
{
    prepare_parameter_unit_policy::<A::Unit, _, _, _>(
        store,
        architecture.static_modules_mut(),
        populator,
        options,
        stream,
        weights_stream,
        ignored,
        layout,
        static_bindings,
        unit_bindings,
        supplementary,
    )
}

/// Prepares the existing completion-safe resident/host/disk parameter mechanism
/// from retained exact units, independently of the caller's invocation driver.
/// Architecture-owned traversal supplies unit construction and semantic addresses.
#[allow(clippy::too_many_arguments)]
pub fn prepare_parameter_unit_policy<U, SM, P, I>(
    store: SharedCheckpointSource,
    static_modules: &mut SM,
    populator: P,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    ignored: I,
    layout: ExecutionUnitLayout,
    static_bindings: Vec<WeightBinding>,
    unit_bindings: Vec<Vec<WeightBinding>>,
    supplementary: Vec<SupplementaryResidencyUnit>,
) -> Result<(MlxLayerwisePolicy<U, P>, LayerwiseModelMetadata), Error>
where
    U: Parameterized<MlxTensor> + 'static,
    SM: Parameterized<MlxTensor>,
    P: MlxUnitPopulator<U>,
    I: Fn(&str) -> bool,
{
    crate::backend::runtime::checkpoint::bounded_quantization::realize_prepared_source(
        &store, stream,
    )?;
    let unit_count = layout.len();
    if unit_count == 0 || unit_bindings.len() != unit_count {
        return Err(Error::Parallel(format!(
            "exact MLX binding definitions contain {} units for a layout of {unit_count}",
            unit_bindings.len()
        )));
    }
    let offload = options.offload()?;
    let depth = options.device_depth(unit_count);
    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();

    let pinned = eredu_runtime::StaticUnitBindings::new("model.static", static_bindings)?
        .prepare::<Error>()?;
    let static_bytes = pinned.bytes;
    let static_parameters = pinned.parameters;
    consumed.extend(pinned.source_keys);
    let static_id = pinned.resource.map(|(definition, spec)| {
        let id = definition.id().clone();
        definitions.push(definition);
        specs.push(spec);
        id
    });

    let mut execution_units =
        eredu_runtime::ExecutionUnitResourcePreparation::new("model", &layout, options);
    for bindings in unit_bindings {
        execution_units = execution_units.push(bindings, |bindings| {
            host_capacity_upper_bound_for_bindings(bindings).map_err(Error::from)
        })?;
    }
    let units = execution_units.finish()?;
    definitions.extend(units.definitions);
    specs.extend(units.specs);
    consumed.extend(units.source_keys);
    let mut extra = eredu_runtime::AuxiliaryWeightRequirements::default();
    let mut sources = std::collections::BTreeMap::new();
    for unit in supplementary {
        let (requirements, spec) =
            extra.admit(&unit.definition, options, unit.shared, |bindings| {
                host_capacity_upper_bound_for_bindings(bindings).map_err(Error::from)
            })?;
        extra = requirements;
        let id = unit.definition.id().clone();
        specs.push(spec);
        sources.insert(id, unit.source);
        definitions.push(unit.definition);
    }
    consumed.extend(store.materialized_source_keys());
    validate_unused(store.as_ref(), &consumed, ignored)?;
    let requirements = eredu_runtime::execution_unit_resource_requirements(
        &layout,
        &units.unit_bytes,
        units.total_host_bytes,
        units.maximum_host_bytes,
        options,
        extra,
    )
    .map_err(Error::from)?;
    validate_host_budget(offload, requirements.required_host_bytes)?;
    validate_device_budget(
        offload,
        static_bytes,
        requirements.required_device_bytes,
        depth,
    )?;

    let plan = OffloadPlan::new(offload, specs)?;
    let metadata = LayerwiseModelMetadata::new(
        "generic",
        None,
        unit_count,
        static_bytes,
        options.execution_residency(),
        units.parameter_bytes,
        requirements.device_window_bytes,
        units.maximum_host_bytes,
        depth,
    );
    materialize_parameter_unit_policy::<U, _, _>(
        store,
        static_modules,
        populator,
        options,
        stream,
        weights_stream,
        plan,
        definitions,
        sources,
        layout,
        units.unit_ids,
        static_id,
        static_parameters,
        metadata,
    )
}

/// Materializes one admitted policy with the existing native completion and transfer owners.
#[allow(clippy::too_many_arguments)]
fn materialize_parameter_unit_policy<U, SM, P>(
    store: SharedCheckpointSource,
    static_modules: &mut SM,
    populator: P,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    plan: OffloadPlan,
    definitions: Vec<OffloadUnit>,
    sources: std::collections::BTreeMap<OffloadUnitId, SharedCheckpointSource>,
    layout: ExecutionUnitLayout,
    unit_ids: Vec<OffloadUnitId>,
    static_id: Option<OffloadUnitId>,
    static_parameters: BTreeSet<String>,
    metadata: LayerwiseModelMetadata,
) -> Result<(MlxLayerwisePolicy<U, P>, LayerwiseModelMetadata), Error>
where
    U: Parameterized<MlxTensor> + 'static,
    SM: Parameterized<MlxTensor>,
    P: MlxUnitPopulator<U>,
{
    let dense = options.dense();
    let residency_stream = if dense.is_some() {
        Stream::new_with_device(&stream.get_device()?)
    } else {
        stream.clone()
    };
    let residency = ResidencyManager::new_shared_sources(
        Arc::clone(&store),
        sources,
        plan,
        definitions,
        weights_stream.clone(),
        residency_stream,
    )?;
    residency.initialize()?;
    let mut static_leases = Vec::new();
    if let Some(id) = static_id {
        let lease = residency.acquire(&id, MemoryTier::Device)?;
        populate_selected_parameterized(static_modules, &lease, &static_parameters)?;
        static_leases.push(lease);
    }
    let dense_controller = dense
        .map(|options| {
            DenseStreamController::new(
                &residency,
                options,
                metadata.layer_count(),
                metadata.layer_parameter_bytes(),
                metadata.maximum_host_layer_bytes(),
                metadata.static_device_bytes(),
                (0..layout.group_count())
                    .filter(|&group| {
                        layout
                            .group_range(group)
                            .is_some_and(|range| !range.is_empty())
                    })
                    .map(|group| {
                        let range = layout
                            .group_range(group)
                            .expect("validated layout covers every execution group");
                        let id = layout
                            .group_id(group)
                            .expect("validated layout names every execution group")
                            .as_str()
                            .to_owned();
                        (id, unit_ids[range].to_vec())
                    }),
            )
            .map(Arc::new)
        })
        .transpose()?;
    let policy = MlxLayerwisePolicy::new(
        residency,
        Arc::clone(&store),
        unit_ids,
        layout,
        metadata.device_layer_capacity(),
        populator,
        static_leases,
        dense_controller,
        options.sample_backend_memory(),
        options.sample_process_memory(),
    )?;
    Ok((policy, metadata))
}
