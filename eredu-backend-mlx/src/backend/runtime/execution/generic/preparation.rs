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
    let fully_resident = options.is_fully_resident();
    let dense = options.dense();
    let offload = options.offload()?;
    let depth = options.device_depth(unit_count);
    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();

    let static_bindings = static_bindings(architecture.static_modules(), store.as_ref())?;
    let static_bytes = binding_bytes(&static_bindings)?;
    let static_parameters = static_bindings
        .iter()
        .map(|binding| binding.name().to_owned())
        .collect();
    consumed.extend(
        static_bindings
            .iter()
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
    );
    // Some architectures place every learned parameter in execution units.
    // An empty static role has no storage identity or residency lease.
    let static_id = if static_bindings.is_empty() {
        None
    } else {
        let id = OffloadUnitId::new("model.static")?;
        definitions.push(OffloadUnit::new(id.clone(), static_bindings)?);
        specs.push(OffloadUnitSpec::new(
            id.clone(),
            static_bytes,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
        )?);
        Some(id)
    };

    let mut unit_ids = Vec::with_capacity(unit_count);
    let mut unit_bytes = Vec::with_capacity(unit_count);
    let mut layer_parameter_bytes = 0u64;
    let mut total_host_bytes = 0u64;
    let mut maximum_host_bytes = 0u64;
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
        let bytes = binding_bytes(&bindings)?;
        layer_parameter_bytes = layer_parameter_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("generic layer bytes overflowed".into()))?;
        let host_bytes = host_capacity_upper_bound_for_bindings(&bindings)?;
        total_host_bytes = total_host_bytes
            .checked_add(host_bytes)
            .ok_or_else(|| Error::Parallel("generic host unit bytes overflowed".into()))?;
        maximum_host_bytes = maximum_host_bytes.max(host_bytes);
        consumed.extend(
            bindings
                .iter()
                .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
        );
        let group_id = layout
            .group_id(address.group())
            .expect("validated layout names every execution group");
        let id = OffloadUnitId::new(format!(
            "model.{}.{:05}",
            group_id.as_str(),
            address.index()
        ))?;
        definitions.push(OffloadUnit::new(id.clone(), bindings)?);
        specs.push(OffloadUnitSpec::new(
            id.clone(),
            bytes,
            if fully_resident {
                ResidencyPolicy::Pinned
            } else if dense.is_some() {
                ResidencyPolicy::Cacheable
            } else {
                ResidencyPolicy::Windowed
            },
            if fully_resident {
                MemoryTier::Device
            } else if dense.is_some() {
                MemoryTier::Disk
            } else {
                MemoryTier::Host
            },
        )?);
        unit_ids.push(id);
        unit_bytes.push(bytes);
    }
    consumed.extend(store.materialized_source_keys());
    validate_unused(store.as_ref(), &consumed, ignored)?;
    let device_window_bytes =
        eredu_runtime::execution_unit_parameter_window_bytes(&layout, &unit_bytes, options)
            .map_err(|error| Error::Parallel(error.to_string()))?;
    let host_required = match dense {
        Some(dense) if dense.host_budget_bytes() > 0 => maximum_host_bytes
            .checked_mul(dense.host_lookahead() as u64)
            .ok_or_else(|| Error::Parallel("generic host window bytes overflowed".into()))?,
        Some(_) => 0,
        None if fully_resident => 0,
        None => total_host_bytes,
    };
    validate_host_budget(offload, host_required)?;
    validate_device_budget(offload, static_bytes, device_window_bytes, depth)?;

    let plan = OffloadPlan::new(offload, specs)?;
    let residency_stream = if dense.is_some() {
        Stream::new_with_device(&stream.get_device()?)
    } else {
        stream.clone()
    };
    let residency = ResidencyManager::new_shared(
        Arc::clone(&store),
        plan,
        definitions,
        weights_stream.clone(),
        residency_stream,
    )?;
    residency.initialize()?;
    let mut static_leases = Vec::new();
    if let Some(id) = static_id {
        let lease = residency.acquire(&id, MemoryTier::Device)?;
        populate_selected_parameterized(
            architecture.static_modules_mut(),
            &lease,
            &static_parameters,
        )?;
        static_leases.push(lease);
    }
    let metadata = LayerwiseModelMetadata::new(
        "generic",
        None,
        unit_count,
        static_bytes,
        options.execution_residency(),
        layer_parameter_bytes,
        device_window_bytes,
        maximum_host_bytes,
        depth,
    );
    let dense_controller = dense
        .map(|options| {
            DenseStreamController::new(
                &residency,
                options,
                unit_count,
                layer_parameter_bytes,
                maximum_host_bytes,
                static_bytes,
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
        depth,
        populator,
        static_leases,
        dense_controller,
        options.sample_backend_memory(),
        options.sample_process_memory(),
    )?;
    Ok((policy, metadata))
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
    let fully_resident = options.is_fully_resident();
    let dense = options.dense();
    let offload = options.offload()?;
    let depth = options.device_depth(unit_count);
    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();

    let static_bytes = binding_bytes(&static_bindings)?;
    let static_parameters = static_bindings
        .iter()
        .map(|binding| binding.name().to_owned())
        .collect();
    consumed.extend(
        static_bindings
            .iter()
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
    );
    // Some architectures place every learned parameter in execution units.
    // An empty static role has no storage identity or residency lease.
    let static_id = if static_bindings.is_empty() {
        None
    } else {
        let id = OffloadUnitId::new("model.static")?;
        definitions.push(OffloadUnit::new(id.clone(), static_bindings)?);
        specs.push(OffloadUnitSpec::new(
            id.clone(),
            static_bytes,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
        )?);
        Some(id)
    };

    let mut unit_ids = Vec::with_capacity(unit_count);
    let mut unit_bytes = Vec::with_capacity(unit_count);
    let mut layer_parameter_bytes = 0u64;
    let mut total_host_bytes = 0u64;
    let mut maximum_host_bytes = 0u64;
    for (index, bindings) in unit_bindings.into_iter().enumerate() {
        let address = layout
            .address(index)
            .expect("validated layout covers every flat unit");
        let bytes = binding_bytes(&bindings)?;
        layer_parameter_bytes = layer_parameter_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("generic layer bytes overflowed".into()))?;
        let host_bytes = host_capacity_upper_bound_for_bindings(&bindings)?;
        total_host_bytes = total_host_bytes
            .checked_add(host_bytes)
            .ok_or_else(|| Error::Parallel("generic host unit bytes overflowed".into()))?;
        maximum_host_bytes = maximum_host_bytes.max(host_bytes);
        consumed.extend(
            bindings
                .iter()
                .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
        );
        let group_id = layout
            .group_id(address.group())
            .expect("validated layout names every execution group");
        let id = OffloadUnitId::new(format!(
            "model.{}.{:05}",
            group_id.as_str(),
            address.index()
        ))?;
        definitions.push(OffloadUnit::new(id.clone(), bindings)?);
        specs.push(OffloadUnitSpec::new(
            id.clone(),
            bytes,
            if fully_resident {
                ResidencyPolicy::Pinned
            } else if dense.is_some() {
                ResidencyPolicy::Cacheable
            } else {
                ResidencyPolicy::Windowed
            },
            if fully_resident {
                MemoryTier::Device
            } else if dense.is_some() {
                MemoryTier::Disk
            } else {
                MemoryTier::Host
            },
        )?);
        unit_ids.push(id);
        unit_bytes.push(bytes);
    }
    let mut extra = eredu_runtime::AuxiliaryWeightRequirements::default();
    let mut sources = std::collections::BTreeMap::new();
    for unit in supplementary {
        let bytes = binding_bytes(unit.definition.bindings())?;
        let host = host_capacity_upper_bound_for_bindings(unit.definition.bindings())?;
        extra = extra
            .checked_add_module(bytes, host, unit.shared)
            .ok_or_else(|| Error::Parallel("auxiliary residency bytes overflowed".into()))?;
        let id = unit.definition.id().clone();
        specs.push(OffloadUnitSpec::new(
            id.clone(),
            bytes,
            if fully_resident {
                ResidencyPolicy::Pinned
            } else {
                ResidencyPolicy::Cacheable
            },
            if fully_resident {
                MemoryTier::Device
            } else if dense.is_some() {
                MemoryTier::Disk
            } else {
                MemoryTier::Host
            },
        )?);
        sources.insert(id, unit.source);
        definitions.push(unit.definition);
    }
    consumed.extend(store.materialized_source_keys());
    validate_unused(store.as_ref(), &consumed, ignored)?;
    let device_window_bytes =
        eredu_runtime::execution_unit_parameter_window_bytes(&layout, &unit_bytes, options)
            .map_err(|error| Error::Parallel(error.to_string()))?;
    let host_required = match dense {
        Some(dense) if dense.host_budget_bytes() > 0 => maximum_host_bytes
            .checked_mul(dense.host_lookahead() as u64)
            .ok_or_else(|| Error::Parallel("generic host window bytes overflowed".into()))?,
        Some(_) => 0,
        None if fully_resident => 0,
        None => total_host_bytes,
    };
    let auxiliary_host = extra
        .host_bytes(options)
        .ok_or_else(|| Error::Parallel("auxiliary host bytes overflowed".into()))?;
    let host_required = if dense.is_some() {
        host_required.max(auxiliary_host)
    } else {
        host_required
            .checked_add(auxiliary_host)
            .ok_or_else(|| Error::Parallel("combined host bytes overflowed".into()))?
    };
    let auxiliary_device = extra
        .device_bytes(options)
        .ok_or_else(|| Error::Parallel("auxiliary device bytes overflowed".into()))?;
    let required_device = if fully_resident {
        device_window_bytes
            .checked_add(auxiliary_device)
            .ok_or_else(|| Error::Parallel("combined device bytes overflowed".into()))?
    } else {
        device_window_bytes.max(auxiliary_device)
    };
    validate_host_budget(offload, host_required)?;
    validate_device_budget(offload, static_bytes, required_device, depth)?;

    let plan = OffloadPlan::new(offload, specs)?;
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
    let metadata = LayerwiseModelMetadata::new(
        "generic",
        None,
        unit_count,
        static_bytes,
        options.execution_residency(),
        layer_parameter_bytes,
        device_window_bytes,
        maximum_host_bytes,
        depth,
    );
    let dense_controller = dense
        .map(|options| {
            DenseStreamController::new(
                &residency,
                options,
                unit_count,
                layer_parameter_bytes,
                maximum_host_bytes,
                static_bytes,
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
        depth,
        populator,
        static_leases,
        dense_controller,
        options.sample_backend_memory(),
        options.sample_process_memory(),
    )?;
    Ok((policy, metadata))
}
