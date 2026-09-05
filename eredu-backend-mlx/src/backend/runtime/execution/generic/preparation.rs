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

    let static_id = OffloadUnitId::new("model.static")?;
    let static_bindings = static_bindings(architecture.static_modules(), store.as_ref())?;
    let static_bytes = binding_bytes(&static_bindings)?;
    consumed.extend(
        static_bindings
            .iter()
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
    );
    definitions.push(OffloadUnit::new(static_id.clone(), static_bindings)?);
    specs.push(OffloadUnitSpec::new(
        static_id.clone(),
        static_bytes,
        ResidencyPolicy::Pinned,
        MemoryTier::Device,
    )?);

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
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let unit = architecture
            .build_unit(address.group(), address.index(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
    let device_window_bytes = (0..layout.group_count())
        .map(|group| {
            let range = layout
                .group_range(group)
                .expect("validated layout covers every execution group");
            largest_window_bytes(&unit_bytes[range], depth)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
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
    let static_lease = residency.acquire(&static_id, MemoryTier::Device)?;
    populate_parameterized(architecture.static_modules_mut(), &static_lease)?;
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
                (0..layout.group_count()).map(|group| {
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
        vec![static_lease],
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
    _state: std::marker::PhantomData<S>,
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
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
    P: MlxUnitPopulator<A::Unit>,
    I: Fn(&str) -> bool,
{
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

    let static_id = OffloadUnitId::new("model.static")?;
    let static_bytes = binding_bytes(&static_bindings)?;
    consumed.extend(
        static_bindings
            .iter()
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
    );
    definitions.push(OffloadUnit::new(static_id.clone(), static_bindings)?);
    specs.push(OffloadUnitSpec::new(
        static_id.clone(),
        static_bytes,
        ResidencyPolicy::Pinned,
        MemoryTier::Device,
    )?);

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
    consumed.extend(store.materialized_source_keys());
    validate_unused(store.as_ref(), &consumed, ignored)?;
    let device_window_bytes = (0..layout.group_count())
        .map(|group| {
            let range = layout
                .group_range(group)
                .expect("validated layout covers every execution group");
            largest_window_bytes(&unit_bytes[range], depth)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
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
    let static_lease = residency.acquire(&static_id, MemoryTier::Device)?;
    populate_parameterized(architecture.static_modules_mut(), &static_lease)?;
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
                (0..layout.group_count()).map(|group| {
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
        vec![static_lease],
        dense_controller,
        options.sample_backend_memory(),
        options.sample_process_memory(),
    )?;
    Ok((policy, metadata))
}
