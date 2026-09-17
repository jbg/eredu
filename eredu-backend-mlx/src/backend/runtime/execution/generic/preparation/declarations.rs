//! Exact cold residency declarations shared by native binding and prepared loading.
use super::*;
use std::collections::BTreeMap;

pub(crate) struct PreparedLayerwiseDeclarations {
    pub(crate) store: RetainedCheckpointSource,
    pub(crate) sources: BTreeMap<OffloadUnitId, RetainedCheckpointSource>,
    pub(crate) plan: OffloadPlan,
    pub(crate) definitions: Vec<OffloadUnit>,
    pub(crate) static_bytes: u64,
    pub(crate) static_id: OffloadUnitId,
    pub(crate) static_parameters: BTreeSet<String>,
    pub(crate) unit_ids: Vec<OffloadUnitId>,
    pub(crate) unit_count: usize,
    pub(crate) layer_parameter_bytes: u64,
    pub(crate) device_window_bytes: u64,
    pub(crate) maximum_host_bytes: u64,
    pub(crate) depth: usize,
}

/// Consumes the same exact bindings as the final native visitor. It performs
/// source/coverage and portable residency validation without creating arrays,
/// streams, a manager, payload leases or a native parameter placeholder.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_layerwise_declarations(
    store: RetainedCheckpointSource,
    options: LayerWeightResidency,
    ignored: impl Fn(&str) -> bool,
    layout: &ExecutionUnitLayout,
    static_bindings: Vec<WeightBinding>,
    unit_bindings: Vec<Vec<WeightBinding>>,
    supplementary: Vec<SupplementaryResidencyUnit>,
) -> Result<PreparedLayerwiseDeclarations, Error> {
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
    let static_parameters = static_bindings
        .iter()
        .map(|binding| binding.name().to_owned())
        .collect();
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
    Ok(PreparedLayerwiseDeclarations {
        store,
        sources,
        plan,
        definitions,
        static_bytes,
        static_id,
        static_parameters,
        unit_ids,
        unit_count,
        layer_parameter_bytes,
        device_window_bytes,
        maximum_host_bytes,
        depth,
    })
}
