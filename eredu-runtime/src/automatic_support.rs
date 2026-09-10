//! Portable resource sizing and telemetry for automatic execution planning.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use eredu_core::{
    residency::{MemoryTier, TransferDirection},
    BoundedResidencyRequirement, DurationSeconds, ResidencyTelemetry, TransferTelemetry,
};

use crate::{
    replicated_text_materialization_tasks, selected_materialization_task_bytes,
    LayerWeightResidency, ReplicatedTextContractError, ReplicatedTextParameterOwner,
    ResidencyReport, SelectedReplicatedTextRealization,
};

/// Invalid exact resource sizing for selected bounded execution.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum BoundedResidencySizingError {
    /// The authoritative selected materialization contract is invalid.
    #[error(transparent)]
    Materialization(#[from] ReplicatedTextContractError),
    /// An exact byte or unit count cannot be represented.
    #[error("selected {0} overflowed")]
    ArithmeticOverflow(&'static str),
    /// An attached executable companion has no retained source/output extent.
    #[error("selected companion {0:?} has no byte geometry")]
    MissingCompanionGeometry(String),
    /// A selected physical output has no authoritative rank-local placement.
    #[error("selected parameter {0:?} has invalid or missing local geometry")]
    InvalidLocalGeometry(String),
}

/// Computes the pinned bytes and largest group-local device window.
///
/// Excluded logical parameters belong to independent storage and contribute no
/// ordinary residency bytes. Missing unit indices retain their zero-byte
/// positions, while memory and time depend on the number of selected tasks,
/// rather than the largest unit index. No payload or native mechanism is used.
pub fn selected_text_bounded_requirement(
    selected: &SelectedReplicatedTextRealization,
    excluded: &BTreeSet<String>,
) -> Result<BoundedResidencyRequirement, BoundedResidencySizingError> {
    let tasks = replicated_text_materialization_tasks(selected)?;
    bounded_requirement(
        task_parameter_bytes(&tasks)?
            .into_iter()
            .map(|(name, owner, bytes)| (name, owner, move || Ok(bytes))),
        selected.residency(),
        excluded,
    )
}

/// Exact parameter totals for the selected executable tasks. Routed storage is
/// identified by task identity, independently of its ordinary residency policy.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedParameterResources {
    /// All executable parameters, including independently acquired banks.
    pub parameter_bytes: u64,
    /// Parameters outside repeated execution units.
    pub pinned_bytes: u64,
    /// Largest ordinary execution unit, excluding independently acquired banks.
    pub largest_unit_bytes: u64,
    /// Largest adjacent pair within one execution group.
    pub largest_adjacent_units_bytes: u64,
    /// All routed-bank parameters; shared feed-forward experts remain ordinary.
    pub expert_bytes: u64,
}

/// Peak simultaneously live recipe values, including the produced source tensor.
/// Native conversion scratch is a separate backend mechanism fact.
pub fn placed_recipe_peak_bytes(
    recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
    target: &str,
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&crate::LocalModelLayout>,
    member_selected: bool,
) -> Result<u64, String> {
    placed_source_recipe(recipe, target, source, layout, member_selected)?
        .peak_materialization_bytes(source)
        .map_err(|e| e.to_string())
}

/// Projects an exact selected recipe through the ordinary or member-local
/// physical placement, using the same bounded source rewrite as materialization.
pub fn placed_source_recipe(
    recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
    target: &str,
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&crate::LocalModelLayout>,
    member_selected: bool,
) -> Result<eredu_checkpoint::recipe::DerivedWeightRecipe, String> {
    let bytes = recipe.infer(source).map_err(|e| e.to_string())?.byte_len();
    let binding = crate::WeightBinding::from_recipe(target, recipe.clone(), bytes)
        .and_then(|binding| binding.with_logical_target(target))
        .map_err(|e| e.to_string())?;
    let bindings = if let Some(layout) = layout {
        if member_selected {
            crate::place_addressable_member_bindings(vec![binding], source, layout)
        } else {
            crate::place_weight_bindings(vec![binding], source, layout)
        }
        .map_err(|e| e.to_string())?
    } else {
        vec![binding]
    };
    Ok(bindings[0].source_recipe())
}

/// Sizes exact selected tasks without reading payloads or constructing modules.
/// Attached physical companions are counted once, whether they also occur as
/// standalone tasks. Transform tasks already include their generated companions.
pub fn selected_parameter_resources(
    tasks: &[crate::ReplicatedTextMaterializationTask],
    routed: &BTreeSet<String>,
    independently_acquired: &BTreeSet<String>,
) -> Result<SelectedParameterResources, BoundedResidencySizingError> {
    parameter_resources(task_parameter_bytes(tasks)?, routed, independently_acquired)
}

/// Sizes a rank's owned physical outputs using the architecture's TP/EP layout.
/// Replicated outputs retain their full extent; packed weights and companions
/// use their own physical shapes. No global-total division is involved.
pub fn selected_parameter_resources_for_layout(
    tasks: &[crate::ReplicatedTextMaterializationTask],
    layout: &crate::LocalModelLayout,
    owned: &BTreeSet<String>,
    routed: &BTreeSet<String>,
    independently_acquired: &BTreeSet<String>,
) -> Result<SelectedParameterResources, BoundedResidencySizingError> {
    use BoundedResidencySizingError::InvalidLocalGeometry;
    let names = tasks
        .iter()
        .map(|task| task.name())
        .collect::<BTreeSet<_>>();
    let mut parameters = Vec::new();
    for task in tasks {
        let target = std::iter::once(task.name())
            .chain(task.aliases().iter().map(String::as_str))
            .find(|name| layout.contains(name))
            .ok_or_else(|| InvalidLocalGeometry(task.name().into()))?;
        if !owned.contains(target) {
            continue;
        }
        let placement = layout.tensor(target).expect("resolved target");
        let bytes = if matches!(
            task.lowering(),
            crate::WeightLoweringKind::Transform | crate::WeightLoweringKind::DerivedTransform
        ) {
            let shape = task
                .logical_shape()
                .iter()
                .zip(placement.global_shape())
                .zip(placement.local_shape())
                .map(|((&logical, &global), &local)| {
                    logical
                        .checked_mul(local)
                        .filter(|_| global != 0)
                        .filter(|scaled| scaled % global == 0)
                        .map(|scaled| scaled / global)
                        .ok_or_else(|| InvalidLocalGeometry(target.into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if shape.len() != task.logical_shape().len() {
                return Err(InvalidLocalGeometry(target.into()));
            }
            let dtype = task
                .source_encoding()
                .scalar_dtype()
                .map(eredu_checkpoint::recipe::RecipeDtype::from)
                .ok_or_else(|| InvalidLocalGeometry(target.into()))?;
            crate::selected_addressable_parameter_bytes(
                task,
                &eredu_checkpoint::recipe::RecipeMetadata {
                    shape,
                    dtype,
                    byte_len: 0,
                },
            )
            .map_err(|_| InvalidLocalGeometry(target.into()))?
        } else {
            local_output_bytes(
                selected_materialization_task_bytes(task)?,
                placement,
                target,
            )?
        };
        parameters.push((task.name(), task.owner(), bytes));
        if matches!(
            task.lowering(),
            crate::WeightLoweringKind::Transform | crate::WeightLoweringKind::DerivedTransform
        ) {
            continue;
        }
        for companion in task.output_companions() {
            if names.contains(companion.name()) {
                continue;
            }
            let bytes = companion_parameter_bytes(companion)?;
            let placement = layout
                .tensor(companion.name())
                .ok_or_else(|| InvalidLocalGeometry(companion.name().into()))?;
            if !owned.contains(companion.name()) {
                return Err(InvalidLocalGeometry(companion.name().into()));
            }
            parameters.push((
                task.name(),
                task.owner(),
                local_output_bytes(bytes, placement, companion.name())?,
            ));
        }
    }
    parameter_resources(parameters, routed, independently_acquired)
}

fn local_output_bytes(
    bytes: u64,
    placement: &crate::LocalTensorLayout,
    name: &str,
) -> Result<u64, BoundedResidencySizingError> {
    let invalid = || BoundedResidencySizingError::InvalidLocalGeometry(name.into());
    let elements = |shape: &[usize]| {
        shape
            .iter()
            .try_fold(1u128, |n, &d| n.checked_mul(d as u128))
            .ok_or_else(invalid)
    };
    let global = elements(placement.global_shape())?;
    let local = elements(placement.local_shape())?;
    let scaled = (bytes as u128).checked_mul(local).ok_or_else(invalid)?;
    if global == 0 || scaled % global != 0 {
        return Err(invalid());
    }
    u64::try_from(scaled / global).map_err(|_| invalid())
}

fn companion_parameter_bytes(
    companion: &crate::ReplicatedTextOutputCompanion,
) -> Result<u64, BoundedResidencySizingError> {
    if let Some(task) = companion.materialization_task() {
        Ok(selected_materialization_task_bytes(task)?)
    } else if let Some(output) = companion.derived_output() {
        Ok(output.byte_len())
    } else if let Some(source) = companion.catalog_source() {
        Ok(source.encoded_byte_len())
    } else {
        Err(BoundedResidencySizingError::MissingCompanionGeometry(
            companion.name().into(),
        ))
    }
}

type ParameterBytes<'a> = (&'a str, &'a ReplicatedTextParameterOwner, u64);

fn task_parameter_bytes(
    tasks: &[crate::ReplicatedTextMaterializationTask],
) -> Result<Vec<ParameterBytes<'_>>, BoundedResidencySizingError> {
    let names = tasks
        .iter()
        .map(|task| task.name())
        .collect::<BTreeSet<_>>();
    let mut parameters = Vec::new();
    for task in tasks {
        parameters.push((
            task.name(),
            task.owner(),
            selected_materialization_task_bytes(task)?,
        ));
        if matches!(
            task.lowering(),
            crate::WeightLoweringKind::Transform | crate::WeightLoweringKind::DerivedTransform
        ) {
            continue;
        }
        for companion in task.output_companions() {
            if names.contains(companion.name()) {
                continue;
            }
            let bytes = companion_parameter_bytes(companion)?;
            // Atomic companions have the primary's execution owner and bank.
            parameters.push((task.name(), task.owner(), bytes));
        }
    }
    Ok(parameters)
}

fn parameter_resources<'a>(
    parameters: impl IntoIterator<Item = (&'a str, &'a ReplicatedTextParameterOwner, u64)>,
    routed: &BTreeSet<String>,
    independently_acquired: &BTreeSet<String>,
) -> Result<SelectedParameterResources, BoundedResidencySizingError> {
    use BoundedResidencySizingError::ArithmeticOverflow;
    let add = |total: &mut u64, bytes| -> Result<(), BoundedResidencySizingError> {
        *total = total
            .checked_add(bytes)
            .ok_or(ArithmeticOverflow("selected parameter bytes"))?;
        Ok(())
    };
    let mut result = SelectedParameterResources {
        parameter_bytes: 0,
        pinned_bytes: 0,
        largest_unit_bytes: 0,
        largest_adjacent_units_bytes: 0,
        expert_bytes: 0,
    };
    let mut groups = BTreeMap::<&str, BTreeMap<usize, u64>>::new();
    for (name, owner, bytes) in parameters {
        add(&mut result.parameter_bytes, bytes)?;
        if routed.contains(name) {
            add(&mut result.expert_bytes, bytes)?;
        }
        if independently_acquired.contains(name) {
            continue;
        }
        match owner {
            ReplicatedTextParameterOwner::StaticRole(_) => add(&mut result.pinned_bytes, bytes)?,
            ReplicatedTextParameterOwner::ExecutionUnit { group, unit } => {
                add(
                    groups.entry(group).or_default().entry(*unit).or_default(),
                    bytes,
                )?;
            }
        }
    }
    for units in groups.values() {
        for (&unit, &bytes) in units {
            result.largest_unit_bytes = result.largest_unit_bytes.max(bytes);
            let next = unit
                .checked_add(1)
                .and_then(|next| units.get(&next))
                .copied()
                .unwrap_or(0);
            let pair = bytes
                .checked_add(next)
                .ok_or(ArithmeticOverflow("adjacent execution-unit bytes"))?;
            result.largest_adjacent_units_bytes = result.largest_adjacent_units_bytes.max(pair);
        }
    }
    Ok(result)
}

fn bounded_requirement<'a, F>(
    parameters: impl IntoIterator<Item = (&'a str, &'a ReplicatedTextParameterOwner, F)>,
    residency: LayerWeightResidency,
    excluded: &BTreeSet<String>,
) -> Result<BoundedResidencyRequirement, BoundedResidencySizingError>
where
    F: FnOnce() -> Result<u64, ReplicatedTextContractError>,
{
    use BoundedResidencySizingError::ArithmeticOverflow;

    let mut static_bytes = 0u64;
    let mut groups = BTreeMap::<&str, BTreeMap<usize, u64>>::new();
    for (name, owner, bytes) in parameters {
        if excluded.contains(name) {
            continue;
        }
        let bytes = bytes()?;
        match owner {
            ReplicatedTextParameterOwner::StaticRole(_) => {
                static_bytes = static_bytes
                    .checked_add(bytes)
                    .ok_or(ArithmeticOverflow("static parameter bytes"))?;
            }
            ReplicatedTextParameterOwner::ExecutionUnit { group, unit } => {
                let total = groups.entry(group).or_default().entry(*unit).or_default();
                *total = total
                    .checked_add(bytes)
                    .ok_or(ArithmeticOverflow("execution-unit bytes"))?;
            }
        }
    }
    let unit_count = groups.values().try_fold(0usize, |total, units| {
        let count = units
            .last_key_value()
            .map_or(Some(0), |(unit, _)| unit.checked_add(1))
            .ok_or(ArithmeticOverflow("execution-group unit count"))?;
        total
            .checked_add(count)
            .ok_or(ArithmeticOverflow("execution unit count"))
    })?;
    let depth = residency.device_depth(unit_count);
    let mut window_bytes = 0u64;
    if depth != 0 {
        for units in groups.values() {
            let mut window = VecDeque::<(usize, u64)>::new();
            let mut current = 0u64;
            for (&unit, &bytes) in units {
                while window
                    .front()
                    .is_some_and(|(first, _)| unit - first >= depth)
                {
                    let (_, expired) = window.pop_front().expect("window is nonempty");
                    current -= expired;
                }
                current = current
                    .checked_add(bytes)
                    .ok_or(ArithmeticOverflow("device-window bytes"))?;
                window.push_back((unit, bytes));
                window_bytes = window_bytes.max(current);
            }
        }
    }
    let required_bytes = static_bytes
        .checked_add(window_bytes)
        .ok_or(ArithmeticOverflow("bounded-residency bytes"))?;
    Ok(BoundedResidencyRequirement {
        static_bytes,
        window_bytes,
        required_bytes,
        depth,
    })
}

/// Projects a neutral residency snapshot into its stable telemetry document.
pub fn residency_telemetry(report: &ResidencyReport) -> ResidencyTelemetry {
    let offload = report.offload();
    let planned = offload.planned_bytes();
    let current = offload.resident_bytes();
    let peak = offload.peak_resident_bytes();
    let transfers = TransferDirection::ALL
        .into_iter()
        .map(|direction| {
            let metrics = offload.transfer(direction);
            TransferTelemetry {
                direction: match direction {
                    TransferDirection::DeviceToHost => "device_to_host",
                    TransferDirection::DeviceToDisk => "device_to_disk",
                    TransferDirection::HostToDevice => "host_to_device",
                    TransferDirection::HostToDisk => "host_to_disk",
                    TransferDirection::DiskToDevice => "disk_to_device",
                    TransferDirection::DiskToHost => "disk_to_host",
                }
                .into(),
                count: metrics.count(),
                bytes: metrics.bytes(),
                seconds: DurationSeconds(metrics.duration().as_secs_f64()),
            }
        })
        .collect();
    ResidencyTelemetry {
        planned_disk_bytes: planned.get(MemoryTier::Disk),
        planned_host_bytes: planned.get(MemoryTier::Host),
        planned_device_bytes: planned.get(MemoryTier::Device),
        current_host_bytes: current.get(MemoryTier::Host),
        current_device_bytes: current.get(MemoryTier::Device),
        peak_host_bytes: peak.get(MemoryTier::Host),
        peak_device_bytes: peak.get(MemoryTier::Device),
        transfers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::residency::{OffloadConfig, OffloadTelemetry, TierByteTotals};

    fn unit(group: &str, unit: usize) -> ReplicatedTextParameterOwner {
        ReplicatedTextParameterOwner::ExecutionUnit {
            group: group.into(),
            unit,
        }
    }

    fn host(depth: usize) -> LayerWeightResidency {
        LayerWeightResidency::LayerwiseHost(crate::LayerwiseLoadOptions::new(
            OffloadConfig::new(None, None, depth).unwrap(),
        ))
    }

    #[test]
    fn resources_keep_both_banks_shared_experts_and_sparse_windows() {
        let pinned = ReplicatedTextParameterOwner::StaticRole("embedding".into());
        let first = unit("text", 0);
        let second = unit("text", 1);
        let fourth = unit("text", 3);
        let parameters = [
            ("embedding", &pinned, 17),
            ("dense", &first, 11),
            ("attention", &second, 13),
            ("shared", &second, 7),
            ("values", &second, 64),
            ("feed_forward", &second, 100),
            ("later", &fourth, 40),
        ];
        let banks = BTreeSet::from(["values".into(), "feed_forward".into()]);
        let independent = parameter_resources(parameters, &banks, &banks).unwrap();
        assert_eq!(
            independent,
            SelectedParameterResources {
                parameter_bytes: 252,
                pinned_bytes: 17,
                largest_unit_bytes: 40,
                largest_adjacent_units_bytes: 40,
                expert_bytes: 164,
            }
        );
        let with_layer = parameter_resources(parameters, &banks, &BTreeSet::new()).unwrap();
        assert_eq!(with_layer.parameter_bytes, independent.parameter_bytes);
        assert_eq!(with_layer.expert_bytes, 164);
        assert_eq!(with_layer.largest_unit_bytes, 184);
        assert_eq!(with_layer.largest_adjacent_units_bytes, 195);
        assert!(parameter_resources(
            [("overflow", &pinned, u64::MAX), ("extra", &first, 1)],
            &banks,
            &banks
        )
        .is_err());
    }

    fn size(
        parameters: &[(&str, ReplicatedTextParameterOwner, u64)],
        residency: LayerWeightResidency,
        excluded: &BTreeSet<String>,
    ) -> Result<BoundedResidencyRequirement, BoundedResidencySizingError> {
        bounded_requirement(
            parameters
                .iter()
                .map(|(name, owner, bytes)| (*name, owner, || Ok(*bytes))),
            residency,
            excluded,
        )
    }

    #[test]
    fn sizing_keeps_group_windows_and_excluded_parameters_exact() {
        let parameters = [
            (
                "embedding",
                ReplicatedTextParameterOwner::StaticRole("input".into()),
                7,
            ),
            ("a", unit("decoder", 0), 2),
            ("b", unit("decoder", 1), 3),
            ("c", unit("decoder", 1), 5),
            ("d", unit("decoder", 2), 11),
            ("e", unit("vision", 0), 13),
            ("f", unit("vision", 2), 17),
            ("bank", unit("decoder", usize::MAX), u64::MAX),
        ];
        let excluded = BTreeSet::from(["bank".into()]);
        assert_eq!(
            size(&parameters, host(2), &excluded).unwrap(),
            BoundedResidencyRequirement {
                static_bytes: 7,
                window_bytes: 19,
                required_bytes: 26,
                depth: 2,
            },
        );
        assert_eq!(
            size(&parameters, host(3), &excluded).unwrap().window_bytes,
            30
        );
        let streamed = LayerWeightResidency::DenseDiskStream(
            crate::DenseDiskStreamLoadOptions::new(1024, 2048, 5, 2).unwrap(),
        );
        assert_eq!(size(&parameters, streamed, &excluded).unwrap().depth, 2);
    }

    #[test]
    fn excluded_parameters_do_not_evaluate_byte_geometry() {
        let owner = unit("decoder", usize::MAX);
        let requirement = bounded_requirement(
            [("bank", &owner, || {
                panic!("excluded byte geometry must not be queried")
            })],
            host(3),
            &BTreeSet::from(["bank".into()]),
        )
        .unwrap();
        assert_eq!(requirement.required_bytes, 0);
    }

    #[test]
    fn sparse_unit_indices_do_not_expand_all_intervening_positions() {
        let parameters = [
            ("a", unit("decoder", 0), 11),
            ("b", unit("decoder", usize::MAX - 1), 17),
        ];
        assert_eq!(
            size(&parameters, host(2), &BTreeSet::new())
                .unwrap()
                .window_bytes,
            17
        );
        assert_eq!(
            size(&parameters, host(usize::MAX), &BTreeSet::new())
                .unwrap()
                .window_bytes,
            28
        );
    }

    #[test]
    fn sparse_windows_equal_the_dense_geometry_for_every_small_occupancy() {
        for occupied in 0u16..256 {
            let bytes = (0..8)
                .map(|unit| {
                    if occupied & (1 << unit) == 0 {
                        0
                    } else {
                        unit as u64 + 1
                    }
                })
                .collect::<Vec<_>>();
            let parameters = bytes
                .iter()
                .enumerate()
                .filter(|(_, bytes)| **bytes != 0)
                .map(|(index, bytes)| ("weight", unit("decoder", index), *bytes))
                .collect::<Vec<_>>();
            for depth in 1..=10 {
                let expected = (0..bytes.len())
                    .map(|start| bytes[start..].iter().take(depth).sum::<u64>())
                    .max()
                    .unwrap_or(0);
                assert_eq!(
                    size(&parameters, host(depth), &BTreeSet::new())
                        .unwrap()
                        .window_bytes,
                    expected
                );
            }
        }
    }

    #[test]
    fn every_byte_and_unit_count_overflow_is_rejected() {
        use BoundedResidencySizingError::ArithmeticOverflow;
        let pinned = ReplicatedTextParameterOwner::StaticRole("input".into());
        for (parameters, expected) in [
            (
                vec![("a", pinned.clone(), u64::MAX), ("b", pinned.clone(), 1)],
                "static parameter bytes",
            ),
            (
                vec![("a", unit("g", 0), u64::MAX), ("b", unit("g", 0), 1)],
                "execution-unit bytes",
            ),
            (
                vec![("a", unit("g", 0), u64::MAX), ("b", unit("g", 1), 1)],
                "device-window bytes",
            ),
            (
                vec![("a", pinned, u64::MAX), ("b", unit("g", 0), 1)],
                "bounded-residency bytes",
            ),
            (
                vec![("a", unit("g", usize::MAX), 1)],
                "execution-group unit count",
            ),
            (
                vec![("a", unit("g", usize::MAX - 1), 1), ("b", unit("h", 0), 1)],
                "execution unit count",
            ),
        ] {
            assert_eq!(
                size(&parameters, host(2), &BTreeSet::new()),
                Err(ArithmeticOverflow(expected))
            );
        }
    }

    #[test]
    fn residency_projection_preserves_every_tier_and_ordered_transfer() {
        let mut source = OffloadTelemetry::default();
        source.set_planned_bytes(TierByteTotals::new(303, 202, 101));
        source.set_resident_bytes(MemoryTier::Host, 90);
        source.set_resident_bytes(MemoryTier::Host, 70);
        source.set_resident_bytes(MemoryTier::Device, 60);
        source.set_resident_bytes(MemoryTier::Device, 40);
        for (index, direction) in TransferDirection::ALL.into_iter().enumerate() {
            source.record_transfer(
                direction,
                10 + index as u64,
                std::time::Duration::from_millis(125),
            );
            source.record_transfer(direction, 20, std::time::Duration::from_millis(375));
        }
        let diagnostics = eredu_checkpoint::store::WeightStoreDiagnostics {
            backend: eredu_checkpoint::store::WeightStoreBackend::Memory,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            currently_cached_shards: 0,
            touched_shard_paths: vec![],
            payload_shard_paths: vec![],
            physical_reads: 0,
            physical_read_bytes: 0,
            coalesced_group_hits: 0,
        };
        let report = ResidencyReport::new(true, source.snapshot(), vec![], vec![], diagnostics);
        assert_eq!(
            residency_telemetry(&report),
            ResidencyTelemetry {
                planned_disk_bytes: 101,
                planned_host_bytes: 202,
                planned_device_bytes: 303,
                current_host_bytes: 70,
                current_device_bytes: 40,
                peak_host_bytes: 90,
                peak_device_bytes: 60,
                transfers: [
                    "device_to_host",
                    "device_to_disk",
                    "host_to_device",
                    "host_to_disk",
                    "disk_to_device",
                    "disk_to_host"
                ]
                .into_iter()
                .enumerate()
                .map(|(index, direction)| TransferTelemetry {
                    direction: direction.into(),
                    count: 2,
                    bytes: 30 + index as u64,
                    seconds: DurationSeconds(0.5),
                })
                .collect(),
            }
        );
    }
}
