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
        tasks.iter().map(|task| {
            (task.name(), task.owner(), || {
                selected_materialization_task_bytes(task)
            })
        }),
        selected.residency(),
        excluded,
    )
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
