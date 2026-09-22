//! Retained cold placement compared with independently prepared ranks and providers.
use super::*;
use eredu_architectures::component_partition::ComponentPartitionLayouts;
use eredu_core::{capture::*, *};
use eredu_runtime::capture::partition::*;

pub(super) fn layouts(
    sources: &eredu_architectures::prepared_sources::PreparedModelSources,
    parameters: &eredu_runtime::ArchitectureParameterDescription,
) -> ComponentPartitionLayouts {
    let discovery = sources
        .prepare_discovery(Default::default(), Default::default())
        .bind_partition_parameters(Some(Arc::new(parameters.clone())))
        .unwrap();
    assert!(!discovery.identity_is_resolved(), "placement is cold");
    let layouts = discovery.component_partition_layouts(64).unwrap().unwrap();
    assert!(
        !discovery.identity_is_resolved(),
        "no checkpoint hashing for placement"
    );
    layouts
}

pub(super) fn plan(descriptor: &ArchitectureDescriptor) -> AdmittedCapturePlan {
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::RoutedUnits],
        max_histogram_bins: 0,
        conditions: vec![],
    };
    let selections = descriptor
        .routed_components
        .iter()
        .flat_map(|component| {
            [&component.activation, &component.effective_activation].map(|path| CaptureSelection {
                id: path.clone(),
                path: path.clone(),
                schedule: CaptureSchedule::default(),
                slices: vec![CaptureSlice {
                    axis: "component".into(),
                    start: 1,
                    end: component.units_per_expert as u64,
                    stride: 2,
                }],
                transform: CaptureTransform::RoutedUnits,
            })
        })
        .collect::<Vec<_>>();
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: capabilities.clone(),
        points: selections
            .iter()
            .map(|selection| ObservationSupport {
                path: selection.path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            })
            .collect(),
    };
    let usage = CaptureUsage {
        captures: 1000,
        retained_bytes: 1_000_000_000,
        host_bytes: 1_000_000_000,
        encoded_bytes: 1_000_000_000,
    };
    CapturePlan {
        schema_version: 1,
        selections,
        limits: CaptureLimits {
            per_step: usage,
            cumulative: CaptureUsage {
                retained_bytes: 4_000_000_000,
                host_bytes: 4_000_000_000,
                encoded_bytes: 4_000_000_000,
                ..usage
            },
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &descriptor.observations,
        &support,
        &capabilities,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 3,
        },
    )
    .unwrap()
}

pub(super) fn limits() -> PartitionCaptureReceiptLimits {
    PartitionCaptureReceiptLimits {
        max_producers: 64,
        max_fragments: 128,
        max_record_bytes: 1 << 20,
    }
}

pub(super) fn verify(layouts: &ComponentPartitionLayouts, descriptor: &ArchitectureDescriptor) {
    let plan = plan(descriptor);
    let topology = layouts.topology();
    for (index, selection) in plan.plan().selections.iter().enumerate() {
        for (phase, prediction) in [
            (CapturePhase::Prefill, 0),
            (CapturePhase::Decode, 1),
            (CapturePhase::Decode, 2),
        ] {
            let placement = layouts
                .routed_capture_placement(&plan, index, phase, prediction, limits())
                .unwrap();
            assert_eq!(placement.effective, index % 2 == 1);
            assert_eq!(
                layouts.observation_site(&selection.path),
                Some(eredu_runtime::inspection::ObservationHookSite::RoutedUnits)
            );
            let dense =
                eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true);
            assert_ne!(
                layouts.capture_hook_support(&selection.path, dense),
                ObservationSupportStatus::Supported
            );
            assert_eq!(
                layouts.capture_hook_support(
                    &selection.path,
                    dense.with_units(false).with_routed_units(true)
                ),
                ObservationSupportStatus::Supported
            );
            assert_eq!(
                layouts.capture_hook_members(&selection.path).unwrap(),
                placement.sources.iter().map(|s| s.rank).collect::<Vec<_>>()
            );
            assert_eq!(
                placement.sources.len(),
                topology.tensor() * topology.expert()
            );
            assert_eq!(
                placement.producers.len(),
                placement.sources.len(),
                "no redundant expert/unit maps in this topology"
            );
            let mut covered = std::collections::BTreeSet::new();
            for producer in &placement.producers {
                let source = placement
                    .sources
                    .iter()
                    .find(|source| source.rank == producer.rank)
                    .unwrap();
                assert_eq!(source.ownership, producer.ownership);
                let ownership = &producer.ownership;
                assert_eq!(ownership.source_peer, (topology.expert() > 1).then_some(0));
                let map = &ownership.coordinates;
                for local_expert in 0..map.experts().local_count() {
                    for local_unit in 0..map.units().local_count() {
                        assert!(
                            covered.insert(map.local_to_global(local_expert, local_unit).unwrap())
                        );
                    }
                }
                for fragment in producer.projection.fragments() {
                    let local = fragment.local();
                    let destination = fragment.destination();
                    for column in 0..local.shape[2] {
                        let actual = map
                            .units()
                            .local_to_global((local.starts[2] + column * local.strides[2]) as usize)
                            .unwrap();
                        let selected = destination.starts[2] + column * destination.strides[2];
                        let slice = producer.projection.global_slice();
                        assert_eq!(actual as u64, slice.starts[2] + selected * slice.strides[2]);
                    }
                }
            }
            let component = &descriptor.routed_components[index / 2];
            assert_eq!(
                covered.len(),
                component.expert_count * component.units_per_expert
            );
            let mut too_few = limits();
            too_few.max_producers = placement.producers.len() - 1;
            assert!(layouts
                .routed_capture_placement(&plan, index, phase, prediction, too_few)
                .is_err());
            let mut no_fragments = limits();
            no_fragments.max_fragments = 0;
            assert!(layouts
                .routed_capture_placement(&plan, index, phase, prediction, no_fragments)
                .is_err());
        }
        assert!(layouts
            .routed_capture_placement(&plan, index, CapturePhase::Decode, 3, limits())
            .is_err());
    }
}

pub(super) fn verify_rejections(
    sources: &eredu_architectures::prepared_sources::PreparedModelSources,
    parameters: &eredu_runtime::ArchitectureParameterDescription,
) {
    for fault in [
        "input width",
        "expert overflow",
        "route count",
        "invocation owner",
    ] {
        let mut descriptor = sources.architecture().architecture_descriptor();
        let component = &mut descriptor.routed_components[0];
        match fault {
            "input width" => component.input_width += 1,
            "expert overflow" => component.expert_count = usize::MAX,
            "route count" => {
                let point = descriptor
                    .observations
                    .points
                    .iter_mut()
                    .find(|point| point.path == component.activation)
                    .unwrap();
                let ObservationValueType::RoutedUnits { geometry, .. } = &mut point.value_type
                else {
                    unreachable!()
                };
                geometry.routes_per_token += 1;
            }
            _ => {
                descriptor
                    .nodes
                    .iter_mut()
                    .find(|node| node.id == component.node_id)
                    .unwrap()
                    .parent = None;
            }
        }
        assert!(
            sources
                .selected()
                .execution()
                .component_partition_layout(&descriptor, parameters)
                .is_err(),
            "cold placement must reject {fault}"
        );
    }
}
