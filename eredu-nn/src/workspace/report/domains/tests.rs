use super::*;
use eredu_core::{MemoryDeviceId, MemoryDomainDescription, MemoryLocation, MemoryPlacement};

fn topology(shared: bool) -> MemoryTopology {
    let gpu = MemoryLocation::Device(MemoryDeviceId {
        backend: "fixture",
        ordinal: 0,
    });
    MemoryTopology::new(if shared {
        vec![MemoryDomainDescription {
            name: "shared".into(),
            locations: vec![MemoryLocation::Host, gpu],
        }]
    } else {
        vec![
            MemoryDomainDescription {
                name: "host".into(),
                locations: vec![MemoryLocation::Host],
            },
            MemoryDomainDescription {
                name: "gpu0".into(),
                locations: vec![gpu],
            },
        ]
    })
    .unwrap()
}
fn input<'a>(
    opening: &'a [usize],
    allocations: &'a [usize],
    closing: &'a [usize],
    borrowed: Option<&'a [usize]>,
) -> WorkspaceDomainReportInputs<'a> {
    WorkspaceDomainReportInputs {
        opening: Some(opening),
        allocations,
        closing,
        borrowed,
        host_workspace: Some(0),
        tensor_complete: true,
    }
}
fn destinations(topology: &MemoryTopology) -> Vec<WorkspaceDomainReportEntry> {
    topology
        .domains()
        .map(|(id, _)| WorkspaceDomainReportEntry::empty(id))
        .collect()
}
fn placement(topology: &MemoryTopology, device: bool) -> MemoryPlacement {
    MemoryPlacement::fixed(
        topology,
        if device {
            topology
                .domain_for(MemoryLocation::Device(MemoryDeviceId {
                    backend: "fixture",
                    ordinal: 0,
                }))
                .unwrap()
        } else {
            topology.host_domain()
        },
    )
    .unwrap()
}

#[test]
fn transfer_overlap_uses_original_lifetimes_in_both_topologies() {
    for shared in [false, true] {
        let topology = topology(shared);
        let nodes = [64, 64].map(|bytes| WorkspaceReportNode {
            bytes: Some(bytes),
            alias_start: 0,
            alias_count: 0,
        });
        let backings = [
            Some(placement(&topology, false)),
            Some(placement(&topology, true)),
        ];
        let scratch = [WorkspaceScratchAllocation {
            maximum_allocations: Some(1),
            bytes: 16,
            host_control_bytes: Some(0),
            placement: Some(placement(&topology, false)),
        }];
        let mut out = destinations(&topology);
        let mut workspace =
            WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(2, 0).unwrap()).unwrap();
        workspace
            .report_domains(
                WorkspaceReportGraph::new(&nodes, &[]).unwrap(),
                input(&[0], &[1], &[1], Some(&[])),
                WorkspaceReportPlacements {
                    topology: &topology,
                    backings: &backings,
                    host_controls: None,
                    scratch: &scratch,
                },
                &mut out,
            )
            .unwrap();
        if shared {
            assert_eq!(out[0].residual.unwrap().total.accounted_bytes, 144);
            assert_eq!(out[0].total.accounted_bytes, 80);
            assert_eq!(out[0].state.unwrap().transient.accounted_bytes, 80);
        } else {
            assert_eq!(out[0].residual.unwrap().total.accounted_bytes, 80);
            assert_eq!(out[1].residual.unwrap().total.accounted_bytes, 64);
            assert_eq!(out[0].total.accounted_bytes, 16);
            assert_eq!(out[1].retained.accounted_bytes, 64);
        }
        // The already registered source is credited by its root identity.
        workspace
            .report_domains(
                WorkspaceReportGraph::new(&nodes, &[]).unwrap(),
                input(&[0], &[1], &[1], Some(&[0])),
                WorkspaceReportPlacements {
                    topology: &topology,
                    backings: &backings,
                    host_controls: None,
                    scratch: &scratch,
                },
                &mut out,
            )
            .unwrap();
        assert_eq!(
            out[0].residual.unwrap().total.accounted_bytes,
            if shared { 80 } else { 16 }
        );
    }
}

#[test]
fn separate_domain_maximum_values_do_not_need_an_aggregate_sum() {
    let topology = topology(false);
    let nodes = [0, 1].map(|_| WorkspaceReportNode {
        bytes: Some(u64::MAX),
        alias_start: 0,
        alias_count: 0,
    });
    let backings = [
        Some(placement(&topology, false)),
        Some(placement(&topology, true)),
    ];
    let mut workspace =
        WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(2, 0).unwrap()).unwrap();
    let mut out = destinations(&topology);
    workspace
        .report_domains(
            WorkspaceReportGraph::new(&nodes, &[]).unwrap(),
            input(&[], &[0, 1], &[0, 1], None),
            WorkspaceReportPlacements {
                topology: &topology,
                backings: &backings,
                host_controls: None,
                scratch: &[],
            },
            &mut out,
        )
        .unwrap();
    assert!(out
        .iter()
        .all(|entry| entry.total.accounted_bytes == u64::MAX));
}

#[test]
fn views_keep_full_backing_and_independent_copy_adds_a_charge() {
    let topology = topology(true);
    let nodes = [
        WorkspaceReportNode {
            bytes: Some(64),
            alias_start: 0,
            alias_count: 0,
        },
        WorkspaceReportNode {
            bytes: Some(0),
            alias_start: 0,
            alias_count: 2,
        },
        WorkspaceReportNode {
            bytes: Some(64),
            alias_start: 0,
            alias_count: 0,
        },
    ];
    let backings = [
        Some(placement(&topology, false)),
        None,
        Some(placement(&topology, true)),
    ];
    let mut out = destinations(&topology);
    let mut workspace =
        WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(3, 2).unwrap()).unwrap();
    workspace
        .report_domains(
            WorkspaceReportGraph::new(&nodes, &[0, 0]).unwrap(),
            input(&[0], &[2], &[1, 2, 1], Some(&[])),
            WorkspaceReportPlacements {
                topology: &topology,
                backings: &backings,
                host_controls: None,
                scratch: &[],
            },
            &mut out,
        )
        .unwrap();
    assert_eq!(out[0].closing.charge.accounted_bytes, 128);
    assert_eq!(out[0].closing.maximum_allocations, 2);
    assert_eq!(out[0].total.accounted_bytes, 64);
}

#[test]
fn managed_candidates_charge_once_per_physical_domain_and_views_share() {
    for shared in [false, true] {
        let topology = topology(shared);
        let managed = MemoryPlacement::possible_locations(
            &topology,
            [
                MemoryLocation::Host,
                MemoryLocation::Device(MemoryDeviceId {
                    backend: "fixture",
                    ordinal: 0,
                }),
            ],
            "backend reports host and registered GPU candidates".into(),
        )
        .unwrap();
        let nodes = [WorkspaceReportNode {
            bytes: Some(64),
            alias_start: 0,
            alias_count: 0,
        }];
        let backings = [Some(managed)];
        let mut workspace =
            WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(1, 0).unwrap()).unwrap();
        let mut out = destinations(&topology);
        workspace
            .report_domains(
                WorkspaceReportGraph::new(&nodes, &[]).unwrap(),
                input(&[], &[0], &[0, 0], None),
                WorkspaceReportPlacements {
                    topology: &topology,
                    backings: &backings,
                    host_controls: None,
                    scratch: &[],
                },
                &mut out,
            )
            .unwrap();
        for entry in out {
            assert_eq!(entry.total.placement_allowance_bytes, 64);
            assert_eq!(entry.total.accounted_bytes, 0);
            assert_eq!(entry.closing.maximum_allocations, 1);
        }
    }
}

#[test]
fn final_domain_overflow_or_missing_placement_leaves_every_destination_unchanged() {
    let topology = topology(false);
    let nodes = [1, u64::MAX, 1].map(|bytes| WorkspaceReportNode {
        bytes: Some(bytes),
        alias_start: 0,
        alias_count: 0,
    });
    let mut backings = [
        Some(placement(&topology, false)),
        Some(placement(&topology, true)),
        Some(placement(&topology, true)),
    ];
    let mut workspace =
        WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(3, 0).unwrap()).unwrap();
    let mut out = destinations(&topology);
    out[0].total.accounted_bytes = 17;
    let before = out.clone();
    let graph = WorkspaceReportGraph::new(&nodes, &[]).unwrap();
    assert_eq!(
        workspace.report_domains(
            graph,
            input(&[], &[0, 1, 2], &[], None),
            WorkspaceReportPlacements {
                topology: &topology,
                backings: &backings,
                host_controls: None,
                scratch: &[]
            },
            &mut out
        ),
        Err(WorkspaceReportError::Domain(
            eredu_core::MemoryDomainError::Overflow
        ))
    );
    assert_eq!(out, before);
    backings[2] = None;
    assert_eq!(
        workspace.report_domains(
            graph,
            input(&[], &[0, 1, 2], &[], None),
            WorkspaceReportPlacements {
                topology: &topology,
                backings: &backings,
                host_controls: None,
                scratch: &[]
            },
            &mut out
        ),
        Err(WorkspaceReportError::IncompletePlacement)
    );
    assert_eq!(out, before);
}

#[derive(Clone, Debug)]
struct SelectedFacts {
    topology: Rc<MemoryTopology>,
    native: MemoryPlacement,
    backing_controls: u64,
    scratch_controls: Option<u64>,
    scratch_births: u64,
    population: Option<WorkspaceAllocationPopulation>,
}
impl WorkspaceMechanisms for SelectedFacts {
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.topology)
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&MemoryPlacement> {
        Some(&self.native)
    }
    fn scratch_allocations(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Error> {
        Ok(self.population.as_ref())
    }
    fn scratch_placement(&self, _: WorkspaceOperationView<'_>) -> Option<&MemoryPlacement> {
        Some(&self.native)
    }
    fn allocation_host_control_bytes(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<u64> {
        Some(self.backing_controls)
    }
    fn scratch_host_control_bytes(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Error> {
        self.scratch_controls
            .map(|bytes| {
                bytes.checked_mul(self.scratch_births).ok_or_else(|| {
                    Error::backend_retained_source(eredu_core::MemoryDomainError::Overflow)
                })
            })
            .transpose()
    }
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|v| v.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: self
                .population
                .as_ref()
                .map_or(4, |p| p.backing_bytes().unwrap()),
            assumptions: "fact".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 8,
            assumptions: "host".into(),
        }))
    }
}
impl WorkspaceFactMechanisms for SelectedFacts {
    type Error = Error;
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.topology)
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&MemoryPlacement> {
        Some(&self.native)
    }
    fn scratch_allocations(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Error> {
        Ok(self.population.as_ref())
    }
    fn scratch_placement(&self, _: WorkspaceOperationView<'_>) -> Option<&MemoryPlacement> {
        Some(&self.native)
    }
    fn allocation_host_control_bytes(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<u64> {
        Some(self.backing_controls)
    }
    fn scratch_host_control_bytes(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Error> {
        self.scratch_controls
            .map(|bytes| {
                bytes.checked_mul(self.scratch_births).ok_or_else(|| {
                    Error::backend_retained_source(eredu_core::MemoryDomainError::Overflow)
                })
            })
            .transpose()
    }
    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Error> {
        Ok(Some(WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: op.outputs.len(),
                aliases: 0,
                assumption_bytes: 4,
            },
            scratch_bytes: self
                .population
                .as_ref()
                .map_or(4, |p| p.backing_bytes().unwrap()),
        }))
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Error> {
        let facts = self.operation_facts(op)?.unwrap();
        out.validate(facts.layout)
            .map_err(Error::backend_retained_source)?;
        for (index, output) in out.outputs.iter_mut().enumerate() {
            *output = WorkspaceOutputEffect::Allocate(
                op.outputs
                    .get(index)
                    .unwrap()
                    .bytes()
                    .map_err(Error::backend_retained_source)?,
            );
        }
        out.assumptions.copy_from_slice(b"fact");
        Ok(Some(facts))
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Error> {
        Ok(Some(WorkspaceHostFacts {
            bytes: 8,
            assumption_bytes: 4,
        }))
    }
    fn write_host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Error> {
        let facts = self.host_facts(op)?.unwrap();
        out.validate(facts)
            .map_err(Error::backend_retained_source)?;
        out.assumptions.copy_from_slice(b"host");
        Ok(Some(facts))
    }
}

#[test]
fn finite_copy_preserves_source_placement_and_charges_independent_destinations() {
    let topology = Rc::new(topology(false));
    let native = placement(&topology, true);
    let facts = SelectedFacts {
        topology: topology.clone(),
        native,
        backing_controls: 0,
        scratch_controls: Some(0),
        scratch_births: 1,
        population: None,
    };
    let context = WorkspaceContext::new_recording_facts(facts.clone());
    let storage =
        WorkspaceExistingStorage::try_new_placed(Some(64), &placement(&topology, false), &context)
            .unwrap();
    let selected = WorkspaceBorrowedStorage::new(&context, [&storage]).unwrap();
    let source = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[4], WorkspaceDtype::Float32).unwrap(),
        &storage,
        &context,
    )
    .unwrap();
    let before = context.metadata_census().unwrap().context_bytes();
    assert_eq!(
        selected
            .requirements(&context)
            .unwrap()
            .get(topology.host_domain())
            .unwrap()
            .accounted_bytes,
        64
    );
    let plan = WorkspaceIsolatedCopyPlan::prepare_finite(&context, &selected, &[source], &facts)
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(plan.incremental_bytes(), Some(56));
    let requirements = plan.incremental_requirements().unwrap();
    assert_eq!(
        requirements
            .get(topology.host_domain())
            .unwrap()
            .accounted_bytes,
        16
    );
    assert_eq!(
        requirements
            .get(placement(&topology, true).domains()[0])
            .unwrap()
            .accounted_bytes,
        40
    );
    assert!(context.metadata_census().unwrap().context_bytes() > before);
    let full = plan.report().physical_domains.as_ref().unwrap();
    assert_eq!(
        full.retained_state
            .as_ref()
            .unwrap()
            .get(topology.host_domain())
            .unwrap()
            .accounted_bytes,
        64
    );
}

#[test]
fn attributed_borrowed_domains_accept_separate_maximum_capacities() {
    let topology = Rc::new(topology(false));
    let context = WorkspaceContext::new(SelectedFacts {
        topology: topology.clone(),
        native: placement(&topology, true),
        backing_controls: 0,
        scratch_controls: Some(0),
        scratch_births: 1,
        population: None,
    });
    let a = WorkspaceExistingStorage::try_new_placed(
        Some(u64::MAX),
        &placement(&topology, false),
        &context,
    )
    .unwrap();
    let b = WorkspaceExistingStorage::try_new_placed(
        Some(u64::MAX),
        &placement(&topology, true),
        &context,
    )
    .unwrap();
    let borrowed = WorkspaceBorrowedStorage::new(&context, [&a, &b, &a]).unwrap();
    assert_eq!(borrowed.roots().len(), 2);
    assert_eq!(borrowed.total_bytes(), None);
}

#[test]
fn backing_controls_follow_identity_and_closing_lifetimes_in_each_topology() {
    for unified in [false, true] {
        let topology = topology(unified);
        let nodes = [
            WorkspaceReportNode {
                bytes: Some(64),
                alias_start: 0,
                alias_count: 0,
            },
            WorkspaceReportNode {
                bytes: Some(0),
                alias_start: 0,
                alias_count: 1,
            },
            WorkspaceReportNode {
                bytes: Some(64),
                alias_start: 0,
                alias_count: 0,
            },
        ];
        let backings = [
            Some(placement(&topology, true)),
            None,
            Some(placement(&topology, true)),
        ];
        let controls = [Some(7), Some(0), Some(7)];
        let scratch = [WorkspaceScratchAllocation {
            maximum_allocations: Some(1),
            bytes: 16,
            host_control_bytes: Some(3),
            placement: Some(placement(&topology, true)),
        }];
        let mut workspace =
            WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(3, 1).unwrap()).unwrap();
        let mut out = destinations(&topology);
        workspace
            .report_domains(
                WorkspaceReportGraph::new(&nodes, &[0]).unwrap(),
                input(&[0, 1], &[2], &[1, 2], Some(&[0])),
                WorkspaceReportPlacements {
                    topology: &topology,
                    backings: &backings,
                    host_controls: Some(&controls),
                    scratch: &scratch,
                },
                &mut out,
            )
            .unwrap();
        let host = &out
            .iter()
            .find(|entry| entry.domain == topology.host_domain())
            .unwrap();
        let native_host = if unified { 80 } else { 0 };
        assert_eq!(host.total.accounted_bytes, native_host + 7 + 3);
        assert_eq!(host.retained.accounted_bytes, if unified { 71 } else { 7 });
        assert_eq!(
            host.closing.charge.accounted_bytes,
            if unified { 142 } else { 14 }
        );
        assert_eq!(
            host.residual.unwrap().total.accounted_bytes,
            native_host + 7 + 3
        );
        assert_eq!(host.tensor_buffers.total.accounted_bytes, native_host);
    }
}

#[test]
fn finite_copy_keeps_source_controls_and_funds_new_backing_controls() {
    let topology = Rc::new(topology(false));
    let facts = SelectedFacts {
        topology: topology.clone(),
        native: placement(&topology, true),
        backing_controls: 5,
        scratch_controls: Some(3),
        scratch_births: 1,
        population: None,
    };
    let context = WorkspaceContext::new_recording_facts(facts.clone());
    let storage = WorkspaceExistingStorage::try_new_placed_with_host_controls(
        Some(64),
        &placement(&topology, false),
        Some(7),
        &context,
    )
    .unwrap();
    let selected = WorkspaceBorrowedStorage::new(&context, [&storage]).unwrap();
    let source = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[4], WorkspaceDtype::Float32).unwrap(),
        &storage,
        &context,
    )
    .unwrap();
    assert_eq!(
        selected
            .requirements(&context)
            .unwrap()
            .get(topology.host_domain())
            .unwrap()
            .accounted_bytes,
        71
    );
    let plan = WorkspaceIsolatedCopyPlan::prepare_finite(&context, &selected, &[source], &facts)
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(plan.incremental_bytes(), Some(72));
    let requirements = plan.incremental_requirements().unwrap();
    assert_eq!(
        requirements
            .get(topology.host_domain())
            .unwrap()
            .accounted_bytes,
        32
    );
    let report = plan.report().physical_domains.as_ref().unwrap();
    assert_eq!(
        report
            .native_allocations
            .get(topology.host_domain())
            .unwrap()
            .accounted_bytes,
        0
    );
    assert_eq!(
        report
            .retained_state
            .as_ref()
            .unwrap()
            .get(topology.host_domain())
            .unwrap()
            .accounted_bytes,
        76
    );
}

#[test]
fn scratch_control_overflow_is_an_error_in_both_trace_paths() {
    let topology = Rc::new(topology(false));
    let facts = SelectedFacts {
        topology: topology.clone(),
        native: placement(&topology, true),
        backing_controls: 0,
        scratch_controls: Some(u64::MAX),
        scratch_births: 2,
        population: None,
    };
    let context = WorkspaceContext::new_recording_facts(facts.clone());
    let storage =
        WorkspaceExistingStorage::try_new_placed(Some(16), &facts.native, &context).unwrap();
    let selected = WorkspaceBorrowedStorage::new(&context, [&storage]).unwrap();
    let source = WorkspaceTensor::existing_with_storage(
        context.layout(&[4], WorkspaceDtype::Float32).unwrap(),
        &storage,
        &context,
    )
    .unwrap();
    let ordinary = context
        .execute(
            WorkspaceOperationKind::DeepCopy,
            &[&source],
            vec![source.layout().clone()],
        )
        .unwrap_err();
    assert!(matches!(
        std::error::Error::source(&ordinary)
            .and_then(|cause| cause.downcast_ref::<eredu_core::MemoryDomainError>()),
        Some(eredu_core::MemoryDomainError::Overflow)
    ));
    let fixed = WorkspaceIsolatedCopyPlan::prepare_finite(&context, &selected, &[source], &facts)
        .unwrap()
        .construct()
        .unwrap_err();
    let WorkspaceCopyPreparationError::Mechanism(cause) = fixed else {
        panic!("{fixed:?}");
    };
    assert!(matches!(
        std::error::Error::source(&cause)
            .and_then(|cause| cause.downcast_ref::<eredu_core::MemoryDomainError>()),
        Some(eredu_core::MemoryDomainError::Overflow)
    ));
}

#[test]
fn missing_scratch_controls_identify_the_operation_without_losing_known_staging() {
    let topology = Rc::new(topology(false));
    let facts = SelectedFacts {
        topology: topology.clone(),
        native: placement(&topology, true),
        backing_controls: 0,
        scratch_controls: None,
        scratch_births: 1,
        population: None,
    };
    let context = WorkspaceContext::new(facts.clone());
    let storage =
        WorkspaceExistingStorage::try_new_placed(Some(16), &facts.native, &context).unwrap();
    let selected = WorkspaceBorrowedStorage::new(&context, [&storage]).unwrap();
    let source = WorkspaceTensor::existing_with_storage(
        context.layout(&[4], WorkspaceDtype::Float32).unwrap(),
        &storage,
        &context,
    )
    .unwrap();
    let output = context
        .execute(
            WorkspaceOperationKind::DeepCopy,
            &[&source],
            vec![source.layout().clone()],
        )
        .unwrap();
    let report = context.report(&output).unwrap();
    assert_eq!(report.host_workspace_bytes, Some(8));
    assert_eq!(report.total_bytes, None);
    assert!(report.physical_domains.is_none());
    assert_eq!(report.unpriced_host_operations, [0]);
    assert!(report.unpriced_operations.is_empty());
    let plan = WorkspaceIsolatedCopyPlan::prepare_finite(&context, &selected, &[source], &facts)
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(plan.report().host_workspace_bytes, Some(16));
    assert_eq!(plan.report().total_bytes, None);
    assert!(plan.report().physical_domains.is_none());
    assert_eq!(plan.report().unpriced_host_operations, [0, 1]);
}

#[test]
fn composite_scratch_keeps_original_domains_candidate_basis_and_independent_births() {
    let topology = Rc::new(topology(false));
    let facts = |native: MemoryPlacement, population: Option<WorkspaceAllocationPopulation>| {
        SelectedFacts {
            topology: topology.clone(),
            native,
            backing_controls: 0,
            scratch_controls: Some(0),
            scratch_births: 1,
            population,
        }
    };
    let host = placement(&topology, false);
    let gpu = placement(&topology, true);
    let possible = MemoryPlacement::possible(
        &topology,
        vec![topology.host_domain(), gpu.domains()[0]],
        "actual child candidate set".into(),
    )
    .unwrap();
    let child = |native: MemoryPlacement| {
        let context = WorkspaceContext::new_recording_facts(facts(native.clone(), None));
        let existing =
            WorkspaceExistingStorage::try_new_placed(Some(1024), &native, &context).unwrap();
        context
            .set_borrowed_storage(WorkspaceBorrowedStorage::new(&context, [&existing]).unwrap())
            .unwrap();
        let input = WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(&[256], WorkspaceDtype::Float32).unwrap(),
            &existing,
            &context,
        )
        .unwrap();
        let outputs = context
            .execute(
                WorkspaceOperationKind::Contiguous,
                &[&input],
                vec![WorkspaceLayout::new(&[4], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap();
        let result = context.new_allocation_scratch().unwrap().unwrap();
        assert_eq!(
            result
                .allocations()
                .iter()
                .map(|row| row.bytes)
                .sum::<u64>(),
            20
        );
        drop(outputs);
        drop(input);
        drop(existing);
        drop(context);
        result
    };
    let a = child(host.clone());
    let b = child(gpu.clone());
    let c = child(possible);
    let collector = WorkspaceContext::new_recording_facts(facts(host.clone(), None));
    let combined = collector
        .combine_scratch_populations(&[(&a, 2), (&b, 1), (&c, 1)])
        .unwrap();
    assert_eq!(combined.allocations().len(), 8);
    for finite in [false, true] {
        let selected = facts(gpu.clone(), Some(combined.clone()));
        let context = if finite {
            WorkspaceContext::new_recording_facts(selected)
        } else {
            WorkspaceContext::new(selected)
        };
        for _ in 0..2 {
            context
                .execute(WorkspaceOperationKind::Contiguous, &[], Vec::new())
                .unwrap();
        }
        let report = context.report_domains(&[]).unwrap();
        let h = report
            .native_allocations
            .get(topology.host_domain())
            .unwrap();
        let g = report.native_allocations.get(gpu.domains()[0]).unwrap();
        assert_eq!((h.accounted_bytes, g.accounted_bytes), (80, 40));
        assert_eq!(
            (h.placement_allowance_bytes, g.placement_allowance_bytes),
            (40, 40)
        );
        assert_eq!(report.native_allocations.placement_allowances().len(), 4);
        for allowance in report.native_allocations.placement_allowances() {
            assert!(
                matches!(allowance.placement.kind(),eredu_core::MemoryPlacementKind::Possible{basis,..} if basis=="actual child candidate set")
            );
        }
    }
    let refused =
        WorkspaceContext::new_with_fact_budget(facts(gpu.clone(), Some(combined.clone())), 0);
    assert!(refused
        .execute(WorkspaceOperationKind::Contiguous, &[], Vec::new())
        .is_err());
    assert_eq!(refused.operation_count(), 0);
    assert!(refused.trace.borrow().placed_scratch.is_empty());
    let foreign = Rc::new(self::topology(false));
    let refused = WorkspaceContext::new_recording_facts(SelectedFacts {
        topology: foreign.clone(),
        native: placement(&foreign, true),
        backing_controls: 0,
        scratch_controls: Some(0),
        scratch_births: 1,
        population: Some(combined),
    });
    assert!(refused
        .execute(WorkspaceOperationKind::Contiguous, &[], Vec::new())
        .is_err());
    assert_eq!(refused.operation_count(), 0);
    assert!(refused.trace.borrow().placed_scratch.is_empty());
}

#[test]
fn alternative_scratch_resolves_each_domain_before_peak_and_preserves_categories() {
    let topology = Rc::new(topology(false));
    let host = placement(&topology, false);
    let gpu = placement(&topology, true);
    let facts = |native: MemoryPlacement, population: Option<WorkspaceAllocationPopulation>| {
        SelectedFacts {
            topology: topology.clone(),
            native,
            backing_controls: 0,
            scratch_controls: Some(0),
            scratch_births: 1,
            population,
        }
    };
    let child = |native: MemoryPlacement| {
        let context = WorkspaceContext::new_recording_facts(facts(native, None));
        context
            .execute(
                WorkspaceOperationKind::Contiguous,
                &[],
                vec![WorkspaceLayout::new(&[4], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap();
        context.new_allocation_scratch().unwrap().unwrap()
    };
    let h = child(host.clone());
    let g = child(gpu.clone());
    let candidate = child(
        MemoryPlacement::possible(
            &topology,
            vec![topology.host_domain(), gpu.domains()[0]],
            "original branch allocation candidates".into(),
        )
        .unwrap(),
    );
    let collector = WorkspaceContext::new_recording_facts(facts(host.clone(), None));
    let larger_scalar = collector
        .combine_scratch_populations(&[(&h, 5), (&g, 1)])
        .unwrap();
    let other_domain = collector
        .combine_scratch_populations(&[(&candidate, 4)])
        .unwrap();
    let branches = collector
        .peak_scratch_populations(&[&larger_scalar, &other_domain])
        .unwrap();
    assert_eq!(branches.backing_bytes(), Some(120));
    let repeated = collector
        .combine_scratch_populations(&[(&branches, 2), (&h, 1)])
        .unwrap();
    assert_eq!(repeated.backing_bytes(), Some(260));
    for finite in [false, true] {
        let selected = facts(gpu.clone(), Some(repeated.clone()));
        let context = if finite {
            WorkspaceContext::new_recording_facts(selected)
        } else {
            WorkspaceContext::new(selected)
        };
        context
            .execute(WorkspaceOperationKind::Contiguous, &[], Vec::new())
            .unwrap();
        let physical = context.report_domains(&[]).unwrap();
        let host = physical
            .native_allocations
            .get(topology.host_domain())
            .unwrap();
        let device = physical.native_allocations.get(gpu.domains()[0]).unwrap();
        assert_eq!(
            (host.accounted_bytes, host.placement_allowance_bytes),
            (220, 0)
        );
        assert_eq!(
            (device.accounted_bytes, device.placement_allowance_bytes),
            (0, 160)
        );
        assert_eq!(physical.native_allocations.placement_allowances().len(), 16);
        for allowance in physical.native_allocations.placement_allowances() {
            assert!(
                matches!(allowance.placement.kind(),eredu_core::MemoryPlacementKind::Possible{basis,..}
                if basis=="original branch allocation candidates")
            );
        }
        // A nested child keeps the same resolved branch envelope and raw owners.
        let nested = context.new_allocation_scratch().unwrap().unwrap();
        assert_eq!(nested.backing_bytes(), Some(260));
    }
    let foreign = Rc::new(self::topology(false));
    let refused = WorkspaceContext::new_recording_facts(SelectedFacts {
        topology: foreign.clone(),
        native: placement(&foreign, true),
        backing_controls: 0,
        scratch_controls: Some(0),
        scratch_births: 1,
        population: Some(repeated),
    });
    assert!(refused
        .execute(WorkspaceOperationKind::Contiguous, &[], Vec::new())
        .is_err());
    assert_eq!(refused.operation_count(), 0);
}

#[derive(Clone, Debug)]
struct OutputPopulationFacts {
    topology: Rc<MemoryTopology>,
    source: WorkspaceAllocationPopulation,
}
impl WorkspaceMechanisms for OutputPopulationFacts {
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.topology)
    }
    fn output_allocations(
        &self,
        op: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Error> {
        Ok(matches!(op.kind, WorkspaceOperationKindView::DeepCopy).then_some(&self.source))
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let output = if matches!(op.kind, WorkspaceOperationKind::DeepCopy) {
            WorkspaceOutputStorage::AllocateOrAliasInputs {
                bytes: self.source.backing_bytes().unwrap(),
                inputs: vec![0],
            }
        } else {
            WorkspaceOutputStorage::AliasInput(0)
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![output],
            scratch_bytes: 0,
            assumptions: "rows".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "host".into(),
        }))
    }
}
impl WorkspaceFactMechanisms for OutputPopulationFacts {
    type Error = Error;
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.topology)
    }
    fn output_allocations(
        &self,
        op: WorkspaceOperationView<'_>,
        index: usize,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Error> {
        WorkspaceMechanisms::output_allocations(self, op, index)
    }
    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Error> {
        Ok(Some(WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: 1,
                aliases: usize::from(matches!(op.kind, WorkspaceOperationKindView::DeepCopy)),
                assumption_bytes: 4,
            },
            scratch_bytes: 0,
        }))
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Error> {
        let facts = self.operation_facts(op)?.unwrap();
        out.validate(facts.layout)
            .map_err(Error::backend_retained_source)?;
        out.outputs[0] = if facts.layout.aliases != 0 {
            out.aliases[0] = 0;
            WorkspaceOutputEffect::AllocateOrAliasInputs {
                bytes: self.source.backing_bytes().unwrap(),
                alias_start: 0,
                alias_count: 1,
            }
        } else {
            WorkspaceOutputEffect::AliasInput(0)
        };
        out.assumptions.copy_from_slice(b"rows");
        Ok(Some(facts))
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Error> {
        Ok(Some(WorkspaceHostFacts {
            bytes: 0,
            assumption_bytes: 4,
        }))
    }
    fn write_host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Error> {
        out.assumptions.copy_from_slice(b"host");
        self.host_facts(op)
    }
}

#[test]
fn output_population_keeps_each_domain_alias_lifetime_and_original_source_exclusion() {
    let topology = Rc::new(topology(false));
    let host = placement(&topology, false);
    let gpu = placement(&topology, true);
    let collector = WorkspaceContext::new_recording_facts(SelectedFacts {
        topology: topology.clone(),
        native: gpu.clone(),
        backing_controls: 0,
        scratch_controls: Some(0),
        scratch_births: 0,
        population: None,
    });
    let source = collector
        .source_scratch_population(&[
            WorkspaceScratchAllocation {
                maximum_allocations: Some(2),
                bytes: 40,
                host_control_bytes: Some(2),
                placement: Some(host.clone()),
            },
            WorkspaceScratchAllocation {
                maximum_allocations: Some(3),
                bytes: 60,
                host_control_bytes: Some(3),
                placement: Some(gpu.clone()),
            },
        ])
        .unwrap();
    for finite in [false, true] {
        let facts = OutputPopulationFacts {
            topology: topology.clone(),
            source: source.clone(),
        };
        let context = if finite {
            WorkspaceContext::new_recording_facts(facts)
        } else {
            WorkspaceContext::new(facts)
        };
        let original = WorkspaceExistingStorage::try_new_placed_with_host_controls(
            Some(80),
            &host,
            Some(7),
            &context,
        )
        .unwrap();
        context
            .set_borrowed_storage(WorkspaceBorrowedStorage::new(&context, [&original]).unwrap())
            .unwrap();
        let input = WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap(),
            &original,
            &context,
        )
        .unwrap();
        context.begin_state_span([&input]).unwrap();
        let mut values = context
            .execute(
                WorkspaceOperationKind::DeepCopy,
                &[&input],
                vec![input.layout().clone()],
            )
            .unwrap();
        let first = values.pop().unwrap();
        let second = context
            .execute(
                WorkspaceOperationKind::DeepCopy,
                &[&input],
                vec![input.layout().clone()],
            )
            .unwrap()
            .pop()
            .unwrap();
        let view = context
            .execute(
                WorkspaceOperationKind::View("source-view"),
                &[&second],
                vec![second.layout().clone()],
            )
            .unwrap()
            .pop()
            .unwrap();
        drop((first, second));
        let report = context.report_domains(&[view.clone()]).unwrap();
        let h = report
            .domains
            .iter()
            .find(|row| row.domain == topology.host_domain())
            .unwrap();
        let g = report
            .domains
            .iter()
            .find(|row| row.domain == gpu.domains()[0])
            .unwrap();
        assert_eq!(
            (h.total.accounted_bytes, g.total.accounted_bytes),
            (90, 120)
        );
        assert_eq!(
            (h.retained.accounted_bytes, g.retained.accounted_bytes),
            (45, 60)
        );
        assert_eq!(
            (h.transient.accounted_bytes, g.transient.accounted_bytes),
            (45, 60)
        );
        assert_eq!(
            (
                h.closing.charge.accounted_bytes,
                g.closing.charge.accounted_bytes
            ),
            (132, 60)
        );
        assert_eq!(
            (h.closing.maximum_allocations, g.closing.maximum_allocations),
            (6, 3)
        );
        assert_eq!(
            (
                h.residual.unwrap().closing.charge.accounted_bytes,
                g.residual.unwrap().closing.charge.accounted_bytes
            ),
            (45, 60)
        );
        context.begin_state_span([&view]).unwrap();
        let retired = context.report_domains(&[]).unwrap();
        let h = retired
            .domains
            .iter()
            .find(|row| row.domain == topology.host_domain())
            .unwrap();
        let g = retired
            .domains
            .iter()
            .find(|row| row.domain == gpu.domains()[0])
            .unwrap();
        assert_eq!(
            (
                h.state.unwrap().displaced.accounted_bytes,
                g.state.unwrap().displaced.accounted_bytes
            ),
            (132, 60)
        );
        assert_eq!(
            (
                h.residual.unwrap().displaced.accounted_bytes,
                g.residual.unwrap().displaced.accounted_bytes
            ),
            (45, 60)
        );
    }
}

#[derive(Debug)]
struct SeededAllocationFacts {
    topology: Rc<MemoryTopology>,
    host: MemoryPlacement,
    device: MemoryPlacement,
}
impl WorkspaceMechanisms for SeededAllocationFacts {
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.topology)
    }
    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&MemoryPlacement> {
        Some(
            if matches!(
                operation.kind,
                WorkspaceOperationKindView::Elementwise("eager_seed")
            ) {
                &self.host
            } else {
                &self.device
            },
        )
    }
    fn scratch_placement(&self, _: WorkspaceOperationView<'_>) -> Option<&MemoryPlacement> {
        Some(&self.host)
    }
    fn allocation_host_control_bytes(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<u64> {
        Some(3)
    }
    fn scratch_host_control_bytes(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Error> {
        Ok(Some(3))
    }
    fn scratch_allocation_count(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<usize>, Error> {
        Ok(Some(1))
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let (output, scratch) = match op.kind {
            WorkspaceOperationKind::Elementwise("eager_seed") => {
                (WorkspaceOutputStorage::Allocate(4), 0)
            }
            WorkspaceOperationKind::Elementwise("seeded_fill") => {
                (WorkspaceOutputStorage::Allocate(64), 4)
            }
            WorkspaceOperationKind::View("seeded_view") => {
                (WorkspaceOutputStorage::AliasInput(0), 0)
            }
            _ => return Ok(None),
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![output],
            scratch_bytes: scratch,
            assumptions: "selected seed and fill source".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "no disjoint host payload".into(),
        }))
    }
}

#[test]
fn host_seed_and_device_fill_keep_independent_placement_and_view_lifetimes() {
    for shared in [false, true] {
        let topology = Rc::new(topology(shared));
        let host = placement(&topology, false);
        let device = placement(&topology, true);
        let context = WorkspaceContext::new(SeededAllocationFacts {
            topology: topology.clone(),
            host: host.clone(),
            device: device.clone(),
        });
        context.begin_span();
        let seed = context
            .execute(
                WorkspaceOperationKind::Elementwise("eager_seed"),
                &[],
                vec![WorkspaceLayout::new(&[], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap()
            .pop()
            .unwrap();
        let filled = context
            .execute(
                WorkspaceOperationKind::Elementwise("seeded_fill"),
                &[],
                vec![WorkspaceLayout::new(&[16], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap()
            .pop()
            .unwrap();
        let view = context
            .execute(
                WorkspaceOperationKind::View("seeded_view"),
                &[&filled],
                vec![WorkspaceLayout::new(&[4], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap()
            .pop()
            .unwrap();
        drop((seed, filled));
        let report = context.report_domains(&[view]).unwrap();
        let h = report
            .domains
            .iter()
            .find(|row| row.domain == host.domains()[0])
            .unwrap();
        assert_eq!(
            h.tensor_buffers.total.accounted_bytes,
            if shared { 72 } else { 8 }
        );
        assert_eq!(h.total.accounted_bytes, if shared { 81 } else { 17 });
        assert_eq!(h.retained.accounted_bytes, if shared { 67 } else { 3 });
        if !shared {
            let d = report
                .domains
                .iter()
                .find(|row| row.domain == device.domains()[0])
                .unwrap();
            assert_eq!(d.total.accounted_bytes, 64);
            assert_eq!(d.retained.accounted_bytes, 64);
        }
    }
}

#[derive(Clone, Debug)]
struct BranchOutputFacts {
    topology: Rc<MemoryTopology>,
    owned: bool,
    preparations: Rc<std::cell::Cell<usize>>,
    scratch: WorkspaceAllocationPopulation,
    output: WorkspaceAllocationPopulation,
}
impl WorkspaceMechanisms for BranchOutputFacts {
    fn prepare_allocation_sources(
        &self,
        op: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceOperationAllocationSources>, Error> {
        if !self.owned || !matches!(op.kind, WorkspaceOperationKindView::DeepCopy) {
            return Ok(None);
        }
        self.preparations.set(self.preparations.get() + 1);
        let mut outputs = context.metadata_vec(1)?;
        outputs.push(Some(self.output.clone()));
        Ok(Some(WorkspaceOperationAllocationSources {
            scratch: Some(self.scratch.clone()),
            outputs,
        }))
    }

    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.topology)
    }
    fn scratch_allocations(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Error> {
        Ok(
            (!self.owned && matches!(op.kind, WorkspaceOperationKindView::DeepCopy))
                .then_some(&self.scratch),
        )
    }
    fn output_allocations(
        &self,
        op: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Error> {
        Ok(
            (!self.owned && matches!(op.kind, WorkspaceOperationKindView::DeepCopy))
                .then_some(&self.output),
        )
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let new = matches!(op.kind, WorkspaceOperationKind::DeepCopy);
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![if new {
                WorkspaceOutputStorage::Allocate(64)
            } else {
                WorkspaceOutputStorage::AliasInput(0)
            }],
            scratch_bytes: if new { 112 } else { 0 },
            assumptions: "actual branch source population".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "no separate host payload".into(),
        }))
    }
}

#[test]
fn branch_scratch_and_escaping_output_keep_source_domains_and_final_alias_lifetimes() {
    for (shared, owned) in [(false, false), (false, true), (true, false), (true, true)] {
        let topology = Rc::new(topology(shared));
        let host = placement(&topology, false);
        let device = placement(&topology, true);
        let collector = WorkspaceContext::new(SelectedFacts {
            topology: topology.clone(),
            native: host.clone(),
            backing_controls: 0,
            scratch_controls: Some(0),
            scratch_births: 0,
            population: None,
        });
        let row = |bytes, births, placement: &MemoryPlacement| WorkspaceScratchAllocation {
            bytes,
            maximum_allocations: Some(births),
            host_control_bytes: Some(births as u64),
            placement: Some(placement.clone()),
        };
        // One completed branch returns Device indices; the other returns Host
        // indices. The eager Host seed and all remaining child backings coexist.
        let gpu_return = collector
            .source_scratch_population(&[row(80, 2, &host), row(32, 1, &device)])
            .unwrap();
        let cpu_return = collector
            .source_scratch_population(&[row(16, 1, &host), row(96, 2, &device)])
            .unwrap();
        let scratch = collector
            .peak_scratch_populations(&[&gpu_return, &cpu_return])
            .unwrap();
        assert_eq!(scratch.backing_bytes(), Some(112));
        let output_placement = if shared {
            host.clone()
        } else {
            MemoryPlacement::possible(
                &topology,
                vec![host.domains()[0], device.domains()[0]],
                "actual returned CPU or Device backing".into(),
            )
            .unwrap()
        };
        let output = collector
            .source_scratch_population(&[WorkspaceScratchAllocation {
                bytes: 64,
                maximum_allocations: Some(1),
                host_control_bytes: Some(7),
                placement: Some(output_placement),
            }])
            .unwrap();
        let preparations = Rc::new(std::cell::Cell::new(0));
        let context = WorkspaceContext::new(BranchOutputFacts {
            topology: topology.clone(),
            owned,
            preparations: preparations.clone(),
            scratch,
            output,
        });
        let result = context
            .execute(
                WorkspaceOperationKind::DeepCopy,
                &[],
                vec![WorkspaceLayout::new(&[16], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap()
            .pop()
            .unwrap();
        let alias = context
            .execute(
                WorkspaceOperationKind::View("partial view"),
                &[&result],
                vec![WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap()
            .pop()
            .unwrap();
        drop(result);
        assert_eq!(preparations.get(), usize::from(owned));
        let report = context.report_domains(&[alias.clone()]).unwrap();
        let h = report
            .domains
            .iter()
            .find(|row| row.domain == host.domains()[0])
            .unwrap();
        assert_eq!(
            h.tensor_buffers.total.accounted_bytes,
            if shared { 176 } else { 80 }
        );
        assert_eq!(
            h.tensor_buffers.total.placement_allowance_bytes,
            if shared { 0 } else { 64 }
        );
        assert_eq!(h.retained.accounted_bytes, if shared { 71 } else { 7 });
        assert_eq!(
            h.retained.placement_allowance_bytes,
            if shared { 0 } else { 64 }
        );
        if !shared {
            let d = report
                .domains
                .iter()
                .find(|row| row.domain == device.domains()[0])
                .unwrap();
            assert_eq!(
                (
                    d.tensor_buffers.total.accounted_bytes,
                    d.tensor_buffers.total.placement_allowance_bytes
                ),
                (96, 64)
            );
            assert_eq!(
                (
                    d.retained.accounted_bytes,
                    d.retained.placement_allowance_bytes
                ),
                (0, 64)
            );
            assert!(report.native_allocations.placement_allowances().iter().all(|allowance| matches!(allowance.placement.kind(),eredu_core::MemoryPlacementKind::Possible {basis,..} if basis=="actual returned CPU or Device backing")));
        }
        drop(alias);
        for row in context.report_domains(&[]).unwrap().domains {
            assert_eq!(
                (
                    row.retained.accounted_bytes,
                    row.retained.placement_allowance_bytes
                ),
                (0, 0)
            );
        }
    }
}

#[test]
fn alternative_scratch_preserves_unknown_controls_without_discarding_known_backing() {
    for (shared, owned) in [(false, false), (false, true), (true, false), (true, true)] {
        let topology = Rc::new(topology(shared));
        let host = placement(&topology, false);
        let device = placement(&topology, true);
        let collector = WorkspaceContext::new(SelectedFacts {
            topology: topology.clone(),
            native: host.clone(),
            backing_controls: 0,
            scratch_controls: Some(0),
            scratch_births: 0,
            population: None,
        });
        let known = collector
            .source_scratch_population(&[WorkspaceScratchAllocation {
                bytes: 112,
                maximum_allocations: Some(2),
                host_control_bytes: Some(6),
                placement: Some(host.clone()),
            }])
            .unwrap();
        let unknown = collector
            .source_scratch_population(&[
                WorkspaceScratchAllocation {
                    bytes: 16,
                    maximum_allocations: Some(1),
                    host_control_bytes: Some(3),
                    placement: Some(host.clone()),
                },
                WorkspaceScratchAllocation {
                    bytes: 96,
                    maximum_allocations: None,
                    host_control_bytes: None,
                    placement: Some(device.clone()),
                },
            ])
            .unwrap();
        let peak = collector
            .peak_scratch_populations(&[&known, &unknown])
            .unwrap();
        assert_eq!(peak.backing_bytes(), Some(112));
        assert_eq!(peak.host_control_bytes(), None);
        let repeated = collector
            .combine_scratch_populations(&[(&peak, 2)])
            .unwrap();
        assert_eq!(repeated.backing_bytes(), Some(224));
        assert_eq!(repeated.host_control_bytes(), None);
        let output = collector
            .source_scratch_population(&[WorkspaceScratchAllocation {
                bytes: 64,
                maximum_allocations: Some(1),
                host_control_bytes: Some(7),
                placement: Some(device),
            }])
            .unwrap();
        let context = WorkspaceContext::new(BranchOutputFacts {
            topology,
            owned,
            preparations: Rc::new(std::cell::Cell::new(0)),
            scratch: peak,
            output,
        });
        let result = context
            .execute(
                WorkspaceOperationKind::DeepCopy,
                &[],
                vec![WorkspaceLayout::new(&[16], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap();
        assert!(context.complete_domain_report(&result).unwrap().is_none());
        let report = context.report(&result).unwrap();
        assert_eq!(report.tensor_buffers.total_bytes, Some(176));
        assert_eq!(report.total_bytes, None);
        assert_eq!(report.unpriced_host_operations, vec![0]);
    }
}
