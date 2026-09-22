//! Physical placement travels through the ordinary sampling and workspace walk.
use eredu_core::*;
use eredu_nn::{workspace::*, Error};
use eredu_runtime::{working_memory::*, ConfiguredTextSampler, GenerationSampler};
use std::sync::Arc;

fn device(ordinal: u32) -> MemoryLocation {
    MemoryLocation::Device(MemoryDeviceId {
        backend: "fixture",
        ordinal,
    })
}
fn topology(unified: bool) -> Arc<MemoryTopology> {
    Arc::new(
        MemoryTopology::new(if unified {
            vec![MemoryDomainDescription {
                name: "shared".into(),
                locations: vec![MemoryLocation::Host, device(0), device(1)],
            }]
        } else {
            vec![
                MemoryDomainDescription {
                    name: "host".into(),
                    locations: vec![MemoryLocation::Host],
                },
                MemoryDomainDescription {
                    name: "gpu0".into(),
                    locations: vec![device(0)],
                },
                MemoryDomainDescription {
                    name: "gpu1".into(),
                    locations: vec![device(1)],
                },
            ]
        })
        .unwrap(),
    )
}
#[derive(Debug)]
struct Facts {
    topology: Arc<MemoryTopology>,
    placement: MemoryPlacement,
}
impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.topology)
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&MemoryPlacement> {
        Some(&self.placement)
    }
    fn scratch_placement(&self, _: WorkspaceOperationView<'_>) -> Option<&MemoryPlacement> {
        Some(&self.placement)
    }
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|output| {
                    if !operation.inputs.is_empty() {
                        Ok(WorkspaceOutputStorage::AliasInput(0))
                    } else {
                        Ok(WorkspaceOutputStorage::Allocate(output.bytes()?))
                    }
                })
                .collect::<Result<_, Error>>()?,
            scratch_bytes: 0,
            assumptions: "selected fixture mechanism".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "no operation host scratch".into(),
        }))
    }
}
fn quote(
    topology: Arc<MemoryTopology>,
    source: &DomainMemoryRequirements,
    steps: u64,
) -> SamplingWorkspaceReport {
    try_quote(topology, source, steps, None).unwrap()
}

fn try_quote(
    topology: Arc<MemoryTopology>,
    source: &DomainMemoryRequirements,
    steps: u64,
    observer: Option<&mut dyn SamplingWorkspaceObserver>,
) -> Result<SamplingWorkspaceReport, Error> {
    let placement =
        MemoryPlacement::fixed(&topology, topology.domain_for(device(0)).unwrap()).unwrap();
    let context = WorkspaceContext::new(Facts {
        topology,
        placement,
    });
    let layout = WorkspaceLayout::new(&[1, 1, 4], WorkspaceDtype::Float32).unwrap();
    let source = WorkspaceSamplingInput {
        layout: &layout,
        backing_capacity_bytes: None,
    }
    .with_backing_population(2)
    .with_physical_domains(Some(source));
    quote_sampling_workspace_with_observer(
        &ConfiguredTextSampler::Standard(GenerationSampler::new()),
        0.0,
        None,
        source,
        &TokenFilter::All,
        steps,
        &context,
        observer,
    )
}

#[test]
fn mixed_score_aliases_keep_each_independent_prior_backing() {
    for unified in [false, true] {
        let topology = topology(unified);
        let mut source = DomainMemoryRequirements::zero(&topology);
        for (location, bytes) in [(MemoryLocation::Host, 64), (device(0), 80)] {
            source
                .add_allocation(
                    bytes,
                    &MemoryPlacement::fixed(&topology, topology.domain_for(location).unwrap())
                        .unwrap(),
                )
                .unwrap();
        }
        let report = quote(topology.clone(), &source, 3);
        let physical = report.physical_domains.as_ref().unwrap();
        let host = report.host_peak_bytes.unwrap();
        assert_eq!(
            physical
                .get(topology.host_domain())
                .unwrap()
                .accounted_bytes,
            host + if unified { 288 } else { 128 }
        );
        assert_eq!(
            physical
                .get(topology.domain_for(device(0)).unwrap())
                .unwrap()
                .accounted_bytes,
            if unified { host + 288 } else { 160 }
        );
        assert_eq!(report.first_gap, None);
    }
}

#[test]
fn managed_score_allowances_survive_retained_aliases_with_their_basis() {
    let topology = topology(false);
    let managed = MemoryPlacement::possible_locations(
        &topology,
        [MemoryLocation::Host, device(0)],
        "fixture allocator can occupy host or gpu0".into(),
    )
    .unwrap();
    let mut source = DomainMemoryRequirements::zero(&topology);
    source.add_allocation(32, &managed).unwrap();
    let report = quote(topology.clone(), &source, 2);
    let physical = report.physical_domains.as_ref().unwrap();
    for location in [MemoryLocation::Host, device(0)] {
        assert_eq!(
            physical
                .get(topology.domain_for(location).unwrap())
                .unwrap()
                .placement_allowance_bytes,
            32
        );
    }
    assert!(physical
        .placement_allowances()
        .iter()
        .all(|allowance| matches!(allowance.placement.kind(),
        MemoryPlacementKind::Possible { basis, .. } if basis.contains("fixture allocator"))));
}

#[test]
fn separate_maximum_score_domains_do_not_require_an_aggregate_value() {
    #[derive(Default)]
    struct Closing(Vec<WorkspaceStoragePopulation>);
    impl SamplingWorkspaceObserver for Closing {
        fn observe(
            &mut self,
            _: SamplingWorkspacePhase,
            _: &WorkspaceTraceReport,
        ) -> Result<(), Error> {
            unreachable!("closing population callback is selected")
        }
        fn observe_with_storage(
            &mut self,
            _: SamplingWorkspacePhase,
            _: &WorkspaceTraceReport,
            closing: WorkspaceStoragePopulation,
        ) -> Result<(), Error> {
            self.0.push(closing);
            Ok(())
        }
    }
    let topology = topology(false);
    let mut source = DomainMemoryRequirements::zero(&topology);
    for location in [device(0), device(1)] {
        source
            .add_allocation(
                u64::MAX,
                &MemoryPlacement::fixed(&topology, topology.domain_for(location).unwrap()).unwrap(),
            )
            .unwrap();
    }
    for observed in [false, true] {
        let mut closing = Closing::default();
        let report = try_quote(
            topology.clone(),
            &source,
            1,
            observed.then_some(&mut closing as &mut dyn SamplingWorkspaceObserver),
        )
        .unwrap();
        let physical = report.physical_domains.as_ref().unwrap();
        // The enclosing equation owns current scores, so the sampler adds no
        // device charge. Closing roots still retain their full backing union.
        for location in [device(0), device(1)] {
            assert_eq!(
                physical
                    .get(topology.domain_for(location).unwrap())
                    .unwrap()
                    .accounted_bytes,
                0
            );
        }
        assert_eq!(report.first_gap, None);
        assert_eq!(report.maximum_closing_storage_allocations, Some(4));
        if observed {
            assert_eq!(closing.0.len(), 2);
            assert_eq!(closing.0[1].maximum_allocations, 4);
            assert_eq!(closing.0[1].bytes, None);
        }

        // A second step retains the first score alias alongside a fresh score
        // backing, which overflows each device domain independently.
        let error = try_quote(
            topology.clone(),
            &source,
            2,
            observed.then_some(&mut closing as &mut dyn SamplingWorkspaceObserver),
        )
        .unwrap_err();
        assert_eq!(
            std::error::Error::source(&error)
                .and_then(|cause| cause.downcast_ref::<WorkspaceMetadataError>()),
            Some(&WorkspaceMetadataError::Report(
                eredu_nn::workspace::WorkspaceReportError::Domain(MemoryDomainError::Overflow)
            ))
        );
    }
}

#[derive(Debug)]
struct SeparateMaximumFacts {
    topology: Arc<MemoryTopology>,
    placements: [MemoryPlacement; 2],
}
impl WorkspaceMechanisms for SeparateMaximumFacts {
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.topology)
    }
    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        index: usize,
    ) -> Option<&MemoryPlacement> {
        Some(&self.placements[usize::from(operation.outputs.get(index)?.shape()[0] == 2)])
    }
    fn scratch_placement(&self, _: WorkspaceOperationView<'_>) -> Option<&MemoryPlacement> {
        Some(&self.placements[0])
    }
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|_| WorkspaceOutputStorage::Allocate(u64::MAX))
                .collect(),
            scratch_bytes: 0,
            assumptions: "two independent selected maximum-capacity device allocations".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "no disjoint host scratch".into(),
        }))
    }
}
#[test]
fn sealing_actual_spans_preserves_independent_maximum_domain_charges() {
    use eredu_nn::Tensor;
    let topology = topology(false);
    let placements = [device(0), device(1)].map(|location| {
        MemoryPlacement::fixed(&topology, topology.domain_for(location).unwrap()).unwrap()
    });
    let context = WorkspaceContext::new(SeparateMaximumFacts {
        topology: topology.clone(),
        placements,
    });
    let ledger = MemoryLedger::new(
        topology.clone(),
        MemoryLimits::unlimited(&topology),
        DomainMemoryRequirements::zero(&topology),
    )
    .unwrap();
    let storage = RegisteredWorkspaceStorage::<u32>::bind(
        &ledger,
        &context,
        std::iter::empty::<RegisteredWorkspaceStorageRow<u32>>(),
    )
    .unwrap();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let report = quote_inference_workspace_with_context(geometry, &context, |_| {
        context.begin_state_span([])?;
        let first = WorkspaceTensor::full_f32(1.0, &[1], &context)?;
        let second = WorkspaceTensor::full_f32(2.0, &[2], &context)?;
        let report = context.finish_report(&[])?;
        drop((first, second));
        Ok::<_, Error>(report)
    })
    .unwrap();
    let zero = || DomainMemoryRequirements::zero(&topology);
    let bound = || WorkspaceBound::bounded(0, "explicit absence in this selected fixture");
    let outside = ExecutionWorkspaceEstimate {
        geometry,
        activations: bound(),
        attention: bound(),
        vocabulary: bound(),
        state_update: bound(),
        materialization: bound(),
        retained: bound(),
        physical_domains: Some(DomainExecutionWorkspaceEstimate {
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
        }),
    };
    let state = estimate_runtime_state(
        &StateMemoryLayout::new(
            LayerSchedule::empty(),
            vec![],
            1,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap(),
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let quote = ResidualInferenceQuote::compose(&report, state, outside, &storage)
        .unwrap()
        .into_incremental()
        .with_span_workspace()
        .unwrap();
    assert_eq!(quote.incremental_bytes(), None);
    for location in [device(0), device(1)] {
        let domain = topology.domain_for(location).unwrap();
        assert_eq!(
            quote.span_workspace().span_domain_bytes(0, domain).unwrap(),
            Some(u64::MAX)
        );
        assert_eq!(
            quote
                .incremental_requirements()
                .unwrap()
                .get(domain)
                .unwrap()
                .total()
                .unwrap(),
            u64::MAX
        );
    }
    assert!(
        quote
            .incremental_requirements()
            .unwrap()
            .get(topology.host_domain())
            .unwrap()
            .total()
            .unwrap()
            > 0
    );
}

#[derive(Debug)]
struct SeparateCopyFacts {
    topology: Arc<MemoryTopology>,
    placements: [MemoryPlacement; 2],
}
impl WorkspaceMechanisms for SeparateCopyFacts {
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.topology)
    }
    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        index: usize,
    ) -> Option<&MemoryPlacement> {
        Some(&self.placements[usize::from(operation.outputs.get(index)?.shape()[0] == 2)])
    }
    fn scratch_placement(&self, _: WorkspaceOperationView<'_>) -> Option<&MemoryPlacement> {
        Some(&self.placements[0])
    }
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let output = match operation.kind {
            WorkspaceOperationKind::Contiguous => WorkspaceOutputStorage::AliasInput(0),
            WorkspaceOperationKind::DeepCopy => WorkspaceOutputStorage::Allocate(u64::MAX / 4),
            _ => unreachable!("closed isolated copy worker"),
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![output],
            scratch_bytes: 0,
            assumptions: "independent device backing selected for each copy".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "no staging allocation".into(),
        }))
    }
}

#[test]
fn terminal_copy_composes_physical_domains_when_aggregate_diagnostic_overflows() {
    for retained in [false, true] {
        terminal_copy_with_separate_retained_domains(retained);
    }
}

fn terminal_copy_with_separate_retained_domains(retained: bool) {
    let topology = topology(false);
    let ledger = MemoryLedger::new(
        topology.clone(),
        MemoryLimits::unlimited(&topology),
        DomainMemoryRequirements::zero(&topology),
    )
    .unwrap();
    let host = ledger.host_placement_handle();
    let storage = StoragePublicationLayout::<u32>::new(2)
        .unwrap()
        .fund(&ledger)
        .unwrap()
        .register_storage([
            (1, StorageAllocation::new(4, host.clone())),
            (2, StorageAllocation::new(8, host.clone())),
        ])
        .unwrap();
    let placements = [device(0), device(1)].map(|location| {
        MemoryPlacement::fixed(&topology, topology.domain_for(location).unwrap()).unwrap()
    });
    let context = WorkspaceContext::new(SeparateCopyFacts {
        topology: topology.clone(),
        placements,
    });
    let roots = [4, 8].map(|bytes| {
        WorkspaceExistingStorage::try_new_placed(Some(bytes), &host, &context).unwrap()
    });
    let binding = RegisteredWorkspaceStorage::bind(
        &ledger,
        &context,
        [(1u32, roots[0].clone()), (2u32, roots[1].clone())],
    )
    .unwrap();
    let tensors = [1, 2]
        .into_iter()
        .zip(&roots)
        .map(|(width, root)| {
            WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[width], WorkspaceDtype::Float32).unwrap(),
                root,
                &context,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, binding.borrowed_storage(), &tensors).unwrap();
    let copy = RegisteredWorkspaceCopy::bind(plan, binding).unwrap();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 1,
        input_positions: 0,
        max_output_tokens: 0,
        prefill_chunk_positions: 0,
        output: OutputDemand::StateOnly,
    };
    let equations =
        quote_inference_workspace(geometry, |_| -> Result<WorkspaceTraceReport, Error> {
            panic!("terminal copy has no inference equations")
        })
        .unwrap();
    let zero = || DomainMemoryRequirements::zero(&topology);
    let mut state = estimate_runtime_state(
        &StateMemoryLayout::new(
            LayerSchedule::empty(),
            vec![],
            1,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap(),
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    state.physical_domains = Some(DomainRuntimeStateEstimate {
        geometry,
        decoder_state: zero(),
        media_embeddings: zero(),
        media_workspace: zero(),
    });
    let mut host_overlap = zero();
    host_overlap.add_allocation(u64::MAX / 2, &host).unwrap();
    let retained_per_device = if retained { u64::MAX / 2 + 1 } else { 0 };
    let mut retained_domains = zero();
    for location in [device(0), device(1)] {
        retained_domains
            .add_allocation(
                retained_per_device,
                &MemoryPlacement::fixed(&topology, topology.domain_for(location).unwrap()).unwrap(),
            )
            .unwrap();
    }
    let bound = || WorkspaceBound::bounded(0, "no additional allocation");
    let outside = ExecutionWorkspaceEstimate {
        geometry,
        activations: bound(),
        attention: bound(),
        vocabulary: bound(),
        state_update: WorkspaceBound::bounded(
            u64::MAX / 2,
            "independent overlapping host allocation",
        ),
        materialization: bound(),
        retained: if retained {
            WorkspaceBound::PerDomain {
                assumptions: "independent retained device buffers exceed one aggregate diagnostic"
                    .into(),
            }
        } else {
            bound()
        },
        physical_domains: Some(DomainExecutionWorkspaceEstimate {
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: host_overlap,
            materialization: zero(),
            retained: retained_domains,
        }),
    };
    let quote = copy
        .compose_inference(&equations, state, outside)
        .unwrap()
        .with_span_workspace()
        .unwrap();
    assert_eq!(quote.incremental_bytes(), None);
    assert!(matches!(
        quote
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .state_update,
        WorkspaceBound::PerDomain { .. }
    ));
    let requirements = quote.incremental_requirements().unwrap();
    for location in [device(0), device(1)] {
        assert_eq!(
            requirements
                .get(topology.domain_for(location).unwrap())
                .unwrap()
                .total()
                .unwrap(),
            u64::MAX / 4 + retained_per_device
        );
    }
    assert!(
        requirements
            .get(topology.host_domain())
            .unwrap()
            .total()
            .unwrap()
            > u64::MAX / 2
    );
    drop((quote, storage));
}
