use super::*;

#[derive(Debug)]
struct Facts {
    host: bool,
    tensor: bool,
}
impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::working_memory::memory_fixture::host_topology_ref())
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }
    fn scratch_placement(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }

    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(if !self.tensor {
            None
        } else {
            Some(WorkspaceOperationBound {
                outputs: operation
                    .outputs
                    .iter()
                    .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                    .collect::<Result<_, _>>()?,
                scratch_bytes: 11,
                assumptions: "test eager initializer allocates its complete output".into(),
            })
        })
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(self.host.then(|| WorkspaceHostBound {
            bytes: 13,
            assumptions: "test operation staging".into(),
        }))
    }
}
fn identity_controls() -> Option<u64> {
    let controls = text_identity_control_bytes();
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_PROMPT_INPUT") == Some("1".into()) {
        assert!(
            controls.is_some(),
            "pinned prompt identity owner must qualify"
        );
    }
    controls
}
fn geometry(chunk: u64) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 5,
        input_positions: 11,
        max_output_tokens: 9,
        prefill_chunk_positions: chunk,
        output: eredu_core::OutputDemand::LastPosition,
    }
}
fn outside(g: InferenceGeometry) -> ExecutionWorkspaceEstimate {
    let zero = || WorkspaceBound::bounded(0, "test component priced elsewhere");
    ExecutionWorkspaceEstimate {
        geometry: g,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
        physical_domains: Some({
            let topology = crate::working_memory::memory_fixture::host_topology_ref();
            let zero = || eredu_core::DomainMemoryRequirements::zero(topology);
            eredu_core::DomainExecutionWorkspaceEstimate {
                geometry: g,
                activations: zero(),
                attention: zero(),
                vocabulary: zero(),
                state_update: zero(),
                materialization: zero(),
                retained: zero(),
            }
        }),
    }
}

#[test]
fn preparation_prices_full_input_and_real_host_capacity_independently_of_chunk_size() {
    let Some(identity_controls) = identity_controls() else {
        let context = WorkspaceContext::new(Facts {
            host: true,
            tensor: true,
        });
        assert_eq!(
            quote_text_prompt_workspace(geometry(4), Some(128), &context)
                .unwrap()
                .peak()
                .bytes(),
            None
        );
        return;
    };
    let context = WorkspaceContext::new(Facts {
        host: true,
        tensor: true,
    });
    for chunk in [1, 4, 11] {
        let g = geometry(chunk);
        let identity_peak =
            crate::input::TextInputIdentityPlan::new(g.batch_size, g.input_positions)
                .unwrap()
                .peak_bytes();
        let report = quote_text_prompt_workspace(g, Some(128), &context).unwrap();
        assert_eq!(report.tensor_peak_bytes(), Some(11 * 4 + 11));
        assert_eq!(
            report.host_peak_bytes(),
            Some(128 + 13 + identity_peak + identity_controls)
        );
        assert_eq!(
            report.peak().bytes(),
            Some(196 + identity_peak + identity_controls)
        );
        let combined = report.compose(outside(g)).unwrap();
        assert_eq!(
            combined.peak_bytes().unwrap(),
            Some(196 + identity_peak + identity_controls)
        );
        assert_eq!(
            context.report(&[]).unwrap().operations[0].outputs[0].shape(),
            [1, 11]
        );
        assert!(matches!(
            context.report(&[]).unwrap().operations[0].kind,
            WorkspaceOperationKind::Elementwise("text_prompt_u32")
        ));
    }
}

#[test]
fn prompt_retains_backing_and_scratch_controls_in_their_host_domain() {
    use eredu_core::{
        MemoryDeviceId, MemoryDomainDescription, MemoryLocation, MemoryPlacement, MemoryTopology,
    };
    #[derive(Debug)]
    struct Placed {
        topology: MemoryTopology,
        placement: MemoryPlacement,
        controls_known: bool,
    }
    impl WorkspaceMechanisms for Placed {
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
            let original = matches!(
                operation.kind,
                WorkspaceOperationKind::Elementwise("prepared_token_input")
            );
            Ok(Some(WorkspaceOperationBound {
                outputs: vec![WorkspaceOutputStorage::Allocate(44)],
                scratch_bytes: if original { 0 } else { 11 },
                assumptions: "selected upload, with original input storage separately admitted"
                    .into(),
            }))
        }
        fn host_workspace_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: if matches!(
                    operation.kind,
                    WorkspaceOperationKind::Elementwise("prepared_token_input")
                ) {
                    0
                } else {
                    13
                },
                assumptions: "selected upload staging".into(),
            }))
        }
        fn allocation_host_control_bytes(
            &self,
            operation: WorkspaceOperationView<'_>,
            _: usize,
        ) -> Option<u64> {
            if matches!(
                operation.kind,
                WorkspaceOperationKindView::Elementwise("prepared_token_input")
            ) {
                Some(0)
            } else {
                self.controls_known.then_some(17)
            }
        }
        fn scratch_host_control_bytes(
            &self,
            operation: WorkspaceOperationView<'_>,
        ) -> Result<Option<u64>, Error> {
            Ok(Some(
                if matches!(
                    operation.kind,
                    WorkspaceOperationKindView::Elementwise("prepared_token_input")
                ) {
                    0
                } else {
                    19
                },
            ))
        }
    }
    let Some(identity_controls) = identity_controls() else {
        return;
    };
    let geometry = geometry(4);
    let identity = crate::input::TextInputIdentityPlan::new(1, 11)
        .unwrap()
        .peak_bytes()
        + identity_controls;
    let input = super::super::OriginalTokenInputLayout::prepare(
        &eredu_core::TokenIdsInputPlan::new(&[7; 11]).unwrap(),
    )
    .unwrap();
    for controls_known in [false, true] {
        let device = MemoryLocation::Device(MemoryDeviceId {
            backend: "fixture",
            ordinal: 0,
        });
        let topology = MemoryTopology::new(vec![
            MemoryDomainDescription {
                name: "host".into(),
                locations: vec![MemoryLocation::Host],
            },
            MemoryDomainDescription {
                name: "device".into(),
                locations: vec![device],
            },
        ])
        .unwrap();
        let host = topology.host_domain();
        let device = topology.domain_for(device).unwrap();
        let placement = MemoryPlacement::fixed(&topology, device).unwrap();
        let context = WorkspaceContext::new(Placed {
            topology,
            placement,
            controls_known,
        });
        let report = quote_text_prompt_workspace(geometry, Some(128), &context).unwrap();
        if controls_known {
            let domains = report.physical_domains().unwrap();
            assert_eq!(domains.get(device).unwrap().total().unwrap(), 44 + 11);
            assert_eq!(
                domains.get(host).unwrap().total().unwrap(),
                128 + identity + 13 + 17 + 19
            );
            assert_eq!(
                report.host_peak_bytes(),
                Some(128 + identity + 13 + 17 + 19)
            );
            assert_eq!(
                report.peak().bytes(),
                Some(128 + identity + 13 + 17 + 19 + 44 + 11)
            );
        } else {
            assert!(report.physical_domains().is_none());
            assert!(matches!(report.peak(), WorkspaceBound::Unknown { .. }));
        }
        let original = quote_original_token_prompt_workspace(geometry, &input, &context).unwrap();
        let domains = original.physical_domains().unwrap();
        assert_eq!(domains.get(device).unwrap().total().unwrap(), 44);
        assert_eq!(domains.get(host).unwrap().total().unwrap(), identity);
        assert_eq!(original.host_peak_bytes(), Some(identity));
        assert_eq!(original.peak().bytes(), Some(identity + 44));
    }
}

#[test]
fn preparation_preserves_missing_backing_mechanisms_and_enclosing_bounds() {
    for (host, tensor, capacity) in [
        (false, true, Some(44)),
        (true, false, Some(44)),
        (true, true, None),
    ] {
        let context = WorkspaceContext::new(Facts { host, tensor });
        let report = quote_text_prompt_workspace(geometry(4), capacity, &context).unwrap();
        assert_eq!(report.peak().bytes(), None);
        assert_eq!(
            report
                .compose(outside(geometry(4)))
                .unwrap()
                .peak_bytes()
                .unwrap(),
            None
        );
    }
    let context = WorkspaceContext::new(Facts {
        host: true,
        tensor: true,
    });
    let report = quote_text_prompt_workspace(geometry(4), Some(44), &context).unwrap();
    let mut missing = outside(geometry(4));
    missing.materialization = WorkspaceBound::Unknown {
        reason: "unpriced weight staging".into(),
    };
    assert_eq!(report.compose(missing).unwrap().peak_bytes().unwrap(), None);
}

#[test]
fn preparation_rejects_invalid_capacity_extent_and_request_composition() {
    let context = WorkspaceContext::new(Facts {
        host: true,
        tensor: true,
    });
    assert!(quote_text_prompt_workspace(geometry(4), Some(43), &context).is_err());
    assert!(
        quote_text_prompt_workspace(
            InferenceGeometry {
                input_positions: i32::MAX as u64 + 1,
                ..geometry(4)
            },
            None,
            &context
        )
        .is_err()
    );
    assert!(context.report(&[]).unwrap().operations.is_empty());
    let report = quote_text_prompt_workspace(geometry(4), Some(44), &context).unwrap();
    assert!(report.compose(outside(geometry(3))).is_err());
    assert!(
        report
            .compose(outside(InferenceGeometry {
                cached_positions: 6,
                ..geometry(4)
            }))
            .is_err()
    );
}

#[test]
fn complete_prompt_capacity_remains_reserved_when_chunk_search_and_concurrent_requests_compete() {
    let Some(identity_controls) = identity_controls() else {
        let context = WorkspaceContext::new(Facts {
            host: true,
            tensor: true,
        });
        assert_eq!(
            quote_text_prompt_workspace(geometry(4), Some(128), &context)
                .unwrap()
                .peak()
                .bytes(),
            None
        );
        return;
    };
    use crate::working_memory::{
        InferenceExecutionIdentity, MemoryLedger, PrefillPlanningError, WorkingMemoryError,
        plan_prefill,
    };
    use eredu_core::{
        AdmissionRequest, CacheStateStrategy, EstimationCompleteness, InputModalities,
        InputTokenCount, LayerSchedule, ModelCapabilities, Observed, StateMemoryLayout,
    };
    let g = geometry(4);
    let capabilities = ModelCapabilities {
        effective_model_type: "stateless preparation fixture".into(),
        native_max_context: Observed::exact(64, "fixture"),
        effective_max_context: Observed::exact(64, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    let request = AdmissionRequest {
        input: InputTokenCount::text(g.cached_positions + g.input_positions),
        max_output_tokens: g.max_output_tokens,
        batch_size: g.batch_size,
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let quote = |g, capacity| {
        let mut state = eredu_core::estimate_runtime_state(
            &layout,
            request.input,
            request.max_output_tokens,
            1,
            std::num::NonZeroU8::new(4).unwrap(),
        )
        .unwrap();
        let topology = crate::working_memory::memory_fixture::host_topology_ref();
        let zero = || eredu_core::DomainMemoryRequirements::zero(topology);
        state.physical_domains = Some(eredu_core::DomainRuntimeStateEstimate {
            geometry: g,
            decoder_state: zero(),
            media_embeddings: zero(),
            media_workspace: zero(),
        });
        let prompt = quote_text_prompt_workspace(
            g,
            capacity,
            &WorkspaceContext::new(Facts {
                host: true,
                tensor: true,
            }),
        )
        .unwrap();
        state.with_execution_workspace(prompt.compose(outside(g))?)
    };
    let execution = InferenceExecutionIdentity::default();
    let identity_peak = crate::input::TextInputIdentityPlan::new(g.batch_size, g.input_positions)
        .unwrap()
        .peak_bytes();
    let probe = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let eredu_core::AdmissionResult::Admitted(admission) = eredu_core::apply_admission_policy(
        &capabilities,
        request.clone(),
        quote(g, Some(128)).unwrap(),
    )
    .unwrap() else {
        panic!("complete fixture")
    };
    let first_bytes = crate::working_memory::memory_fixture::reservation_bytes(&probe, &admission);
    let pool = crate::working_memory::memory_fixture::host_ledger(first_bytes, 0).unwrap();
    let (_, first) = plan_prefill(&execution, &pool, &capabilities, request.clone(), g, |g| {
        quote(g, Some(128))
    })
    .unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        196 + identity_peak + identity_controls
    );
    let mut attempted_chunks = Vec::new();
    let second = plan_prefill(&execution, &pool, &capabilities, request.clone(), g, |g| {
        attempted_chunks.push(g.prefill_chunk_positions);
        quote(g, Some(44))
    });
    assert!(matches!(
        second,
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
        ))
    ));
    assert_eq!(attempted_chunks, [4, 3, 2, 1]);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        196 + identity_peak + identity_controls
    );
    drop(first);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let (_, second) = plan_prefill(&execution, &pool, &capabilities, request.clone(), g, |g| {
        quote(g, Some(44))
    })
    .unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        112 + identity_peak + identity_controls
    );
    assert!(
        plan_prefill(&execution, &pool, &capabilities, request.clone(), g, |g| {
            quote(g, None)
        })
        .is_err()
    );
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        112 + identity_peak + identity_controls
    );
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(pool.payload_peak_bytes().unwrap(), first_bytes);
}

#[test]
fn original_token_input_report_preserves_native_identity_and_staging_without_legacy_capacity() {
    if identity_controls().is_none() {
        let context = WorkspaceContext::new(Facts {
            host: true,
            tensor: true,
        });
        assert_eq!(
            quote_text_prompt_workspace(geometry(4), Some(128), &context)
                .unwrap()
                .host_peak_bytes(),
            None
        );
        return;
    }
    let ids = [7; 11];
    let input = crate::working_memory::OriginalTokenInputLayout::prepare(
        &eredu_core::TokenIdsInputPlan::new(&ids).unwrap(),
    )
    .unwrap();
    for chunk in [1, 4, 11] {
        let context = WorkspaceContext::new(Facts {
            host: true,
            tensor: true,
        });
        let original =
            quote_original_token_prompt_workspace(geometry(chunk), &input, &context).unwrap();
        let legacy = quote_text_prompt_workspace(geometry(chunk), Some(128), &context).unwrap();
        assert_eq!(original.tensor_peak_bytes(), legacy.tensor_peak_bytes());
        assert_eq!(
            original.host_peak_bytes().unwrap() + 128,
            legacy.host_peak_bytes().unwrap()
        );
        assert_eq!(
            original.peak().bytes().unwrap() + 128,
            legacy.peak().bytes().unwrap()
        );
        assert_eq!(
            context.report(&[]).unwrap().operations[0].outputs[0].shape(),
            [1, 11]
        );
    }
    for (host, tensor) in [(false, true), (true, false)] {
        let context = WorkspaceContext::new(Facts { host, tensor });
        assert_eq!(
            quote_original_token_prompt_workspace(geometry(4), &input, &context)
                .unwrap()
                .peak()
                .bytes(),
            None
        );
    }
    let context = WorkspaceContext::new(Facts {
        host: true,
        tensor: true,
    });
    assert!(
        quote_original_token_prompt_workspace(
            InferenceGeometry {
                input_positions: 12,
                ..geometry(4)
            },
            &input,
            &context
        )
        .is_err()
    );
}
