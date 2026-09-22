use super::*;
use crate::working_memory::{
    InferenceExecutionIdentity, MemoryLedger, PrefillPlanningError, WorkingMemoryError,
    plan_prefill,
};
use eredu_core::{
    AdmissionRequest, AdmissionResult, CacheStateStrategy, EstimationCompleteness, InputModalities,
    InputTokenCount, LayerSchedule, ModelCapabilities, Observed, StateMemoryLayout,
};
use eredu_nn::{Error, Tensor, workspace::*};
use std::num::NonZeroU8;

#[derive(Debug)]
struct Facts {
    tensor: u64,
    host: Option<u64>,
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
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![WorkspaceOutputStorage::Allocate(4)],
            scratch_bytes: self.tensor - 4,
            assumptions: "one scalar output plus exact fixture scratch".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(self.host.map(|bytes| WorkspaceHostBound {
            bytes,
            assumptions: "disjoint fixture host payload".into(),
        }))
    }
}
fn trace(tensor: u64, host: Option<u64>, retain: bool) -> WorkspaceTraceReport {
    let context = WorkspaceContext::new(Facts { tensor, host });
    context.begin_state_span([]).unwrap();
    let input = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap(),
        &context,
    )
    .unwrap();
    let output = input.square(&context).unwrap();
    context
        .report(if retain {
            std::slice::from_ref(&output)
        } else {
            &[]
        })
        .unwrap()
}
fn fixture_requirements(bytes: u64) -> eredu_core::DomainMemoryRequirements {
    let topology = crate::working_memory::memory_fixture::host_topology();
    let mut result = eredu_core::DomainMemoryRequirements::zero(&topology);
    result
        .add_allocation(
            bytes,
            crate::working_memory::memory_fixture::host_placement(),
        )
        .unwrap();
    result
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 2,
        cached_positions: 2,
        input_positions: 7,
        max_output_tokens: 3,
        prefill_chunk_positions: 3,
        output: OutputDemand::LastPosition,
    }
}
fn outside(geometry: InferenceGeometry) -> ExecutionWorkspaceEstimate {
    let zero = || WorkspaceBound::bounded(0, "fixture has no untraced allocations");
    ExecutionWorkspaceEstimate {
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
        physical_domains: Some(eredu_core::DomainExecutionWorkspaceEstimate {
            geometry,
            activations: fixture_requirements(0),
            attention: fixture_requirements(0),
            vocabulary: fixture_requirements(0),
            state_update: fixture_requirements(0),
            materialization: fixture_requirements(0),
            retained: fixture_requirements(0),
        }),
    }
}
fn request(g: InferenceGeometry) -> AdmissionRequest {
    AdmissionRequest {
        input: InputTokenCount::text(g.cached_positions + g.input_positions),
        max_output_tokens: g.max_output_tokens,
        batch_size: g.batch_size,
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
    }
}
fn state(g: InferenceGeometry) -> RuntimeStateEstimate {
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    eredu_core::estimate_runtime_state(
        &layout,
        request(g).input,
        g.max_output_tokens,
        g.batch_size,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
}
fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        effective_model_type: "shared equation inspection".into(),
        native_max_context: Observed::exact(128, "fixture"),
        effective_max_context: Observed::exact(128, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::PersistentStateOnly,
    }
}

#[test]
fn request_quote_prices_overlapping_cache_generations_and_rejects_the_old_underestimate() {
    use crate::{ArchitectureStateFactory, RuntimeLayerState};
    use eredu_core::{AdmissionResult, AttentionPolicy, cache::LayerCachePolicy};
    use eredu_nn::AttentionCache;
    #[derive(Debug)]
    struct StateFacts;
    impl WorkspaceMechanisms for StateFacts {
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
            Ok(Some(WorkspaceOperationBound {
                outputs: operation
                    .outputs
                    .iter()
                    .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                    .collect::<Result<_, _>>()?,
                scratch_bytes: 0,
                assumptions: "fixture allocates exact outputs without donation or extra scratch"
                    .into(),
            }))
        }
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "fixture has no disjoint host payload".into(),
            }))
        }
    }
    let g = geometry();
    let policy = LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 1).unwrap();
    let schedule = LayerSchedule::new(1, vec![policy]).unwrap();
    let context = WorkspaceContext::new(StateFacts);
    let mut factory = crate::working_memory::WorkspaceConcatStateFactory::new(
        std::num::NonZeroU32::new(2).unwrap(),
        &context,
    )
    .unwrap();
    let mut state = factory
        .realize(&crate::StateLayout::new(schedule.clone()).unwrap())
        .unwrap();
    let cache = &mut state.as_mut()[0];
    let prefix = WorkspaceTensor::full_f32(0.5, &[2, 1, 2, 1], &context).unwrap();
    cache
        .update_for_attention(prefix.clone(), prefix, &context)
        .unwrap();
    let report = quote_inference_workspace(g, |span| {
        let count = match span {
            InferenceWorkspaceSpan::Sampling(_) => unreachable!("model-only traversal fixture"),
            InferenceWorkspaceSpan::Prefill(chunk) => chunk.input.end - chunk.input.start,
            InferenceWorkspaceSpan::Decode { .. } => 1,
        };
        context.begin_state_span(cache.retained_values())?;
        let input = WorkspaceTensor::full_f32(0.25, &[2, 1, count as i32, 1], &context)?;
        cache.update_for_attention(input.clone(), input, &context)?;
        context.report(&cache.retained_values().cloned().collect::<Vec<_>>())
    })
    .unwrap();
    // Twelve final positions, two batches, one F32 key. The last decode
    // overlaps 88 old bytes, its 8 input bytes, and the 96-byte new cache.
    assert_eq!(report.retained_peak_bytes(), Some(96));
    assert_eq!(report.transient().bytes(), Some(96));
    assert_eq!(report.tensor_transient_peak_bytes(), Some(96));
    assert!(matches!(
        report.peak_span(),
        Some(InferenceWorkspaceSpan::Decode {
            index: 2,
            position: 11,
            ..
        })
    ));
    let layout =
        StateMemoryLayout::new(schedule, vec![0], 1, 1, EstimationCompleteness::Complete).unwrap();
    let persistent = eredu_core::estimate_runtime_state(
        &layout,
        request(g).input,
        g.max_output_tokens,
        g.batch_size,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    assert_eq!(persistent.requested_state_bytes, 96);
    let estimate = report.compose(persistent, outside(g)).unwrap();
    let mut request = request(g);
    // Displaced state contributes to both persistent and transient demand.
    request.memory_limits = crate::working_memory::memory_fixture::host_limits(128);
    let AdmissionResult::Admitted(short) =
        eredu_core::apply_admission_policy(&capabilities(), request.clone(), estimate.clone())
            .unwrap()
    else {
        panic!("complete domain requirements")
    };
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &short),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    request.memory_limits = Default::default();
    let AdmissionResult::Admitted(admission) =
        eredu_core::apply_admission_policy(&capabilities(), request, estimate).unwrap()
    else {
        panic!("complete overlap bound must fit exact capacity");
    };
    assert_eq!(admission.incremental_required_bytes, Some(192));
    let total = crate::working_memory::memory_fixture::reservation_bytes(&pool, &admission);
    let pool = crate::working_memory::memory_fixture::host_ledger(total, 0).unwrap();
    let owner = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission)
        .unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 192);
    drop(owner);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn primitive_only_trace_cannot_authorize_an_inference_request() {
    let g = geometry();
    let context = WorkspaceContext::new(Facts {
        tensor: 4,
        host: Some(0),
    });
    let report = quote_inference_workspace(g, |_| {
        context.begin_span();
        context.report(&[])
    })
    .unwrap();
    assert!(report.first_gap().is_some());
    assert_eq!(report.transient().bytes(), None);
    let estimate = report.compose(state(g), outside(g)).unwrap();
    assert!(matches!(
        eredu_core::apply_admission_policy(&capabilities(), request(g), estimate).unwrap(),
        eredu_core::AdmissionResult::Rejected(
            eredu_core::AdmissionRejection::EstimationUnsupported { .. }
        )
    ));
}

#[test]
fn scheduler_quotes_all_semantic_spans_and_keeps_simultaneous_domains_together() {
    for demand in [
        OutputDemand::LastPosition,
        OutputDemand::StateOnly,
        OutputDemand::Sequence,
    ] {
        let g = InferenceGeometry {
            output: demand,
            ..geometry()
        };
        let mut spans = Vec::new();
        let report = quote_inference_workspace(g, |span| {
            spans.push(span.clone());
            Ok::<_, Error>(if spans.len() == 2 {
                trace(100, Some(10), false)
            } else {
                trace(4, Some(80), false)
            })
        })
        .unwrap();
        let intermediate = if demand == OutputDemand::Sequence {
            OutputDemand::Sequence
        } else {
            OutputDemand::StateOnly
        };
        assert_eq!(
            spans,
            vec![
                InferenceWorkspaceSpan::Prefill(PrefillChunk {
                    input: 0..3,
                    position: 2,
                    output: intermediate
                }),
                InferenceWorkspaceSpan::Prefill(PrefillChunk {
                    input: 3..6,
                    position: 5,
                    output: intermediate
                }),
                InferenceWorkspaceSpan::Prefill(PrefillChunk {
                    input: 6..7,
                    position: 8,
                    output: demand
                }),
                InferenceWorkspaceSpan::Decode {
                    index: 0,
                    position: 9,
                    output: OutputDemand::LastPosition
                },
                InferenceWorkspaceSpan::Decode {
                    index: 1,
                    position: 10,
                    output: OutputDemand::LastPosition
                },
                InferenceWorkspaceSpan::Decode {
                    index: 2,
                    position: 11,
                    output: OutputDemand::LastPosition
                },
            ]
        );
        assert_eq!(report.completed_spans(), 6);
        assert_eq!(report.transient().bytes(), Some(110));
        assert_eq!(report.tensor_transient_peak_bytes(), Some(100));
        assert_eq!(report.host_peak_bytes(), Some(80));
        assert_eq!(report.peak_span(), Some(&spans[1]));
        let composed = report.compose(state(g), outside(g)).unwrap();
        assert_eq!(
            composed.execution_workspace.unwrap().peak_bytes().unwrap(),
            Some(110)
        );
    }
}

#[test]
fn one_unpriced_late_decode_cannot_be_hidden_by_larger_complete_spans() {
    let g = geometry();
    let gap = InferenceWorkspaceSpan::Decode {
        index: 1,
        position: 10,
        output: OutputDemand::LastPosition,
    };
    let report = quote_inference_workspace(g, |span| {
        Ok::<_, Error>(if span == &gap {
            trace(4, None, false)
        } else if matches!(span, InferenceWorkspaceSpan::Decode { index: 2, .. }) {
            trace(160, Some(4), false)
        } else {
            trace(96, Some(4), false)
        })
    })
    .unwrap();
    assert_eq!(report.completed_spans(), 6);
    assert_eq!(report.transient().bytes(), None);
    assert_eq!(report.tensor_transient_peak_bytes(), Some(160));
    assert_eq!(report.host_peak_bytes(), None);
    assert_eq!(report.first_gap(), Some(&gap));
    let result = eredu_core::apply_admission_policy(
        &capabilities(),
        request(g),
        report.compose(state(g), outside(g)).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        result,
        eredu_core::AdmissionResult::Rejected(
            eredu_core::AdmissionRejection::EstimationUnsupported { .. }
        )
    ));
}

#[test]
fn full_request_quotes_select_chunks_and_reserve_against_concurrent_work() {
    let g = InferenceGeometry {
        prefill_chunk_positions: 5,
        ..geometry()
    };
    let execution = InferenceExecutionIdentity::default();
    let mut candidates = Vec::new();
    let mut quote = |g: InferenceGeometry| {
        candidates.push(g.prefill_chunk_positions);
        let report = quote_inference_workspace(g, |span| {
            let bytes = match span {
                InferenceWorkspaceSpan::Sampling(_) => unreachable!("model-only traversal fixture"),
                InferenceWorkspaceSpan::Prefill(chunk) => match chunk.input.end - chunk.input.start
                {
                    5 => 150,
                    4 => 160,
                    3 => 96,
                    _ => 64,
                },
                // The later decode peak dominates every smaller chunk.
                InferenceWorkspaceSpan::Decode { index: 2, .. } => 90,
                _ => 4,
            };
            Ok::<_, Error>(trace(bytes, Some(0), false))
        })
        .unwrap();
        report.compose(state(g), outside(g))
    };
    let probe = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let sample_geometry = InferenceGeometry {
        prefill_chunk_positions: 3,
        ..g
    };
    let sample = quote(sample_geometry).unwrap();
    let AdmissionResult::Admitted(sample) =
        eredu_core::apply_admission_policy(&capabilities(), request(sample_geometry), sample)
            .unwrap()
    else {
        panic!("complete fixture")
    };
    let metadata = crate::working_memory::memory_fixture::reservation_bytes(&probe, &sample) - 96;
    let pool = crate::working_memory::memory_fixture::host_ledger(100 + metadata, 0).unwrap();
    let (_, first) = plan_prefill(
        &execution,
        &pool,
        &capabilities(),
        request(g),
        g,
        &mut quote,
    )
    .unwrap();
    assert_eq!(first.geometry().prefill_chunk_positions, 3);
    assert_eq!(
        first
            .requirements()
            .get(crate::working_memory::memory_fixture::host_topology_ref().host_domain())
            .ok()
            .and_then(|charge| charge.total().ok())
            .unwrap(),
        96 + metadata
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 96);
    assert!(matches!(
        plan_prefill(
            &execution,
            &pool,
            &capabilities(),
            request(g),
            g,
            &mut quote
        ),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
        ))
    ));
    drop(first);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let (_, next) = plan_prefill(
        &execution,
        &pool,
        &capabilities(),
        request(g),
        g,
        &mut quote,
    )
    .unwrap();
    assert_eq!(
        next.requirements()
            .get(crate::working_memory::memory_fixture::host_topology_ref().host_domain())
            .ok()
            .and_then(|charge| charge.total().ok())
            .unwrap(),
        96 + metadata
    );
    assert_eq!(&candidates[1..4], &[5, 4, 3]);
    assert_eq!(pool.payload_peak_bytes().unwrap(), 96 + metadata);
    drop(next);
    // Smaller prompt chunks cannot eliminate an over-budget late cache update.
    // No reservation (and therefore no authorized preparation) may result.
    let rejected = plan_prefill(
        &execution,
        &pool,
        &capabilities(),
        request(g),
        g,
        |candidate| {
            let report = quote_inference_workspace(candidate, |span| {
                Ok::<_, Error>(
                    if matches!(span, InferenceWorkspaceSpan::Decode { index: 2, .. }) {
                        trace(120, Some(0), false)
                    } else {
                        trace(32, Some(0), false)
                    },
                )
            })
            .unwrap();
            report.compose(state(candidate), outside(candidate))
        },
    );
    assert!(matches!(
        rejected,
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
        ))
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn failed_quote_stops_traversal_and_composition_keeps_geometry_and_state_guards() {
    let g = geometry();
    let mut visited = 0;
    let error = quote_inference_workspace(g, |span| {
        visited += 1;
        if matches!(span, InferenceWorkspaceSpan::Decode { index: 0, .. }) {
            Err(Error::backend("fixture failure"))
        } else {
            Ok(trace(4, Some(0), false))
        }
    })
    .unwrap_err();
    assert!(matches!(
        error,
        InferenceWorkspaceError::Quote {
            span: InferenceWorkspaceSpan::Decode { index: 0, .. },
            ..
        }
    ));
    assert_eq!(visited, 4);
    let invalid_geometry = InferenceGeometry {
        prefill_chunk_positions: 0,
        ..g
    };
    assert!(
        quote_inference_workspace(
            invalid_geometry,
            |_| -> Result<WorkspaceTraceReport, Error> {
                panic!("invalid geometry must not inspect equations")
            }
        )
        .is_err()
    );
    let report = quote_inference_workspace(g, |_| Ok::<_, Error>(trace(4, Some(0), true))).unwrap();
    let actual = report.compose(state(g), outside(g)).unwrap();
    assert_eq!(
        actual
            .physical_domains
            .as_ref()
            .unwrap()
            .decoder_state
            .get(crate::working_memory::memory_fixture::host_topology().host_domain())
            .unwrap()
            .total()
            .unwrap(),
        4,
        "canonical retained backing refines the logical state estimate"
    );
    let report =
        quote_inference_workspace(g, |_| Ok::<_, Error>(trace(4, Some(0), false))).unwrap();
    assert!(
        report
            .compose(
                state(g),
                outside(InferenceGeometry {
                    prefill_chunk_positions: 1,
                    ..g
                })
            )
            .is_err()
    );
    let mut unknown = outside(g);
    unknown.materialization = WorkspaceBound::Unknown {
        reason: "fixture missing materialization".into(),
    };
    assert_eq!(
        report.compose(state(g), unknown).unwrap().completeness,
        EstimationCompleteness::PersistentStateOnly
    );
}

#[test]
fn invalid_request_identity_or_context_rejects_before_inspecting_equations() {
    let execution = InferenceExecutionIdentity::default();
    let pool = crate::working_memory::memory_fixture::host_ledger(100, 0).unwrap();
    let g = geometry();
    let mut wrong = request(g);
    wrong.batch_size += 1;
    assert!(matches!(
        plan_prefill(&execution, &pool, &capabilities(), wrong, g, |_| {
            panic!("different request must not invoke its workspace provider")
        }),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    for (input, output) in [(129, 0), (125, 2), (7, u64::MAX - 9)] {
        let oversized = InferenceGeometry {
            input_positions: input,
            max_output_tokens: output,
            ..g
        };
        assert!(matches!(
            plan_prefill(
                &execution,
                &pool,
                &capabilities(),
                request(oversized),
                oversized,
                |_| {
                    panic!("context rejection must precede inspection of a large decode allowance")
                }
            ),
            Err(PrefillPlanningError::Admission(_))
        ));
    }
    let mut unknown = capabilities();
    unknown.effective_max_context = Observed::Unavailable {
        reason: "fixture missing context limit".into(),
    };
    assert!(matches!(
        plan_prefill(&execution, &pool, &unknown, request(g), g, |_| {
            panic!("missing context must reject before equation inspection")
        }),
        Err(PrefillPlanningError::Admission(
            eredu_core::AdmissionRejection::EstimationUnsupported { .. }
        ))
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

mod span_workspace;

#[test]
fn native_prefill_envelope_keeps_initial_state_escaped_outputs_and_unknown_fallback() {
    use crate::working_memory::{NativeEquationStorage, NativePrefillEnvelopeBuilder};
    let report = quote_inference_workspace(geometry(), |_| {
        Ok::<_, std::convert::Infallible>(trace(10, Some(0), false))
    })
    .unwrap();
    let plan = report.span_workspace_plan();
    assert_eq!(plan.records().len(), 6);
    for (initial, expected) in [(Some(160), 560), (Some(100), 530), (None, 600)] {
        let mut envelope = NativePrefillEnvelopeBuilder::new(plan);
        for (index, record) in plan.records().iter().enumerate() {
            envelope
                .push(
                    record.span(),
                    NativeEquationStorage {
                        construction_bytes: 100,
                        opening_state_bytes: if index == 0 {
                            initial
                        } else {
                            Some(20 + index as u64 * 5)
                        },
                        closing_state_bytes: Some(25 + index as u64 * 5),
                        output_bytes: Some(30),
                        validation_producer_bytes: Some(20),
                    },
                )
                .unwrap();
        }
        let envelope = envelope.finish().unwrap();
        assert_eq!(envelope.original_equation_bytes(), 600);
        assert_eq!(envelope.equation_bytes(), expected);
        if initial.is_none() {
            assert_eq!(envelope.successful_prefill_candidate(), None);
        }
    }
    assert!(matches!(
        NativePrefillEnvelopeBuilder::new(plan).finish(),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}

#[test]
fn native_completed_equation_envelope_carries_escaped_scores_and_late_unknowns() {
    use crate::working_memory::{NativeEquationStorage, NativePrefillEnvelopeBuilder};
    let report = quote_inference_workspace(geometry(), |_| {
        Ok::<_, std::convert::Infallible>(trace(10, Some(0), false))
    })
    .unwrap();
    let plan = report.span_workspace_plan();
    for (unknown_decode, expected) in [(false, 395), (true, 600)] {
        let mut envelope = NativePrefillEnvelopeBuilder::new_completed_equations(plan);
        for (index, record) in plan.records().iter().enumerate() {
            envelope
                .push(
                    record.span(),
                    NativeEquationStorage {
                        construction_bytes: 100,
                        opening_state_bytes: Some(if index == 0 {
                            100
                        } else {
                            20 + index as u64 * 5
                        }),
                        closing_state_bytes: Some(25 + index as u64 * 5),
                        output_bytes: Some(30),
                        validation_producer_bytes: if unknown_decode && index == 4 {
                            None
                        } else {
                            Some(20)
                        },
                    },
                )
                .unwrap();
        }
        let envelope = envelope.finish().unwrap();
        assert_eq!(envelope.original_equation_bytes(), 600);
        assert_eq!(envelope.equation_bytes(), expected);
        assert_eq!(envelope.successful_prefill_candidate(), Some(230));
        assert_eq!(
            envelope.successful_equation_candidate(),
            if unknown_decode { None } else { Some(395) }
        );
    }
}

#[test]
fn terminal_copy_geometry_schedules_no_forward_or_sampling_attempt() {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 7,
        input_positions: 0,
        max_output_tokens: 0,
        prefill_chunk_positions: 0,
        output: OutputDemand::StateOnly,
    };
    geometry.validate_fixed().unwrap();
    let report = quote_inference_workspace(geometry, |_| -> Result<WorkspaceTraceReport, Error> {
        panic!("terminal state placement has no model equation");
    })
    .unwrap();
    assert_eq!(report.completed_spans, 0);
    assert!(report.span_workspace_plan().records().is_empty());
    assert_eq!(
        report.span_workspace_plan().generation_forward_count(),
        Some(0)
    );
    for invalid in [
        InferenceGeometry {
            max_output_tokens: 1,
            ..geometry
        },
        InferenceGeometry {
            prefill_chunk_positions: 1,
            ..geometry
        },
        InferenceGeometry {
            output: OutputDemand::LastPosition,
            ..geometry
        },
    ] {
        assert!(invalid.validate_fixed().is_err());
    }
}
