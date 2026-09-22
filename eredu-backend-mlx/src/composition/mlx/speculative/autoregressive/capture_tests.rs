//! Nonzero observed execution through real selected AR claims and model sources.
use super::*;
use crate::backend::{
    MlxAcceleratorFamily, MlxBackend, MlxDeviceIdentity,
    managed_memory::gpu_stream::PreparedExecutionStreams,
};
use crate::composition::mlx::MlxPreparedInputMaterializer;
use crate::composition::mlx::speculative::sampling::numerical::native_tests::{
    load, settle, source_configs_with_capacity,
};
use crate::memory_fixture::LedgerFixture as _;
use eredu_core::{
    GenerationCancellationToken, SpeculativeConfig, SpeculativePrefillOutcome,
    SpeculativeSchedulerOptions, capture::*, speculative::*,
};
use eredu_runtime::speculative::autoregressive::{
    AutoregressiveInvocation, AutoregressiveSchedulePlan,
};
use std::num::{NonZeroU64, NonZeroUsize};

const OBSERVED_PASSES: [(AutoregressivePass, u64, usize); 3] = [
    (AutoregressivePass::TargetPrefill, 0, 2),
    (AutoregressivePass::Verification, 2, 3),
    (AutoregressivePass::TargetCommit, 5, 1),
];

/// Each local layer can retain its live pages while the next invocation owns
/// prepared Host destinations for every page in its complete recorded frontier.
/// The manager's eviction tiers remain independent from this shared pool census.
pub(crate) fn prepared_cache_pool_populations(local_layers: usize) -> u64 {
    let frontier = OBSERVED_PASSES
        .iter()
        .map(|(_, start, width)| start.checked_add(u64::try_from(*width).unwrap()).unwrap())
        .max()
        .unwrap();
    u64::try_from(local_layers)
        .unwrap()
        .checked_mul(frontier)
        .and_then(|pages| pages.checked_mul(2)) // live backing and prepared destination
        .unwrap()
}

#[test]
#[ignore = "requires managed native allocator and CPU model execution"]
fn observed_original_autoregressive_cpu_keeps_two_three_one_geometry() {
    exercise(safemlx::DeviceType::Cpu)
}
#[test]
#[ignore = "requires Metal model execution"]
fn observed_original_autoregressive_metal_keeps_two_three_one_geometry() {
    exercise(safemlx::DeviceType::Gpu)
}

fn exercise(device: safemlx::DeviceType) {
    if !crate::tests::support::native_process::enter("original-speculative-source") {
        return;
    }
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let streams = match device {
        safemlx::DeviceType::Cpu => PreparedExecutionStreams::for_cpu_factory_with_matmul(
            &pool,
            crate::backend::nn::workspace::MlxCpuMatmulMechanism::select(
                eredu_nn::CpuMatmulImplementation::Float32Tiles,
            )
            .unwrap(),
        ),
        safemlx::DeviceType::Gpu => PreparedExecutionStreams::for_factory(&pool),
    }
    .unwrap()
    .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(
            &safemlx::Device::new(device, 0),
            matches!(device, safemlx::DeviceType::Gpu).then_some(MlxAcceleratorFamily::Metal),
        )
        .unwrap(),
    );
    let baseline = pool.fixture_host_charge().unwrap();
    let (target_config, draft_config, selected) = source_configs_with_capacity(
        &backend,
        artifact.path(),
        2,
        if matches!(device, safemlx::DeviceType::Cpu) {
            "cpu:0"
        } else {
            "gpu:0"
        },
    );
    let mut target = load(&backend, &target_config);
    let draft = load(&backend, &draft_config);
    let records = observed_sequence(
        &backend,
        &mut target,
        &draft,
        &selected,
        &pool,
        false,
        false,
        2,
    );
    drop((
        records,
        target,
        draft,
        target_config,
        draft_config,
        selected,
    ));
    settle(&pool, baseline);
}

fn observed_sequence(
    backend: &MlxBackend<'_>,
    target: &mut crate::composition::mlx::MlxModelSession,
    draft: &crate::composition::mlx::MlxModelSession,
    selected: &eredu_runtime::SelectedSpeculativeRealization,
    pool: &eredu_runtime::working_memory::MemoryLedger,
    partitioned: bool,
    evidence: bool,
    prefill_chunk: u64,
) -> Vec<SpeculativeActivationCapture> {
    let config = SpeculativeConfig {
        max_tokens: 5,
        max_draft_tokens: 2,
        temperature: 0.0,
        eos_token_ids: Vec::new(),
    };
    let schedule = AutoregressiveSchedulePlan::new(
        selected,
        NonZeroUsize::new(2).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let ids = [1u32, 2];
    let parts = [eredu_runtime::input::host::HostInputPart {
        modality: eredu_core::InputModality::Text,
        kind: eredu_core::InputPayloadKind::TokenIds,
        payload: eredu_runtime::input::host::HostTensorView {
            shape: &[1, 2],
            values: eredu_runtime::input::host::HostTensorValues::U32(&ids),
        },
        metadata: &[],
        extents: &[],
    }];
    let source = pool
        .compile_prepared_host_input(
            eredu_runtime::input::host::PreparedHostInputPlan::prepare(&parts).unwrap(),
        )
        .unwrap();
    let prompt = MlxPreparedInputMaterializer::prepare_admitted(pool)
        .unwrap()
        .text_input_plan(&source)
        .unwrap()
        .materialize(pool)
        .unwrap()
        .into_prompt(NonZeroU64::new(prefill_chunk));
    drop(source);
    // This finite parity budget covers the actual capture transport and
    // construction sources across all observed invocation roles.
    const OBSERVED_REQUEST_CEILING: u64 = 16 << 30;
    let pair = AutoregressiveSourcePair::prepare(
        target.original_model_source().unwrap(),
        draft.original_model_source().unwrap(),
        &schedule,
        pool,
        crate::memory_fixture::resolved_limits(OBSERVED_REQUEST_CEILING),
    )
    .unwrap();
    let declaration = target
        .original_intervention_declaration(pair.metadata_funding())
        .unwrap();
    let pair = pair.with_intervention_declaration(declaration).unwrap();
    let partition = target
        .prepare_original_model_partition_source(pair.metadata_funding())
        .unwrap();
    assert_eq!(partition.is_some(), partitioned);
    let pair = pair
        .with_partition_source(
            eredu_runtime::speculative::autoregressive::AutoregressiveSource::Target,
            partition,
        )
        .unwrap();
    let discovery = pair
        .numerical_sources()
        .activation_control_declaration()
        .unwrap()
        .activation_discovery()
        .unwrap();
    let selections = discovery
        .captures
        .catalog
        .points
        .iter()
        .filter_map(|point| {
            let axis = point
                .axes
                .as_ref()?
                .iter()
                .find(|axis| axis.name == "component")?;
            let eredu_core::SymbolicDimension::Known(width) = axis.dimension else {
                return None;
            };
            Some(CaptureSelection {
                id: point.path.clone(),
                path: point.path.clone(),
                schedule: CaptureSchedule::default(),
                transform: CaptureTransform::Slice,
                slices: vec![CaptureSlice {
                    axis: "component".into(),
                    start: 0,
                    end: width as u64,
                    stride: 3,
                }],
            })
        })
        .collect::<Vec<_>>();
    assert!(selections.len() >= 4);
    // Give each selected point a finite fixture quota for producer records,
    // protocol encoding and assembled output. Evidence has before/after rows.
    // The shared quota workers still charge the actual source-derived bounds.
    const ENCODED_BYTES_PER_POINT: u64 = 256 << 10;
    let selected_records = u64::try_from(selections.len())
        .unwrap()
        .checked_add(if evidence { 2 } else { 0 })
        .unwrap();
    let encoded_per_step = selected_records
        .checked_mul(ENCODED_BYTES_PER_POINT)
        .unwrap();
    // This cap also sizes the initial padded U32 transport. Check the actual
    // TP2 gather dimension before handing the plan to the cold native query.
    assert!(i32::try_from(encoded_per_step.div_ceil(4).checked_mul(2).unwrap()).is_ok());
    let observed_invocations = 2u64.div_ceil(prefill_chunk).checked_add(2).unwrap();
    let usage = CaptureUsage {
        captures: 1000,
        // Retention and Host quotas cover the actual receiver, gather and
        // assembly lifetimes; the ledger independently admits physical charges.
        retained_bytes: OBSERVED_REQUEST_CEILING,
        host_bytes: OBSERVED_REQUEST_CEILING,
        encoded_bytes: encoded_per_step,
    };
    let cumulative = CaptureUsage {
        encoded_bytes: encoded_per_step.checked_mul(observed_invocations).unwrap(),
        ..usage
    };
    let plan = SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        captures: CapturePlan {
            schema_version: 1,
            selections,
            limits: CaptureLimits {
                per_step: usage,
                cumulative,
                on_limit: CaptureLimitPolicy::Fail,
            },
        },
        interventions: if evidence {
            use eredu_core::intervention::*;
            let point = discovery
                .interventions
                .points
                .iter()
                .find(|point| {
                    point.stage == InterventionStage::Activation
                        && point.axes.iter().any(|axis| axis.name == "component")
                        && point.operations.contains(&InterventionKind::Scale)
                        && point.prefill == eredu_core::ObservationSupportStatus::Supported
                        && point.decode == eredu_core::ObservationSupportStatus::Supported
                })
                .expect("actual dense component edit");
            InterventionPlan {
                schema_version: INTERVENTION_SCHEMA_VERSION,
                operations: vec![InterventionOperation {
                    id: "scale-component".into(),
                    target: point.path.clone(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![],
                    action: InterventionAction::Scale {
                        dtype: InterventionDtype::Float32,
                        factor: 1.25,
                    },
                    evidence: InterventionEvidence::Summary,
                }],
            }
        } else {
            eredu_core::intervention::InterventionPlan::none()
        },
        bounds: CaptureInvocationBounds {
            batch: 1,
            max_sequence: 3,
            max_context: None,
            max_predictions: 8,
        },
    }
    .admit(&discovery)
    .unwrap();
    let environment = backend.original_copy_environment().unwrap();
    let context = SpeculativeExecutionStreams::single(backend.stream())
        .with_original_sources(&pair, &environment)
        .unwrap();
    let mut observer = MlxAutoregressiveMechanisms::activation_observer(
        &plan,
        eredu_core::SpeculativeRequestId::new(0),
        context,
    )
    .unwrap()
    .unwrap();
    let passes = OBSERVED_PASSES;
    let mut invocations = Vec::new();
    for (pass, frontier, width) in passes {
        let mut found: Option<AutoregressiveInvocation> = None;
        schedule
            .domains()
            .iter()
            .find(|domain| domain.pass() == pass)
            .unwrap()
            .visit(|position, value| {
                if position == frontier && value.positions() == width {
                    found = Some(value);
                }
                Ok::<_, std::convert::Infallible>(())
            })
            .unwrap();
        invocations.push(found.expect("actual schedule contains exact geometry"));
    }
    let mut cursor = schedule.into_cursor();
    let mut records = Vec::new();
    target
        .with_model_operation_funded(pair.metadata_funding().clone(), |model| {
            let mut state = MlxAutoregressiveMechanisms::empty(
                model,
                AutoregressivePass::TargetPrefill,
                context,
            ).map_err(|cause| {
                eprintln!("AR provider startup failed: {cause:?}");
                cause
            })?;
            for (step, ((pass, frontier, width), invocation)) in
                passes.into_iter().zip(invocations).enumerate()
            {
                let phase = match pass {
                    AutoregressivePass::TargetPrefill => SpeculativeActivationPhase::TargetPrefill,
                    AutoregressivePass::Verification => SpeculativeActivationPhase::Verification,
                    _ => SpeculativeActivationPhase::TargetReplay,
                };
                observer.set_activation_origin(Some(SpeculativeActivationOrigin {
                    request: eredu_core::SpeculativeRequestId::new(0),
                    committed_tokens: 0,
                    prediction: 4,
                    prefix_digest: [7; 32],
                    optimistic: false,
                }));
                let claim = cursor.claim(frontier, invocation).unwrap();
                let entered_numerical = std::cell::Cell::new(false);
                MlxAutoregressiveMechanisms::with_observed_invocation(
                    model,
                    &mut state,
                    (step == 0).then_some(&prompt),
                    claim,
                    context,
                    phase,
                    Some(&mut *observer),
                    |model, state, observer| {
                        entered_numerical.set(true);
                        if step == 0 {
                            let result = MlxAutoregressiveMechanisms::prefill_with_observer(
                                model,
                                &prompt,
                                state,
                                pass,
                                &GenerationCancellationToken::new(),
                                context,
                                observer,
                            )?;
                            let SpeculativePrefillOutcome::Complete(result) = result else {
                                panic!("uncancelled prefill")
                            };
                            assert_eq!(result.evaluated_tokens, 2);
                            assert!(result.logits.is_some());
                            drop(result);
                        } else {
                            let tokens = if width == 3 {
                                &[3u32, 4, 5][..]
                            } else {
                                &[6u32][..]
                            };
                            let output = MlxAutoregressiveMechanisms::decode(
                                model, tokens, state, pass, context,
                            )?;
                            assert_eq!(output.logits.shape()[1], width as i32);
                            drop(output);
                        }
                        Ok(())
                    },
                ).map_err(|cause| {
                    eprintln!("AR provider invocation {step}, frontier {frontier}, width {width}, evidence {evidence}, chunk {prefill_chunk}, numerical {} failed: {cause:?}", entered_numerical.get());
                    cause
                })?;
                assert_eq!(
                    state.native.generation_fixed(),
                    Some(frontier + width as u64)
                );
                let before = records.len();
                while let Some(record) = observer.take_activation_capture() {
                    assert!(record.completed);
                    assert_eq!(record.invocation, records.len() as u64);
                    let sequence = if step == 0 {
                        prefill_chunk.min(2)
                    } else {
                        width as u64
                    };
                    assert_eq!(
                        record.captures.as_step().invocation.unwrap().sequence,
                        sequence
                    );
                    assert!(
                        record
                            .captures
                            .as_step()
                            .records
                            .iter()
                            .all(|record| record.outcome == CaptureOutcome::Captured)
                    );
                    if evidence {
                        let edits = &record.captures.as_step().interventions;
                        assert_eq!(edits.len(), 1);
                        assert_eq!(
                            edits[0].outcome,
                            eredu_core::intervention::InterventionOutcome::Applied
                        );
                        assert_eq!(edits[0].evidence.len(), 2);
                        for side in &edits[0].evidence {
                            assert_eq!(side.outcome, CaptureOutcome::Captured);
                            let Some(CapturePayload::Summary(summary)) = side.payload.as_ref()
                            else {
                                panic!("actual component summary")
                            };
                            assert!(summary.elements > 0 && summary.finite == summary.elements);
                            assert!(summary.rms.is_some_and(|r| r > 0.0));
                        }
                    }
                    records.push(record);
                }
                assert_eq!(
                    records.len() - before,
                    if step == 0 {
                        2usize.div_ceil(prefill_chunk as usize)
                    } else {
                        1
                    }
                );
            }
            Ok(())
        })
        .unwrap();
    assert!(records.iter().flat_map(|value|&value.captures.as_step().records).filter_map(|value|value.payload.as_ref()?.as_tensor()).any(|value|matches!(value.data(),eredu_core::TensorObservationData::F32(values) if values.iter().any(|x|x.abs()>1e-6))));
    pair.request().close().unwrap();
    drop((observer, pair, prompt, plan, discovery));
    records
}

/// Compare the same admitted traversal against a loaded Ring target. The
/// source owner and all communication groups come from that actual session.
pub(crate) fn verify_partition_provider(
    backend: &MlxBackend<'_>,
    target: &mut crate::composition::mlx::MlxModelSession,
    path: &std::path::Path,
) {
    let pool = crate::backend::managed_memory::ledger();
    let baseline = pool.fixture_host_charge().unwrap();
    let device = backend.stream().get_device().unwrap().get_type().unwrap();
    let (reference_config, draft_config, selected) = source_configs_with_capacity(
        backend,
        path,
        2,
        if device == safemlx::DeviceType::Cpu {
            "cpu:0"
        } else {
            "gpu:0"
        },
    );
    let mut reference = load(backend, &reference_config);
    let draft = load(backend, &draft_config);
    for (evidence, chunk) in [(false, 2), (true, 2), (true, 1)] {
        let actual = observed_sequence(
            backend, target, &draft, &selected, &pool, true, evidence, chunk,
        );
        let expected = observed_sequence(
            backend,
            &mut reference,
            &draft,
            &selected,
            &pool,
            false,
            evidence,
            chunk,
        );
        assert_eq!(actual.len(), 2 + 2usize.div_ceil(chunk as usize));
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.phase, expected.phase);
            assert_eq!(actual.prefill_span, expected.prefill_span);
            let window = actual.prefill_span.map(|span| CaptureInvocationWindow {
                logical_sequence: span.prompt_tokens,
                start: span.hidden_start,
            });
            let actual = actual.captures.as_step();
            let expected = expected.captures.as_step();
            assert_eq!(actual.invocation, expected.invocation);
            assert_eq!(actual.records.len(), expected.records.len());
            assert_eq!(actual.partitions.len(), actual.records.len());
            assert!(expected.partitions.is_empty());
            assert!(
                actual
                    .partitions
                    .iter()
                    .any(|evidence| evidence.producers.len() == 2)
            );
            for evidence in &actual.partitions {
                assert!(
                    evidence
                        .context
                        .matches_invocation(actual.invocation, window)
                );
                assert_eq!(evidence.context.phase, actual.phase);
                assert_eq!(evidence.context.prediction, actual.prediction_index);
                assert!(evidence.context.forward_epoch > 0);
            }
            assert!(actual.capture_seconds > 0.0);
            assert_eq!(actual.interventions.len(), expected.interventions.len());
            for (actual, expected) in actual.interventions.iter().zip(&expected.interventions) {
                assert_eq!(actual.target, expected.target);
                assert_eq!(actual.outcome, expected.outcome);
                for (actual, expected) in actual.evidence.iter().zip(&expected.evidence) {
                    assert_eq!(actual.source_shape, expected.source_shape);
                    let (
                        Some(CapturePayload::Summary(actual)),
                        Some(CapturePayload::Summary(expected)),
                    ) = (actual.payload.as_ref(), expected.payload.as_ref())
                    else {
                        panic!("paired summary")
                    };
                    assert_eq!(
                        (actual.elements, actual.finite, actual.non_finite),
                        (expected.elements, expected.finite, expected.non_finite)
                    );
                    for (a, b) in [actual.min, actual.max, actual.mean, actual.rms]
                        .into_iter()
                        .zip([expected.min, expected.max, expected.mean, expected.rms])
                    {
                        let (Some(a), Some(b)) = (a, b) else {
                            panic!("nonzero finite fixture")
                        };
                        assert!(
                            (a - b).abs() <= 3e-4 + b.abs() * 3e-4,
                            "partition evidence {a} differs from ordinary {b}"
                        );
                    }
                }
            }

            for (actual, expected) in actual.records.iter().zip(&expected.records) {
                assert_eq!(actual.path, expected.path);
                assert_eq!(actual.source_shape, expected.source_shape);
                assert_eq!(actual.selected_shape, expected.selected_shape);
                assert_eq!(actual.outcome, CaptureOutcome::Captured);
                let actual = actual
                    .payload
                    .as_ref()
                    .and_then(CapturePayload::as_tensor)
                    .unwrap();
                let expected = expected
                    .payload
                    .as_ref()
                    .and_then(CapturePayload::as_tensor)
                    .unwrap();
                assert_eq!(actual.shape(), expected.shape());
                let (
                    eredu_core::TensorObservationData::F32(actual),
                    eredu_core::TensorObservationData::F32(expected),
                ) = (actual.data(), expected.data())
                else {
                    panic!("shared component F32 observation")
                };
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() <= 3e-4 + expected.abs() * 3e-4,
                        "partition component {actual} differs from ordinary {expected}"
                    );
                }
            }
        }
        drop((actual, expected));
    }
    drop((reference, draft, reference_config, draft_config, selected));
    settle(&pool, baseline);
}
