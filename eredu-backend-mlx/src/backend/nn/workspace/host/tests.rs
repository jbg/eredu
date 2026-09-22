use super::*;
use eredu_nn::{NeuralBackend, RotaryOperator, RotaryPosition, RotarySpec, Tensor};

fn selected() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts {
            page_size: 16_384,
            cpu_header: false,
            original_storage: false,
        },
        sdpa_blocks: None,
    }
}
fn layout(shape: &[i32]) -> WorkspaceLayout {
    WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap()
}

#[test]
fn rotary_host_vectors_are_separate_from_native_frequency_copies() {
    for dimensions in [8, 128] {
        for arithmetic in [RotaryArithmetic::Native, RotaryArithmetic::InputProducts] {
            for (algorithm, native_vectors, extra_vectors) in [
                (RotaryAlgorithm::Default, 0, 1),
                (RotaryAlgorithm::Linear { factor: 2.0 }, 0, 1),
                (
                    RotaryAlgorithm::Llama3 {
                        factor: 8.0,
                        low_frequency_factor: 1.0,
                        high_frequency_factor: 4.0,
                        original_max_positions: 128,
                    },
                    0,
                    0,
                ),
                (
                    RotaryAlgorithm::Proportional {
                        factor: 2.0,
                        rotary_fraction: 0.5,
                    },
                    1,
                    0,
                ),
                (
                    RotaryAlgorithm::Yarn {
                        factor: 8.0,
                        original_max_positions: 128,
                        beta_fast: 32.0,
                        beta_slow: 1.0,
                        amplitude: 1.125,
                        truncate: true,
                    },
                    2,
                    1,
                ),
            ] {
                let context = WorkspaceContext::new(selected());
                let input =
                    WorkspaceTensor::existing(layout(&[2, 3, 7, dimensions]), &context).unwrap();
                let mut rotary = WorkspaceBackend::rotary(
                    RotarySpec {
                        algorithm,
                        arithmetic,
                        traditional: false,
                        dimensions,
                        base: 10_000.0,
                    },
                    &context,
                )
                .unwrap();
                let output = rotary
                    .forward(&input, RotaryPosition::Offset(3), &context)
                    .unwrap();
                let report = context.report(&[output]).unwrap();
                let vectors = native_vectors
                    + if arithmetic == RotaryArithmetic::InputProducts {
                        extra_vectors
                    } else {
                        0
                    };
                let host = dimensions as u64 / 2 * 4 * vectors;
                assert_eq!(report.host_workspace_bytes, Some(host));
                assert_eq!(
                    report.total_bytes,
                    report.tensor_buffers.total_bytes.map(|tensor| tensor
                        + host
                        + crate::backend::nn::workspace::test_backing_controls(
                            &selected(),
                            &report
                        ))
                );
                assert!(report
                    .assumptions
                    .iter()
                    .any(|a| a.contains("host payload")));
                // Caller-provided embeddings do not undo frequency storage
                // already allocated when the rotary operator was constructed.
                context.begin_span();
                let embedding =
                    WorkspaceTensor::existing(layout(&[7, dimensions]), &context).unwrap();
                let output = rotary
                    .forward(
                        &input,
                        RotaryPosition::Embeddings {
                            cosine: &embedding,
                            sine: &embedding,
                        },
                        &context,
                    )
                    .unwrap();
                let explicit = context.report(&[output]).unwrap();
                assert_eq!(explicit.host_workspace_bytes, Some(host));
                let operation = explicit.operations.last().unwrap();
                let mut default = operation.clone();
                default.kind = WorkspaceOperationKind::Rotary(
                    RotarySpec {
                        algorithm: RotaryAlgorithm::Default,
                        arithmetic: RotaryArithmetic::Native,
                        traditional: false,
                        dimensions,
                        base: 10_000.0,
                    },
                    None,
                );
                let base = selected()
                    .operation_bound(&default)
                    .unwrap()
                    .unwrap()
                    .scratch_bytes;
                let actual = selected()
                    .operation_bound(operation)
                    .unwrap()
                    .unwrap()
                    .scratch_bytes;
                let minimum_frequency_copies = match algorithm {
                    RotaryAlgorithm::Yarn { .. } | RotaryAlgorithm::Proportional { .. } => 2,
                    _ => 0,
                };
                assert!(
                    actual
                        >= base
                            + minimum_frequency_copies
                                * selected()
                                    .allocation
                                    .buffer_capacity(dimensions as u64 / 2 * 4)
                                    .unwrap()
                );
            }
        }
    }
}

#[test]
fn attention_host_masks_follow_native_row_threshold_and_completed_key_tiles() {
    for keys in [1, 8192, 8193, 16385] {
        for queries in [1, 7, 257] {
            for arithmetic in [AttentionArithmetic::Fused, AttentionArithmetic::InputScores] {
                let op = WorkspaceOperation {
                    kind: WorkspaceOperationKind::Attention {
                        causal: false,
                        window: None,
                        sinks: false,
                        softcap: false,
                        arithmetic,
                    },
                    inputs: vec![
                        layout(&[2, 4, queries, 16]),
                        layout(&[2, 2, keys, 16]),
                        layout(&[2, 2, keys, 8]),
                    ],
                    outputs: vec![layout(&[2, 4, queries, 8])],
                };
                let host = selected().host_workspace_bound(&op).unwrap().unwrap();
                let input_controls = if arithmetic == AttentionArithmetic::InputScores {
                    super::super::super::matrix::batched_input_control_bytes().unwrap() as u64
                } else {
                    0
                };
                assert_eq!(
                    host.bytes,
                    input_controls
                        + if keys > 8192 && arithmetic == AttentionArithmetic::InputScores {
                            256
                        } else {
                            0
                        }
                );
                let mut invalid = op;
                invalid.outputs[0] = layout(&[2, 4, queries, 16]);
                assert!(selected().host_workspace_bound(&invalid).is_err());
            }
        }
    }
}

#[test]
#[cfg(not(feature = "cuda"))]
fn dense_and_affine_hybrid_spans_have_independent_tensor_and_host_bounds() {
    use eredu_architectures::qwen::hybrid;
    use eredu_core::OutputDemand;
    use eredu_runtime::working_memory::{
        quote_inference_workspace, InferenceWorkspaceSpan, WorkspaceResidentStateFactory,
    };
    use eredu_runtime::RuntimeStateComponents;
    use eredu_runtime::{
        ArchitectureStateFactory, ExpertPass, LayerRuntimeState, LayerwiseRuntime, NoopObserver,
        ResidentExpertProvider, ResidentUnitWindow, RuntimeLayerState,
    };
    use std::num::NonZeroU32;
    let mut trajectories = 0;
    let mut completed_spans = 0;
    for packed in [false, true] {
        for tied in [false, true] {
            for chunk in [1, 3, 7] {
                for demand in [
                    OutputDemand::StateOnly,
                    OutputDemand::LastPosition,
                    OutputDemand::Sequence,
                ] {
                    let mut config = serde_json::json!({
                        "model_type":"qwen3_5_text", "vocab_size":37,"hidden_size":64,
                        "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
                        "head_dim":16,"max_position_embeddings":128,"linear_conv_kernel_dim":3,
                        "linear_key_head_dim":16,"linear_value_head_dim":16,
                        "linear_num_key_heads":2,"linear_num_value_heads":4,
                        "intermediate_size":64,"layer_types":["linear_attention","full_attention"],
                        "tie_word_embeddings":tied,
                        "rope_parameters":{"partial_rotary_factor":1.0,"rope_theta":10000.0}
                    });
                    if packed {
                        config["quantization"] = serde_json::json!({"group_size":32,"bits":4});
                    }
                    let args = hybrid::model_args_from_config_value(&config).unwrap().text;
                    let context = WorkspaceContext::new(selected());
                    let architecture =
                        hybrid::LayeredModel::<WorkspaceBackend>::new(args, &context).unwrap();
                    let geometry = architecture.state_layout().unwrap();
                    let units = (0..2)
                        .map(|i| architecture.construct_unit(0, i, &context).unwrap())
                        .collect();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let mut state = WorkspaceResidentStateFactory::new(
                        NonZeroU32::new(2).unwrap(),
                        NonZeroU32::new(256).unwrap(),
                        &context,
                    )
                    .unwrap()
                    .realize(&geometry)
                    .unwrap();
                    let mut offset = 0;
                    let mut inspect = |position: usize,
                                       count: usize,
                                       output_demand: OutputDemand,
                                       pass: ExpertPass| {
                        assert_eq!(position, offset);
                        context
                            .begin_state_span(
                                state
                                    .as_ref()
                                    .iter()
                                    .flat_map(|layer| layer.retained_values()),
                            )
                            .unwrap();
                        let ids = (0..2 * count)
                            .map(|i| ((offset + 3 * i + 1) % 37) as i32)
                            .collect::<Vec<_>>();
                        let tokens =
                            WorkspaceTensor::from_i32_slice(&ids, &[2, count as i32], &context)
                                .unwrap();
                        let (scores, _) = runtime
                            .forward_with_provider_and_observer_and_context_with_readout(
                                hybrid::EmbeddedInput::target(&tokens, None),
                                &mut state,
                                pass,
                                &mut ResidentExpertProvider,
                                &context,
                                &mut NoopObserver,
                                output_demand,
                            )
                            .unwrap();
                        offset += count;
                        assert_eq!(scores.is_some(), output_demand != OutputDemand::StateOnly);
                        if let Some(scores) = scores {
                            assert_eq!(
                                scores.shape(),
                                [2, output_demand.positions(count as u64) as i32, 37]
                            );
                        }
                        let mut roots = Vec::new();
                        for layer in 0..2 {
                            let lane = state.layer(layer).unwrap();
                            assert_eq!(lane.position(), offset as i32);
                            roots.extend(lane.retained_values().cloned());
                        }
                        let report = context.report(&roots).unwrap();
                        let gaps = report
                            .unpriced_operations
                            .iter()
                            .chain(&report.unpriced_host_operations)
                            .map(|&i| &report.operations[i].kind)
                            .collect::<Vec<_>>();
                        assert!(gaps.is_empty(), "hybrid packed={packed} gaps: {gaps:?}");
                        assert_eq!(report.host_workspace_bytes, Some(0));
                        assert!(report.total_bytes.unwrap() >= report.retained_bytes.unwrap());
                        assert_eq!(
                            report.total_bytes,
                            report.tensor_buffers.total_bytes.map(|n| n
                                + crate::backend::nn::workspace::test_backing_controls(
                                    &selected(),
                                    &report
                                ))
                        );
                        completed_spans += 1;
                        report
                    };
                    inspect(0, 2, OutputDemand::StateOnly, ExpertPass::Prefill);
                    let geometry = eredu_core::InferenceGeometry {
                        batch_size: 2,
                        cached_positions: 2,
                        input_positions: 7,
                        max_output_tokens: 3,
                        prefill_chunk_positions: chunk as u64,
                        output: demand,
                    };
                    let request = quote_inference_workspace(geometry, |span| {
                        let report = match span {
                            InferenceWorkspaceSpan::Sampling(_) => {
                                panic!("model scheduler emitted a sampling phase")
                            }
                            InferenceWorkspaceSpan::Prefill(chunk) => inspect(
                                chunk.position as usize,
                                (chunk.input.end - chunk.input.start) as usize,
                                chunk.output,
                                ExpertPass::Prefill,
                            ),
                            InferenceWorkspaceSpan::Decode {
                                position, output, ..
                            } => inspect(*position as usize, 1, *output, ExpertPass::Decode),
                        };
                        Ok::<_, Error>(report)
                    })
                    .unwrap();
                    assert_eq!(request.geometry(), geometry);
                    assert_eq!(request.completed_spans(), 7_u64.div_ceil(chunk as u64) + 3);
                    assert!(request.transient().bytes().is_some());
                    assert_eq!(request.host_peak_bytes(), Some(0));
                    assert!(request.first_gap().is_none());
                    trajectories += 1;
                }
            }
        }
    }
    assert_eq!(trajectories, 36);
    assert_eq!(completed_spans, 276);
}
