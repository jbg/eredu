use super::*;
use eredu_runtime::{
    ArchitectureStateFactory, ExpertPass, LayerRuntimeState, LayerwiseRuntime, NoopObserver,
    ResidentExpertProvider, ResidentUnitWindow, RuntimeLayerState, RuntimeStateComponents,
};
use std::num::NonZeroU32;

fn args(family: &str, tied: bool, packed: bool) -> qwen::hybrid::HybridConfig {
    let routed = family != "qwen3_5_text";
    let mut value = serde_json::json!({
        "model_type":family,"vocab_size":37,"hidden_size":64,
        "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":16,"max_position_embeddings":128,"linear_conv_kernel_dim":3,
        "linear_key_head_dim":16,"linear_value_head_dim":16,
        "linear_num_key_heads":2,"linear_num_value_heads":4,
        "intermediate_size":64,"moe_intermediate_size":64,
        "shared_expert_intermediate_size":64,"num_experts_per_tok":2,
        "num_experts":if routed {4} else {0},"norm_topk_prob":true,
        "layer_types":["linear_attention","full_attention"],"tie_word_embeddings":tied,
        "rope_parameters":{"partial_rotary_factor":1.0,"rope_theta":10000.0}
    });
    if packed {
        value["quantization"] = serde_json::json!({"group_size":32,"bits":4});
    }
    qwen::hybrid::model_args_from_config_value(&value)
        .unwrap()
        .text
}

fn run(
    family: &str,
    tied: bool,
    packed: bool,
    chunk: usize,
    demand: OutputDemand,
) -> Vec<WorkspaceTraceReport> {
    let context = WorkspaceContext::new(EquationMechanism {
        omit_attention: false,
    });
    let args = args(family, tied, packed);
    let architecture = qwen::hybrid::LayeredModel::<WorkspaceBackend>::new(args, &context).unwrap();
    let layout = architecture.state_layout().unwrap();
    let units = (0..2)
        .map(|index| architecture.construct_unit(0, index, &context).unwrap())
        .collect();
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let mut state = WorkspaceResidentStateFactory::new(
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(256).unwrap(),
        &context,
    )
    .unwrap()
    .realize(&layout)
    .unwrap();
    // Warm through the actual equations. Inventing logical existing tensors
    // would lose cache backing/copy and fixed-state initialization behavior.
    let mut spans = vec![(2, OutputDemand::StateOnly, ExpertPass::Prefill)];
    spans.extend((0..7).step_by(chunk).map(|start| {
        let count = chunk.min(7 - start);
        (
            count,
            demand.for_chunk(start + count == 7),
            ExpertPass::Prefill,
        )
    }));
    spans.extend([(1, OutputDemand::LastPosition, ExpertPass::Decode); 3]);
    let mut offset = 0;
    let mut reports = Vec::new();
    for (span, (count, demand, pass)) in spans.into_iter().enumerate() {
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
        let tokens = WorkspaceTensor::from_i32_slice(&ids, &[2, count as i32], &context).unwrap();
        let (scores, _) = runtime
            .forward_with_provider_and_observer_and_context_with_readout(
                qwen::hybrid::EmbeddedInput::target(&tokens, None),
                &mut state,
                pass,
                &mut ResidentExpertProvider,
                &context,
                &mut NoopObserver,
                demand,
            )
            .unwrap();
        assert_eq!(scores.is_some(), demand != OutputDemand::StateOnly);
        if let Some(scores) = scores {
            assert_eq!(
                scores.shape(),
                [2, demand.positions(count as u64) as i32, 37]
            );
        }
        offset += count;
        let mut roots = Vec::new();
        for layer in 0..2 {
            let lane = state.layer(layer).unwrap();
            assert_eq!(lane.position(), offset as i32);
            for component in layout.layer(layer).unwrap().fixed_state() {
                let value = lane
                    .fixed_component(component.role)
                    .unwrap()
                    .as_ref()
                    .unwrap();
                assert_eq!(value.shape(), component.resolved_shape(2, offset).unwrap());
            }
            let retained = lane.retained_values().cloned().collect::<Vec<_>>();
            assert_eq!(retained.len(), 2);
            if layer == 1 {
                assert!(retained
                    .iter()
                    .all(|value| value.shape() == [2, 2, offset as i32, 16]));
            }
            roots.extend(retained);
        }
        let report = context.report(&roots).unwrap();
        assert!(report.total_bytes.is_some());
        assert!(report.retained_bytes.unwrap() > 0);
        assert!(report
            .operations
            .iter()
            .any(|op| matches!(op.kind, WorkspaceOperationKind::GatedDeltaScan)));
        let readouts = report
            .operations
            .iter()
            .filter(|op| {
                matches!(op.kind, WorkspaceOperationKind::Projection(_))
                    && op.outputs[0].shape().last() == Some(&37)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            readouts.len(),
            usize::from(demand != OutputDemand::StateOnly)
        );
        if let Some(readout) = readouts.first() {
            assert_eq!(
                readout.inputs[0].shape(),
                [2, demand.positions(count as u64) as i32, 64]
            );
            assert_eq!(
                readout.inputs[1].shape(),
                if packed { vec![37, 8] } else { vec![37, 64] }
            );
        }
        let routed = report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
            .count();
        assert_eq!(routed, if family == "qwen3_5_text" { 0 } else { 2 });
        if span != 0 {
            reports.push(report);
        }
    }
    reports
}

#[test]
fn hybrid_equations_trace_attention_and_fixed_state_with_every_chunk_and_readout() {
    let mut trajectories = 0;
    let mut spans = 0;
    for family in ["qwen3_5_text", "qwen3_5_moe_text", "qwen3_next"] {
        for tied in [false, true] {
            for packed in [false, true] {
                for chunk in [1, 3, 7] {
                    for demand in [
                        OutputDemand::StateOnly,
                        OutputDemand::LastPosition,
                        OutputDemand::Sequence,
                    ] {
                        spans += run(family, tied, packed, chunk, demand).len();
                        trajectories += 1;
                    }
                }
            }
        }
    }
    assert_eq!(trajectories, 108);
    assert_eq!(spans, 720);
}
