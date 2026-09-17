use super::*;
use eredu_architectures::{deepseek, kimi_linear, replicated_text::ReplicatedTextStateProfiles};
use eredu_core::cache::LayerCachePolicy;
use eredu_runtime::{
    ArchitectureStateFactory, ExpertPass, LayerRuntimeState, LayerwiseRuntime, NoopObserver,
    ResidentExpertProvider, ResidentUnitWindow, RuntimeLayerState, RuntimeStateComponents,
    StateLayout,
};
use std::num::NonZeroU32;

type State = <WorkspaceResidentStateFactory as ReplicatedTextStateProfiles<WorkspaceBackend>>::CompressedComponentState;

fn run(
    context: &WorkspaceContext,
    layout: StateLayout,
    step: u32,
    chunk: usize,
    demand: OutputDemand,
    mut forward: impl FnMut(
        &WorkspaceTensor,
        &mut State,
        ExpertPass,
        OutputDemand,
    ) -> Result<Option<WorkspaceTensor>, Error>,
) -> usize {
    let mut state = WorkspaceResidentStateFactory::new(
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(step).unwrap(),
        context,
    )
    .unwrap()
    .realize(&layout)
    .unwrap();
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
    let count_spans = spans.len() - 1;
    let mut position = 0;
    for (count, demand, pass) in spans {
        context
            .begin_state_span(
                state
                    .as_ref()
                    .iter()
                    .flat_map(|layer| layer.retained_values()),
            )
            .unwrap();
        let ids = (0..2 * count)
            .map(|n| ((1 + position + n * 3) % 37) as i32)
            .collect::<Vec<_>>();
        let tokens = WorkspaceTensor::from_i32_slice(&ids, &[2, count as i32], context).unwrap();
        let scores = forward(&tokens, &mut state, pass, demand).unwrap();
        assert_eq!(scores.is_some(), demand != OutputDemand::StateOnly);
        if let Some(scores) = scores {
            assert_eq!(
                scores.shape(),
                [2, demand.positions(count as u64) as i32, 37]
            );
        }
        position += count;
        let mut roots = Vec::new();
        let mut compressed = 0;
        for (index, policy) in layout.layers().iter().enumerate() {
            let lane = state.layer(index).unwrap();
            assert_eq!(lane.position(), position as i32);
            match lane {
                WorkspaceResidentLayerState::Compressed(cache) => {
                    compressed += 1;
                    assert!(matches!(
                        policy,
                        LayerCachePolicy::CompressedLatentRotary { .. }
                    ));
                    assert_eq!(
                        cache.capacity(),
                        (position as u32).div_ceil(step) as i32 * step as i32
                    );
                }
                WorkspaceResidentLayerState::Ordinary(_) => {
                    for component in policy.fixed_state() {
                        let tensor = lane
                            .fixed_component(component.role)
                            .unwrap()
                            .as_ref()
                            .unwrap();
                        assert_eq!(
                            tensor.shape(),
                            component.resolved_shape(2, position).unwrap()
                        );
                    }
                }
                WorkspaceResidentLayerState::Pooling(_) => {
                    panic!("compressed-attention fixture unexpectedly selected pooling state")
                }
                WorkspaceResidentLayerState::Paged(_) => {
                    panic!("compressed-attention fixture unexpectedly selected paged state")
                }
            }
            roots.extend(lane.retained_values().cloned());
        }
        assert!(compressed > 0);
        let report = context.report(&roots).unwrap();
        assert!(report.total_bytes.is_some());
        assert!(report.retained_bytes.unwrap() > 0);
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
        }
        let updates = report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::SliceUpdate { .. }))
            .collect::<Vec<_>>();
        assert_eq!(updates.len(), 2 * compressed);
        for update in updates {
            assert_eq!(
                update.inputs[0].shape()[1],
                (position as u32).div_ceil(step) as i32 * step as i32
            );
            assert_eq!(update.inputs[1].shape()[1], count as i32);
        }
    }
    count_spans
}

#[test]
fn complete_kimi_hybrid_traces_recurrent_and_capacity_backed_latent_state() {
    let mut total = 0;
    let mut trajectories = 0;
    for packed in [false, true] {
        for low_rank in [false, true] {
            for step in [4, 256] {
                for chunk in [1, 3, 7] {
                    for demand in [
                        OutputDemand::StateOnly,
                        OutputDemand::LastPosition,
                        OutputDemand::Sequence,
                    ] {
                        let context = WorkspaceContext::new(EquationMechanism {
                            omit_attention: false,
                        });
                        let mut args = super::blockwise::args(low_rank, false, packed);
                        // KDA's low-rank gate projections consume head_dim;
                        // retain complete 32-column affine groups there too.
                        args.kda_config.head_dim = 32;
                        let architecture =
                            kimi_linear::LayeredModel::<WorkspaceBackend>::new(args, &context)
                                .unwrap();
                        let layout = architecture.state_layout().unwrap();
                        let units = (0..2)
                            .map(|i| architecture.construct_unit(0, i, &context).unwrap())
                            .collect();
                        let mut runtime =
                            LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                        total += run(
                            &context,
                            layout,
                            step,
                            chunk,
                            demand,
                            |tokens, state, pass, demand| {
                                runtime
                                    .forward_with_provider_and_observer_and_context_with_readout(
                                        decoder::LayeredInput { tokens, mask: None },
                                        state,
                                        pass,
                                        &mut ResidentExpertProvider,
                                        &context,
                                        &mut NoopObserver,
                                        demand,
                                    )
                                    .map(|(scores, _)| scores)
                                    .map_err(Error::backend)
                            },
                        );
                        trajectories += 1;
                    }
                }
            }
        }
    }
    assert_eq!((trajectories, total), (72, 480));
}

fn v3_args(packed: bool, low_rank: bool) -> deepseek::V3Args {
    let mut value = serde_json::json!({
        "model_type":"deepseek_v3","vocab_size":37,"hidden_size":64,
        "intermediate_size":64,"moe_intermediate_size":64,"num_hidden_layers":2,
        "num_attention_heads":4,"max_position_embeddings":128,
        "q_lora_rank":if low_rank { Some(32) } else { None },"kv_lora_rank":32,"qk_nope_head_dim":16,
        "qk_rope_head_dim":16,"v_head_dim":16,"first_k_dense_replace":1,
        "n_routed_experts":4,"n_shared_experts":1,"num_experts_per_tok":2,
        "n_group":2,"topk_group":1,"num_nextn_predict_layers":0,"tie_word_embeddings":false
    });
    if packed {
        value["quantization"] = serde_json::json!({"group_size":32,"bits":4});
    }
    deepseek::parse_v3_config(&value).unwrap()
}

#[test]
fn complete_deepseek_v3_traces_capacity_backed_latents_and_routed_layers() {
    let mut total = 0;
    let mut trajectories = 0;
    for packed in [false, true] {
        for low_rank in [false, true] {
            for step in [4, 256] {
                for chunk in [1, 3, 7] {
                    for demand in [
                        OutputDemand::StateOnly,
                        OutputDemand::LastPosition,
                        OutputDemand::Sequence,
                    ] {
                        let context = WorkspaceContext::new(EquationMechanism {
                            omit_attention: false,
                        });
                        let args = v3_args(packed, low_rank);
                        let layout = deepseek::v3::state_layout(&args).unwrap();
                        let architecture =
                            deepseek::v3::Model::<WorkspaceBackend>::new(args, &context).unwrap();
                        let units = (0..2)
                            .map(|i| architecture.construct_unit(0, i, &context).unwrap())
                            .collect();
                        let mut runtime =
                            LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                        total += run(
                            &context,
                            layout,
                            step,
                            chunk,
                            demand,
                            |tokens, state, pass, demand| {
                                runtime
                                    .forward_with_provider_and_observer_and_context_with_readout(
                                        deepseek::mtp::EmbeddedInput::Target { tokens, mask: None },
                                        state,
                                        pass,
                                        &mut ResidentExpertProvider,
                                        &context,
                                        &mut NoopObserver,
                                        demand,
                                    )
                                    .map(|(scores, _)| scores)
                                    .map_err(Error::backend)
                            },
                        );
                        trajectories += 1;
                    }
                }
            }
        }
    }
    assert_eq!((trajectories, total), (72, 480));
}
