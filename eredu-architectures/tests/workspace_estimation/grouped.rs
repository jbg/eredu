use super::*;
use eredu_runtime::{
    DeviceState, ExpertPass, LayerRuntimeState, LayerwiseRuntime, NoopObserver,
    ResidentExpertProvider, ResidentUnitWindow, RuntimeLayerState,
};

#[derive(Debug)]
struct RoutedMechanism {
    omit_grouped: bool,
}
impl WorkspaceMechanisms for RoutedMechanism {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::memory_fixture::topology())
    }
    fn output_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory_fixture::placement())
    }
    fn scratch_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory_fixture::placement())
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "test mechanism has no disjoint host workspace".into(),
        }))
    }

    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if self.omit_grouped && matches!(op.kind, WorkspaceOperationKind::Grouped { .. }) {
            return Ok(None);
        }
        EquationMechanism {
            omit_attention: false,
        }
        .operation_bound(op)
    }
}
fn args(tied: bool, packed: bool) -> qwen::ModelArgs {
    let mut configuration = config("qwen3_moe", tied, packed);
    configuration["hidden_size"] = 64.into();
    configuration["head_dim"] = 16.into();
    configuration["num_experts"] = 4.into();
    configuration["num_experts_per_tok"] = 2.into();
    configuration["moe_intermediate_size"] = 64.into();
    configuration["norm_topk_prob"] = true.into();
    qwen::model_args_from_config_value(&configuration).unwrap()
}
fn run(
    tied: bool,
    packed: bool,
    chunk: usize,
    demand: OutputDemand,
    omit_grouped: bool,
) -> Vec<WorkspaceTraceReport> {
    let args = args(tied, packed);
    let context = WorkspaceContext::new(RoutedMechanism { omit_grouped });
    let layout = decoder::state_layout(&args).unwrap();
    let architecture = qwen::RoutedLayeredModel::<WorkspaceBackend>::new(args, &context).unwrap();
    let units = (0..2)
        .map(|i| architecture.construct_unit(i, &context).unwrap())
        .collect::<Vec<_>>();
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let mut state = DeviceState::<WorkspaceBackend, _>::create(layout, |_, _| {
        Ok::<_, Error>(AppendCache {
            keys: Some(WorkspaceTensor::unloaded_f32(&[2, 2, 2, 16], &context)?),
            values: Some(WorkspaceTensor::unloaded_f32(&[2, 2, 2, 16], &context)?),
            offset: 2,
        })
    })
    .unwrap();
    let mut spans = (0..7)
        .step_by(chunk)
        .map(|start| {
            let count = chunk.min(7 - start);
            (
                count,
                demand.for_chunk(start + count == 7),
                ExpertPass::Prefill,
            )
        })
        .collect::<Vec<_>>();
    spans.extend([(1, OutputDemand::LastPosition, ExpertPass::Decode); 3]);
    let mut reports = Vec::new();
    let mut offset = 2;
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
            .map(|i| ((offset + 3 * i) % 37) as i32)
            .collect::<Vec<_>>();
        let tokens = WorkspaceTensor::from_i32_slice(&ids, &[2, count as i32], &context).unwrap();
        let (scores, _) = runtime
            .forward_with_provider_and_observer_and_context_with_readout(
                decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
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
            let cache = state.layer(layer).unwrap();
            assert_eq!(cache.offset, offset as i32);
            for value in cache.retained_values() {
                assert_eq!(value.shape(), [2, 2, offset as i32, 16]);
                roots.push(value.clone());
            }
        }
        let report = context.report(&roots).unwrap();
        assert_eq!(report.total_bytes.is_some(), !omit_grouped);
        let routers = report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::GroupSelection { .. }))
            .collect::<Vec<_>>();
        let groups = report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
            .collect::<Vec<_>>();
        assert_eq!((routers.len(), groups.len()), (2, 2));
        for router in routers {
            assert_eq!(router.outputs[0].shape(), [2 * count as i32, 2]);
        }
        for op in groups {
            assert_eq!(op.outputs[0].shape(), [2, count as i32, 64]);
            assert_eq!(
                op.inputs[4].shape(),
                if packed {
                    vec![4, 128, 8]
                } else {
                    vec![4, 128, 64]
                }
            );
            assert!(
                matches!(&op.kind,WorkspaceOperationKind::Grouped{bank,phase:WorkspaceGroupedPhase::Whole,partitions:None} if matches!(bank.as_ref(),WorkspaceGroupedBank::GatedProduct(spec) if spec.group_count()==4 && spec.intermediate_dimensions()==64))
            );
        }
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
        if let Some(op) = readouts.first() {
            assert_eq!(
                op.inputs[0].shape()[1],
                demand.positions(count as u64) as i32
            );
        }
        if omit_grouped {
            assert_eq!(report.unpriced_operations.len(), 2);
            assert_eq!(report.transient_bytes, None);
        } else {
            assert_eq!(
                report.retained_bytes,
                Some(2 * 2 * 2 * 2 * offset as u64 * 16 * 4)
            );
        }
        reports.push(report);
    }
    reports
}
#[test]
fn routed_qwen_existing_provider_traces_all_chunks_readouts_and_cached_decodes() {
    for tied in [false, true] {
        for packed in [false, true] {
            for chunk in [1, 3, 7] {
                for demand in [
                    OutputDemand::StateOnly,
                    OutputDemand::LastPosition,
                    OutputDemand::Sequence,
                ] {
                    assert_eq!(
                        run(tied, packed, chunk, demand, false).len(),
                        7_usize.div_ceil(chunk) + 3
                    );
                }
            }
        }
    }
}
#[test]
fn missing_grouped_native_facts_remain_unknown_after_the_full_routed_model() {
    for chunk in [1, 3, 7] {
        run(true, false, chunk, OutputDemand::LastPosition, true);
    }
}

#[test]
fn tensor_parallel_routed_provider_retains_global_router_and_local_expert_width() {
    for packed in [false, true] {
        for partitions in [1, 2] {
            for rank in 0..partitions {
                let context = WorkspaceContext::new(RoutedMechanism {
                    omit_grouped: false,
                });
                let global = args(true, packed);
                let mut local = global.clone();
                local.moe_intermediate_size /= partitions as i32;
                let mut mlp = qwen::FeedForward::<WorkspaceBackend>::new_partitioned(
                    &global, &local, 0, &context,
                )
                .unwrap();
                let parallel = WorkspaceParallelContext::new(rank, partitions).unwrap();
                for positions in [3, 1] {
                    context.begin_span();
                    let input =
                        WorkspaceTensor::unloaded_f32(&[2, positions, 64], &context).unwrap();
                    let output = mlp
                        .forward_with_provider_parallel(
                            0,
                            if positions > 1 {
                                ExpertPass::Prefill
                            } else {
                                ExpertPass::Decode
                            },
                            &input,
                            &parallel,
                            &context,
                            &mut ResidentExpertProvider,
                        )
                        .unwrap();
                    assert_eq!(output.shape(), input.shape());
                    let report = context.report(&[]).unwrap();
                    let groups = report
                        .operations
                        .iter()
                        .find(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
                        .unwrap();
                    assert!(
                        matches!(&groups.kind,WorkspaceOperationKind::Grouped{bank,partitions:Some(n),..} if *n==partitions && matches!(bank.as_ref(),WorkspaceGroupedBank::GatedProduct(s) if s.group_count()==4 && s.intermediate_dimensions()==64/partitions as i32))
                    );
                    assert_eq!(
                        groups.inputs[4].shape(),
                        if packed {
                            vec![4, 128 / partitions as i32, 8]
                        } else {
                            vec![4, 128 / partitions as i32, 64]
                        }
                    );
                    assert!(
                        matches!(&report.operations.last().unwrap().kind,WorkspaceOperationKind::Collective(WorkspaceCollective::Sum{partitions:n,rank:r}) if *n==partitions && *r==rank)
                    );
                }
            }
        }
    }
}
