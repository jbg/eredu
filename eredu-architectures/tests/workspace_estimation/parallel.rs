use super::*;
use eredu_runtime::{
    DeviceState, LayerRuntimeState, LayeredTraversalHook, LayerwiseRuntime, LocalModelLayout,
    LocalTensorLayout, ParameterRole, ResidentUnitWindow, RuntimeLayerState, TensorPlacement,
};

impl RuntimeLayerState<WorkspaceBackend> for AppendCache {
    type RetainedValues<'a> = std::iter::Chain<
        std::option::Iter<'a, WorkspaceTensor>,
        std::option::Iter<'a, WorkspaceTensor>,
    >;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.keys.iter().chain(self.values.iter())
    }
}

struct Traversal;
impl<C> LayeredTraversalHook<WorkspaceBackend, C, Error> for Traversal {}

// Fixture planner outputs. The architecture itself derives local modules and
// state from these tensor placements; the backend receives no family policy.
fn layout(rank: usize, partitions: usize, tied: bool) -> LocalModelLayout {
    let mut layout = LocalModelLayout::default();
    let mut insert = |name: &str, rows: usize, columns: usize, role: ParameterRole| {
        let base = rows / partitions;
        let remainder = rows % partitions;
        let start = rank * base + rank.min(remainder);
        let end = start + base + usize::from(rank < remainder);
        layout.insert(
            format!("{name}.weight"),
            LocalTensorLayout::new(
                name,
                role,
                vec![rows, columns],
                vec![end - start, columns],
                TensorPlacement::Range {
                    axis: 0,
                    start,
                    end,
                },
                None,
                Some(start..end),
                false,
            ),
        );
    };
    insert("model.embed_tokens", 37, 64, ParameterRole::Vocabulary);
    if !tied {
        insert("lm_head", 37, 64, ParameterRole::Vocabulary);
    }
    for layer in 0..2 {
        insert(
            &format!("model.layers.{layer}.self_attn.q_proj"),
            64,
            64,
            ParameterRole::AttentionHeads,
        );
        insert(
            &format!("model.layers.{layer}.self_attn.k_proj"),
            32,
            64,
            ParameterRole::AttentionHeads,
        );
        insert(
            &format!("model.layers.{layer}.mlp.gate_proj"),
            64,
            64,
            ParameterRole::FeedForwardIntermediate,
        );
    }
    layout
}

fn run(
    tied: bool,
    packed: bool,
    rank: usize,
    partitions: usize,
    chunk: usize,
    demand: OutputDemand,
) -> Vec<WorkspaceTraceReport> {
    let mut configuration = config("llama", tied, packed);
    // TP=2 leaves 32 local attention-output columns, preserving complete
    // published 32-column affine quantization groups.
    configuration["hidden_size"] = 64.into();
    configuration["head_dim"] = 16.into();
    let args = llama::model_args_from_config_value(&configuration).unwrap();
    let context = WorkspaceContext::new(EquationMechanism {
        omit_attention: false,
    });
    let geometry = llama::local_geometry(&args, &layout(rank, partitions, tied)).unwrap();
    let heads = geometry.block(0).unwrap().num_key_value_heads;
    let state_layout = geometry.state_layout().clone();
    let architecture =
        llama::LayeredModel::<WorkspaceBackend>::new_parallel(args, geometry, &context).unwrap();
    let units = (0..2)
        .map(|layer| architecture.construct_unit(layer, &context).unwrap())
        .collect::<Vec<_>>();
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let parallel = WorkspaceParallelContext::new(rank, partitions).unwrap();
    let mut state = DeviceState::<WorkspaceBackend, _>::create(state_layout, |_, _| {
        Ok::<_, Error>(AppendCache {
            keys: Some(WorkspaceTensor::unloaded_f32(&[2, heads, 2, 16], &context)?),
            values: Some(WorkspaceTensor::unloaded_f32(&[2, heads, 2, 16], &context)?),
            offset: 2,
        })
    })
    .unwrap();
    let mut spans = (0..7)
        .step_by(chunk)
        .map(|start| {
            let count = chunk.min(7 - start);
            (count, demand.for_chunk(start + count == 7))
        })
        .collect::<Vec<_>>();
    spans.extend([(1, OutputDemand::LastPosition); 3]);
    let mut reports = Vec::new();
    let mut expected_offset = 2;
    for (count, demand) in spans {
        context
            .begin_state_span(
                state
                    .as_ref()
                    .iter()
                    .flat_map(|layer| layer.retained_values()),
            )
            .unwrap();
        let ids = (0..2 * count)
            .map(|i| ((expected_offset + 3 * i) % 37) as i32)
            .collect::<Vec<_>>();
        let tokens = WorkspaceTensor::from_i32_slice(&ids, &[2, count as i32], &context).unwrap();
        let (scores, _) = runtime
            .forward_parallel_with_traversal_hook_with_readout(
                decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                &mut state,
                &parallel,
                &context,
                &mut Traversal,
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
        expected_offset += count;
        let mut roots = Vec::new();
        for layer in 0..2 {
            let cache = state.layer(layer).unwrap();
            assert_eq!(cache.offset, expected_offset as i32);
            for value in cache.retained_values() {
                assert_eq!(value.shape(), [2, heads, expected_offset as i32, 16]);
                roots.push(value.clone());
            }
        }
        let report = context.report(&roots).unwrap();
        assert!(report.total_bytes.is_some());
        assert_eq!(
            report.retained_bytes,
            Some(2 * 2 * 2 * heads as u64 * expected_offset as u64 * 16 * 4)
        );
        let local_vocabulary = 37 / partitions + usize::from(rank < 37 % partitions);
        let readout_projections = report
            .operations
            .iter()
            .filter(|op| {
                matches!(op.kind, WorkspaceOperationKind::Projection(_))
                    && op.outputs[0].shape().last() == Some(&(local_vocabulary as i32))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            readout_projections.len(),
            usize::from(demand != OutputDemand::StateOnly)
        );
        if let Some(projection) = readout_projections.first() {
            assert_eq!(
                projection.inputs[0].shape()[1],
                demand.positions(count as u64) as i32
            );
            assert!(
                matches!(&projection.kind, WorkspaceOperationKind::Projection(format)
                if matches!(format.encoding(), eredu_checkpoint::LinearFormat::Affine(_)) == packed)
            );
        }
        let gathers = report
            .operations
            .iter()
            .filter(|op| {
                matches!(
                    op.kind,
                    WorkspaceOperationKind::Collective(WorkspaceCollective::GatherFirstAxis { .. })
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            gathers.len(),
            usize::from(demand != OutputDemand::StateOnly)
        );
        if let Some(gather) = gathers.first() {
            assert_eq!(
                gather.inputs[0].shape()[1],
                demand.positions(count as u64) as i32
            );
            let widths = if partitions == 1 {
                vec![37]
            } else {
                vec![19, 18]
            };
            assert_eq!(gather.inputs[0].shape()[2], *widths.iter().max().unwrap());
            assert!(
                matches!(&gather.kind, WorkspaceOperationKind::Collective(WorkspaceCollective::GatherFirstAxis { axis:2, rank:r, peer_widths }) if *r==rank && *peer_widths==widths.iter().map(|n|*n as usize).collect::<Vec<_>>())
            );
            let gather_index=report.operations.iter().position(|op|std::ptr::eq(op,*gather)).unwrap();
            let projection=report.operations[..gather_index].iter().rev()
                .find(|op|matches!(op.kind,WorkspaceOperationKind::Projection(_))).unwrap();
            assert!(matches!(
                projection.kind,
                WorkspaceOperationKind::Projection(_)
            ));
            assert_eq!(
                projection.inputs[0].shape()[1],
                demand.positions(count as u64) as i32
            );
        }
        // Four block reductions plus the vocabulary lookup contribution are
        // explicit collectives in the same native equation order.
        assert_eq!(report.operations.iter().filter(|op| matches!(op.kind,
            WorkspaceOperationKind::Collective(WorkspaceCollective::Sum{partitions:n,rank:r})
                if n==partitions&&r==rank)).count(), 5);
        reports.push(report);
    }
    reports
}

#[test]
fn parallel_decoder_equations_trace_uneven_vocabulary_chunked_prefill_and_cached_decode() {
    for tied in [false, true] {
        for packed in [false, true] {
            for partitions in [1, 2] {
                for rank in 0..partitions {
                    for chunk in [1, 3, 7] {
                        for demand in [
                            OutputDemand::StateOnly,
                            OutputDemand::LastPosition,
                            OutputDemand::Sequence,
                        ] {
                            let reports = run(tied, packed, rank, partitions, chunk, demand);
                            assert_eq!(reports.len(), 7_usize.div_ceil(chunk) + 3);
                        }
                    }
                }
            }
        }
    }
}
