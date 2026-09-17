use super::*;
use eredu_architectures::kimi_linear;
use eredu_nn::{
    CompressedAttentionBlock, CompressedAttentionCache, CompressedAttentionScan,
    CompressedAttentionState, CompressedAttentionView, Index,
};

// A deliberately specified fixture mechanism: immutable blocks alias appended
// metadata tensors. This is not a description of native compressed cache
// capacity, paging transfers, or isolated snapshot copies.
#[derive(Debug, Clone, Default)]
struct Blocks {
    blocks: Vec<CompressedAttentionBlock<WorkspaceTensor>>,
    offset: i32,
    last_scan: CompressedAttentionScan,
}
impl CompressedAttentionCache<WorkspaceTensor> for Blocks {
    type Checkpoint = Self;
    fn offset(&self) -> i32 {
        self.offset
    }
    fn is_paged(&self) -> bool {
        true
    }
    fn append(
        &mut self,
        state: CompressedAttentionState<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<CompressedAttentionView<WorkspaceTensor>, Error> {
        let tokens = state.latent.shape()[1];
        for start in (0..tokens).step_by(3) {
            let end = (start + 3).min(tokens);
            self.blocks.push(CompressedAttentionBlock {
                start: (self.offset + start) as i64,
                end: (self.offset + end) as i64,
                state: CompressedAttentionState {
                    latent: state
                        .latent
                        .index(&[Index::Full, Index::Range(start, end)], context)?,
                    rotary: state
                        .rotary
                        .index(&[Index::Full, Index::Range(start, end)], context)?,
                },
            });
        }
        self.offset += tokens;
        Ok(CompressedAttentionView::Paged { appended: state })
    }
    fn visit_blocks<F>(
        &mut self,
        _: i32,
        _: &WorkspaceContext,
        mut visitor: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<WorkspaceTensor>) -> Result<u64, Error>,
    {
        let mut scan = CompressedAttentionScan::default();
        for block in &self.blocks {
            scan.bytes +=
                block.state.latent.layout().bytes()? + block.state.rotary.layout().bytes()?;
            scan.reconstruction_scratch_bytes = scan
                .reconstruction_scratch_bytes
                .max(visitor(block.clone())?);
            scan.blocks += 1;
        }
        self.last_scan = scan;
        Ok(scan)
    }
    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }
    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        context.validate_values(
            checkpoint
                .blocks
                .iter()
                .flat_map(|b| [&b.state.latent, &b.state.rotary]),
        )?;
        self.clone_from(checkpoint);
        Ok(())
    }
    fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn clear(&mut self) -> Result<(), Error> {
        *self = Self::default();
        Ok(())
    }
}

pub(super) fn args(low_rank: bool, split: bool, packed: bool) -> kimi_linear::ModelArgs {
    let mut value = serde_json::json!({
        "model_type":"kimi_linear","vocab_size":37,"hidden_size":64,
        "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":4,
        "intermediate_size":64,"head_dim":16,"model_max_length":128,
        "linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":4,"head_dim":16,"short_conv_kernel_size":3},
        "num_experts":2,"moe_intermediate_size":64,"kv_lora_rank":32,
        "q_lora_rank":if low_rank { Some(32) } else { None },
        "qk_nope_head_dim":16,"qk_rope_head_dim":8,"v_head_dim":16,
        "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,
        "routed_scaling_factor":1.0,"first_k_dense_replace":1,
        "num_expert_group":1,"topk_group":1,"tie_word_embeddings":false
    });
    if packed {
        value["quantization"] = serde_json::json!({"group_size":32,"bits":4});
    }
    let mut args = kimi_linear::model_args_from_config_value(&value).unwrap();
    args.split_kv_b = split;
    args
}

#[test]
fn latent_attention_equations_reconstruct_bounded_blocks_with_complete_absolute_coordinates() {
    let mut trajectories = 0;
    let mut spans = 0;
    for low_rank in [false, true] {
        for split in [false, true] {
            for packed in [false, true] {
                for chunk in [1, 3, 7] {
                    for explicit_mask in [false, true] {
                        let context = WorkspaceContext::new(EquationMechanism {
                            omit_attention: false,
                        });
                        let mut module = kimi_linear::KimiLatentAttention::<WorkspaceBackend>::new(
                            &args(low_rank, split, packed),
                            1,
                            &context,
                        )
                        .unwrap();
                        let mut state = Blocks::default();
                        let counts = std::iter::once(2)
                            .chain((0..7).step_by(chunk).map(|start| chunk.min(7 - start)))
                            .chain([1; 3]);
                        for (span, count) in counts.enumerate() {
                            context
                                .begin_state_span(
                                    state.blocks.iter().flat_map(|block| {
                                        [&block.state.latent, &block.state.rotary]
                                    }),
                                )
                                .unwrap();
                            let offset = state.offset;
                            let input = WorkspaceTensor::from_f32_slice(
                                &vec![0.25; 2 * count * 64],
                                &[2, count as i32, 64],
                                &context,
                            )
                            .unwrap();
                            let mask = if explicit_mask {
                                Some(
                                    WorkspaceBackend::causal_mask(
                                        count as i32,
                                        offset,
                                        None,
                                        &context,
                                    )
                                    .unwrap(),
                                )
                            } else {
                                None
                            };
                            let output = module
                                .forward(&input, mask.as_ref(), Some(&mut state), &context)
                                .unwrap();
                            assert_eq!(output.shape(), [2, count as i32, 64]);
                            assert_eq!(state.offset, offset + count as i32);
                            let roots = state
                                .blocks
                                .iter()
                                .flat_map(|b| [b.state.latent.clone(), b.state.rotary.clone()])
                                .collect::<Vec<_>>();
                            let report = context.report(&roots).unwrap();
                            assert!(report.total_bytes.is_some());
                            assert!(report.retained_bytes.unwrap() > 0);
                            let stages = report
                                .operations
                                .iter()
                                .filter_map(|op| match &op.kind {
                                    WorkspaceOperationKind::BlockwiseAttention {
                                        policy,
                                        stage,
                                    } => Some((op, policy, stage)),
                                    _ => None,
                                })
                                .collect::<Vec<_>>();
                            assert_eq!(stages.len(), state.blocks.len() + 2);
                            assert_eq!(*stages[0].2, WorkspaceBlockwiseStage::Begin);
                            assert_eq!(*stages.last().unwrap().2, WorkspaceBlockwiseStage::Finish);
                            let mut frontier = 0;
                            let mut maximum_reconstruction = 0;
                            for (op, policy, stage) in stages {
                                assert_eq!(
                                    (policy.query_start, policy.context_end),
                                    (offset as i64, state.offset as i64)
                                );
                                assert_eq!(policy.mask_origin, explicit_mask.then_some(0));
                                if let WorkspaceBlockwiseStage::Accumulate {
                                    start,
                                    end,
                                    previous,
                                    ..
                                } = stage
                                {
                                    assert_eq!(*start, frontier);
                                    assert!(*end - *start <= 3);
                                    assert_eq!(*previous, *start != 0);
                                    assert_eq!(
                                        op.inputs[1].shape(),
                                        [2, 4, (*end - *start) as i32, 24]
                                    );
                                    assert_eq!(
                                        op.inputs[2].shape(),
                                        [2, 4, (*end - *start) as i32, 16]
                                    );
                                    maximum_reconstruction = maximum_reconstruction
                                        .max(2 * 4 * (*end - *start) as u64 * (24 + 16) * 4);
                                    frontier = *end;
                                }
                            }
                            assert_eq!(frontier, state.offset as i64);
                            assert_eq!(
                                state.last_scan.reconstruction_scratch_bytes,
                                maximum_reconstruction
                            );
                            if span != 0 {
                                spans += 1;
                            }
                        }
                        trajectories += 1;
                    }
                }
            }
        }
    }
    assert_eq!((trajectories, spans), (48, 320));
}
