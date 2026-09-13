//! Cold resource totals compared with actual prediction-module tensor geometry.
use super::*;

#[test]
fn prediction_rank_resources_count_tensor_shards_and_pipeline_expert_replicas() {
    let config = serde_json::json!({
        "model_type":"deepseek_v3", "hidden_size":8, "vocab_size":16,
        "num_hidden_layers":2, "num_attention_heads":2, "intermediate_size":10,
        "moe_intermediate_size":4, "q_lora_rank":4, "kv_lora_rank":4,
        "qk_nope_head_dim":2, "qk_rope_head_dim":2, "v_head_dim":4,
        "first_k_dense_replace":0, "n_routed_experts":2, "n_shared_experts":1,
        "num_experts_per_tok":1, "n_group":1, "topk_group":1,
        "max_position_embeddings":64, "num_nextn_predict_layers":2
    });
    let args = deepseek::parse_v3_config(&config).unwrap();
    let parameters = deepseek::parallel::v3_parameter_description(&args).unwrap();
    let (root, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let context = NumericContext::default();
    struct Bytes(u64);
    impl<'a> ParameterVisitor<'a, NumericTensor> for Bytes {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a NumericTensor) {
            if metadata.alias_of.is_none() {
                self.0 += value.shape.iter().map(|&n| n as u64).product::<u64>() * 4;
            }
        }
    }
    for (tp, pp, ep) in [(2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2)] {
        let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
        for rank in 0..topology.world_size() {
            let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
            let parallel = eredu_runtime::ParallelLoadRequest::new(
                rank_topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                8,
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(2),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            )
            .unwrap();
            let select = |drafting| {
                let plan = prepared_adapter::plan(None)
                    .with_topology(topology)
                    .with_drafting(drafting);
                let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
                    &plan,
                    eredu_runtime::ResidencyDiagnostics::new(false, false),
                    Some(parallel.clone()),
                )
                .unwrap();
                eredu_architectures::select_preparation(&inspection, &request, &Provider).unwrap()
            };
            let enabled = select(eredu_core::DraftingPlan::Embedded {
                max_draft_tokens: 1,
                lookahead: false,
                adaptive_lookahead: false,
            });
            let disabled = select(eredu_core::DraftingPlan::Disabled);
            let tensor_topology = ParallelRankTopology::new(
                ParallelTopology::new(tp, 1, 1, 1).unwrap(),
                rank_topology.tensor_parallel_rank(),
            )
            .unwrap();
            let layout =
                eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                    &parameters,
                    tensor_topology,
                )
                .unwrap();
            let model = deepseek::v3::Model::<NumericBackend>::new_parallel(
                args.clone(),
                deepseek::parallel::v3_local_geometry(&args, &layout).unwrap(),
                &context,
            )
            .unwrap();
            let module_bytes = (1..=2)
                .map(|group| {
                    let unit = model.construct_unit(group, 0, &context).unwrap();
                    let mut bytes = Bytes(0);
                    unit.visit_parameters(&mut bytes);
                    bytes.0
                })
                .collect::<Vec<_>>();
            let resources = |selection: &eredu_architectures::SelectedPreparation| {
                selection
                    .execution()
                    .partition_parameter_resources(&parameters)
                    .unwrap()
                    .unwrap()
                    .0
            };
            let ordinary = resources(&disabled);
            let complete = resources(&enabled);
            // Prediction consumes original-token embeddings on every pipeline
            // stage, so enabling it also changes target static ownership.
            let mut replicated_embedding = Bytes(0);
            if rank_topology.pipeline_parallel_rank() != 0 {
                model
                    .static_modules()
                    .embeddings
                    .visit_parameters(&mut replicated_embedding);
            }
            assert_eq!(complete.parameter_bytes - ordinary.parameter_bytes, module_bytes.iter().sum::<u64>() + replicated_embedding.0, "{tp}/{pp}/{ep} rank {rank}: every rank owns its full tensor-sharded prediction schedule");
            assert_eq!(
                complete.pinned_bytes - ordinary.pinned_bytes,
                replicated_embedding.0
            );
            assert_eq!(
                complete.largest_unit_bytes,
                ordinary
                    .largest_unit_bytes
                    .max(*module_bytes.iter().max().unwrap())
            );
            assert_eq!(
                complete.largest_adjacent_units_bytes,
                ordinary
                    .largest_adjacent_units_bytes
                    .max(*module_bytes.iter().max().unwrap())
            );
        }
    }
}
