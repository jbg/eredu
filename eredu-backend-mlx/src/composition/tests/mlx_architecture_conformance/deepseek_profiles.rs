#[test]
fn neutral_deepseek_v3_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::deepseek::v3::Model<MlxNeuralBackend>;
    type State = DeviceState<MlxNeuralBackend, CompressedLatentCache>;
    let args = eredu_architectures::deepseek::parse_v3_config(&serde_json::json!({
        "hidden_size":8,"intermediate_size":16,"moe_intermediate_size":8,
        "num_hidden_layers":2,"num_attention_heads":2,"vocab_size":31,
        "max_position_embeddings":64,"kv_lora_rank":4,"qk_nope_head_dim":2,
        "qk_rope_head_dim":2,"v_head_dim":2,"first_k_dense_replace":1,
        "n_routed_experts":4,"n_shared_experts":1,"num_experts_per_tok":2,
        "n_group":2,"topk_group":1,"num_nextn_predict_layers":1,
        "tie_word_embeddings":false
    }))
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state = State::create(
        eredu_architectures::deepseek::v3::state_layout(&args).unwrap(),
        |_, _| Ok::<_, std::convert::Infallible>(CompressedLatentCache::new()),
    )
    .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        State,
        architecture,
        state,
        eredu_architectures::deepseek::mtp::EmbeddedInput::target(&tokens, None),
        &[1, 2, 31],
        stream
    );
}

#[test]
fn neutral_deepseek_v4_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::deepseek::v4::Model<MlxNeuralBackend>;
    type State = crate::backend::runtime::cache::state::MlxPoolingAttentionState;
    let args = eredu_architectures::deepseek::parse_v4_config(&serde_json::json!({
        "hidden_size":8,"moe_intermediate_size":8,"num_hidden_layers":3,
        "num_attention_heads":2,"head_dim":4,"qk_rope_head_dim":2,
        "q_lora_rank":4,"o_lora_rank":2,"o_groups":2,"vocab_size":31,
        "max_position_embeddings":64,"sliding_window":8,
        "compress_ratios":[0,4,128,0],"index_n_heads":2,"index_head_dim":4,
        "index_topk":1,"hc_mult":2,"hc_sinkhorn_iters":2,
        "n_routed_experts":4,"num_experts_per_tok":2,
        "scoring_func":"sqrtsoftplus","topk_method":"noaux_tc",
        "norm_topk_prob":true,"num_nextn_predict_layers":1
    }))
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state = MlxPoolingAttentionStateFactory::device(
        eredu_architectures::deepseek::v4::state_layout(&args).unwrap(),
    )
    .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        State,
        architecture,
        state,
        eredu_architectures::deepseek::mtp::EmbeddedInput::target(&tokens, None),
        &[1, 2, 31],
        stream
    );
}

#[test]
fn deepseek_unit_factories_use_installed_expert_realizations() {
    use eredu_nn::GroupedGatedProductOperator;

    let execution = mlx_execution();
    let stream = execution.stream();
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, 2, 1).unwrap(),
        0,
    )
    .unwrap();

    let v3_args = eredu_architectures::deepseek::parse_v3_config(&serde_json::json!({
        "hidden_size":8,"intermediate_size":16,"moe_intermediate_size":8,
        "num_hidden_layers":2,"num_attention_heads":2,"vocab_size":31,
        "max_position_embeddings":64,"kv_lora_rank":4,"qk_nope_head_dim":2,
        "qk_rope_head_dim":2,"v_head_dim":2,"first_k_dense_replace":1,
        "n_routed_experts":4,"n_shared_experts":1,"num_experts_per_tok":2,
        "n_group":2,"topk_group":1,"num_nextn_predict_layers":1,
        "tie_word_embeddings":false
    }))
    .unwrap();
    let expected_v3_target = eredu_architectures::deepseek::v3::expert_bank_spec(&v3_args, 1)
        .unwrap()
        .with_group_geometry(2, 3)
        .unwrap();
    let expected_v3_mtp = eredu_architectures::deepseek::v3::expert_bank_spec(&v3_args, 2)
        .unwrap()
        .with_group_geometry(2, 3)
        .unwrap();
    let v3_plan = eredu_architectures::ExpertRealizationPlan::balanced(
        4,
        topology,
        std::collections::BTreeMap::from([
            (
                (eredu_runtime::ExecutionGroupId::new("target").unwrap(), 1),
                expected_v3_target.clone(),
            ),
            (
                (eredu_runtime::ExecutionGroupId::new("mtp.0").unwrap(), 0),
                expected_v3_mtp.clone(),
            ),
        ]),
    )
    .unwrap();
    let mut v3 =
        eredu_architectures::deepseek::v3::Model::<MlxNeuralBackend>::new(v3_args, stream).unwrap();
    v3.install_expert_realization(v3_plan);
    let v3_target = v3.construct_unit(0, 1, stream).unwrap();
    let eredu_architectures::deepseek::v3::Unit::Target(v3_target) = v3_target else {
        panic!("V3 target factory returned a prediction unit")
    };
    let eredu_architectures::deepseek::block::V3FeedForward::Routed(v3_target) =
        v3_target.feed_forward
    else {
        panic!("V3 sparse target factory returned a dense MLP")
    };
    assert_eq!(
        v3_target.experts.spec().group_count(),
        expected_v3_target.group_count()
    );
    assert_eq!(
        v3_target.experts.spec().intermediate_dimensions(),
        expected_v3_target.intermediate_dimensions()
    );
    let v3_mtp = v3.construct_unit(1, 0, stream).unwrap();
    assert_eq!(
        v3_mtp.expert_bank_spec().unwrap().group_count(),
        expected_v3_mtp.group_count()
    );

    let v4_args = eredu_architectures::deepseek::parse_v4_config(&serde_json::json!({
        "hidden_size":8,"moe_intermediate_size":8,"num_hidden_layers":3,
        "num_attention_heads":2,"head_dim":4,"qk_rope_head_dim":2,
        "q_lora_rank":4,"o_lora_rank":2,"o_groups":2,"vocab_size":31,
        "max_position_embeddings":64,"sliding_window":8,
        "compress_ratios":[0,4,128,0],"index_n_heads":2,"index_head_dim":4,
        "index_topk":1,"hc_mult":2,"hc_sinkhorn_iters":2,
        "n_routed_experts":4,"num_experts_per_tok":2,
        "scoring_func":"sqrtsoftplus","topk_method":"noaux_tc",
        "norm_topk_prob":true,"num_nextn_predict_layers":1
    }))
    .unwrap();
    let expected_v4 = eredu_architectures::deepseek::v4::expert_bank_spec(&v4_args, 0)
        .unwrap()
        .with_group_geometry(2, 3)
        .unwrap();
    let v4_plan = eredu_architectures::ExpertRealizationPlan::balanced(
        4,
        topology,
        std::collections::BTreeMap::from([(
            (eredu_runtime::ExecutionGroupId::new("target").unwrap(), 0),
            expected_v4.clone(),
        )]),
    )
    .unwrap();
    let mut v4 =
        eredu_architectures::deepseek::v4::Model::<MlxNeuralBackend>::new(v4_args, stream).unwrap();
    v4.install_expert_realization(v4_plan);
    let v4_target = v4.construct_unit(0, 0, stream).unwrap();
    let eredu_architectures::deepseek::v4::Unit::Target(v4_target) = v4_target else {
        panic!("V4 target factory returned a prediction unit")
    };
    assert_eq!(
        v4_target.feed_forward.experts.spec().group_count(),
        expected_v4.group_count()
    );
    assert_eq!(
        v4_target
            .feed_forward
            .experts
            .spec()
            .intermediate_dimensions(),
        expected_v4.intermediate_dimensions()
    );
}
