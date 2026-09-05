fn deepseek_gguf_metadata() -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("deepseek2".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        ("deepseek2.block_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "deepseek2.context_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        (
            "deepseek2.embedding_length".into(),
            GgufMetadataValue::Uint32(12),
        ),
        (
            "deepseek2.feed_forward_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        (
            "deepseek2.attention.head_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.000001),
        ),
        (
            "deepseek2.rope.freq_base".into(),
            GgufMetadataValue::Float32(10_000.0),
        ),
        (
            "deepseek2.rope.dimension_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.attention.q_lora_rank".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.kv_lora_rank".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.key_length_mla".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.value_length_mla".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.leading_dense_block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "deepseek2.expert_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.expert_shared_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "deepseek2.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.expert_used_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.expert_group_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.expert_group_used_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "deepseek2.expert_gating_func".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.expert_weights_norm".into(),
            GgufMetadataValue::Bool(true),
        ),
        (
            "deepseek2.expert_weights_scale".into(),
            GgufMetadataValue::Float32(1.0),
        ),
        ("deepseek2.vocab_size".into(), GgufMetadataValue::Uint32(13)),
    ])
}

fn deepseek_gguf_tensors() -> Vec<GgufFixtureTensor> {
    let tensor = |name: &str, row_major_shape: &[u64], phase: usize| {
        let mut dimensions = row_major_shape.to_vec();
        dimensions.reverse();
        let elements = dimensions.iter().product::<u64>() as usize;
        f32_gguf_tensor(name, dimensions, patterned_values(elements, 0.003, phase))
    };
    let norm =
        |name: &str, width: u64| f32_gguf_tensor(name, vec![width], vec![1.0; width as usize]);
    let mut tensors = vec![
        tensor("token_embd.weight", &[13, 12], 1),
        norm("output_norm.weight", 12),
        tensor("output.weight", &[13, 12], 2),
    ];
    for layer in 0..2 {
        let phase = 3 + layer * 9;
        tensors.extend([
            norm(&format!("blk.{layer}.attn_norm.weight"), 12),
            norm(&format!("blk.{layer}.ffn_norm.weight"), 12),
            tensor(&format!("blk.{layer}.attn_q_a.weight"), &[4, 12], phase),
            norm(&format!("blk.{layer}.attn_q_a_norm.weight"), 4),
            tensor(&format!("blk.{layer}.attn_q_b.weight"), &[16, 4], phase + 1),
            tensor(
                &format!("blk.{layer}.attn_kv_a_mqa.weight"),
                &[6, 12],
                phase + 2,
            ),
            norm(&format!("blk.{layer}.attn_kv_a_norm.weight"), 4),
            tensor(
                &format!("blk.{layer}.attn_k_b.weight"),
                &[4, 4, 2],
                phase + 3,
            ),
            tensor(
                &format!("blk.{layer}.attn_v_b.weight"),
                &[4, 2, 4],
                phase + 4,
            ),
            tensor(
                &format!("blk.{layer}.attn_output.weight"),
                &[12, 8],
                phase + 5,
            ),
        ]);
    }
    tensors.extend([
        tensor("blk.0.ffn_gate.weight", &[16, 12], 21),
        tensor("blk.0.ffn_up.weight", &[16, 12], 22),
        tensor("blk.0.ffn_down.weight", &[12, 16], 23),
        tensor("blk.1.ffn_gate_inp.weight", &[4, 12], 24),
        f32_gguf_tensor(
            "blk.1.exp_probs_b.bias",
            vec![4],
            patterned_values(4, 0.001, 25),
        ),
        tensor("blk.1.ffn_gate_shexp.weight", &[4, 12], 26),
        tensor("blk.1.ffn_up_shexp.weight", &[4, 12], 27),
        tensor("blk.1.ffn_down_shexp.weight", &[12, 4], 28),
        tensor("blk.1.ffn_gate_exps.weight", &[4, 4, 12], 29),
        tensor("blk.1.ffn_up_exps.weight", &[4, 4, 12], 30),
        tensor("blk.1.ffn_down_exps.weight", &[4, 12, 4], 31),
    ]);
    tensors
}

fn write_deepseek_gguf_fixture(path: &Path) {
    let tensors = deepseek_gguf_tensors();
    let inputs = tensors
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &deepseek_gguf_metadata(),
            &inputs,
        )
        .unwrap();
}

fn kimi_linear_gguf_metadata() -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("kimi-linear".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (
            "kimi-linear.block_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.embedding_length".into(),
            GgufMetadataValue::Uint32(12),
        ),
        (
            "kimi-linear.attention.head_count".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "kimi-linear.attention.head_count_kv".into(),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![0, 1])),
        ),
        (
            "kimi-linear.rope.dimension_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.attention.key_length_mla".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.vocab_size".into(),
            GgufMetadataValue::Uint32(13),
        ),
        (
            "kimi-linear.feed_forward_length".into(),
            GgufMetadataValue::Uint32(17),
        ),
        (
            "kimi-linear.context_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        (
            "kimi-linear.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.00001),
        ),
        (
            "kimi-linear.kda.head_dim".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.ssm.conv_kernel".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.expert_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(9),
        ),
        (
            "kimi-linear.attention.kv_lora_rank".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.attention.value_length_mla".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.leading_dense_block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "kimi-linear.expert_used_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.expert_shared_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
    ])
}

fn kimi_linear_gguf_specs() -> Vec<GgufFixtureTensor> {
    let tensor = |name: &str, row_major_shape: &[u64], phase: usize| {
        let mut dimensions = row_major_shape.to_vec();
        dimensions.reverse();
        let elements = dimensions.iter().product::<u64>() as usize;
        f32_gguf_tensor(name, dimensions, patterned_values(elements, 0.003, phase))
    };
    let norm =
        |name: &str, width: u64| f32_gguf_tensor(name, vec![width], vec![1.0; width as usize]);
    let mut specs = vec![
        tensor("token_embd.weight", &[13, 12], 1),
        norm("output_norm.weight", 12),
        tensor("output.weight", &[13, 12], 2),
    ];
    for layer in 0..2 {
        specs.push(norm(&format!("blk.{layer}.attn_norm.weight"), 12));
        specs.push(norm(&format!("blk.{layer}.ffn_norm.weight"), 12));
    }
    specs.extend([
        tensor("blk.0.attn_q.weight", &[12, 12], 3),
        tensor("blk.0.attn_k.weight", &[12, 12], 4),
        tensor("blk.0.attn_v.weight", &[12, 12], 5),
        tensor("blk.0.ssm_conv1d_q.weight", &[12, 2], 6),
        tensor("blk.0.ssm_conv1d_k.weight", &[12, 2], 7),
        tensor("blk.0.ssm_conv1d_v.weight", &[12, 2], 8),
        tensor("blk.0.ssm_f_a.weight", &[4, 12], 9),
        tensor("blk.0.ssm_f_b.weight", &[12, 4], 10),
        tensor("blk.0.ssm_beta.weight", &[3, 12], 11),
        tensor("blk.0.ssm_g_a.weight", &[4, 12], 12),
        tensor("blk.0.ssm_g_b.weight", &[12, 4], 13),
        f32_gguf_tensor("blk.0.ssm_a", vec![3], vec![-0.7, -0.9, -1.1]),
        f32_gguf_tensor(
            "blk.0.ssm_dt.bias",
            vec![12],
            patterned_values(12, 0.002, 14),
        ),
        norm("blk.0.ssm_norm.weight", 4),
        tensor("blk.0.attn_output.weight", &[12, 12], 15),
        tensor("blk.0.ffn_gate.weight", &[17, 12], 16),
        tensor("blk.0.ffn_up.weight", &[17, 12], 17),
        tensor("blk.0.ffn_down.weight", &[12, 17], 18),
        tensor("blk.1.attn_q.weight", &[12, 12], 19),
        tensor("blk.1.attn_kv_a_mqa.weight", &[6, 12], 20),
        norm("blk.1.attn_kv_a_norm.weight", 4),
        tensor("blk.1.attn_kv_b.weight", &[12, 4], 21),
        tensor("blk.1.attn_output.weight", &[12, 6], 22),
        tensor("blk.1.ffn_gate_inp.weight", &[4, 12], 23),
        f32_gguf_tensor(
            "blk.1.exp_probs_b.bias",
            vec![4],
            patterned_values(4, 0.001, 24),
        ),
        tensor("blk.1.ffn_gate_shexp.weight", &[9, 12], 25),
        tensor("blk.1.ffn_up_shexp.weight", &[9, 12], 26),
        tensor("blk.1.ffn_down_shexp.weight", &[12, 9], 27),
        tensor("blk.1.ffn_gate_exps.weight", &[4, 9, 12], 28),
        tensor("blk.1.ffn_up_exps.weight", &[4, 9, 12], 29),
        tensor("blk.1.ffn_down_exps.weight", &[4, 12, 9], 30),
    ]);
    specs
}

fn write_kimi_linear_gguf_fixture(path: &Path) {
    let specs = kimi_linear_gguf_specs();
    let tensors = specs
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &kimi_linear_gguf_metadata(),
            &tensors,
        )
        .unwrap();
}

fn render_failure(rank: usize, output: &Output) -> String {
    format!(
        "pipeline Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}
