#[test]
fn neutral_moshi_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::moshi::LayeredModel<MlxNeuralBackend>;
    let config = eredu_architectures::moshi::MoshiConfig::from_json(
        r#"{
            "model_type":"moshi","dim":16,"text_card":31,
            "n_q":4,"dep_q":3,"generated_audio_codebooks":2,"card":32,
            "num_heads":4,"num_layers":1,"dim_feedforward":24,
            "causal":true,"context":7,"max_period":10000.0,
            "positional_embedding":"rope","depformer_dim":16,
            "depformer_dim_feedforward":24,"depformer_num_heads":4,
            "depformer_num_layers":1,"depformer_context":3,
            "depformer_max_period":10000.0,"depformer_pos_emb":"none",
            "delays":[0,0,1,2,1]
        }"#,
    )
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(config.clone(), stream).unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::moshi::state_layout(&config).unwrap())
            .unwrap();
    let text = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    let audio_tokens = (0..4)
        .map(|_| MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2])))
        .collect::<Vec<_>>();
    let audio = audio_tokens.iter().collect::<Vec<_>>();
    execute_target_group!(
        Architecture,
        MlxKeyValueState,
        architecture,
        state,
        eredu_architectures::moshi::Input {
            text: &text,
            audio: &audio,
            mask: None,
        },
        &[1, 2, 31],
        stream
    );
}
