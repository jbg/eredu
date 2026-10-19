//! Mixed media traverse the same prepared target, distributed state and providers.
use super::*;

#[test]
#[ignore = "spawns local CPU Ring ranks and opens loopback sockets; run explicitly"]
fn native_spark_multimodal_parallel_execution_plan_target() {
    assert!(distributed::is_available(Backend::Ring));
    for (axes, residency) in [
        ("tp", WorkerResidency::FullyResident),
        ("pp", WorkerResidency::LayerwiseHost),
        ("ep", WorkerResidency::DenseDiskStream),
        ("tp-pp-ep", WorkerResidency::DenseDiskStream),
    ] {
        let checkpoint = tempfile::tempdir().unwrap();
        let mut config = spark_text_config();
        let official: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../validation/fixtures/muse-spark/bf16-config.json"
        ))
        .unwrap();
        config["vision_config"] = official["vision_config"].clone();
        for (name, value) in [
            ("hidden_size", 8),
            ("intermediate_size", 12),
            ("num_hidden_layers", 1),
            ("num_attention_heads", 2),
            ("patch_size", 2),
            ("patch_temporal", 1),
            ("pos_emb_height", 2),
            ("pos_emb_width", 2),
            ("max_position_embeddings", 4),
        ] {
            config["vision_config"][name] = value.into();
        }
        config["vision_config"]["layer_types"] = serde_json::json!(["full_attention"]);
        config["projector_hidden_size"] = 16.into();
        config["out_hidden_size"] = 32.into();
        config["audio_config"] = official["audio_config"].clone();
        config["audio_config"]["hidden_sizes"] = serde_json::json!([4, 8, 8, 16]);
        config["audio_config"]["depths"] = serde_json::json!([1, 1, 1, 1]);
        config["audio_config"]["mlp_ratio"] = 2.0.into();
        config["audio_config"]["out_hidden_size"] = 16.into();
        write_spark_config(checkpoint.path(), Dtype::F32, &config);
        let path = checkpoint.path().to_path_buf();
        run_ring_pipeline_processes(
            residency,
            FixtureFamily::MuseSpark,
            WorkerMode::ExecutionPlanMediaTarget,
            checkpoint,
            path,
            Some(axes),
        );
    }
}

pub(super) fn input() -> (
    Vec<crate::backend::runtime::media::input::InputPart>,
    Vec<u32>,
) {
    let before = Array::from_slice(&[1u32], &[1, 1]);
    let after = Array::from_slice(&[2u32], &[1, 1]);
    let pixels = Array::from_slice(
        &(0..48)
            .map(|i| (i as f32 - 23.) * 0.013)
            .collect::<Vec<_>>(),
        &[4, 12],
    );
    let video = Array::from_slice(
        &(0..96)
            .map(|i| (i as f32 - 45.) * 0.007)
            .collect::<Vec<_>>(),
        &[8, 12],
    );
    let waveform = Array::from_slice(
        &(0..1920)
            .map(|i| ((i * 17 % 101) as f32 - 50.) * 0.004)
            .collect::<Vec<_>>(),
        &[1, 1920],
    );
    let image_grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
    let video_grid = Array::from_slice(&[2i32, 2, 2], &[1, 3]);
    let parts = vec![
        text_input_part(&before),
        input_part(
            InputModality::Image,
            InputPayload::Tensor(pixels),
            [(InputMetadataKey::PatchGrid, image_grid)],
            [],
        ),
        input_part(
            InputModality::Audio,
            InputPayload::Tensor(waveform),
            [],
            [InputExtent::AudioValidSamples(1920)],
        ),
        input_part(
            InputModality::Video,
            InputPayload::Tensor(video),
            [(InputMetadataKey::PatchGrid, video_grid)],
            [],
        ),
        text_input_part(&after),
    ];
    (parts, vec![1, 42, 44, 43, 43, 2])
}
