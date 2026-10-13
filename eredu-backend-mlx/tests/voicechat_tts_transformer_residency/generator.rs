//! The same generator owns prompt fusion and refinement for selected causal stages.
use super::*;
use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    tts_causal_pipeline::PipelineCausalTransformer,
    tts_generator::{TtsGenerator, TtsGeneratorParts},
    tts_ingress::{PromptLayout, SpeakerPrompt},
};
use eredu_backend_mlx::native::{
    MlxReplicatedCoordinator, MlxTtsTransformerPipeline, MlxTtsTransformerPipelineState,
};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized};

fn coordinator(world: &safemlx::distributed::Group, stream: &Stream) -> MlxReplicatedCoordinator {
    let group = CollectiveGroupId::new(91);
    let descriptor = CommunicationGroupDescriptor::new(
        group,
        0,
        (0..world.size()).collect(),
        Some(world.rank()),
        CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::failure_agreement(true),
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::I32],
                CommunicationTensorLimits::new(1, 1, 1, None).unwrap(),
                true,
            )
            .unwrap(),
        ])
        .unwrap(),
    )
    .unwrap();
    let manifest = CommunicationManifest::new(world.size(), world.rank(), vec![descriptor], vec![])
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(30),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    MlxReplicatedCoordinator::from_manifest(manifest, group, 0, world, stream).unwrap()
}

fn codes(value: &Value, stream: &Stream) -> MlxTensor {
    let shape: Vec<_> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let values: Vec<_> = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    MlxTensor::from_i32_slice(&values, &shape, stream).unwrap()
}
struct Bind<'v>(&'v Value);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
        if self.0.get(metadata.id.as_str()).is_none()
            && metadata
                .id
                .as_str()
                .starts_with("tts_model.tts_model.embed_subword.")
        {
            return;
        }
        let replacement = tensor(&self.0[metadata.id.as_str()]);
        assert_eq!(
            replacement.shape(),
            value.shape(),
            "{}",
            metadata.id.as_str()
        );
        *value = replacement;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    world: &safemlx::distributed::Group,
    pipeline: &mut MlxTtsTransformerPipeline,
    state: &mut MlxTtsTransformerPipelineState,
    selected: &TransformerConfig,
    blocks: std::ops::Range<usize>,
    fixture: &Value,
    mode: u64,
    stream: &Stream,
) {
    let mut config = VoiceChatConfig::from_json(eredu_evaluation::voicechat::OFFICIAL_CONFIG_JSON)
        .unwrap()
        .tts;
    config.backbone = geometry(&fixture["transformer_config"], true);
    let transformer: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    config.character_encoder = geometry(&transformer["t5gemma"]["config"]["encoder"], false);
    config.latent_size = 4;
    config.codebook_size = 5;
    config.mog.intermediate_size = 12;
    config.mog.num_layers = 3;
    config.mog.low_rank = 2;
    config.mog.num_predictions = 4;
    let mut ordinary = TtsGenerator::<MlxNeuralBackend>::new(&config, 5, 5, stream).unwrap();
    let mut bind = Bind(&fixture["parameters"]);
    ordinary.conditioning.visit_parameters_mut(&mut bind);
    ordinary.transformer.visit_parameters_mut(&mut bind);
    ordinary.refinement.visit_parameters_mut(&mut bind);
    ordinary.mog.visit_parameters_mut(&mut bind);
    let mut caches = MlxKeyValueState::device(
        eredu_architectures::decoder::state_layout(
            &TransformerConfig::new(&config.backbone).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    // Independently bind generator components; only its causal provider differs.
    let mut local = TtsGenerator::<MlxNeuralBackend>::new(&config, 5, 5, stream).unwrap();
    local.conditioning.visit_parameters_mut(&mut bind);
    local.refinement.visit_parameters_mut(&mut bind);
    local.mog.visit_parameters_mut(&mut bind);
    let parts = local.into_parts();
    let (mut conditioning, mut character, mut refinement, mut mog) = (
        parts.conditioning,
        parts.character_transformer,
        parts.refinement,
        parts.mog,
    );
    config.backbone = selected.geometry().clone();
    let mut agreement = coordinator(world, stream);
    for (index, call) in fixture["calls"].as_array().unwrap().iter().enumerate() {
        let epoch = DistributedCommitEpoch::new(index as u64 + 1).unwrap();
        if index == 1 && mode == 0 {
            let position = state.causal_position().unwrap();
            let mut rejected = PipelineCausalTransformer::new(
                pipeline,
                &mut agreement,
                [41, mode, world.rank() as u64],
                epoch,
            );
            assert!(rejected
                .forward_causal(&tensor(&call["hidden"]), state, stream)
                .is_err());
            assert_eq!(state.causal_position().unwrap(), position);
            compare_local_cache(state.caches(), &caches, blocks.clone(), fixture, stream);
        }
        let authority = PipelineCausalTransformer::new(
            pipeline,
            &mut agreement,
            [41, mode, index as u64],
            epoch,
        );
        let mut generator = TtsGenerator::from_parts(
            &config,
            5,
            5,
            TtsGeneratorParts::new(conditioning, character, authority, refinement, mog),
        )
        .unwrap();
        let input = codes(&call["input_codes"], stream);
        let text = tensor(&call["supplied_subwords"]);
        if index == 0 {
            let mask: Vec<_> = call["audio_mask"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect();
            let layout = PromptLayout::new(1, 5, &mask).unwrap();
            let expected = ordinary
                .prefill(
                    &input,
                    &layout,
                    SpeakerPrompt::Aria,
                    Some(&text),
                    caches.as_mut(),
                    stream,
                )
                .unwrap();
            compare(
                &expected.to_f32_vec(stream).unwrap(),
                &call["hidden"],
                fixture,
            );
            let actual = generator
                .prefill(
                    &input,
                    &layout,
                    SpeakerPrompt::Aria,
                    Some(&text),
                    state,
                    stream,
                )
                .unwrap();
            compare(
                &actual.to_f32_vec(stream).unwrap(),
                &call["hidden"],
                fixture,
            );
            assert!(generator.transformer.is_consumed());
        } else {
            let selected = call["subword_mask"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_bool().unwrap());
            let mut expected = ordinary
                .begin_frame(&input, selected.then_some(&text), caches.as_mut(), stream)
                .unwrap();
            let mut actual = generator
                .begin_frame(&input, selected.then_some(&text), state, stream)
                .unwrap();
            assert_eq!(actual.model_position(), expected.model_position());
            let lanes = MlxTensor::concatenate(
                &[
                    actual.conditional().clone(),
                    actual.unconditional().unwrap().clone(),
                ],
                0,
                stream,
            )
            .unwrap();
            compare(&lanes.to_f32_vec(stream).unwrap(), &call["hidden"], fixture);
            for step in call["refinement"].as_array().unwrap() {
                let sample = |slot: usize, logits: &MlxTensor, _: f64| {
                    assert_eq!(slot, step["slot"].as_u64().unwrap() as usize);
                    compare(
                        &logits.to_f32_vec(stream).unwrap(),
                        &step["logits"],
                        fixture,
                    );
                    Ok((
                        codes(&step["mixture_indices"], stream)
                            .reshape(&[logits.dim(0), logits.dim(1)], stream)
                            .unwrap(),
                        tensor(&step["gaussian_draw"]),
                    ))
                };
                actual = generator.advance_frame(&actual, sample, stream).unwrap();
                expected = ordinary.advance_frame(&expected, sample, stream).unwrap();
                assert_eq!(
                    actual.refinement().codes().to_i32_vec(stream).unwrap(),
                    expected.refinement().codes().to_i32_vec(stream).unwrap()
                );
            }
            assert_eq!(
                actual.codes().unwrap().to_i32_vec(stream).unwrap(),
                codes(&call["codes"], stream).to_i32_vec(stream).unwrap()
            );
            // Spent authority rejects a second invocation before cache mutation.
            let position = state.causal_position().unwrap();
            assert!(generator
                .begin_frame(&input, selected.then_some(&text), state, stream)
                .is_err());
            assert_eq!(state.causal_position().unwrap(), position);
        }
        compare_local_cache(state.caches(), &caches, blocks.clone(), fixture, stream);
        assert_eq!(
            state.causal_position().unwrap(),
            eredu_nn::AttentionCache::offset(&caches.as_ref()[0])
        );
        let parts = generator.into_parts();
        conditioning = parts.conditioning;
        character = parts.character_transformer;
        refinement = parts.refinement;
        mog = parts.mog;
    }
}
