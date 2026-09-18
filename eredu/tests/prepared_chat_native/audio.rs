//! Actual nonzero Gemma image/audio ingress joined to ordinary semantic tools.
//! The decoder scripts token transitions; this is not an audio-understanding oracle.
use super::*;
use eredu_core::{InputExtent, InputMetadataKey, InputModality, InputPayloadKind, TokenInputRejection};
use eredu_runtime::input::host::{HostInputPart, HostTensorValues, HostTensorView};
use std::ops::ControlFlow;

// Gemma's authenticated chat projection uses these architecture-declared
// spellings. The configured numeric IDs alone do not authorize other markers.
const IMAGE: &str = "<|image|>";
const AUDIO: &str = "<|audio|>";

#[test]
#[ignore = "requires an accessible Metal device and actual image/audio encoders"]
fn authenticated_image_audio_tools_preserve_ordinary_manual_and_recorded_semantics() {
    for choice in [ToolChoice::Required, ToolChoice::Auto] {
        let (root, tokenizer, script) = fixture();
        // No drafting plan, drafter or speculative generation capability is used.
        let execution = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "metal:0").unwrap());
        let (mut model, _) = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(), root.path(), &execution,
        ).unwrap_or_else(fail).into_parts();
        let cancellation = GenerationCancellationToken::new();
        let tokens = model.compile_managed_plain_text_source(
            std::fs::File::open(root.path().join("tokenizer.json")).unwrap(),
        ).unwrap_or_else(fail);
        let source = model.compile_managed_chat_source(
            &tokens, std::fs::File::open(root.path().join("tokenizer_config.json")).unwrap(),
            true, &cancellation,
        ).unwrap_or_else(fail).unwrap();
        let policy = ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":
                format!("Read value 17. Image: {IMAGE} Audio: {AUDIO}")})],
            tools: vec![serde_json::json!({"type":"function","function":{"name":"reading",
                "parameters":{"type":"object","properties":{"value":{"enum":[17]}},
                "required":["value"],"additionalProperties":false}}})],
            tool_choice: choice, add_generation_prompt: true, ..Default::default()
        };
        let chat = model.prepare_chat(&source, &policy, CAPACITY, &cancellation)
            .unwrap_or_else(fail).unwrap();
        let encoded = tokenizer.encode(chat.rendered_prompt(), false).unwrap();
        let ids = encoded.get_ids();
        let marker = |spelling| {
            let token = tokenizer.token_to_id(spelling).unwrap();
            let found = ids.iter().enumerate().filter(|(_, id)| **id == token)
                .map(|(index, _)| index).collect::<Vec<_>>();
            assert_eq!(found.len(), 1, "one exact rendered media marker");
            found[0]
        };
        let image = marker(IMAGE);
        let audio = marker(AUDIO);
        assert!(image + 1 < audio && audio + 1 < ids.len());
        // Same Gemma Unified geometry/masks as the original_media native fixture:
        // four image patches pool to one row; four audio frames subsample to one.
        let prompt = Prompt::new([&ids[..image], &ids[image + 1..audio], &ids[audio + 1..]]);
        let parts = prompt.parts();
        let positions = ids.len() as u64;
        let chunk = if positions % 17 == 0 { 19 } else { 17 };
        assert!(positions > chunk && positions % chunk != 0);
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(0.0), max_new_tokens: Some(8), ..Default::default()
            },
            inference: TextInferencePolicy {
                managed_memory_capacity_bytes: Some(CAPACITY),
                prefill_chunk_positions: NonZeroU64::new(chunk), ..Default::default()
            }, seed: 37, ..Default::default()
        };
        let cancelled = GenerationCancellationToken::new();
        cancelled.cancel();
        assert!(model.prepare_chat_input(&chat, &parts, &cancelled).unwrap_or_else(fail).is_none());
        let input = model.prepare_chat_input(&chat, &parts, &cancellation)
            .unwrap_or_else(fail).unwrap();
        let binding = input.chat_binding().unwrap();
        assert_eq!(binding.coordinates().len(), 5);
        for (segment, at) in [(1, image), (3, audio)] {
            assert_eq!(binding.coordinates()[segment].rendered_range(), [at, at + 1]);
            assert_eq!(binding.coordinates()[segment].decoder_range(), Some([at as u64, at as u64 + 1]));
        }
        assert_eq!(binding.coordinates()[4].decoder_range().unwrap()[1], positions);
        let foreign = model.prepare_chat(&source, &policy, CAPACITY, &cancellation)
            .unwrap_or_else(fail).unwrap();
        let mut request = PreparedChatRequest::new(&foreign, settings.clone());
        request.input = PreparedChatPrompt::Media(input);
        let error = model.start_prepared_chat(request, &cancellation).err()
            .expect("equal rendered text does not replace the authenticated chat owner");
        assert_eq!(error.input_rejection(), Some(TokenInputRejection::IdentityMismatch));
        drop((error, foreign));

        let mut expected = None;
        for mode in 0..3 {
            if mode != 0 {
                model.prepare_reset_ordinary().unwrap_or_else(fail)
                    .reset_admitted(eredu_core::SessionResetLimits::new(CAPACITY)).unwrap_or_else(fail);
                model.synchronize().unwrap_or_else(fail);
            }
            let input = model.prepare_chat_input(&chat, &parts, &cancellation)
                .unwrap_or_else(|error| panic!("{choice:?} audio mode {mode} input: {error:?}"))
                .unwrap();
            let mut request = PreparedChatRequest::new(&chat, settings.clone());
            request.input = PreparedChatPrompt::Media(input);
            let mut events = Vec::new();
            let (output_ids, finish) = if mode == 2 {
                let mut emit = |record: ControlledGenerationRecord| {
                    if let Some(ObservedGenerationEvent::Semantic { event, .. }) = record.event.progress() {
                        events.push(event.clone());
                    }
                    ControlFlow::Continue(())
                };
                let mut session = model.start_controlled_chat(request, TraceLimits {
                    per_record_bytes: 65536, total_bytes: 1 << 20,
                }, Default::default(), &mut emit)
                    .unwrap_or_else(|error| panic!("{choice:?} audio recorded start: {error:?}"))
                    .unwrap();
                assert_eq!(session.prompt_attribution().decoder_positions, positions);
                session.run(&mut emit).unwrap_or_else(fail);
                (session.token_ids().to_vec(), session.finish_reason().unwrap())
            } else {
                let mut session = model.start_prepared_chat(request, &cancellation)
                    .unwrap_or_else(|error| panic!("{choice:?} audio mode {mode} start: {error:?}"))
                    .unwrap();
                let report = session.preparation_report().unwrap();
                assert_eq!(report.geometry.input_positions, positions);
                assert_eq!(report.geometry.prefill_chunk_positions, chunk);
                let attribution = session.prompt_attribution().unwrap().attribution();
                assert_eq!(attribution.segments.len(), 5);
                assert_eq!(attribution.decoder_positions, positions);
                assert!(attribution.complete_token_ids().is_none());
                for (segment, at) in [(1, image), (3, audio)] {
                    assert_eq!(attribution.segments[segment].plan.decoder_range, [at as u64, at as u64 + 1]);
                    assert!(matches!(attribution.segments[segment].tokens,
                        eredu_core::PromptTokenAttribution::NotTokenized));
                }
                let text_ids = prompt.text.iter().flat_map(|part| part.iter().copied()).collect::<Vec<_>>();
                assert_eq!(attribution.canonical_token_ids.as_slice(), text_ids.as_slice());
                let output = if mode == 0 {
                    session.run(&cancellation, &mut |event| events.push(event)).unwrap_or_else(fail)
                } else {
                    while session.finish_reason().is_none() {
                        session = session.advance(&cancellation, &mut |event| events.push(event))
                            .unwrap_or_else(fail);
                    }
                    session.into_output().unwrap_or_else(|_| panic!("terminal audio session"))
                };
                (output.token_ids.to_vec(), output.finish_reason)
            };
            assert!(output_ids.starts_with(&script[..5]), "{choice:?} mode {mode}: {output_ids:?}");
            assert!(output_ids.len() >= 5, "prefill followed by multiple cached predictions");
            assert_eq!(finish, eredu_core::FinishReason::GrammarComplete);
            assert!(!events.iter().any(|event| matches!(event,
                SemanticEvent::TextDelta(_) | SemanticEvent::ReasoningDelta(_))));
            let serialized = serde_json::to_value(&events).unwrap();
            let rows = serialized.as_array().unwrap();
            let starts = rows.iter().filter_map(|event| event.get("ToolCallStart")).collect::<Vec<_>>();
            assert_eq!(starts.len(), 1);
            assert_eq!(starts[0]["name"], "reading");
            assert_eq!(starts[0]["id"], "call_0");
            assert_eq!(starts[0]["index"], 0);
            let fragments = rows.iter().filter_map(|event| event.get("ToolArgumentsDelta"))
                .map(|delta| { assert_eq!(delta["index"], 0); delta["json_fragment"].as_str().unwrap() })
                .collect::<Vec<_>>();
            assert!(fragments.len() >= 2, "arguments must span committed token events");
            assert_eq!(serde_json::from_str::<serde_json::Value>(&fragments.concat()).unwrap(), serde_json::json!({"value":17}));
            assert_eq!(events.iter().filter(|event| matches!(event, SemanticEvent::ToolCallEnd)).count(), 1);
            assert_eq!(events.iter().filter(|event| matches!(event, SemanticEvent::Finished { .. })).count(), 1);
            assert!(matches!(events.first(), Some(SemanticEvent::ToolCallStart { .. })));
            assert!(matches!(events.get(events.len() - 2), Some(SemanticEvent::ToolCallEnd)));
            assert!(events[1..events.len() - 2].iter().all(|event|
                matches!(event, SemanticEvent::ToolArgumentsDelta { .. })));
            assert!(matches!(events.last(), Some(SemanticEvent::Finished {
                reason: eredu_core::FinishReason::GrammarComplete,
            })));
            let result = serde_json::json!({"ids":output_ids,"finish":finish,"events":serialized});
            if let Some(expected) = &expected { assert_eq!(&result, expected, "{choice:?} mode {mode}"); }
            else { expected = Some(result); }
        }
    }
}

struct Prompt<'a> {
    text: [&'a [u32]; 3],
    shapes: [[usize; 2]; 3],
    patches: [f32; 192],
    audio: [f32; 512],
    image_metadata: [(InputMetadataKey, HostTensorView<'static>); 2],
    audio_metadata: [(InputMetadataKey, HostTensorView<'static>); 1],
}
impl<'a> Prompt<'a> {
    fn new(text: [&'a [u32]; 3]) -> Self {
        Self {
            shapes: text.map(|part| [1, part.len()]), text,
            patches: std::array::from_fn(|i| (i as f32 - 93.0) / 193.0),
            audio: std::array::from_fn(|i| (i as f32 - 251.0) / 513.0),
            image_metadata: [
                (InputMetadataKey::PatchGrid, HostTensorView { shape: &[1, 3], values: HostTensorValues::I32(&[1, 2, 2]) }),
                (InputMetadataKey::PatchPositions, HostTensorView { shape: &[1, 4, 2], values: HostTensorValues::I32(&[0, 0, 0, 1, 1, 0, 1, 1]) }),
            ],
            audio_metadata: [(InputMetadataKey::AudioMask, HostTensorView { shape: &[1, 4], values: HostTensorValues::Bool(&[true, true, true, true]) })],
        }
    }
    fn parts(&self) -> [HostInputPart<'_>; 5] {
        let text = |index: usize| HostInputPart {
            modality: InputModality::Text, kind: InputPayloadKind::TokenIds,
            payload: HostTensorView { shape: self.shapes[index].as_slice(), values: HostTensorValues::U32(self.text[index]) },
            metadata: &[], extents: &[],
        };
        [text(0), HostInputPart {
            modality: InputModality::Image, kind: InputPayloadKind::Tensor,
            payload: HostTensorView { shape: &[1, 4, 48], values: HostTensorValues::F32(&self.patches) },
            metadata: &self.image_metadata,
            extents: &[InputExtent::PatchGrid { time: 1, height: 2, width: 2 }],
        }, text(1), HostInputPart {
            modality: InputModality::Audio, kind: InputPayloadKind::Tensor,
            payload: HostTensorView { shape: &[1, 4, 128], values: HostTensorValues::F32(&self.audio) },
            metadata: &self.audio_metadata, extents: &[InputExtent::AudioValidFrames(4)],
        }, text(2)]
    }
}

fn fixture() -> (tempfile::TempDir, Tokenizer, Vec<u32>) {
    let (root, mut tokenizer, script) = super::fixture(false);
    tokenizer.add_special_tokens([IMAGE, AUDIO].map(|value| AddedToken::from(value, true).normalized(false))).unwrap();
    tokenizer.save(root.path().join("tokenizer.json"), false).unwrap();
    let hidden = 16;
    let config = serde_json::json!({
        "model_type":"gemma4_unified", "tie_word_embeddings":false,
        "image_token_id":tokenizer.token_to_id(IMAGE).unwrap(),
        "audio_token_id":tokenizer.token_to_id(AUDIO).unwrap(),
        "eos_token_id":tokenizer.token_to_id("<|im_end|>").unwrap(),
        "text_config":{
            "model_type":"gemma4_text", "hidden_size":hidden, "num_hidden_layers":2,
            "intermediate_size":16, "num_attention_heads":4, "num_key_value_heads":2,
            "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":tokenizer.get_vocab_size(true),
            "max_position_embeddings":2048, "attention_k_eq_v":false,
            "layer_types":["full_attention","sliding_attention"], "sliding_window":8,
            "tie_word_embeddings":false
        },
        "vision_config":{
            "hidden_size":16, "intermediate_size":16, "num_hidden_layers":1,
            "num_attention_heads":4, "num_key_value_heads":2, "head_dim":4,
            "patch_size":4, "pooling_kernel_size":2, "position_embedding_size":4,
            "rms_norm_eps":0.00001
        },
        "audio_config":{
            "hidden_size":16, "num_hidden_layers":1, "num_attention_heads":4,
            "output_proj_dims":8, "conv_kernel_size":3, "attention_chunk_size":4,
            "attention_context_left":5, "attention_context_right":0,
            "attention_invalid_logits_value":-1000000000.0, "attention_logit_cap":50.0,
            "residual_weight":0.5, "rms_norm_eps":0.00001, "subsampling_conv_channels":[4,8]
        }
    });
    std::fs::write(root.path().join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let checkpoint = resolved.architecture.checkpoint();
    let newline = tokenizer.encode("\n", false).unwrap().get_ids()[0];
    let previous = std::iter::once(newline).chain(script[..script.len() - 1].iter().copied()).collect::<Vec<_>>();
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    for tensor in checkpoint.common_tensors.iter().chain(checkpoint.layout_groups.iter()
        .filter_map(|group| group.variants.first()).flat_map(|variant| &variant.tensors)) {
        let start = data.len();
        let media = tensor.key.contains("vision") || tensor.key.contains("audio");
        for index in 0..tensor.shape.iter().product::<usize>() {
            let value: f32 = if tensor.key.ends_with("input_min") || tensor.key.ends_with("output_min") { -4.0 }
            else if tensor.key.ends_with("input_max") || tensor.key.ends_with("output_max") { 4.0 }
            else if tensor.key.contains("norm") || tensor.key.ends_with(".layer_scalar") { 1.0 }
            else if tensor.key.ends_with(".embed_tokens.weight") {
                if previous.get(index % hidden).copied() == Some((index / hidden) as u32) { 1.0 } else { 0.0 }
            } else if tensor.key.ends_with("lm_head.weight") {
                if script.get(index % hidden).copied() == Some((index / hidden) as u32) { 10.0 } else { 0.0 }
            } else if media { (index % 17) as f32 * 0.01 - 0.08 }
            else { 0.0 };
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(tensor.key.clone(), serde_json::json!({"dtype":"F32","shape":tensor.shape,"data_offsets":[start,data.len()]}));
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) { header.push(b' '); }
    let mut file = std::fs::File::create(root.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&data).unwrap();
    (root, tokenizer, script)
}
