//! NVIDIA channel fusion, prompt embedding and dual readout on one supplied hidden state.
#[path = "support/component_coordinator.rs"]
mod component_coordinator;
#[path = "voicechat_channels/parallel.rs"]
mod parallel;

use eredu_architectures::voicechat::function_head_layered::FunctionHeadExecution;
use eredu_architectures::{
    decoder::{StaticModuleSpec, StaticModules},
    voicechat::{
        channels::{ChannelError, ChannelFusion, FunctionHead, PromptPolicy},
        config::ChannelPolicy,
    },
};
use eredu_backend_mlx::{MlxTensor, backend::nn::shared::MlxNeuralBackend as Backend};
use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    LinearFormatSpec, NormalizationOperator, ParameterMetadata, ParameterVisitorMut, Parameterized,
    Tensor, TensorElementType,
};
use safemlx::{Array, Device, DeviceType, Stream};
use serde_json::Value;

fn tensor(value: &Value) -> MlxTensor {
    let shape = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let values = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    MlxTensor::from_array(Array::from_slice(&values, &shape))
}
fn ids(value: &Value, stream: &Stream) -> MlxTensor {
    let rows = value.as_array().unwrap();
    let width = rows[0].as_array().unwrap().len();
    let flat: Vec<_> = rows
        .iter()
        .flat_map(|row| row.as_array().unwrap())
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    MlxTensor::from_i32_slice(&flat, &[rows.len() as i32, width as i32], stream).unwrap()
}
struct Bind<'a> {
    values: &'a Value,
    dtype: TensorElementType,
    stream: &'a Stream,
    rows: Option<std::ops::Range<i32>>,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut MlxTensor) {
        let mut replacement = tensor(&self.values[metadata.id.as_str()])
            .cast_float(self.dtype, self.stream)
            .unwrap();
        if replacement.shape().len() == 2 {
            if let Some(rows) = &self.rows {
                replacement = replacement
                    .index(
                        &[
                            eredu_nn::Index::Range(rows.start, rows.end),
                            eredu_nn::Index::Full,
                        ],
                        self.stream,
                    )
                    .unwrap();
            }
        }
        assert_eq!(output.shape(), replacement.shape());
        *output = replacement;
    }
}
fn fusion() -> ChannelFusion {
    ChannelFusion::from_config(&ChannelPolicy {
        delay_source_text_by: 15,
        delay_text_channel_by: 2,
        duplex_asr_text_weight: 1.0,
        duplex_function_channel_weight: 2.0,
        duplex_text_channel_weight: 1.0,
        duplex_user_channel_weight: 1.0,
        text_embedding_weight: 1.0,
        max_fc_total_tokens: 3000,
        use_function_head: true,
        use_separate_asr_head: false,
        predict_user_text: false,
        allow_user_text_in_agent_turn: false,
        eval_text_turn_taking: true,
    })
    .unwrap()
}
fn specification(fixture: &Value) -> StaticModuleSpec {
    let geometry = &fixture["geometry"];
    StaticModuleSpec {
        normalization_groups: Some(1),
        embedding_weight: "stt_model.embed_tokens.weight".into(),
        normalization_weight: "stt_model.llm.norm_f.weight".into(),
        head_weight: "stt_model.lm_head.weight".into(),
        vocabulary: geometry["vocab_size"].as_i64().unwrap() as i32,
        hidden_size: geometry["width"].as_i64().unwrap() as i32,
        normalization_epsilon: geometry["norm_epsilon"].as_f64().unwrap() as f32,
        normalization_offset: 0.0,
        embedding_quantization: None,
        head_format: LinearFormat::Dense,
        tied_head: false,
    }
}
fn modules(fixture: &Value, dtype: TensorElementType, stream: &Stream) -> StaticModules<Backend> {
    let mut result = StaticModules::<Backend>::from_spec(specification(fixture), stream).unwrap();
    result.visit_parameters_mut(&mut Bind {
        values: &fixture["parameters"],
        dtype,
        stream,
        rows: None,
    });
    result
}
fn compare(actual: &MlxTensor, expected: &Value, tolerance: &Value, bf16: bool, stream: &Stream) {
    assert_eq!(actual.shape(), tensor(expected).shape());
    for (index, (actual, expected)) in actual
        .to_f32_vec(stream)
        .unwrap()
        .into_iter()
        .zip(expected["values"].as_array().unwrap())
        .enumerate()
    {
        let expected = expected.as_f64().unwrap() as f32;
        let mut limit = tolerance["absolute"].as_f64().unwrap() as f32
            + tolerance["relative"].as_f64().unwrap() as f32 * expected.abs();
        if bf16 {
            limit += f32::from_bits(expected.abs().to_bits() + 0x10000) - expected.abs();
        }
        assert!(
            (actual - expected).abs() <= limit,
            "element {index}: {actual} vs NVIDIA {expected}, bound {limit}"
        );
    }
}
#[test]
fn cpu_channels_preserve_cast_order_and_project_both_heads_from_one_hidden() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for (name, dtype) in [
        ("f32", TensorElementType::F32),
        ("bf16", TensorElementType::Bf16),
    ] {
        let case = &fixture[name];
        let mut modules = modules(&fixture, dtype, &stream);
        let fused = fusion()
            .fuse(
                &mut modules,
                &tensor(&case["audio_input_f32"]),
                &ids(&case["agent_tokens"], &stream),
                &ids(&case["function_tokens"], &stream),
                &stream,
            )
            .unwrap();
        if dtype == TensorElementType::Bf16 {
            assert_eq!(
                fused.to_f32_vec(&stream).unwrap(),
                tensor(&case["fused_input"]).to_f32_vec(&stream).unwrap(),
                "audio must round before the two BF16 additions"
            );
        } else {
            compare(
                &fused,
                &case["fused_input"],
                &fixture["tolerance"],
                false,
                &stream,
            );
        }
        let mut head = FunctionHead::<Backend>::new(
            4,
            24,
            LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
            &stream,
        )
        .unwrap();
        head.visit_parameters_mut(&mut Bind {
            values: &fixture["parameters"],
            dtype,
            stream: &stream,
            rows: None,
        });
        let residual = tensor(&case["residual_hidden"])
            .cast_float(dtype, &stream)
            .unwrap();
        let normalized =
            NormalizationOperator::forward(&mut modules.boundary.norm, &residual, &stream).unwrap();
        compare(
            &normalized,
            &case["normalized_hidden"],
            &fixture["tolerance"],
            dtype == TensorElementType::Bf16,
            &stream,
        );
        let logits = head
            .project_channels(&normalized, &mut modules, &stream)
            .unwrap();
        compare(
            &logits.agent,
            &case["text_logits"],
            &fixture["tolerance"],
            dtype == TensorElementType::Bf16,
            &stream,
        );
        compare(
            &logits.function,
            &case["function_logits"],
            &fixture["tolerance"],
            dtype == TensorElementType::Bf16,
            &stream,
        );
        assert_eq!(
            logits
                .greedy_function(&stream)
                .unwrap()
                .to_i32_vec(&stream)
                .unwrap(),
            ids(&case["function_argmax"], &stream)
                .to_i32_vec(&stream)
                .unwrap()
        );
    }
}
#[test]
fn cpu_prompt_repetition_matches_pinned_shared_embedding_and_pad_feedback() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut modules = modules(&fixture, TensorElementType::F32, &stream);
    for case in fixture["prompts"].as_array().unwrap() {
        let prompt = fusion()
            .prefill(
                &mut modules,
                &[5, 7],
                PromptPolicy {
                    repeat: case["repeat_n"].as_u64().unwrap() as usize,
                    max_tokens: 8,
                },
                &stream,
            )
            .unwrap();
        compare(
            &prompt.embedded,
            &case["fused_input"],
            &fixture["tolerance"],
            false,
            &stream,
        );
        let expected: Vec<_> = case["prompt_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(prompt.tokens.to_i32_vec(&stream).unwrap(), expected);
    }
    assert!(matches!(
        fusion().prefill(
            &mut modules,
            &[5, 7],
            PromptPolicy {
                repeat: 2,
                max_tokens: 7
            },
            &stream
        ),
        Err(ChannelError::PromptCapacity {
            required: 8,
            capacity: 7
        })
    ));
}
