//! Actual pinned HF transformer outputs using native shared attention caches.
use super::super::*;
use eredu_architectures::voicechat::tts_transformer_layered::{
    CausalTransformerExecution, TransformerExecution,
};
use eredu_architectures::voicechat::{
    config::GemmaConfig,
    tts_transformer::{TransformerConfig, TtsTransformer},
};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
use safemlx::{Device, DeviceType};
use serde_json::Value;
pub(super) fn tensor(value: &Value) -> MlxTensor {
    let shape = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let values = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    MlxTensor::from_array(Array::from_slice(&values, &shape))
}
pub(super) fn geometry(value: &Value, causal: bool) -> GemmaConfig {
    let n = |key: &str| value[key].as_u64().unwrap() as usize;
    GemmaConfig {
        hidden_size: n("hidden_size"),
        intermediate_size: n("intermediate_size"),
        num_hidden_layers: n("num_hidden_layers"),
        num_attention_heads: n("num_attention_heads"),
        num_key_value_heads: n("num_key_value_heads"),
        head_dim: n("head_dim"),
        hidden_activation: value["hidden_activation"].as_str().unwrap().into(),
        max_position_embeddings: n("max_position_embeddings"),
        rms_norm_eps: value["rms_norm_eps"].as_f64().unwrap(),
        rope_theta: value["rope_theta"].as_f64().unwrap(),
        rope_local_base_freq: value["rope_local_base_freq"].as_f64(),
        attention_bias: value["attention_bias"].as_bool().unwrap(),
        query_pre_attn_scalar: n("query_pre_attn_scalar"),
        sliding_window: n("sliding_window"),
        layer_types: serde_json::from_value(value["layer_types"].clone()).unwrap(),
        final_logit_softcapping: value["final_logit_softcapping"].as_f64(),
        attn_logit_softcapping: value["attn_logit_softcapping"].as_f64(),
        rope_scaling: None,
        causal,
    }
}

struct Bind<'a>(&'a Value);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
        *value = tensor(&self.0[metadata.id.as_str()]);
    }
}
pub(super) fn compare(actual: &MlxTensor, expected: &Value, fixture: &Value, stream: &Stream) {
    assert_eq!(actual.shape(), tensor(expected).shape());
    let a = fixture["comparison"]["absolute_tolerance"]
        .as_f64()
        .unwrap() as f32;
    let r = fixture["comparison"]["relative_tolerance"]
        .as_f64()
        .unwrap() as f32;
    for (index, (v, e)) in actual
        .to_f32_vec(stream)
        .unwrap()
        .into_iter()
        .zip(expected["values"].as_array().unwrap())
        .enumerate()
    {
        let e = e.as_f64().unwrap() as f32;
        assert!((v - e).abs() <= a + r * e.abs(), "{index}: {v} vs HF {e}");
    }
}
#[test]
fn cpu_voicechat_transformers_match_pinned_hf_cached_and_bidirectional_outputs() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let data = &fixture["gemma3"];
    let mut model = TtsTransformer::<MlxNeuralBackend>::new(
        TransformerConfig::new(&geometry(&data["config"], true)).unwrap(),
        &stream,
    )
    .unwrap();
    model.visit_parameters_mut(&mut Bind(&data["parameters"]));
    for execution in ["full", "chunks"] {
        let mut caches = [
            ConcatKeyValueCache::new_for_sliding_attention(3),
            ConcatKeyValueCache::new(),
        ];
        for step in data[execution].as_array().unwrap() {
            let output = model
                .forward_causal(&tensor(&step["input"]), caches.as_mut_slice(), &stream)
                .unwrap();
            compare(&output, &step["output"], &fixture, &stream);
            for (cache, expected) in caches.iter().zip(step["cache"].as_array().unwrap()) {
                assert_eq!(
                    KeyValueCache::offset(cache),
                    expected["seen"].as_i64().unwrap() as i32
                );
                let arrays = KeyValueCache::retained_arrays(cache);
                assert_eq!(arrays.len(), 2);
                compare(
                    &MlxTensor::from_array(arrays[0].clone()),
                    &expected["keys"],
                    &fixture,
                    &stream,
                );
                compare(
                    &MlxTensor::from_array(arrays[1].clone()),
                    &expected["values"],
                    &fixture,
                    &stream,
                );
            }
        }
    }
    let data = &fixture["t5gemma"];
    let mut model = TtsTransformer::<MlxNeuralBackend>::new(
        TransformerConfig::new(&geometry(&data["config"]["encoder"], false)).unwrap(),
        &stream,
    )
    .unwrap();
    model.visit_parameters_mut(&mut Bind(&data["parameters"]));
    let mask = tensor(&data["attention_mask"])
        .equal_i32(1, &stream)
        .unwrap();
    let batch = &data["batches"][0];
    let output = model
        .forward_encoder(&tensor(&batch["input"]), &mask, &stream)
        .unwrap();
    compare(&output, &batch["output"], &fixture, &stream);
}
