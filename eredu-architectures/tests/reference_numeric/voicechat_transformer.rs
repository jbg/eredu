//! Actual pinned HF local/global transformer outputs and retained KV state.
use super::*;
use eredu_architectures::voicechat::tts_transformer_layered::CausalTransformerExecution;
use eredu_architectures::voicechat::{
    config::{GemmaConfig, VoiceChatConfig},
    tts_transformer::{TransformerConfig, TtsTransformer},
};
use serde_json::Value;

fn tensor(value: &Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    )
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
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        *value = tensor(&self.0[metadata.id.as_str()]);
    }
}
fn compare(actual: &NumericTensor, expected: &Value, fixture: &Value) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let a = fixture["comparison"]["absolute_tolerance"]
        .as_f64()
        .unwrap() as f32;
    let r = fixture["comparison"]["relative_tolerance"]
        .as_f64()
        .unwrap() as f32;
    for (index, (v, e)) in actual.data.iter().zip(expected.data).enumerate() {
        assert!((v - e).abs() <= a + r * e.abs(), "{index}: {v} vs HF {e}");
    }
}
// Reuse the neutral harness cache arithmetic with the caller-selected exact
// sliding retention convention (current token is included in window size).
#[derive(Clone)]
pub(super) struct CausalCache(pub(super) NumericCache);
impl RuntimeLayerState<NumericBackend> for CausalCache {
    type RetainedValues<'a> =
        <NumericCache as RuntimeLayerState<NumericBackend>>::RetainedValues<'a>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.0.retained_values()
    }
}

impl AttentionCache<NumericTensor> for CausalCache {
    fn offset(&self) -> i32 {
        self.0.offset
    }
    fn max_size(&self) -> Option<i32> {
        self.0.window
    }
    fn update_for_attention(
        &mut self,
        keys: NumericTensor,
        values: NumericTensor,
        ctx: &NumericContext,
    ) -> Result<(NumericTensor, NumericTensor), Error> {
        let output = self.0.update_for_attention(keys, values, ctx)?;
        if let Some(window) = self.0.window {
            for value in [&mut self.0.keys, &mut self.0.values] {
                let t = value.as_ref().unwrap();
                let end = t.dim(2) as usize;
                *value = Some(t.axis_slice(2, end.saturating_sub((window - 1) as usize), end));
            }
        }
        Ok(output)
    }
    fn attention(
        &mut self,
        request: AttentionRequest<'_, NumericTensor>,
        ctx: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.0.attention(request, ctx)
    }
}

#[test]
fn voicechat_transformers_match_pinned_hf_cached_and_bidirectional_outputs() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    TransformerConfig::new(&config.tts.backbone).unwrap();
    TransformerConfig::new(&config.tts.character_encoder).unwrap();
    let data = &fixture["gemma3"];
    let mut model = TtsTransformer::<NumericBackend>::new(
        TransformerConfig::new(&geometry(&data["config"], true)).unwrap(),
        &context,
    )
    .unwrap();
    model.visit_parameters_mut(&mut Bind(&data["parameters"]));
    for execution in ["full", "chunks"] {
        let mut caches = [
            CausalCache(NumericCache::new(Some(3))),
            CausalCache(NumericCache::new(None)),
        ];
        for step in data[execution].as_array().unwrap() {
            let output = model
                .forward_causal(&tensor(&step["input"]), &mut caches[..], &context)
                .unwrap();
            compare(&output, &step["output"], &fixture);
            for (cache, expected) in caches.iter().zip(step["cache"].as_array().unwrap()) {
                assert_eq!(cache.offset(), expected["seen"].as_i64().unwrap() as i32);
                compare(cache.0.keys.as_ref().unwrap(), &expected["keys"], &fixture);
                compare(
                    cache.0.values.as_ref().unwrap(),
                    &expected["values"],
                    &fixture,
                );
            }
        }
        if execution == "chunks" {
            let capacity = data["config"]["max_position_embeddings"].as_i64().unwrap() as i32;
            let oversized = NumericTensor::zeros(vec![1, capacity, 8]);
            assert!(
                model
                    .forward_causal(&oversized, &mut caches[..], &context)
                    .is_err()
            );
            let last = data["chunks"].as_array().unwrap().last().unwrap();
            for (cache, expected) in caches.iter().zip(last["cache"].as_array().unwrap()) {
                assert_eq!(cache.offset(), expected["seen"].as_i64().unwrap() as i32);
                compare(cache.0.keys.as_ref().unwrap(), &expected["keys"], &fixture);
                compare(
                    cache.0.values.as_ref().unwrap(),
                    &expected["values"],
                    &fixture,
                );
            }
        }
    }
    let data = &fixture["t5gemma"];
    let mut model = TtsTransformer::<NumericBackend>::new(
        TransformerConfig::new(&geometry(&data["config"]["encoder"], false)).unwrap(),
        &context,
    )
    .unwrap();
    model.visit_parameters_mut(&mut Bind(&data["parameters"]));
    let mask = tensor(&data["attention_mask"])
        .equal_i32(1, &context)
        .unwrap();
    let batch = &data["batches"][0];
    let output = model
        .forward_encoder(&tensor(&batch["input"]), &mask, &context)
        .unwrap();
    compare(&output, &batch["output"], &fixture);
}

#[test]
fn voicechat_gemma_tp2_preserves_rotary_qk_norm_and_cached_hidden() {
    use eredu_architectures::decoder::{self, AttentionInput, PartitionedConfig};
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let data = &fixture["gemma3"];
    let mut dimensions = geometry(&data["config"], true);
    // Duplicating the fixture's single KV head is mathematically equivalent
    // for its two query heads and gives each ordinary TP rank one KV head.
    dimensions.num_key_value_heads = 2;
    let config = TransformerConfig::new(&dimensions).unwrap();
    let context = NumericContext::default();
    let groups = (0..dimensions.num_hidden_layers)
        .flat_map(|layer| {
            let block = config
                .build_block::<NumericBackend>(layer, &context)
                .unwrap();
            decoder::layer_parallel_parameter_groups::<NumericBackend>(&block, &config, layer)
                .unwrap()
        })
        .collect::<Vec<_>>();
    struct LocalBind<'a> {
        values: &'a Value,
        layout: &'a LocalModelLayout,
    }
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for LocalBind<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, target: &'a mut NumericTensor) {
            let name = metadata.id.as_str();
            let mut value = tensor(&self.values[name]);
            if name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight") {
                value = NumericTensor::concatenate(
                    &[value.clone(), value],
                    0,
                    &NumericContext::default(),
                )
                .unwrap();
            }
            if let Some(layout) = self.layout.tensor(name) {
                value = select_parameter(&value, layout.placement()).unwrap();
            }
            assert_eq!(target.shape(), value.shape(), "{name}");
            *target = value;
        }
    }
    let collective = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let handles = (0..2)
            .map(|rank| {
                let fixture = &fixture;
                let config = config.clone();
                let layout = numeric_local_layout(&groups, 2, rank).unwrap();
                let collective = Arc::clone(&collective);
                scope.spawn(move || {
                    let data = &fixture["gemma3"];
                    let context = NumericContext::with_local_layout(layout.clone());
                    let parallel = NumericParallelContext::new(rank, collective);
                    let mut blocks = (0..2)
                        .map(|layer| {
                            let local = config.local_block_config(layer, &layout).unwrap();
                            let mut block = local
                                .build_block::<NumericBackend>(layer, &context)
                                .unwrap();
                            block.visit_parameters_mut(&mut LocalBind {
                                values: &data["parameters"],
                                layout: &layout,
                            });
                            block
                        })
                        .collect::<Vec<_>>();
                    let mut norm = config.build_final_norm::<NumericBackend>(&context).unwrap();
                    norm.visit_parameters_mut(&mut Bind(&data["parameters"]));
                    let mut caches = [
                        CausalCache(NumericCache::new(Some(3))),
                        CausalCache(NumericCache::new(None)),
                    ];
                    for step in data["chunks"].as_array().unwrap() {
                        let mut hidden = tensor(&step["input"]);
                        let offset = caches[0].offset();
                        for (layer, (block, cache)) in
                            blocks.iter_mut().zip(caches.iter_mut()).enumerate()
                        {
                            let window = (layer == 0).then_some(3);
                            let past = window.map_or(offset, |w| offset.min(w - 1));
                            let mask = NumericBackend::causal_mask(
                                hidden.dim(1),
                                past,
                                window.map(|w| w - 1),
                                &context,
                            )
                            .unwrap();
                            hidden = block
                                .forward_tensor_parallel(
                                    AttentionInput {
                                        hidden: &hidden,
                                        selected_positions: None,
                                        mask: Some(&mask),
                                        cache: Some(cache),
                                        allow_sliding_prefill: false,
                                        rotary_position: None,
                                    },
                                    &parallel,
                                    &context,
                                )
                                .unwrap();
                        }
                        hidden = norm.forward(&hidden, &context).unwrap();
                        compare(&hidden, &step["output"], fixture);
                        for (cache, expected) in
                            caches.iter().zip(step["cache"].as_array().unwrap())
                        {
                            compare(cache.0.keys.as_ref().unwrap(), &expected["keys"], fixture);
                            compare(
                                cache.0.values.as_ref().unwrap(),
                                &expected["values"],
                                fixture,
                            );
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}
