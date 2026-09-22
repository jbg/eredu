//! Nonzero component reconstruction and intervention behavior on shared equations.
use super::*;
use eredu_architectures::decoder::ComponentInstrumentation;
use eredu_runtime::ActivationObserver;

#[path = "components/deepseek_v3.rs"]
mod compressed_v3;

#[path = "components/deepseek_v4.rs"]
mod compressed_v4;

#[path = "components/partition_sum.rs"]
mod partition_sum;

#[path = "components/composite_banks.rs"]
mod composite_banks;
#[path = "components/composite_routed.rs"]
mod composite_routed;
#[path = "components/gemma4_components.rs"]
mod gemma4_components;
#[path = "components/gemma4_placement.rs"]
mod gemma4_placement;
#[path = "components/muse.rs"]
mod muse;
#[path = "components/muse_placement.rs"]
mod muse_placement;
#[path = "components/qwen_conditional.rs"]
mod qwen_conditional;
#[path = "components/qwen_prediction.rs"]
mod qwen_prediction;
#[path = "components/qwen_recurrent.rs"]
mod qwen_recurrent;
#[path = "components/qwen_vl.rs"]
mod qwen_vl;

#[path = "components/nemotron_prediction.rs"]
mod nemotron_prediction;

#[path = "components/inkling.rs"]
mod inkling;
#[path = "components/inkling_placement.rs"]
mod inkling_placement;
#[path = "components/kimi_linear.rs"]
mod kimi_linear;
#[path = "components/lfm2.rs"]
mod lfm2_components;

#[derive(Default)]
pub(super) struct Components {
    pub(super) values: BTreeMap<String, NumericTensor>,
    reject_duplicates: bool,
    zero: Option<(&'static str, usize, usize)>,
    keep: Option<(&'static str, usize, usize)>,
}
impl Components {
    pub(super) fn strict() -> Self {
        Self {
            reject_duplicates: true,
            ..Default::default()
        }
    }
}
impl ActivationObserver<NumericTensor, Error> for Components {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if self.values.insert(path.into(), value.clone()).is_some() && self.reject_duplicates {
            return Err(Error::backend(format!(
                "duplicate component observation: {path}"
            )));
        }
        Ok(())
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        let mut changed = None;
        for (selection, keep) in [(self.zero, false), (self.keep, true)] {
            if let Some((target, sequence, component)) = selection {
                if path == target {
                    let mut out = value.clone();
                    let width = *out.shape.last().unwrap() as usize;
                    for i in 0..width {
                        if (i == component) != keep {
                            out.data[sequence * width + i] = 0.0;
                        }
                    }
                    changed = Some(out);
                }
            }
        }
        Ok(changed)
    }
}

fn forward(
    block: &mut decoder::TransformerBlock<NumericBackend>,
    input: &NumericTensor,
    capture: &mut Components,
) -> NumericTensor {
    block
        .forward_observed(
            decoder::AttentionInput {
                hidden: input,
                mask: None,
                cache: None::<&mut NumericCache>,
                allow_sliding_prefill: true,
                rotary_position: None,
            },
            &NumericContext::default(),
            &mut ComponentInstrumentation::new("model.layers.0", capture),
        )
        .unwrap()
}

#[test]
fn component_sums_reconstruct_writes_and_masks_change_only_the_selected_values() {
    let args = llama::model_args_from_config_value(&config("llama", false)).unwrap();
    let context = NumericContext::default();
    let block = decoder::TransformerBlock::<NumericBackend>::new(&args, 0, &context).unwrap();
    let input = NumericTensor::new(
        vec![1, 3, 8],
        (0..24).map(|i| (i as f32 * 0.4 - 2.0).sin()).collect(),
    );
    let ordinary = block
        .clone()
        .forward(
            decoder::AttentionInput {
                hidden: &input,
                mask: None,
                cache: None::<&mut NumericCache>,
                allow_sliding_prefill: true,
                rotary_position: None,
            },
            &context,
        )
        .unwrap();
    let mut baseline = Components::default();
    let observed = forward(&mut block.clone(), &input, &mut baseline);
    assert_tensor_exact(&ordinary, &observed, "no-op component observation");
    for (suffix, output, projection) in [
        (
            "attention.channels",
            "attention.write",
            &block.self_attention.output,
        ),
        ("feed_forward.units", "feed_forward.write", &block.mlp.down),
    ] {
        let values = &baseline.values[&format!("model.layers.0.{suffix}")];
        assert!(values.data.iter().any(|x| x.abs() > 1e-4));
        let width = values.shape[2] as usize;
        let hidden = projection.weight.shape[0] as usize;
        let mut reconstructed = vec![0.0; 3 * hidden];
        for row in 0..3 {
            for out in 0..hidden {
                let mut total = 0.0_f64;
                for component in 0..width {
                    total += values.data[row * width + component] as f64
                        * projection.weight.data[out * width + component] as f64;
                }
                total += projection
                    .bias
                    .as_ref()
                    .map_or(0.0, |(b, _)| b.data[out] as f64);
                reconstructed[row * hidden + out] = total as f32;
            }
        }
        assert_tensor_close(
            &NumericTensor::new(vec![1, 3, hidden as i32], reconstructed),
            &baseline.values[&format!("model.layers.0.{output}")],
            "component sum plus affine bias",
        );
    }
    for target in [
        "model.layers.0.attention.channels",
        "model.layers.0.feed_forward.units",
    ] {
        let mut capture = Components {
            zero: Some((target, 1, 2)),
            ..Default::default()
        };
        let changed = forward(&mut block.clone(), &input, &mut capture);
        let before = &capture.values[target];
        let after = &capture.values[&format!("{target}.effective")];
        let width = before.shape[2] as usize;
        for i in 0..before.data.len() {
            assert_eq!(
                after.data[i],
                if i == width + 2 { 0.0 } else { before.data[i] }
            );
        }
        assert!(changed
            .data
            .iter()
            .zip(&ordinary.data)
            .any(|(a, b)| (a - b).abs() > 1e-5));
        // Future attention reads are unchanged: this hook is after KV insertion and aggregation.
        assert_eq!(&changed.data[..8], &ordinary.data[..8]);
        assert_eq!(&changed.data[16..], &ordinary.data[16..]);
    }
}

#[test]
fn keep_only_recomputes_surviving_ffn_units_from_the_changed_residual() {
    let args = llama::model_args_from_config_value(&config("llama", false)).unwrap();
    let block =
        decoder::TransformerBlock::<NumericBackend>::new(&args, 0, &NumericContext::default())
            .unwrap();
    let input = NumericTensor::new(
        vec![1, 3, 8],
        (0..24).map(|i| (i as f32 * 0.3).cos()).collect(),
    );
    let mut baseline = Components::default();
    forward(&mut block.clone(), &input, &mut baseline);
    let mut trial = Components {
        zero: Some(("model.layers.0.attention.channels", 2, 1)),
        keep: Some(("model.layers.0.feed_forward.units", 2, 3)),
        ..Default::default()
    };
    forward(&mut block.clone(), &input, &mut trial);
    let units = &trial.values["model.layers.0.feed_forward.units"];
    let effective = &trial.values["model.layers.0.feed_forward.units.effective"];
    let width = units.shape[2] as usize;
    assert_ne!(
        units.data[2 * width + 3],
        baseline.values["model.layers.0.feed_forward.units"].data[2 * width + 3]
    );
    assert_eq!(effective.data[2 * width + 3], units.data[2 * width + 3]);
    for i in 0..width {
        if i != 3 {
            assert_eq!(effective.data[2 * width + i], 0.0);
        }
    }
    let mut replay = Components::default();
    forward(&mut block.clone(), &input, &mut replay);
    assert_eq!(
        baseline.values["model.layers.0.feed_forward.units"].data,
        replay.values["model.layers.0.feed_forward.units"].data
    );
}

#[test]
fn final_readout_hooks_observe_real_values_and_intervene_before_projection() {
    let args = llama::model_args_from_config_value(&config("llama", false)).unwrap();
    let context = NumericContext::default();
    let tokens = NumericTensor::token_ids(&[1, 4, 2]);
    let run = |capture: &mut Components| {
        let architecture =
            llama::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
        let mut runtime = LayerwiseRuntime::new(architecture, RebuildingUnitPolicy::default());
        let mut state = DeviceState::<NumericBackend, _>::create(
            llama::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap();
        runtime
            .forward_with_observer(
                decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                &mut state,
                &context,
                capture,
            )
            .unwrap()
    };
    let mut baseline = Components::default();
    let logits = run(&mut baseline);
    assert_tensor_exact(
        &baseline.values["readout.linear"],
        &logits,
        "linear readout without softcap",
    );
    assert_tensor_exact(
        &baseline.values["readout.embedding.effective"],
        &baseline.values["model.layers.0.input"],
        "effective embeddings enter decoder",
    );
    let mut trial = Components {
        keep: Some(("readout.normalized", 1, usize::MAX)),
        ..Default::default()
    };
    let changed = run(&mut trial);
    assert_tensor_exact(
        &baseline.values["readout.normalized"],
        &trial.values["readout.normalized"],
        "original normalized values preserved",
    );
    let width = logits.shape[2] as usize;
    assert!(logits.data[width..2 * width].iter().any(|x| x.abs() > 1e-5));
    assert!(changed.data[width..2 * width].iter().all(|x| *x == 0.0));
    assert_eq!(&changed.data[..width], &logits.data[..width]);
    assert_eq!(&changed.data[2 * width..], &logits.data[2 * width..]);
}

#[test]
fn gelu_gated_units_follow_the_declared_activation_before_projection() {
    let args = llama::model_args_from_config_value(&config("llama", false)).unwrap();
    let context = NumericContext::default();
    let mut block = decoder::TransformerBlock::<NumericBackend>::new(&args, 0, &context).unwrap();
    block.mlp.limit = Some(GatedProductPolicy::ordinary_gelu_approximate());
    let input = NumericTensor::new(
        vec![1, 3, 8],
        (0..24).map(|i| (i as f32 * 0.7 - 1.1).cos()).collect(),
    );
    let mut captured = Components::default();
    forward(&mut block, &input, &mut captured);
    let normalized = &captured.values["model.layers.0.feed_forward.input"];
    let decoder::GatedInputProjection::Split { gate, up } = &mut block.mlp.input_projection else {
        panic!("split fixture")
    };
    let gates = gate.forward(normalized, &context).unwrap();
    let values = up.forward(normalized, &context).unwrap();
    let units = &captured.values["model.layers.0.feed_forward.units"];
    assert!(gates.data.iter().any(|x| *x < -1e-3));
    assert!(gates.data.iter().any(|x| *x > 1e-3));
    for ((g, v), actual) in gates.data.iter().zip(&values.data).zip(&units.data) {
        let x = *g as f64;
        let gelu = 0.5
            * x
            * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
        let expected = gelu * *v as f64;
        assert!((*actual as f64 - expected).abs() <= 2e-6 + 2e-6 * expected.abs());
    }
    let reconstructed = block.mlp.down.forward(units, &context).unwrap();
    assert_tensor_exact(
        &reconstructed,
        &captured.values["model.layers.0.feed_forward.write"],
        "GELU units are consumed by down projection",
    );
}

#[test]
fn exact_gelu_fixture_matches_independent_erf_values() {
    // Python math.erf evaluated at x / sqrt(2), distinct from the tanh GELU.
    let input = NumericTensor::new(
        [11],
        vec![-10.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 10.0],
    );
    let expected = [
        0.0,
        -0.00012668496733247991,
        -0.04550026389635842,
        -0.15865525393145707,
        -0.15426876936299347,
        0.0,
        0.3457312306370065,
        0.8413447460685429,
        1.9544997361036416,
        3.9998733150326675,
        10.0,
    ];
    let actual = NumericTensor::gelu(&input, &NumericContext::default()).unwrap();
    for (&actual, expected) in actual.data.iter().zip(expected) {
        assert!((f64::from(actual) - expected).abs() < 2e-7);
    }
}

#[test]
fn non_gated_gelu_and_relu_units_include_bias_and_consume_effective_values() {
    use decoder::unary::{self, Activation};
    let context = NumericContext::default();
    let mut read = explicit_numeric_linear(
        "read.weight",
        4,
        2,
        &[0.7, -0.3, -0.5, 0.9, 1.1, 0.2, -0.4, -0.8],
    );
    let mut write = explicit_numeric_linear(
        "write.weight",
        2,
        4,
        &[0.2, -0.4, 0.7, -0.1, -0.8, 0.3, 0.5, 0.6],
    );
    let bias = |name: &str, values: Vec<f32>| {
        let spec = ParameterSpec::trainable(name).unwrap();
        (
            NumericTensor::new([values.len() as i32], values),
            ParameterMetadata::from_spec(&spec, true),
        )
    };
    read.bias = Some(bias("read.bias", vec![0.1, -0.2, 0.3, -0.4]));
    write.bias = Some(bias("write.bias", vec![-0.25, 0.15]));
    let input = NumericTensor::new([1, 3, 2], vec![-1.0, 0.5, 0.7, -0.4, 1.5, 1.0]);
    let projected = read.forward(&input, &context).unwrap();
    assert!(projected.data.iter().any(|v| *v < -0.1));
    assert!(projected.data.iter().any(|v| *v > 0.1));
    for activation in [
        Activation::Gelu,
        Activation::GeluApproximate,
        Activation::Relu,
        Activation::ReluSquared,
    ] {
        let ordinary = unary::forward::<NumericBackend>(
            &input,
            &mut read,
            &mut write,
            activation,
            None,
            &context,
            &mut ComponentInstrumentation::disabled(),
        )
        .unwrap();
        let mut captured = Components::default();
        let observed = unary::forward::<NumericBackend>(
            &input,
            &mut read,
            &mut write,
            activation,
            None,
            &context,
            &mut ComponentInstrumentation::new("unit", &mut captured),
        )
        .unwrap();
        assert_tensor_exact(&ordinary, &observed, "non-gated all-keep");
        let units = &captured.values["unit.feed_forward.units"];
        for (x, actual) in projected.data.iter().zip(&units.data) {
            let x = *x as f64;
            let expected = match activation {
                Activation::GeluApproximate => {
                    0.5 * x
                        * (1.0
                            + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3)))
                                .tanh())
                }
                Activation::Relu => x.max(0.0),
                Activation::ReluSquared => x.max(0.0).powi(2),
                Activation::Gelu => {
                    // Independent Simpson integration of the normal density.
                    let step = x / 256.0;
                    let density =
                        |v: f64| (-0.5 * v * v).exp() / (2.0 * std::f64::consts::PI).sqrt();
                    let interior = (1..256)
                        .map(|i| density(f64::from(i) * step) * if i % 2 == 0 { 2.0 } else { 4.0 })
                        .sum::<f64>();
                    x * (0.5 + step / 3.0 * (density(0.0) + density(x) + interior))
                }
            };
            assert!((*actual as f64 - expected).abs() < 2e-6);
        }
        let mut trial = Components {
            keep: Some(("unit.feed_forward.units", 1, 2)),
            ..Default::default()
        };
        let changed = unary::forward::<NumericBackend>(
            &input,
            &mut read,
            &mut write,
            activation,
            None,
            &context,
            &mut ComponentInstrumentation::new("unit", &mut trial),
        )
        .unwrap();
        let effective = &trial.values["unit.feed_forward.units.effective"];
        for (i, v) in effective.data.iter().enumerate() {
            assert_eq!(
                *v,
                if (4..8).contains(&i) && i != 6 {
                    0.0
                } else {
                    units.data[i]
                }
            );
        }
        let expected = write.forward(effective, &context).unwrap();
        assert_tensor_exact(
            &changed,
            &expected,
            "effective non-gated units plus write bias",
        );
        assert_eq!(&ordinary.data[..2], &changed.data[..2]);
        assert_eq!(&ordinary.data[4..], &changed.data[4..]);
        assert_ne!(&ordinary.data[2..4], &changed.data[2..4]);
    }
}

pub(super) fn nemotron_config() -> serde_json::Value {
    serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":16, "hidden_size":8,
        "intermediate_size":10, "num_hidden_layers":2,
        "hybrid_override_pattern":"*-", "num_attention_heads":2,
        "num_key_value_heads":1, "head_dim":4, "mamba_num_heads":2,
        "n_groups":1, "mamba_head_dim":4, "ssm_state_size":3,
        "conv_kernel":3, "chunk_size":2, "n_routed_experts":2,
        "n_shared_experts":1, "moe_intermediate_size":6,
        "moe_shared_expert_intermediate_size":6, "num_experts_per_tok":1,
        "n_group":1, "topk_group":1, "num_nextn_predict_layers":0,
        "tie_word_embeddings":false, "residual_in_fp32":true,
        "mlp_bias":true, "attention_bias":true
    })
}

impl<C> eredu_runtime::LayeredTraversalHook<NumericBackend, C, Error> for Components {
    fn observes_activations(&self) -> bool {
        true
    }
    fn observe_activation(&mut self, path: &str, tensor: &NumericTensor) -> Result<(), Error> {
        ActivationObserver::observe(self, path, tensor)
    }
    fn intervene_activation(
        &mut self,
        path: &str,
        tensor: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        ActivationObserver::intervene(self, path, tensor)
    }
}

#[test]
fn nemotron_non_gated_components_match_resident_and_rebuilt_execution() {
    let args = nemotron_h::model_args_from_config_value(&nemotron_config()).unwrap();
    let context = NumericContext::default();
    let model = || nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let state = || {
        DeviceState::<NumericBackend, _>::create(
            nemotron_h::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap()
    };
    for masked in [false, true] {
        let mut resident = ResidentRuntime::new(model(), &context).unwrap();
        let mut rebuilt = LayerwiseRuntime::new(model(), RebuildingUnitPolicy::default());
        let (mut rs, mut ls) = (state(), state());
        for ids in [vec![1, 4, 2], vec![3], vec![5]] {
            let input = NumericTensor::token_ids(&ids);
            let capture = || Components {
                zero: masked.then_some(("model.layers.0.attention.channels", ids.len() - 1, 1)),
                keep: masked.then_some(("model.layers.1.feed_forward.units", ids.len() - 1, 2)),
                ..Default::default()
            };
            let (mut rc, mut lc) = (capture(), capture());
            let (r, _) = resident
                .forward_with_traversal_hook(
                    nemotron_h::EmbeddedInput::target(&input, None),
                    &mut rs,
                    &context,
                    &mut rc,
                )
                .unwrap();
            let l = rebuilt
                .forward_with_observer(
                    nemotron_h::EmbeddedInput::target(&input, None),
                    &mut ls,
                    &context,
                    &mut lc,
                )
                .unwrap();
            assert_tensor_exact(&r, &l, "resident/rebuilt non-gated component intervention");
            for (path, value) in &rc.values {
                assert_tensor_exact(value, &lc.values[path], path);
            }
            for path in [
                "model.layers.0.attention.channels",
                "model.layers.1.feed_forward.units",
            ] {
                assert!(rc.values[path].data.iter().any(|v| v.abs() > 1e-5));
            }
            if masked {
                let units = &rc.values["model.layers.1.feed_forward.units.effective"];
                let last = (ids.len() - 1) * 10;
                for i in 0..10 {
                    if i != 2 {
                        assert_eq!(units.data[last + i], 0.0);
                    }
                }
            }
        }
    }
}

#[test]
fn shared_invocation_parameter_slots_carry_the_discovered_aliases() {
    let config = super::nanbeige::tiny_config(false);
    let args = eredu_architectures::nanbeige::model_args_from_config_value(&config).unwrap();
    let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    struct Aliases(BTreeMap<String, String>);
    impl<'a> ParameterVisitor<'a, NumericTensor> for Aliases {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, _: &'a NumericTensor) {
            self.0.insert(
                metadata.id().to_string(),
                metadata.alias_of().unwrap_or(metadata.id()).to_string(),
            );
        }
    }
    let context = NumericContext::default();
    let mut aliases = Aliases(BTreeMap::new());
    for layer in 0..args.state_layer_count() {
        let block =
            decoder::TransformerBlock::<NumericBackend>::new(&args, layer, &context).unwrap();
        block.visit_parameters(&mut aliases);
    }
    for group in &descriptor.components {
        assert_eq!(aliases.0[&group.write_weight], group.shared_write_weight);
        for read in &group.reads {
            assert_eq!(aliases.0[&read.weight], read.shared_weight);
        }
    }
    assert_eq!(
        aliases.0["model.layers.1.output_norm.weight"],
        "model.norm.weight"
    );
    assert_ne!(
        descriptor.components[0].activation,
        descriptor.components[4].activation
    );
}

#[test]
fn component_query_key_normalizations_match_constructed_head_equations() {
    use eredu_core::component::{ComponentNormalizationKind, ComponentReadRole};
    fn verify<C: decoder::Config>(args: &C, config: &serde_json::Value, independent: bool) {
        let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let group = &descriptor.components[0];
        let context = NumericContext::default();
        let mut attention = decoder::Attention::<NumericBackend>::new(args, 0, &context).unwrap();
        for (role, actual) in [
            (
                ComponentReadRole::Query,
                attention.query_norm.as_mut().unwrap(),
            ),
            (ComponentReadRole::Key, attention.key_norm.as_mut().unwrap()),
        ] {
            let read = group.reads.iter().find(|read| read.role == role).unwrap();
            let declared = read.head_normalization.as_ref().unwrap();
            assert_eq!(declared.independent_gains, independent);
            assert_eq!(declared.head_width, args.head_dim() as usize);
            assert_eq!(
                declared.heads,
                if role == ComponentReadRole::Query {
                    args.num_attention_heads()
                } else {
                    args.num_key_value_heads()
                } as usize
            );
            assert_eq!(declared.normalization.kind, ComponentNormalizationKind::Rms);
            assert_eq!(
                declared.normalization.gain.as_deref(),
                Some(actual.metadata.id.as_str())
            );
            assert_eq!(declared.normalization.groups, actual.groups);
            assert_eq!(declared.normalization.epsilon.value(), actual.epsilon);
            assert_eq!(declared.normalization.gain_offset.value(), actual.offset);
            let width = declared.head_width;
            let heads = declared.heads;
            assert_eq!(
                actual.weight.data.len(),
                width * if independent { heads } else { 1 }
            );
            for (index, gain) in actual.weight.data.iter_mut().enumerate() {
                *gain = 0.6 + index as f32 * 0.11;
            }
            let values: Vec<f32> = (0..2 * heads * width)
                .map(|i| (0.31 * i as f32 + 0.2).sin())
                .collect();
            let shape = if independent {
                vec![1, 2, (heads * width) as i32]
            } else {
                vec![1, 2, heads as i32, width as i32]
            };
            let input = NumericTensor::new(shape, values.clone());
            let normalized = actual.forward(&input, &context).unwrap();
            let mut expected = Vec::new();
            for (row, values) in values.chunks_exact(width).enumerate() {
                let denominator = (values
                    .iter()
                    .map(|value| (*value as f64).powi(2))
                    .sum::<f64>()
                    / width as f64
                    + declared.normalization.epsilon.value() as f64)
                    .sqrt();
                for (channel, value) in values.iter().enumerate() {
                    let gain_index = if independent {
                        (row % heads) * width + channel
                    } else {
                        channel
                    };
                    let gain = actual.weight.data[gain_index] as f64
                        + declared.normalization.gain_offset.value() as f64;
                    expected.push((*value as f64 * gain / denominator) as f32);
                }
            }
            assert_tensor_close(
                &normalized,
                &NumericTensor::new(input.shape.clone(), expected),
                "declared query/key head normalization",
            );
        }
        assert!(group
            .reads
            .iter()
            .filter(|read| !matches!(read.role, ComponentReadRole::Query | ComponentReadRole::Key))
            .all(|read| read.head_normalization.is_none()));
    }
    let qwen_config = config("qwen3", false);
    let qwen_args = qwen::model_args_from_config_value(&qwen_config).unwrap();
    verify(&qwen_args, &qwen_config, false);
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/k2_horizon/reference.json")).unwrap();
    let mut k2_config = fixture["dense"]["config"].clone();
    k2_config["query_key_norm"] = true.into();
    let k2_args =
        eredu_architectures::k2_horizon::model_args_from_config_value(&k2_config).unwrap();
    verify(&k2_args, &k2_config, true);
}

pub(super) fn lfm2_component_config() -> serde_json::Value {
    serde_json::json!({
        "model_type":"lfm2", "vocab_size":16, "hidden_size":4,
        "intermediate_size":8, "num_hidden_layers":3,
        "num_attention_heads":2, "num_key_value_heads":1,
        "max_position_embeddings":32, "layer_types":["conv","full_attention","conv"],
        "conv_L_cache":3, "block_multiple_of":2, "block_ffn_dim_multiplier":1.0,
        "block_auto_adjust_ff_dim":true, "tie_word_embeddings":true
    })
}

#[test]
fn lfm2_components_preserve_mixed_state_and_reconstruct_residual_writes() {
    let config = lfm2_component_config();
    let args = lfm2::model_args_from_config_value(&config).unwrap();
    let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    assert_eq!(descriptor.components.len(), 4);
    let context = NumericContext::default();
    let model = || lfm2::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let state = || {
        DeviceState::<NumericBackend, _>::create(lfm2::state_layout(&args).unwrap(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    let mut ordinary = ResidentRuntime::new(model(), &context).unwrap();
    let mut os = state();
    let mut baselines = vec![];
    for masked in [false, true] {
        let mut resident = ResidentRuntime::new(model(), &context).unwrap();
        let mut rebuilt = LayerwiseRuntime::new(model(), RebuildingUnitPolicy::default());
        let (mut rs, mut ls) = (state(), state());
        let mut position = 0;
        for (step, ids) in [vec![1, 3, 2], vec![4], vec![5]].into_iter().enumerate() {
            let tokens = NumericTensor::token_ids(&ids);
            let input = || decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            };
            let capture = || Components {
                zero: masked.then_some(("model.layers.1.attention.channels", ids.len() - 1, 1)),
                keep: masked.then_some(("model.layers.2.feed_forward.units", ids.len() - 1, 2)),
                ..Default::default()
            };
            let (mut rc, mut lc) = (capture(), capture());
            let (r, _) = resident
                .forward_with_traversal_hook(input(), &mut rs, &context, &mut rc)
                .unwrap();
            let l = rebuilt
                .forward_with_observer(input(), &mut ls, &context, &mut lc)
                .unwrap();
            assert_tensor_exact(&r, &l, "LFM2 observed resident/rebuilt execution");
            for (path, value) in &rc.values {
                assert_tensor_exact(value, &lc.values[path], path);
            }
            if !masked {
                let baseline = ordinary.forward(input(), &mut os, &context).unwrap();
                assert_tensor_exact(&r, &baseline, "LFM2 observation does not change execution");
                baselines.push(rc.values.clone());
            }
            let mut residual = rc.values["readout.embedding.effective"].clone();
            for layer in 0..3 {
                let mut block = lfm2::Block::<NumericBackend>::new(&args, layer, &context).unwrap();
                let boundary = if layer == 1 { "attention" } else { "mixer" };
                if let lfm2::TokenMixer::Attention(attention) = &mut block.mixer {
                    let expected = attention
                        .output
                        .forward(
                            &rc.values
                                [&format!("model.layers.{layer}.attention.channels.effective")],
                            &context,
                        )
                        .unwrap();
                    assert_tensor_close(
                        &expected,
                        &rc.values[&format!("model.layers.{layer}.attention.write")],
                        "LFM2 attention component sum",
                    );
                }
                let lfm2::FeedForward::Dense(ffn) = &mut block.feed_forward else {
                    panic!("dense fixture");
                };
                let units =
                    &rc.values[&format!("model.layers.{layer}.feed_forward.units.effective")];
                assert!(units.data.iter().any(|value| value.abs() > 1e-7));
                let expected = ffn.down.forward(units, &context).unwrap();
                assert_tensor_close(
                    &expected,
                    &rc.values[&format!("model.layers.{layer}.feed_forward.write")],
                    "LFM2 feed-forward component sum",
                );
                for suffix in [boundary, "feed_forward"] {
                    residual = residual
                        .add(
                            &rc.values[&format!("model.layers.{layer}.{suffix}.output.effective")],
                            &context,
                        )
                        .unwrap();
                    assert_tensor_close(
                        &residual,
                        &rc.values[&format!("model.layers.{layer}.{suffix}.residual")],
                        "LFM2 declared residual order",
                    );
                }
            }
            assert_tensor_close(
                &residual,
                &rc.values["readout.residual"],
                "embedding plus all component and convolution writes",
            );
            let readout = descriptor.component_readout.as_ref().unwrap();
            assert_eq!(
                readout.normalization.gain.as_deref(),
                Some("model.embedding_norm.weight")
            );
            let mut final_model = model();
            let gain = &final_model.static_modules().norm.weight.data;
            let expected_normalized: Vec<f32> = residual
                .data
                .chunks_exact(4)
                .flat_map(|row| {
                    let denominator = (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 4.0
                        + readout.normalization.epsilon.value() as f64)
                        .sqrt();
                    row.iter()
                        .zip(gain)
                        .map(move |(v, gain)| (*v as f64 * *gain as f64 / denominator) as f32)
                })
                .collect();
            assert_tensor_close(
                &NumericTensor::new(residual.shape.clone(), expected_normalized),
                &rc.values["readout.normalized"],
                "declared LFM2 final RMS normalization",
            );
            let expected = final_model
                .static_modules_mut()
                .embeddings
                .as_linear(&rc.values["readout.normalized.effective"], &context)
                .unwrap();
            assert_tensor_close(&expected, &r, "LFM2 tied readout decomposition stage");
            if masked {
                let units = &rc.values["model.layers.2.feed_forward.units.effective"];
                let width = args.dense_intermediate_size as usize;
                for i in 0..width {
                    if i != 2 {
                        assert_eq!(units.data[(ids.len() - 1) * width + i], 0.0);
                    }
                }
                let effective = &rc.values["model.layers.1.attention.channels.effective"];
                assert_eq!(effective.data[(ids.len() - 1) * 4 + 1], 0.0);
                assert_ne!(
                    units.data[(ids.len() - 1) * width + 2],
                    baselines[step]["model.layers.2.feed_forward.units"].data
                        [(ids.len() - 1) * width + 2],
                    "survivor must be recomputed from changed residual"
                );
            }
            position += ids.len() as i32;
            for layer in 0..3 {
                assert_eq!(rs.layer(layer).unwrap().position(), position);
                assert_eq!(ls.layer(layer).unwrap().position(), position);
            }
        }
    }
}

#[test]
fn lfm2_component_observers_preserve_routed_provider_events() {
    let mut config = lfm2_component_config();
    config["model_type"] = "lfm2_moe".into();
    config["num_dense_layers"] = 1.into();
    config["moe_intermediate_size"] = 6.into();
    config["num_experts"] = 2.into();
    config["num_experts_per_tok"] = 1.into();
    let args = lfm2::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let model = || lfm2::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let state = || {
        DeviceState::<NumericBackend, _>::create(lfm2::state_layout(&args).unwrap(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    #[derive(Default)]
    struct RoutedComponents {
        components: Components,
        paths: Vec<String>,
    }
    impl ActivationObserver<NumericTensor, Error> for RoutedComponents {
        fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
            self.components.observe(path, value)
        }
        fn observe_routing(
            &mut self,
            routing: eredu_runtime::RoutingObservation<'_, NumericTensor>,
        ) -> Result<(), Error> {
            self.paths.push(routing.path.into());
            assert!(routing
                .coefficients
                .data
                .iter()
                .all(|v| *v > 0.0 && v.is_finite()));
            assert!(routing.routed_output.data.iter().any(|v| v.abs() > 1e-7));
            Ok(())
        }
    }
    let mut observed = LayerwiseRuntime::new(model(), RebuildingUnitPolicy::default());
    let mut ordinary = ResidentRuntime::new(model(), &context).unwrap();
    let (mut observed_state, mut ordinary_state) = (state(), state());
    for ids in [vec![1, 4, 2], vec![3]] {
        let tokens = NumericTensor::token_ids(&ids);
        let input = || decoder::LayeredInput {
            tokens: &tokens,
            mask: None,
        };
        let mut capture = RoutedComponents::default();
        let actual = observed
            .forward_with_observer(input(), &mut observed_state, &context, &mut capture)
            .unwrap();
        let expected = ordinary
            .forward(input(), &mut ordinary_state, &context)
            .unwrap();
        assert_tensor_exact(&actual, &expected, "nested component/routing observers");
        assert_eq!(capture.paths.len(), 2);
        for layer in [1, 2] {
            let points = args
                .routed_observation_points(&format!("model.layers.{layer}"), layer, None)
                .unwrap()
                .unwrap();
            let path = points
                .bank(eredu_runtime::RoutedBankId::new(0))
                .unwrap()
                .path();
            assert!(capture.paths.iter().any(|p| p == path));
        }
        assert!(capture
            .components
            .values
            .contains_key("model.layers.0.feed_forward.units"));
        assert!(capture
            .components
            .values
            .contains_key("model.layers.1.attention.channels"));
        assert!(!capture
            .components
            .values
            .contains_key("model.layers.1.feed_forward.units"));
        assert!(capture
            .components
            .values
            .contains_key("model.layers.2.mixer.residual"));
    }
}

#[test]
fn lfm2_published_configuration_preserves_legacy_width_and_attention_schedule() {
    let released: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/lfm2/released-config.json")).unwrap();
    let args = lfm2::model_args_from_config_value(&released).unwrap();
    assert_eq!(args.hidden_size, 1024);
    assert_eq!(args.dense_intermediate_size, 4608);
    assert_eq!(args.num_hidden_layers, 16);
    let attention = args
        .layer_schedule
        .iter()
        .enumerate()
        .filter_map(|(layer, policy)| {
            matches!(policy.operator, lfm2::OperatorPolicy::SelfAttention(_)).then_some(layer)
        })
        .collect::<Vec<_>>();
    assert_eq!(attention, [2, 5, 8, 10, 12, 14]);
    let mut modern = released.clone();
    let object = modern.as_object_mut().unwrap();
    let width = object.remove("block_ff_dim").unwrap();
    object.insert("intermediate_size".into(), width);
    object.remove("full_attn_idxs");
    object.insert(
        "layer_types".into(),
        serde_json::json!((0..16)
            .map(|layer| {
                if attention.contains(&layer) {
                    "full_attention"
                } else {
                    "conv"
                }
            })
            .collect::<Vec<_>>()),
    );
    let normalized = lfm2::model_args_from_config_value(&modern).unwrap();
    assert_eq!(
        args.dense_intermediate_size,
        normalized.dense_intermediate_size
    );
    assert_eq!(
        args.layer_schedule_fingerprint(),
        normalized.layer_schedule_fingerprint()
    );
    assert_eq!(
        lfm2::state_layout(&args).unwrap(),
        lfm2::state_layout(&normalized).unwrap()
    );
    let descriptor = |config: &serde_json::Value| {
        eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor()
    };
    assert_eq!(descriptor(&released), descriptor(&modern));
    for indices in [
        serde_json::json!([-1]),
        serde_json::json!([16]),
        serde_json::json!([2, 2]),
    ] {
        let mut invalid = released.clone();
        invalid["full_attn_idxs"] = indices;
        assert!(lfm2::model_args_from_config_value(&invalid).is_err());
    }
    let mut conflicting = modern.clone();
    conflicting["full_attn_idxs"] = serde_json::json!([1]);
    assert!(lfm2::model_args_from_config_value(&conflicting).is_err());
}

pub(super) fn nemotron_mixed_config() -> serde_json::Value {
    let mut config = nemotron_config();
    config["num_hidden_layers"] = 4.into();
    config["hybrid_override_pattern"] = "M*E-".into();
    config
}

#[test]
fn nemotron_shared_units_join_parameters_and_reconstruct_the_sparse_branch() {
    let config = nemotron_mixed_config();
    let args = nemotron_h::model_args_from_config_value(&config).unwrap();
    let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    let group = graph
        .components
        .iter()
        .find(|g| g.id == "decoder.layers.2.operator.shared.units")
        .unwrap();
    let context = NumericContext::default();
    let mut base = nemotron_h::SparseMoe::<NumericBackend>::new(
        &args,
        2,
        args.moe_intermediate_size,
        args.moe_shared_expert_intermediate_size,
        &context,
    )
    .unwrap();
    let read = &mut base.shared_experts.up_proj;
    for (i, value) in read.weight.data.iter_mut().enumerate() {
        *value = 0.03 + (i % 11) as f32 * 0.007;
    }
    for (i, value) in read.bias.as_mut().unwrap().0.data.iter_mut().enumerate() {
        *value = 0.1 + i as f32 * 0.01;
    }
    let write = &mut base.shared_experts.down_proj;
    for (i, value) in write.weight.data.iter_mut().enumerate() {
        *value = (i as f32 * 0.3).sin() * 0.2;
    }
    for (i, value) in write.bias.as_mut().unwrap().0.data.iter_mut().enumerate() {
        *value = 0.01 * (i + 1) as f32;
    }
    let (read, write) = (&base.shared_experts.up_proj, &base.shared_experts.down_proj);
    assert_eq!(group.reads[0].weight, read.weight_metadata.id.as_str());
    assert_eq!(
        group.reads[0].bias.as_deref(),
        Some(read.bias.as_ref().unwrap().1.id.as_str())
    );
    assert_eq!(group.write_weight, write.weight_metadata.id.as_str());
    assert_eq!(
        group.write_bias.as_deref(),
        Some(write.bias.as_ref().unwrap().1.id.as_str())
    );
    let input = NumericTensor::new(
        vec![1, 3, 8],
        (0..24).map(|i| 0.2 + i as f32 * 0.013).collect(),
    );
    struct Observe(Components);
    impl ActivationObserver<NumericTensor, Error> for Observe {
        fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
            assert!(
                !self.0.values.contains_key(path),
                "duplicate observation {path}"
            );
            self.0.observe(path, value)
        }
        fn intervene(
            &mut self,
            path: &str,
            value: &NumericTensor,
        ) -> Result<Option<NumericTensor>, Error> {
            self.0.intervene(path, value)
        }
        fn observe_routing(
            &mut self,
            event: eredu_runtime::RoutingObservation<'_, NumericTensor>,
        ) -> Result<(), Error> {
            event.for_each_tensor(|path, value| self.observe(&path, value).unwrap());
            Ok(())
        }
    }
    let ordinary = base.clone().forward(&input, &context).unwrap();
    let mut baseline = None;
    for mode in 0..3 {
        let mut observer = Observe(Components {
            zero: match mode {
                1 => Some(("model.layers.2.shared.feed_forward.units", 1, 1)),
                2 => Some(("model.layers.2.shared.feed_forward.input", 1, 2)),
                _ => None,
            },
            keep: (mode == 2).then_some(("model.layers.2.shared.feed_forward.units", 1, 1)),
            ..Default::default()
        });
        let output = base
            .clone()
            .forward_observed_with_provider(
                "model.layers.2.routing",
                args.n_routed_experts,
                &input,
                eredu_runtime::ExpertPass::Prefill,
                &context,
                &mut observer,
                &mut eredu_runtime::ResidentExpertProvider,
            )
            .unwrap();
        let values = &observer.0.values;
        let units = &values[&group.activation];
        let effective = &values[&group.effective_activation];
        assert!(units.data.iter().all(|value| *value > 0.0));
        let normalized = &values[&format!("{}.effective", group.input)];
        let mut reconstructed = vec![0.; 24];
        for row in 0..3 {
            for component in 0..group.count {
                let read_row = group.reads[0].rows.row(component).unwrap();
                let projected = (0..8)
                    .map(|i| {
                        normalized.data[row * 8 + i] as f64
                            * read.weight.data[read_row * 8 + i] as f64
                    })
                    .sum::<f64>()
                    + read.bias.as_ref().unwrap().0.data[component] as f64;
                assert!(
                    (units.data[row * group.count + component] as f64 - projected.max(0.).powi(2))
                        .abs()
                        < 2e-6
                );
                assert_eq!(
                    effective.data[row * group.count + component],
                    if row == 1 && ((mode == 1 && component == 1) || (mode == 2 && component != 1))
                    {
                        0.
                    } else {
                        units.data[row * group.count + component]
                    }
                );
            }
            for out in 0..8 {
                reconstructed[row * 8 + out] = ((0..group.count)
                    .map(|component| {
                        effective.data[row * group.count + component] as f64
                            * write.weight.data[out * group.count + component] as f64
                    })
                    .sum::<f64>()
                    + write.bias.as_ref().unwrap().0.data[out] as f64)
                    as f32;
            }
        }
        assert_tensor_close(
            &NumericTensor::new(vec![1, 3, 8], reconstructed),
            &values[group.write_output.as_ref().unwrap()],
            "shared unit writes plus affine bias",
        );
        let shared = &values["model.layers.2.shared.feed_forward.output.effective"];
        assert_tensor_exact(
            shared,
            &values
                [&eredu_core::RoutingObservationField::SharedOutput.path("model.layers.2.routing")],
            "routing uses effective shared output",
        );
        assert_tensor_exact(
            &output,
            &values
                [&eredu_core::RoutingObservationField::RoutedOutput.path("model.layers.2.routing")]
                .add(shared, &context)
                .unwrap(),
            "shared contribution included once",
        );
        if mode == 0 {
            assert_tensor_exact(&ordinary, &output, "disabled shared hooks");
            baseline = Some(units.clone());
        } else {
            assert_eq!(&ordinary.data[..8], &output.data[..8]);
            assert_eq!(&ordinary.data[16..], &output.data[16..]);
            assert!(ordinary.data[8..16]
                .iter()
                .zip(&output.data[8..16])
                .any(|(a, b)| (a - b).abs() > 1e-6));
            if mode == 2 {
                assert_ne!(
                    effective.data[group.count + 1],
                    baseline.as_ref().unwrap().data[group.count + 1],
                    "survivor recomputed from changed shared input"
                );
            }
        }
    }
}

#[test]
fn nemotron_mixed_writes_reconstruct_residual_and_preserve_routing_interventions() {
    let args = nemotron_h::model_args_from_config_value(&nemotron_mixed_config()).unwrap();
    let context = NumericContext::default();
    let model = || nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let state = || {
        DeviceState::<NumericBackend, _>::create(
            nemotron_h::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap()
    };
    let mut ordinary = ResidentRuntime::new(model(), &context).unwrap();
    for unit in ordinary.units_mut().iter_mut().flatten() {
        populate_mixed_component_unit(unit);
    }
    let mut ordinary_state = state();
    let mut baseline = vec![];
    for mode in 0..3 {
        let mut resident = ResidentRuntime::new(model(), &context).unwrap();
        for unit in resident.units_mut().iter_mut().flatten() {
            populate_mixed_component_unit(unit);
        }
        let mut rebuilt = LayerwiseRuntime::new(model(), MixedComponentUnitPolicy);
        let (mut rs, mut ls) = (state(), state());
        for (step, ids) in [vec![1, 4, 2], vec![3], vec![5]].into_iter().enumerate() {
            let tokens = NumericTensor::token_ids(&ids);
            let input = || nemotron_h::EmbeddedInput::target(&tokens, None);
            let capture = || Components {
                zero: match mode {
                    1 => Some(("model.layers.0.mixer.output", ids.len() - 1, 2)),
                    2 => Some(("model.layers.2.routing.output", ids.len() - 1, 2)),
                    _ => None,
                },
                keep: (mode != 0).then_some((
                    "model.layers.3.feed_forward.units",
                    ids.len() - 1,
                    1,
                )),
                ..Default::default()
            };
            let (mut rc, mut lc) = (capture(), capture());
            let (actual, _) = resident
                .forward_with_traversal_hook(input(), &mut rs, &context, &mut rc)
                .unwrap();
            let rebuilt = rebuilt
                .forward_with_observer(input(), &mut ls, &context, &mut lc)
                .unwrap();
            assert_tensor_exact(
                &actual,
                &rebuilt,
                "Nemotron mixed component residency parity",
            );
            for (path, value) in &rc.values {
                assert_tensor_exact(value, &lc.values[path], path);
            }
            if mode == 0 {
                let expected = ordinary
                    .forward(input(), &mut ordinary_state, &context)
                    .unwrap();
                assert_tensor_exact(
                    &actual,
                    &expected,
                    "disabled and enabled mixed instrumentation",
                );
                baseline.push(rc.values.clone());
            }
            let mut residual = rc.values["readout.embedding.effective"].clone();
            for (layer, boundary) in ["mixer", "attention", "feed_forward", "feed_forward"]
                .into_iter()
                .enumerate()
            {
                let output =
                    &rc.values[&format!("model.layers.{layer}.{boundary}.output.effective")];
                assert!(output.data.iter().any(|value| value.abs() > 1e-7), "nonzero write: mode={mode}, step={step}, layer={layer}, boundary={boundary}, values={:?}", output.data);
                residual = residual.add(output, &context).unwrap();
                assert_tensor_close(
                    &residual,
                    &rc.values[&format!("model.layers.{layer}.{boundary}.residual")],
                    "declared mixed residual writes",
                );
            }
            assert_tensor_close(
                &residual,
                &rc.values["readout.residual"],
                "embedding plus all component, Mamba and routed writes",
            );
            if mode != 0 {
                let units = &rc.values["model.layers.3.feed_forward.units.effective"];
                assert_ne!(
                    units.data[(ids.len() - 1) * 10 + 1],
                    baseline[step]["model.layers.3.feed_forward.units"].data
                        [(ids.len() - 1) * 10 + 1],
                    "survivor is recomputed"
                );
                for i in 0..10 {
                    if i != 1 {
                        assert_eq!(units.data[(ids.len() - 1) * 10 + i], 0.0);
                    }
                }
                let path = if mode == 1 {
                    "model.layers.0.mixer.output.effective"
                } else {
                    "model.layers.2.feed_forward.output"
                };
                let effective = &rc.values[path];
                assert_eq!(
                    effective.data[(ids.len() - 1) * 8 + 2],
                    0.0,
                    "whole write consumes the intervention at the requested row"
                );
                if step == 0 {
                    assert_eq!(&effective.data[..16], &baseline[step][path].data[..16]);
                }
            }
        }
    }
}

fn populate_mixed_component_unit<U: Parameterized<NumericTensor>>(unit: &mut U) {
    struct Populate;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate {
        fn visit_mut(
            &mut self,
            metadata: eredu_nn::ParameterMetadataView<'_>,
            value: &'a mut NumericTensor,
        ) {
            // Standalone recurrent parameters are intentionally unloaded by the
            // neutral constructor. Supply nonzero fixture values on every rebuild.
            let id = metadata.id().as_str();
            let base = if id.ends_with(".mamba.norm.weight") || id.ends_with(".mamba.D") {
                Some(0.9)
            } else if id.ends_with(".mamba.dt_bias") {
                Some(-1.5)
            } else if id.ends_with(".mamba.A_log") {
                Some(-0.3)
            } else if id.ends_with(".mamba.conv1d.weight") {
                Some(0.2)
            } else if id.ends_with(".mlp.up_proj.bias") {
                Some(0.4)
            } else {
                None
            };
            if let Some(base) = base {
                for (i, value) in value.data.iter_mut().enumerate() {
                    *value = base + (i % 7) as f32 * 0.017;
                }
            }
        }
    }
    unit.visit_parameters_mut(&mut Populate);
}

pub(super) struct MixedComponentUnitPolicy;
impl<U: Parameterized<NumericTensor>> LayerwisePolicy<NumericBackend, U>
    for MixedComponentUnitPolicy
{
    type Lease = RebuiltUnitLease<U>;
    type Error = &'static str;
    fn begin(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Self::Error> {
        Ok(())
    }
    fn acquire<E, F>(
        &mut self,
        _: usize,
        _: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&NumericContext) -> Result<U, E>,
    {
        let mut unit = build(context).map_err(LayerwiseAcquireError::Architecture)?;
        populate_mixed_component_unit(&mut unit);
        Ok(RebuiltUnitLease(unit))
    }
    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        _: usize,
        _: ExecutionUnitAddress,
        _: Self::Lease,
        _: &'a NumericTensor,
        _: StateValues,
        _: ContextValues,
        _: &NumericContext,
    ) -> Result<(), Self::Error>
    where
        NumericTensor: 'a,
        StateValues: Iterator<Item = &'a NumericTensor>,
        ContextValues: Iterator<Item = &'a NumericTensor>,
    {
        Ok(())
    }
    fn finish(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[test]
fn qwen_hybrid_components_preserve_mixed_state_and_interleaved_reads() {
    for config in heterogeneous_replicated_configs().into_iter().filter(|c| {
        matches!(
            c["model_type"].as_str(),
            Some("qwen3_next" | "qwen3_5_text")
        )
    }) {
        let args = qwen::hybrid::model_args_from_config_value(&config)
            .unwrap()
            .text;
        let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        assert_eq!(descriptor.components.len(), 4);
        let attention = descriptor
            .components
            .iter()
            .find(|g| {
                matches!(
                    g.activation_equation,
                    eredu_core::component::ComponentActivation::Attention { .. }
                )
            })
            .unwrap();
        let query = attention
            .reads
            .iter()
            .find(|r| r.role == eredu_core::component::ComponentReadRole::Query)
            .unwrap();
        let gate = attention
            .reads
            .iter()
            .find(|r| r.role == eredu_core::component::ComponentReadRole::OutputGate)
            .unwrap();
        assert_eq!(query.weight, gate.weight);
        assert_eq!(
            query
                .head_normalization
                .as_ref()
                .unwrap()
                .normalization
                .gain_offset
                .value(),
            1.0
        );
        for (component, query_rows, gate_row) in [
            (0, 0..2, 2),
            (1, 0..2, 3),
            (2, 4..6, 6),
            (3, 4..6, 7),
            (6, 12..14, 14),
            (7, 12..14, 15),
        ] {
            assert_eq!(query.rows.row_range(component), Some(query_rows));
            assert_eq!(gate.rows.row(component), Some(gate_row));
        }
        assert_eq!(
            descriptor
                .component_readout
                .as_ref()
                .unwrap()
                .other_writes
                .len(),
            0
        );

        let context = NumericContext::default();
        let model =
            || qwen::hybrid::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
        let state = || {
            DeviceState::<NumericBackend, _>::create(
                qwen::hybrid::state_layout(&args).unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap()
        };
        let mut ordinary = ResidentRuntime::new(model(), &context).unwrap();
        let mut os = state();
        let mut baselines = vec![];
        for masked in [false, true] {
            let mut resident = ResidentRuntime::new(model(), &context).unwrap();
            let mut rebuilt = LayerwiseRuntime::new(model(), RebuildingUnitPolicy::default());
            let (mut rs, mut ls) = (state(), state());
            let mut position = 0;
            for (step, ids) in [vec![1, 3, 2], vec![4], vec![5]].into_iter().enumerate() {
                let tokens = NumericTensor::token_ids(&ids);
                let input = || qwen::hybrid::EmbeddedInput::target(&tokens, None);
                let capture = || Components {
                    zero: masked.then_some(("model.layers.1.attention.channels", ids.len() - 1, 1)),
                    keep: masked.then_some(("model.layers.1.feed_forward.units", ids.len() - 1, 2)),
                    ..Default::default()
                };
                let (mut rc, mut lc) = (capture(), capture());
                let (r, _) = resident
                    .forward_with_traversal_hook(input(), &mut rs, &context, &mut rc)
                    .unwrap();
                let l = rebuilt
                    .forward_with_observer(input(), &mut ls, &context, &mut lc)
                    .unwrap();
                assert_tensor_exact(&r, &l, "Qwen hybrid observed resident/rebuilt execution");
                for (path, value) in &rc.values {
                    assert_tensor_exact(value, &lc.values[path], path);
                }
                if !masked {
                    let baseline = ordinary.forward(input(), &mut os, &context).unwrap();
                    assert_tensor_exact(
                        &r,
                        &baseline,
                        "Qwen hybrid observation does not change execution",
                    );
                    baselines.push(rc.values.clone());
                }
                let mut residual = rc.values["readout.embedding.effective"].clone();
                for layer in 0..2 {
                    let mut block =
                        qwen::hybrid::Block::<NumericBackend>::new(&args, layer, &context).unwrap();
                    let boundary = if layer == 1 { "attention" } else { "mixer" };
                    if let qwen::hybrid::TokenMixer::Attention(attention) = &mut block.mixer {
                        let expected = attention
                            .output
                            .forward(
                                &rc.values
                                    [&format!("model.layers.{layer}.attention.channels.effective")],
                                &context,
                            )
                            .unwrap();
                        assert_tensor_close(
                            &expected,
                            &rc.values[&format!("model.layers.{layer}.attention.write")],
                            "Qwen hybrid attention component sum",
                        );
                    }
                    let qwen::hybrid::FeedForward::Dense(ffn) = &mut block.feed_forward else {
                        panic!("dense fixture");
                    };
                    let units =
                        &rc.values[&format!("model.layers.{layer}.feed_forward.units.effective")];
                    assert!(units.data.iter().any(|value| value.abs() > 1e-7));
                    let expected = ffn.down.forward(units, &context).unwrap();
                    assert_tensor_close(
                        &expected,
                        &rc.values[&format!("model.layers.{layer}.feed_forward.write")],
                        "Qwen hybrid feed-forward component sum",
                    );
                    for suffix in [boundary, "feed_forward"] {
                        residual = residual
                            .add(
                                &rc.values
                                    [&format!("model.layers.{layer}.{suffix}.output.effective")],
                                &context,
                            )
                            .unwrap();
                        assert_tensor_close(
                            &residual,
                            &rc.values[&format!("model.layers.{layer}.{suffix}.residual")],
                            "Qwen hybrid declared residual order",
                        );
                    }
                }
                assert_tensor_close(
                    &residual,
                    &rc.values["readout.residual"],
                    "embedding plus all component and recurrent writes",
                );
                let readout = descriptor.component_readout.as_ref().unwrap();
                assert_eq!(
                    readout.normalization.gain.as_deref(),
                    Some("model.norm.weight")
                );
                let mut final_model = model();
                let gain = &final_model.static_modules().norm.weight.data;
                let expected_normalized: Vec<f32> = residual
                    .data
                    .chunks_exact(8)
                    .flat_map(|row| {
                        let denominator = (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                            / 8.0
                            + readout.normalization.epsilon.value() as f64)
                            .sqrt();
                        row.iter().zip(gain).map(move |(v, gain)| {
                            (*v as f64 * (*gain as f64 + 1.0) / denominator) as f32
                        })
                    })
                    .collect();
                assert_tensor_close(
                    &NumericTensor::new(residual.shape.clone(), expected_normalized),
                    &rc.values["readout.normalized"],
                    "declared Qwen hybrid final RMS normalization",
                );
                let expected = final_model
                    .static_modules_mut()
                    .lm_head
                    .as_mut()
                    .unwrap()
                    .forward(&rc.values["readout.normalized.effective"], &context)
                    .unwrap();
                assert_tensor_close(
                    &expected,
                    &r,
                    "Qwen hybrid untied readout decomposition stage",
                );
                if masked {
                    let units = &rc.values["model.layers.1.feed_forward.units.effective"];
                    let width = args.intermediate_size as usize;
                    for i in 0..width {
                        if i != 2 {
                            assert_eq!(units.data[(ids.len() - 1) * width + i], 0.0);
                        }
                    }
                    let effective = &rc.values["model.layers.1.attention.channels.effective"];
                    assert_eq!(effective.data[(ids.len() - 1) * 8 + 1], 0.0);
                    assert_ne!(
                        units.data[(ids.len() - 1) * width + 2],
                        baselines[step]["model.layers.1.feed_forward.units"].data
                            [(ids.len() - 1) * width + 2],
                        "survivor must be recomputed from changed residual"
                    );
                }
                position += ids.len() as i32;
                for layer in 0..2 {
                    assert_eq!(rs.layer(layer).unwrap().position(), position);
                    assert_eq!(ls.layer(layer).unwrap().position(), position);
                }
            }
        }
    }
}

#[test]
fn tensor_parallel_component_boundaries_match_global_masks_and_recomputed_survivors() {
    let mut config = config("llama", false);
    config["num_attention_heads"] = 4.into();
    config["num_key_value_heads"] = 2.into();
    config["head_dim"] = 2.into();
    let args = llama::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let block = decoder::TransformerBlock::<NumericBackend>::new(&args, 0, &context).unwrap();
    let groups = decoder::layer_parallel_parameter_groups(&block, &args, 0).unwrap();
    let input = NumericTensor::new(
        vec![1, 3, 8],
        (0..24).map(|i| (i as f32 * 0.4 - 2.0).sin()).collect(),
    );
    for case in 0..4 {
        let mut expected_capture = Components {
            zero: (case == 1 || case == 3).then_some(("model.layers.0.attention.channels", 1, 1)),
            keep: (case == 2 || case == 3).then_some(("model.layers.0.feed_forward.units", 1, 4)),
            ..Default::default()
        };
        let expected = forward(&mut block.clone(), &input, &mut expected_capture);
        let collective = NumericParallelGroup::new(2);
        let outputs = std::thread::scope(|scope| {
            let handles = (0..2)
                .map(|rank| {
                    let layout = numeric_local_layout(&groups, 2, rank).unwrap();
                    let local_args = llama::local_block_args(&args, 0, &layout).unwrap();
                    let context = NumericContext::with_local_layout(layout);
                    let mut local =
                        decoder::TransformerBlock::<NumericBackend>::new(&local_args, 0, &context)
                            .unwrap();
                    let input = input.clone();
                    let parallel = NumericParallelContext::new(rank, Arc::clone(&collective));
                    scope.spawn(move || {
                        let attention_width =
                            (local_args.num_attention_heads * local_args.head_dim) as usize;
                        let ffn_width = local_args.intermediate_size as usize;
                        let mut capture = Components {
                            zero: if case == 1 || case == 3 {
                                1usize
                                    .checked_sub(rank * attention_width)
                                    .filter(|i| *i < attention_width)
                                    .map(|i| ("model.layers.0.attention.channels", 1, i))
                            } else {
                                None
                            },
                            keep: (case == 2 || case == 3).then(|| {
                                let local_index = 4usize
                                    .checked_sub(rank * ffn_width)
                                    .filter(|i| *i < ffn_width)
                                    .unwrap_or(ffn_width);
                                ("model.layers.0.feed_forward.units", 1, local_index)
                            }),
                            ..Default::default()
                        };
                        let output = local
                            .forward_tensor_parallel_observed(
                                decoder::AttentionInput {
                                    hidden: &input,
                                    mask: None,
                                    cache: None::<&mut NumericCache>,
                                    allow_sliding_prefill: true,
                                    rotary_position: None,
                                },
                                &parallel,
                                &context,
                                &mut ComponentInstrumentation::new("model.layers.0", &mut capture),
                            )
                            .unwrap();
                        (
                            output,
                            capture,
                            parallel.trace(),
                            attention_width,
                            ffn_width,
                        )
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        for (rank, (output, capture, trace, attention_width, ffn_width)) in
            outputs.iter().enumerate()
        {
            assert_tensor_close(output, &expected, "TP component intervention output");
            assert_eq!(trace.len(), 2, "component hooks add no collectives");
            assert!(trace
                .iter()
                .all(|event| event.kind == NumericCollectiveKind::Sum));
            assert_eq!(capture.values.len(), expected_capture.values.len());
            for (path, value) in &capture.values {
                let width = if path.contains("attention.channels")
                    || path.ends_with("attention.write_input")
                {
                    Some(*attention_width)
                } else if path.contains("feed_forward.units")
                    || path.ends_with("feed_forward.write_input")
                {
                    Some(*ffn_width)
                } else {
                    None
                };
                let expected = &expected_capture.values[path];
                if let Some(width) = width {
                    let expected = expected.axis_slice(2, rank * width, (rank + 1) * width);
                    assert_tensor_close(value, &expected, path);
                } else {
                    assert_tensor_close(value, expected, path);
                }
            }
        }
    }
}

#[test]
fn tensor_parallel_runtime_preserves_component_interventions_through_cached_decode() {
    for tied in [false, true] {
        verify_tensor_parallel_component_decode(tied, 0, false);
    }
}

#[test]
fn tensor_parallel_readout_interventions_match_global_execution_through_cached_decode() {
    for tied in [false, true] {
        for readout in [1, 2] {
            verify_tensor_parallel_component_decode(tied, readout, false);
        }
    }
}

struct ComponentTraversalHook<'a>(&'a mut Components);

impl<C> eredu_runtime::LayeredTraversalHook<NumericBackend, C, Error>
    for ComponentTraversalHook<'_>
{
    fn observes_activations(&self) -> bool {
        true
    }
    fn observe_activation(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.0.observe(path, value)
    }
    fn observe_generated_activation(
        &mut self,
        path: &str,
        prototype: &NumericTensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<NumericTensor, Error>,
    ) -> Result<(), Error> {
        self.0.observe_generated(path, prototype, source, generate)
    }
    fn intervene_activation(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        self.0.intervene(path, value)
    }
}

#[test]
fn tensor_parallel_traversal_hooks_preserve_internal_components_through_cached_decode() {
    for tied in [false, true] {
        for readout in [0, 1] {
            verify_tensor_parallel_component_decode(tied, readout, true);
        }
    }
}

fn verify_tensor_parallel_component_decode(tied: bool, readout: usize, traversal: bool) {
    let mut config = config("llama", tied);
    config["num_attention_heads"] = 4.into();
    config["num_key_value_heads"] = 2.into();
    config["head_dim"] = 2.into();
    config["num_hidden_layers"] = 2.into();
    let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    let args = llama::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let architecture = llama::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut groups = decoder::static_parallel_parameter_groups::<NumericBackend>(
        &architecture.static_modules().embeddings,
        &architecture.static_modules().norm,
        architecture.static_modules().lm_head.as_ref(),
        "model",
    )
    .unwrap();
    for layer in 0..2 {
        groups.extend(
            decoder::layer_parallel_parameter_groups(
                &architecture.construct_unit(layer, &context).unwrap(),
                &args,
                layer,
            )
            .unwrap(),
        );
    }
    let mut state = DeviceState::<NumericBackend, _>::create(
        architecture.state_layout(None).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let units = (0..2)
        .map(|layer| architecture.construct_unit(layer, &context).unwrap())
        .collect();
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let inputs = [
        NumericTensor::token_ids(&[1, 2, 5]),
        NumericTensor::token_ids(&[3]),
        NumericTensor::token_ids(&[4]),
    ];
    let expected = inputs
        .iter()
        .enumerate()
        .map(|(step, tokens)| {
            let position = usize::from(step == 0);
            let (zero, keep) = match readout {
                0 => (
                    ("model.layers.0.attention.channels", position, 1),
                    ("model.layers.0.feed_forward.units", position, 7),
                ),
                1 => (
                    ("readout.embedding", position, 3),
                    ("readout.normalized", position, 2),
                ),
                2 => (
                    ("readout.residual", position, 4),
                    ("readout.linear", position, 7),
                ),
                _ => unreachable!(),
            };
            let mut capture = Components {
                zero: Some(zero),
                keep: Some(keep),
                ..Default::default()
            };
            let output = if traversal {
                runtime
                    .forward_with_traversal_hook(
                        decoder::LayeredInput { tokens, mask: None },
                        &mut state,
                        &context,
                        &mut ComponentTraversalHook(&mut capture),
                    )
                    .unwrap()
                    .0
            } else {
                runtime
                    .forward_with_observer(
                        decoder::LayeredInput { tokens, mask: None },
                        &mut state,
                        &context,
                        &mut capture,
                    )
                    .unwrap()
            };
            assert!(capture
                .values
                .contains_key("model.layers.0.attention.channels"));
            assert!(capture
                .values
                .contains_key("model.layers.1.feed_forward.units"));
            (output, capture)
        })
        .collect::<Vec<_>>();
    let collective = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = (0..2)
            .map(|rank| {
                let layout = numeric_local_layout(&groups, 2, rank).unwrap();
                let coordinates = |path: &str| {
                    let component = descriptor
                        .components
                        .iter()
                        .find(|component| component.activation == path)
                        .unwrap();
                    eredu_architectures::component_partition::derive_component_coordinates(
                        component,
                        layout.tensor(&component.write_weight).unwrap(),
                    )
                    .unwrap()
                };
                let attention_coordinates = coordinates("model.layers.0.attention.channels");
                let ffn_coordinates = coordinates("model.layers.0.feed_forward.units");
                let geometry = llama::local_geometry(&args, &layout).unwrap();
                let state_layout = geometry.state_layout().clone();
                let local = geometry.block(0).unwrap();
                let attention_width = (local.num_attention_heads * local.head_dim) as usize;
                let ffn_width = local.intermediate_size as usize;
                let context = NumericContext::with_local_layout(layout);
                let architecture = llama::LayeredModel::<NumericBackend>::new_parallel(
                    args.clone(),
                    geometry,
                    &context,
                )
                .unwrap();
                let inputs = inputs.clone();
                let parallel = NumericParallelContext::new(rank, Arc::clone(&collective));
                scope.spawn(move || {
                    let mut state =
                        DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
                            Ok::<_, Error>(NumericHybridLayerState::new(policy))
                        })
                        .unwrap();
                    let units = (0..2)
                        .map(|layer| architecture.construct_unit(layer, &context).unwrap())
                        .collect();
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let values = inputs
                        .iter()
                        .enumerate()
                        .map(|(step, tokens)| {
                            let position = usize::from(step == 0);
                            let mut capture = Components {
                                zero: attention_coordinates
                                    .global_to_local(1)
                                    .map(|i| ("model.layers.0.attention.channels", position, i)),
                                keep: Some((
                                    "model.layers.0.feed_forward.units",
                                    position,
                                    ffn_coordinates.global_to_local(7).unwrap_or(ffn_width),
                                )),
                                ..Default::default()
                            };
                            if readout == 1 {
                                capture.zero = Some(("readout.embedding", position, 3));
                                capture.keep = Some(("readout.normalized", position, 2));
                            } else if readout == 2 {
                                capture.zero = Some(("readout.residual", position, 4));
                                capture.keep = Some(("readout.linear", position, 7));
                            }
                            let output = if traversal {
                                runtime
                                    .forward_parallel_with_traversal_hook(
                                        decoder::LayeredInput { tokens, mask: None },
                                        &mut state,
                                        &parallel,
                                        &context,
                                        &mut ComponentTraversalHook(&mut capture),
                                    )
                                    .unwrap()
                                    .0
                            } else {
                                runtime
                                    .forward_parallel_with_observer(
                                        decoder::LayeredInput { tokens, mask: None },
                                        &mut state,
                                        &parallel,
                                        &context,
                                        &mut capture,
                                    )
                                    .unwrap()
                            };
                            (output, capture)
                        })
                        .collect::<Vec<_>>();
                    (values, attention_width, ffn_width, parallel.trace())
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (rank, (steps, attention_width, ffn_width, trace)) in outputs.iter().enumerate() {
        assert_eq!(
            trace.len(),
            18,
            "only ordinary embedding, two writes per layer and vocabulary collectives per step"
        );
        for ((output, capture), (expected_output, expected_capture)) in steps.iter().zip(&expected)
        {
            assert_tensor_close(output, expected_output, "observed TP runtime cached logits");
            for (path, expected) in &expected_capture.values {
                let width = if path.contains("attention.channels")
                    || path.ends_with("attention.write_input")
                {
                    Some(*attention_width)
                } else if path.contains("feed_forward.units")
                    || path.ends_with("feed_forward.write_input")
                {
                    Some(*ffn_width)
                } else {
                    None
                };
                let actual = capture
                    .values
                    .get(path)
                    .unwrap_or_else(|| panic!("missing internal runtime point {path}"));
                if let Some(width) = width {
                    assert_tensor_close(
                        actual,
                        &expected.axis_slice(2, rank * width, (rank + 1) * width),
                        path,
                    );
                } else {
                    assert_tensor_close(actual, expected, path);
                }
            }
        }
    }
}

#[test]
fn prepared_component_interventions_cross_pipeline_cuts_and_bounded_residency() {
    let mut config = config("llama", false);
    config["num_attention_heads"] = 4.into();
    config["num_key_value_heads"] = 2.into();
    config["head_dim"] = 2.into();
    config["num_hidden_layers"] = 2.into();
    let args = llama::model_args_from_config_value(&config).unwrap();
    verify_prepared_component_partitions::<_, decoder::DenseBlockFactory>(&config, args);
}

#[test]
fn prepared_qwen_components_cross_pipeline_cuts_and_bounded_residency() {
    let mut config = config("qwen3", true);
    config["num_attention_heads"] = 4.into();
    config["num_key_value_heads"] = 2.into();
    config["head_dim"] = 2.into();
    config["num_hidden_layers"] = 2.into();
    let args = qwen::model_args_from_config_value(&config).unwrap();
    verify_prepared_component_partitions::<_, qwen::QwenBlockFactory>(&config, args);
}

#[test]
fn prepared_gemma_components_preserve_final_softcap_across_partitions_and_residency() {
    let mut config = super::gemma2::tiny_config();
    config["num_attention_heads"] = 4.into();
    config["num_key_value_heads"] = 2.into();
    config["head_dim"] = 2.into();
    config["num_hidden_layers"] = 2.into();
    let args = eredu_architectures::gemma2::model_args_from_config_value(&config).unwrap();
    verify_prepared_component_partitions::<_, decoder::DenseBlockFactory>(&config, args);
}

#[test]
fn prepared_nanbeige_components_and_loop_normalization_cross_pipeline_cuts() {
    let mut config = super::nanbeige::tiny_config(false);
    config["head_dim"] = 2.into();
    let args = eredu_architectures::nanbeige::model_args_from_config_value(&config).unwrap();
    verify_prepared_component_partitions_with_reference::<_, decoder::DenseBlockFactory>(
        &config,
        args,
        |unit| unit.visit_parameters_mut(&mut super::nanbeige::FixtureAliases),
    );
}

#[test]
fn prepared_lfm2_components_and_convolution_cross_partitions_and_residency() {
    verify_prepared_lfm2_components(false);
}

#[test]
fn prepared_qwen_moe_attention_and_complete_expert_writes_cross_partitions_and_residency() {
    let mut config = config("qwen3_moe", false);
    config["num_attention_heads"] = 4.into();
    config["num_key_value_heads"] = 2.into();
    config["head_dim"] = 2.into();
    config["num_hidden_layers"] = 2.into();
    let args = qwen::model_args_from_config_value(&config).unwrap();
    let (root, fixture) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let descriptor = inspection.architecture_plan().architecture_descriptor();
    assert_eq!(descriptor.components.len(), 2);
    assert_eq!(
        descriptor
            .component_readout
            .as_ref()
            .unwrap()
            .other_writes
            .len(),
        2
    );
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let architecture =
        qwen::RoutedLayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let parameters = architecture
        .parameter_description(&context)
        .unwrap()
        .into_owned();
    let inputs = [
        NumericTensor::token_ids(&[1, 2, 5]),
        NumericTensor::token_ids(&[3]),
        NumericTensor::token_ids(&[4]),
    ];
    let targets = [
        None,
        Some("model.layers.0.attention.channels"),
        Some("model.layers.1.feed_forward.contribution"),
        Some("model.layers.1.feed_forward.input"),
        Some("readout.embedding"),
        Some("readout.normalized"),
        Some("readout.linear"),
    ]
    .map(|target| (target, vec![]))
    .into_iter()
    .chain([
        (
            None,
            vec![("model.layers.0.attention.channels", vec![1, 6], false)],
        ),
        (
            None,
            vec![("model.layers.1.attention.channels", vec![2, 5], true)],
        ),
    ])
    .collect::<Vec<_>>();
    struct Populate<'a>(&'a prepared_adapter::ParameterBits);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(
            &mut self,
            metadata: eredu_nn::ParameterMetadataView<'_>,
            value: &'a mut NumericTensor,
        ) {
            let (shape, bits) = &self.0[metadata.id().as_str()];
            assert_eq!(&value.shape, shape);
            value.data = bits.iter().map(|bits| f32::from_bits(*bits)).collect();
        }
    }
    let expected = targets
        .iter()
        .map(|(target, masks)| {
            let mut architecture =
                qwen::RoutedLayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
            architecture
                .static_modules_mut()
                .visit_parameters_mut(&mut Populate(&fixture));
            let mut state = DeviceState::<NumericBackend, _>::create(
                architecture.state_layout(None).unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap();
            let units = (0..2)
                .map(|layer| {
                    let mut unit = architecture.construct_unit(layer, &context).unwrap();
                    unit.visit_parameters_mut(&mut Populate(&fixture));
                    unit
                })
                .collect();
            let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
            inputs
                .iter()
                .enumerate()
                .map(|(step, tokens)| {
                    let mut observer = GlobalComponentObserver {
                        inner: NumericLifecycleObserver {
                            zero_path: target.map(str::to_owned),
                            ..Default::default()
                        },
                        values: BTreeMap::new(),
                        layout: None,
                        masks,
                        position: usize::from(step == 0),
                        projection_plan: None,
                        producer_receipts: None,
                        prediction: step as u64,
                    };
                    let output = runtime
                        .forward_with_observer(
                            decoder::LayeredInput { tokens, mask: None },
                            &mut state,
                            &context,
                            &mut observer,
                        )
                        .unwrap();
                    let output =
                        eredu_runtime::observe_model_logits(&mut observer, &output).unwrap();
                    let vocabulary = *output.shape.last().unwrap() as usize;
                    (
                        NumericTensor::new(
                            vec![1, 1, vocabulary as i32],
                            output.data[output.data.len() - vocabulary..].to_vec(),
                        ),
                        observer.values,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (index, trial) in expected.iter().enumerate().skip(1) {
        let difference = trial
            .iter()
            .zip(&expected[0])
            .flat_map(|(a, b)| a.0.data.iter().zip(&b.0.data).map(|(a, b)| (a - b).abs()))
            .fold(0.0_f32, f32::max);
        assert!(
            difference > 1e-5,
            "Qwen MoE intervention {index}, difference={difference}"
        );
    }
    verify_prepared_component_execution(
        &inspection,
        parameters,
        &inputs,
        &targets,
        &expected,
        |sources, context| {
            partitioned_adapter::routed(sources, context, Arc::new(AtomicUsize::new(0)), None)
        },
    );
}

#[test]
fn gpt_oss_sink_attention_and_biased_expert_terms_reconstruct_scores() {
    // Expanded scalar experts test equations; the native Ring fixture separately
    // binds the published packed MXFP4 source and checks every captured value.
    let config = serde_json::json!({
        "model_type":"gpt_oss", "hidden_size":32, "intermediate_size":64,
        "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":2,
        "head_dim":8, "vocab_size":17, "num_local_experts":4, "num_experts_per_tok":2,
        "rms_norm_eps":0.00001, "sliding_window":2, "max_position_embeddings":64,
        "rope_theta":150000.0, "quantization_config":{"quant_method":"mxfp4"},
        "swiglu_limit":7.0, "layer_types":["sliding_attention", "full_attention"]
    });
    let args = gpt_oss::model_args_from_config_value(&config).unwrap();
    let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    assert_eq!(descriptor.components.len(), 2);
    let readout = descriptor.component_readout.as_ref().unwrap();
    assert_eq!(readout.other_writes.len(), 2);
    let context = NumericContext::default();
    let model = || gpt_oss::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    for layer in 0..2 {
        let attention = &descriptor.components[layer];
        let node = descriptor.node(&attention.node_id).unwrap();
        assert_eq!(
            node.attention.as_ref().unwrap().sink_logits.as_deref(),
            Some(format!("model.layers.{layer}.self_attn.sinks").as_str())
        );
        assert!(attention.reads.iter().all(|read| read.bias.is_some()));
        assert_eq!(
            attention.write_bias.as_deref(),
            Some(format!("model.layers.{layer}.self_attn.o_proj.bias").as_str())
        );
        assert_eq!(
            readout.other_writes[layer].input.as_deref(),
            Some(format!("model.layers.{layer}.feed_forward.input").as_str())
        );
    }
    // Learned sinks must affect actual aggregation, not just discovery.
    let input = NumericTensor::new(
        vec![1, 3, 32],
        (0..96).map(|i| (i as f32 * 0.13).sin()).collect(),
    );
    let mut block = gpt_oss::new_block::<NumericBackend>(&args, 0, &context).unwrap();
    let mut no_sinks = block.clone();
    no_sinks.self_attention.sinks = None;
    let run_block = |block: &mut gpt_oss::TransformerBlock<NumericBackend>,
                     observer: &mut Components| {
        block
            .forward_observed(
                decoder::AttentionInput {
                    hidden: &input,
                    mask: None,
                    cache: None::<&mut NumericCache>,
                    allow_sliding_prefill: true,
                    rotary_position: None,
                },
                &context,
                &mut ComponentInstrumentation::new("model.layers.0", observer),
            )
            .unwrap()
    };
    let (mut with, mut without) = (Components::default(), Components::default());
    run_block(&mut block, &mut with);
    run_block(&mut no_sinks, &mut without);
    assert!(with.values["model.layers.0.attention.channels"]
        .data
        .iter()
        .zip(&without.values["model.layers.0.attention.channels"].data)
        .any(|(a, b)| (a - b).abs() > 1e-5));
    let mut ordinary = ResidentRuntime::new(model(), &context).unwrap();
    let new_state = || {
        DeviceState::<NumericBackend, _>::create(
            model().state_layout(None).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap()
    };
    let mut ordinary_state = new_state();
    let mut baseline = vec![];
    for masked in [false, true] {
        let mut runtime = ResidentRuntime::new(model(), &context).unwrap();
        let mut state = new_state();
        for (step, ids) in [vec![1, 2, 5], vec![3], vec![4]].into_iter().enumerate() {
            let tokens = NumericTensor::token_ids(&ids);
            let input = || decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            };
            let mut observer = Components {
                zero: masked.then_some(("model.layers.0.attention.channels", ids.len() - 1, 3)),
                keep: masked.then_some(("model.layers.1.attention.channels", ids.len() - 1, 7)),
                ..Default::default()
            };
            let (output, _) = runtime
                .forward_with_traversal_hook(input(), &mut state, &context, &mut observer)
                .unwrap();
            if !masked {
                let expected = ordinary
                    .forward(input(), &mut ordinary_state, &context)
                    .unwrap();
                assert_tensor_exact(&output, &expected, "GPT-OSS observation no-op");
                baseline.push(output.clone());
            } else {
                assert!(output
                    .data
                    .iter()
                    .zip(&baseline[step].data)
                    .any(|(a, b)| (a - b).abs() > 1e-5));
            }
            let values = &observer.values;
            let mut residual = values["readout.embedding.effective"].clone();
            for layer in 0..2 {
                let attention = &descriptor.components[layer];
                let mut block =
                    gpt_oss::new_block::<NumericBackend>(&args, layer, &context).unwrap();
                let write = block
                    .self_attention
                    .output
                    .forward(&values[&attention.effective_activation], &context)
                    .unwrap();
                assert_tensor_close(
                    &write,
                    &values[&format!("model.layers.{layer}.attention.write")],
                    "GPT-OSS component sum includes output bias once",
                );
                let whole = &readout.other_writes[layer];
                let expert_input = &values[whole.input.as_ref().unwrap()];
                assert!(expert_input.data.iter().any(|v| v.abs() > 1e-5));
                residual = residual
                    .add(&write, &context)
                    .unwrap()
                    .add(&values[&whole.effective_output], &context)
                    .unwrap();
            }
            assert_tensor_close(
                &residual,
                &values["readout.residual"],
                "attention, embedding and completed biased expert writes",
            );
            let final_model = model();
            let modules = final_model.static_modules();
            let gain = &modules.norm.weight.data;
            let head = modules.lm_head.as_ref().unwrap();
            let normalized = residual
                .data
                .chunks_exact(32)
                .flat_map(|row| {
                    let denominator = (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 32.0
                        + readout.normalization.epsilon.value() as f64)
                        .sqrt();
                    row.iter()
                        .zip(gain)
                        .map(move |(v, g)| (*v as f64 * *g as f64 / denominator) as f32)
                })
                .collect::<Vec<_>>();
            assert_tensor_close(
                &NumericTensor::new(residual.shape.clone(), normalized.clone()),
                &values["readout.normalized"],
                "declared final RMS equation",
            );
            let scores = normalized
                .chunks_exact(32)
                .flat_map(|row| {
                    head.weight.data.chunks_exact(32).map(move |weights| {
                        row.iter()
                            .zip(weights)
                            .map(|(v, w)| *v as f64 * *w as f64)
                            .sum::<f64>() as f32
                    })
                })
                .collect::<Vec<_>>();
            assert_tensor_close(
                &NumericTensor::new(output.shape.clone(), scores.clone()),
                &output,
                "GPT-OSS reconstructed full scores",
            );
            for (expected, actual) in scores.chunks_exact(17).zip(output.data.chunks_exact(17)) {
                assert!(((expected[2] - expected[7]) - (actual[2] - actual[7])).abs() < 2e-5);
            }
        }
    }
}

#[test]
fn prepared_lfm2_moe_components_and_complete_expert_writes_cross_partitions_and_residency() {
    verify_prepared_lfm2_components(true);
}

#[test]
fn prepared_lfm2_width_one_no_state_preserves_all_partition_and_residency_paths() {
    for routed in [false, true] {
        verify_prepared_lfm2_components_with_kernel(routed, 1);
    }
}

fn verify_prepared_lfm2_components(routed: bool) {
    verify_prepared_lfm2_components_with_kernel(routed, 3);
}

fn verify_prepared_lfm2_components_with_kernel(routed: bool, kernel: i32) {
    let mut config = lfm2_component_config();
    config["conv_L_cache"] = kernel.into();
    config["hidden_size"] = 8.into();
    config["intermediate_size"] = 12.into();
    config["num_attention_heads"] = 4.into();
    config["block_auto_adjust_ff_dim"] = false.into();
    config["num_hidden_layers"] = 4.into();
    config["layer_types"] = serde_json::json!(["conv", "full_attention", "conv", "full_attention"]);
    config["num_key_value_heads"] = 2.into();
    if routed {
        config["model_type"] = "lfm2_moe".into();
        config["num_dense_layers"] = 1.into();
        config["moe_intermediate_size"] = 6.into();
        config["num_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
    }
    let args = lfm2::model_args_from_config_value(&config).unwrap();
    let (root, fixture) =
        prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
            name.ends_with(".conv.conv.weight").then(|| {
                parameter(
                    &ParameterSpec::trainable(name).unwrap(),
                    shape.to_vec(),
                    false,
                )
                .map(|value| 16.0 * value)
            })
        });
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let inputs = [
        NumericTensor::token_ids(&[1, 2, 5]),
        NumericTensor::token_ids(&[3]),
        NumericTensor::token_ids(&[4]),
    ];
    let feed_forward_target = if routed {
        "model.layers.0.feed_forward.units"
    } else {
        "model.layers.2.feed_forward.units"
    };
    let mut targets = [
        None,
        Some("model.layers.0.mixer.output"),
        Some("model.layers.1.attention.channels"),
        Some("readout.embedding"),
        Some("readout.normalized"),
        Some("readout.linear"),
    ]
    .map(|target| (target, vec![]))
    .into_iter()
    .chain([
        (
            None,
            vec![("model.layers.1.attention.channels", vec![1], false)],
        ),
        (None, vec![(feed_forward_target, vec![1, 2], true)]),
        (
            None,
            vec![
                ("model.layers.1.attention.channels", vec![1, 3], true),
                (feed_forward_target, vec![0, 2], true),
            ],
        ),
    ])
    .collect::<Vec<_>>();
    if routed {
        targets.push((Some("model.layers.1.feed_forward.contribution"), vec![]));
    }
    let context = NumericContext {
        bind_checkpoint_values: routed,
        ..Default::default()
    };
    let parameters = lfm2::LayeredModel::<NumericBackend>::new(args.clone(), &context)
        .unwrap()
        .parameter_description(&context)
        .unwrap()
        .into_owned();
    struct Populate<'a>(&'a prepared_adapter::ParameterBits);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(
            &mut self,
            metadata: eredu_nn::ParameterMetadataView<'_>,
            value: &'a mut NumericTensor,
        ) {
            let (shape, bits) = &self.0[metadata.id().as_str()];
            assert_eq!(&value.shape, shape);
            value.data = bits.iter().map(|bits| f32::from_bits(*bits)).collect();
        }
    }
    let expected = targets
        .iter()
        .map(|(target, masks)| {
            let architecture =
                lfm2::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
            let mut state = DeviceState::<NumericBackend, _>::create(
                lfm2::state_layout(&args).unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap();
            let units = (0..4)
                .map(|layer| {
                    let mut unit =
                        lfm2::Block::<NumericBackend>::new(&args, layer, &context).unwrap();
                    unit.visit_parameters_mut(&mut Populate(&fixture));
                    unit
                })
                .collect();
            let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
            inputs
                .iter()
                .enumerate()
                .map(|(step, tokens)| {
                    let mut observer = GlobalComponentObserver {
                        inner: NumericLifecycleObserver {
                            zero_path: target.map(str::to_owned),
                            ..Default::default()
                        },
                        values: BTreeMap::new(),
                        layout: None,
                        masks,
                        position: usize::from(step == 0),
                        projection_plan: None,
                        producer_receipts: None,
                        prediction: step as u64,
                    };
                    let output = runtime
                        .forward_with_observer(
                            decoder::LayeredInput { tokens, mask: None },
                            &mut state,
                            &context,
                            &mut observer,
                        )
                        .unwrap();
                    let output =
                        eredu_runtime::observe_model_logits(&mut observer, &output).unwrap();
                    let vocabulary = *output.shape.last().unwrap() as usize;
                    (
                        NumericTensor::new(
                            vec![1, 1, vocabulary as i32],
                            output.data[output.data.len() - vocabulary..].to_vec(),
                        ),
                        observer.values,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (trial_index, trial) in expected.iter().enumerate().skip(1) {
        let difference = trial
            .iter()
            .zip(&expected[0])
            .flat_map(|(actual, baseline)| {
                actual
                    .0
                    .data
                    .iter()
                    .zip(&baseline.0.data)
                    .map(|(a, b)| (a - b).abs())
            })
            .fold(0.0_f32, f32::max);
        assert!(
            difference > 1e-5,
            "LFM2 intervention {trial_index} must affect fixture, difference={difference}"
        );
    }
    verify_prepared_component_execution(
        &inspection,
        parameters,
        &inputs,
        &targets,
        &expected,
        if routed {
            |sources, context| {
                partitioned_adapter::routed(sources, context, Arc::new(AtomicUsize::new(0)), None)
            }
        } else {
            partitioned_adapter::dense
        },
    );
}

/// The same global mask plan is used on every partition; only production
/// coordinate lowering changes its local scalar indices.
struct GlobalComponentObserver<'a> {
    inner: NumericLifecycleObserver,
    values: BTreeMap<String, NumericTensor>,
    layout: Option<&'a eredu_architectures::component_partition::ComponentPartitionLayout>,
    masks: &'a [(&'static str, Vec<u32>, bool)],
    position: usize,
    projection_plan: Option<&'a eredu_core::capture::AdmittedCapturePlan>,
    producer_receipts: Option<&'a [eredu_runtime::capture::partition::PartitionCaptureReceiptPlan]>,
    prediction: u64,
}

impl ActivationObserver<NumericTensor, Error> for GlobalComponentObserver<'_> {
    fn observe_replica(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.observe(path, value)
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.values.insert(path.into(), value.clone());
        if let (Some(plan), Some(layout)) = (self.projection_plan, self.layout) {
            let phase = if self.prediction == 0 {
                eredu_core::capture::CapturePhase::Prefill
            } else {
                eredu_core::capture::CapturePhase::Decode
            };
            for (selection_index, selection) in plan.plan().selections.iter().enumerate() {
                if selection.path != path {
                    continue;
                }
                let projection = layout
                    .project_capture(plan, selection_index, phase, self.prediction, 4)
                    .unwrap()
                    .expect("an emitted component invocation must be local");
                assert_eq!(
                    projection.local_shape(),
                    value
                        .shape
                        .iter()
                        .map(|extent| *extent as u64)
                        .collect::<Vec<_>>()
                );
                if let Some(receipts) = self.producer_receipts {
                    let rank = layout.topology().global_rank();
                    match receipts[selection_index].producer(rank) {
                        Some(producer) => assert_eq!(producer, &projection),
                        None => assert!(
                            receipts[selection_index]
                                .producers()
                                .any(|(owner, declared)| declared == &projection
                                    && (owner < rank
                                        || !layout.observation(path).unwrap().exports())),
                            "a replica must match its eligible or authoritative producer"
                        ),
                    }
                }
                let coordinates = layout.observation(path).unwrap().coordinates().unwrap();
                let axis = projection.axis();
                for fragment in projection.fragments() {
                    for index in 0..fragment.local().shape[axis] {
                        let local =
                            fragment.local().starts[axis] + index * fragment.local().strides[axis];
                        let destination = fragment.destination().starts[axis]
                            + index * fragment.destination().strides[axis];
                        let global = projection.global_slice().starts[axis]
                            + destination * projection.global_slice().strides[axis];
                        assert_eq!(
                            coordinates.local_to_global(local as usize),
                            Some(global as usize)
                        );
                    }
                }
            }
        }
        self.inner.observe(path, value)
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        use eredu_core::{
            capture::ResolvedCaptureSlice,
            component::ComponentCoordinateMap,
            intervention::{InterventionAction, InterventionDtype},
        };
        let Some((_, indices, keep_selected)) =
            self.masks.iter().find(|(target, _, _)| *target == path)
        else {
            return self.inner.intervene(path, value);
        };
        let full;
        let coordinates = if let Some(layout) = self.layout {
            layout
                .observation(path)
                .expect("declared global observation point")
                .coordinates()
                .expect("only owning pipeline stage emits the point")
        } else {
            full = ComponentCoordinateMap::range(
                *value.shape.last().unwrap() as usize,
                0..*value.shape.last().unwrap() as usize,
            )
            .unwrap();
            &full
        };
        assert_eq!(
            coordinates.local_count(),
            *value.shape.last().unwrap() as usize
        );
        let count = coordinates.global_count() as u64;
        let slice = ResolvedCaptureSlice {
            starts: vec![0, self.position as u64, 0],
            ends: vec![1, self.position as u64 + 1, count],
            strides: vec![1, 1, 1],
            shape: vec![1, 1, count],
        };
        let action = InterventionAction::MaskComponents {
            dtype: InterventionDtype::Float32,
            indices: indices.clone(),
            keep_selected: *keep_selected,
        };
        let (action, slice) =
            eredu_runtime::intervention::localize_component_mask(&action, &slice, coordinates)
                .unwrap();
        let InterventionAction::MaskComponents {
            indices,
            keep_selected,
            ..
        } = action
        else {
            unreachable!()
        };
        let mut result = value.clone();
        let width = coordinates.local_count();
        for sequence in slice.starts[1]..slice.ends[1] {
            for component in 0..width {
                if indices.contains(&(component as u32)) != keep_selected {
                    result.data[sequence as usize * width + component] = 0.0;
                }
            }
        }
        Ok(Some(result))
    }
}

fn component_partition_capture_plan(
    descriptor: &eredu_core::ArchitectureDescriptor,
) -> eredu_core::capture::AdmittedCapturePlan {
    use eredu_core::{capture::*, *};
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Slice],
        max_histogram_bins: 0,
        conditions: vec![],
    };
    let mut selections = descriptor
        .components
        .iter()
        .flat_map(|component| {
            [&component.activation, &component.effective_activation]
                .into_iter()
                .chain(component.write_input.iter())
                .map(move |path| CaptureSelection {
                    id: path.clone(),
                    path: path.clone(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![CaptureSlice {
                        axis: "component".into(),
                        start: 1,
                        end: component.count as u64,
                        stride: 2,
                    }],
                    transform: CaptureTransform::Slice,
                })
        })
        .collect::<Vec<_>>();
    let mut replicated = descriptor
        .components
        .iter()
        .map(|component| component.input.as_str())
        .collect::<Vec<_>>();
    for component in &descriptor.components {
        replicated.extend(component.write_output.as_deref());
        replicated.extend(component.output.as_deref());
        if let Some(gate) = &component.output_gate {
            replicated.extend([
                gate.input.as_str(),
                gate.projection_input.as_str(),
                gate.output.as_str(),
                gate.effective_output.as_str(),
            ]);
        }
        for stage in component
            .reads
            .iter()
            .flat_map(|read| &read.input_projections)
        {
            replicated.push(
                stage
                    .output
                    .strip_suffix(".effective")
                    .unwrap_or(&stage.output),
            );
        }
    }
    if let Some(readout) = &descriptor.component_readout {
        replicated.extend([
            readout.embedding.as_str(),
            readout.residual.as_str(),
            readout.normalized.as_str(),
            readout.linear_scores.as_str(),
            readout.logits.as_str(),
        ]);
        for normalization in &readout.block_normalizations {
            replicated.extend([normalization.input.as_str(), normalization.output.as_str()]);
        }
        replicated.extend(readout.other_writes.iter().flat_map(|write| {
            std::iter::once(write.output.as_str()).chain(write.input.as_deref())
        }));
    }
    let paths = replicated
        .into_iter()
        .flat_map(|path| [path.to_owned(), format!("{path}.effective")])
        .collect::<std::collections::BTreeSet<_>>();
    for point in descriptor
        .observations
        .points
        .iter()
        .filter(|point| paths.contains(&point.path))
    {
        let axis = point.axes.as_ref().unwrap().last().unwrap();
        let SymbolicDimension::Known(width) = axis.dimension else {
            panic!("known normalized/readout width")
        };
        selections.push(CaptureSelection {
            id: point.path.clone(),
            path: point.path.clone(),
            schedule: CaptureSchedule::default(),
            slices: vec![CaptureSlice {
                axis: axis.name.clone(),
                start: 1,
                end: width as u64,
                stride: 2,
            }],
            transform: CaptureTransform::Slice,
        });
    }
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: capabilities.clone(),
        points: selections
            .iter()
            .map(|selection| ObservationSupport {
                path: selection.path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            })
            .collect(),
    };
    let usage = CaptureUsage {
        captures: 1000,
        retained_bytes: 10_000_000,
        host_bytes: 10_000_000,
        encoded_bytes: 10_000_000,
    };
    CapturePlan {
        schema_version: 1,
        selections,
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &descriptor.observations,
        &support,
        &capabilities,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 3,
        },
    )
    .unwrap()
}

fn verify_prepared_component_partitions<C, P>(config: &serde_json::Value, args: C)
where
    C: decoder::Config + Clone + Sync,
    P: decoder::BlockFactory<NumericBackend, C>,
{
    verify_prepared_component_partitions_with_reference::<C, P>(config, args, |_| {});
}

fn verify_prepared_component_partitions_with_reference<C, P>(
    config: &serde_json::Value,
    args: C,
    bind_reference: impl Fn(&mut decoder::TransformerBlock<NumericBackend, P::FeedForward>),
) where
    C: decoder::Config + Clone + Sync,
    P: decoder::BlockFactory<NumericBackend, C>,
{
    let layer_count = args.num_hidden_layers() as usize;
    let (root, _) = prepared_adapter::payload_fixture_config(config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let inputs = [
        NumericTensor::token_ids(&[1, 2, 5]),
        NumericTensor::token_ids(&[3]),
        NumericTensor::token_ids(&[4]),
    ];
    let targets = [
        None,
        Some("model.layers.0.attention.channels"),
        Some("model.layers.1.feed_forward.units"),
        Some("readout.embedding"),
        Some("readout.normalized"),
        Some("readout.linear"),
    ]
    .map(|target| (target, vec![]))
    .into_iter()
    .chain([
        (
            None,
            vec![("model.layers.0.attention.channels", vec![1, 6], false)],
        ),
        (
            None,
            vec![("model.layers.1.feed_forward.units", vec![7], true)],
        ),
        (
            None,
            vec![
                ("model.layers.0.attention.channels", vec![6, 1], true),
                ("model.layers.1.feed_forward.units", vec![10, 1], true),
            ],
        ),
    ])
    .collect::<Vec<_>>();
    let expected = targets
        .iter()
        .map(|(target, masks)| {
            let context = NumericContext::default();
            let architecture =
                decoder::LayeredModel::<NumericBackend, C, P>::new(args.clone(), &context).unwrap();
            let mut state = DeviceState::<NumericBackend, _>::create(
                architecture.state_layout(None).unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap();
            let units = (0..layer_count)
                .map(|index| {
                    let mut unit = architecture.construct_unit(index, &context).unwrap();
                    bind_reference(&mut unit);
                    unit
                })
                .collect();
            let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
            inputs
                .iter()
                .enumerate()
                .map(|(step, tokens)| {
                    let mut observer = GlobalComponentObserver {
                        inner: NumericLifecycleObserver {
                            zero_path: target.map(str::to_owned),
                            ..Default::default()
                        },
                        values: BTreeMap::new(),
                        layout: None,
                        masks,
                        position: usize::from(step == 0),
                        projection_plan: None,
                        producer_receipts: None,
                        prediction: step as u64,
                    };
                    let output = runtime
                        .forward_with_observer(
                            decoder::LayeredInput { tokens, mask: None },
                            &mut state,
                            &context,
                            &mut observer,
                        )
                        .unwrap();
                    // The lower-level ordinary traversal returns the full
                    // scores; the session's final observation seam follows it.
                    let output =
                        eredu_runtime::observe_model_logits(&mut observer, &output).unwrap();
                    let vocabulary = *output.shape.last().unwrap() as usize;
                    (
                        NumericTensor::new(
                            vec![1, 1, vocabulary as i32],
                            output.data[output.data.len() - vocabulary..].to_vec(),
                        ),
                        observer.values,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if let Some(cap) = args.output_softcap() {
        for (_, values) in &expected[0] {
            let affine = &values["readout.linear"];
            let final_scores = &values[eredu_core::MODEL_LOGITS_OBSERVATION_PATH];
            assert!(
                affine
                    .data
                    .iter()
                    .zip(&final_scores.data)
                    .any(|(before, after)| (before - after).abs() > 1e-4),
                "nonlinear fixture must distinguish affine and final scores"
            );
            for (before, after) in affine.data.iter().zip(&final_scores.data) {
                assert!((cap * (before / cap).tanh() - after).abs() < 2e-6);
            }
        }
    }
    for trial in &expected[1..] {
        assert!(
            trial.iter().zip(&expected[0]).any(|(a, b)| a
                .0
                .data
                .iter()
                .zip(&b.0.data)
                .any(|(a, b)| (a - b).abs() > 1e-4)),
            "component deletion must have a nonzero downstream effect"
        );
    }
    verify_prepared_component_execution(
        &inspection,
        decoder::dense_parameter_description(&args).unwrap(),
        &inputs,
        &targets,
        &expected,
        partitioned_adapter::dense,
    );
}

fn verify_prepared_component_execution(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    parameters: eredu_runtime::ArchitectureParameterDescription,
    inputs: &[NumericTensor],
    targets: &[(Option<&'static str>, Vec<(&'static str, Vec<u32>, bool)>)],
    expected: &[Vec<(NumericTensor, BTreeMap<String, NumericTensor>)>],
    construct: fn(
        eredu_architectures::prepared_sources::PreparedModelSources,
        &NumericContext,
    ) -> Result<NumericPartitionExecutable, String>,
) {
    verify_prepared_component_execution_with_topologies(
        inspection,
        parameters,
        inputs,
        targets,
        expected,
        construct,
        &[
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            ParallelTopology::new(1, 2, 1, 1).unwrap(),
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
        ],
    );
}

fn verify_prepared_component_execution_with_topologies(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    parameters: eredu_runtime::ArchitectureParameterDescription,
    inputs: &[NumericTensor],
    targets: &[(Option<&'static str>, Vec<(&'static str, Vec<u32>, bool)>)],
    expected: &[Vec<(NumericTensor, BTreeMap<String, NumericTensor>)>],
    construct: fn(
        eredu_architectures::prepared_sources::PreparedModelSources,
        &NumericContext,
    ) -> Result<NumericPartitionExecutable, String>,
    topologies: &[ParallelTopology],
) {
    for residency in [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(1 << 20),
            host_budget_bytes: Some(1 << 20),
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 20,
            host_budget_bytes: 1 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ] {
        for &topology in topologies {
            let world = Arc::new(NumericPartitionWorld::default());
            // One retained rank-zero selection declares every other rank. Compare
            // it with each independently admitted rank and the actual forward hooks.
            let reference_plan = prepared_adapter::plan(None)
                .with_topology(topology)
                .with_residency(residency.clone());
            let reference_sources = partitioned_adapter::prepare_plan(
                &inspection,
                &reference_plan,
                0,
                std::time::Duration::from_secs(10),
            )
            .unwrap();
            let reference_selection = reference_sources.selected().execution().clone();
            let reference_descriptor = reference_sources.architecture().architecture_descriptor();
            let global_capture_plan = component_partition_capture_plan(&reference_descriptor);
            let reference_parameters = parameters.clone();
            let discovery =
                reference_sources.prepare_discovery(Default::default(), Default::default());
            assert!(!discovery.identity_is_resolved());
            assert_eq!(
                discovery.execution_identity(),
                reference_sources.execution_identity()
            );
            assert!(discovery
                .component_partition_layouts(topology.world_size())
                .is_err());
            assert!(discovery.clone().bind_partition_parameters(None).is_err());
            let wrong_graph = eredu_runtime::ExecutionGraph::chain(["another-model"]).unwrap();
            let wrong_units = eredu_runtime::ExecutionUnitLayout::new(&wrong_graph, [2]).unwrap();
            let wrong_parameters = eredu_runtime::ArchitectureParameterDescription::new(
                &wrong_graph,
                &wrong_units,
                [],
                [],
            )
            .unwrap();
            assert!(discovery
                .clone()
                .bind_partition_parameters(Some(Arc::new(wrong_parameters)))
                .is_err());
            let discovery = discovery
                .bind_partition_parameters(Some(Arc::new(reference_parameters.clone())))
                .unwrap();
            drop(reference_sources);
            assert!(reference_selection
                .component_partition_layouts(
                    &reference_descriptor,
                    &reference_parameters,
                    topology.world_size() - 1
                )
                .is_err());
            let expected_layouts = reference_selection
                .component_partition_layouts(
                    &reference_descriptor,
                    &reference_parameters,
                    topology.world_size(),
                )
                .unwrap()
                .unwrap();
            assert!(discovery
                .component_partition_layouts(topology.world_size() - 1)
                .is_err());
            let reference_layouts = discovery
                .component_partition_layouts(topology.world_size())
                .unwrap()
                .unwrap();
            assert_eq!(reference_layouts, expected_layouts);
            assert!(
                !discovery.identity_is_resolved(),
                "bound discovery must not hash weights or allocate native tensors"
            );
            let reference_selection = &reference_selection;
            let reference_layouts = &reference_layouts;
            let outputs = std::thread::scope(|scope| {
                let handles = (0..topology.world_size()).map(|rank| {
                    let world = Arc::clone(&world);
                    let inspection = inspection;
                    let parameters = &parameters;
                    let inputs = inputs;
                    let targets = targets;
                    let expected = expected;
                    let residency = residency.clone();
                    scope.spawn(move || {
                        use eredu_architectures::partitioned_execution::derive_partitioned_local_layout;
                        let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                        let plan = prepared_adapter::plan(None).with_topology(topology).with_residency(residency);
                        let sources = partitioned_adapter::prepare_plan(inspection, &plan, rank, std::time::Duration::from_secs(10)).unwrap();
                        let description = parameters.clone();
                        let descriptor = sources.architecture().architecture_descriptor();
                        let capture_plan = component_partition_capture_plan(&descriptor);
                        let component_layout = sources.selected().execution().component_partition_layout(&descriptor, &description).unwrap().unwrap();
                        assert_eq!(component_layout.topology(), rank_topology);
                        assert_eq!(reference_selection.component_partition_layout_for_rank(&descriptor, &description, rank).unwrap().unwrap(), component_layout,
                            "remote projection must match this rank's independently retained admission");
                        assert!(reference_selection.component_partition_layout_for_rank(&descriptor, &description, topology.world_size()).is_err());
                        assert_eq!(reference_layouts.rank(rank), Some(&component_layout));
                        verify_component_producer_limits(reference_layouts, &capture_plan, topology);
                        let mut shared_descriptor = descriptor.clone();
                        for component in &mut shared_descriptor.components {
                            if let Some(original) = descriptor.components.iter().find(|original| original.layer_index < component.layer_index && original.activation_equation == component.activation_equation) {
                                component.shared_write_weight = original.write_weight.clone();
                            }
                        }
                        assert_eq!(sources.selected().execution().component_partition_layout(&shared_descriptor, &description).unwrap().unwrap(), component_layout,
                            "shared source storage must not transfer invocation ownership");
                        if let Some(write) = descriptor.component_readout.as_ref().and_then(|readout| readout.other_writes.first()) {
                            let mut missing = descriptor.clone();
                            missing.nodes.iter_mut().find(|node| node.id == write.node_id).unwrap().parameter_groups.clear();
                            assert!(sources.selected().execution().component_partition_layout(&missing, &description).is_err(), "whole write without invocation ownership must reject");
                            let readout = descriptor.component_readout.as_ref().unwrap();
                            let embedding = descriptor.parameter_groups.iter().find(|group| readout.embedding_weight.strip_prefix(&group.canonical_prefix).is_some_and(|suffix| suffix.starts_with('.'))).unwrap();
                            let mut ambiguous = descriptor.clone();
                            ambiguous.nodes.iter_mut().find(|node| node.id == write.node_id).unwrap().parameter_groups.push(embedding.id.clone());
                            assert!(sources.selected().execution().component_partition_layout(&ambiguous, &description).is_err(), "static whole-write ownership must reject");
                            if let Some(other_write) = readout.other_writes.iter().find(|other| other.layer_index != write.layer_index) {
                                let other = descriptor.nodes.iter().find(|node| node.id == other_write.node_id).unwrap();
                                let mut ambiguous = descriptor.clone();
                                ambiguous.nodes.iter_mut().find(|node| node.id == write.node_id).unwrap().parameter_groups.extend(other.parameter_groups.clone());
                                assert!(sources.selected().execution().component_partition_layout(&ambiguous, &description).is_err(), "conflicting whole-write units must reject");
                            }
                        }
                        let mut publication_descriptor = descriptor.clone();
                        publication_descriptor.components.clear();
                        publication_descriptor.component_transforms.clear();
                        publication_descriptor.component_readout = None;
                        let publication_layout = sources.selected().execution().component_partition_layout(&publication_descriptor, &description).unwrap().unwrap();
                        assert_eq!(publication_layout.observation(eredu_core::MODEL_LOGITS_OBSERVATION_PATH), component_layout.observation(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
                            "final publication cannot depend on component/readout decomposition metadata");
                        let layout = derive_partitioned_local_layout(&description, rank_topology).unwrap();
                        let mut context = NumericContext::with_partition(layout, rank, world);
                        context.bind_checkpoint_values = true;
                        let mut executable = construct(sources, &context).unwrap();
                        targets.iter().enumerate().map(|(trial, (target, masks))| {
                            executable.reset().unwrap();
                            inputs.iter().enumerate().map(|(step, tokens)| {
                                let receipts = component_producer_receipts(reference_layouts, &capture_plan, step as u64);
                                let mut observer = GlobalComponentObserver {
                                    inner: NumericLifecycleObserver { zero_path: target.map(str::to_owned), ..Default::default() },
                                    values: BTreeMap::new(),
                                    layout: Some(&component_layout), masks, position: usize::from(step == 0),
                                    projection_plan: Some(&capture_plan), producer_receipts: Some(&receipts), prediction: step as u64,
                                };
                                let output = executable.forward_observed(tokens, step == 0, &mut observer).unwrap();
                                for component in &descriptor.components {
                                    let path = &component.activation;
                                    let mapped = component_layout.point(path).unwrap();
                                    assert_eq!(observer.inner.paths.contains(path), mapped.coordinates().is_some(), "rank {rank} invocation of {path}");
                                }
                                for (point, owner) in [
                                    ("readout.embedding", 0),
                                    ("readout.normalized", topology.pipeline() - 1),
                                    ("readout.projection_input", topology.pipeline() - 1),
                                    ("readout.linear", topology.pipeline() - 1),
                                ] {
                                    assert_eq!(observer.inner.paths.iter().filter(|path| path.as_str() == point).count(),
                                        usize::from(rank_topology.pipeline_parallel_rank() == owner),
                                        "rank {rank} ownership of {point}");
                                }
                                let mut summed_terms = BTreeMap::new();
                                for selection in &capture_plan.plan().selections {
                                    let placement = component_layout.observation(&selection.path).unwrap();
                                    let actual = observer.values.get(&selection.path);
                                    let Some(coordinates) = placement.coordinates() else {
                                        assert!(actual.is_none(), "inactive invocation {}", selection.path);
                                        continue;
                                    };
                                    if placement.combination() == eredu_core::capture::PartitionCaptureCombination::SumF64ToF32 {
                                        summed_terms.insert(selection.path.clone(), actual.expect("owned sum term").clone());
                                        continue;
                                    }
                                    let expected = expected[trial][step].1.get(&selection.path).unwrap_or_else(|| panic!("serial reference omitted {} at trial {trial}, step {step}", selection.path));
                                    let point = descriptor.observations.points.iter().find(|point| point.path == selection.path).unwrap();
                                    let axis = point.axes.as_ref().unwrap().iter().position(|axis| axis.name == placement.axis()).unwrap();
                                    let range = coordinates.contiguous_range().unwrap();
                                    let expected = expected.axis_slice(axis, range.start, range.end);
                                    assert_tensor_close(actual.expect("owned observation"), &expected, &selection.path);
                                }
                                (output, summed_terms)
                            }).collect::<Vec<_>>()
                        }).collect::<Vec<_>>()
                    })
                }).collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for (trial_index, trial) in expected.iter().enumerate() {
                for (step, (_, expected_values)) in trial.iter().enumerate() {
                    for (index, receipt) in component_producer_receipts(
                        reference_layouts,
                        &global_capture_plan,
                        step as u64,
                    )
                    .into_iter()
                    .enumerate()
                    {
                        if receipt.combination()
                            != eredu_core::capture::PartitionCaptureCombination::SumF64ToF32
                        {
                            continue;
                        }
                        let path = &global_capture_plan.plan().selections[index].path;
                        partition_sum::verify(
                            &global_capture_plan,
                            receipt,
                            |rank| &outputs[rank][trial_index][step].1[path],
                            &expected_values[path],
                        );
                    }
                }
            }
            for trials in outputs {
                for (actual, expected) in trials.iter().zip(expected) {
                    for (actual, expected) in actual.iter().zip(expected) {
                        assert_tensor_close(
                            &actual.0,
                            &expected.0,
                            "component intervention across prepared TP/PP/bounded traversal",
                        );
                    }
                }
            }
        }
    }
}

fn component_producer_receipts(
    layouts: &eredu_architectures::component_partition::ComponentPartitionLayouts,
    plan: &eredu_core::capture::AdmittedCapturePlan,
    prediction: u64,
) -> Vec<eredu_runtime::capture::partition::PartitionCaptureReceiptPlan> {
    use eredu_architectures::component_partition::ComponentCaptureProjectionRequest;
    use eredu_core::capture::*;
    use eredu_runtime::capture::partition::*;
    let phase = if prediction == 0 {
        CapturePhase::Prefill
    } else {
        CapturePhase::Decode
    };
    let world = layouts.topology().world_size();
    let plan = SharedCapturePlan::new(plan.clone());
    let mut ledger = CaptureLedger::new(&plan);
    (0..plan.plan().selections.len())
        .map(|selection_index| {
            let producers = layouts
                .capture_producers(ComponentCaptureProjectionRequest {
                    invocation: None,
                    plan: &plan,
                    selection_index,
                    phase,
                    prediction,
                    max_producers: world,
                    max_fragments: 16,
                })
                .unwrap();
            let constructor = match layouts
                .observation_combination(&plan.plan().selections[selection_index].path)
                .unwrap()
            {
                PartitionCaptureCombination::Disjoint => PartitionCaptureReceiptPlan::new,
                PartitionCaptureCombination::SumF64ToF32 => PartitionCaptureReceiptPlan::new_sum,
            };
            constructor(
                plan.clone(),
                PartitionCaptureContext {
                    invocation: None,
                invocation_window: None,
                    artifact_identity: "pinned-numeric-source".into(),
                    execution_identity: "retained-partition-selection".into(),
                    run_identity: "numerical-forward".into(),
                    overlay_identity: None,
                    capture_plan_identity: plan.identity().into(),
                    selection_index,
                    phase,
                    prediction,
                    forward_epoch: prediction,
                },
                producers,
                world,
                PartitionCaptureReceiptLimits {
                    max_producers: world,
                    max_fragments: 16,
                    max_record_bytes: 65536,
                },
                &mut ledger,
            )
            .unwrap()
        })
        .collect()
}

fn verify_component_producer_limits(
    layouts: &eredu_architectures::component_partition::ComponentPartitionLayouts,
    plan: &eredu_core::capture::AdmittedCapturePlan,
    topology: eredu_core::ParallelTopology,
) {
    use eredu_architectures::component_partition::ComponentCaptureProjectionRequest;
    use eredu_core::capture::*;
    let project = |plan: &AdmittedCapturePlan, max_producers, max_fragments, prediction| {
        layouts.capture_producers(ComponentCaptureProjectionRequest {
            invocation: None,
            plan,
            selection_index: 0,
            phase: CapturePhase::Prefill,
            prediction,
            max_producers,
            max_fragments,
        })
    };
    let valid = project(plan, topology.tensor(), topology.tensor(), 0).unwrap();
    assert_eq!(valid.len(), topology.tensor());
    assert!(project(plan, 0, 16, 0).is_err());
    assert!(project(plan, topology.tensor(), 0, 0).is_err());
    assert!(project(plan, topology.tensor(), 16, plan.request().max_predictions).is_err());
    if topology.tensor() > 1 {
        assert!(project(plan, topology.tensor() - 1, 16, 0).is_err());
    }
    // A globally empty selector still acknowledges every distinct TP shard.
    let mut empty = plan.plan().clone();
    empty.selections[0].slices[0].end = empty.selections[0].slices[0].start;
    let support = eredu_core::ObservationSupportReport {
        schema_version: 1,
        capture: CaptureCapabilities {
            transformations: vec![CaptureTransformKind::Slice],
            max_histogram_bins: 0,
            conditions: vec![],
        },
        points: plan
            .points()
            .iter()
            .map(|point| eredu_core::ObservationSupport {
                path: point.path.clone(),
                prefill: eredu_core::ObservationSupportStatus::Supported,
                decode: eredu_core::ObservationSupportStatus::Supported,
                floating_to_f32: true,
            })
            .collect(),
    };
    let empty = empty
        .admit(
            &eredu_core::ObservationCatalog {
                schema_version: 1,
                points: plan.points().to_vec(),
                completeness: eredu_core::DescriptionCompleteness::Complete,
            },
            &support,
            &support.capture,
            plan.request(),
        )
        .unwrap();
    let producers = project(&empty, topology.tensor(), 0, 0).unwrap();
    assert_eq!(producers.len(), topology.tensor());
    assert!(producers
        .iter()
        .all(|producer| producer.projection.fragments().is_empty()));
}

#[test]
fn prepared_nemotron_mixed_whole_writes_have_exact_partition_owners() {
    let mut config = nemotron_mixed_config();
    config["num_key_value_heads"] = 2.into();
    config["n_groups"] = 2.into();
    let (root, _) = prepared_adapter::payload_fixture_config_with(&config, 1., |_, _| None);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let context = NumericContext::default();
    let parameters = nemotron_h::LayeredModel::<NumericBackend>::new(
        nemotron_h::model_args_from_config_value(&config).unwrap(),
        &context,
    )
    .unwrap()
    .parameter_description(&context)
    .unwrap()
    .into_owned();
    for (tp, pp, ep) in [
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 1),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
    ] {
        let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
        let plan = prepared_adapter::plan(None).with_topology(topology);
        let sources = partitioned_adapter::prepare_plan(
            &inspection,
            &plan,
            0,
            std::time::Duration::from_secs(10),
        )
        .unwrap();
        let descriptor = sources.architecture().architecture_descriptor();
        let capture_plan = component_partition_capture_plan(&descriptor);
        let discovery = sources
            .prepare_discovery(Default::default(), Default::default())
            .bind_partition_parameters(Some(Arc::new(parameters.clone())))
            .unwrap();
        let layouts = discovery
            .component_partition_layouts(topology.world_size())
            .unwrap()
            .unwrap();
        for path in [
            "model.layers.0.mixer.output",
            "model.layers.2.feed_forward.output",
        ] {
            let mut invocations = 0;
            let mut exporters = 0;
            for rank in 0..topology.world_size() {
                let observation = layouts.rank(rank).unwrap().observation(path).unwrap();
                invocations += usize::from(observation.coordinates().is_some());
                exporters += usize::from(observation.exports());
                assert!(!observation.exports() || observation.coordinates().is_some());
            }
            assert_eq!(
                invocations,
                tp * ep,
                "one physical invocation stage for {path}"
            );
            assert_eq!(exporters, tp * ep, "eligible replicas for {path}");
            let selection_index = capture_plan
                .plan()
                .selections
                .iter()
                .position(|selection| selection.path == path)
                .unwrap();
            for (phase, prediction) in [
                (eredu_core::capture::CapturePhase::Prefill, 0),
                (eredu_core::capture::CapturePhase::Decode, 1),
            ] {
                let producers = layouts.capture_producers(
                    eredu_architectures::component_partition::ComponentCaptureProjectionRequest {
                        invocation: None,
                        plan: &capture_plan, selection_index, phase, prediction,
                        max_producers: topology.world_size(), max_fragments: 16,
                    },
                ).unwrap();
                assert_eq!(
                    producers.len(),
                    1,
                    "replicas export one complete write for {path}"
                );
            }
        }
    }
}

#[test]
fn prepared_nemotron_components_and_complete_writes_cross_all_partitions_and_residency() {
    let mut config = nemotron_mixed_config();
    config["num_key_value_heads"] = 2.into();
    config["n_groups"] = 2.into();
    let args = nemotron_h::model_args_from_config_value(&config).unwrap();
    let (root, fixture) =
        prepared_adapter::payload_fixture_config_with(&config, 1., |name, shape| {
            name.ends_with(".A_log").then(|| {
                NumericTensor::new(
                    shape.to_vec(),
                    (0..shape.iter().product::<i32>())
                        .map(|i| -1.0 - 0.1 * i as f32)
                        .collect(),
                )
            })
        });
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let parameters = nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context)
        .unwrap()
        .parameter_description(&context)
        .unwrap()
        .into_owned();
    let inputs = [
        NumericTensor::token_ids(&[1, 2, 5]),
        NumericTensor::token_ids(&[3]),
        NumericTensor::token_ids(&[4]),
    ];
    let targets = [
        None,
        Some("model.layers.0.mixer.input"),
        Some("model.layers.0.mixer.output"),
        Some("model.layers.2.feed_forward.input"),
        Some("model.layers.2.feed_forward.output"),
        Some("model.layers.2.shared.feed_forward.input"),
        Some("model.layers.2.shared.feed_forward.output"),
        Some("readout.embedding"),
        Some("readout.normalized"),
        Some("readout.linear"),
    ]
    .map(|target| (target, vec![]))
    .into_iter()
    .chain([
        (
            None,
            vec![("model.layers.1.attention.channels", vec![1, 3], false)],
        ),
        (
            None,
            vec![("model.layers.3.feed_forward.units", vec![1, 2], true)],
        ),
        (
            None,
            vec![("model.layers.2.shared.feed_forward.units", vec![1, 3], true)],
        ),
        (
            None,
            vec![(
                "model.layers.2.shared.feed_forward.units",
                vec![1, 3],
                false,
            )],
        ),
        (
            Some("model.layers.0.mixer.output"),
            vec![
                ("model.layers.1.attention.channels", vec![0, 2], true),
                ("model.layers.2.shared.feed_forward.units", vec![1], true),
                ("model.layers.3.feed_forward.units", vec![1], true),
            ],
        ),
    ])
    .collect::<Vec<_>>();
    struct Populate<'a>(&'a prepared_adapter::ParameterBits);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(
            &mut self,
            metadata: eredu_nn::ParameterMetadataView<'_>,
            value: &'a mut NumericTensor,
        ) {
            let canonical = metadata.id().as_str();
            let source = if let Some(rest) = canonical.strip_prefix("model.layers.") {
                let (layer, rest) = rest.split_once('.').unwrap();
                let rest = if rest.starts_with("norm.") {
                    rest.to_string()
                } else {
                    format!("mixer.{}", rest.split_once('.').unwrap().1)
                };
                format!("backbone.layers.{layer}.{rest}")
            } else if let Some(rest) = canonical.strip_prefix("model.") {
                format!("backbone.{rest}")
            } else {
                canonical.to_string()
            };
            let (shape, bits) = self
                .0
                .get(&source)
                .unwrap_or_else(|| panic!("source {source} for {canonical}"));
            // Official depthwise storage moves only its singleton axis.
            assert_eq!(
                value.shape.iter().product::<i32>(),
                shape.iter().product::<i32>()
            );
            if !canonical.ends_with(".conv1d.weight") {
                assert_eq!(&value.shape, shape, "{canonical}");
            }
            value.data = bits
                .iter()
                .map(|bits| {
                    let v = f32::from_bits(*bits);
                    if canonical.ends_with(".A_log") {
                        (-v).ln()
                    } else {
                        v
                    }
                })
                .collect();
        }
    }
    let expected = targets
        .iter()
        .map(|(target, masks)| {
            let mut architecture =
                nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
            architecture
                .static_modules_mut()
                .visit_parameters_mut(&mut Populate(&fixture));
            let mut state = DeviceState::<NumericBackend, _>::create(
                nemotron_h::state_layout(&args).unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap();
            let units = (0..args.num_hidden_layers as usize)
                .map(|layer| {
                    let mut unit = architecture.construct_unit(0, layer, &context).unwrap();
                    unit.visit_parameters_mut(&mut Populate(&fixture));
                    unit
                })
                .collect();
            let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
            inputs
                .iter()
                .enumerate()
                .map(|(step, tokens)| {
                    let mut observer = GlobalComponentObserver {
                        inner: NumericLifecycleObserver {
                            zero_path: target.map(str::to_owned),
                            ..Default::default()
                        },
                        values: BTreeMap::new(),
                        layout: None,
                        masks,
                        position: usize::from(step == 0),
                        projection_plan: None,
                        producer_receipts: None,
                        prediction: step as u64,
                    };
                    let output = runtime
                        .forward_with_observer(
                            nemotron_h::EmbeddedInput::target(tokens, None),
                            &mut state,
                            &context,
                            &mut observer,
                        )
                        .unwrap();
                    let output =
                        eredu_runtime::observe_model_logits(&mut observer, &output).unwrap();
                    let vocabulary = *output.shape.last().unwrap() as usize;
                    (
                        NumericTensor::new(
                            vec![1, 1, vocabulary as i32],
                            output.data[output.data.len() - vocabulary..].to_vec(),
                        ),
                        observer.values,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (index, trial) in expected.iter().enumerate().skip(1) {
        let difference = trial
            .iter()
            .zip(&expected[0])
            .flat_map(|(a, b)| a.0.data.iter().zip(&b.0.data).map(|(a, b)| (a - b).abs()))
            .fold(0.0_f32, f32::max);
        assert!(
            difference > 1e-7,
            "Nemotron intervention {index}, difference={difference}"
        );
    }
    verify_prepared_component_execution_with_topologies(
        &inspection,
        parameters,
        &inputs,
        &targets,
        &expected,
        |sources, context| {
            partitioned_adapter::routed(sources, context, Arc::new(AtomicUsize::new(0)), None)
        },
        &[
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (2, 2, 1),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
        ]
        .map(|(tp, pp, ep)| ParallelTopology::new(tp, pp, ep, 1).unwrap()),
    );
}

#[path = "components/outer_boundaries.rs"]
mod outer_boundaries;
