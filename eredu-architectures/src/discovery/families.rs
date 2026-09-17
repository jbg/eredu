use super::routed_components::{self as routed_units, Site as RoutedUnitSite};
use super::*;
use crate::decoder::Config;
use eredu_core::component::{ComponentResidualWrite, ComponentScalar};

pub(super) fn gguf_composite(
    g: &mut Builder,
    config: &crate::gguf_companion::GgufMediaProjectorConfig,
) {
    use crate::gguf_companion::GgufMediaProjectorConfig as C;
    match config {
        C::Gemma4(c) => gemma(g, c),
        C::Inkling(c) => inkling(g, c),
        C::MuseGlimmer(c) => muse(g, c),
        C::Qwen35(c) | C::Qwen35Pending(c) => qwen_hybrid(g, c),
        C::Qwen3Vl(c) => {
            qwen_with_deepstack(g, &c.text, c.vision.deepstack_layers().len());
            media(g, "vision");
        }
        C::Qwen3VlPending(c) => {
            qwen_with_deepstack(g, &c.text, c.vision.deepstack_layers().len());
            media(g, "vision");
        }
    }
}

fn prepared_vision_addition(
    g: &mut Builder,
    block: &str,
    previous: &str,
    path: &str,
    layer: usize,
    width: usize,
) -> String {
    let addition = format!("{block}.deepstack");
    g.node(
        &addition,
        ArchitectureNodeKind::ResidualAdd,
        Some(block),
        Some(path),
        Some(width),
    );
    g.get_mut(&addition).layer_index = Some(layer);
    g.edge(previous, &addition, ArchitectureEdgeKind::Data);
    let start = g.descriptor.observations.points.len();
    g.read_only_observation(
        &addition,
        format!("{path}.deepstack.input"),
        "Residual before adding prepared vision features",
        axes(width),
    );
    for (suffix, meaning) in [
        (
            "output",
            "Prepared vision contribution at decoder positions",
        ),
        ("residual", "Residual after adding prepared vision features"),
    ] {
        g.component_observation(
            &addition,
            format!("{path}.deepstack.{suffix}"),
            meaning,
            axes(width),
        );
    }
    for point in &mut g.descriptor.observations.points[start..] {
        point.decode = false;
        point.requirements.push(ObservationRequirement::MediaInput);
    }
    addition
}

pub(super) fn dense<C: Config>(g: &mut Builder, c: &C, moe_policy: Option<MoeAttributes>) {
    dense_with_deepstack(g, c, moe_policy, 0);
}

fn dense_with_deepstack<C: Config>(
    g: &mut Builder,
    c: &C,
    moe_policy: Option<MoeAttributes>,
    deepstack_count: usize,
) {
    let root = c.parameter_root();
    let width = c.hidden_size() as usize;
    let embedding = format!("{root}.embed_tokens");
    let mut previous = g.start(&embedding, width);
    let fields = c.block_parameter_fields();
    let mut other_writes = Vec::new();
    for (i, policy) in c.attention_schedule().iter().enumerate() {
        let path = format!("{root}.layers.{i}");
        let block = g.block(&previous, i, &path, width);
        let (op, join) = g.sublayer(
            &block,
            &block,
            "attention",
            &format!("{path}.{}", fields.input_norm),
            &format!("{path}.{}", fields.attention),
            width,
            ArchitectureNodeKind::Attention,
        );
        let mut a = attention(
            c.num_attention_heads(),
            c.num_key_value_heads(),
            c.head_dim(),
            *policy,
        );
        a.positional_encoding = Some(if c.rotary_enabled() {
            PositionalEncoding::Rotary
        } else {
            PositionalEncoding::None
        });
        a.sink_logits = c
            .learned_attention_sinks()
            .then(|| format!("{path}.{}.{}", fields.attention, fields.attention_sinks));
        g.get_mut(&op).attention = Some(a);
        post_sublayer_norm(
            g,
            &block,
            &op,
            &join,
            c.attention_output_normalization(i),
            width,
        );
        let (ff, output) = g.sublayer(
            &block,
            &join,
            "feed_forward",
            &format!("{path}.{}", fields.post_attention_norm),
            &format!("{path}.{}", fields.feed_forward),
            width,
            ArchitectureNodeKind::FeedForward,
        );
        post_sublayer_norm(
            g,
            &block,
            &ff,
            &output,
            c.feed_forward_output_normalization(i),
            width,
        );
        let point = c.routed_observation_points(&path, i);
        let dense_ffn = point.is_none();
        if let Some(attributes) = moe_policy.as_ref().filter(|_| !dense_ffn) {
            g.moe(
                &ff,
                &format!("{path}.{}", fields.feed_forward),
                attributes.clone(),
                point
                    .as_ref()
                    .and_then(|p| p.bank(eredu_runtime::RoutedBankId::new(0)))
                    .map(|p| p.path()),
                width,
            );
        }
        {
            super::components::decoder_unit(g, c, i, &path, &op, &ff, dense_ffn);
            for (node, suffix, meaning, extent, axis) in [
                (
                    &op,
                    "attention.input",
                    "Normalized attention input",
                    width,
                    "hidden",
                ),
                (
                    &op,
                    "attention.channels",
                    "Aggregated attention channels after gating, before output projection",
                    (c.num_attention_heads() * c.head_dim()) as usize,
                    "component",
                ),
                (
                    &op,
                    "attention.write",
                    "Attention affine write before output normalization",
                    width,
                    "hidden",
                ),
                (
                    &op,
                    "attention.output",
                    "Attention write after output normalization",
                    width,
                    "hidden",
                ),
                (
                    &join,
                    "attention.residual",
                    "Residual after attention addition",
                    width,
                    "hidden",
                ),
                (
                    &ff,
                    "feed_forward.input",
                    "Normalized feed-forward input",
                    width,
                    "hidden",
                ),
                (
                    &ff,
                    "feed_forward.units",
                    "Feed-forward unit values after activation/gating, before down projection",
                    c.intermediate_size() as usize,
                    "component",
                ),
                (
                    &ff,
                    "feed_forward.write",
                    "Feed-forward affine write before output normalization",
                    width,
                    "hidden",
                ),
                (
                    &ff,
                    if dense_ffn {
                        "feed_forward.output"
                    } else {
                        "feed_forward.contribution"
                    },
                    "Complete feed-forward write after output normalization",
                    width,
                    "hidden",
                ),
                (
                    &output,
                    "feed_forward.residual",
                    "Residual after feed-forward addition, before block normalization",
                    width,
                    "hidden",
                ),
            ] {
                if !dense_ffn && suffix == "feed_forward.units" {
                    continue;
                }
                let mut shape = axes(extent);
                shape[2].name = axis.into();
                g.component_observation(node, format!("{path}.{suffix}"), meaning, shape);
            }
        }
        if !dense_ffn {
            other_writes.push(eredu_core::component::ComponentResidualWrite {
                input: Some(format!("{path}.feed_forward.input")),
                layer_index: i,
                node_id: ff.clone(),
                output: format!("{path}.feed_forward.contribution"),
                effective_output: format!("{path}.feed_forward.contribution.effective"),
                residual_scale: eredu_core::component::ComponentScalar::new(1.0),
            });
        }
        previous = output;
        if let Some(name) = c.block_output_normalization(i) {
            let id = format!("{block}.output_norm");
            g.node(
                &id,
                ArchitectureNodeKind::Normalization,
                Some(&block),
                Some(name.trim_end_matches(".weight")),
                Some(width),
            );
            g.edge(&previous, &id, ArchitectureEdgeKind::Data);
            previous = id;
        }
        if i < deepstack_count {
            let addition = prepared_vision_addition(g, &block, &previous, &path, i, width);
            other_writes.push(ComponentResidualWrite {
                input: None,
                layer_index: i,
                node_id: addition.clone(),
                output: format!("{path}.deepstack.output"),
                effective_output: format!("{path}.deepstack.output.effective"),
                residual_scale: ComponentScalar::new(1.0),
            });
            previous = addition;
        }
    }
    g.output(
        &previous,
        &format!("{root}.norm"),
        if c.tie_word_embeddings() {
            &embedding
        } else {
            "lm_head"
        },
        width,
        c.vocabulary_size() as usize,
    );
    super::components::readout(g, c);
    g.descriptor
        .component_readout
        .as_mut()
        .expect("shared decoder readout")
        .other_writes = other_writes;
}

pub(super) fn nanbeige(g: &mut Builder, c: &crate::nanbeige::ModelArgs) {
    dense(g, c, None);
    g.repeated_decoder(c.parameter_root(), c.physical_layer_count(), c.num_loops());
}

pub(super) fn qwen(g: &mut Builder, c: &crate::qwen::ModelArgs) {
    qwen_with_deepstack(g, c, 0);
}

fn qwen_with_deepstack(g: &mut Builder, c: &crate::qwen::ModelArgs, deepstack_count: usize) {
    dense_with_deepstack(
        g,
        c,
        (c.num_experts > 0).then(|| {
            moe(
                c.num_experts,
                c.num_experts_per_tok,
                0,
                c.norm_topk_prob,
                RoutingScoreTransform::Softmax,
            )
        }),
        deepstack_count,
    );
    routed_units::decoder(g, c, |layer| crate::qwen::expert_bank_spec(c, layer));
    if let Ok(spec) = c.routing_spec() {
        for layer in 0..c.num_hidden_layers as usize {
            let unit = format!("{}.layers.{layer}", c.parameter_root);
            if let Some(point) = c.routed_observation_points(&unit, layer) {
                g.routing_control(
                    &format!("decoder.layers.{layer}.feed_forward"),
                    point
                        .bank(eredu_runtime::RoutedBankId::new(0))
                        .expect("declared feed-forward observation")
                        .path(),
                    spec,
                    0,
                    false,
                );
            }
        }
    }
}

pub(super) fn gpt_oss(g: &mut Builder, c: &crate::gpt_oss::ModelArgs) {
    dense(
        g,
        c,
        Some(moe(
            c.num_local_experts,
            c.num_experts_per_tok,
            0,
            false,
            RoutingScoreTransform::SelectedSoftmax,
        )),
    );
    routed_units::decoder(g, c, |layer| crate::gpt_oss::expert_bank_spec(c, layer));
}

pub(super) fn qwen_hybrid(g: &mut Builder, parsed: &crate::qwen::hybrid::ParsedHybridConfig) {
    use crate::qwen::hybrid::HybridLayerPolicy;
    let c = &parsed.text;
    let width = c.hidden_size as usize;
    let mut previous = g.start("model.embed_tokens", width);
    for (i, policy) in c.layer_schedule.iter().enumerate() {
        let path = format!("model.layers.{i}");
        let block = g.block(&previous, i, &path, width);
        let linear = matches!(policy, HybridLayerPolicy::LinearAttention);
        let (op, join) = g.sublayer(
            &block,
            &block,
            "mixer",
            &format!("{path}.input_layernorm"),
            &format!(
                "{path}.{}",
                if linear { "linear_attn" } else { "self_attn" }
            ),
            width,
            if linear {
                ArchitectureNodeKind::Mixer
            } else {
                ArchitectureNodeKind::Attention
            },
        );
        match policy {
            HybridLayerPolicy::LinearAttention => {
                g.get_mut(&op).mixer = Some(MixerAttributes {
                    mechanism: MixerMechanism::GatedDelta,
                    recurrent: true,
                    convolution_width: Some(c.linear_conv_kernel_dim as usize),
                });
                g.get_mut(&op).attention = Some(AttentionAttributes {
                    mechanism: Some(AttentionMechanism::Linear),
                    recurrent: Some(true),
                    causal: Some(true),
                    key_head_dimension: Some(c.linear_key_head_dim as usize),
                    value_head_dimension: Some(c.linear_value_head_dim as usize),
                    positional_encoding: Some(PositionalEncoding::None),
                    ..Default::default()
                });
            }
            HybridLayerPolicy::SelfAttention(policy) => {
                g.get_mut(&op).attention = Some(attention(
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    *policy,
                ))
            }
        }
        let (ff, output) = qwen_hybrid_ffn(g, c, i, &block, &path, &join);
        super::components::qwen_hybrid::unit(g, c, i, &path, &op, &ff);
        if parsed
            .vision
            .as_ref()
            .is_some_and(|vision| i < vision.deepstack_layers().len())
        {
            let addition = format!("{block}.deepstack");
            g.node(
                &addition,
                ArchitectureNodeKind::ResidualAdd,
                Some(&block),
                Some(&path),
                Some(width),
            );
            g.get_mut(&addition).layer_index = Some(i);
            g.edge(&output, &addition, ArchitectureEdgeKind::Data);
            let first_observation = g.descriptor.observations.points.len();
            g.read_only_observation(
                &addition,
                format!("{path}.deepstack.input"),
                "Residual before adding prepared vision features",
                axes(width),
            );
            for (suffix, meaning) in [
                (
                    "output",
                    "Prepared vision contribution at decoder positions",
                ),
                ("residual", "Residual after adding prepared vision features"),
            ] {
                g.component_observation(
                    &addition,
                    format!("{path}.deepstack.{suffix}"),
                    meaning,
                    axes(width),
                );
            }
            for point in &mut g.descriptor.observations.points[first_observation..] {
                point.decode = false;
                point.requirements.push(ObservationRequirement::MediaInput);
            }
            previous = addition;
        } else {
            previous = output;
        }
    }
    g.output(
        &previous,
        "model.norm",
        if c.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        },
        width,
        c.vocab_size as usize,
    );
    super::components::qwen_hybrid::readout(g, c);
    if let Some(vision) = &parsed.vision {
        let readout = g
            .descriptor
            .component_readout
            .as_mut()
            .expect("hybrid readout");
        for layer in 0..vision
            .deepstack_layers()
            .len()
            .min(c.num_hidden_layers as usize)
        {
            readout.other_writes.push(ComponentResidualWrite {
                input: None,
                layer_index: layer,
                node_id: format!("decoder.layers.{layer}.deepstack"),
                output: format!("model.layers.{layer}.deepstack.output"),
                effective_output: format!("model.layers.{layer}.deepstack.output.effective"),
                residual_scale: ComponentScalar::new(1.0),
            });
        }
    }
    if c.mtp_num_hidden_layers > 0 {
        super::components::qwen_hybrid::prediction::declare(g, c, &previous);
    }
    if let Some(vision) = &parsed.vision {
        media(g, "vision");
        // The primary residual starts after text/media assembly. The embedding
        // parameter still identifies the token lookup branch of that assembly.
        for point in &mut g.descriptor.observations.points {
            if matches!(
                point.path.as_str(),
                "readout.embedding" | "readout.embedding.effective"
            ) {
                point.node_id = "assembly".into();
                point.meaning = "Actual assembled text and media input to the decoder".into();
            }
        }
        let paths = ["readout.embedding", "readout.embedding.effective"];
        g.get_mut("embedding")
            .observation_paths
            .retain(|path| !paths.contains(&path.as_str()));
        for path in paths {
            g.get_mut("assembly").observation_paths.push(path.into());
        }
        for layer in 0..vision
            .deepstack_layers()
            .len()
            .min(c.num_hidden_layers as usize)
        {
            g.edge(
                "vision.projector",
                &format!("decoder.layers.{layer}.deepstack"),
                ArchitectureEdgeKind::Data,
            );
        }
    }
}

pub(in crate::discovery) fn qwen_hybrid_ffn(
    g: &mut Builder,
    c: &crate::qwen::hybrid::HybridConfig,
    i: usize,
    block: &str,
    path: &str,
    join: &str,
) -> (String, String) {
    let width = c.hidden_size as usize;
    let (ff, output) = g.sublayer(
        &block,
        &join,
        "feed_forward",
        &format!("{path}.post_attention_layernorm"),
        &format!("{path}.mlp"),
        width,
        ArchitectureNodeKind::FeedForward,
    );
    if c.num_experts > 0 {
        let mut attrs = moe(
            c.num_experts,
            c.num_experts_per_tok,
            1,
            c.norm_topk_prob,
            RoutingScoreTransform::Softmax,
        );
        attrs.shared_expert_width = Some(c.shared_expert_intermediate_size as usize);
        attrs.shared_expert_gated = Some(true);
        if let Ok(spec) = c.routing_spec() {
            g.routing_control(&ff, &format!("{path}.mlp"), spec, 1, false);
        }
        g.moe(
            &ff,
            &format!("{path}.mlp"),
            attrs,
            Some(&format!("{path}.mlp")),
            width,
        );
        routed_units::gated(
            g,
            RoutedUnitSite {
                node: &ff,
                layer: i,
                routing: &format!("{path}.mlp"),
                input: Some(format!("{path}.feed_forward.input")),
                normalization: Some(routed_units::rms(
                    format!("{path}.post_attention_layernorm.weight"),
                    c.rms_norm_eps,
                    1.0,
                )),
                residual_scale: Some(ComponentScalar::new(1.0)),
            },
            crate::qwen::hybrid::block::expert_bank_spec(c, i),
        );
    }
    (ff, output)
}

fn prediction(g: &mut Builder, previous: &str) {
    g.node(
        "prediction",
        ArchitectureNodeKind::Prediction,
        None,
        None,
        None,
    );
    g.edge(previous, "prediction", ArchitectureEdgeKind::Data);
    g.get_mut("prediction").completeness = DescriptionCompleteness::Partial(vec![
        "Prediction subgraph and its execution-dependent captures are not yet expanded".into(),
    ]);
    g.partial("Embedded prediction is represented as an opaque component");
    g.catalog_partial("Embedded prediction capture paths are not enumerated");
}

fn media(g: &mut Builder, name: &str) {
    g.node(name, ArchitectureNodeKind::Encoder, None, None, None);
    g.get_mut(name).completeness = DescriptionCompleteness::Partial(vec![
        "Media encoder internal operations are not expanded".into(),
    ]);
    g.partial("Media encoder internal operations are not expanded");
    g.catalog_partial("Media request-indexed and encoder capture paths are not enumerated");
    let projector = format!("{name}.projector");
    g.node(
        &projector,
        ArchitectureNodeKind::Projector,
        None,
        None,
        None,
    );
    if g.descriptor.node("assembly").is_none() {
        g.node(
            "assembly",
            ArchitectureNodeKind::ModalityMerge,
            None,
            None,
            None,
        );
        for edge in &mut g.descriptor.edges {
            if edge.from == "embedding" {
                edge.from = "assembly".into();
            }
        }
        g.edge("embedding", "assembly", ArchitectureEdgeKind::Data);
    }
    g.edge(name, &projector, ArchitectureEdgeKind::Data);
    g.edge(&projector, "assembly", ArchitectureEdgeKind::Data);
    media_captures(g, &projector, name == "vision");
    merge_capture(g, "assembly");
}

fn media_captures(g: &mut Builder, projector: &str, vision: bool) {
    let path = if vision {
        VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH
    } else {
        AUDIO_PROJECTOR_OUTPUT_OBSERVATION_PATH
    };
    g.observation(
        projector,
        path.into(),
        "Projected media features before merging with token embeddings",
        ObservationDtype::Floating,
        None,
        false,
    );
    let point = g.descriptor.observations.points.last_mut().unwrap();
    point.decode = false;
    point.requirements.push(ObservationRequirement::MediaInput);
}

fn merge_capture(g: &mut Builder, assembly: &str) {
    if g.descriptor
        .observations
        .get(MODALITY_MERGE_OUTPUT_OBSERVATION_PATH)
        .is_none()
    {
        g.observation(
            assembly,
            MODALITY_MERGE_OUTPUT_OBSERVATION_PATH.into(),
            "Decoder input after text/media assembly",
            ObservationDtype::Floating,
            None,
            false,
        );
    }
}

pub(super) fn remaining_safetensors(g: &mut Builder, c: &SafetensorsModelConfig) {
    match c {
        SafetensorsModelConfig::K2Horizon(c) => k2_horizon(g, c),
        SafetensorsModelConfig::Nanbeige(c) => nanbeige(g, c),
        SafetensorsModelConfig::Lfm2(c) => lfm2(g, c),
        SafetensorsModelConfig::KimiLinear(c) => kimi(g, c),
        SafetensorsModelConfig::NemotronH(c) => nemotron(g, c),
        SafetensorsModelConfig::DeepSeekV3(c) => deepseek_v3(g, c),
        SafetensorsModelConfig::DeepSeekV4(c) => deepseek_v4(g, c),
        SafetensorsModelConfig::Gemma4(c) => gemma(g, c),
        SafetensorsModelConfig::MuseGlimmer(c) => muse(g, c),
        SafetensorsModelConfig::Inkling(c) => inkling(g, c),
        SafetensorsModelConfig::QwenVl(c) => {
            qwen_with_deepstack(g, &c.text, c.vision.deepstack_layers().len());
            media(g, "vision");
        }
        SafetensorsModelConfig::Moshi(c) => moshi(g, c),
        SafetensorsModelConfig::Gemma2(_)
        | SafetensorsModelConfig::Llama(_)
        | SafetensorsModelConfig::Qwen(_)
        | SafetensorsModelConfig::GptOss(_)
        | SafetensorsModelConfig::QwenHybrid(_) => unreachable!("handled by caller"),
    }
}

pub(super) fn remaining_gguf(g: &mut Builder, c: &GgufModelConfig) {
    match c {
        GgufModelConfig::K2Horizon(c) => k2_horizon(g, c),
        GgufModelConfig::Lfm2(c) => lfm2(g, c),
        GgufModelConfig::KimiLinear(c) => kimi(g, c),
        GgufModelConfig::NemotronH(c) => nemotron(g, c),
        GgufModelConfig::DeepSeekV3(c) => deepseek_v3(g, c),
        GgufModelConfig::DeepSeekV4(c) => deepseek_v4(g, c),
        GgufModelConfig::Gemma4(c) => gemma(g, c),
        GgufModelConfig::MuseGlimmer(c) => muse(g, c),
        GgufModelConfig::Inkling(c) => inkling(g, c),
        GgufModelConfig::Gemma2(_)
        | GgufModelConfig::Llama(_)
        | GgufModelConfig::Nanbeige(_)
        | GgufModelConfig::Qwen(_)
        | GgufModelConfig::GptOss(_)
        | GgufModelConfig::QwenHybrid(_) => unreachable!("handled by caller"),
    }
}

fn lfm2(g: &mut Builder, c: &crate::lfm2::ModelArgs) {
    use crate::lfm2::{FeedForwardPolicy, OperatorPolicy};
    let width = c.hidden_size as usize;
    let mut previous = g.start("model.embed_tokens", width);
    for (i, policy) in c.layer_schedule.iter().enumerate() {
        let path = format!("model.layers.{i}");
        let block = g.block(&previous, i, &path, width);
        let conv = matches!(policy.operator, OperatorPolicy::CausalConvolution);
        let (op, join) = g.sublayer(
            &block,
            &block,
            "mixer",
            &format!("{path}.operator_norm"),
            &format!("{path}.{}", if conv { "conv" } else { "self_attn" }),
            width,
            if conv {
                ArchitectureNodeKind::Mixer
            } else {
                ArchitectureNodeKind::Attention
            },
        );
        match policy.operator {
            OperatorPolicy::CausalConvolution => {
                g.get_mut(&op).mixer = Some(MixerAttributes {
                    mechanism: MixerMechanism::ShortConvolution,
                    recurrent: true,
                    convolution_width: Some(c.conv_l_cache as usize),
                })
            }
            OperatorPolicy::SelfAttention(p) => {
                g.get_mut(&op).attention = Some(attention(
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.hidden_size / c.num_attention_heads,
                    p,
                ))
            }
        }
        let (ff, output) = g.sublayer(
            &block,
            &join,
            "feed_forward",
            &format!("{path}.ffn_norm"),
            &format!("{path}.feed_forward"),
            width,
            ArchitectureNodeKind::FeedForward,
        );
        if policy.feed_forward == FeedForwardPolicy::SparseMoe {
            let point = c.routed_observation_points(&path, i).expect("sparse layer");
            g.moe(
                &ff,
                &format!("{path}.feed_forward"),
                moe(
                    c.num_experts,
                    c.num_experts_per_tok,
                    0,
                    c.norm_topk_prob,
                    RoutingScoreTransform::Sigmoid,
                ),
                Some(
                    point
                        .bank(eredu_runtime::RoutedBankId::new(0))
                        .expect("declared feed-forward observation")
                        .path(),
                ),
                width,
            );
            routed_units::gated(
                g,
                RoutedUnitSite {
                    node: &ff,
                    layer: i,
                    routing: point
                        .bank(eredu_runtime::RoutedBankId::new(0))
                        .unwrap()
                        .path(),
                    input: Some(format!("{path}.feed_forward.input")),
                    normalization: Some(routed_units::rms(
                        format!("{path}.ffn_norm.weight"),
                        c.norm_eps,
                        0.0,
                    )),
                    residual_scale: Some(ComponentScalar::new(1.0)),
                },
                crate::lfm2::moe::expert_bank_spec(c, i),
            );
        }
        super::components::lfm2_unit(g, c, i, &path, &op, &ff);
        previous = output;
    }
    g.output(
        &previous,
        "model.embedding_norm",
        if c.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        },
        width,
        c.vocab_size as usize,
    );
    super::components::lfm2_readout(g, c);
    super::components::lfm2_other_writes(g, c);
}

fn kimi(g: &mut Builder, c: &crate::kimi_linear::ModelArgs) {
    use crate::kimi_linear::{AttentionKind, FeedForwardPolicy};
    let width = c.hidden_size as usize;
    let mut previous = g.start("model.embed_tokens", width);
    for (i, policy) in c.layer_schedule.iter().enumerate() {
        let path = format!("model.layers.{i}");
        let block = g.block(&previous, i, &path, width);
        let kda = policy.attention == AttentionKind::Kda;
        let (op, join) = g.sublayer(
            &block,
            &block,
            "mixer",
            &format!("{path}.input_layernorm"),
            &format!("{path}.self_attn"),
            width,
            if kda {
                ArchitectureNodeKind::Mixer
            } else {
                ArchitectureNodeKind::Attention
            },
        );
        g.get_mut(&op).attention = Some(AttentionAttributes {
            mechanism: Some(if kda {
                AttentionMechanism::Linear
            } else {
                AttentionMechanism::Latent
            }),
            recurrent: Some(kda),
            causal: Some(true),
            query_heads: Some(if kda {
                c.kda_config.num_heads
            } else {
                c.num_attention_heads
            } as usize),
            key_head_dimension: Some(if kda {
                c.kda_config.head_dim
            } else {
                c.qk_nope_head_dim + c.qk_rope_head_dim
            } as usize),
            value_head_dimension: Some(if kda {
                c.kda_config.head_dim
            } else {
                c.v_head_dim
            } as usize),
            positional_encoding: Some(PositionalEncoding::None),
            receptive_field: (!kda).then_some(ReceptiveField::Full),
            ..Default::default()
        });
        if kda {
            g.get_mut(&op).mixer = Some(MixerAttributes {
                mechanism: MixerMechanism::GatedDelta,
                recurrent: true,
                convolution_width: Some(c.kda_config.short_conv_kernel_size as usize),
            });
        }
        let (ff, output) = g.sublayer(
            &block,
            &join,
            "feed_forward",
            &format!("{path}.post_attention_layernorm"),
            &format!("{path}.mlp"),
            width,
            ArchitectureNodeKind::FeedForward,
        );
        if policy.feed_forward == FeedForwardPolicy::SparseMoe {
            let point = c.routed_observation_points(&path, i).expect("sparse layer");
            let mut attrs = moe(
                c.num_experts,
                c.num_experts_per_token,
                c.num_shared_experts,
                c.moe_renormalize,
                RoutingScoreTransform::Sigmoid,
            );
            attrs.shared_expert_width = Some(c.moe_intermediate_size as usize);
            g.moe(
                &ff,
                &format!("{path}.mlp"),
                attrs,
                Some(
                    point
                        .bank(eredu_runtime::RoutedBankId::new(0))
                        .expect("declared feed-forward observation")
                        .path(),
                ),
                width,
            );
            routed_units::gated(
                g,
                RoutedUnitSite {
                    node: &ff,
                    layer: i,
                    routing: point
                        .bank(eredu_runtime::RoutedBankId::new(0))
                        .unwrap()
                        .path(),
                    input: Some(format!("{path}.feed_forward.input")),
                    normalization: Some(routed_units::rms(
                        format!("{path}.post_attention_layernorm.weight"),
                        c.rms_norm_eps,
                        0.0,
                    )),
                    residual_scale: Some(ComponentScalar::new(1.0)),
                },
                crate::kimi_linear::moe::expert_bank_spec(c, i),
            );
        }
        super::components::kimi_linear::unit(g, c, i, &path, &op, &ff);
        previous = output;
    }
    g.output(
        &previous,
        "model.norm",
        if c.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        },
        width,
        c.vocab_size as usize,
    );
    super::components::kimi_linear::readout(g, c);
    if c.num_nextn_predict_layers > 0 {
        prediction(g, &previous);
    }
}

fn nemotron(g: &mut Builder, c: &crate::nemotron_h::ModelArgs) {
    use crate::nemotron_h::LayerPolicy;
    let width = c.hidden_size as usize;
    let mut previous = g.start("model.embeddings", width);
    for (i, policy) in c.layer_schedule.iter().enumerate() {
        let path = format!("model.layers.{i}");
        let block = g.block(&previous, i, &path, width);
        let (kind, field) = match policy {
            LayerPolicy::Mamba => (ArchitectureNodeKind::Mixer, "mamba"),
            LayerPolicy::SelfAttention(_) => (ArchitectureNodeKind::Attention, "attention"),
            LayerPolicy::DenseMlp => (ArchitectureNodeKind::FeedForward, "mlp"),
            LayerPolicy::SparseMoe => (ArchitectureNodeKind::MixtureOfExperts, "moe"),
        };
        let (op, output) = g.sublayer(
            &block,
            &block,
            "operator",
            &format!("{path}.norm"),
            &format!("{path}.{field}"),
            width,
            kind,
        );
        match policy {
            LayerPolicy::Mamba => {
                g.get_mut(&op).mixer = Some(MixerAttributes {
                    mechanism: MixerMechanism::SelectiveStateSpace,
                    recurrent: true,
                    convolution_width: Some(c.conv_kernel as usize),
                });
                super::components::nemotron_other_unit(g, width, &path, &op, "mixer");
            }
            LayerPolicy::SelfAttention(p) => {
                let mut attributes =
                    attention(c.num_attention_heads, c.num_key_value_heads, c.head_dim, *p);
                attributes.positional_encoding = Some(PositionalEncoding::None);
                g.get_mut(&op).attention = Some(attributes);
                super::components::nemotron_unit(g, c, i, &path, &op, true);
            }
            LayerPolicy::SparseMoe => {
                super::components::nemotron_other_unit(g, width, &path, &op, "feed_forward");
                let mut attrs = moe(
                    c.n_routed_experts,
                    c.num_experts_per_tok,
                    c.n_shared_experts,
                    c.norm_topk_prob,
                    RoutingScoreTransform::Sigmoid,
                );
                attrs.shared_expert_width = Some(c.moe_shared_expert_intermediate_size as usize);
                g.moe(
                    &op,
                    &format!("{path}.moe"),
                    attrs,
                    Some(&format!("{path}.routing")),
                    width,
                );
                super::components::nemotron::shared_units(g, c, i, &path, &op);
                routed_units::relu2(
                    g,
                    RoutedUnitSite {
                        node: &op,
                        layer: i,
                        routing: &format!("{path}.routing"),
                        input: Some(format!("{path}.feed_forward.input")),
                        normalization: Some(routed_units::rms(
                            format!("{path}.norm.weight"),
                            c.layer_norm_epsilon,
                            0.0,
                        )),
                        residual_scale: Some(ComponentScalar::new(1.0)),
                    },
                    crate::nemotron_h::expert_bank_spec(c, i),
                );
            }
            LayerPolicy::DenseMlp => super::components::nemotron_unit(g, c, i, &path, &op, false),
        }
        previous = output;
    }
    g.output(
        &previous,
        "model.norm_f",
        if c.tie_word_embeddings {
            "model.embeddings"
        } else {
            "lm_head"
        },
        width,
        c.vocab_size as usize,
    );
    super::components::nemotron_readout(g, c);
    if c.num_nextn_predict_layers > 0 {
        super::components::nemotron::prediction::declare(g, c, &previous);
    }
}

fn deepseek_v3(g: &mut Builder, c: &crate::deepseek::V3Args) {
    let width = c.hidden_size as usize;
    let mut previous = g.start("model.embed_tokens", width);
    for i in 0..c.layer_schedule.len() {
        let path = format!("model.layers.{i}");
        let block = g.block(&previous, i, &path, width);
        previous = deepseek_v3_unit(g, c, i, &path, &block);
    }
    g.output(
        &previous,
        "model.norm",
        "lm_head",
        width,
        c.vocab_size as usize,
    );
    super::components::deepseek_v3::readout(g, c);
    if c.num_nextn_predict_layers > 0 {
        super::components::deepseek_v3::prediction_scopes(g, c, &previous);
    }
}

pub(super) fn deepseek_v3_unit(
    g: &mut Builder,
    c: &crate::deepseek::V3Args,
    i: usize,
    path: &str,
    block: &str,
) -> String {
    let width = c.hidden_size as usize;
    let (op, join) = g.sublayer(
        block,
        block,
        "attention",
        &format!("{path}.input_layernorm"),
        &format!("{path}.self_attn"),
        width,
        ArchitectureNodeKind::Attention,
    );
    g.get_mut(&op).attention = Some(AttentionAttributes {
        mechanism: Some(AttentionMechanism::Latent),
        recurrent: Some(false),
        causal: Some(true),
        query_heads: Some(c.num_attention_heads as usize),
        key_head_dimension: Some((c.qk_nope_head_dim + c.qk_rope_head_dim) as usize),
        value_head_dimension: Some(c.v_head_dim as usize),
        receptive_field: Some(ReceptiveField::Full),
        positional_encoding: Some(PositionalEncoding::Rotary),
        ..Default::default()
    });
    let (ff, output) = g.sublayer(
        block,
        &join,
        "feed_forward",
        &format!("{path}.post_attention_layernorm"),
        &format!("{path}.mlp"),
        width,
        ArchitectureNodeKind::FeedForward,
    );
    super::components::deepseek_v3::unit(g, c, i, path, &op, &ff);
    if c.layer_schedule.get(i) != Some(&crate::deepseek::LayerPolicy::DenseMlp) {
        let attrs = moe(
            c.n_routed_experts,
            c.num_experts_per_tok,
            c.n_shared_experts,
            c.norm_topk_prob,
            RoutingScoreTransform::Sigmoid,
        );
        g.moe(
            &ff,
            &format!("{path}.mlp"),
            attrs,
            Some(&format!("{path}.feed_forward")),
            width,
        );
        routed_units::gated(
            g,
            RoutedUnitSite {
                node: &ff,
                layer: i,
                routing: &format!("{path}.feed_forward"),
                input: Some(format!("{path}.feed_forward.input")),
                normalization: Some(routed_units::rms(
                    format!("{path}.post_attention_layernorm.weight"),
                    c.rms_norm_eps,
                    0.0,
                )),
                residual_scale: Some(ComponentScalar::new(1.0)),
            },
            crate::deepseek::v3::expert_bank_spec(c, i),
        );
        super::components::deepseek_v3::shared_units(g, c, i, path, &ff);
    }
    output
}

fn deepseek_v4(g: &mut Builder, c: &crate::deepseek::V4Args) {
    let width = c.hidden_size as usize;
    let mut previous = g.start("embed", width);
    for (i, policy) in c
        .attention_schedule
        .iter()
        .take(c.num_hidden_layers as usize)
        .enumerate()
    {
        let path = format!("layers.{i}");
        let block = g.block(&previous, i, &path, width);
        let mut streams = axes(width);
        streams.insert(
            2,
            TensorAxis {
                name: "stream".into(),
                dimension: SymbolicDimension::Known(c.hc_mult as usize),
            },
        );
        g.descriptor
            .observations
            .points
            .retain(|p| p.node_id != block);
        g.interventions.retain(|p| p.node_id != block);
        g.get_mut(&block).observation_paths.clear();
        g.get_mut(&block).output_axes = Some(streams.clone());
        for boundary in ["input", "output"] {
            g.component_observation(
                &block,
                format!("{path}.{boundary}"),
                "Actual V4 residual streams at the block boundary",
                streams.clone(),
            );
        }
        let op = format!("{block}.attention");
        g.node(
            &op,
            ArchitectureNodeKind::Attention,
            Some(&block),
            Some(&format!("{path}.attn")),
            Some(width),
        );
        g.get_mut(&op).attention = Some(AttentionAttributes {
            mechanism: Some(AttentionMechanism::CompressedSparse),
            causal: Some(true),
            recurrent: Some(false),
            query_heads: Some(c.num_attention_heads as usize),
            key_value_heads: Some(c.num_key_value_heads as usize),
            key_head_dimension: Some(c.head_dim as usize),
            value_head_dimension: Some(c.head_dim as usize),
            // Compressed layers combine local and pooled history. A sliding-only
            // field would misrepresent that second branch.
            receptive_field: matches!(policy, crate::deepseek::V4AttentionPolicy::Local).then_some(
                ReceptiveField::Sliding {
                    window: c.sliding_window as usize,
                },
            ),
            positional_encoding: Some(PositionalEncoding::Rotary),
            ..Default::default()
        });
        let ff = format!("{block}.feed_forward");
        g.node(
            &ff,
            ArchitectureNodeKind::MixtureOfExperts,
            Some(&block),
            Some(&format!("{path}.ffn")),
            Some(width),
        );
        g.moe(
            &ff,
            &format!("{path}.ffn"),
            moe(
                c.n_routed_experts,
                c.num_experts_per_tok,
                c.n_shared_experts,
                c.norm_topk_prob,
                RoutingScoreTransform::SqrtSoftplus,
            ),
            Some(&format!("{path}.feed_forward")),
            width,
        );
        routed_units::gated(
            g,
            RoutedUnitSite {
                node: &ff,
                layer: i,
                routing: &format!("{path}.feed_forward"),
                input: Some(format!("{path}.feed_forward.input")),
                normalization: Some(routed_units::rms(
                    format!("{path}.ffn_norm.weight"),
                    c.rms_norm_eps,
                    0.0,
                )),
                residual_scale: None,
            },
            crate::deepseek::v4::expert_bank_spec(c, i),
        );
        super::components::deepseek_v4::unit(g, c, i, &path, &path, &op, &ff);
        g.get_mut(&block).completeness = DescriptionCompleteness::Partial(vec![
            "Compressed-history provenance and hash selection policy remain partially described"
                .into(),
        ]);
        previous = block;
    }
    g.output(&previous, "norm", "head", width, c.vocab_size as usize);
    super::components::deepseek_v4::readout(g, c);
    g.partial(
        "V4 compressed-history provenance and hash selection policy remain partially described",
    );
    g.catalog_partial("V4 compressed-index operation details are not fully enumerated");
    if c.num_nextn_predict_layers > 0 {
        if c.dspark.is_some() {
            super::components::deepseek_v4::declare_dspark(g, c, &previous);
        } else {
            super::components::deepseek_v4::declare_prediction(g, c, &previous);
        }
    }
}

/// Retains the existing architecture component graph, including media branches.
fn components(
    g: &mut Builder,
    graph: eredu_runtime::ComponentGraph,
    root: &str,
    width: usize,
    vocab: usize,
) {
    use eredu_runtime::ComponentKind as K;
    for component in graph.units() {
        let kind = match component.kind {
            K::StaticText => ArchitectureNodeKind::Embedding,
            K::Vision | K::Audio => ArchitectureNodeKind::Encoder,
            K::Assembly => ArchitectureNodeKind::ModalityMerge,
            K::Decoder => ArchitectureNodeKind::DecoderBlock,
            K::OutputProjection => ArchitectureNodeKind::OutputHead,
            K::Prediction | K::Assistant => ArchitectureNodeKind::Prediction,
        };
        g.node(&component.id, kind, None, None, None);
        for dependency in &component.dependencies {
            g.edge(dependency, &component.id, ArchitectureEdgeKind::Data);
        }
    }
    for component in graph
        .units()
        .iter()
        .filter(|c| matches!(c.kind, K::Vision | K::Audio))
    {
        let projector = format!("{}.projector", component.id);
        g.node(
            &projector,
            ArchitectureNodeKind::Projector,
            None,
            None,
            None,
        );
        for edge in &mut g.descriptor.edges {
            if edge.from == component.id {
                edge.from = projector.clone();
            }
        }
        g.edge(&component.id, &projector, ArchitectureEdgeKind::Data);
        media_captures(g, &projector, component.kind == K::Vision);
    }
    for component in graph.units().iter().filter(|c| c.kind == K::Assembly) {
        merge_capture(g, &component.id);
    }
    for (index, component) in graph
        .units()
        .iter()
        .filter(|c| c.kind == K::Decoder)
        .enumerate()
    {
        let path = format!("{root}.layers.{index}");
        // Retain the logical invocation namespace at the same declaration site
        // as its real outer hooks; shared physical sources do not choose owners.
        g.attach_parameters(&component.id, &path);
        g.get_mut(&component.id).layer_index = Some(index);
        g.decoder_execution(&component.id, index);
        g.get_mut(&component.id).output_axes = Some(axes(width));
        g.get_mut(&component.id).completeness = DescriptionCompleteness::Partial(vec!["Block equations, additional normalizations, state sharing, and internal data flow are not expanded".into()]);
        for (boundary, meaning) in [
            (UnitObservation::Input, "Block input"),
            (UnitObservation::Output, "Block output"),
        ] {
            g.component_observation(&component.id, boundary.path(&path), meaning, axes(width));
        }
    }
    let mut shape = axes(vocab);
    shape[2].name = "vocabulary".into();
    g.observation(
        "output",
        MODEL_LOGITS_OBSERVATION_PATH.into(),
        "Final vocabulary logits",
        ObservationDtype::Floating,
        Some(shape),
        false,
    );
    g.partial(
        "Component graph is exact; complex decoder and encoder interiors are partially described",
    );
    g.catalog_partial("Media, routing, and prediction internal capture points are not enumerated");
}

fn contained_attention(g: &mut Builder, layer: usize, attrs: AttentionAttributes) {
    let block = format!("decoder.{layer}");
    let id = format!("{block}.attention");
    g.node(
        &id,
        ArchitectureNodeKind::Attention,
        Some(&block),
        None,
        None,
    );
    g.get_mut(&id).attention = Some(attrs);
    g.node(
        &format!("{block}.norm"),
        ArchitectureNodeKind::Normalization,
        Some(&block),
        None,
        None,
    );
    g.get_mut(&format!("{block}.norm")).completeness = DescriptionCompleteness::Partial(vec![
        "Normalization parameters and ordering are not expanded".into(),
    ]);
}

fn contained_ff(g: &mut Builder, layer: usize, attrs: Option<MoeAttributes>) {
    let block = format!("decoder.{layer}");
    let id = format!("{block}.feed_forward");
    g.node(
        &id,
        if attrs.is_some() {
            ArchitectureNodeKind::MixtureOfExperts
        } else {
            ArchitectureNodeKind::FeedForward
        },
        Some(&block),
        None,
        None,
    );
    g.get_mut(&id).moe = attrs;
}

fn gemma(g: &mut Builder, c: &crate::gemma4::FamilyConfig) {
    let text = &c.text;
    let graph = crate::gemma4::component_graph(
        text,
        crate::gemma4::ComponentOptions {
            vision: c.vision.is_some(),
            audio: c.audio.is_some(),
            ..Default::default()
        },
    )
    .expect("admitted component geometry");
    components(
        g,
        graph,
        "model.language_model",
        text.hidden_size as usize,
        text.vocab_size as usize,
    );
    for (i, policy) in text.layer_schedule.iter().enumerate() {
        contained_attention(
            g,
            i,
            attention(
                text.num_attention_heads,
                policy.num_key_value_heads.get() as i32,
                policy.head_dim.get() as i32,
                policy.attention,
            ),
        );
        let sparse = policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe;
        contained_ff(
            g,
            i,
            sparse.then(|| {
                let mut attrs = moe(
                    text.num_experts.expect("sparse experts"),
                    text.top_k_experts.expect("sparse top k"),
                    0,
                    false,
                    RoutingScoreTransform::Softmax,
                );
                attrs.score_transform = Some(RoutingScoreTransform::SelectedSoftmax);
                attrs.normalization = Some(RoutingNormalization::None);
                attrs
            }),
        );
        if sparse {
            let block = format!("decoder.{i}");
            g.node(
                &format!("{block}.dense_feed_forward"),
                ArchitectureNodeKind::FeedForward,
                Some(&block),
                None,
                None,
            );
        }
        super::components::gemma4::unit(g, text, i);
    }
    super::components::gemma4::readout(g, text);
}

fn muse(g: &mut Builder, c: &crate::muse_glimmer::DecoderConfig) {
    components(
        g,
        crate::muse_glimmer::component_graph(c).expect("admitted component geometry"),
        "model",
        c.hidden_size as usize,
        c.vocab_size as usize,
    );
    for (i, policy) in c.attention_schedule.iter().enumerate() {
        let mut attrs = attention(
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            *policy,
        );
        attrs.positional_encoding = Some(if c.layer_uses_rope[i] {
            PositionalEncoding::Rotary
        } else {
            PositionalEncoding::None
        });
        contained_attention(g, i, attrs);
        contained_ff(
            g,
            i,
            (c.num_experts > 0).then(|| {
                moe(
                    c.num_experts,
                    c.num_experts_per_tok,
                    0,
                    c.norm_topk_prob,
                    RoutingScoreTransform::Softmax,
                )
            }),
        );
        super::components::muse::unit(g, c, i);
    }
    super::components::muse::readout(g, c);
}

fn inkling(g: &mut Builder, c: &crate::inkling::ModelArgs) {
    let text = &c.text_config;
    components(
        g,
        crate::inkling::component_graph(c).expect("admitted component geometry"),
        "model",
        text.hidden_size as usize,
        text.vocab_size as usize,
    );
    for (i, policy) in text.layer_schedule.iter().enumerate() {
        let sliding = matches!(policy.attention, AttentionPolicy::Sliding { .. });
        let mut attrs = attention(
            if sliding {
                text.swa_num_attention_heads
                    .unwrap_or(text.num_attention_heads)
            } else {
                text.num_attention_heads
            },
            if sliding {
                text.swa_num_key_value_heads
                    .unwrap_or(text.num_key_value_heads)
            } else {
                text.num_key_value_heads
            },
            if sliding {
                text.swa_head_dim.unwrap_or(text.head_dim)
            } else {
                text.head_dim
            },
            policy.attention,
        );
        attrs.positional_encoding = Some(PositionalEncoding::Relative);
        contained_attention(g, i, attrs);
        contained_ff(
            g,
            i,
            (policy.feed_forward == crate::inkling::FeedForwardPolicy::SparseMoe).then(|| {
                let mut attrs = moe(
                    text.n_routed_experts,
                    text.num_experts_per_tok,
                    text.n_shared_experts,
                    text.norm_after_topk,
                    RoutingScoreTransform::Softmax,
                );
                attrs.score_transform = Some(RoutingScoreTransform::Sigmoid);
                attrs.normalization = Some(RoutingNormalization::SelectedAndSharedSum);
                attrs.shared_expert_gated = Some(true);
                attrs
            }),
        );
        super::components::inkling::decoder_scalars(
            g,
            text,
            *policy,
            i,
            &format!("decoder.{i}"),
            &format!("model.layers.{i}"),
        );
        super::components::inkling::decoder_routed(
            g,
            c,
            i,
            &format!("decoder.{i}"),
            &format!("model.layers.{i}"),
        );
        super::components::inkling::decoder_transforms(
            g,
            text,
            *policy,
            &format!("decoder.{i}"),
            &format!("model.layers.{i}"),
        );
    }
    super::components::inkling::target_readout(g, text);
    super::components::inkling::prediction(g, c);
}

fn moshi(g: &mut Builder, c: &crate::moshi::MoshiConfig) {
    g.node("realtime", ArchitectureNodeKind::Realtime, None, None, None);
    g.node(
        "temporal",
        ArchitectureNodeKind::DecoderBlock,
        Some("realtime"),
        Some(c.temporal().parameter_root()),
        None,
    );
    g.node(
        "depth",
        ArchitectureNodeKind::DecoderBlock,
        Some("realtime"),
        Some(c.depth_template().parameter_root()),
        None,
    );
    g.edge("temporal", "depth", ArchitectureEdgeKind::Data);
    g.get_mut("realtime").completeness = DescriptionCompleteness::Partial(vec!["Temporal/depth frame schedule, codebook branches, and per-frame observations require the separate realtime protocol".into()]);
    g.partial(
        "Realtime frame architecture is described only at temporal/depth component granularity",
    );
    g.descriptor.observations.completeness = DescriptionCompleteness::Unsupported(vec![
        "Model prefill/decode discovery does not describe realtime frame observation identities"
            .into(),
    ]);
}

fn post_sublayer_norm(
    g: &mut Builder,
    block: &str,
    op: &str,
    join: &str,
    name: Option<String>,
    width: usize,
) {
    if let Some(name) = name {
        let id = format!("{op}.post_norm");
        g.node(
            &id,
            ArchitectureNodeKind::Normalization,
            Some(block),
            Some(name.trim_end_matches(".weight")),
            Some(width),
        );
        g.descriptor.edges.retain(|edge| {
            !(edge.from == op && edge.to == join && edge.kind == ArchitectureEdgeKind::Data)
        });
        g.edge(op, &id, ArchitectureEdgeKind::Data);
        g.edge(&id, join, ArchitectureEdgeKind::Data);
    }
}

fn k2_horizon(g: &mut Builder, c: &crate::k2_horizon::ModelArgs) {
    dense(g, c, None);
    let score = if c.router_score_func == "sigmoid" {
        RoutingScoreTransform::Sigmoid
    } else {
        RoutingScoreTransform::Softmax
    };
    for layer in 0..c.num_hidden_layers as usize {
        let path = format!("model.layers.{layer}");
        let Some(points) = c.routed_observation_points(&path, layer) else {
            continue;
        };
        for (bank, node, count, topk, shared, normalized) in [
            (
                crate::k2_horizon::ExpertBank::FeedForward,
                format!("decoder.layers.{layer}.feed_forward"),
                c.num_experts,
                c.num_experts_per_tok,
                c.num_shared_experts,
                c.norm_topk_prob,
            ),
            (
                crate::k2_horizon::ExpertBank::AttentionValue,
                format!("decoder.layers.{layer}.attention.values"),
                c.mova_num_experts,
                c.mova_num_experts_per_tok,
                0,
                c.mova_num_experts_per_tok > 1,
            ),
        ] {
            let Some(point) = points.bank(bank.id()) else {
                continue;
            };
            let width = if bank == crate::k2_horizon::ExpertBank::AttentionValue {
                let attention = format!("decoder.layers.{layer}.attention");
                g.node(
                    &node,
                    ArchitectureNodeKind::MixtureOfExperts,
                    Some(&attention),
                    Some(&format!("{path}.self_attn.v_experts")),
                    Some((c.num_key_value_heads * c.head_dim) as usize),
                );
                (c.num_key_value_heads * c.head_dim) as usize
            } else {
                c.hidden_size as usize
            };
            let mut attributes = moe(count, topk, shared, normalized, score);
            attributes.shared_expert_width =
                (shared > 0).then_some((shared * c.moe_intermediate_size) as usize);
            g.moe(&node, point.path(), attributes, Some(point.path()), width);
            if let Ok(spec) = c.routing_spec(bank) {
                g.routing_control(&node, point.path(), spec, shared as u32, false);
            }
        }
    }
    routed_units::decoder(g, c, |layer| {
        crate::k2_horizon::feed_forward_expert_spec(c, layer)
    });
    super::components::k2_horizon::complete(g, c);
}
