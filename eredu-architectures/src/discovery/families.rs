use super::routed_components::{self as routed_units, Site as RoutedUnitSite};
use super::*;
use crate::decoder::Config;
use eredu_core::component::{ComponentResidualWrite, ComponentScalar};

pub(super) fn gguf_composite(
    g: &mut Builder,
    config: &crate::gguf_companion::GgufMediaProjectorConfig,
    primary: &GgufModelConfig,
    text_declaration: Option<&crate::qwen::Declaration>,
) {
    use crate::gguf_companion::GgufMediaProjectorConfig as C;
    match config {
        C::NemotronLabsDiffusion { config, .. } => {
            super::nemotron_labs_diffusion::describe(g, config)
        }
        C::Qwen4Exp(_) | C::Qwen4ExpPending { .. } => {
            if let GgufModelConfig::Qwen4Exp(target) = primary {
                qwen4_exp(
                    g,
                    target.text_plan().config(),
                    target.text_plan().parameter_declaration(),
                );
                media(g, "vision");
            }
        }
        C::Gemma4(c) => gemma(g, c),
        C::Inkling(c) => inkling(g, c),
        C::Muse(c) => muse(g, c),
        C::Qwen35(c) | C::Qwen35Pending(c) => qwen_hybrid(g, c),
        C::Qwen3Vl(c) => {
            qwen_with_deepstack(
                g,
                text_declaration.expect("admitted VL text scopes"),
                c.vision.deepstack_layers().len(),
            );
            media(g, "vision");
        }
        C::Qwen3VlPending(c) => {
            qwen_with_deepstack(
                g,
                text_declaration.expect("admitted VL text scopes"),
                c.vision.deepstack_layers().len(),
            );
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

fn declared_static_input(
    g: &mut Builder,
    statics: &crate::decoder::StaticModuleConstruction,
) -> String {
    let embedding = statics.embedding_spec();
    g.start(
        super::components::declared_weight_prefix(&embedding.weight),
        embedding.dimensions as usize,
    )
}

fn declared_static_output(
    g: &mut Builder,
    previous: &str,
    statics: &crate::decoder::StaticModuleConstruction,
) {
    let output = statics.output_projection_spec();
    g.output(
        previous,
        super::components::declared_normalization_prefix(statics.normalization_spec())
            .expect("declared final normalization weight"),
        super::components::declared_weight_prefix(&output.weight),
        output.input as usize,
        output.output as usize,
    );
}

pub(super) fn dense<C: Config>(
    g: &mut Builder,
    c: &C,
    moe_policy: Option<MoeAttributes>,
    declaration: &crate::decoder::DecoderDeclaration,
) {
    dense_with_deepstack(g, c, moe_policy, 0, declaration);
}

fn dense_with_deepstack<C: Config>(
    g: &mut Builder,
    c: &C,
    moe_policy: Option<MoeAttributes>,
    deepstack_count: usize,
    declaration: &crate::decoder::DecoderDeclaration,
) {
    let root = c.parameter_root();
    declaration.validate_for(c).expect("admitted dense policy");
    let statics = declaration.static_construction();
    let width = statics.embedding_spec().dimensions as usize;
    let mut previous = declared_static_input(g, &statics);
    let fields = c.block_parameter_fields();
    let mut other_writes = Vec::new();
    for (i, policy) in c.attention_schedule().iter().enumerate() {
        let scope = declaration.block_parameters(i);
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
            scope.attention.geometry.query_heads(),
            scope.attention.geometry.key_value_heads(),
            scope.attention.geometry.key_dimensions(),
            *policy,
        );
        a.positional_encoding = Some(if c.rotary_enabled() {
            PositionalEncoding::Rotary
        } else {
            PositionalEncoding::None
        });
        a.sink_logits = scope.sinks.as_ref().map(|sink| sink.weight.id.to_string());
        g.get_mut(&op).attention = Some(a);
        post_sublayer_norm(
            g,
            &block,
            &op,
            &join,
            scope
                .normalizations
                .attention_output
                .as_ref()
                .and_then(super::components::declared_normalization_prefix)
                .map(|prefix| format!("{prefix}.weight")),
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
            scope
                .normalizations
                .feed_forward_output
                .as_ref()
                .and_then(super::components::declared_normalization_prefix)
                .map(|prefix| format!("{prefix}.weight")),
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
            super::components::decoder_unit(g, c, i, &path, &op, &ff, dense_ffn, scope);
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
                    scope.attention.geometry.attended_width() as usize,
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
                    scope
                        .feed_forward
                        .as_ref()
                        .map_or(0, |spec| spec.down().input as usize),
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
        let terminal = scope
            .terminal
            .as_ref()
            .and_then(super::components::declared_normalization_prefix)
            .map(str::to_owned);
        if let Some(name) = terminal {
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
    declared_static_output(g, &previous, &statics);
    super::components::readout(g, c, declaration);
    g.descriptor
        .component_readout
        .as_mut()
        .expect("shared decoder readout")
        .other_writes = other_writes;
}

pub(super) fn nanbeige(
    g: &mut Builder,
    c: &crate::nanbeige::ModelArgs,
    declaration: &crate::decoder::DecoderDeclaration,
) {
    dense(g, c, None, declaration);
    use eredu_nn::DeclareLogicalParameters;
    let parameters = declaration
        .logical_parameters()
        .expect("admitted repeated parameter slots");
    g.repeated_decoder(c.physical_layer_count(), c.num_loops());
    g.declared_parameter_sharing(&parameters)
        .expect("admitted repeated parameter sharing");
}

pub(super) fn qwen(g: &mut Builder, declaration: &crate::qwen::Declaration) {
    qwen_with_deepstack(g, declaration, 0);
}
fn qwen_with_deepstack(
    g: &mut Builder,
    declaration: &crate::qwen::Declaration,
    deepstack_count: usize,
) {
    let c = declaration.args();
    dense_with_deepstack(
        g,
        c,
        (c.num_experts > 0).then(|| {
            let spec = declaration
                .routed(0)
                .expect("admitted Qwen routed declaration");
            let selection = spec.selector.selection();
            moe(
                selection.group_count(),
                selection.top_k(),
                0,
                selection.normalize_selected(),
                RoutingScoreTransform::Softmax,
            )
        }),
        deepstack_count,
        declaration.common(),
    );
    routed_units::decoder(g, c, |layer| declaration.expert_bank(layer).cloned());
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

pub(super) fn gpt_oss(g: &mut Builder, declaration: &crate::gpt_oss::Declaration) {
    let c = declaration.args();
    let spec = declaration
        .projections()
        .parameters(0)
        .expect("admitted GPT-OSS routed declaration");
    let selection = spec.selector.selection();
    dense(
        g,
        c,
        Some(moe(
            selection.group_count(),
            selection.top_k(),
            0,
            selection.normalize_selected(),
            RoutingScoreTransform::SelectedSoftmax,
        )),
        declaration.common(),
    );
    routed_units::decoder(g, c, |layer| declaration.expert_bank(layer).cloned());
}

pub(super) fn qwen_hybrid(g: &mut Builder, parsed: &crate::qwen::hybrid::ParsedHybridConfig) {
    use crate::qwen::hybrid::HybridLayerPolicy;
    let c = &parsed.text;
    let statics = c.statics();
    let width = statics.embedding_spec().dimensions as usize;
    let mut previous = declared_static_input(g, &statics);
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
                let scope = crate::qwen::hybrid::BlockParameters::target(c, i)
                    .expect("admitted Qwen hybrid block declaration");
                let crate::qwen::hybrid::MixerParameters::Attention(parameters) = &scope.mixer
                else {
                    unreachable!()
                };
                let geometry = parameters.projection.geometry;
                g.get_mut(&op).attention = Some(attention(
                    geometry.query_heads(),
                    geometry.key_value_heads(),
                    geometry.key_dimensions(),
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
    declared_static_output(g, &previous, &statics);
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
        block,
        join,
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
    g.partial(super::OPAQUE_PREDICTION_DESCRIPTION);
    g.catalog_partial(super::UNENUMERATED_PREDICTION_CAPTURES);
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

pub(super) fn remaining_safetensors(
    g: &mut Builder,
    c: &SafetensorsModelConfig,
    plan: &crate::processor_plan::ArtifactArchitecturePlan,
) {
    match c {
        SafetensorsModelConfig::NemotronLabsDiffusion(c) => {
            super::nemotron_labs_diffusion::describe(g, c)
        }
        SafetensorsModelConfig::Qwen4Exp(c, _) => {
            qwen4_exp(g, c.configuration(), c.target_parameters())
        }
        SafetensorsModelConfig::DeepSeekV3(c) => deepseek_v3(g, c),
        SafetensorsModelConfig::DeepSeekV4(c) => deepseek_v4(g, c),
        SafetensorsModelConfig::DiffusionGemma(c) => diffusion_gemma(g, c),
        SafetensorsModelConfig::Gemma4(c) => gemma(g, c),
        SafetensorsModelConfig::Muse(c) => muse(g, c),
        SafetensorsModelConfig::Inkling(c) => inkling(g, c),
        SafetensorsModelConfig::QwenVl(c) => {
            qwen_with_deepstack(
                g,
                plan.vl_declaration(c)
                    .expect("admitted VL scopes")
                    .text_declaration(),
                c.vision.deepstack_layers().len(),
            );
            media(g, "vision");
        }
        SafetensorsModelConfig::Moshi(c) => moshi(g, c),
        SafetensorsModelConfig::VoiceChat(c) => super::voicechat::describe(g, c),
        SafetensorsModelConfig::K2Horizon(_)
        | SafetensorsModelConfig::Nanbeige(_)
        | SafetensorsModelConfig::KimiLinear(_)
        | SafetensorsModelConfig::Lfm2(_)
        | SafetensorsModelConfig::NemotronH(_)
        | SafetensorsModelConfig::Gemma2(_)
        | SafetensorsModelConfig::Llama(_)
        | SafetensorsModelConfig::Qwen(_)
        | SafetensorsModelConfig::GptOss(_)
        | SafetensorsModelConfig::QwenHybrid(_) => unreachable!("handled by caller"),
    }
}

pub(super) fn remaining_gguf(g: &mut Builder, c: &GgufModelConfig) {
    match c {
        GgufModelConfig::NemotronLabsDiffusion(plan) => {
            super::nemotron_labs_diffusion::describe(g, plan.text_config())
        }
        GgufModelConfig::Qwen4Exp(plan) => qwen4_exp(
            g,
            plan.text_plan().config(),
            plan.text_plan().parameter_declaration(),
        ),
        GgufModelConfig::VoiceChat(c) => super::voicechat::describe(g, c),
        GgufModelConfig::DeepSeekV3(c) => deepseek_v3(g, c),
        GgufModelConfig::DeepSeekV4(c) => deepseek_v4(g, c),
        GgufModelConfig::DiffusionGemma(c) => diffusion_gemma(g, c),
        GgufModelConfig::Gemma4(c) => gemma(g, c),
        GgufModelConfig::Muse(c) => muse(g, c),
        GgufModelConfig::Inkling(c) => inkling(g, c),
        GgufModelConfig::K2Horizon(_)
        | GgufModelConfig::KimiLinear(_)
        | GgufModelConfig::Lfm2(_)
        | GgufModelConfig::NemotronH(_)
        | GgufModelConfig::Gemma2(_)
        | GgufModelConfig::Llama(_)
        | GgufModelConfig::Nanbeige(_)
        | GgufModelConfig::Qwen(_)
        | GgufModelConfig::GptOss(_)
        | GgufModelConfig::QwenHybrid(_) => unreachable!("handled by caller"),
    }
}

pub(super) fn lfm2(g: &mut Builder, declaration: &crate::lfm2::Declaration) {
    use crate::lfm2::declarations::{FeedForwardParameters as F, MixerParameters as M};
    let statics = &declaration.statics;
    let width = statics.embedding_spec().dimensions as usize;
    let mut previous = declared_static_input(g, statics);
    for (i, scope) in declaration.blocks.iter().enumerate() {
        let path = format!("model.layers.{i}");
        let block = g.block(&previous, i, &path, width);
        let conv = matches!(scope.mixer, M::Convolution(_));
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
        match &scope.mixer {
            M::Convolution(spec) => {
                g.get_mut(&op).mixer = Some(MixerAttributes {
                    mechanism: MixerMechanism::ShortConvolution,
                    recurrent: true,
                    convolution_width: Some(spec.convolution.kernel_size as usize),
                })
            }
            M::Attention(spec, p) => {
                g.get_mut(&op).attention = Some(attention(
                    spec.projection.geometry.query_heads(),
                    spec.projection.geometry.key_value_heads(),
                    spec.projection.geometry.key_dimensions(),
                    *p,
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
        if let F::Routed(spec) = &scope.feed_forward {
            let selection = spec.selector.selection();
            let point = declaration
                .routed_observation_points(&path, i)
                .expect("sparse layer");
            g.moe(
                &ff,
                &format!("{path}.feed_forward"),
                moe(
                    selection.group_count(),
                    selection.top_k(),
                    0,
                    selection.normalize_selected(),
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
                    normalization: Some(super::components::declared_normalization(
                        &scope.feed_forward_norm,
                    )),
                    residual_scale: Some(ComponentScalar::new(1.0)),
                },
                Ok(spec.experts.clone()),
            );
        }
        super::components::lfm2_unit(g, declaration, i, &path, &op, &ff);
        previous = output;
    }
    declared_static_output(g, &previous, statics);
    super::components::lfm2_readout(g, declaration);
    super::components::lfm2_other_writes(g, declaration);
}

pub(super) fn kimi(g: &mut Builder, declaration: &crate::kimi_linear::Declaration) {
    use crate::kimi_linear::declarations::{FeedForwardParameters as F, MixerParameters as M};
    let statics = &declaration.statics;
    let width = statics.embedding_spec().dimensions as usize;
    let mut previous = declared_static_input(g, statics);
    for (i, scope) in declaration.blocks.iter().enumerate() {
        let path = format!("model.layers.{i}");
        let block = g.block(&previous, i, &path, width);
        let kda = matches!(scope.mixer, M::Kda(_));
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
        let (heads, key_dim, value_dim, kernel) = match &scope.mixer {
            M::Kda(spec) => (
                spec.geometry.value_heads() as usize,
                spec.geometry.key_dimensions() as usize,
                spec.geometry.value_dimensions() as usize,
                Some(spec.convolutions[0].kernel_size as usize),
            ),
            M::Mla(spec) => (
                spec.geometry.heads() as usize,
                spec.geometry.query_head_dimensions() as usize,
                spec.geometry.value_dimensions() as usize,
                None,
            ),
        };
        g.get_mut(&op).attention = Some(AttentionAttributes {
            mechanism: Some(if kda {
                AttentionMechanism::Linear
            } else {
                AttentionMechanism::Latent
            }),
            recurrent: Some(kda),
            causal: Some(true),
            query_heads: Some(heads),
            key_head_dimension: Some(key_dim),
            value_head_dimension: Some(value_dim),
            positional_encoding: Some(PositionalEncoding::None),
            receptive_field: (!kda).then_some(ReceptiveField::Full),
            ..Default::default()
        });
        if kda {
            g.get_mut(&op).mixer = Some(MixerAttributes {
                mechanism: MixerMechanism::GatedDelta,
                recurrent: true,
                convolution_width: kernel,
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
        if let F::Sparse(parameters) = &scope.feed_forward {
            let selection = parameters.routed.selector.selection();
            let point = declaration
                .routed_observation_points(&path, i)
                .expect("sparse layer");
            let mut attrs = moe(
                selection.group_count(),
                selection.top_k(),
                1,
                selection.normalize_selected(),
                RoutingScoreTransform::Sigmoid,
            );
            attrs.shared_expert_width = Some(parameters.shared.gate.output as usize);
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
                    normalization: Some(super::components::declared_normalization(
                        &scope.post_attention_norm,
                    )),
                    residual_scale: Some(ComponentScalar::new(1.0)),
                },
                Ok(parameters.routed.experts.clone()),
            );
        }
        super::components::kimi_linear::unit(g, declaration, i, &path, &op, &ff);
        previous = output;
    }
    declared_static_output(g, &previous, statics);
    super::components::kimi_linear::readout(g, declaration);
    if declaration.args().num_nextn_predict_layers > 0 {
        prediction(g, &previous);
    }
}

pub(super) fn nemotron(g: &mut Builder, declaration: &crate::nemotron_h::Declaration) {
    use crate::nemotron_h::{declarations::TargetOperatorParameters as P, LayerPolicy};
    let c = declaration.args();
    let statics = &declaration.statics;
    let width = statics.embedding_spec().dimensions as usize;
    let mut previous = declared_static_input(g, &statics);
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
        let scope = &declaration.blocks[i];
        match (&scope.operator, policy) {
            (P::Mamba(spec), LayerPolicy::Mamba) => {
                g.get_mut(&op).mixer = Some(MixerAttributes {
                    mechanism: MixerMechanism::SelectiveStateSpace,
                    recurrent: true,
                    convolution_width: Some(spec.convolution.kernel_size as usize),
                });
                super::components::nemotron_other_unit(g, width, &path, &op, "mixer");
            }
            (P::Attention(parameters), LayerPolicy::SelfAttention(p)) => {
                let geometry = parameters.projection.geometry;
                let mut attributes = attention(
                    geometry.query_heads(),
                    geometry.key_value_heads(),
                    geometry.key_dimensions(),
                    *p,
                );
                attributes.positional_encoding = Some(PositionalEncoding::None);
                g.get_mut(&op).attention = Some(attributes);
                super::components::nemotron_unit(g, scope, i, &path, &op);
            }
            (P::Sparse(spec), LayerPolicy::SparseMoe) => {
                super::components::nemotron_other_unit(g, width, &path, &op, "feed_forward");
                let mut attrs = declared_routing(spec.selector.selection(), c.n_shared_experts)
                    .expect("admitted Nemotron routing equation");
                attrs.shared_expert_width = Some(spec.shared.network.up.output as usize);
                g.moe(
                    &op,
                    &format!("{path}.moe"),
                    attrs,
                    Some(&format!("{path}.routing")),
                    width,
                );
                super::components::nemotron::shared_units(
                    g,
                    spec,
                    &scope.norm,
                    i,
                    &path,
                    &op,
                    &format!("{path}.moe"),
                );
                routed_units::relu2(
                    g,
                    RoutedUnitSite {
                        node: &op,
                        layer: i,
                        routing: &format!("{path}.routing"),
                        input: Some(format!("{path}.feed_forward.input")),
                        normalization: Some(super::components::declared_normalization(&scope.norm)),
                        residual_scale: Some(ComponentScalar::new(1.0)),
                    },
                    Ok(spec.experts.clone()),
                );
            }
            (P::Dense(_), LayerPolicy::DenseMlp) => {
                super::components::nemotron_unit(g, scope, i, &path, &op)
            }
            _ => unreachable!("validated Nemotron invocation and scope agree"),
        }
        previous = output;
    }
    declared_static_output(g, &previous, &statics);
    super::components::nemotron_readout(g, declaration);
    if c.num_nextn_predict_layers > 0 {
        super::components::nemotron::prediction::declare(g, declaration, &previous);
    }
}

fn deepseek_v3(g: &mut Builder, c: &crate::deepseek::V3Args) {
    let statics = &c.parameters().statics;
    let width = statics.embedding_spec().dimensions as usize;
    let mut previous = declared_static_input(g, &statics);
    for i in 0..c.layer_schedule.len() {
        let path = format!("model.layers.{i}");
        let block = g.block(&previous, i, &path, width);
        previous = deepseek_v3_unit(g, c, i, &path, &block);
    }
    declared_static_output(g, &previous, &statics);
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
    let spec = crate::deepseek::v3_specs::attention_spec(c, &format!("{path}.self_attn"))
        .expect("admitted V3 MLA declaration");
    g.get_mut(&op).attention = Some(AttentionAttributes {
        mechanism: Some(AttentionMechanism::Latent),
        recurrent: Some(false),
        causal: Some(true),
        query_heads: Some(spec.geometry.heads() as usize),
        key_head_dimension: Some(spec.geometry.query_head_dimensions() as usize),
        value_head_dimension: Some(spec.geometry.value_dimensions() as usize),
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
    let statics = &c.parameters().statics;
    let width = statics.embedding_spec().dimensions as usize;
    let mut previous = declared_static_input(g, &statics);
    for (i, policy) in c
        .attention_schedule
        .iter()
        .take(c.num_hidden_layers as usize)
        .enumerate()
    {
        let path = format!("layers.{i}");
        let attention_spec =
            crate::deepseek::v4_specs::attention_spec(c, i, &format!("{path}.attn"))
                .expect("validated V4 attention declaration");
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
            query_heads: Some(attention_spec.geometry.query_heads() as usize),
            key_value_heads: Some(attention_spec.geometry.key_value_heads() as usize),
            key_head_dimension: Some(attention_spec.geometry.key_dimensions() as usize),
            value_head_dimension: Some(attention_spec.geometry.value_dimensions() as usize),
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
        let routed_spec = crate::deepseek::v4_specs::moe_spec(c, i, &format!("{path}.ffn"), None)
            .expect("validated V4 routed declaration");
        let selector = &routed_spec.routed.parameters.selector;
        g.moe(
            &ff,
            &format!("{path}.ffn"),
            moe(
                selector.selection().group_count(),
                selector.selection().top_k(),
                routed_spec.routed.shared.gate.output
                    / routed_spec
                        .routed
                        .parameters
                        .experts
                        .intermediate_dimensions(),
                selector.selection().normalize_selected(),
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
                normalization: Some(super::components::declared_normalization(
                    &crate::deepseek::v4_specs::block_normalizations(c, &path)
                        .expect("validated V4 block normalization")[1],
                )),
                residual_scale: None,
            },
            Ok(routed_spec.routed.parameters.experts),
        );
        super::components::deepseek_v4::unit(g, c, i, &path, &path, &op, &ff);
        g.get_mut(&block).completeness = DescriptionCompleteness::Partial(vec![
            "Compressed-history provenance and hash selection policy remain partially described"
                .into(),
        ]);
        previous = block;
    }
    declared_static_output(g, &previous, &statics);
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
    statics: &crate::decoder::StaticModuleConstruction,
) {
    use eredu_runtime::ComponentKind as K;
    let width = statics.embedding_spec().dimensions as usize;
    let vocab = statics.output_projection_spec().output as usize;
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
        g.get_mut(&component.id).layer_index = Some(index);
        g.decoder_execution(&component.id, index);
        g.get_mut(&component.id).output_axes = Some(axes(width));
        g.get_mut(&component.id).completeness = DescriptionCompleteness::Partial(vec!["Block equations, additional normalizations, state sharing, and internal data flow are not expanded".into()]);
        for (boundary, meaning) in [
            (UnitObservation::Input, "Block input"),
            (UnitObservation::Output, "Block output"),
        ] {
            g.observation(
                &component.id,
                boundary.path(&path),
                meaning,
                ObservationDtype::Floating,
                Some(axes(width)),
                false,
            );
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

fn diffusion_gemma(g: &mut Builder, c: &crate::diffusion_gemma::FamilyConfig) {
    let text = &c.text;
    let declaration = c.pinned();
    let width = declaration.embedding().dimensions as usize;
    g.node(
        "text.embedding",
        ArchitectureNodeKind::Embedding,
        None,
        Some(super::components::declared_weight_prefix(
            &declaration.embedding().weight,
        )),
        Some(width),
    );
    g.node(
        "assembly",
        ArchitectureNodeKind::Sum,
        None,
        None,
        Some(width),
    );
    g.edge("text.embedding", "assembly", ArchitectureEdgeKind::Data);
    let mut previous = "assembly".to_owned();
    g.node(
        "context",
        ArchitectureNodeKind::Encoder,
        None,
        None,
        Some(width),
    );
    let conditioning = c.conditioning();
    g.node(
        "refinement",
        ArchitectureNodeKind::DecoderBlock,
        None,
        Some(crate::diffusion_gemma::conditioning::Declaration::ROOT),
        Some(conditioning.product.gate.input as usize),
    );
    g.edge("context", "text.embedding", ArchitectureEdgeKind::Data);
    g.edge("refinement", "text.embedding", ArchitectureEdgeKind::Data);
    for (i, policy) in text.layer_schedule.iter().enumerate() {
        let declaration = c
            .attention(i)
            .expect("admitted shared attention declaration");
        let geometry = declaration.geometry;
        let block = format!("decoder.{i}");
        g.node(
            &block,
            ArchitectureNodeKind::DecoderBlock,
            None,
            Some(&format!("model.decoder.layers.{i}")),
            Some(width),
        );
        g.get_mut(&block).layer_index = Some(i);
        g.decoder_execution(&block, i);
        g.edge(&previous, &block, ArchitectureEdgeKind::Data);
        contained_attention(
            g,
            i,
            attention(
                geometry.query_heads(),
                geometry.key_value_heads(),
                geometry.key_dimensions(),
                policy.attention,
            ),
        );
        let sparse = policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe;
        contained_ff(
            g,
            i,
            sparse.then(|| {
                let declaration = c
                    .routed(i)
                    .expect("admitted shared layer")
                    .expect("declared routed source");
                declared_routing(declaration.routed.selector.selection(), 0)
                    .expect("admitted routed scoring equation")
            }),
        );
        if sparse {
            g.node(
                &format!("{block}.dense_feed_forward"),
                ArchitectureNodeKind::FeedForward,
                Some(&block),
                None,
                None,
            );
        }
        super::components::gemma4::unit_at(
            g,
            text,
            i,
            "model.decoder",
            c.block(i).expect("admitted shared block"),
            c.dense(i).expect("admitted shared dense branch"),
            c.attention_scopes(),
            c.routed(i).expect("admitted routed scope"),
        );
        g.attach_parameters(
            &block,
            c.encoder_scalar(i)
                .expect("validated encoding scale")
                .weight
                .id
                .as_str(),
        );
        previous = block;
    }
    let group = &mut g.descriptor.layer_groups[0];
    group.id = "shared_text".into();
    group.label = "Shared context and refinement layers".into();
    group.weight_sharing = LayerWeightSharing::SharedAcrossPasses;
    group.passes.push(ArchitectureExecutionPass {
        index: 1,
        executions: group.passes[0].executions.clone(),
    });
    declared_static_output(g, &previous, &declaration.ordinary);
    super::components::gemma4::readout(g, text, declaration);
    if let Some(spec) = c.vision_projector() {
        media(g, "vision");
        super::components::gemma4::modality_projector(g, "vision.projector", spec);
    }
}

fn gemma(g: &mut Builder, c: &crate::gemma4::FamilyConfig) {
    let text = &c.text;
    let declaration = c.pinned();
    let graph = crate::gemma4::component_graph(
        text,
        crate::gemma4::ComponentOptions {
            vision: c.vision.is_some(),
            audio: c.audio.is_some(),
            ..Default::default()
        },
    )
    .expect("admitted component geometry");
    components(g, graph, "model.language_model", &declaration.ordinary);
    if let Some(spec) = c.projector(crate::gemma4::media::ProjectionKind::Vision) {
        super::components::gemma4::modality_projector(g, "vision.projector", spec);
    }
    if let Some(spec) = c.projector(crate::gemma4::media::ProjectionKind::Audio) {
        super::components::gemma4::modality_projector(g, "audio.projector", spec);
    }
    for (i, policy) in text.layer_schedule.iter().enumerate() {
        let declaration = c.attention(i).expect("admitted attention declaration");
        let geometry = declaration.geometry;
        contained_attention(
            g,
            i,
            attention(
                geometry.query_heads(),
                geometry.key_value_heads(),
                geometry.key_dimensions(),
                policy.attention,
            ),
        );
        let sparse = policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe;
        contained_ff(
            g,
            i,
            sparse.then(|| {
                let declaration = c
                    .routed(i)
                    .expect("admitted layer")
                    .expect("declared routed source");
                declared_routing(declaration.routed.selector.selection(), 0)
                    .expect("admitted routed scoring equation")
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
        super::components::gemma4::unit(
            g,
            text,
            i,
            c.block(i).expect("admitted block"),
            c.dense(i).expect("admitted dense branch"),
            c.attention_scopes(),
            c.routed(i).expect("admitted routed scope"),
        );
    }
    super::components::gemma4::readout(g, text, declaration);
}

fn muse(g: &mut Builder, c: &crate::muse::DecoderConfig) {
    let statics = c.pinned();
    components(
        g,
        crate::muse::component_graph(c).expect("admitted component geometry"),
        "model",
        statics,
    );
    if let Some(vision) = &c.vision_config {
        use eredu_nn::DeclareLogicalParameters;
        let declarations = vision.declaration();
        let projector_parameters = declarations
            .statics
            .adapter
            .logical_parameters()
            .expect("validated Muse adapter slots")
            .into_iter()
            .chain(
                declarations
                    .statics
                    .projection
                    .logical_parameters()
                    .expect("validated Muse projection slots"),
            )
            .map(|parameter| parameter.parameter.id.as_str().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        for parameter in declarations
            .logical_parameters()
            .expect("validated Muse vision slots")
        {
            let name = parameter.parameter.id.as_str();
            let node = if projector_parameters.contains(name) {
                "vision.projector"
            } else {
                "vision"
            };
            let prefix = name
                .strip_suffix(".weight")
                .or_else(|| name.strip_suffix(".bias"))
                .unwrap_or(name);
            g.attach_parameters(node, prefix);
        }
    }
    if let Some(audio) = &c.audio_config {
        use eredu_nn::DeclareLogicalParameters;
        let declarations = audio.declaration();
        for (kind, spec) in &declarations.units {
            let node = if *kind == crate::muse::audio::AudioUnitKind::Projection {
                "audio.projector"
            } else {
                "audio"
            };
            for parameter in spec
                .logical_parameters()
                .expect("validated Muse audio slots")
            {
                let name = parameter.parameter.id.as_str();
                let prefix = name
                    .strip_suffix(".weight")
                    .or_else(|| name.strip_suffix(".bias"))
                    .unwrap_or(name);
                g.attach_parameters(node, prefix);
            }
        }
    }
    for (i, policy) in c.attention_schedule.iter().enumerate() {
        let parameters = &c
            .attention(i)
            .expect("admitted Muse attention declaration")
            .parameters;
        let geometry = parameters.projection.geometry;
        let mut attrs = attention(
            geometry.query_heads(),
            geometry.key_value_heads(),
            geometry.key_dimensions(),
            *policy,
        );
        attrs.positional_encoding = Some(if c.layer_uses_rope[i] {
            PositionalEncoding::Rotary
        } else {
            PositionalEncoding::None
        });
        contained_attention(g, i, attrs);
        let selection = c
            .routed(i)
            .map(|declaration| declaration.selector().selection());
        contained_ff(
            g,
            i,
            selection
                .as_ref()
                .ok()
                .and_then(|selection| declared_routing(*selection, 0)),
        );
        if c.is_moe() {
            match selection {
                Ok(spec) => g.routing_control(
                    &format!("decoder.{i}.feed_forward"),
                    &format!("model.layers.{i}.routing"),
                    spec,
                    0,
                    false,
                ),
                Err(error) => {
                    g.partial(format!("Muse routing control declaration failed: {error}"))
                }
            }
        }
        super::components::muse::unit(g, c, i);
    }
    super::components::muse::readout(g, c, statics);
}

fn inkling(g: &mut Builder, c: &crate::inkling::ModelArgs) {
    let text = &c.text_config;
    let statics = &c.pinned().ordinary;
    components(
        g,
        crate::inkling::component_graph(c).expect("admitted component geometry"),
        "model",
        &statics,
    );
    super::components::inkling::media(g, c);
    for (i, policy) in text.layer_schedule.iter().enumerate() {
        let common = c.common(i).expect("admitted Inkling common declaration");
        let parameters = &common.attention;
        let dense = (policy.feed_forward == crate::inkling::FeedForwardPolicy::Dense)
            .then(|| c.dense(i).expect("admitted Inkling dense declaration"));
        let mut attrs = attention(
            parameters.projection.geometry.query_heads(),
            parameters.projection.geometry.key_value_heads(),
            parameters.projection.geometry.key_dimensions(),
            policy.attention,
        );
        attrs.positional_encoding = Some(PositionalEncoding::Relative);
        contained_attention(g, i, attrs);
        contained_ff(
            g,
            i,
            (policy.feed_forward == crate::inkling::FeedForwardPolicy::SparseMoe).then(|| {
                let parameters = c.sparse(i).expect("admitted Inkling routing declaration");
                let selection = parameters.router.selection();
                let mut attrs = moe(
                    selection.selectable_groups(),
                    selection.top_k(),
                    selection.always_on_groups(),
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
            *policy,
            i,
            &format!("decoder.{i}"),
            &format!("model.layers.{i}"),
            &common,
            dense,
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
            *policy,
            &format!("decoder.{i}"),
            &format!("model.layers.{i}"),
            &common,
            dense,
        );
    }
    super::components::inkling::target_readout(g, c);
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
    let declarations = c.declarations();
    for (node, parameters) in [
        (
            "temporal",
            declarations
                .temporal_parameters()
                .expect("validated Moshi temporal declarations"),
        ),
        (
            "depth",
            declarations
                .depth_parameters()
                .expect("validated Moshi depth declarations"),
        ),
    ] {
        for parameter in parameters {
            let prefix = parameter.parameter.id.as_str().trim_end_matches(".weight");
            g.attach_parameters(node, prefix);
            if let Some(owner) = &parameter.parameter.alias_of {
                let shared_with = g.parameter_group(owner.as_str().trim_end_matches(".weight"));
                let id = format!("parameters:{prefix}");
                g.descriptor
                    .parameter_groups
                    .iter_mut()
                    .find(|group| group.id == id)
                    .expect("declared Moshi parameter group")
                    .shared_with = Some(shared_with);
            }
        }
    }
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

pub(super) fn k2_horizon(g: &mut Builder, source: &crate::k2_horizon::Declaration) {
    let c = source.args();
    let common = source
        .common_projection()
        .expect("admitted K2 common projection");
    dense(g, c, None, &common);

    for layer in 0..c.num_hidden_layers as usize {
        let path = format!("model.layers.{layer}");
        let Some(points) = c.routed_observation_points(&path, layer) else {
            continue;
        };
        let declaration = source
            .block(layer)
            .expect("validated K2 routed declaration");
        for (bank, node, shared) in [
            (
                crate::k2_horizon::ExpertBank::FeedForward,
                format!("decoder.layers.{layer}.feed_forward"),
                c.num_shared_experts,
            ),
            (
                crate::k2_horizon::ExpertBank::AttentionValue,
                format!("decoder.layers.{layer}.attention.values"),
                0,
            ),
        ] {
            let Some(point) = points.bank(bank.id()) else {
                continue;
            };
            let selector = match bank {
                crate::k2_horizon::ExpertBank::FeedForward => {
                    &declaration
                        .routed
                        .as_ref()
                        .expect("declared feed-forward bank")
                        .parameters
                        .selector
                }
                crate::k2_horizon::ExpertBank::AttentionValue => {
                    &declaration
                        .values
                        .as_ref()
                        .expect("declared value bank")
                        .router
                }
            };
            let selection = selector.selection();
            let width = if bank == crate::k2_horizon::ExpertBank::AttentionValue {
                let value = &declaration
                    .values
                    .as_ref()
                    .expect("declared value bank")
                    .experts;
                let attention = format!("decoder.layers.{layer}.attention");
                g.node(
                    &node,
                    ArchitectureNodeKind::MixtureOfExperts,
                    Some(&attention),
                    Some(&format!("{path}.self_attn.v_experts")),
                    Some(value.output_dimensions() as usize),
                );
                value.output_dimensions() as usize
            } else {
                selector.input_dimensions() as usize
            };
            let mut attributes =
                declared_routing(selection, shared).expect("validated K2 routing equation");
            attributes.shared_expert_width = if shared > 0 {
                declaration
                    .routed
                    .as_ref()
                    .and_then(|routed| routed.shared.as_ref())
                    .map(|spec| spec.gate.output as usize)
            } else {
                None
            };
            g.moe(&node, point.path(), attributes, Some(point.path()), width);
            g.routing_control(&node, point.path(), selection, shared as u32, false);
        }
    }
    routed_units::decoder(g, c, |layer| {
        source.block(layer).and_then(|block| {
            block
                .routed
                .map(|routed| routed.parameters.experts)
                .ok_or_else(|| eredu_nn::Error::backend("K2 layer has no feed-forward bank"))
        })
    });
    super::components::k2_horizon::complete(g, source);
}

// Configuration-only projection: bounds and native state mechanisms belong to
// retained selection. Do not manufacture a TargetSpec with arbitrary limits here.
fn qwen4_exp(
    g: &mut Builder,
    c: &crate::qwen4_exp::config::Config,
    declaration: &crate::qwen4_exp::target::declarations::TargetParameters,
) {
    use crate::qwen4_exp::{config::LayerKind, target::declarations::MixerParameters};
    use ArchitectureEdgeKind as E;
    use ArchitectureNodeKind as K;
    let statics = &declaration.static_parameters;
    let output = statics.output_projection();
    let hidden = statics.embedding.dimensions as usize;
    let mut residual_axes = axes(hidden);
    residual_axes.insert(
        2,
        TensorAxis {
            name: "stream".into(),
            dimension: SymbolicDimension::Known(statics.readout.geometry.streams() as usize),
        },
    );
    let embedding = g.start(
        statics
            .embedding
            .weight
            .id
            .as_str()
            .strip_suffix(".weight")
            .expect("declared vocabulary weight"),
        hidden,
    );
    g.node("tokens", K::Processor, None, None, None);
    g.get_mut("tokens").label = "Original token IDs retained through media replacement".into();
    g.edge("tokens", &embedding, E::Data);
    g.node("residual.expansion", K::Projector, None, None, None);
    g.get_mut("residual.expansion").label = "Replicate embeddings into residual streams".into();
    g.get_mut("residual.expansion").output_axes = Some(residual_axes.clone());
    g.edge(&embedding, "residual.expansion", E::Data);
    let mut previous = "residual.expansion".to_owned();
    for (layer, kind) in c.layers.iter().enumerate() {
        let (mixer, feed_forward) = declaration
            .decoder(layer)
            .expect("declared Qwen4 invocation");
        let path = format!("model.layers.{layer}");
        if c.ngram.layers.contains(&layer) {
            let lexical = declaration
                .units
                .iter()
                .find_map(|unit| match unit {
                    crate::qwen4_exp::target::declarations::UnitParameters::Lexical {
                        layer: owner,
                        parameters,
                    } if *owner == layer => Some(parameters),
                    _ => None,
                })
                .expect("declared Qwen4 lexical invocation");
            let id = format!("decoder.layers.{layer}.lexical");
            g.node(&id, K::Mixer, None, Some(&format!("{path}.ple")), None);
            g.get_mut(&id).label = format!(
                "Gated n-gram injection; convolution dilation {}",
                lexical.convolution.dilation
            );
            g.get_mut(&id).layer_index = Some(layer);
            g.get_mut(&id).output_axes = Some(residual_axes.clone());
            g.get_mut(&id).mixer = Some(MixerAttributes {
                mechanism: MixerMechanism::ShortConvolution,
                recurrent: true,
                convolution_width: Some(lexical.convolution.kernel_size as usize),
            });
            let table = format!("{id}.table");
            g.node(
                &table,
                K::Embedding,
                Some(&id),
                Some(&format!("{path}.ple.ple_embedding.ngram_embedding")),
                Some(lexical.key.input as usize),
            );
            g.get_mut(&table).label = "Logical row-addressable n-gram table".into();
            g.edge("tokens", &table, E::Data);
            g.edge(&table, &id, E::Data);
            g.edge(&previous, &id, E::Residual);
            qwen4_state(
                g,
                &id,
                "Original token history, dilated convolution history and sequence position",
            );
            previous = id;
        }
        let block = g.block(&previous, layer, &path, hidden);
        g.get_mut(&block).output_axes = Some(residual_axes.clone());
        for point in &mut g.descriptor.observations.points {
            if point.node_id == block {
                point.axes = Some(residual_axes.clone());
            }
        }
        let mut input = block.clone();
        for (suffix, parameter) in [
            ("mixer", "attn_hyper_connection"),
            ("feed_forward", "mlp_hyper_connection"),
        ] {
            let collapse = format!("{block}.{suffix}.collapse");
            let residual = format!("{block}.{suffix}.residual");
            let op = format!("{block}.{suffix}");
            g.node(
                &collapse,
                K::Projector,
                Some(&block),
                Some(&format!("{path}.{parameter}")),
                Some(hidden),
            );
            g.get_mut(&collapse).label = "Gated multi-stream residual input collapse".into();
            g.edge(&input, &collapse, E::Data);
            let recurrent = *kind == LayerKind::Recurrent;
            let op_path = if suffix == "feed_forward" {
                format!("{path}.mlp")
            } else {
                format!(
                    "{path}.{}",
                    if recurrent {
                        "linear_attn"
                    } else {
                        "self_attn"
                    }
                )
            };
            g.node(
                &op,
                if suffix == "feed_forward" {
                    K::FeedForward
                } else if recurrent {
                    K::Mixer
                } else {
                    K::Attention
                },
                Some(&block),
                Some(&op_path),
                Some(hidden),
            );
            g.edge(&collapse, &op, E::Data);
            g.node(
                &residual,
                K::ResidualAdd,
                Some(&block),
                Some(&format!("{path}.{parameter}")),
                None,
            );
            g.get_mut(&residual).label = "Gated write to every retained residual stream".into();
            g.get_mut(&residual).output_axes = Some(residual_axes.clone());
            g.edge(&input, &residual, E::Residual);
            g.edge(&op, &residual, E::Data);
            if suffix == "feed_forward" {
                let parameters = &feed_forward.feed_forward.parameters;
                let selection = parameters.routed.selector.selection();
                g.moe(
                    &op,
                    &op_path,
                    moe(
                        parameters.routed.experts.group_count(),
                        selection.top_k(),
                        1,
                        selection.normalize_selected(),
                        RoutingScoreTransform::Softmax,
                    ),
                    None,
                    hidden,
                );
            } else if recurrent {
                let MixerParameters::Recurrent(spec) = mixer else {
                    unreachable!("declared recurrent invocation")
                };
                g.get_mut(&op).label =
                    "Gated delta recurrent mixer with sigmoid output gate".into();
                g.get_mut(&op).mixer = Some(MixerAttributes {
                    mechanism: MixerMechanism::GatedDelta,
                    recurrent: true,
                    convolution_width: Some(spec.mixer.convolution.kernel_size as usize),
                });
                qwen4_state(g, &op, "Recurrent matrix and causal convolution history");
            } else {
                let MixerParameters::Indexed { parameters, .. } = mixer else {
                    unreachable!("declared indexed invocation")
                };
                let geometry = parameters.projection.geometry;
                g.get_mut(&op).label = format!(
                    "QSA-selected original K/V; block ratio {}, selection budget {}",
                    c.attention.ratio, c.attention.budget
                );
                g.get_mut(&op).attention = Some(AttentionAttributes {
                    head_sharing: Some(if geometry.query_heads() == geometry.key_value_heads() {
                        HeadSharing::MultiHead
                    } else if geometry.key_value_heads() == 1 {
                        HeadSharing::MultiQuery
                    } else {
                        HeadSharing::GroupedQuery
                    }),
                    query_heads: Some(geometry.query_heads() as usize),
                    key_value_heads: Some(geometry.key_value_heads() as usize),
                    key_head_dimension: Some(geometry.key_dimensions() as usize),
                    value_head_dimension: Some(geometry.value_dimensions() as usize),
                    mechanism: Some(AttentionMechanism::Softmax),
                    causal: Some(true),
                    recurrent: Some(false),
                    positional_encoding: Some(PositionalEncoding::Rotary),
                    ..Default::default()
                });
                qwen4_state(
                    g,
                    &op,
                    "Full K/V history, completed index summaries, positions and partial micro-block",
                );
            }
            input = residual;
        }
        previous = input;
    }
    g.node(
        "output.collapse",
        K::Projector,
        None,
        Some(
            statics
                .readout
                .down
                .weight
                .group
                .as_deref()
                .expect("declared residual owner"),
        ),
        Some(hidden),
    );
    g.get_mut("output.collapse").label =
        "Gated final residual collapse without another RMSNorm".into();
    g.edge(&previous, "output.collapse", E::Data);
    g.node(
        "output",
        K::OutputHead,
        None,
        Some(
            output
                .weight
                .id
                .as_str()
                .strip_suffix(".weight")
                .expect("declared vocabulary weight"),
        ),
        Some(output.output as usize),
    );
    g.edge("output.collapse", "output", E::Data);
    let mut logits = axes(output.output as usize);
    logits[2].name = "vocabulary".into();
    g.get_mut("output").output_axes = Some(logits.clone());
    g.observation(
        "output",
        MODEL_LOGITS_OBSERVATION_PATH.into(),
        "Final vocabulary logits",
        ObservationDtype::Floating,
        Some(logits),
        false,
    );
    if c.prediction.is_some() {
        prediction(g, &previous);
    }
    if c.vision.is_some() {
        media(g, "vision");
    }
    g.partial("Residual and state ownership are described; component decomposition and prediction internals require retained execution specifications");
    g.catalog_partial("Configuration discovery declares unit boundaries and logits; internal capture and intervention points require retained execution discovery");
}

fn qwen4_state(g: &mut Builder, owner: &str, label: &str) {
    let state = format!("{owner}.state");
    g.node(
        &state,
        ArchitectureNodeKind::Opaque,
        Some(owner),
        None,
        None,
    );
    g.get_mut(&state).label = label.into();
    g.edge(&state, owner, ArchitectureEdgeKind::State);
    g.edge(owner, &state, ArchitectureEdgeKind::State);
}
