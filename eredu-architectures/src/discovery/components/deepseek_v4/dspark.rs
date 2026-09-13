//! Fused proposal equations and separate accepted-context cache invocations.
use super::*;
use crate::discovery::routed_components::{self as routed_units, Site as RoutedUnitSite};
use eredu_core::speculative::{SpeculativeCaptureBinding, SpeculativeCaptureScope};

pub(in crate::discovery) fn declare(g: &mut Builder, c: &V4Args, target_output: &str) {
    let config = c.dspark.as_ref().expect("selected fused configuration");
    let width = c.hidden_size as usize;
    let count = c.num_nextn_predict_layers as usize;
    let last = count - 1;
    let root = "prediction.proposal";
    let base = "dspark.proposal";
    g.node(
        "prediction",
        ArchitectureNodeKind::Prediction,
        None,
        None,
        None,
    );
    g.edge(target_output, "prediction", ArchitectureEdgeKind::Data);
    context(g, c);
    g.node(
        root,
        ArchitectureNodeKind::Prediction,
        Some("prediction"),
        None,
        None,
    );
    let embedding = format!("{root}.embedding");
    g.node(
        &embedding,
        ArchitectureNodeKind::Embedding,
        Some(root),
        Some("embed"),
        Some(width),
    );
    g.component_observation(
        &embedding,
        format!("{base}.embedding"),
        "Anchor and proposal-noise token embeddings before stream broadcast",
        axes(width),
    );
    g.component_observation(
        root,
        format!("{base}.input"),
        "Embedding residual expanded over proposal streams",
        stream_axes(c),
    );
    g.edge(&embedding, root, ArchitectureEdgeKind::Data);
    let component_start = g.descriptor.components.len();
    let routed_start = g.descriptor.routed_components.len();
    let mut cycles = Vec::new();
    let mut groups = Vec::new();
    let mut previous = root.to_owned();
    for depth in 0..count {
        let layer = c.num_hidden_layers as usize + depth;
        let parameters = format!("mtp.{depth}");
        let node = format!("{root}.layers.{depth}");
        let path = format!("{base}.layers.{depth}");
        let attention = format!("{node}.attention");
        let ffn = format!("{node}.feed_forward");
        g.node(
            &node,
            ArchitectureNodeKind::DecoderBlock,
            Some(root),
            Some(&parameters),
            None,
        );
        g.get_mut(&node).layer_index = Some(layer);
        g.get_mut(&node).output_axes = Some(stream_axes(c));
        g.edge(&previous, &node, ArchitectureEdgeKind::Data);
        previous = node.clone();
        g.descriptor.layer_groups.push(ArchitectureLayerGroup {
            id: parameters.clone(),
            label: format!("Fused proposal block {depth}"),
            physical_layer_count: 1,
            passes: vec![ArchitectureExecutionPass {
                index: 0,
                executions: vec![ArchitectureLayerExecution {
                    node_id: node.clone(),
                    physical_layer_index: 0,
                }],
            }],
            weight_sharing: LayerWeightSharing::None,
        });
        groups.push(parameters.clone());
        for suffix in ["input", "output"] {
            g.component_observation(
                &node,
                format!("{path}.{suffix}"),
                "Fused proposal decoder residual streams",
                stream_axes(c),
            );
        }
        g.node(
            &attention,
            ArchitectureNodeKind::Attention,
            Some(&node),
            Some(&format!("{parameters}.attn")),
            Some(width),
        );
        g.get_mut(&attention).attention = Some(AttentionAttributes {
            mechanism: Some(AttentionMechanism::CompressedSparse),
            causal: Some(false),
            recurrent: Some(false),
            query_heads: Some(c.num_attention_heads as usize),
            key_value_heads: Some(1),
            key_head_dimension: Some(c.head_dim as usize),
            value_head_dimension: Some(c.head_dim as usize),
            positional_encoding: Some(PositionalEncoding::Rotary),
            ..Default::default()
        });
        g.node(
            &ffn,
            ArchitectureNodeKind::MixtureOfExperts,
            Some(&node),
            Some(&format!("{parameters}.ffn")),
            Some(width),
        );
        g.moe(
            &ffn,
            &format!("{parameters}.ffn"),
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
                node: &ffn,
                layer,
                routing: &format!("{path}.feed_forward"),
                input: Some(format!("{path}.feed_forward.input")),
                normalization: Some(routed_units::rms(
                    format!("{parameters}.ffn_norm.weight"),
                    c.rms_norm_eps,
                    0.0,
                )),
                residual_scale: None,
            },
            crate::deepseek::v4::expert_bank_spec(c, layer),
        );
        unit(g, c, layer, &path, &parameters, &attention, &ffn);
        cycles.extend(stream_cycles(
            c,
            layer,
            &path,
            &parameters,
            &attention,
            &ffn,
        ));
    }
    let components = g.descriptor.components.split_off(component_start);
    let routed_components = g.descriptor.routed_components.split_off(routed_start);
    let readout = format!("{base}.readout");
    let head = format!("{root}.head");
    let collapse = format!("{head}.collapse");
    let head_norm = format!("{head}.norm");
    g.node(
        &collapse,
        ArchitectureNodeKind::Sum,
        Some(root),
        Some(&format!("mtp.{last}")),
        Some(width),
    );
    g.node(
        &head_norm,
        ArchitectureNodeKind::Normalization,
        Some(root),
        Some(&format!("mtp.{last}.norm")),
        Some(width),
    );
    g.node(
        &head,
        ArchitectureNodeKind::OutputHead,
        Some(root),
        Some("head"),
        None,
    );
    g.edge(&previous, &collapse, ArchitectureEdgeKind::Data);
    g.edge(&collapse, &head_norm, ArchitectureEdgeKind::Data);
    g.edge(&head_norm, &head, ArchitectureEdgeKind::Data);
    g.component_observation(
        &collapse,
        format!("{readout}.streams"),
        "Proposal streams consumed by final learned collapse",
        stream_axes(c),
    );
    let mut coefficient_axes = axes(c.hc_mult as usize);
    coefficient_axes[2].name = "stream".into();
    g.read_only_observation(
        &collapse,
        format!("{readout}.stream_coefficients"),
        "Measured proposal head stream-collapse coefficients",
        coefficient_axes,
    );
    g.component_observation(
        &collapse,
        format!("{readout}.residual"),
        "Collapsed proposal residual before normalization",
        axes(width),
    );
    g.component_observation(
        &head_norm,
        format!("{readout}.normalized"),
        "Normalized proposal residual consumed by shared head",
        axes(width),
    );
    let mut vocabulary = axes(c.vocab_size as usize);
    vocabulary[2].name = "vocabulary".into();
    g.get_mut(&head).output_axes = Some(vocabulary.clone());
    g.component_observation(
        &head,
        format!("{readout}.linear"),
        "Primary proposal head scores before dynamic score additions",
        vocabulary.clone(),
    );
    let markov = format!("{root}.markov");
    let markov_embedding = format!("{markov}.embedding");
    let markov_input = format!("{base}.markov.input");
    let markov_output = format!("{base}.markov.output");
    let first = format!("mtp.{last}.markov_head.markov_w1.weight");
    let second = format!("mtp.{last}.markov_head.markov_w2.weight");
    g.node(
        &markov_embedding,
        ArchitectureNodeKind::Embedding,
        Some(root),
        Some(&format!("mtp.{last}.markov_head.markov_w1")),
        Some(config.markov_rank as usize),
    );
    g.node(
        &markov,
        ArchitectureNodeKind::Projector,
        Some(root),
        Some(&format!("mtp.{last}.markov_head.markov_w2")),
        None,
    );
    g.edge(&markov_embedding, &markov, ArchitectureEdgeKind::Data);
    let mut anchor_axes = axes(config.markov_rank as usize);
    anchor_axes[1].dimension = SymbolicDimension::Known(1);
    g.component_observation(
        &markov_embedding,
        markov_input.clone(),
        "Anchor-dependent Markov embedding consumed by its vocabulary projection",
        anchor_axes.clone(),
    );
    g.projection_input_observation(
        &markov,
        format!("{base}.markov.projection_input"),
        anchor_axes,
    );
    let mut markov_axes = vocabulary.clone();
    markov_axes[1].dimension = SymbolicDimension::Known(1);
    g.component_observation(
        &markov,
        markov_output.clone(),
        "Dynamic anchor-dependent scores before proposal-row broadcast",
        markov_axes,
    );
    let scores = format!("{root}.scores");
    g.node(&scores, ArchitectureNodeKind::Sum, Some(root), None, None);
    g.get_mut(&scores).output_axes = Some(vocabulary.clone());
    g.edge(&head, &scores, ArchitectureEdgeKind::Data);
    g.edge(&markov, &scores, ArchitectureEdgeKind::Data);
    g.component_observation(
        &scores,
        format!("{base}.logits"),
        "Primary proposal scores plus broadcast Markov scores, before sampling",
        vocabulary,
    );
    g.descriptor.component_scopes.push(ComponentExecutionScope {
        id: "prediction.fused".into(),
        node_id: root.into(),
        execution_groups: groups,
        static_parameter_roles: vec!["embedding".into(), "output".into(), "mtp".into()],
        kind: ComponentExecutionScopeKind::FusedPrediction,
        components,
        routed_components,
        residual_base: ComponentResidualBase::Source {
            source: ComponentFusionSource::TokenEmbedding {
                normalization: None,
                weight: "embed.weight".into(),
                shared_weight: "embed.weight".into(),
                parameter_group: "parameters:embed".into(),
                scale: ComponentScalar::new(1.0),
            },
            input: format!("{base}.embedding.effective"),
            expansion: ComponentFusionExpansion::BroadcastAxis {
                axis: 2,
                name: "stream".into(),
                extent: c.hc_mult as usize,
            },
            output: format!("{base}.input"),
            effective_output: format!("{base}.input.effective"),
        },
        readout: ComponentReadoutEquation {
            block_transforms: vec![],
            stream_residual: Some(ComponentStreamResidual {
                streams: c.hc_mult as usize,
                base: ComponentStreamBase::Streams {
                    input: format!("{base}.input.effective"),
                },
                cycles,
                head: ComponentStreamHead {
                    input: format!("{readout}.streams.effective"),
                    coefficients: format!("{readout}.stream_coefficients"),
                    parameters: coefficients(
                        c,
                        &format!("parameters:mtp.{last}"),
                        &format!("mtp.{last}.hc_head"),
                    ),
                },
            }),
            residual: format!("{readout}.residual"),
            normalized: format!("{readout}.normalized"),
            projection_input: Some(format!("{readout}.projection_input")),
            linear_scores: format!("{readout}.linear"),
            logits: format!("{base}.logits.effective"),
            normalization: norm(c, Some(format!("mtp.{last}.norm.weight"))),
            weight: "head.weight".into(),
            bias: None,
            score_writes: vec![ComponentScoreWrite {
                node_id: markov,
                source: ComponentFusionSource::TokenEmbedding {
                    normalization: None,
                    weight: first.clone(),
                    shared_weight: first,
                    parameter_group: format!("parameters:mtp.{last}.markov_head.markov_w1"),
                    scale: ComponentScalar::new(1.0),
                },
                input: format!("{markov_input}.effective"),
                projection_input: format!("{base}.markov.projection_input"),
                weight: second.clone(),
                shared_weight: second,
                parameter_group: format!("parameters:mtp.{last}.markov_head.markov_w2"),
                bias: None,
                output: markov_output.clone(),
                effective_output: format!("{markov_output}.effective"),
                broadcast_axes: vec!["sequence".into()],
            }],
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes: vec![],
        },
    });
}

fn context(g: &mut Builder, c: &V4Args) {
    let root = "prediction.context";
    let path = "dspark.context";
    let width = c.hidden_size as usize;
    g.node(
        root,
        ArchitectureNodeKind::Prediction,
        Some("prediction"),
        None,
        None,
    );
    g.descriptor
        .speculative_invocations
        .push(SpeculativeCaptureBinding {
            node_id: root.into(),
            scope: SpeculativeCaptureScope::PredictionContext,
        });
    let projection = format!("{root}.projection");
    let normalization = format!("{root}.norm");
    g.node(
        &projection,
        ArchitectureNodeKind::Projector,
        Some(root),
        Some("mtp.0.main_proj"),
        Some(width),
    );
    g.node(
        &normalization,
        ArchitectureNodeKind::Normalization,
        Some(root),
        Some("mtp.0.main_norm"),
        Some(width),
    );
    g.edge(&projection, &normalization, ArchitectureEdgeKind::Data);
    let input_axes = axes(
        width
            * c.target_capture_policy
                .as_ref()
                .expect("DSpark capture policy")
                .len(),
    );
    g.read_only_observation(
        &projection,
        format!("{path}.input"),
        "Ordered accepted target captures before context projection",
        input_axes.clone(),
    );
    g.projection_input_observation(&projection, format!("{path}.projection_input"), input_axes);
    g.component_observation(
        &projection,
        format!("{path}.projected"),
        "Context projection before learned normalization",
        axes(width),
    );
    g.component_observation(
        &normalization,
        format!("{path}.normalized"),
        "Normalized accepted context before stream broadcast",
        axes(width),
    );
    g.component_observation(
        root,
        format!("{path}.streams"),
        "Accepted context repeated over streams independently for each cache",
        stream_axes(c),
    );
    for depth in 0..c.num_nextn_predict_layers as usize {
        let node = format!("{root}.layers.{depth}");
        let prefix = format!("{path}.layers.{depth}");
        g.node(
            &node,
            ArchitectureNodeKind::Attention,
            Some(root),
            Some(&format!("mtp.{depth}.attn")),
            None,
        );
        g.edge(&normalization, &node, ArchitectureEdgeKind::Data);
        g.component_observation(
            &node,
            format!("{prefix}.input"),
            "Independent context input used only to update this block's cache",
            stream_axes(c),
        );
        g.component_observation(
            &node,
            format!("{prefix}.collapsed"),
            "Hyper-connection collapse before context attention normalization",
            axes(width),
        );
        g.component_observation(
            &node,
            format!("{prefix}.normalized"),
            "Actual normalized input used to update context key/value cache",
            axes(width),
        );
    }
}
