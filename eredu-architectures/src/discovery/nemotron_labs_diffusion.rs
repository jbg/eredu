//! Exact shared text, draft-only low-rank, and independently owned Pixtral semantics.
use super::*;
use crate::{decoder::Config, nemotron_labs_diffusion::FamilyConfig};

pub(super) fn describe(g: &mut Builder, config: &FamilyConfig) {
    families::dense(g, config, None);
    for layer in 0..config.num_hidden_layers() as usize {
        let node = format!("decoder.layers.{layer}.attention");
        g.get_mut(&node).label =
            "Shared causal or prefix-and-block attention; request-local visibility".into();
        if let Some(attention) = &mut g.get_mut(&node).attention {
            attention.causal = None;
        }
        if let Some(adapter) = config.draft_adapter() {
            let root = format!("draft_adapter.layers.{layer}.o_proj");
            let down = format!("{root}.down");
            let up = format!("{root}.up");
            let sum = format!("{root}.sum");
            let parent = format!("decoder.layers.{layer}");
            g.node(
                &down,
                ArchitectureNodeKind::Projector,
                Some(&parent),
                Some(&format!("{root}.lora_A")),
                Some(adapter.rank() as usize),
            );
            g.get_mut(&down).label = "Draft-only low-rank reduction of attention channels".into();
            g.node(
                &up,
                ArchitectureNodeKind::Projector,
                Some(&parent),
                Some(&format!("{root}.lora_B")),
                Some(config.hidden_size() as usize),
            );
            g.get_mut(&up).label =
                format!("Draft-only expansion, alpha/rank scale {}", adapter.scale());
            g.node(
                &sum,
                ArchitectureNodeKind::Sum,
                Some(&parent),
                None,
                Some(config.hidden_size() as usize),
            );
            g.get_mut(&sum).label = "Shared attention output plus draft-only low-rank write".into();
            g.edge(&node, &down, ArchitectureEdgeKind::Data);
            g.edge(&down, &up, ArchitectureEdgeKind::Data);
            g.edge(&node, &sum, ArchitectureEdgeKind::Data);
            g.edge(&up, &sum, ArchitectureEdgeKind::Data);
            let residual = format!("decoder.layers.{layer}.attention.residual");
            for edge in &mut g.descriptor.edges {
                if edge.from == node && edge.to == residual {
                    edge.from = sum.clone();
                }
            }
        }
    }
    let Some(vision) = &config.vision else {
        return;
    };
    let width = vision.hidden_size as usize;
    g.node(
        "vision",
        ArchitectureNodeKind::Encoder,
        None,
        None,
        Some(width),
    );
    g.node(
        "vision.processor",
        ArchitectureNodeKind::Processor,
        Some("vision"),
        None,
        None,
    );
    g.get_mut("vision.processor").label =
        "RGB conversion, bounded cubic resize, published normalization and channel-major patches"
            .into();
    g.node(
        "vision.embedding",
        ArchitectureNodeKind::Embedding,
        Some("vision"),
        Some("model.vision_tower.patch_conv"),
        Some(width),
    );
    g.edge(
        "vision.processor",
        "vision.embedding",
        ArchitectureEdgeKind::Data,
    );
    g.node(
        "vision.pre_norm",
        ArchitectureNodeKind::Normalization,
        Some("vision"),
        Some("model.vision_tower.ln_pre"),
        Some(width),
    );
    g.edge(
        "vision.embedding",
        "vision.pre_norm",
        ArchitectureEdgeKind::Data,
    );
    let mut previous = "vision.pre_norm".to_owned();
    let mut executions = Vec::new();
    for layer in 0..vision.num_hidden_layers as usize {
        let block = format!("vision.layers.{layer}");
        let root = format!("model.vision_tower.transformer.layers.{layer}");
        g.node(
            &block,
            ArchitectureNodeKind::Encoder,
            Some("vision"),
            Some(&root),
            Some(width),
        );
        g.get_mut(&block).layer_index = Some(layer);
        g.edge(&previous, &block, ArchitectureEdgeKind::Data);
        let (attention, residual) = g.sublayer(
            &block,
            &block,
            "attention",
            &format!("{root}.attention_norm"),
            &format!("{root}.attention"),
            width,
            ArchitectureNodeKind::Attention,
        );
        g.get_mut(&attention).attention = Some(AttentionAttributes {
            head_sharing: Some(HeadSharing::MultiHead),
            query_heads: Some(vision.num_attention_heads as usize),
            key_value_heads: Some(vision.num_attention_heads as usize),
            key_head_dimension: Some(vision.head_dim as usize),
            value_head_dimension: Some(vision.head_dim as usize),
            receptive_field: Some(ReceptiveField::Local { window: None }),
            mechanism: Some(AttentionMechanism::Softmax),
            recurrent: Some(false),
            causal: Some(false),
            positional_encoding: Some(PositionalEncoding::Rotary),
            ..Default::default()
        });
        g.get_mut(&attention).label = "Bidirectional attention within each image; independent two-dimensional rotary positions".into();
        let (ffn, output) = g.sublayer(
            &block,
            &residual,
            "feed_forward",
            &format!("{root}.ffn_norm"),
            &format!("{root}.feed_forward"),
            width,
            ArchitectureNodeKind::FeedForward,
        );
        g.get_mut(&ffn).label = "SwiGLU image feed-forward".into();
        previous = output;
        executions.push(ArchitectureLayerExecution {
            node_id: block,
            physical_layer_index: layer,
        });
    }
    g.descriptor.layer_groups.push(ArchitectureLayerGroup {
        id: "vision".into(),
        label: "Pixtral".into(),
        physical_layer_count: vision.num_hidden_layers as usize,
        passes: vec![ArchitectureExecutionPass {
            index: 0,
            executions,
        }],
        weight_sharing: LayerWeightSharing::None,
    });
    g.node(
        "vision.projector",
        ArchitectureNodeKind::Projector,
        None,
        None,
        Some(config.hidden_size() as usize),
    );
    for (node, kind, parameter, label) in [
        (
            "vision.projector.norm",
            ArchitectureNodeKind::Normalization,
            "model.multi_modal_projector.norm",
            "Pre-merger RMS normalization",
        ),
        (
            "vision.projector.merge",
            ArchitectureNodeKind::Projector,
            "model.multi_modal_projector.patch_merger.merging_layer",
            "Channel-major spatial unfold and learned merger",
        ),
        (
            "vision.projector.linear_1",
            ArchitectureNodeKind::Projector,
            "model.multi_modal_projector.linear_1",
            "Image-to-text projection followed by GELU",
        ),
        (
            "vision.projector.linear_2",
            ArchitectureNodeKind::Projector,
            "model.multi_modal_projector.linear_2",
            "Final image-to-text projection",
        ),
    ] {
        g.node(node, kind, Some("vision.projector"), Some(parameter), None);
        g.get_mut(node).label = label.into();
        g.edge(&previous, node, ArchitectureEdgeKind::Data);
        previous = node.into();
    }
    g.edge(&previous, "vision.projector", ArchitectureEdgeKind::Data);
    g.node(
        "assembly",
        ArchitectureNodeKind::ModalityMerge,
        None,
        None,
        Some(config.hidden_size() as usize),
    );
    g.get_mut("assembly").label = "Projected image features replace image-pad positions; row and end markers remain token embeddings".into();
    for edge in &mut g.descriptor.edges {
        if edge.from == "embedding" {
            edge.from = "assembly".into();
        }
    }
    g.edge("embedding", "assembly", ArchitectureEdgeKind::Data);
    g.edge("vision.projector", "assembly", ArchitectureEdgeKind::Data);
    for (node, path, hidden, projected) in [
        (
            "vision.projector.norm",
            "model.multi_modal_projector.norm.output",
            width,
            false,
        ),
        (
            "vision.projector.merge",
            "model.multi_modal_projector.patch_merger.output",
            width,
            true,
        ),
        (
            "vision.projector.linear_1",
            "model.multi_modal_projector.linear_1.output",
            config.hidden_size() as usize,
            true,
        ),
        (
            "vision.projector.linear_1",
            "model.multi_modal_projector.gelu.output",
            config.hidden_size() as usize,
            true,
        ),
        (
            "vision.projector.linear_2",
            "model.multi_modal_projector.linear_2.output",
            config.hidden_size() as usize,
            true,
        ),
    ] {
        media_hook(g, node, path.into(), hidden, projected);
    }
    for index in 0..vision.num_hidden_layers as usize {
        let node = format!("vision.layers.{index}");
        let path = format!("model.vision_tower.transformer.layers.{index}");
        for (suffix, boundary) in [
            ("attention.norm", "attention_norm.output"),
            ("attention", "attention.output"),
            ("attention.residual", "attention.residual"),
            ("feed_forward.norm", "ffn_norm.output"),
            ("feed_forward", "feed_forward.output"),
            ("feed_forward.residual", "feed_forward.residual"),
        ] {
            media_hook(
                g,
                &format!("{node}.{suffix}"),
                format!("{path}.{boundary}"),
                width,
                false,
            );
        }
    }
}

fn media_hook(g: &mut Builder, node: &str, path: String, width: usize, projected: bool) {
    let mut shape = axes(width);
    shape[1].name = "media".into();
    shape[1].dimension = if projected {
        SymbolicDimension::ProjectedMediaPositions
    } else {
        SymbolicDimension::MediaPositions
    };
    let first = g.descriptor.observations.points.len();
    g.component_observation(node, path, "Packed image activation", shape);
    for point in &mut g.descriptor.observations.points[first..] {
        point.decode = false;
        point.requirements.push(ObservationRequirement::MediaInput);
    }
}
