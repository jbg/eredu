//! Logical ownership checks against the actual nonzero prepared pair.
use super::*;
use eredu_architectures::speculative_execution::speculative_capture_scope;
use eredu_core::{ArchitectureDescriptor, ArchitectureEdgeKind as E, ArchitectureNodeKind as K};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn check(
    joined: &eredu_architectures::qwen4_exp::prepared::PredictionDiscovery,
    prepared: &PreparedTarget,
) {
    let graph = &joined.architecture;
    let captures = &joined.activations;
    assert_eq!(graph.observations, captures.captures.catalog);
    assert!(matches!(
        graph.completeness,
        eredu_core::DescriptionCompleteness::Partial(_)
    ));
    let ids: BTreeSet<_> = graph.nodes.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(ids.len(), graph.nodes.len());
    let groups: BTreeMap<_, _> = graph
        .parameter_groups
        .iter()
        .map(|g| (g.id.as_str(), g))
        .collect();
    assert_eq!(groups.len(), graph.parameter_groups.len());
    for node in &graph.nodes {
        if let Some(parent) = &node.parent {
            assert!(ids.contains(parent.as_str()));
        }
        for group in &node.parameter_groups {
            assert!(groups.contains_key(group.as_str()));
        }
        for path in &node.observation_paths {
            assert_eq!(graph.observations.get(path).unwrap().node_id, node.id);
        }
        // Every ancestry chain terminates, including non-observable state nodes.
        let mut seen = BTreeSet::new();
        let mut next = Some(node);
        while let Some(node) = next {
            assert!(seen.insert(&node.id));
            next = node.parent.as_ref().map(|id| graph.node(id).unwrap());
        }
    }
    for group in &graph.parameter_groups {
        if let Some(shared) = &group.shared_with {
            let owner = groups[shared.as_str()];
            assert_eq!(owner.canonical_prefix, group.canonical_prefix);
            assert!(
                owner.shared_with.is_none(),
                "sharing names the physical owner directly"
            );
        }
    }
    for point in &graph.observations.points {
        let node = graph.node(&point.node_id).unwrap();
        assert_eq!(
            node.observation_paths
                .iter()
                .filter(|p| *p == &point.path)
                .count(),
            1
        );
        let scope = speculative_capture_scope(graph, &point.node_id).unwrap();
        assert_eq!(
            captures
                .bindings
                .iter()
                .find(|b| b.node_id == point.node_id)
                .unwrap()
                .scope,
            scope
        );
    }
    for point in &captures.interventions.points {
        assert_eq!(
            speculative_capture_scope(graph, &point.node_id).unwrap(),
            captures
                .bindings
                .iter()
                .find(|b| b.node_id == point.node_id)
                .unwrap()
                .scope
        );
    }
    for edge in &graph.edges {
        assert!(ids.contains(edge.from.as_str()) && ids.contains(edge.to.as_str()));
    }
    let mut unique_edges = BTreeSet::new();
    for edge in &graph.edges {
        assert!(unique_edges.insert((&edge.from, &edge.to, format!("{:?}", edge.kind))));
    }
    let has = |from: &str, to: &str, kind| {
        graph
            .edges
            .iter()
            .any(|e| e.from == from && e.to == to && e.kind == kind)
    };
    let owner = |path: &str| graph.observations.get(path).unwrap().node_id.as_str();
    assert_eq!(owner("model.layers.1.ple.input"), "target.units.1");
    assert_eq!(owner("model.layers.1.lexical.write"), "target.units.1");
    assert_eq!(owner("model.layers.1.input"), "target.units.2");
    assert!(has("target.units.0", "target.units.1", E::Data));
    assert!(has("target.units.1", "target.units.2", E::Data));
    assert!(has("target.tokens", "target.units.1.table", E::Data));
    let target_group = graph
        .layer_groups
        .iter()
        .find(|g| g.id == "target.decoder")
        .unwrap();
    assert_eq!(target_group.physical_layer_count, 4);
    assert_eq!(
        target_group.passes[0]
            .executions
            .iter()
            .map(|e| (
                e.physical_layer_index,
                graph.node(&e.node_id).unwrap().layer_index.unwrap()
            ))
            .collect::<Vec<_>>(),
        [(0, 0), (1, 2), (2, 3), (3, 4)]
    );
    for (ordinal, unit) in prepared.spec().units.iter().enumerate() {
        let id = format!("target.units.{ordinal}");
        assert_eq!(graph.node(&id).unwrap().layer_index, Some(ordinal));
        match unit {
            eredu_architectures::qwen4_exp::target::UnitSpec::Lexical { .. } => {
                assert_eq!(graph.node(&id).unwrap().kind, K::Mixer)
            }
            eredu_architectures::qwen4_exp::target::UnitSpec::Decoder { mixer, .. } => {
                let node = graph.node(&format!("{id}.mixer")).unwrap();
                match mixer {
                    eredu_architectures::qwen4_exp::target::MixerSpec::Recurrent(spec) => {
                        assert_eq!(node.kind, K::Mixer);
                        assert_eq!(
                            node.mixer.as_ref().unwrap().convolution_width,
                            Some(spec.mixer.convolution.kernel_size as usize)
                        );
                    }
                    eredu_architectures::qwen4_exp::target::MixerSpec::Indexed(spec) => {
                        assert_eq!(node.kind, K::Attention);
                        let attention = node.attention.as_ref().unwrap();
                        assert_eq!(attention.query_heads, Some(spec.heads as usize));
                        assert_eq!(attention.key_value_heads, Some(spec.kv_heads as usize));
                        assert!(
                            attention.receptive_field.is_none(),
                            "selected K/V is not dense full-prefix attention"
                        );
                    }
                }
                let ff = format!("{id}.feed_forward");
                assert!(has(
                    &format!("{ff}.router"),
                    &format!("{ff}.routed"),
                    E::Routing
                ));
                assert_eq!(
                    graph
                        .node(&ff)
                        .unwrap()
                        .moe
                        .as_ref()
                        .unwrap()
                        .routed_experts,
                    3
                );
            }
        }
    }
    let canonical = |node: &str| {
        graph
            .node(node)
            .unwrap()
            .parameter_groups
            .iter()
            .map(|id| groups[id.as_str()].canonical_prefix.as_str())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        canonical("target.units.1.table"),
        ["model.layers.1.ple.ple_embedding.ngram_embedding"]
    );
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|n| n.label == "Logical sharded n-gram row lookup")
            .count(),
        1
    );
    assert_eq!(graph.speculative_invocations.len(), 3);
    for depth in 0..2 {
        let root = format!("prediction.{depth}");
        let scope = SpeculativeCaptureScope::Prediction { depth };
        assert_eq!(speculative_capture_scope(graph, &root).unwrap(), scope);
        for suffix in ["embedding", "head"] {
            let target = format!("target.{suffix}");
            let node = format!("{root}.{suffix}");
            assert_eq!(canonical(&node), canonical(&target));
            let group = groups[graph.node(&node).unwrap().parameter_groups[0].as_str()];
            assert_eq!(
                group.shared_with.as_ref().unwrap(),
                &graph.node(&target).unwrap().parameter_groups[0]
            );
        }
        assert_eq!(
            owner(&format!("mtp.layers.{depth}.mlp.routing.selected_experts")),
            format!("{root}.feed_forward.router")
        );
        assert_eq!(
            owner(&format!("mtp.layers.{depth}.mlp.units")),
            format!("{root}.feed_forward.routed")
        );
        assert!(has(
            &format!("{root}.mixer.state"),
            &format!("{root}.mixer"),
            E::State
        ));
        let state = graph.node(&format!("{root}.mixer.state")).unwrap();
        assert!(state.parameter_groups.is_empty());
    }
    assert_eq!(
        canonical("prediction.0.readout"),
        canonical("prediction.1.readout")
    );
    assert_ne!(
        canonical("target.readout"),
        canonical("prediction.0.readout")
    );
    assert!(has("target.units.4", "prediction.0.fusion", E::Data));
    assert!(has(
        "prediction.0.feed_forward.residual",
        "prediction.1.fusion",
        E::Data
    ));
    assert!(
        !has("target.readout", "prediction.0.fusion", E::Data),
        "prediction receives uncollapsed residuals"
    );
    // Wire documents preserve typed identities and explicit sharing/scopes.
    let serialized = serde_json::to_vec(graph).unwrap();
    let decoded: ArchitectureDescriptor = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(&decoded, graph);
    let mut invalid = graph.clone();
    invalid
        .speculative_invocations
        .retain(|b| b.node_id != "prediction.1");
    assert!(speculative_capture_scope(&invalid, "prediction.1.feed_forward.router").is_err());
    let mut invalid = graph.clone();
    invalid
        .speculative_invocations
        .push(invalid.speculative_invocations[1].clone());
    assert!(speculative_capture_scope(&invalid, "prediction.0.logits").is_err());
    let mut invalid = graph.clone();
    invalid
        .nodes
        .iter_mut()
        .find(|n| n.id == "prediction.0.feed_forward")
        .unwrap()
        .parent = Some("prediction.0.feed_forward.router".into());
    assert!(speculative_capture_scope(&invalid, "prediction.0.feed_forward.router").is_err());

    // The capture passed to MTP retains streams, while final collapse removes them.
    let shape = |node: &str| graph.node(node).unwrap().output_axes.as_ref().unwrap();
    assert_eq!(shape("target.units.4"), shape("prediction.0.fusion"));
    assert_eq!(shape("prediction.0.feed_forward.residual").len(), 4);
    assert_eq!(shape("target.readout").len(), 3);
    assert_eq!(shape("prediction.0.readout"), shape("target.readout"));
    assert_eq!(shape("prediction.1.logits"), shape("target.logits"));
    assert_eq!(
        shape("target.units.4")[2].dimension,
        eredu_core::SymbolicDimension::Known(prepared.spec().boundary.geometry.streams() as usize)
    );
}
