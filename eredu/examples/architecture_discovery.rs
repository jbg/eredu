//! Cold architecture discovery and named observation catalog.
fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: architecture_discovery <artifact>"))?;
    let architecture = eredu::api::inspect_architecture(path)?;
    for group in &architecture.layer_groups {
        let sharing = match group.weight_sharing {
            eredu_core::LayerWeightSharing::SharedAcrossPasses => " · shared weights",
            eredu_core::LayerWeightSharing::None => "",
        };
        println!(
            "{}: {} layers × {} passes{}",
            group.label,
            group.physical_layer_count,
            group.passes.len(),
            sharing
        );
        for pass in &group.passes {
            for execution in &pass.executions {
                let node = architecture
                    .node(&execution.node_id)
                    .expect("declared execution");
                println!(
                    "  Pass {}, layer {} ({}): captures {:?}",
                    pass.index + 1,
                    execution.physical_layer_index + 1,
                    node.id,
                    node.observation_paths
                );
            }
        }
    }
    for node in &architecture.nodes {
        println!(
            "{}: {:?}; captures {:?}",
            node.id, node.kind, node.observation_paths
        );
    }
    for edge in &architecture.edges {
        println!("{} -> {} ({:?})", edge.from, edge.to, edge.kind);
    }
    println!("Coverage: {:?}", architecture.completeness);
    println!(
        "Catalog: {}",
        serde_json::to_string_pretty(&architecture.observations)?
    );
    Ok(())
}
