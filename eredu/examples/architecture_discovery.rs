//! Cold discovery plus a portable capture helper for an application's existing runtime.
use eredu_core::{
    BackendProvider, BackendSession, InspectableBackendSession, ModelInspectionReport,
    ModelRuntime, ObservationRequest, ObservationSelector, ObservationSet,
    ObservationSupportStatus,
};

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

/// Captures one advertised path from an already loaded runtime matching `report`.
/// The report comes from selected-backend inspection with the same load options.
pub fn capture_advertised<B>(
    runtime: &mut ModelRuntime<B>,
    report: &ModelInspectionReport,
    input: <B::Session as BackendSession<B>>::PrefillInput,
) -> Result<Option<ObservationSet>, B::Error>
where
    B: BackendProvider,
    B::Session: InspectableBackendSession<B>,
{
    let Some(support) = &report.observation_support else {
        return Ok(None);
    };
    for point in &support.points {
        println!(
            "{}: prefill {:?}, decode {:?}",
            point.path, point.prefill, point.decode
        );
    }
    let Some(point) = support
        .points
        .iter()
        .find(|p| p.prefill == ObservationSupportStatus::Supported)
    else {
        return Ok(None);
    };
    let request = ObservationRequest::selected([ObservationSelector::Exact(point.path.clone())]);
    runtime
        .inspect_prefill(input, &request)
        .map(|output| Some(output.observations))
}
