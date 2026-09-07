//! Complete selected-execution discovery and capture using explicit native resources.
use eredu_backend_mlx::{native, MlxLoadRequest};
use eredu_core::{ModelRuntime, ObservationRequest, ObservationSelector, ObservationSupportStatus};
use safemlx::{Array, Device, DeviceType};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: capture_discovery <artifact> [token-id ...]"))?;
    let tokens = std::env::args()
        .skip(2)
        .map(|v| v.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(
        !tokens.is_empty(),
        "provide one or more valid token IDs for this artifact"
    );
    let options = MlxLoadRequest::default();
    // This step only reads headers/configuration. It creates no device or tensors.
    let report = native::inspect_model(&path, native::MlxInspectionOptions::new(options.clone()))?;
    let architecture = report.architecture_descriptor.as_ref().ok_or_else(|| {
        anyhow::anyhow!("artifact has no admitted architecture: {:?}", report.issues)
    })?;
    for node in &architecture.nodes {
        println!("{}: {:?}", node.id, node.kind);
    }
    for edge in &architecture.edges {
        println!("{} -> {} ({:?})", edge.from, edge.to, edge.kind);
    }
    let support = report
        .observation_support
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no selected execution support report"))?;
    for point in &support.points {
        println!(
            "{}: prefill {:?}; decode {:?}",
            point.path, point.prefill, point.decode
        );
    }
    let point = support
        .points
        .iter()
        .find(|p| {
            p.prefill == ObservationSupportStatus::Supported
                && p.decode == ObservationSupportStatus::Supported
        })
        .ok_or_else(|| anyhow::anyhow!("no observation supported in both phases"))?;
    let request = ObservationRequest::selected([ObservationSelector::Exact(point.path.clone())]);

    let execution = native::ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let backend = native::backend(stream, stream);
    let prepared = eredu_core::load_model(&backend, &path, options)?;
    let mut runtime = ModelRuntime::from_prepared(backend, prepared)?;
    let ids = Array::from_slice(&tokens, &[1, i32::try_from(tokens.len())?]);
    let part = eredu_backend_mlx::backend::runtime::media::input::token_ids_part(&ids)?;
    let parts = [part];
    let input = eredu_backend_mlx::backend::runtime::media::input::ModelInput::new(&parts);
    let prefill = runtime.inspect_prefill(input.into(), &request)?;
    println!(
        "Captured {}: {:?}",
        point.path,
        prefill.observations.get(&point.path)
    );
    let decode = runtime.inspect_decode(Array::from_slice(&tokens[..1], &[1, 1]), &request)?;
    println!(
        "Decode {}: {:?}",
        point.path,
        decode.observations.get(&point.path)
    );
    Ok(())
}
