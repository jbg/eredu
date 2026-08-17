//! Inspect a pairwise Cartesian distributed topology before loading weights.

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{DeviceAssignment, MlxBackend, ParallelTopology};

fn parse(index: usize, name: &str) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(std::env::args()
        .nth(index)
        .ok_or_else(|| format!("missing {name}"))?
        .parse()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tp = parse(1, "TP")?;
    let pp = parse(2, "PP")?;
    let ep = parse(3, "EP")?;
    let layers = parse(4, "DECODER_LAYERS")?;
    let experts = parse(5, "ROUTED_EXPERTS")?;
    let world = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = ParallelTopology::from_group(
        &world,
        tp,
        pp,
        ep,
        DeviceAssignment::new(DeviceType::Gpu, local_index),
    )?;
    let stream = Stream::new_with_device(&topology.device.device()?);
    let report = topology.preflight((pp > 1).then_some(layers), (ep > 1).then_some(experts))?;
    let _execution = MlxBackend::new(&stream).create_communication_session(topology, &world)?;
    eprintln!(
        "global={}/{} coordinates={:?} layers={:?} experts={:?} embedding={} head={} TP={:?} PP={:?} EP={:?}",
        topology.global_rank,
        topology.world_size,
        topology.coordinates(),
        report.local_layer_range,
        report.local_expert_range,
        report.owns_embedding,
        report.owns_output_head,
        report.tensor_subgroup.global_ranks,
        report.pipeline_subgroup.global_ranks,
        report.expert_subgroup.global_ranks,
    );
    Ok(())
}
