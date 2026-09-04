//! Inspect neutral Cartesian rank topology construction for MLX execution.

use eredu_backend_mlx::native::DeviceAssignment;
use eredu_core::{ParallelAxis, ParallelRankTopology, ParallelTopology};
use safemlx::{
    distributed::{self, Backend},
    DeviceType,
};

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
    let topology = ParallelRankTopology::new(ParallelTopology::new(tp, pp, ep, 1)?, world.rank())?;
    let device = DeviceAssignment::new(DeviceType::Gpu, local_index);
    let coordinates = topology.coordinates();
    let local_layers =
        layers * coordinates.pipeline() / pp..layers * (coordinates.pipeline() + 1) / pp;
    let local_experts =
        experts * coordinates.expert() / ep..experts * (coordinates.expert() + 1) / ep;
    eprintln!(
        "global={}/{} coordinates={:?} layers={:?} experts={:?} embedding={} head={} TP={:?} PP={:?} EP={:?}",
        topology.global_rank(),
        topology.world_size(),
        topology.coordinates(),
        local_layers,
        local_experts,
        coordinates.pipeline() == 0,
        coordinates.pipeline() + 1 == pp,
        topology.subgroup(ParallelAxis::Tensor)?.global_ranks(),
        topology.subgroup(ParallelAxis::Pipeline)?.global_ranks(),
        topology.subgroup(ParallelAxis::Expert)?.global_ranks(),
    );
    let _ = device;
    Ok(())
}
