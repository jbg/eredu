//! Device policy shared by the standalone examples.

use anyhow::{Context, Result};
use eredu_core::{BackendHardware, DevicePlan, HardwareDeviceKind};

pub fn default_device(factory: &impl BackendHardware) -> Result<DevicePlan> {
    select_device(factory, &[HardwareDeviceKind::Gpu, HardwareDeviceKind::Cpu])
}

fn select_device(
    factory: &impl BackendHardware,
    preferred: &[HardwareDeviceKind],
) -> Result<DevicePlan> {
    let backend = factory.backend_id();
    let hardware = factory.discover_hardware()?;
    let profile = hardware
        .backends
        .iter()
        .find(|profile| profile.backend == backend && profile.available)
        .with_context(|| format!("backend {backend} is unavailable"))?;
    let device = preferred
        .iter()
        .find_map(|kind| {
            profile
                .devices
                .iter()
                .filter(|device| &device.kind == kind)
                .min_by_key(|device| (device.index, &device.id))
        })
        .with_context(|| format!("backend {backend} reported no matching device"))?;
    Ok(DevicePlan::new(backend.as_str(), &device.id)?)
}
