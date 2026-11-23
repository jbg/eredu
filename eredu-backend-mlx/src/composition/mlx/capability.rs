//! MLX model capability derivation, resource observation, and admission adapter.

use eredu_core::{AvailableMemory, CapabilityError, ObservationKind, Observed};

fn unavailable_counter(error: safemlx::error::Exception) -> Observed<u64> {
    Observed::Unavailable {
        reason: error.to_string(),
    }
}

fn runtime_counter(
    function: fn() -> Result<usize, safemlx::error::Exception>,
    source: &'static str,
) -> Observed<u64> {
    match function() {
        Ok(value) => match u64::try_from(value) {
            Ok(value) => Observed::Available {
                value,
                kind: ObservationKind::Observational,
                source: source.into(),
            },
            Err(_) => Observed::Unavailable {
                reason: "counter does not fit u64".into(),
            },
        },
        Err(error) => unavailable_counter(error),
    }
}

fn native_memory_facts() -> eredu_core::capability::NativeMemoryFacts {
    eredu_core::capability::NativeMemoryFacts {
        active_allocation_bytes: runtime_counter(
            safemlx::memory::active_memory,
            "process-global MLX active allocation counter",
        ),
        allocator_cache_bytes: runtime_counter(
            safemlx::memory::cache_memory,
            "process-global MLX allocator cache counter",
        ),
        device_allocated_bytes: Observed::unsupported(
            "MLX reports its allocator counters separately",
        ),
        physical_semantics: eredu_core::host_memory::apple_metal_semantics(),
    }
}

/// Queries system memory that can be used as an admission signal.
///
/// Apple Silicon reports one unified physical capacity; logical host/device
/// residency tiers must not be added as independent physical capacities.
pub fn available_memory() -> Result<AvailableMemory, CapabilityError> {
    let memory = eredu_core::host_memory::observe(eredu_core::host_memory::apple_metal_semantics());
    #[cfg(any(target_os = "ios", target_os = "tvos", target_os = "visionos"))]
    let memory = AvailableMemory {
        available_memory_bytes: Observed::Available {
            value: safemlx::system::app_memory_headroom(),
            kind: ObservationKind::Estimated,
            source: "point-in-time app allocation headroom from os_proc_available_memory".into(),
        },
        ..memory
    };
    Ok(memory)
}

/// Observations on the device retained by a model execution stream.
pub(super) fn placement_memory(
    queue: &safemlx::Stream,
) -> Result<eredu_core::capability::ParameterMemoryFacts, eredu_core::BackendFailure> {
    use eredu_core::{
        capability::{ExecutionMemoryPlacement, ParameterMemoryFacts},
        BackendFailure,
    };
    let device = queue.get_device().map_err(BackendFailure::from_error)?;
    let host_execution =
        device.get_type().map_err(BackendFailure::from_error)? == safemlx::DeviceType::Cpu;
    let index = device.get_index().map_err(BackendFailure::from_error)?;
    let accelerator = if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "metal"
    };
    let family = if host_execution { "cpu" } else { accelerator };
    Ok(ParameterMemoryFacts {
        placement: ExecutionMemoryPlacement {
            device: eredu_core::DevicePlan::new("mlx", format!("{family}:{index}"))
                .map_err(BackendFailure::from_error)?,
            host_execution,
            allocator: Some("mlx:process".into()),
        },
        native: native_memory_facts(),
        available: available_memory().map_err(BackendFailure::from_error)?,
        allocator_cache_limit: match crate::allocator_cache_policy() {
            Ok(policy) => Observed::exact(
                policy.limit_bytes,
                format!(
                    "MLX allocator-cache policy {:?}: {} bytes",
                    policy.source, policy.limit_bytes
                ),
            ),
            Err(error) => Observed::unavailable(error.to_string()),
        },
    })
}
