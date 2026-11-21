//! MLX model capability derivation, resource observation, and admission adapter.

use eredu_core::{
    AvailableMemory, CapabilityError, ObservationKind, Observed, PhysicalMemorySemantics,
};

fn physical_memory_semantics(os: &str, arch: &str) -> PhysicalMemorySemantics {
    // Supported Apple silicon devices and ARM64 simulators share physical RAM
    // between CPU and Metal. Other targets need device facts before we can
    // claim a relationship (CUDA managed memory alone does not establish one).
    if arch == "aarch64" && matches!(os, "macos" | "ios" | "tvos" | "visionos") {
        PhysicalMemorySemantics::Unified
    } else {
        PhysicalMemorySemantics::Unknown
    }
}

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

pub(super) fn native_memory_facts() -> eredu_core::capability::NativeMemoryFacts {
    eredu_core::capability::NativeMemoryFacts {
        active_allocation_bytes: runtime_counter(
            safemlx::memory::active_memory,
            "process-global MLX active allocation counter",
        ),
        allocator_cache_bytes: runtime_counter(
            safemlx::memory::cache_memory,
            "process-global MLX allocator cache counter",
        ),
        physical_semantics: physical_memory_semantics(std::env::consts::OS, std::env::consts::ARCH),
    }
}

/// Queries system memory that can be used as an admission signal.
///
/// Apple Silicon reports one unified physical capacity; logical host/device
/// residency tiers must not be added as independent physical capacities.
pub fn available_memory() -> Result<AvailableMemory, CapabilityError> {
    let memory = safemlx::system::system_memory()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let physical_memory_bytes = memory.total.map_or_else(
        || Observed::Unavailable {
            reason: "physical-memory capacity is unavailable on this platform".into(),
        },
        |value| Observed::Available {
            value,
            kind: ObservationKind::Exact,
            source: "host physical-memory observation".into(),
        },
    );
    let available_memory_bytes = memory.available.map_or_else(
        || Observed::Unavailable {
            reason: "available physical memory is unavailable on this platform".into(),
        },
        |value| Observed::Available {
            value,
            kind: ObservationKind::Estimated,
            source: if cfg!(target_os = "macos") {
                "point-in-time Mach host VM free (including speculative) plus inactive pages"
            } else if cfg!(any(
                target_os = "ios",
                target_os = "tvos",
                target_os = "visionos"
            )) {
                "point-in-time app allocation headroom from os_proc_available_memory"
            } else {
                "point-in-time host available-memory observation"
            }
            .into(),
        },
    );
    Ok(AvailableMemory {
        physical_memory_bytes,
        available_memory_bytes,
        physical_semantics: physical_memory_semantics(std::env::consts::OS, std::env::consts::ARCH),
    })
}

/// Native allocator observations for MLX placements.
pub struct MlxAllocatorObservations;

impl eredu_core::capability::NativeAllocatorObservations for MlxAllocatorObservations {
    fn native_memory_facts() -> eredu_core::capability::NativeMemoryFacts {
        native_memory_facts()
    }

    fn available_memory() -> Result<eredu_core::AvailableMemory, eredu_core::BackendFailure> {
        available_memory().map_err(eredu_core::BackendFailure::from_error)
    }

    fn allocator_cache_limit() -> eredu_core::Observed<u64> {
        match crate::allocator_cache_policy() {
            Ok(policy) => Observed::exact(
                policy.limit_bytes,
                format!(
                    "MLX allocator-cache policy {:?}: {} bytes",
                    policy.source, policy.limit_bytes
                ),
            ),
            Err(error) => eredu_core::Observed::unavailable(error.to_string()),
        }
    }
}
