//! Host memory probes shared by backend adapters and portable residency reporting.

use crate::{AvailableMemory, ObservationKind, Observed, PhysicalMemorySemantics};

/// Measures installed RAM and advisory allocation headroom with a safe OS probe.
pub fn observe(semantics: PhysicalMemorySemantics) -> AvailableMemory {
    let mut system = sysinfo::System::new();
    system.refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());
    let total = system.total_memory();
    #[cfg(target_os = "macos")]
    let (available, source) = macos_available_memory().map_or_else(
        || (system.free_memory(), "point-in-time macOS free pages"),
        |bytes| (bytes, "point-in-time Mach free plus inactive pages"),
    );
    #[cfg(not(target_os = "macos"))]
    let (available, source) = (
        system.available_memory(),
        "point-in-time host available memory",
    );
    AvailableMemory {
        physical_memory_bytes: if total > 0 {
            Observed::exact(total, "host installed physical memory")
        } else {
            Observed::unavailable("host physical-memory capacity is unavailable")
        },
        available_memory_bytes: if total > 0 && available <= total {
            Observed::Available {
                value: available,
                kind: ObservationKind::Estimated,
                source: source.into(),
            }
        } else {
            Observed::unavailable("host available memory is unavailable")
        },
        device: None,
        physical_semantics: semantics,
    }
}

/// Observes current process RSS. This includes allocations outside any backend.
pub fn process_rss_bytes() -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    let mut system = sysinfo::System::new();
    system.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        true,
        sysinfo::ProcessRefreshKind::nothing().with_memory(),
    );
    Some(system.process(pid)?.memory()).filter(|bytes| *bytes > 0)
}

/// Apple ARM hosts share CPU and Metal physical capacity.
pub const fn apple_metal_semantics() -> PhysicalMemorySemantics {
    if cfg!(all(target_arch = "aarch64", target_vendor = "apple")) {
        PhysicalMemorySemantics::Unified
    } else {
        PhysicalMemorySemantics::Unknown
    }
}

#[cfg(target_os = "macos")]
fn macos_available_memory() -> Option<u64> {
    let output = std::process::Command::new("/usr/bin/vm_stat")
        .env("LC_ALL", "C")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = std::str::from_utf8(&output.stdout).ok()?;
    let mut lines = text.lines();
    let page_size = lines
        .next()?
        .split_once("(page size of ")?
        .1
        .strip_suffix(" bytes)")?
        .parse::<u64>()
        .ok()?;
    let pages = |name: &str| {
        text.lines().find_map(|line| {
            line.strip_prefix(name)?
                .trim()
                .strip_suffix('.')?
                .parse::<u64>()
                .ok()
        })
    };
    // Keep speculative pages out of the sum as a separate category.
    pages("Pages free:")?
        .checked_add(pages("Pages inactive:")?)?
        .checked_mul(page_size)
}
