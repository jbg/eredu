//! Safe process and operating-system resource observations used by runtimes.

/// Process resource observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessUsage {
    /// Peak resident-set size in bytes.
    pub peak_rss: u64,
    /// Minor page faults observed for the process.
    pub minor_page_faults: u64,
    /// Major page faults observed for the process.
    pub major_page_faults: u64,
}

/// Reads the current app's allocation headroom on Apple mobile targets.
#[cfg(any(target_os = "ios", target_os = "tvos", target_os = "visionos"))]
pub fn app_memory_headroom() -> u64 {
    unsafe extern "C" {
        fn os_proc_available_memory() -> libc::size_t;
    }
    // SAFETY: This argument-free libSystem query transfers no resources.
    unsafe { os_proc_available_memory() as u64 }
}

/// Observes resource usage for the current process.
#[cfg(unix)]
pub fn process_usage() -> Option<ProcessUsage> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let peak_rss = u64::try_from(usage.ru_maxrss).ok()?;
    Some(ProcessUsage {
        peak_rss: if cfg!(target_os = "macos") {
            peak_rss
        } else {
            peak_rss.saturating_mul(1024)
        },
        minor_page_faults: u64::try_from(usage.ru_minflt).ok()?,
        major_page_faults: u64::try_from(usage.ru_majflt).ok()?,
    })
}

/// Observes resource usage for the current process.
#[cfg(not(unix))]
pub fn process_usage() -> Option<ProcessUsage> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_usage_reports_a_nonzero_resident_set_when_supported() {
        if let Some(usage) = process_usage() {
            assert!(usage.peak_rss > 0);
        }
    }
}
