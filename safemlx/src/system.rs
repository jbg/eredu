//! Safe process and operating-system resource observations used by runtimes.

/// Host physical-memory observations in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMemory {
    /// Installed physical memory, when available.
    pub total: Option<u64>,
    /// Point-in-time estimate of available host physical memory, when available.
    /// This is advisory capacity, not a reservation or a process allocation limit.
    pub available: Option<u64>,
}

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

/// Observes host physical memory.
#[cfg(target_os = "macos")]
pub fn system_memory() -> std::io::Result<SystemMemory> {
    let name = c"hw.memsize";
    let mut total = 0u64;
    let mut size = std::mem::size_of::<u64>();
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut total as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    let total = (status == 0 && size == std::mem::size_of::<u64>()).then_some(total);
    let available = macos_available_memory();
    Ok(SystemMemory { total, available })
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc's Mach port accessors avoid an additional FFI dependency.
fn macos_available_memory() -> Option<u64> {
    unsafe extern "C" {
        fn host_page_size(host: libc::host_t, size: *mut libc::vm_size_t) -> libc::kern_return_t;
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }

    let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let mut page_size = 0;
    // SAFETY: Both output buffers are valid for the declared sizes. The Mach
    // host send right is released after both calls, including on query failure.
    let (page_status, stats_status) = unsafe {
        let host = libc::mach_host_self();
        let page_status = host_page_size(host, &mut page_size);
        let stats_status = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            stats.as_mut_ptr().cast(),
            &mut count,
        );
        mach_port_deallocate(libc::mach_task_self(), host);
        (page_status, stats_status)
    };
    // Older kernels return a shorter revision than libc's current struct. Only
    // the initial fields through inactive_count are needed, and the rest of the
    // buffer was zero-initialized before the call.
    let required_bytes = std::mem::offset_of!(libc::vm_statistics64, inactive_count)
        + std::mem::size_of::<libc::natural_t>();
    if page_status != libc::KERN_SUCCESS
        || stats_status != libc::KERN_SUCCESS
        || (count as usize) < required_bytes / std::mem::size_of::<libc::integer_t>()
    {
        return None;
    }
    // SAFETY: The integer-only struct was zeroed and the query filled the fields
    // read below, as checked by the returned count.
    let stats = unsafe { stats.assume_init() };
    macos_available_bytes(&stats, u64::try_from(page_size).ok()?)
}

#[cfg(target_os = "macos")]
fn macos_available_bytes(stats: &libc::vm_statistics64, page_size: u64) -> Option<u64> {
    if page_size == 0 {
        return None;
    }
    // Mach free_count already includes speculative_count. Inactive pages are
    // potentially reclaimable; this is an estimate, not guaranteed free space.
    (u64::from(stats.free_count) + u64::from(stats.inactive_count)).checked_mul(page_size)
}

/// Observes host physical memory.
#[cfg(target_os = "linux")]
pub fn system_memory() -> std::io::Result<SystemMemory> {
    let contents = std::fs::read_to_string("/proc/meminfo")?;
    let value = |name: &str| -> Option<u64> {
        contents.lines().find_map(|line| {
            let (key, rest) = line.split_once(':')?;
            if key != name {
                return None;
            }
            rest.split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()?
                .checked_mul(1024)
        })
    };
    Ok(SystemMemory {
        total: value("MemTotal"),
        available: value("MemAvailable"),
    })
}

/// Observes host physical memory.
#[cfg(target_os = "windows")]
pub fn system_memory() -> std::io::Result<SystemMemory> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(SystemMemory {
        total: Some(status.ullTotalPhys),
        available: Some(status.ullAvailPhys),
    })
}

/// Observes host physical memory.
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub fn system_memory() -> std::io::Result<SystemMemory> {
    Ok(SystemMemory {
        total: None,
        available: None,
    })
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

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_available_memory_counts_reclaimable_pages_once() {
        // SAFETY: The Mach statistics struct contains only integer fields.
        let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        stats.free_count = 7;
        stats.speculative_count = 3; // Included in the seven free pages.
        stats.inactive_count = 5;
        stats.active_count = 11;
        stats.wire_count = 13;
        stats.compressor_page_count = 17;
        for page_size in [4096, 16384] {
            assert_eq!(
                macos_available_bytes(&stats, page_size),
                Some(12 * page_size)
            );
        }
        assert_eq!(macos_available_bytes(&stats, 0), None);
        assert_eq!(macos_available_bytes(&stats, u64::MAX), None);
        stats.free_count = 0;
        stats.inactive_count = 0;
        assert_eq!(macos_available_bytes(&stats, 16384), Some(0));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_host_memory_observation_is_available() {
        let memory = system_memory().unwrap();
        let total = memory.total.expect("macOS physical capacity");
        let available = memory.available.expect("macOS host VM statistics");
        assert!(total > 0);
        assert!(available <= total);
        eprintln!("macOS host memory: {available} available / {total} total bytes");
    }

    #[test]
    fn system_memory_is_ordered_when_available() {
        let memory = system_memory().unwrap();
        if let (Some(total), Some(available)) = (memory.total, memory.available) {
            assert!(total > 0);
            assert!(available <= total);
        }
    }

    #[test]
    fn process_usage_reports_a_nonzero_resident_set_when_supported() {
        if let Some(usage) = process_usage() {
            assert!(usage.peak_rss > 0);
        }
    }
}
