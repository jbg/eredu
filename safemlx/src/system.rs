//! Safe process and operating-system resource observations used by runtimes.

/// Host physical-memory observations in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMemory {
    /// Installed physical memory, when available.
    pub total: Option<u64>,
    /// Physical memory currently available to the process, when available.
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
    unsafe extern "C" {
        fn os_proc_available_memory() -> usize;
    }

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
    let available = u64::try_from(unsafe { os_proc_available_memory() })
        .ok()
        .filter(|value| *value > 0);
    Ok(SystemMemory { total, available })
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
