//! Fresh process ownership for tests of native singleton admission.

/// Runs one named case in a fresh process, entering its body only in the child.
pub(crate) fn enter(case: &str) -> bool {
    const KEY: &str = "EREDU_NATIVE_TEST_PROCESS";
    let thread = std::thread::current();
    let name = thread.name().expect("named native test");
    let identity = format!("{name}:{case}");
    if let Ok(selected) = std::env::var(KEY) {
        if selected != identity { return false; }
        println!("NATIVE_TEST_ENTERED:{identity}");
        return true;
    }
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--test-threads=1", "--nocapture", "--include-ignored"])
        .env(KEY, &identity)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(result.status.success(), "{stdout}\n{}",
        String::from_utf8_lossy(&result.stderr));
    assert!(stdout.contains(&format!("NATIVE_TEST_ENTERED:{identity}")), "{stdout}");
    false
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
pub(crate) fn metal() -> (
    crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams,
    eredu_runtime::working_memory::WorkingMemoryPool,
    u64,
) {
    let pool = crate::backend::managed_memory::domain();
    let streams = crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_factory(&pool)
        .unwrap().expect("qualified native stream factory");
    let baseline = pool.used_bytes().unwrap();
    (streams, pool, baseline)
}
