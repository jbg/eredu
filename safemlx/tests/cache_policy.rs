//! Fresh processes are required: native allocator policy is process-global.
use safemlx::memory::{self, CacheLimitSource as Source};

#[test]
fn native_cache_policy_preserves_explicit_ownership() {
    let Ok(scenario) = std::env::var("EREDU_CACHE_POLICY_TEST_SCENARIO") else {
        for scenario in ["automatic", "preserve", "explicit", "race", "smaller"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "native_cache_policy_preserves_explicit_ownership",
                    "--nocapture",
                ])
                .env("EREDU_CACHE_POLICY_TEST_SCENARIO", scenario)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{scenario}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    };
    let initial = match memory::cache_policy() {
        Ok(policy) => policy,
        Err(error)
            if cfg!(feature = "metal")
                && error.to_string().contains("No Metal device available") =>
        {
            return
        }
        Err(error) => panic!("native cache query failed: {error}"),
    };
    assert_eq!(initial.source, Source::NativeDefault);
    assert_eq!(memory::cache_policy().unwrap(), initial);
    assert_eq!(memory::cache_limit().unwrap(), initial.bytes);
    match scenario.as_str() {
        "automatic" => {
            let configured = memory::configure_default_cache_policy(1024, false).unwrap();
            assert_eq!(configured.bytes, initial.bytes.min(1024));
            assert_eq!(configured.source, Source::ManagedDefault);
            assert_eq!(
                memory::configure_default_cache_policy(0, false).unwrap(),
                configured
            );
            assert_eq!(
                memory::configure_default_cache_policy(0, true).unwrap(),
                configured
            );
            assert_eq!(memory::set_cache_limit(2048).unwrap(), configured.bytes);
            assert_eq!(memory::cache_policy().unwrap().source, Source::Explicit);
            assert_eq!(
                memory::configure_default_cache_policy(0, false)
                    .unwrap()
                    .bytes,
                2048
            );
        }
        "preserve" => {
            let kept = memory::configure_default_cache_policy(0, true).unwrap();
            assert_eq!(kept.bytes, initial.bytes);
            assert_eq!(kept.source, Source::Preserved);
            assert_eq!(
                memory::configure_default_cache_policy(0, false).unwrap(),
                kept
            );
        }
        "explicit" => {
            // Bypass the safe wrapper, and set the SAME value as the native
            // default. Provenance must come from the allocator, not a Rust flag
            // or comparison with the default's numeric value.
            let mut previous = 0;
            assert_eq!(
                unsafe { safemlx_sys::mlx_set_cache_limit(&mut previous, initial.bytes) },
                0
            );
            assert_eq!(previous, initial.bytes);
            assert_eq!(memory::cache_policy().unwrap().source, Source::Explicit);
            assert_eq!(
                memory::configure_default_cache_policy(0, false)
                    .unwrap()
                    .bytes,
                initial.bytes
            );
        }
        "race" => {
            let setter = std::thread::spawn(|| {
                let mut previous = 0;
                assert_eq!(
                    unsafe { safemlx_sys::mlx_set_cache_limit(&mut previous, 4096) },
                    0
                );
            });
            let default =
                std::thread::spawn(|| memory::configure_default_cache_policy(0, false).unwrap());
            setter.join().unwrap();
            default.join().unwrap();
            let policy = memory::cache_policy().unwrap();
            assert_eq!(policy.source, Source::Explicit);
            assert_eq!(policy.bytes, 4096);
        }
        "smaller" => {
            let policy = memory::configure_default_cache_policy(usize::MAX, false).unwrap();
            assert_eq!(policy.bytes, initial.bytes);
            assert_eq!(policy.source, Source::ManagedDefault);
        }
        _ => panic!("unknown scenario"),
    }
}
