fn main() {
    // Model preparation and CLI dispatch need more than Windows' default 1 MiB
    // main-thread stack in unoptimized builds. Reserve the same 8 MiB available
    // to the main thread on our Unix CLI targets; pages are committed on demand.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        match std::env::var("CARGO_CFG_TARGET_ENV").as_deref() {
            Ok("msvc") => println!("cargo:rustc-link-arg-bin=eredu=/STACK:8388608"),
            Ok("gnu") => println!("cargo:rustc-link-arg-bin=eredu=-Wl,--stack,8388608"),
            _ => {}
        }
    }
}
