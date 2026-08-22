//! Named Cargo target for the crate-private distributed pipeline Ring suite.

use std::process::Command;

fn run_library_ring_selection(selection: &str) {
    let status = Command::new(env!("CARGO"))
        .args([
            "test",
            "-p",
            "eredu",
            "--features",
            "mlx-test-support",
            "--lib",
            selection,
            "--",
            "--ignored",
            "--test-threads=1",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("launch crate-private distributed pipeline Ring suite");
    assert!(status.success(), "Ring selection {selection:?} failed");
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn qwen3_next() {
    run_library_ring_selection("qwen3_next");
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn qwen35() {
    run_library_ring_selection("qwen35");
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn qwen3_vl() {
    run_library_ring_selection("qwen3_vl");
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn gpt_oss() {
    run_library_ring_selection("gpt_oss");
}
