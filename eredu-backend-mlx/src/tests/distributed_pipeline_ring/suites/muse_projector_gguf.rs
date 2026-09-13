fn run_muse_projector_gguf_component_case(
    source: &Path,
    routed: bool,
    axes: &'static str,
    residency: WorkerResidency,
    mode: WorkerMode,
) {
    eprintln!("Muse packed projector routed={routed} {axes} {residency:?} {mode:?}");
    let checkpoint = tempfile::tempdir().unwrap();
    // Share immutable fixture bytes while giving each run its own source paths,
    // prompt cache and experimental state. Edits never write these artifacts.
    for name in [
        "model.gguf",
        "mmproj.gguf",
        "config.json",
        "component-media-fixture.json",
    ] {
        std::fs::hard_link(source.join(name), checkpoint.path().join(name)).unwrap();
    }
    let path = checkpoint.path().join("model.gguf");
    run_ring_pipeline_processes(
        residency,
        if routed {
            FixtureFamily::MuseGlimmerMoe
        } else {
            FixtureFamily::MuseGlimmer
        },
        mode,
        checkpoint,
        path,
        Some(axes),
    );
}

fn run_muse_projector_gguf_component_matrix(routed: bool, mode: WorkerMode) {
    let source = tempfile::tempdir().unwrap();
    write_muse_projector_gguf_component_fixture(&source.path().join("model.gguf"), routed);
    let axes: &[&str] = if routed {
        &["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"]
    } else {
        &["tp", "pp", "tp-pp"]
    };
    for &axes in axes {
        for residency in [
            WorkerResidency::FullyResident,
            WorkerResidency::LayerwiseHost,
            WorkerResidency::DenseDiskStream,
        ] {
            run_muse_projector_gguf_component_case(source.path(), routed, axes, residency, mode);
        }
    }
}

#[test]
#[ignore = "uses a published-geometry packed projector and local Ring ranks; run explicitly"]
fn ring_muse_projector_gguf_components_focused() {
    for routed in [true, false] {
        let source = tempfile::tempdir().unwrap();
        write_muse_projector_gguf_component_fixture(&source.path().join("model.gguf"), routed);
        if routed {
            run_muse_projector_gguf_component_case(
                source.path(),
                true,
                "pp",
                WorkerResidency::DenseDiskStream,
                WorkerMode::OpaqueComponentCapture,
            );
            run_muse_projector_gguf_component_case(
                source.path(),
                true,
                "tp-pp-ep",
                WorkerResidency::LayerwiseHost,
                WorkerMode::OpaqueComponentCaptureAddressableBank,
            );
        } else {
            run_muse_projector_gguf_component_case(
                source.path(),
                false,
                "tp",
                WorkerResidency::FullyResident,
                WorkerMode::OpaqueComponentCapture,
            );
        }
    }
}

#[test]
#[ignore = "uses a published-geometry packed projector and two to eight local Ring ranks; run explicitly"]
fn ring_muse_projector_gguf_components_independent_pipeline_focused() {
    let source = tempfile::tempdir().unwrap();
    write_muse_projector_gguf_component_fixture(&source.path().join("model.gguf"), true);
    run_muse_projector_gguf_component_case(
        source.path(),
        true,
        "tp-pp-ep",
        WorkerResidency::LayerwiseHost,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}

#[test]
#[ignore = "uses a published-geometry packed projector and two to eight local Ring ranks; run explicitly"]
fn ring_muse_projector_gguf_components_matrix() {
    for routed in [false, true] {
        run_muse_projector_gguf_component_matrix(routed, WorkerMode::OpaqueComponentCapture);
    }
}

#[test]
#[ignore = "uses a published-geometry packed projector and two to eight local Ring ranks; run explicitly"]
fn ring_muse_projector_gguf_components_independent_bank_matrix() {
    run_muse_projector_gguf_component_matrix(
        true,
        WorkerMode::OpaqueComponentCaptureAddressableBank,
    );
}
