//! Scalar storage adapters for the existing selected realtime constructor.
//! A host payload is not an executable unit; disk retains recipes, not payloads.
include!("../../support/numeric/frame_residency.rs");

#[test]
fn moshi_selected_host_and_disk_frames_match_resident_with_real_unit_eviction() {
    for depth in [2, 3] {
        let (artifact, config) = fixture(depth, true, 1.0);
        for batch in [1, 2] {
            let case = Case {
                batch,
                bounded: true,
                ..Default::default()
            };
            let reference = run(artifact.path(), &config, case);
            for mode in [
                ExecutionResidency::LayerwiseHost,
                ExecutionResidency::DenseDiskStream,
            ] {
                let actual = run(
                    artifact.path(),
                    &config,
                    Case {
                        residency: mode,
                        ..case
                    },
                );
                same_report(&reference, &actual);
                assert_eq!(actual.bound_parameters, reference.bound_parameters);
                storage_proof(&actual, mode);
                let expected = (0..actual.residency.unit_count).collect::<Vec<_>>();
                assert_eq!(actual.residency.acquired, expected.repeat(10));
            }
        }
    }
}

#[test]
fn moshi_selected_host_and_disk_forced_tail_match_resident_execution_boundary() {
    for depth in [2, 3] {
        let (artifact, config) = fixture(depth, false, 1.0);
        for diagnostics in [false, true] {
            let case = Case {
                forced: true,
                diagnostics,
                bounded: true,
                ..Default::default()
            };
            let reference = run(artifact.path(), &config, case);
            for mode in [
                ExecutionResidency::LayerwiseHost,
                ExecutionResidency::DenseDiskStream,
            ] {
                let actual = run(
                    artifact.path(),
                    &config,
                    Case {
                        residency: mode,
                        ..case
                    },
                );
                same_report(&reference, &actual);
                storage_proof(&actual, mode);
                // Existing user audio also becomes a Force directive. Both
                // depth configurations omit the fully forced depth tail when
                // diagnostics are disabled, leaving only the temporal units.
                let units = if diagnostics {
                    actual.residency.unit_count
                } else {
                    2
                };
                assert_eq!(
                    actual.residency.acquired,
                    (0..units).collect::<Vec<_>>().repeat(10)
                );
            }
        }
    }
}

#[test]
fn moshi_selected_host_and_disk_failure_and_cancellation_retire_leased_prefix() {
    let (artifact, config) = fixture(3, false, 1.0);
    for (fail, cancel) in [
        (Some("text_linear.logits"), false),
        (Some("depformer.slices.1.logits"), false),
        (None, true),
    ] {
        let case = Case {
            fail,
            cancel,
            bounded: true,
            ..Default::default()
        };
        let reference = run(artifact.path(), &config, case);
        for mode in [
            ExecutionResidency::LayerwiseHost,
            ExecutionResidency::DenseDiskStream,
        ] {
            let actual = run(
                artifact.path(),
                &config,
                Case {
                    residency: mode,
                    ..case
                },
            );
            same_report(&reference, &actual);
            storage_proof(&actual, mode);
            assert_eq!(actual.committed_work, 2);
        }
    }
}
