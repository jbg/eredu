//! Actual nonzero projection inputs, state, and frame-driver demand conformance.
use super::*;
use eredu_core::OutputDemand;
use eredu_runtime::{PredictionDirective, SequentialDecisionDriver, SequentialDecisionPlan};

const RESIDENCIES: [ExecutionResidency; 3] = [
    ExecutionResidency::FullyResident,
    ExecutionResidency::LayerwiseHost,
    ExecutionResidency::DenseDiskStream,
];
include!("../../support/numeric/vocabulary.rs");
fn same_decoded_frames(a: &Report, b: &Report) {
    assert_eq!(a.frames.len(), b.frames.len());
    assert_eq!(a.completion_calls, b.completion_calls);
    assert_eq!(a.committed_work, b.committed_work);
    for (a, b) in a.frames.iter().zip(&b.frames) {
        same_snapshot(&a.snapshot, &b.snapshot);
        same_outputs(&a.outputs, &b.outputs);
    }
}

#[test]
fn moshi_frame_demand_omits_only_unconsumed_forced_projections_across_residencies() {
    let (artifact, config) = fixture(2, false, 1.0);
    for residency in RESIDENCIES {
        for force_text_only in [false, true] {
            let base = Case {
                residency,
                forced: !force_text_only,
                force_text_only,
                diagnostics: false,
                ..Default::default()
            };
            let reference = run(artifact.path(), &config, base);
            let lean = run(
                artifact.path(),
                &config,
                Case {
                    readout: OutputDemand::StateOnly,
                    bounded: true,
                    ..base
                },
            );
            same_decoded_frames(&reference, &lean);
            let full = vocabulary(&reference.projections);
            let selected = vocabulary(&lean.projections);
            assert_eq!(
                full.iter()
                    .filter(|(p, _)| p == "text_linear.weight")
                    .count(),
                10
            );
            assert!(selected
                .iter()
                .all(|(p, shape)| p != "text_linear.weight" && shape == &[1, 1, 4]));
            assert_eq!(selected.len(), if force_text_only { 20 } else { 0 });
            for frame in &lean.frames {
                assert!(frame
                    .values
                    .iter()
                    .all(|(p, _)| p != "model.logits" && p != "text_linear.logits"));
                assert!(frame.diagnostics.is_empty());
                assert!(frame.snapshot.state.as_ref()[..2].iter().all(|s| {
                    s.attention
                        .as_ref()
                        .unwrap()
                        .keys
                        .as_ref()
                        .unwrap()
                        .data
                        .iter()
                        .any(|v| v.abs() > 1e-9)
                }));
            }
            if force_text_only {
                assert!(lean.final_state.calls[1..].iter().all(|n| *n > 0));
                assert!(lean.final_state.state.as_ref()[2..]
                    .iter()
                    .all(|s| s.position() > 0));
            } else {
                assert!(lean.final_state.calls.iter().all(|n| *n == 0));
                assert!(lean.final_state.state.as_ref()[2..]
                    .iter()
                    .all(|s| s.position() == 0));
            }
        }
    }
}

#[test]
fn moshi_required_final_observers_execute_forced_tail_and_intervene_before_decisions() {
    let (artifact, config) = fixture(2, false, 1.0);
    for residency in RESIDENCIES {
        let reference = run(
            artifact.path(),
            &config,
            Case {
                residency,
                forced: true,
                ..Default::default()
            },
        );
        let required = run(
            artifact.path(),
            &config,
            Case {
                residency,
                forced: true,
                diagnostics: false,
                readout: OutputDemand::StateOnly,
                required_observer: true,
                bounded: true,
                ..Default::default()
            },
        );
        same_decoded_frames(&reference, &required);
        let projected = vocabulary(&required.projections);
        assert_eq!(projected.len(), 30);
        assert!(projected.iter().all(|(_, shape)| shape == &[1, 1, 4]));
        for (a, b) in reference.frames.iter().zip(&required.frames) {
            same_values(&a.values, &b.values);
            assert!(b.diagnostics.is_empty());
        }
        let intervened = run(
            artifact.path(),
            &config,
            Case {
                residency,
                diagnostics: false,
                readout: OutputDemand::StateOnly,
                required_observer: true,
                intervene_text: true,
                ..Default::default()
            },
        );
        assert_eq!(vocabulary(&intervened.projections).len(), 30);
        for frame in &intervened.frames {
            assert_eq!(
                frame.outputs[0].data,
                [3.0],
                "real sampler consumed the intervened text projection"
            );
            let logits = &frame
                .values
                .iter()
                .find(|(p, _)| p == "model.logits")
                .unwrap()
                .1;
            assert_eq!(logits.shape, [1, 1, 7]);
            assert_eq!(logits.data[3], 100.0);
        }
    }
}

#[test]
fn moshi_state_only_public_scores_preserve_diagnostics_and_terminal_prefix() {
    let (artifact, config) = fixture(3, false, 1.0);
    for residency in RESIDENCIES {
        let base = Case {
            residency,
            ..Default::default()
        };
        let reference = run(artifact.path(), &config, base);
        let lean = run(
            artifact.path(),
            &config,
            Case {
                readout: OutputDemand::StateOnly,
                bounded: true,
                ..base
            },
        );
        same_decoded_frames(&reference, &lean);
        assert_eq!(
            vocabulary(&lean.projections),
            vocabulary(&reference.projections)
        );
        for (a, b) in reference.frames.iter().zip(&lean.frames) {
            same_outputs(&a.diagnostics, &b.diagnostics);
            assert_eq!(b.diagnostics.len(), 4);
        }
        for (fail, cancel) in [(Some("depformer.slices.1.logits"), false), (None, true)] {
            let terminal = run(
                artifact.path(),
                &config,
                Case {
                    readout: OutputDemand::StateOnly,
                    fail,
                    cancel,
                    bounded: true,
                    ..base
                },
            );
            assert_eq!(terminal.frames.len(), 2);
            for (a, b) in reference.frames.iter().zip(&terminal.frames) {
                same_snapshot(&a.snapshot, &b.snapshot);
                same_outputs(&a.outputs, &b.outputs);
            }
            assert_eq!(terminal.committed_work, 2);
            assert!(terminal.retired_state.is_some());
        }
    }
}

include!("../../support/numeric/frame_low.rs");

#[test]
fn moshi_wide_low_level_demand_selects_hidden_before_vocab_and_keeps_complete_cached_state() {
    let (artifact, config) = fixture(2, false, 1.0);
    for residency in RESIDENCIES {
        let reference = low(
            artifact.path(),
            &config,
            residency,
            OutputDemand::Sequence,
            true,
            false,
        );
        for (demand, observed) in [
            (OutputDemand::Sequence, false),
            (OutputDemand::LastPosition, false),
            (OutputDemand::StateOnly, false),
            (OutputDemand::StateOnly, true),
        ] {
            let actual = low(artifact.path(), &config, residency, demand, false, observed);
            for (index, (a, b)) in reference.iter().zip(&actual).enumerate() {
                let width = [3, 2, 1, 1][index];
                same_state(&a.state, &b.state);
                assert_tensor_exact(&a.hidden, &b.hidden, "complete temporal dependency");
                assert!(b.state.as_ref().iter().all(|layer| layer
                    .attention
                    .as_ref()
                    .unwrap()
                    .keys
                    .as_ref()
                    .unwrap()
                    .data
                    .iter()
                    .any(|v| v.abs() > 1e-9)));
                let effective = if observed {
                    OutputDemand::LastPosition
                } else {
                    demand
                };
                let expected_width = effective.positions(width as u64) as i32;
                assert_eq!(a.projections.len(), 3);
                assert!(a
                    .projections
                    .iter()
                    .all(|(_, shape)| shape == &[1, width, 4]));
                assert_eq!(
                    b.projections.len(),
                    if observed {
                        3
                    } else {
                        usize::from(expected_width != 0)
                    }
                );
                assert!(b
                    .projections
                    .iter()
                    .all(|(_, shape)| shape == &[1, expected_width, 4]));
                assert_eq!(a.diagnostics.len(), 3);
                assert!(b.diagnostics.is_empty());
                match &b.output {
                    Some(output) => {
                        let full = a.output.as_ref().unwrap();
                        let expected = if expected_width == width {
                            full.clone()
                        } else {
                            full.axis_slice(1, width as usize - 1, width as usize)
                        };
                        assert_tensor_exact(output, &expected, "actual public selected rows");
                    }
                    None => assert_eq!(effective, OutputDemand::StateOnly),
                }
                if observed {
                    assert_eq!(b.values.len(), 7);
                    for (path, value) in &b.values {
                        let expected = if path.ends_with("logits") { 1 } else { width };
                        assert_eq!(value.shape[1], expected, "{path}");
                    }
                }
            }
        }
    }
}
