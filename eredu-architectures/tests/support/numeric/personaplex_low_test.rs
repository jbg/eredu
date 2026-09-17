#[test]
fn personaplex_wide_demand_preserves_full_hidden_and_complete_kv_through_three_decodes() {
    let (artifact, config) = fixture(None);
    for mode in MODES {
        let full = run_low(
            artifact.path(),
            &config,
            mode,
            OutputDemand::Sequence,
            true,
            false,
        );
        assert_eq!(full.len(), 5);
        for (demand, observed) in [
            (OutputDemand::Sequence, false),
            (OutputDemand::LastPosition, false),
            (OutputDemand::StateOnly, false),
            (OutputDemand::StateOnly, true),
        ] {
            let actual = run_low(artifact.path(), &config, mode, demand, false, observed);
            for (index, (a, b)) in full.iter().zip(&actual).enumerate() {
                let width = [2, 3, 1, 1, 1][index];
                same_state(&a.state, &b.state);
                assert_tensor_exact(
                    &a.hidden,
                    &b.hidden,
                    "full temporal hidden still feeds all sixteen depth bodies",
                );
                assert_eq!(a.hidden.shape, [1, width, 8]);
                assert!(b.state.as_ref().iter().all(|s| s
                    .attention
                    .as_ref()
                    .unwrap()
                    .keys
                    .as_ref()
                    .unwrap()
                    .data
                    .iter()
                    .any(|v| v.abs() > 1e-9)));
                assert_eq!(a.projections.len(), 17);
                assert!(a.projections.iter().all(|(_, s)| s == &[1, width, 8]));
                assert_eq!(a.diagnostics.len(), 17);
                let effective = if observed {
                    OutputDemand::LastPosition
                } else {
                    demand
                };
                let positions = effective.positions(width as u64) as i32;
                assert_eq!(
                    b.projections.len(),
                    if observed {
                        17
                    } else {
                        usize::from(positions != 0)
                    }
                );
                assert!(b.projections.iter().all(|(_, s)| s == &[1, positions, 8]));
                assert!(b.diagnostics.is_empty());
                match &b.output {
                    Some(output) => {
                        let expected = if positions == width {
                            a.output.as_ref().unwrap().clone()
                        } else {
                            a.output.as_ref().unwrap().axis_slice(
                                1,
                                width as usize - 1,
                                width as usize,
                            )
                        };
                        assert_tensor_exact(output, &expected, "public final rows");
                    }
                    None => assert_eq!(effective, OutputDemand::StateOnly),
                }
                if observed {
                    assert_eq!(b.values.len(), moshi::observation_points(&config).len() + 1);
                    for (path, value) in &b.values {
                        assert_eq!(
                            value.shape[1],
                            if path.ends_with("logits") { 1 } else { width },
                            "{path}"
                        );
                    }
                }
            }
        }
    }
}
