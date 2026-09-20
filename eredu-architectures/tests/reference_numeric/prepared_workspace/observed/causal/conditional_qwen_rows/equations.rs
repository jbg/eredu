use super::*;

fn compare(f: &Fixture) {
    let c = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let (mut model, d, paths) = f.model(&c);
    let mut full = f.state();
    model
        .run(
            &[4, 2],
            &mut full,
            &c,
            &mut Rows::new(&[]),
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
    populated(&full, 2);
    let prefix = full.clone();
    let mut chunked = prefix.clone();
    let mut expected_rows = Rows::new(&d);
    let expected = model
        .run(
            &[1, 3, 5, 2, 6],
            &mut full,
            &c,
            &mut expected_rows,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap()
        .unwrap();
    expected_rows.complete(5);
    populated(&full, 7);
    let mut joined: BTreeMap<String, Vec<NumericTensor>> = BTreeMap::new();
    let mut outputs = Vec::new();
    let mut consumed = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut actual = Rows::new(&d);
        let out = model
            .run(
                ids,
                &mut chunked,
                &c,
                &mut actual,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap()
            .unwrap();
        actual.complete(ids.len());
        let start = consumed.len();
        consumed.extend_from_slice(ids);
        for (path, value) in actual.values {
            assert_tensor_close(
                &value,
                &expected_rows.values[&path].axis_slice(1, start, consumed.len()),
                &path,
            );
            joined.entry(path).or_default().push(value);
        }
        outputs.push(out);
        let mut reference = prefix.clone();
        model
            .run(
                &consumed,
                &mut reference,
                &c,
                &mut Rows::new(&[]),
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
        same_state(&chunked, &reference);
        populated(&chunked, 2 + consumed.len() as i32);
    }
    assert_tensor_close(
        &NumericTensor::concatenate(&outputs, 1, &c).unwrap(),
        &expected,
        "full/chunked conditional scores",
    );
    for (path, values) in joined {
        assert_tensor_close(
            &NumericTensor::concatenate(&values, 1, &c).unwrap(),
            &expected_rows.values[&path],
            &path,
        );
    }
    same_state(&full, &chunked);
    for (step, id) in [1, 3, 2].into_iter().enumerate() {
        let mut a = Rows::new(&d);
        let mut b = Rows::new(&d);
        let ao = model
            .run(&[id], &mut full, &c, &mut a, &paths, OutputDemand::Sequence)
            .unwrap()
            .unwrap();
        let bo = model
            .run(
                &[id],
                &mut chunked,
                &c,
                &mut b,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap()
            .unwrap();
        a.complete(1);
        b.complete(1);
        assert_tensor_close(&ao, &bo, "next cached decode");
        for (path, value) in a.values {
            assert_tensor_close(&value, &b.values[&path], &path);
        }
        same_state(&full, &chunked);
        populated(&chunked, 8 + step as i32);
    }
}
#[test]
fn conditional_qwen_vl_and_hybrid_rows_preserve_all_eighteen_callbacks_and_complete_state() {
    for hybrid in [false, true] {
        for routed in [false, true] {
            for edge in [false, true] {
                for tied in [false, true] {
                    compare(&Fixture::new(
                        configuration(hybrid, routed, edge, tied),
                        hybrid,
                    ));
                }
            }
        }
    }
}
#[test]
fn conditional_qwen_body_rows_precede_state_only_and_last_position_readout() {
    for hybrid in [false, true] {
        for routed in [false, true] {
            for edge in [false, true] {
                let f = Fixture::new(configuration(hybrid, routed, edge, false), hybrid);
                let c = NumericContext {
                    bind_checkpoint_values: true,
                    ..Default::default()
                };
                let (mut model, d, paths) = f.model(&c);
                let body = d
                    .iter()
                    .filter(|d| d.readout_stage() == Stage::BeforeReadout)
                    .cloned()
                    .collect::<Vec<_>>();
                assert!(body.len() >= 10);
                let mut state = f.state();
                model
                    .run(
                        &[4, 2],
                        &mut state,
                        &c,
                        &mut Rows::new(&[]),
                        &paths,
                        OutputDemand::Sequence,
                    )
                    .unwrap();
                let prefix = state.clone();
                let mut complete = Rows::new(&d);
                let scores = model
                    .run(
                        &[1, 3, 5, 2, 6],
                        &mut state,
                        &c,
                        &mut complete,
                        &paths,
                        OutputDemand::Sequence,
                    )
                    .unwrap()
                    .unwrap();
                complete.complete(5);
                for demand in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
                    let mut trial = prefix.clone();
                    let mut consumed = Vec::new();
                    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
                        let mut rows = Rows::new(&body);
                        let out = model
                            .run(ids, &mut trial, &c, &mut rows, &paths, demand)
                            .unwrap();
                        rows.complete(ids.len());
                        let start = consumed.len();
                        consumed.extend_from_slice(ids);
                        for (path, value) in rows.values {
                            assert_tensor_close(
                                &value,
                                &complete.values[&path].axis_slice(1, start, consumed.len()),
                                &path,
                            );
                        }
                        match demand {
                            OutputDemand::StateOnly => assert!(out.is_none()),
                            OutputDemand::LastPosition => assert_tensor_close(
                                &out.unwrap(),
                                &scores.axis_slice(1, consumed.len() - 1, consumed.len()),
                                "actual last position",
                            ),
                            _ => unreachable!(),
                        }
                        let mut reference = prefix.clone();
                        model
                            .run(
                                &consumed,
                                &mut reference,
                                &c,
                                &mut Rows::new(&[]),
                                &paths,
                                OutputDemand::Sequence,
                            )
                            .unwrap();
                        same_state(&trial, &reference);
                    }
                    same_state(&trial, &state);
                    let mut expected = state.clone();
                    for id in [1, 3, 2] {
                        let a = model
                            .run(
                                &[id],
                                &mut trial,
                                &c,
                                &mut Rows::new(&d),
                                &paths,
                                OutputDemand::Sequence,
                            )
                            .unwrap()
                            .unwrap();
                        let b = model
                            .run(
                                &[id],
                                &mut expected,
                                &c,
                                &mut Rows::new(&d),
                                &paths,
                                OutputDemand::Sequence,
                            )
                            .unwrap()
                            .unwrap();
                        assert_tensor_close(&a, &b, "body-mode resumed decode");
                        same_state(&trial, &expected);
                    }
                }
            }
        }
    }
}
