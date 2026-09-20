use super::*;

#[test]
fn inkling_prepared_composite_retains_direct_text_rows_and_all_mutable_state() {
    for sparse in [false, true] {
        for kernel in [1, 3] {
            let f = fixture(configuration(sparse, kernel, true));
            let context = NumericContext {
                bind_checkpoint_values: true,
                ..Default::default()
            };
            let (mut prepared, declarations, paths) = make_model(&f, &context);
            let mut direct = ResidentRuntime::<Architecture, NumericBackend, State>::new(
                Architecture::new(f.args.clone(), &context).unwrap(),
                &context,
            )
            .unwrap();
            let mut populate = Populate(&f.parameters, BTreeMap::new());
            <Architecture as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(
                direct.architecture_mut(),
            )
            .visit_parameters_mut(&mut populate);
            for unit in direct.units_mut().iter_mut().flatten() {
                unit.visit_parameters_mut(&mut populate);
            }
            let direct_paths = direct.prepare_observation_paths().unwrap();
            for d in &declarations {
                assert_eq!(direct_paths.source().prefill_observation(d.path()), Some(d));
            }
            assert!(!direct_paths.source().same_storage(paths.source()));
            let mut expected = state(&f);
            let mut actual = state(&f);
            for ids in [
                &[4, 2][..],
                &[1, 3][..],
                &[5][..],
                &[2, 6][..],
                &[7][..],
                &[8][..],
                &[9][..],
            ] {
                let mut a = Rows::new(&declarations);
                let mut b = Rows::new(&declarations);
                let scores = run(
                    &mut prepared,
                    &f,
                    ids,
                    &mut actual,
                    &context,
                    &mut a,
                    &paths,
                    OutputDemand::Sequence,
                )
                .unwrap();
                let tokens = NumericTensor::token_ids(ids);
                let parts = [inkling::DecoderInputPart::Text(&tokens)];
                let (raw, _) = direct
                    .forward_with_prepared_observer_and_context_with_readout(
                        inkling::ModelInput {
                            parts: &parts,
                            vision_patches: None,
                            audio: None,
                        },
                        &mut expected,
                        &context,
                        &mut b,
                        &direct_paths,
                        OutputDemand::Sequence,
                    )
                    .unwrap();
                let raw = eredu_runtime::observe_model_logits(&mut b, &raw.unwrap()).unwrap();
                assert_tensor_close(&scores, &raw, "actual direct/prepared text scores");
                a.complete(ids.len());
                b.complete(ids.len());
                for d in &declarations {
                    assert_tensor_close(&a.values[d.path()], &b.values[d.path()], d.path());
                }
                compare_state(&actual, &expected);
            }
            populated(&actual, &f, 10);
        }
    }
}

#[test]
fn inkling_prepared_sources_bind_actual_group_two_and_scaled_readout() {
    for sparse in [false, true] {
        let f = fixture(configuration(sparse, 1, true));
        let inspection =
            eredu_architectures::configuration::inspect_artifact(f.artifact.path()).unwrap();
        let sources = prepared_adapter::prepare(
            &inspection,
            &prepared_adapter::plan(None),
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let discovery = sources.prepare_discovery(
            ObservationMechanisms {
                activation_tensors: true,
                floating_to_f32: true,
                ..Default::default()
            },
            CaptureCapabilities {
                transformations: vec![
                    CaptureTransformKind::FullTensor,
                    CaptureTransformKind::Preview,
                ],
                ..Default::default()
            },
        );
        let context = NumericContext::default();
        let (_, declarations, paths) = make_model(&f, &context);
        let (_, _, other_paths) = make_model(&f, &context);
        let before = sources.target().source_diagnostics().unwrap();
        for d in &declarations {
            for preview in [false, true] {
                let source = admission(&discovery, &[d.path()], preview, true);
                let selected = discovery
                    .prepare_capture_selection(&source, paths.source())
                    .unwrap();
                assert_eq!(selected.declaration(0).unwrap(), Some(d));
                assert!(selected.source().same_storage(&source));
                assert!(selected.paths().same_storage(paths.source()));
                selected
                    .validate_sources(&source.clone(), &paths.source().clone())
                    .unwrap();
                assert!(matches!(
                    selected.validate_sources(&source, other_paths.source()),
                    Err(SelectionError::Identity)
                ));
                let distinct = admission(&discovery, &[d.path()], preview, true);
                assert!(matches!(
                    selected.validate_sources(&distinct, paths.source()),
                    Err(SelectionError::Identity)
                ));
                for logical in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
                    let physical = if d.readout_stage() == Stage::BeforeReadout {
                        logical
                    } else {
                        OutputDemand::Sequence
                    };
                    assert_eq!(selected.physical_output(logical), physical);
                    let mut geometry = request(4);
                    geometry.prefill_chunk_positions = 2;
                    geometry.output = physical;
                    let bound = selected.bind_geometry(geometry).unwrap();
                    assert!(std::ptr::eq(bound.selection(), &selected));
                    if physical == OutputDemand::Sequence {
                        geometry.output = OutputDemand::LastPosition;
                        assert!(matches!(
                            selected.bind_geometry(geometry),
                            Err(SelectionError::Readout)
                        ));
                    }
                }
            }
        }
        let internal = "model.layers.0.attention.input";
        assert!(paths.source().prefill_observation(internal).is_none());
        let source = admission(&discovery, &[internal], false, true);
        assert!(matches!(
            discovery.prepare_capture_selection(&source, paths.source()),
            Err(SelectionError::Undeclared { index: 0 })
        ));
        assert_eq!(
            before.physical_reads,
            sources
                .target()
                .source_diagnostics()
                .unwrap()
                .physical_reads
        );
    }
}

#[test]
fn inkling_body_rows_preserve_complete_state_before_readout_selection() {
    for sparse in [false, true] {
        let f = fixture(configuration(sparse, 1, true));
        let context = NumericContext {
            bind_checkpoint_values: true,
            cache_owned_attention: false,
            ..Default::default()
        };
        let (mut model, declarations, paths) = make_model(&f, &context);
        let body = declarations
            .into_iter()
            .filter(|d| d.readout_stage() == Stage::BeforeReadout)
            .collect::<Vec<_>>();
        assert!(body.len() >= 10);
        let mut prefix = state(&f);
        run(
            &mut model,
            &f,
            &[4, 2],
            &mut prefix,
            &context,
            &mut Rows::new(&[]),
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        let mut expected_state = prefix.clone();
        let mut expected = Rows::new(&body);
        let scores = run(
            &mut model,
            &f,
            &[1, 3, 5, 2, 6],
            &mut expected_state,
            &context,
            &mut expected,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        expected.complete(5);
        for demand in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
            let mut state = prefix.clone();
            let mut start = 0;
            for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
                let end = start + ids.len();
                let mut rows = Rows::new(&body);
                assert!(!rows.requires_sequence_readout());
                let result = run(
                    &mut model, &f, ids, &mut state, &context, &mut rows, &paths, demand,
                );
                rows.complete(ids.len());
                for d in &body {
                    assert_tensor_close(
                        &rows.values[d.path()],
                        &expected.values[d.path()].axis_slice(1, start, end),
                        d.path(),
                    );
                }
                if demand == OutputDemand::StateOnly {
                    assert!(result.is_none());
                } else {
                    assert_tensor_close(
                        &result.unwrap(),
                        &scores.axis_slice(1, end - 1, end),
                        "selected actual unpadded last row",
                    );
                }
                populated(&state, &f, 2 + end as i32);
                start = end;
            }
            compare_state(&state, &expected_state);
        }
    }
}
