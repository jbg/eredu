use super::*;
use eredu_architectures::qwen;
use eredu_runtime::layered::PrefillObservationDeclaration;

type QwenState = DeviceState<NumericBackend, NumericHybridLayerState>;
type QwenModel<P> = decoder::LayeredModel<NumericBackend, qwen::ModelArgs, P>;

fn configuration(kind: &str, tied: bool) -> serde_json::Value {
    let mut value = config(kind, tied);
    value["num_hidden_layers"] = 2.into();
    if kind == "qwen2" {
        // Cross the two-position window while retaining a full-attention layer.
        value["layer_types"] = serde_json::json!(["sliding_attention", "full_attention"]);
    }
    value
}

fn assert_rows(rows: &Rows, declarations: &[PrefillObservationDeclaration], positions: i32) {
    for declaration in declarations {
        let path = declaration.path();
        let value = rows.0.get(path).unwrap_or_else(|| panic!("missing {path}"));
        assert_eq!(declaration.sequence_axis(), 1);
        assert_eq!(value.shape.len(), 3, "{path}");
        assert_eq!(value.shape[..2], [1, positions], "{path}");
        assert!(value.data.iter().all(|v| v.is_finite()), "finite {path}");
        assert!(value.data.iter().any(|v| v.abs() > 1e-6), "nonzero {path}");
    }
}

fn assert_state_close(actual: &QwenState, expected: &QwenState) {
    assert_eq!(actual.as_ref().len(), expected.as_ref().len());
    for (actual, expected) in actual.as_ref().iter().zip(expected.as_ref()) {
        assert_eq!(actual.position(), expected.position());
        let a = RuntimeLayerState::<NumericBackend>::retained_values(actual).collect::<Vec<_>>();
        let b = RuntimeLayerState::<NumericBackend>::retained_values(expected).collect::<Vec<_>>();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.into_iter().zip(b) {
            assert_tensor_close(a, b, "Qwen cached state");
        }
    }
}

fn compare_equations<P>(config: serde_json::Value)
where
    P: decoder::BlockFactory<NumericBackend, qwen::ModelArgs>,
{
    let args = qwen::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let architecture = QwenModel::<P>::new(args.clone(), &context).unwrap();
    let declarations = tensor_row_declarations(<QwenModel<P> as LayeredArchitecture<NumericBackend, QwenState>>::
        prefill_observation_declarations(&architecture, None).unwrap());
    // Check every declared hook, including effective and final vocabulary rows.
    assert!(declarations.len() >= 18);
    let mut model = ResidentRuntime::new(architecture, &context).unwrap();
    let paths = model.prepare_observation_paths().unwrap();
    for declaration in &declarations {
        assert_eq!(
            paths.source().prefill_observation(declaration.path()),
            Some(declaration)
        );
    }
    let mut full_state = QwenState::create(qwen::state_layout(&args).unwrap(), |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    model
        .forward(
            decoder::LayeredInput {
                tokens: &NumericTensor::token_ids(&[4, 2]),
                mask: None,
            },
            &mut full_state,
            &context,
        )
        .unwrap();
    let mut chunk_state = full_state.clone();
    let mut full = Rows::default();
    let (scores, _) = model
        .forward_with_prepared_observer_and_context_with_readout(
            decoder::LayeredInput {
                tokens: &NumericTensor::token_ids(&[1, 3, 5, 2, 6]),
                mask: None,
            },
            &mut full_state,
            &context,
            &mut full,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
    eredu_runtime::observe_model_logits(&mut full, &scores.unwrap()).unwrap();
    assert_rows(&full, &declarations, 5);
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = Rows::default();
        let (scores, _) = model
            .forward_with_prepared_observer_and_context_with_readout(
                decoder::LayeredInput {
                    tokens: &NumericTensor::token_ids(ids),
                    mask: None,
                },
                &mut chunk_state,
                &context,
                &mut rows,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
        eredu_runtime::observe_model_logits(&mut rows, &scores.unwrap()).unwrap();
        assert_rows(&rows, &declarations, ids.len() as i32);
        for declaration in &declarations {
            assembled
                .entry(declaration.path().into())
                .or_default()
                .extend_from_slice(&rows.0[declaration.path()].data);
        }
    }
    assert_eq!(assembled.len(), declarations.len());
    for declaration in &declarations {
        let path = declaration.path();
        let expected = &full.0[path];
        assert_tensor_close(
            &NumericTensor::new(expected.shape.clone(), assembled.remove(path).unwrap()),
            expected,
            path,
        );
    }
    assert_state_close(&chunk_state, &full_state);
    for id in [7, 8, 9] {
        let tokens = NumericTensor::token_ids(&[id]);
        let mut expected = Rows::default();
        let mut actual = Rows::default();
        for (state, rows) in [
            (&mut full_state, &mut expected),
            (&mut chunk_state, &mut actual),
        ] {
            let (scores, _) = model
                .forward_with_prepared_observer_and_context_with_readout(
                    decoder::LayeredInput {
                        tokens: &tokens,
                        mask: None,
                    },
                    state,
                    &context,
                    rows,
                    &paths,
                    OutputDemand::Sequence,
                )
                .unwrap();
            eredu_runtime::observe_model_logits(rows, &scores.unwrap()).unwrap();
            assert_rows(rows, &declarations, 1);
        }
        for declaration in &declarations {
            let path = declaration.path();
            assert_tensor_close(&actual.0[path], &expected.0[path], path);
        }
        assert_state_close(&chunk_state, &full_state);
    }
    assert!(full_state
        .as_ref()
        .iter()
        .all(|state| state.position() == 10));
}

#[test]
fn dense_qwen_causal_rows_preserve_uneven_prefill_and_repeated_cached_decode() {
    for (kind, tied) in [("qwen2", false), ("qwen3", true)] {
        compare_equations::<qwen::QwenBlockFactory>(configuration(kind, tied));
    }
}

#[test]
fn routed_qwen_causal_rows_preserve_dense_and_moe_equations() {
    for (kind, tied) in [("qwen2", false), ("qwen3", true), ("qwen3_moe", false)] {
        compare_equations::<qwen::RoutedQwenBlockFactory>(configuration(kind, tied));
    }
    let mut unnormalized = configuration("qwen3_moe", true);
    unnormalized["norm_topk_prob"] = false.into();
    compare_equations::<qwen::RoutedQwenBlockFactory>(unnormalized);
}

fn check_binding<P>(config: serde_json::Value)
where
    P: decoder::BlockFactory<NumericBackend, qwen::ModelArgs>,
{
    let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
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
            transformations: vec![CaptureTransformKind::FullTensor],
            ..Default::default()
        },
    );
    let args = qwen::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let runtime = ResidentRuntime::<_, NumericBackend, QwenState>::new(
        QwenModel::<P>::new(args, &context).unwrap(),
        &context,
    )
    .unwrap();
    let paths = runtime.prepare_observation_paths().unwrap();
    for (path, stage) in [
        ("readout.embedding", Stage::BeforeReadout),
        ("model.layers.1.output", Stage::BeforeReadout),
        ("readout.projection_input", Stage::ReadoutInput),
        ("model.logits", Stage::VocabularyScores),
    ] {
        let source = admission(&discovery, &[path], false, true);
        let selected = discovery
            .prepare_capture_selection(&source, paths.source())
            .unwrap();
        assert!(selected.source().same_storage(&source));
        assert!(selected.paths().same_storage(paths.source()));
        assert_eq!(
            selected.declaration(0).unwrap().unwrap().readout_stage(),
            stage
        );
        let mut geometry = request(4);
        geometry.prefill_chunk_positions = 2;
        geometry.output = selected.physical_output(OutputDemand::LastPosition);
        assert_eq!(
            geometry.output,
            if stage == Stage::BeforeReadout {
                OutputDemand::LastPosition
            } else {
                OutputDemand::Sequence
            }
        );
        let bound = selected.bind_geometry(geometry).unwrap();
        assert_eq!(bound.geometry(), geometry);
        if stage != Stage::BeforeReadout {
            geometry.output = OutputDemand::LastPosition;
            assert!(matches!(
                selected.bind_geometry(geometry),
                Err(SelectionError::Readout)
            ));
        }
    }
}

#[test]
fn qwen_actual_source_binding_preserves_physical_readout_requirements() {
    for (kind, tied) in [("qwen2", false), ("qwen3", true)] {
        check_binding::<qwen::QwenBlockFactory>(configuration(kind, tied));
    }
    for (kind, tied) in [("qwen2", false), ("qwen3", true), ("qwen3_moe", false)] {
        check_binding::<qwen::RoutedQwenBlockFactory>(configuration(kind, tied));
    }
}
