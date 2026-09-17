use super::*;
use eredu_architectures::lfm2;
use eredu_runtime::layered::PrefillObservationDeclaration;

type HybridState = DeviceState<NumericBackend, NumericHybridLayerState>;
type HybridModel = lfm2::LayeredModel<NumericBackend>;

// Every declared value, including effective companions, must be emitted by the
// real component/traversal observer. This collector never manufactures aliases.
struct DeclaredRows<'a> {
    declarations: &'a [PrefillObservationDeclaration],
    values: BTreeMap<String, NumericTensor>,
}
impl<'a> DeclaredRows<'a> {
    fn new(declarations: &'a [PrefillObservationDeclaration]) -> Self {
        Self {
            declarations,
            values: BTreeMap::new(),
        }
    }
    fn record(&mut self, path: &str, value: &NumericTensor) {
        if self.declarations.iter().any(|d| d.path() == path) {
            assert!(
                self.values.insert(path.into(), value.clone()).is_none(),
                "duplicate {path}"
            );
        }
    }
    fn assert_complete(&self, positions: i32) {
        assert_eq!(self.values.len(), self.declarations.len());
        for declaration in self.declarations {
            let path = declaration.path();
            let value = self
                .values
                .get(path)
                .unwrap_or_else(|| panic!("missing {path}"));
            assert_eq!(declaration.sequence_axis(), 1);
            assert_eq!(value.shape.len(), 3, "{path}");
            assert_eq!(value.shape[..2], [1, positions], "{path}");
            assert!(value.data.iter().all(|v| v.is_finite()), "finite {path}");
            assert!(value.data.iter().any(|v| v.abs() > 1e-6), "nonzero {path}");
        }
    }
}
impl ActivationObserver<NumericTensor, Error> for DeclaredRows<'_> {
    fn requires_sequence_readout(&self) -> bool {
        self.declarations
            .iter()
            .any(|d| d.readout_stage() != Stage::BeforeReadout)
    }
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.record(path, value);
        Ok(())
    }
    fn observe_generated(
        &mut self,
        path: &str,
        _: &NumericTensor,
        _: &GeneratedCaptureSource,
        _: &mut dyn FnMut() -> Result<NumericTensor, Error>,
    ) -> Result<(), Error> {
        assert!(self.declarations.iter().all(|d| d.path() != path));
        Ok(())
    }
}

fn assert_hybrid_state(actual: &HybridState, expected: &HybridState) {
    assert_eq!(actual.layout(), expected.layout());
    assert_eq!(actual.as_ref().len(), expected.as_ref().len());
    for (layer, (a, b)) in actual.as_ref().iter().zip(expected.as_ref()).enumerate() {
        assert_eq!(a.position(), b.position());
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        assert_eq!(
            a.fixed.keys().collect::<Vec<_>>(),
            b.fixed.keys().collect::<Vec<_>>()
        );
        for (role, a) in &a.fixed {
            let b = &b.fixed[role];
            assert_eq!(a.is_some(), b.is_some(), "layer {layer} {role:?}");
            if let (Some(a), Some(b)) = (a, b) {
                assert_tensor_close(a, b, &format!("layer {layer} {role:?}"));
            }
        }
        // These ordinary configurations have only the declared convolution fixed
        // slots or contiguous full KV storage; absence is checked explicitly.
        assert!(a.compressed.is_none() && b.compressed.is_none());
        assert!(a.pooling.is_none() && b.pooling.is_none());
        assert_eq!(a.attention.is_some(), b.attention.is_some());
        if let (Some(a), Some(b)) = (&a.attention, &b.attention) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.window, b.window);
            assert!(a.attention_history.is_none() && b.attention_history.is_none());
            for (name, a, b) in [("keys", &a.keys, &b.keys), ("values", &a.values, &b.values)] {
                assert_eq!(a.is_some(), b.is_some());
                if let (Some(a), Some(b)) = (a, b) {
                    assert_tensor_close(a, b, &format!("layer {layer} {name}"));
                }
            }
        }
    }
}

fn assert_populated(state: &HybridState, config: &lfm2::ModelArgs, positions: i32) {
    assert_eq!(state.as_ref().len(), config.layer_schedule.len());
    for (state, policy) in state.as_ref().iter().zip(config.layer_schedule.iter()) {
        assert!(state.compressed.is_none() && state.pooling.is_none());
        match policy.operator {
            lfm2::OperatorPolicy::CausalConvolution => {
                assert!(state.attention.is_none());
                if config.conv_l_cache == 1 {
                    assert!(state.fixed.is_empty());
                    assert_eq!(state.position(), 0);
                    assert_eq!(state.fixed_offset, 0);
                } else {
                    assert_eq!(state.position(), positions);
                    assert_eq!(state.fixed_offset, positions);
                    assert_eq!(state.fixed.len(), 1);
                    let history = state.fixed[&StateTensorRole::Convolution { slot: 0 }]
                        .as_ref()
                        .unwrap();
                    assert_eq!(
                        history.shape,
                        [1, config.conv_l_cache - 1, config.hidden_size]
                    );
                    assert!(history.data.iter().all(|v| v.is_finite()));
                    assert!(history.data.iter().any(|v| *v != 0.0));
                }
            }
            lfm2::OperatorPolicy::SelfAttention(_) => {
                assert_eq!(state.position(), positions);
                assert!(state.fixed.is_empty());
                assert_eq!(state.fixed_offset, 0);
                let attention = state.attention.as_ref().unwrap();
                assert_eq!(attention.offset, positions);
                assert_eq!(attention.window, None);
                assert!(attention.attention_history.is_none());
                for tensor in [&attention.keys, &attention.values] {
                    let tensor = tensor.as_ref().unwrap();
                    assert_eq!(
                        tensor.shape,
                        [
                            1,
                            config.num_key_value_heads,
                            positions,
                            config.hidden_size / config.num_attention_heads
                        ]
                    );
                    assert!(tensor.data.iter().all(|v| v.is_finite()));
                    assert!(tensor.data.iter().any(|v| *v != 0.0));
                }
            }
        }
    }
}

fn compare_equations(config: serde_json::Value) {
    let args = lfm2::model_args_from_config_value(&config).unwrap();
    let layout = lfm2::state_layout(&args).unwrap();
    let vocabulary = args.vocab_size;
    let context = NumericContext::default();
    let architecture = HybridModel::new(args.clone(), &context).unwrap();
    let declarations = <HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
        prefill_observation_declarations(&architecture).unwrap();
    assert_eq!(declarations.len(), 10 + 4 * args.layer_schedule.len());
    let mut model = ResidentRuntime::new(architecture, &context).unwrap();
    let paths = model.prepare_observation_paths().unwrap();
    for declaration in &declarations {
        assert_eq!(
            paths.source().prefill_observation(declaration.path()),
            Some(declaration)
        );
    }
    let mut full_state = HybridState::create(layout, |_, policy| {
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
    let prefix_state = full_state.clone();
    assert_populated(&prefix_state, &args, 2);
    let mut chunk_state = full_state.clone();
    let mut full = DeclaredRows::new(&declarations);
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
    let scores = eredu_runtime::observe_model_logits(&mut full, &scores.unwrap()).unwrap();
    assert_eq!(scores.shape, [1, 5, vocabulary]);
    full.assert_complete(5);
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    let mut consumed = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = DeclaredRows::new(&declarations);
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
        let scores = eredu_runtime::observe_model_logits(&mut rows, &scores.unwrap()).unwrap();
        assert_eq!(scores.shape, [1, ids.len() as i32, vocabulary]);
        rows.assert_complete(ids.len() as i32);
        consumed.extend_from_slice(ids);
        let mut direct_prefix = prefix_state.clone();
        model
            .forward(
                decoder::LayeredInput {
                    tokens: &NumericTensor::token_ids(&consumed),
                    mask: None,
                },
                &mut direct_prefix,
                &context,
            )
            .unwrap();
        assert_hybrid_state(&chunk_state, &direct_prefix);
        assert_populated(&chunk_state, &args, 2 + consumed.len() as i32);
        for declaration in &declarations {
            assembled
                .entry(declaration.path().into())
                .or_default()
                .extend_from_slice(&rows.values[declaration.path()].data);
        }
    }
    assert_eq!(assembled.len(), declarations.len());
    for declaration in &declarations {
        let path = declaration.path();
        let expected = &full.values[path];
        assert_tensor_close(
            &NumericTensor::new(expected.shape.clone(), assembled.remove(path).unwrap()),
            expected,
            path,
        );
    }
    assert_hybrid_state(&chunk_state, &full_state);
    for id in [7, 8, 9] {
        let tokens = NumericTensor::token_ids(&[id]);
        let mut expected = DeclaredRows::new(&declarations);
        let mut actual = DeclaredRows::new(&declarations);
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
            let scores = eredu_runtime::observe_model_logits(rows, &scores.unwrap()).unwrap();
            assert_eq!(scores.shape, [1, 1, vocabulary]);
            rows.assert_complete(1);
        }
        for declaration in &declarations {
            let path = declaration.path();
            assert_tensor_close(&actual.values[path], &expected.values[path], path);
        }
        assert_hybrid_state(&chunk_state, &full_state);
    }
    assert_populated(&full_state, &args, 10);
    assert_populated(&chunk_state, &args, 10);
}

fn configuration(routed: bool, tied: bool) -> serde_json::Value {
    let mut config = heterogeneous_replicated_configs()
        .into_iter()
        .find(|config| config["model_type"] == "lfm2")
        .unwrap();
    // A wider rotary pair and a third independently addressed convolution make
    // the continuation check cover frequency-dependent positions and unit paths.
    config["hidden_size"] = 16.into();
    config["num_hidden_layers"] = 3.into();
    config["layer_types"] = serde_json::json!(["conv", "full_attention", "conv"]);
    config["tie_word_embeddings"] = tied.into();
    if routed {
        config["model_type"] = "lfm2_moe".into();
        config["num_dense_layers"] = 1.into();
        config["num_experts"] = 4.into();
        config["num_experts_per_tok"] = 2.into();
        config["moe_intermediate_size"] = 6.into();
        config["routed_scaling_factor"] = 1.25.into();
        config["norm_topk_prob"] = true.into();
    }
    config
}

#[test]
fn lfm2_dense_rows_preserve_complete_convolution_and_kv_across_chunks() {
    for kernel in [1, 2, 3, 4] {
        for tied in [false, true] {
            let mut config = configuration(false, tied);
            config["conv_L_cache"] = kernel.into();
            config["conv_bias"] = tied.into();
            config["rope_parameters"] = serde_json::json!({
                "rope_type": "default", "rope_theta": 12345.0
            });
            compare_equations(config);
        }
    }
    for schedule in [
        serde_json::json!(["conv", "conv", "conv"]),
        serde_json::json!(["full_attention", "full_attention", "full_attention"]),
    ] {
        let mut config = configuration(false, false);
        config["layer_types"] = schedule;
        compare_equations(config);
    }
}

#[test]
fn lfm2_moe_rows_preserve_dense_and_sparse_equations_across_chunks() {
    for normalized in [false, true] {
        for tied in [false, true] {
            let mut config = configuration(true, tied);
            config["norm_topk_prob"] = normalized.into();
            config["use_expert_bias"] = tied.into();
            config["conv_L_cache"] = if tied { 1 } else { 3 }.into();
            config["conv_bias"] = tied.into();
            compare_equations(config);
        }
    }
}

#[test]
fn lfm2_actual_sources_bind_all_target_rows_and_original_physical_readout() {
    for routed in [false, true] {
        let config = configuration(routed, false);
        let (artifact, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
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
        let args = lfm2::model_args_from_config_value(&config).unwrap();
        let context = NumericContext::default();
        let architecture = HybridModel::new(args, &context).unwrap();
        let declarations = <HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
            prefill_observation_declarations(&architecture).unwrap();
        assert_eq!(declarations.len(), 22);
        let runtime =
            ResidentRuntime::<_, NumericBackend, HybridState>::new(architecture, &context).unwrap();
        let paths = runtime.prepare_observation_paths().unwrap();
        let before = sources.target().source_diagnostics().unwrap();
        for declaration in &declarations {
            for preview in [false, true] {
                let source = admission(&discovery, &[declaration.path()], preview, true);
                let selected = discovery
                    .prepare_capture_selection(&source, paths.source())
                    .unwrap();
                assert!(selected.source().same_storage(&source));
                assert!(selected.paths().same_storage(paths.source()));
                assert_eq!(selected.declaration(0).unwrap(), Some(declaration));
                let mut geometry = request(4);
                geometry.prefill_chunk_positions = 2;
                geometry.output = selected.physical_output(OutputDemand::LastPosition);
                let expected = match declaration.readout_stage() {
                    Stage::BeforeReadout => OutputDemand::LastPosition,
                    Stage::ReadoutInput | Stage::VocabularyScores => OutputDemand::Sequence,
                };
                assert_eq!(geometry.output, expected);
                let bound = selected.bind_geometry(geometry).unwrap();
                assert_eq!(bound.geometry(), geometry);
                assert!(std::ptr::eq(bound.selection(), &selected));
                if expected == OutputDemand::Sequence {
                    geometry.output = OutputDemand::LastPosition;
                    assert!(matches!(
                        selected.bind_geometry(geometry),
                        Err(SelectionError::Readout)
                    ));
                }
                let independent = admission(&discovery, &[declaration.path()], preview, true);
                assert!(matches!(
                    selected.validate_sources(&independent, paths.source()),
                    Err(SelectionError::Identity)
                ));
            }
        }
        // Actual internal hook semantics are not inferred from their axes.
        let internal = "model.layers.1.attention.channels";
        assert!(paths.source().prefill_observation(internal).is_none());
        let source = admission(&discovery, &[internal], false, true);
        assert!(matches!(
            discovery.prepare_capture_selection(&source, paths.source()),
            Err(SelectionError::Undeclared { index: 0 })
        ));
        let after = sources.target().source_diagnostics().unwrap();
        assert_eq!(before.physical_reads, after.physical_reads);
    }
}

#[test]
fn lfm2_body_rows_preserve_sequence_before_state_only_or_last_position_readout() {
    for routed in [false, true] {
        let args = lfm2::model_args_from_config_value(&configuration(routed, false)).unwrap();
        let context = NumericContext::default();
        let architecture = HybridModel::new(args.clone(), &context).unwrap();
        let declarations = <HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
            prefill_observation_declarations(&architecture).unwrap()
            .into_iter().filter(|d| d.readout_stage() == Stage::BeforeReadout)
            .collect::<Vec<_>>();
        assert_eq!(declarations.len(), 14);
        let mut model = ResidentRuntime::new(architecture, &context).unwrap();
        let paths = model.prepare_observation_paths().unwrap();
        let mut prefix = HybridState::create(lfm2::state_layout(&args).unwrap(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap();
        model
            .forward(
                decoder::LayeredInput {
                    tokens: &NumericTensor::token_ids(&[4, 2]),
                    mask: None,
                },
                &mut prefix,
                &context,
            )
            .unwrap();
        let tokens = NumericTensor::token_ids(&[1, 3, 5]);
        let mut expected_state = prefix.clone();
        let mut expected = DeclaredRows::new(&declarations);
        let (full, _) = model
            .forward_with_prepared_observer_and_context_with_readout(
                decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                &mut expected_state,
                &context,
                &mut expected,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
        let full = full.unwrap();
        expected.assert_complete(3);
        for demand in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
            let mut state = prefix.clone();
            let mut actual = DeclaredRows::new(&declarations);
            assert!(!actual.requires_sequence_readout());
            let (output, _) = model
                .forward_with_prepared_observer_and_context_with_readout(
                    decoder::LayeredInput {
                        tokens: &tokens,
                        mask: None,
                    },
                    &mut state,
                    &context,
                    &mut actual,
                    &paths,
                    demand,
                )
                .unwrap();
            actual.assert_complete(3);
            assert_hybrid_state(&state, &expected_state);
            for declaration in &declarations {
                let path = declaration.path();
                assert_tensor_close(&actual.values[path], &expected.values[path], path);
            }
            if demand == OutputDemand::StateOnly {
                assert!(output.is_none());
            } else {
                let last = full
                    .index(&[Index::Full, Index::Range(2, 3), Index::Full], &context)
                    .unwrap();
                assert_tensor_close(&output.unwrap(), &last, "last prediction");
            }
        }
    }
}

#[test]
fn lfm2_nonzero_history_still_requires_the_declared_state_component() {
    let args = lfm2::model_args_from_config_value(&configuration(false, false)).unwrap();
    assert_eq!(args.conv_l_cache, 3);
    let context = NumericContext::default();
    let mut block = lfm2::Block::<NumericBackend>::new(&args, 0, &context).unwrap();
    let mut state = NumericHybridLayerState::new(&LayerCachePolicy::NoState);
    let hidden = NumericTensor::new(
        vec![1, 2, args.hidden_size],
        (0..2 * args.hidden_size)
            .map(|i| 0.1 + i as f32 * 0.01)
            .collect(),
    );
    let error = block
        .forward(&hidden, None, &mut state, &context)
        .err()
        .expect("missing history");
    // Preserve this established neutral error bridge: Error::backend carries
    // the exact StateError diagnostic but does not retain a source object.
    let expected = StateError::UnknownComponent {
        role: StateTensorRole::Convolution { slot: 0 },
    };
    assert_eq!(error.to_string(), expected.to_string());
    assert!(std::error::Error::source(&error).is_none());
    assert!(state.fixed.is_empty());
    assert_eq!(state.position(), 0);
}

#[path = "lfm2_rows/width_one_partition.rs"]
mod width_one_partition;
