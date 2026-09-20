use super::*;
use eredu_architectures::nemotron_h;
use eredu_runtime::layered::PrefillObservationDeclaration;

type HybridState = DeviceState<NumericBackend, NumericHybridLayerState>;
type HybridModel = nemotron_h::LayeredModel<NumericBackend>;

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
        assert_hybrid_layer(a, b, layer);
    }
}

fn assert_hybrid_layer(a: &NumericHybridLayerState, b: &NumericHybridLayerState, layer: usize) {
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
    // These ordinary configurations have only the declared convolution/recurrent fixed
    // slots or contiguous full/sliding KV storage; absence is checked explicitly.
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

fn assert_populated(state: &HybridState, config: &nemotron_h::ModelArgs, positions: i32) {
    assert_eq!(
        state.as_ref().len(),
        config.layer_schedule.len() + config.mtp_policies().unwrap().len()
    );
    for (state, policy) in state.as_ref().iter().zip(config.layer_schedule.iter()) {
        assert!(state.compressed.is_none() && state.pooling.is_none());
        match policy {
            nemotron_h::LayerPolicy::Mamba => {
                assert!(state.attention.is_none());
                assert_eq!(state.position(), positions);
                assert_eq!(state.fixed_offset, positions);
                assert_eq!(
                    state.fixed.len(),
                    if config.conv_kernel == 1 { 1 } else { 2 }
                );
                if config.conv_kernel == 1 {
                    assert!(!state
                        .fixed
                        .contains_key(&StateTensorRole::Convolution { slot: 0 }));
                } else {
                    let history = state.fixed[&StateTensorRole::Convolution { slot: 0 }]
                        .as_ref()
                        .unwrap();
                    assert_eq!(
                        history.shape,
                        [
                            1,
                            config.conv_kernel - 1,
                            config.mamba_num_heads * config.mamba_head_dim
                                + 2 * config.n_groups * config.ssm_state_size
                        ]
                    );
                    assert_nonzero(history);
                }
                let recurrent = state.fixed[&StateTensorRole::Recurrent].as_ref().unwrap();
                assert_eq!(
                    recurrent.shape,
                    [
                        1,
                        config.mamba_num_heads,
                        config.mamba_head_dim,
                        config.ssm_state_size
                    ]
                );
                assert_nonzero(recurrent);
            }
            nemotron_h::LayerPolicy::SelfAttention(policy) => {
                assert!(state.fixed.is_empty());
                assert_eq!(state.fixed_offset, 0);
                let attention = state.attention.as_ref().unwrap();
                assert_eq!(attention.offset, positions);
                assert_eq!(attention.window, policy.sliding_window_i32().unwrap());
                assert!(attention.attention_history.is_none());
                let retained = attention
                    .window
                    .map_or(positions, |window| positions.min(window));
                for tensor in [&attention.keys, &attention.values] {
                    let tensor = tensor.as_ref().unwrap();
                    assert_eq!(
                        tensor.shape,
                        [1, config.num_key_value_heads, retained, config.head_dim]
                    );
                    assert_nonzero(tensor);
                }
            }
            nemotron_h::LayerPolicy::DenseMlp | nemotron_h::LayerPolicy::SparseMoe => {
                assert!(state.fixed.is_empty() && state.attention.is_none());
                assert_eq!(state.position(), 0);
                assert_eq!(state.fixed_offset, 0);
            }
        }
    }
    // A target invocation never consumes the separate prediction state.
    for state in &state.as_ref()[config.layer_schedule.len()..] {
        assert_eq!(state.position(), 0);
        assert_eq!(state.fixed_offset, 0);
        assert!(state.fixed.values().all(Option::is_none));
        if let Some(attention) = &state.attention {
            assert!(attention.keys.is_none() && attention.values.is_none());
            assert!(attention.attention_history.is_none());
        }
    }
}

fn assert_nonzero(value: &NumericTensor) {
    assert!(value.data.iter().all(|v| v.is_finite()));
    assert!(value.data.iter().any(|v| v.abs() > 1e-9));
}

// Factory-created linear/embedding weights are already deterministic. Raw
// Parameter::unloaded slots use NumericTensor::unloaded_f32 (all zeros), so
// load the actual convolution/scan controls before any state or path token.
// Mutable state itself is never seeded or replaced by this fixture.
fn load_recurrent_parameters(
    model: &mut ResidentRuntime<HybridModel, NumericBackend, HybridState>,
    expected_parameters: usize,
) {
    struct Load {
        visited: BTreeMap<String, ()>,
    }
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Load {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            let Some((_, field)) = metadata.id().as_str().rsplit_once(".mamba.") else {
                return;
            };
            let (base, step) = match field {
                "conv1d.weight" => (0.15, 0.01),
                "dt_bias" => (-1.0, 0.01),
                "A_log" => (-0.6, 0.01),
                "norm.weight" => (1.0, 0.001),
                "D" => (0.4, 0.01),
                "conv1d.bias" => (0.02, 0.001),
                _ => return, // Existing deterministic factory parameters stay intact.
            };
            assert!(self
                .visited
                .insert(metadata.id().as_str().into(), ())
                .is_none());
            assert!(
                value.data.iter().all(|value| *value == 0.0),
                "unloaded {}",
                metadata.id().as_str()
            );
            for (index, value) in value.data.iter_mut().enumerate() {
                *value = base + step * (index % 7) as f32;
            }
        }
    }
    let mut loader = Load {
        visited: BTreeMap::new(),
    };
    for unit in model.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut loader);
    }
    assert_eq!(loader.visited.len(), expected_parameters);
}

fn compare_equations(config: serde_json::Value) {
    let args = nemotron_h::model_args_from_config_value(&config).unwrap();
    let layout = nemotron_h::state_layout(&args).unwrap();
    let vocabulary = args.vocab_size;
    let context = NumericContext::default();
    let architecture = HybridModel::new(args.clone(), &context).unwrap();
    let declarations = tensor_row_declarations(<HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
        prefill_observation_declarations(&architecture, None).unwrap());
    assert!(declarations.len() >= 10 + 4 * args.layer_schedule.len());
    let mut model = ResidentRuntime::new(architecture, &context).unwrap();
    let mamba_units = args
        .layer_schedule
        .iter()
        .filter(|policy| matches!(policy, nemotron_h::LayerPolicy::Mamba))
        .count();
    load_recurrent_parameters(
        &mut model,
        mamba_units * (5 + usize::from(args.use_conv_bias)),
    );
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
            nemotron_h::EmbeddedInput::target(&NumericTensor::token_ids(&[4, 2]), None),
            &mut full_state,
            &context,
        )
        .unwrap();
    let prefix_state = full_state.clone();
    assert_populated(&prefix_state, &args, 2);
    let mut chunk_state = full_state.clone();
    let mut full = DeclaredRows::new(&declarations);
    let (scores, forward) = model
        .forward_with_prepared_observer_and_context_with_readout(
            nemotron_h::EmbeddedInput::target(&NumericTensor::token_ids(&[1, 3, 5, 2, 6]), None),
            &mut full_state,
            &context,
            &mut full,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
    assert_target_context(&forward, &full);
    let scores = eredu_runtime::observe_model_logits(&mut full, &scores.unwrap()).unwrap();
    assert_eq!(scores.shape, [1, 5, vocabulary]);
    full.assert_complete(5);
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    let mut consumed = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = DeclaredRows::new(&declarations);
        let (scores, forward) = model
            .forward_with_prepared_observer_and_context_with_readout(
                nemotron_h::EmbeddedInput::target(&NumericTensor::token_ids(ids), None),
                &mut chunk_state,
                &context,
                &mut rows,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
        assert_target_context(&forward, &rows);
        let scores = eredu_runtime::observe_model_logits(&mut rows, &scores.unwrap()).unwrap();
        assert_eq!(scores.shape, [1, ids.len() as i32, vocabulary]);
        rows.assert_complete(ids.len() as i32);
        consumed.extend_from_slice(ids);
        let mut direct_prefix = prefix_state.clone();
        model
            .forward(
                nemotron_h::EmbeddedInput::target(&NumericTensor::token_ids(&consumed), None),
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
            let (scores, forward) = model
                .forward_with_prepared_observer_and_context_with_readout(
                    nemotron_h::EmbeddedInput::target(&tokens, None),
                    state,
                    &context,
                    rows,
                    &paths,
                    OutputDemand::Sequence,
                )
                .unwrap();
            assert_target_context(&forward, rows);
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
}
fn assert_target_context(
    forward: &nemotron_h::ForwardContext<NumericTensor>,
    rows: &DeclaredRows<'_>,
) {
    assert!(matches!(forward.mode(), nemotron_h::ForwardMode::Target));
    assert!(forward.draft_logits().is_none());
    let last = rows
        .declarations
        .iter()
        .filter(|d| d.readout_stage() == Stage::BeforeReadout)
        .last()
        .unwrap()
        .path();
    assert_tensor_close(
        forward.target_capture().unwrap(),
        &rows.values[last],
        "target hidden before readout",
    );
}

fn configuration(pattern: &str, kernel: i32, sliding: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": "nemotron_h", "vocab_size": 32, "hidden_size": 16,
        "intermediate_size": 24, "num_hidden_layers": pattern.len(),
        "hybrid_override_pattern": pattern, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 4, "mamba_num_heads": 4,
        "n_groups": 2, "mamba_head_dim": 4, "ssm_state_size": 3,
        "conv_kernel": kernel, "chunk_size": 2, "n_routed_experts": 4,
        "n_shared_experts": 1, "moe_intermediate_size": 8,
        "moe_shared_expert_intermediate_size": 8, "num_experts_per_tok": 2,
        "n_group": 2, "topk_group": 1, "num_nextn_predict_layers": 0,
        "tie_word_embeddings": false, "residual_in_fp32": true
    });
    if sliding {
        config["sliding_window"] = 3.into();
    }
    config
}

#[test]
fn nemotron_rows_preserve_complete_mamba_and_full_or_sliding_kv_across_chunks() {
    for kernel in [1, 3, 4] {
        for sliding in [false, true] {
            let mut config = configuration("M*-E", kernel, sliding);
            config["tie_word_embeddings"] = sliding.into();
            config["norm_topk_prob"] = sliding.into();
            config["use_bias"] = sliding.into();
            config["attention_bias"] = sliding.into();
            config["mlp_bias"] = sliding.into();
            config["use_conv_bias"] = (!sliding).into();
            config["chunk_size"] = if sliding { 3 } else { 2 }.into();
            compare_equations(config);
        }
    }
}

#[test]
fn nemotron_rows_use_stateful_frontiers_after_stateless_units_and_in_homogeneous_schedules() {
    for pattern in ["-M*E", "E*-M", "--M*", "MMMM", "****", "----", "EEEE"] {
        let mut config = configuration(pattern, 3, pattern == "****");
        config["n_groups"] = 1.into();
        config["residual_in_fp32"] = false.into();
        compare_equations(config);
    }
}

#[test]
fn nemotron_actual_sources_bind_target_rows_and_original_physical_readout() {
    for pattern in ["M*-E", "-M*E"] {
        let config = configuration(pattern, 3, false);
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
        let args = nemotron_h::model_args_from_config_value(&config).unwrap();
        let attention_index = args
            .layer_schedule
            .iter()
            .position(|p| matches!(p, nemotron_h::LayerPolicy::SelfAttention(_)))
            .unwrap();
        let context = NumericContext::default();
        let architecture = HybridModel::new(args, &context).unwrap();
        let declarations = tensor_row_declarations(<HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
            prefill_observation_declarations(&architecture, None).unwrap());
        assert!(declarations.len() >= 26);
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
        let internal = format!("model.layers.{attention_index}.attention.channels");
        assert!(paths.source().prefill_observation(&internal).is_none());
        let source = admission(&discovery, &[&internal], false, true);
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
fn nemotron_target_rows_preserve_mtp_scope_and_state() {
    use eredu_runtime::inspection::ObservationHookSite;
    let mut config = configuration("-M*E", 1, false);
    config["num_nextn_predict_layers"] = 1.into();
    config["mtp_hybrid_override_pattern"] = "E*".into();
    // Target numerical invariance must include all untouched prediction slots.
    compare_equations(config.clone());
    let args = nemotron_h::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let architecture = HybridModel::new(args.clone(), &context).unwrap();
    let declarations = tensor_row_declarations(<HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
        prefill_observation_declarations(&architecture, None).unwrap());
    assert!(declarations.len() >= 26);
    assert!(declarations
        .iter()
        .all(|d| !d.path().starts_with("model.mtp.")));
    let hooks =
        <HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::observation_hooks(
            &architecture,
        );
    for site in [
        ObservationHookSite::Input,
        ObservationHookSite::Unit,
        ObservationHookSite::RoutedUnits,
        ObservationHookSite::Readout,
    ] {
        assert!(!hooks.supports(site), "existing support gate unchanged");
    }
    let tokens = NumericTensor::token_ids(&[11]);
    let prior = NumericTensor::new(
        vec![1, 1, args.hidden_size],
        (0..args.hidden_size)
            .map(|i| 0.1 + i as f32 * 0.01)
            .collect(),
    );
    assert_eq!(
        <HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::inference_input_shape(
            &nemotron_h::EmbeddedInput::draft(&tokens, &prior, 0)
        )
        .unwrap(),
        None
    );
    let mut runtime = ResidentRuntime::new(architecture, &context).unwrap();
    let mut state = HybridState::create(nemotron_h::state_layout(&args).unwrap(), |_, p| {
        Ok::<_, Error>(NumericHybridLayerState::new(p))
    })
    .unwrap();
    runtime
        .forward(
            nemotron_h::EmbeddedInput::target(&NumericTensor::token_ids(&[1, 2]), None),
            &mut state,
            &context,
        )
        .unwrap();
    let target = state.clone();
    let output = runtime
        .forward(
            nemotron_h::EmbeddedInput::draft(&tokens, &prior, 0),
            &mut state,
            &context,
        )
        .unwrap();
    assert_eq!(output.shape, [1, 1, args.vocab_size]);
    assert_nonzero(&output);
    for (layer, (actual, expected)) in state.as_ref()[..4]
        .iter()
        .zip(&target.as_ref()[..4])
        .enumerate()
    {
        assert_hybrid_layer(actual, expected, layer);
    }
    assert_eq!(state.as_ref()[4].position(), 0);
    assert_eq!(state.as_ref()[5].position(), 1);
    let draft_prefix = state.clone();
    let continuation_tokens = NumericTensor::token_ids(&[12, 13]);
    let continuation_hidden = NumericTensor::new(
        vec![1, 2, args.hidden_size],
        (0..2 * args.hidden_size)
            .map(|i| 0.2 + i as f32 * 0.01)
            .collect(),
    );
    let full = runtime
        .forward(
            nemotron_h::EmbeddedInput::draft(&continuation_tokens, &continuation_hidden, 0),
            &mut state,
            &context,
        )
        .unwrap();
    let mut split = draft_prefix.clone();
    let mut pieces = Vec::new();
    for (index, token) in [12, 13].into_iter().enumerate() {
        let tokens = NumericTensor::token_ids(&[token]);
        let hidden = continuation_hidden.axis_slice(1, index, index + 1);
        pieces.push(
            runtime
                .forward(
                    nemotron_h::EmbeddedInput::draft(&tokens, &hidden, 0),
                    &mut split,
                    &context,
                )
                .unwrap(),
        );
        let mut direct = draft_prefix.clone();
        runtime
            .forward(
                nemotron_h::EmbeddedInput::draft(
                    &continuation_tokens.axis_slice(1, 0, index + 1),
                    &continuation_hidden.axis_slice(1, 0, index + 1),
                    0,
                ),
                &mut direct,
                &context,
            )
            .unwrap();
        assert_hybrid_state(&split, &direct);
    }
    assert_tensor_close(
        &NumericTensor::concatenate(&pieces, 1, &context).unwrap(),
        &full,
        "MTP own cached frontier",
    );
    assert_hybrid_state(&split, &state);
    assert_eq!(state.as_ref()[5].position(), 3);
}

#[test]
fn nemotron_wider_convolution_preserves_missing_history_error() {
    let args = nemotron_h::model_args_from_config_value(&configuration("M*-E", 3, false)).unwrap();
    let context = NumericContext::default();
    let mut mixer = nemotron_h::Mamba2::<NumericBackend>::new(&args, 0, &context).unwrap();
    let layout = nemotron_h::state_layout(&args).unwrap();
    let mut state = NumericHybridLayerState::new(layout.layer(0).unwrap());
    state
        .fixed
        .remove(&StateTensorRole::Convolution { slot: 0 });
    let input = NumericTensor::new(
        vec![1, 2, args.hidden_size],
        (0..2 * args.hidden_size)
            .map(|i| 0.1 + i as f32 * 0.01)
            .collect(),
    );
    let error = mixer
        .forward(&input, &mut state, &context)
        .err()
        .expect("missing history");
    let expected = StateError::UnknownComponent {
        role: StateTensorRole::Convolution { slot: 0 },
    };
    assert_eq!(error.to_string(), expected.to_string());
    assert!(std::error::Error::source(&error).is_none());
    assert_eq!(state.position(), 0);
    assert_eq!(state.fixed.len(), 1);
    assert!(state.fixed[&StateTensorRole::Recurrent].is_none());
}

#[test]
fn nemotron_body_rows_preserve_sequence_before_state_only_or_last_position_readout() {
    for sliding in [false, true] {
        let args =
            nemotron_h::model_args_from_config_value(&configuration("-M*E", 1, sliding)).unwrap();
        let context = NumericContext::default();
        let architecture = HybridModel::new(args.clone(), &context).unwrap();
        let declarations = tensor_row_declarations(<HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::
            prefill_observation_declarations(&architecture, None).unwrap())
            .into_iter().filter(|d| d.readout_stage() == Stage::BeforeReadout)
            .collect::<Vec<_>>();
        assert!(declarations.len() >= 18);
        let mut model = ResidentRuntime::new(architecture, &context).unwrap();
        let paths = model.prepare_observation_paths().unwrap();
        let mut prefix =
            HybridState::create(nemotron_h::state_layout(&args).unwrap(), |_, policy| {
                Ok::<_, Error>(NumericHybridLayerState::new(policy))
            })
            .unwrap();
        model
            .forward(
                nemotron_h::EmbeddedInput::target(&NumericTensor::token_ids(&[4, 2]), None),
                &mut prefix,
                &context,
            )
            .unwrap();
        let tokens = NumericTensor::token_ids(&[1, 3, 5]);
        let mut expected_state = prefix.clone();
        let mut expected = DeclaredRows::new(&declarations);
        let (full, _) = model
            .forward_with_prepared_observer_and_context_with_readout(
                nemotron_h::EmbeddedInput::target(&tokens, None),
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
                    nemotron_h::EmbeddedInput::target(&tokens, None),
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
fn nemotron_embedded_tp1_target_and_draft_use_their_own_attention_frontiers() {
    let mut config = configuration("-M*E", 1, false);
    config["num_nextn_predict_layers"] = 1.into();
    config["mtp_hybrid_override_pattern"] = "E*".into();
    let args = nemotron_h::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let architecture = HybridModel::new(args.clone(), &context).unwrap();
    let description = architecture.parameter_description(&context).unwrap().into_owned();
    let groups = description
        .groups()
        .iter()
        .map(|owned| owned.group().clone())
        .collect::<Vec<_>>();
    let addresses = (0..description.unit_layout().len())
        .map(|ordinal| {
            let address = description.unit_layout().address(ordinal).unwrap();
            (address.group(), address.index())
        })
        .collect::<Vec<_>>();
    assert_eq!(addresses, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]);
    let mut ordinary_state =
        HybridState::create(architecture.state_layout().unwrap(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap();
    let mut ordinary = ResidentRuntime::new(architecture, &context).unwrap();
    let local = numeric_local_layout(&groups, 1, 0).unwrap();
    let parallel_context = NumericContext::with_local_layout(local.clone());
    let geometry = nemotron_h::local_geometry(&args, &local).unwrap();
    let local_state = geometry.state_layout().clone();
    let architecture =
        HybridModel::new_parallel(args.clone(), geometry, &parallel_context).unwrap();
    let units = addresses
        .into_iter()
        .map(|(group, index)| {
            <HybridModel as LayeredArchitecture<NumericBackend, HybridState>>::build_unit(
                &architecture,
                group,
                index,
                &parallel_context,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut parallel = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let mut parallel_state = HybridState::create(local_state, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let group = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let mut last_hidden = None;
    for ids in [
        &[4, 2][..],
        &[1, 3][..],
        &[5][..],
        &[2, 6][..],
        &[7][..],
        &[8][..],
        &[9][..],
    ] {
        let tokens = NumericTensor::token_ids(ids);
        let (expected, expected_forward) = ordinary
            .forward_with_context(
                nemotron_h::EmbeddedInput::target(&tokens, None),
                &mut ordinary_state,
                &context,
            )
            .unwrap();
        let (actual, actual_forward) = parallel
            .forward_parallel_with_context_hook(
                nemotron_h::EmbeddedInput::target(&tokens, None),
                &mut parallel_state,
                &group,
                &parallel_context,
                |_, _, _| Ok(()),
            )
            .unwrap();
        assert_tensor_close(&actual, &expected, "TP1 target scores");
        assert_tensor_close(
            actual_forward.target_capture().unwrap(),
            expected_forward.target_capture().unwrap(),
            "TP1 target hidden",
        );
        assert_hybrid_state(&parallel_state, &ordinary_state);
        last_hidden = expected_forward.target_capture().cloned();
    }
    let hidden = last_hidden.unwrap();
    for ids in [&[11][..], &[12, 13][..], &[14][..]] {
        let tokens = NumericTensor::token_ids(ids);
        let prior = hidden
            .broadcast_to(&[1, ids.len() as i32, args.hidden_size], &context)
            .unwrap();
        let expected = ordinary
            .forward(
                nemotron_h::EmbeddedInput::draft(&tokens, &prior, 0),
                &mut ordinary_state,
                &context,
            )
            .unwrap();
        let actual = parallel
            .forward_parallel(
                nemotron_h::EmbeddedInput::draft(&tokens, &prior, 0),
                &mut parallel_state,
                &group,
                &parallel_context,
            )
            .unwrap();
        assert_tensor_close(&actual, &expected, "TP1 draft scores");
        assert_hybrid_state(&parallel_state, &ordinary_state);
    }
    assert_eq!(ordinary_state.as_ref()[0].position(), 0);
    assert_eq!(ordinary_state.as_ref()[2].position(), 10);
    assert_eq!(ordinary_state.as_ref()[4].position(), 0);
    assert_eq!(ordinary_state.as_ref()[5].position(), 4);
}

#[path = "nemotron_rows/partition.rs"]
mod partition;
