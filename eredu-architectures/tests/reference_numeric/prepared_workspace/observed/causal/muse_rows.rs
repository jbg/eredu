use super::*;
use eredu_architectures::composite_execution::{
    CompositeArchitecture, PreparedCompositeArchitecture, PreparedCompositeInput,
};
use eredu_architectures::muse_glimmer;
use eredu_runtime::layered::{PrefillObservationDeclaration, PreparedLayeredObservationPaths};

type MuseState = DeviceState<NumericBackend, NumericHybridLayerState>;
type MuseModel = muse_glimmer::LayeredModel<NumericBackend>;
type PreparedMuse = PreparedCompositeArchitecture<MuseModel>;
type Model = ResidentRuntime<PreparedMuse, NumericBackend, MuseState>;

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
        assert!(
            !path.starts_with("model.vision_tower."),
            "text ran vision hook {path}"
        );
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

fn assert_state(actual: &MuseState, expected: &MuseState) {
    assert_eq!(actual.layout(), expected.layout());
    assert_eq!(actual.as_ref().len(), expected.as_ref().len());
    for (layer, (a, b)) in actual.as_ref().iter().zip(expected.as_ref()).enumerate() {
        assert_layer(a, b, layer);
    }
}

fn assert_layer(a: &NumericHybridLayerState, b: &NumericHybridLayerState, layer: usize) {
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
    // These ordinary text states contain only full/sliding KV storage; every
    // other mutable representation is compared and its absence checked explicitly.
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

fn assert_populated(state: &MuseState, config: &muse_glimmer::DecoderConfig, positions: i32) {
    assert_eq!(state.as_ref().len(), config.num_hidden_layers as usize);
    for (layer, policy) in state.as_ref().iter().zip(config.attention_schedule.iter()) {
        assert!(layer.fixed.is_empty());
        assert_eq!(layer.fixed_offset, 0);
        assert!(layer.compressed.is_none() && layer.pooling.is_none());
        let attention = layer.attention.as_ref().unwrap();
        assert_eq!(attention.offset, positions);
        assert_eq!(attention.window, policy.sliding_window_i32().unwrap());
        assert!(attention.attention_history.is_none());
        let retained = attention.window.map_or(positions, |w| positions.min(w));
        for tensor in [&attention.keys, &attention.values] {
            let tensor = tensor.as_ref().unwrap();
            assert_eq!(
                tensor.shape,
                [1, config.num_key_value_heads, retained, config.head_dim]
            );
            assert!(tensor.data.iter().all(|v| v.is_finite()));
            assert!(tensor.data.iter().any(|v| v.abs() > 1e-9));
        }
    }
}

fn make_state(config: &muse_glimmer::DecoderConfig) -> MuseState {
    MuseState::create(muse_glimmer::state_layout(config).unwrap(), |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap()
}

fn make_model(
    config: &muse_glimmer::DecoderConfig,
    context: &NumericContext,
) -> (
    Model,
    Vec<PrefillObservationDeclaration>,
    PreparedLayeredObservationPaths,
) {
    let architecture = MuseModel::new(config.clone(), context).unwrap();
    let declarations = <MuseModel as LayeredArchitecture<NumericBackend, MuseState>>::
        prefill_observation_declarations(&architecture, None).unwrap();
    assert_eq!(
        declarations.len(),
        10 + 4 * config.num_hidden_layers as usize
    );
    assert_eq!(
        <MuseModel as LayeredArchitecture<NumericBackend, MuseState>>::group_unit_count(
            &architecture,
            0, None)
        .unwrap(),
        1
    );
    assert!(declarations
        .iter()
        .all(|d| !d.path().starts_with("model.vision_tower.")));
    for index in 0..config.num_hidden_layers as usize {
        let path = <MuseModel as LayeredArchitecture<NumericBackend, MuseState>>::unit_path(
            &architecture,
            1,
            index, None)
        .unwrap();
        for suffix in ["input", "input.effective", "output", "output.effective"] {
            assert!(declarations
                .iter()
                .any(|d| d.path() == format!("{path}.{suffix}")));
        }
    }
    let prepared = PreparedCompositeArchitecture::new(architecture);
    assert_eq!(<PreparedMuse as LayeredArchitecture<NumericBackend, MuseState>>::
        prefill_observation_declarations(&prepared, None).unwrap(), declarations);
    let runtime = ResidentRuntime::new(prepared, context).unwrap();
    let paths = runtime.prepare_observation_paths().unwrap();
    for declaration in &declarations {
        assert_eq!(
            paths.source().prefill_observation(declaration.path()),
            Some(declaration)
        );
    }
    (runtime, declarations, paths)
}

#[allow(clippy::too_many_arguments)]
fn run(
    model: &mut Model,
    config: &muse_glimmer::DecoderConfig,
    ids: &[usize],
    state: &mut MuseState,
    context: &NumericContext,
    rows: &mut DeclaredRows<'_>,
    paths: &PreparedLayeredObservationPaths,
    demand: OutputDemand,
) -> Option<NumericTensor> {
    let part = eredu_runtime::PreparedInputPart::new(
        eredu_core::InputModality::Text,
        eredu_runtime::PreparedInputPayload::TokenIds(NumericTensor::token_ids(ids)),
        [],
    )
    .unwrap();
    let input = eredu_runtime::PreparedModelInput::new(vec![part], |tensor| {
        eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
    })
    .unwrap();
    let admitted =
        <MuseModel as CompositeArchitecture<NumericBackend, MuseState>>::admit_prepared_input(
            config,
            &input,
            &NumericInputInspector,
        )
        .unwrap();
    let paired = PreparedCompositeInput::new(&input, &admitted).unwrap();
    assert_eq!(
        <PreparedMuse as LayeredArchitecture<NumericBackend, MuseState>>::inference_input_shape(
            &paired
        )
        .unwrap(),
        Some([1, ids.len() as u64])
    );
    let (scores, _) = model
        .forward_with_prepared_observer_and_context_with_readout(
            paired, state, context, rows, paths, demand,
        )
        .unwrap();
    // The enclosing runtime owns this publication, after the model's real softcap.
    scores.map(|scores| eredu_runtime::observe_model_logits(rows, &scores).unwrap())
}

fn compare_equations(value: serde_json::Value) {
    let config = muse_glimmer::DecoderConfig::from_hf_value(&value).unwrap();
    let context = NumericContext::default();
    let (mut model, declarations, paths) = make_model(&config, &context);
    let mut full_state = make_state(&config);
    run(
        &mut model,
        &config,
        &[4, 2],
        &mut full_state,
        &context,
        &mut DeclaredRows::new(&[]),
        &paths,
        OutputDemand::Sequence,
    )
    .unwrap();
    let prefix = full_state.clone();
    assert_populated(&prefix, &config, 2);
    let mut split_state = prefix.clone();
    let mut full = DeclaredRows::new(&declarations);
    let scores = run(
        &mut model,
        &config,
        &[1, 3, 5, 2, 6],
        &mut full_state,
        &context,
        &mut full,
        &paths,
        OutputDemand::Sequence,
    )
    .unwrap();
    assert_eq!(scores.shape, [1, 5, config.vocab_size]);
    full.assert_complete(5);
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    let mut consumed = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = DeclaredRows::new(&declarations);
        let scores = run(
            &mut model,
            &config,
            ids,
            &mut split_state,
            &context,
            &mut rows,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        assert_eq!(scores.shape, [1, ids.len() as i32, config.vocab_size]);
        rows.assert_complete(ids.len() as i32);
        consumed.extend_from_slice(ids);
        let mut direct = prefix.clone();
        run(
            &mut model,
            &config,
            &consumed,
            &mut direct,
            &context,
            &mut DeclaredRows::new(&[]),
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        assert_state(&split_state, &direct);
        assert_populated(&split_state, &config, 2 + consumed.len() as i32);
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
    assert_state(&split_state, &full_state);
    for id in [7, 8, 9] {
        let mut expected = DeclaredRows::new(&declarations);
        let mut actual = DeclaredRows::new(&declarations);
        for (state, rows) in [
            (&mut full_state, &mut expected),
            (&mut split_state, &mut actual),
        ] {
            let scores = run(
                &mut model,
                &config,
                &[id],
                state,
                &context,
                rows,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
            assert_eq!(scores.shape, [1, 1, config.vocab_size]);
            rows.assert_complete(1);
        }
        for declaration in &declarations {
            let path = declaration.path();
            assert_tensor_close(&actual.values[path], &expected.values[path], path);
        }
        assert_state(&split_state, &full_state);
    }
    assert_populated(&full_state, &config, 10);
}

fn configuration(routed: bool, scaling: &str, tied: bool) -> serde_json::Value {
    let mut value = serde_json::json!({
        "architectures":["MuseGlimmerForConditionalGeneration"],"model_type":"muse_glimmer",
        "image_token_id":22,"video_token_id":23,"out_hidden_size":32,"projector_hidden_size":8,
        "text_config":{"model_type":"muse_glimmer_text","hidden_size":8,"num_hidden_layers":2,
          "intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
          "rms_norm_eps":0.00001,"post_norm_eps":0.00002,"vocab_size":24,
          "max_position_embeddings":64,"rope_theta":10000.0,
          "layer_types":["sliding_attention","full_attention"],
          "layer_rope_theta":[10000.0,0.0],"sliding_window":4,"tie_word_embeddings":tied,
          "hidden_act":"silu","attention_dropout":0.0,"qk_scale_factor":1.3,
          "output_multiplier":1.2,"final_logit_softcapping":7.0},
        "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,
          "intermediate_size":12,"num_attention_heads":2,"num_hidden_layers":1,
          "patch_size":2,"patch_temporal":1,"merge_size":2,"pos_emb_height":2,
          "pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
          "hidden_act":"gelu","layer_types":["full_attention"],
          "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
    });
    if routed {
        value["text_config"]["num_experts"] = 4.into();
        value["text_config"]["num_experts_per_tok"] = 2.into();
        value["text_config"]["moe_intermediate_size"] = 8.into();
        value["text_config"]["norm_topk_prob"] = tied.into();
    }
    match scaling {
        "default" => (),
        "linear" => {
            value["text_config"]["rope_scaling"] =
                serde_json::json!({"rope_type":"linear", "factor":2.0})
        }
        "yarn" => {
            value["text_config"]["rope_scaling"] = serde_json::json!({"rope_type":"yarn", "factor":2.0, "original_max_position_embeddings":32.0})
        }
        _ => unreachable!(),
    }
    value
}

#[test]
fn muse_dense_rows_preserve_all_real_hooks_and_complete_cached_kv() {
    for scaling in ["default", "linear", "yarn"] {
        for tied in [false, true] {
            compare_equations(configuration(false, scaling, tied));
        }
    }
}

#[test]
fn muse_routed_rows_preserve_token_local_routing_across_uneven_chunks() {
    for scaling in ["default", "linear", "yarn"] {
        for tied in [false, true] {
            compare_equations(configuration(true, scaling, tied));
        }
    }
}

#[test]
fn muse_prepared_composite_paths_bind_actual_sources_and_original_readout_demand() {
    for routed in [false, true] {
        let value = configuration(routed, "default", false);
        let (artifact, _) = prepared_adapter::payload_fixture_config(&value, 1.0);
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
        let config = muse_glimmer::DecoderConfig::from_hf_value(&value).unwrap();
        let context = NumericContext::default();
        let (_, declarations, paths) = make_model(&config, &context);
        let before = sources.target().source_diagnostics().unwrap();
        assert_eq!(declarations.len(), 18);
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
        for path in ["model.layers.0.attention.input"] {
            assert!(paths.source().prefill_observation(path).is_none());
            let source = admission(&discovery, &[path], false, true);
            assert!(matches!(
                discovery.prepare_capture_selection(&source, paths.source()),
                Err(SelectionError::Undeclared { index: 0 })
            ));
        }
        // Media remains in discovery with its own input requirement; ordinary
        // text declarations neither hide it nor promise causal media assembly.
        let capture = discovery.capture().unwrap();
        let vision = capture
            .catalog
            .get(eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH)
            .unwrap();
        assert!(!vision.decode);
        assert!(vision
            .requirements
            .contains(&eredu_core::ObservationRequirement::MediaInput));
        assert!(paths.source().prefill_observation(&vision.path).is_none());
        assert!(paths
            .source()
            .prefill_observation(eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH)
            .is_none());
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
fn muse_body_rows_keep_every_position_before_state_only_and_last_position_readout() {
    for routed in [false, true] {
        let config =
            muse_glimmer::DecoderConfig::from_hf_value(&configuration(routed, "linear", true))
                .unwrap();
        let context = NumericContext::default();
        let (mut model, declarations, paths) = make_model(&config, &context);
        let body = declarations
            .into_iter()
            .filter(|d| d.readout_stage() == Stage::BeforeReadout)
            .collect::<Vec<_>>();
        assert_eq!(body.len(), 10);
        let mut prefix = make_state(&config);
        run(
            &mut model,
            &config,
            &[4, 2],
            &mut prefix,
            &context,
            &mut DeclaredRows::new(&[]),
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        let mut expected_state = prefix.clone();
        let mut expected = DeclaredRows::new(&body);
        let full = run(
            &mut model,
            &config,
            &[1, 3, 5],
            &mut expected_state,
            &context,
            &mut expected,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        expected.assert_complete(3);
        for demand in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
            let mut state = prefix.clone();
            let mut actual = DeclaredRows::new(&body);
            assert!(!actual.requires_sequence_readout());
            let output = run(
                &mut model,
                &config,
                &[1, 3, 5],
                &mut state,
                &context,
                &mut actual,
                &paths,
                demand,
            );
            actual.assert_complete(3);
            assert_state(&state, &expected_state);
            for declaration in &body {
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
fn muse_prepared_text_adapter_preserves_direct_equations_and_existing_target_scope() {
    use eredu_runtime::inspection::ObservationHookSite;
    for routed in [false, true] {
        let config =
            muse_glimmer::DecoderConfig::from_hf_value(&configuration(routed, "yarn", false))
                .unwrap();
        let context = NumericContext::default();
        let (mut prepared, declarations, paths) = make_model(&config, &context);
        let architecture = MuseModel::new(config.clone(), &context).unwrap();
        let hooks =
            <MuseModel as LayeredArchitecture<NumericBackend, MuseState>>::observation_hooks(
                &architecture,
            );
        for site in [
            ObservationHookSite::Input,
            ObservationHookSite::Unit,
            ObservationHookSite::Readout,
            ObservationHookSite::RoutedUnits,
        ] {
            assert!(hooks.supports(site));
        }
        assert!(<MuseModel as CompositeArchitecture<NumericBackend, MuseState>>::external_assistant_target_profile(&config).is_some());
        assert!(<MuseModel as LayeredArchitecture<NumericBackend, MuseState>>::prediction_execution_groups(&architecture).is_empty());
        let mut direct = ResidentRuntime::new(architecture, &context).unwrap();
        let mut direct_state = make_state(&config);
        let mut prepared_state = make_state(&config);
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
            let parts = [muse_glimmer::DecoderInputPart::Text(&tokens)];
            let expected = direct
                .forward(
                    muse_glimmer::ModelInput {
                        parts: &parts,
                        vision: None,
                        mask: None,
                    },
                    &mut direct_state,
                    &context,
                )
                .unwrap();
            let mut rows = DeclaredRows::new(&declarations);
            let actual = run(
                &mut prepared,
                &config,
                ids,
                &mut prepared_state,
                &context,
                &mut rows,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
            rows.assert_complete(ids.len() as i32);
            assert_tensor_close(&actual, &expected, "prepared ordinary text");
            assert_state(&prepared_state, &direct_state);
        }
        assert_populated(&prepared_state, &config, 10);
    }
}
