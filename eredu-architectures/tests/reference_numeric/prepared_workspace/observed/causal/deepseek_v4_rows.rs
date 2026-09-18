use super::*;
use eredu_runtime::layered::{PrefillObservationDeclaration, PreparedLayeredObservationPaths};

type V4State = DeviceState<NumericBackend, NumericHybridLayerState>;
type Architecture = deepseek::v4::Model<NumericBackend>;
type Model = ResidentRuntime<Architecture, NumericBackend, V4State>;

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
            assert!((3..=4).contains(&value.shape.len()), "{path}");
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
            !path.starts_with("mtp.") && !path.starts_with("dspark."),
            "ordinary hook {path}"
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

fn configuration(hash: bool, yarn: bool, top_k: i32) -> serde_json::Value {
    let mut value = tiny_v4_config();
    value["max_position_embeddings"] = 512.into();
    value["num_hash_layers"] = i32::from(hash).into();
    value["index_topk"] = top_k.into();
    if yarn {
        value["rope_scaling"] = serde_json::json!({
            "type":"yarn", "factor":2.0, "original_max_position_embeddings":128,
            "beta_fast":32.0, "beta_slow":1.0
        });
    }
    value
}

fn make_state(args: &deepseek::V4Args) -> V4State {
    V4State::create(deepseek::v4::state_layout(args).unwrap(), |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap()
}

fn make_model(
    args: &deepseek::V4Args,
    context: &NumericContext,
) -> (
    Model,
    Vec<PrefillObservationDeclaration>,
    PreparedLayeredObservationPaths,
) {
    let architecture = Architecture::new(args.clone(), context).unwrap();
    let declarations = <Architecture as LayeredArchitecture<NumericBackend, V4State>>::
        prefill_observation_declarations(&architecture, None).unwrap();
    assert_eq!(declarations.len(), 13 + 4 * args.num_hidden_layers as usize);
    for index in 0..args.num_hidden_layers as usize {
        let path = <Architecture as LayeredArchitecture<NumericBackend, V4State>>::unit_path(
            &architecture,
            0,
            index, None)
        .unwrap();
        for suffix in ["input", "input.effective", "output", "output.effective"] {
            assert!(declarations
                .iter()
                .any(|d| d.path() == format!("{path}.{suffix}")));
        }
    }
    let mut model = Model::new(architecture, context).unwrap();
    // Raw Parameter::unloaded slots are distinct from NumericBackend's already
    // nonzero linear/norm/hyper factories. Load only these actual declared slots,
    // before creating any path token or cache; no state values are manufactured.
    struct Load {
        names: BTreeMap<String, ()>,
        experts: usize,
    }
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Load {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            let name = metadata.id().as_str();
            let kind = if name.ends_with(".ape") {
                0
            } else if name.ends_with(".attn_sink") {
                1
            } else if name.ends_with(".tid2eid") {
                2
            } else {
                return;
            };
            assert!(self.names.insert(name.into(), ()).is_none());
            assert!(value.data.iter().all(|v| *v == 0.0), "unloaded {name}");
            for (index, value) in value.data.iter_mut().enumerate() {
                *value = match kind {
                    0 => 0.02 + 0.01 * (index % 11) as f32,
                    1 => -0.2 + 0.03 * index as f32,
                    _ => (index % self.experts) as f32,
                };
            }
        }
    }
    let mut load = Load {
        names: BTreeMap::new(),
        experts: args.n_routed_experts as usize,
    };
    for unit in model.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut load);
    }
    // Three target sinks, two ratio4 APEs, one ratio128 APE, optional hash table.
    assert_eq!(load.names.len(), 6 + args.num_hash_layers as usize);
    let paths = model.prepare_observation_paths().unwrap();
    for declaration in &declarations {
        assert_eq!(
            paths.source().prefill_observation(declaration.path()),
            Some(declaration)
        );
    }
    (model, declarations, paths)
}

fn assert_optional(a: &Option<NumericTensor>, b: &Option<NumericTensor>, name: &str) {
    assert_eq!(a.is_some(), b.is_some(), "{name}");
    if let (Some(a), Some(b)) = (a, b) {
        assert_tensor_close(a, b, name);
    }
}

fn assert_state(a: &V4State, b: &V4State, same_last_width: bool) {
    assert_eq!(a.layout(), b.layout());
    assert_eq!(a.as_ref().len(), b.as_ref().len());
    for (index, (a, b)) in a.as_ref().iter().zip(b.as_ref()).enumerate() {
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        assert!(
            a.compressed.is_none() && b.compressed.is_none(),
            "V4 uses pooling, not V3 MLA"
        );
        assert_eq!(
            a.fixed.keys().collect::<Vec<_>>(),
            b.fixed.keys().collect::<Vec<_>>()
        );
        for (role, value) in &a.fixed {
            assert_optional(value, &b.fixed[role], &format!("{index} {role:?}"));
        }
        assert_eq!(a.attention.is_some(), b.attention.is_some());
        if let (Some(a), Some(b)) = (&a.attention, &b.attention) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.window, b.window);
            assert_optional(&a.keys, &b.keys, "unused auxiliary keys");
            assert_optional(&a.values, &b.values, "unused auxiliary values");
            assert!(a.attention_history.is_none() && b.attention_history.is_none());
        }
        let (a, b) = (a.pooling.as_ref().unwrap(), b.pooling.as_ref().unwrap());
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.window, b.window);
        // append_local overwrites this per-invocation mask width BEFORE local_mask.
        // run() checks its exact derivation for every call. probe_next checks that
        // both saved states yield identical next outputs and this field too.
        if same_last_width {
            assert_eq!(a.attention_local_tokens, b.attention_local_tokens);
        }
        assert_optional(&a.local, &b.local, "local key/value channels");
        assert_eq!(a.streams.len(), b.streams.len());
        for (stream, (a, b)) in a.streams.iter().zip(&b.streams).enumerate() {
            assert_eq!(a.ratio, b.ratio);
            assert_eq!(a.processed, b.processed);
            for (name, a, b) in [
                ("pending values", &a.pending_values, &b.pending_values),
                ("pending gates", &a.pending_gates, &b.pending_gates),
                ("pooled", &a.pooled, &b.pooled),
                ("overlap values", &a.overlap_values, &b.overlap_values),
                ("overlap gates", &a.overlap_gates, &b.overlap_gates),
            ] {
                assert_optional(a, b, &format!("layer {index} stream {stream} {name}"));
            }
        }
    }
}

fn assert_nonzero(t: &NumericTensor) {
    assert!(t.data.iter().all(|v| v.is_finite()));
    assert!(t.data.iter().any(|v| v.abs() > 1e-9));
}
fn assert_populated(state: &V4State, args: &deepseek::V4Args, positions: i32) {
    for (index, state) in state.as_ref().iter().enumerate() {
        assert_eq!(state.fixed_offset, 0);
        assert_eq!(state.resets, 0);
        assert!(state.compressed.is_none());
        assert!(
            state.fixed.values().all(Option::is_none),
            "pool fields have their dedicated owner"
        );
        let auxiliary = state.attention.as_ref().unwrap();
        assert_eq!(auxiliary.offset, 0);
        assert!(
            auxiliary.keys.is_none()
                && auxiliary.values.is_none()
                && auxiliary.attention_history.is_none()
        );
        let pool = state.pooling.as_ref().unwrap();
        assert_eq!(pool.offset, positions);
        assert_eq!(pool.window, args.sliding_window);
        let local = pool.local.as_ref().unwrap();
        assert_eq!(
            local.shape,
            [1, positions.min(args.sliding_window), args.head_dim]
        );
        assert_nonzero(local);
        let ratios = match args.attention_policy(index).unwrap() {
            deepseek::V4AttentionPolicy::Local => vec![],
            deepseek::V4AttentionPolicy::Compressed { ratio: 4 } => vec![4, 4],
            deepseek::V4AttentionPolicy::Compressed { ratio } => vec![ratio],
        };
        assert_eq!(pool.streams.len(), ratios.len());
        for (stream, ratio) in pool.streams.iter().zip(ratios) {
            assert_eq!(stream.ratio, ratio);
            assert_eq!(stream.processed, positions);
            let width = if ratio == 4 {
                2 * args.head_dim
            } else {
                args.head_dim
            };
            for pending in [&stream.pending_values, &stream.pending_gates] {
                assert_eq!(pending.is_some(), positions % ratio != 0);
                if let Some(t) = pending {
                    assert_eq!(t.shape, [1, positions % ratio, width]);
                    assert_nonzero(t);
                }
            }
            assert_eq!(stream.pooled.is_some(), positions >= ratio);
            if let Some(t) = &stream.pooled {
                assert_eq!(t.shape, [1, positions / ratio, args.head_dim]);
                assert_nonzero(t);
            }
            for overlap in [&stream.overlap_values, &stream.overlap_gates] {
                assert_eq!(overlap.is_some(), ratio == 4 && positions >= ratio);
                if let Some(t) = overlap {
                    assert_eq!(t.shape, [1, ratio, args.head_dim]);
                    assert_nonzero(t);
                }
            }
        }
    }
}

fn run(
    model: &mut Model,
    ids: &[usize],
    state: &mut V4State,
    context: &NumericContext,
    rows: &mut DeclaredRows<'_>,
    paths: &PreparedLayeredObservationPaths,
    demand: OutputDemand,
) -> Option<NumericTensor> {
    let before = state
        .as_ref()
        .iter()
        .map(|s| {
            let p = s.pooling.as_ref().unwrap();
            (p.offset, p.local.as_ref().map_or(0, |t| t.dim(1)))
        })
        .collect::<Vec<_>>();
    let tokens = NumericTensor::token_ids(ids);
    let (scores, _) = model
        .forward_with_prepared_observer_and_context_with_readout(
            deepseek::mtp::EmbeddedInput::target(&tokens, None),
            state,
            context,
            rows,
            paths,
            demand,
        )
        .unwrap();
    for (state, (offset, local)) in state.as_ref().iter().zip(before) {
        let pool = state.pooling.as_ref().unwrap();
        assert_eq!(pool.offset, offset + ids.len() as i32);
        assert_eq!(pool.attention_local_tokens, local + ids.len() as i32);
    }
    scores.map(|scores| eredu_runtime::observe_model_logits(rows, &scores).unwrap())
}

fn probe_next(
    model: &mut Model,
    a: &V4State,
    b: &V4State,
    context: &NumericContext,
    declarations: &[PrefillObservationDeclaration],
    paths: &PreparedLayeredObservationPaths,
) {
    let (mut a, mut b) = (a.clone(), b.clone());
    let (mut ar, mut br) = (
        DeclaredRows::new(declarations),
        DeclaredRows::new(declarations),
    );
    for (state, rows) in [(&mut a, &mut ar), (&mut b, &mut br)] {
        run(
            model,
            &[11],
            state,
            context,
            rows,
            paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        rows.assert_complete(1);
    }
    for declaration in declarations {
        let path = declaration.path();
        assert_tensor_close(&ar.values[path], &br.values[path], path);
    }
    assert_state(&a, &b, true);
}

fn compare_equations(value: serde_json::Value, prefix_len: usize) {
    let args = deepseek::parse_v4_config(&value).unwrap();
    let context = NumericContext::default();
    let (mut model, declarations, paths) = make_model(&args, &context);
    let mut full_state = make_state(&args);
    let prefix_ids = (0..prefix_len)
        .map(|i| (i * 3 + 2) % args.vocab_size as usize)
        .collect::<Vec<_>>();
    run(
        &mut model,
        &prefix_ids,
        &mut full_state,
        &context,
        &mut DeclaredRows::new(&[]),
        &paths,
        OutputDemand::Sequence,
    )
    .unwrap();
    let prefix = full_state.clone();
    assert_populated(&prefix, &args, prefix_len as i32);
    let mut split = prefix.clone();
    let mut full = DeclaredRows::new(&declarations);
    run(
        &mut model,
        &[1, 3, 5, 2, 6],
        &mut full_state,
        &context,
        &mut full,
        &paths,
        OutputDemand::Sequence,
    )
    .unwrap();
    full.assert_complete(5);
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    let mut consumed = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = DeclaredRows::new(&declarations);
        run(
            &mut model,
            ids,
            &mut split,
            &context,
            &mut rows,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        rows.assert_complete(ids.len() as i32);
        consumed.extend_from_slice(ids);
        let mut direct = prefix.clone();
        run(
            &mut model,
            &consumed,
            &mut direct,
            &context,
            &mut DeclaredRows::new(&[]),
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        assert_state(&split, &direct, false);
        assert_populated(&split, &args, (prefix_len + consumed.len()) as i32);
        probe_next(&mut model, &split, &direct, &context, &declarations, &paths);
        for declaration in &declarations {
            assembled
                .entry(declaration.path().into())
                .or_default()
                .extend_from_slice(&rows.values[declaration.path()].data);
        }
    }
    for declaration in &declarations {
        let path = declaration.path();
        let expected = &full.values[path];
        assert_tensor_close(
            &NumericTensor::new(expected.shape.clone(), assembled.remove(path).unwrap()),
            expected,
            path,
        );
    }
    assert!(assembled.is_empty());
    assert_state(&split, &full_state, false);
    for id in [7, 8, 9] {
        let (mut expected, mut actual) = (
            DeclaredRows::new(&declarations),
            DeclaredRows::new(&declarations),
        );
        for (state, rows) in [(&mut full_state, &mut expected), (&mut split, &mut actual)] {
            run(
                &mut model,
                &[id],
                state,
                &context,
                rows,
                &paths,
                OutputDemand::Sequence,
            )
            .unwrap();
            rows.assert_complete(1);
        }
        for declaration in &declarations {
            let path = declaration.path();
            assert_tensor_close(&actual.values[path], &expected.values[path], path);
        }
        assert_state(&split, &full_state, true);
    }
    assert_populated(&full_state, &args, (prefix_len + 8) as i32);
}

#[test]
fn v4_local_indexed_and_compressed_rows_match_uneven_chunks_and_cached_decode() {
    for hash in [false, true] {
        for yarn in [false, true] {
            compare_equations(configuration(hash, yarn, 1), 2);
        }
    }
}
#[test]
fn v4_populated_ratio128_and_index_topk_branch_preserve_every_causal_row() {
    compare_equations(configuration(true, true, 2), 126);
}

#[test]
fn v4_body_rows_keep_full_sequence_before_state_only_or_last_readout() {
    let args = deepseek::parse_v4_config(&configuration(true, true, 1)).unwrap();
    let context = NumericContext::default();
    let (mut model, declarations, paths) = make_model(&args, &context);
    let body = declarations
        .iter()
        .filter(|d| d.readout_stage() == Stage::BeforeReadout)
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(body.len(), 14);
    let mut prefix = make_state(&args);
    run(
        &mut model,
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
    let scores = run(
        &mut model,
        &[1, 3, 5, 2, 6],
        &mut expected_state,
        &context,
        &mut expected,
        &paths,
        OutputDemand::Sequence,
    )
    .unwrap();
    expected.assert_complete(5);
    for demand in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
        let mut state = prefix.clone();
        let mut rows = DeclaredRows::new(&body);
        assert!(!rows.requires_sequence_readout());
        let output = run(
            &mut model,
            &[1, 3, 5, 2, 6],
            &mut state,
            &context,
            &mut rows,
            &paths,
            demand,
        );
        rows.assert_complete(5);
        for d in &body {
            assert_tensor_close(&rows.values[d.path()], &expected.values[d.path()], d.path());
        }
        assert_state(&state, &expected_state, true);
        probe_next(
            &mut model,
            &state,
            &expected_state,
            &context,
            &declarations,
            &paths,
        );
        match demand {
            OutputDemand::StateOnly => assert!(output.is_none()),
            _ => assert_tensor_close(
                &output.unwrap(),
                &scores.axis_slice(1, 4, 5),
                "last prediction",
            ),
        }
    }
    // The learned collapse really executes only after physical row selection.
    // A caller cannot ask the shared prepared driver for last-only scores while
    // its observer requires every declared stream/readout position.
    let unchanged = prefix.clone();
    let mut state = prefix;
    let tokens = NumericTensor::token_ids(&[1, 3]);
    let mut rows = DeclaredRows::new(&declarations);
    assert!(matches!(
        model.forward_with_prepared_observer_and_context_with_readout(
            deepseek::mtp::EmbeddedInput::target(&tokens, None),
            &mut state,
            &context,
            &mut rows,
            &paths,
            OutputDemand::LastPosition,
        ),
        Err(eredu_runtime::layered::PreparedLayeredObservationError::ReadoutDemand)
    ));
    assert!(rows.values.is_empty());
    assert_state(&state, &unchanged, true);
}

#[test]
fn v4_actual_catalogue_and_retained_paths_bind_every_target_and_stream_readout() {
    let value = configuration(true, true, 1);
    let (artifact, _) = prepared_adapter::payload_fixture_config(&value, 1.0);
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
            transformations: vec![
                CaptureTransformKind::FullTensor,
                CaptureTransformKind::Preview,
            ],
            ..Default::default()
        },
    );
    let args = deepseek::parse_v4_config(&value).unwrap();
    let context = NumericContext::default();
    let (_, declarations, paths) = make_model(&args, &context);
    let (_, _, other_paths) = make_model(&args, &context);
    let before = sources.target().source_diagnostics().unwrap();
    let catalogue = discovery.capture().unwrap();
    for declaration in &declarations {
        let point = catalogue.catalog.get(declaration.path()).unwrap();
        let axes = point.axes.as_ref().unwrap();
        assert_eq!(axes[1].dimension, eredu_core::SymbolicDimension::Sequence);
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
            let expected = if declaration.readout_stage() == Stage::BeforeReadout {
                OutputDemand::LastPosition
            } else {
                OutputDemand::Sequence
            };
            assert_eq!(geometry.output, expected);
            let bound = selected.bind_geometry(geometry).unwrap();
            assert!(std::ptr::eq(bound.selection(), &selected));
            assert_eq!(bound.geometry(), geometry);
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
            assert!(matches!(
                selected.validate_sources(&source, other_paths.source()),
                Err(SelectionError::Identity)
            ));
        }
    }
    for path in [
        "readout.streams",
        "readout.streams.effective",
        "readout.stream_coefficients",
    ] {
        assert_eq!(
            paths
                .source()
                .prefill_observation(path)
                .unwrap()
                .readout_stage(),
            Stage::ReadoutInput
        );
    }
    // Internal attention/index/cache values keep their distinct contracts; no
    // causal declaration is inferred merely from sequence-shaped axes.
    let internal = "layers.1.compressed_attention.input";
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

#[test]
fn v4_target_declarations_do_not_expand_mtp_dspark_or_intervention_policy() {
    use eredu_runtime::inspection::ObservationHookSite;
    for dspark in [false, true] {
        let mut value = configuration(true, false, 1);
        value["num_nextn_predict_layers"] = 1.into();
        value["compress_ratios"] = serde_json::json!([0, 4, 128, 0]);
        if dspark {
            value["dspark_block_size"] = 2.into();
            value["dspark_noise_token_id"] = 0.into();
            value["dspark_target_layer_ids"] = serde_json::json!([0, 1]);
            value["dspark_markov_rank"] = 2.into();
        }
        let args = deepseek::parse_v4_config(&value).unwrap();
        let model = Architecture::new(args, &NumericContext::default()).unwrap();
        let declarations = <Architecture as LayeredArchitecture<NumericBackend, V4State>>::prefill_observation_declarations(&model, None).unwrap();
        assert_eq!(declarations.len(), 25);
        assert!(declarations
            .iter()
            .all(|d| !d.path().starts_with("mtp.") && !d.path().starts_with("dspark.")));
        assert_eq!(<Architecture as LayeredArchitecture<NumericBackend,V4State>>::prediction_execution_groups(&model).len(), 1);
        let hooks =
            <Architecture as LayeredArchitecture<NumericBackend, V4State>>::observation_hooks(
                &model,
            );
        for site in [
            ObservationHookSite::Input,
            ObservationHookSite::Unit,
            ObservationHookSite::Readout,
        ] {
            assert_eq!(hooks.supports(site), !dspark);
        }
        // Generic ordinary declaration constructors retain their invocation
        // restriction; this production change does not modify hook capability,
        // prediction equations, proposal masks or any intervention implementation.
        let runtime = Model::new(model, &NumericContext::default()).unwrap();
        let paths = runtime.prepare_observation_paths().unwrap();
        assert!(paths.source().prefill_observation("mtp.0.output").is_none());
    }
}
