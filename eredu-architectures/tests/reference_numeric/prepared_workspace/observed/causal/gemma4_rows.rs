use super::*;
use eredu_architectures::gemma4;
use eredu_runtime::layered::{PrefillObservationDeclaration, PreparedLayeredObservationPaths};
use eredu_runtime::ArchitectureParameters;

type State = DeviceState<NumericBackend, NumericHybridLayerState>;
type Architecture = gemma4::LayeredModel<NumericBackend>;
type Model = ResidentRuntime<Architecture, NumericBackend, State>;

// Both ordinary and replica callbacks are real emissions. No effective value is
// synthesized from an intervention callback or another collected observation.
struct Rows<'a> {
    declarations: &'a [PrefillObservationDeclaration],
    values: BTreeMap<String, NumericTensor>,
    prepared: bool,
}
impl<'a> Rows<'a> {
    fn new(declarations: &'a [PrefillObservationDeclaration]) -> Self {
        Self {
            declarations,
            values: BTreeMap::new(),
            prepared: true,
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
    fn complete(&self, positions: usize) {
        assert_eq!(self.values.len(), self.declarations.len());
        for d in self.declarations {
            let t = self
                .values
                .get(d.path())
                .unwrap_or_else(|| panic!("missing {}", d.path()));
            assert_eq!(d.sequence_axis(), 1);
            assert_eq!(t.shape.len(), 3);
            assert_eq!(t.shape[..2], [1, positions as i32]);
            nonzero(t);
        }
    }
}
impl ActivationObserver<NumericTensor, Error> for Rows<'_> {
    fn requires_prepared_traversal(&self) -> bool {
        self.prepared
    }
    fn requires_sequence_readout(&self) -> bool {
        self.declarations
            .iter()
            .any(|d| d.readout_stage() != Stage::BeforeReadout)
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.record(path, value);
        Ok(())
    }
    fn observe_replica(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
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
fn nonzero(t: &NumericTensor) {
    assert!(t.data.iter().all(|v| v.is_finite()));
    assert!(t.data.iter().any(|v| v.abs() > 1e-9));
}
fn configuration(sparse: bool, wide: bool, per_layer: bool, tied: bool) -> serde_json::Value {
    let layers = if wide { 8 } else { 4 };
    let mut value = serde_json::json!({
        "model_type":"gemma4_unified", "tie_word_embeddings":tied,
        "text_config": {
            "model_type":"gemma4_text", "hidden_size":8,"num_hidden_layers":layers,
            "intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":2,
            "head_dim":4,"rms_norm_eps":0.00001,"vocab_size":19,
            "max_position_embeddings":64,"attention_bias":true,"attention_k_eq_v":sparse,
            "num_kv_shared_layers":layers/2,
            "layer_types":(0..layers).map(|i|if i%2==0 {"sliding_attention"} else {"full_attention"}).collect::<Vec<_>>(),
            "sliding_window":4,"enable_moe_block":sparse,"num_experts":sparse.then_some(4),
            "top_k_experts":sparse.then_some(2),"moe_intermediate_size":sparse.then_some(6),
            "hidden_size_per_layer_input":if per_layer {4} else {0},
            "vocab_size_per_layer_input":19,"final_logit_softcapping":7.0
        }
    });
    if tied {
        value["text_config"]["rope_parameters"] = serde_json::json!({
            "full_attention":{"rope_type":"proportional","partial_rotary_factor":0.5,"rope_theta":10000.0},
            "sliding_attention":{"rope_type":"default","rope_theta":1000.0}
        });
    }
    value
}
struct Fixture {
    artifact: tempfile::TempDir,
    args: gemma4::FamilyConfig,
    parameters: BTreeMap<String, NumericTensor>,
    value: serde_json::Value,
}
fn fixture_for(value: serde_json::Value) -> Fixture {
    let (artifact, bits) =
        prepared_adapter::payload_fixture_config_with(&value, 1.0, |name, shape| {
            let seed = name
                .bytes()
                .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            Some(NumericTensor::new(
                shape.to_vec(),
                (0..shape.iter().product::<i32>() as usize)
                    .map(|i| {
                        let delta = ((i * 7 + (seed % 97) as usize) % 41) as f32 - 20.;
                        if name.ends_with("layer_scalar") {
                            1.15
                        } else if name.contains("norm") && name.ends_with("weight") {
                            0.9 + delta * 0.002
                        } else {
                            delta * 0.025
                        }
                    })
                    .collect(),
            ))
        });
    let args = gemma4::FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).unwrap();
    let mut parameters = bits
        .into_iter()
        .map(|(name, (shape, bits))| {
            (
                name,
                NumericTensor::new(shape, bits.into_iter().map(f32::from_bits).collect()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let store = eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap();
    let context = NumericContext::default();
    for flat in 0..args.text.num_hidden_layers() {
        for (target, recipe) in gemma4::unit_recipes(&store, &args, flat).unwrap() {
            parameters.insert(
                target,
                payload::recipe_value(&recipe, &store, &context).unwrap(),
            );
        }
    }
    Fixture {
        artifact,
        args,
        parameters,
        value,
    }
}
fn make_model(
    f: &Fixture,
    context: &NumericContext,
) -> (
    Model,
    Vec<PrefillObservationDeclaration>,
    PreparedLayeredObservationPaths,
) {
    let mut model =
        ResidentRuntime::new(Architecture::new(f.args.clone(), context).unwrap(), context).unwrap();
    struct Populate<'a>(
        &'a BTreeMap<String, NumericTensor>,
        BTreeMap<String, Vec<usize>>,
    );
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            let name = metadata.id.as_str();
            let original = self.0.get(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(value.shape, original.shape, "{name}");
            value.data.clone_from(&original.data);
            nonzero(value);
            assert!(self
                .1
                .insert(
                    name.into(),
                    value.shape.iter().map(|n| *n as usize).collect()
                )
                .is_none());
        }
    }
    let mut populate = Populate(&f.parameters, BTreeMap::new());
    <Architecture as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(
        model.architecture_mut(),
    )
    .visit_parameters_mut(&mut populate);
    for unit in model.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut populate);
    }
    let description = model.architecture().parameter_description(context).unwrap();
    let expected = description
        .groups()
        .iter()
        .flat_map(|g| g.group().members())
        .map(|m| (m.target().to_owned(), m.global_shape().to_vec()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        populate.1, expected,
        "all physical parameter slots initialized before token/state"
    );
    let declarations=<Architecture as LayeredArchitecture<NumericBackend,State>>::prefill_observation_declarations(model.architecture()).unwrap();
    assert_eq!(declarations.len(), 10 + 4 * f.args.text.num_hidden_layers());
    for i in 0..f.args.text.num_hidden_layers() {
        let base = <Architecture as LayeredArchitecture<NumericBackend, State>>::unit_path(
            model.architecture(),
            2,
            i,
        )
        .unwrap();
        assert_eq!(base, format!("model.language_model.layers.{i}"));
        for suffix in ["input", "input.effective", "output", "output.effective"] {
            assert!(declarations
                .iter()
                .any(|d| d.path() == format!("{base}.{suffix}")));
        }
    }
    let paths = model.prepare_observation_paths().unwrap();
    for d in &declarations {
        assert_eq!(paths.source().prefill_observation(d.path()), Some(d));
    }
    (model, declarations, paths)
}
fn state(f: &Fixture) -> State {
    State::create(gemma4::state_layout(&f.args.text).unwrap(), |_, p| {
        Ok::<_, Error>(NumericHybridLayerState::new(p))
    })
    .unwrap()
}
fn run(
    model: &mut Model,
    ids: &[usize],
    state: &mut State,
    context: &NumericContext,
    rows: &mut Rows<'_>,
    paths: &PreparedLayeredObservationPaths,
    demand: OutputDemand,
) -> Option<NumericTensor> {
    // Numeric packed expert parameters and executable scalar views are separate.
    // Every direct reference must consume the payload populated by make_model,
    // just as the prepared partition binder does.
    assert!(context.bind_checkpoint_values);
    let tokens = NumericTensor::token_ids(ids);
    let parts = [gemma4::DecoderInputPart::Text(&tokens)];
    let input = gemma4::ModelInput {
        parts: &parts,
        vision: None,
        audio: None,
        per_layer_tokens: None,
        mask: None,
    };
    let (scores, _) = model
        .forward_with_prepared_observer_and_context_with_readout(
            input, state, context, rows, paths, demand,
        )
        .unwrap();
    scores.map(|s| eredu_runtime::observe_model_logits(rows, &s).unwrap())
}
fn compare_cache(
    a: &NumericCache,
    b: &NumericCache,
    a_chunk: usize,
    b_chunk: usize,
    heads: std::ops::Range<usize>,
) {
    assert_eq!(a.offset, b.offset);
    assert_eq!(a.window, b.window);
    for (x, y) in [(&a.keys, &b.keys), (&a.values, &b.values)] {
        assert_eq!(x.is_some(), y.is_some());
        if let (Some(x), Some(y)) = (x, y) {
            let y = y.axis_slice(1, heads.start, heads.end);
            assert_eq!(x.dtype, y.dtype);
            assert_tensor_close(x, &y, "all persistent bounded KV");
            nonzero(x);
        }
    }
    assert_eq!(a.attention_history.is_some(), b.attention_history.is_some());
    if let (Some((ak, av)), Some((bk, bv))) = (&a.attention_history, &b.attention_history) {
        let extent = |chunk: usize| {
            a.window.map_or(a.offset, |w| {
                (a.offset - chunk as i32).min(w) + chunk as i32
            }) as usize
        };
        let ae = extent(a_chunk);
        let be = extent(b_chunk);
        assert_eq!(ak.shape[2] as usize, ae);
        assert_eq!(av.shape[2] as usize, ae);
        assert_eq!(bk.shape[2] as usize, be);
        assert_eq!(bv.shape[2] as usize, be);
        assert!(ae <= be);
        // Invocation-local scratch may have different width. Every retained
        // actual history element matches its absolute-position reference slice.
        for (x, y) in [(ak, bk), (av, bv)] {
            let y = y
                .axis_slice(1, heads.start, heads.end)
                .axis_slice(2, be - ae, be);
            assert_tensor_close(x, &y, "entire invocation history absolute slice");
            nonzero(x);
        }
    }
}
fn compare_state(a: &State, b: &State, a_chunk: usize, b_chunk: usize) {
    assert_eq!(a.layout(), b.layout());
    assert_eq!(a.as_ref().len(), b.as_ref().len());
    for (a, b) in a.as_ref().iter().zip(b.as_ref()) {
        assert_eq!(a.position(), b.position());
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        assert!(a.compressed.is_none() && b.compressed.is_none());
        assert!(a.pooling.is_none() && b.pooling.is_none());
        assert_eq!(
            a.fixed.keys().collect::<Vec<_>>(),
            b.fixed.keys().collect::<Vec<_>>()
        );
        for (role, value) in &a.fixed {
            let expected = &b.fixed[role];
            assert_eq!(value.is_some(), expected.is_some());
            if let (Some(value), Some(expected)) = (value, expected) {
                assert_tensor_close(value, expected, "actual fixed role");
            }
        }
        assert_eq!(a.attention.is_some(), b.attention.is_some());
        if let (Some(a), Some(b)) = (&a.attention, &b.attention) {
            compare_cache(
                a,
                b,
                a_chunk,
                b_chunk,
                0..b.keys.as_ref().unwrap().shape[1] as usize,
            );
        }
    }
}
fn populated(state: &State, f: &Fixture, position: i32) {
    assert_eq!(state.as_ref().len(), f.args.text.num_hidden_layers());
    for (i, layer) in state.as_ref().iter().enumerate() {
        assert_eq!(layer.resets, 0);
        assert_eq!(layer.fixed_offset, 0);
        assert!(layer.compressed.is_none() && layer.pooling.is_none());
        if i == 0 {
            assert_eq!(layer.fixed.len(), 1);
            assert!(layer.fixed[&StateTensorRole::PrefixEmbedding].is_none());
        } else {
            assert!(layer.fixed.is_empty());
        }
        let policy = f.args.text.layer_policy(i).unwrap();
        if policy.key_value == eredu_nn::AttentionStateSource::Shared {
            assert_eq!(layer.position(), 0);
            assert!(layer.attention.is_none());
        } else {
            let cache = layer.attention.as_ref().unwrap();
            assert_eq!(cache.offset, position);
            assert_eq!(layer.position(), position);
            for t in [cache.keys.as_ref().unwrap(), cache.values.as_ref().unwrap()] {
                nonzero(t);
            }
        }
    }
}
fn equations(f: &Fixture, owned: bool) {
    let context = NumericContext {
        bind_checkpoint_values: true,
        cache_owned_attention: owned,
        ..Default::default()
    };
    let (mut model, declarations, paths) = make_model(f, &context);
    let mut prefix = state(f);
    run(
        &mut model,
        &[4, 2],
        &mut prefix,
        &context,
        &mut Rows::new(&[]),
        &paths,
        OutputDemand::Sequence,
    )
    .unwrap();
    populated(&prefix, f, 2);
    let mut full_state = prefix.clone();
    let mut split_state = prefix.clone();
    let mut full = Rows::new(&declarations);
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
    full.complete(5);
    let linear = &full.values["readout.linear"];
    let logits = &full.values["model.logits"];
    let capped = NumericTensor::new(
        linear.shape.clone(),
        linear.data.iter().map(|v| 7. * (v / 7.).tanh()).collect(),
    );
    assert_tensor_close(logits, &capped, "post-softcap model.logits");
    assert!(linear
        .data
        .iter()
        .zip(&logits.data)
        .any(|(a, b)| (a - b).abs() > 1e-6));
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    let mut consumed = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = Rows::new(&declarations);
        run(
            &mut model,
            ids,
            &mut split_state,
            &context,
            &mut rows,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        rows.complete(ids.len());
        consumed.extend_from_slice(ids);
        let mut oracle = prefix.clone();
        run(
            &mut model,
            &consumed,
            &mut oracle,
            &context,
            &mut Rows::new(&[]),
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        compare_state(&split_state, &oracle, ids.len(), consumed.len());
        populated(&split_state, f, 2 + consumed.len() as i32);
        for d in &declarations {
            assembled
                .entry(d.path().into())
                .or_default()
                .extend_from_slice(&rows.values[d.path()].data);
        }
    }
    for d in &declarations {
        let b = &full.values[d.path()];
        assert_tensor_close(
            &NumericTensor::new(b.shape.clone(), assembled.remove(d.path()).unwrap()),
            b,
            d.path(),
        );
    }
    assert!(assembled.is_empty());
    compare_state(&split_state, &full_state, 2, 5);
    for id in [7, 8, 9] {
        let mut a = Rows::new(&declarations);
        let mut b = Rows::new(&declarations);
        run(
            &mut model,
            &[id],
            &mut split_state,
            &context,
            &mut a,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        run(
            &mut model,
            &[id],
            &mut full_state,
            &context,
            &mut b,
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        a.complete(1);
        b.complete(1);
        for d in &declarations {
            assert_tensor_close(&a.values[d.path()], &b.values[d.path()], d.path());
        }
        // Same next invocation overwrites scratch: entire raw histories now agree.
        compare_state(&split_state, &full_state, 1, 1);
    }
    populated(&split_state, f, 10);
}
#[test]
fn gemma4_dense_group_two_rows_preserve_shared_kv_and_all_real_callbacks() {
    for (per_layer, tied) in [(true, false), (false, true)] {
        let f = fixture_for(configuration(false, false, per_layer, tied));
        for owned in [false, true] {
            equations(&f, owned);
        }
    }
}
#[test]
fn gemma4_sparse_group_two_rows_preserve_normalized_shared_values_and_causal_offsets() {
    let f = fixture_for(configuration(true, false, true, false));
    for owned in [false, true] {
        equations(&f, owned);
    }
}

#[test]
fn gemma4_prepared_sources_bind_every_target_row_and_preserve_physical_readout() {
    for sparse in [false, true] {
        let f = fixture_for(configuration(sparse, false, true, false));
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
        let internal = "model.language_model.layers.0.attention.input";
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
fn gemma4_body_rows_precede_state_only_and_last_position_readout() {
    for sparse in [false, true] {
        let f = fixture_for(configuration(sparse, false, true, false));
        let context = NumericContext {
            bind_checkpoint_values: true,
            cache_owned_attention: true,
            ..Default::default()
        };
        let (mut model, declarations, paths) = make_model(&f, &context);
        let body = declarations
            .into_iter()
            .filter(|d| d.readout_stage() == Stage::BeforeReadout)
            .collect::<Vec<_>>();
        assert_eq!(body.len(), 18);
        let mut prefix = state(&f);
        run(
            &mut model,
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
                    &mut model, ids, &mut state, &context, &mut rows, &paths, demand,
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
                        "selected post-softcap last row",
                    );
                }
                populated(&state, &f, 2 + end as i32);
                start = end;
            }
            compare_state(&state, &expected_state, 2, 5);
        }
    }
}

fn compare_partition_state(
    actual: &State,
    expected: &State,
    f: &Fixture,
    identity: &eredu_core::cache::PromptCacheModelIdentity,
    rank: ParallelRankTopology,
    chunk: usize,
) {
    let range = identity.global_layer_start()..identity.global_layer_end();
    assert_eq!(actual.as_ref().len(), range.len());
    assert_eq!(actual.layout().len(), range.len());
    let mut seen_receivers = BTreeSet::new();
    for (local, a) in actual.as_ref().iter().enumerate() {
        let global = range.start + local;
        let policy = f.args.text.layer_policy(global).unwrap();
        let source = if policy.key_value != eredu_nn::AttentionStateSource::Shared {
            Some(global)
        } else {
            let publisher = (0..global)
                .rev()
                .find(|&i| {
                    let p = f.args.text.layer_policy(i).unwrap();
                    p.attention == policy.attention && p.key_value.publishes_state()
                })
                .unwrap();
            // The first local shared consumer owns the actual receiver replica;
            // later consumers of that publisher are NoState views.
            if range.contains(&publisher) || !seen_receivers.insert(publisher) {
                None
            } else {
                Some(publisher)
            }
        };
        assert_eq!(a.fixed_offset, 0);
        assert_eq!(a.resets, 0);
        assert!(a.compressed.is_none() && a.pooling.is_none());
        if global == 0 {
            assert_eq!(a.fixed.len(), 1);
            assert!(a.fixed[&StateTensorRole::PrefixEmbedding].is_none());
        } else {
            assert!(a.fixed.is_empty());
        }
        if let Some(source) = source {
            let b = expected.as_ref()[source].attention.as_ref().unwrap();
            let a = a
                .attention
                .as_ref()
                .expect("actual local owner or receiver replica");
            let heads = eredu_core::balanced_contiguous_range(
                policy.num_key_value_heads.get() as usize,
                rank.tensor_parallel_size(),
                rank.tensor_parallel_rank(),
                false,
            )
            .unwrap();
            compare_cache(a, b, chunk, chunk, heads);
        } else {
            assert!(a.attention.is_none());
            assert_eq!(a.position(), 0);
        }
    }
}

// TP computes rank-local f32 partial sums before reduction. Eight residual
// blocks can amplify their rounding error in unnormalized observed activations.
// Use the usual combined absolute/relative comparison only for these partition
// observations; direct execution, public scores and full state keep the shared
// absolute comparator. Repeated full-matrix diagnostics are recorded with74.
fn assert_partition_observation_close(
    actual: &NumericTensor,
    expected: &NumericTensor,
    label: &str,
) {
    assert_eq!(actual.shape, expected.shape, "{label} shape");
    assert_eq!(actual.dtype, expected.dtype, "{label} dtype");
    assert_eq!(actual.data.len(), expected.data.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label}[{index}] finite"
        );
        let error = (f64::from(actual) - f64::from(expected)).abs();
        let tolerance = 1.0e-4 + 1.0e-4 * f64::from(expected).abs();
        assert!(
            error <= tolerance,
            "{label}[{index}]: expected {expected}, got {actual}, error {error}, tolerance {tolerance}"
        );
    }
}

// These boundary trials consume actual prepared partition sources. Their
// observation callbacks exercise the existing partition executor, not a new
// partition capture admission or a fabricated prepared direct-runtime token.
#[test]
fn gemma4_causal_rows_cross_prepublication_and_shared_consumer_pipeline_cuts() {
    for sparse in [false, true] {
        let f = fixture_for(configuration(sparse, true, true, false));
        let inspection =
            eredu_architectures::configuration::inspect_artifact(f.artifact.path()).unwrap();
        let descriptor = inspection.architecture_plan().architecture_descriptor();
        let parameters = numeric_composite_parameter_description(&f.value);
        let context = NumericContext {
            bind_checkpoint_values: true,
            cache_owned_attention: true,
            ..Default::default()
        };
        let (mut model, declarations, paths) = make_model(&f, &context);
        let inputs: Vec<Vec<usize>> = vec![
            vec![4, 2],
            vec![1, 3],
            vec![5],
            vec![2, 6],
            vec![7],
            vec![8],
            vec![9],
        ];
        let mut direct_state = state(&f);
        let reference = inputs
            .iter()
            .map(|ids| {
                let mut rows = Rows::new(&declarations);
                let scores = run(
                    &mut model,
                    ids,
                    &mut direct_state,
                    &context,
                    &mut rows,
                    &paths,
                    OutputDemand::Sequence,
                )
                .unwrap();
                rows.complete(ids.len());
                (scores, rows.values, direct_state.clone())
            })
            .collect::<Vec<_>>();
        for (tp, pp) in [(2, 1), (1, 4), (2, 4)] {
            let topology = ParallelTopology::new(tp, pp, 1, 1).unwrap();
            for residency in [
                eredu_core::ResidencyPlan::FullyResident,
                eredu_core::ResidencyPlan::LayerwiseHost {
                    device_layer_window: 1,
                    device_budget_bytes: Some(1 << 20),
                    host_budget_bytes: Some(1 << 20),
                },
                eredu_core::ResidencyPlan::DenseDiskStream {
                    device_budget_bytes: 1 << 20,
                    host_budget_bytes: 1 << 20,
                    host_lookahead: 1,
                    background_queue: 1,
                },
            ] {
                let world_label =
                    format!("sparse {sparse} tp {tp} pp {pp} residency {residency:?}");
                let plan = prepared_adapter::plan(None)
                    .with_topology(topology)
                    .with_residency(residency);
                let world = Arc::new(NumericPartitionWorld::default());
                std::thread::scope(|scope| {
                    let workers=(0..topology.world_size()).map(|rank| {
                        let (inspection,parameters,descriptor,plan,inputs,reference,declarations,f)=(&inspection,&parameters,&descriptor,&plan,&inputs,&reference,&declarations,&f);
                        let world=Arc::clone(&world);
                        let world_label=&world_label;
                        std::thread::Builder::new()
                            .name(format!("reference-gemma4-causal-{rank}"))
                            .stack_size(32 * 1024 * 1024)
                            .spawn_scoped(scope, move || {
                            let rank_topology=ParallelRankTopology::new(topology,rank).unwrap();
                            let sources=partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(inspection,plan,rank,std::time::Duration::from_secs(30),None,8).unwrap();
                            let components=sources.selected().execution().component_partition_layout(descriptor,parameters).unwrap().unwrap();
                            let layout=eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters,rank_topology).unwrap();
                            let mut context=NumericContext::with_partition(layout,rank,Arc::clone(&world));context.bind_checkpoint_values=true;context.cache_owned_attention=true;
                            let mut executable=partitioned_adapter::composite(sources,&context).unwrap();
                            let identity=world.prompt_cache_identities().remove(&rank).unwrap();
                            if pp==4 {assert_eq!(identity.global_layer_start(),2*rank_topology.pipeline_parallel_rank());assert_eq!(identity.global_layer_end(),identity.global_layer_start()+2);}
                            for (step,ids) in inputs.iter().enumerate() {
                                let mut rows=Rows::new(declarations);rows.prepared=false;
                                let output=executable.forward_observed(&numeric_text_prepared_input(ids),step<4,&mut rows).unwrap();
                                assert_tensor_close(&output,&reference[step].0.axis_slice(1,ids.len()-1,ids.len()),&format!("{world_label} rank {rank} step {step} selected global scores"));
                                for d in declarations {
                                    let path=d.path();
                                    let site=components.observation(path).unwrap_or_else(||panic!("missing actual partition path {path}"));
                                    assert_eq!(rows.values.contains_key(path),site.coordinates().is_some(),"rank {rank} {path}");
                                    if let Some(actual)=rows.values.get(path) {
                                        nonzero(actual);
                                        // Sequence demand preserves every observation row, including
                                        // model.logits. The public return is indexed only afterward.
                                        assert_partition_observation_close(actual,&reference[step].1[path],&format!("{world_label} rank {rank} step {step} {path}"));
                                    }
                                }
                                let snapshot=executable.snapshot().unwrap();
                                compare_partition_state(&snapshot,&reference[step].2,f,&identity,rank_topology,ids.len());
                                assert_eq!(executable.positions().unwrap(),snapshot.as_ref().iter().map(|l|l.position()).collect::<Vec<_>>());
                            }
                        }).expect("spawn reference gemma4-causal rank worker")
                    }).collect::<Vec<_>>();
                    for worker in workers {
                        worker.join().unwrap();
                    }
                });
            }
        }
    }
}

#[test]
fn gemma4_shared_consumers_reject_missing_or_preceding_publications() {
    let mut missing = configuration(false, false, true, false);
    missing["text_config"]["num_kv_shared_layers"] = 4.into();
    assert!(gemma4::FamilyConfig::from_hf_json(&serde_json::to_vec(&missing).unwrap()).is_err());
    let f = fixture_for(configuration(false, false, true, false));
    let context = NumericContext::default();
    let policy = f.args.text.layer_policy(2).unwrap();
    assert_eq!(policy.key_value, eredu_nn::AttentionStateSource::Shared);
    let mut consumer =
        gemma4::text::Attention::<NumericBackend>::new(&f.args.text, 2, policy, &context).unwrap();
    let hidden = NumericTensor::new([1, 2, 8], (0..16).map(|i| 0.2 + i as f32 * 0.03).collect());
    let mut shared = gemma4::text::SharedAttentionStates::new();
    let absent = consumer.forward::<NumericCache>(
        gemma4::text::AttentionInput {
            hidden: &hidden,
            mask: None,
            cache: None,
            shared: &mut shared,
            rotary_position: None,
        },
        &context,
    );
    assert!(absent.is_err());
    assert!(shared.is_empty());
    let mut cache = NumericCache::new(Some(4));
    let before = cache.clone();
    let preceding = consumer.forward(
        gemma4::text::AttentionInput {
            hidden: &hidden,
            mask: None,
            cache: Some(&mut cache),
            shared: &mut shared,
            rotary_position: None,
        },
        &context,
    );
    assert!(preceding.is_err());
    assert_eq!(cache.offset, before.offset);
    assert!(cache.keys.is_none() && cache.values.is_none() && cache.attention_history.is_none());
    assert!(shared.is_empty());
}

#[path = "gemma4_rows/sliding.rs"]
mod sliding;
