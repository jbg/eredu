use super::*;
use eredu_architectures::composite_execution::{
    CompositeArchitecture, PreparedCompositeArchitecture, PreparedCompositeInput,
};
use eredu_architectures::inkling;
use eredu_runtime::layered::{PrefillObservationDeclaration, PreparedLayeredObservationPaths};
use eredu_runtime::ArchitectureParameters;
type State = DeviceState<NumericBackend, NumericHybridLayerState>;
type Architecture = inkling::LayeredModel<NumericBackend>;
type Prepared = PreparedCompositeArchitecture<Architecture>;
type Model = ResidentRuntime<Prepared, NumericBackend, State>;
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

fn configuration(sparse: bool, kernel: i32, local_heads: bool) -> serde_json::Value {
    let mut value = serde_json::json!({
        "model_type":"inkling_mm_model", "image_token_id":15,
        "text_config":{
            "hidden_size":8,"num_hidden_layers":2,"vocab_size":19,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":2,
            "sliding_window_size":4,"layer_types":["sliding_attention","full_attention"],
            "mlp_layer_types":if sparse {vec!["dense","moe"]} else {vec!["dense","dense"]},
            "sconv_kernel_size":kernel,"d_rel":2,"rel_extent":3,
            "intermediate_size":12,"dense_intermediate_size":12,"moe_intermediate_size":6,
            "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
            "unpadded_vocab_size":17,"logits_mup_width_multiplier":1.5,
            "log_scaling_n_floor":3,"log_scaling_alpha":0.7
        }
    });
    if local_heads {
        value["text_config"]["swa_num_attention_heads"] = 2.into();
        value["text_config"]["swa_num_key_value_heads"] = 1.into();
        value["text_config"]["swa_head_dim"] = 2.into();
    }
    value
}

struct Fixture {
    artifact: tempfile::TempDir,
    args: inkling::ModelArgs,
    parameters: BTreeMap<String, NumericTensor>,
    value: serde_json::Value,
}
fn fixture(value: serde_json::Value) -> Fixture {
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
                        if name.ends_with("global_scale") {
                            1.3
                        } else if name.contains("norm") && name.ends_with("weight") {
                            0.9 + delta * 0.002
                        } else if name.contains("sconv") {
                            0.08 + delta * 0.007
                        } else {
                            delta * 0.025
                        }
                    })
                    .collect(),
            ))
        });
    let args = inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&value).unwrap()).unwrap();
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
    for (target, recipe) in inkling::safetensors_recipes(&args, &store).unwrap() {
        parameters.insert(
            target,
            payload::recipe_value(&recipe, &store, &context).unwrap(),
        );
    }
    Fixture {
        artifact,
        args,
        parameters,
        value,
    }
}
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
fn make_model(
    f: &Fixture,
    context: &NumericContext,
) -> (
    Model,
    Vec<PrefillObservationDeclaration>,
    PreparedLayeredObservationPaths,
) {
    let architecture = Architecture::new(f.args.clone(), context).unwrap();
    let declarations=<Architecture as LayeredArchitecture<NumericBackend,State>>::prefill_observation_declarations(&architecture).unwrap();
    let count = f.args.text_config.num_hidden_layers as usize;
    assert_eq!(declarations.len(), 11 + 4 * count);
    for index in 0..count {
        let path = <Architecture as LayeredArchitecture<NumericBackend, State>>::unit_path(
            &architecture,
            2,
            index,
        )
        .unwrap();
        assert_eq!(path, format!("model.layers.{index}"));
        for suffix in ["input", "input.effective", "output", "output.effective"] {
            assert!(declarations
                .iter()
                .any(|d| d.path() == format!("{path}.{suffix}")));
        }
    }
    assert_eq!(
        declarations
            .iter()
            .find(|d| d.path() == "readout.scaled")
            .unwrap()
            .readout_stage(),
        Stage::ReadoutInput
    );
    let prepared = Prepared::new(architecture);
    assert_eq!(
        <Prepared as LayeredArchitecture<NumericBackend, State>>::prefill_observation_declarations(
            &prepared
        )
        .unwrap(),
        declarations
    );
    let mut model = ResidentRuntime::new(prepared, context).unwrap();
    let mut populate = Populate(&f.parameters, BTreeMap::new());
    <Prepared as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(
        model.architecture_mut(),
    )
    .visit_parameters_mut(&mut populate);
    for unit in model.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut populate);
    }
    let description = model
        .architecture()
        .inner()
        .parameter_description(context)
        .unwrap();
    let expected = description
        .groups()
        .iter()
        .flat_map(|g| g.group().members())
        .map(|m| (m.target().to_owned(), m.global_shape().to_vec()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        populate.1, expected,
        "every actual physical slot initialized before paths/state"
    );
    let paths = model.prepare_observation_paths().unwrap();
    for d in &declarations {
        assert_eq!(paths.source().prefill_observation(d.path()), Some(d));
    }
    assert!(paths
        .source()
        .prefill_observation(eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH)
        .is_none());
    (model, declarations, paths)
}
fn state(f: &Fixture) -> State {
    State::create(inkling::state_layout(&f.args).unwrap(), |_, p| {
        Ok::<_, Error>(NumericHybridLayerState::new(p))
    })
    .unwrap()
}
#[allow(clippy::too_many_arguments)]
fn run(
    model: &mut Model,
    f: &Fixture,
    ids: &[usize],
    state: &mut State,
    context: &NumericContext,
    rows: &mut Rows<'_>,
    paths: &PreparedLayeredObservationPaths,
    demand: OutputDemand,
) -> Option<NumericTensor> {
    // Loaded packed expert parameters have separate executable scalar views.
    assert!(context.bind_checkpoint_values);
    let input = numeric_text_prepared_input(ids);
    let admission =
        <Architecture as CompositeArchitecture<NumericBackend, State>>::admission_config(
            model.architecture().inner(),
        );
    let admitted =
        <Architecture as CompositeArchitecture<NumericBackend, State>>::admit_prepared_input(
            &admission,
            &input,
            &NumericInputInspector,
        )
        .unwrap();
    let paired = PreparedCompositeInput::new(&input, &admitted).unwrap();
    assert_eq!(
        <Prepared as LayeredArchitecture<NumericBackend, State>>::inference_input_shape(&paired)
            .unwrap(),
        Some([1, ids.len() as u64])
    );
    let (scores, _) = model
        .forward_with_prepared_observer_and_context_with_readout(
            paired, state, context, rows, paths, demand,
        )
        .unwrap();
    scores.map(|s| eredu_runtime::observe_model_logits(rows, &s).unwrap())
}
fn compare_layer(a: &NumericHybridLayerState, b: &NumericHybridLayerState) {
    assert_eq!(a.position(), b.position());
    assert_eq!(a.fixed_offset, b.fixed_offset);
    assert_eq!(a.resets, b.resets);
    assert!(a.compressed.is_none() && b.compressed.is_none());
    assert!(a.pooling.is_none() && b.pooling.is_none());
    assert_eq!(
        a.fixed.keys().collect::<Vec<_>>(),
        b.fixed.keys().collect::<Vec<_>>()
    );
    for (role, a) in &a.fixed {
        let b = &b.fixed[role];
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            assert_eq!(a.dtype, b.dtype);
            assert_tensor_close(a, b, &format!("{role:?}"));
        }
    }
    assert_eq!(a.attention.is_some(), b.attention.is_some());
    if let (Some(a), Some(b)) = (&a.attention, &b.attention) {
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.window, b.window);
        // This actual resident/sliding owner returns full visible history.
        // The separate existing BlockCache fixture proves its own relative override.
        assert!(a.attention_history.is_none() && b.attention_history.is_none());
        for (a, b) in [(&a.keys, &b.keys), (&a.values, &b.values)] {
            assert_eq!(a.is_some(), b.is_some());
            if let (Some(a), Some(b)) = (a, b) {
                assert_eq!(a.dtype, b.dtype);
                assert_tensor_close(a, b, "complete KV");
            }
        }
    }
}
fn compare_state(a: &State, b: &State) {
    assert_eq!(a.layout(), b.layout());
    assert_eq!(a.as_ref().len(), b.as_ref().len());
    for (a, b) in a.as_ref().iter().zip(b.as_ref()) {
        compare_layer(a, b);
    }
}
fn populated(state: &State, f: &Fixture, position: i32) {
    assert_eq!(
        state.as_ref().len(),
        f.args.text_config.num_hidden_layers as usize
    );
    for (layer, policy) in state
        .as_ref()
        .iter()
        .zip(f.args.text_config.layer_schedule.iter())
    {
        assert_eq!(layer.resets, 0);
        assert_eq!(layer.fixed_offset, 0);
        assert!(layer.compressed.is_none() && layer.pooling.is_none());
        let cache = layer.attention.as_ref().unwrap();
        assert_eq!(cache.offset, position);
        assert_eq!(layer.position(), position);
        let window = policy.attention.window().map(|w| w.get() as i32);
        assert_eq!(cache.window, window);
        assert!(cache.attention_history.is_none());
        let local = window.is_some();
        let heads = f.args.text_config.key_value_heads(local);
        let dim = f.args.text_config.attention_head_dim(local);
        for value in [&cache.keys, &cache.values] {
            let value = value.as_ref().unwrap();
            assert_eq!(
                value.shape,
                [1, heads, window.map_or(position, |w| position.min(w)), dim]
            );
            nonzero(value);
        }
        let history = f.args.text_config.sconv_kernel_size - 1;
        assert_eq!(layer.fixed.len(), if history == 0 { 0 } else { 4 });
        if history != 0 {
            for (slot, width) in [
                heads * dim,
                heads * dim,
                f.args.text_config.hidden_size,
                f.args.text_config.hidden_size,
            ]
            .into_iter()
            .enumerate()
            {
                let value = layer.fixed[&StateTensorRole::Convolution { slot: slot as u32 }]
                    .as_ref()
                    .unwrap();
                assert_eq!(value.shape, [1, history, width]);
                nonzero(value);
            }
        }
    }
}
fn equations(f: &Fixture) {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let (mut model, declarations, paths) = make_model(f, &context);
    let mut prefix = state(f);
    run(
        &mut model,
        f,
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
    let output = run(
        &mut model,
        f,
        &[1, 3, 5, 2, 6],
        &mut full_state,
        &context,
        &mut full,
        &paths,
        OutputDemand::Sequence,
    )
    .unwrap();
    full.complete(5);
    assert_eq!(output.shape, [1, 5, 17]);
    assert_tensor_close(
        &full.values["readout.scaled"],
        &full.values["readout.normalized.effective"].map(|x| x / 1.5),
        "real muP-scaled input",
    );
    assert_tensor_close(
        &output,
        &full.values["readout.linear.effective"],
        "actual final unpadded scores",
    );
    let mut assembled = BTreeMap::<String, Vec<f32>>::new();
    let mut consumed = Vec::new();
    for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
        let mut rows = Rows::new(&declarations);
        run(
            &mut model,
            f,
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
            f,
            &consumed,
            &mut oracle,
            &context,
            &mut Rows::new(&[]),
            &paths,
            OutputDemand::Sequence,
        )
        .unwrap();
        compare_state(&split_state, &oracle);
        populated(&split_state, f, 2 + consumed.len() as i32);
        for d in &declarations {
            assembled
                .entry(d.path().into())
                .or_default()
                .extend_from_slice(&rows.values[d.path()].data);
        }
    }
    for d in &declarations {
        let expected = &full.values[d.path()];
        assert_tensor_close(
            &NumericTensor::new(expected.shape.clone(), assembled.remove(d.path()).unwrap()),
            expected,
            d.path(),
        );
    }
    assert!(assembled.is_empty());
    compare_state(&split_state, &full_state);
    for id in [7, 8, 9] {
        let mut a = Rows::new(&declarations);
        let mut b = Rows::new(&declarations);
        run(
            &mut model,
            f,
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
            f,
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
        compare_state(&split_state, &full_state);
    }
    populated(&split_state, f, 10);
}
#[test]
fn inkling_dense_group_two_rows_preserve_four_histories_and_absolute_relative_positions() {
    for kernel in [1, 3, 4] {
        equations(&fixture(configuration(false, kernel, true)));
    }
}
#[test]
fn inkling_routed_group_two_rows_preserve_shared_experts_and_complete_cached_state() {
    for kernel in [1, 3, 4] {
        equations(&fixture(configuration(true, kernel, true)));
    }
}

#[path = "inkling_rows/history.rs"]
mod history;
#[path = "inkling_rows/partition.rs"]
mod partition;
#[path = "inkling_rows/source.rs"]
mod source;
