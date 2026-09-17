//! The two actual conditional group1 owners, using admitted text-only inputs.
use super::qwen_hybrid_rows::recurrent_parameter_pattern;
use super::*;
use eredu_architectures::composite_execution::{
    CompositeArchitecture, PreparedCompositeArchitecture, PreparedCompositeInput,
};
use eredu_architectures::qwen::{hybrid, vl};
use eredu_runtime::layered::{
    PrefillObservationDeclaration as Declaration, PreparedLayeredObservationPaths,
};

type State = DeviceState<NumericBackend, NumericHybridLayerState>;
type Vl = vl::LayeredModel<NumericBackend>;
type Hybrid = hybrid::ConditionalLayeredModel<NumericBackend>;
type PreparedVl = PreparedCompositeArchitecture<Vl>;
type PreparedHybrid = PreparedCompositeArchitecture<Hybrid>;

#[derive(Clone)]
enum Args {
    Vl(vl::ModelArgs),
    Hybrid(hybrid::ParsedHybridConfig),
}
struct Fixture {
    value: serde_json::Value,
    artifact: tempfile::TempDir,
    parameters: BTreeMap<String, NumericTensor>,
    args: Args,
}
fn parameters(name: &str, shape: &[i32]) -> Option<NumericTensor> {
    let (base, step) = recurrent_parameter_pattern(name)?;
    let count: usize = shape.iter().map(|n| usize::try_from(*n).unwrap()).product();
    Some(NumericTensor::new(
        shape.to_vec(),
        (0..count).map(|i| base + step * (i % 7) as f32).collect(),
    ))
}
fn configuration(hybrid_owner: bool, routed: bool, edge: bool, tied: bool) -> serde_json::Value {
    let mut value = if hybrid_owner {
        conditional_qwen_partition_config(routed)
    } else {
        qwen_vl_partition_config(routed)
    };
    value["tie_word_embeddings"] = tied.into();
    value["text_config"]["tie_word_embeddings"] = tied.into();
    if hybrid_owner {
        value["text_config"]["linear_conv_kernel_dim"] = if edge { 1.into() } else { 4.into() };
    } else if edge {
        value["text_config"]["rope_scaling"]["mrope_section"] = serde_json::json!([0, 2, 2]);
    }
    value
}
impl Fixture {
    fn new(value: serde_json::Value, hybrid_owner: bool) -> Self {
        let (artifact, bits) =
            prepared_adapter::payload_fixture_config_with(&value, 1.0, parameters);
        let args = if hybrid_owner {
            Args::Hybrid(hybrid::model_args_from_config_value(&value).unwrap())
        } else {
            Args::Vl(vl::model_args_from_config_value(&value).unwrap())
        };
        let context = NumericContext::default();
        let store = eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap();
        let mut values = bits
            .into_iter()
            .map(|(n, (s, b))| {
                (
                    n,
                    NumericTensor::new(s, b.into_iter().map(f32::from_bits).collect()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let recipes = match &args {
            Args::Vl(a) => {
                let mut recipes = vl::static_recipes(&store);
                for flat in 0..a.vision.layer_count() + a.text.num_hidden_layers as usize {
                    recipes.extend(vl::unit_recipes(&store, a, flat).unwrap());
                }
                recipes
            }
            Args::Hybrid(a) => {
                let mut recipes = hybrid::static_recipes(&store).unwrap();
                for flat in 0..a.vision.as_ref().unwrap().layer_count()
                    + a.text.num_hidden_layers as usize
                    + a.text.mtp_num_hidden_layers as usize
                {
                    recipes.extend(hybrid::conditional_unit_recipes(&store, a, flat).unwrap());
                }
                recipes
            }
        };
        for (name, recipe) in recipes {
            values.insert(
                name,
                payload::recipe_value(&recipe, &store, &context).unwrap(),
            );
        }
        Self {
            value,
            artifact,
            parameters: values,
            args,
        }
    }
    fn state(&self) -> State {
        let layout = match &self.args {
            Args::Vl(a) => vl::state_layout(a).unwrap(),
            Args::Hybrid(a) => hybrid::state_layout(&a.text).unwrap(),
        };
        State::create(layout, |_, p| {
            Ok::<_, Error>(NumericHybridLayerState::new(p))
        })
        .unwrap()
    }
    fn model(
        &self,
        c: &NumericContext,
    ) -> (Model, Vec<Declaration>, PreparedLayeredObservationPaths) {
        match &self.args {
            Args::Vl(a) => {
                let mut m = ResidentRuntime::new(
                    PreparedCompositeArchitecture::new(Vl::new(a.clone(), c).unwrap()),
                    c,
                )
                .unwrap();
                populate(&mut m, &self.parameters);
                let (d, p) = paths(&m);
                (Model::Vl(m, a.clone()), d, p)
            }
            Args::Hybrid(a) => {
                let mut m = ResidentRuntime::new(
                    PreparedCompositeArchitecture::new(Hybrid::new(a.clone(), c).unwrap()),
                    c,
                )
                .unwrap();
                populate(&mut m, &self.parameters);
                let (d, p) = paths(&m);
                (Model::Hybrid(m, a.clone()), d, p)
            }
        }
    }
}
fn populate<A>(
    model: &mut ResidentRuntime<A, NumericBackend, State>,
    values: &BTreeMap<String, NumericTensor>,
) where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    A::StaticModules: Parameterized<NumericTensor>,
    A::Unit: Parameterized<NumericTensor>,
{
    struct Populate<'a>(&'a BTreeMap<String, NumericTensor>, usize);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, m: ParameterMetadata, v: &'a mut NumericTensor) {
            let source = self
                .0
                .get(m.id.as_str())
                .unwrap_or_else(|| panic!("missing actual parameter {}", m.id.as_str()));
            assert_eq!(v.shape, source.shape);
            v.data.clone_from(&source.data);
            nonzero(v);
            self.1 += 1;
        }
    }
    let mut p = Populate(values, 0);
    A::static_modules_mut(model.architecture_mut()).visit_parameters_mut(&mut p);
    for unit in model.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut p);
    }
    assert!(p.1 > 0);
}
fn paths<A>(
    model: &ResidentRuntime<A, NumericBackend, State>,
) -> (Vec<Declaration>, PreparedLayeredObservationPaths)
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
{
    let a = model.architecture();
    let d = A::prefill_observation_declarations(a).unwrap();
    assert_eq!(d.len(), 18);
    assert_eq!(A::group_unit_count(a, 1).unwrap(), 2);
    for i in 0..2 {
        let p = A::unit_path(a, 1, i).unwrap();
        for suffix in ["input", "input.effective", "output", "output.effective"] {
            assert!(d.iter().any(|d| d.path() == format!("{p}.{suffix}")));
        }
    }
    let p = model.prepare_observation_paths().unwrap();
    for d in &d {
        assert_eq!(p.source().prefill_observation(d.path()), Some(d));
    }
    (d, p)
}
enum Model {
    Vl(
        ResidentRuntime<PreparedVl, NumericBackend, State>,
        vl::ModelArgs,
    ),
    Hybrid(
        ResidentRuntime<PreparedHybrid, NumericBackend, State>,
        hybrid::ParsedHybridConfig,
    ),
}
impl Model {
    fn run(
        &mut self,
        ids: &[usize],
        state: &mut State,
        c: &NumericContext,
        rows: &mut Rows<'_>,
        paths: &PreparedLayeredObservationPaths,
        demand: OutputDemand,
    ) -> Result<Option<NumericTensor>, PreparedLayeredObservationError<Error>> {
        // Loaded references refresh packed scalar expert views on every route.
        assert!(c.bind_checkpoint_values);
        let input = numeric_text_prepared_input(ids);
        let out = match self {
            Self::Vl(m, a) => {
                let admitted =
                    eredu_architectures::media_plan::admit_qwen_vl_input(
                        a,
                        &input,
                        &NumericInputInspector,
                    )
                    .unwrap();
                let paired = PreparedCompositeInput::new(&input, &admitted).unwrap();
                m.forward_with_prepared_observer_and_context_with_readout(
                    paired, state, c, rows, paths, demand,
                )?
                .0
            }
            Self::Hybrid(m, a) => {
                let admitted =
                    eredu_architectures::media_plan::admit_qwen_hybrid_input(
                        a,
                        &input,
                        &NumericInputInspector,
                    )
                    .unwrap();
                let paired = PreparedCompositeInput::new(&input, &admitted).unwrap();
                m.forward_with_prepared_observer_and_context_with_readout(
                    paired, state, c, rows, paths, demand,
                )?
                .0
            }
        };
        out.map(|v| {
            eredu_runtime::observe_model_logits(rows, &v)
                .map_err(PreparedLayeredObservationError::Execution)
        })
        .transpose()
    }
    fn invalidate(&mut self) {
        match self {
            Self::Vl(m, _) => {
                let _ = m.architecture_mut();
            }
            Self::Hybrid(m, _) => {
                let _ = m.architecture_mut();
            }
        }
    }
}
struct Rows<'a> {
    declarations: &'a [Declaration],
    values: BTreeMap<String, NumericTensor>,
    prepared: bool,
}
impl<'a> Rows<'a> {
    fn new(d: &'a [Declaration]) -> Self {
        Self {
            declarations: d,
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
    fn complete(&self, n: usize) {
        assert_eq!(self.values.len(), self.declarations.len());
        for d in self.declarations {
            let v = self.values.get(d.path()).unwrap();
            assert_eq!(d.sequence_axis(), 1);
            assert_eq!(v.shape[..2], [1, n as i32]);
            nonzero(v);
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
    fn observe(&mut self, p: &str, v: &NumericTensor) -> Result<(), Error> {
        self.record(p, v);
        Ok(())
    }
    fn observe_replica(&mut self, p: &str, v: &NumericTensor) -> Result<(), Error> {
        self.record(p, v);
        Ok(())
    }
    fn observe_generated(
        &mut self,
        p: &str,
        _: &NumericTensor,
        _: &GeneratedCaptureSource,
        _: &mut dyn FnMut() -> Result<NumericTensor, Error>,
    ) -> Result<(), Error> {
        assert!(self.declarations.iter().all(|d| d.path() != p));
        Ok(())
    }
}
fn nonzero(v: &NumericTensor) {
    assert!(!v.data.is_empty());
    assert!(v.data.iter().all(|v| v.is_finite()));
    assert!(v.data.iter().any(|v| v.abs() > 1e-9));
}
fn same_state(a: &State, b: &State) {
    assert_eq!(a.layout(), b.layout());
    assert_eq!(a.as_ref().len(), b.as_ref().len());
    for (a, b) in a.as_ref().iter().zip(b.as_ref()) {
        assert_eq!(a.position(), b.position());
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert_eq!(a.resets, b.resets);
        assert_eq!(
            a.fixed.keys().collect::<Vec<_>>(),
            b.fixed.keys().collect::<Vec<_>>()
        );
        for (role, a) in &a.fixed {
            let b = &b.fixed[role];
            assert_eq!(a.is_some(), b.is_some());
            if let (Some(a), Some(b)) = (a, b) {
                assert_eq!(a.dtype, b.dtype);
                assert_tensor_close(a, b, "all fixed state");
            }
        }
        assert!(
            a.compressed.is_none()
                && b.compressed.is_none()
                && a.pooling.is_none()
                && b.pooling.is_none()
        );
        assert_eq!(a.attention.is_some(), b.attention.is_some());
        if let (Some(a), Some(b)) = (&a.attention, &b.attention) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.window, b.window);
            assert_eq!(a.attention_history.is_some(), b.attention_history.is_some());
            if let (Some((ak, av)), Some((bk, bv))) = (&a.attention_history, &b.attention_history) {
                assert_eq!(ak.dtype, bk.dtype);
                assert_eq!(av.dtype, bv.dtype);
                assert_tensor_close(ak, bk, "attention-owned keys");
                assert_tensor_close(av, bv, "attention-owned values");
            }
            for (a, b) in [(&a.keys, &b.keys), (&a.values, &b.values)] {
                assert_eq!(a.is_some(), b.is_some());
                if let (Some(a), Some(b)) = (a, b) {
                    assert_eq!(a.dtype, b.dtype);
                    assert_tensor_close(a, b, "complete KV");
                }
            }
        }
    }
}
fn populated(state: &State, positions: i32) {
    for layer in state.as_ref() {
        let stateful = layer.attention.is_some() || !layer.fixed.is_empty();
        assert_eq!(layer.position(), if stateful { positions } else { 0 });
        assert_eq!(layer.resets, 0);
        for (role, value) in &layer.fixed {
            let v = value.as_ref().expect("actual fixed slot populated");
            if *role == StateTensorRole::PositionDelta {
                assert_eq!(v.dtype, eredu_core::checkpoint::TensorDtype::I32);
                assert_eq!(v.data, [0.]);
            } else {
                nonzero(v);
            }
        }
        if let Some(a) = &layer.attention {
            assert_eq!(a.offset, positions);
            assert_eq!(a.window, None);
            nonzero(a.keys.as_ref().unwrap());
            nonzero(a.values.as_ref().unwrap());
        }
    }
}

#[path = "conditional_qwen_rows/binding.rs"]
mod binding;
#[path = "conditional_qwen_rows/equations.rs"]
mod equations;
#[path = "conditional_qwen_rows/partition.rs"]
mod partition;
