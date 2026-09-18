//! Section validation parity; no conditional causal declaration is added here.
use super::*;
use eredu_architectures::composite_execution::{
    CompositeArchitecture, PreparedCompositeArchitecture, PreparedCompositeInput,
};
use eredu_architectures::qwen::vl;

#[test]
fn qwen_zero_mrope_tensor_values_keep_distinct_coordinates_and_cached_offsets() {
    let context = NumericContext::default();
    for ids in [
        [vec![2, 3], vec![5, 7], vec![9, 11]],
        [vec![5, 6], vec![5, 6], vec![5, 6]],
    ] {
        let positions = vl::position_ids_tensor::<NumericTensor>(&ids, &context).unwrap();
        assert_eq!(positions.shape, [2, 3]);
        for sections in [
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 4],
            [0, 2, 2],
            [2, 0, 2],
            [2, 2, 0],
            [1, 1, 2],
        ] {
            let expected = vl::mrope_values(&ids, 8, 100., &sections).unwrap();
            let (cos, sin) =
                vl::mrope_embeddings(&positions, 8, 100., &sections, &context).unwrap();
            assert_eq!(cos.shape, [2, 8]);
            assert_eq!(sin.shape, [2, 8]);
            assert_tensor_close(
                &cos,
                &NumericTensor::new([2, 8], expected.0),
                "independent Qwen scalar cosine",
            );
            assert_tensor_close(
                &sin,
                &NumericTensor::new([2, 8], expected.1),
                "independent Qwen scalar sine",
            );
        }
        for (head, sections) in [
            (0, [0, 0, 0]),
            (7, [1, 1, 1]),
            (8, [0, 0, 0]),
            (8, [-1, 2, 3]),
            (8, [1, 1, 1]),
            (8, [i32::MAX, i32::MAX, 2]),
        ] {
            assert!(vl::mrope_embeddings(&positions, head, 100., &sections, &context).is_err());
        }
        let missing_axis = NumericTensor::from_i32_slice(&[1, 2, 3, 4], &[2, 2], &context).unwrap();
        assert!(vl::mrope_embeddings(&missing_axis, 8, 100., &[0, 2, 2], &context).is_err());
    }
}

type State = DeviceState<NumericBackend, NumericHybridLayerState>;
type Architecture = vl::LayeredModel<NumericBackend>;
type Prepared = PreparedCompositeArchitecture<Architecture>;
type Model = ResidentRuntime<Prepared, NumericBackend, State>;

fn model(config: &serde_json::Value, context: &NumericContext) -> (Model, vl::ModelArgs) {
    let (artifact, bits) = prepared_adapter::payload_fixture_config(config, 1.0);
    let args = vl::model_args_from_config_value(config).unwrap();
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
    for (target, recipe) in vl::static_recipes(&store) {
        parameters.insert(
            target,
            payload::recipe_value(&recipe, &store, context).unwrap(),
        );
    }
    for flat in 0..args.vision.layer_count() + args.text.num_hidden_layers as usize {
        for (target, recipe) in vl::unit_recipes(&store, &args, flat).unwrap() {
            parameters.insert(
                target,
                payload::recipe_value(&recipe, &store, context).unwrap(),
            );
        }
    }
    let architecture =
        PreparedCompositeArchitecture::new(Architecture::new(args.clone(), context).unwrap());
    let mut model = ResidentRuntime::new(architecture, context).unwrap();
    struct Populate<'a>(&'a BTreeMap<String, NumericTensor>, usize);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            let original = self
                .0
                .get(metadata.id().as_str())
                .unwrap_or_else(|| panic!("missing actual {}", metadata.id().as_str()));
            assert_eq!(value.shape, original.shape);
            value.data.clone_from(&original.data);
            assert!(value.data.iter().all(|v| v.is_finite()));
            assert!(value.data.iter().any(|v| v.abs() > 1e-9));
            self.1 += 1;
        }
    }
    let mut populate = Populate(&parameters, 0);
    <Prepared as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(
        model.architecture_mut(),
    )
    .visit_parameters_mut(&mut populate);
    for unit in model.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut populate);
    }
    assert!(populate.1 > 0);
    (model, args)
}

fn run(
    model: &mut Model,
    args: &vl::ModelArgs,
    ids: &[usize],
    state: &mut State,
    context: &NumericContext,
) -> NumericTensor {
    // Execute the populated canonical expert tensors, not constructor views.
    assert!(context.bind_checkpoint_values);
    let input = numeric_text_prepared_input(ids);
    let admitted =
        eredu_architectures::media_plan::admit_qwen_vl_input(
            args,
            &input,
            &NumericInputInspector,
        )
        .unwrap();
    let paired = PreparedCompositeInput::new(&input, &admitted).unwrap();
    model.forward(paired, state, context).unwrap()
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
        for (role, value) in &a.fixed {
            assert_eq!(value.is_some(), b.fixed[role].is_some());
            if let (Some(value), Some(expected)) = (value, &b.fixed[role]) {
                assert_eq!(value.dtype, expected.dtype);
                assert_tensor_close(value, expected, "all fixed state");
                assert_eq!(*role, StateTensorRole::PositionDelta);
                assert_eq!(value.dtype, eredu_core::checkpoint::TensorDtype::I32);
                assert_eq!(value.data, [0.]);
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
            assert!(a.attention_history.is_none() && b.attention_history.is_none());
            for (a, b) in [(&a.keys, &b.keys), (&a.values, &b.values)] {
                assert_eq!(a.is_some(), b.is_some());
                if let (Some(a), Some(b)) = (a, b) {
                    assert_tensor_close(a, b, "complete retained KV");
                    assert!(a.data.iter().any(|v| v.abs() > 1e-9));
                }
            }
        }
    }
}

#[test]
fn prepared_qwen_vl_zero_sections_preserve_complete_cached_text_state() {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    for routed in [false, true] {
        for sections in [
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 4],
            [0, 2, 2],
            [2, 0, 2],
            [2, 2, 0],
            [1, 1, 2],
        ] {
            let mut config = qwen_vl_partition_config(routed);
            config["text_config"]["rope_scaling"]["mrope_section"] = serde_json::json!(sections);
            let (mut model, args) = model(&config, &context);
            let mut full = State::create(vl::state_layout(&args).unwrap(), |_, policy| {
                Ok::<_, Error>(NumericHybridLayerState::new(policy))
            })
            .unwrap();
            run(&mut model, &args, &[4, 2], &mut full, &context);
            let prefix = full.clone();
            let mut chunked = prefix.clone();
            let expected = run(&mut model, &args, &[1, 3, 5, 2, 6], &mut full, &context);
            let mut outputs = Vec::new();
            let mut consumed = Vec::new();
            for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
                outputs.push(run(&mut model, &args, ids, &mut chunked, &context));
                consumed.extend_from_slice(ids);
                let mut reference = prefix.clone();
                run(&mut model, &args, &consumed, &mut reference, &context);
                same_state(&chunked, &reference);
            }
            assert_tensor_close(
                &NumericTensor::concatenate(&outputs, 1, &context).unwrap(),
                &expected,
                "zero-section full versus chunked logits",
            );
            same_state(&full, &chunked);
            for id in [1, 3, 2] {
                let a = run(&mut model, &args, &[id], &mut full, &context);
                let b = run(&mut model, &args, &[id], &mut chunked, &context);
                assert_tensor_close(&a, &b, "zero-section continued decode");
                same_state(&full, &chunked);
            }
        }
    }
}
