//! Published Pixtral and VLM projector comparison, including packed image isolation.
use super::*;
#[path = "nemotron_pixtral/prepared_partitions.rs"]
mod prepared_partitions;
use eredu_architectures::nemotron_labs_diffusion::{
    FamilyConfig,
    vision::{ImageEmbedding, ImageInput, ImageLayer, ImageProjector},
};

type State = DeviceState<NumericBackend, NumericCache>;
type Composite = eredu_architectures::nemotron_labs_diffusion::composite::Model<NumericBackend>;
type CompositeUnit = eredu_architectures::nemotron_labs_diffusion::composite::Unit<NumericBackend>;
use eredu_architectures::composite_execution::PreparedCompositeInput;
use eredu_core::decoding::AttentionInvocation;

fn composite_state(config: &FamilyConfig) -> State {
    State::create(decoder::state_layout(config).unwrap(), |_, _| {
        Ok::<_, Error>(NumericCache::new(None))
    })
    .unwrap()
}
fn composite_storage(
    runtime: &ResidentRuntime<Composite, NumericBackend, State>,
) -> BTreeMap<String, usize> {
    struct Storage(BTreeMap<String, usize>);
    impl<'a> ParameterVisitor<'a, NumericTensor> for Storage {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a NumericTensor) {
            assert!(
                self.0
                    .insert(metadata.id.to_string(), value.data.as_ptr() as usize)
                    .is_none(),
                "one shared parameter has multiple storage owners"
            );
        }
    }
    let mut storage = Storage(BTreeMap::new());
    runtime
        .architecture()
        .static_modules()
        .visit_parameters(&mut storage);
    for unit in runtime.units().iter().flatten() {
        unit.visit_parameters(&mut storage);
    }
    storage.0
}
fn populate_vision<M: Parameterized<NumericTensor>>(
    module: &mut M,
    fixture: &serde_json::Value,
    layout: Option<&LocalModelLayout>,
) {
    struct Populate<'f> {
        fixture: &'f serde_json::Value,
        layout: Option<&'f LocalModelLayout>,
    }
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            let Some(data) = self.fixture["parameters"].get(metadata.id.as_str()) else {
                return;
            };
            let global = tensor(data);
            let selected = match self
                .layout
                .and_then(|layout| layout.tensor(metadata.id.as_str()))
            {
                Some(layout) => select_parameter(&global, layout.placement()).unwrap(),
                None => global,
            };
            assert_eq!(value.shape(), selected.shape(), "{}", metadata.id);
            *value = selected;
        }
    }
    module.visit_parameters_mut(&mut Populate { fixture, layout });
}
fn prepared_prompt(
    fixture: &serde_json::Value,
    projected: bool,
) -> eredu_runtime::PreparedModelInput<NumericTensor> {
    use eredu_core::{InputExtent, InputMetadataKey, InputModality};
    use eredu_runtime::{PreparedInputPart, PreparedInputPayload, PreparedModelInput};
    let text = |ids: &[usize]| {
        PreparedInputPart::new(
            InputModality::Text,
            PreparedInputPayload::TokenIds(NumericTensor::token_ids(ids)),
            [],
        )
        .unwrap()
    };
    let patches = tensor(&fixture["patches"]);
    let features = tensor(&fixture["outputs"]["projected"]);
    let image = |index: usize, height, width| {
        PreparedInputPart::new_with_extents(
            InputModality::Image,
            if projected {
                PreparedInputPayload::Embeddings(features.axis_slice(1, index * 2, (index + 1) * 2))
            } else {
                PreparedInputPayload::Tensor(patches.axis_slice(0, index * 8, (index + 1) * 8))
            },
            [(
                InputMetadataKey::PatchGrid,
                NumericTensor::from_i32_slice(
                    &[1, height, width],
                    &[1, 3],
                    &NumericContext::default(),
                )
                .unwrap(),
            )],
            [InputExtent::PatchGrid {
                time: 1,
                height: height as usize,
                width: width as usize,
            }],
        )
        .unwrap()
    };
    PreparedModelInput::new(
        vec![
            text(&[1]),
            image(0, 2, 4),
            text(&[4, 5]),
            image(1, 4, 2),
            text(&[7]),
        ],
        |tensor: &NumericTensor| {
            eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
        },
    )
    .unwrap()
}
fn prepared_text(ids: &[usize]) -> eredu_runtime::PreparedModelInput<NumericTensor> {
    eredu_runtime::PreparedModelInput::new(
        vec![
            eredu_runtime::PreparedInputPart::new(
                eredu_core::InputModality::Text,
                eredu_runtime::PreparedInputPayload::TokenIds(NumericTensor::token_ids(ids)),
                [],
            )
            .unwrap(),
        ],
        |tensor: &NumericTensor| {
            eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
        },
    )
    .unwrap()
}

struct FixturePolicy {
    lifecycle: RebuildingUnitPolicy,
    fixture: Arc<serde_json::Value>,
}
impl LayerwisePolicy<NumericBackend, CompositeUnit> for FixturePolicy {
    type Lease = RebuiltUnitLease<CompositeUnit>;
    type Error = Error;
    fn begin(&mut self, input: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, CompositeUnit>>::begin(
            &mut self.lifecycle,
            input,
            context,
        )
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&NumericContext) -> Result<CompositeUnit, E>,
    {
        let fixture = &self.fixture;
        self.lifecycle.acquire(
            ordinal,
            address,
            |context| {
                let mut unit = build(context)?;
                populate_vision(&mut unit, fixture, None);
                Ok(unit)
            },
            context,
        )
    }
    fn complete<'a, SV, CV>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a NumericTensor,
        state: SV,
        context_values: CV,
        context: &NumericContext,
    ) -> Result<(), Error>
    where
        SV: Iterator<Item = &'a NumericTensor>,
        CV: Iterator<Item = &'a NumericTensor>,
    {
        self.lifecycle.complete(
            ordinal,
            address,
            lease,
            output,
            state,
            context_values,
            context,
        )
    }
    fn finish(&mut self, output: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, CompositeUnit>>::finish(
            &mut self.lifecycle,
            output,
            context,
        )
    }
}

#[test]
fn prepared_pixtral_ingress_observes_independent_patch_embeddings() {
    use eredu_architectures::composite_execution::PreparedCompositeArchitecture;
    let fixture: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON,
    )
    .unwrap();
    let config = FamilyConfig::from_hf_value(&fixture["config"]).unwrap();
    let context = NumericContext::default();
    let prepared = prepared_prompt(&fixture, false);
    let admitted = eredu_architectures::nemotron_labs_diffusion::input::admit_input(
        &config,
        &prepared,
        &NumericInputInspector,
    )
    .unwrap();
    let mut runtime =
        ResidentRuntime::<PreparedCompositeArchitecture<Composite>, NumericBackend, State>::new(
            PreparedCompositeArchitecture::new(Composite::new(config.clone(), &context).unwrap()),
            &context,
        )
        .unwrap();
    populate_vision(
        runtime.architecture_mut().inner_mut().static_modules_mut(),
        &fixture,
        None,
    );
    for unit in runtime.units_mut().iter_mut().flatten() {
        populate_vision(unit, &fixture, None);
    }
    struct Ingress(Option<NumericTensor>);
    impl
        eredu_runtime::LayeredTraversalHook<
            NumericBackend,
            eredu_architectures::nemotron_labs_diffusion::composite::ForwardContext<NumericTensor>,
            Error,
        > for Ingress
    {
        fn observes_activations(&self) -> bool {
            true
        }
        fn observe_activation(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
            if path == "model.vision_tower.ln_pre.output" {
                assert!(self.0.is_none(), "image ingress is observed once");
                self.0 = Some(value.clone());
            }
            Ok(())
        }
    }
    let mut ingress = Ingress(None);
    runtime
        .forward_with_traversal_hook(
            PreparedCompositeInput::new(&prepared, &admitted).unwrap(),
            &mut composite_state(&config),
            &context,
            &mut ingress,
        )
        .unwrap();
    assert_tensor_close(
        &ingress.0.expect("prepared adapter exposes image ingress"),
        &tensor(&fixture["outputs"]["embedding"]),
        "independent Pixtral ingress",
    );
}

#[test]
fn vlm_image_markers_and_cached_modes_match_independent_context_in_resident_bounded_and_tp2() {
    let fixture: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON,
    )
    .unwrap();
    let mut config_json = fixture["config"].clone();
    config_json["num_key_value_heads"] = 2.into();
    let config = FamilyConfig::from_hf_value(&config_json).unwrap();
    let context = NumericContext::default();
    let prepared = prepared_prompt(&fixture, false);
    let admitted = eredu_architectures::nemotron_labs_diffusion::input::admit_input(
        &config,
        &prepared,
        &NumericInputInspector,
    )
    .unwrap();
    assert_eq!(admitted.decoder_positions(), 13);
    let mut resident = ResidentRuntime::<Composite, NumericBackend, State>::new(
        Composite::new(config.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    populate_vision(
        resident.architecture_mut().static_modules_mut(),
        &fixture,
        None,
    );
    for unit in resident.units_mut().iter_mut().flatten() {
        populate_vision(unit, &fixture, None);
    }
    let storage = composite_storage(&resident);
    let mut state = composite_state(&config);
    let actual = resident
        .forward(
            PreparedCompositeInput::new(&prepared, &admitted).unwrap(),
            &mut state,
            &context,
        )
        .unwrap();

    // This sequence and these replacement positions are independently enumerated
    // from the two published image grids, including the intermediate row break.
    let ids = NumericTensor::token_ids(&[1, 18, 19, 19, 21, 4, 5, 18, 19, 20, 19, 21, 7]);
    let mut reference =
        decoder::LayeredModel::<NumericBackend, FamilyConfig>::new(config.clone(), &context)
            .unwrap();
    let mut embedded = reference
        .static_modules_mut()
        .embed(&ids, &context)
        .unwrap();
    let features = tensor(&fixture["outputs"]["projected"]);
    for (row, position) in [2usize, 3, 8, 10].into_iter().enumerate() {
        embedded.data[position * 6..(position + 1) * 6]
            .copy_from_slice(&features.data[row * 6..(row + 1) * 6]);
    }
    let mut reference_state = composite_state(&config);
    let mut forward = reference
        .begin_embedded(embedded, None, &mut reference_state, &context)
        .unwrap();
    for index in 0..2 {
        let mut unit = reference.construct_unit(index, &context).unwrap();
        forward.hidden = reference
            .forward_unit(
                0,
                index,
                &mut unit,
                &forward.hidden,
                &mut reference_state,
                &mut forward.context,
                &context,
            )
            .unwrap();
    }
    let expected = reference
        .finish_forward(
            &forward.hidden,
            &mut reference_state,
            &forward.context,
            &context,
        )
        .unwrap();
    assert_tensor_close(
        &actual,
        &expected,
        "independently enumerated image placement and causal context",
    );

    let mut bounded_model = Composite::new(config.clone(), &context).unwrap();
    populate_vision(bounded_model.static_modules_mut(), &fixture, None);
    let mut bounded = LayerwiseRuntime::<Composite, NumericBackend, State, _>::new(
        bounded_model,
        FixturePolicy {
            lifecycle: RebuildingUnitPolicy::default(),
            fixture: Arc::new(fixture.clone()),
        },
    );
    let mut bounded_state = composite_state(&config);
    assert_tensor_close(
        &bounded
            .forward(
                PreparedCompositeInput::new(&prepared, &admitted).unwrap(),
                &mut bounded_state,
                &context,
            )
            .unwrap(),
        &expected,
        "individually rebuilt image/text units",
    );
    let projected = prepared_prompt(&fixture, true);
    let projected_admission = eredu_architectures::nemotron_labs_diffusion::input::admit_input(
        &config,
        &projected,
        &NumericInputInspector,
    )
    .unwrap();
    let mut projected_state = composite_state(&config);
    assert_tensor_close(
        &resident
            .forward(
                PreparedCompositeInput::new(&projected, &projected_admission).unwrap(),
                &mut projected_state,
                &context,
            )
            .unwrap(),
        &expected,
        "projected images retain identical marker and cache positions",
    );

    let modes = [
        (vec![7, 0, 0], AttentionInvocation::DIFFUSION),
        (vec![8], AttentionInvocation::CAUSAL),
        (vec![8, 0, 0], AttentionInvocation::DRAFT),
        (vec![9], AttentionInvocation::CAUSAL),
    ];
    let mut outputs = vec![expected];
    for (ids, mode) in &modes {
        let prepared = prepared_text(ids);
        let admitted = eredu_architectures::nemotron_labs_diffusion::input::admit_input(
            &config,
            &prepared,
            &NumericInputInspector,
        )
        .unwrap();
        let before = state.layer(0).unwrap().offset;
        let actual = resident
            .forward(
                PreparedCompositeInput::new(&prepared, &admitted)
                    .unwrap()
                    .with_attention_invocation(*mode),
                &mut state,
                &context,
            )
            .unwrap();
        let bounded_output = bounded
            .forward(
                PreparedCompositeInput::new(&prepared, &admitted)
                    .unwrap()
                    .with_attention_invocation(*mode),
                &mut bounded_state,
                &context,
            )
            .unwrap();
        assert_tensor_close(
            &actual,
            &bounded_output,
            "bounded shared text invocation retains media cache",
        );
        assert_eq!(
            state.layer(0).unwrap().offset,
            before
                + if *mode == AttentionInvocation::CAUSAL {
                    1
                } else {
                    0
                }
        );
        outputs.push(actual);
    }
    let groups: Vec<_> =
        eredu_architectures::nemotron_labs_diffusion::parallel::composite_parameter_description(
            &config,
        )
        .unwrap()
        .groups()
        .iter()
        .map(|group| group.group().clone())
        .collect();
    let world = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let fixture = &fixture;
                let config = &config;
                let groups = &groups;
                let modes = &modes;
                let outputs = &outputs;
                let world = Arc::clone(&world);
                scope.spawn(move || {
                    let layout = numeric_local_layout(groups, 2, rank).unwrap();
                    let context = NumericContext::with_local_layout(layout.clone());
                    let parallel = NumericParallelContext::new(rank, world);
                    let model =
                        Composite::new_parallel(config.clone(), layout.clone(), &context).unwrap();
                    let mut state = State::create(model.state_layout().unwrap(), |_, _| {
                        Ok::<_, Error>(NumericCache::new(None))
                    })
                    .unwrap();
                    let mut runtime =
                        ResidentRuntime::<Composite, NumericBackend, State>::new(model, &context)
                            .unwrap();
                    populate_vision(
                        runtime.architecture_mut().static_modules_mut(),
                        fixture,
                        Some(&layout),
                    );
                    for unit in runtime.units_mut().iter_mut().flatten() {
                        populate_vision(unit, fixture, Some(&layout));
                    }
                    let (model, units) = runtime.into_parts();
                    let mut runtime = LayerwiseRuntime::<Composite, NumericBackend, State, _>::new(
                        model,
                        ResidentUnitWindow::new(units),
                    );
                    let prepared = prepared_prompt(fixture, false);
                    let admitted =
                        eredu_architectures::nemotron_labs_diffusion::input::admit_input(
                            config,
                            &prepared,
                            &NumericInputInspector,
                        )
                        .unwrap();
                    assert_tensor_close(
                        &runtime
                            .forward_parallel(
                                PreparedCompositeInput::new(&prepared, &admitted).unwrap(),
                                &mut state,
                                &parallel,
                                &context,
                            )
                            .unwrap(),
                        &outputs[0],
                        "joint Pixtral/projector/text TP2 context",
                    );
                    let waves = <Composite as eredu_architectures::composite_execution::CompositeArchitecture<NumericBackend, State>>::prepared_group_collective_waves(
                        runtime.architecture(), 0, PreparedCompositeInput::new(&prepared, &admitted).unwrap(), 2, &[0..1, 1..2, 0..0],
                    ).unwrap().unwrap();
                    assert_eq!(waves.iter().map(Vec::len).collect::<Vec<_>>(), [2, 2, 0], "idle vision stages declare no tensor reductions");
                    let expected: Vec<_> = waves.into_iter().flatten().map(|operation| match operation {
                        eredu_architectures::composite_execution::CompositeTensorCollective::Sum { shape } => (NumericCollectiveKind::Sum, shape),
                        eredu_architectures::composite_execution::CompositeTensorCollective::Gather { .. } => panic!("vision layer wave contains a readout gather"),
                    }).collect();
                    let actual: Vec<_> = parallel.trace().into_iter().take(4).map(|operation| (operation.kind, operation.input_shape)).collect();
                    assert_eq!(actual, expected, "pipeline vision schedule describes the actual nonzero TP equations");
                    for (index, (ids, mode)) in modes.iter().enumerate() {
                        let prepared = prepared_text(ids);
                        let admitted =
                            eredu_architectures::nemotron_labs_diffusion::input::admit_input(
                                config,
                                &prepared,
                                &NumericInputInspector,
                            )
                            .unwrap();
                        assert_tensor_close(
                            &runtime
                                .forward_parallel(
                                    PreparedCompositeInput::new(&prepared, &admitted)
                                        .unwrap()
                                        .with_attention_invocation(*mode),
                                    &mut state,
                                    &parallel,
                                    &context,
                                )
                                .unwrap(),
                            &outputs[index + 1],
                            "TP2 media prefix survives text invocation changes",
                        );
                    }
                })
            })
            .collect();
        assert_eq!(
            composite_storage(&resident),
            storage,
            "image-conditioned attention alternation replaces no base or vision storage"
        );
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

#[test]
fn vlm_rejects_conflicting_image_positions_and_projected_patch_counts() {
    use eredu_core::{CapabilityError, InputExtent, InputModality};
    use eredu_runtime::{PreparedInputPart, PreparedInputPayload, PreparedModelInput};
    let fixture: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON,
    )
    .unwrap();
    let config = FamilyConfig::from_hf_value(&fixture["config"]).unwrap();
    for projected in [false, true] {
        let prompt = prepared_prompt(&fixture, projected);
        let mut parts = prompt.parts().to_vec();
        let image = &parts[1];
        let payload = if projected {
            PreparedInputPayload::Embeddings(image.payload().value().axis_slice(1, 0, 1))
        } else {
            image.payload().clone()
        };
        let extent = if projected {
            InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 4,
            }
        } else {
            InputExtent::PatchGrid {
                time: 1,
                height: 4,
                width: 2,
            }
        };
        parts[1] = PreparedInputPart::new_with_extents(
            InputModality::Image,
            payload,
            image
                .metadata()
                .iter()
                .map(|(key, value)| (*key, value.clone())),
            [extent],
        )
        .unwrap();
        let malformed = PreparedModelInput::new(parts, |tensor: &NumericTensor| {
            eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
        })
        .unwrap();
        assert!(
            matches!(
                eredu_architectures::nemotron_labs_diffusion::input::admit_input(
                    &config,
                    &malformed,
                    &NumericInputInspector
                ),
                Err(CapabilityError::UnsupportedInput { .. })
            ),
            "image identity cannot silently change rotary positions or marker counts"
        );
    }
}

fn tensor(value: &serde_json::Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect(),
    )
}
fn populate<M: Parameterized<NumericTensor>>(module: &mut M, fixture: &serde_json::Value) {
    struct Populate<'f>(&'f serde_json::Value);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            let expected = tensor(&self.0[metadata.id.as_str()]);
            assert_eq!(value.shape(), expected.shape(), "{}", metadata.id);
            *value = expected;
        }
    }
    module.visit_parameters_mut(&mut Populate(&fixture["parameters"]));
}

#[test]
fn pixtral_head_and_channel_shards_match_published_image_outputs() {
    let fixture: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON,
    )
    .unwrap();
    let config = FamilyConfig::from_hf_value(&fixture["config"]).unwrap();
    let vision = config.vision.as_ref().unwrap();
    let groups: Vec<_> = (0..vision.num_hidden_layers as usize)
        .flat_map(|index| {
            eredu_architectures::nemotron_labs_diffusion::parallel::vision_layer_groups(
                vision, index,
            )
            .unwrap()
        })
        .collect();
    let world = NumericParallelGroup::new(2);
    let mut inconsistent = numeric_local_layout(&groups, 2, 0).unwrap();
    let other = numeric_local_layout(&groups, 2, 1).unwrap();
    let key = "model.vision_tower.transformer.layers.0.attention.k_proj.weight";
    inconsistent.insert(key.into(), other.tensor(key).unwrap().clone());
    assert!(
        ImageLayer::<NumericBackend>::new_parallel(
            vision,
            0,
            &inconsistent,
            &NumericContext::with_local_layout(inconsistent.clone())
        )
        .is_err(),
        "equal-sized Q/K shards must select the same global heads"
    );
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let fixture = &fixture;
                let config = &config;
                let groups = &groups;
                let world = Arc::clone(&world);
                scope.spawn(move || {
                    let layout = numeric_local_layout(groups, 2, rank).unwrap();
                    let context = NumericContext::with_local_layout(layout.clone());
                    let parallel = NumericParallelContext::new(rank, world);
                    let mut image =
                        ImageEmbedding::<NumericBackend>::new(config, &context).unwrap();
                    populate(&mut image, fixture);
                    let patches = tensor(&fixture["patches"]);
                    let (mut hidden, state) = image
                        .begin(
                            ImageInput {
                                patches: &patches,
                                grids: &[(2, 4), (4, 2)],
                            },
                            &context,
                        )
                        .unwrap();
                    for index in 0..config.vision.as_ref().unwrap().num_hidden_layers as usize {
                        let mut layer = ImageLayer::<NumericBackend>::new_parallel(
                            config.vision.as_ref().unwrap(),
                            index,
                            &layout,
                            &context,
                        )
                        .unwrap();
                        struct PopulateShard<'f> {
                            fixture: &'f serde_json::Value,
                            layout: &'f LocalModelLayout,
                        }
                        impl<'a> ParameterVisitorMut<'a, NumericTensor> for PopulateShard<'_> {
                            fn visit_mut(
                                &mut self,
                                metadata: ParameterMetadata,
                                value: &'a mut NumericTensor,
                            ) {
                                let global =
                                    tensor(&self.fixture["parameters"][metadata.id.as_str()]);
                                let placement = self.layout.tensor(metadata.id.as_str()).unwrap();
                                let selected =
                                    select_parameter(&global, placement.placement()).unwrap();
                                assert_eq!(value.shape(), selected.shape(), "{}", metadata.id);
                                *value = selected;
                            }
                        }
                        layer.visit_parameters_mut(&mut PopulateShard {
                            fixture,
                            layout: &layout,
                        });
                        hidden = layer
                            .forward_parallel(&hidden, &state, &parallel, &context)
                            .unwrap();
                        assert_tensor_close(
                            &hidden,
                            &tensor(&fixture["outputs"][format!("layer.{index}")]),
                            "published Pixtral TP2 layer",
                        );
                    }
                    let mut projector =
                        ImageProjector::<NumericBackend>::new(config, &context).unwrap();
                    populate(&mut projector, fixture);
                    let projected = projector.finish(&hidden, &state, &context).unwrap();
                    assert_tensor_close(
                        &projected,
                        &tensor(&fixture["outputs"]["projected"]),
                        "published Pixtral TP2 projector",
                    );
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

#[test]
fn pixtral_layers_and_spatial_projector_match_published_equations() {
    let fixture: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON,
    )
    .unwrap();
    let config = FamilyConfig::from_hf_value(&fixture["config"]).unwrap();
    let context = NumericContext::default();
    let mut image = ImageEmbedding::<NumericBackend>::new(&config, &context).unwrap();
    populate(&mut image, &fixture);
    let patches = tensor(&fixture["patches"]);
    let grids = [(2, 4), (4, 2)];
    let (mut hidden, state) = image
        .begin(
            ImageInput {
                patches: &patches,
                grids: &grids,
            },
            &context,
        )
        .unwrap();
    assert_tensor_close(
        &hidden,
        &tensor(&fixture["outputs"]["embedding"]),
        "published patch convolution and pre-norm",
    );
    let mut layers = Vec::new();
    for index in 0..config.vision.as_ref().unwrap().num_hidden_layers as usize {
        let mut layer =
            ImageLayer::<NumericBackend>::new(config.vision.as_ref().unwrap(), index, &context)
                .unwrap();
        populate(&mut layer, &fixture);
        hidden = layer.forward(&hidden, &state, &context).unwrap();
        assert_tensor_close(
            &hidden,
            &tensor(&fixture["outputs"][format!("layer.{index}")]),
            "published two-axis attention and SwiGLU",
        );
        layers.push(layer);
    }
    let mut projector = ImageProjector::<NumericBackend>::new(&config, &context).unwrap();
    populate(&mut projector, &fixture);
    let projected = projector.finish(&hidden, &state, &context).unwrap();
    assert_tensor_close(
        &projected,
        &tensor(&fixture["outputs"]["projected"]),
        "published normalize/unfold/GELU projector",
    );
    // Different images cannot influence each other's attention or spatial merger.
    let mut altered = patches.clone();
    for value in &mut altered.data[8 * 12..] {
        *value = -3.0 * (*value) + 0.73;
    }
    let (mut other, other_state) = image
        .begin(
            ImageInput {
                patches: &altered,
                grids: &grids,
            },
            &context,
        )
        .unwrap();
    for layer in &mut layers {
        other = layer.forward(&other, &other_state, &context).unwrap();
    }
    let other = projector.finish(&other, &other_state, &context).unwrap();
    assert_tensor_exact(
        &projected
            .index(&[Index::Full, Index::Range(0, 2), Index::Full], &context)
            .unwrap(),
        &other
            .index(&[Index::Full, Index::Range(0, 2), Index::Full], &context)
            .unwrap(),
        "independent image attention and merger spans",
    );
    assert_ne!(&projected.data[12..], &other.data[12..]);
}

#[test]
fn pixtral_internal_hooks_consume_interventions_at_independent_media_extents() {
    use eredu_runtime::ActivationObserver;
    let fixture: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON,
    )
    .unwrap();
    let config = FamilyConfig::from_hf_value(&fixture["config"]).unwrap();
    let context = NumericContext::default();
    let mut image = ImageEmbedding::<NumericBackend>::new(&config, &context).unwrap();
    populate(&mut image, &fixture);
    let patches = tensor(&fixture["patches"]);
    let (hidden, state) = image
        .begin(
            ImageInput {
                patches: &patches,
                grids: &[(2, 4), (4, 2)],
            },
            &context,
        )
        .unwrap();
    struct Hooks {
        zero: &'static str,
        residual: Option<NumericTensor>,
        records: Vec<(String, Vec<i32>)>,
    }
    impl ActivationObserver<NumericTensor, Error> for Hooks {
        fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
            if path.ends_with("attention.residual.effective") {
                self.residual = Some(value.clone());
            }
            self.records.push((path.into(), value.shape().to_vec()));
            Ok(())
        }
        fn intervene(
            &mut self,
            path: &str,
            value: &NumericTensor,
        ) -> Result<Option<NumericTensor>, Error> {
            if path == self.zero {
                let mut zero = value.clone();
                zero.data.fill(0.0);
                Ok(Some(zero))
            } else {
                Ok(None)
            }
        }
    }
    let mut layer =
        ImageLayer::<NumericBackend>::new(config.vision.as_ref().unwrap(), 0, &context).unwrap();
    populate(&mut layer, &fixture);
    let mut hooks = Hooks {
        zero: "model.vision_tower.transformer.layers.0.feed_forward.output",
        residual: None,
        records: vec![],
    };
    let output = layer
        .forward_observed(&hidden, &state, None, &context, &mut hooks)
        .unwrap();
    assert_tensor_exact(
        &output,
        hooks.residual.as_ref().unwrap(),
        "zero MLP write preserves the independent attention residual",
    );
    assert_eq!(hooks.records.len(), 12);
    assert!(hooks.records.iter().all(|(_, shape)| shape == &[1, 16, 8]));
    let mut projector = ImageProjector::<NumericBackend>::new(&config, &context).unwrap();
    populate(&mut projector, &fixture);
    let mut hooks = Hooks {
        zero: "model.multi_modal_projector.linear_1.output",
        residual: None,
        records: vec![],
    };
    let output = projector
        .finish_observed(&output, &state, &context, &mut hooks)
        .unwrap();
    assert_eq!(output.shape(), &[1, 4, 6]);
    assert!(
        output.data.iter().all(|value| *value == 0.0),
        "zero projection is consumed by GELU and bias-free readout"
    );
    assert_eq!(hooks.records.len(), 10);
    assert_eq!(hooks.records[0].1, [1, 16, 8]);
    assert_eq!(hooks.records[2].1, [1, 4, 8]);
    assert_eq!(hooks.records[4].1, [1, 4, 6]);
}

#[test]
fn vlm_pipeline_cuts_keep_vision_projector_and_text_storage_local_across_all_modes() {
    use eredu_architectures::decoder::Config;
    use eredu_architectures::nemotron_labs_diffusion::composite::partition::{
        Model as Part, geometry,
    };
    use eredu_architectures::nemotron_labs_diffusion::partitioned::TextBoundarySchema;
    use eredu_runtime::{
        ArchitecturePartition, ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule,
        LayeredPartitionInput, LayeredPartitionOutput, PartitionOwnership,
        PartitionedLayeredArchitecture,
    };
    let fixture: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON,
    )
    .unwrap();
    let mut value = fixture["config"].clone();
    value["num_key_value_heads"] = 2.into();
    let config = FamilyConfig::from_hf_value(&value).unwrap();
    let modes = [
        (vec![7, 0, 0], AttentionInvocation::DIFFUSION),
        (vec![8, 0, 0], AttentionInvocation::DRAFT),
        (vec![8], AttentionInvocation::CAUSAL),
        (vec![9, 0, 0], AttentionInvocation::DRAFT),
        (vec![9, 0, 0], AttentionInvocation::DIFFUSION),
        (vec![9], AttentionInvocation::CAUSAL),
    ];
    let context = NumericContext::default();
    let mut reference = ResidentRuntime::<Composite, NumericBackend, State>::new(
        Composite::new(config.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    populate_vision(
        reference.architecture_mut().static_modules_mut(),
        &fixture,
        None,
    );
    for unit in reference.units_mut().iter_mut().flatten() {
        populate_vision(unit, &fixture, None);
    }
    let mut reference_state = composite_state(&config);
    let prepared = prepared_prompt(&fixture, false);
    let admitted = eredu_architectures::nemotron_labs_diffusion::input::admit_input(
        &config,
        &prepared,
        &NumericInputInspector,
    )
    .unwrap();
    let mut expected = vec![
        reference
            .forward(
                PreparedCompositeInput::new(&prepared, &admitted).unwrap(),
                &mut reference_state,
                &context,
            )
            .unwrap(),
    ];
    for (ids, mode) in &modes {
        let prepared = prepared_text(ids);
        let admitted = eredu_architectures::nemotron_labs_diffusion::input::admit_input(
            &config,
            &prepared,
            &NumericInputInspector,
        )
        .unwrap();
        expected.push(
            reference
                .forward(
                    PreparedCompositeInput::new(&prepared, &admitted)
                        .unwrap()
                        .with_attention_invocation(*mode),
                    &mut reference_state,
                    &context,
                )
                .unwrap(),
        );
    }
    let description =
        eredu_architectures::nemotron_labs_diffusion::parallel::composite_parameter_description(
            &config,
        )
        .unwrap();
    let groups: Vec<_> = description
        .groups()
        .iter()
        .map(|group| group.group().clone())
        .collect();
    for tensor_parts in [1, 2] {
        let world = NumericParallelGroup::new(tensor_parts);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..tensor_parts).map(|rank| {
                let world = Arc::clone(&world); let config = &config; let fixture = &fixture; let description = &description; let groups = &groups; let expected = &expected; let modes = &modes;
                scope.spawn(move || {
                    let layout = numeric_local_layout(groups, tensor_parts, rank).unwrap();
                    let context = NumericContext::with_local_layout(layout.clone());
                    let parallel = NumericParallelContext::new(rank, world);
                    let parallel_ref = (tensor_parts > 1).then_some(&parallel);
                    let mut parts = Vec::new(); let mut states = Vec::new(); let mut units = Vec::new();
                    for stage in 0..4 {
                        let vision = match stage { 0 => 0..1, 1 => 1..2, _ => 0..0 };
                        let text = (stage >= 2).then(|| stage-2..stage-1);
                        let local = geometry(config, layout.clone(), vision.clone(), stage == 2, text.clone()).unwrap();
                        let mut ranges = Vec::new();
                        if !vision.is_empty() { ranges.push(("vision", vision.clone())); }
                        if stage == 2 { ranges.push(("vision_projection", 0..1)); }
                        if let Some(text) = text { ranges.push((decoder::TEXT_DECODER_EXECUTION_GROUP, text)); }
                        let roles: Vec<&str> = match stage { 0 => vec!["vision_embedding"], 2 => vec!["embedding"], 3 => vec!["norm", "output"], _ => vec![] };
                        let selected = ArchitecturePartition::from_description(description, ranges, PartitionOwnership::new(stage == 0, stage == 3, roles).unwrap(), local.complete_state(), &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(2, 0..local.complete_state().len())]), local.clone(), TextBoundarySchema::new(config.hidden_size())).unwrap();
                        let mut model = Part::<NumericBackend>::new(config.clone(), description, &selected, &context).unwrap();
                        let state = match selected.state() { Some(state) => State::create(state.layout().clone(), |_, _| Ok::<_, Error>(NumericCache::new(None))).unwrap(), None => State::stateless() };
                        populate_vision(model.static_modules_mut(), fixture, Some(&layout));
                        let mut owned = Vec::new();
                        for address in selected.units() { let mut unit = model.construct_unit(address.group(), address.index(), &context).unwrap(); populate_vision(&mut unit, fixture, Some(&layout)); owned.push((address.group(), address.index(), unit)); }
                        assert!(model.construct_unit(0, 2, &context).is_err());
                        if stage < 2 { assert!(state.as_ref().is_empty(), "vision owners allocate no fake text cache"); }
                        parts.push(model); states.push(state); units.push(owned);
                    }
                    let storage = |parts: &[Part<NumericBackend>], units: &[Vec<(usize,usize,CompositeUnit)>]| {
                        struct Ids(BTreeMap<String,usize>);
                        impl<'a> ParameterVisitor<'a,NumericTensor> for Ids {
                            fn visit(&mut self, metadata: ParameterMetadata, value: &'a NumericTensor) { assert!(self.0.insert(metadata.id.to_string(),value.data.as_ptr() as usize).is_none(), "pipeline owners cannot duplicate base storage"); }
                        }
                        let mut ids = Ids(BTreeMap::new());
                        for part in parts { part.static_modules().visit_parameters(&mut ids); }
                        for unit in units.iter().flatten() { unit.2.visit_parameters(&mut ids); }
                        ids.0
                    };
                    let before = storage(&parts, &units);
                    let expected_ids: BTreeSet<_> = description.groups().iter().flat_map(|group| group.group().members().iter().map(|member| member.target().to_owned())).collect();
                    assert_eq!(before.keys().cloned().collect::<BTreeSet<_>>(), expected_ids);
                    for invocation in 0..=modes.len() {
                        let prepared = if invocation == 0 { prepared_prompt(fixture, false) } else { prepared_text(&modes[invocation-1].0) };
                        let admitted = eredu_architectures::nemotron_labs_diffusion::input::admit_input(config, &prepared, &NumericInputInspector).unwrap();
                        let mode = if invocation == 0 { AttentionInvocation::CAUSAL } else { modes[invocation-1].1 };
                        let mut frames = Vec::new();
                        for (part,state) in parts.iter_mut().zip(states.iter_mut()) { frames.push(part.begin_forward(PreparedCompositeInput::new(&prepared,&admitted).unwrap().with_attention_invocation(mode),state,&context).unwrap()); }
                        let mut projected = None;
                        if invocation == 0 {
                            let mut hidden = frames[0].hidden.clone();
                            for stage in 0..2 {
                                hidden = parts[stage].begin_execution_group(0,&hidden,&[],&mut states[stage],&mut frames[stage].context,&context).unwrap();
                                let (group,index,unit) = &mut units[stage][0];
                                hidden = match parallel_ref { Some(parallel) => parts[stage].forward_unit_parallel(*group,*index,unit,&hidden,&mut states[stage],&mut frames[stage].context,parallel,&context), None => parts[stage].forward_unit(*group,*index,unit,&hidden,&mut states[stage],&mut frames[stage].context,&context) }.unwrap();
                                let destination = if stage == 0 { 0 } else { 1 };
                                let schema = eredu_architectures::nemotron_labs_diffusion::composite::group_boundary_schema(config,0,destination).unwrap().resolve(1,hidden.dim(1)).unwrap();
                                assert_eq!(schema.primary().shape(), hidden.shape);
                            }
                            let initial = frames[2].hidden.clone();
                            let mut hidden = parts[2].begin_execution_group(1,&initial,&[&hidden],&mut states[2],&mut frames[2].context,&context).unwrap();
                            let (_,_,unit) = &mut units[2][0];
                            hidden = parts[2].forward_unit(1,0,unit,&hidden,&mut states[2],&mut frames[2].context,&context).unwrap();
                            let schema = eredu_architectures::nemotron_labs_diffusion::composite::group_boundary_schema(config,1,2).unwrap().resolve(1,hidden.dim(1)).unwrap();
                            assert_eq!(schema.primary().shape(), hidden.shape);
                            projected = Some(hidden);
                        }
                        let dependencies: Vec<_> = projected.iter().collect();
                        let initial = frames[2].hidden.clone();
                        let mut hidden = match parallel_ref { Some(parallel) => parts[2].begin_execution_group_parallel(2,&initial,&dependencies,&mut states[2],&mut frames[2].context,parallel,&context), None => parts[2].begin_execution_group(2,&initial,&dependencies,&mut states[2],&mut frames[2].context,&context) }.unwrap();
                        let (_,_,unit) = units[2].last_mut().unwrap();
                        hidden = match parallel_ref { Some(parallel) => parts[2].forward_unit_parallel(2,0,unit,&hidden,&mut states[2],&mut frames[2].context,parallel,&context), None => parts[2].forward_unit(2,0,unit,&hidden,&mut states[2],&mut frames[2].context,&context) }.unwrap();
                        let boundary = parts[2].finish_partition(&hidden,&mut states[2],&frames[2].context,false,parallel_ref,&context).unwrap();
                        let LayeredPartitionOutput::Boundary { hidden, auxiliary } = boundary else { panic!("middle text owner returns a complete canvas boundary") };
                        let expected_state = states[3].layout().clone();
                        let mut forward = parts[3].begin_partition_observed(LayeredPartitionInput::Hidden { hidden, auxiliary },None,&mut states[3],&expected_state,1,parallel_ref,&context,&mut eredu_runtime::NoopObserver).unwrap();
                        let (_,_,unit) = &mut units[3][0];
                        let output = match parallel_ref { Some(parallel) => parts[3].forward_unit_parallel(2,1,unit,&forward.hidden,&mut states[3],&mut forward.context,parallel,&context), None => parts[3].forward_unit(2,1,unit,&forward.hidden,&mut states[3],&mut forward.context,&context) }.unwrap();
                        let output = parts[3].finish_partition(&output,&mut states[3],&forward.context,true,parallel_ref,&context).unwrap();
                        let LayeredPartitionOutput::Final { output, .. } = output else { panic!("output owner returns logits") };
                        assert_tensor_close(&output,&expected[invocation],"vision/projector/text PP and TP+PP preserve shared invocation equations");
                        let first_frontier = states[2].layer(0).unwrap().offset;
                        let second_frontier = states[3].layer(0).unwrap().offset;
                        assert_eq!(first_frontier, second_frontier);
                    }
                    assert_eq!(storage(&parts,&units),before,"mode alternation replaces no pipeline base storage");
                })
            }).collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }
}
