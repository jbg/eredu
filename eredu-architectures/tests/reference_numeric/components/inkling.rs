//! Component values feed Inkling's projections and its causal histories.
use super::*;
use eredu_architectures::inkling;

#[test]
fn relative_attention_rejects_invalid_geometry_before_backend_work() {
    let make = |shape: &[i32]| NumericTensor {
        shape: shape.to_vec(),
        data: Vec::new(),
        dtype: eredu_core::checkpoint::TensorDtype::F32,
        retirement_probe: None,
        publication_funding: None,
    };
    let mut q = make(&[1, 4, 3, 8]);
    let mut k = make(&[1, 2, 3, 8]);
    let profiles = make(&[1, 4, 3, 5]);
    let check = |q: &NumericTensor, k: &NumericTensor, query_offset, key_offset| {
        eredu_nn::RelativeAttentionInput {
            queries: q,
            keys: k,
            values: k,
            profiles: &profiles,
            query_offset,
            key_offset,
            window: None,
            log_scaling_floor: None,
            log_scaling_alpha: 1.0,
        }
        .validate()
    };
    assert!(check(&q, &k, 0, 0).is_ok());
    for axis in 0..4 {
        for invalid in [0, -1] {
            let original = k.shape[axis];
            k.shape[axis] = invalid;
            assert!(check(&q, &k, 0, 0).is_err());
            k.shape[axis] = original;
            let original = q.shape[axis];
            q.shape[axis] = invalid;
            assert!(check(&q, &k, 0, 0).is_err());
            q.shape[axis] = original;
        }
    }
    for (query, key) in [(-1, 0), (0, -1), (i32::MAX, 0), (0, i32::MAX)] {
        assert!(check(&q, &k, query, key).is_err());
    }
    assert!(check(&q, &k, i32::MAX - 3, i32::MAX - 3).is_ok());
}

#[test]
fn inkling_relative_attention_consumes_cache_owned_history() {
    struct BlockCache {
        resident: NumericCache,
        visible: Option<(NumericTensor, NumericTensor)>,
        calls: usize,
    }
    impl AttentionCache<NumericTensor> for BlockCache {
        fn uses_blockwise_attention(&self) -> bool {
            true
        }
        fn offset(&self) -> i32 {
            self.resident.offset()
        }
        fn max_size(&self) -> Option<i32> {
            self.resident.max_size()
        }
        fn update_for_attention(
            &mut self,
            keys: NumericTensor,
            values: NumericTensor,
            context: &NumericContext,
        ) -> Result<(NumericTensor, NumericTensor), Error> {
            self.visible = Some(self.resident.update_for_attention(
                keys.clone(),
                values.clone(),
                context,
            )?);
            Ok((keys, values))
        }
        fn attention(
            &mut self,
            _: AttentionRequest<'_, NumericTensor>,
            _: &NumericContext,
        ) -> Result<NumericTensor, Error> {
            panic!("relative profiles require the cache-aware relative operation")
        }
        fn relative_attention<B: NeuralBackend<Tensor = NumericTensor>>(
            &mut self,
            input: eredu_nn::RelativeAttentionInput<'_, NumericTensor>,
            context: &NumericContext,
        ) -> Result<NumericTensor, Error> {
            self.calls += 1;
            assert_eq!(
                input.keys.shape[2], input.queries.shape[2],
                "only submitted keys were returned"
            );
            let (keys, values) = self.visible.as_ref().unwrap();
            B::relative_attention(
                eredu_nn::RelativeAttentionInput {
                    keys,
                    values,
                    key_offset: self.resident.offset() - keys.shape[2],
                    ..input
                },
                context,
            )
        }
    }
    let config = routed_inkling_partition_fixture();
    let args = inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
    let context = NumericContext::default();
    for layer_index in 0..2 {
        let mut layer =
            inkling::DecoderLayer::<NumericBackend>::new(&args.text_config, layer_index, &context)
                .unwrap();
        layer.visit_parameters_mut(&mut Initialize);
        let window = args
            .text_config
            .layer_policy(layer_index)
            .unwrap()
            .attention
            .window()
            .map(|window| window.get() as i32);
        let mut resident = inkling::LayerState {
            attention: NumericCache::new(window),
            convolutions: Default::default(),
        };
        let mut block = inkling::LayerState {
            attention: BlockCache {
                resident: NumericCache::new(window),
                visible: None,
                calls: 0,
            },
            convolutions: Default::default(),
        };
        for (step, sequence) in [3, 2, 1, 1].into_iter().enumerate() {
            let input = NumericTensor::new(
                [1, sequence, 8],
                (0..sequence * 8)
                    .map(|i| ((i * 7 + step as i32 * 3) % 23) as f32 / 13.0 - 0.5)
                    .collect(),
            );
            let expected = layer
                .attention
                .forward(&input, Some(&mut resident), &context)
                .unwrap();
            let actual = layer
                .attention
                .forward(&input, Some(&mut block), &context)
                .unwrap();
            assert_tensor_close(&actual, &expected, "cache-owned relative history");
            assert_eq!(block.attention.calls, step + 1);
            assert_eq!(block.attention.offset(), resident.attention.offset());
        }
    }
}

#[derive(Default)]
struct Observer {
    values: BTreeMap<String, NumericTensor>,
    zero: Option<String>,
    routed_path: String,
    routed: Vec<UnitEvidence>,
    original_units: Option<NumericTensor>,
}

struct UnitEvidence {
    path: String,
    original: NumericTensor,
    effective: NumericTensor,
    groups: Vec<f32>,
    tokens: Vec<f32>,
    slots: Vec<f32>,
    coefficients: Vec<f32>,
}

impl ActivationObserver<NumericTensor, Error> for Observer {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        assert!(self.values.insert(path.into(), value.clone()).is_none());
        Ok(())
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        Ok((self.zero.as_deref() == Some(path)).then(|| {
            let mut value = value.clone();
            let width = *value.shape.last().unwrap() as usize;
            let selected = value.data.len() - width + 1;
            assert_ne!(value.data[selected], 0.0, "nonzero {path}");
            value.data[selected] = 0.0;
            value
        }))
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<NumericTensor>>, Error> {
        self.routed_path = path.into();
        Ok(Some(self))
    }
}

impl eredu_runtime::RoutedUnitObserver<NumericTensor> for Observer {
    fn observe(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert!(
            self.original_units
                .replace(batch.units.values.clone())
                .is_none()
        );
        Ok(())
    }
    fn intervene(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<Option<NumericTensor>, Error> {
        if self.zero.as_deref() != Some(&self.routed_path) {
            return Ok(None);
        }
        let mut value = batch.units.values.clone();
        let width = value.shape[1] as usize;
        let selected_token = batch.units.total_token_count - 1;
        for (row, token) in batch.units.token_indices.data.iter().enumerate() {
            if *token as usize == selected_token {
                assert_ne!(value.data[row * width + 1], 0.0);
                value.data[row * width + 1] = 0.0;
            }
        }
        Ok(Some(value))
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        let original = self.original_units.take().unwrap();
        let effective = batch.units.values.clone();
        let width = effective.shape[1] as usize;
        for (index, (&before, &after)) in original.data.iter().zip(&effective.data).enumerate() {
            let selected = self.zero.as_deref() == Some(&self.routed_path)
                && index % width == 1
                && batch.units.token_indices.data[index / width] as usize
                    == batch.units.total_token_count - 1;
            assert_eq!(after, if selected { 0.0 } else { before });
        }
        self.routed.push(UnitEvidence {
            path: self.routed_path.clone(),
            original,
            effective,
            groups: batch.units.group_indices.data.clone(),
            tokens: batch.units.token_indices.data.clone(),
            slots: batch.units.selection_indices.data.clone(),
            coefficients: batch.units.coefficients.data.clone(),
        });
        Ok(())
    }
}

struct Initialize;
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Initialize {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut NumericTensor,
    ) {
        let name = metadata.id().as_str();
        value.data = deterministic_values(
            &ParameterSpec::trainable(name).unwrap(),
            value.data.len(),
            name.contains("norm"),
        );
        if name.ends_with("global_scale") {
            value.data.fill(1.3);
        }
        if name.contains("sconv") {
            for (i, value) in value.data.iter_mut().enumerate() {
                *value = [0.11, -0.07, 0.23][i % 3] + (i / 3) as f32 * 0.002;
            }
        }
    }
}

#[derive(Default)]
struct Parameters(BTreeMap<String, NumericTensor>);
impl<'a> ParameterVisitor<'a, NumericTensor> for Parameters {
    fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a NumericTensor) {
        self.0.insert(metadata.id().as_str().into(), value.clone());
    }
}

fn convolution_reference(
    input: &NumericTensor,
    kernel: &NumericTensor,
    past: &mut Vec<f32>,
) -> NumericTensor {
    let width = input.shape[2] as usize;
    let start = past.len() / width;
    past.extend_from_slice(&input.data);
    let mut result = input.clone();
    for row in 0..input.shape[1] as usize {
        for column in 0..width {
            let mut sum = f64::from(input.data[row * width + column]);
            for tap in 0..3 {
                if let Some(position) = (start + row + tap).checked_sub(2) {
                    sum += f64::from(past[position * width + column])
                        * f64::from(kernel.data[column * 3 + tap]);
                }
            }
            result.data[row * width + column] = sum as f32;
        }
    }
    result
}

fn trial(
    sparse: bool,
    parallel: bool,
    observed: bool,
    mode: usize,
) -> (
    Vec<NumericTensor>,
    DeviceState<NumericBackend, NumericHybridLayerState>,
) {
    let mut config = routed_inkling_partition_fixture();
    if !sparse {
        config["text_config"]["mlp_layer_types"] = serde_json::json!(["dense", "dense"]);
    }
    let args = inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let architecture =
        inkling::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut state = DeviceState::<NumericBackend, _>::create(
        architecture.ingress_state_layout().unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let mut layer =
        inkling::DecoderLayer::<NumericBackend>::new(&args.text_config, 0, &context).unwrap();
    layer.visit_parameters_mut(&mut Initialize);
    let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    let transforms = graph
        .component_transforms
        .iter()
        .filter(|transform| transform.input.starts_with("model.layers.0."))
        .collect::<Vec<_>>();
    assert_eq!(transforms.len(), if sparse { 5 } else { 6 });
    let mut parameters = Parameters::default();
    layer.visit_parameters(&mut parameters);
    let mut transform_histories = BTreeMap::<String, Vec<f32>>::new();
    for transform in &transforms {
        assert!(graph.node(&transform.node_id).is_some());
        for path in [&transform.input, &transform.output]
            .into_iter()
            .chain(transform.effective_output.iter())
        {
            assert!(
                graph
                    .observations
                    .points
                    .iter()
                    .any(|point| &point.path == path),
                "declared transform boundary {path}"
            );
        }
    }
    let parallel = parallel.then(|| NumericParallelContext::new(0, NumericParallelGroup::new(1)));
    let mut outputs = Vec::new();
    let mut key_past = Vec::new();
    let mut value_past = Vec::new();
    let mut attention_past = Vec::new();
    let mut feed_forward_past = Vec::new();
    for (step, sequence) in [3, 1, 1].into_iter().enumerate() {
        let input = NumericTensor::new(
            [1, sequence, 8],
            (0..sequence * 8)
                .map(|i| ((i * 7 + step as i32 * 3) % 23) as f32 / 13.0 - 0.5)
                .collect(),
        );
        let zero = match mode {
            0 => None,
            1 => Some("model.layers.0.attention.channels"),
            2 => Some(if sparse {
                "model.layers.0.routing"
            } else {
                "model.layers.0.feed_forward.units"
            }),
            3 => Some(if sparse {
                "model.layers.0.shared.routing"
            } else {
                "model.layers.0.attention.input"
            }),
            4 => Some("model.layers.0.feed_forward.input"),
            _ => unreachable!(),
        };
        let mut observer = Observer {
            zero: zero.map(str::to_owned),
            ..Default::default()
        };
        let mut provider = eredu_runtime::ResidentExpertProvider;
        let output = match (parallel.as_ref(), observed) {
            (None, false) => layer.forward(&input, Some(&mut state.as_mut()[0]), &context),
            (Some(parallel), false) => {
                layer.forward_parallel(&input, Some(&mut state.as_mut()[0]), parallel, &context)
            }
            (None, true) => layer.forward_with_provider_instrumented(
                &input,
                Some(&mut state.as_mut()[0]),
                eredu_runtime::ExpertPass::Prefill,
                &mut provider,
                &context,
                &mut ComponentInstrumentation::new("model.layers.0", &mut observer),
            ),
            (Some(parallel), true) => layer.forward_parallel_with_provider_instrumented(
                &input,
                Some(&mut state.as_mut()[0]),
                eredu_runtime::ExpertPass::Prefill,
                &mut provider,
                parallel,
                &context,
                &mut ComponentInstrumentation::new("model.layers.0", &mut observer),
            ),
        }
        .unwrap();
        if observed {
            let get = |suffix: &str| &observer.values[&format!("model.layers.0.{suffix}")];
            let groups = graph
                .components
                .iter()
                .filter(|group| group.layer_index == 0)
                .collect::<Vec<_>>();
            assert_eq!(groups.len(), if sparse { 1 } else { 2 });
            for group in groups {
                let write = linear(
                    &observer.values[group.write_input.as_ref().unwrap()],
                    &parameters.0[&group.write_weight],
                    group.write_bias.as_ref().map(|name| &parameters.0[name]),
                )
                .unwrap();
                assert_tensor_close(
                    &write,
                    &observer.values[group.write_output.as_ref().unwrap()],
                    "discovered scalar write reconstructs actual projection",
                );
                for read in &group.reads {
                    let weight = &parameters.0[&read.weight];
                    assert!(
                        read.rows.row_range(group.count - 1).unwrap().end
                            <= weight.shape[0] as usize
                    );
                    if let Some(output) = &read.projection_output {
                        let projected = linear(
                            &observer.values[&format!("{}.effective", group.input)],
                            weight,
                            read.bias.as_ref().map(|name| &parameters.0[name]),
                        )
                        .unwrap();
                        assert_tensor_close(
                            &projected,
                            &observer.values[output],
                            "discovered attention read matches actual affine output",
                        );
                    }
                }
            }
            for transform in &transforms {
                use eredu_core::component::ComponentTensorTransformEquation;
                let input = &observer.values[&transform.input];
                let (reference, parameter) = match &transform.equation {
                    ComponentTensorTransformEquation::CausalDepthwiseConvolution {
                        kernel,
                        channels,
                        taps,
                        residual,
                        activation,
                    } => {
                        assert!(activation.is_none());
                        assert!(*residual);
                        assert_eq!(*taps, 3);
                        assert_eq!(input.shape[2] as usize, *channels);
                        let weight = &parameters.0[&kernel.parameter];
                        assert_eq!(weight.shape, [*channels as i32, 1, *taps as i32]);
                        (
                            convolution_reference(
                                input,
                                weight,
                                transform_histories.entry(transform.id.clone()).or_default(),
                            ),
                            kernel,
                        )
                    }
                    ComponentTensorTransformEquation::SharedGroupedProjection {
                        weight,
                        groups,
                        input_width,
                        output_width,
                    } => {
                        let table = &parameters.0[&weight.parameter];
                        assert_eq!(table.shape, [*input_width as i32, *output_width as i32]);
                        assert_eq!(input.shape[2] as usize, groups * input_width);
                        let rows = input.data.len() / input_width;
                        let mut values = Vec::new();
                        for row in 0..rows {
                            for column in 0..*output_width {
                                values.push(
                                    (0..*input_width)
                                        .map(|i| {
                                            f64::from(input.data[row * input_width + i])
                                                * f64::from(table.data[i * output_width + column])
                                        })
                                        .sum::<f64>() as f32,
                                );
                            }
                        }
                        (
                            NumericTensor::new(
                                [
                                    input.shape[0],
                                    input.shape[1],
                                    *groups as i32,
                                    *output_width as i32,
                                ],
                                values,
                            ),
                            weight,
                        )
                    }
                    ComponentTensorTransformEquation::LearnedScale { scale } => {
                        let weight = &parameters.0[&scale.parameter];
                        assert_eq!(weight.shape, [1]);
                        (input.multiply(weight, &context).unwrap(), scale)
                    }
                    equation => panic!("unexpected decoder transform {equation:?}"),
                };
                assert_eq!(parameter.parameter, parameter.shared_parameter);
                assert!(
                    graph
                        .parameter_groups
                        .iter()
                        .any(|group| group.id == parameter.parameter_group)
                );
                assert_tensor_close(
                    &reference,
                    &observer.values[&transform.output],
                    "discovered transform reconstructs actual output",
                );
            }

            for (role, projection, kernel, past) in [
                (
                    "key",
                    &layer.attention.key,
                    &layer.attention.key_convolution,
                    &mut key_past,
                ),
                (
                    "value",
                    &layer.attention.value,
                    &layer.attention.value_convolution,
                    &mut value_past,
                ),
            ] {
                let projected = linear(
                    get("attention.input.effective"),
                    &projection.weight,
                    projection.bias.as_ref().map(|bias| &bias.0),
                )
                .unwrap();
                assert_tensor_close(
                    &projected,
                    get(&format!("attention.{role}.projected")),
                    "Inkling current-position affine read",
                );
                let convolved = convolution_reference(&projected, kernel.weight.as_ref(), past);
                assert_tensor_close(
                    &convolved,
                    get(&format!("attention.{role}.convolved")),
                    "Inkling current-position causal read",
                );
            }
            let attention = linear(
                get("attention.channels.effective"),
                &layer.attention.output.weight,
                layer.attention.output.bias.as_ref().map(|bias| &bias.0),
            )
            .unwrap();
            assert_tensor_close(
                &attention,
                get("attention.write"),
                "Inkling attention component reconstruction",
            );
            if let inkling::FeedForward::Dense(dense) = &layer.feed_forward {
                let projection = linear(
                    get("feed_forward.units.effective"),
                    &dense.down.weight,
                    None,
                )
                .unwrap();
                assert_tensor_close(
                    &projection,
                    get("feed_forward.projection"),
                    "Inkling dense component reconstruction",
                );
                let scaled = projection
                    .multiply(dense.global_scale.as_ref(), &context)
                    .unwrap();
                assert_tensor_close(
                    &scaled,
                    get("feed_forward.write"),
                    "Inkling learned FFN scale",
                );
            }
            let attention = convolution_reference(
                get("attention.write.effective"),
                layer.attention_convolution.weight.as_ref(),
                &mut attention_past,
            );
            let feed_forward = convolution_reference(
                get("feed_forward.write.effective"),
                layer.feed_forward_convolution.weight.as_ref(),
                &mut feed_forward_past,
            );
            assert_tensor_close(
                &attention,
                get("attention.contribution"),
                "Inkling causal attention contribution",
            );
            assert_tensor_close(
                &feed_forward,
                get("feed_forward.contribution"),
                "Inkling causal FFN contribution",
            );
            let expected = input
                .add(&attention, &context)
                .unwrap()
                .add(&feed_forward, &context)
                .unwrap();
            assert_tensor_close(&expected, &output, "Inkling full residual reconstruction");
            assert_eq!(observer.routed.len(), if sparse { 2 } else { 0 });
            if let inkling::FeedForward::Sparse(_) = &layer.feed_forward {
                let mut sum = vec![0.0f64; input.data.len()];
                for evidence in &observer.routed {
                    if observer.zero.as_deref() == Some(&evidence.path) {
                        assert_ne!(evidence.original.data, evidence.effective.data);
                    }
                    let group = graph
                        .routed_components
                        .iter()
                        .find(|group| group.routing == evidence.path)
                        .unwrap();
                    assert_eq!(group.layer_index, 0);
                    let routes = group.routes_per_token.unwrap();
                    assert_eq!(
                        evidence.coefficients.len(),
                        input.shape[1] as usize * routes
                    );
                    let eredu_core::ObservationValueType::RoutedUnits { geometry, .. } = graph
                        .observations
                        .get(&group.activation)
                        .unwrap()
                        .value_type
                    else {
                        panic!("routed units")
                    };
                    assert_eq!(geometry.routes_per_token, routes as u64);
                    let eredu_core::component::RoutedComponentParameter::Packed { name } =
                        &group.write_weight
                    else {
                        panic!("packed fixture")
                    };
                    let weight = &parameters.0[&name.parameter];
                    assert_eq!(
                        weight.shape,
                        [group.expert_count as i32, 8, group.units_per_expert as i32]
                    );
                    let width = evidence.effective.shape[1] as usize;
                    for (row, units) in evidence.effective.data.chunks_exact(width).enumerate() {
                        let expert = evidence.groups[row] as usize;
                        assert!(expert < group.expert_count);
                        let coefficient =
                            f64::from(evidence.coefficients[evidence.slots[row] as usize]);
                        let token = evidence.tokens[row] as usize;
                        for channel in 0..8 {
                            let projection = units
                                .iter()
                                .zip(
                                    &weight.data[(expert * 8 + channel) * width
                                        ..(expert * 8 + channel + 1) * width],
                                )
                                .map(|(unit, weight)| f64::from(*unit) * f64::from(*weight))
                                .sum::<f64>();
                            sum[token * 8 + channel] += coefficient * projection;
                        }
                    }
                }
                assert_tensor_close(
                    &NumericTensor::new(
                        input.shape.clone(),
                        sum.into_iter().map(|value| value as f32).collect(),
                    ),
                    get("feed_forward.write"),
                    "Inkling routed/shared component reconstruction",
                );
            }
            for (path, original) in &observer.values {
                if let Some(effective) = observer.values.get(&format!("{path}.effective")) {
                    if observer.zero.as_deref() == Some(path) {
                        let selected =
                            original.data.len() - *original.shape.last().unwrap() as usize + 1;
                        for (i, (&before, &after)) in
                            original.data.iter().zip(&effective.data).enumerate()
                        {
                            assert_eq!(after, if i == selected { 0.0 } else { before });
                        }
                    } else {
                        assert_tensor_exact(original, effective, "unchanged component");
                    }
                }
            }
        }
        outputs.push(output);
    }
    (outputs, state)
}

#[test]
fn inkling_component_masks_feed_projections_and_convolution_histories() {
    for sparse in [false, true] {
        let (baseline, baseline_state) = trial(sparse, false, false, 0);
        for parallel in [false, true] {
            let (ordinary, ordinary_state) = trial(sparse, parallel, false, 0);
            let (captured, captured_state) = trial(sparse, parallel, true, 0);
            for (ordinary, captured) in ordinary.iter().zip(&captured) {
                assert_tensor_exact(ordinary, captured, "no-op instrumentation");
            }
            assert_state_exact(
                &ordinary_state,
                &captured_state,
                2,
                "no-op convolution state",
            );
            for (a, b) in ordinary.iter().zip(&baseline) {
                assert_tensor_close(a, b, "Inkling serial/TP1 components");
            }
            assert_state_exact(
                &baseline_state,
                &ordinary_state,
                2,
                "Inkling serial/TP1 state",
            );
            for mode in 1..=4 {
                let (masked, _) = trial(sparse, parallel, true, mode);
                for (actual, baseline) in masked.iter().zip(&ordinary) {
                    assert_ne!(
                        actual.data, baseline.data,
                        "causal mode {mode}, sparse={sparse}"
                    );
                }
            }
        }
    }
}

fn model_trial(sparse: bool, observed: bool, mode: usize) -> Vec<NumericTensor> {
    type Model = inkling::LayeredModel<NumericBackend>;
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    let mut config = routed_inkling_partition_fixture();
    config.as_object_mut().unwrap().remove("vision_config");
    config["text_config"]["logits_mup_width_multiplier"] = 1.7.into();
    config["text_config"]["unpadded_vocab_size"] = 13.into();
    if !sparse {
        config["text_config"]["mlp_layer_types"] = serde_json::json!(["dense", "dense"]);
    }
    let args = inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
    let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    let readout = graph.component_readout.as_ref().unwrap();
    let scale = graph
        .component_transforms
        .iter()
        .find_map(|transform| {
            if transform.input == format!("{}.effective", readout.normalized) {
                if let eredu_core::component::ComponentTensorTransformEquation::ConstantScale {
                    scale,
                } = transform.equation
                {
                    return Some(scale.value());
                }
            }
            None
        })
        .unwrap();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut model = Model::new(args, &context).unwrap();
    let layout = model.ingress_state_layout().unwrap();
    let statics =
        <Model as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut model);
    statics.visit_parameters_mut(&mut Initialize);
    let head = statics.output.weight.clone();
    let mut parameters = Parameters::default();
    statics.visit_parameters(&mut parameters);
    let normalize =
        |input: &NumericTensor, normalization: &eredu_core::component::ComponentNormalization| {
            assert_eq!(
                normalization.kind,
                eredu_core::component::ComponentNormalizationKind::Rms
            );
            assert_eq!(normalization.groups, 1);
            let gain = &parameters.0[normalization.gain.as_ref().unwrap()];
            let mut output = input.clone();
            let width = input.shape[2] as usize;
            for (source, target) in input
                .data
                .chunks_exact(width)
                .zip(output.data.chunks_exact_mut(width))
            {
                let denominator = (source
                    .iter()
                    .map(|value| f64::from(*value).powi(2))
                    .sum::<f64>()
                    / width as f64
                    + f64::from(normalization.epsilon.value()))
                .sqrt();
                for (i, value) in source.iter().enumerate() {
                    target[i] = (f64::from(*value)
                        * f64::from(gain.data[i] + normalization.gain_offset.value())
                        / denominator) as f32;
                }
            }
            output
        };
    let units = (0..2)
        .map(|index| {
            let mut unit = <Model as LayeredArchitecture<NumericBackend, State>>::build_unit(
                &model, 2, index, &context,
            )
            .unwrap();
            unit.visit_parameters_mut(&mut Initialize);
            unit
        })
        .collect();
    let mut runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(units));
    let mut state = State::create(layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let mut outputs = Vec::new();
    for ids in [vec![1, 2, 3], vec![4], vec![2]] {
        let tokens = NumericTensor::token_ids(&ids);
        let parts = [inkling::DecoderInputPart::Text(&tokens)];
        let input = inkling::ModelInput {
            parts: &parts,
            vision_patches: None,
            audio: None,
        };
        let zero = match mode {
            0 => None,
            1 => Some("readout.embedding"),
            2 => Some("model.layers.0.attention.channels"),
            3 => Some("readout.normalized"),
            4 => Some(if sparse {
                "model.layers.1.routing"
            } else {
                "model.layers.1.feed_forward.units"
            }),
            _ => unreachable!(),
        };
        let mut observer = Observer {
            zero: zero.map(str::to_owned),
            ..Default::default()
        };
        let output = if observed {
            runtime.forward_with_observer(input, &mut state, &context, &mut observer)
        } else {
            runtime.forward(input, &mut state, &context)
        }
        .unwrap();
        if observed {
            let get = |path: &str| &observer.values[path];
            let embedding = &parameters.0[&readout.embedding_weight];
            let width = embedding.shape[1] as usize;
            let values = ids
                .iter()
                .flat_map(|token| {
                    embedding.data[*token * width..(*token + 1) * width]
                        .iter()
                        .map(|value| *value * readout.embedding_scale.value())
                })
                .collect();
            let raw = NumericTensor::new([1, ids.len() as i32, width as i32], values);
            let expected = normalize(&raw, readout.embedding_normalization.as_ref().unwrap());
            assert_tensor_close(
                &expected,
                get(&readout.embedding),
                "Declared embedding normalization matches the actual residual base",
            );
            let mut residual = get(&format!("{}.effective", readout.embedding)).clone();
            for write in &readout.other_writes {
                residual = residual
                    .add(get(&write.effective_output), &context)
                    .unwrap();
            }
            assert_tensor_close(
                &residual,
                get(&readout.residual),
                "Inkling model residual reconstruction",
            );
            let normalized = normalize(get(&readout.residual), &readout.normalization);
            assert_tensor_close(
                &normalized,
                get(&readout.normalized),
                "Declared final norm matches actual output normalization",
            );
            let scaled = get(&format!("{}.effective", readout.normalized))
                .multiply_scalar(scale, &context)
                .unwrap();
            assert_tensor_exact(
                &scaled,
                get("readout.scaled"),
                "Inkling declared muP head scaling",
            );
            let scores = linear(get(readout.projection_input.as_ref().unwrap()), &head, None)
                .unwrap()
                .axis_slice(2, 0, 13);
            assert_tensor_close(
                &scores,
                get("readout.linear"),
                "Inkling selected score reconstruction",
            );
            assert_tensor_exact(
                &output,
                get("readout.linear.effective"),
                "Inkling effective readout reaches output",
            );
        }
        outputs.push(output);
    }
    outputs
}

#[test]
fn inkling_layered_component_and_readout_hooks_preserve_causal_execution() {
    for sparse in [false, true] {
        let baseline = model_trial(sparse, false, 0);
        let captured = model_trial(sparse, true, 0);
        for (baseline, captured) in baseline.iter().zip(&captured) {
            assert_tensor_exact(baseline, captured, "Inkling layered no-op components");
        }
        for mode in 1..=4 {
            let masked = model_trial(sparse, true, mode);
            for (baseline, masked) in baseline.iter().zip(&masked) {
                assert_ne!(
                    baseline.data, masked.data,
                    "causal Inkling layered mode {mode}, sparse={sparse}"
                );
            }
        }
    }
}

#[test]
fn inkling_prediction_fusion_components_preserve_double_norm_and_depth_state() {
    for chain_norm in [false, true] {
        let mut config = routed_inkling_partition_fixture();
        config.as_object_mut().unwrap().remove("vision_config");
        config["text_config"]["layer_types"] =
            serde_json::json!(["full_attention", "sliding_attention"]);
        config["mtp_config"] = serde_json::json!({
            "num_nextn_predict_layers": 2, "local_layer_ids": [1],
            "chain_hidden_post_norm": chain_norm,
        });
        let args = inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
        for depth in 0..2 {
            let trial = |observed, mode| {
                let context = NumericContext {
                    bind_checkpoint_values: true,
                    ..Default::default()
                };
                let mut model = inkling::MtpModel::<NumericBackend>::new(&args, &context)
                    .unwrap()
                    .unwrap();
                model.visit_parameters_mut(&mut Initialize);
                let mut state = DeviceState::<NumericBackend, _>::create(
                    inkling::mtp_state_layout(&args).unwrap().unwrap(),
                    |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
                )
                .unwrap();
                let path = format!("model.mtp.layers.{depth}");
                let zero = match mode {
                    0 => None,
                    1 => Some("prediction.hidden.first_normalized"),
                    2 => Some("prediction.hidden.normalized"),
                    3 => Some("prediction.embedding.normalized"),
                    4 => Some("prediction.fusion.output"),
                    5 => Some("transformer_block.attention.channels"),
                    6 => Some("transformer_block.feed_forward.units"),
                    7 => Some("prediction.readout.normalized"),
                    _ => unreachable!(),
                }
                .map(|suffix| format!("{path}.{suffix}"));
                let mut outputs = Vec::new();
                for (step, sequence) in [3, 1, 1].into_iter().enumerate() {
                    let hidden = NumericTensor::new(
                        [1, sequence, 8],
                        (0..sequence * 8)
                            .map(|index| ((index * 7 + step as i32 * 3) % 23) as f32 / 13.0 - 0.5)
                            .collect(),
                    );
                    let embedded = NumericTensor::new(
                        [1, sequence, 8],
                        (0..sequence * 8)
                            .map(|index| ((index * 11 + step as i32 * 5) % 29) as f32 / 17.0 - 0.4)
                            .collect(),
                    );
                    let tokens = NumericTensor::token_ids(&vec![2; sequence as usize]);
                    let mut observer = Observer {
                        zero: zero.clone(),
                        ..Default::default()
                    };
                    let output = if observed {
                        model.forward_step_instrumented(
                            &hidden,
                            &embedded,
                            &tokens,
                            depth + 2,
                            state.as_mut(),
                            &context,
                            &mut ComponentInstrumentation::new(&path, &mut observer),
                        )
                    } else {
                        model.forward_step(
                            &hidden,
                            &embedded,
                            &tokens,
                            depth + 2,
                            state.as_mut(),
                            &context,
                        )
                    }
                    .unwrap();
                    assert_tensor_exact(
                        &output.tokens,
                        &tokens,
                        "Inkling prediction token identity",
                    );
                    if observed {
                        let get = |suffix: &str| &observer.values[&format!("{path}.{suffix}")];
                        let fused_input = NumericTensor::concatenate(
                            &[
                                get("prediction.hidden.normalized.effective").clone(),
                                get("prediction.embedding.normalized.effective").clone(),
                            ],
                            -1,
                            &context,
                        )
                        .unwrap();
                        assert_tensor_exact(
                            &fused_input,
                            get("prediction.fusion.input"),
                            "Inkling hidden-before-embedding fusion order",
                        );
                        let expected = linear(
                            &fused_input,
                            &model.layers[depth].input_projection.weight,
                            None,
                        )
                        .unwrap();
                        assert_tensor_close(
                            &expected,
                            get("prediction.fusion.output"),
                            "Inkling prediction fusion reconstruction",
                        );
                        assert_tensor_exact(
                            &output.hidden,
                            get("prediction.readout.normalized.effective"),
                            "Inkling effective prediction continuation",
                        );
                        assert_tensor_exact(
                            &hidden,
                            get("prediction.hidden"),
                            "Inkling raw prediction input",
                        );
                        if let Some(zero) = &zero {
                            assert_ne!(
                                observer.values[zero].data,
                                observer.values[&format!("{zero}.effective")].data
                            );
                        }
                    }
                    outputs.push(output.hidden);
                }
                assert_eq!(state.layer(depth).unwrap().position(), 5);
                assert_eq!(state.layer(1 - depth).unwrap().position(), 0);
                (outputs, state)
            };
            let (baseline, baseline_state) = trial(false, 0);
            let (captured, captured_state) = trial(true, 0);
            assert_state_exact(
                &baseline_state,
                &captured_state,
                2,
                "Inkling prediction no-op state",
            );
            for (baseline, captured) in baseline.iter().zip(&captured) {
                assert_tensor_exact(baseline, captured, "Inkling prediction no-op components");
            }
            for mode in 1..=7 {
                let (masked, _) = trial(true, mode);
                for (baseline, masked) in baseline.iter().zip(&masked) {
                    assert_ne!(
                        baseline.data, masked.data,
                        "causal Inkling prediction mode {mode}, depth={depth}"
                    );
                }
            }
        }
    }
}
