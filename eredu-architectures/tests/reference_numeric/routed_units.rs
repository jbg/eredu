use super::*;
use eredu_nn::{GroupedUnitBatch, GroupedUnitObserver};
use eredu_runtime::{RoutedUnitBatch, RoutedUnitObserver};
#[path = "routed_units/partition.rs"]
mod partition;

fn observe_units(
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    groups: usize,
    mut values: Vec<NumericTensor>,
    observer: &mut dyn GroupedUnitObserver<NumericTensor>,
) -> Result<Vec<NumericTensor>, Error> {
    let tokens = routes.group_indices().shape[0] as usize;
    let top_k = routes.group_indices().shape[1] as usize;
    let width = values[0].data.len();
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by_key(|&row| routes.group_indices().data[row] as usize);
    let sorted = NumericTensor::new(
        vec![order.len() as i32, width as i32],
        order
            .iter()
            .flat_map(|&row| values[row].data.iter().copied())
            .collect(),
    );
    let indices = NumericTensor::new(
        vec![order.len() as i32],
        order.iter().map(|&row| row as f32).collect(),
    );
    let token_indices = NumericTensor::new(
        indices.shape.clone(),
        order.iter().map(|&row| (row / top_k) as f32).collect(),
    );
    let groups_tensor = NumericTensor::new(
        indices.shape.clone(),
        order
            .iter()
            .map(|&row| routes.group_indices().data[row])
            .collect(),
    );
    let batch = GroupedUnitBatch {
        values: &sorted,
        group_indices: &groups_tensor,
        selection_indices: &indices,
        token_indices: &token_indices,
        coefficients: routes.coefficients(),
        token_offset: 0,
        total_token_count: tokens,
        group_count: groups,
    };
    observer.observe(&batch)?;
    let replacement = observer.intervene(&batch)?;
    let effective = replacement.as_ref().unwrap_or(&sorted);
    if effective.shape != sorted.shape {
        return Err(Error::backend_source(
            eredu_nn::GroupedUnitError::ReplacementShape {
                expected: sorted.shape.clone(),
                actual: effective.shape.clone(),
            },
        ));
    }
    observer.observe_effective(&batch.with_values(effective))?;
    for (row, &original) in order.iter().enumerate() {
        values[original] = NumericTensor::new(
            vec![1, width as i32],
            effective.data[row * width..(row + 1) * width].to_vec(),
        );
    }
    assert_eq!(
        input.data.len() / input.shape.last().copied().unwrap() as usize,
        tokens
    );
    Ok(values)
}

pub(super) fn gated(
    bank: &mut NumericExpertBank,
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    context: &NumericContext,
    observer: &mut dyn GroupedUnitObserver<NumericTensor>,
    tp: bool,
) -> Result<TensorParallelGroupedOutput<NumericTensor>, Error> {
    if context.bind_checkpoint_values {
        bank.refresh_bound_experts()?;
    }
    let hidden = *input.shape.last().unwrap() as usize;
    let tokens = input.data.len() / hidden;
    let top_k = routes.group_indices().shape[1] as usize;
    let mut values = Vec::new();
    for token in 0..tokens {
        let row = NumericTensor::new(
            vec![1, hidden as i32],
            input.data[token * hidden..(token + 1) * hidden].to_vec(),
        );
        for slot in 0..top_k {
            let expert = &bank.experts[routes.group_indices().data[token * top_k + slot] as usize];
            let gate = linear(&row, &expert.gate, expert.gate_bias.as_ref())?
                .map(|v| bank.policy.gate_upper_bound().map_or(v, |b| v.min(b)));
            let gate = gate.map(|v| match bank.policy.activation() {
                eredu_nn::GatedProductActivation::Silu => {
                    v / (1.0 + (-bank.policy.sigmoid_multiplier() * v).exp())
                }
                eredu_nn::GatedProductActivation::GeluApproximate => {
                    0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044_715 * v.powi(3))).tanh())
                }
                _ => panic!("unsupported fixture activation"),
            });
            let up = linear(&row, &expert.up, expert.up_bias.as_ref())?.map(|v| {
                bank.policy
                    .up_absolute_bound()
                    .map_or(v, |b| v.clamp(-b, b))
                    + bank.policy.up_offset()
            });
            values.push(gate.zip(&up, |a, b| a * b)?);
        }
    }
    let values = observe_units(input, routes, bank.experts.len(), values, observer)?;
    let mut output = NumericTensor::zeros(input.shape.clone());
    let mut bias = bank
        .experts
        .iter()
        .any(|e| e.down_bias.is_some())
        .then(|| NumericTensor::zeros(input.shape.clone()));
    for token in 0..tokens {
        let mut order: Vec<usize> = (0..top_k).collect();
        if bank.spec.reduction() == eredu_nn::GroupReduction::SequentialGroupOrder {
            order.sort_by_key(|&s| routes.group_indices().data[token * top_k + s] as usize);
        }
        for slot in order {
            let index = token * top_k + slot;
            let expert = &bank.experts[routes.group_indices().data[index] as usize];
            let result = linear(
                &values[index],
                &expert.down,
                if tp { None } else { expert.down_bias.as_ref() },
            )?;
            for c in 0..hidden {
                output.data[token * hidden + c] +=
                    routes.coefficients().data[index] * result.data[c];
                if let Some(bias) = &mut bias {
                    bias.data[token * hidden + c] += routes.coefficients().data[index]
                        * expert.down_bias.as_ref().unwrap().data[c];
                }
            }
        }
    }
    Ok(TensorParallelGroupedOutput::new(
        output,
        if tp { bias } else { None },
    ))
}

pub(super) fn relu(
    bank: &mut NumericRelu2Groups,
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    observer: &mut dyn GroupedUnitObserver<NumericTensor>,
) -> Result<NumericTensor, Error> {
    let tokens = input.data.len() / bank.hidden;
    let top_k = routes.group_indices().shape[1] as usize;
    let mut values = Vec::new();
    let member = |tensor: &NumericTensor, expert| {
        let value = tensor.axis_slice(0, expert, expert + 1);
        NumericTensor::new(value.shape[1..].to_vec(), value.data)
    };
    for token in 0..tokens {
        let row = NumericTensor::new(
            vec![1, bank.hidden as i32],
            input.data[token * bank.hidden..(token + 1) * bank.hidden].to_vec(),
        );
        for slot in 0..top_k {
            let expert = routes.group_indices().data[token * top_k + slot] as usize;
            values
                .push(linear(&row, &member(&bank.up.0, expert), None)?.map(|v| v.max(0.0).powi(2)));
        }
    }
    let values = observe_units(input, routes, bank.expert_count, values, observer)?;
    let mut output = NumericTensor::zeros(input.shape.clone());
    for token in 0..tokens {
        for slot in 0..top_k {
            let index = token * top_k + slot;
            let expert = routes.group_indices().data[index] as usize;
            let result = linear(&values[index], &member(&bank.down.0, expert), None)?;
            for c in 0..bank.hidden {
                output.data[token * bank.hidden + c] +=
                    routes.coefficients().data[index] * result.data[c];
            }
        }
    }
    Ok(output)
}

#[derive(Default)]
pub(super) struct Capture {
    pub values: BTreeMap<(Option<usize>, usize, usize, usize), Vec<f32>>,
    pub effective: BTreeMap<(Option<usize>, usize, usize, usize), Vec<f32>>,
    pub zero: Option<(usize, usize)>,
    pub chunks: std::collections::BTreeSet<usize>,
}
impl Capture {
    fn keys(
        batch: &RoutedUnitBatch<'_, NumericTensor>,
    ) -> Vec<(Option<usize>, usize, usize, usize)> {
        let source_width = *batch.source_groups.shape.last().unwrap() as usize;
        let local_width = *batch.units.coefficients.shape.last().unwrap() as usize;
        (0..batch.units.values.shape[0] as usize)
            .map(|row| {
                let native_token = batch.units.token_indices.data[row] as usize;
                let slot = batch.units.selection_indices.data[row] as usize % local_width;
                let source_token = batch.source_token(native_token).unwrap();
                let expert = batch
                    .global_group(
                        batch.source_groups.data[source_token * source_width + slot] as usize,
                    )
                    .unwrap();
                let origin = batch.route_origin(native_token, slot).unwrap();
                (origin.source_peer, origin.token, origin.slot, expert)
            })
            .collect()
    }
}
impl RoutedUnitObserver<NumericTensor> for Capture {
    fn observe(&mut self, b: &RoutedUnitBatch<'_, NumericTensor>) -> Result<(), Error> {
        self.chunks.insert(b.provider_token_offset);
        let width = b.units.values.shape[1] as usize;
        for (r, key) in Self::keys(b).into_iter().enumerate() {
            assert!(self
                .values
                .insert(
                    key,
                    b.units.values.data[r * width..(r + 1) * width].to_vec()
                )
                .is_none());
        }
        Ok(())
    }
    fn intervene(
        &mut self,
        b: &RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<Option<NumericTensor>, Error> {
        let Some((selected_token, selected_expert)) = self.zero else {
            return Ok(None);
        };
        let mut value = b.units.values.clone();
        let width = value.shape[1] as usize;
        for (r, (_, token, _, expert)) in Self::keys(b).into_iter().enumerate() {
            if token == selected_token && expert == selected_expert {
                value.data[r * width] = 0.0;
            }
        }
        Ok(Some(value))
    }
    fn observe_effective(&mut self, b: &RoutedUnitBatch<'_, NumericTensor>) -> Result<(), Error> {
        let width = b.units.values.shape[1] as usize;
        for (r, key) in Self::keys(b).into_iter().enumerate() {
            assert!(self
                .effective
                .insert(
                    key,
                    b.units.values.data[r * width..(r + 1) * width].to_vec()
                )
                .is_none());
        }
        Ok(())
    }
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Capture {
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
    fn routed_unit_observer(
        &mut self,
        _: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<NumericTensor>>, Error> {
        Ok(Some(self))
    }
}

fn invoke_gated<P: TensorParallelRoutedExpertProvider<NumericBackend>>(
    provider: &mut P,
    bank: &mut NumericExpertBank,
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    context: &NumericContext,
    capture: &mut Capture,
    pass: ExpertPass,
    tp: bool,
) -> NumericTensor
where
    P::Error: std::fmt::Display,
{
    let id = eredu_runtime::RoutedBankId::new(0);
    let observed = eredu_runtime::ObservedExpertProvider::<_, _, Error>::new(
        provider,
        capture,
        eredu_runtime::RoutedObservationPoints::new(id, "fixture.experts", bank.spec.group_count()),
    );
    let mut boxed = eredu_runtime::RoutedBankProviders::new([(id, Box::new(observed))]).unwrap();
    let request = RoutedExpertRequest {
        bank: id,
        layer: 0,
        input,
        routes,
        pass,
        unit_observer: None,
    };
    if tp {
        let result =
            TensorParallelRoutedExpertProvider::<NumericBackend>::forward_grouped_tensor_parallel(
                &mut boxed, bank, request, 1, context,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        match result {
            RoutedExpertTensorParallelOutput::Complete(v) => v,
            RoutedExpertTensorParallelOutput::Partial(p) => {
                let (v, b) = p.into_parts();
                if let Some(b) = b {
                    v.add(&b, context).unwrap()
                } else {
                    v
                }
            }
        }
    } else {
        RoutedExpertProvider::<NumericBackend>::forward_grouped(&mut boxed, bank, request, context)
            .unwrap_or_else(|e| panic!("{e}"))
    }
}
fn invoke_relu<P: TensorParallelRoutedExpertProvider<NumericBackend>>(
    provider: &mut P,
    bank: &mut NumericRelu2Groups,
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    context: &NumericContext,
    capture: &mut Capture,
    pass: ExpertPass,
    tp: bool,
) -> NumericTensor
where
    P::Error: std::fmt::Display,
{
    let id = eredu_runtime::RoutedBankId::new(0);
    let observed = eredu_runtime::ObservedExpertProvider::<_, _, Error>::new(
        provider,
        capture,
        eredu_runtime::RoutedObservationPoints::new(id, "fixture.experts", bank.spec.group_count()),
    );
    let mut boxed = eredu_runtime::RoutedBankProviders::new([(id, Box::new(observed))]).unwrap();
    let request = RoutedExpertRequest {
        bank: id,
        layer: 0,
        input,
        routes,
        pass,
        unit_observer: None,
    };
    if tp {
        let result=TensorParallelRoutedExpertProvider::<NumericBackend>::forward_relu2_routed_tensor_parallel(&mut boxed,bank,request,1,context).unwrap_or_else(|e|panic!("{e}"));
        match result {
            RoutedExpertTensorParallelOutput::Complete(v) => v,
            RoutedExpertTensorParallelOutput::Partial(p) => {
                let (v, b) = p.into_parts();
                if let Some(b) = b {
                    v.add(&b, context).unwrap()
                } else {
                    v
                }
            }
        }
    } else {
        RoutedExpertProvider::<NumericBackend>::forward_relu2_routed(
            &mut boxed, bank, request, context,
        )
        .unwrap_or_else(|e| panic!("{e}"))
    }
}

pub(super) fn verify_gated<P: TensorParallelRoutedExpertProvider<NumericBackend>>(
    provider: &mut P,
    bank: &mut NumericExpertBank,
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    context: &NumericContext,
) where
    P::Error: std::fmt::Display,
{
    let baseline = bank.forward_grouped(input, routes, context).unwrap();
    for pass in [ExpertPass::Prefill, ExpertPass::Decode] {
        for tp in [false, true] {
            for zero in [None, Some((1, 0))] {
                let mut reference = Capture {
                    zero,
                    ..Default::default()
                };
                let expected = invoke_gated(
                    &mut eredu_runtime::ResidentExpertProvider,
                    bank,
                    input,
                    routes,
                    context,
                    &mut reference,
                    pass,
                    tp,
                );
                let mut actual = Capture {
                    zero,
                    ..Default::default()
                };
                let result = invoke_gated(
                    provider,
                    bank,
                    input,
                    routes,
                    context,
                    &mut actual,
                    pass,
                    tp,
                );
                assert_tensor_close(&result, &expected, "compacted gated unit intervention");
                assert_eq!(
                    actual.values, reference.values,
                    "original units must retain global expert/token/slot identity"
                );
                assert_eq!(actual.effective, reference.effective);
                assert_eq!(actual.values.len(), routes.group_indices().data.len());
                assert!(
                    actual.chunks.len() > 1,
                    "fixture must cross provider chunks"
                );
                if zero.is_none() {
                    assert_tensor_close(&result, &baseline, "all keep provider parity")
                } else {
                    assert_ne!(
                        result.data, baseline.data,
                        "edit must change a real contribution"
                    );
                }
            }
        }
    }
}
pub(super) fn verify_relu<P: TensorParallelRoutedExpertProvider<NumericBackend>>(
    provider: &mut P,
    bank: &mut NumericRelu2Groups,
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    context: &NumericContext,
) where
    P::Error: std::fmt::Display,
{
    let baseline = bank.forward_grouped(input, routes, context).unwrap();
    for pass in [ExpertPass::Prefill, ExpertPass::Decode] {
        for tp in [false, true] {
            for zero in [None, Some((1, 2))] {
                let mut reference = Capture {
                    zero,
                    ..Default::default()
                };
                let expected = invoke_relu(
                    &mut eredu_runtime::ResidentExpertProvider,
                    bank,
                    input,
                    routes,
                    context,
                    &mut reference,
                    pass,
                    tp,
                );
                let mut actual = Capture {
                    zero,
                    ..Default::default()
                };
                let result = invoke_relu(
                    provider,
                    bank,
                    input,
                    routes,
                    context,
                    &mut actual,
                    pass,
                    tp,
                );
                assert_tensor_close(&result, &expected, "compacted ReLU2 unit intervention");
                assert_eq!(actual.values, reference.values);
                assert_eq!(actual.effective, reference.effective);
                assert_eq!(actual.values.len(), routes.group_indices().data.len());
                assert!(actual.chunks.len() > 1);
                if zero.is_none() {
                    assert_tensor_close(&result, &baseline, "all keep provider parity")
                } else {
                    assert_ne!(result.data, baseline.data);
                }
            }
        }
    }
}

#[test]
fn exchanged_unit_origins_preserve_idle_peers_and_original_slots() {
    let origins = eredu_runtime::RoutedUnitOrigins::new(&[2, 0, 3], &[7, 2, 4, 1, 7], 3).unwrap();
    let expected = [(0, 2, 1), (0, 0, 2), (2, 1, 1), (2, 0, 1), (2, 2, 1)];
    for (row, (peer, token, slot)) in expected.into_iter().enumerate() {
        assert_eq!(
            origins.resolve(row),
            Some(eredu_runtime::RoutedUnitOrigin {
                source_peer: Some(peer),
                token,
                slot
            })
        );
    }
    assert_eq!(origins.resolve(5), None);
    assert!(eredu_runtime::RoutedUnitOrigins::new(&[usize::MAX, 1], &[], 1).is_err());
    assert!(eredu_runtime::RoutedUnitOrigins::new(&[1], &[], 1).is_err());
    assert!(eredu_runtime::RoutedUnitOrigins::new(&[0], &[], 0).is_err());
}

#[test]
fn partition_unit_coordinates_survive_provider_chunks_exchange_and_actual_replacement() {
    use eredu_core::{capture::*, component::*, intervention::*};
    struct Sink {
        calls: usize,
        effective: Vec<f32>,
    }
    impl RoutedUnitObserver<NumericTensor> for Sink {
        fn observe(&mut self, batch: &RoutedUnitBatch<'_, NumericTensor>) -> Result<(), Error> {
            assert_eq!(batch.unit_coordinates.unwrap().local_to_global(0), Some(4));
            assert_eq!(
                batch.route_origin(2, 0).unwrap(),
                eredu_runtime::RoutedUnitOrigin {
                    source_peer: Some(2),
                    token: 0,
                    slot: 1
                }
            );
            self.calls += 1;
            Ok(())
        }
        fn intervene(
            &mut self,
            batch: &RoutedUnitBatch<'_, NumericTensor>,
        ) -> Result<Option<NumericTensor>, Error> {
            let rows: Vec<_> = Capture::keys(batch)
                .into_iter()
                .map(|(peer, token, slot, expert)| RoutedUnitLocation {
                    source_peer: peer.map(|n| n as u64),
                    token: token as u64,
                    slot: slot as u64,
                    expert: expert as u64,
                })
                .collect();
            assert_eq!(rows.iter().map(|r| r.expert).collect::<Vec<_>>(), [1, 1, 3]);
            let coordinates = RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::indices(4, vec![3, 1]).unwrap(),
                batch.unit_coordinates.unwrap().clone(),
            );
            let recipe = eredu_runtime::intervention::lower_partition_routed_intervention(
                RoutedUnitGeometry {
                    experts: 4,
                    units_per_expert: 5,
                    routes_per_token: 2,
                },
                2,
                &rows,
                &coordinates,
                &ResolvedCaptureSlice {
                    starts: vec![0, 0],
                    ends: vec![2, 20],
                    strides: vec![1, 1],
                    shape: vec![2, 20],
                },
                &InterventionAction::MaskComponents {
                    dtype: InterventionDtype::Float32,
                    indices: vec![9, 15],
                    keep_selected: true,
                },
            )
            .map_err(Error::backend_source)?;
            assert!(matches!(
                recipe.action,
                Some(InterventionAction::Zero { .. })
            ));
            let mut result = batch.units.values.clone();
            for index in recipe.indices {
                result.data[index as usize] = 0.;
            }
            self.calls += 1;
            Ok(Some(result))
        }
        fn observe_effective(
            &mut self,
            batch: &RoutedUnitBatch<'_, NumericTensor>,
        ) -> Result<(), Error> {
            assert_eq!(batch.unit_coordinates.unwrap().local_to_global(2), Some(2));
            self.effective = batch.units.values.data.clone();
            self.calls += 1;
            Ok(())
        }
    }
    let units = ComponentCoordinateMap::indices(5, vec![4, 0, 2]).unwrap();
    let values = NumericTensor::new(
        vec![3, 3],
        vec![10., 11., 12., 20., 21., 22., 30., 31., 32.],
    );
    let sorted = NumericTensor::new(vec![3], vec![2., 0., 1.]);
    let groups = NumericTensor::new(vec![4, 1], vec![0., 1., 0., 1.]);
    let compact_groups = NumericTensor::new(vec![3], vec![1., 1., 0.]);
    let coefficients = NumericTensor::new(vec![3, 1], vec![0.5, 0.75, 0.25]);
    let batch = GroupedUnitBatch {
        values: &values,
        group_indices: &compact_groups,
        selection_indices: &sorted,
        token_indices: &sorted,
        coefficients: &coefficients,
        token_offset: 1,
        total_token_count: 4,
        group_count: 2,
    };
    let origins = eredu_runtime::RoutedUnitOrigins::new(&[2, 0, 2], &[0, 3, 2, 1], 2).unwrap();
    let mut sink = Sink {
        calls: 0,
        effective: vec![],
    };
    let mut observer: Option<&mut dyn RoutedUnitObserver<NumericTensor>> = Some(&mut sink);
    eredu_runtime::with_partition_unit_observer(&mut observer, &units, |mut observer| {
        eredu_runtime::with_exchanged_unit_observer(&mut observer, origins, |mut observer| {
            eredu_runtime::with_provider_unit_observer(
                &mut observer,
                &groups,
                Some(&[3, 1]),
                0,
                |observer| {
                    let observer = observer.unwrap();
                    observer.observe(&batch)?;
                    let replacement = observer.intervene(&batch)?.unwrap();
                    observer.observe_effective(&batch.with_values(&replacement))
                },
            )
        })
    })
    .unwrap();
    assert_eq!(sink.calls, 3);
    assert_eq!(sink.effective, [10., 0., 0., 20., 0., 0., 0., 31., 0.]);
    // A malformed local unit axis is rejected before the consumer callback.
    let short = ComponentCoordinateMap::range(5, 0..2).unwrap();
    let mut observer: Option<&mut dyn RoutedUnitObserver<NumericTensor>> = Some(&mut sink);
    assert!(
        eredu_runtime::with_partition_unit_observer(&mut observer, &short, |mut observer| {
            eredu_runtime::with_provider_unit_observer(
                &mut observer,
                &groups,
                None,
                0,
                |observer| observer.unwrap().observe(&batch),
            )
        })
        .is_err()
    );
    assert_eq!(sink.calls, 3);
    eredu_runtime::with_partition_unit_observer::<NumericTensor, _>(
        &mut None,
        &short,
        |observer| assert!(observer.is_none()),
    );
}

#[derive(Default)]
struct Invocations {
    values: BTreeMap<String, Capture>,
    zero: Option<(String, usize, usize)>,
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Invocations {
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<NumericTensor>>, Error> {
        let capture = self.values.entry(path.to_owned()).or_default();
        capture.zero = self
            .zero
            .as_ref()
            .filter(|(p, _, _)| p == path)
            .map(|(_, token, expert)| (*token, *expert));
        Ok(Some(capture))
    }
}
#[test]
fn shared_hybrid_model_delivers_actual_routed_units_during_prefill_and_cached_decode() {
    let mut config = heterogeneous_replicated_configs()
        .into_iter()
        .find(|v| v["model_type"] == "qwen3_next")
        .unwrap();
    config["num_experts"] = 4.into();
    config["num_experts_per_tok"] = 2.into();
    config["moe_intermediate_size"] = 6.into();
    config["shared_expert_intermediate_size"] = 8.into();
    config["norm_topk_prob"] = true.into();
    let args = qwen::hybrid::model_args_from_config_value(&config)
        .unwrap()
        .text;
    let context = NumericContext::default();
    let run = |zero: Option<(String, usize, usize)>, instrument: bool| {
        let architecture =
            qwen::hybrid::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
        let mut runtime = LayerwiseRuntime::new(architecture, RebuildingUnitPolicy::default());
        let mut state = DeviceState::<NumericBackend, _>::create(
            qwen::hybrid::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap();
        let mut result = Vec::new();
        for (step, tokens) in [vec![1, 4, 2], vec![3], vec![2]].into_iter().enumerate() {
            let input = NumericTensor::token_ids(&tokens);
            let mut capture = Invocations {
                zero: if step == 0 { zero.clone() } else { None },
                ..Default::default()
            };
            let output = if instrument {
                runtime
                    .forward_with_provider_and_observer(
                        qwen::hybrid::EmbeddedInput::target(&input, None),
                        &mut state,
                        if step == 0 {
                            ExpertPass::Prefill
                        } else {
                            ExpertPass::Decode
                        },
                        &mut eredu_runtime::ResidentExpertProvider,
                        &context,
                        &mut capture,
                    )
                    .unwrap()
            } else {
                runtime
                    .forward(
                        qwen::hybrid::EmbeddedInput::target(&input, None),
                        &mut state,
                        &context,
                    )
                    .unwrap()
            };
            result.push((output, capture));
        }
        result
    };
    let ordinary = run(None, false);
    let observed = run(None, true);
    for ((expected, _), (actual, capture)) in ordinary.iter().zip(&observed) {
        assert_tensor_close(actual, expected, "model observer all-keep parity");
        assert!(
            !capture.values.is_empty(),
            "actual routed invocation must deliver units"
        );
        for c in capture.values.values() {
            assert!(!c.values.is_empty());
            assert_eq!(c.values, c.effective);
        }
    }
    let (path, capture) = observed[0].1.values.iter().next().unwrap();
    let &(_, token, _, expert) = capture
        .values
        .iter()
        .find(|((_, token, _, _), values)| *token == 1 && values[0].abs() > 1e-8)
        .unwrap()
        .0;
    let edited = run(Some((path.clone(), token, expert)), true);
    assert_ne!(
        edited[0].0.data, ordinary[0].0.data,
        "one actual expert unit must change model computation"
    );
    let restored = run(None, true);
    for ((actual, _), (expected, _)) in restored.iter().zip(&ordinary) {
        assert_tensor_close(
            actual,
            expected,
            "fresh trial restores cached model behavior",
        );
    }
    let architecture =
        qwen::hybrid::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut runtime = LayerwiseRuntime::new(architecture, RebuildingUnitPolicy::default());
    let mut state = DeviceState::<NumericBackend, _>::create(
        qwen::hybrid::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let input = NumericTensor::token_ids(&[1, 4, 2]);
    let error = runtime
        .forward_with_provider_and_observer(
            qwen::hybrid::EmbeddedInput::target(&input, None),
            &mut state,
            ExpertPass::Prefill,
            &mut eredu_runtime::ResidentExpertProvider,
            &context,
            &mut Failing,
        )
        .expect_err("unit failure must stop the model");
    assert_unit_source(&error);
}

#[derive(Debug, thiserror::Error)]
#[error("routed unit sentinel")]
struct UnitSentinel;
fn assert_unit_source(error: &(dyn std::error::Error + 'static)) {
    let mut cause = Some(error);
    while let Some(error) = cause {
        if error.is::<UnitSentinel>() {
            return;
        }
        cause = error.source();
    }
    panic!("unit callback's original source was lost: {error}");
}
struct Failing;
impl RoutedUnitObserver<NumericTensor> for Failing {
    fn observe(&mut self, _: &RoutedUnitBatch<'_, NumericTensor>) -> Result<(), Error> {
        Err(Error::backend_source(UnitSentinel))
    }
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Failing {
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
    fn routed_unit_observer(
        &mut self,
        _: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<NumericTensor>>, Error> {
        Ok(Some(self))
    }
}
pub(super) fn verify_failure<P: RoutedExpertProvider<NumericBackend>>(
    provider: &mut P,
    bank: &mut NumericExpertBank,
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    context: &NumericContext,
) where
    P::Error: std::fmt::Display + std::error::Error + 'static,
{
    use std::error::Error as _;
    let id = eredu_runtime::RoutedBankId::new(0);
    let mut observer = Failing;
    let mut observed = eredu_runtime::ObservedExpertProvider::new(
        provider,
        &mut observer,
        eredu_runtime::RoutedObservationPoints::new(id, "bank", bank.spec.group_count()),
    );
    let error = RoutedExpertProvider::<NumericBackend>::forward_grouped(
        &mut observed,
        bank,
        RoutedExpertRequest {
            bank: id,
            layer: 0,
            input,
            routes,
            pass: ExpertPass::Prefill,
            unit_observer: None,
        },
        context,
    )
    .expect_err("unit failure must stop provider");
    assert!(matches!(
        error,
        eredu_runtime::ObservedExpertProviderError::Unit(_)
    ));
    assert!(
        error
            .source()
            .unwrap()
            .source()
            .unwrap()
            .is::<UnitSentinel>(),
        "provider string adaptation must not lose the observer's source"
    );
    let error = eredu_runtime::ObservedExpertProvider::new(
        provider,
        &mut Failing,
        eredu_runtime::RoutedObservationPoints::new(id, "bank", bank.spec.group_count()),
    )
    .execute_neural(|observed| {
        RoutedExpertProvider::<NumericBackend>::forward_grouped(
            observed,
            bank,
            RoutedExpertRequest {
                bank: id,
                layer: 0,
                input,
                routes,
                pass: ExpertPass::Prefill,
                unit_observer: None,
            },
            context,
        )
        .map_err(|error| Error::backend(error.to_string()))
    })
    .expect_err("neural adapter must preserve failed unit work");
    assert_unit_source(&error);

    let mut existing = Capture::default();
    let mut invoked = false;
    let result = eredu_runtime::with_routed_unit_observer(
        &mut Failing,
        "bank",
        RoutedExpertRequest {
            bank: id,
            layer: 0,
            input,
            routes,
            pass: ExpertPass::Prefill,
            unit_observer: Some(&mut existing),
        },
        |_| {
            invoked = true;
            Ok::<_, Error>(())
        },
    );
    assert!(matches!(
        result,
        Err(eredu_runtime::ObservedExpertProviderError::DuplicateUnitObserver)
    ));
    assert!(
        !invoked,
        "conflicting observer authority must fail before provider work"
    );
}

#[test]
fn exchanged_and_chunked_units_preserve_source_identity_and_modify_consumed_rows() {
    let origins = eredu_runtime::RoutedUnitOrigins::new(&[1, 0, 3], &[2, 4, 1, 5], 2).unwrap();
    let source = NumericTensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]);
    let groups = NumericTensor::new(vec![2], vec![0.0, 1.0]);
    let selection = NumericTensor::new(vec![2], vec![1.0, 0.0]);
    let coefficients = NumericTensor::new(vec![2, 1], vec![0.25, 0.75]);
    let values = NumericTensor::new(vec![2, 2], vec![-2.0, 4.0, 3.0, -5.0]);
    let batch = GroupedUnitBatch {
        values: &values,
        group_indices: &groups,
        selection_indices: &selection,
        token_indices: &selection,
        coefficients: &coefficients,
        token_offset: 0,
        total_token_count: 2,
        group_count: 2,
    };
    let mut capture = Capture {
        zero: Some((2, 5)),
        ..Default::default()
    };
    let mut admitted: Option<&mut dyn RoutedUnitObserver<NumericTensor>> = Some(&mut capture);
    let effective =
        eredu_runtime::with_exchanged_unit_observer(&mut admitted, origins, |mut observer| {
            eredu_runtime::with_provider_unit_observer(
                &mut observer,
                &source,
                Some(&[5, 9]),
                2,
                |observer| {
                    let observer = observer.unwrap();
                    observer.observe(&batch).unwrap();
                    let effective = observer.intervene(&batch).unwrap().unwrap();
                    observer
                        .observe_effective(&batch.with_values(&effective))
                        .unwrap();
                    effective
                },
            )
        });
    assert_eq!(capture.values[&(Some(2), 2, 1, 5)], [-2.0, 4.0]);
    assert_eq!(capture.values[&(Some(2), 0, 1, 9)], [3.0, -5.0]);
    assert_eq!(effective.data, [0.0, 4.0, 3.0, -5.0]);
    assert_eq!(capture.effective[&(Some(2), 2, 1, 5)], [0.0, 4.0]);
}
