use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use eredu_nn::{
    GatedProductGroupLayout, GroupSelection, GroupedGatedProductSpec, GroupedNeuralBackend,
    GroupedProjectionSpec, GroupedRelu2Spec, LinearFormatSpec, ParameterSpec,
};
use eredu_runtime::capture::partition::{
    PartitionCaptureReceiptLimits, PartitionCaptureReceiptPlan, PartitionRoutedCaptureProducer,
};
use eredu_runtime::{RoutedExpertProvider, RoutedUnitBatch, RoutedUnitObserver};
mod live;

fn input(peer: usize, token: usize, channel: usize) -> f32 {
    ((token + channel * 3) % 9) as f32 * 0.125 - 0.5 + peer as f32 * 0.25
}
fn weight(expert: usize, unit: usize, half: usize, channel: usize) -> f32 {
    (expert + 1) as f32 * 0.125 + (unit + 1) as f32 * 0.0625 + half as f32 * 0.125
        - channel as f32 * 0.25
}
fn activation(token: usize, unit: usize, gated: bool) -> f32 {
    let expert = token % 3;
    let read = |half| {
        (0..2)
            .map(|channel| {
                f64::from(input(0, token, channel)) * f64::from(weight(expert, unit, half, channel))
            })
            .sum::<f64>()
    };
    let gate = read(0);
    if gated {
        (gate / (1. + (-gate).exp()) * read(1)) as f32
    } else {
        gate.max(0.).powi(2) as f32
    }
}
fn projection(name: &str) -> GroupedProjectionSpec {
    GroupedProjectionSpec::new(
        ParameterSpec::trainable(name).unwrap(),
        None,
        LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
    )
    .unwrap()
}

struct Collector<'a> {
    backend: NativeCapture<'a>,
    request: PartitionRoutedUnitCaptureRequest<'a>,
    original: Vec<RoutedUnitCapture>,
    effective: Vec<RoutedUnitCapture>,
}
impl RoutedUnitObserver<MlxTensor> for Collector<'_> {
    fn observe(&mut self, batch: &RoutedUnitBatch<'_, MlxTensor>) -> Result<(), eredu_nn::Error> {
        let source = batch
            .partition_capture_source()
            .map_err(eredu_nn::Error::backend_retained_source)?;
        self.original.push(
            self.backend
                .capture_partition_routed_units(&source, &self.request)
                .unwrap()
                .map_err(eredu_nn::Error::backend_retained_source)?,
        );
        Ok(())
    }
    fn intervene(
        &mut self,
        batch: &RoutedUnitBatch<'_, MlxTensor>,
    ) -> Result<Option<MlxTensor>, eredu_nn::Error> {
        let two = Array::from_slice(&[2f32], &[]);
        Ok(Some(MlxTensor::from_array(
            batch
                .units
                .values
                .as_array()
                .multiply(&two, self.backend.stream)
                .map_err(eredu_nn::Error::backend_retained_source)?,
        )))
    }
    fn observe_effective(
        &mut self,
        batch: &RoutedUnitBatch<'_, MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        let source = batch
            .partition_capture_source()
            .map_err(eredu_nn::Error::backend_retained_source)?;
        self.effective.push(
            self.backend
                .capture_partition_routed_units(&source, &self.request)
                .unwrap()
                .map_err(eredu_nn::Error::backend_retained_source)?,
        );
        Ok(())
    }
}

fn verify(device: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    let geometry = RoutedUnitGeometry {
        experts: 3,
        units_per_expert: 4,
        routes_per_token: 2,
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 0, 0],
        ends: vec![17, 2, 4],
        strides: vec![16, 1, 1],
        shape: vec![2, 2, 4],
    };
    let plan = admission(geometry, 17, &slice);
    let context = PartitionCaptureContext {
        invocation: None,
        artifact_identity: "native-fixture".into(),
        execution_identity: "tp-unit-columns".into(),
        run_identity: "provider".into(),
        overlay_identity: None,
        capture_plan_identity: plan.identity().into(),
        selection_index: 0,
        phase: CapturePhase::Prefill,
        prediction: 0,
        forward_epoch: 1,
        invocation_window: None,
    };
    let tags = (0..34).chain(0..34).collect::<Vec<_>>();
    let origins = eredu_runtime::RoutedUnitOrigins::new(&[34, 34], &tags, 2).unwrap();
    let input_values = (0..2)
        .flat_map(|peer| {
            (0..17).flat_map(move |token| {
                (0..2).flat_map(move |_| [input(peer, token, 0), input(peer, token, 1)])
            })
        })
        .collect::<Vec<_>>();
    let inputs = MlxTensor::from_array(Array::from_slice(&input_values, &[68, 2]));
    let groups = MlxTensor::from_array(Array::from_slice(
        &(0..2)
            .flat_map(|_| (0..17).flat_map(|token| [token % 3, token % 3]))
            .collect::<Vec<i32>>(),
        &[68, 1],
    ));
    let coefficients = MlxTensor::from_array(Array::from_slice(
        &(0..34).flat_map(|_| [0.25f32, 0.75]).collect::<Vec<_>>(),
        &[68, 1],
    ));
    let routes = GroupSelection::new(groups, coefficients.clone(), coefficients);

    for gated in [true, false] {
        let mut ledger = CaptureLedger::new(&plan);
        let producers = (0..2)
            .map(|rank| {
                let units = ComponentCoordinateMap::range(4, rank * 2..rank * 2 + 2).unwrap();
                PartitionRoutedCaptureProducer {
                    rank,
                    projection: CaptureSlicePartition::new(&[17, 2, 4], &slice, 2, &units, 4)
                        .unwrap(),
                    ownership: RoutedUnitCaptureOwnership {
                        coordinates: RoutedComponentCoordinateMap::new(
                            ComponentCoordinateMap::range(3, 0..3).unwrap(),
                            units,
                        ),
                        source_peer: Some(0),
                        source_peers: 2,
                    },
                }
            })
            .collect();
        let receipt = PartitionCaptureReceiptPlan::new_routed(
            eredu_core::capture::SharedCapturePlan::new(plan.clone()),
            context.clone(),
            producers,
            2,
            PartitionCaptureReceiptLimits {
                max_producers: 2,
                max_fragments: 4,
                max_record_bytes: 32768,
            },
            &mut ledger,
        )
        .unwrap();
        let mut records = vec![];
        for rank in 0..2 {
            let owned = receipt.routed_producer(rank).unwrap();
            let local_slice = ResolvedCaptureSlice {
                starts: vec![0, 0, (rank * 2) as u64],
                ends: vec![17, 2, (rank * 2 + 2) as u64],
                strides: vec![16, 1, 1],
                shape: vec![2, 2, 2],
            };
            let request = PartitionRoutedUnitCaptureRequest {
                geometry,
                source_tokens: 17,
                ownership: owned,
                slice: &local_slice,
            };
            let backend = NativeCapture {
                stream: &stream,
                domain: None,
                partition: None,
            };
            let usage = backend.estimate_partition_routed_units(&request).unwrap();
            assert_eq!(ledger.reserve(usage.checked_mul(2).unwrap()).unwrap(), None);
            let mut collector = Collector {
                backend,
                request,
                original: vec![],
                effective: vec![],
            };
            let up = (0..3)
                .flat_map(|expert| {
                    (0..if gated { 2 } else { 1 }).flat_map(move |half| {
                        (rank * 2..rank * 2 + 2).flat_map(move |unit| {
                            [weight(expert, unit, half, 0), weight(expert, unit, half, 1)]
                        })
                    })
                })
                .collect::<Vec<_>>();
            let down = (0..3)
                .flat_map(|expert| {
                    (0..2).flat_map(move |channel| {
                        (rank * 2..rank * 2 + 2)
                            .map(move |unit| (expert + channel + unit + 1) as f32 * 0.125)
                    })
                })
                .collect::<Vec<_>>();
            let bindings = std::collections::BTreeMap::from([
                (
                    (if gated { "gate_up_proj" } else { "up_proj" }).into(),
                    Array::from_slice(&up, &[3, if gated { 4 } else { 2 }, 2]),
                ),
                ("down_proj".into(), Array::from_slice(&down, &[3, 2, 2])),
            ]);
            let mut provider = eredu_runtime::ResidentExpertProvider;
            macro_rules! request {
                ($observer:expr) => {
                    eredu_runtime::RoutedExpertRequest {
                        bank: eredu_runtime::RoutedBankId::new(1),
                        layer: 0,
                        input: &inputs,
                        routes: &routes,
                        pass: eredu_runtime::ExpertPass::Prefill,
                        unit_observer: $observer,
                    }
                };
            }
            let (baseline, effective) = if gated {
                let mut bank = MlxNeuralBackend::grouped_gated_product(
                    GroupedGatedProductSpec::new(
                        3,
                        2,
                        2,
                        2,
                        eredu_nn::GatedProductPolicy::ordinary_silu(),
                        GatedProductGroupLayout::Packed {
                            gate_up: projection("read"),
                            down: projection("write"),
                        },
                    )
                    .unwrap(),
                    &stream,
                )
                .unwrap();
                bank.bind_local_parameters(bindings).unwrap();
                let baseline = RoutedExpertProvider::<MlxNeuralBackend>::forward_grouped(
                    &mut provider,
                    &mut bank,
                    request!(None),
                    &stream,
                )
                .unwrap();
                let mut observer: Option<&mut dyn RoutedUnitObserver<MlxTensor>> =
                    Some(&mut collector);
                let effective = eredu_runtime::with_exchanged_unit_observer(
                    &mut observer,
                    origins,
                    |mut observer| {
                        eredu_runtime::with_partition_unit_observer(
                            &mut observer,
                            owned.coordinates.units(),
                            |observer| {
                                RoutedExpertProvider::<MlxNeuralBackend>::forward_grouped(
                                    &mut provider,
                                    &mut bank,
                                    request!(observer),
                                    &stream,
                                )
                            },
                        )
                    },
                )
                .unwrap();
                (baseline, effective)
            } else {
                let mut bank = MlxNeuralBackend::grouped_relu2(
                    GroupedRelu2Spec::new(3, 2, 2, projection("read"), projection("write"))
                        .unwrap(),
                    &stream,
                )
                .unwrap();
                bank.bind_local_parameters(bindings).unwrap();
                let baseline = RoutedExpertProvider::<MlxNeuralBackend>::forward_relu2_routed(
                    &mut provider,
                    &mut bank,
                    request!(None),
                    &stream,
                )
                .unwrap();
                let mut observer: Option<&mut dyn RoutedUnitObserver<MlxTensor>> =
                    Some(&mut collector);
                let effective = eredu_runtime::with_exchanged_unit_observer(
                    &mut observer,
                    origins,
                    |mut observer| {
                        eredu_runtime::with_partition_unit_observer(
                            &mut observer,
                            owned.coordinates.units(),
                            |observer| {
                                RoutedExpertProvider::<MlxNeuralBackend>::forward_relu2_routed(
                                    &mut provider,
                                    &mut bank,
                                    request!(observer),
                                    &stream,
                                )
                            },
                        )
                    },
                )
                .unwrap();
                (baseline, effective)
            };
            let baseline = baseline.as_array().evaluated().unwrap();
            let effective = effective.as_array().evaluated().unwrap();
            for (&before, &after) in baseline
                .as_slice::<f32>()
                .iter()
                .zip(effective.as_slice::<f32>())
            {
                assert!((after - 2. * before).abs() < 2e-6);
            }
            assert!(!collector.original.is_empty());
            if gated {
                assert!(
                    collector.original.len() > 1,
                    "exercise actual native token chunking"
                );
            }
            assert_eq!(collector.original.len(), collector.effective.len());
            for (original, effective) in collector.original.iter().zip(&collector.effective) {
                for (a, b) in original.rows.iter().zip(&effective.rows) {
                    assert_eq!(a.coefficient, if a.slot == 0 { 0.25 } else { 0.75 });
                    assert_eq!(b.coefficient, a.coefficient);
                    assert_eq!(
                        (a.source_peer, a.token, a.slot, a.expert),
                        (Some(0), a.token, a.slot, a.token % 3)
                    );
                    let TensorObservationData::F32(actual) = a.values.data() else {
                        panic!("original")
                    };
                    let TensorObservationData::F32(edited) = b.values.data() else {
                        panic!("effective")
                    };
                    for (unit, (&actual, &edited)) in
                        (rank * 2..rank * 2 + 2).zip(actual.iter().zip(edited))
                    {
                        assert!((actual - activation(a.token as usize, unit, gated)).abs() < 2e-6);
                        assert!((edited - 2. * actual).abs() < 2e-6);
                    }
                }
            }
            let payload = RoutedUnitCapture {
                geometry,
                source_token_ranges: collector
                    .original
                    .iter()
                    .flat_map(|p| p.source_token_ranges.iter().copied())
                    .collect(),
                rows: collector
                    .original
                    .into_iter()
                    .flat_map(|p| p.rows)
                    .collect(),
            };
            let projection = receipt.producer(rank).unwrap();
            records.push(PartitionCaptureProducerRecord {
                combination: eredu_core::capture::PartitionCaptureCombination::Disjoint,
                schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
                receipt_plan_identity: receipt.identity().into(),
                context: context.clone(),
                producer_rank: rank,
                source_dtype: Some(eredu_core::checkpoint::TensorDtype::F32),
                fragments: vec![PartitionCaptureFragmentRecord {
                    fragment_index: 0,
                    record: CaptureRecord {
                        schema_version: CAPTURE_SCHEMA_VERSION,
                        selection_id: "units".into(),
                        path: "bank.units".into(),
                        node_id: "bank".into(),
                        position: ObservationPosition::BeforeIntervention,
                        source_shape: Some(projection.local_shape().to_vec()),
                        source_dtype: Some(eredu_core::checkpoint::TensorDtype::F32),
                        selected_shape: Some(projection.fragments()[0].local().shape.clone()),
                        outcome: CaptureOutcome::Captured,
                        payload: Some(CapturePayload::RoutedUnits(payload)),
                        charged: usage,
                    },
                }],
            });
        }
        let mut quota = ledger
            .reserve_quota(receipt.delivery_usage().unwrap())
            .unwrap();
        let mut delivery = receipt.into_delivery();
        for record in records {
            delivery
                .receive(
                    record.producer_rank,
                    &serde_json::to_vec(&record).unwrap(),
                    &mut quota,
                )
                .unwrap();
        }
        let result = delivery.finish(&mut quota).unwrap();
        let Some(CapturePayload::RoutedUnits(payload)) = &result.capture().record().payload else {
            panic!("assembled")
        };
        assert_eq!(payload.rows.len(), 4);
        for row in &payload.rows {
            let TensorObservationData::F32(actual) = row.values.data() else {
                panic!("values")
            };
            for (unit, &actual) in actual.iter().enumerate() {
                assert!((actual - activation(row.token as usize, unit, gated)).abs() < 2e-6);
            }
        }
    }
}

#[test]
#[ignore = "requires local native MLX execution"]
fn native_provider_chunks_feed_partition_receipts_cpu() {
    verify(DeviceType::Cpu);
}

#[test]
#[ignore = "requires local MLX Metal execution"]
#[cfg(feature = "metal")]
fn native_provider_chunks_feed_partition_receipts_metal() {
    verify(DeviceType::Gpu);
}
