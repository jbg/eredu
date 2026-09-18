use super::*;
use crate::backend::distributed::MlxDistributedSession;
use eredu_runtime::capture::{partition::*, CaptureSession};
use eredu_runtime::ActivationObserver;

struct Layout {
    slice: ResolvedCaptureSlice,
    units: ComponentCoordinateMap,
}
impl PartitionCaptureLayout for Layout {
    fn capture_placement(
        &self,
        _: &AdmittedCapturePlan,
        _: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        panic!("routed units require sparse placement")
    }
    fn routed_capture_placement(
        &self,
        _: &AdmittedCapturePlan,
        _: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionRoutedCapturePlacement, CaptureError> {
        let ownership = RoutedUnitCaptureOwnership {
            coordinates: RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::range(3, 0..3).unwrap(),
                self.units.clone(),
            ),
            source_peer: Some(0),
            source_peers: 2,
        };
        Ok(PartitionRoutedCapturePlacement {
            routing: "bank".into(),
            effective: false,
            producers: vec![PartitionRoutedCaptureProducer {
                rank: 0,
                projection: CaptureSlicePartition::new(
                    &[17, 2, 4],
                    &self.slice,
                    2,
                    &self.units,
                    4,
                )?,
                ownership: ownership.clone(),
            }],
            sources: vec![PartitionRoutedCaptureSource {
                rank: 0,
                ownership,
                input_width: 2,
            }],
        })
    }
}

// Real grouped operators, provider chunks, native transforms and live session
// delivery; supplied EP tags model two source peers. This singleton transport
// test does not establish multiprocess EP invocation/failure integration.
fn verify_live(device: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let manifest = eredu_runtime::CommunicationManifest::new(1, 0, vec![], vec![])
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(5),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    let transport = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
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
    let tags = (0..34).chain(0..34).collect::<Vec<_>>();
    let origins = eredu_runtime::RoutedUnitOrigins::new(&[34, 34], &tags, 2).unwrap();
    let inputs = MlxTensor::from_array(Array::from_slice(
        &(0..2)
            .flat_map(|peer| {
                (0..17).flat_map(move |token| {
                    (0..2).flat_map(move |_| [input(peer, token, 0), input(peer, token, 1)])
                })
            })
            .collect::<Vec<_>>(),
        &[68, 2],
    ));
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
    let units = ComponentCoordinateMap::range(4, 0..4).unwrap();
    let layout = Layout {
        slice: slice.clone(),
        units: units.clone(),
    };
    let mut expected_usage = None;
    for gated in [false, true] {
        for committed in [false, true] {
            let plan = admission(geometry, 17, &slice);
            let mut capture = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(plan));
            let epoch = DistributedCommitEpoch::FIRST;
            let native = NativeCapture {
                stream: &stream,
                domain: None,
                partition: Some(&transport),
            };
            let mut observer = PartitionCaptureObserver::for_step(
                &mut capture,
                native,
                &transport,
                &layout,
                0,
                PartitionCaptureReceiptLimits {
                    max_producers: 1,
                    max_fragments: 4,
                    max_record_bytes: 1 << 20,
                },
                |_: &[u64],
                 _: &CaptureSelection,
                 _: &ResolvedCaptureSlice|
                 -> Result<PartitionCaptureNativeEstimate, CaptureError> {
                    panic!("dense estimate")
                },
                |error: PartitionCaptureObserverError<Error>| error,
            )
            .with_session_identity(
                PartitionCaptureIdentity::for_session(
                    "native-fixture".into(),
                    "native-grouped".into(),
                    transport.session_identity(),
                    None,
                )
                .unwrap(),
            );
            observer
                .prepare_transaction(epoch, eredu_runtime::ExpertPass::Prefill)
                .unwrap();
            observer.coordinate_transaction(epoch).unwrap();
            let up = (0..3)
                .flat_map(|expert| {
                    (0..if gated { 2 } else { 1 }).flat_map(move |half| {
                        (0..4).flat_map(move |unit| {
                            [weight(expert, unit, half, 0), weight(expert, unit, half, 1)]
                        })
                    })
                })
                .collect::<Vec<_>>();
            let down = (0..3)
                .flat_map(|expert| {
                    (0..2).flat_map(move |channel| {
                        (0..4).map(move |unit| (expert + channel + unit + 1) as f32 * 0.125)
                    })
                })
                .collect::<Vec<_>>();
            let bindings = std::collections::BTreeMap::from([
                (
                    (if gated { "gate_up_proj" } else { "up_proj" }).into(),
                    Array::from_slice(&up, &[3, if gated { 8 } else { 4 }, 2]),
                ),
                ("down_proj".into(), Array::from_slice(&down, &[3, 2, 4])),
            ]);
            {
                let mut units_observer = observer.routed_unit_observer("bank").unwrap();
                let result = eredu_runtime::with_exchanged_unit_observer(
                    &mut units_observer,
                    origins,
                    |mut observer| {
                        eredu_runtime::with_partition_unit_observer(
                            &mut observer,
                            &units,
                            |observer| {
                                eredu_runtime::with_routed_unit_invocation(
                                    observer,
                                    eredu_runtime::RoutedUnitInvocation {
                                        input: &inputs,
                                        origins: None,
                                        unit_coordinates: None,
                                    },
                                    |observer| {
                                        let request = eredu_runtime::RoutedExpertRequest {
                                            bank: eredu_runtime::RoutedBankId::new(1),
                                            layer: 0,
                                            input: &inputs,
                                            routes: &routes,
                                            pass: eredu_runtime::ExpertPass::Prefill,
                                            unit_observer: observer,
                                        };
                                        let mut provider = eredu_runtime::ResidentExpertProvider;
                                        if gated {
                                            let mut bank = MlxNeuralBackend::grouped_gated_product(
                                                GroupedGatedProductSpec::new(
                                                    3,
                                                    2,
                                                    4,
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
                                            RoutedExpertProvider::<MlxNeuralBackend>::forward_grouped(
                                        &mut provider,
                                        &mut bank,
                                        request,
                                        &stream,
                                    )
                                        } else {
                                            let mut bank = MlxNeuralBackend::grouped_relu2(
                                                GroupedRelu2Spec::new(
                                                    3,
                                                    2,
                                                    4,
                                                    projection("read"),
                                                    projection("write"),
                                                )
                                                .unwrap(),
                                                &stream,
                                            )
                                            .unwrap();
                                            bank.bind_local_parameters(bindings).unwrap();
                                            RoutedExpertProvider::<MlxNeuralBackend>::forward_relu2_routed(
                                        &mut provider,
                                        &mut bank,
                                        request,
                                        &stream,
                                    )
                                        }
                                    },
                                    |error| error,
                                )
                            },
                        )
                    },
                );
                result.unwrap().as_array().evaluated().unwrap();
            }
            observer.complete_transaction(epoch).unwrap();
            observer.finish_transaction(epoch, committed);
            drop(observer);
            let step = capture.take_shared_step().map(|frame| frame.as_step().clone()).unwrap();
            if let Some(expected) = expected_usage {
                assert_eq!(step.cumulative_usage, expected);
            } else {
                expected_usage = Some(step.cumulative_usage);
            }
            if !committed {
                assert!(step.records[0].payload.is_none());
                assert!(step.partitions.is_empty());
                continue;
            }
            let Some(CapturePayload::RoutedUnits(payload)) = &step.records[0].payload else {
                panic!("sparse")
            };
            assert_eq!(payload.rows.len(), 4);
            for row in &payload.rows {
                assert_eq!((row.source_peer, row.expert), (Some(0), row.token % 3));
                assert_eq!(row.coefficient, if row.slot == 0 { 0.25 } else { 0.75 });
                let TensorObservationData::F32(values) = row.values.data() else {
                    panic!("values")
                };
                for (unit, &value) in values.iter().enumerate() {
                    assert!((value - activation(row.token as usize, unit, gated)).abs() < 2e-6);
                }
            }
            let evidence = step.partitions[0].contributions[0].routed.as_ref().unwrap();
            assert!(!evidence.source_token_ranges.is_empty());
            if gated {
                assert!(
                    evidence.source_token_ranges.len() > 1,
                    "actual native chunks"
                );
            }
            assert_eq!(evidence.source_token_ranges.last().unwrap()[1], 68);
        }
    }
}

#[test]
#[ignore = "requires local native MLX execution"]
fn native_sparse_live_session_cpu() {
    verify_live(DeviceType::Cpu);
}
#[test]
#[ignore = "requires local MLX Metal execution"]
#[cfg(feature = "metal")]
fn native_sparse_live_session_metal() {
    verify_live(DeviceType::Gpu);
}
