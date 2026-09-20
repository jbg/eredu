use super::*;

#[test]
fn native_intervention_estimates_account_for_group_masks_and_host_lookup() {
    let mut policy = InterventionRoutingPolicy {
        expert_count: 64,
        top_k: 2,
        scoring: RoutingScoring::Softmax,
        normalize_selected: true,
        normalization_epsilon: 0.,
        coefficient_scale: 1.,
        groups: 1,
        selected_groups: 1,
        learned_coefficient_scale: false,
        shared_experts: 0,
    };
    let estimate = |p: &InterventionRoutingPolicy, rows| {
        NativeInterventionEstimator
            .original_route_usage(p, rows)
            .unwrap()
    };
    let plain = estimate(&policy, 3);
    assert_eq!(plain.host_bytes, 0);
    policy.groups = 8;
    let grouped = estimate(&policy, 3);
    assert!(grouped.host_bytes > 0 && grouped.retained_bytes > plain.retained_bytes);
    policy.selected_groups = 7;
    let wider = estimate(&policy, 3);
    assert!(wider.retained_bytes > grouped.retained_bytes);
    assert_eq!(wider.host_bytes, grouped.host_bytes);
    assert!(estimate(&policy, 6).retained_bytes > wider.retained_bytes);
    assert_eq!(wider.captures, 0);
    assert_eq!(wider.encoded_bytes, 0);
    assert!(NativeInterventionEstimator
        .original_route_usage(&policy, u64::MAX)
        .is_err());
}

#[test]
fn native_activation_primitives_pass_shared_conformance() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    eredu_evaluation::intervention::activation_conformance(
        &mut NativeCapture {
            partition: None,
            stream: &stream,
            domain: None,
        },
        |value| value.to_f32_vec(&stream).unwrap(),
    );
}

fn row() -> ResolvedCaptureSlice {
    ResolvedCaptureSlice {
        starts: vec![1, 0],
        ends: vec![2, 3],
        strides: vec![1, 1],
        shape: vec![1, 3],
    }
}

#[test]
fn native_intervention_patch_scale_mask_and_bias_preserve_other_rows() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut native = NativeCapture {
        partition: None,
        stream: &stream,
        domain: None,
    };
    let input = MlxTensor::from_array(Array::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    ));
    let cases = [
        (
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
            vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        ),
        (
            InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 2.0,
            },
            vec![1.0, 2.0, 3.0, 8.0, 10.0, 12.0],
        ),
        (
            InterventionAction::Mask {
                dtype: InterventionDtype::Float32,
                shape: vec![1, 3],
                keep: vec![true, false, true],
            },
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0],
        ),
        (
            InterventionAction::Replace {
                tensor: InterventionTensor {
                    shape: vec![1, 3],
                    values: InterventionValues::Float32(vec![7.0, 8.0, 9.0]),
                },
            },
            vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
        ),
        (
            InterventionAction::Add {
                tensor: InterventionTensor {
                    shape: vec![1, 3],
                    values: InterventionValues::Float32(vec![7.0, 8.0, 9.0]),
                },
            },
            vec![1.0, 2.0, 3.0, 11.0, 13.0, 15.0],
        ),
    ];
    for (action, expected) in cases {
        let output =
            eredu_runtime::intervention::apply_activation(&mut native, &input, &action, &row())
                .unwrap();
        assert_eq!(output.as_array().dtype(), Dtype::Float32);
        assert_eq!(output.to_f32_vec(&stream).unwrap(), expected);
        assert_eq!(
            input.to_f32_vec(&stream).unwrap(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }
    let output = eredu_runtime::intervention::apply_activation(
        &mut native,
        &input,
        &InterventionAction::MaskLogits {
            dtype: InterventionDtype::Float32,
            token_ids: vec![1],
        },
        &row(),
    )
    .unwrap();
    assert_eq!(
        output.to_f32_vec(&stream).unwrap(),
        [1.0, 2.0, 3.0, 4.0, f32::NEG_INFINITY, 6.0]
    );
}

#[test]
fn native_intervention_half_payload_bits_are_exact_and_casts_are_rejected() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut native = NativeCapture {
        partition: None,
        stream: &stream,
        domain: None,
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0],
        ends: vec![2],
        strides: vec![1],
        shape: vec![2],
    };
    for (dtype, values) in [
        (
            Dtype::Float16,
            InterventionValues::Float16(vec![0x3555, 0x8000]),
        ),
        (
            Dtype::Bfloat16,
            InterventionValues::Bfloat16(vec![0x3eab, 0x8000]),
        ),
    ] {
        let input = MlxTensor::from_array(
            Array::from_slice(&[1.0f32, 2.0], &[2])
                .as_dtype(dtype, &stream)
                .unwrap(),
        );
        let output = eredu_runtime::intervention::apply_activation(
            &mut native,
            &input,
            &InterventionAction::Replace {
                tensor: InterventionTensor {
                    shape: vec![2],
                    values: values.clone(),
                },
            },
            &slice,
        )
        .unwrap();
        let evaluated = output.as_array().evaluated().unwrap();
        match values {
            InterventionValues::Float16(bits) => assert_eq!(
                evaluated
                    .as_slice::<half::f16>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                bits
            ),
            InterventionValues::Bfloat16(bits) => assert_eq!(
                evaluated
                    .as_slice::<half::bf16>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                bits
            ),
            _ => unreachable!(),
        }
        assert!(eredu_runtime::intervention::apply_activation(
            &mut native,
            &input,
            &InterventionAction::Replace {
                tensor: InterventionTensor {
                    shape: vec![2],
                    values: InterventionValues::Float32(vec![1.0; 2])
                }
            },
            &slice
        )
        .is_err());
    }
}

#[test]
fn native_prepaid_partition_activation_preserves_payloads_and_global_masks_cpu() {
    verify_prepaid_partition_activation(safemlx::DeviceType::Cpu);
}

#[cfg(feature = "metal")]
#[test]
fn native_prepaid_partition_activation_preserves_payloads_and_global_masks_metal() {
    verify_prepaid_partition_activation(safemlx::DeviceType::Gpu);
}

fn verify_prepaid_partition_activation(device: safemlx::DeviceType) {
    use eredu_core::{component::ComponentCoordinateMap, *};
    use eredu_runtime::intervention::PartitionActivationProjection;
    let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
    let request = CaptureRequestShape {
        batch: 1,
        prompt_tokens: 3,
        max_predictions: 1,
    };
    let mut capture = CapturePlan::none();
    let usage = CaptureUsage {
        captures: 0,
        retained_bytes: 1 << 20,
        host_bytes: 1 << 20,
        encoded_bytes: 0,
    };
    capture.limits.per_step = usage;
    capture.limits.cumulative = usage;
    let capabilities = CaptureCapabilities::default();
    let capture = capture
        .admit(
            &ObservationCatalog {
                schema_version: 1,
                points: vec![],
                completeness: DescriptionCompleteness::Complete,
            },
            &ObservationSupportReport {
                schema_version: 1,
                capture: capabilities.clone(),
                points: vec![],
            },
            &capabilities,
            request,
        )
        .unwrap();
    let plan = |action: InterventionAction, partial| {
        let logits = matches!(action, InterventionAction::MaskLogits { .. });
        let axis = if logits { "vocabulary" } else { "component" };
        let mut slices = vec![CaptureSlice {
            axis: "sequence".into(),
            start: 1,
            end: 3,
            stride: 1,
        }];
        if partial {
            slices.push(CaptureSlice {
                axis: axis.into(),
                start: 1,
                end: 8,
                stride: 2,
            });
        }
        let point = InterventionPoint {
            path: "units".into(),
            node_id: "block".into(),
            stage: if logits {
                InterventionStage::LogitsBeforeSampling
            } else {
                InterventionStage::Activation
            },
            axes: vec![
                TensorAxis {
                    name: "sequence".into(),
                    dimension: SymbolicDimension::Sequence,
                },
                TensorAxis {
                    name: axis.into(),
                    dimension: SymbolicDimension::Known(8),
                },
            ],
            dtypes: vec![action.dtype().unwrap()],
            operations: vec![action.kind()],
            score_stages: vec![],
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            conditions: vec![],
            routed_units: None,
            routing: None,
        };
        InterventionPlan {
            schema_version: 1,
            operations: vec![InterventionOperation {
                id: "edit".into(),
                target: point.path.clone(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices,
                action,
                evidence: InterventionEvidence::None,
            }],
        }
        .admit(
            &InterventionDiscovery {
                schema_version: 1,
                artifact_identity: "fixture".into(),
                session_identity: Some("native-fixture".into()),
                points: vec![point],
            },
            request,
            "trial",
        )
        .unwrap()
    };
    for (dtype, values) in [
        (
            Dtype::Float32,
            InterventionValues::Float32(vec![0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75, 2.0]),
        ),
        (
            Dtype::Float16,
            InterventionValues::Float16(vec![
                0x3400, 0xb800, 0x3a00, 0x3c00, 0xbd00, 0x3e00, 0xbf00, 0x4000,
            ]),
        ),
        (
            Dtype::Bfloat16,
            InterventionValues::Bfloat16(vec![
                0x3e80, 0xbf00, 0x3f40, 0x3f80, 0xbfa0, 0x3fc0, 0xbfe0, 0x4000,
            ]),
        ),
    ] {
        let admitted = plan(
            InterventionAction::Replace {
                tensor: InterventionTensor {
                    shape: vec![2, 4],
                    values,
                },
            },
            true,
        );
        let map = ComponentCoordinateMap::indices(8, vec![7, 1, 5, 3]).unwrap();
        let source: Vec<f32> = (0..3)
            .flat_map(|row| {
                let map = &map;
                (0..4).map(move |column| (row * 8 + map.local_to_global(column).unwrap()) as f32)
            })
            .collect();
        let input = MlxTensor::from_array(
            Array::from_slice(&source, &[3, 4])
                .as_dtype(dtype, &stream)
                .unwrap(),
        );
        let projection = PartitionActivationProjection::new(
            &admitted,
            0,
            CapturePhase::Prefill,
            0,
            &[3, 8],
            1,
            &map,
            4,
        )
        .unwrap();
        let mut ledger = CaptureLedger::new(&capture);
        let work = projection
            .reserve(&mut ledger, &NativeInterventionEstimator)
            .unwrap();
        let charged = ledger.total();
        assert!(charged.retained_bytes > 0 && charged.host_bytes > 0);
        let mut native = NativeCapture {
            partition: None,
            stream: &stream,
            domain: None,
        };
        let output = work.apply(&mut native, &input).unwrap().unwrap();
        assert_eq!(
            output.to_f32_vec(&stream).unwrap(),
            [7.0, 1.0, 5.0, 3.0, 1.0, 0.25, 0.75, -0.5, 2.0, -1.25, -1.75, 1.5]
        );
        assert_eq!(input.to_f32_vec(&stream).unwrap(), source);
        assert_eq!(ledger.total(), charged);
    }
    for (action, expected_tail) in [
        (
            InterventionAction::MaskComponents {
                dtype: InterventionDtype::Float32,
                indices: vec![0, 1],
                keep_selected: true,
            },
            0.0,
        ),
        (
            InterventionAction::MaskLogits {
                dtype: InterventionDtype::Float32,
                token_ids: vec![4, 5, 6, 7],
            },
            f32::NEG_INFINITY,
        ),
    ] {
        let admitted = plan(action, false);
        let map = ComponentCoordinateMap::range(8, 4..8).unwrap();
        let input = MlxTensor::from_array(Array::from_slice(&[2.0f32; 12], &[3, 4]));
        let work = PartitionActivationProjection::new(
            &admitted,
            0,
            CapturePhase::Prefill,
            0,
            &[3, 8],
            1,
            &map,
            1,
        )
        .unwrap()
        .reserve(
            &mut CaptureLedger::new(&capture),
            &NativeInterventionEstimator,
        )
        .unwrap();
        let output = work
            .apply(
                &mut NativeCapture {
                    partition: None,
                    stream: &stream,
                    domain: None,
                },
                &input,
            )
            .unwrap()
            .unwrap();
        let data = output.to_f32_vec(&stream).unwrap();
        assert_eq!(&data[..4], &[2.0; 4]);
        assert_eq!(&data[4..], &[expected_tail; 8]);
    }
}


#[test]
fn static_activation_all_actions_match_strided_host_oracle_and_trace_population() {
    use eredu_nn::workspace::{WorkspaceContext,WorkspaceTensor,WorkspaceDtype,WorkspaceMechanisms};
    use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
    let stream=Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
    let values:Vec<f32>=(0..14).map(|i|i as f32*0.25-1.5).collect();
    let transposed:Vec<f32>=(0..7).flat_map(|column|[values[column],values[7+column]]).collect();
    let input=MlxTensor::from_array(Array::from_slice(&transposed,&[7,2]).transpose_axes(&[1,0],&stream).unwrap());
    for case in 0..7 {
        let whole=case==3 || case==6;
        let slice=ResolvedCaptureSlice {starts:vec![1,if whole {0}else{1}],ends:vec![2,7],
            strides:vec![1,if whole {1}else{2}],shape:vec![1,if whole {7}else{3}]};
        let dtype=InterventionDtype::Float32;
        let tensor=||InterventionTensor {shape:vec![1,3],values:InterventionValues::Float32(vec![0.125,-0.5,1.75])};
        let action=match case {
            0=>InterventionAction::Zero {dtype},
            1=>InterventionAction::Scale {dtype,factor:-0.5},
            2=>InterventionAction::Mask {dtype,shape:vec![1,3],keep:vec![true,false,true]},
            3=>InterventionAction::MaskComponents {dtype,indices:vec![1,5],keep_selected:true},
            4=>InterventionAction::Replace {tensor:tensor()},
            5=>InterventionAction::Add {tensor:tensor()},
            _=>InterventionAction::MaskLogits {dtype,token_ids:vec![0,4]},
        };
        let mut expected=values.clone();
        for column in 0..7 {
            if !whole && ![1,3,5].contains(&column) {continue;}
            let ordinal=column/2;let index=7+column;
            expected[index]=match case {
                0=>0.0,
                1=>values[index]*-0.5,
                2=>if ordinal==1 {0.0}else{values[index]},
                3=>if [1,5].contains(&column) {values[index]}else{0.0},
                4=>[0.125,-0.5,1.75][ordinal],
                5=>values[index]+[0.125,-0.5,1.75][ordinal],
                _=>if [0,4].contains(&column) {f32::NEG_INFINITY}else{values[index]},
            };
        }
        let mut native=NativeCapture {stream:&stream,domain:None,partition:None};
        let actual=eredu_runtime::intervention::apply_activation(&mut native,&input,&action,&slice).unwrap();
        assert_eq!(actual.to_f32_vec(&stream).unwrap(),expected,"action {case}");
        assert_eq!(input.to_f32_vec(&stream).unwrap(),values,"source mutated");
        let program=PreparedStaticActivation::new(&action,&slice,&[2,7],dtype).unwrap();
        let population=program.population().unwrap();
        #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
        {
        let mechanism=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context=WorkspaceContext::new(mechanism);
        let source=WorkspaceTensor::existing(context.layout(&[2,7],WorkspaceDtype::Float32).unwrap(),&context).unwrap();
        context.begin_span();
        let mut retained=Vec::new();
        let output=program.trace(&source,&context,&mut retained).unwrap();
        assert_eq!(output.shape(),&[2,7]);
        assert_eq!(retained.len(),population.retained_roots);
        assert_eq!(population.completions,population.retained_roots);
        assert_eq!(population.host_bytes,if whole {7}else{0});
        assert!(population.controls>0);
        let report=context.report(&retained).unwrap();
        for operation in &report.operations {
            assert!(mechanism.operation_bound(operation).unwrap().is_some(),"unpriced {operation:?}");
        }
        let wrong=WorkspaceTensor::existing(context.layout(&[1,14],WorkspaceDtype::Float32).unwrap(),&context).unwrap();
        assert!(program.trace(&wrong,&context,&mut Vec::new()).is_err());
        }
    }
}

#[path = "tests/precision.rs"]
mod precision;

#[cfg(all(target_vendor = "apple", not(feature = "cuda")))]
#[path = "tests/zero.rs"]
mod zero;
