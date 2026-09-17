use super::*;
use crate::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};

fn fixed(rank: usize) -> ResolvedCaptureSlice {
    ResolvedCaptureSlice {
        starts: vec![0; rank],
        ends: vec![0; rank],
        strides: vec![0; rank],
        shape: vec![0; rank],
    }
}
fn prefix(value: &Value, start: u64, end: u64, context: u64) -> Value {
    let region = ResolvedCaptureSlice {
        starts: vec![0, start, 0],
        ends: vec![1, end, context],
        strides: vec![1; 3],
        shape: vec![1, end - start, context],
    };
    Value {
        shape: region.shape.clone(),
        data: indices(&value.shape, &region)
            .into_iter()
            .map(|i| value.data[i])
            .collect(),
    }
}
#[test]
fn context_prefix_projection_shares_original_payload_and_rejects_wrong_reached_axes() {
    let (_, base) = plans(
        vec![operation(
            "base",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        )],
        false,
    );
    let mut discovery = session::discovery(&base);
    discovery.points[0].axes = vec![
        TensorAxis {
            name: "batch".into(),
            dimension: SymbolicDimension::Batch,
        },
        TensorAxis {
            name: "query".into(),
            dimension: SymbolicDimension::Sequence,
        },
        TensorAxis {
            name: "context".into(),
            dimension: SymbolicDimension::Context,
        },
    ];
    let input = Value {
        shape: vec![1, 5, 9],
        data: (1..=45).map(|n| n as f32).collect(),
    };
    let inference = InferenceGeometry {
        batch_size: 1,
        cached_positions: 4,
        input_positions: 5,
        prefill_chunk_positions: 2,
        max_output_tokens: 1,
        output: OutputDemand::Sequence,
    };
    let pool = WorkingMemoryPool::new(1 << 22, 17).unwrap();
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 21)
        .unwrap();
    let mut owners = Vec::new();
    for context_start in [1, 7] {
        let width = (9u64 - context_start).div_ceil(2);
        let payload = InterventionTensor {
            shape: vec![1, 2, width],
            values: InterventionValues::Float32((0..2 * width).map(|n| 100.0 + n as f32).collect()),
        };
        for action in [
            InterventionAction::Add {
                tensor: payload.clone(),
            },
            InterventionAction::Replace { tensor: payload },
        ] {
            let mut edit = operation("context", action);
            edit.schedule = CaptureSchedule {
                decode: false,
                ..Default::default()
            };
            edit.evidence = InterventionEvidence::None;
            edit.slices = vec![
                CaptureSlice {
                    axis: "query".into(),
                    start: 1,
                    end: 5,
                    stride: 2,
                },
                CaptureSlice {
                    axis: "context".into(),
                    start: context_start,
                    end: 9,
                    stride: 2,
                },
            ];
            let plan = InterventionPlan {
                schema_version: 1,
                operations: vec![edit],
            }
            .admit_with_text_origin(
                &discovery,
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 5,
                    max_predictions: 1,
                },
                CaptureTextOrigin {
                    cached_positions: 4,
                },
                "context-prefix",
            )
            .unwrap();
            let full = plan
                .validate_at(
                    0,
                    CapturePhase::Prefill,
                    0,
                    None,
                    &input.shape,
                    Some(InterventionDtype::Float32),
                )
                .unwrap();
            let expected = apply_activation(
                &mut Backend::default(),
                &input,
                &plan.plan().operations[0].action,
                &full,
            )
            .unwrap();
            for (start, end) in [(0, 2), (2, 4), (4, 5)] {
                let chunk = crate::prefill::PrefillChunk {
                    input: start..end,
                    position: 4 + start,
                    output: inference.output.for_chunk(end == 5),
                };
                let span = InterventionPrefillWindow::new(&plan, inference, &chunk).unwrap();
                let actual = prefix(&input, start, end, 4 + end);
                let mut global = vec![0; 3];
                let mut selected = fixed(3);
                let mut local = fixed(3);
                let mut destination = fixed(3);
                let untouched = (local.clone(), destination.clone());
                let (row, overlap) = span
                    .resolve_projection_into(
                        &plan,
                        0,
                        &actual.shape,
                        InterventionDtype::Float32,
                        &mut global,
                        &mut selected,
                        &mut local,
                        &mut destination,
                    )
                    .unwrap();
                assert_eq!(row, 1);
                assert_eq!(global, input.shape);
                assert_eq!(selected, full);
                assert_eq!(overlap, end != 5 && (context_start == 1 || start == 2));
                let edited = if overlap {
                    let owner = PreparedWindowInterventionPayload::prepare(
                        &plan.plan().operations[0].action,
                        &destination,
                        funding.clone(),
                    )
                    .unwrap();
                    let edited = apply_activation(
                        &mut Backend::default(),
                        &actual,
                        owner.projected_action().unwrap(),
                        &local,
                    )
                    .unwrap();
                    owners.push(owner);
                    edited
                } else {
                    assert_eq!((local.clone(), destination.clone()), untouched);
                    actual.clone()
                };
                assert_eq!(edited.data, prefix(&expected, start, end, 4 + end).data);
                let mut wrong = actual.shape.clone();
                wrong[2] += 1;
                assert!(matches!(
                    span.resolve_projection_into(
                        &plan,
                        0,
                        &wrong,
                        InterventionDtype::Float32,
                        &mut global,
                        &mut selected,
                        &mut local,
                        &mut destination
                    ),
                    Err(InterventionPrefillProjectionError::Window(
                        CaptureWindowSourceError::Axes(CaptureAxisError::Extent { index: 2, .. })
                    ))
                ));
                let mut wrong_chunk = chunk.clone();
                wrong_chunk.position += 1;
                assert_eq!(
                    InterventionPrefillWindow::new(&plan, inference, &wrong_chunk),
                    Err(InterventionPrefillSourceError::Chunk)
                );
            }
        }
    }
    let paid = pool.used_bytes().unwrap();
    drop(funding);
    assert_eq!(pool.used_bytes().unwrap(), paid);
    drop(owners);
    assert_eq!(pool.used_bytes().unwrap(), 17);
}
