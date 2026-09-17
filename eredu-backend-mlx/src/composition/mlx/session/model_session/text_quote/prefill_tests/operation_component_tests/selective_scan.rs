//! The actual scan report admits both recurrent state and output through the
//! same original component harness; no native capacity fixture replaces it.
use super::*;
use eredu_nn::{SelectiveStateSpaceScanInput, SelectiveStateSpaceScanOutput};

fn input<'a, T>(
    values: &'a [T; 8],
    initial: bool,
    chunk: usize,
) -> SelectiveStateSpaceScanInput<'a, T> {
    SelectiveStateSpaceScanInput {
        values: &values[0],
        input_state: &values[1],
        output_state: &values[2],
        time_step: &values[3],
        time_step_bias: &values[4],
        transition_log: &values[5],
        skip: &values[6],
        initial_state: initial.then_some(&values[7]),
        time_step_floor: 0.075,
        chunk_size: chunk,
    }
}
fn join(output: SelectiveStateSpaceScanOutput<MlxTensor>, stream: &safemlx::Stream) -> Array {
    let state = output.state.reshape(&[-1], stream).unwrap();
    let output = output.output.reshape(&[-1], stream).unwrap();
    MlxTensor::concatenate(&[state, output], 0, stream)
        .unwrap()
        .into()
}

#[test]
fn original_selective_scan_matches_reference_with_tail_and_cached_state() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    // Uneven prefill (5/2), cached decode and a caller chunk cap larger than
    // i32::MAX all consume the ordinary recurrence. Both source dtypes are real.
    for (sequence, chunk, initial, dtype) in [
        (5, 2, false, Dtype::Float32),
        (5, 2, true, Dtype::Bfloat16),
        (1, usize::MAX, true, Dtype::Float32),
    ] {
        let shapes: [&[i32]; 8] = [
            &[2, sequence, 2, 3],
            &[2, sequence, 2, 4],
            &[2, sequence, 2, 4],
            &[2, sequence, 2],
            &[2],
            &[2],
            &[2],
            &[2, 2, 3, 4],
        ];
        let values: [MlxTensor; 8] = std::array::from_fn(|source| {
            let size = shapes[source].iter().map(|d| *d as usize).product();
            let data: Vec<f32> = (0..size)
                .map(|i| {
                    let x = ((i * 7 + source * 3) % 29) as f32 - 13.0;
                    if source == 5 {
                        x / 19.0 - 0.75
                    } else {
                        x / 23.0
                    }
                })
                .collect();
            let value = Array::from_slice(&data, shapes[source]);
            let value = if source == 0 {
                value.as_dtype(dtype, &stream).unwrap()
            } else {
                value
            };
            MlxTensor::from_array(value)
        });
        let host: [Vec<f32>; 8] = std::array::from_fn(|i| {
            values[i]
                .as_array()
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap()
        });
        let (state_ref, output_ref) = eredu_nn::reference_selective_state_space_scan(
            2,
            sequence as usize,
            2,
            3,
            4,
            &host[0],
            &host[1],
            &host[2],
            &host[3],
            &host[4],
            &host[5],
            &host[6],
            0.075,
            initial.then_some(host[7].as_slice()),
        )
        .unwrap();
        let expected = join(
            MlxNeuralBackend::selective_state_space_scan(input(&values, initial, chunk), &stream)
                .unwrap(),
            &stream,
        )
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
        assert_eq!(expected.len(), state_ref.len() + output_ref.len());
        for (index, (actual, reference)) in expected
            .iter()
            .zip(state_ref.iter().chain(&output_ref))
            .enumerate()
        {
            let tolerance = if index >= state_ref.len() && dtype == Dtype::Bfloat16 {
                0.008
            } else {
                2e-5
            };
            assert!(
                (actual - reference).abs() <= tolerance + tolerance * reference.abs(),
                "native {actual}, host reference {reference}"
            );
        }
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let metadata: [WorkspaceTensor; 8] = std::array::from_fn(|i| {
            let format = if i == 0 && dtype == Dtype::Bfloat16 {
                WorkspaceFloatingType::Bfloat16
            } else {
                WorkspaceFloatingType::Float32
            };
            WorkspaceTensor::existing(
                context
                    .layout(shapes[i], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(format, true))),
                &context,
            )
            .unwrap()
        });
        context.begin_span();
        let result = WorkspaceBackend::selective_state_space_scan(
            input(&metadata, initial, chunk),
            &context,
        )
        .unwrap();
        let state = result.state.reshape(&[-1], &context).unwrap();
        let output = result.output.reshape(&[-1], &context).unwrap();
        let output = WorkspaceTensor::concatenate(&[state, output], 0, &context).unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        let leaves: Vec<&Array> = values[..if initial { 8 } else { 7 }]
            .iter()
            .map(MlxTensor::as_array)
            .collect();
        exercise_component(&mut prepared, &plan, &leaves, &expected, || {
            join(
                MlxNeuralBackend::selective_state_space_scan(
                    input(&values, initial, chunk),
                    &stream,
                )
                .unwrap(),
                &stream,
            )
        });
    }
    drop(stream);
    prepared.finish();
}
