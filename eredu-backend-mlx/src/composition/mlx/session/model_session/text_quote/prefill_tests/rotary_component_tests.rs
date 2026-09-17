use super::*;
use super::operation_component_tests::exercise_component;
use crate::{
    MlxTensor,
    backend::nn::{
        shared::MlxNeuralBackend,
        workspace::{MlxMetalWorkspaceMechanisms, OriginalComponentTestPlan},
    },
};
use eredu_nn::{
    NeuralBackend, RotaryAlgorithm, RotaryArithmetic, RotaryOperator, RotaryPosition, RotarySpec,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceDtype, WorkspaceTensor},
};
use safemlx::Array;

#[test]
fn original_rotary_frequency_ancestors_match_ordinary_workers() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    let values: Vec<f32> = (0..72)
        .map(|i| ((i * 7 % 29) as f32 - 14.0) / 16.0)
        .collect();
    let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 2, 3, 12]));
    let wavelength = RotaryAlgorithm::Llama3 {
        factor: 8.0,
        low_frequency_factor: 1.0,
        high_frequency_factor: 4.0,
        original_max_positions: 128,
    };
    for (algorithm, arithmetic) in [
        (wavelength, RotaryArithmetic::Native),
        (wavelength, RotaryArithmetic::InputProducts),
        (
            RotaryAlgorithm::Proportional {
                factor: 2.0,
                rotary_fraction: 0.5,
            },
            RotaryArithmetic::InputProducts,
        ),
    ] {
        for traditional in [false, true] {
            // Four frequency slots cover low, medium and high wavelengths;
            // the unrotated tail remains present in this nonzero fixture.
            let spec = RotarySpec {
                dimensions: 8,
                traditional,
                base: 10_000.0,
                algorithm,
                arithmetic,
            };
            let invoke = || -> Array {
                let mut rotary = MlxNeuralBackend::rotary(spec, &stream).unwrap();
                rotary
                    .forward(&input, RotaryPosition::Offset(17), &stream)
                    .unwrap()
                    .into()
            };
            let expected = invoke().evaluated().unwrap().try_to_vec::<f32>().unwrap();
            let context =
                WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
            let metadata = WorkspaceTensor::existing(
                context
                    .layout(&[1, 2, 3, 12], WorkspaceDtype::Float32)
                    .unwrap(),
                &context,
            )
            .unwrap();
            let mut rotary = WorkspaceBackend::rotary(spec, &context).unwrap();
            context.begin_span();
            let output = rotary
                .forward(&metadata, RotaryPosition::Offset(17), &context)
                .unwrap();
            let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
            // Constructor and application both run inside the admitted native
            // Graph in this witness, so first-use ancestors cannot be hidden by
            // warming or detaching an ordinary frequency graph beforehand.
            exercise_component(&mut prepared, &plan, &[input.as_array()], &expected, invoke);
        }
    }
    drop((input, stream));
    prepared.finish();
}
