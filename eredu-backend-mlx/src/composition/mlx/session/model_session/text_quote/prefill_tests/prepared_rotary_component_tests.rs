use super::operation_component_tests::exercise_component;
use super::*;
use crate::{
    MlxTensor,
    backend::nn::workspace::{MlxMetalWorkspaceMechanisms, OriginalComponentTestPlan},
};
use eredu_nn::{
    Tensor,
    multimodal::{
        MultiAxisRotaryLayout, MultiAxisRotarySpec, PreparedMultiAxisRotary, RotaryAxisSpec,
        reference_multi_axis_rotary_embeddings_prepared,
    },
    workspace::{WorkspaceContext, WorkspaceDtype, WorkspaceTensor},
};
use safemlx::Array;

#[test]
fn original_prepared_spatial_rotary_matches_all_inline_layouts() {
    exercise_profiles([
        (MultiAxisRotaryLayout::IndependentAxes, vec![4, 8], 2),
        (MultiAxisRotaryLayout::SplitHalves, vec![8, 8], 2),
        (MultiAxisRotaryLayout::RoundRobinSections, vec![4, 2, 2], 2),
    ]);
}

#[test]
fn original_prepared_spatial_rotary_uses_paid_larger_rows() {
    exercise_profiles([
        (
            MultiAxisRotaryLayout::IndependentAxes,
            vec![4, 8, 4, 8, 4],
            2,
        ),
        (
            MultiAxisRotaryLayout::SplitHalves,
            vec![4, 8, 4, 8, 4, 8],
            2,
        ),
        (
            MultiAxisRotaryLayout::RoundRobinSections,
            vec![32, 16, 16],
            2,
        ),
    ]);
}

#[test]
fn original_prepared_spatial_rotary_uses_paid_larger_shapes() {
    // Independent case: large row populations must still execute if a future
    // rank-dependent worker change regresses this distinct allocation path.
    exercise_profiles_with_source(
        [(MultiAxisRotaryLayout::IndependentAxes, vec![4, 8], 12)],
        true,
    );
}

fn exercise_profiles(profiles: impl IntoIterator<Item = (MultiAxisRotaryLayout, Vec<i32>, usize)>) {
    exercise_profiles_with_source(profiles, false);
}

fn exercise_profiles_with_source(
    profiles: impl IntoIterator<Item = (MultiAxisRotaryLayout, Vec<i32>, usize)>,
    transposed_source: bool,
) {
    let mut fixture = OriginalOperationFixture::prepare(true);
    let stream = fixture.stream.clone();
    for (layout, widths, rank) in profiles {
        let spec = MultiAxisRotarySpec {
            axes: widths
                .into_iter()
                .enumerate()
                .map(|(axis, dimensions)| RotaryAxisSpec {
                    dimensions,
                    position_offset: axis as i32 - 2,
                })
                .collect(),
            base: 10000.0,
            minimum_position: -1,
            layout,
        };
        let axes = spec.axes.len();
        let positions: Vec<i32> = (0..2 * axes).map(|i| i as i32 * 3 - 1).collect();
        let mut shape = vec![1; rank];
        shape[0] = 2;
        shape[rank - 1] = axes as i32;
        let mut native_input = Array::from_slice(&positions, &shape);
        if transposed_source {
            assert_eq!(axes, 2);
            let mut permutation: Vec<i32> = (0..rank as i32).collect();
            permutation.swap(0, rank - 1);
            native_input = native_input.transpose_axes(&permutation, &stream).unwrap();
        }
        let input = MlxTensor::from_array(native_input);
        // A noncontiguous rank-12 source makes the actual flattening reshape
        // exercise its metadata/copy alternative; use its logical order in the
        // independent scalar reference, without assuming a contiguous source.
        let positions = input
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<i32>()
            .unwrap();
        let mut frequencies = vec![0.0; spec.as_ref().frequency_count().unwrap()];
        spec.as_ref().fill_frequencies(&mut frequencies).unwrap();
        let prepared = PreparedMultiAxisRotary::new(spec.as_ref(), &frequencies).unwrap();
        let invoke = || -> Array {
            let (cosine, sine) =
                MlxTensor::multi_axis_rotary_embeddings_prepared(&input, prepared, &stream)
                    .unwrap();
            // One real shared concatenation keeps both native output branches
            // in the existing component harness's single completion root.
            MlxTensor::concatenate(&[cosine, sine], -1, &stream)
                .unwrap()
                .into()
        };
        let expected = invoke().evaluated().unwrap().try_to_vec::<f32>().unwrap();
        let (reference_cosine, reference_sine) =
            reference_multi_axis_rotary_embeddings_prepared(&positions, 2, prepared).unwrap();
        let width = spec.dimensions().unwrap() as usize;
        let reference: Vec<f32> = reference_cosine
            .chunks(width)
            .zip(reference_sine.chunks(width))
            .flat_map(|(cosine, sine)| cosine.iter().chain(sine).copied())
            .collect();
        assert_eq!(expected.len(), reference.len());
        for (actual, scalar) in expected.iter().zip(reference) {
            assert!(
                (actual - scalar).abs() <= 1e-5,
                "{layout:?}: {actual} vs {scalar}"
            );
        }
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let metadata = WorkspaceTensor::existing(
            context.layout(&shape, WorkspaceDtype::Int32).unwrap(),
            &context,
        )
        .unwrap();
        context.begin_span();
        let (cosine, sine) =
            WorkspaceTensor::multi_axis_rotary_embeddings_prepared(&metadata, prepared, &context)
                .unwrap();
        let output = WorkspaceTensor::concatenate(&[cosine, sine], -1, &context).unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        // The actual supplied frequency table is copied inside the original
        // scope; no generated ordinary frequency graph is used as a warm root.
        exercise_component(&mut fixture, &plan, &[input.as_array()], &expected, invoke);
    }
    drop(stream);
    fixture.finish();
}
