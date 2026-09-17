use super::*;
use crate::sequence_layout::{
    attention_chunk_lengths, bilinear_interpolation_samples, inverse_permutation, patch_positions,
    window_partition,
};

#[test]
fn bounded_grids_match_unchanged_pre_factor_equations() {
    use crate::sequence_layout::legacy_reference as old;
    for time in [1, 2] {
        for merge in [1, 2, 3] {
            for height in [merge, 2 * merge] {
                for width in [merge, 3 * merge] {
                    let grid = [(time, height, width), (1, merge, merge)];
                    assert_eq!(
                        attention_chunk_lengths(&grid).unwrap(),
                        old::attention_chunk_lengths(&grid).unwrap()
                    );
                    for traversal in [PatchTraversal::Raster, PatchTraversal::MergeMajor(merge)] {
                        assert_eq!(
                            patch_positions(&grid, traversal).unwrap(),
                            old::patch_positions(&grid, traversal).unwrap()
                        );
                        for mode in [
                            InterpolationMode::AlignCorners,
                            InterpolationMode::HalfPixel,
                        ] {
                            let prior =
                                old::bilinear_interpolation_samples(&grid, 3, 4, mode, traversal)
                                    .unwrap();
                            let mut indices = vec![-1; prior.len() * 4];
                            let mut weights = vec![f32::NAN; prior.len() * 4];
                            fill_bilinear_planes(
                                grid.iter().copied(),
                                3,
                                4,
                                mode,
                                traversal,
                                &mut indices,
                                &mut weights,
                            )
                            .unwrap();
                            for (position, sample) in prior.iter().enumerate() {
                                for corner in 0..4 {
                                    assert_eq!(
                                        indices[corner * prior.len() + position],
                                        sample.indices[corner] as i32
                                    );
                                    assert_eq!(
                                        weights[corner * prior.len() + position].to_bits(),
                                        sample.weights[corner].to_bits()
                                    );
                                }
                            }
                        }
                    }
                    for window in [merge, 2 * merge, 5 * merge] {
                        let prior = old::window_partition(&grid, merge, window, 1).unwrap();
                        let (mut permutation, mut inverse, mut chunks) = (
                            vec![-1; prior.permutation.len()],
                            vec![-1; prior.permutation.len()],
                            vec![-1; prior.chunk_lengths.len()],
                        );
                        fill_window_partition(
                            grid.iter().copied(),
                            merge,
                            window,
                            1,
                            &mut permutation,
                            &mut inverse,
                            &mut chunks,
                        )
                        .unwrap();
                        assert_eq!(permutation, prior.permutation);
                        assert_eq!(chunks, prior.chunk_lengths);
                        for (position, source) in permutation.iter().enumerate() {
                            assert_eq!(inverse[*source as usize], position as i32);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fixed_window_tables_preserve_independent_order_inverse_and_temporal_chunks() {
    let grid = [(1, 4, 6)];
    let population = window_partition_population(grid.iter().copied(), 2, 4, 1).unwrap();
    assert_eq!(
        population,
        WindowPartitionPopulation {
            groups: 6,
            windows: 2
        }
    );
    let (mut permutation, mut inverse, mut chunks) = ([99; 7], [99; 7], [99; 3]);
    fill_window_partition(
        grid.iter().copied(),
        2,
        4,
        1,
        &mut permutation,
        &mut inverse,
        &mut chunks,
    )
    .unwrap();
    assert_eq!(permutation, [0, 1, 3, 4, 2, 5, 99]);
    assert_eq!(inverse, [0, 1, 4, 2, 3, 5, 99]);
    assert_eq!(chunks, [16, 8, 99]);
    let ordinary = window_partition(&grid, 2, 4, 1).unwrap();
    assert_eq!(ordinary.permutation, permutation[..6]);
    assert_eq!(ordinary.chunk_lengths, chunks[..2]);
    assert_eq!(
        inverse_permutation(&ordinary.permutation).unwrap(),
        inverse[..6]
    );

    let rows = std::iter::once((2, 2, 2)).chain(std::iter::once((1, 2, 4)));
    let mut lengths = [99; 4];
    assert_eq!(fill_attention_chunk_lengths(rows, &mut lengths).unwrap(), 3);
    assert_eq!(lengths, [4, 4, 8, 99]);
    assert_eq!(
        attention_chunk_lengths(&[(2, 2, 2), (1, 2, 4)]).unwrap(),
        lengths[..3]
    );
}

#[test]
fn fixed_interpolation_planes_preserve_corner_arithmetic_and_borrowed_grid_order() {
    for mode in [
        InterpolationMode::AlignCorners,
        InterpolationMode::HalfPixel,
    ] {
        for traversal in [PatchTraversal::Raster, PatchTraversal::MergeMajor(2)] {
            let grid = [(1, 2, 4), (2, 2, 2)];
            let samples = bilinear_interpolation_samples(&grid, 3, 3, mode, traversal).unwrap();
            let mut indices = vec![-77; 4 * samples.len() + 1];
            let mut weights = vec![-77.; 4 * samples.len() + 1];
            assert_eq!(
                fill_bilinear_planes(
                    grid.iter().copied(),
                    3,
                    3,
                    mode,
                    traversal,
                    &mut indices,
                    &mut weights,
                )
                .unwrap(),
                samples.len()
            );
            for (position, sample) in samples.iter().enumerate() {
                for corner in 0..4 {
                    assert_eq!(
                        indices[corner * samples.len() + position],
                        sample.indices[corner] as i32
                    );
                    assert_eq!(
                        weights[corner * samples.len() + position].to_bits(),
                        sample.weights[corner].to_bits()
                    );
                }
            }
            assert_eq!(indices.last(), Some(&-77));
            assert_eq!(weights.last(), Some(&-77.));
        }
    }
    // Independent nontrivial edge values, in the existing 00/01/10/11 order.
    let mut indices = [0; 16];
    let mut weights = [0.; 16];
    fill_bilinear_planes(
        std::iter::once((1, 2, 2)),
        1,
        1,
        InterpolationMode::HalfPixel,
        PatchTraversal::Raster,
        &mut indices,
        &mut weights,
    )
    .unwrap();
    assert_eq!(
        [weights[0], weights[4], weights[8], weights[12]],
        [0., 0., 0., 0.5625]
    );
    fill_bilinear_planes(
        std::iter::once((1, 2, 2)),
        3,
        3,
        InterpolationMode::AlignCorners,
        PatchTraversal::Raster,
        &mut indices,
        &mut weights,
    )
    .unwrap();
    assert_eq!(
        [indices[0], indices[4], indices[8], indices[12]],
        [0, 1, 3, 4]
    );
    assert_eq!(
        [indices[3], indices[7], indices[11], indices[15]],
        [8, 8, 8, 8]
    );
    assert_eq!(
        [weights[3], weights[7], weights[11], weights[15]],
        [1., 0., 0., 0.]
    );
}

#[test]
fn fixed_population_rejections_leave_all_output_ranges_untouched() {
    let (mut indices, mut weights) = ([-5; 16], [-5.; 15]);
    assert!(matches!(
        fill_bilinear_planes(
            std::iter::once((1, 2, 2)),
            3,
            3,
            InterpolationMode::AlignCorners,
            PatchTraversal::Raster,
            &mut indices,
            &mut weights,
        ),
        Err(SequenceLayoutError::DestinationTooShort {
            required: 16,
            actual: 15
        })
    ));
    assert_eq!(indices, [-5; 16]);
    assert_eq!(weights, [-5.; 15]);

    let mut weights = [-5.; 16];
    assert!(matches!(
        fill_bilinear_planes(
            std::iter::once((1, 2, 2)),
            65536,
            65536,
            InterpolationMode::AlignCorners,
            PatchTraversal::Raster,
            &mut indices,
            &mut weights,
        ),
        Err(SequenceLayoutError::Overflow)
    ));
    assert_eq!(indices, [-5; 16]);
    assert_eq!(weights, [-5.; 16]);

    // f32 rounding at the signed maximum must reject the actual corner-add
    // overflow before writing, rather than panic or wrap in the fill pass.
    assert!(matches!(
        fill_bilinear_planes(
            std::iter::once((1, 2, 2)),
            i32::MAX,
            1,
            InterpolationMode::AlignCorners,
            PatchTraversal::Raster,
            &mut indices,
            &mut weights,
        ),
        Err(SequenceLayoutError::Overflow)
    ));
    assert_eq!(indices, [-5; 16]);
    assert_eq!(weights, [-5.; 16]);

    let (mut permutation, mut inverse, mut chunks) = ([-5; 6], [-5; 5], [-5; 2]);
    assert!(matches!(
        fill_window_partition(
            std::iter::once((1, 4, 6)),
            2,
            4,
            1,
            &mut permutation,
            &mut inverse,
            &mut chunks,
        ),
        Err(SequenceLayoutError::DestinationTooShort {
            required: 6,
            actual: 5
        })
    ));
    assert_eq!(permutation, [-5; 6]);
    assert_eq!(inverse, [-5; 5]);
    assert_eq!(chunks, [-5; 2]);
    assert!(matches!(
        fill_window_partition(
            std::iter::once((1, 1, i32::MAX)),
            1,
            i32::MAX,
            1,
            &mut permutation,
            &mut inverse,
            &mut chunks,
        ),
        Err(SequenceLayoutError::Overflow)
    ));
    assert_eq!(permutation, [-5; 6]);
    assert_eq!(inverse, [-5; 5]);
    assert_eq!(chunks, [-5; 2]);

    let mut positions = [-5; 8];
    assert!(matches!(
        fill_spatial_patch_positions(
            [(1, 2, 2), (i32::MAX, 1, 2)].into_iter(),
            PatchTraversal::Raster,
            &mut positions,
        ),
        Err(SequenceLayoutError::Overflow)
    ));
    assert_eq!(positions, [-5; 8]);
}

#[test]
fn gathered_tables_preserve_raster_order_and_reject_late_rows_before_writes() {
    let mut output = [-3; 9];
    assert_eq!(
        fill_gathered_patch_positions([(2, 1, 3), (1, 2, 1)].into_iter(), 3, 4, &mut output,)
            .unwrap(),
        8
    );
    assert_eq!(output, [0, 1, 2, 0, 1, 2, 0, 4, -3]);
    output.fill(-3);
    assert!(matches!(
        fill_gathered_patch_positions([(1, 1, 1), (1, 4, 1)].into_iter(), 3, 4, &mut output,),
        Err(SequenceLayoutError::GridExceedsSourceTable { index: 1, .. })
    ));
    assert_eq!(output, [-3; 9]);
    assert!(matches!(
        fill_gathered_patch_positions(std::iter::once((1, 2, 2)), 65536, i32::MAX, &mut output,),
        Err(SequenceLayoutError::Overflow)
    ));
    assert_eq!(output, [-3; 9]);
}

#[test]
fn spatial_fill_preserves_axes_and_unused_source_extent_is_not_a_new_restriction() {
    let grid = [(2, 2, 4)];
    let expected = crate::sequence_layout::legacy_reference::patch_positions(
        &grid,
        PatchTraversal::MergeMajor(2),
    )
    .unwrap();
    let mut spatial = [0; 32];
    assert_eq!(
        fill_spatial_patch_positions(
            grid.iter().copied(),
            PatchTraversal::MergeMajor(2),
            &mut spatial,
        )
        .unwrap(),
        spatial.len()
    );
    for (position, value) in expected.iter().enumerate() {
        assert_eq!(&spatial[position * 2..position * 2 + 2], &value[1..]);
    }
    // A one-position AlignCorners target accesses only cell zero even when
    // the unaccessed full table area exceeds a signed flattened index.
    let mut indices = [-5; 4];
    let mut weights = [-5.; 4];
    fill_bilinear_planes(
        std::iter::once((1, 1, 1)),
        65536,
        65536,
        InterpolationMode::AlignCorners,
        PatchTraversal::Raster,
        &mut indices,
        &mut weights,
    )
    .unwrap();
    assert_eq!(indices, [0; 4]);
    assert_eq!(weights, [1., 0., 0., 0.]);
}
