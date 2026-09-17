use super::*;

fn spec(windows: PatchAttentionWindows) -> PatchEncoderTableSpec {
    PatchEncoderTableSpec {
        merge: 2,
        positions: PatchPositionTableSpec::Interpolated {
            height: 3,
            width: 3,
            mode: InterpolationMode::AlignCorners,
            traversal: PatchTraversal::MergeMajor(2),
        },
        windows,
        rotary_axes: [RotaryAxisSpec {
            dimensions: 4,
            position_offset: 0,
        }; 2],
        rotary_base: 10000.,
        rotary_minimum: 0,
        rotary_layout: MultiAxisRotaryLayout::SplitHalves,
    }
}

#[test]
fn complete_patch_tables_preserve_real_aliases_and_all_ranges() {
    let grid = [(2, 2, 4), (1, 2, 2)];
    for windows in [
        PatchAttentionWindows::Full,
        PatchAttentionWindows::Windowed {
            window_size: 4,
            patch_size: 1,
        },
    ] {
        let plan = PatchEncoderTableLayout::new(spec(windows), grid.into_iter(), Some(20)).unwrap();
        let (mut integers, mut floats) = (
            vec![-2; plan.integer_count()],
            vec![-2.; plan.float_count()],
        );
        plan.fill(grid.into_iter(), &mut integers, &mut floats)
            .unwrap();
        let tables = PatchEncoderTables::new(plan, &integers, &floats).unwrap();
        assert_eq!(tables.full_chunks(), [8, 8, 4]);
        assert_eq!(tables.learned_indices().len(), 80);
        assert_eq!(tables.learned_weights().len(), 80);
        let legacy_spatial = crate::sequence_layout::legacy_reference::patch_positions(
            &grid,
            PatchTraversal::MergeMajor(2),
        )
        .unwrap()
        .into_iter()
        .flat_map(|[_, y, x]| [y, x])
        .collect::<Vec<_>>();
        assert_eq!(tables.spatial_positions(), legacy_spatial);
        assert_eq!(tables.rotary().frequencies(), [1., 0.01, 1., 0.01]);
        assert!(tables.learned_weights().iter().any(|value| *value > 0.));
        if windows == PatchAttentionWindows::Full {
            assert!(std::ptr::eq(
                tables.full_chunks().as_ptr(),
                tables.window_chunks().as_ptr()
            ));
            assert!(std::ptr::eq(
                tables.permutation().as_ptr(),
                tables.inverse().as_ptr()
            ));
            assert_eq!(tables.permutation(), [0, 1, 2, 3, 4]);
        } else {
            assert!(!std::ptr::eq(
                tables.full_chunks().as_ptr(),
                tables.window_chunks().as_ptr()
            ));
            assert!(!std::ptr::eq(
                tables.permutation().as_ptr(),
                tables.inverse().as_ptr()
            ));
            for (i, j) in tables.permutation().iter().enumerate() {
                assert_eq!(tables.inverse()[*j as usize], i as i32);
            }
        }
    }
}

#[test]
fn complete_patch_plan_short_float_buffer_and_late_source_reject_before_integer_writes() {
    let plan = PatchEncoderTableLayout::new(
        spec(PatchAttentionWindows::Full),
        std::iter::once((1, 2, 4)),
        Some(8),
    )
    .unwrap();
    let mut integers = vec![-11; plan.integer_count()];
    let mut floats = vec![-11.; plan.float_count() - 1];
    assert!(plan
        .fill(std::iter::once((1, 2, 4)), &mut integers, &mut floats)
        .is_err());
    assert!(integers.iter().all(|v| *v == -11));
    assert!(floats.iter().all(|v| *v == -11.));
    floats.push(-11.);
    assert!(plan
        .fill(
            [(1, 2, 4), (i32::MAX, 2, 2)].into_iter(),
            &mut integers,
            &mut floats
        )
        .is_err());
    assert!(integers.iter().all(|v| *v == -11));
    assert!(floats.iter().all(|v| *v == -11.));
}
