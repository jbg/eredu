use super::*;
use crate::multimodal::{reference_multi_axis_rotary_embeddings, MultiAxisRotarySpec};

#[test]
fn fixed_rotary_frequencies_preserve_prior_formulas_and_source_values() {
    for layout in [
        MultiAxisRotaryLayout::IndependentAxes,
        MultiAxisRotaryLayout::SplitHalves,
        MultiAxisRotaryLayout::RoundRobinSections,
    ] {
        for dimensions in [[4, 8, 2], [2, 2, 6], [0, 4, 4]] {
            if dimensions[0] == 0 && layout != MultiAxisRotaryLayout::RoundRobinSections {
                continue;
            }
            let axes = dimensions.map(|dimensions| RotaryAxisSpec {
                dimensions,
                position_offset: 1,
            });
            let spec = MultiAxisRotarySpecRef {
                axes: &axes,
                base: 10000.,
                minimum_position: 0,
                layout,
            };
            let count = spec.frequency_count().unwrap();
            let mut frequencies = vec![-13.; count + 1];
            assert_eq!(spec.fill_frequencies(&mut frequencies).unwrap(), count);
            assert_eq!(frequencies[count], -13.);
            let prior = if layout == MultiAxisRotaryLayout::RoundRobinSections {
                let width: i32 = dimensions.iter().sum();
                (0..width / 2)
                    .map(|index| 1.0 / spec.base.powf(2.0 * index as f32 / width as f32))
                    .collect::<Vec<_>>()
            } else {
                dimensions
                    .into_iter()
                    .flat_map(|width| {
                        (0..width)
                            .step_by(2)
                            .map(move |index| 1.0 / spec.base.powf(index as f32 / width as f32))
                    })
                    .collect()
            };
            assert_eq!(
                frequencies[..count]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                prior.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let prepared = PreparedMultiAxisRotary::new(spec, &frequencies[..count]).unwrap();
            let positions = [-3, 2, 7, 4, 8, 1];
            let actual =
                reference_multi_axis_rotary_embeddings_prepared(&positions, 2, prepared).unwrap();
            let ordinary = MultiAxisRotarySpec {
                axes: axes.to_vec(),
                base: spec.base,
                minimum_position: spec.minimum_position,
                layout,
            };
            let expected =
                reference_multi_axis_rotary_embeddings(&positions, 2, &ordinary).unwrap();
            for (a, e) in actual
                .0
                .iter()
                .chain(&actual.1)
                .zip(expected.0.iter().chain(&expected.1))
            {
                assert!((a - e).abs() <= 2e-6, "actual={a} expected={e}");
            }
            // Supplied values are real numerical input, not merely a geometry tag.
            let zeros = vec![0.; count];
            let zero = reference_multi_axis_rotary_embeddings_prepared(
                &positions,
                2,
                PreparedMultiAxisRotary::new(spec, &zeros).unwrap(),
            )
            .unwrap();
            assert!(zero.0.iter().all(|value| *value == 1.));
            assert!(zero.1.iter().all(|value| *value == 0.));
            assert!(actual.1.iter().any(|value| value.abs() > 0.01));
        }
    }
}

#[test]
fn rotary_rejects_late_axes_overflow_and_short_tables_without_writing() {
    for dimensions in [[4, -2], [4, 3], [4, 0], [i32::MAX - 1, 2]] {
        let axes = dimensions.map(|dimensions| RotaryAxisSpec {
            dimensions,
            position_offset: 0,
        });
        let spec = MultiAxisRotarySpecRef {
            axes: &axes,
            base: 10000.,
            minimum_position: 0,
            layout: MultiAxisRotaryLayout::SplitHalves,
        };
        let mut output = [-7.; 8];
        assert!(spec.fill_frequencies(&mut output).is_err());
        assert_eq!(output, [-7.; 8]);
    }
    let axes = [RotaryAxisSpec {
        dimensions: 4,
        position_offset: i32::MAX,
    }; 2];
    let spec = MultiAxisRotarySpecRef {
        axes: &axes,
        base: 100.,
        minimum_position: 0,
        layout: MultiAxisRotaryLayout::SplitHalves,
    };
    let mut short = [-7.; 3];
    assert_eq!(
        spec.fill_frequencies(&mut short),
        Err(RotaryTableError::Capacity {
            required: 4,
            actual: 3
        })
    );
    assert_eq!(short, [-7.; 3]);
    assert!(PreparedMultiAxisRotary::new(spec, &[0.; 5]).is_err());
    let source = [1., 0., 1., 0.];
    let prepared = PreparedMultiAxisRotary::new(spec, &source).unwrap();
    let (cosine, sine) =
        reference_multi_axis_rotary_embeddings_prepared(&[i32::MAX, i32::MIN], 1, prepared)
            .unwrap();
    assert_eq!(cosine[0], (i32::MAX as f32).cos());
    assert_eq!(sine[0], (i32::MAX as f32).sin());
    assert_eq!(cosine[2], 1.);
    assert_eq!(sine[2], 0.);
}
