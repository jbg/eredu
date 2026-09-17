//! Qwen3-VL temporal/height/width position construction and section interleaving.

use eredu_nn::{
    multimodal::{
        multi_axis_rotary_embeddings, MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
    },
    Error, Tensor,
};

/// One ordered prepared-input component used for position construction.
#[derive(Debug, Clone, Copy)]
pub enum PositionPart<'a> {
    /// Consecutive text tokens.
    Text(i32),
    /// One or more image/video patch grids before spatial merging.
    Media(&'a [(i32, i32, i32)]),
}

mod kernel;
pub(crate) use kernel::{
    emit_positions, validate_positions, GridRows, PositionComponent, PositionDestination,
};

/// Constructs three position axes and the persisted decode-time delta. The
/// shared count pass rejects every overflow before any destination allocation.
pub fn multimodal_position_ids(
    parts: &[PositionPart<'_>],
    merge: i32,
    expected: i32,
) -> Result<([Vec<i32>; 3], i32), String> {
    positions_with_destination(parts, merge, expected, |count| Ok(vec![0; count]), |message| message.to_string())
}

pub(crate) fn multimodal_position_ids_with_metadata(
    parts: &[PositionPart<'_>], merge: i32, expected: i32,
    context: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<([Vec<i32>; 3], i32), Error> {
    let metadata = crate::decoder::identity::Metadata::new(context);
    metadata.controls::<([Vec<i32>; 3], PositionDestination<'_>, PositionComponent<'_>, i32)>()?;
    positions_with_destination(parts, merge, expected, |count| {
        let mut values = metadata.vector(count)?;
        values.resize(count, 0);
        Ok(values)
    }, |message| metadata.error(message))
}

fn positions_with_destination<E>(
    parts: &[PositionPart<'_>], merge: i32, expected: i32,
    vector: impl Fn(usize) -> Result<Vec<i32>, E>,
    error: impl Fn(std::fmt::Arguments<'_>) -> E,
) -> Result<([Vec<i32>; 3], i32), E> {
    let source = parts.iter().map(|part| {
        Ok(match part {
            PositionPart::Text(length) => PositionComponent::Text(i64::from(*length)),
            PositionPart::Media(rows) => PositionComponent::Media(GridRows::Tuples(rows)),
        })
    });
    let count = usize::try_from(expected)
        .map_err(|_| error(format_args!("multimodal positions require positive geometry")))?;
    validate_positions(source.clone(), merge, count).map_err(|cause| error(format_args!("{cause}")))?;
    let mut positions = [vector(count)?, vector(count)?, vector(count)?];
    let [first, second, third] = &mut positions;
    let delta = emit_positions(
        source, merge, count,
        PositionDestination { axes: [first, second, third], prefix: None },
    ).map_err(|cause| error(format_args!("{cause}")))?;
    Ok((positions, delta))
}

/// Scalar reference for Qwen section-interleaved mRoPE cosine/sine values.
pub fn mrope_values(
    position_ids: &[Vec<i32>; 3],
    head_dim: i32,
    theta: f32,
    sections: &[i32; 3],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let length = position_ids[0].len();
    if head_dim <= 0
        || head_dim % 2 != 0
        || !theta.is_finite()
        || theta <= 0.0
        || position_ids.iter().any(|axis| axis.len() != length)
        || sections.iter().any(|v| *v < 0)
        || sections
            .iter()
            .try_fold(0_i32, |sum, section| sum.checked_add(*section))
            != Some(head_dim / 2)
    {
        return Err("invalid mRoPE geometry".into());
    }
    let half = head_dim / 2;
    let frequency = (0..half)
        .map(|index| 1.0 / theta.powf(2.0 * index as f32 / head_dim as f32))
        .collect::<Vec<_>>();
    let mut cosine = Vec::with_capacity(length * head_dim as usize);
    let mut sine = Vec::with_capacity(length * head_dim as usize);
    for ((&first, &second), &third) in position_ids[0]
        .iter()
        .zip(&position_ids[1])
        .zip(&position_ids[2])
        .take(length)
    {
        let axes = [first, second, third];
        let angles = frequency
            .iter()
            .enumerate()
            .map(|(index, inv)| {
                let axis = if index % 3 == 1 && index < sections[1] as usize * 3 {
                    1
                } else if index % 3 == 2 && index < sections[2] as usize * 3 {
                    2
                } else {
                    0
                };
                axes[axis] as f32 * inv
            })
            .collect::<Vec<_>>();
        for angle in angles.iter().chain(&angles) {
            cosine.push(angle.cos());
            sine.push(angle.sin());
        }
    }
    Ok((cosine, sine))
}

/// Converts host position metadata to a backend-native `[sequence, 3]` tensor.
pub fn position_ids_tensor<T: Tensor>(
    position_ids: &[Vec<i32>; 3],
    context: &T::Context,
) -> Result<T, Error> {
    position_ids_tensor_with_metadata(std::array::from_fn(|axis| position_ids[axis].as_slice()), context, None)
}

pub(crate) fn position_ids_tensor_with_metadata<T: Tensor>(
    position_ids: [&[i32]; 3],
    context: &T::Context,
    metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<T, Error> {
    let metadata = crate::decoder::identity::Metadata::new(metadata);
    metadata.controls::<(T, Vec<i32>, [&[i32]; 3], [i32; 2])>()?;
    let length = position_ids[0].len();
    if position_ids.iter().any(|axis| axis.len() != length) {
        return Err(metadata.error(format_args!("mRoPE position axes have different lengths")));
    }
    let count = length.checked_mul(3)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
    let mut values = metadata.vector(count)?;
    values.extend((0..length).flat_map(|token| position_ids.iter().map(move |axis| axis[token])));
    T::from_i32_slice(&values, &[length as i32, 3], context)
}

/// Builds exact section-interleaved Qwen multimodal rotary embeddings.
pub fn mrope_embeddings<T: Tensor>(
    position_ids: &T,
    head_dim: i32,
    theta: f32,
    sections: &[i32; 3],
    context: &T::Context,
) -> Result<(T, T), Error> {
    mrope_embeddings_with_metadata(position_ids, head_dim, theta, sections, context, None)
}

pub(crate) fn mrope_embeddings_with_metadata<T: Tensor>(
    position_ids: &T,
    head_dim: i32,
    theta: f32,
    sections: &[i32; 3],
    context: &T::Context,
    metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<(T, T), Error> {
    let metadata = crate::decoder::identity::Metadata::new(metadata);
    metadata.controls::<((T, T), MultiAxisRotarySpec, Vec<RotaryAxisSpec>)>()?;
    if head_dim <= 0
        || head_dim % 2 != 0
        || sections.iter().any(|section| *section < 0)
        || sections.iter().try_fold(0_i32, |sum, section| sum.checked_add(*section)) != Some(head_dim / 2)
    {
        return Err(metadata.error(format_args!("invalid section-interleaved mRoPE geometry")));
    }
    let mut axes = metadata.vector(sections.len())?;
    axes.extend(sections.iter().map(|section| RotaryAxisSpec {
        dimensions: section * 2,
        position_offset: 0,
    }));
    let spec = MultiAxisRotarySpec {
        axes,
        base: theta,
        minimum_position: 0,
        layout: MultiAxisRotaryLayout::RoundRobinSections,
    };
    match metadata.context() {
        Some(metadata) => eredu_nn::multimodal::multi_axis_rotary_embeddings_with_metadata(
            position_ids, &spec, context, metadata,
        ),
        None => multi_axis_rotary_embeddings(position_ids, &spec, context),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positions_preserve_text_media_order_and_delta() {
        let grid = [(1, 4, 4)];
        let (ids, delta) = multimodal_position_ids(
            &[
                PositionPart::Text(2),
                PositionPart::Media(&grid),
                PositionPart::Text(1),
            ],
            2,
            7,
        )
        .unwrap();
        assert_eq!(ids[0], [0, 1, 2, 2, 2, 2, 4]);
        assert_eq!(ids[1], [0, 1, 2, 2, 3, 3, 4]);
        assert_eq!(delta, -2);
    }
    #[test]
    fn mrope_rejects_bad_sections_and_emits_full_heads() {
        let ids = [vec![0, 1], vec![0, 2], vec![0, 3]];
        let (cos, sin) = mrope_values(&ids, 12, 10_000.0, &[2, 2, 2]).unwrap();
        assert_eq!(cos.len(), 24);
        assert_eq!(sin.len(), 24);
        assert!(mrope_values(&ids, 12, 10_000.0, &[2, 2, 1]).is_err());
    }

    #[test]
    fn round_robin_general_layout_matches_qwen_reference() {
        use eredu_nn::multimodal::reference_multi_axis_rotary_embeddings;
        let ids = [vec![1, 2], vec![3, 4], vec![5, 6]];
        let expected = mrope_values(&ids, 12, 10_000.0, &[2, 2, 2]).unwrap();
        let positions = (0..2)
            .flat_map(|token| ids.iter().map(move |axis| axis[token]))
            .collect::<Vec<_>>();
        let actual = reference_multi_axis_rotary_embeddings(
            &positions,
            2,
            &MultiAxisRotarySpec {
                axes: vec![
                    RotaryAxisSpec {
                        dimensions: 4,
                        position_offset: 0,
                    },
                    RotaryAxisSpec {
                        dimensions: 4,
                        position_offset: 0,
                    },
                    RotaryAxisSpec {
                        dimensions: 4,
                        position_offset: 0,
                    },
                ],
                base: 10_000.0,
                minimum_position: 0,
                layout: MultiAxisRotaryLayout::RoundRobinSections,
            },
        )
        .unwrap();
        assert!(expected
            .0
            .iter()
            .zip(actual.0)
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(expected
            .1
            .iter()
            .zip(actual.1)
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn qwen_scalar_zero_sections_keep_the_temporal_fallback() {
        let ids = [vec![2], vec![5], vec![9]];
        for (sections, selected) in [
            ([4, 0, 0], [0, 0, 0, 0]),
            ([0, 4, 0], [0, 1, 0, 0]),
            ([0, 0, 4], [0, 0, 2, 0]),
            ([0, 2, 2], [0, 1, 2, 0]),
            ([2, 0, 2], [0, 0, 2, 0]),
            ([2, 2, 0], [0, 1, 0, 0]),
        ] {
            let (cos, sin) = mrope_values(&ids, 8, 100., &sections).unwrap();
            for i in 0..4 {
                let angle = f64::from(ids[selected[i]][0]) / 100_f64.powf(i as f64 / 4.);
                for j in [i, i + 4] {
                    assert!((f64::from(cos[j]) - angle.cos()).abs() < 2e-6);
                    assert!((f64::from(sin[j]) - angle.sin()).abs() < 2e-6);
                }
            }
        }
        for sections in [[0, 0, 0], [-1, 2, 3], [1, 1, 1], [i32::MAX, i32::MAX, 2]] {
            assert!(mrope_values(&ids, 8, 100., &sections).is_err());
        }
    }
}
