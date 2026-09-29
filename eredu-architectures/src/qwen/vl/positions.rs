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

/// Constructs three position axes and the persisted decode-time delta.
pub fn multimodal_position_ids(
    parts: &[PositionPart<'_>],
    merge: i32,
    expected: i32,
) -> Result<([Vec<i32>; 3], i32), String> {
    if parts.is_empty() || merge <= 0 || expected <= 0 {
        return Err("multimodal positions require parts and positive geometry".into());
    }
    // Validate the entire extent before allocating or expanding grid coordinates.
    let count = parts.iter().try_fold(0i32, |count, part| {
        let length = match part {
            PositionPart::Text(length) if *length > 0 => *length,
            PositionPart::Media(grid) if !grid.is_empty() => {
                grid.iter().try_fold(0i32, |sum, &(time, height, width)| {
                    if time <= 0
                        || height <= 0
                        || width <= 0
                        || height % merge != 0
                        || width % merge != 0
                    {
                        return Err("invalid merged media grid");
                    }
                    time.checked_mul(height / merge)
                        .and_then(|v| v.checked_mul(width / merge))
                        .and_then(|v| sum.checked_add(v))
                        .ok_or("position extent overflow")
                })?
            }
            _ => return Err("empty or invalid position part"),
        };
        count.checked_add(length).ok_or("position extent overflow")
    })?;
    if count != expected {
        return Err(format!(
            "position metadata describes {count} tokens, expected {expected}"
        ));
    }
    let mut positions = std::array::from_fn(|_| Vec::with_capacity(expected as usize));
    let mut current = 0i32;
    for part in parts {
        match part {
            PositionPart::Text(length) => {
                let end = current.checked_add(*length).ok_or("position overflow")?;
                for position in current..end {
                    for axis in &mut positions {
                        axis.push(position);
                    }
                }
                current = end;
            }
            PositionPart::Media(grid) => {
                for &(time, height, width) in *grid {
                    let (height, width) = (height / merge, width / merge);
                    current
                        .checked_add(time.max(height).max(width))
                        .ok_or("position overflow")?;
                    for temporal in 0..time {
                        for y in 0..height {
                            for x in 0..width {
                                positions[0].push(current + temporal);
                                positions[1].push(current + y);
                                positions[2].push(current + x);
                            }
                        }
                    }
                    current = current
                        .checked_add(height.max(width))
                        .ok_or("position overflow")?;
                }
            }
        }
    }
    let maximum = positions.iter().flatten().copied().max().unwrap_or(0);
    Ok((
        positions,
        maximum
            .checked_add(1)
            .and_then(|v| v.checked_sub(expected))
            .ok_or("position delta overflow")?,
    ))
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
        || sections.iter().sum::<i32>() != head_dim / 2
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
                let axis = if index % 3 == 1 && index < (sections[1] * 3) as usize {
                    1
                } else if index % 3 == 2 && index < (sections[2] * 3) as usize {
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
    let length = position_ids[0].len();
    if position_ids.iter().any(|axis| axis.len() != length) {
        return Err(Error::backend("mRoPE position axes have different lengths"));
    }
    let values = (0..length)
        .flat_map(|token| position_ids.iter().map(move |axis| axis[token]))
        .collect::<Vec<_>>();
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
    multi_axis_rotary_embeddings(
        position_ids,
        &mrope_spec(head_dim, theta, sections)?,
        context,
    )
}

/// Exact section policy shared by request preparation and mechanism reporting.
pub fn mrope_spec(
    head_dim: i32,
    theta: f32,
    sections: &[i32; 3],
) -> Result<MultiAxisRotarySpec, Error> {
    if head_dim <= 0
        || head_dim % 2 != 0
        || sections.iter().any(|section| *section <= 0)
        || sections
            .iter()
            .map(|section| i64::from(*section))
            .sum::<i64>()
            * 2
            != i64::from(head_dim)
    {
        return Err(Error::backend("invalid section-interleaved mRoPE geometry"));
    }
    let spec = MultiAxisRotarySpec {
        axes: sections
            .iter()
            .map(|section| RotaryAxisSpec {
                dimensions: section * 2,
                position_offset: 0,
            })
            .collect(),
        base: theta,
        minimum_position: 0,
        layout: MultiAxisRotaryLayout::RoundRobinSections,
    };
    spec.dimensions()?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grid_extent_is_checked_before_expanding_positions() {
        assert!(
            multimodal_position_ids(&[PositionPart::Media(&[(i32::MAX, 4, 4)])], 2, 1).is_err()
        );
        assert!(multimodal_position_ids(
            &[PositionPart::Text(i32::MAX), PositionPart::Text(1)],
            2,
            1
        )
        .is_err());
        assert!(multimodal_position_ids(&[PositionPart::Media(&[(1, 4, 4)])], 2, 3).is_err());
    }
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
            &mrope_spec(12, 10_000.0, &[2, 2, 2]).unwrap(),
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
    fn rotary_spec_rejects_odd_overflowing_and_nonfinite_geometry() {
        for (width, theta, sections) in [
            (7, 10_000.0, [1, 1, 1]),
            (6, 10_000.0, [i32::MAX, i32::MAX, i32::MAX]),
            (6, f32::NAN, [1, 1, 1]),
            (6, 0.0, [1, 1, 1]),
        ] {
            assert!(mrope_spec(width, theta, &sections).is_err());
        }
    }
}
