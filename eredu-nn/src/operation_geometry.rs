//! Tensor-independent checks shared by native neural-operator adapters.

use crate::Error;

/// Validated final-axis normalization geometry and scalar stability policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NormalizationGeometry {
    width: i32,
}

impl NormalizationGeometry {
    /// Checks a nonempty feature axis and a finite, positive epsilon.
    pub fn new(shape: &[i32], epsilon: f32) -> Result<Self, Error> {
        let width = shape.last().copied().unwrap_or(0);
        if width <= 0
            || shape.iter().any(|dimension| *dimension < 0)
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(Error::backend("invalid final-axis normalization geometry"));
        }
        Ok(Self { width })
    }

    /// Number of final-axis features.
    pub const fn width(self) -> i32 {
        self.width
    }
}

/// Validated matching input/gate tensors split into equal feature groups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedNormalizationGeometry {
    width: i32,
    group_width: i32,
}

impl GroupedNormalizationGeometry {
    /// Checks normalization scalars, exact gate shape, and positive divisibility.
    ///
    /// Learned scale layout remains explicit at the caller: some mechanisms
    /// accept a repeated group scale while others require a full feature scale.
    pub fn new(
        input_shape: &[i32],
        gate_shape: &[i32],
        groups: i32,
        epsilon: f32,
    ) -> Result<Self, Error> {
        let width = NormalizationGeometry::new(input_shape, epsilon)?.width();
        if input_shape != gate_shape || groups <= 0 || width % groups != 0 {
            return Err(Error::backend("invalid grouped normalization geometry"));
        }
        Ok(Self {
            width,
            group_width: width / groups,
        })
    }

    /// Complete final-axis feature count.
    pub const fn width(self) -> i32 {
        self.width
    }

    /// Features normalized together within each group.
    pub const fn group_width(self) -> i32 {
        self.group_width
    }
}

/// Checked absolute positions for a causal mask, without tensor allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CausalMaskGeometry {
    sequence: i32,
    offset: i32,
    keys: i32,
    max_past: Option<i32>,
}

impl CausalMaskGeometry {
    /// Checks nonnegative positions and a checked sequence endpoint.
    ///
    /// `max_past` is an inclusive backward distance: zero retains only the
    /// current position, and one additionally permits its predecessor. Sliding
    /// attention APIs that specify a token count pass `window - 1` here.
    pub fn new(sequence: i32, offset: i32, max_past: Option<i32>) -> Result<Self, Error> {
        let keys = offset
            .checked_add(sequence)
            .ok_or_else(|| Error::backend("causal mask position endpoint overflows"))?;
        if sequence < 0 || offset < 0 || max_past.is_some_and(|distance| distance < 0) {
            return Err(Error::backend("invalid causal mask position geometry"));
        }
        Ok(Self {
            sequence,
            offset,
            keys,
            max_past,
        })
    }

    /// Number of query rows.
    pub const fn sequence(self) -> i32 {
        self.sequence
    }

    /// Absolute position of the first query.
    pub const fn offset(self) -> i32 {
        self.offset
    }

    /// Number of key columns, including positions before the current sequence.
    pub const fn keys(self) -> i32 {
        self.keys
    }

    /// Whether one local query row can observe an absolute key position.
    pub fn allows(self, query: i32, key: i32) -> bool {
        if query < 0 || query >= self.sequence || key < 0 || key >= self.keys {
            return false;
        }
        let position = self.offset + query;
        key <= position
            && self
                .max_past
                .is_none_or(|distance| position - key <= distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_checks_rank_features_and_stability_before_tensor_work() {
        assert_eq!(
            NormalizationGeometry::new(&[2, 4], 0.25).unwrap().width(),
            4
        );
        assert!(NormalizationGeometry::new(&[0, 4], 0.25).is_ok());
        for shape in [&[][..], &[0], &[-1], &[-1, 4]] {
            assert!(NormalizationGeometry::new(shape, 0.25).is_err());
        }
        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(NormalizationGeometry::new(&[4], epsilon).is_err());
        }
    }

    #[test]
    fn grouped_geometry_has_exact_shape_and_checked_divisibility() {
        for width in 1..32 {
            for groups in -1..35 {
                let geometry =
                    GroupedNormalizationGeometry::new(&[2, width], &[2, width], groups, 1e-5);
                assert_eq!(geometry.is_ok(), groups > 0 && width % groups == 0);
                if let Ok(geometry) = geometry {
                    assert_eq!(geometry.width(), width);
                    assert_eq!(geometry.group_width() * groups, width);
                }
            }
        }
        assert!(GroupedNormalizationGeometry::new(&[2, 4], &[1, 4], 2, 1e-5).is_err());
        assert!(GroupedNormalizationGeometry::new(&[2, 4], &[2, 4], 2, f32::NAN).is_err());
    }

    #[test]
    fn causal_distance_matches_an_independent_integer_oracle() {
        for sequence in 0..6 {
            for offset in 0..6 {
                for distance in [None, Some(0), Some(1), Some(3), Some(i32::MAX)] {
                    let geometry = CausalMaskGeometry::new(sequence, offset, distance).unwrap();
                    assert_eq!(geometry.keys(), sequence + offset);
                    for query in -1..=sequence {
                        for key in -1..=geometry.keys() {
                            let position = i64::from(offset) + i64::from(query);
                            let expected = query >= 0
                                && query < sequence
                                && key >= 0
                                && key < geometry.keys()
                                && i64::from(key) <= position
                                && distance.is_none_or(|distance| {
                                    i64::from(key) >= position - i64::from(distance)
                                });
                            assert_eq!(geometry.allows(query, key), expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn causal_geometry_rejects_negative_and_overflowed_endpoints() {
        for (sequence, offset, distance) in [
            (-1, 0, None),
            (1, -1, None),
            (1, 0, Some(-1)),
            (1, i32::MAX, None),
        ] {
            assert!(CausalMaskGeometry::new(sequence, offset, distance).is_err());
        }
        let geometry = CausalMaskGeometry::new(1, i32::MAX - 1, Some(i32::MAX)).unwrap();
        assert!(geometry.allows(0, 0));
        assert!(geometry.allows(0, i32::MAX - 1));
    }
}
