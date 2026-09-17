//! Tensor-independent checks shared by native neural-operator adapters.

use crate::Error;

/// Exact shapes for centroid-selected vocabulary projection. Empty batch or
/// sequence axes are valid; feature, vocabulary and centroid axes are positive.
pub fn validate_masked_output_geometry(
    hidden: &[i32],
    weight: &[i32],
    centroids: &[i32],
    ordering: &[i32],
    top: i32,
    margin: f32,
) -> Result<(), Error> {
    validate_masked_output_geometry_fixed(hidden, weight, centroids, ordering, top, margin)
        .map_err(|_| Error::backend(format!(
            "invalid masked-output geometry: hidden={hidden:?} weight={weight:?} centroids={centroids:?} ordering={ordering:?} top={top} margin={margin}"
        )))
}

/// The selected masked-output shapes or scalar policy are inconsistent.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("invalid masked-output geometry")]
pub struct MaskedOutputGeometryError;

/// Checks the identical shape and scalar rules without formatting or storage.
/// Callers retain the actual borrowed shapes for ordinary detailed diagnostics.
pub fn validate_masked_output_geometry_fixed(
    hidden: &[i32],
    weight: &[i32],
    centroids: &[i32],
    ordering: &[i32],
    top: i32,
    margin: f32,
) -> Result<(), MaskedOutputGeometryError> {
    if hidden.len() != 3
        || weight.len() != 2
        || centroids.len() != 3
        || ordering.len() != 1
        || hidden
            .iter()
            .chain(weight)
            .chain(centroids)
            .chain(ordering)
            .any(|n| *n < 0)
        || hidden[..2] != centroids[..2]
        || hidden[2] <= 0
        || hidden[2] != weight[1]
        || weight[0] <= 0
        || ordering[0] != weight[0]
        || centroids[2] <= 0
        || weight[0] % centroids[2] != 0
        || top <= 0
        || top > centroids[2]
        || !margin.is_finite()
        || margin <= 0.0
    {
        return Err(MaskedOutputGeometryError);
    }
    Ok(())
}

/// Fixed causal/sliding geometry failures with the ordinary diagnostic text.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum AttentionGeometryError {
    /// Adding a sliding query span to its absolute offset overflows.
    #[error("sliding attention position endpoint overflows")]
    SlidingEndpoint,
    /// The retained sliding history cannot cover the selected queries.
    #[error("invalid sliding attention retained history")]
    SlidingHistory,
    /// Adding causal mask rows to their absolute offset overflows.
    #[error("causal mask position endpoint overflows")]
    CausalEndpoint,
    /// A causal row count, offset, or backward distance is negative.
    #[error("invalid causal mask position geometry")]
    CausalPosition,
}

/// Retained causal history for a sliding attention submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlidingAttentionGeometry {
    key_origin: i32,
}
impl SlidingAttentionGeometry {
    /// Keys must include all current queries and end at the query endpoint.
    /// `window` counts tokens, including the current position.
    pub fn new(queries: i32, keys: i32, window: i32, offset: i32) -> Result<Self, Error> {
        Self::new_fixed(queries, keys, window, offset).map_err(Error::backend)
    }
    /// Checks the identical retained-history rules without a diagnostic allocation.
    pub fn new_fixed(
        queries: i32,
        keys: i32,
        window: i32,
        offset: i32,
    ) -> Result<Self, AttentionGeometryError> {
        let endpoint = offset
            .checked_add(queries)
            .ok_or(AttentionGeometryError::SlidingEndpoint)?;
        if queries <= 0 || keys < queries || window <= 0 || offset < 0 || keys > endpoint {
            return Err(AttentionGeometryError::SlidingHistory);
        }
        Ok(Self {
            key_origin: endpoint - keys,
        })
    }
    /// Absolute position represented by the first retained key.
    pub const fn key_origin(self) -> i32 {
        self.key_origin
    }
}

/// Fixed normalization shape/scalar refusal with the ordinary diagnostic text.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum NormalizationGeometryError {
    /// The final feature axis or numerical epsilon is invalid.
    #[error("invalid final-axis normalization geometry")]
    FinalAxis,
    /// The gate shape or equal feature grouping is invalid.
    #[error("invalid grouped normalization geometry")]
    Groups,
}

/// Validated final-axis normalization geometry and scalar stability policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NormalizationGeometry {
    width: i32,
}

impl NormalizationGeometry {
    /// Checks a nonempty feature axis and a finite, positive epsilon.
    pub fn new(shape: &[i32], epsilon: f32) -> Result<Self, Error> {
        Self::new_fixed(shape, epsilon).map_err(Error::backend)
    }

    /// Checks the same shape and epsilon without allocating a diagnostic.
    pub fn new_fixed(shape: &[i32], epsilon: f32) -> Result<Self, NormalizationGeometryError> {
        let width = shape.last().copied().unwrap_or(0);
        if width <= 0
            || shape.iter().any(|dimension| *dimension < 0)
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(NormalizationGeometryError::FinalAxis);
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
        Self::new_fixed(input_shape, gate_shape, groups, epsilon).map_err(Error::backend)
    }

    /// Checks normalization first, then gate/group geometry, without creating
    /// a diagnostic string or any shape destination.
    pub fn new_fixed(
        input_shape: &[i32],
        gate_shape: &[i32],
        groups: i32,
        epsilon: f32,
    ) -> Result<Self, NormalizationGeometryError> {
        let width = NormalizationGeometry::new_fixed(input_shape, epsilon)?.width();
        if input_shape != gate_shape || groups <= 0 || width % groups != 0 {
            return Err(NormalizationGeometryError::Groups);
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

/// Fixed pooling-mask validation failure with its ordinary diagnostic text.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("invalid pooling mask position geometry")]
pub struct PoolingMaskGeometryError;

/// Checked coordinates for eligibility of complete append-only pooling windows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoolingMaskGeometry {
    queries: i32,
    pooled: i32,
    offset: i32,
    ratio: i32,
}

impl PoolingMaskGeometry {
    /// Checks the native half-open query endpoint, positive ratio and extents.
    /// Retained pooled columns may include windows invisible to early queries.
    pub fn new(queries: i32, pooled: i32, offset: i32, ratio: i32) -> Result<Self, Error> {
        Self::new_fixed(queries, pooled, offset, ratio).map_err(Error::backend)
    }
    /// Checks the identical coordinates without constructing a diagnostic owner.
    pub fn new_fixed(
        queries: i32,
        pooled: i32,
        offset: i32,
        ratio: i32,
    ) -> Result<Self, PoolingMaskGeometryError> {
        if queries < 0
            || pooled < 0
            || offset < 0
            || ratio <= 0
            || offset
                .checked_add(queries)
                .and_then(|end| end.checked_add(1))
                .is_none()
        {
            return Err(PoolingMaskGeometryError);
        }
        Ok(Self {
            queries,
            pooled,
            offset,
            ratio,
        })
    }
    /// Query rows.
    pub const fn queries(self) -> i32 {
        self.queries
    }
    /// Retained pooled columns.
    pub const fn pooled(self) -> i32 {
        self.pooled
    }
    /// Absolute first source-token query position.
    pub const fn offset(self) -> i32 {
        self.offset
    }
    /// Source tokens per complete pooled window.
    pub const fn ratio(self) -> i32 {
        self.ratio
    }
    /// Whether this query may consume the complete source window of this column.
    pub fn allows(self, query: i32, pooled: i32) -> bool {
        query >= 0
            && query < self.queries
            && pooled >= 0
            && pooled < self.pooled
            && pooled < (self.offset + query + 1) / self.ratio
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
        Self::new_fixed(sequence, offset, max_past).map_err(Error::backend)
    }
    /// Checks the identical ordered position rules without diagnostic allocation.
    pub fn new_fixed(
        sequence: i32,
        offset: i32,
        max_past: Option<i32>,
    ) -> Result<Self, AttentionGeometryError> {
        let keys = offset
            .checked_add(sequence)
            .ok_or(AttentionGeometryError::CausalEndpoint)?;
        if sequence < 0 || offset < 0 || max_past.is_some_and(|distance| distance < 0) {
            return Err(AttentionGeometryError::CausalPosition);
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

    /// Inclusive backward distance, when the mask has a sliding window.
    pub const fn max_past(self) -> Option<i32> {
        self.max_past
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
    fn fixed_masked_output_checks_the_same_source_shapes_and_retains_ordinary_details() {
        for (hidden, weight, centroids, ordering, top, margin, accepted) in [
            (
                &[2, 3, 7][..],
                &[8, 7][..],
                &[2, 3, 2][..],
                &[8][..],
                1,
                1.0,
                true,
            ),
            (
                &[0, 3, 7][..],
                &[8, 7][..],
                &[0, 3, 2][..],
                &[8][..],
                2,
                1.0,
                true,
            ),
            (
                &[][..],
                &[8, 7][..],
                &[2, 3, 2][..],
                &[8][..],
                0,
                f32::NAN,
                false,
            ),
            (
                &[2, -1, 7][..],
                &[8, 7][..],
                &[2, 3, 2][..],
                &[8][..],
                1,
                1.0,
                false,
            ),
            (
                &[2, 3, 7][..],
                &[8, 7][..],
                &[2, 3, 3][..],
                &[8][..],
                1,
                1.0,
                false,
            ),
            (
                &[2, 3, 7][..],
                &[8, 7][..],
                &[2, 3, 2][..],
                &[8][..],
                3,
                1.0,
                false,
            ),
            (
                &[2, 3, 7][..],
                &[8, 7][..],
                &[2, 3, 2][..],
                &[8][..],
                1,
                f32::INFINITY,
                false,
            ),
        ] {
            let fixed = validate_masked_output_geometry_fixed(
                hidden, weight, centroids, ordering, top, margin,
            );
            assert_eq!(fixed.is_ok(), accepted);
            let ordinary =
                validate_masked_output_geometry(hidden, weight, centroids, ordering, top, margin);
            assert_eq!(ordinary.is_ok(), accepted);
            if !accepted {
                assert_eq!(fixed, Err(MaskedOutputGeometryError));
                let detail = format!(
                    "invalid masked-output geometry: hidden={hidden:?} weight={weight:?} centroids={centroids:?} ordering={ordering:?} top={top} margin={margin}"
                );
                assert_eq!(
                    ordinary.unwrap_err().to_string(),
                    Error::backend(detail).to_string()
                );
            }
        }
    }

    #[test]
    fn pooling_mask_matches_completed_group_visibility_and_checks_overflow() {
        for queries in 0..7 {
            for offset in 0..13 {
                for ratio in 1..7 {
                    let pooled = (offset + queries) / ratio;
                    let geometry =
                        PoolingMaskGeometry::new(queries, pooled, offset, ratio).unwrap();
                    for query in -1..=queries {
                        for column in -1..=pooled {
                            let expected = query >= 0
                                && query < queries
                                && column >= 0
                                && column < pooled
                                && (i64::from(column) + 1) * i64::from(ratio)
                                    <= i64::from(offset) + i64::from(query) + 1;
                            assert_eq!(geometry.allows(query, column), expected);
                        }
                    }
                }
            }
        }
        for (queries, pooled, offset, ratio) in [
            (-1, 1, 0, 4),
            (1, -1, 0, 4),
            (1, 1, -1, 4),
            (1, 1, 0, 0),
            (1, 1, i32::MAX - 1, 4),
        ] {
            assert!(PoolingMaskGeometry::new(queries, pooled, offset, ratio).is_err());
        }
        assert!(PoolingMaskGeometry::new(1, 1, i32::MAX - 2, 4).is_ok());
    }

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

/// Fixed geometry refusal for the shared selective recurrent scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("invalid selective state-space scan geometry")]
pub struct SelectiveScanGeometryError;

/// Validated borrowed scan dimensions. Carries no tensor or execution authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectiveScanGeometry {
    values: [i32; 4],
    state_dimensions: i32,
    chunk: i32,
}
impl SelectiveScanGeometry {
    /// Checks the actual seven inputs, optional state and retained scalar policy.
    pub fn new(
        shapes: [&[i32]; 7],
        initial: Option<&[i32]>,
        chunk: usize,
        floor: f32,
    ) -> Result<Self, SelectiveScanGeometryError> {
        let [values, input, output, step, bias, transition, skip] = shapes;
        let v: [i32; 4] = values.try_into().map_err(|_| SelectiveScanGeometryError)?;
        if v.iter().any(|n| *n < 0)
            || input.len() != 4
            || input[..3] != v[..3]
            || input[3] < 0
            || output != input
            || step != &v[..3]
            || bias != [v[2]]
            || transition != [v[2]]
            || skip != [v[2]]
            || chunk == 0
            || !floor.is_finite()
            || floor < 0.0
        {
            return Err(SelectiveScanGeometryError);
        }
        let state = [v[0], v[2], v[3], input[3]];
        if initial.is_some_and(|shape| shape != state) {
            return Err(SelectiveScanGeometryError);
        }
        Ok(Self {
            values: v,
            state_dimensions: input[3],
            chunk: i32::try_from(chunk).unwrap_or(i32::MAX),
        })
    }
    /// Batch, sequence, heads and per-head value width.
    pub const fn values(self) -> [i32; 4] {
        self.values
    }
    /// FP32 recurrent state shape.
    pub const fn state(self) -> [i32; 4] {
        [
            self.values[0],
            self.values[2],
            self.values[3],
            self.state_dimensions,
        ]
    }
    /// Number of token rows in the actual source.
    pub const fn sequence(self) -> i32 {
        self.values[1]
    }
    /// Checked next chunk endpoint; avoids overflowing at a large caller cap.
    pub fn chunk_end(self, start: i32) -> Option<i32> {
        (0..=self.sequence())
            .contains(&start)
            .then(|| start + self.chunk.min(self.sequence() - start))
    }
}

#[cfg(test)]
mod selective_scan_geometry_tests {
    use super::*;

    #[test]
    fn scan_geometry_preserves_shapes_and_large_chunk_endpoints() {
        let values = [2, i32::MAX, 3, 5];
        let vectors = [2, i32::MAX, 3, 7];
        let steps = [2, i32::MAX, 3];
        let heads = [3];
        let state = [2, 3, 5, 7];
        let source = [
            &values[..],
            &vectors,
            &vectors,
            &steps,
            &heads,
            &heads,
            &heads,
        ];
        let geometry = SelectiveScanGeometry::new(source, Some(&state), usize::MAX, 0.01).unwrap();
        assert_eq!(geometry.state(), state);
        assert_eq!(geometry.chunk_end(0), Some(i32::MAX));
        assert_eq!(geometry.chunk_end(i32::MAX - 1), Some(i32::MAX));
        assert_eq!(geometry.chunk_end(-1), None);
        assert!(SelectiveScanGeometry::new(source, Some(&[2, 3, 7, 5]), 2, 0.01).is_err());
        assert!(SelectiveScanGeometry::new(source, None, 0, 0.01).is_err());
        assert!(SelectiveScanGeometry::new(source, None, 2, f32::NAN).is_err());
        let mut incompatible = source;
        incompatible[3] = &[2, 1, 3];
        assert!(SelectiveScanGeometry::new(incompatible, None, 2, 0.01).is_err());
    }
}

mod absolute_mask;
pub use absolute_mask::{AbsoluteAttentionMaskError, AbsoluteAttentionMaskGeometry};
