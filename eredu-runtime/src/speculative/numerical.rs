//! Fixed speculative probability programs, separate from arbitrary sampler policy.

/// One deterministic numerical program applied to actual processed logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeNumericalKind {
    /// Exact eager U32 token input, with no model or sampler callback.
    TokenIds { length: u32 },
    /// Exact architecture-declared placeholder value and admitted extent.
    RepeatedToken { token: u32, length: u32 },
    /// Exact axis-one concatenation of the ordered completed token segments.
    TokenConcatenate { parts: u32, positions: u32 },
    /// Exact sequence-preserving static view of a completed capture tensor.
    TensorRange { start: u32, end: u32 },
    /// Exact non-batch-axis static view of a completed capture tensor. The
    /// axis is part of the paid numerical program, not inferred from a family.
    TensorAxisRange { axis: u8, start: u32, end: u32 },
    /// Exact axis-one concatenation of two completed capture sequences.
    TensorConcatenate { right_positions: u32 },
    /// Exact sequence-preserving static view of completed U32 token input.
    TokenRange { start: u32, end: u32 },
    /// Actual explicit scalar seed; creates exactly two U32 key words.
    CreateKey { seed: u64 },
    /// Existing state split: retained row zero and returned row one.
    NextKey,
    /// Sequential split followed by a completed F32 uniform acceptance draw.
    UniformUnitInterval,
    /// Exact existing position worker: split position+1 keys, select position.
    KeyAt { position: u32 },
    /// Actual known-policy categorical worker with an independently owned key.
    Categorical(crate::generation::SpeculativeCategoricalProgram),
    /// Exact known-policy greedy argmax, followed by completed U32 observation.
    Greedy(crate::generation::SpeculativeGreedyProgram),
    /// Actual frozen known sampler policy; arbitrary callbacks have no variant.
    ProcessLogits(crate::generation::SpeculativeLogitProgram),
    /// One lazily requested sequence row from an already completed logit source.
    LogitsRow { row: u32 },
    /// F32 conversion followed by precise last-axis softmax.
    Normalize,
    /// Select coordinate `[0, ..., token]` from completed probabilities.
    ProbabilityAt { token: u32 },
    /// Positive difference of normalized distributions, its total mass and,
    /// only when that mass exceeds F32 epsilon, logarithmic corrected logits.
    Correction,
}
/// Validated fixed shape and operation. No account or native authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeNumericalProgram {
    kind: SpeculativeNumericalKind,
    shape: [i32; 4],
    rank: u8,
}
/// Allocation-free program validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SpeculativeNumericalError {
    /// Native PRNG keys have exactly two U32 words; positions fit split's i32 count.
    #[error("speculative key geometry or position is invalid")]
    Key,
    /// One scalar choice requires exactly one source score row.
    #[error("speculative greedy choice requires exactly one score row")]
    GreedyRows,
    /// Fixed known-policy source geometry refusal.
    #[error(transparent)]
    Policy(#[from] crate::generation::PreparedLogitPolicyError),
    /// The selected source has invalid distribution or captured-sequence geometry.
    #[error("speculative source has invalid geometry for the selected numerical operation")]
    Shape,
    /// The source is not one batch of sequence logits or the row is out of bounds.
    #[error("speculative readout row exceeds the actual sequence")]
    Row,
    /// A selected token exceeds the actual vocabulary axis.
    #[error("speculative probability token exceeds the actual vocabulary")]
    Token,
    /// Correction sources differ from the program's actual shape.
    #[error("speculative numerical sources differ from the selected program")]
    Source,
}
impl SpeculativeNumericalProgram {
    /// Selects this exact source shape without allocating a shape vector.
    pub fn new(
        kind: SpeculativeNumericalKind,
        shape: &[i32],
    ) -> Result<Self, SpeculativeNumericalError> {
        if let SpeculativeNumericalKind::TokenIds { length } | SpeculativeNumericalKind::RepeatedToken { length, .. } = kind {
            let width = i32::try_from(length).map_err(|_| SpeculativeNumericalError::Shape)?;
            if width == 0 || shape != [1, width] {
                return Err(SpeculativeNumericalError::Shape);
            }
            return Ok(Self { kind, shape: [1, width, 1, 1], rank: 2 });
        }
        if let SpeculativeNumericalKind::TokenConcatenate { parts, positions } = kind {
            let width = i32::try_from(positions).map_err(|_| SpeculativeNumericalError::Shape)?;
            if parts == 0 || width == 0 || shape != [1, width] { return Err(SpeculativeNumericalError::Shape); }
            return Ok(Self { kind, shape: [1, width, 1, 1], rank: 2 });
        }
        if matches!(kind, SpeculativeNumericalKind::CreateKey { .. } | SpeculativeNumericalKind::NextKey | SpeculativeNumericalKind::UniformUnitInterval | SpeculativeNumericalKind::KeyAt { .. }) {
            if shape != [2] || matches!(kind, SpeculativeNumericalKind::KeyAt { position } if position >= i32::MAX as u32) {
                return Err(SpeculativeNumericalError::Key);
            }
            return Ok(Self { kind, shape: [2,1,1,1], rank: 1 });
        }
        let capture = matches!(kind, SpeculativeNumericalKind::TensorRange { .. } | SpeculativeNumericalKind::TensorAxisRange { .. } | SpeculativeNumericalKind::TensorConcatenate { .. });
        if !(2..=if capture { 4 } else { 3 }).contains(&shape.len())
            || shape.iter().any(|&d| d <= 0)
            || shape
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
                .is_none()
        {
            return Err(SpeculativeNumericalError::Shape);
        }
        if let SpeculativeNumericalKind::TensorRange { start, end }
            | SpeculativeNumericalKind::TokenRange { start, end } = kind {
            let valid_rank = if matches!(kind, SpeculativeNumericalKind::TokenRange { .. }) { shape.len() == 2 } else { (3..=4).contains(&shape.len()) };
            if !valid_rank || shape[0] != 1 || start > end || u64::from(end) > shape[1] as u64 {
                return Err(SpeculativeNumericalError::Row);
            }
        }
        if let SpeculativeNumericalKind::TensorAxisRange { axis, start, end } = kind {
            let axis = usize::from(axis);
            if shape[0] != 1 || axis == 0 || axis >= shape.len() || start > end
                || u64::from(end) > shape[axis] as u64 {
                return Err(SpeculativeNumericalError::Row);
            }
        }
        if let SpeculativeNumericalKind::TensorConcatenate { right_positions } = kind {
            if !(3..=4).contains(&shape.len()) || shape[0] != 1 || right_positions == 0
                || i32::try_from(right_positions).ok().and_then(|right| shape[1].checked_add(right)).is_none() {
                return Err(SpeculativeNumericalError::Shape);
            }
        }
        if let SpeculativeNumericalKind::LogitsRow { row } = kind {
            if shape.len() != 3 || shape[0] != 1 || u64::from(row) >= shape[1] as u64 {
                return Err(SpeculativeNumericalError::Row);
            }
        }
        if let SpeculativeNumericalKind::ProbabilityAt { token } = kind {
            if u64::from(token) >= *shape.last().unwrap() as u64 {
                return Err(SpeculativeNumericalError::Token);
            }
        }
        if matches!(kind, SpeculativeNumericalKind::Greedy(_) | SpeculativeNumericalKind::Categorical(_))
            && shape[..shape.len()-1].iter().any(|&d| d != 1) {
            return Err(SpeculativeNumericalError::GreedyRows);
        }
        if let SpeculativeNumericalKind::ProcessLogits(policy) = kind {
            policy.validate_layout(shape)?;
        }
        let mut extents = [1; 4];
        extents[..shape.len()].copy_from_slice(shape);
        Ok(Self {
            kind,
            shape: extents,
            rank: shape.len() as u8,
        })
    }
    /// Actual shape used by the producer.
    pub fn shape(&self) -> &[i32] {
        &self.shape[..usize::from(self.rank)]
    }
    /// Fixed selected program.
    pub fn kind(self) -> SpeculativeNumericalKind {
        self.kind
    }
    /// Number of actual source distributions, without callback inference.
    pub fn source_count(self) -> usize {
        match self.kind {
            SpeculativeNumericalKind::CreateKey { .. } | SpeculativeNumericalKind::TokenIds { .. } | SpeculativeNumericalKind::RepeatedToken { .. } => 0,
            SpeculativeNumericalKind::Correction | SpeculativeNumericalKind::Categorical(_)
                | SpeculativeNumericalKind::TensorConcatenate { .. } => 2,
            SpeculativeNumericalKind::TokenConcatenate { parts, .. } => parts as usize,
            _ => 1,
        }
    }
    /// Exact second-source geometry for the selected shared numerical program.
    pub fn validate_source(self, shape: &[i32]) -> Result<(), SpeculativeNumericalError> {
        let matches = match self.kind {
            SpeculativeNumericalKind::TensorConcatenate { right_positions } =>
                shape.len() == usize::from(self.rank) && shape[0] == 1
                    && shape[1] == right_positions as i32 && shape[2..] == self.shape()[2..],
            SpeculativeNumericalKind::TokenConcatenate { positions, .. } =>
                matches!(shape, [1, width] if *width > 0 && *width as u32 <= positions),
            _ => shape == self.shape(),
        };
        if matches {
            Ok(())
        } else {
            Err(SpeculativeNumericalError::Source)
        }
    }
}

/// Primitive realization of the fixed probability equations. Implementing
/// this trait does not qualify an arbitrary policy callback or issue a grant.
pub trait SpeculativeProbabilityBackend {
    /// Actual numerical value or metadata value.
    type Value;
    /// Selected native or metadata context.
    type Context: ?Sized;
    /// Realization failure.
    type Error;
    /// Convert to F32 while preserving shape.
    fn f32(value: &Self::Value, context: &Self::Context) -> Result<Self::Value, Self::Error>;
    /// Precise last-axis softmax.
    fn softmax(value: &Self::Value, context: &Self::Context) -> Result<Self::Value, Self::Error>;
    /// Select `[0, ..., token]` as a scalar, without host observation.
    fn select(
        value: &Self::Value,
        token: u32,
        context: &Self::Context,
    ) -> Result<Self::Value, Self::Error>;
    /// Elementwise difference.
    fn subtract(
        left: &Self::Value,
        right: &Self::Value,
        context: &Self::Context,
    ) -> Result<Self::Value, Self::Error>;
    /// Elementwise maximum with F32 zero.
    fn positive(value: &Self::Value, context: &Self::Context) -> Result<Self::Value, Self::Error>;
    /// Sum every axis into one scalar; completion/observation is separate.
    fn total(value: &Self::Value, context: &Self::Context) -> Result<Self::Value, Self::Error>;
    /// Elementwise natural logarithm.
    fn logarithm(value: &Self::Value, context: &Self::Context) -> Result<Self::Value, Self::Error>;
}
/// Same normalization for ordinary, admitted and metadata execution.
pub fn normalize<B: SpeculativeProbabilityBackend>(
    value: &B::Value,
    context: &B::Context,
) -> Result<B::Value, B::Error> {
    B::softmax(&B::f32(value, context)?, context)
}
/// First correction cut. The mass is observed only after real completion.
pub fn correction<B: SpeculativeProbabilityBackend>(
    left: &B::Value,
    right: &B::Value,
    context: &B::Context,
) -> Result<(B::Value, B::Value), B::Error> {
    let left = normalize::<B>(left, context)?;
    let right = normalize::<B>(right, context)?;
    let difference = B::positive(&B::subtract(&left, &right, context)?, context)?;
    let mass = B::total(&difference, context)?;
    Ok((difference, mass))
}
/// Existing shared mass decision; NaN preserves the ordinary branch behavior.
pub fn correction_has_mass(mass: f32) -> bool {
    !(mass <= f32::EPSILON)
}
/// Second correction cut, constructed only after the positive-mass decision.
pub fn correction_logits<B: SpeculativeProbabilityBackend>(
    difference: &B::Value,
    context: &B::Context,
) -> Result<B::Value, B::Error> {
    B::logarithm(difference, context)
}

#[cfg(test)]
mod capture_geometry_tests {
    use super::*;
    #[test]
    fn static_capture_range_keeps_the_selected_non_batch_axis() {
        let shape=[1,2,7,4];
        let kind=SpeculativeNumericalKind::TensorAxisRange { axis:2,start:1,end:6 };
        let program=SpeculativeNumericalProgram::new(kind,&shape).unwrap();
        assert_eq!(program.kind(),kind);
        assert_eq!(program.shape(),shape);
        assert_eq!(program.source_count(),1);
        program.validate_source(&shape).unwrap();
        assert!(program.validate_source(&[1,7,2,4]).is_err());
        for (axis,start,end) in [(0,0,1),(4,0,1),(1,1,6),(2,2,8),(2,5,4)] {
            assert_eq!(SpeculativeNumericalProgram::new(
                SpeculativeNumericalKind::TensorAxisRange{axis,start,end},&shape),
                Err(SpeculativeNumericalError::Row));
        }
        assert!(SpeculativeNumericalProgram::new(kind,&[2,2,7,4]).is_err());
        assert!(SpeculativeNumericalProgram::new(SpeculativeNumericalKind::Normalize,&shape).is_err());
    }
    #[test]
    fn captured_stream_axis_survives_ranges_and_concatenation_without_relaxing_logits() {
        let shape = [1, 5, 3, 8];
        let range = SpeculativeNumericalProgram::new(
            SpeculativeNumericalKind::TensorRange { start: 1, end: 4 }, &shape).unwrap();
        assert_eq!(range.shape(), shape);
        range.validate_source(&shape).unwrap();
        let join = SpeculativeNumericalProgram::new(
            SpeculativeNumericalKind::TensorConcatenate { right_positions: 2 }, &shape).unwrap();
        join.validate_source(&[1, 2, 3, 8]).unwrap();
        for invalid in [&[1, 2, 8][..], &[1, 2, 4, 8], &[1, 2, 3, 7]] {
            assert_eq!(join.validate_source(invalid), Err(SpeculativeNumericalError::Source));
        }
        assert_eq!(SpeculativeNumericalProgram::new(SpeculativeNumericalKind::Normalize, &shape).unwrap_err(), SpeculativeNumericalError::Shape);
        assert!(SpeculativeNumericalProgram::new(SpeculativeNumericalKind::TensorRange { start: 1, end: 6 }, &shape).is_err());
    }
}
