//! Architecture-declared hidden/token pairing for the shared prefill driver.
use super::{SpeculativeActivationPhase, SpeculativePrefillSpan};

/// Semantic alignment of the actual prediction extension. This is independent
/// of sequential/fused scheduling and grants no execution or memory authority.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PredictionPrefillAlignment {
    /// Each hidden row predicts the following input token.
    NextToken,
    /// Each accepted hidden row and input token seed the same position.
    Aligned,
}
impl PredictionPrefillAlignment {
    /// Physical prediction rows in a complete target prefix.
    pub const fn sequence_len(self, target_sequence: u64) -> u64 {
        match self {
            Self::NextToken => target_sequence.saturating_sub(1),
            Self::Aligned => target_sequence,
        }
    }
    /// Derives the prediction seed from an actual target span and the retained
    /// initial prediction frontier. A shifted singleton first span has no seed
    /// invocation, but still has a frontier and a target carry to complete.
    pub fn seed(self, target: SpeculativePrefillSpan, base: u64) -> Option<PredictionPrefillSeed> {
        if !target.validate(
            SpeculativeActivationPhase::TargetPrefill,
            usize::try_from(target.sequence).ok()?,
        ) {
            return None;
        }
        let shifted = self == Self::NextToken;
        let skip = u64::from(shifted && target.input_start == 0);
        let sequence = target.sequence.checked_sub(skip)?;
        let seed_start = base.checked_add(self.sequence_len(target.input_start))?;
        let after = seed_start.checked_add(sequence)?;
        Some(PredictionPrefillSeed {
            span: SpeculativePrefillSpan {
                hidden_start: if shifted {
                    target.input_start.saturating_sub(1)
                } else {
                    target.input_start
                },
                token_start: target.input_start.checked_add(skip)?,
                sequence,
                seed_start,
                ..target
            },
            after,
        })
    }
}

/// Checked seed coordinates shared by execution and occurrence accounting.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PredictionPrefillSeed {
    span: SpeculativePrefillSpan,
    after: u64,
}
impl PredictionPrefillSeed {
    /// Exact prediction cache frontier required before this span.
    pub const fn frontier_before(self) -> u64 {
        self.span.seed_start
    }
    /// Exact frontier after successful seed completion, even for an empty seed.
    pub const fn frontier_after(self) -> u64 {
        self.after
    }
    /// Number of actual hidden/token pairs consumed.
    pub const fn sequence(self) -> u64 {
        self.span.sequence
    }
    /// Number of leading input tokens omitted from the enclosing target span.
    pub const fn token_skip(self) -> u64 {
        self.span.token_start - self.span.input_start
    }
    /// Nonempty physical invocation. Empty first shifted spans produce none.
    pub fn invocation(self) -> Option<SpeculativePrefillSpan> {
        (self.span.sequence != 0).then_some(self.span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shifted_and_aligned_chunks_preserve_all_nonzero_pairs_and_actual_frontiers() {
        let hidden = [11, 23, 37, 41, 59, 67, 73];
        let tokens = [2, 3, 5, 7, 11, 13, 17];
        for alignment in [
            PredictionPrefillAlignment::NextToken,
            PredictionPrefillAlignment::Aligned,
        ] {
            let expected: Vec<_> = match alignment {
                PredictionPrefillAlignment::NextToken => hidden[..6]
                    .iter()
                    .copied()
                    .zip(tokens[1..].iter().copied())
                    .collect(),
                PredictionPrefillAlignment::Aligned => hidden.into_iter().zip(tokens).collect(),
            };
            for chunk in [1, 3, 7] {
                let mut actual = Vec::new();
                let mut frontier = 19;
                for start in (0..7).step_by(chunk) {
                    let end = (start + chunk).min(7);
                    let target = SpeculativePrefillSpan {
                        prompt_tokens: 7,
                        input_start: start as u64,
                        input_end: end as u64,
                        position: 101 + start as u64,
                        hidden_start: start as u64,
                        token_start: start as u64,
                        sequence: (end - start) as u64,
                        seed_start: 101 + start as u64,
                    };
                    let seed = alignment.seed(target, 19).unwrap();
                    assert_eq!(seed.frontier_before(), frontier);
                    if let Some(span) = seed.invocation() {
                        assert!(span.validate(
                            SpeculativeActivationPhase::PredictionPrefill,
                            span.sequence as usize
                        ));
                        for row in 0..span.sequence {
                            actual.push((
                                hidden[(span.hidden_start + row) as usize],
                                tokens[(span.token_start + row) as usize],
                            ));
                        }
                    }
                    frontier = seed.frontier_after();
                }
                assert_eq!(actual, expected);
                assert_eq!(frontier, 19 + expected.len() as u64);
            }
        }
    }
    #[test]
    fn seed_rejects_invalid_target_and_prediction_frontier_overflow() {
        let target = SpeculativePrefillSpan {
            prompt_tokens: 3,
            input_start: 0,
            input_end: 3,
            position: 0,
            hidden_start: 0,
            token_start: 0,
            sequence: 3,
            seed_start: 0,
        };
        assert!(PredictionPrefillAlignment::NextToken
            .seed(target, u64::MAX - 1)
            .is_none());
        assert!(PredictionPrefillAlignment::Aligned
            .seed(
                SpeculativePrefillSpan {
                    sequence: 2,
                    ..target
                },
                0
            )
            .is_none());
    }
}
