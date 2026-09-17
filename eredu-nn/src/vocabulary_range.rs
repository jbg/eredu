//! Fixed scalar checks shared with ordinary vocabulary ownership validation.
use crate::VocabularyParallelRange;

/// A vocabulary range is invalid; no diagnostic or source identity is allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VocabularyRangeError {
    /// Ownership is empty, reversed, or outside the declared global vocabulary.
    #[error("invalid vocabulary-parallel range {start}..{end} of {global_vocabulary}")]
    InvalidRange {
        /// Complete declared vocabulary.
        global_vocabulary: usize,
        /// Inclusive local start.
        start: usize,
        /// Exclusive local end.
        end: usize,
    },
    /// The operator's signed row count differs from the validated global count.
    #[error("vocabulary-parallel operator declares {rows} rows but ownership covers {global_vocabulary}")]
    GlobalRows {
        /// Actual operator declaration.
        rows: i32,
        /// Complete validated vocabulary.
        global_vocabulary: usize,
    },
    /// The requested rank is outside a nonempty balanced partition.
    #[error("invalid vocabulary partition rank {rank} of {partitions}")]
    PartitionRank {
        /// Requested rank.
        rank: usize,
        /// Declared partition count.
        partitions: usize,
    },
    /// The retained local range differs from the selected balanced partition.
    #[error("vocabulary-parallel range {start}..{end} differs from balanced rank {rank} ownership {expected_start}..{expected_end}")]
    BalancedRange {
        /// Actual local start.
        start: usize,
        /// Actual local end.
        end: usize,
        /// Requested rank.
        rank: usize,
        /// Balanced local start.
        expected_start: usize,
        /// Balanced local end.
        expected_end: usize,
    },
}

/// Validated scalar balanced-width plan. It owns no peer list, allocation or
/// execution authority; ordinary and finite destinations use the same widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BalancedVocabularyWidths {
    partitions: usize,
    base: usize,
    remainder: usize,
}

impl BalancedVocabularyWidths {
    /// Exact number of peer entries required by a destination.
    pub fn len(self) -> usize {
        self.partitions
    }

    /// Whether there are no peers. Validated plans are always nonempty.
    pub fn is_empty(self) -> bool {
        self.partitions == 0
    }

    /// Width at one peer index, without constructing other entries.
    pub fn width(self, peer: usize) -> Option<usize> {
        (peer < self.partitions).then(|| self.base + usize::from(peer < self.remainder))
    }

    /// Exact widths in rank order, borrowing no source allocation.
    pub fn widths(self) -> impl ExactSizeIterator<Item = usize> {
        (0..self.partitions).map(move |peer| self.base + usize::from(peer < self.remainder))
    }
}

impl VocabularyParallelRange {
    /// Validates the range, rank and exact balanced ownership, in that order,
    /// without allocating a destination or a diagnostic string.
    pub fn balanced_peer_widths_plan(
        &self,
        partitions: usize,
        rank: usize,
    ) -> Result<BalancedVocabularyWidths, VocabularyRangeError> {
        self.validate_fixed()?;
        if partitions == 0 || rank >= partitions {
            return Err(VocabularyRangeError::PartitionRank { rank, partitions });
        }
        let plan = BalancedVocabularyWidths {
            partitions,
            base: self.global_vocabulary / partitions,
            remainder: self.global_vocabulary % partitions,
        };
        // Since rank < partitions, each term and their sum is bounded by the
        // validated global vocabulary. This is the prefix sum of these widths.
        let expected_start = plan.base * rank + rank.min(plan.remainder);
        let expected_end = expected_start + plan.width(rank).expect("validated rank");
        if self.local != (expected_start..expected_end) {
            return Err(VocabularyRangeError::BalancedRange {
                start: self.local.start,
                end: self.local.end,
                rank,
                expected_start,
                expected_end,
            });
        }
        Ok(plan)
    }

    /// Validates nonempty in-bounds ownership without allocating a diagnostic.
    pub fn validate_fixed(&self) -> Result<(), VocabularyRangeError> {
        if self.global_vocabulary == 0
            || self.local.is_empty()
            || self.local.end > self.global_vocabulary
        {
            return Err(VocabularyRangeError::InvalidRange {
                global_vocabulary: self.global_vocabulary,
                start: self.local.start,
                end: self.local.end,
            });
        }
        Ok(())
    }

    /// Validates ownership first, then the operator's exact signed global rows.
    pub fn validate_global_rows_fixed(&self, rows: i32) -> Result<(), VocabularyRangeError> {
        self.validate_fixed()?;
        if usize::try_from(rows).ok() != Some(self.global_vocabulary) {
            return Err(VocabularyRangeError::GlobalRows {
                rows,
                global_vocabulary: self.global_vocabulary,
            });
        }
        Ok(())
    }
}

impl From<VocabularyRangeError> for crate::Error {
    fn from(cause: VocabularyRangeError) -> Self {
        // Keep the ordinary literal formatting and message-only adapter, including
        // its existing intermediate diagnostic String. Fixed callers avoid it.
        match cause {
            VocabularyRangeError::InvalidRange {
                global_vocabulary,
                start,
                end,
            } => {
                let local = start..end;
                crate::Error::backend(format!(
                    "invalid vocabulary-parallel range {:?} of {}",
                    local, global_vocabulary
                ))
            }
            VocabularyRangeError::GlobalRows {
                rows,
                global_vocabulary,
            } => crate::Error::backend(format!(
                "vocabulary-parallel operator declares {rows} rows but ownership covers {}",
                global_vocabulary
            )),
            VocabularyRangeError::PartitionRank { rank, partitions } => crate::Error::backend(
                format!("invalid vocabulary partition rank {rank} of {partitions}"),
            ),
            VocabularyRangeError::BalancedRange {
                start,
                end,
                rank,
                expected_start,
                expected_end,
            } => {
                let local = start..end;
                let expected = expected_start..expected_end;
                crate::Error::backend(format!(
                    "vocabulary-parallel range {local:?} differs from balanced rank {rank} ownership {expected:?}"
                ))
            }
        }
    }
}
