//! Checked role geometry for the existing driver. These scalar facts are not a
//! grant: only an accepted original control bank can issue a native role owner.
use crate::working_memory::WorkingMemoryError;
use eredu_core::{InferenceGeometry, OutputDemand};

/// Actual enclosing scopes used by one local shared-prefill span.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PrefillSpanControlPhase {
    /// Cancellation/retirement agreement before submission.
    PreBoundary,
    /// Source chunk preparation and the enclosing session operation.
    SpanOuter,
    /// The existing transaction scope, only for an actually retained admission.
    InputTransaction,
    /// Completion and publication agreement after the outer scope settles.
    SettlementAgreement,
    /// Post-completion cancellation/retirement agreement.
    PostBoundary,
}

/// A named role with exact prompt-relative and absolute span coordinates.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PrefillControlRole {
    /// The source factory's existing enclosing scope.
    SourcePreparation,
    /// One phase of a specific physical decoder invocation.
    Span {
        /// Scope's actual operation.
        phase: PrefillSpanControlPhase,
        /// Inclusive prompt-relative start.
        input_start: u64,
        /// Exclusive prompt-relative end.
        input_end: u64,
        /// Absolute first decoder position.
        position: u64,
        /// Readout demand actually selected for this span.
        output: OutputDemand,
    },
    /// Final score indexing after every physical span has settled.
    FinalIndex,
}

/// Cold, allocation-free representation of the finite local scope geometry.
/// Construction validates geometry before original comparison. It does not
/// certify source identity, native support, scope storage or a complete request.
/// Captured auxiliary and distributed profiles require their own actual facts.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PrefillControlPlan {
    geometry: InferenceGeometry,
    retained_input_transaction: bool,
    spans: u64,
    scopes: u64,
}
impl PrefillControlPlan {
    /// Derive the exact successful-run scope population. A caller must bind the
    /// selected retained-admission branch; presence of a capture ticket is not
    /// that branch. No original authority is created by these diagnostic facts.
    pub fn new(
        geometry: InferenceGeometry,
        retained_input_transaction: bool,
    ) -> Result<Self, WorkingMemoryError> {
        geometry
            .validate_fixed()
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        // Valid empty geometry is a terminal saved-state placement. It has no
        // source factory, decoder invocation or final score-indexing role.
        if geometry.input_positions == 0 {
            return Ok(Self {
                geometry,
                retained_input_transaction,
                spans: 0,
                scopes: 0,
            });
        }
        // Quotient/remainder avoids overflowing input + chunk - 1.
        let spans = geometry.input_positions / geometry.prefill_chunk_positions
            + u64::from(geometry.input_positions % geometry.prefill_chunk_positions != 0);
        let scopes = spans
            .checked_mul(if retained_input_transaction { 5 } else { 4 })
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_add(u64::from(geometry.output != OutputDemand::StateOnly)))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            geometry,
            retained_input_transaction,
            spans,
            scopes,
        })
    }
    /// The exact request geometry whose source/selection must also authenticate.
    pub const fn geometry(self) -> InferenceGeometry {
        self.geometry
    }
    /// Whether the selected session retains the admitted inner transaction.
    pub const fn retained_input_transaction(self) -> bool {
        self.retained_input_transaction
    }
    /// Number of physical spans in the shared driver.
    pub const fn span_count(self) -> u64 {
        self.spans
    }
    /// Total distinct successful-run scopes. Cancelled/failed runs consume only
    /// reached roles; their unconsumed owners remain held rather than refunded.
    pub const fn scope_count(self) -> u64 {
        self.scopes
    }
    /// Exact role at a fixed bank position. Out-of-range positions never wrap.
    pub fn role(self, index: u64) -> Option<PrefillControlRole> {
        if index >= self.scopes {
            return None;
        }
        if index == 0 {
            return Some(PrefillControlRole::SourcePreparation);
        }
        let phases = if self.retained_input_transaction {
            5
        } else {
            4
        };
        let position = index - 1;
        let span = position / phases;
        if span == self.spans {
            return Some(PrefillControlRole::FinalIndex);
        }
        let phase = match (position % phases, self.retained_input_transaction) {
            (0, _) => PrefillSpanControlPhase::PreBoundary,
            (1, _) => PrefillSpanControlPhase::SpanOuter,
            (2, true) => PrefillSpanControlPhase::InputTransaction,
            (2, false) | (3, true) => PrefillSpanControlPhase::SettlementAgreement,
            (3, false) | (4, true) => PrefillSpanControlPhase::PostBoundary,
            _ => unreachable!("fixed phase population"),
        };
        self.chunk(span)
            .map(|chunk| PrefillControlRole::for_chunk(phase, &chunk))
    }
    /// Exact physical invocation geometry used by the shared driver. Auxiliary
    /// occurrence plans borrow these coordinates rather than reproducing chunks.
    pub fn chunk(self, span: u64) -> Option<super::PrefillChunk> {
        if span >= self.spans {
            return None;
        }
        let input_start = span * self.geometry.prefill_chunk_positions;
        let input_end = input_start
            + self
                .geometry
                .prefill_chunk_positions
                .min(self.geometry.input_positions - input_start);
        Some(super::PrefillChunk {
            input: input_start..input_end,
            position: self.geometry.cached_positions + input_start,
            output: self
                .geometry
                .output
                .for_chunk(input_end == self.geometry.input_positions),
        })
    }
}

impl PrefillControlRole {
    /// Descriptor of an already selected physical span. The original bank must
    /// compare it with its retained plan before issuing its non-refillable role.
    pub fn for_chunk(phase: PrefillSpanControlPhase, chunk: &super::PrefillChunk) -> Self {
        Self::Span {
            phase,
            input_start: chunk.input.start,
            input_end: chunk.input.end,
            position: chunk.position,
            output: chunk.output,
        }
    }
}
impl PrefillControlPlan {
    /// Translate the actual driver boundary without consulting optional capture
    /// state. Invalid/unaligned coordinates do not produce a role descriptor.
    pub fn boundary_role(self, boundary: super::PrefillBoundary) -> Option<PrefillControlRole> {
        let phases = if self.retained_input_transaction {
            5
        } else {
            4
        };
        let ordinal = match boundary {
            super::PrefillBoundary::Before { input_position }
                if input_position < self.geometry.input_positions
                    && input_position % self.geometry.prefill_chunk_positions == 0 =>
            {
                1 + (input_position / self.geometry.prefill_chunk_positions) * phases
            }
            super::PrefillBoundary::After { input_position }
                if input_position != 0
                    && input_position <= self.geometry.input_positions
                    && (input_position == self.geometry.input_positions
                        || input_position % self.geometry.prefill_chunk_positions == 0) =>
            {
                ((input_position - 1) / self.geometry.prefill_chunk_positions + 1) * phases
            }
            _ => return None,
        };
        self.role(ordinal)
    }
}
