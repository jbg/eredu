//! Finite occurrences for the existing embedded-prediction executor.
//! These coordinates are descriptive: source binding, state copies, native
//! construction, completion and memory authority remain separately required.
use crate::{
    prefill::PrefillControlPlan, SelectedSpeculativeRealization, SpeculativeStrategyClass,
};
use eredu_core::{
    generation::{SpeculativeConfig, SpeculativeRequestStatus, SpeculativeSchedulerOptions},
    speculative::{
        PredictionPrefillAlignment, SpeculativeActivationPhase as Phase, SpeculativePrefillSpan,
        SpeculativeRequestGeometry,
    },
    GenerationError, OutputDemand,
};
use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_SCHEDULE: AtomicU64 = AtomicU64::new(1);

/// Architecture-declared iteration shape, independent of a requested cap.
/// This is not a physical module census: one depth may contain a group, and
/// shared modules may sit outside it. Source banks use actual module visitors.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EmbeddedPredictionShape {
    /// One invocation per proposal depth, including that depth's actual modules.
    Sequential { depth: NonZeroUsize },
    /// One call traverses all depths with an anchor expanded to the requested width.
    Fused {
        depth: NonZeroUsize,
        maximum_proposals: NonZeroUsize,
    },
}
impl EmbeddedPredictionShape {
    /// Architecture iteration count, including unrequested deeper iterations.
    /// Prefill/replay visit all depths; physical membership remains the actual
    /// architecture invocation and its module/retained-resource visitors.
    pub const fn depth(self) -> NonZeroUsize {
        match self {
            Self::Sequential { depth } | Self::Fused { depth, .. } => depth,
        }
    }
    fn admits(self, selected: &SelectedSpeculativeRealization) -> bool {
        let strategy = selected.requirements().strategy();
        match self {
            Self::Sequential { depth } => {
                strategy.class() == SpeculativeStrategyClass::EmbeddedSequential
                    && strategy.proposal_capacity() <= depth
            }
            Self::Fused {
                maximum_proposals, ..
            } => {
                strategy.class() == SpeculativeStrategyClass::EmbeddedFused
                    && strategy.proposal_capacity() <= maximum_proposals
            }
        }
    }
}

/// Finite phase populations. Proposal depth remains part of each invocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[repr(usize)]
pub enum EmbeddedOccurrenceKind {
    TargetPrefill,
    PredictionPrefill,
    SequentialProposal,
    FusedProposal,
    Verification,
    TargetReplay,
    PredictionReplay,
}
impl EmbeddedOccurrenceKind {
    /// Stable finite enumeration, independent of family names and file layout.
    pub const ALL: [Self; 7] = [
        Self::TargetPrefill,
        Self::PredictionPrefill,
        Self::SequentialProposal,
        Self::FusedProposal,
        Self::Verification,
        Self::TargetReplay,
        Self::PredictionReplay,
    ];
    fn of(phase: Phase) -> Self {
        match phase {
            Phase::TargetPrefill => Self::TargetPrefill,
            Phase::PredictionPrefill => Self::PredictionPrefill,
            Phase::Proposal { .. } => Self::SequentialProposal,
            Phase::FusedProposal => Self::FusedProposal,
            Phase::Verification => Self::Verification,
            Phase::TargetReplay => Self::TargetReplay,
            Phase::PredictionReplay => Self::PredictionReplay,
        }
    }
}
/// Exact logical call at an existing executor boundary. Prediction cache
/// frontiers are architecture-authenticated separately from target coordinates.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct EmbeddedInvocation {
    phase: Phase,
    target_frontier: u64,
    positions: usize,
    span: Option<SpeculativePrefillSpan>,
    target_output: Option<OutputDemand>,
}
impl EmbeddedInvocation {
    pub const fn phase(self) -> Phase {
        self.phase
    }
    pub const fn target_frontier(self) -> u64 {
        self.target_frontier
    }
    /// Physical sequence width. A fused call expands one anchor to this width.
    pub const fn positions(self) -> usize {
        self.positions
    }
    pub const fn prefill_span(self) -> Option<SpeculativePrefillSpan> {
        self.span
    }
    /// Target vocabulary demand; prediction readout follows its actual equation.
    pub const fn target_output(self) -> Option<OutputDemand> {
        self.target_output
    }
    /// Supplied token rows, distinct from a fused call's expanded sequence.
    pub const fn source_positions(self) -> usize {
        if matches!(self.phase, Phase::FusedProposal) {
            1
        } else {
            self.positions
        }
    }
}
#[derive(Debug, thiserror::Error)]
pub enum EmbeddedOccurrenceError {
    #[error(transparent)]
    Generation(#[from] GenerationError),
    #[error("embedded speculative source selection differs")]
    Selection,
    #[error("embedded speculative occurrence geometry differs")]
    Geometry,
    #[error("embedded speculative occurrence geometry overflow")]
    Overflow,
    #[error("embedded speculative context capacity exceeded")]
    Context,
    #[error("embedded speculative occurrence capacity exhausted")]
    Exhausted,
}
/// Identity of a geometry plan, never native authorization or a source epoch.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct EmbeddedScheduleIdentity(u64);
/// Cold plan over actual shared prefill chunks and shared proposal count policy.
#[derive(Debug)]
pub struct EmbeddedSchedulePlan<'a> {
    selected: &'a SelectedSpeculativeRealization,
    identity: EmbeddedScheduleIdentity,
    shape: EmbeddedPredictionShape,
    alignment: PredictionPrefillAlignment,
    seed_base: u64,
    prefill: PrefillControlPlan,
    geometry: SpeculativeRequestGeometry,
    proposal_blocks: usize,
    first_frontier: u64,
    closing: u64,
    limits: [usize; 7],
}
impl<'a> EmbeddedSchedulePlan<'a> {
    /// Constructs finite alternatives without allocating, opening a source, or
    /// running a scheduler. Native binding must authenticate these source facts.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        selected: &'a SelectedSpeculativeRealization,
        shape: EmbeddedPredictionShape,
        alignment: PredictionPrefillAlignment,
        seed_base: u64,
        prefill: PrefillControlPlan,
        context: u64,
        config: &SpeculativeConfig,
        options: SpeculativeSchedulerOptions,
    ) -> Result<Self, EmbeddedOccurrenceError> {
        config.validate()?;
        let options = options.validate()?;
        if !shape.admits(selected) {
            return Err(EmbeddedOccurrenceError::Selection);
        }
        let geometry = SpeculativeRequestGeometry::new(
            config,
            selected.requirements().strategy().proposal_capacity().get(),
        );
        let input = prefill.geometry();
        if input.batch_size != 1
            || input.max_output_tokens
                != u64::try_from(geometry.output_positions())
                    .map_err(|_| EmbeddedOccurrenceError::Overflow)?
        {
            return Err(EmbeddedOccurrenceError::Geometry);
        }
        // Every candidate becomes an actual host/native invocation with a
        // usize sequence width. Check the largest chunk once so a valid plan
        // cannot later fail prefill_invocations and panic during enumeration.
        usize::try_from(input.prefill_chunk_positions)
            .map_err(|_| EmbeddedOccurrenceError::Overflow)?;
        let first_frontier = input
            .cached_positions
            .checked_add(input.input_positions)
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        let closing = first_frontier
            .checked_add(input.max_output_tokens)
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        if closing > context {
            return Err(EmbeddedOccurrenceError::Context);
        }
        seed_base
            .checked_add(alignment.sequence_len(input.input_positions))
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        let proposal_blocks = 1usize
            .checked_add(options.lookahead_blocks)
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        let rounds = geometry.verification_attempts();
        let maximum = geometry.proposal_count(1);
        let blocks = rounds
            .checked_mul(proposal_blocks)
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        let proposals = blocks
            .checked_mul(maximum)
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        let target_spans =
            usize::try_from(prefill.span_count()).map_err(|_| EmbeddedOccurrenceError::Overflow)?;
        // Only the first shifted singleton target span has no prediction call.
        let empty_seed = alignment == PredictionPrefillAlignment::NextToken
            && input.prefill_chunk_positions == 1;
        let prediction_spans = target_spans - usize::from(empty_seed);
        let (sequential, fused) = match shape {
            EmbeddedPredictionShape::Sequential { .. } => (proposals, 0),
            EmbeddedPredictionShape::Fused { .. } => (0, if maximum == 0 { 0 } else { blocks }),
        };
        let identity = NEXT_SCHEDULE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map(EmbeddedScheduleIdentity)
            .map_err(|_| EmbeddedOccurrenceError::Overflow)?;
        let limits = [
            target_spans,
            prediction_spans,
            sequential,
            fused,
            rounds,
            rounds,
            rounds,
        ];
        limits
            .iter()
            .try_fold(0usize, |n, v| n.checked_add(*v))
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        Ok(Self {
            selected,
            identity,
            shape,
            alignment,
            seed_base,
            prefill,
            geometry,
            proposal_blocks,
            first_frontier,
            closing,
            limits,
        })
    }
    pub const fn identity(&self) -> EmbeddedScheduleIdentity {
        self.identity
    }
    pub const fn selected(&self) -> &SelectedSpeculativeRealization {
        self.selected
    }
    pub const fn shape(&self) -> EmbeddedPredictionShape {
        self.shape
    }
    pub const fn geometry(&self) -> SpeculativeRequestGeometry {
        self.geometry
    }
    pub const fn attempts(&self, kind: EmbeddedOccurrenceKind) -> usize {
        self.limits[kind as usize]
    }
    /// Exact target and optional prediction invocation of one shared chunk.
    pub fn prefill_invocations(
        &self,
        index: u64,
    ) -> Option<(EmbeddedInvocation, Option<EmbeddedInvocation>)> {
        let chunk = self.prefill.chunk(index)?;
        let span = SpeculativePrefillSpan {
            prompt_tokens: self.prefill.geometry().input_positions,
            input_start: chunk.input.start,
            input_end: chunk.input.end,
            position: chunk.position,
            hidden_start: chunk.input.start,
            token_start: chunk.input.start,
            sequence: chunk.input.end - chunk.input.start,
            seed_start: chunk.position,
        };
        let target = EmbeddedInvocation {
            phase: Phase::TargetPrefill,
            target_frontier: chunk.position,
            positions: usize::try_from(span.sequence).ok()?,
            span: Some(span),
            target_output: Some(chunk.output),
        };
        let prediction = self
            .alignment
            .seed(span, self.seed_base)?
            .invocation()
            .map(|seed| EmbeddedInvocation {
                phase: Phase::PredictionPrefill,
                target_frontier: chunk.position,
                positions: seed.sequence as usize,
                span: Some(seed),
                target_output: None,
            });
        Some((target, prediction))
    }
    /// Normalizes the existing scheduler's prefix coordinate and the shared
    /// driver's exact span. The generated prefix includes the unevaluated anchor:
    /// one token precedes every proposal/verification/replay invocation.
    /// These coordinates are descriptive and grant no native source authority.
    pub fn scheduler_invocation(
        &self,
        phase: Phase,
        positions: usize,
        span: Option<SpeculativePrefillSpan>,
        origin: eredu_core::speculative::SpeculativeActivationOrigin,
    ) -> Result<EmbeddedInvocation, EmbeddedOccurrenceError> {
        if matches!(phase, Phase::TargetPrefill | Phase::PredictionPrefill) {
            if origin.prediction != 0 || origin.committed_tokens != 0 || origin.optimistic {
                return Err(EmbeddedOccurrenceError::Geometry);
            }
            let span = span.ok_or(EmbeddedOccurrenceError::Geometry)?;
            let width = self.prefill.geometry().prefill_chunk_positions;
            let index = span.input_start.checked_div(width)
                .ok_or(EmbeddedOccurrenceError::Geometry)?;
            let (target, prediction) = self.prefill_invocations(index)
                .ok_or(EmbeddedOccurrenceError::Geometry)?;
            let invocation = if phase == Phase::TargetPrefill { Some(target) } else { prediction }
                .ok_or(EmbeddedOccurrenceError::Geometry)?;
            if invocation.prefill_span() != Some(span) || invocation.positions() != positions {
                return Err(EmbeddedOccurrenceError::Geometry);
            }
            return Ok(invocation);
        }
        if span.is_some() || origin.prediction < origin.committed_tokens {
            return Err(EmbeddedOccurrenceError::Geometry);
        }
        let prefix = origin.prediction.checked_sub(1)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(EmbeddedOccurrenceError::Geometry)?;
        let frontier = self.first_frontier.checked_add(prefix)
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        self.decode_invocation(phase, frontier, positions)
    }

    /// Validated non-prefill call. This uses canonical target coordinates, never
    /// an invented independent-draft frontier for prediction-local state.
    pub fn decode_invocation(
        &self,
        phase: Phase,
        target_frontier: u64,
        positions: usize,
    ) -> Result<EmbeddedInvocation, EmbeddedOccurrenceError> {
        let maximum = self.geometry.proposal_count(1);
        let valid = match phase {
            Phase::Proposal { depth } => {
                matches!(self.shape, EmbeddedPredictionShape::Sequential { .. })
                    && positions == 1
                    && depth < maximum
            }
            Phase::FusedProposal => {
                matches!(self.shape, EmbeddedPredictionShape::Fused { .. })
                    && positions != 0
                    && positions <= maximum
            }
            Phase::Verification => positions >= 2 && positions <= maximum.saturating_add(1),
            Phase::TargetReplay | Phase::PredictionReplay => positions != 0 && positions <= maximum,
            _ => false,
        };
        let closing = u64::try_from(positions)
            .ok()
            .and_then(|n| target_frontier.checked_add(n));
        if !valid
            || self.limits[EmbeddedOccurrenceKind::of(phase) as usize] == 0
            || target_frontier < self.first_frontier
            || !closing.is_some_and(|n| n <= self.closing)
        {
            return Err(EmbeddedOccurrenceError::Geometry);
        }
        Ok(EmbeddedInvocation {
            phase,
            target_frontier,
            positions,
            span: None,
            target_output: matches!(phase, Phase::Verification | Phase::TargetReplay)
                .then_some(OutputDemand::Sequence),
        })
    }
    fn contains(&self, invocation: EmbeddedInvocation) -> bool {
        if let Some(span) = invocation.span {
            let size = self.prefill.geometry().prefill_chunk_positions;
            if span.input_start % size != 0 {
                return false;
            }
            return self
                .prefill_invocations(span.input_start / size)
                .is_some_and(|(target, prediction)| {
                    invocation == target || prediction == Some(invocation)
                });
        }
        self.decode_invocation(
            invocation.phase,
            invocation.target_frontier,
            invocation.positions,
        )
        .is_ok_and(|actual| actual == invocation)
    }
    /// Every possible geometry, not an assertion that cost grows with width.
    /// Quotes reduce actual source-dependent costs by maximum before attempts.
    pub fn visit_candidates<E>(
        &self,
        kind: EmbeddedOccurrenceKind,
        mut visit: impl FnMut(EmbeddedInvocation) -> Result<(), E>,
    ) -> Result<(), E> {
        if self.attempts(kind) == 0 {
            return Ok(());
        }
        if matches!(
            kind,
            EmbeddedOccurrenceKind::TargetPrefill | EmbeddedOccurrenceKind::PredictionPrefill
        ) {
            for index in 0..self.prefill.span_count() {
                let (target, prediction) = self
                    .prefill_invocations(index)
                    .expect("validated finite prefill geometry");
                if kind == EmbeddedOccurrenceKind::TargetPrefill {
                    visit(target)?;
                } else if let Some(prediction) = prediction {
                    visit(prediction)?;
                }
            }
            return Ok(());
        }
        let maximum = self.geometry.proposal_count(1);
        for frontier in self.first_frontier..self.closing {
            let (widths, depths) = match kind {
                EmbeddedOccurrenceKind::SequentialProposal => (1..=1, 0..maximum),
                EmbeddedOccurrenceKind::FusedProposal => (1..=maximum, 0..1),
                EmbeddedOccurrenceKind::Verification => (2..=maximum.saturating_add(1), 0..1),
                _ => (1..=maximum, 0..1),
            };
            for positions in widths {
                for depth in depths.clone() {
                    let phase = match kind {
                        EmbeddedOccurrenceKind::SequentialProposal => Phase::Proposal { depth },
                        EmbeddedOccurrenceKind::FusedProposal => Phase::FusedProposal,
                        EmbeddedOccurrenceKind::Verification => Phase::Verification,
                        EmbeddedOccurrenceKind::TargetReplay => Phase::TargetReplay,
                        EmbeddedOccurrenceKind::PredictionReplay => Phase::PredictionReplay,
                        _ => unreachable!(),
                    };
                    if let Ok(invocation) = self.decode_invocation(phase, frontier, positions) {
                        visit(invocation)?;
                    }
                }
            }
        }
        Ok(())
    }
    /// Move into a request owner kept outside all model rollback/snapshot state.
    pub fn into_cursor(self) -> EmbeddedOccurrenceCursor<'a> {
        EmbeddedOccurrenceCursor {
            plan: self,
            spent: [0; 7],
            attempted: 0,
        }
    }
}
/// Monotonic attempted work, including errors and discarded optimistic branches.
#[derive(Debug)]
pub struct EmbeddedOccurrenceCursor<'a> {
    plan: EmbeddedSchedulePlan<'a>,
    spent: [usize; 7],
    attempted: usize,
}
#[derive(Debug)]
pub struct EmbeddedOccurrenceClaim<'a> {
    selected: &'a SelectedSpeculativeRealization,
    identity: EmbeddedScheduleIdentity,
    invocation: EmbeddedInvocation,
    ordinal: usize,
    phase_ordinal: usize,
}
impl EmbeddedOccurrenceClaim<'_> {
    pub const fn selected(&self) -> &SelectedSpeculativeRealization {
        self.selected
    }
    pub const fn identity(&self) -> EmbeddedScheduleIdentity {
        self.identity
    }
    pub const fn invocation(&self) -> EmbeddedInvocation {
        self.invocation
    }
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }
    pub const fn phase_ordinal(&self) -> usize {
        self.phase_ordinal
    }
}
/// One fresh future's finite additional slots. This does not fund their storage.
#[derive(Debug)]
pub struct EmbeddedContinuation {
    identity: EmbeddedScheduleIdentity,
    previous: [usize; 7],
    next: [usize; 7],
    previous_slots: usize,
    next_slots: usize,
}
impl EmbeddedContinuation {
    pub const fn identity(&self) -> EmbeddedScheduleIdentity {
        self.identity
    }
    pub const fn previous_slots(&self) -> usize {
        self.previous_slots
    }
    pub const fn next_slots(&self) -> usize {
        self.next_slots
    }
    /// Concrete scalar/cursor control frames of planning this continuation.
    /// The actual replacement slot buffer is priced by its accounting owner.
    pub fn control_bytes(&self) -> Option<usize> {
        let parts = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, EmbeddedOccurrenceError>>(),
            std::mem::size_of::<std::cell::RefMut<'_, EmbeddedOccurrenceCursor<'_>>>(),
            std::mem::size_of::<(usize, SpeculativeRequestStatus)>(),
            std::mem::size_of::<([usize; 7], [usize; 7], usize, usize, usize)>(),
        ];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}
impl<'a> EmbeddedOccurrenceCursor<'a> {
    pub const fn plan(&self) -> &EmbeddedSchedulePlan<'a> {
        &self.plan
    }
    pub const fn attempted(&self) -> usize {
        self.attempted
    }
    /// Claim before entering the existing source-specific native producer.
    pub fn claim(
        &mut self,
        invocation: EmbeddedInvocation,
    ) -> Result<EmbeddedOccurrenceClaim<'a>, EmbeddedOccurrenceError> {
        if !self.plan.contains(invocation) {
            return Err(EmbeddedOccurrenceError::Geometry);
        }
        let index = EmbeddedOccurrenceKind::of(invocation.phase) as usize;
        let (ordinal, phase_ordinal) = super::attempts::consume(
            &mut self.spent,
            &mut self.attempted,
            index,
            self.plan.limits[index],
        )
        .map_err(|e| match e {
            super::attempts::AttemptError::Exhausted => EmbeddedOccurrenceError::Exhausted,
            super::attempts::AttemptError::Overflow => EmbeddedOccurrenceError::Overflow,
        })?;
        Ok(EmbeddedOccurrenceClaim {
            selected: self.plan.selected,
            identity: self.plan.identity,
            invocation,
            ordinal,
            phase_ordinal,
        })
    }
    /// A restored committed boundary adds freshly funded future slots; it never
    /// resets spent counters or repeats initial target/prediction prefill.
    pub fn continuation(
        &self,
        committed: usize,
        status: SpeculativeRequestStatus,
    ) -> Result<EmbeddedContinuation, EmbeddedOccurrenceError> {
        let output = self.plan.geometry.output_positions();
        if committed > output {
            return Err(EmbeddedOccurrenceError::Geometry);
        }
        let rounds = match status {
            SpeculativeRequestStatus::Completed => 0,
            SpeculativeRequestStatus::ReadyToDraft if committed != 0 && committed < output => {
                output - committed
            }
            _ => return Err(EmbeddedOccurrenceError::Geometry),
        };
        let maximum = self.plan.geometry.proposal_count(committed);
        let blocks = rounds
            .checked_mul(self.plan.proposal_blocks)
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        let proposals = blocks
            .checked_mul(maximum)
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        let (sequential, fused) = match self.plan.shape {
            EmbeddedPredictionShape::Sequential { .. } => (proposals, 0),
            EmbeddedPredictionShape::Fused { .. } => (0, blocks),
        };
        let mut next = self.plan.limits;
        for (n, added) in next
            .iter_mut()
            .zip([0, 0, sequential, fused, rounds, rounds, rounds])
        {
            *n = n
                .checked_add(added)
                .ok_or(EmbeddedOccurrenceError::Overflow)?;
        }
        next.iter()
            .try_fold(0usize, |n, v| n.checked_add(*v))
            .ok_or(EmbeddedOccurrenceError::Overflow)?;
        Ok(EmbeddedContinuation {
            identity: self.plan.identity,
            previous: self.plan.limits,
            next,
            previous_slots: self.plan.limits.iter().try_fold(0usize, |n, v| n.checked_add(*v)).ok_or(EmbeddedOccurrenceError::Overflow)?,
            next_slots: next.iter().try_fold(0usize, |n, v| n.checked_add(*v)).ok_or(EmbeddedOccurrenceError::Overflow)?,
        })
    }
    /// Install only after the enclosing source-specific adapter funds its actual
    /// new storage. Comparing the previous limits prevents stale/double install.
    pub fn install_continuation(
        &mut self,
        continuation: EmbeddedContinuation,
    ) -> Result<(), EmbeddedOccurrenceError> {
        if continuation.identity != self.plan.identity || continuation.previous != self.plan.limits
        {
            return Err(EmbeddedOccurrenceError::Geometry);
        }
        self.plan.limits = continuation.next;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

mod workspace;
pub use workspace::EmbeddedInvocationWorkspace;
