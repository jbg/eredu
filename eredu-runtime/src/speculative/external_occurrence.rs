//! Finite calls made by the existing external-assistant strategies.
//! A claim describes work; source binding and native admission remain separate.
use crate::{prefill::PrefillControlPlan, SelectedSpeculativeRealization, SpeculativeStrategyClass};
use eredu_core::{
    generation::{SpeculativeConfig, SpeculativeRequestStatus, SpeculativeSchedulerOptions},
    speculative::{SpeculativeActivationOrigin, SpeculativePrefillSpan, SpeculativeRequestGeometry},
    GenerationError, InferenceGeometry,
};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SCHEDULE: AtomicU64 = AtomicU64::new(1);

/// Iteration of the retained architecture's actual external strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalPredictionShape {
    /// Each requested proposal executes an assistant step and target embedding.
    Sequential,
    /// One block executes context preparation, embeddings, assistant and readout.
    Fused,
}

/// Existing equation boundaries, independent of checkpoint family names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum ExternalInvocationKind {
    /// One physical target prefill chunk.
    TargetPrefill,
    /// Verification or its exact retained-prefix replay through the same worker.
    TargetVerification,
    /// Target embedding used by an actual proposal or fused block.
    TargetTokenEmbeddings,
    /// Target readout of an actual fused assistant block.
    TargetProjectLogits,
    /// One sequential assistant step.
    AssistantStep,
    /// Assembly of captures at a completed target-state boundary.
    AssembleContext,
    /// Optional update of the pending fused context before one block.
    UpdateContext,
    /// One fused assistant block.
    FusedProposal,
}
impl ExternalInvocationKind {
    /// Fixed populations used by the actual shared strategies.
    pub const ALL: [Self; 8] = [Self::TargetPrefill, Self::TargetVerification,
        Self::TargetTokenEmbeddings, Self::TargetProjectLogits, Self::AssistantStep,
        Self::AssembleContext, Self::UpdateContext, Self::FusedProposal];
}

/// Actual equation geometry and scheduler coordinate; no native authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalInvocation {
    kind: ExternalInvocationKind,
    geometry: InferenceGeometry,
    span: Option<SpeculativePrefillSpan>,
    origin: SpeculativeActivationOrigin,
}
impl ExternalInvocation {
    /// Actual operation boundary.
    pub const fn kind(self) -> ExternalInvocationKind { self.kind }
    /// Full quoted equation geometry, including its real cache frontier.
    pub const fn geometry(self) -> InferenceGeometry { self.geometry }
    /// Physical prefill chunk, when this is a target prefill call.
    pub const fn prefill_span(self) -> Option<SpeculativePrefillSpan> { self.span }
    /// Exact origin supplied by the shared scheduler.
    pub const fn origin(self) -> SpeculativeActivationOrigin { self.origin }
}

/// Identity of one retained external schedule, not its memory account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalScheduleIdentity(u64);

/// Fixed descriptive refusal without allocating a diagnostic.
#[derive(Debug, thiserror::Error)]
pub enum ExternalOccurrenceError {
    /// Shared request configuration refused.
    #[error(transparent)]
    Generation(#[from] GenerationError),
    /// The retained selection is not an external strategy.
    #[error("external speculative source selection differs")]
    Selection,
    /// Actual operation geometry or scheduler coordinate differs.
    #[error("external speculative occurrence geometry differs")]
    Geometry,
    /// Checked extent or occurrence population overflowed.
    #[error("external speculative occurrence overflow")]
    Overflow,
    /// Actual maximum frontier exceeds the declared context.
    #[error("external speculative context capacity exceeded")]
    Context,
    /// All attempts for this operation have been consumed.
    #[error("external speculative occurrence capacity exhausted")]
    Exhausted,
}

/// Retains the actual selected contract and the shared driver's finite geometry.
#[derive(Debug)]
pub struct ExternalSchedulePlan<'a> {
    selected: &'a SelectedSpeculativeRealization,
    identity: ExternalScheduleIdentity,
    shape: ExternalPredictionShape,
    prefill: PrefillControlPlan,
    geometry: SpeculativeRequestGeometry,
    proposal_blocks: usize,
    first_frontier: u64,
    closing: u64,
    limits: [usize; 8],
}
impl<'a> ExternalSchedulePlan<'a> {
    /// Enumerates actual loop populations without constructing native resources.
    pub fn new(selected: &'a SelectedSpeculativeRealization, shape: ExternalPredictionShape,
        prefill: PrefillControlPlan, context: u64, config: &SpeculativeConfig,
        options: SpeculativeSchedulerOptions) -> Result<Self, ExternalOccurrenceError>
    {
        config.validate()?;
        let options = options.validate()?;
        if selected.requirements().strategy().class() != SpeculativeStrategyClass::External {
            return Err(ExternalOccurrenceError::Selection);
        }
        let geometry = SpeculativeRequestGeometry::new(config,
            selected.requirements().strategy().proposal_capacity().get());
        let input = prefill.geometry();
        if input.batch_size != 1 || input.input_positions == 0
            || input.max_output_tokens != u64::try_from(geometry.output_positions())
                .map_err(|_| ExternalOccurrenceError::Overflow)? {
            return Err(ExternalOccurrenceError::Geometry);
        }
        usize::try_from(input.prefill_chunk_positions).map_err(|_| ExternalOccurrenceError::Overflow)?;
        let first_frontier = input.cached_positions.checked_add(input.input_positions)
            .ok_or(ExternalOccurrenceError::Overflow)?;
        let closing = first_frontier.checked_add(input.max_output_tokens)
            .ok_or(ExternalOccurrenceError::Overflow)?;
        if closing > context { return Err(ExternalOccurrenceError::Context); }
        let proposal_blocks = 1usize.checked_add(options.lookahead_blocks)
            .ok_or(ExternalOccurrenceError::Overflow)?;
        let limits = populations(shape,
            usize::try_from(prefill.span_count()).map_err(|_| ExternalOccurrenceError::Overflow)?,
            geometry.verification_attempts(), geometry.proposal_count(1), proposal_blocks)?;
        let identity = NEXT_SCHEDULE.fetch_update(Ordering::Relaxed, Ordering::Relaxed,
            |n| n.checked_add(1)).map(ExternalScheduleIdentity)
            .map_err(|_| ExternalOccurrenceError::Overflow)?;
        Ok(Self { selected, identity, shape, prefill, geometry, proposal_blocks,
            first_frontier, closing, limits })
    }
    /// Retained exact selected realization.
    pub const fn selected(&self) -> &'a SelectedSpeculativeRealization { self.selected }
    /// Distinct issuance identity, even for equal schedules.
    pub const fn identity(&self) -> ExternalScheduleIdentity { self.identity }
    /// Actual declared strategy iteration.
    pub const fn shape(&self) -> ExternalPredictionShape { self.shape }
    /// Shared proposal/output caps.
    pub const fn geometry(&self) -> SpeculativeRequestGeometry { self.geometry }
    /// Finite number of attempted calls, including failures and discarded work.
    pub const fn attempts(&self, kind: ExternalInvocationKind) -> usize { self.limits[kind as usize] }
    /// Checked population of the request's fixed role destination.
    pub fn total_attempts(&self) -> Result<usize, ExternalOccurrenceError> { total(self.limits) }

    /// Validates the actual native quote geometry against the retained shared
    /// driver coordinate. Stateful frontiers must be exact; stateless target
    /// projections retain a zero cached frontier in their own equation reports.
    pub fn invocation(&self, kind: ExternalInvocationKind, geometry: InferenceGeometry,
        span: Option<SpeculativePrefillSpan>, origin: SpeculativeActivationOrigin)
        -> Result<ExternalInvocation, ExternalOccurrenceError>
    {
        let fail = || ExternalOccurrenceError::Geometry;
        if self.attempts(kind) == 0 || geometry.batch_size != self.prefill.geometry().batch_size
            || geometry.max_output_tokens != 0 || geometry.input_positions == 0
            || geometry.prefill_chunk_positions != geometry.input_positions
            || origin.prediction < origin.committed_tokens { return Err(fail()); }
        if kind == ExternalInvocationKind::TargetPrefill {
            if origin.prediction != 0 || origin.committed_tokens != 0 || origin.optimistic {
                return Err(fail());
            }
            let start = geometry.cached_positions.checked_sub(self.prefill.geometry().cached_positions)
                .ok_or_else(fail)?;
            let width = self.prefill.geometry().prefill_chunk_positions;
            if start % width != 0 { return Err(fail()); }
            let chunk = self.prefill.chunk(start / width).ok_or_else(fail)?;
            if geometry.cached_positions != chunk.position
                || geometry.input_positions != chunk.input.end - chunk.input.start {
                return Err(fail());
            }
            if let Some(span) = span {
                if span.prompt_tokens != self.prefill.geometry().input_positions
                    || span.input_start != chunk.input.start || span.input_end != chunk.input.end
                    || span.position != chunk.position || span.sequence != geometry.input_positions
                    || span.hidden_start != chunk.input.start || span.token_start != chunk.input.start
                    || span.seed_start != chunk.position { return Err(fail()); }
            } else if self.prefill.span_count() != 1 { return Err(fail()); }
        } else {
            if span.is_some() { return Err(fail()); }
            let maximum = u64::try_from(self.geometry.proposal_count(1))
                .map_err(|_| ExternalOccurrenceError::Overflow)?;
            let width = geometry.input_positions;
            // Context assembly may follow a prefill chunk before a generated
            // prefix exists. It is separately bounded by the physical chunks.
            if kind == ExternalInvocationKind::AssembleContext && origin.prediction == 0 {
                if self.shape != ExternalPredictionShape::Fused || origin.committed_tokens != 0
                    || origin.optimistic || geometry.cached_positions != 0
                    || width > self.prefill.geometry().prefill_chunk_positions { return Err(fail()); }
            } else {
                let prefix = origin.prediction.checked_sub(1).and_then(|v| u64::try_from(v).ok())
                    .ok_or_else(fail)?;
                let frontier = self.first_frontier.checked_add(prefix)
                    .ok_or(ExternalOccurrenceError::Overflow)?;
                if frontier >= self.closing { return Err(fail()); }
                let block_width = maximum.checked_add(1).ok_or(ExternalOccurrenceError::Overflow)?;
                let (stateful, valid_width) = match kind {
                    ExternalInvocationKind::TargetVerification => (true, width <= block_width),
                    ExternalInvocationKind::AssistantStep => (true, width == 1),
                    ExternalInvocationKind::TargetTokenEmbeddings => (false,
                        if self.shape == ExternalPredictionShape::Sequential { width == 1 }
                        else { width >= 2 && width <= block_width }),
                    ExternalInvocationKind::TargetProjectLogits => (false, width <= block_width),
                    ExternalInvocationKind::AssembleContext => (false, width <= block_width),
                    // Pending context may include the retained prompt suffix;
                    // its exact physical source is authenticated by the quote.
                    ExternalInvocationKind::UpdateContext => (false, width <= frontier),
                    ExternalInvocationKind::FusedProposal => (true, width >= 2 && width <= block_width),
                    ExternalInvocationKind::TargetPrefill => unreachable!(),
                };
                if !valid_width || geometry.cached_positions != if stateful { frontier } else { 0 }
                    || (stateful && !frontier.checked_add(width).is_some_and(|end| end <= self.closing)) {
                    return Err(fail());
                }
            }
        }
        Ok(ExternalInvocation { kind, geometry, span, origin })
    }
    /// Move counters outside all model/cache snapshot owners.
    pub fn into_cursor(self) -> ExternalOccurrenceCursor<'a> {
        ExternalOccurrenceCursor { plan: self, spent: [0; 8], attempted: 0 }
    }
}

fn populations(shape: ExternalPredictionShape, prefill: usize, rounds: usize,
    maximum: usize, proposal_blocks: usize) -> Result<[usize; 8], ExternalOccurrenceError>
{
    let overflow = || ExternalOccurrenceError::Overflow;
    let blocks = rounds.checked_mul(proposal_blocks).ok_or_else(overflow)?;
    let verification = rounds.checked_mul(2).ok_or_else(overflow)?;
    let values = match shape {
        ExternalPredictionShape::Sequential => {
            let steps = blocks.checked_mul(maximum).ok_or_else(overflow)?;
            [prefill, verification, steps, 0, steps, 0, 0, 0]
        }
        ExternalPredictionShape::Fused => {
            let blocks = if maximum == 0 { 0 } else { blocks };
            // Each completed target boundary constructs one state; a rejected
            // suffix is replayed before that same state boundary, not twice.
            [prefill, verification, blocks, blocks, 0,
                prefill.checked_add(rounds).ok_or_else(overflow)?, blocks, blocks]
        }
    };
    total(values)?;
    Ok(values)
}
fn total(values: [usize; 8]) -> Result<usize, ExternalOccurrenceError> {
    values.into_iter().try_fold(0usize, usize::checked_add).ok_or(ExternalOccurrenceError::Overflow)
}

/// Move-only proof of one consumed actual attempt.
#[derive(Debug)]
pub struct ExternalOccurrenceClaim<'a> {
    selected: &'a SelectedSpeculativeRealization,
    identity: ExternalScheduleIdentity,
    invocation: ExternalInvocation,
    ordinal: usize,
}
impl ExternalOccurrenceClaim<'_> {
    /// Exact retained selected source contract.
    pub const fn selected(&self) -> &SelectedSpeculativeRealization { self.selected }
    /// Exact schedule issuer.
    pub const fn identity(&self) -> ExternalScheduleIdentity { self.identity }
    /// Actual validated geometry and coordinate.
    pub const fn invocation(&self) -> ExternalInvocation { self.invocation }
    /// Monotonic attempted-work slot.
    pub const fn ordinal(&self) -> usize { self.ordinal }
}
/// Attempt counters, never stored in a model snapshot.
#[derive(Debug)]
pub struct ExternalOccurrenceCursor<'a> {
    plan: ExternalSchedulePlan<'a>,
    spent: [usize; 8],
    attempted: usize,
}
impl<'a> ExternalOccurrenceCursor<'a> {
    /// Retained source-derived plan.
    pub const fn plan(&self) -> &ExternalSchedulePlan<'a> { &self.plan }
    /// Failed or discarded calls remain spent.
    pub const fn attempted(&self) -> usize { self.attempted }
    /// Consume before source-dependent quoting or account acceptance.
    pub fn claim(&mut self, invocation: ExternalInvocation)
        -> Result<ExternalOccurrenceClaim<'a>, ExternalOccurrenceError>
    {
        if self.plan.invocation(invocation.kind, invocation.geometry, invocation.span, invocation.origin)? != invocation {
            return Err(ExternalOccurrenceError::Geometry);
        }
        let index = invocation.kind as usize;
        let (ordinal, _) = super::attempts::consume(&mut self.spent, &mut self.attempted,
            index, self.plan.limits[index]).map_err(|error| match error {
                super::attempts::AttemptError::Exhausted => ExternalOccurrenceError::Exhausted,
                super::attempts::AttemptError::Overflow => ExternalOccurrenceError::Overflow,
            })?;
        Ok(ExternalOccurrenceClaim { selected: self.plan.selected, identity: self.plan.identity,
            invocation, ordinal })
    }
    /// Describe fresh future slots without rewinding attempts or prefill.
    pub fn continuation(&self, committed: usize, status: SpeculativeRequestStatus)
        -> Result<ExternalContinuation, ExternalOccurrenceError>
    {
        let output = self.plan.geometry.output_positions();
        if committed > output { return Err(ExternalOccurrenceError::Geometry); }
        let rounds = match status {
            SpeculativeRequestStatus::Completed => 0,
            SpeculativeRequestStatus::ReadyToDraft if committed != 0 && committed < output => output - committed,
            _ => return Err(ExternalOccurrenceError::Geometry),
        };
        let added = populations(self.plan.shape, 0, rounds,
            self.plan.geometry.proposal_count(committed), self.plan.proposal_blocks)?;
        let mut next = self.plan.limits;
        for (slot, count) in next.iter_mut().zip(added) {
            *slot = slot.checked_add(count).ok_or(ExternalOccurrenceError::Overflow)?;
        }
        Ok(ExternalContinuation { identity: self.plan.identity, previous: self.plan.limits, next,
            previous_slots: total(self.plan.limits)?, next_slots: total(next)? })
    }
    /// Install only after the exact request account funds replacement storage.
    pub fn install_continuation(&mut self, continuation: ExternalContinuation)
        -> Result<(), ExternalOccurrenceError>
    {
        if continuation.identity != self.plan.identity || continuation.previous != self.plan.limits {
            return Err(ExternalOccurrenceError::Geometry);
        }
        self.plan.limits = continuation.next;
        Ok(())
    }
}
/// Move-only extension of the same schedule's finite destination.
#[derive(Debug)]
pub struct ExternalContinuation {
    identity: ExternalScheduleIdentity,
    previous: [usize; 8],
    next: [usize; 8],
    previous_slots: usize,
    next_slots: usize,
}
impl ExternalContinuation {
    /// Same issuer as the original request.
    pub const fn identity(&self) -> ExternalScheduleIdentity { self.identity }
    /// Exact current slot count.
    pub const fn previous_slots(&self) -> usize { self.previous_slots }
    /// Exact funded replacement slot count.
    pub const fn next_slots(&self) -> usize { self.next_slots }
    /// Scalar planning and cursor-loan controls; destination is charged by owner.
    pub fn control_bytes(&self) -> Option<usize> {
        let parts = [std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, ExternalOccurrenceError>>(),
            std::mem::size_of::<std::cell::RefMut<'_, ExternalOccurrenceCursor<'_>>>(),
            std::mem::size_of::<(usize, SpeculativeRequestStatus)>(),
            std::mem::size_of::<([usize; 8], [usize; 8], usize, usize)>()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}
