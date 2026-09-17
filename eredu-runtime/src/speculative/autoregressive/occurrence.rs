//! Fresh independent-model request occurrences, consumed without rollback refunds.
use super::AutoregressivePass;
use crate::SelectedSpeculativeRealization;
use eredu_core::{
    generation::{SpeculativeConfig, SpeculativeSchedulerOptions},
    speculative::SpeculativeRequestGeometry,
    GenerationError,
};
use std::{
    num::{NonZeroU64, NonZeroUsize},
    sync::atomic::{AtomicU64, Ordering},
};

// Request identity only, not an execution grant or a source-epoch identity.
// Refuse exhaustion instead of reusing an identity belonging to a live claim.
static NEXT_SCHEDULE: AtomicU64 = AtomicU64::new(1);

/// Exact selected model role; this is descriptive and issues no execution grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AutoregressiveSource {
    /// Canonical target and its tentative verification/replay branches.
    Target,
    /// Independent draft, including discardable optimistic branches.
    Draft,
}
/// One actual decoder call at the existing shared mechanism boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoregressiveInvocation {
    pass: AutoregressivePass,
    positions: usize,
}
impl AutoregressiveInvocation {
    pub(super) fn decode(pass: AutoregressivePass, positions: usize) -> Option<Self> {
        if positions == 0
            || matches!(
                pass,
                AutoregressivePass::TargetPrefill | AutoregressivePass::DraftPrefill
            )
            || matches!(pass, AutoregressivePass::Proposal) && positions != 1
        {
            return None;
        }
        Some(Self { pass, positions })
    }
    pub(super) fn prefill(pass: AutoregressivePass, positions: usize) -> Option<Self> {
        (positions != 0
            && matches!(
                pass,
                AutoregressivePass::TargetPrefill | AutoregressivePass::DraftPrefill
            ))
        .then_some(Self { pass, positions })
    }
    /// The existing independent-draft worker's semantic operation.
    pub const fn pass(self) -> AutoregressivePass {
        self.pass
    }
    /// Number of actual input tokens, including the seed in verification.
    pub const fn positions(self) -> usize {
        self.positions
    }
    /// Exact execution pass sent by the shared independent-decoder mechanism.
    /// In particular, verification and replay remain Decode at widths above one.
    pub const fn execution_pass(self) -> crate::ExpertPass {
        match self.pass {
            AutoregressivePass::TargetPrefill | AutoregressivePass::DraftPrefill => {
                crate::ExpertPass::Prefill
            }
            _ => crate::ExpertPass::Decode,
        }
    }
    /// Model selected by the existing pass, never by a backend family name.
    pub const fn source(self) -> AutoregressiveSource {
        match self.pass {
            AutoregressivePass::TargetPrefill
            | AutoregressivePass::Verification
            | AutoregressivePass::TargetCommit => AutoregressiveSource::Target,
            AutoregressivePass::DraftPrefill
            | AutoregressivePass::Proposal
            | AutoregressivePass::DraftCommit => AutoregressiveSource::Draft,
        }
    }
}

/// A source-dependent set of possible invocations, not a monotonicity claim.
/// A quote must run the actual equations for the contained frontier/width pairs
/// and reduce alternative costs by maximum before multiplying by attempts.
/// Pricing only the largest shape is not sufficient without a mechanism proof.
#[derive(Clone, Copy, Debug)]
pub struct AutoregressiveInvocationDomain {
    pass: AutoregressivePass,
    attempts: usize,
    first_frontier: u64,
    last_frontier: u64,
    minimum_positions: usize,
    maximum_positions: usize,
    closing_ceiling: u64,
}
impl AutoregressiveInvocationDomain {
    /// Semantic pass and therefore exact target/draft source role.
    pub const fn pass(self) -> AutoregressivePass {
        self.pass
    }
    /// Cumulative attempted calls, including failed or discarded prefixes.
    pub const fn attempts(self) -> usize {
        self.attempts
    }
    /// Visits each possible geometry without allocating an inventory or running
    /// a scheduler. The same membership check validates actual cursor claims.
    pub fn visit<E>(
        self,
        mut visitor: impl FnMut(u64, AutoregressiveInvocation) -> Result<(), E>,
    ) -> Result<(), E> {
        if self.attempts == 0 {
            return Ok(());
        }
        for frontier in self.first_frontier..=self.last_frontier {
            for positions in self.minimum_positions..=self.maximum_positions {
                let invocation = AutoregressiveInvocation {
                    pass: self.pass,
                    positions,
                };
                if self.contains(frontier, invocation) {
                    visitor(frontier, invocation)?;
                }
            }
        }
        Ok(())
    }
    fn contains(self, frontier: u64, invocation: AutoregressiveInvocation) -> bool {
        self.attempts != 0
            && invocation.pass == self.pass
            && (self.first_frontier..=self.last_frontier).contains(&frontier)
            && (self.minimum_positions..=self.maximum_positions).contains(&invocation.positions)
            && u64::try_from(invocation.positions)
                .ok()
                .and_then(|n| frontier.checked_add(n))
                .is_some_and(|end| end <= self.closing_ceiling)
    }
}

/// Fixed failure before the requested invocation may allocate or mutate state.
#[derive(Debug, thiserror::Error)]
pub enum AutoregressiveOccurrenceError {
    /// Existing configuration/scheduler validation is preserved unchanged.
    #[error(transparent)]
    Generation(#[from] GenerationError),
    /// The selected independent executor and retained strategy disagree.
    #[error("independent speculative source selection differs")]
    Selection,
    /// A finite count or position cannot be represented.
    #[error("independent speculative occurrence geometry overflow")]
    Overflow,
    /// Prompt plus the complete tentative frontier exceeds the actual context cap.
    #[error("independent speculative context capacity exceeded")]
    Context,
    /// The actual source frontier/pass/width is outside its selected domain.
    #[error("independent speculative invocation differs from its selected geometry")]
    Geometry,
    /// An occurrence owner is already active on this synchronous call stack.
    #[error("independent speculative occurrence owner is reentrant")]
    Reentrant,
    /// Every attempt in this pass has already been consumed, including failures.
    #[error("independent speculative occurrence capacity exhausted")]
    Exhausted,
}

/// Opaque allocation-free identity of one prepared occurrence schedule.
/// Equality binds descriptive preparations, never native work permission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoregressiveScheduleIdentity(u64);

/// Move-only source loan and finite geometry for one fresh independent-draft
/// request. This is not H/Q admission, a native recipe or a source-epoch proof.
/// The enclosing caller must retain/revalidate both actual model epochs and
/// supply complete copy, sampling, completion and publication contributions.
#[derive(Debug)]
pub struct AutoregressiveSchedulePlan<'a> {
    selected: &'a SelectedSpeculativeRealization,
    identity: AutoregressiveScheduleIdentity,
    geometry: SpeculativeRequestGeometry,
    proposal_blocks: usize,
    domains: [AutoregressiveInvocationDomain; 6],
}
impl<'a> AutoregressiveSchedulePlan<'a> {
    /// Builds the finite occurrence plan from the retained selection and actual
    /// driver policy before native source binding. This creates no device,
    /// executable, mutable state or work authority. The source-pair binder must
    /// separately authenticate the concrete target/draft frontiers.
    pub fn new(
        selected: &'a SelectedSpeculativeRealization,
        capacity: NonZeroUsize,
        input: NonZeroU64,
        context: NonZeroU64,
        config: &SpeculativeConfig,
        options: SpeculativeSchedulerOptions,
    ) -> Result<Self, AutoregressiveOccurrenceError> {
        config.validate()?;
        let options = options.validate()?;
        if selected.requirements().strategy().proposal_capacity() != capacity
            || selected.requirements().strategy().class()
                != crate::SpeculativeStrategyClass::External
            || !selected.requirements().capture().entries().is_empty()
        {
            return Err(AutoregressiveOccurrenceError::Selection);
        }
        let geometry = SpeculativeRequestGeometry::new(config, capacity.get());
        let output = u64::try_from(geometry.output_positions())
            .map_err(|_| AutoregressiveOccurrenceError::Overflow)?;
        let closing = input
            .get()
            .checked_add(output)
            .ok_or(AutoregressiveOccurrenceError::Overflow)?;
        if closing > context.get() {
            return Err(AutoregressiveOccurrenceError::Context);
        }
        let input_positions =
            usize::try_from(input.get()).map_err(|_| AutoregressiveOccurrenceError::Overflow)?;
        let rounds = geometry.verification_attempts();
        let maximum = geometry.proposal_count(1);
        let verification = maximum
            .checked_add(1)
            .ok_or(AutoregressiveOccurrenceError::Overflow)?;
        // At most one canonical and one optimistic block per verification.
        // Promotion, EOS, adaptive disabling and cancellation can only remove
        // work. No failed or discarded invocation returns an occurrence.
        let proposal_attempts = rounds
            .checked_mul(maximum)
            .and_then(|n| n.checked_mul(1 + options.lookahead_blocks))
            .ok_or(AutoregressiveOccurrenceError::Overflow)?;
        let domain =
            |pass, attempts, minimum_positions, maximum_positions| AutoregressiveInvocationDomain {
                pass,
                attempts,
                first_frontier: input.get(),
                last_frontier: closing.saturating_sub(1),
                minimum_positions,
                maximum_positions,
                closing_ceiling: closing,
            };
        let prefill = |pass| AutoregressiveInvocationDomain {
            pass,
            attempts: usize::from(geometry.output_positions() != 0),
            first_frontier: 0,
            last_frontier: 0,
            minimum_positions: input_positions,
            maximum_positions: input_positions,
            closing_ceiling: input.get(),
        };
        let domains = [
            prefill(AutoregressivePass::TargetPrefill),
            prefill(AutoregressivePass::DraftPrefill),
            domain(AutoregressivePass::Proposal, proposal_attempts, 1, 1),
            domain(AutoregressivePass::Verification, rounds, 2, verification),
            domain(AutoregressivePass::TargetCommit, rounds, 1, maximum),
            domain(AutoregressivePass::DraftCommit, rounds, 1, verification),
        ];
        domains
            .iter()
            .try_fold(0usize, |n, d| n.checked_add(d.attempts))
            .ok_or(AutoregressiveOccurrenceError::Overflow)?;
        let identity = NEXT_SCHEDULE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| AutoregressiveOccurrenceError::Overflow)?;
        Ok(Self {
            selected,
            identity: AutoregressiveScheduleIdentity(identity),
            geometry,
            proposal_blocks: 1 + options.lookahead_blocks,
            domains,
        })
    }
    /// Concrete fixed neutral plan/cursor/claim transports for enclosing host
    /// preparation. No heap, native inspector, error erasure or grant is included.
    pub fn control_bytes(&self) -> Option<usize> {
        let parts = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<AutoregressiveOccurrenceCursor<'_>>(),
            std::mem::size_of::<AutoregressiveInvocation>(),
            std::mem::size_of::<AutoregressiveInvocationDomain>(),
            std::mem::size_of::<AutoregressiveOccurrenceError>(),
            std::mem::size_of::<Result<Self, AutoregressiveOccurrenceError>>(),
            std::mem::size_of::<AutoregressiveOccurrenceClaim<'_>>(),
            std::mem::size_of::<
                Result<AutoregressiveOccurrenceClaim<'_>, AutoregressiveOccurrenceError>,
            >(),
            std::mem::size_of::<std::cell::RefCell<AutoregressiveOccurrenceCursor<'_>>>(),
            std::mem::size_of::<std::cell::RefMut<'_, AutoregressiveOccurrenceCursor<'_>>>(),
            std::mem::size_of::<eredu_core::InferenceGeometry>(),
            std::mem::size_of::<Result<eredu_core::InferenceGeometry, AutoregressiveOccurrenceError>>(
            ),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// Same request identity after this move-only plan enters its cursor.
    pub const fn identity(&self) -> AutoregressiveScheduleIdentity {
        self.identity
    }

    /// Same immutable selection borrowed during typed executor preparation.
    pub const fn selected(&self) -> &SelectedSpeculativeRealization {
        self.selected
    }
    /// Shared canonical/optimistic count policy, not a second scheduling rule.
    pub const fn geometry(&self) -> SpeculativeRequestGeometry {
        self.geometry
    }
    /// Finite alternative domains in the exact pass order used by this mechanism.
    pub fn domains(&self) -> &[AutoregressiveInvocationDomain; 6] {
        &self.domains
    }
    /// Converts one validated source/pass candidate to the ordinary equation
    /// driver's exact geometry. Decoder calls retain full sequence logits even
    /// when a replay caller later discards them. Prefill uses its actual supplied
    /// chunk cap and existing target/draft readout demand.
    pub fn workspace_geometry(
        &self,
        frontier: u64,
        invocation: AutoregressiveInvocation,
        prefill_chunk_positions: NonZeroU64,
    ) -> Result<eredu_core::InferenceGeometry, AutoregressiveOccurrenceError> {
        if !self
            .domains
            .iter()
            .any(|d| d.contains(frontier, invocation))
        {
            return Err(AutoregressiveOccurrenceError::Geometry);
        }
        let count = u64::try_from(invocation.positions)
            .map_err(|_| AutoregressiveOccurrenceError::Overflow)?;
        let prefill = matches!(
            invocation.pass,
            AutoregressivePass::TargetPrefill | AutoregressivePass::DraftPrefill
        );
        let output = match invocation.pass {
            AutoregressivePass::TargetPrefill => eredu_core::OutputDemand::LastPosition,
            AutoregressivePass::DraftPrefill => eredu_core::OutputDemand::StateOnly,
            _ => eredu_core::OutputDemand::Sequence,
        };
        Ok(eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: frontier,
            input_positions: count,
            max_output_tokens: 0,
            prefill_chunk_positions: if prefill {
                prefill_chunk_positions.get().min(count)
            } else {
                count
            },
            output,
        })
    }

    /// Consumes this plan into one monotonic request cursor; no reset/clone API.
    pub fn into_cursor(self) -> AutoregressiveOccurrenceCursor<'a> {
        AutoregressiveOccurrenceCursor {
            plan: self,
            spent: [0; 6],
            attempted: 0,
            cache_started: false,
        }
    }
}

/// One request's attempt counters. Keep this owner outside rollback/snapshot
/// state so restoring a model branch cannot restore spent occurrence capacity.
#[derive(Debug)]
pub struct AutoregressiveOccurrenceCursor<'a> {
    plan: AutoregressiveSchedulePlan<'a>,
    spent: [usize; 6],
    attempted: usize,
    cache_started: bool,
}
/// Finite additional attempts for one restored canonical future. This is only
/// geometry: the same request must fund its actual slot destination, and each
/// later invocation must independently admit its complete native requirements.
/// The private prior limits prevent installing the same extension twice.
#[derive(Debug)]
pub struct AutoregressiveContinuation {
    identity: AutoregressiveScheduleIdentity,
    previous: [usize; 6],
    next: [usize; 6],
    previous_slots: usize,
    next_slots: usize,
}
impl AutoregressiveContinuation {
    /// Immutable schedule lineage; native source/account validation is separate.
    pub const fn identity(&self) -> AutoregressiveScheduleIdentity { self.identity }
    /// Exact existing slot population, including previously consumed attempts.
    pub const fn previous_slots(&self) -> usize { self.previous_slots }
    /// Exact replacement population, not a heuristic capacity multiplier.
    pub const fn next_slots(&self) -> usize { self.next_slots }
    /// Concrete fixed geometry/transport controls for the admitted caller.
    pub fn control_bytes(&self) -> Option<usize> {
        let parts = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, AutoregressiveOccurrenceError>>(),
            std::mem::size_of::<std::cell::RefMut<'_, AutoregressiveOccurrenceCursor<'_>>>(),
            std::mem::size_of::<(usize, eredu_core::generation::SpeculativeRequestStatus)>(),
            std::mem::size_of::<([usize; 6], [usize; 6], usize, usize, usize)>(),
        ];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}
impl<'a> AutoregressiveOccurrenceCursor<'a> {
    pub(super) fn continuation(
        &self,
        committed: usize,
        status: eredu_core::generation::SpeculativeRequestStatus,
    ) -> Result<AutoregressiveContinuation, AutoregressiveOccurrenceError> {
        use eredu_core::generation::SpeculativeRequestStatus;
        let output = self.plan.geometry.output_positions();
        if !self.cache_started || committed > output {
            return Err(AutoregressiveOccurrenceError::Geometry);
        }
        let rounds = match status {
            SpeculativeRequestStatus::Completed => 0,
            SpeculativeRequestStatus::ReadyToDraft if committed != 0 && committed < output => output - committed,
            _ => return Err(AutoregressiveOccurrenceError::Geometry),
        };
        let maximum = self.plan.geometry.proposal_count(committed);
        let proposals = rounds.checked_mul(maximum)
            .and_then(|n| n.checked_mul(self.plan.proposal_blocks))
            .ok_or(AutoregressiveOccurrenceError::Overflow)?;
        // Restoring a committed boundary never repeats target/draft prefill.
        let additional = [0, 0, proposals, rounds, rounds, rounds];
        let previous = self.plan.domains.map(|domain| domain.attempts);
        let mut next = previous;
        for (total, added) in next.iter_mut().zip(additional) {
            *total = total.checked_add(added).ok_or(AutoregressiveOccurrenceError::Overflow)?;
        }
        let count = |parts: [usize; 6]| parts.into_iter().try_fold(0usize, usize::checked_add)
            .ok_or(AutoregressiveOccurrenceError::Overflow);
        Ok(AutoregressiveContinuation {
            identity: self.plan.identity,
            previous, next,
            previous_slots: count(previous)?, next_slots: count(next)?,
        })
    }
    pub(super) fn install_continuation(&mut self, continuation: AutoregressiveContinuation) {
        // Only the parent executor calls this while holding the same exclusive
        // cursor loan used to compute and fund the extension.
        debug_assert_eq!(continuation.identity, self.plan.identity);
        debug_assert_eq!(continuation.previous, self.plan.domains.map(|domain| domain.attempts));
        for (domain, attempts) in self.plan.domains.iter_mut().zip(continuation.next) {
            domain.attempts = attempts;
        }
    }
    pub(super) fn begin_cache(&mut self) -> Result<(), AutoregressiveOccurrenceError> {
        if self.cache_started {
            return Err(AutoregressiveOccurrenceError::Exhausted);
        }
        self.cache_started = true;
        Ok(())
    }
    /// Validates the actual state's frontier and consumes one attempt before
    /// the source-specific native/Workspace producer starts. A subsequent error
    /// or rollback cannot refund it. The move-only claim retains the exact
    /// request geometry; it is not work authority or a native fit.
    pub fn claim(
        &mut self,
        frontier: u64,
        invocation: AutoregressiveInvocation,
    ) -> Result<AutoregressiveOccurrenceClaim<'a>, AutoregressiveOccurrenceError> {
        let index = self
            .plan
            .domains
            .iter()
            .position(|d| d.pass == invocation.pass)
            .ok_or(AutoregressiveOccurrenceError::Geometry)?;
        let domain = self.plan.domains[index];
        if !domain.contains(frontier, invocation) {
            return Err(AutoregressiveOccurrenceError::Geometry);
        }
        let (ordinal, pass_ordinal) = crate::speculative::attempts::consume(
            &mut self.spent, &mut self.attempted, index, domain.attempts,
        ).map_err(|error| match error {
            crate::speculative::attempts::AttemptError::Exhausted => AutoregressiveOccurrenceError::Exhausted,
            crate::speculative::attempts::AttemptError::Overflow => AutoregressiveOccurrenceError::Overflow,
        })?;
        Ok(AutoregressiveOccurrenceClaim {
            schedule: AutoregressiveSchedulePlan {
                selected: self.plan.selected,
                identity: self.plan.identity,
                geometry: self.plan.geometry,
                proposal_blocks: self.plan.proposal_blocks,
                domains: self.plan.domains,
            },
            invocation,
            frontier,
            ordinal,
            pass_ordinal,
        })
    }
    /// Cumulative calls actually claimed, including work later discarded.
    pub const fn attempted(&self) -> usize {
        self.attempted
    }
    /// Exact retained source selection; this loan grants no native role.
    pub fn selected(&self) -> &SelectedSpeculativeRealization {
        self.plan.selected()
    }
}

/// One permanently spent invocation from the shared request cursor. The private
/// geometry copy cannot be moved out to create another cursor. Model/cache
/// epochs, storage admission and native roles must still be bound by the caller.
#[derive(Debug)]
pub struct AutoregressiveOccurrenceClaim<'a> {
    schedule: AutoregressiveSchedulePlan<'a>,
    invocation: AutoregressiveInvocation,
    frontier: u64,
    ordinal: usize,
    pass_ordinal: usize,
}
impl AutoregressiveOccurrenceClaim<'_> {
    /// Geometry of this same request, borrowed without reset/clone capability.
    pub fn schedule(&self) -> &AutoregressiveSchedulePlan<'_> {
        &self.schedule
    }
    /// Actual source/pass/width whose attempt has already been spent.
    pub const fn invocation(&self) -> AutoregressiveInvocation {
        self.invocation
    }
    /// Actual state frontier inspected before native work.
    pub const fn frontier(&self) -> u64 {
        self.frontier
    }
    /// Monotonic attempted invocation across all passes, including failures.
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }
    /// Monotonic attempted invocation within this semantic pass.
    pub const fn pass_ordinal(&self) -> usize {
        self.pass_ordinal
    }
    /// Reject a claim from another preparation of even the same selection.
    pub fn belongs_to(&self, identity: AutoregressiveScheduleIdentity) -> bool {
        self.schedule.identity == identity
    }
}

#[cfg(test)]
mod account_tests;
