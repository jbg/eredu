//! One frozen target choice and canonical prefix progress for every verifier.
use super::{GenerationError, SpeculativeCommitPlan, VerificationRound};

/// Causal target choice, after the selected source's verification policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationDecision {
    /// The next proposal is accepted.
    Accept(u32),
    /// The first rejected proposal is replaced by this target token.
    Replacement(u32),
    /// Full acceptance, including the zero-proposal causal case, emits a bonus.
    Bonus(u32),
}
impl VerificationDecision {
    /// Authoritative target token; no second draw occurs before commitment.
    pub const fn token(self) -> u32 {
        match self {
            Self::Accept(token) | Self::Replacement(token) | Self::Bonus(token) => token,
        }
    }
}

/// Native prefix that must settle before the staged target becomes committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationBoundary {
    /// Exact accepted input count, including the preceding committed token.
    pub verified_inputs: usize,
    /// A matching, nonterminal proposal leaves another causal row available.
    pub continues: bool,
}

/// Shared bookkeeping for batched and explicitly advanced verification.
/// Sampling, semantic state and native evidence remain with their own owners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationCursor {
    round: VerificationRound,
    pending: Option<(VerificationDecision, bool)>,
}
impl VerificationCursor {
    /// Retains the admitted proposal allowance independently of model parameters.
    pub fn new(proposals: usize) -> Result<Self, GenerationError> {
        Ok(Self {
            round: VerificationRound::new(proposals)?,
            pending: None,
        })
    }
    /// Next causal score row. Pending decisions and resolved cursors reject reads.
    pub fn position(&self) -> Result<usize, GenerationError> {
        self.check_live()?;
        if self.pending.is_some() {
            return Err(GenerationError::VerificationDecisionPending);
        }
        Ok(self.round.accepted)
    }
    fn check_live(&self) -> Result<(), GenerationError> {
        if self.round.terminal || self.round.tail.is_some() {
            return Err(GenerationError::VerificationFinished);
        }
        Ok(())
    }
    /// Freezes one choice without changing accepted progress or committed IDs.
    pub fn stage(
        &mut self,
        decision: VerificationDecision,
        terminal: bool,
    ) -> Result<(), GenerationError> {
        self.position()?;
        let full = self.round.is_full_acceptance();
        if matches!(decision, VerificationDecision::Bonus(_)) != full {
            return Err(GenerationError::InvalidSpeculativeTransition);
        }
        self.pending = Some((decision, terminal));
        Ok(())
    }
    /// Read-only cache commitment plan. Failed native work leaves this choice frozen.
    pub fn boundary(&self, token: u32) -> Result<VerificationBoundary, GenerationError> {
        self.check_live()?;
        let (decision, terminal) = self
            .pending
            .ok_or(GenerationError::VerificationCommitmentMismatch)?;
        if decision.token() != token {
            return Err(GenerationError::VerificationCommitmentMismatch);
        }
        let matching = matches!(decision, VerificationDecision::Accept(_));
        Ok(VerificationBoundary {
            verified_inputs: 1 + self.round.accepted + usize::from(matching),
            continues: matching && !terminal,
        })
    }
    /// Advances bookkeeping only after the enclosing native/output commitment succeeds.
    pub fn commit(&mut self, token: u32) -> Result<(), GenerationError> {
        self.boundary(token)?;
        let (decision, terminal) = self.pending.expect("validated target choice");
        match decision {
            VerificationDecision::Accept(token) => self.round.accept(token, terminal)?,
            VerificationDecision::Replacement(token) => self.round.reject_with(token, terminal)?,
            VerificationDecision::Bonus(token) => self.round.bonus(token, terminal)?,
        }
        self.pending = None;
        Ok(())
    }
    /// Whether the accepted prefix has consumed every proposal.
    pub const fn is_full_acceptance(&self) -> bool {
        self.round.is_full_acceptance()
    }
    /// Complete transaction plan; staging a choice alone cannot publish it.
    pub fn commit_plan(&self) -> Result<SpeculativeCommitPlan<'_>, GenerationError> {
        if self.pending.is_some() {
            return Err(GenerationError::VerificationDecisionPending);
        }
        self.round.commit_plan()
    }
    /// Additional cloned host backing, beyond the inline cursor structure.
    pub fn heap_bytes(&self) -> Option<u64> {
        (self.round.committed_tokens.capacity() as u64).checked_mul(4)
    }
}
