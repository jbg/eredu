//! One prospective canonical choice through the existing ordinary sampler.
use super::SnapshotTokenController;
use crate::TokenDomain;
use eredu_core::{TokenFilter, TokenFilterController};

/// Rejected canonical choice or error from the original grammar controller.
#[derive(Debug, thiserror::Error)]
pub enum TokenChoiceError<E: std::error::Error + 'static> {
    /// The existing grammar/filter failed.
    #[error("token constraint failed: {0}")]
    Constraint(#[source] E),
    /// ID is outside the canonical vocabulary domain.
    #[error("forced token {0} is outside the canonical vocabulary")]
    InvalidToken(u32),
    /// The active grammar/filter disallows this canonical ID.
    #[error("forced token {0} conflicts with the active token constraints")]
    Forbidden(u32),
    /// A choice is already waiting for commitment.
    #[error("a forced choice is already pending; clear it before replacing it")]
    AlreadyPending,
    /// Backend commitment violated the filter it received.
    #[error("forced token {expected} was requested, but backend committed {actual}")]
    UnexpectedCommit {
        /// The sole allowed canonical ID.
        expected: u32,
        /// The ID returned by the backend.
        actual: u32,
    },
}

/// Adds a one-decision restriction to an existing canonical constraint owner.
/// Native sampling, penalties, history and RNG still use the ordinary path. A
/// forced decision consumes exactly that sampler's one-candidate decision: at
/// nonzero temperature RNG advances as usual, and Mirostat observes probability
/// one. It does not edit an already committed prefix or replay text.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenChoiceController<C> {
    inner: C,
    domain: TokenDomain,
    pending: Option<u32>,
    last_forced: bool,
}

impl<C: TokenFilterController> TokenChoiceController<C> {
    /// Wraps the existing grammar with the canonical tokenizer vocabulary domain.
    pub fn new(inner: C, domain: TokenDomain) -> Self {
        Self {
            inner,
            domain,
            pending: None,
            last_forced: false,
        }
    }
    /// Stages a choice after checking vocabulary and the current active grammar.
    /// No token is committed and no sampler/RNG or model state advances here.
    pub fn force_next(&mut self, token: u32) -> Result<(), TokenChoiceError<C::Error>> {
        if self.pending.is_some() {
            return Err(TokenChoiceError::AlreadyPending);
        }
        if token as usize >= self.domain.cardinality() {
            return Err(TokenChoiceError::InvalidToken(token));
        }
        let filter = self
            .inner
            .current_filter()
            .map_err(TokenChoiceError::Constraint)?;
        Self::check_filter(&filter, token)?;
        self.pending = Some(token);
        Ok(())
    }
    /// Removes an uncommitted choice without changing history or randomness.
    pub fn clear_forced(&mut self) -> bool {
        self.pending.take().is_some()
    }
    /// Choice still waiting for the next ordinary commitment.
    pub fn pending_forced(&self) -> Option<u32> {
        self.pending
    }
    /// Whether the most recent canonical commitment consumed a forced choice.
    pub fn last_committed_was_forced(&self) -> bool {
        self.last_forced
    }
    /// Read-only access to the original canonical grammar owner.
    pub fn inner(&self) -> &C {
        &self.inner
    }
    fn check_filter(filter: &TokenFilter, token: u32) -> Result<(), TokenChoiceError<C::Error>> {
        if filter
            .allowed_mask()
            .is_some_and(|mask| !mask.get(token as usize).copied().unwrap_or(false))
        {
            Err(TokenChoiceError::Forbidden(token))
        } else {
            Ok(())
        }
    }
}
impl<C: TokenFilterController> TokenFilterController for TokenChoiceController<C> {
    type Error = TokenChoiceError<C::Error>;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let filter = self
            .inner
            .current_filter()
            .map_err(TokenChoiceError::Constraint)?;
        let Some(token) = self.pending else {
            return Ok(filter);
        };
        Self::check_filter(&filter, token)?;
        let mut allowed = vec![
            false;
            filter
                .allowed_mask()
                .map_or(self.domain.cardinality(), <[bool]>::len)
        ];
        allowed[token as usize] = true;
        Ok(TokenFilter::allowed(allowed).expect("one canonical candidate is allowed"))
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        if let Some(expected) = self.pending {
            if expected != token {
                return Err(TokenChoiceError::UnexpectedCommit {
                    expected,
                    actual: token,
                });
            }
        }
        self.inner
            .commit_token(token)
            .map_err(TokenChoiceError::Constraint)?;
        self.last_forced = self.pending.take().is_some();
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.inner
            .is_complete()
            .map_err(TokenChoiceError::Constraint)
    }
}
impl<C: SnapshotTokenController> SnapshotTokenController for TokenChoiceController<C> {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        self.inner
            .snapshot_storage_bytes()?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(Self {
            inner: self.inner.fork_snapshot()?,
            domain: self.domain,
            pending: self.pending,
            last_forced: self.last_forced,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    #[derive(Clone)]
    struct Grammar(Vec<u32>);
    impl TokenFilterController for Grammar {
        type Error = Infallible;
        fn current_filter(&mut self) -> Result<TokenFilter, Infallible> {
            let mut allowed = vec![true; 4];
            allowed[self.0.len() % 4] = false;
            Ok(TokenFilter::allowed(allowed).unwrap())
        }
        fn commit_token(&mut self, token: u32) -> Result<(), Infallible> {
            self.0.push(token);
            Ok(())
        }
        fn is_complete(&mut self) -> Result<bool, Infallible> {
            Ok(false)
        }
    }
    impl SnapshotTokenController for Grammar {
        fn snapshot_storage_bytes(&self) -> Option<u64> {
            Some(24 + 4 * self.0.len() as u64)
        }
        fn fork_snapshot(&self) -> Result<Self, String> {
            Ok(self.clone())
        }
    }

    #[test]
    fn choices_validate_constraints_and_commit_once_with_snapshot_isolation() {
        let mut parent = TokenChoiceController::new(Grammar(vec![]), TokenDomain::new(4));
        assert!(matches!(
            parent.force_next(4),
            Err(TokenChoiceError::InvalidToken(4))
        ));
        assert!(matches!(
            parent.force_next(0),
            Err(TokenChoiceError::Forbidden(0))
        ));
        assert!(parent.pending_forced().is_none());
        parent.force_next(2).unwrap();
        assert_eq!(
            parent.current_filter().unwrap().allowed_mask(),
            Some(&[false, false, true, false][..])
        );
        assert!(matches!(
            parent.force_next(3),
            Err(TokenChoiceError::AlreadyPending)
        ));
        let mut child = parent.fork_snapshot().unwrap();
        assert!(matches!(
            parent.commit_token(3),
            Err(TokenChoiceError::UnexpectedCommit { .. })
        ));
        assert!(parent.inner().0.is_empty());
        parent.commit_token(2).unwrap();
        assert!(parent.last_committed_was_forced());
        assert!(parent.pending_forced().is_none());
        assert_eq!(parent.inner().0, [2]);
        assert_eq!(child.pending_forced(), Some(2));
        assert!(child.clear_forced());
        child.commit_token(3).unwrap();
        assert!(!child.last_committed_was_forced());
        assert_eq!(child.inner().0, [3]);
        assert_eq!(parent.inner().0, [2]);
        assert!(matches!(
            parent.force_next(1),
            Err(TokenChoiceError::Forbidden(1))
        ));
        parent.commit_token(3).unwrap();
        assert!(!parent.last_committed_was_forced());
        assert_eq!(parent.inner().0, [2, 3]);
    }
}
