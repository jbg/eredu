//! Paid provisional grammar decisions over the shared canonical suffix worker.
use super::{ControllerHistoryError, ControllerHistorySuffix};
use eredu_core::{
    speculative::{PreparedGrammarController, PreparedGrammarSource},
    HostMetadataFunding, HostMetadataFundingError, PackedTokenFilter, PackedTokenFilterError,
};
use std::mem::{size_of, size_of_val};

/// First real refusal while constructing or advancing a provisional grammar.
#[derive(Debug, thiserror::Error)]
pub enum PreparedGrammarBranchCause<C: PreparedGrammarController> {
    /// Checked control extent overflowed before the next operation.
    #[error("prepared grammar branch control extent overflow")]
    Overflow,
    /// The independent copy did not preserve its exact sources/history/account.
    #[error("prepared grammar branch source or destination changed")]
    Source,
    /// Canonical proposed history diverged before a copy was made.
    #[error(transparent)]
    History(#[from] ControllerHistoryError),
    /// The actual account refused before the next constructor or operation.
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    /// The real copy failure owns its partial destination and immutable sources.
    #[error("prepared grammar copy failed: {0}")]
    Copy(#[source] C::CopyError),
    /// The real operation failure owns its mutated parser/history prefix.
    #[error("prepared grammar operation failed: {0}")]
    Operation(#[source] C::Error),
    /// The completed parser did not expose a valid nonempty packed mask.
    #[error(transparent)]
    Mask(#[from] PackedTokenFilterError),
}
/// Completed destinations and nested failed prefixes retire before their account.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PreparedGrammarBranchError<C: PreparedGrammarController> {
    #[source]
    cause: PreparedGrammarBranchCause<C>,
    prefix: Option<C>,
    funding: HostMetadataFunding,
}
impl<C: PreparedGrammarController> PreparedGrammarBranchError<C> {
    /// Exact first refusal; nested failures retain their actual partial state.
    pub fn cause(&self) -> &PreparedGrammarBranchCause<C> {
        &self.cause
    }
}
/// An independent semantic branch; a completed mask can be loaned after computation.
/// No native sampler authority is granted by this owner or its borrowed filter.
#[derive(Debug)]
pub struct PreparedGrammarBranch<C: PreparedGrammarController> {
    controller: C,
    funding: HostMetadataFunding,
}
impl<C: PreparedGrammarController> PreparedGrammarBranch<C> {
    /// Complete branch payment for copying the current committed history only.
    /// Prospective suffix commitment and mask/terminal operations are separate
    /// producers and must not use this amount as their execution allowance.
    pub fn committed_copy_bytes(source: &C, capacity: usize) -> Option<usize> {
        if capacity < source.prepared_grammar_source().history().len() { return None; }
        Self::control_bytes()?.checked_add(source.prepared_grammar_copy_bytes(capacity)?)
    }
    fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<C>(),
            size_of::<Option<C>>(),
            size_of::<PreparedGrammarBranchCause<C>>(),
            size_of::<PreparedGrammarBranchError<C>>(),
            size_of::<Result<Self, PreparedGrammarBranchError<C>>>(),
            size_of::<Result<(Self, bool), PreparedGrammarBranchError<C>>>(),
            size_of::<Result<C, C::Error>>(),
            size_of::<Result<C, C::CopyError>>(),
            size_of::<Result<(C, bool), C::Error>>(),
            size_of::<Result<(), PreparedGrammarBranchCause<C>>>(),
            size_of::<ControllerHistorySuffix<'_>>(),
            size_of::<Result<ControllerHistorySuffix<'_>, ControllerHistoryError>>(),
            size_of::<(&C, &[u32], usize, &HostMetadataFunding)>(),
            size_of::<(&mut Option<C>,)>(),
            size_of::<u32>(),
            size_of::<bool>(),
            size_of::<std::slice::Iter<'_, u32>>(),
            PreparedGrammarSource::control_bytes()?,
            PackedTokenFilter::control_bytes(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Copies the committed owner and visits the same canonical suffix as ordinary
    /// controllers. Terminal queries need not compute a token mask first.
    #[inline(never)]
    pub fn fork_at(
        source: &C,
        history: &[u32],
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, PreparedGrammarBranchError<C>> {
        let mut prefix = None;
        let result = (|| -> Result<(), PreparedGrammarBranchCause<C>> {
            funding.reserve_metadata(
                Self::control_bytes().ok_or(PreparedGrammarBranchCause::Overflow)?,
            )?;
            let original = source.prepared_grammar_source();
            let suffix = ControllerHistorySuffix::new(original.history(), history)?;
            if capacity < history.len() {
                return Err(PreparedGrammarBranchCause::Source);
            }
            prefix = Some(
                source
                    .copy_prepared_grammar(capacity, funding)
                    .map_err(PreparedGrammarBranchCause::Copy)?,
            );
            if !original.matches_copy(
                prefix
                    .as_ref()
                    .expect("copied grammar")
                    .prepared_grammar_source(),
                capacity,
                funding,
            ) {
                return Err(PreparedGrammarBranchCause::Source);
            }
            suffix.visit(|token| {
                let controller = prefix.take().expect("complete grammar prefix");
                prefix = Some(
                    controller
                        .commit_prepared_grammar(token)
                        .map_err(PreparedGrammarBranchCause::Operation)?,
                );
                Ok::<_, PreparedGrammarBranchCause<C>>(())
            })?;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(Self {
                controller: prefix.take().expect("complete grammar decision"),
                funding: funding.clone(),
            }),
            Err(cause) => Err(PreparedGrammarBranchError {
                cause,
                prefix,
                funding: funding.clone(),
            }),
        }
    }
    /// Builds the same provisional history and then computes its actual mask.
    pub fn at(
        source: &C,
        history: &[u32],
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, PreparedGrammarBranchError<C>> {
        Self::fork_at(source, history, capacity, funding)?.compute_mask()
    }
    /// Computes the actual parser mask while retaining every failed prefix.
    pub fn compute_mask(self) -> Result<Self, PreparedGrammarBranchError<C>> {
        let reservation = Self::control_bytes()
            .ok_or(HostMetadataFundingError::Overflow)
            .and_then(|bytes| self.funding.reserve_metadata(bytes));
        if let Err(cause) = reservation {
            return Err(PreparedGrammarBranchError {
                cause: cause.into(),
                prefix: Some(self.controller),
                funding: self.funding,
            });
        }
        let controller = self
            .controller
            .compute_prepared_grammar_mask()
            .map_err(|cause| PreparedGrammarBranchError {
                cause: PreparedGrammarBranchCause::Operation(cause),
                prefix: None,
                funding: self.funding.clone(),
            })?;
        if let Err(cause) = controller.prepared_grammar_mask().map(|_| ()) {
            return Err(PreparedGrammarBranchError {
                cause: cause.into(),
                prefix: Some(controller),
                funding: self.funding,
            });
        }
        Ok(Self {
            controller,
            funding: self.funding,
        })
    }
    /// Actual completed packed semantic mask, borrowed until the next mutation.
    pub fn mask(&self) -> Result<PackedTokenFilter<'_>, PackedTokenFilterError> {
        self.controller.prepared_grammar_mask()
    }
    /// Exact independent controller retained by this branch.
    pub fn controller(&self) -> &C {
        &self.controller
    }
    /// Moves the independent owner into the caller's canonical commitment path.
    pub fn into_controller(self) -> C {
        self.controller
    }
    /// Evaluates the ordinary terminal policy on this actual independent owner.
    pub fn terminal(self) -> Result<(Self, bool), PreparedGrammarBranchError<C>> {
        let reservation = Self::control_bytes()
            .ok_or(HostMetadataFundingError::Overflow)
            .and_then(|bytes| self.funding.reserve_metadata(bytes));
        if let Err(cause) = reservation {
            return Err(PreparedGrammarBranchError {
                cause: cause.into(),
                prefix: Some(self.controller),
                funding: self.funding,
            });
        }
        match self.controller.prepared_grammar_terminal() {
            Ok((controller, terminal)) => Ok((
                Self {
                    controller,
                    funding: self.funding,
                },
                terminal,
            )),
            Err(cause) => Err(PreparedGrammarBranchError {
                cause: PreparedGrammarBranchCause::Operation(cause),
                prefix: None,
                funding: self.funding,
            }),
        }
    }
}
