//! One expression representation; checked mutation retains its finite table.
use super::{Expr, ExprEncodingError, ExprRef, ExprSet};
use crate::hashcons::{
    HashConsCopyFailure, HashConsPreparedSourcePlan, PreparedVecHashCons, VecHashCons,
};
use std::{fmt, ops::Deref};

pub(super) enum ExpressionStorage {
    Ordinary(VecHashCons),
    Prepared(PreparedVecHashCons),
}
impl Deref for ExpressionStorage {
    type Target = VecHashCons;
    fn deref(&self) -> &VecHashCons {
        match self {
            Self::Ordinary(source) => source,
            Self::Prepared(source) => source.source(),
        }
    }
}
impl ExpressionStorage {
    pub(super) fn prepared_source_plan(
        &self,
    ) -> Result<HashConsPreparedSourcePlan<'_>, HashConsCopyFailure> {
        match self {
            Self::Ordinary(source) => source.prepared_source_plan(),
            Self::Prepared(source) => source.prepared_source_plan(),
        }
    }
    pub(super) fn ordinary_mut(&mut self) -> &mut VecHashCons {
        match self {
            Self::Ordinary(source) => source,
            Self::Prepared(_) => {
                panic!("ordinary growth cannot mutate a prepared expression owner")
            }
        }
    }
}

/// A complete independently copied expression source with finite mutation
/// storage. Only read access is public; shared checked constructors own mutation.
/// It does not provide derivative, simplifier, lexer or parser admission.
pub struct PreparedExprSet {
    pub(super) inner: ExprSet,
}
impl fmt::Debug for PreparedExprSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedExprSet")
            .field("expressions", &self.inner.len())
            .finish()
    }
}
impl PreparedExprSet {
    /// Borrows the same IDs, metadata, Unicode source and printer configuration.
    /// This immutable loan cannot invoke ordinary mutable arena constructors.
    pub fn source(&self) -> &ExprSet {
        &self.inner
    }

    /// Binds a separately paid backing owner to this exact quiescent source.
    /// Copies retain their independent storage policy and do not inherit it.
    pub fn bind_backing_funding(
        &mut self,
        funding: crate::hashcons::PreparedHashConsFunding,
    ) -> Result<(), PreparedExprError> {
        let ExpressionStorage::Prepared(storage) = &mut self.inner.exprs else {
            return Err(PreparedExprError::Storage);
        };
        storage
            .bind_backing_funding(funding)
            .map_err(|e| PreparedExprError::Encoding(ExprEncodingError::Storage(e)))
    }

    pub(crate) fn source_mut(&mut self) -> &mut ExprSet {
        &mut self.inner
    }
}

/// A checked mutation could not use its exact prepared expression destination.
#[derive(Debug)]
pub enum PreparedExprError {
    /// The owner does not hold the required finite storage.
    Storage,
    /// An expression references a child outside this exact arena.
    Source,
    /// Actual operation-cost accounting overflowed before emission.
    Cost,
    /// Fixed operation workspace is exhausted or does not match its source.
    Capacity,
    /// The operation scope retains a failed mutation prefix.
    Failed,
    /// This source requests the separate optimized prefix-tree producer.
    Optimized,
    /// Encoding or fixed insertion failed, preserving committed expressions.
    Encoding(ExprEncodingError),
}
impl fmt::Display for PreparedExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage => f.write_str("expression owner has no prepared mutation storage"),
            Self::Source => f.write_str("expression child is outside the prepared arena"),
            Self::Cost => f.write_str("expression operation cost overflow"),
            Self::Capacity => {
                f.write_str("expression workspace exceeds its prepared source destinations")
            }
            Self::Failed => f.write_str("expression operation scope retains a failed prefix"),
            Self::Optimized => {
                f.write_str("expression source requires the optimized prefix-tree producer")
            }
            Self::Encoding(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for PreparedExprError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encoding(e) => Some(e),
            _ => None,
        }
    }
}
impl ExprSet {
    pub(crate) fn grow_prepared_workspace<T>(
        &self,
        values: &mut Vec<T>,
        total: usize,
    ) -> Result<(), PreparedExprError> {
        let ExpressionStorage::Prepared(storage) = &self.exprs else {
            return Err(PreparedExprError::Storage);
        };
        storage
            .grow_workspace(values, total)
            .map_err(|e| PreparedExprError::Encoding(ExprEncodingError::Storage(e)))
    }

    // Exact committed graph geometry. Appended expressions retain topological
    // IDs; no constructor allowance or guessed future nodes enter this scan.
    pub(crate) fn reached_workspace_geometry(
        &self,
    ) -> Result<(usize, usize, usize), PreparedExprError> {
        let mut stack = 1usize;
        let mut width = 0usize;
        for index in 1..self.len() {
            let root = ExprRef::new(u32::try_from(index).map_err(|_| PreparedExprError::Capacity)?);
            let args = self.get_args(root);
            if args
                .iter()
                .any(|child| !child.is_valid() || child.as_usize() >= index)
            {
                return Err(PreparedExprError::Source);
            }
            stack = stack
                .checked_add(args.len())
                .ok_or(PreparedExprError::Capacity)?;
            width = width.max(args.len());
        }
        Ok((self.len(), stack, width))
    }
    /// Quotes an independent empty intern table from this retained source's
    /// exact physical table/backing layout. The encoded width is a separate
    /// operation-shape destination; no expression or DFA identity is transferred.
    pub fn empty_table_source_plan(
        &self,
        encoding_words: usize,
    ) -> Result<crate::hashcons::HashConsEmptySourcePlan<'_>, HashConsCopyFailure> {
        self.exprs
            .prepared_source_plan()?
            .empty_destination(encoding_words)
    }
    pub(crate) fn prepared_extents(&self) -> Result<(usize, usize, usize), PreparedExprError> {
        match &self.exprs {
            ExpressionStorage::Prepared(storage) => Ok((
                storage.max_words(),
                storage.max_entries(),
                storage.max_encoded_words(),
            )),
            ExpressionStorage::Ordinary(_) => Err(PreparedExprError::Storage),
        }
    }
    pub(crate) fn require_prepared(&self) -> Result<(), PreparedExprError> {
        if matches!(&self.exprs, ExpressionStorage::Prepared(_)) {
            Ok(())
        } else {
            Err(PreparedExprError::Storage)
        }
    }
    /// Private emission is used only by shared constructors that compute their
    /// own flags. Raw caller encodings cannot introduce arbitrary ExprRefs.
    pub(crate) fn emit_prepared(
        &mut self,
        expression: Expr<'_>,
    ) -> Result<ExprRef, PreparedExprError> {
        if expression.args().iter().any(|&child| !self.is_valid(child)) {
            return Err(PreparedExprError::Source);
        }
        let ExpressionStorage::Prepared(storage) = &mut self.exprs else {
            return Err(PreparedExprError::Storage);
        };
        let id = expression
            .try_intern_encoded(storage)
            .map_err(PreparedExprError::Encoding)?;
        Ok(ExprRef::new(id))
    }
    pub(crate) fn pay_prepared(&mut self, amount: usize) -> Result<(), PreparedExprError> {
        let amount = u64::try_from(amount).map_err(|_| PreparedExprError::Cost)?;
        self.cost = self
            .cost
            .checked_add(amount)
            .ok_or(PreparedExprError::Cost)?;
        Ok(())
    }
}
