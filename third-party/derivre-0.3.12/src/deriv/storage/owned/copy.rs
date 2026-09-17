//! Exact mutable cache copies around the existing expression constructors.
use super::PreparedExpressionMachine;
use crate::{
    ast::{ExprSet, ExprSetCopyFailure, ExprSetPreparedSourcePlan, PreparedExprSet, PreparedExprError},
    copy_storage as storage,
    hashcons::PreparedHashConsFunding,
};
use std::{fmt, mem::{size_of, size_of_val}};
#[derive(Debug)]
enum Cause<E> {
    Storage(storage::Error<E>),
    Source(ExprSetCopyFailure),
    Binding(PreparedExprError),
}
impl<E> From<storage::Error<E>> for Cause<E> {
    fn from(cause: storage::Error<E>) -> Self { Self::Storage(cause) }
}
enum Pending {
    Empty,
    Source(PreparedExprSet),
    Machine(PreparedExpressionMachine),
}
/// An independent expression-copy failure retains every constructed destination
/// and the explicitly supplied new backing account. The original is borrowed.
pub struct PreparedExpressionCopyFailure<E> {
    cause: Cause<E>,
    pending: Pending,
    backing: Option<PreparedHashConsFunding>,
}
impl<E: fmt::Debug> fmt::Debug for PreparedExpressionCopyFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedExpressionCopyFailure").field("cause", &self.cause)
            .field("retains_prefix", &!matches!(self.pending, Pending::Empty)).finish()
    }
}
impl<E: fmt::Display> fmt::Display for PreparedExpressionCopyFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Storage(e) => fmt::Display::fmt(e, f),
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Binding(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedExpressionCopyFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Storage(e) => e, Cause::Source(e) => e,
            Cause::Binding(e) => e,
        })
    }
}
impl PreparedExpressionMachine {
    fn copy_controls<E>() -> Option<usize> {
        let parts = [
                ExprSet::prepared_source_inspection_control_bytes()?,
                size_of::<Self>(), size_of::<&Self>(), size_of::<Pending>(), size_of::<Cause<E>>(),
                size_of::<PreparedExpressionCopyFailure<E>>(),
                size_of::<Option<PreparedHashConsFunding>>(),
                size_of::<Result<Self, PreparedExpressionCopyFailure<E>>>(),
                size_of::<Result<(), Cause<E>>>(), size_of::<Result<(), E>>(),
                size_of::<Result<ExprSetPreparedSourcePlan<'_>, ExprSetCopyFailure>>(),
                size_of::<Result<PreparedExprSet, ExprSetCopyFailure>>(),
                size_of::<Result<(), PreparedExprError>>(),
                size_of::<(&Self, &())>(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn copy_destination_controls<E>(&self) -> Option<usize> {
        self.parts.as_ref()?.copy_destination_controls::<E>()?
            .checked_add(self.symbolic.as_ref()?.copy_destination_controls::<E>()?)?
            .checked_add(self.weights.as_ref()?.copy_destination_controls::<E>()?)?
            .checked_add(self.relevance.as_ref()?.copy_destination_controls::<E>()?)
    }
    /// Complete prospective independent-copy payment for this exact saved
    /// owner and funding-error representation. Reads actual source/table and
    /// cache capacities; does not allocate, invoke funding, or mutate a cache.
    /// None means a failed/incomplete source or overflowing storage geometry.
    pub fn copy_required_bytes<E>(&self) -> Option<usize> {
        if self.failed { return None; }
        let plan = self.source.source().prepared_source_plan().ok()?;
        Self::copy_controls::<E>()?
            .checked_add(plan.requirements().required_bytes())?
            .checked_add(self.copy_destination_controls::<E>()?)?
            .checked_add(self.parts.as_ref()?.copy_required_bytes::<E>()?)?
            .checked_add(self.symbolic.as_ref()?.copy_required_bytes::<E>()?)?
            .checked_add(self.weights.as_ref()?.copy_required_bytes::<E>()?)?
            .checked_add(self.relevance.as_ref()?.copy_required_bytes::<E>()?)
    }
    /// Copies this complete expression graph and every persistent mutable
    /// traversal/cache through the existing independent source constructors.
    /// The old backing authority is never copied: future growth requires the
    /// explicitly supplied new account; None leaves the independent copy fixed.
    pub fn try_copy<F: Fn(usize) -> Result<(), E>, E>(
        &self, backing: Option<PreparedHashConsFunding>, funding: &F,
    ) -> Result<Self, PreparedExpressionCopyFailure<E>> {
        let mut pending = Pending::Empty;
        let result = (|| -> Result<(), Cause<E>> {
            funding(Self::copy_controls::<E>().ok_or(storage::Error::Overflow)?)
                .map_err(storage::Error::Funding)?;
            if self.failed || self.parts.is_none() || self.symbolic.is_none()
                || self.weights.is_none() || self.relevance.is_none() {
                return Err(storage::Error::Source.into());
            }
            let plan = self.source.source().prepared_source_plan().map_err(Cause::Source)?;
            funding(plan.requirements().required_bytes()).map_err(storage::Error::Funding)?;
            pending = Pending::Source(plan.compile().map_err(Cause::Source)?);
            if let Some(backing) = &backing {
                let Pending::Source(source) = &mut pending else { unreachable!() };
                source.bind_backing_funding(backing.clone()).map_err(Cause::Binding)?;
            }
            // Retain the independently copied expression source before paying
            // for every allocation-free mutable destination constructor.
            funding(self.copy_destination_controls::<E>().ok_or(storage::Error::Source)?)
                .map_err(storage::Error::Funding)?;
            let Pending::Source(source) = std::mem::replace(&mut pending, Pending::Empty) else { unreachable!() };
            pending = Pending::Machine(Self {
                source,
                parts: Some(self.parts.as_ref().expect("checked derivative source").copy_destination()),
                symbolic: Some(self.symbolic.as_ref().expect("checked symbolic source").copy_destination()),
                weights: Some(self.weights.as_ref().expect("checked weight source").copy_destination()),
                relevance: Some(self.relevance.as_ref().expect("checked relevance source").copy_destination()),
                failed: true,
            });
            let Pending::Machine(copied) = &mut pending else { unreachable!() };
            copied.parts.as_mut().ok_or(storage::Error::Source)?
                .restore_copy(self.parts.as_ref().ok_or(storage::Error::Source)?, funding)?;
            copied.symbolic.as_mut().ok_or(storage::Error::Source)?
                .restore_copy(self.symbolic.as_ref().ok_or(storage::Error::Source)?, funding)?;
            copied.weights.as_mut().ok_or(storage::Error::Source)?
                .restore_copy(self.weights.as_ref().ok_or(storage::Error::Source)?, funding)?;
            copied.relevance.as_mut().ok_or(storage::Error::Source)?
                .restore_copy(self.relevance.as_ref().ok_or(storage::Error::Source)?, funding)?;
            copied.failed = self.failed;
            Ok(())
        })();
        match result {
            Ok(()) => { let Pending::Machine(copied) = pending else { unreachable!() }; Ok(copied) }
            Err(cause) => Err(PreparedExpressionCopyFailure { cause, pending, backing }),
        }
    }
}
