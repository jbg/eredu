//! Selected independent bank reads through existing completion-safe mechanisms.
use super::{
    ParameterSlotOperation, PreparedBankParameter, PreparedBankParameterMember,
    PreparedParameterLocation,
};
use std::collections::BTreeSet;

/// A selected canonical parameter is absent from its exact retained bank owner.
#[derive(Debug, thiserror::Error)]
#[error("selected bank parameter {parameter:?} is absent from bank {bank} unit {unit}")]
pub struct BankParameterReadSelectionError {
    /// Selected canonical parameter identity.
    pub parameter: String,
    /// Required physical bank owner.
    pub bank: usize,
    /// Architecture-global unit owner.
    pub unit: usize,
}

/// Borrowed exact selection, admitted completely before any tensor work occurs.
/// This plan selects no execution mechanism and grants no memory or edit authority.
pub struct BankParameterReadPlan<'a> {
    parameters: Vec<&'a PreparedBankParameter>,
}
impl<'a> BankParameterReadPlan<'a> {
    /// Resolves every selected canonical identity at its exact physical owner,
    /// retaining deterministic request order and prepared member assembly order.
    pub fn new(
        bank: usize,
        unit: usize,
        parameters: &'a [PreparedBankParameter],
        selected: &BTreeSet<String>,
    ) -> Result<Self, BankParameterReadSelectionError> {
        let location = PreparedParameterLocation::Bank { bank, unit };
        let parameters = selected
            .iter()
            .map(|id| {
                parameters
                    .iter()
                    .find(|parameter| {
                        parameter.slot.parameter.id.as_str() == id
                            && parameter.slot.location == location
                    })
                    .ok_or_else(|| BankParameterReadSelectionError {
                        parameter: id.clone(),
                        bank,
                        unit,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { parameters })
    }

    /// Runs the ordinary parameter traversal over replacement values or
    /// sequentially copied members. The caller reserves all outputs and assembly
    /// workspace before entry. `copy_member` must finish or safely detach each
    /// copy from its source loan before returning; native completion and failure
    /// recovery stay with that mechanism. `assemble` returns a safely retained
    /// value. All completed values live through the operation, including failure.
    /// No parameter handle is published by this traversal.
    pub fn with_slots<T, M, E>(
        &self,
        mut replacement: impl FnMut(&str) -> Option<T>,
        mut copy_member: impl FnMut(&PreparedBankParameterMember) -> Result<M, E>,
        mut assemble: impl FnMut(Vec<M>) -> Result<T, E>,
        operation: &mut ParameterSlotOperation<'_, T, E>,
    ) -> Result<(), E> {
        let mut values = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            let value = if let Some(value) = replacement(parameter.slot.parameter.id.as_str()) {
                value
            } else {
                let mut copies = Vec::with_capacity(parameter.members.len());
                for member in &parameter.members {
                    copies.push(copy_member(member)?);
                }
                assemble(copies)?
            };
            values.push((&parameter.slot, value));
        }
        operation(&mut |visitor| {
            for (slot, value) in &mut values {
                visitor.visit_slot(slot.parameter.clone(), value);
            }
        })
    }
}

/// Native snapshots and copy operations for one already selected bank.
pub trait BankParameterReadSource<T, C: ?Sized, E> {
    /// Installed cache used for member loans.
    type Residency: crate::residency::ResidencyAccess<
        Error: Into<E>,
        Transfer: eredu_core::residency::ResidencyTransferOrdering<C, Error: Into<E>>,
    >;
    /// Native scope that holds the exact loan through completion.
    type Loan: eredu_core::residency::ParameterLoan<T, C, Error = E>;

    /// Snapshots the installed cache and replacement handles at the same boundary.
    fn read_resources(
        &self,
        selected: &BTreeSet<String>,
    ) -> Result<(Self::Residency, std::collections::BTreeMap<String, T>), E>;

    /// Opens native retention for the selected and ordered transfer.
    fn begin_loan(
        transfer: <Self::Residency as crate::residency::ResidencyAccess>::Transfer,
        context: &C,
    ) -> Result<Self::Loan, E>;

    /// Assembles copied members in canonical order.
    fn assemble_members(copies: Vec<T>, context: &C) -> Result<T, E>;
}

impl BankParameterReadPlan<'_> {
    /// Acquires member leases in portable order and hands each to the copy mechanism.
    pub fn with_resident_slots<T: Clone, C: ?Sized, E, S: BankParameterReadSource<T, C, E>>(
        &self,
        source: &S,
        selected: &BTreeSet<String>,
        context: &C,
        operation: &mut ParameterSlotOperation<'_, T, E>,
    ) -> Result<(), E> {
        use crate::residency::ResidencyAccess;
        use eredu_core::residency::{MemoryTier, ParameterLoan, ResidencyTransferOrdering};

        let (residency, replacements) = source.read_resources(selected)?;
        self.with_slots(
            |id| replacements.get(id).cloned(),
            |member| {
                let transfer = residency
                    .acquire_many_with_transfer(&[(member.key.unit_id(), 1)], MemoryTier::Device)
                    .map_err(Into::into)?;
                transfer.order_after(context).map_err(Into::into)?;
                let mut loan = S::begin_loan(transfer, context)?;
                let value = loan.copy_binding(&member.binding, context)?;
                loan.finish(&value)?;
                Ok(value)
            },
            |copies| S::assemble_members(copies, context),
            operation,
        )
    }
}
