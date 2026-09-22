//! Loans from the installed request's native bank and accepted host account.
use super::*;
use crate::backend::runtime::residency::{
    manager::OriginalMaterializedLoan, storage::native_storage::BankOwner,
};
use eredu_nn::workspace::HostMetadataFunding;

#[derive(Clone)]
pub(super) struct TextMaterializationSource {
    bank: BankOwner,
    controls: OriginalTextControlGuard,
    funding: HostMetadataFunding,
}

#[derive(Clone)]
pub(super) enum MaterializationSource {
    Text(TextMaterializationSource),
    Role {
        owner: MaterializationOwner,
        observer: safemlx::OriginalScopeObserver,
    },
}
impl MaterializationSource {
    fn borrow(&self, registry: &Registry) -> Result<MaterializationOwner, Error> {
        match self {
            Self::Text(source) => source.borrow(registry),
            Self::Role { owner, observer } => {
                if !registry.authenticate()?.same_scope(observer) {
                    return Err(identity());
                }
                Ok(owner.clone())
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct MaterializationOwner {
    budget: safemlx::OriginalBufferBudget,
    funding: HostMetadataFunding,
}

impl MaterializationOwner {
    pub(super) fn from_loan(loan: OriginalMaterializedLoan<'_>) -> Self {
        Self {
            budget: loan.budget.clone(),
            funding: loan.funding.clone(),
        }
    }
    pub(super) fn same_source(&self, other: &Self) -> bool {
        self.budget.same_budget(&other.budget) && self.funding.same_account(&other.funding)
    }

    pub(super) fn from_role(
        role: &crate::backend::submission_recovery::native_role::NativeRoleContext<'_>,
        funding: &HostMetadataFunding,
    ) -> Self {
        Self {
            budget: role.budget().clone(),
            funding: funding.clone(),
        }
    }

    pub(super) fn loan(&self) -> OriginalMaterializedLoan<'_> {
        OriginalMaterializedLoan {
            budget: &self.budget,
            funding: &self.funding,
        }
    }
}

impl TextMaterializationSource {
    pub(super) fn new(
        bank: BankOwner,
        controls: OriginalTextControlGuard,
        funding: HostMetadataFunding,
    ) -> Result<Self, Error> {
        bank.try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .budget_for_controls(&controls)
            .map_err(memory)?;
        Ok(Self {
            bank,
            controls,
            funding,
        })
    }

    pub(super) fn borrow(&self, registry: &Registry) -> Result<MaterializationOwner, Error> {
        registry.authenticate()?;
        self.controls
            .validate_reservation(registry.request.memory_reservation().ok_or_else(identity)?)
            .map_err(memory)?;
        let budget = self
            .bank
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .budget_for_controls(&self.controls)
            .map_err(memory)?
            .clone();
        Ok(MaterializationOwner {
            budget,
            funding: self.funding.clone(),
        })
    }

    pub(super) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<MaterializationSource>(),
            size_of::<MaterializationOwner>(),
            size_of::<Option<MaterializationOwner>>(),
            size_of::<Result<MaterializationOwner, Error>>(),
            size_of::<OriginalMaterializedLoan<'_>>(),
            size_of::<(&Self, &Registry)>(),
            size_of::<
                std::cell::Ref<
                    '_,
                    crate::backend::runtime::residency::storage::native_storage::Bank,
                >,
            >(),
            safemlx::OriginalBufferBudget::inspection_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}

impl Registry {
    pub(super) fn materialization_owner(&self) -> Result<Option<MaterializationOwner>, Error> {
        self.authenticate()?;
        let source = self
            .materialized_recipe
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .clone();
        source
            .as_ref()
            .map(|source| source.borrow(self))
            .transpose()
    }
}
