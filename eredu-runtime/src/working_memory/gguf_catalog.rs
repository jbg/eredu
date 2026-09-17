//! One source-owned immutable GGUF catalog through the existing cold account.
use super::{
    original_prepared_native_input::Account, qualified_storage, WorkingMemoryError,
    WorkingMemoryPool,
};
use eredu_checkpoint::{
    gguf_store::{GgufCatalogCompileFailure, GgufCatalogPlan, PreparedGgufCatalog},
    store::StoreError,
};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

// Only this fixed compiler can mint this source-specific origin. A caller of
// the generic checkpoint worker cannot construct or clone this private type.
#[derive(Debug)]
struct CatalogCustody(Account);

/// Exact completed catalog plus its original constructor account. The catalog
/// independently retains that same account through all actual source aliases.
#[derive(Debug)]
pub struct OriginalGgufCatalog {
    catalog: PreparedGgufCatalog,
    account: Account,
}
impl OriginalGgufCatalog {
    /// Continue only this exact checkpoint-owning builder. This moves no raw
    /// allowance and certifies no reader/catalog-wrapper/full-model work.
    pub fn into_prepared(self) -> PreparedGgufCatalog {
        let Self { catalog, account } = self;
        drop(account);
        catalog
    }
    /// Validate actual original accounting, without manufacturing a new hold.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        self.catalog
            .catalog_control_owner::<CatalogCustody>()
            .ok_or(WorkingMemoryError::UnknownBound)?
            .0
            .matches_pool(pool)
            .then_some(())
            .ok_or(WorkingMemoryError::IdentityMismatch)
    }
}
/// Uncalled exact plan, actual failed row prefix, or completed catalog after an
/// accounting refusal. Value storage always retires before its raw account.
#[derive(Debug)]
pub struct OriginalGgufCatalogError {
    input: Option<StoreError>,
    accounting: Option<WorkingMemoryError>,
    compilation: Option<GgufCatalogCompileFailure<CatalogCustody>>,
    checkpoint: Option<eredu_checkpoint::gguf_store::GgufCatalogInput>,
    completed: Option<PreparedGgufCatalog>,
    account: Option<Account>,
}
impl OriginalGgufCatalogError {
    fn rejected(
        plan: GgufCatalogPlan<'_>,
        input: Option<StoreError>,
        accounting: Option<WorkingMemoryError>,
    ) -> Self {
        Self {
            input,
            accounting,
            compilation: None,
            checkpoint: Some(plan.into_input()),
            completed: None,
            account: None,
        }
    }
    /// The actual source limit refusal, before allocation/admission.
    pub fn input_failure(&self) -> Option<&StoreError> {
        self.input.as_ref()
    }
    /// The original qualification/comparison/accounting refusal.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.accounting.as_ref()
    }
    /// The exact uncalled checkpoint-owning plan.
    pub fn rejected_input(&self) -> Option<&eredu_checkpoint::gguf_store::GgufCatalogInput> {
        self.checkpoint.as_ref()
    }
    /// Actual source compiler prefix and input, never reconstructed for error.
    pub fn compilation_failure(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.compilation
            .as_ref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
    /// Same failed original checkpoint, with no native/custody extraction.
    pub fn failed_input(&self) -> Option<&eredu_checkpoint::gguf_store::GgufCatalogInput> {
        self.compilation.as_ref().map(|e| e.input())
    }
    /// Actual installed catalog prefix, for diagnostics without retrying work.
    pub fn completed_rows(&self) -> Option<usize> {
        self.compilation.as_ref().map(|e| e.completed_rows())
    }
}
impl fmt::Display for OriginalGgufCatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(e) = &self.input {
            e.fmt(f)
        } else if let Some(e) = &self.compilation {
            e.fmt(f)
        } else {
            self.accounting.as_ref().expect("catalog failure").fmt(f)
        }
    }
}
impl std::error::Error for OriginalGgufCatalogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Some(e) = &self.input {
            Some(e)
        } else if let Some(e) = &self.compilation {
            std::error::Error::source(e)
        } else {
            self.accounting
                .as_ref()
                .map(|e| e as &(dyn std::error::Error + 'static))
        }
    }
}
impl WorkingMemoryPool {
    pub(super) fn validate_prepared_gguf_catalog(
        &self,
        catalog: &PreparedGgufCatalog,
    ) -> Result<(), WorkingMemoryError> {
        let custody = catalog
            .catalog_control_owner::<CatalogCustody>()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if custody.0.matches_pool(self) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    /// Complete managed catalog producer request. The existing pinned storage
    /// qualifier covers the concrete fresh clone/reserve/Box/Arc workers only.
    pub fn gguf_catalog_required_bytes(
        plan: &GgufCatalogPlan<'_>,
    ) -> Result<u64, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let requests = plan
            .requested_storage::<CatalogCustody>()
            .ok_or(WorkingMemoryError::Overflow)?;
        let shared = usize::try_from(qualified_storage::shared_layout_bytes(
            requests.shared_body(),
        )?)
        .map_err(|_| WorkingMemoryError::Overflow)?;
        let controls = [
            Account::storage_bytes().ok_or(WorkingMemoryError::UnknownBound)?,
            size_of::<super::loaded_decode_source::Allowance>(),
            size_of::<Account>(),
            size_of::<CatalogCustody>(),
            size_of::<OriginalGgufCatalog>(),
            size_of::<OriginalGgufCatalogError>(),
            size_of::<Result<OriginalGgufCatalog, OriginalGgufCatalogError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Option<WorkingMemoryError>>(),
            size_of::<eredu_checkpoint::gguf_store::GgufCatalogStorageRequest>(),
            size_of::<Option<eredu_checkpoint::gguf_store::GgufCatalogStorageRequest>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(
                requests
                    .other_bytes()
                    .checked_add(shared)
                    .ok_or(WorkingMemoryError::Overflow)?,
                usize::checked_add,
            )
            .and_then(|n| n.checked_add(size_of_val(&controls)))
            .ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
    }
    /// Admit before any catalog allocation, then execute the one built-in
    /// checkpoint worker. No callback or destructor runs with Usage borrowed.
    pub fn compile_gguf_catalog<'a>(
        &self,
        plan: GgufCatalogPlan<'a>,
    ) -> Result<OriginalGgufCatalog, OriginalGgufCatalogError> {
        if let Err(input) = plan.validate_input() {
            return Err(OriginalGgufCatalogError::rejected(plan, Some(input), None));
        }
        let bytes = match Self::gguf_catalog_required_bytes(&plan) {
            Ok(n) => n,
            Err(e) => return Err(OriginalGgufCatalogError::rejected(plan, None, Some(e))),
        };
        let allowance = match self.admit_source_compiler(bytes) {
            Ok(a) => a,
            Err(e) => return Err(OriginalGgufCatalogError::rejected(plan, None, Some(e))),
        };
        let account = allowance.into_prepared_native_account();
        let result = plan.compile(CatalogCustody(account.share()));
        let accounting = account.finish().err();
        match result {
            Ok(catalog) if accounting.is_none() => Ok(OriginalGgufCatalog { catalog, account }),
            Ok(catalog) => Err(OriginalGgufCatalogError {
                input: None,
                accounting,
                compilation: None,
                checkpoint: None,
                completed: Some(catalog),
                account: Some(account),
            }),
            Err(error) => Err(OriginalGgufCatalogError {
                input: None,
                accounting,
                compilation: Some(error),
                checkpoint: None,
                completed: None,
                account: Some(account),
            }),
        }
    }
}

impl WorkingMemoryPool {
    /// Recognize only a catalog born through this module's fixed compiler and
    /// validate its original pool. Ordinary/arbitrary generic custody is unknown.
    pub fn validate_gguf_catalog_source(
        &self,
        source: &eredu_checkpoint::gguf_store::GgufWeightStore,
    ) -> Result<(), WorkingMemoryError> {
        let origin = source
            .catalog_control_owner::<CatalogCustody>()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if origin.0.matches_pool(self) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl super::WorkingMemoryReservation {
    /// Owning fixed transports of the built-in catalog-domain observation.
    /// There is no amount input or allocation/authority output.
    pub fn gguf_catalog_validation_control_bytes() -> Option<usize> {
        [
            size_of::<&Self>(),
            size_of::<&eredu_checkpoint::gguf_store::GgufConversionPlan>(),
            size_of::<Option<&CatalogCustody>>(),
            size_of::<&CatalogCustody>(),
            size_of::<&WorkingMemoryPool>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<WorkingMemoryError>(),
            size_of::<bool>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// The retained physical plan's built-in source supplies its actual catalog;
    /// no arbitrary source callback or external origin assertion participates.
    pub fn validate_gguf_catalog_plan(
        &self,
        plan: &eredu_checkpoint::gguf_store::GgufConversionPlan,
    ) -> Result<(), WorkingMemoryError> {
        let origin = plan
            .catalog_control_owner::<CatalogCustody>()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if origin.0.matches_pool(&self.0.pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}

#[cfg(test)]
mod tests;
