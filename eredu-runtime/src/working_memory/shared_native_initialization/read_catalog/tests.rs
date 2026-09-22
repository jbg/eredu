use super::*;
use eredu_checkpoint::{
    StoredDtype,
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeInferenceInput, RecipeInferencePlan},
    store::{TensorMetadata, TensorSelection},
};

fn tensors() -> [TensorMetadata; 3] {
    [("雪", 2), ("", 5), ("雪", 7)].map(|(name, width)| TensorMetadata {
        name: name.into(),
        logical_shape: vec![width],
        physical_shape: vec![width],
        stored_dtype: StoredDtype::U8,
        encoded_byte_len: width as u64,
        backing_shard: None,
    })
}
fn required(plan: &ReadBatchCatalogPlan<'_>) -> Option<u64> {
    let result = MemoryLedger::shared_native_initialization_required_bytes(plan);
    if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => None,
        Err(cause) => panic!("{cause}"),
    }
}

#[test]
fn catalog_admission_preserves_borrowed_identity_and_ordinary_inference() {
    let tensors = tensors();
    let plan = || ReadBatchCatalogPlan::new(&tensors).unwrap();
    let Some(bytes) = required(&plan()) else {
        return;
    };
    let short = crate::working_memory::memory_fixture::host_ledger(bytes - 1, 0).unwrap();
    let error = short.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert!(error.rejected_plan().is_some());
    assert!(error.constructor_failure().is_none());
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    drop(error);
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let exclusion = pool.acquire_unquoted().unwrap();
    let error = pool.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    drop((error, exclusion));
    let catalog = pool.initialize_shared_native(plan()).unwrap();
    assert_eq!(catalog.original_bytes(), bytes);
    assert!(matches!(
        catalog.validate_pool(&short),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(std::ptr::eq(
        catalog.output().tensor_metadata_borrowed("雪").unwrap(),
        &tensors[2]
    ));
    assert_eq!(
        catalog
            .output()
            .tensor_metadata_borrowed("")
            .unwrap()
            .logical_shape,
        [5]
    );
    assert!(
        catalog
            .output()
            .tensor_metadata_borrowed("missing")
            .is_none()
    );
    let competitor = pool.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        competitor.accounting_failure(),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    drop(competitor);
    let recipe = DerivedWeightRecipe::source("雪", TensorSelection::Full);
    let inferred =
        RecipeInferencePlan::new(RecipeInferenceInput::Derived(&recipe), catalog.output())
            .unwrap()
            .infer()
            .unwrap();
    assert_eq!(inferred.shape, [7]);
    assert_eq!(inferred.byte_len, 7);
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    drop(catalog);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    drop(tensors);
    assert_eq!(inferred.shape, [7]);
}

#[derive(Debug, thiserror::Error)]
enum Failure<'a> {
    #[error("catalog construction: {0}")]
    Build(#[from] ReadBatchCatalogBuildError<SharedNativeInitializationCustody>),
    #[error("later producer failure")]
    Later(ReadBatchCatalog<'a, SharedNativeInitializationCustody>),
}
struct Later<'a>(ReadBatchCatalogPlan<'a>);
impl<'a> SharedNativeInitializer for Later<'a> {
    type Output = ReadBatchCatalog<'a, SharedNativeInitializationCustody>;
    type Error = Failure<'a>;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        self.0
            .required_storage_bytes()?
            .checked_add(size_of::<Failure<'a>>())
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        Err(Failure::Later(self.0.initialize(custody)?))
    }
}

#[test]
fn later_failure_keeps_index_custody_and_borrowed_metadata() {
    let tensors = tensors();
    let plan = ReadBatchCatalogPlan::new(&tensors).unwrap();
    let Some(_) = required(&plan) else { return };
    let plan = Later(plan);
    let bytes = MemoryLedger::shared_native_initialization_required_bytes(&plan).unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let (uncalled, failure) = pool
        .initialize_shared_native(plan)
        .unwrap_err()
        .into_parts();
    assert!(uncalled.is_none());
    assert!(failure.accounting_failure().is_none());
    let Failure::Later(catalog) = failure.constructor_failure().unwrap() else {
        panic!("expected completed catalog")
    };
    assert!(std::ptr::eq(
        catalog.tensor_metadata_borrowed("雪").unwrap(),
        &tensors[2]
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    drop(failure);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
