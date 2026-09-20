use super::*;
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};

fn recipe() -> DerivedWeightRecipe {
    DerivedWeightRecipe::Concatenate {
        axis: 0,
        inputs: ["雪", "b", "雪"]
            .map(|key| DerivedWeightRecipe::source(key, TensorSelection::Full))
            .into(),
    }
}
fn required(plan: &EncodedRecipeKeysPlan<'_>) -> Option<u64> {
    let result = WorkingMemoryPool::shared_native_initialization_required_bytes(plan);
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
fn keys_use_exact_original_admission_and_survive_recipe_retirement() {
    let recipe = recipe();
    let plan = || EncodedRecipeKeysPlan::new(&recipe).unwrap().unwrap();
    let Some(bytes) = required(&plan()) else {
        return;
    };
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let error = short.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert!(error.rejected_plan().is_some());
    assert!(error.constructor_failure().is_none());
    assert_eq!(short.used_bytes().unwrap(), 0);
    drop(error);
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let exclusion = pool.acquire_unquoted().unwrap();
    let error = pool.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    drop((error, exclusion));
    let keys = pool.initialize_shared_native(plan()).unwrap();
    assert_eq!(keys.original_bytes(), bytes);
    assert!(matches!(
        keys.validate_pool(&short),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let competitor = pool.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        competitor.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    drop(competitor);
    drop(recipe);
    assert_eq!(keys.output().keys(), ["雪", "b", "雪"]);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(keys);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error("source-key construction: {0}")]
    Build(#[from] EncodedRecipeKeysBuildError<SharedNativeInitializationCustody>),
    #[error("later producer failure")]
    Later(PreparedEncodedRecipeKeys<SharedNativeInitializationCustody>),
}
struct Later<'a>(EncodedRecipeKeysPlan<'a>);
impl SharedNativeInitializer for Later<'_> {
    type Output = PreparedEncodedRecipeKeys<SharedNativeInitializationCustody>;
    type Error = Failure;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        self.0
            .required_storage_bytes()?
            .checked_add(size_of::<Failure>())
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
fn later_failure_keeps_constructed_keys_and_account_after_recipe_is_gone() {
    let recipe = recipe();
    let plan = EncodedRecipeKeysPlan::new(&recipe).unwrap().unwrap();
    let Some(_) = required(&plan) else { return };
    let plan = Later(plan);
    let bytes = WorkingMemoryPool::shared_native_initialization_required_bytes(&plan).unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let (uncalled, failure) = pool
        .initialize_shared_native(plan)
        .unwrap_err()
        .into_parts();
    assert!(uncalled.is_none());
    drop(uncalled);
    drop(recipe);
    assert!(failure.accounting_failure().is_none());
    let Failure::Later(keys) = failure.constructor_failure().unwrap() else {
        panic!("expected completed key prefix")
    };
    assert_eq!(keys.keys(), ["雪", "b", "雪"]);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
