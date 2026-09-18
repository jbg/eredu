use super::*;
use crate::backend::runtime::execution::generic::MlxParameterExclusions;
use std::{convert::Infallible, sync::atomic::{AtomicUsize, Ordering}};

struct Constructor {
    names: Vec<String>,
    calls: Arc<AtomicUsize>,
}
impl SharedNativeInitializer for Constructor {
    type Output = MlxParameterExclusions;
    type Error = Infallible;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        MlxParameterExclusions::construction_bytes(&self.names)?
            .checked_add(usize::try_from(ManagerCustody::storage_bytes()?)
                .map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(self, custody: SharedNativeInitializationCustody) -> Result<Self::Output, Infallible> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(MlxParameterExclusions::construct(&self.names, ManagerCustody::new(custody)))
    }
}
fn constructor(calls: &Arc<AtomicUsize>) -> Constructor {
    Constructor { names: vec!["a.independent.bank".into(), "z.shared.bank".into()], calls: calls.clone() }
}

#[test]
fn exact_exclusion_source_refuses_before_birth_and_keeps_escaped_account() {
    let calls = Arc::new(AtomicUsize::new(0));
    let bytes = WorkingMemoryPool::shared_native_initialization_required_bytes(&constructor(&calls)).unwrap();
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let failure = short.initialize_shared_native(constructor(&calls)).unwrap_err();
    assert!(matches!(failure.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { .. })));
    assert!(failure.rejected_plan().is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(short.used_bytes().unwrap(), 0);
    drop(failure);

    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let ready = pool.initialize_shared_native(constructor(&calls)).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(ready.original_bytes(), bytes);
    let selected = constructor(&calls).names.into_iter().collect();
    let native = ready.output().for_selection(&selected).unwrap();
    let cold = native.clone();
    assert!(native.contains("a.independent.bank"));
    assert!(!native.contains("a.missing.bank"));
    assert_eq!(native.unfunded_retained_bytes(), Some(0), "source backing is not charged again by each quote");
    let foreign = ["a.independent.bank".to_owned(), "z.foreign.bank".to_owned()].into();
    assert!(matches!(native.for_selection(&foreign), Err(WorkingMemoryError::IdentityMismatch)));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(ready);
    drop(native);
    assert_eq!(pool.used_bytes().unwrap(), bytes, "cold alias outlives native owner without releasing its payer");
    assert!(cold.contains("z.shared.bank"));
    drop(cold);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unpriced_nonempty_exclusions_cannot_qualify_an_original_binding() {
    let selected = ["independent.bank".to_owned()].into();
    let ordinary = MlxParameterExclusions::ordinary(selected);
    assert!(ordinary.contains("independent.bank"));
    assert!(ordinary.unfunded_retained_bytes().unwrap() > 0);
    assert!(matches!(ordinary.for_selection(&["independent.bank".to_owned()].into()),
        Err(WorkingMemoryError::IdentityMismatch)));
    let empty = MlxParameterExclusions::ordinary(BTreeSet::new());
    assert!(empty.is_source_funded());
    assert!(empty.names().is_empty());
    assert_eq!(empty.unfunded_retained_bytes(), Some(0));
}
