use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
#[derive(Debug)]
struct Hold(Arc<AtomicUsize>);
impl Drop for Hold {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn descriptor() -> TensorDescriptor {
    TensorDescriptor {
        name: "weight".into(),
        shape: vec![2],
        dtype: TensorDtype::U8,
        storage: Some(TensorStorage {
            member: "weights".into(),
            offset: 8,
            length: 2,
        }),
    }
}
#[test]
fn clones_share_entries_and_custody_while_serialization_preserves_values() {
    let retired = Arc::new(AtomicUsize::new(0));
    let catalog = TensorCatalog::with_custody([descriptor()], Hold(retired.clone())).unwrap();
    let alias = catalog.clone();
    assert!(std::ptr::eq(
        catalog.get("weight").unwrap(),
        alias.get("weight").unwrap()
    ));
    let expected = serde_json::json!({"tensors":{"weight":{"name":"weight","shape":[2],"dtype":"u8","storage":{"member":"weights","offset":8,"length":2}}}});
    assert_eq!(serde_json::to_value(&catalog).unwrap(), expected);
    let restored: TensorCatalog = serde_json::from_value(expected).unwrap();
    assert_eq!(catalog, restored);
    assert!(!std::ptr::eq(
        catalog.get("weight").unwrap(),
        restored.get("weight").unwrap()
    ));
    drop(catalog);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert_eq!(restored.get("weight").unwrap().shape, [2]);
}
#[test]
fn failed_validation_retires_accepted_custody_and_keeps_typed_errors() {
    let retired = Arc::new(AtomicUsize::new(0));
    assert!(
        matches!(TensorCatalog::with_custody([descriptor(),descriptor()], Hold(retired.clone())), Err(CatalogError::Duplicate(name)) if name=="weight")
    );
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    let mut invalid = descriptor();
    invalid.shape = vec![0];
    assert!(
        matches!(TensorCatalog::with_custody([invalid], Hold(retired.clone())), Err(CatalogError::InvalidShape(name)) if name=="weight")
    );
    assert_eq!(retired.load(Ordering::SeqCst), 2);
}
