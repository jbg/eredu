use super::*;
use crate::{
    recipe::DerivedWeightRecipe,
    store::{MemoryWeightStore, TensorSelection},
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Custody(Arc<AtomicUsize>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn mixed_custody_views_keep_exact_source_identity_and_detached_lifetimes() {
    let values = [13u32, 29, 47, 61];
    let source = || {
        MemoryWeightStore::from_safetensors([(
            "weights".into(),
            safetensors::Dtype::U32,
            vec![4],
            values.into_iter().flat_map(u32::to_le_bytes).collect(),
        )])
        .unwrap()
    };
    let store = source();
    let recipe = DerivedWeightRecipe::source(
        "weights",
        TensorSelection::Indices {
            axis: 0,
            indices: vec![3, 1, 3],
        },
    );
    let first = recipe.prepare_encoded_read(&store).unwrap().unwrap();
    let plain = recipe.prepare_encoded_read(&store).unwrap().unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let held = EncodedRecipeRead {
        output: first.output,
        batch: first.batch,
        _custody: Custody(drops.clone()),
    };
    let views = [held.borrowed(), plain.borrowed()];
    assert_eq!(views[0].output(), held.output());
    assert_eq!(views[0].sources(), held.sources());
    let plan = EncodedRecipeReadView::prepare_detached(views.into_iter()).unwrap();
    let scratch = plan.read_layout::<Arc<()>>().unwrap().required_bytes();
    let detached_custody = Arc::new(());
    let weak = Arc::downgrade(&detached_custody);
    let detached = plan.construct(detached_custody).unwrap();
    assert!(scratch >= detached.read_layout().unwrap().required_bytes());
    assert!(detached.matches_read(0, &held));
    assert!(detached.matches_read_view(1, plain.borrowed()));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let other_store = source();
    let other = recipe.prepare_encoded_read(&other_store).unwrap().unwrap();
    assert_eq!(other.output(), held.output());
    assert!(!detached.matches_read_view(0, other.borrowed()));
    drop((held, plain, store));
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    let mut first = [0u8; 12];
    let mut second = [0u8; 12];
    detached
        .read_many_into(&mut [&mut first, &mut second])
        .unwrap();
    let expected: Vec<_> = [61u32, 29, 61]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    assert_eq!(first.as_slice(), expected);
    assert_eq!(second, first);
    let mut short = [99u8; 11];
    let error = detached
        .read_many_into(&mut [&mut short, &mut second])
        .unwrap_err();
    assert_eq!(short, [99; 11]);
    drop(detached);
    assert!(weak.upgrade().is_some());
    drop(error);
    assert!(weak.upgrade().is_none());
}
