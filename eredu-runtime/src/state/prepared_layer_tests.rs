use super::*;
use eredu_nn::workspace::WorkspaceBackend;
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

struct Value(u32, Arc<AtomicUsize>);
impl Drop for Value {
    fn drop(&mut self) {
        self.1.fetch_add(1, Ordering::SeqCst);
    }
}
struct Charge(Arc<AtomicUsize>);
impl Drop for Charge {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn layout() -> SharedStateLayout {
    SharedStateLayout::new(
        StateLayout::new(
            eredu_core::LayerSchedule::new(2, vec![LayerCachePolicy::NoState; 2]).unwrap(),
        )
        .unwrap(),
    )
}

#[test]
fn prepared_layers_move_actual_buffer_and_retire_payload_before_metadata_charge() {
    let payload = Arc::new(AtomicUsize::new(0));
    let charges = Arc::new(AtomicUsize::new(0));
    let table = crate::HostSlotTable::new(
        vec![Value(3, payload.clone()), Value(17, payload.clone())].into_boxed_slice(),
    );
    let pointer = table.slots().as_ptr();
    let token = table.metadata().clone();
    token
        .try_attach(&eredu_core::SharedStorageAccountingId::default(), || {
            Ok::<_, Infallible>(Box::new(Charge(charges.clone())))
        })
        .unwrap();
    let layout = layout();
    let state = DeviceState::<WorkspaceBackend, Value>::from_prepared_layers(layout.clone(), table)
        .unwrap();
    assert_eq!(state.as_ref().as_ptr(), pointer);
    assert_eq!(
        state.as_ref().iter().map(|v| v.0).collect::<Vec<_>>(),
        [3, 17]
    );
    assert!(state.layout.as_ref().unwrap().same_storage(&layout));
    assert!(state.layer_slot_metadata().unwrap().same_storage(&token));
    assert!(state.inference_retention.is_empty());
    drop(state);
    assert_eq!(payload.load(Ordering::SeqCst), 2);
    assert_eq!(charges.load(Ordering::SeqCst), 0);
    drop(token);
    assert_eq!(charges.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_layers_reject_wrong_extent_and_preserve_stateless_absence() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let table = crate::HostSlotTable::new(vec![Value(7, dropped.clone())].into_boxed_slice());
    let error = DeviceState::<WorkspaceBackend, Value>::from_prepared_layers(layout(), table)
        .err()
        .unwrap();
    assert!(matches!(
        error,
        StateError::PreparedLayerCount {
            expected: 2,
            actual: 1
        }
    ));
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    let absent = DeviceState::<WorkspaceBackend, Value>::stateless();
    assert!(absent.layout.is_none());
    assert!(absent.prepare_layer_copy_slots().unwrap().is_none());
    assert!(absent.inference_retention.is_empty());
}
