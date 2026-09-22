use super::*;
use eredu_core::{AttentionPolicy, LayerSchedule, SharedStorageAccountingId};
use eredu_nn::workspace::WorkspaceBackend;
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn shared_layout_construction_preserves_published_owner_and_independent_layers() {
    let layout = StateLayout::new(
        LayerSchedule::new(
            2,
            vec![
                LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap(),
                LayerCachePolicy::key_value(AttentionPolicy::sliding(7).unwrap(), 1, 4).unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let mut original = DeviceState::<WorkspaceBackend, Vec<u32>>::create(layout, |index, _| {
        Ok::<_, Infallible>(vec![index as u32 + 3])
    })
    .unwrap();
    let owner = original.layout.as_ref().unwrap().clone();
    let retired = Arc::new(AtomicUsize::new(0));
    owner
        .try_attach(&SharedStorageAccountingId::default(), || {
            Ok::<_, Infallible>(Box::new(Retired(retired.clone())))
        })
        .unwrap();
    let mut visited = Vec::new();
    let copy = DeviceState::<WorkspaceBackend, Vec<u32>>::create_with_shared_layout(
        owner.clone(),
        |index, policy| {
            visited.push(index);
            assert_eq!(policy, owner.layout().layers().get(index).unwrap());
            Ok::<_, Infallible>(original.as_ref()[index].clone())
        },
    )
    .unwrap();
    assert_eq!(visited, [0, 1]);
    assert!(owner.same_storage(copy.layout.as_ref().unwrap()));
    assert!(std::ptr::eq(
        owner.layout(),
        copy.layout.as_ref().unwrap().layout()
    ));
    original.as_mut()[0].push(19);
    assert_eq!(copy.as_ref(), [vec![3], vec![4]]);
    drop((owner, original));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(copy);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn failed_shared_layout_construction_keeps_existing_owner_and_typed_failure() {
    let layout = SharedStateLayout::new(
        StateLayout::new(LayerSchedule::new(2, vec![LayerCachePolicy::NoState; 2]).unwrap())
            .unwrap(),
    );
    let retired = Arc::new(AtomicUsize::new(0));
    layout
        .try_attach(&SharedStorageAccountingId::default(), || {
            Ok::<_, Infallible>(Box::new(Retired(retired.clone())))
        })
        .unwrap();
    #[derive(Debug, PartialEq)]
    struct Rejected(usize);
    let mut visits = 0;
    let result = DeviceState::<WorkspaceBackend, usize>::create_with_shared_layout(
        layout.clone(),
        |index, _| {
            visits += 1;
            if index == 1 {
                Err(Rejected(index))
            } else {
                Ok(index + 7)
            }
        },
    );
    assert_eq!(result.unwrap_err(), Rejected(1));
    assert_eq!(visits, 2);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(layout);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn device_layer_tables_keep_independent_identity_and_reuse_only_the_same_extent() {
    fn state(count: usize, seed: u32) -> DeviceState<WorkspaceBackend, u32> {
        let layout = StateLayout::new(
            LayerSchedule::new(count, vec![LayerCachePolicy::NoState; count]).unwrap(),
        )
        .unwrap();
        DeviceState::create(layout, |index, _| Ok::<_, Infallible>(seed + index as u32)).unwrap()
    }
    let original = state(2, 3);
    let mut copy = original.clone();
    let source_token = original.layer_slot_metadata().unwrap().clone();
    let copied_token = copy.layer_slot_metadata().unwrap().clone();
    assert!(!source_token.same_storage(&copied_token));
    assert_eq!(
        source_token.capacity_bytes(),
        Some(2 * std::mem::size_of::<u32>() as u64)
    );
    copy.as_mut()[0] = 91;
    assert_eq!(original.as_ref(), [3, 4]);
    assert_eq!(copy.as_ref(), [91, 4]);
    copy.clone_from(&state(2, 11));
    assert!(copied_token.same_storage(copy.layer_slot_metadata().unwrap()));
    assert_eq!(copy.as_ref(), [11, 12]);
    copy.clone_from(&state(3, 21));
    assert!(!copied_token.same_storage(copy.layer_slot_metadata().unwrap()));
    assert_eq!(copy.as_ref(), [21, 22, 23]);
    assert!(matches!(
        copied_token.try_attach(&SharedStorageAccountingId::default(), || Ok::<
            Box<dyn Send + Sync>,
            Infallible,
        >(
            Box::new(())
        )),
        Err(crate::HostSlotAttachmentError::Retired)
    ));
    const EMPTY: DeviceState<WorkspaceBackend, u32> = DeviceState::stateless();
    assert!(EMPTY.layer_slot_metadata().is_none());
    copy.clone_from(&EMPTY);
    assert!(copy.as_ref().is_empty());
    assert!(copy.layer_slot_metadata().is_none());
}

#[test]
fn device_layer_token_retains_only_custody_after_actual_elements_retire() {
    struct Payload(Arc<AtomicUsize>);
    impl Drop for Payload {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct Charge {
        payload: Arc<AtomicUsize>,
        retired: Arc<AtomicUsize>,
    }
    impl Drop for Charge {
        fn drop(&mut self) {
            assert_eq!(self.payload.load(Ordering::SeqCst), 2);
            self.retired.fetch_add(1, Ordering::SeqCst);
        }
    }
    let payload = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicUsize::new(0));
    let layout =
        StateLayout::new(LayerSchedule::new(2, vec![LayerCachePolicy::NoState; 2]).unwrap())
            .unwrap();
    let state = DeviceState::<WorkspaceBackend, Payload>::create(layout, |_, _| {
        Ok::<_, Infallible>(Payload(payload.clone()))
    })
    .unwrap();
    let token = state.layer_slot_metadata().unwrap().clone();
    token
        .try_attach(&SharedStorageAccountingId::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Charge {
                payload: payload.clone(),
                retired: retired.clone(),
            }))
        })
        .unwrap();
    drop(state);
    assert_eq!(payload.load(Ordering::SeqCst), 2);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert!(matches!(
        token.try_attach(&SharedStorageAccountingId::default(), || Ok::<
            Box<dyn Send + Sync>,
            Infallible,
        >(
            Box::new(())
        )),
        Err(crate::HostSlotAttachmentError::Retired)
    ));
    drop(token);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn layer_copy_plan_borrows_actual_owner_and_preserves_stateless_absence() {
    struct NoClone(u32);
    let layout =
        StateLayout::new(LayerSchedule::new(2, vec![LayerCachePolicy::NoState; 2]).unwrap())
            .unwrap();
    let state = DeviceState::<WorkspaceBackend, NoClone>::create(layout, |i, _| {
        Ok::<_, Infallible>(NoClone(i as u32 + 13))
    })
    .unwrap();
    let plan = state.prepare_layer_copy_slots().unwrap().unwrap();
    assert!(plan
        .source_metadata()
        .same_storage(state.layer_slot_metadata().unwrap()));
    assert_eq!(plan.len(), 2);
    for index in 0..2 {
        assert!(std::ptr::eq(
            plan.source_at(index).unwrap(),
            &state.as_ref()[index]
        ));
        assert_eq!(plan.source_at(index).unwrap().0, index as u32 + 13);
    }
    assert_eq!(
        plan.retained_bytes(),
        2 * std::mem::size_of::<Option<NoClone>>() as u64
    );
    let stateless = DeviceState::<WorkspaceBackend, NoClone>::stateless();
    assert!(stateless.prepare_layer_copy_slots().unwrap().is_none());
    assert!(stateless.layer_slot_metadata().is_none());
    assert!(stateless.layout.is_none());
}
