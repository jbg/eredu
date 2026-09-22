use super::*;
use eredu_core::{
    cache::{MutableStateResidency, StateTensorDtype, StateTensorPolicy},
    LayerSchedule, SharedStorageAccountingId,
};
use eredu_runtime::{HostSlotAttachmentError, HostSlotMetadata};
use safemlx::{Device, DeviceType};
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

const ROLE: StateTensorRole = StateTensorRole::Recurrent;

fn state() -> MlxHybridState {
    let fixed = LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new(
        ROLE,
        vec![StateTensorDimension::fixed(2).unwrap()],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap()])
    .unwrap();
    let layout = StateLayout::new(
        LayerSchedule::new(3, vec![fixed.clone(), LayerCachePolicy::NoState, fixed]).unwrap(),
    )
    .unwrap();
    let mut state = MlxHybridState::device_with_global_layer_start(layout, 9).unwrap();
    *state.layers_mut()[0].fixed_component(ROLE).unwrap() = Some(MlxTensor::from_array(
        Array::from_slice(&[3.25f32, 7.5], &[2]),
    ));
    state.layers_mut()[0].advance_fixed(5).unwrap();
    state
}

fn array(state: &MlxHybridState) -> &Array {
    state.layers.slots()[0]
        .fixed
        .values()
        .next()
        .unwrap()
        .as_ref()
        .unwrap()
        .as_array()
}

fn identities(state: &MlxHybridState) -> Vec<HostSlotMetadata> {
    std::iter::once(state.layer_slot_metadata())
        .chain(state.fixed_slot_metadata())
        .cloned()
        .collect()
}

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn layer_clone_owns_new_boxes_and_restore_reuses_the_existing_tables() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut source = state();
    let original = identities(&source);
    assert_eq!(original.len(), 4);
    assert_eq!(
        original[0].capacity_bytes(),
        Some(3 * std::mem::size_of::<MlxHybridLayerState>() as u64)
    );
    assert_eq!(original[2].capacity_bytes(), Some(0));
    let checkpoint = source.clone();
    assert!(original
        .iter()
        .zip(identities(&checkpoint))
        .all(|(a, b)| !a.same_storage(&b)));
    assert_eq!(checkpoint.global_layer_start, 9);
    array(&source).evaluated().unwrap();
    array(&checkpoint).evaluated().unwrap();
    assert_eq!(
        array(&source)
            .allocation_info()
            .unwrap()
            .unwrap()
            .identity(),
        array(&checkpoint)
            .allocation_info()
            .unwrap()
            .unwrap()
            .identity()
    );
    source.clear().unwrap();
    assert!(source.retained_arrays().is_empty());
    assert_eq!(checkpoint.layer_positions().collect::<Vec<_>>(), [5, 0, 0]);
    source.restore_checkpoint(&checkpoint, &stream).unwrap();
    assert!(original
        .iter()
        .zip(identities(&source))
        .all(|(a, b)| a.same_storage(&b)));
    assert_eq!(source.layer_positions().collect::<Vec<_>>(), [5, 0, 0]);
    assert_eq!(
        array(&source).evaluated().unwrap().as_slice::<f32>(),
        &[3.25, 7.5]
    );
    // The previous derived Clone used whole-state replacement for clone_from.
    source.clone_from(&checkpoint);
    assert!(original
        .iter()
        .zip(identities(&source))
        .all(|(a, b)| !a.same_storage(&b)));
    for token in original {
        assert!(matches!(
            token
                .try_attach::<Infallible>(&SharedStorageAccountingId::default(), || unreachable!()),
            Err(HostSlotAttachmentError::Retired)
        ));
    }
}

#[test]
fn escaped_layer_tokens_retain_custody_without_retaining_layer_payloads() {
    let source = state();
    let copy = source.clone();
    // Aliases captured before registration share later attachments.
    let tokens = identities(&source);
    let domain = SharedStorageAccountingId::default();
    let retired = Arc::new(AtomicUsize::new(0));
    for token in std::iter::once(source.layer_slot_metadata()).chain(source.fixed_slot_metadata()) {
        assert!(token
            .try_attach::<Infallible>(&domain, || Ok(Box::new(Retired(retired.clone()))))
            .unwrap());
    }
    drop(source);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    for token in &tokens {
        assert!(matches!(
            token.try_attach::<Infallible>(&domain, || unreachable!()),
            Err(HostSlotAttachmentError::Retired)
        ));
    }
    // Independent source boxes have retired even though the clone shares native
    // tensor handles. Those handles are not payloads of the escaped tokens.
    assert_eq!(
        array(&copy).evaluated().unwrap().as_slice::<f32>(),
        &[3.25, 7.5]
    );
    drop(tokens);
    assert_eq!(retired.load(Ordering::SeqCst), 4);
    assert_eq!(copy.layer_positions().collect::<Vec<_>>(), [5, 0, 0]);
    drop(copy);
    assert_eq!(retired.load(Ordering::SeqCst), 4);
}
