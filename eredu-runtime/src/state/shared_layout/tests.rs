use super::*;
use crate::state::{DeviceState, RuntimeLayerState, RuntimeState, StateSegmentLifetime};
use eredu_core::{
    cache::{MutableStateResidency, StateTensorDtype, StateTensorRole},
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceTensor};
use std::{
    convert::Infallible,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

pub(super) struct PayloadRetired(pub Arc<AtomicBool>);

impl Drop for PayloadRetired {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn tensor(slot: u32, shape_capacity: usize) -> StateTensorPolicy {
    let mut shape = Vec::with_capacity(shape_capacity);
    shape.extend([
        StateTensorDimension::Batch,
        StateTensorDimension::fixed(7 + slot as i32).unwrap(),
    ]);
    StateTensorPolicy::new(
        StateTensorRole::Convolution { slot },
        shape,
        StateTensorDtype::Float32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap()
}

fn spare_layout() -> StateLayout {
    let mut fixed = Vec::with_capacity(7);
    fixed.push(tensor(0, 11));
    let mut hybrid = Vec::with_capacity(9);
    hybrid.extend([tensor(0, 13), tensor(1, 17)]);
    let mut key_only = Vec::with_capacity(5);
    key_only.push(tensor(0, 19));
    let layers = LayerSchedule::new(
        4,
        vec![
            LayerCachePolicy::fixed_only(fixed).unwrap(),
            LayerCachePolicy::key_value_with_fixed_state(AttentionPolicy::Full, 2, 8, hybrid)
                .unwrap(),
            LayerCachePolicy::key_only_with_fixed_state(AttentionPolicy::Full, 1, 7, key_only)
                .unwrap(),
            LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 16, 8).unwrap(),
        ],
    )
    .unwrap();
    let mut persistent_name = String::with_capacity(113);
    persistent_name.push_str("persistent");
    let mut frame_name = String::with_capacity(197);
    frame_name.push_str("frame");
    let mut segments = Vec::with_capacity(9);
    segments.extend([
        StateSegmentSpec::new(persistent_name, 0..2, StateSegmentLifetime::Persistent, 0).unwrap(),
        StateSegmentSpec::new(frame_name, 2..4, StateSegmentLifetime::FrameLocal, -1).unwrap(),
    ]);
    let mut layout = StateLayout::segmented(layers, segments).unwrap();
    layout.components.reserve_exact(13);
    for components in &mut layout.components {
        components.reserve_exact(7);
    }
    layout
}

#[test]
fn exact_live_layout_charge_includes_all_nested_spare_capacity() {
    let layout = spare_layout();
    let layer_pointer = layout.layers.iter().next().unwrap() as *const LayerCachePolicy;
    let name_pointer = layout.segments[0].id.0.as_ptr();
    let component_pointer = layout.components[1].as_ptr();
    let mut expected = layout.logical_metadata_bytes().unwrap();
    // Start with the existing logical-content diagnostic and add independently
    // observed spare allocations; a len-based charge cannot satisfy this sum.
    expected += ((layout.components.capacity() - layout.components.len())
        * size_of::<Vec<StateComponentPolicy>>()) as u64;
    expected += ((layout.segments.capacity() - layout.segments.len())
        * size_of::<StateSegmentSpec>()) as u64;
    assert!(layout.components.capacity() > layout.components.len());
    assert!(layout.segments.capacity() > layout.segments.len());
    for layer in layout.layers.iter().take(3) {
        let tensors = match layer {
            LayerCachePolicy::FixedState { tensors }
            | LayerCachePolicy::KeyValueWithFixedState { tensors, .. }
            | LayerCachePolicy::KeyOnlyWithFixedState { tensors, .. } => tensors,
            _ => panic!("fixture must cover every fixed-state container"),
        };
        assert!(tensors.capacity() > tensors.len());
        expected += ((tensors.capacity() - tensors.len()) * size_of::<StateTensorPolicy>()) as u64;
        for tensor in tensors {
            assert!(tensor.shape.capacity() > tensor.shape.len());
            expected += ((tensor.shape.capacity() - tensor.shape.len())
                * size_of::<StateTensorDimension>()) as u64;
        }
    }
    for components in &layout.components {
        assert!(components.capacity() > components.len());
        expected +=
            ((components.capacity() - components.len()) * size_of::<StateComponentPolicy>()) as u64;
        for component in components {
            expected += ((component.shape_capacity() - component.shape().len())
                * size_of::<StateTensorDimension>()) as u64;
        }
    }
    for segment in &layout.segments {
        assert!(segment.id.0.capacity() > segment.id.0.len());
        expected += (segment.id.0.capacity() - segment.id.0.len()) as u64;
    }
    let logical = layout.logical_metadata_bytes().unwrap();
    let shared = SharedStateLayout::new(layout);
    assert_eq!(shared.capacity_bytes(), Some(expected));
    assert!(expected > logical);
    assert_eq!(
        shared.layout().layers.iter().next().unwrap() as *const LayerCachePolicy,
        layer_pointer
    );
    assert_eq!(shared.layout().segments[0].id.0.as_ptr(), name_pointer);
    assert_eq!(shared.layout().components[1].as_ptr(), component_pointer);
    assert_eq!(shared.layout().layer_prefix_offsets(), [0, 0, -1, -1]);
    assert!(std::ptr::eq(shared.layout(), shared.as_ref()));
}

#[test]
fn semantic_equality_and_debug_ignore_owner_identity_and_spare_capacity() {
    let shared = SharedStateLayout::new(spare_layout());
    let alias = shared.clone();
    let distinct = SharedStateLayout::new(shared.layout().clone());
    assert_eq!(shared, alias);
    assert_eq!(shared, distinct);
    assert_eq!(format!("{shared:?}"), format!("{distinct:?}"));
    assert_eq!(shared.identity(), alias.identity());
    assert_ne!(shared.identity(), distinct.identity());
    assert!(shared.same_storage(&alias));
    assert!(!shared.same_storage(&distinct));
    assert!(shared.capacity_bytes().unwrap() > distinct.capacity_bytes().unwrap());
    let changed =
        StateLayout::new(LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap()).unwrap();
    assert_ne!(shared, SharedStateLayout::new(changed));
}

struct RetireCharge {
    payload_retired: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
    reenter: SharedStateLayout,
}

impl Drop for RetireCharge {
    fn drop(&mut self) {
        assert!(self.payload_retired.load(Ordering::SeqCst));
        // The prior owner's custody must not retain any mutex while invoking
        // the provider destructor. Reentry into an independent owner is valid.
        self.reenter
            .try_attach(&SharedStorageAccountingId::default(), || {
                Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(()))
            })
            .unwrap();
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn earlier_aliases_retain_all_domains_until_payload_retires_before_custody() {
    let retired = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut shared = SharedStateLayout::new(spare_layout());
    Arc::get_mut(&mut shared.0).unwrap().payload_retired = Some(PayloadRetired(retired.clone()));
    let alias = shared.clone();
    let identity = shared.identity().clone();
    let domain = SharedStorageAccountingId::default();
    let other = SharedStateLayout::new(spare_layout());
    for domain in [&domain, &SharedStorageAccountingId::default()] {
        assert!(shared
            .try_attach(domain, || {
                Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(RetireCharge {
                    payload_retired: retired.clone(),
                    drops: drops.clone(),
                    reenter: other.clone(),
                }))
            })
            .unwrap());
    }
    assert!(!alias
        .try_attach::<Infallible>(&domain, || panic!("already attached domain"))
        .unwrap());
    drop(shared);
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(alias.identity(), &identity);
    assert_eq!(alias.layout().segments().len(), 2);
    drop(alias);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    // Keeping only the registry key cannot retain payload or attached charges.
    drop(identity);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[derive(Debug, Clone)]
struct Layer {
    value: usize,
    retained: Vec<WorkspaceTensor>,
}

impl RuntimeLayerState<WorkspaceBackend> for Layer {
    type RetainedValues<'a> = std::slice::Iter<'a, WorkspaceTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.retained.iter()
    }
}

#[test]
fn device_state_clone_shares_actual_layout_and_keeps_layer_values_independent() {
    let layout = spare_layout();
    let pointer = layout.components[1].as_ptr();
    let state = DeviceState::<WorkspaceBackend, Layer>::create(layout, |index, _| {
        Ok::<_, Infallible>(Layer {
            value: index + 11,
            retained: Vec::new(),
        })
    })
    .unwrap();
    let owner = state.shared_layout().unwrap();
    assert_eq!(owner.layout().components[1].as_ptr(), pointer);
    assert!(std::ptr::eq(state.layout(), owner.layout()));
    let mut cloned = state.clone();
    assert!(cloned.shared_layout().unwrap().same_storage(owner));
    cloned.as_mut()[1].value = 29;
    assert_eq!(state.as_ref()[1].value, 12);
    assert_eq!(cloned.as_ref()[1].value, 29);
    drop(state);
    assert_eq!(
        cloned.optional_layout().unwrap().layer_prefix_offsets(),
        [0, 0, -1, -1]
    );
    let empty = DeviceState::<WorkspaceBackend, Layer>::stateless();
    assert!(empty.shared_layout().is_none());
    assert!(empty.optional_layout().is_none());
}

#[test]
fn checked_layout_byte_geometry_does_not_wrap() {
    assert_eq!(bytes::<StateTensorDimension>(usize::MAX), None);
    assert_eq!(bytes::<StateSegmentSpec>(usize::MAX), None);
    assert_eq!(bytes::<StateTensorPolicy>(0), Some(0));
}
