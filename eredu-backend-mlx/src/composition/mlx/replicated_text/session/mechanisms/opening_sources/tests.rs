//! Native source/collector tests. No fabricated canonical context, source stamp,
//! account grant, or claim of SessionPrefill/install integration.
use super::*;
use crate::backend::runtime::cache::kv::KeyValueCache;
use eredu_core::{
    AttentionPolicy, InferenceGeometry, LayerSchedule, OutputDemand, cache::LayerCachePolicy,
};
use eredu_runtime::{StateLayout, working_memory::InferenceExecutionIdentity};
use safemlx::{Device, DeviceType};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}
fn state(stream: &Stream) -> MlxKeyValueState {
    let layout = StateLayout::new(
        LayerSchedule::new(
            2,
            vec![
                LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 4).unwrap(),
                LayerCachePolicy::NoState,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = MlxKeyValueState::device_with_global_layer_start(layout, 17).unwrap();
    KeyValueCache::update_for_attention(
        state.layer(0).unwrap(),
        Array::from_slice(&[1_f32, 3., 5., 7.], &[1, 1, 1, 4]),
        Array::from_slice(&[11_f32, 13., 15., 17.], &[1, 1, 1, 4]),
        stream,
    )
    .unwrap();
    for array in state.retained_arrays() {
        let _ = array.evaluated().unwrap();
    }
    state
}
fn tensor() -> MlxTensor {
    let array = Array::from_slice(&[2_f32, 4., 8., 16.], &[2, 2]);
    let _ = array.evaluated().unwrap();
    MlxTensor::from_array(array)
}
fn parameters(value: &MlxTensor) -> Result<RetainedStorage, Error> {
    let mut storage = RetainedStorage::default();
    storage.include_array(value.as_array())?;
    Ok(storage)
}
thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct Housekeeping;
impl Housekeeping {
    fn start() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.set(0);
        Self
    }
}
impl Drop for Housekeeping {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn actual_nonzero_state_host_and_execution_domains_preserve_aliases_cold() {
    let stream = stream();
    let state = state(&stream);
    let value = tensor();
    let mut expected_state = RetainedStorage::default();
    for array in state.retained_arrays() {
        expected_state.include_array(array).unwrap();
    }
    let expected = parameters(&value).unwrap().array_allocation_facts();
    let narrow = value
        .as_array()
        .try_index_device((..1, ..), &stream)
        .unwrap();
    let _ = narrow.evaluated().unwrap();
    let narrow = MlxTensor::from_array(narrow);
    assert_eq!(narrow.as_array().shape(), &[1, 2]);
    let mut parts = OpeningSourceParts::default();
    let guard = Housekeeping::start();
    parts
        .collect(
            &state,
            || parameters(&value),
            |visitor| {
                visitor(&narrow);
                visitor(&narrow);
                true
            },
        )
        .unwrap();
    assert_eq!(HOUSEKEEPING.get(), 0);
    drop(guard);
    assert!(parts.complete);
    assert_eq!(
        parts.state.as_ref().unwrap().array_allocation_facts(),
        expected_state.array_allocation_facts()
    );
    assert_eq!(
        parts.parameters.as_ref().unwrap().array_allocation_facts(),
        expected
    );
    assert_eq!(parts.execution.array_allocation_facts(), expected);
    let host = parts.host.as_ref().unwrap();
    assert!(
        host.slot_metadata_sources()
            .any(|token| token.same_storage(state.layer_slot_metadata()))
    );
    assert_eq!(host.metadata_sources().count(), 1);
    assert!(host.byte_bound().unwrap().unwrap() > 0);
    assert_eq!(MlxStateMechanisms::offset(&state), 1);
    // Alias overlap between domains is deliberate. No sum of those bounds is
    // advertised as a unique inventory or as remaining-account credit.
}

#[test]
fn false_execution_visitor_keeps_nonempty_prefix_and_incomplete_evidence() {
    let stream = stream();
    let state = state(&stream);
    let value = tensor();
    let mut parts = OpeningSourceParts::default();
    let error = parts
        .collect(
            &state,
            || parameters(&value),
            |visitor| {
                visitor(&value);
                false
            },
        )
        .unwrap_err();
    assert!(
        matches!(error, Error::Other(ref cause) if matches!(cause.downcast_ref::<NativeOpeningSourceError>(), Some(NativeOpeningSourceError::Incomplete)))
    );
    assert!(!parts.complete);
    assert_eq!(parts.execution.array_allocation_facts().len(), 1);
    assert_eq!(parts.execution.byte_bound().unwrap(), None);
    assert!(parts.state.is_some() && parts.host.is_some() && parts.parameters.is_some());
}

#[test]
fn lazy_execution_owner_stays_unknown_after_later_external_evaluation() {
    let stream = stream();
    let state = state(&stream);
    let value = tensor();
    let lazy = MlxTensor::from_array(value.as_array().square(&stream).unwrap());
    let mut parts = OpeningSourceParts::default();
    assert!(
        parts
            .collect(
                &state,
                || parameters(&value),
                |visitor| {
                    visitor(&lazy);
                    true
                }
            )
            .is_err()
    );
    assert_eq!(parts.execution.unknown_arrays().len(), 1);
    assert_eq!(parts.execution.byte_bound().unwrap(), None);
    let _ = lazy.as_array().evaluated().unwrap();
    assert_eq!(parts.execution.byte_bound().unwrap(), None);
    assert!(!parts.complete);
}

#[derive(Debug, thiserror::Error)]
#[error("original parameter collector sentinel")]
struct Sentinel;
#[test]
fn original_parameter_error_keeps_completed_state_domains_and_skips_execution() {
    let stream = stream();
    let state = state(&stream);
    let mut parts = OpeningSourceParts::default();
    let error = parts
        .collect(
            &state,
            || Err(Error::Other(Box::new(Sentinel))),
            |_| panic!("execution after failed parameter inventory"),
        )
        .unwrap_err();
    assert!(matches!(error, Error::Other(ref cause) if cause.is::<Sentinel>()));
    assert_eq!(
        parts.state.as_ref().unwrap().array_allocation_facts().len(),
        2
    );
    assert!(parts.host.is_some());
    assert!(parts.parameters.is_none());
    assert!(!parts.complete);
}

#[test]
fn visitor_unwind_retains_actual_array_prefix_until_owner_moves_outside_loan() {
    let stream = stream();
    let state = state(&stream);
    let value = tensor();
    // Real source collection behind the same RefCell ownership pattern; this
    // does not manufacture a canonical coordinate or test the runtime hook.
    let owner = RefCell::new(OpeningSourceParts::default());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let mut parts = owner.borrow_mut();
            let _ = parts.collect(
                &state,
                || parameters(&value),
                |visitor| {
                    visitor(&value);
                    panic!("after real source prefix");
                },
            );
        }))
        .is_err()
    );
    let parts = owner.into_inner();
    assert!(!parts.complete);
    assert_eq!(parts.execution.array_allocation_facts().len(), 1);
    drop(value);
    drop(state);
    let arrays = parts
        .execution
        .into_retained_arrays()
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(arrays.len(), 1);
    assert_eq!(
        arrays[0].evaluated().unwrap().try_to_vec::<f32>().unwrap(),
        [2., 4., 8., 16.]
    );
}

fn request() -> InferenceRequest {
    crate::memory_fixture::empty_admitted_request(
        &InferenceExecutionIdentity::default(),
        InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 2,
            max_output_tokens: 1,
            prefill_chunk_positions: 1,
            output: OutputDemand::LastPosition,
        },
    )
    .unwrap()
}
#[test]
fn private_guard_is_exclusive_and_drop_closes_without_native_work() {
    let mut binding = NativeOpeningSourceBinding::default();
    assert!(binding.slot.is_none());
    assert!(!binding.is_active());
    let request = request();
    let guard = binding.bind(&request).unwrap();
    assert!(binding.is_active());
    assert!(
        matches!(binding.bind(&request), Err(Error::Other(ref cause)) if matches!(cause.downcast_ref::<NativeOpeningSourceError>(), Some(NativeOpeningSourceError::Pending)))
    );
    let old = guard.slot.clone();
    drop(guard);
    assert!(!binding.is_active());
    let borrowed = old.pending.borrow_mut();
    assert!(
        matches!(binding.bind(&request), Err(Error::Other(ref cause)) if matches!(cause.downcast_ref::<NativeOpeningSourceError>(), Some(NativeOpeningSourceError::Busy)))
    );
    drop(borrowed);
    let replacement = binding.bind(&request).unwrap();
    assert!(!Rc::ptr_eq(&old, &replacement.slot));
    assert!(!old.active.get());
    assert!(NativeOpeningSourceBinding::inline_control_bytes().unwrap() > 0);
    assert!(
        NativeOpeningSourceBinding::active_fixed_control_peak_bytes().unwrap()
            > NativeOpeningSourceBinding::inline_control_bytes().unwrap()
    );
}

#[test]
fn incomplete_manager_domain_does_not_hide_current_execution_roots() {
    let stream = stream();
    let state = state(&stream);
    let value = tensor();
    let mut parts = OpeningSourceParts::default();
    let result = parts.collect(
        &state,
        || {
            let mut storage = parameters(&value)?;
            storage.mark_incomplete();
            Ok(storage)
        },
        |visitor| {
            visitor(&value);
            true
        },
    );
    assert!(result.is_err());
    assert!(!parts.complete);
    assert_eq!(
        parts.parameters.as_ref().unwrap().byte_bound().unwrap(),
        None
    );
    assert_eq!(parts.execution.array_allocation_facts().len(), 1);
}
