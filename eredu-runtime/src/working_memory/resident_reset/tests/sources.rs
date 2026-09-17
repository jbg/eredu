use super::*;

pub(super) fn release_ordinary_source(data: &Rc<RefCell<Data>>) -> (WorkingMemoryPool, u64) {
    let mut data = data.borrow_mut();
    let pool = data.pool.clone();
    let bytes = data.state.layout.capacity_bytes().unwrap();
    data.displaced.take();
    let empty = pool
        .register_storage(std::iter::empty::<(Key, u64)>())
        .unwrap();
    drop(std::mem::replace(&mut data.registration, empty));
    (pool, bytes)
}

#[test]
fn repeated_original_reset_has_exact_overlap_and_retires_each_predecessor_independently() {
    let (mut runtime, data, _, required) = fixture(None, 3);
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    let (pool, layout_bytes) = release_ordinary_source(&data);
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    for _ in 0..4 {
        let previous_required = data.borrow().original_bytes;
        let required = data.borrow().plan().required_bytes();
        let (identity, revision, pointer, before_fills) = {
            let data = data.borrow();
            (
                data.state.layers.metadata().identity().clone(),
                data.state.retention.revision().clone(),
                data.state.layers.slots().as_ptr(),
                FILLS.get(),
            )
        };
        let before = pool.used_bytes().unwrap();
        let failure = runtime
            .reset_admitted(SessionResetLimits::new(before + required - 1))
            .unwrap_err();
        let error = std::error::Error::source(&failure)
            .unwrap()
            .downcast_ref::<ResidentResetError<State>>()
            .unwrap();
        assert!(matches!(
            error.cause,
            ResetCause::Memory(WorkingMemoryError::BudgetExceeded { .. })
        ));
        assert_eq!(error.retained_bytes(), 0);
        assert_eq!(FILLS.get(), before_fills);
        assert_eq!(pool.used_bytes().unwrap(), before);
        assert_eq!(data.borrow().state.layers.slots().as_ptr(), pointer);
        assert_eq!(data.borrow().state.retention.revision(), &revision);
        drop(failure);
        runtime
            .reset_admitted(SessionResetLimits::new(before + required))
            .unwrap();
        assert_eq!(FILLS.get(), before_fills + 3);
        assert_eq!(
            pool.used_bytes().unwrap(),
            layout_bytes + previous_required + required
        );
        assert_ne!(data.borrow().state.layers.metadata().identity(), &identity);
        assert!(data
            .borrow()
            .state
            .layers
            .slots()
            .iter()
            .all(|s| s.position == 0 && s.values == [0; 4]));
        assert_eq!(data.borrow().state.global_start, 5);
        // Actual old table disappears first; escaped identity/revision still
        // retain precisely that predecessor account, independently of the new one.
        data.borrow_mut().displaced.take();
        assert_eq!(
            pool.used_bytes().unwrap(),
            layout_bytes + previous_required + required
        );
        drop(identity);
        assert_eq!(
            pool.used_bytes().unwrap(),
            layout_bytes + previous_required + required
        );
        drop(revision);
        assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    }
    drop(runtime);
    drop(data);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn partial_repeated_reset_retains_only_actual_predecessor_until_error_retires() {
    let (mut runtime, data, _, required) = fixture(None, 3);
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    let (pool, layout_bytes) = release_ordinary_source(&data);
    let next_required = data.borrow().plan().required_bytes();
    FAIL_AT.set(Some(2));
    let failure = runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap_err();
    FAIL_AT.set(None);
    let error = std::error::Error::source(&failure)
        .unwrap()
        .downcast_ref::<ResidentResetError<State>>()
        .unwrap();
    assert!(matches!(error.cause, ResetCause::Allocation(_)));
    assert_eq!(error.initialized_count(), 2);
    assert_eq!(error.retained_bytes(), next_required);
    assert_eq!(
        pool.used_bytes().unwrap(),
        layout_bytes + required + next_required
    );
    drop(runtime);
    drop(data);
    assert_eq!(
        pool.used_bytes().unwrap(),
        layout_bytes + required + next_required
    );
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_source_rejects_foreign_current_session_without_fill() {
    let (mut runtime, data, _, _) = fixture(None, 2);
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    let (pool, _) = release_ordinary_source(&data);
    let (mut other, foreign, _, _) = fixture(None, 2);
    other
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    let before = pool.used_bytes().unwrap();
    let before_fills = FILLS.get();
    data.borrow_mut().foreign_source = Some(foreign);
    let failure = runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap_err();
    let error = std::error::Error::source(&failure)
        .unwrap()
        .downcast_ref::<ResidentResetError<State>>()
        .unwrap();
    assert!(matches!(
        error.cause,
        ResetCause::Memory(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(FILLS.get(), before_fills);
    assert_eq!(pool.used_bytes().unwrap(), before);
    data.borrow_mut().foreign_source = None;
    drop(failure);
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    assert_eq!(FILLS.get(), before_fills + 2);
}

#[test]
fn original_layout_pin_avoids_provider_callbacks_and_destroys_last_key_outside_usage() {
    let (mut runtime, data, _, _) = fixture(None, 2);
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    let (pool, layout_bytes) = release_ordinary_source(&data);
    let required = data.borrow().plan().required_bytes();
    FORBID_KEY_CALLBACKS.set(true);
    KEY_DROP_CHECK.with_borrow_mut(|check| *check = Some((pool.clone(), required, 0)));
    // Failures restore thread-local probes before any ordinary fixture cleanup.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime
            .reset_admitted(SessionResetLimits::new(10_000_000))
            .unwrap();
        data.borrow_mut().displaced.take();
        assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
        drop(runtime);
        drop(data);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }));
    FORBID_KEY_CALLBACKS.set(false);
    let (_, _, drops) = KEY_DROP_CHECK.with_borrow_mut(Option::take).unwrap();
    result.unwrap();
    assert_eq!(
        drops, 1,
        "only the independent canonical layout key survived"
    );
}

#[test]
fn closed_slot_classification_and_ordinary_transport_preserve_original_capacity_once() {
    let (mut runtime, data, _, required) = fixture(None, 2);
    let pool = data.borrow().pool.clone();
    let ordinary = pool
        .classify_host_slot_source(data.borrow().state.layers.metadata())
        .unwrap();
    assert!(ordinary.registered().is_some());
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    let (_, layout_bytes) = release_ordinary_source(&data);
    let metadata = data.borrow().state.layers.metadata().clone();
    let classified = pool.classify_host_slot_source(&metadata).unwrap();
    assert!(classified.registered().is_none());
    let source = classified.into_original().unwrap();
    assert_eq!(source.original_bytes(), required);
    drop(source);
    let unquoted = pool.acquire_unquoted().unwrap();
    let mut sources = UnquotedOriginalSlotSources::prepare(&unquoted);
    sources.push(&metadata).unwrap();
    assert!(matches!(
        sources.push(&metadata),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(sources.sources().len(), 1);
    let foreign = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    let foreign_unquoted = foreign.acquire_unquoted().unwrap();
    let mut foreign_sources = UnquotedOriginalSlotSources::prepare(&foreign_unquoted);
    assert_eq!(foreign_sources.0.as_ref().unwrap().sources.capacity(), 0);
    assert!(matches!(
        foreign_sources.push(&metadata),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(
        foreign_sources.0.as_ref().unwrap().sources.capacity(),
        0,
        "foreign rejection precedes Vec allocation"
    );
    let rejected = foreign
        .pin_registered_storage(std::iter::empty::<(Key, u64)>())
        .unwrap();
    assert!(matches!(
        rejected.with_original_reset_sources(&mut sources),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(
        sources.sources().len(),
        1,
        "rejection preserves caller staging and its lease"
    );
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    let storage = pool
        .pin_registered_storage(std::iter::empty::<(Key, u64)>())
        .unwrap()
        .with_original_reset_sources(&mut sources)
        .unwrap();
    assert!(
        !sources.has_custody(),
        "successful handoff moves the actual lease"
    );
    assert_eq!(storage.bytes(), metadata.capacity_bytes().unwrap());
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    {
        let usage = pool.0.usage.lock().unwrap();
        storage.validate_copy_source(&pool, &usage).unwrap();
    }
    let alias = storage.clone();
    drop(unquoted);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(runtime);
    drop(data);
    assert!(
        pool.classify_host_slot_source(&metadata).is_err(),
        "retired metadata cannot authorize another source use"
    );
    {
        let usage = pool.0.usage.lock().unwrap();
        assert!(storage.validate_copy_source(&pool, &usage).is_err());
    }
    drop(metadata);
    drop(storage);
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(foreign_unquoted);
    assert_eq!(foreign.unquoted_owner_count().unwrap(), 1);
    drop(foreign_sources);
    assert_eq!(foreign.unquoted_owner_count().unwrap(), 0);
}
