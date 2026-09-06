#[test]
fn caller_owned_transfer_publishes_only_after_exact_completion() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();

    let mut transfer = manager
        .acquire_many_with_transfer(&[(id("a"), 2)], MemoryTier::Device)
        .unwrap();
    let lease = &transfer.leases()[0];
    {
        let state = manager.lock().unwrap();
        let copy = state
            .control
            .ledger()
            .copy_status(&id("a"), MemoryTier::Device)
            .unwrap()
            .unwrap();
        assert_eq!(copy.pins(), 1);
        assert!(copy.in_flight().is_some());
    }

    let consumer = cpu_stream();
    transfer.order_after(&consumer).unwrap();
    let dependent = lease
        .device_value("weight")
        .unwrap()
        .add(Array::from_int(1), &consumer)
        .unwrap();
    eval([&dependent]).unwrap();
    transfer.synchronize().unwrap();
    {
        let state = manager.lock().unwrap();
        let copy = state
            .control
            .ledger()
            .copy_status(&id("a"), MemoryTier::Device)
            .unwrap()
            .unwrap();
        assert!(copy.in_flight().is_none());
    }
    let ready = manager
        .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
        .unwrap();
    assert!(ready.is_complete().unwrap());
    drop(ready);
    drop(transfer);
    assert_eq!(state(&manager.report().unwrap(), "a").device_pins(), 0);
}

#[test]
fn empty_caller_owned_transfer_is_immediately_complete() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();

    let mut transfer = manager
        .acquire_many_with_transfer(&[], MemoryTier::Device)
        .unwrap();
    assert!(transfer.is_empty());
    assert!(transfer.is_complete().unwrap());
    transfer.order_after(&cpu_stream()).unwrap();
    transfer.synchronize().unwrap();
    assert!(!manager.is_resident(&id("a"), MemoryTier::Device).unwrap());
}

fn contended_transfer_retains_manager_pins(poison: bool) {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    let transfer = manager
        .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
        .unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let holder = std::thread::spawn(move || loop {
        if safemlx::try_with_submission_retirement(|| {
            ready_tx.send(()).unwrap();
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
        })
        .is_some()
        {
            break;
        }
        std::thread::yield_now();
    });
    ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let state = manager.inner.state.lock().unwrap();
    if poison {
        transfer.mark_failed_for_test();
    }
    let started = Instant::now();
    assert!(!transfer.is_complete().unwrap());
    drop(transfer);
    assert!(started.elapsed() < Duration::from_secs(1));
    let copy = state
        .control
        .ledger()
        .copy_status(&id("a"), MemoryTier::Device)
        .unwrap()
        .unwrap();
    assert_eq!(
        copy.pins(),
        1,
        "unknown terminal status must retain application pins"
    );
    assert!(
        copy.in_flight().is_some(),
        "Drop must not publish completion"
    );
    if poison {
        // Known failure rejects new acquisition without needing this held manager lock.
        assert!(manager.acquire(&id("a"), MemoryTier::Device).is_err());
    }
    drop(state);
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    if poison {
        assert!(manager.acquire(&id("a"), MemoryTier::Device).is_err());
    } else {
        // Ordinary unlocked entry advances the exact receipt, then resolves the
        // old generation and releases its pin before acquiring a new lease.
        let lease = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
        assert_eq!(
            manager
                .inner
                .state
                .lock()
                .unwrap()
                .control
                .ledger()
                .copy_status(&id("a"), MemoryTier::Device)
                .unwrap()
                .unwrap()
                .pins(),
            1
        );
        drop(lease);
    }
}

#[test]
fn transfer_poll_and_drop_are_nonblocking_under_runtime_and_manager_locks() {
    contended_transfer_retains_manager_pins(false);
}

#[test]
fn failed_transfer_rejects_acquisition_without_releasing_unproven_generation() {
    contended_transfer_retains_manager_pins(true);
}

#[test]
fn dropping_transfer_retains_queued_consumer_and_publishes_copy() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();

    let transfer = manager
        .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
        .unwrap();
    let consumer = cpu_stream();
    transfer.order_after(&consumer).unwrap();
    let dependent = transfer.leases()[0]
        .device_value("weight")
        .unwrap()
        .add(Array::from_int(1), &consumer)
        .unwrap();
    drop(transfer);
    // Drop no longer synchronizes. An ordinary acquisition advances pending
    // recovery and publishes only after its exact producer/consumer frontier.
    let completed = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
    assert!(manager
        .lock()
        .unwrap()
        .control
        .ledger()
        .copy_status(&id("a"), MemoryTier::Device)
        .unwrap()
        .unwrap()
        .in_flight()
        .is_none());
    assert_eq!(dependent.evaluated().unwrap().as_slice::<i32>(), &[2, 3]);
    drop(completed);
}

#[test]
fn synchronous_acquisition_waits_for_caller_owned_transfer() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();

    let mut transfer = manager
        .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
        .unwrap();
    let waiting_manager = manager.clone();
    let (sender, receiver) = mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let result = waiting_manager
            .acquire(&id("a"), MemoryTier::Device)
            .map(|lease| {
                lease
                    .device_value("weight")
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<i32>()
                    .to_vec()
            });
        sender.send(result).unwrap();
    });
    assert!(matches!(
        receiver.recv_timeout(Duration::from_millis(20)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    transfer.synchronize().unwrap();
    assert_eq!(
        receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap(),
        vec![1, 2]
    );
    waiter.join().unwrap();
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn promotes_host_buffers_to_a_real_metal_stream() {
    let (_dir, store) = fixture_store();
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(8), Some(fixture_binding_capacity()), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Host)],
    )
    .unwrap();
    let manager = ResidencyManager::new(
        store,
        plan,
        [single("a", "a")],
        cpu_stream(),
        Stream::new_with_device(&Device::new(DeviceType::Gpu, 0)),
    )
    .unwrap();
    manager.initialize().unwrap();
    let lease = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
    assert_eq!(
        lease
            .device_value("weight")
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[1, 2]
    );
    assert_eq!(
        manager
            .report()
            .unwrap()
            .offload()
            .transfer(TransferDirection::HostToDevice)
            .count(),
        1
    );
}
