use super::*;
use eredu_core::load_model;
use eredu_runtime::working_memory::WorkingMemoryPool;
use std::{sync::mpsc, time::Duration};

#[test]
fn contended_initial_publication_keeps_actual_loading_owner_and_source_storage() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let backend = crate::backend::MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let prepared = load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let owner = prepared.memory_owner().unwrap().clone();
    let mut executable = prepared.into_inner().into_executable();
    stream.synchronize().unwrap();
    let before = executable
        .retained_idle_storage(Some(RetainedStorage::default()))
        .unwrap();
    let bytes = before.nonstate_bytes().unwrap().unwrap();
    assert!(bytes > 0);
    assert!(before.has_empty_decoder_storage().unwrap());
    let source_allocations = before.nonstate.array_allocation_facts();
    assert!(!source_allocations.is_empty());
    let positions = executable.erased().state_snapshot();
    let charged = pool.used_bytes().unwrap();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _runtime = safemlx::RuntimeCallDeadline::new(Duration::from_secs(10))
            .unwrap()
            .enter()
            .unwrap();
        entered_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let started = std::time::Instant::now();
    let result = executable.publish_initial_idle_storage(RetainedStorage::default(), &owner);
    let elapsed = started.elapsed();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(!result.unwrap());
    assert!(elapsed < Duration::from_secs(1));
    assert!(executable
        ._memory_owner
        .as_ref()
        .unwrap()
        .same_authority(&owner));
    assert_eq!(pool.used_bytes().unwrap(), charged);
    assert_eq!(executable.erased().state_snapshot(), positions);
    drop(owner);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    // Retained source facts and the same owner remain valid after contention.
    let after = executable
        .retained_idle_storage(Some(RetainedStorage::default()))
        .unwrap();
    assert_eq!(after.nonstate_bytes().unwrap(), Some(bytes));
    assert!(after.has_empty_decoder_storage().unwrap());
    assert_eq!(after.nonstate.array_allocation_facts(), source_allocations);
    let owner = executable._memory_owner.as_ref().unwrap().clone();
    assert!(executable
        .publish_initial_idle_storage(RetainedStorage::default(), &owner)
        .unwrap());
    assert!(executable._memory_owner.is_none());
    drop(owner);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop((before, after, executable));
    crate::backend::ordinary_retirement::reclaim_all();
    safemlx::reclaim_allocation_owners();
    crate::backend::submission_recovery::wait_for_retirement(|| pool.used_bytes().unwrap() == 0);
}

#[test]
fn initial_publication_only_defers_actual_array_inspection_contention() {
    fn wrapped(source: safemlx::ArrayMetadataError) -> Error {
        crate::backend::runtime::residency::manager::ResidencyError::Mlx {
            id: eredu_core::residency::OffloadUnitId::new("retained-storage").unwrap(),
            operation: "retained array allocation inspection",
            source: Exception::from_source(source),
        }
        .into()
    }
    assert!(initial_publication_inspection_busy(&wrapped(
        safemlx::ArrayMetadataError::RuntimeBusy
    )));
    let actual = wrapped(safemlx::ArrayMetadataError::Native(Exception::from_source(
        std::io::Error::new(std::io::ErrorKind::InvalidData, "actual metadata failure"),
    )));
    assert!(!initial_publication_inspection_busy(&actual));
    let mut source: &(dyn std::error::Error + 'static) = &actual;
    loop {
        if let Some(error) = source.downcast_ref::<std::io::Error>() {
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            break;
        }
        source = source.source().expect("preserve actual metadata cause");
    }
    let same_words = Error::Exception(Exception::custom(
        "native runtime is busy during array metadata inspection",
    ));
    assert!(!initial_publication_inspection_busy(&same_words));
}

#[test]
fn optional_publication_recognizes_actual_wrapped_busy_but_keeps_native_boundary() {
    use std::convert::Infallible;
    type SessionError = eredu_runtime::replicated_session::ReplicatedTextSessionError<
        Infallible,
        Infallible,
        Error,
    >;
    fn wrapped(cause: safemlx::ArrayMetadataError) -> Error {
        let native = crate::backend::runtime::residency::manager::ResidencyError::Mlx {
            id: eredu_core::residency::OffloadUnitId::new("retained-storage").unwrap(),
            operation: "retained array allocation inspection",
            source: Exception::from_source(cause),
        };
        Error::Other(Box::new(SessionError::Mechanism(native.into())))
    }
    let busy = wrapped(safemlx::ArrayMetadataError::RuntimeBusy);
    assert!(initial_publication_inspection_busy(&busy));
    // A failed native query must not become a deferrable Busy just because an
    // inner exception retains a metadata error as its original source.
    let native = wrapped(safemlx::ArrayMetadataError::Native(Exception::from_source(
        safemlx::ArrayMetadataError::RuntimeBusy,
    )));
    assert!(!initial_publication_inspection_busy(&native));
    let mut source: &(dyn std::error::Error + 'static) = &native;
    loop {
        if let Some(safemlx::ArrayMetadataError::Native(actual)) = source.downcast_ref() {
            assert!(matches!(
                std::error::Error::source(actual).unwrap().downcast_ref(),
                Some(safemlx::ArrayMetadataError::RuntimeBusy)
            ));
            break;
        }
        source = source
            .source()
            .expect("same actual native query source is retained");
    }
}
