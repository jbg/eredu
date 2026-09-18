//! A real completed leaf followed by a host refusal must not quarantine Q.
use super::*;
use crate::backend::{
    MlxAcceleratorFamily, MlxBackend, MlxDeviceIdentity,
    managed_memory::gpu_stream::PreparedExecutionStreams,
    runtime::cache::state::resident_copy::{
        OriginalResidentState, PreparedResidentDecoderCopy, copy_original_resident_state,
    },
};
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{
    HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError,
};

thread_local! {
    static ARM: Cell<bool> = const { Cell::new(false) };
    static REFUSE: Cell<bool> = const { Cell::new(false) };
    static COMPLETED: Cell<usize> = const { Cell::new(0) };
    static REFUSED_BYTES: Cell<usize> = const { Cell::new(0) };
}

// Called only between the actual attention-copy worker and the existing fixed
// child-table reservation. The receipt reads current native completion; it
// neither evaluates a root nor invents a completion result.
pub(in crate::backend::runtime::cache::state::hybrid::resident_grouped_copy) fn before_child_reservation(
    roots: &RefCell<Vec<Array>>,
) {
    if ARM.replace(false) {
        let observer = safemlx::OriginalScopeObserver::require_current().unwrap();
        let roots = roots.borrow();
        let last = roots.last().expect("actual attention copy produced a root");
        last.completed_in_original_scope(&observer).unwrap();
        COMPLETED.set(COMPLETED.get() + 1);
        REFUSE.set(true);
    }
}

#[derive(Debug)]
struct RefuseOnce(HostMetadataFunding);
impl HostMetadataAccount for RefuseOnce {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        if REFUSE.replace(false) {
            REFUSED_BYTES.set(bytes);
            return Err(HostMetadataFundingError::Capacity {
                required: u64::try_from(bytes).unwrap(),
                available: 0,
            });
        }
        self.0.reserve_metadata(bytes)
    }
}
fn attempt_funding(
    pool: &WorkingMemoryPool,
) -> (HostMetadataFunding, HostPreparationAuthority) {
    let real = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), u64::MAX)
        .unwrap();
    let funding = HostMetadataFunding::new(RefuseOnce(real)).unwrap();
    funding
        .reserve_metadata(
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().unwrap(),
        )
        .unwrap();
    let host = HostPreparationAuthority::retain(funding.clone());
    (funding, host)
}
fn identities(state: &MlxHybridState) -> Vec<safemlx::AllocationIdentity> {
    let mut result = Vec::new();
    PreparedHybridGroupedCopy::prepare_fixed(state)
        .unwrap()
        .visit_operands(&mut |array| {
            result.push(array.allocation_info().unwrap().unwrap().identity())
        });
    result
}

#[test]
fn late_host_refusal_retires_completed_copy_and_fresh_attempt_retries() {
    const NAME: &str = "late_host_refusal_retires_completed_copy_and_fresh_attempt_retries";
    if std::env::var("EREDU_ORIGINAL_COPY_DISCARD_CASE").as_deref() != Ok(NAME) {
        let test = format!(
            "backend::runtime::cache::state::hybrid::resident_grouped_copy::tests::original_discard::{NAME}"
        );
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &test, "--nocapture"])
            .env("EREDU_ORIGINAL_COPY_DISCARD_CASE", NAME)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(stdout.contains("ORIGINAL_COPY_DISCARD_OK"), "{stdout}");
        return;
    }

    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let streams = PreparedExecutionStreams::for_factory(&pool)
        .unwrap()
        .expect("original copy test requires qualified fresh native stream owners");
    let identity = MlxDeviceIdentity::from_realized_device(
        &Device::new(DeviceType::Gpu, 0),
        Some(MlxAcceleratorFamily::Metal),
    )
    .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(streams, identity);
    let initialized = safemlx::PrefillRootsRuntime::prepare_for_stream(
        backend.stream(),
        backend.weights_stream(),
    )
    .unwrap();
    let environment = backend.original_copy_environment().unwrap();
    let mechanisms = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let source = source(backend.stream());
    publish(&source, &loading);
    drop(loading);
    backend.synchronize().unwrap();
    settle(&pool, pool.used_bytes().unwrap());
    let original = values(&PreparedHybridGroupedCopy::prepare_fixed(&source).unwrap());
    let original_controls = controls(&PreparedHybridGroupedCopy::prepare_fixed(&source).unwrap());
    let original_ids = identities(&source);
    let baseline = pool.used_bytes().unwrap();

    let (funding, host) = attempt_funding(&pool);
    ARM.set(true);
    let failure = match copy_original_resident_state(
        PreparedResidentDecoderCopy::hybrid_fixed(&source).unwrap(),
        &environment,
        &initialized,
        mechanisms,
        &funding,
        &host,
        u64::MAX,
    ) {
        Err(error) => error,
        Ok(_) => panic!("child table must refuse after a completed attention copy"),
    };
    assert_eq!(COMPLETED.get(), 1);
    assert!(!ARM.get() && !REFUSE.get());
    assert!(REFUSED_BYTES.get() > 0);
    let mut cause: &(dyn std::error::Error + 'static) = &failure;
    let refused = loop {
        if let Some(value) = cause.downcast_ref::<HostMetadataFundingError>() {
            break *value;
        }
        cause = cause
            .source()
            .expect("typed metadata capacity remains in source chain");
    };
    assert_eq!(
        refused,
        HostMetadataFundingError::Capacity {
            required: u64::try_from(REFUSED_BYTES.get()).unwrap(),
            available: 0,
        }
    );
    assert_eq!(identities(&source), original_ids);
    assert_eq!(
        values(&PreparedHybridGroupedCopy::prepare_fixed(&source).unwrap()),
        original
    );
    assert_eq!(
        controls(&PreparedHybridGroupedCopy::prepare_fixed(&source).unwrap()),
        original_controls
    );
    drop((failure, host, funding));
    settle(&pool, baseline);
    assert_eq!(pool.used_bytes().unwrap(), baseline);

    // The failed attempt's host and numerical accounts are gone. A genuinely
    // fresh planning account pays the retry; rollback refunds no earlier work.
    let (funding, host) = attempt_funding(&pool);
    let copied = copy_original_resident_state(
        PreparedResidentDecoderCopy::hybrid_fixed(&source).unwrap(),
        &environment,
        &initialized,
        mechanisms,
        &funding,
        &host,
        u64::MAX,
    )
    .unwrap();
    let OriginalResidentState::Hybrid(copied) = copied else {
        panic!("same native source representation")
    };
    let copied_plan = PreparedHybridGroupedCopy::prepare_fixed(&copied).unwrap();
    assert_eq!(values(&copied_plan), original);
    assert_eq!(controls(&copied_plan), original_controls);
    assert!(
        identities(&copied)
            .iter()
            .all(|id| !original_ids.contains(id))
    );
    assert_eq!(identities(&source), original_ids);
    drop(copied_plan);
    assert!(pool.used_bytes().unwrap() > baseline);
    drop((copied, host, funding));
    settle(&pool, baseline);
    assert_eq!(pool.used_bytes().unwrap(), baseline);
    println!("ORIGINAL_COPY_DISCARD_OK");
}
