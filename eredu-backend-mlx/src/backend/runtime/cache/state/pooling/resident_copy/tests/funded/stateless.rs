use super::*;
use crate::backend::{
    array_copy::IsolatedArrayCopy,
    error::Error,
    runtime::cache::state::{InitializedResidentDecoderCopy, SavedResidentDecoderCopy},
};
use crate::memory_fixture::LedgerFixture;
use eredu_core::TokenFilter;
use eredu_runtime::{
    working_memory::{FundedSamplerCopy, WorkspaceCopyCustody},
    ConfiguredTextSampler, PenaltyConfig, SamplingBackend, TokenDomain,
};

// Portable scalar primitives populate the real run-owned host sampler without
// pretending that this mechanism fixture has a model/native execution grant.
struct Scalar;
impl SamplingBackend for Scalar {
    type Logits = u32;
    type Token = u32;
    type RandomState = ();
    type Context = ();
    type Error = String;
    fn error(message: String) -> String {
        message
    }
    fn validate_token(token: &u32, _: TokenDomain, _: &()) -> Result<u32, String> {
        Ok(*token)
    }
    fn scale_temperature(logits: &u32, _: f32, _: &()) -> Result<u32, String> {
        Ok(*logits)
    }
    fn apply_penalties(logits: &u32, _: &[u32], _: PenaltyConfig, _: &()) -> Result<u32, String> {
        Ok(*logits)
    }
    fn apply_top_k(logits: u32, _: i32, _: &()) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_top_p(logits: u32, _: f32, _: &()) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_min_p(logits: u32, _: f32, _: &()) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_token_filter(logits: &u32, _: &TokenFilter, _: &()) -> Result<u32, String> {
        Ok(*logits)
    }
    fn apply_mirostat(
        logits: &u32,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        _: &(),
    ) -> Result<u32, String> {
        Ok(*logits)
    }
    fn sample_raw(logits: &u32, _: f32, _: Option<&mut ()>, _: &()) -> Result<u32, String> {
        Ok(*logits)
    }
    fn sample_processed(logits: &u32, _: f32, _: Option<&mut ()>, _: &()) -> Result<u32, String> {
        Ok(*logits)
    }
    fn token_id(token: &u32, _: &()) -> Result<u32, String> {
        Ok(*token)
    }
    fn token_probability(_: &u32, _: u32, _: &()) -> Result<f32, String> {
        Ok(0.125)
    }
}
fn grow(sampler: &mut RunOwnedTextSampler) {
    for token in [11, 23] {
        assert_eq!(
            sampler
                .prepare_sample()
                .unwrap()
                .sample::<Scalar>(&token, 0.7, None, &())
                .unwrap(),
            token
        );
    }
}
fn history(sampler: &ConfiguredTextSampler) -> &[u32] {
    match sampler {
        ConfiguredTextSampler::Standard(sampler) => sampler.generated_tokens(),
        ConfiguredTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
}
fn words(array: &Array) -> Vec<u32> {
    array.evaluated().unwrap().try_to_vec::<u32>().unwrap()
}
fn id(array: &Array) -> safemlx::AllocationIdentity {
    array.allocation_info().unwrap().unwrap().identity()
}
fn sources(pool: &MemoryLedger) -> (Array, Array) {
    let loading = NativeMemoryOwner::acquire(pool).unwrap();
    let key = Array::from_slice(&[0x1020_3040u32, 0x5060_7080], &[2]);
    let pending = Array::from_slice(&[23u32], &[]);
    let mut storage = RetainedStorage::default();
    for array in [&key, &pending] {
        array.evaluated().unwrap();
        storage.include_array(array).unwrap();
    }
    drop(storage.publish_unquoted(&loading).unwrap());
    drop(loading);
    (key, pending)
}
fn joined<'a>(
    sampler: BorrowedFundedSampler<'a>,
    key: &Array,
    pending: &Array,
    pool: &MemoryLedger,
) -> (
    RegisteredSamplingCopy<'a, StorageIdentity>,
    eredu_runtime::working_memory::WorkingMemoryStorage<StorageIdentity>,
) {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut projection = ExistingArrayProjection::new(&context);
    let inputs = [&key, &pending]
        .into_iter()
        .map(|array| projection.project(array).unwrap())
        .collect::<Vec<_>>();
    let native = projection.into_storage();
    assert!(native.is_complete());
    let source = RegisteredWorkspaceStorage::bind(
        pool,
        &context,
        native
            .iter()
            .map(|(id, _, root)| crate::backend::nn::workspace::registered_storage_row(id, root)),
    )
    .unwrap();
    let program =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &inputs).unwrap();
    let arrays = RegisteredWorkspaceCopy::bind(program, source).unwrap();
    // Actual absent decoder has neither numerical roots nor host metadata.
    // This is a genuine empty inventory pin, not a fabricated table owner;
    // the operand registrations above still retain the key and pending origins.
    let complete = RetainedStorage::default().pin_registered(pool).unwrap();
    (
        RegisteredSamplingCopy::prepare(sampler, arrays).unwrap(),
        complete,
    )
}
fn memory_error(error: &Error) -> &WorkingMemoryError {
    let mut current: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(error) = current.downcast_ref::<WorkingMemoryError>() {
            return error;
        }
        current = current.source().expect("typed working-memory source");
    }
}
struct Copied {
    native: SavedResidentDecoderCopy,
    sampler: FundedSamplerCopy,
    key: Array,
    pending: Array,
    custody: WorkspaceCopyCustody,
}
fn finish<'a>(
    plan: PreparedResidentDecoderCopy<'a>,
    slots: InitializedResidentDecoderCopy<'a>,
    sampler: FundedSamplerCopy,
    native: eredu_runtime::working_memory::AdmittedWorkspaceCopy,
    key: &Array,
    pending: &Array,
    stream: &Stream,
) -> Copied {
    let (custody, scope) = native.into_parts();
    let roots = RefCell::new(Vec::with_capacity(4));
    let native = plan.copy_retained(slots, stream, &roots).unwrap();
    assert!(roots.borrow().is_empty());
    let key = IsolatedArrayCopy::new(key)
        .copy_retained(stream, &roots)
        .unwrap();
    let pending = IsolatedArrayCopy::new(pending)
        .copy_retained(stream, &roots)
        .unwrap();
    let mut storage = RetainedStorage::default();
    for array in [&key, &pending] {
        array.evaluated().unwrap();
        storage.include_array(array).unwrap();
    }
    drop(storage.publish_funded(&scope).unwrap());
    roots.borrow_mut().clear();
    scope.certify().unwrap();
    Copied {
        native,
        sampler,
        key,
        pending,
        custody,
    }
}
fn assert_absent(native: &SavedResidentDecoderCopy) {
    assert!(native.shared_layout().is_none());
    assert_eq!(native.global_layer_start(), None);
    assert_eq!(native.retained_slot_bytes(), 0);
    assert_eq!(native.protected_slot_bytes(), 0);
    let plan = native.prepare_copy().unwrap();
    assert!(plan.shared_layout().is_none());
    let mut count = 0;
    plan.visit_operands(&mut |_| count += 1).unwrap();
    plan.visit_retained_arrays(&mut |_| count += 1).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn stateless_dispatch_admits_exact_preserves_live_ceiling_and_recopies_saved_absence() {
    for (exact_first, drop_original_first) in [(true, false), (false, false), (false, true)] {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let stream = metal();
        let (key, pending) = sources(&pool);
        let source = MlxPoolingAttentionState::stateless();
        let (mut sampler, preparation, run) = sampler(&pool);
        grow(&mut sampler);
        let plan = PreparedResidentDecoderCopy::pooling(&source).unwrap();
        assert!(plan.shared_layout().is_none());
        assert_eq!(
            plan.host_copy(&pool).unwrap().initialization_peak_bytes(),
            0
        );
        let (sampling, complete) = joined(sampler.borrow_funded(), &key, &pending, &pool);
        let limits = crate::memory_fixture::publication_copy_limits(&pool, 2, u64::MAX);
        let payload = sampling
            .required_bytes()
            .unwrap()
            .checked_add(limits.additional_host_metadata_bytes)
            .unwrap();
        let descriptor = sampling.with_complete_source(complete);
        let required = crate::memory_fixture::host_total(
            &pool
                .sampling_copy_with_source_requirements(&descriptor, &limits)
                .unwrap(),
        );
        drop(descriptor);
        let (sampling, complete) = joined(sampler.borrow_funded(), &key, &pending, &pool);
        let physical_before = pool.fixture_host_current().unwrap();
        let before = (
            pool.fixture_funded_charge().unwrap(),
            pool.fixture_host_peak().unwrap(),
        );
        let error = plan
            .host_copy(&pool)
            .unwrap()
            .admit(
                &pool,
                sampling,
                complete,
                crate::memory_fixture::publication_copy_limits(
                    &pool,
                    2,
                    physical_before.checked_add(required).unwrap() - 1,
                ),
            )
            .err()
            .unwrap();
        assert!(
            matches!(memory_error(&error), WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }) if *required_bytes == required && limit_bytes.checked_sub(*existing_bytes).unwrap() == required - 1)
        );
        assert_eq!(
            (
                pool.fixture_funded_charge().unwrap(),
                pool.fixture_host_peak().unwrap()
            ),
            before
        );
        assert_eq!(history(sampler.as_sampler()), &[11, 23]);
        assert!(source.layer_slot_metadata().is_none());
        let (sampling, complete) = joined(sampler.borrow_funded(), &key, &pending, &pool);
        let (copied_sampler, slots, native) = plan
            .host_copy(&pool)
            .unwrap()
            .admit(
                &pool,
                sampling,
                complete,
                crate::memory_fixture::publication_copy_limits(
                    &pool,
                    2,
                    if exact_first {
                        physical_before.checked_add(required).unwrap()
                    } else {
                        u64::MAX
                    },
                ),
            )
            .unwrap();
        assert_eq!(
            native
                .requirements()
                .get(crate::memory_fixture::topology().host_domain())
                .unwrap()
                .total()
                .unwrap(),
            required
        );
        let first = finish(plan, slots, copied_sampler, native, &key, &pending, &stream);
        assert_absent(&first.native);
        assert_eq!(history(first.sampler.as_sampler()), &[11, 23]);
        assert_eq!(words(&first.key), words(&key));
        assert_eq!(words(&first.pending), vec![23]);
        assert_ne!(id(&first.key), id(&key));
        assert_ne!(id(&first.pending), id(&pending));
        assert_ne!(
            history(first.sampler.as_sampler()).as_ptr(),
            history(sampler.as_sampler()).as_ptr()
        );
        drop((key, pending, sampler, preparation, run, source));
        settle(
            &pool,
            first
                .custody
                .requirements()
                .get(crate::memory_fixture::topology().host_domain())
                .unwrap()
                .total()
                .unwrap()
                .checked_sub(crate::memory_fixture::publication_control_bytes(4))
                .unwrap(),
        );
        let plan = first.native.prepare_copy().unwrap();
        let (sampling, complete) = joined(
            first.sampler.borrow_funded(),
            &first.key,
            &first.pending,
            &pool,
        );
        let limits = crate::memory_fixture::publication_copy_limits(&pool, 2, u64::MAX);
        let next_payload = sampling
            .required_bytes()
            .unwrap()
            .checked_add(limits.additional_host_metadata_bytes)
            .unwrap();
        let descriptor = sampling.with_complete_source(complete);
        let next_required = crate::memory_fixture::host_total(
            &pool
                .sampling_copy_with_source_requirements(&descriptor, &limits)
                .unwrap(),
        );
        drop(descriptor);
        let (sampling, complete) = joined(
            first.sampler.borrow_funded(),
            &first.key,
            &first.pending,
            &pool,
        );
        let copied = plan.host_copy(&pool).unwrap().admit(
            &pool,
            sampling,
            complete,
            crate::memory_fixture::publication_copy_limits(
                &pool,
                2,
                pool.fixture_host_current()
                    .unwrap()
                    .checked_add(next_required)
                    .unwrap(),
            ),
        );
        if exact_first {
            // A retained source's live finite ceiling constrains the next copy.
            // Copy custody does not confer request-ceiling succession authority.
            let error = copied.err().unwrap();
            assert!(matches!(
                memory_error(&error),
                WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
            ));
            drop(error);
            drop(first);
            settle(&pool, 0);
            continue;
        }
        let (sampler, slots, native) = copied.unwrap();
        let second = finish(
            plan,
            slots,
            sampler,
            native,
            &first.key,
            &first.pending,
            &stream,
        );
        assert_absent(&second.native);
        assert_eq!(history(second.sampler.as_sampler()), &[11, 23]);
        assert_eq!(words(&second.key), words(&first.key));
        assert_ne!(id(&second.key), id(&first.key));
        assert_ne!(
            history(second.sampler.as_sampler()).as_ptr(),
            history(first.sampler.as_sampler()).as_ptr()
        );
        let (retained, account_controls) = if drop_original_first {
            drop(first);
            (second, next_required.checked_sub(next_payload).unwrap())
        } else {
            drop(second);
            (first, required.checked_sub(payload).unwrap())
        };
        settle(
            &pool,
            retained
                .custody
                .requirements()
                .get(crate::memory_fixture::topology().host_domain())
                .unwrap()
                .total()
                .unwrap()
                .checked_sub(crate::memory_fixture::publication_control_bytes(4))
                .unwrap(),
        );
        let escaped = retained.key.clone();
        let info = escaped.allocation_info().unwrap().unwrap();
        let escaped_bytes = (info.bytes() as u64)
            .checked_add(info.host_control_bytes() as u64)
            .unwrap();
        drop(retained);
        settle(&pool, escaped_bytes + account_controls);
        assert_eq!(words(&escaped), vec![0x1020_3040, 0x5060_7080]);
        drop(escaped);
        settle(&pool, 0);
    }
}

#[test]
fn different_actual_absence_rejects_before_native_copy_and_releases_admitted_work() {
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let stream = metal();
    let (key, pending) = sources(&pool);
    let source = MlxPoolingAttentionState::stateless();
    let other = MlxPoolingAttentionState::stateless();
    let (sampler, preparation, run) = sampler(&pool);
    let plan = PreparedResidentDecoderCopy::pooling(&source).unwrap();
    let (sampling, complete) = joined(sampler.borrow_funded(), &key, &pending, &pool);
    let (copied_sampler, slots, native) = plan
        .host_copy(&pool)
        .unwrap()
        .admit(
            &pool,
            sampling,
            complete,
            crate::memory_fixture::publication_copy_limits(&pool, 2, u64::MAX),
        )
        .unwrap();
    let (custody, scope) = native.into_parts();
    let roots = RefCell::new(Vec::new());
    let other = PreparedResidentDecoderCopy::pooling(&other).unwrap();
    let error = other.copy_retained(slots, &stream, &roots).err().unwrap();
    assert!(matches!(
        memory_error(&error),
        WorkingMemoryError::IdentityMismatch
    ));
    assert!(roots.borrow().is_empty());
    scope.certify().unwrap();
    drop((
        copied_sampler,
        custody,
        sampler,
        preparation,
        run,
        key,
        pending,
    ));
    settle(&pool, 0);
}

#[derive(Debug, thiserror::Error)]
#[error("injected stateless pair failure after native key copy")]
struct AfterKey;
fn fail_after_key(key: &Array, stream: &Stream, roots: &RefCell<Vec<Array>>) -> Result<(), Error> {
    let _key = IsolatedArrayCopy::new(key).copy_retained(stream, roots)?;
    Err(Error::Other(Box::new(AfterKey)))
}
#[test]
fn stateless_late_failure_keeps_partial_roots_and_account_until_exact_settlement() {
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let stream = metal();
    let (key, pending) = sources(&pool);
    let source = MlxPoolingAttentionState::stateless();
    let (mut sampler, preparation, run) = sampler(&pool);
    grow(&mut sampler);
    let plan = PreparedResidentDecoderCopy::pooling(&source).unwrap();
    let (sampling, complete) = joined(sampler.borrow_funded(), &key, &pending, &pool);
    let (copied_sampler, slots, native) = plan
        .host_copy(&pool)
        .unwrap()
        .admit(
            &pool,
            sampling,
            complete,
            crate::memory_fixture::publication_copy_limits(&pool, 2, u64::MAX),
        )
        .unwrap();
    let (custody, scope) = native.into_parts();
    let roots = RefCell::new(Vec::new());
    let saved = plan.copy_retained(slots, &stream, &roots).unwrap();
    let error = fail_after_key(&key, &stream, &roots).unwrap_err();
    assert!(std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<AfterKey>()
        .is_some());
    assert_eq!(roots.borrow().len(), 2);
    assert_eq!(words(&key), vec![0x1020_3040, 0x5060_7080]);
    assert_eq!(history(sampler.as_sampler()), &[11, 23]);
    let charged = pool.fixture_funded_charge().unwrap();
    drop((saved, copied_sampler));
    assert_eq!(pool.fixture_funded_charge().unwrap(), charged);
    for array in roots.borrow().iter() {
        array.evaluated().unwrap();
    }
    assert_eq!(words(&roots.borrow()[1]), words(&key));
    roots.borrow_mut().clear();
    // Every submitted partial is now settled and retired; failure did not
    // publish a saved destination. Only now can the native scope certify.
    scope.certify().unwrap();
    drop((custody, sampler, preparation, run, key, pending));
    settle(&pool, 0);
}
