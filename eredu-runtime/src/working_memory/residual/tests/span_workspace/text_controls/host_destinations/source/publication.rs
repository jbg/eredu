//! Neutral accounting oracle only; native birth truth is tested by the actual
//! GGUF/native component. This private fixture cannot issue production origins.
use super::*;
use crate::working_memory::{funding, storage::native_publication::PreparedNativePublication};
use crate::working_memory::{
    NativeStorageObservation, NativeStorageRegistration, OriginalHostSourceConstruction,
    OriginalHostSourceFailureCause, OriginalHostSourceReceipt, OriginalTextControlGuard,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

type Registration = NativeStorageRegistration<u32>;
struct Completed {
    attached: RefCell<Option<Registration>>,
    key: u32,
    bytes: u64,
    fail_attach: bool,
    pool: WorkingMemoryPool,
    // Test payload/canonical registration retire before the actual host receipt.
    _receipt: OriginalHostSourceReceipt,
}
impl Drop for Completed {
    fn drop(&mut self) {
        assert!(
            self.pool.0.usage.try_lock().is_ok(),
            "completed provider retired under Usage"
        );
    }
}
struct Producer {
    controls: OriginalTextControlGuard,
    pool: WorkingMemoryPool,
    calls: Rc<Cell<usize>>,
    total: u64,
    key: u32,
    bytes: u64,
    fail_attach: bool,
}
impl OriginalHostSourceConstruction for Producer {
    type Key = u32;
    type Completed = Completed;
    type Output = Completed;
    type Attachment = Registration;
    type Error = WorkingMemoryError;
    type Observation<'a> = &'a Completed;
    fn source_custody(&self) -> crate::working_memory::OriginalHostSourceCustody {
        self.controls.clone().into()
    }
    fn storage_bytes(&self) -> Result<(u64, u64), WorkingMemoryError> {
        Ok((self.total, self.bytes))
    }
    fn create(self, receipt: OriginalHostSourceReceipt) -> Result<Completed, WorkingMemoryError> {
        assert!(self.pool.0.usage.try_lock().is_ok());
        assert!(receipt.belongs_to(&self.controls));
        self.calls.set(self.calls.get() + 1);
        Ok(Completed {
            attached: RefCell::new(None),
            key: self.key,
            bytes: self.bytes,
            fail_attach: self.fail_attach,
            pool: self.pool,
            _receipt: receipt,
        })
    }
    fn observe(value: &Completed) -> Result<&Completed, WorkingMemoryError> {
        assert!(value.pool.0.usage.try_lock().is_ok());
        Ok(value)
    }
    fn describe(value: &&Completed) -> Option<(u32, u64)> {
        Some((value.key, value.bytes))
    }
    fn prepare_attachment(value: Registration) -> Result<Registration, WorkingMemoryError> {
        Ok(value)
    }
    fn registration(value: &Registration) -> &Registration {
        value
    }
    fn attach(
        value: &Completed,
        attachment: Registration,
    ) -> Result<(), (WorkingMemoryError, Registration)> {
        assert!(value.pool.0.usage.try_lock().is_ok());
        if value.fail_attach {
            return Err((WorkingMemoryError::AlreadyStarted, attachment));
        }
        *value.attached.borrow_mut() = Some(attachment);
        Ok(())
    }
    fn into_output(value: Completed) -> Completed {
        value
    }
}
fn producer(
    controls: &OriginalTextControlGuard,
    pool: &WorkingMemoryPool,
    calls: &Rc<Cell<usize>>,
    key: u32,
    fail_attach: bool,
) -> Producer {
    Producer {
        controls: controls.clone(),
        pool: pool.clone(),
        calls: calls.clone(),
        total: total().expect("selected qualified fixture"),
        key,
        bytes: 16,
        fail_attach,
    }
}
fn qualification_unknown() {
    assert_ne!(
        std::env::var("EREDU_REQUIRE_IMMUTABLE_SOURCE_QUALIFICATION").as_deref(),
        Ok("1"),
        "the selected validation environment must exercise positive immutable-source qualification"
    );
}
fn qualified_request(pool: &WorkingMemoryPool, bytes: u64) -> Option<IncrementalInferenceQuote> {
    let request = source_request(pool, bytes, 1, 0);
    if request.is_none() {
        qualification_unknown();
    }
    request
}
fn total() -> Option<u64> {
    // Closed neutral representation plus exact initialized stand-in payload;
    // no native allocator/Graph certificate is inferred from this fixture.
    match crate::working_memory::OriginalHostSourceBank::publication_control_bytes::<Producer>(0) {
        Ok(bytes) => Some(bytes + std::mem::size_of::<Completed>() as u64 + 16),
        Err(WorkingMemoryError::UnknownBound) => {
            qualification_unknown();
            None
        }
        Err(cause) => panic!("unexpected source layout: {cause}"),
    }
}
fn balances(pool: &WorkingMemoryPool) -> (u64, u64) {
    let usage = pool.0.usage.lock().unwrap();
    (usage.reserved, usage.registered)
}

#[test]
fn source_publication_preserves_prepaid_a_origin_through_b_existing_alias() {
    let Some(total) = total() else {
        return;
    };
    for drop_birth_first in [false, true] {
        let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let Some(quote) = qualified_request(&pool, total) else {
            return;
        };
        let (r, run, accepted) = accept(&pool, quote);
        let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
        let protected = span.protected_host_bytes();
        let controls = span.control_guard();
        let mut host = span.take_host_destinations().unwrap().unwrap();
        let mut bank = host.take_source_constructions().unwrap();
        let calls = Rc::new(Cell::new(0));
        let before = balances(&pool);
        let output = bank
            .construct(producer(&controls, &pool, &calls, 2, false))
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(
            balances(&pool),
            before,
            "initial prepaid row must not debit H or charge B twice"
        );
        assert_eq!((bank.remaining_bytes(), bank.remaining_attempts()), (0, 0));
        drop((bank, host, controls, span, r, run, root));
        assert_eq!(pool.used_bytes().unwrap(), protected);
        let (b, br) = funding::tests::reservation(&pool, 200, 10_000_000)
            .into_funding()
            .unwrap();
        let bp = br
            .take_native_partition(funding::native_partition::test_receipt(&br, 100))
            .unwrap();
        let bs = br.scope().unwrap();
        let mut key_only = PreparedNativePublication::prepare_slots(bp.clone(), 1);
        key_only.push_source(&2u32, 16, None, &pool).unwrap();
        let before = balances(&pool);
        assert_eq!(key_only.publish(&bs), Err(WorkingMemoryError::IdentityMismatch));
        assert_eq!(balances(&pool), before);
        assert!(key_only.take_input(0).is_none());
        drop(key_only);
        // Same generation may not cross immutable/mutable coverage kinds, and
        // an unseen immutable generation cannot originate through B.
        for input in [
            NativeStorageObservation::Existing(2u32, 16),
            NativeStorageObservation::ExistingImmutable(3u32, 16),
        ] {
            let mut refused = PreparedNativePublication::prepare_slots(bp.clone(), 1);
            refused.push_observation(input).unwrap();
            let before = balances(&pool);
            assert_eq!(
                refused.publish(&bs),
                Err(WorkingMemoryError::IdentityMismatch)
            );
            assert_eq!(balances(&pool), before);
        }
        let mut publication = PreparedNativePublication::prepare_slots(bp.clone(), 2);
        publication
            .push_observation(NativeStorageObservation::ExistingImmutable(2u32, 16))
            .unwrap();
        publication
            .push_observation(NativeStorageObservation::ExistingImmutable(2u32, 16))
            .unwrap();
        let before = balances(&pool);
        publication.publish(&bs).unwrap();
        assert_eq!(
            balances(&pool),
            before,
            "B recognizes A without another payload debit"
        );
        let alias = publication.take_input(0).unwrap();
        bs.certify().unwrap();
        drop((publication, bp, b, br));
        assert_eq!(pool.used_bytes().unwrap(), protected);
        if drop_birth_first {
            drop(output);
            assert_eq!(pool.used_bytes().unwrap(), protected);
            drop(alias);
        } else {
            drop(alias);
            assert_eq!(pool.used_bytes().unwrap(), protected);
            drop(output);
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}


#[test]
fn immutable_existing_alias_refuses_foreign_pool_and_quarantined_donor_without_credit() {
    let Some(total) = total() else { return; };
    for quarantine in [false, true] {
        let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let Some(quote) = qualified_request(&pool, total) else { return; };
        let (r, run, accepted) = accept(&pool, quote);
        let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
        let controls = span.control_guard();
        let mut host = span.take_host_destinations().unwrap().unwrap();
        let mut bank = host.take_source_constructions().unwrap();
        let calls = Rc::new(Cell::new(0));
        let output = bank.construct(producer(&controls, &pool, &calls, 2, false)).unwrap();
        assert_eq!((calls.get(), bank.remaining_bytes(), bank.remaining_attempts()), (1, 0, 0));
        if quarantine {
            // Real unfinished native custody fences A; no private account mutation.
            drop(run.scope().unwrap());
        }
        drop((bank, host, controls, span, r, run, root));
        let donor = pool.used_bytes().unwrap();
        let target = if quarantine { pool.clone() } else {
            WorkingMemoryPool::new(10_000_000, 0).unwrap()
        };
        let (b, br) = funding::tests::reservation(&target, 200, 10_000_000).into_funding().unwrap();
        let bp = br.take_native_partition(funding::native_partition::test_receipt(&br, 100)).unwrap();
        let bs = br.scope().unwrap();
        let mut attempt = PreparedNativePublication::prepare_slots(bp.clone(), 1);
        attempt.push_observation(NativeStorageObservation::ExistingImmutable(2u32, 16)).unwrap();
        let before = balances(&target);
        assert_eq!(attempt.publish(&bs), Err(if quarantine {
            WorkingMemoryError::ExecutionFenced
        } else {
            WorkingMemoryError::IdentityMismatch
        }));
        assert_eq!(balances(&target), before);
        assert!(attempt.take_input(0).is_none());
        assert_eq!(attempt.publish(&bs), Err(WorkingMemoryError::PreparationAlreadyStarted));
        bs.certify().unwrap();
        drop((attempt, bp, b, br));
        assert_eq!(pool.used_bytes().unwrap(), donor);
        drop(output);
        if quarantine {
            assert!(pool.used_bytes().unwrap() > 0, "failed source cannot refund its account");
        } else {
            assert_eq!(pool.used_bytes().unwrap(), 0);
            assert_eq!(target.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn source_publication_short_debit_never_enters_producer_and_keeps_refused_receipt() {
    let Some(total) = total() else {
        return;
    };
    let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some(quote) = qualified_request(&pool, total - 1) else {
        return;
    };
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let protected = span.protected_host_bytes();
    let controls = span.control_guard();
    let mut host = span.take_host_destinations().unwrap().unwrap();
    let mut bank = host.take_source_constructions().unwrap();
    let calls = Rc::new(Cell::new(0));
    let error = bank
        .construct(producer(&controls, &pool, &calls, 2, false))
        .err()
        .unwrap();
    let (unstarted, retained) = error.into_parts();
    assert!(unstarted.is_some());
    assert!(retained.completed().is_none());
    assert_eq!(calls.get(), 0);
    assert!(
        matches!(retained.cause(), OriginalHostSourceFailureCause::Funding(c) if c.retains_receipt())
    );
    assert!(pool.pin_registered_storage([(2u32, 16)]).is_err());
    drop((unstarted, bank, host, controls, span, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(retained);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_publication_conflicting_row_and_post_commit_failure_retain_truthful_prefix() {
    let Some(total) = total() else {
        return;
    };
    for (key, fail_attach, committed) in [(1u32, false, false), (2, true, true)] {
        let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let Some(quote) = qualified_request(&pool, total) else {
            return;
        };
        let (r, run, accepted) = accept(&pool, quote);
        let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
        let protected = span.protected_host_bytes();
        let controls = span.control_guard();
        let mut host = span.take_host_destinations().unwrap().unwrap();
        let mut bank = host.take_source_constructions().unwrap();
        let calls = Rc::new(Cell::new(0));
        let before = balances(&pool);
        let failure = bank
            .construct(producer(&controls, &pool, &calls, key, fail_attach))
            .err()
            .unwrap();
        let (unstarted, retained) = failure.into_parts();
        assert!(unstarted.is_none());
        assert!(retained.completed().is_some());
        assert_eq!(calls.get(), 1);
        assert_eq!(balances(&pool), before);
        if committed {
            assert!(matches!(
                retained.cause(),
                OriginalHostSourceFailureCause::Native(WorkingMemoryError::AlreadyStarted)
            ));
            let pin = pool.pin_registered_storage([(key, 16)]).unwrap();
            drop(pin);
        } else {
            assert!(matches!(
                retained.cause(),
                OriginalHostSourceFailureCause::Memory(
                    WorkingMemoryError::StorageCapacityMismatch {
                        expected_bytes: 64,
                        actual_bytes: 16
                    }
                )
            ));
        }
        drop((bank, host, controls, span, r, run, root));
        assert_eq!(pool.used_bytes().unwrap(), protected);
        drop(retained);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn immutable_output_sources_publish_independently_and_do_not_refund_host_banks() {
    let Some(total)=total() else { return; };
    let pool=WorkingMemoryPool::new(10_000_000,0).unwrap();
    let root=pool.register_storage([(1u32,64)]).unwrap();
    let output_facts=HostSourceConstructionFacts::new(total*2,2,0).unwrap();
    let host_facts=HostDestinationFacts::new(4,1).unwrap()
        .with_source_constructions(HostSourceConstructionFacts::new(total,1,0).unwrap()).unwrap();
    let quote=replacement_quote(&pool,geometry(),0).into_incremental();
    let controls=PreparedTextControlWorkspace::prepare_controls(geometry(),quote.span_workspace().plan(),super::super::facts()).unwrap()
        .with_host_destinations(host_facts).unwrap().with_output_source_constructions(output_facts).unwrap();
    assert!(matches!(controls.clone().with_output_source_constructions(output_facts),Err(WorkingMemoryError::AlreadyStarted)));
    let quote=quote.with_span_workspace_and_text_controls(controls).unwrap();
    let (r,run,accepted)=accept(&pool,quote);
    let (mut span,_)=accepted.into_funded_text_span_workspace(&run,&r).unwrap();
    let protected=span.protected_host_bytes();let controls=span.control_guard();
    let mut outputs=span.take_output_source_constructions().unwrap().unwrap();
    assert!(matches!(span.take_output_source_constructions(),Err(WorkingMemoryError::AlreadyStarted)));
    let mut host=span.take_host_destinations().unwrap().unwrap();
    let mut cache=host.take_source_constructions().unwrap();let calls=Rc::new(Cell::new(0));
    let before=balances(&pool);
    let output=outputs.construct(producer(&controls,&pool,&calls,2,false)).unwrap();
    assert_eq!(balances(&pool),before);
    assert_eq!((outputs.remaining_bytes(),outputs.remaining_attempts()),(total,1));
    assert_eq!((cache.remaining_bytes(),cache.remaining_attempts()),(total,1));
    let cached=cache.construct(producer(&controls,&pool,&calls,3,false)).unwrap();
    let mut vector=host.try_vec::<u32>(1).unwrap();vector.try_fill(1,[0x12345678]).unwrap();
    assert_eq!(calls.get(),2);assert_eq!(vector.as_slice(),&[0x12345678]);
    let (b,br)=funding::tests::reservation(&pool,200,10_000_000).into_funding().unwrap();
    let bp=br.take_native_partition(funding::native_partition::test_receipt(&br,100)).unwrap();
    let bs=br.scope().unwrap();let mut published=PreparedNativePublication::prepare_slots(bp.clone(),1);
    published.push_observation(NativeStorageObservation::ExistingImmutable(2u32,16)).unwrap();
    let before=balances(&pool);published.publish(&bs).unwrap();assert_eq!(balances(&pool),before);
    bs.certify().unwrap();
    drop(output);assert_eq!((outputs.remaining_bytes(),outputs.remaining_attempts()),(total,1));
    let second=outputs.construct(producer(&controls,&pool,&calls,4,false)).unwrap();
    assert_eq!((outputs.remaining_bytes(),outputs.remaining_attempts()),(0,0));
    assert!(outputs.construct(producer(&controls,&pool,&calls,5,false)).is_err());assert_eq!(calls.get(),3);
    drop((published,bp,b,br,cached,vector,cache,host,outputs,controls,span,r,run,root));
    assert_eq!(pool.used_bytes().unwrap(),protected);
    drop(second);assert_eq!(pool.used_bytes().unwrap(),0);
}
