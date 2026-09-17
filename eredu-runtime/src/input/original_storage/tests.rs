use super::*;
use crate::{
    input::host::{HostInputPart, PreparedHostInputPlan},
    working_memory::PreparedNativeInputCompiler,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn source(pool: &crate::working_memory::WorkingMemoryPool) -> OriginalPreparedHostInput {
    let tokens = [7u32, 19];
    let pixels = [0.25f32, -1.5, 3.0, 2.75];
    let grid = [1i32, 2, 2];
    let metadata = [(
        InputMetadataKey::PatchGrid,
        HostTensorView {
            shape: &[1, 3],
            values: HostTensorValues::I32(&grid),
        },
    )];
    let parts = [
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 2],
                values: HostTensorValues::U32(&tokens),
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: InputModality::Image,
            kind: InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[4, 1],
                values: HostTensorValues::F32(&pixels),
            },
            metadata: &metadata,
            extents: &[InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 2,
            }],
        },
    ];
    pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
        .unwrap()
}
#[derive(Debug, thiserror::Error)]
#[error("actual closed slot refusal")]
struct Failed;
struct Native(OriginalPreparedInputCustody);
#[derive(Clone, Debug, PartialEq, Eq)]
struct Slot(usize);
type Output = PreparedModelInputSource<Slot, Slot, Native>;
type SourcePlan<'a> = PreparedModelInputSourcePlan<'a, Slot, Slot, Native, Failed>;
struct Plan<'a> {
    source: &'a OriginalPreparedHostInput,
    calls: &'a AtomicUsize,
    fail: Option<usize>,
}
impl PreparedNativeInputCompiler for Plan<'_> {
    type Output = Output;
    type Error = PreparedModelInputSourceError<Failed>;
    fn source(&self) -> &OriginalPreparedHostInput {
        self.source
    }
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        Ok(SourcePlan::new(self.source)?.required_storage_bytes())
    }
    fn compile(self, owner: OriginalPreparedInputCustody) -> Result<Output, (Output, Self::Error)> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        SourcePlan::new(self.source).unwrap().construct(
            owner,
            |owner| Ok(Native(owner)),
            |_, slot, _| {
                if self.fail == Some(slot) {
                    Err(Failed)
                } else {
                    Ok((Slot(slot), Slot(slot)))
                }
            },
        )
    }
}
fn plan<'a>(source: &'a OriginalPreparedHostInput, calls: &'a AtomicUsize) -> Plan<'a> {
    Plan {
        source,
        calls,
        fail: None,
    }
}
#[test]
fn original_prepared_storage_exact_short_foreign_and_unquoted_compare_before_constructor() {
    let seed = crate::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&seed);
    let calls = AtomicUsize::new(0);
    let b = crate::working_memory::WorkingMemoryPool::prepared_native_input_required_bytes(&plan(
        &i, &calls,
    ))
    .unwrap();
    let ibytes = i.original_bytes();
    drop(i);
    drop(seed);
    for short in [true, false] {
        let pool = crate::working_memory::WorkingMemoryPool::new(ibytes + b - u64::from(short), 0)
            .unwrap();
        let i = source(&pool);
        let result = pool.compile_prepared_native_input(plan(&i, &calls));
        if short {
            let e = result.unwrap_err();
            assert_eq!(e.retained_bytes(), 0);
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert!(
                matches!(e.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes}) if *required_bytes==b&&*available_bytes==b-1)
            );
        } else {
            let output = result.unwrap();
            assert_eq!(output.original_bytes(), b);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            drop(output);
        }
        assert_eq!(pool.used_bytes().unwrap(), ibytes);
        let other = crate::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        assert_eq!(
            other
                .compile_prepared_native_input(plan(&i, &calls))
                .unwrap_err()
                .accounting_failure(),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        let before = calls.load(Ordering::SeqCst);
        let lease = pool.acquire_unquoted().unwrap();
        assert_eq!(
            pool.compile_prepared_native_input(plan(&i, &calls))
                .unwrap_err()
                .accounting_failure(),
            Some(&WorkingMemoryError::UnknownBound)
        );
        assert_eq!(calls.load(Ordering::SeqCst), before);
        drop(lease);
        drop(i);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn original_prepared_storage_matches_ordinary_order_descriptors_and_cache_equation() {
    let pool = crate::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let calls = AtomicUsize::new(0);
    let output = pool
        .compile_prepared_native_input(plan(&i, &calls))
        .unwrap();
    let body = output.storage();
    let prepared = body.prepared().unwrap();
    let parts = body.parts().unwrap();
    assert_eq!(prepared.len(), 2);
    assert_eq!(parts.as_ref(), prepared.parts());
    assert_eq!(parts.as_ref()[0].payload().value(), &Slot(0));
    assert_eq!(parts.as_ref()[1].payload().value(), &Slot(1));
    assert_eq!(
        parts.as_ref()[1].metadata_value(InputMetadataKey::PatchGrid),
        Some(&Slot(2))
    );
    let descriptors = i
        .parts()
        .map(|p| {
            InputPartDescriptor::new_with_extents(
                p.modality(),
                p.kind(),
                identity::<Failed>(p.payload()).unwrap(),
                p.metadata()
                    .map(|(key, v)| (key, identity::<Failed>(v).unwrap())),
                p.extents().iter().copied(),
            )
            .unwrap()
        })
        .collect();
    let ordinary = PreparedInputCacheIdentity::new(
        PreparedInputIdentity::new(descriptors).unwrap(),
        super::super::text_identity::fixed_hex(*i.content_digest()),
    )
    .unwrap();
    assert_eq!(body.cache().unwrap().as_ref(), &ordinary);
    let raw = PreparedModelInput::new(prepared.parts().to_vec(), |slot| {
        identity::<Failed>(i.slot(slot.0).unwrap()).map_err(|e| match e {
            PreparedModelInputSourceError::Descriptor(e) => e,
            _ => panic!("ordinary fixture"),
        })
    })
    .unwrap();
    let legacy: PreparedModelInputOwner<_> = raw.into();
    assert!(legacy.original_source().is_none());
    assert!(prepared.original_source().unwrap().same_source(&i));
}
#[test]
fn original_cache_profile_requires_the_exact_materialization_account() {
    let pool = crate::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let host = source(&pool);
    let calls = AtomicUsize::new(0);
    let first = pool.compile_prepared_native_input(plan(&host, &calls)).unwrap();
    let second = pool.compile_prepared_native_input(plan(&host, &calls)).unwrap();
    let first_body = first.storage();
    let second_body = second.storage();
    let cache = first_body.cache().unwrap();
    let other = second_body.cache().unwrap();
    let first_custody = first_body.prepared().unwrap().workspace_custody_ref().unwrap();
    let second_custody = second_body.prepared().unwrap().workspace_custody_ref().unwrap();
    assert!(cache.original_source().unwrap().same_source(other.original_source().unwrap()));
    assert_eq!(cache.as_ref(), other.as_ref());
    assert!(cache.original_residence(&pool).unwrap().is_ok());
    assert!(other.original_residence(&pool).unwrap().is_ok());
    assert!(cache.matches_original_account(first_custody));
    assert!(cache.clone().matches_original_account(first_custody));
    assert!(!cache.matches_original_account(second_custody));
    assert!(!other.matches_original_account(first_custody));
    let ordinary = crate::SharedPreparedInputCacheIdentity::new(cache.as_ref().clone());
    assert!(!ordinary.matches_original_account(first_custody));
    let foreign = crate::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    assert!(matches!(cache.original_residence(&foreign),
        Some(Err(WorkingMemoryError::IdentityMismatch))));
}

#[test]
fn original_prepared_cache_all_aliases_branch_before_provider_and_keep_final_identity_custody() {
    let pool = crate::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let calls = AtomicUsize::new(0);
    let output = pool
        .compile_prepared_native_input(plan(&i, &calls))
        .unwrap();
    let b = output.original_bytes();
    let cache = output.storage().cache().unwrap().clone();
    let alias = crate::SharedHostMetadata::Input(cache.clone());
    let identity = cache.identity().clone();
    let key = identity.registry_key().clone();
    let foreign = eredu_core::SharedStorageDomain::default();
    let provider =
        || -> Result<Box<dyn Send + Sync>, Infallible> { panic!("original alias called provider") };
    assert!(!cache
        .try_attach(pool.shared_storage_domain(), provider)
        .unwrap());
    assert!(!alias
        .try_attach(pool.shared_storage_domain(), provider)
        .unwrap());
    assert!(matches!(
        alias.try_attach(&foreign, provider),
        Err(eredu_core::SharedStorageAttachmentError::AttachmentMismatch)
    ));
    drop(output);
    drop(i);
    drop(cache);
    drop(alias);
    assert!(pool.used_bytes().unwrap() >= b);
    drop(identity);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(key);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn original_prepared_views_and_cache_concurrent_final_aliases_retire_the_complete_account() {
    let pool = crate::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let calls = AtomicUsize::new(0);
    let output = pool
        .compile_prepared_native_input(plan(&i, &calls))
        .unwrap();
    let prepared = output.storage().prepared().unwrap().clone();
    let parts = output.storage().parts().unwrap().clone();
    let cache = output.storage().cache().unwrap().clone();
    let twins = (prepared.clone(), parts.clone(), cache.clone());
    let held = pool.used_bytes().unwrap();
    drop(output);
    drop(i);
    assert_eq!(pool.used_bytes().unwrap(), held);
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let other = barrier.clone();
    let thread = std::thread::spawn(move || {
        other.wait();
        drop(twins);
    });
    barrier.wait();
    drop((prepared, parts, cache));
    thread.join().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn original_prepared_failed_late_slot_retains_real_prefix_and_source_without_retry() {
    let pool = crate::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let calls = AtomicUsize::new(0);
    let mut p = plan(&i, &calls);
    p.fail = Some(2);
    let bytes =
        crate::working_memory::WorkingMemoryPool::prepared_native_input_required_bytes(&p).unwrap();
    let error = pool.compile_prepared_native_input(p).unwrap_err();
    assert_eq!(error.retained_bytes(), bytes);
    assert!(matches!(
        error.compiler_failure(),
        Some(PreparedModelInputSourceError::Native(Failed))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(i);
    assert!(pool.used_bytes().unwrap() > bytes);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_prepared_host_failure_keeps_committed_prefix_and_tears_down_values_before_account() {
    #[derive(Debug)]
    struct Tracked {
        pool: crate::working_memory::WorkingMemoryPool,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Tracked {
        fn drop(&mut self) {
            assert!(
                self.pool.used_bytes().unwrap() > 0,
                "source/account retired before host/native slot"
            );
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct TrackedPlan<'a> {
        source: &'a OriginalPreparedHostInput,
        pool: &'a crate::working_memory::WorkingMemoryPool,
        drops: Arc<AtomicUsize>,
    }
    impl PreparedNativeInputCompiler for TrackedPlan<'_> {
        type Output = PreparedModelInputSource<Tracked, Tracked, Native>;
        type Error = PreparedModelInputSourceError<Failed>;
        fn source(&self) -> &OriginalPreparedHostInput {
            self.source
        }
        fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
            Ok(
                PreparedModelInputSourcePlan::<'_, Tracked, Tracked, Native, Failed>::new(
                    self.source,
                )?
                .required_storage_bytes(),
            )
        }
        fn compile(
            self,
            owner: OriginalPreparedInputCustody,
        ) -> Result<Self::Output, (Self::Output, Self::Error)> {
            PreparedModelInputSourcePlan::new(self.source)
                .unwrap()
                .construct(
                    owner,
                    |owner| Ok(Native(owner)),
                    |_, _, _| {
                        Ok((
                            Tracked {
                                pool: self.pool.clone(),
                                drops: self.drops.clone(),
                            },
                            Tracked {
                                pool: self.pool.clone(),
                                drops: self.drops.clone(),
                            },
                        ))
                    },
                )
        }
    }
    let pool = crate::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let drops = Arc::new(AtomicUsize::new(0));
    // Four outer populations, two zero-length text extents, and both text
    // descriptors finish before the first image extent's real reserve fails.
    FAIL_HOST_RESERVATION.set(Some(8));
    let error = pool
        .compile_prepared_native_input(TrackedPlan {
            source: &i,
            pool: &pool,
            drops: drops.clone(),
        })
        .unwrap_err();
    assert!(matches!(
        error.compiler_failure(),
        Some(PreparedModelInputSourceError::Allocation(_))
    ));
    assert_eq!(FAIL_HOST_RESERVATION.get(), None);
    assert_eq!(
        drops.load(Ordering::SeqCst),
        4,
        "only unpublished image handles retire on failure"
    );
    let b = error.retained_bytes();
    drop(i);
    assert!(pool.used_bytes().unwrap() > b);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 6);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod encoder;
