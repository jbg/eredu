use super::*;
use eredu_core::{BackendFailure, ControllerDeclarationData, SharedControllerDeclaration};

#[derive(Debug)]
struct Declaration {
    words: Vec<u32>,
    // Independently owned test instrumentation, outside the declaration's data.
    retired: Arc<AtomicBool>,
}
impl Declaration {
    fn new(capacity: usize, retired: &Arc<AtomicBool>) -> Self {
        let mut words = Vec::with_capacity(capacity);
        words.extend_from_slice(&[3, 7, 11]);
        assert_eq!(words.capacity(), capacity);
        Self {
            words,
            retired: retired.clone(),
        }
    }
}
impl ControllerDeclarationData for Declaration {
    fn owned_capacity_bytes(&self) -> Option<u64> {
        u64::try_from(self.words.capacity()).ok()?.checked_mul(4)
    }
}
impl Drop for Declaration {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.words));
        self.retired.store(true, Ordering::SeqCst);
    }
}
struct DeclaredController {
    sources: Vec<SharedControllerDeclaration>,
    additional: u64,
}
impl TokenFilterController for DeclaredController {
    type Error = Infallible;
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwnedWithSharedDeclarations {
            filters: &[],
            bytes: &[],
            declarations: &self.sources,
        }
    }
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: self.additional,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("declaration inspection must not advance a controller")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        unreachable!()
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        unreachable!()
    }
}

#[test]
fn immutable_declaration_inventory_preserves_capacity_identity_and_final_alias_custody() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let retired = Arc::new(AtomicBool::new(false));
    let source = pool
        .prepare_shared_controller_declaration(|| {
            assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
            Ok(Declaration::new(13, &retired))
        })
        .unwrap();
    assert_eq!(source.capacity_bytes(), Some(52));
    assert_eq!(
        source.declaration::<Declaration>().unwrap().words,
        [3, 7, 11]
    );
    assert_eq!(balances(&pool), (0, 52, 52));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let escaped = source.clone();
    let mut controller = DeclaredController {
        sources: vec![source.clone(), source],
        additional: 52,
    };
    assert!(controller.inference_storage().shared_filters().is_none());
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    assert_eq!(contract.shared.len(), 1);
    assert_eq!(contract.shared_bytes, 52);
    assert!(
        contract
            .shared
            .values()
            .all(|row| row.kind == SourceKind::Declaration)
    );
    contract
        .validate_workspace(controller.inference_workspace(1).unwrap())
        .unwrap();
    controller.additional = 51;
    assert!(matches!(
        contract.validate_workspace(controller.inference_workspace(1).unwrap()),
        Err(ControllerStorageError::UnpricedSharedStorage {
            required_bytes: 52,
            available_bytes: 51
        })
    ));
    controller.additional = 52;
    contract
        .validate_decision(
            &TokenSamplingDecision::new(TokenFilter::All)
                .with_controller_storage(controller.inference_storage()),
        )
        .unwrap();
    assert!(matches!(
        contract.validate_decision(&TokenSamplingDecision::new(TokenFilter::All)),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let foreign_retired = Arc::new(AtomicBool::new(false));
    let foreign = SharedControllerDeclaration::new(Declaration::new(13, &foreign_retired), eredu_core::HostPreparationAuthority::unmanaged());
    assert!(!foreign.same_storage(&escaped));
    let replacement = DeclaredController {
        sources: vec![foreign],
        additional: 52,
    };
    assert!(contract.validate(&replacement).is_err());
    drop(replacement);
    assert!(foreign_retired.load(Ordering::SeqCst));

    struct AfterPayload(Arc<AtomicBool>);
    impl Drop for AfterPayload {
        fn drop(&mut self) {
            assert!(self.0.load(Ordering::SeqCst));
        }
    }
    escaped
        .try_attach(&SharedStorageDomain::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(AfterPayload(retired.clone())))
        })
        .unwrap();
    let (metadata, run) = funding(&pool, 128);
    let scope = run.scope().unwrap();
    contract.adopt(&controller, &scope).unwrap();
    assert_eq!(balances(&pool), (128, 52, 180));
    scope.certify().unwrap();
    drop((run, controller));
    assert_eq!(pool.used_bytes().unwrap(), 52);
    assert!(!retired.load(Ordering::SeqCst));
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    // Contract and reservation diagnostics contain no source/account aliases.
    drop((contract, metadata));
}

#[test]
fn declaration_factory_rejection_and_failed_prefix_keep_their_exact_construction_exclusion() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission(0))
        .unwrap();
    let calls = Cell::new(0);
    let retired = Arc::new(AtomicBool::new(false));
    assert!(matches!(
        pool.prepare_shared_controller_declaration(|| {
            calls.set(calls.get() + 1);
            Ok(Declaration::new(13, &retired))
        }),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::ReservedWorkActive
        ))
    ));
    assert_eq!(calls.get(), 0);
    drop(reservation);

    #[derive(Debug, thiserror::Error)]
    #[error("declaration constructor failed after its first destination")]
    struct Partial(Declaration);
    let failure = pool
        .prepare_shared_controller_declaration::<Declaration>(|| {
            assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
            Err(BackendFailure::from_error(Partial(Declaration::new(
                13, &retired,
            ))))
        })
        .unwrap_err();
    assert!(matches!(
        failure,
        ControllerStorageError::Construction { .. }
    ));
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(balances(&pool), (0, 0, 0));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &admission(0)),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop(failure);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);

    let short = WorkingMemoryPool::new(51, 0).unwrap();
    let retired = Arc::new(AtomicBool::new(false));
    let failure = short
        .prepare_shared_controller_declaration(|| Ok(Declaration::new(13, &retired)))
        .unwrap_err();
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(short.unquoted_owner_count().unwrap(), 0);
    assert_eq!(short.used_bytes().unwrap(), 0);
    drop(failure);
}
