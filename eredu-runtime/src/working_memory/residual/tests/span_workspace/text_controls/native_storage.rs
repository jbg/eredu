//! Real accepted-plan issuance and canonical publication through a neutral provider.
use super::*;
use crate::working_memory::{
    NativeStorageError, NativeStorageObservation, NativeStorageRegistration,
    NativeStorageSelection, OriginalNativeBudgetCustody, OriginalNativeStorageBank,
    OriginalNativeStorageMechanism, PreparedNativeStoragePlan,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[derive(Debug)]
struct Failure(&'static str);
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Failure {}
struct Budget(OriginalNativeBudgetCustody);
struct Root {
    // Physical birth retires before sidecars, as the native contract requires.
    budget: RefCell<Option<Rc<Budget>>>,
    key: u32,
    bytes: u64,
    values: [f32; 2],
    generation: Cell<u64>,
    replace_after_observation: Cell<bool>,
    owners: RefCell<[Option<NativeStorageRegistration<u32>>; 4]>,
}
impl Root {
    fn native(key: u32, bytes: u64, budget: &Rc<Budget>) -> Self {
        assert!(bytes <= budget.0.capacity_bytes());
        Self {
            budget: RefCell::new(Some(budget.clone())),
            key,
            bytes,
            values: [1.25, -3.5],
            generation: Cell::new(0),
            replace_after_observation: Cell::new(false),
            owners: RefCell::new(std::array::from_fn(|_| None)),
        }
    }
    fn ordinary(key: u32, bytes: u64) -> Self {
        Self {
            budget: RefCell::new(None),
            key,
            bytes,
            values: [4.0, 2.5],
            generation: Cell::new(0),
            replace_after_observation: Cell::new(false),
            owners: RefCell::new(std::array::from_fn(|_| None)),
        }
    }
    fn attachments(&self) -> usize {
        self.owners.borrow().iter().flatten().count()
    }
}
struct Observation<'a> {
    root: &'a Root,
    kind: u8,
    generation: u64,
}
#[derive(Clone)]
struct Mechanism {
    selection: NativeStorageSelection,
    pool: WorkingMemoryPool,
    calls: Rc<Cell<usize>>,
    fail_install: Rc<Cell<bool>>,
    reuse_attachments: Rc<Cell<bool>>,
    nested_key_bytes: Option<u64>,
}
impl Mechanism {
    fn new(pool: &WorkingMemoryPool) -> Self {
        Self {
            selection: NativeStorageSelection::default(),
            pool: pool.clone(),
            calls: Rc::new(Cell::new(0)),
            fail_install: Rc::new(Cell::new(false)),
            reuse_attachments: Rc::new(Cell::new(false)),
            nested_key_bytes: None,
        }
    }
    fn callback(&self) {
        // Deterministic failure instead of a hanging nested-lock regression.
        drop(
            self.pool
                .0
                .usage
                .try_lock()
                .expect("provider callback outside Usage"),
        );
        assert!(self.pool.used_bytes().unwrap() > 0);
        self.calls.set(self.calls.get() + 1);
    }
}
impl OriginalNativeStorageMechanism for Mechanism {
    type Key = u32;
    type Budget = Rc<Budget>;
    type Root<'a> = &'a Root;
    type Attachment = NativeStorageRegistration<u32>;
    type Error = Failure;
    type Observation<'a> = Observation<'a>;
    fn selection(&self) -> &NativeStorageSelection {
        &self.selection
    }
    fn key_clone_storage_bytes(&self) -> Option<u64> {
        self.nested_key_bytes
    }
    fn create_budget(&self, custody: OriginalNativeBudgetCustody) -> Result<Self::Budget, Failure> {
        self.callback();
        if self.fail_install.get() {
            return Err(Failure("budget refused before native publication"));
        }
        Ok(Rc::new(Budget(custody)))
    }
    fn observe<'a, 'root: 'a>(
        &'a self,
        budget: &'a Rc<Budget>,
        root: &'root Root,
    ) -> Result<Observation<'a>, Failure> {
        self.callback();
        assert_eq!(root.values.len(), 2);
        let kind = match root.budget.borrow().as_ref() {
            Some(actual) if Rc::ptr_eq(actual, budget) => 0,
            Some(_) => 1,
            None => 2,
        };
        let generation = root.generation.get();
        if root.replace_after_observation.replace(false) {
            // Deterministic descriptor replacement through another fixture
            // alias after the borrowed facts were read. Existing sidecars
            // still name the old physical generation.
            root.generation.set(generation + 1);
        }
        Ok(Observation { root, kind, generation })
    }
    fn describe(observed: &Observation<'_>) -> NativeStorageObservation<u32> {
        let r = observed.root;
        match observed.kind {
            0 => NativeStorageObservation::Originating(r.key, r.bytes),
            1 => NativeStorageObservation::Existing(r.key, r.bytes),
            _ => NativeStorageObservation::Ordinary(r.key, r.bytes),
        }
    }
    fn prepare_attachment(&self, owner: Self::Attachment) -> Result<Self::Attachment, Failure> {
        self.callback();
        Ok(owner)
    }
    fn has_retained_attachment(&self, previous: &Observation<'_>, current: &Observation<'_>, pool: &WorkingMemoryPool) -> bool {
        self.reuse_attachments.get()
            && self.pool.shared_storage_domain().same_identity(pool.shared_storage_domain())
            && previous.kind == current.kind
            && previous.generation == current.generation
            && current.generation == 0
            && std::ptr::eq(previous.root, current.root)
            && current.root.attachments() > 0
    }
    fn registration(owner: &Self::Attachment) -> &Self::Attachment {
        owner
    }
    fn attach(
        observed: Observation<'_>,
        owner: Self::Attachment,
    ) -> Result<(), (Failure, Self::Attachment)> {
        // This closed fixture uses key 99 as the explicit reached native refusal.
        if observed.root.key == 99 {
            return Err((Failure("second physical attachment refused"), owner));
        }
        let mut owners = observed.root.owners.borrow_mut();
        *owners
            .iter_mut()
            .find(|slot| slot.is_none())
            .expect("cold fixed sidecar population") = Some(owner);
        Ok(())
    }
}

fn selected(
    pool: &WorkingMemoryPool,
    mechanism: &Mechanism,
    cap: u64,
    attempts: usize,
    rows: usize,
) -> IncrementalInferenceQuote {
    let q = replacement_quote(pool, geometry(), 0).into_incremental();
    let full = (0..q.span_workspace().plan().records().len())
        .map(|i| q.span_workspace().span_bytes(i))
        .collect::<Vec<_>>();
    let plan = PreparedNativeStoragePlan::<Mechanism>::prepare(
        q.span_workspace(),
        &mechanism.selection,
        Some(cap),
        Some((attempts, rows)),
        full.iter().map(|n| n.map(|n| n.min(cap))),
        Some(4096),
        Some(4096),
    )
    .unwrap();
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        q.span_workspace().plan(),
        facts(),
    )
    .unwrap()
    .with_native_storage(plan)
    .unwrap();
    let q = q.with_span_workspace_and_text_controls(controls).unwrap();
    assert_eq!(
        (0..full.len())
            .map(|i| q.span_workspace().span_bytes(i))
            .collect::<Vec<_>>(),
        full,
        "separate Q never changes the recorded full span requirement"
    );
    q
}
fn setup(
    pool: &WorkingMemoryPool,
    cap: u64,
    attempts: usize,
    rows: usize,
) -> (
    WorkingMemoryReservation,
    WorkingMemoryFundingRun,
    OwnedTextSpanWorkspace,
    OriginalNativeStorageBank<Mechanism>,
    Mechanism,
) {
    let mechanism = Mechanism::new(pool);
    let (r, run, accepted) = accept(pool, selected(pool, &mechanism, cap, attempts, rows));
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let mut bank = span
        .take_native_storage_bank::<Mechanism>(&run, &mechanism.selection)
        .unwrap()
        .unwrap();
    bank.install(mechanism.clone()).unwrap();
    (r, run, span, bank, mechanism)
}
fn balances(pool: &WorkingMemoryPool) -> (u64, u64) {
    let usage = pool.0.usage.lock().unwrap();
    (usage.reserved, usage.registered)
}

#[test]
fn attached_registration_compares_its_exact_metadata_origin_without_retaining_or_reissuing() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let foreign_pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let foreign_namespace = foreign_pool.register_storage([(1u32, 64)]).unwrap();
    let (ar, a_run, a_span, mut a, mechanism) = setup(&pool, 32, 1, 1);
    let (br, b_run, b_span, b, _) = setup(&pool, 32, 1, 1);
    let (fr, f_run, f_span, f, _) = setup(&foreign_pool, 32, 1, 1);
    let a_origin = a_span.control_guard().metadata_custody().into();
    let b_origin = b_span.control_guard().metadata_custody().into();
    let foreign_origin = f_span.control_guard().metadata_custody().into();
    let mut scope = a_run.scope().unwrap();
    let root = Root::native(7, 16, a.budget_for_scope(&scope).unwrap());
    let mut publication = a.claim_publication(&mut scope).unwrap();
    publication.publish(&scope, [&root], &[]).unwrap();
    assert_eq!(root.attachments(), 1);
    assert_eq!(root.values, [1.25, -3.5]);
    let before = balances(&pool);
    let calls = mechanism.calls.get();
    let foreign_before = balances(&foreign_pool);
    {
        let owners = root.owners.borrow();
        let registration = owners.iter().flatten().next().unwrap();
        assert!(registration.has_metadata_origin(&a_origin));
        assert!(!registration.has_metadata_origin(&b_origin));
        assert!(!registration.has_metadata_origin(&foreign_origin));
        assert!(registration.has_metadata_origin(&a_origin));
    }
    assert_eq!(balances(&pool), before);
    assert_eq!(balances(&foreign_pool), foreign_before);
    assert_eq!(mechanism.calls.get(), calls);
    assert_eq!(a.remaining_publications(), 0);
    assert_eq!(b.remaining_publications(), 1);
    assert!(NativeStorageRegistration::<u32>::metadata_origin_control_bytes().unwrap() > 0);
    scope.certify().unwrap();
    drop((publication, a, a_span, ar, a_run));
    {
        let owners = root.owners.borrow();
        let registration = owners.iter().flatten().next().unwrap();
        assert!(registration.has_metadata_origin(&a_origin), "closed retained account remains the same origin");
        a_origin.validate_retained_origin(&pool).unwrap();
    }
    drop((root, a_origin, b_origin, foreign_origin, b, b_span, br, b_run, f, f_span, fr, f_run));
    drop((namespace, foreign_namespace));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(foreign_pool.used_bytes().unwrap(), 0);
}

#[test]
fn selected_plan_keeps_unknowns_and_checks_actual_coverage_cardinality_before_any_native_work() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let mechanism = Mechanism::new(&pool);
    let q = replacement_quote(&pool, geometry(), 0).into_incremental();
    let n = q.span_workspace().plan().records().len();
    let before = pool.used_bytes().unwrap();
    for coverage in [vec![Some(0); n - 1], vec![Some(0); n + 1]] {
        assert!(matches!(
            PreparedNativeStoragePlan::<Mechanism>::prepare(
                q.span_workspace(),
                &mechanism.selection,
                Some(32),
                Some((2, 3)),
                coverage,
                Some(4096),
                Some(4096)
            ),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
    let unknown = PreparedNativeStoragePlan::<Mechanism>::prepare(
        q.span_workspace(),
        &mechanism.selection,
        Some(32),
        None,
        vec![Some(0); n],
        Some(4096),
        None,
    )
    .unwrap();
    assert_eq!(unknown.control_bytes(), None);
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        q.span_workspace().plan(),
        facts(),
    )
    .unwrap()
    .with_native_storage(unknown)
    .unwrap();
    assert!(q.with_span_workspace_and_text_controls(controls).is_err());
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert_eq!(mechanism.calls.get(), 0);
    drop(namespace);
}

#[test]
fn accepted_selection_is_once_only_even_for_zero_and_failed_installation() {
    for cap in [0, 32] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let namespace = pool.register_storage([(1u32, 64)]).unwrap();
        let mechanism = Mechanism::new(&pool);
        let (r, run, accepted) = accept(&pool, selected(&pool, &mechanism, cap, 1, 1));
        let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
        let mut bank = span
            .take_native_storage_bank::<Mechanism>(&run, &mechanism.selection)
            .unwrap()
            .unwrap();
        mechanism.fail_install.set(true);
        assert!(matches!(
            bank.install(mechanism.clone()),
            Err(NativeStorageError::Native(Failure(_)))
        ));
        assert!(matches!(
            bank.install(mechanism.clone()),
            Err(NativeStorageError::Memory(
                WorkingMemoryError::AlreadyStarted
            ))
        ));
        assert!(matches!(
            span.take_native_storage_bank::<Mechanism>(&run, &mechanism.selection),
            Err(WorkingMemoryError::AlreadyStarted)
        ));
        assert_eq!(mechanism.calls.get(), 1);
        drop((bank, span, r, run, namespace));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn same_account_other_work_refuses_before_observation_but_same_scope_moves_and_claims_succeed() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let (r, run, span, mut bank, mechanism) = setup(&pool, 32, 3, 2);
    let mut a = run.scope().unwrap();
    let b = run.scope().unwrap();
    let root = Root::native(7, 16, bank.budget_for_scope(&a).unwrap());
    let mut wrong = bank.claim_publication(&mut a).unwrap();
    let calls = mechanism.calls.get();
    assert!(matches!(
        wrong.publish(&b, [&root], &[]),
        Err(NativeStorageError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(mechanism.calls.get(), calls);
    assert_eq!(root.attachments(), 0);
    let mut moved = a;
    let mut first = bank.claim_publication(&mut moved).unwrap();
    let mut second = bank.claim_publication(&mut moved).unwrap();
    first.publish(&moved, [&root], &[]).unwrap();
    second.publish(&moved, [&root], &[]).unwrap();
    assert_eq!(root.attachments(), 2);
    assert!(matches!(
        bank.claim_publication(&mut moved),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    moved.certify().unwrap();
    b.certify().unwrap();
    drop((first, second, wrong, root, bank, span, r, run, namespace));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn selected_native_and_ordinary_rows_share_registry_without_double_charge_and_keep_a_under_b() {
    for birth_first in [false, true] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let namespace = pool.register_storage([(1u32, 64)]).unwrap();
        let (ar, a_run, a_span, mut a, _) = setup(&pool, 32, 1, 3);
        let (br, b_run, b_span, mut b, _) = setup(&pool, 32, 1, 3);
        let protected = a_span.protected_host_bytes();
        let b_protected = b_span.protected_host_bytes();
        let mut ascope = a_run.scope().unwrap();
        let mut bscope = b_run.scope().unwrap();
        let native = Root::native(7, 16, a.budget_for_scope(&ascope).unwrap());
        let ordinary = Root::ordinary(8, 7);
        let before = balances(&pool);
        let mut ap = a.claim_publication(&mut ascope).unwrap();
        ap.publish(&ascope, [&native, &ordinary, &native], &[])
            .unwrap();
        assert_eq!(
            native.attachments(),
            1,
            "duplicate inventory adds one sidecar"
        );
        assert_eq!(balances(&pool), (before.0 - 7, before.1 + 7));
        let mut bp = b.claim_publication(&mut bscope).unwrap();
        bp.publish(&bscope, [&native], &[]).unwrap();
        assert_eq!(balances(&pool), (before.0 - 7, before.1 + 7));
        assert_eq!(native.values, [1.25, -3.5]);
        ascope.certify().unwrap();
        bscope.certify().unwrap();
        drop((ordinary, ap, bp, a, b, a_span, b_span, ar, br, a_run, b_run));
        // Both A-native sidecars carry A raw metadata. B's bank/control never
        // becomes the canonical donor and can retire completely here.
        assert_eq!(
            pool.used_bytes().unwrap(),
            64 + protected + b_protected + 32
        );
        if birth_first {
            drop(native.budget.borrow_mut().take());
        } else {
            native
                .owners
                .borrow_mut()
                .iter_mut()
                .for_each(|owner| drop(owner.take()));
        }
        assert_eq!(
            pool.used_bytes().unwrap(),
            64 + protected + 32 + if birth_first { b_protected } else { 0 }
        );
        drop(native);
        drop(namespace);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn later_attachment_refusal_keeps_successful_prefix_failed_preparation_and_origin_until_retirement()
{
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let (r, run, span, mut bank, mechanism) = setup(&pool, 32, 1, 2);
    let protected = span.protected_host_bytes();
    let mut scope = run.scope().unwrap();
    let first = Root::native(7, 16, bank.budget_for_scope(&scope).unwrap());
    let second = Root::native(99, 16, bank.budget_for_scope(&scope).unwrap());
    let before = balances(&pool);
    let mut attempt = bank.claim_publication(&mut scope).unwrap();
    assert!(matches!(
        attempt.publish(&scope, [&first, &second], &[]),
        Err(NativeStorageError::Native(Failure(
            "second physical attachment refused"
        )))
    ));
    assert_eq!((first.attachments(), second.attachments()), (1, 0));
    assert_eq!(balances(&pool), before);
    assert!(
        mechanism.calls.get() >= 5,
        "actual observations and preparations reentered the pool"
    );
    assert!(matches!(
        attempt.publish(&scope, [&first], &[]),
        Err(NativeStorageError::Memory(
            WorkingMemoryError::PreparationAlreadyStarted
        ))
    ));
    assert!(attempt.take_source(0).is_none());
    // Explicit fixture settlement: physical roots are retired before attempting
    // certification; an attachment failure itself was never a completion proof.
    drop((first, second));
    scope.certify().unwrap();
    drop((bank, span, r, run, namespace));
    assert_eq!(pool.used_bytes().unwrap(), protected + 32);
    drop(attempt);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn late_publication_preserves_sources_and_rejects_missing_foreign_birth_without_spending_credit() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let (ar, a_run, a_span, a, _) = setup(&pool, 32, 1, 1);
    let (br, b_run, b_span, mut b, _) = setup(&pool, 32, 1, 1);
    let ascope = a_run.scope().unwrap();
    let mut bscope = b_run.scope().unwrap();
    let unpublished = Root::native(13, 16, a.budget_for_scope(&ascope).unwrap());
    drop(b_run);
    let before = balances(&pool);
    let mut attempt = b.claim_publication(&mut bscope).unwrap();
    assert!(matches!(
        attempt.publish(&bscope, [&unpublished], &[]),
        Err(NativeStorageError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(balances(&pool), before);
    assert_eq!(unpublished.attachments(), 0);
    drop(unpublished);
    ascope.certify().unwrap();
    bscope.certify().unwrap();
    drop((attempt, a, b, ar, br, a_run, a_span, b_span, namespace));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn excess_actual_root_iterator_and_wrong_selection_are_refused_without_native_callbacks() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let (r, run, span, mut bank, mechanism) = setup(&pool, 32, 1, 1);
    let mut scope = run.scope().unwrap();
    let root = Root::native(7, 16, bank.budget_for_scope(&scope).unwrap());
    let mut attempt = bank.claim_publication(&mut scope).unwrap();
    let calls = mechanism.calls.get();
    struct Liar<'a>(&'a Root, usize);
    impl<'a> Iterator for Liar<'a> {
        type Item = &'a Root;
        fn next(&mut self) -> Option<Self::Item> {
            if self.1 == 0 {
                None
            } else {
                self.1 -= 1;
                Some(self.0)
            }
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, Some(0))
        }
    }
    assert!(matches!(
        attempt.publish(&scope, Liar(&root, 2), &[]),
        Err(NativeStorageError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(mechanism.calls.get(), calls);
    let other = Mechanism::new(&pool);
    let (r2, run2, accepted2) = accept(&pool, selected(&pool, &mechanism, 0, 1, 1));
    let (mut span2, _) = accepted2
        .into_funded_text_span_workspace(&run2, &r2)
        .unwrap();
    assert!(matches!(
        span2.take_native_storage_bank::<Mechanism>(&run2, &other.selection),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        span2.take_native_storage_bank::<Mechanism>(&run2, &mechanism.selection),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    scope.certify().unwrap();
    drop((
        attempt, root, bank, span, r, run, span2, r2, run2, namespace,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod qualified_controls;

mod sampling_extension;

#[test]
fn retained_attachment_revalidates_descriptor_at_commit_without_an_extra_sidecar() {
    for changed in [false, true] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let namespace = pool.register_storage([(1u32, 64)]).unwrap();
        let (r, run, span, mut bank, mechanism) = setup(&pool, 32, 2, 1);
        let mut scope = run.scope().unwrap();
        let root = Root::native(7, 16, bank.budget_for_scope(&scope).unwrap());
        let mut first = bank.claim_publication(&mut scope).unwrap();
        first.publish(&scope, [&root], &[]).unwrap();
        assert_eq!(root.attachments(), 1);
        mechanism.reuse_attachments.set(true);
        root.replace_after_observation.set(changed);
        let mut repeated = bank.claim_publication(&mut scope).unwrap();
        let before = balances(&pool);
        let result = repeated.publish(&scope, [&root], &[]);
        if changed {
            assert!(matches!(result, Err(NativeStorageError::Memory(WorkingMemoryError::IdentityMismatch))));
            assert!(!repeated.is_published());
        } else {
            result.unwrap();
            assert!(repeated.is_published());
        }
        assert_eq!(root.attachments(), 1, "existing attachment is never duplicated");
        assert_eq!(balances(&pool), before);
        drop(root); // explicit actual fixture settlement before scope certification
        scope.certify().unwrap();
        drop((first, repeated, bank, span, r, run, namespace));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
