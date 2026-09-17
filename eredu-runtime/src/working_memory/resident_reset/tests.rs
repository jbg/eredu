use super::*;
use crate::{HostMetadataKey, HostSlotMetadata, StateLayout};
use eredu_core::{
    AttentionPolicy, BackendDescriptor, BackendFailure, BackendProvider, BackendSession,
    Completion, DeviceCapabilities, DeviceDescriptor, LayerSchedule, ModelRuntime, ObservationSet,
    PendingTextInput, PreparedModel, SessionCapabilities, SessionResetLimits, Submission,
    TextGenerationBackend, TextGenerationConfig, TextPreparationInput, TextStepContext,
    TokenFilter, TokenFilterController, TokenOutput,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
mod backend;
mod direct;
mod sources;
mod empty;
#[path = "tests/construction.rs"]
mod construction;
use backend::Session;

thread_local! {
    static FAIL_AT: Cell<Option<usize>> = const { Cell::new(None) };
    static DROP_CHECK: RefCell<Option<(WorkingMemoryPool, u64, usize)>> = const { RefCell::new(None) };
    static PANIC_AT: Cell<Option<usize>> = const { Cell::new(None) };
    static RETIRE_PAUSE: RefCell<Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>> = const { RefCell::new(None) };
    static POISON_ENTRY: Cell<bool> = const { Cell::new(false) };
    static FORBID_KEY_CALLBACKS: Cell<bool> = const { Cell::new(false) };
    static KEY_DROP_CHECK: RefCell<Option<(WorkingMemoryPool, u64, usize)>> = const { RefCell::new(None) };
    static FILLS: Cell<usize> = const { Cell::new(0) };
    static RETIRE_CHECK: RefCell<Option<(WorkingMemoryPool, u64)>> = const { RefCell::new(None) };
}
pub(super) fn before_entry_publication(pool: &WorkingMemoryPool) {
    if POISON_ENTRY.replace(false) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _usage = pool.0.usage.lock().unwrap();
            panic!("injected post-acceptance accounting poison");
        }));
    }
}
pub(super) fn fail_at(index: usize) -> bool {
    FAIL_AT.get() == Some(index)
}
pub(super) fn after_entry_retirement(pool: &WorkingMemoryPool) {
    let pause = RETIRE_PAUSE.with_borrow_mut(Option::take);
    if let Some((ready, release)) = pause {
        ready.send(()).unwrap();
        release.recv().unwrap();
    }
    RETIRE_CHECK.with_borrow(|value| {
        if let Some((expected, ceiling)) = value {
            if expected.same_domain(pool) {
                let available = {
                    let usage = pool
                        .0
                        .usage
                        .try_lock()
                        .expect("entry retirement is outside Usage");
                    assert_eq!(pool.0.capacity(&usage, None), *ceiling);
                    assert!(usage.reset_retiring_count > 0);
                    assert!(usage.reserved > 0);
                    pool.0.available(&usage, None).unwrap()
                };
                // Without the retiring floor this independently attempted
                // admission fits the pool's much higher configured ceiling.
                assert!(matches!(
                    pool.register_storage([(99_u16, available + 1)]),
                    Err(WorkingMemoryError::BudgetExceeded { .. })
                ));
            }
        }
    });
}
#[derive(Debug)]
struct Slot {
    window: Option<i32>,
    position: u64,
    values: [u64; 4],
}
impl Drop for Slot {
    fn drop(&mut self) {
        if self.position == 0 {
            DROP_CHECK.with_borrow_mut(|slot| {
                if let Some((pool, minimum, count)) = slot {
                    assert!(
                        pool.used_bytes().unwrap() >= *minimum,
                        "partial payload precedes refund"
                    );
                    *count += 1;
                }
            });
        }
    }
}
struct State {
    layers: HostSlotTable<Slot>,
    layout: SharedStateLayout,
    global_start: usize,
    retention: super::super::InferenceRetention,
}
impl InferenceStateRetention for State {
    fn inference_retention(&self) -> &super::super::InferenceRetention {
        &self.retention
    }
    fn inference_retention_mut(&mut self) -> &mut super::super::InferenceRetention {
        &mut self.retention
    }
    fn retain_inference(&mut self, request: &super::super::InferenceRequest) {
        self.retention.retain(request);
    }
}
impl ResidentKvResetState for State {
    type Layer = Slot;
    type ResetPlan = (usize, usize);
    type ResetContext = ();
    fn resident_reset_plan(&self) -> Result<(Self::ResetPlan, usize), WorkingMemoryError> {
        if construction::enabled() {
            Ok(((self.global_start, self.layers.len()), construction::BYTES))
        } else { Ok(((0, 0), 0)) }
    }
    fn prepare_resident_reset_context(&self, plan: &Self::ResetPlan,
        funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>) -> Result<(), BackendFailure> {
        if construction::enabled() {
            if *plan != (self.global_start, self.layers.len()) {
                return Err(BackendFailure::from_error(WorkingMemoryError::IdentityMismatch));
            }
            construction::prepare(funding.expect("accepted source context"))?;
        }
        Ok(())
    }
    type Child = ();
    fn resident_reset_layers(&self) -> &HostSlotTable<Slot> {
        &self.layers
    }
    fn resident_reset_layout(&self) -> &SharedStateLayout {
        &self.layout
    }
    fn resident_reset_global_start(&self) -> usize {
        self.global_start
    }
    fn resident_fork_is_empty(&self) -> bool {
        self.layers.slots().iter().all(|slot| slot.position == 0 && slot.values == [0; 4])
    }
    fn validate_resident_reset_layer(layer: &Slot, policy: &LayerCachePolicy) -> bool {
        match policy {
            LayerCachePolicy::KeyValue { attention, .. } => attention
                .sliding_window_i32()
                .is_ok_and(|w| w == layer.window),
            _ => false,
        }
    }
    fn empty_resident_reset_layer(policy: &LayerCachePolicy) -> Slot {
        if PANIC_AT.get() == Some(FILLS.get()) {
            panic!("injected empty-layer constructor unwind");
        }
        FILLS.set(FILLS.get() + 1);
        let LayerCachePolicy::KeyValue { attention, .. } = policy else {
            unreachable!()
        };
        Slot {
            window: attention.sliding_window_i32().unwrap(),
            position: 0,
            values: [0; 4],
        }
    }
    fn from_resident_reset(
        layout: SharedStateLayout,
        global_start: usize,
        layers: HostSlotTable<Slot>,
    ) -> Self {
        Self {
            layers,
            layout,
            global_start,
            retention: Default::default(),
        }
    }
}
#[derive(Debug, PartialEq, Eq)]
struct Key(HostMetadataKey);
impl Clone for Key {
    fn clone(&self) -> Self {
        assert!(
            !FORBID_KEY_CALLBACKS.get(),
            "reset must not clone provider keys"
        );
        Self(self.0.clone())
    }
}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        assert!(
            !FORBID_KEY_CALLBACKS.get(),
            "reset must not compare provider keys"
        );
        self.0.cmp(&other.0)
    }
}
impl Drop for Key {
    fn drop(&mut self) {
        KEY_DROP_CHECK.with_borrow_mut(|check| {
            if let Some((pool, minimum, drops)) = check {
                let valid = pool
                    .0
                    .usage
                    .try_lock()
                    .is_ok_and(|usage| usage.reserved >= *minimum);
                *drops = if valid {
                    drops.saturating_add(1)
                } else {
                    usize::MAX
                };
            }
        });
    }
}
impl HostSlotStorageKey for Key {
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        assert!(
            !FORBID_KEY_CALLBACKS.get(),
            "reset must not project provider keys"
        );
        Some(&self.0)
    }
}
struct Data {
    state: State,
    displaced: Option<State>,
    selected: SelectedStateRealization,
    registration: WorkingMemoryStorage<Key>,
    execution: InferenceExecutionIdentity,
    control: Arc<()>,
    pool: WorkingMemoryPool,
    foreign: bool,
    foreign_source: Option<Rc<RefCell<Data>>>,
    original_bytes: u64,
}
impl Data {
    fn plan(&self) -> PreparedResidentKvReset<'_, State, Key> {
        // This unit fixture binds actual source fields directly. The production
        // mint is independently covered by the shared-session conformance test.
        let source = ResidentResetSource {
            state: &self.state,
            selected: &self.selected,
            execution: &self.execution,
            control: &self.control,
            revision: self.state.retention.initialized_revision(),
        };
        if self
            .state
            .layers
            .metadata()
            .original_reset_custody()
            .is_some()
        {
            PreparedResidentKvReset::prepare_original(source)
        } else if direct::requested() {
            let table = Key(self
                .state
                .layers
                .metadata()
                .identity()
                .registry_key()
                .clone());
            let layout = Key(self.state.layout.identity().registry_key().clone());
            PreparedResidentKvReset::prepare_registered(source, table, layout)
        } else {
            PreparedResidentKvReset::prepare(source, &self.registration)
        }
        .unwrap()
    }
}
struct Backend {
    data: Rc<RefCell<Data>>,
}
impl ResidentResetSession<State> for Session {
    fn validate_resident_reset_source(
        &self,
        source: &ResidentResetSource<'_, State>,
    ) -> Result<(), WorkingMemoryError> {
        let data = self.0.borrow();
        if data.plan().source.same_source(source) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl Backend {
    fn construct(
        &self,
        session: &Session,
        claim: SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        let (replacement, held) = {
            let data = self.data.borrow();
            let alternate = data.foreign_source.as_ref().map(|owner| owner.borrow());
            let plan = alternate.as_deref().unwrap_or(&data).plan();
            let held = plan
                .required_bytes()
                .checked_add(claim.limits().safety_reserve_bytes);
            let foreign = Session(self.data.clone());
            let target = if data.foreign { &foreign } else { session };
            (
                plan.construct(target, claim, &data.pool)
                    .map_err(BackendFailure::from_error)?,
                held.expect("successful original comparison checked the safety sum"),
            )
        };
        let mut data = self.data.borrow_mut();
        data.original_bytes = held;
        let previous = std::mem::replace(&mut data.state, replacement);
        data.displaced = Some(previous);
        Ok(())
    }
}

fn fixture(
    capacity: Option<u64>,
    count: usize,
) -> (ModelRuntime<Backend>, Rc<RefCell<Data>>, u64, u64) {
    fixture_in_pool(
        WorkingMemoryPool::new(capacity.unwrap_or(10_000_000), 0).unwrap(),
        count,
    )
}
fn fixture_in_pool(
    pool: WorkingMemoryPool,
    count: usize,
) -> (ModelRuntime<Backend>, Rc<RefCell<Data>>, u64, u64) {
    FILLS.set(0);
    FAIL_AT.set(None);
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 4).unwrap();
    let layout = SharedStateLayout::new(
        StateLayout::new(LayerSchedule::new(count, vec![policy; count]).unwrap()).unwrap(),
    );
    let state = State {
        layers: HostSlotTable::new(
            (0..count)
                .map(|i| Slot {
                    window: None,
                    position: 19,
                    values: [3 + i as u64, 7, 11, 23],
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        layout: layout.clone(),
        global_start: 5,
        retention: Default::default(),
    };
    let source_bytes =
        state.layers.metadata().capacity_bytes().unwrap() + layout.capacity_bytes().unwrap();
    let registration = pool
        .register_storage([
            (
                Key(state.layers.metadata().identity().registry_key().clone()),
                state.layers.metadata().capacity_bytes().unwrap(),
            ),
            (
                Key(layout.identity().registry_key().clone()),
                layout.capacity_bytes().unwrap(),
            ),
        ])
        .unwrap();
    let data = Rc::new(RefCell::new(Data {
        selected: crate::replicated_text::resident_reset_test_selection(layout.layout().clone()),
        state,
        displaced: None,
        registration,
        execution: Default::default(),
        control: Arc::new(()),
        pool,
        foreign: false,
        foreign_source: None,
        original_bytes: 0,
    }));
    let required = data.borrow().plan().required_bytes();
    let runtime = ModelRuntime::prepare(Backend { data: data.clone() }, ()).unwrap();
    (runtime, data, source_bytes, required)
}
#[test]
fn genuine_original_reset_exact_and_one_short_preserve_source_before_construction() {
    let (_, _, source, required) = fixture(None, 3);
    for short in [true, false] {
        let (mut runtime, data, actual_source, actual_required) =
            fixture(Some(source + required - u64::from(short)), 3);
        assert_eq!((actual_source, actual_required), (source, required));
        let result = runtime.reset_admitted(SessionResetLimits::new(source + required));
        assert_eq!(result.is_err(), short);
        let data = data.borrow();
        if short {
            assert_eq!(FILLS.get(), 0);
            assert_eq!(data.pool.used_bytes().unwrap(), source);
            assert!(data
                .state
                .layers
                .slots()
                .iter()
                .all(|s| s.position == 19 && s.values[1] == 7));
        } else {
            assert_eq!(FILLS.get(), 3);
            assert_eq!(data.pool.used_bytes().unwrap(), source + required);
            assert_eq!(data.state.global_start, 5);
            assert!(data.state.retention.is_empty());
            assert!(data
                .state
                .layers
                .slots()
                .iter()
                .all(|s| s.position == 0 && s.values == [0; 4]));
            assert!(data
                .state
                .layout
                .same_storage(&data.displaced.as_ref().unwrap().layout));
        }
    }
}
#[test]
fn escaped_token_identity_and_payload_free_key_have_distinct_final_lifetimes() {
    let (mut runtime, data, _source, required) = fixture(None, 2);
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    let (pool, token, identity, key) = {
        let data = data.borrow();
        let token = data.state.layers.metadata().clone();
        let identity = token.identity().clone();
        let key = identity.registry_key().clone();
        (data.pool.clone(), token, identity, key)
    };
    let layout_bytes = data.borrow().state.layout.capacity_bytes().unwrap();
    drop(runtime);
    drop(data);
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    assert!(matches!(
        token.try_attach(pool.shared_storage_domain(), || Ok::<
            Box<dyn Send + Sync>,
            (),
        >(Box::new(()))),
        Err(crate::HostSlotAttachmentError::Retired)
    ));
    drop(token);
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    drop(identity);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    // The still-live registry key cannot hold the original account or identity
    // allocation, and remains unique instead of allowing an address ABA.
    assert_eq!(key, key.clone());
}
#[test]
fn every_partial_fill_frontier_keeps_real_capacity_error_and_original_prefix() {
    for frontier in 0..3 {
        let (mut runtime, data, source, required) = fixture(None, 3);
        FAIL_AT.set(Some(frontier));
        let failure = runtime
            .reset_admitted(SessionResetLimits::new(10_000_000))
            .unwrap_err();
        FAIL_AT.set(None);
        let exact = std::error::Error::source(&failure)
            .unwrap()
            .downcast_ref::<ResidentResetError<State>>()
            .unwrap();
        assert_eq!(exact.initialized_count(), frontier);
        assert_eq!(exact.retained_bytes(), required);
        assert!(matches!(exact.cause, ResetCause::Allocation(_)));
        let pool = data.borrow().pool.clone();
        assert_eq!(pool.used_bytes().unwrap(), source + required);
        assert!(data
            .borrow()
            .state
            .layers
            .slots()
            .iter()
            .all(|s| s.position == 19));
        drop(runtime);
        drop(data);
        assert_eq!(pool.used_bytes().unwrap(), source + required);
        drop(failure);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn foreign_genuine_claim_rejects_before_any_bank_or_constructor_and_next_claim_succeeds() {
    let (mut runtime, data, source, _) = fixture(None, 2);
    data.borrow_mut().foreign = true;
    let failures = (0..20)
        .map(|_| {
            let failure = runtime
                .reset_admitted(SessionResetLimits::new(10_000_000))
                .unwrap_err();
            assert_eq!(
                std::error::Error::source(&failure)
                    .unwrap()
                    .downcast_ref::<ResidentResetError<State>>()
                    .unwrap()
                    .retained_bytes(),
                0
            );
            failure
        })
        .collect::<Vec<_>>();
    assert_eq!(data.borrow().pool.used_bytes().unwrap(), source);
    assert_eq!(FILLS.get(), 0);
    data.borrow_mut().foreign = false;
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    assert_eq!(FILLS.get(), 2);
    let pool = data.borrow().pool.clone();
    drop(runtime);
    drop(data);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(failures);
}
#[test]
fn final_fixed_entry_retirement_preserves_old_ceiling_outside_usage_loan() {
    let (mut runtime, data, source, required) = fixture(None, 2);
    let ceiling = source + required;
    runtime
        .reset_admitted(SessionResetLimits::new(ceiling))
        .unwrap();
    let pool = data.borrow().pool.clone();
    RETIRE_CHECK.with_borrow_mut(|slot| *slot = Some((pool.clone(), ceiling)));
    drop(runtime);
    drop(data);
    RETIRE_CHECK.with_borrow_mut(|slot| *slot = None);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn current_session_rejects_equal_geometry_foreign_source_before_original_acceptance() {
    let (mut runtime, data, source, _) = fixture(None, 3);
    let (_foreign_runtime, foreign, foreign_source, _) = fixture(None, 3);
    data.borrow_mut().foreign_source = Some(foreign.clone());
    let failure = runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap_err();
    let error = std::error::Error::source(&failure)
        .unwrap()
        .downcast_ref::<ResidentResetError<State>>()
        .unwrap();
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(data.borrow().pool.used_bytes().unwrap(), source);
    assert_eq!(foreign.borrow().pool.used_bytes().unwrap(), foreign_source);
    assert_eq!(FILLS.get(), 0);
    data.borrow_mut().foreign_source = None;
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    assert_eq!(FILLS.get(), 3);
}
#[test]
fn application_safety_overflow_and_foreign_domain_reject_without_fill() {
    let (mut runtime, data, source, required) = fixture(None, 2);
    for limits in [
        SessionResetLimits {
            capacity_bytes: 10_000_000,
            application_memory_budget_bytes: Some(required),
            safety_reserve_bytes: 1,
        },
        SessionResetLimits {
            capacity_bytes: 10_000_000,
            application_memory_budget_bytes: None,
            safety_reserve_bytes: u64::MAX,
        },
    ] {
        let failure = runtime.reset_admitted(limits).unwrap_err();
        let error = std::error::Error::source(&failure)
            .unwrap()
            .downcast_ref::<ResidentResetError<State>>()
            .unwrap();
        let expected = if limits.safety_reserve_bytes == u64::MAX {
            SessionResetRejection::Overflow
        } else {
            SessionResetRejection::ApplicationBudgetExceeded {
                required_bytes: required + 1,
                budget_bytes: required,
            }
        };
        assert!(matches!(&error.cause, ResetCause::Claim(actual) if *actual == expected));
        assert_eq!(error.retained_bytes(), 0);
    }
    let original = data.borrow().pool.clone();
    let foreign = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    data.borrow_mut().pool = foreign.clone();
    assert!(runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .is_err());
    assert_eq!(original.used_bytes().unwrap(), source);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    assert_eq!(FILLS.get(), 0);
}
#[test]
fn original_source_witness_and_new_revision_retain_actual_account_without_registry_cycle() {
    use super::super::OriginalStorageSourcesLayout;
    use eredu_core::HostPreparationAuthority;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct HostCustody {
        pool: WorkingMemoryPool,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for HostCustody {
        fn drop(&mut self) {
            // The final carrier retires its source tokens before the enclosing
            // host-preparation authority. No source charge becomes a new grant.
            assert_eq!(self.pool.used_bytes().unwrap(), 0);
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    let (mut runtime, data, _source, required) = fixture(None, 2);
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    let host_drops = Arc::new(AtomicUsize::new(0));
    let (pool, witness, revision, pin) = {
        let data = data.borrow();
        let metadata = data.state.layers.metadata();
        assert!(!metadata
            .try_attach(
                data.pool.shared_storage_domain(),
                || -> Result<Box<dyn Send + Sync>, ()> {
                    panic!("original same-domain mode cannot invoke attachment provider")
                }
            )
            .unwrap());
        let foreign = WorkingMemoryPool::new(10_000_000, 0).unwrap();
        assert!(matches!(
            metadata.try_attach(
                foreign.shared_storage_domain(),
                || -> Result<Box<dyn Send + Sync>, ()> {
                    panic!("original foreign-domain mode cannot invoke attachment provider")
                }
            ),
            Err(crate::HostSlotAttachmentError::OriginalDomainMismatch)
        ));
        let witness = data.pool.pin_original_reset_slots(metadata).unwrap();
        assert_eq!(witness.original_bytes(), required);
        assert!(WorkingMemoryPool::new(10_000_000, 0)
            .unwrap()
            .pin_original_reset_slots(data.state.layers.metadata())
            .is_err());

        let used = data.pool.used_bytes().unwrap();
        let host = HostPreparationAuthority::retain(HostCustody {
            pool: data.pool.clone(),
            drops: host_drops.clone(),
        });
        let key = metadata.identity().registry_key();
        let mut empty = OriginalStorageSourcesLayout::new(0)
            .unwrap()
            .construct(&data.pool, &host)
            .unwrap();
        assert!(matches!(
            empty.retain_table(metadata),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert!(!empty.contains(key));
        drop(empty);

        let mut foreign_carrier = OriginalStorageSourcesLayout::new(1)
            .unwrap()
            .construct(&foreign, &host)
            .unwrap();
        assert!(matches!(
            foreign_carrier.retain_table(metadata),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert!(!foreign_carrier.contains(key));
        drop(foreign_carrier);
        assert_eq!(foreign.used_bytes().unwrap(), 0);

        let mut carrier = OriginalStorageSourcesLayout::new(1)
            .unwrap()
            .construct(&data.pool, &host)
            .unwrap();
        carrier.retain_table(metadata).unwrap();
        // The same real table is an alias, even at the exact one-row limit.
        carrier.retain_table(metadata).unwrap();
        assert!(carrier.contains(key));
        let shared = data
            .pool
            .pin_registered_storage(std::iter::empty::<(Key, u64)>())
            .unwrap();
        let shared_alias = shared.clone();
        assert!(matches!(
            shared.with_retained_original_sources(&mut carrier),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert!(
            carrier.contains(key),
            "refusal preserves the source carrier"
        );
        drop(shared_alias);
        let pin = data
            .pool
            .pin_registered_storage(std::iter::empty::<(Key, u64)>())
            .unwrap()
            .with_retained_original_sources(&mut carrier)
            .unwrap();
        assert!(
            !carrier.contains(key),
            "successful association moves custody"
        );
        assert_eq!(pin.bytes(), metadata.capacity_bytes().unwrap());
        assert_eq!(data.pool.used_bytes().unwrap(), used);
        drop(host);
        assert_eq!(host_drops.load(Ordering::SeqCst), 0);
        (
            data.pool.clone(),
            witness,
            data.state.retention.revision().clone(),
            pin,
        )
    };
    let layout_bytes = data.borrow().state.layout.capacity_bytes().unwrap();
    drop(runtime);
    drop(data);
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    drop(witness);
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    drop(revision);
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    let pin_alias = pin.clone();
    drop(pin);
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    assert_eq!(host_drops.load(Ordering::SeqCst), 0);
    drop(pin_alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(host_drops.load(Ordering::SeqCst), 1);
}
#[test]
fn exact_capacity_contenders_share_one_original_acceptance_without_overcommit() {
    let (_, _, source, required) = fixture(None, 2);
    let capacity = source * 2 + required;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let ready = Arc::new(std::sync::Barrier::new(2));
    let held = Arc::new(std::sync::Barrier::new(2));
    let workers = (0..2)
        .map(|_| {
            let pool = pool.clone();
            let ready = ready.clone();
            let held = held.clone();
            std::thread::spawn(move || {
                let (mut runtime, _data, actual_source, actual_required) = fixture_in_pool(pool, 2);
                assert_eq!((actual_source, actual_required), (source, required));
                ready.wait();
                let accepted = runtime
                    .reset_admitted(SessionResetLimits::new(capacity))
                    .is_ok();
                held.wait();
                accepted
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        workers
            .into_iter()
            .map(|w| usize::from(w.join().unwrap()))
            .sum::<usize>(),
        1
    );
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn post_acceptance_entry_failure_retains_actual_original_owner_and_poison_never_refunds() {
    let (mut runtime, data, source, required) = fixture(None, 2);
    POISON_ENTRY.set(true);
    let failure = runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap_err();
    let exact = std::error::Error::source(&failure)
        .unwrap()
        .downcast_ref::<ResidentResetError<State>>()
        .unwrap();
    assert_eq!(exact.retained_bytes(), required);
    assert!(matches!(
        exact.cause,
        ResetCause::Memory(WorkingMemoryError::Poisoned)
    ));
    assert!(exact.entry.is_some());
    let pool = data.borrow().pool.clone();
    {
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.reserved, required);
        assert_eq!(usage.registered, source);
        assert!(usage.reset_pending.is_some());
    }
    assert_eq!(FILLS.get(), 0);
    drop(runtime);
    drop(data);
    drop(failure);
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.reserved, required);
    assert_eq!(usage.reservations, 1);
    assert!(usage.reset_pending.is_some());
}

#[test]
fn constructor_unwind_retires_partial_buffer_before_original_account_refund() {
    let (mut runtime, data, source, required) = fixture(None, 3);
    let pool = data.borrow().pool.clone();
    DROP_CHECK.with_borrow_mut(|slot| *slot = Some((pool.clone(), source + required, 0)));
    PANIC_AT.set(Some(2));
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = runtime.reset_admitted(SessionResetLimits::new(10_000_000));
    }));
    PANIC_AT.set(None);
    assert_eq!(DROP_CHECK.with_borrow_mut(Option::take).unwrap().2, 2);
    assert!(unwind.is_err());
    assert_eq!(FILLS.get(), 2);
    assert_eq!(pool.used_bytes().unwrap(), source);
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    assert!(data
        .borrow()
        .state
        .layers
        .slots()
        .iter()
        .all(|slot| slot.position == 19));
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    drop(runtime);
    drop(data);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn concurrent_retirements_keep_lowest_ceiling_until_last_entry_refund() {
    let (_, _, source, required) = fixture(None, 2);
    let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    let low = source * 2 + required * 2 + 100;
    let high = low + 100_000;
    let (mut a, adata, _, _) = fixture_in_pool(pool.clone(), 2);
    let (mut b, bdata, _, _) = fixture_in_pool(pool.clone(), 2);
    a.reset_admitted(SessionResetLimits::new(low)).unwrap();
    b.reset_admitted(SessionResetLimits::new(high)).unwrap();
    let aowner = adata.borrow().state.layers.metadata().clone();
    let bowner = bdata.borrow().state.layers.metadata().clone();
    drop(a);
    drop(adata);
    drop(b);
    drop(bdata);
    let spawn = |owner: HostSlotMetadata| {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            RETIRE_PAUSE.with_borrow_mut(|slot| *slot = Some((ready_tx, release_rx)));
            drop(owner);
        });
        (worker, ready_rx, release_tx)
    };
    let (aworker, aready, arelease) = spawn(aowner);
    aready.recv().unwrap();
    let (bworker, bready, brelease) = spawn(bowner);
    bready.recv().unwrap();
    let reject_above_floor = || {
        let available = {
            let usage = pool.0.usage.lock().unwrap();
            assert_eq!(pool.0.capacity(&usage, None), low);
            pool.0.available(&usage, None).unwrap()
        };
        assert!(matches!(
            pool.register_storage([(71_u16, available + 1)]),
            Err(WorkingMemoryError::BudgetExceeded { .. })
        ));
    };
    reject_above_floor();
    arelease.send(()).unwrap();
    aworker.join().unwrap();
    assert_eq!(pool.0.usage.lock().unwrap().reset_retiring_count, 1);
    reject_above_floor();
    brelease.send(()).unwrap();
    bworker.join().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let usage = pool.0.usage.lock().unwrap();
    assert_eq!(usage.reset_retiring_count, 0);
    assert_eq!(pool.0.capacity(&usage, None), 10_000_000);
}
