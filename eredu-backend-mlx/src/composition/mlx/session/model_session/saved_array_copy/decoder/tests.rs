use super::*;

#[derive(Clone, Copy)]
pub(in crate::composition::mlx) enum Limit {
    Unlimited,
    Exact,
    OneShort,
}
#[derive(Default)]
struct Observation {
    active: bool,
    required: Option<eredu_core::DomainMemoryRequirements>,
    identities: Vec<safemlx::AllocationIdentity>,
    alias: Option<Array>,
    layout: Option<eredu_runtime::SharedStateLayout>,
}
thread_local! {
    static COPIES: Cell<usize> = const { Cell::new(0) };
    static FAIL_AFTER_KEY: Cell<bool> = const { Cell::new(false) };
    static LIMIT: Cell<Limit> = const { Cell::new(Limit::Unlimited) };
    static OBSERVATION: RefCell<Observation> = RefCell::new(Observation::default());
}
#[derive(Debug, thiserror::Error)]
#[error("injected paired decoder failure after the real key copy")]
pub(in crate::composition::mlx) struct InjectedPairedCopyFailure;
pub(super) fn after_key_copy() -> Result<(), Error> {
    COPIES.with(|copies| copies.set(copies.get() + 1));
    if FAIL_AFTER_KEY.with(|flag| flag.replace(false)) {
        return Err(Error::Other(Box::new(InjectedPairedCopyFailure)));
    }
    Ok(())
}
pub(in crate::composition::mlx) struct Observe;
impl Drop for Observe {
    fn drop(&mut self) {
        OBSERVATION.with(|slot| {
            slot.replace(Observation::default());
        });
        LIMIT.set(Limit::Unlimited);
        FAIL_AFTER_KEY.set(false);
    }
}
pub(in crate::composition::mlx) fn observe() -> Observe {
    OBSERVATION.with(|slot| {
        assert!(!slot.borrow().active);
        slot.borrow_mut().active = true;
    });
    Observe
}
pub(in crate::composition::mlx) fn limit(limit: Limit) {
    LIMIT.set(limit);
}
pub(in crate::composition::mlx) fn fail_next_key_copy() {
    FAIL_AFTER_KEY.set(true);
}
pub(in crate::composition::mlx) fn copies() -> usize {
    COPIES.get()
}
pub(in crate::composition::mlx) fn identities() -> Vec<safemlx::AllocationIdentity> {
    OBSERVATION.with(|slot| slot.borrow().identities.clone())
}
pub(in crate::composition::mlx) fn take_alias() -> Array {
    OBSERVATION.with(|slot| slot.borrow_mut().alias.take().expect("copied key"))
}

pub(in crate::composition::mlx) fn take_layout() -> eredu_runtime::SharedStateLayout {
    OBSERVATION.with(|slot| {
        slot.borrow_mut()
            .layout
            .take()
            .expect("copied shared state layout")
    })
}
pub(in crate::composition::mlx) fn installed_layout(
    runtime: &ModelRuntime<MlxBackend<'_>>,
) -> eredu_runtime::SharedStateLayout {
    let storage = runtime
        .session()
        .payload
        .model
        .erased()
        .retained_idle_auxiliary_storage()
        .unwrap();
    let mut layouts = storage
        .metadata_sources()
        .filter_map(|source| match source {
            SharedHostMetadata::Layout(layout) => Some(layout.clone()),
            _ => None,
        });
    let layout = layouts.next().expect("actual shared state layout");
    assert!(layouts.next().is_none());
    layout
}

pub(super) fn limits(
    pool: &eredu_runtime::working_memory::MemoryLedger,
    mut limits: WorkspaceCopyLimits,
) -> WorkspaceCopyLimits {
    let adjustment = match LIMIT.get() {
        Limit::Unlimited => return limits,
        Limit::Exact => 0,
        Limit::OneShort => 1,
    };
    // The first complete copy records the real account requirement; a repeated
    // unchanged source must accept exactly that amount and reject one byte less.
    let required = OBSERVATION.with(|slot| {
        slot.borrow()
            .required
            .clone()
            .expect("observed complete copy")
    });
    let snapshot = pool.snapshot().unwrap();
    let host = pool.topology().host_domain();
    limits.memory_limits =
        eredu_core::MemoryLimitDeclarations::new(snapshot.domains.iter().map(|domain| {
            let bytes = domain
                .current_charge_bytes
                .checked_add(required.get(domain.domain).unwrap().total().unwrap())
                .unwrap();
            let bytes = if domain.domain == host {
                bytes.checked_sub(adjustment).unwrap()
            } else {
                bytes
            };
            (
                pool.topology()
                    .domains()
                    .find(|(id, _)| *id == domain.domain)
                    .unwrap()
                    .1
                    .name
                    .clone(),
                eredu_core::MemoryLimit::Finite(bytes),
            )
        }));
    limits
}
pub(super) fn charged(custody: &WorkspaceCopyCustody) {
    OBSERVATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.active {
            slot.required = Some(custody.requirements().clone());
        }
    });
}
pub(super) fn completed(saved: &CopiedTextComponents) {
    OBSERVATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        if !slot.active {
            return;
        }
        let mut ids = Vec::new();
        let plan = saved.decoder.native.prepare_copy_fixed().unwrap();
        plan.visit_operands(&mut |array| {
            ids.push(array.allocation_info().unwrap().unwrap().identity());
        })
        .unwrap();
        for array in saved
            .sampling
            .arrays
            .key
            .iter()
            .chain(saved.sampling.arrays.pending.iter())
        {
            ids.push(array.allocation_info().unwrap().unwrap().identity());
        }
        slot.identities = ids;
        slot.alias = saved.sampling.arrays.key.clone();
        slot.layout = saved.decoder.native.shared_layout().cloned();
    });
}
