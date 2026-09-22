//! Pure fixed-static facts. Thread and selected-context lifetimes are separate.

/// Fixed storage from the owning submission, C handler and safe-wrapper modules.
/// These are managed object extents, excluding opaque allocator/process/driver
/// bookkeeping. They are a partial native-domain inventory: TLS, allocator,
/// Device, stream and dynamically installed owner storage remain
/// separate. No runtime Rc is included here or duplicated per request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeStaticBaseline {
    /// Actual native submission module objects, including a static Registry
    /// only when the native ABI qualifies its constant initialization.
    pub submission_module_bytes: usize,
    /// Actual C handler function/shared-owner slots, not an installed payload.
    pub error_handler_static_bytes: usize,
    /// The one safe runtime mutex representation, not its per-thread identity.
    pub runtime_lock_static_bytes: usize,
    /// The safe error-handler initialization Once representation.
    pub error_once_static_bytes: usize,
    /// The safe retirement queue's static head, not queued owners.
    pub retirement_head_static_bytes: usize,
    /// Fixed cold snapshot of the actual prepared allocator placement.
    pub prepared_placement_static_bytes: usize,
    /// Native physical-root registry and immutable observer bridge.
    pub physical_backing_static_bytes: usize,
    /// The actual shared input allocator slot/scalar snapshot and CPU static
    /// object. The Metal heap object belongs to its separate admitted owner.
    pub input_allocator_static_bytes: usize,
    /// Whether those allocator module objects have qualified constant storage.
    pub input_allocator_constant_storage: bool,
    /// Actual Device/source module objects and generated static compressed bytes.
    /// Dynamic Device/library construction belongs to its separate source account.
    pub metal_device_static_bytes: usize,
    /// Whether the compiled Device/source producer qualifies its constant storage.
    pub metal_device_constant_storage: bool,
    /// One native Scheduler slot, constructor instruction data and first-caller
    /// thread identity/guard. The dynamic singleton has a separate source owner.
    pub scheduler_static_bytes: usize,
    /// Whether those statics have a qualified constant representation. This is
    /// independent of the actual loaded dynamic shared_mutex constructor code.
    pub scheduler_constant_storage: bool,
    /// Fixed stream-registration tree, synchronization object and publication
    /// storage. Entries/CPU encoders are separately owned dynamic allocations.
    pub stream_registration_static_bytes: usize,
    /// Qualification of those fixed extents and the compiler guard; dynamic
    /// constructor code is independently checked by the registration producer.
    pub stream_registration_static_qualified: bool,
    /// Fixed worker qualification data and libc++ TLS key/guard storage.
    /// Per-worker startup and later diagnostics remain separately owned.
    pub cpu_worker_static_bytes: usize,
    /// Static ABI qualification, not the loaded worker-constructor qualification.
    pub cpu_worker_static_qualified: bool,
    /// Native submission TLS subset only. This is not the full submitting- or
    /// worker-thread extent, and is not part of the fixed static charge.
    pub submission_thread_bytes: usize,
    /// Whether Registry is actually constant-initialized with static lifetime.
    pub constant_registry: bool,
    /// One fallback Registry candidate's inline size, not a population bound.
    pub dynamic_registry_candidate_bytes: usize,
}

impl RuntimeStaticBaseline {
    /// Known static subtotal, also available when Registry is dynamically born.
    /// This alone must not certify the complete required baseline.
    pub fn known_static_storage_bytes(&self) -> Option<usize> {
        [
            self.submission_module_bytes,
            self.error_handler_static_bytes,
            self.runtime_lock_static_bytes,
            self.error_once_static_bytes,
            self.retirement_head_static_bytes,
            self.prepared_placement_static_bytes,
            self.physical_backing_static_bytes,
            self.input_allocator_static_bytes,
            self.metal_device_static_bytes,
            self.scheduler_static_bytes,
            self.stream_registration_static_bytes,
            self.cpu_worker_static_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// The fixed static component is closed only for the qualified constant
    /// Registry. This still makes no claim about TLS or selected contexts.
    pub fn fixed_storage_bytes(&self) -> Option<usize> {
        (self.constant_registry
            && self.input_allocator_constant_storage
            && self.metal_device_constant_storage
            && self.scheduler_constant_storage
            && self.stream_registration_static_qualified
            && self.cpu_worker_static_qualified)
            .then(|| self.known_static_storage_bytes())
            .flatten()
    }
}

/// Reads only owning layout facts: no device, runtime lock, TLS initialization,
/// Registry walk, reclamation, handler setup, native allocation or authority.
pub fn runtime_static_baseline() -> RuntimeStaticBaseline {
    let mut native = safemlx_sys::mlx_submission_static_layout::default();
    let mut allocator = safemlx_sys::mlx_input_allocator_layout::default();
    // SAFETY: both C queries read type/static layout only. The output is an
    // initialized, aligned local value with the exact repr(C) layout.
    let error_handler_static_bytes = unsafe {
        safemlx_sys::mlx_submission_static_layout_for(&mut native);
        safemlx_sys::mlx_input_allocator_layout_for(&mut allocator);
        safemlx_sys::mlx_error_static_storage_bytes()
    };
    let device = crate::allocation_retention::metal_device_static_layout();
    let scheduler = crate::allocation_retention::scheduler_static_layout();
    let streams = crate::allocation_retention::stream_registration_static_layout();
    let workers = crate::allocation_retention::cpu_worker_static_layout();
    RuntimeStaticBaseline {
        submission_module_bytes: native.module_bytes,
        error_handler_static_bytes,
        runtime_lock_static_bytes: crate::utils::runtime_lock::static_storage_bytes(),
        error_once_static_bytes: crate::error::static_storage_bytes(),
        retirement_head_static_bytes: crate::allocation_retention::static_storage_bytes(),
        prepared_placement_static_bytes: crate::prepared_input::placement_static_storage_bytes(),
        // SAFETY: pure native static layout query.
        physical_backing_static_bytes: unsafe { safemlx_sys::mlx_physical_backing_static_bytes() },
        input_allocator_static_bytes: allocator.static_bytes,
        input_allocator_constant_storage: allocator.qualified == 1,
        metal_device_static_bytes: device.bytes,
        metal_device_constant_storage: device.qualified,
        scheduler_static_bytes: scheduler.bytes,
        scheduler_constant_storage: scheduler.qualified,
        stream_registration_static_bytes: streams.bytes,
        stream_registration_static_qualified: streams.qualified,
        cpu_worker_static_bytes: workers.bytes,
        cpu_worker_static_qualified: workers.qualified,
        submission_thread_bytes: native.thread_bytes,
        constant_registry: native.constant_registry,
        dynamic_registry_candidate_bytes: native.dynamic_registry_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_static_query_is_shared_and_needs_no_runtime_loan() {
        let first = runtime_static_baseline();
        assert!(first.known_static_storage_bytes().unwrap() > 0);
        assert!(first.submission_thread_bytes > 0);
        if std::env::var_os("EREDU_REQUIRE_STATIC_BASELINE_QUALIFICATION").is_some() {
            assert!(first.fixed_storage_bytes().is_some());
            assert_eq!(first.dynamic_registry_candidate_bytes, 0);
        }
        // A foreign thread can query while the runtime is owned here. It does
        // not need to initialize native owner identity or wait for this loan.
        let _loan = crate::utils::runtime_lock::enter();
        let other = std::thread::spawn(runtime_static_baseline).join().unwrap();
        assert_eq!(first, other);
    }
}
