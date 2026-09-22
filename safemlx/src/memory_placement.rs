//! Physical facts supplied by the linked allocation mechanisms.
use crate::{
    error::{Exception, Result},
    utils::{guard::Guarded, runtime_lock},
};

/// Exact backing location, or conservative CUDA managed-memory candidates.
/// Candidate allowances describe possible occupancy, not measured residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationPlacement {
    /// The backing allocator or mechanism could not certify placement.
    Unknown,
    /// Physical host memory, including pinned and Metal shared storage.
    Host,
    /// Physical memory of one registered accelerator.
    Device {
        /// Native GPU ordinal in this process.
        ordinal: u32,
    },
    /// Full-capacity allowance in host memory and every registered GPU below
    /// `device_count`. The count is captured by the actual allocation mechanism.
    CudaManaged {
        /// Finite candidate ordinal range `0..device_count`.
        device_count: u32,
    },
    /// Prospective CUDA GPU allocator alternatives, including fixed target
    /// storage and managed or pinned host allocations. This is an
    /// allocation envelope; actual backing witnesses report the chosen mechanism.
    CudaAllocatorCandidates {
        /// Finite registered GPU ordinal range `0..device_count`.
        device_count: u32,
    },
}
impl AllocationPlacement {
    pub(crate) const fn from_native(value: safemlx_sys::mlx_memory_placement) -> Self {
        match value.kind {
            1 => Self::Host,
            2 if value.device >= 0 => Self::Device {
                ordinal: value.device as u32,
            },
            3 if value.device_count > 0 => Self::CudaManaged {
                device_count: value.device_count,
            },
            4 if value.device_count > 0 => Self::CudaAllocatorCandidates {
                device_count: value.device_count,
            },
            _ => Self::Unknown,
        }
    }
}

/// Placement selected by the linked ordinary allocation worker, independent of
/// any stream. GPU-specific asynchronous allocators may select different facts.
pub fn default_allocation_placement() -> Result<AllocationPlacement> {
    let _guard = runtime_lock::enter();
    let mut value = safemlx_sys::mlx_memory_placement {
        kind: 0,
        device: -1,
        device_count: 0,
    };
    <() as Guarded>::try_from_op(|_| unsafe {
        safemlx_sys::mlx_default_memory_placement(&mut value)
    })?;
    Ok(AllocationPlacement::from_native(value))
}

/// Complete prospective placement envelope of the linked GPU allocation worker.
/// CPU-only libraries report Unknown. Managed and pinned alternatives remain
/// represented even when the currently selected GPU has private memory.
pub fn gpu_allocation_placement() -> Result<AllocationPlacement> {
    let _guard = runtime_lock::enter();
    let mut value = safemlx_sys::mlx_memory_placement {
        kind: 0,
        device: -1,
        device_count: 0,
    };
    <() as Guarded>::try_from_op(|_| unsafe {
        safemlx_sys::mlx_gpu_allocation_placement(&mut value)
    })?;
    Ok(AllocationPlacement::from_native(value))
}

/// Hardware sharing fact for one registered accelerator; ordinals alone do not
/// establish sharing. The topology owner retains this immutable snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalMemoryDevice {
    /// Native GPU ordinal.
    pub ordinal: u32,
    /// Native hardware explicitly establishes sharing of physical host memory.
    pub shares_host_memory: bool,
}

/// Discover currently registered native accelerators and physical sharing.
/// This may initialize the native device runtime and belongs before admission.
/// Device hot-plug and changes to CUDA visibility are outside a retained topology.
pub fn physical_memory_topology() -> Result<Vec<PhysicalMemoryDevice>> {
    let _guard = runtime_lock::enter();
    let count = i32::try_from_op(|out| unsafe { safemlx_sys::mlx_memory_device_count(out) })?;
    let count = u32::try_from(count).map_err(|_| Exception::custom("invalid GPU population"))?;
    let mut devices = Vec::with_capacity(count as usize);
    for ordinal in 0..count {
        let shared = bool::try_from_op(|out| unsafe {
            safemlx_sys::mlx_memory_device_shares_host(out, ordinal as i32)
        })?;
        devices.push(PhysicalMemoryDevice {
            ordinal,
            shares_host_memory: shared,
        });
    }
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::indexing::TryIndexOp;
    use crate::{Array, Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy, Stream};

    #[test]
    fn placement_follows_backing_through_views_and_independent_copies() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let root = Array::from_slice(&[2.0f32, -3.0, 5.0, 7.0], &[2, 2]);
        let expected = root.allocation_info().unwrap().unwrap();
        assert_ne!(expected.placement(), AllocationPlacement::Unknown);
        let view = root.try_index_device((1.., ..), &stream).unwrap();
        view.evaluated().unwrap();
        assert_eq!(view.allocation_info().unwrap(), Some(expected));
        let copy = view.deep_clone().unwrap();
        let copied = copy.allocation_info().unwrap().unwrap();
        assert_ne!(copied.identity(), expected.identity());
        assert_eq!(copied.placement(), expected.placement());
        assert_eq!(copy.evaluated().unwrap().as_slice::<f32>(), &[5.0, 7.0]);
    }

    #[test]
    fn host_transfer_witness_certifies_physical_host_storage() {
        let host = HostTransferBuffer::new(&[4], Dtype::Float32, HostTransferPolicy::Transfer)
            .unwrap()
            .freeze();
        let facts = host.allocation_info().unwrap();
        assert_eq!(facts.placement(), AllocationPlacement::Host);
        let witness = host.inspect_original_source().unwrap();
        assert_eq!(witness.allocation(), facts);
    }

    #[test]
    fn topology_has_distinct_registered_device_ordinals() {
        let devices = physical_memory_topology().unwrap();
        for (slot, device) in devices.iter().enumerate() {
            assert_eq!(usize::try_from(device.ordinal).unwrap(), slot);
        }
    }

    #[test]
    fn prepared_allocator_retains_its_selected_host_placement() {
        let runtime = crate::PreparedInputRuntime::prepare().unwrap();
        let kind = runtime.raw().storage_kind;
        assert!(
            matches!(kind,
            safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CPU |
            safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_METAL_SHARED |
            safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CUDA_PINNED)
        );
        assert_eq!(runtime.allocation_placement(), AllocationPlacement::Host);
        assert_eq!(
            crate::PreparedInputRuntime::established_allocation_placement(),
            Some(runtime.allocation_placement())
        );
        let request = crate::OriginalBufferBudget::request_layout(&runtime, 17).unwrap();
        assert!(request.capacity() >= 17);
        assert_eq!(
            runtime.inspection_alias().allocation_placement(),
            AllocationPlacement::Host
        );
    }

    #[test]
    fn prospective_gpu_placement_covers_the_selected_copy_backing() {
        let devices = physical_memory_topology().unwrap();
        let prospective = gpu_allocation_placement().unwrap();
        println!(
            "Registered physical GPU facts: {devices:?}; allocation envelope: {prospective:?}"
        );
        if devices.is_empty() {
            assert_eq!(prospective, AllocationPlacement::Unknown);
            return;
        }
        for device in devices {
            let stream =
                Stream::new_with_device(&Device::new(DeviceType::Gpu, device.ordinal as i32));
            let source = Array::from_slice(&[2.0f32, -3.0, 5.0, 7.0], &[4]);
            let copy = source.copy(&stream).unwrap();
            copy.evaluated().unwrap();
            let actual = copy.allocation_info().unwrap().unwrap().placement();
            match prospective {
                AllocationPlacement::Host => assert_eq!(actual, AllocationPlacement::Host),
                AllocationPlacement::CudaAllocatorCandidates { device_count } => match actual {
                    AllocationPlacement::Host => {}
                    AllocationPlacement::Device { ordinal } => assert!(ordinal < device_count),
                    AllocationPlacement::CudaManaged {
                        device_count: actual,
                    } => assert_eq!(actual, device_count),
                    _ => panic!("a completed backing must identify its actual mechanism"),
                },
                _ => panic!("a registered GPU must expose its allocator alternatives"),
            }
            let host =
                HostTransferBuffer::copy_from_array(&copy, HostTransferPolicy::Transfer, &stream)
                    .unwrap()
                    .synchronize()
                    .unwrap();
            let values: Vec<f32> = host
                .as_bytes()
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect();
            assert_eq!(values, [2.0, -3.0, 5.0, 7.0]);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn managed_host_transfer_retains_all_registered_candidates() {
        let devices = physical_memory_topology().unwrap();
        assert!(
            !devices.is_empty(),
            "CUDA qualification requires a visible GPU"
        );
        let mut host =
            HostTransferBuffer::new(&[4], Dtype::Float32, HostTransferPolicy::Managed).unwrap();
        let values = [2.0f32, -3.0, 5.0, 7.0];
        for (slot, value) in host.as_bytes_mut().unwrap().chunks_exact_mut(4).zip(values) {
            slot.copy_from_slice(&value.to_ne_bytes());
        }
        let host = host.freeze();
        let facts = host.allocation_info().unwrap();
        assert_eq!(
            facts.placement(),
            AllocationPlacement::CudaManaged {
                device_count: u32::try_from(devices.len()).unwrap()
            }
        );
        assert_eq!(host.inspect_original_source().unwrap().allocation(), facts);
        for device in devices {
            let stream =
                Stream::new_with_device(&Device::new(DeviceType::Gpu, device.ordinal as i32));
            let copied = host.copy_to_array(&stream).unwrap().synchronize().unwrap();
            let observed =
                HostTransferBuffer::copy_from_array(&copied, HostTransferPolicy::Transfer, &stream)
                    .unwrap()
                    .synchronize()
                    .unwrap();
            let decoded: Vec<f32> = observed
                .as_bytes()
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect();
            assert_eq!(decoded, values);
            assert_eq!(
                host.allocation_info().unwrap(),
                facts,
                "access on another registered device keeps the entire managed candidate set"
            );
        }
    }
}

/// Exact persistent controls of one ordinary physical allocation root, including
/// the selected allocator's separate native buffer descriptor.
/// Original guarded allocations use their separately quoted arena controls.
pub fn physical_backing_control_bytes() -> usize {
    // SAFETY: a pure native layout fact; no allocator/device initialization.
    unsafe { safemlx_sys::mlx_physical_backing_control_bytes() }
}
/// Process-local admission for the linked ordinary allocator. Its roots retain
/// custody through cache reuse and release it only after physical eviction.
pub trait PhysicalBackingObserver: Sync + 'static {
    /// Called before native root controls and backing allocation. The returned
    /// owner must retain their admitted charge, never any native backing itself.
    fn admit(&self, facts: crate::AllocationInfo) -> Result<crate::PhysicalBackingCustody>;
}
/// Installs one immutable observer and attributes existing roots before return.
/// The observer must remain process-live; installation grants no execution scope.
pub fn observe_physical_backings<T: PhysicalBackingObserver>(observer: &'static T) -> Result<()> {
    let _guard = runtime_lock::enter();
    <() as Guarded>::try_from_op(|_| unsafe {
        safemlx_sys::mlx_observe_physical_backings(
            std::ptr::from_ref(observer).cast_mut().cast(),
            Some(backing_callback::<T>),
        )
    })
}

unsafe extern "C" fn backing_callback<T: PhysicalBackingObserver>(
    context: *mut std::ffi::c_void,
    identity: u64,
    capacity: usize,
    controls: usize,
    placement: safemlx_sys::mlx_memory_placement,
    owner: *mut *mut std::ffi::c_void,
    release: *mut Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    publish: *mut safemlx_sys::mlx_physical_backing_publish,
) -> bool {
    // SAFETY: immutable process-live T and fixed writable native outputs.
    let observer = unsafe { &*context.cast::<T>() };
    let facts = crate::AllocationInfo::from_native(identity, capacity, placement)
        .with_host_controls(controls);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| observer.admit(facts)));
    let result = match outcome {
        Ok(result) => result,
        Err(_) => Err(Exception::custom("physical backing observer unwound")),
    };
    let accepted = result.is_ok();
    let custody = result
        .unwrap_or_else(|error| crate::PhysicalBackingCustody::new(std::sync::Arc::new(error)));
    let (payload, publication) = custody.into_raw();
    unsafe {
        *owner = payload;
        *release = Some(crate::PhysicalBackingCustody::release);
        *publish = publication;
    }
    accepted
}

struct DeferredPhysicalObserver<T>(Option<std::sync::Arc<T>>);
impl<T> Drop for DeferredPhysicalObserver<T> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // Closed native aliases share the bridge, not independent Arc blocks.
            // Deallocate the final shared block before its paid owner retires.
            drop(std::sync::Arc::into_inner(owner));
        }
    }
}

unsafe extern "C" fn scoped_backing_callback<T: PhysicalBackingObserver + Send>(
    context: *mut std::ffi::c_void,
    identity: u64,
    capacity: usize,
    controls: usize,
    placement: safemlx_sys::mlx_memory_placement,
    owner: *mut *mut std::ffi::c_void,
    release: *mut Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    publish: *mut safemlx_sys::mlx_physical_backing_publish,
) -> bool {
    // SAFETY: this constructor installs exactly this deferred node type. The
    // native bridge owns it throughout this callback and retires it only once.
    let retained = unsafe {
        crate::PhysicalBackingCustody::borrowed_owner::<DeferredPhysicalObserver<T>>(context)
    };
    let observer = retained.0.as_ref().expect("live physical observer");
    // SAFETY: the shared T and the native output slots remain live for the call.
    unsafe {
        backing_callback::<T>(
            std::sync::Arc::as_ptr(observer).cast_mut().cast(),
            identity,
            capacity,
            controls,
            placement,
            owner,
            release,
            publish,
        )
    }
}

/// Shared, prepaid physical observer for real ordinary submission scopes.
/// This retains no tensor and grants no execution authority. Constructor and
/// callback owner controls must be funded before `new`; clones allocate nothing.
#[derive(Debug)]
pub struct ScopedPhysicalBackingObserver(safemlx_sys::mlx_scoped_physical_observer);
// SAFETY: constructors accept only Send+Sync callback owners; native aliases use
// an atomic count. Binding itself remains constrained by SubmissionScope affinity.
unsafe impl Send for ScopedPhysicalBackingObserver {}
unsafe impl Sync for ScopedPhysicalBackingObserver {}
impl ScopedPhysicalBackingObserver {
    /// Whether current native scope/worker ancestry has a physical observer.
    /// This read-only fact identifies a required source loan; it supplies no
    /// observer handle, storage credit, or permission to execute.
    pub fn has_current() -> bool {
        // SAFETY: native reads thread-local ancestry without mutation or hooks.
        unsafe { safemlx_sys::mlx_scoped_physical_observer_has_current() }
    }
    /// Whether the actual current scope or native worker inherits this exact
    /// observer. This allocation-free comparison creates no scope, reservation,
    /// or execution authority and does not select an observer from a registry.
    pub fn is_current(&self) -> bool {
        // SAFETY: this alias is live; native reads only thread-local source
        // pointers and compares both the retained bridge and its callback.
        unsafe { safemlx_sys::mlx_scoped_physical_observer_is_current(self.0) }
    }
    /// Exact native shared bridge allocation, excluding the caller's Arc owner.
    pub fn control_bytes() -> usize {
        // SAFETY: pure sizeof query, no runtime work.
        unsafe { safemlx_sys::mlx_submission_scope_physical_observer_control_bytes() }
    }
    /// Exact deferred Rust owner node, excluding the caller's shared allocation.
    /// Reserve this alongside `control_bytes` before constructing an observer.
    pub fn retirement_control_bytes<T: PhysicalBackingObserver + Send>() -> usize {
        crate::PhysicalBackingCustody::control_bytes::<DeferredPhysicalObserver<T>>()
    }
    /// Constructs the separately funded observer before entering native work.
    /// Final native release queues its owner for ordinary unlocked reclamation.
    pub fn new<T: PhysicalBackingObserver + Send>(observer: std::sync::Arc<T>) -> Result<Self> {
        Self::with_cache_policy(observer, true, true)
    }
    /// Constructs an observer whose newly born backing retires at its final
    /// physical alias instead of entering the native cache. Existing cached
    /// backing keeps its original owner and cache policy when reused.
    /// Constructor controls and deferred owner custody match `new`.
    pub fn new_uncached<T: PhysicalBackingObserver + Send>(
        observer: std::sync::Arc<T>,
    ) -> Result<Self> {
        Self::with_cache_policy(observer, false, true)
    }
    /// Constructs an observer that bypasses cached backing and retires newly
    /// created roots after their last physical alias. Previously cached roots
    /// keep their identity, payer and cache policy until their own eviction.
    /// The same allocator, placement and admission callbacks remain selected;
    /// native worker descendants inherit this source through their real scope.
    /// Constructor controls and deferred owner custody match `new`.
    pub fn new_fresh<T: PhysicalBackingObserver + Send>(
        observer: std::sync::Arc<T>,
    ) -> Result<Self> {
        Self::with_cache_policy(observer, false, false)
    }
    fn with_cache_policy<T: PhysicalBackingObserver + Send>(
        observer: std::sync::Arc<T>,
        retain_in_cache: bool,
        reuse_cached_backing: bool,
    ) -> Result<Self> {
        let custody = crate::PhysicalBackingCustody::new(DeferredPhysicalObserver(Some(observer)));
        let (context, _) = custody.into_raw();
        let mut raw = safemlx_sys::mlx_scoped_physical_observer {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: initialized output and immutable Send+Sync owner. This same
        // prepaid retirement node survives either acceptance or construction error.
        let status = unsafe {
            let construct = if !reuse_cached_backing {
                safemlx_sys::mlx_scoped_physical_observer_new_fresh
            } else if retain_in_cache {
                safemlx_sys::mlx_scoped_physical_observer_new
            } else {
                safemlx_sys::mlx_scoped_physical_observer_new_uncached
            };
            construct(
                &mut raw,
                context,
                Some(scoped_backing_callback::<T>),
                Some(crate::PhysicalBackingCustody::release),
            )
        };
        if status != 0 {
            unsafe { crate::PhysicalBackingCustody::release(context) };
            return Err(Exception::custom("physical observer construction failed"));
        }
        Ok(Self(raw))
    }
}
impl Clone for ScopedPhysicalBackingObserver {
    fn clone(&self) -> Self {
        // SAFETY: this live handle is retained atomically, without allocation.
        unsafe { safemlx_sys::mlx_scoped_physical_observer_retain(self.0) };
        Self(self.0)
    }
}
impl Drop for ScopedPhysicalBackingObserver {
    fn drop(&mut self) {
        // SAFETY: one release of this alias; the shared block retires before owner.
        unsafe { safemlx_sys::mlx_scoped_physical_observer_free(self.0) };
    }
}
impl crate::SubmissionScope {
    /// Binds the prepaid observer to this actual empty ordinary scope. CPU tasks
    /// retain that same scope through execution and destruction. No allocation.
    pub fn bind_physical_observer(
        &mut self,
        observer: &ScopedPhysicalBackingObserver,
    ) -> Result<()> {
        // SAFETY: actual live scope and observer, validated by native owner-thread
        // and empty-scope checks. Success acquires its independent alias.
        let status = unsafe {
            safemlx_sys::mlx_submission_scope_bind_physical_observer(self.raw(), observer.0)
        };
        if status == 0 {
            Ok(())
        } else {
            Err(Exception::custom(
                "physical allocation observer scope mismatch",
            ))
        }
    }
}
