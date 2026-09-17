//! Once-admitted finite source family; each declared cache slot compiles lazily.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
mod metal {
    use eredu_runtime::working_memory::{
        InitializedSharedNative, SharedNativeInitializationCustody,
        SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
        WorkingMemoryPool,
    };
    use safemlx::fast::{
        BorrowedKernelTemplate, KernelDefinitionError, KernelInputClass, KernelInputSignature,
        KernelSpecialization, MetalKernelDefinitionPlan, MetalKernelFamilyPlan,
        PreparedMetalKernelFamily,
    };
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock,
    };

    pub(crate) type Definition = PreparedMetalKernelFamily<SharedNativeInitializationCustody>;
    pub(crate) static PLAN: MetalKernelDefinitionPlan<'static, 1, 1> = MetalKernelDefinitionPlan {
        name: "f32_pointwise",
        inputs: ["input"],
        outputs: ["output"],
        source: "uint i=thread_position_in_grid.x; if(i>=threads_per_grid.x) return; float x=pointwise_value(input,i); float denominator=1.0f+stable_exp_f32(-x); output[i]=(OP==1 ? 1.0f : x)/denominator;",
        header: concat!(include_str!("../nn/exp_f32.metal"), "\ninline float pointwise_value(constant const float& x, uint i) { return x; }\ninline float pointwise_value(constant const float* x, uint i) { return x[i]; }\ninline float pointwise_value(device const float* x, uint i) { return x[i]; }\n"),
        ensure_row_contiguous: true,
        atomic_outputs: false,
    };
    static OP1: [BorrowedKernelTemplate<'static>; 1] = [BorrowedKernelTemplate::Int(c"OP", 1)];
    static OP2: [BorrowedKernelTemplate<'static>; 1] = [BorrowedKernelTemplate::Int(c"OP", 2)];
    const fn specialization(
        class: KernelInputClass,
        templates: &'static [BorrowedKernelTemplate<'static>],
    ) -> KernelSpecialization<'static, 1, 1> {
        KernelSpecialization {
            inputs: [KernelInputSignature {
                dtype: safemlx::Dtype::Float32,
                class,
            }],
            outputs: [safemlx::Dtype::Float32],
            templates,
        }
    }
    pub(crate) static FAMILY: MetalKernelFamilyPlan<'static, 1, 1, 6> = MetalKernelFamilyPlan {
        definition: PLAN,
        specializations: [
            specialization(KernelInputClass::Scalar, &OP1),
            specialization(KernelInputClass::Small, &OP1),
            specialization(KernelInputClass::Device, &OP1),
            specialization(KernelInputClass::Scalar, &OP2),
            specialization(KernelInputClass::Small, &OP2),
            specialization(KernelInputClass::Device, &OP2),
        ],
    };
    // Pure declared producer fact, independent of cache warmth/initialization.
    static SOURCE_QUALIFIED: OnceLock<bool> = OnceLock::new();
    pub(crate) fn source_qualified() -> bool {
        *SOURCE_QUALIFIED
            .get_or_init(|| FAMILY.layout::<SharedNativeInitializationCustody>().is_ok())
    }
    static INITIALIZED: OnceLock<InitializedSharedNative<Definition>> = OnceLock::new();
    static INITIALIZING: AtomicBool = AtomicBool::new(false);
    pub(crate) fn static_storage_bytes() -> usize {
        // These are actual fixed owners/literal bytes, not a per-request grant.
        std::mem::size_of_val(&INITIALIZED)
            + std::mem::size_of_val(&SOURCE_QUALIFIED)
            + std::mem::size_of_val(&INITIALIZING)
            + std::mem::size_of_val(&PLAN)
            + std::mem::size_of_val(&FAMILY)
            + std::mem::size_of_val(&OP1)
            + std::mem::size_of_val(&OP2)
            + c"OP".to_bytes_with_nul().len()
            + PLAN.name.len()
            + PLAN.source.len()
            + PLAN.header.len()
            + PLAN.inputs[0].len()
            + PLAN.outputs[0].len()
            + Definition::static_storage_bytes()
    }
    struct Winner;
    impl Drop for Winner {
        fn drop(&mut self) {
            INITIALIZING.store(false, Ordering::Release);
        }
    }
    #[derive(Debug)]
    pub(crate) struct Initializer;
    impl SharedNativeInitializer for Initializer {
        type Output = Definition;
        type Error = KernelDefinitionError<SharedNativeInitializationCustody>;
        fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
            FAMILY
                .layout::<SharedNativeInitializationCustody>()
                .map_err(|cause| match cause {
                    safemlx::fast::KernelDefinitionCause::Overflow => WorkingMemoryError::Overflow,
                    _ => WorkingMemoryError::UnknownBound,
                })?
                .required_bytes()
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Winner>()))
                .and_then(|bytes| {
                    bytes.checked_add(std::mem::size_of::<MlxPointwiseDefinitionError>())
                })
                .and_then(|bytes| {
                    bytes.checked_add(std::mem::size_of::<
                        super::super::input_allocator::MlxInputAllocatorInitializationError,
                    >())
                })
                .ok_or(WorkingMemoryError::Overflow)
        }
        fn initialize(
            self,
            custody: SharedNativeInitializationCustody,
        ) -> Result<Definition, Self::Error> {
            FAMILY.realize(custody)
        }
    }
    #[derive(Debug, thiserror::Error)]
    pub(crate) enum MlxPointwiseDefinitionError {
        #[error("pointwise definition admission: {0}")]
        Policy(#[source] WorkingMemoryError),
        #[error("pointwise definition constructor: {0}")]
        Constructor(#[source] SharedNativeInitializationError<Initializer>),
        #[error("pointwise definition initialization is busy")]
        Busy,
    }
    pub(crate) fn prepare_admitted(
        pool: &WorkingMemoryPool,
    ) -> Result<(), MlxPointwiseDefinitionError> {
        if !pool.same_domain(&super::super::domain()) {
            return Err(MlxPointwiseDefinitionError::Policy(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        if let Some(owner) = INITIALIZED.get() {
            return owner
                .validate_pool(pool)
                .map_err(MlxPointwiseDefinitionError::Policy);
        }
        if INITIALIZING
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(MlxPointwiseDefinitionError::Busy);
        }
        let _winner = Winner;
        if let Some(owner) = INITIALIZED.get() {
            return owner
                .validate_pool(pool)
                .map_err(MlxPointwiseDefinitionError::Policy);
        }
        let owner = pool
            .initialize_shared_native(Initializer)
            .map_err(MlxPointwiseDefinitionError::Constructor)?;
        INITIALIZED
            .set(owner)
            .expect("exclusive pointwise definition initializer");
        Ok(())
    }
    // No ordinary installation/promotion exists. Successful original cold entry
    // has already validated this process domain; this borrow grants no credit.
    pub(crate) fn definition() -> Option<&'static Definition> {
        INITIALIZED.get().map(InitializedSharedNative::output)
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn family_exact_admission_aliases_and_refusal_preserve_account() {
            let bytes =
                WorkingMemoryPool::shared_native_initialization_required_bytes(&Initializer);
            if std::env::var_os("EREDU_REQUIRE_QUALIFIED_KERNEL_FAMILY").is_some() {
                assert!(bytes.is_ok(), "pinned kernel definition must qualify");
            }
            let Ok(bytes) = bytes else { return };
            let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
            let refused = short.initialize_shared_native(Initializer).unwrap_err();
            assert!(refused.accounting_failure().is_some());
            assert!(refused.rejected_plan().is_some());
            assert!(refused.completed_output().is_none());
            assert_eq!(short.used_bytes().unwrap(), 0);
            drop(refused);
            let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
            let owner = pool.initialize_shared_native(Initializer).unwrap();
            assert_eq!(owner.original_bytes(), bytes);
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            owner.validate_pool(&pool).unwrap();
            assert!(matches!(
                owner.validate_pool(&short),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            // Aliases retain this same actual native definition/account; no
            // second shared initialization or per-invocation allowance exists.
            let first = std::sync::Arc::new(owner);
            let last = first.clone();
            let a = first.output() as *const _;
            let b = last.output() as *const _;
            assert_eq!(a, b);
            drop(first);
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            drop(last);
            safemlx::reclaim_allocation_owners();
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) use metal::*;
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn static_storage_bytes() -> usize {
    0
}

#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn source_qualified() -> bool {
    false
}
