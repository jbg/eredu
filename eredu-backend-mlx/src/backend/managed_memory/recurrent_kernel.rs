//! Finite source owners for the shared six-input gated-delta recurrence.
#[cfg(not(feature = "cuda"))]
mod implementation {
    use eredu_runtime::working_memory::{
        InitializedSharedNative, MemoryLedger, SharedNativeInitializationCustody,
        SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
    };
    use safemlx::fast::{
        BorrowedKernelOutput, KernelDefinitionCause, KernelDefinitionError, KernelFamilyLayout,
        KernelInputClass, KernelInputSignature, KernelSpecialization, MetalKernelDefinitionPlan,
        MetalKernelFamilyPlan, PreparedMetalKernelFamily,
    };
    use safemlx::{error::Exception, Array, Dtype, OriginalScopeObserver, Stream};
    use std::{
        mem::size_of,
        sync::{
            atomic::{AtomicBool, Ordering},
            OnceLock,
        },
    };

    pub(crate) type Family = PreparedMetalKernelFamily<SharedNativeInitializationCustody>;
    use crate::backend::managed_memory::kernel_family::UnenforcedFamilyCache;
    static UNENFORCED: [UnenforcedFamilyCache; 4] = [const { UnenforcedFamilyCache::new() }; 4];
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum ScanKernel {
        DecodeScalar,
        DecodeVector,
        PrefillScalar,
        PrefillVector,
    }
    impl ScanKernel {
        pub(crate) const fn select(decode: bool, vector_decay: bool) -> Self {
            match (decode, vector_decay) {
                (true, false) => Self::DecodeScalar,
                (true, true) => Self::DecodeVector,
                (false, false) => Self::PrefillScalar,
                (false, true) => Self::PrefillVector,
            }
        }
        const fn index(self) -> usize {
            self as usize
        }
    }
    const fn definition(
        name: &'static str,
        source: &'static str,
    ) -> MetalKernelDefinitionPlan<'static, 6, 2> {
        MetalKernelDefinitionPlan {
            name,
            source,
            header: "",
            inputs: ["state", "query", "key", "value", "g", "beta"],
            outputs: ["out", "state_out"],
            ensure_row_contiguous: true,
            atomic_outputs: false,
        }
    }
    // Every input is rank three or four after the actual F32 casts. The two
    // non-scalar address spaces therefore give exactly 2^6 source signatures.
    // Dimensions come from existing query/value shape arguments, not templates.
    const fn signatures() -> [KernelSpecialization<'static, 6, 2>; 64] {
        let small = KernelInputSignature {
            dtype: Dtype::Float32,
            class: KernelInputClass::Small,
        };
        let mut values = [KernelSpecialization {
            inputs: [small; 6],
            outputs: [Dtype::Float32; 2],
            templates: &[],
        }; 64];
        let mut bits = 0;
        while bits < values.len() {
            let mut input = 0;
            while input < 6 {
                values[bits].inputs[input].class = if bits & (1 << input) == 0 {
                    KernelInputClass::Small
                } else {
                    KernelInputClass::Device
                };
                input += 1;
            }
            bits += 1;
        }
        values
    }
    static PLANS: [MetalKernelFamilyPlan<'static, 6, 2, 64>; 4] = [
        MetalKernelFamilyPlan {
            definition: definition(
                "gated_delta_decode_scalar",
                include_str!("recurrent_kernel/decode_scalar.metal"),
            ),
            specializations: signatures(),
        },
        MetalKernelFamilyPlan {
            definition: definition(
                "gated_delta_decode_vector",
                include_str!("recurrent_kernel/decode_vector.metal"),
            ),
            specializations: signatures(),
        },
        MetalKernelFamilyPlan {
            definition: definition(
                "gated_delta_prefill_scalar",
                include_str!("recurrent_kernel/prefill_scalar.metal"),
            ),
            specializations: signatures(),
        },
        MetalKernelFamilyPlan {
            definition: definition(
                "gated_delta_prefill_vector",
                include_str!("recurrent_kernel/prefill_vector.metal"),
            ),
            specializations: signatures(),
        },
    ];
    static INITIALIZED: [OnceLock<InitializedSharedNative<Family>>; 4] =
        [const { OnceLock::new() }; 4];
    static INITIALIZING: AtomicBool = AtomicBool::new(false);
    struct Winner;
    impl Drop for Winner {
        fn drop(&mut self) {
            INITIALIZING.store(false, Ordering::Release);
        }
    }
    fn layout(kind: ScanKernel) -> Result<KernelFamilyLayout, KernelDefinitionCause> {
        PLANS[kind.index()].layout::<SharedNativeInitializationCustody>()
    }
    // All four source definitions and the native platform layout are fixed for
    // this binary. Actual initialization and admission remain separate.
    static SOURCE_QUALIFIED: OnceLock<bool> = OnceLock::new();
    pub(crate) fn source_qualified() -> bool {
        *SOURCE_QUALIFIED.get_or_init(|| {
            [
                ScanKernel::DecodeScalar,
                ScanKernel::DecodeVector,
                ScanKernel::PrefillScalar,
                ScanKernel::PrefillVector,
            ]
            .into_iter()
            .all(|kind| layout(kind).is_ok())
        })
    }
    pub(crate) fn static_storage_bytes() -> usize {
        // Shared source preamble, ABI and retirement statics are already in
        // the pointwise family's process baseline. These are this owner's rows.
        size_of::<[OnceLock<InitializedSharedNative<Family>>; 4]>()
            + std::mem::size_of_val(&UNENFORCED)
            + std::mem::size_of_val(&SOURCE_QUALIFIED)
            + size_of::<AtomicBool>()
            + std::mem::size_of_val(&PLANS)
            + PLANS
                .iter()
                .map(|p| {
                    p.definition.name.len()
                        + p.definition.source.len()
                        + p.definition.header.len()
                        + p.definition.inputs.iter().map(|s| s.len()).sum::<usize>()
                        + p.definition.outputs.iter().map(|s| s.len()).sum::<usize>()
                })
                .sum::<usize>()
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let native = Family::control_bytes::<6, 2>(0, 4)?
            .checked_add(Stream::device_type_control_bytes()?)?;
        [
            size_of::<ScanKernel>(),
            size_of::<&Family>(),
            size_of::<[i32; 4]>() * 2,
            size_of::<[Array; 6]>(),
            size_of::<(Array, Array)>(),
            size_of::<Result<(Array, Array), Exception>>(),
            size_of::<Option<OriginalScopeObserver>>(),
            size_of::<[i32; 3]>(),
            size_of::<i32>() * 5,
            size_of::<bool>() * 2,
        ]
        .into_iter()
        .try_fold(native, usize::checked_add)
    }
    pub(crate) fn validate_call(kind: ScanKernel) -> Result<(), Exception> {
        if let Some(observer) = OriginalScopeObserver::try_current()? {
            if control_bytes().is_none() || INITIALIZED[kind.index()].get().is_none() {
                return Err(observer.capacity_error());
            }
        }
        Ok(())
    }
    pub(crate) fn apply(
        kind: ScanKernel,
        inputs: [&Array; 6],
        outputs: [BorrowedKernelOutput<'_>; 2],
        grid: [i32; 3],
        stream: &Stream,
    ) -> Result<[Array; 2], Exception> {
        let observer = OriginalScopeObserver::try_current()?;
        if let Some(family) = INITIALIZED[kind.index()].get() {
            return family.output().apply_fixed_device(
                inputs,
                outputs,
                &[],
                grid,
                [256, 1, 1],
                stream,
            );
        }
        if let Some(observer) = observer {
            return Err(observer.capacity_error());
        }
        let family =
            UNENFORCED[kind.index()].get_or_try_init(|owner| PLANS[kind.index()].realize(owner))?;
        family.apply_fixed_device(inputs, outputs, &[], grid, [256, 1, 1], stream)
    }
    #[derive(Debug)]
    pub(crate) struct Initializer(ScanKernel);
    impl SharedNativeInitializer for Initializer {
        type Output = Family;
        type Error = KernelDefinitionError<SharedNativeInitializationCustody>;
        fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
            layout(self.0)
                .map_err(|e| match e {
                    KernelDefinitionCause::Overflow => WorkingMemoryError::Overflow,
                    _ => WorkingMemoryError::UnknownBound,
                })?
                .required_bytes()
                .and_then(|n| n.checked_add(size_of::<Self>()))
                .and_then(|n| n.checked_add(size_of::<Winner>()))
                .and_then(|n| n.checked_add(size_of::<MlxRecurrentKernelError>()))
                .and_then(|n| {
                    n.checked_add(size_of::<
                        super::super::input_allocator::MlxInputAllocatorInitializationError,
                    >())
                })
                .ok_or(WorkingMemoryError::Overflow)
        }
        fn initialize(
            self,
            custody: SharedNativeInitializationCustody,
        ) -> Result<Family, Self::Error> {
            PLANS[self.0.index()].realize(custody)
        }
    }
    #[derive(Debug, thiserror::Error)]
    pub(crate) enum MlxRecurrentKernelError {
        #[error("recurrent kernel admission: {0}")]
        Policy(#[source] WorkingMemoryError),
        #[error("recurrent kernel construction: {0}")]
        Constructor(#[source] SharedNativeInitializationError<Initializer>),
        #[error("recurrent kernel initialization is busy")]
        Busy,
    }
    pub(crate) fn prepare_admitted(pool: &MemoryLedger) -> Result<(), MlxRecurrentKernelError> {
        if !pool.same_ledger(&super::super::ledger()) {
            return Err(MlxRecurrentKernelError::Policy(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        if INITIALIZED.iter().all(|slot| slot.get().is_some()) {
            for slot in &INITIALIZED {
                slot.get()
                    .expect("completed recurrent family")
                    .validate_pool(pool)
                    .map_err(MlxRecurrentKernelError::Policy)?;
            }
            return Ok(());
        }
        if INITIALIZING
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(MlxRecurrentKernelError::Busy);
        }
        let _winner = Winner;
        for kind in [
            ScanKernel::DecodeScalar,
            ScanKernel::DecodeVector,
            ScanKernel::PrefillScalar,
            ScanKernel::PrefillVector,
        ] {
            let slot = &INITIALIZED[kind.index()];
            if let Some(owner) = slot.get() {
                owner
                    .validate_pool(pool)
                    .map_err(MlxRecurrentKernelError::Policy)?;
            } else {
                slot.set(
                    pool.initialize_shared_native(Initializer(kind))
                        .map_err(MlxRecurrentKernelError::Constructor)?,
                )
                .expect("exclusive recurrent family initializer");
            }
        }
        Ok(())
    }
    #[cfg(all(test, feature = "metal"))]
    include!("recurrent_kernel/tests.rs");
}
#[cfg(not(feature = "cuda"))]
pub(crate) use implementation::*;
#[cfg(feature = "cuda")]
pub(crate) fn static_storage_bytes() -> usize {
    0
}
#[cfg(feature = "cuda")]
pub(crate) fn source_qualified() -> bool {
    false
}
#[cfg(feature = "cuda")]
pub(crate) fn control_bytes() -> Option<usize> {
    None
}
