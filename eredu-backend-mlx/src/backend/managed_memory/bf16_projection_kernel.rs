//! Finite source/cache ownership for the existing BF16 row reduction worker.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
mod metal {
    use eredu_runtime::working_memory::{
        InitializedSharedNative, SharedNativeInitializationCustody,
        SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
        WorkingMemoryPool,
    };
    use safemlx::fast::{
        BorrowedKernelOutput, BorrowedKernelTemplate, KernelDefinitionCause, KernelDefinitionError,
        KernelInputClass, KernelInputSignature, KernelSpecialization, MetalKernelDefinitionPlan,
        MetalKernelFamilyPlan, PreparedMetalKernelFamily,
    };
    use safemlx::{Array, Dtype, OriginalScopeObserver, Stream, error::Exception};
    use std::{
        mem::size_of,
        sync::{
            OnceLock,
            atomic::{AtomicBool, Ordering},
        },
    };

    type Family = PreparedMetalKernelFamily<SharedNativeInitializationCustody>;
    const DEFINITION: MetalKernelDefinitionPlan<'static, 3, 1> = MetalKernelDefinitionPlan {
        name: "bf16_row_projection",
        inputs: ["input", "weight", "groups"],
        outputs: ["output"],
        source: include_str!("bf16_projection_kernel/reduction.metal"),
        header: "",
        ensure_row_contiguous: true,
        atomic_outputs: false,
    };
    const fn templates(columns: bool, grouped: bool) -> [BorrowedKernelTemplate<'static>; 2] {
        [
            BorrowedKernelTemplate::Int(c"COLUMNS", columns as i32),
            BorrowedKernelTemplate::Int(c"GROUPED", grouped as i32),
        ]
    }
    static TEMPLATES: [[BorrowedKernelTemplate<'static>; 2]; 4] = [
        templates(false, false),
        templates(true, false),
        templates(false, true),
        templates(true, true),
    ];
    // Inputs have positive rank: matrix values/weights and a rank-one I32 ID
    // vector. Small and Device are the two possible native address classes.
    // Width and output count are invocation data, never source specializations.
    const fn signatures() -> [KernelSpecialization<'static, 3, 1>; 32] {
        let small = KernelInputSignature {
            dtype: Dtype::Bfloat16,
            class: KernelInputClass::Small,
        };
        let mut values = [KernelSpecialization {
            inputs: [
                small,
                small,
                KernelInputSignature {
                    dtype: Dtype::Int32,
                    class: KernelInputClass::Small,
                },
            ],
            outputs: [Dtype::Bfloat16],
            templates: &TEMPLATES[0],
        }; 32];
        let mut index = 0;
        while index < values.len() {
            let mut input = 0;
            while input < 3 {
                values[index].inputs[input].class = if index & (1 << input) == 0 {
                    KernelInputClass::Small
                } else {
                    KernelInputClass::Device
                };
                input += 1;
            }
            values[index].templates = &TEMPLATES[index / 8];
            index += 1;
        }
        values
    }
    static PLAN: MetalKernelFamilyPlan<'static, 3, 1, 32> = MetalKernelFamilyPlan {
        definition: DEFINITION,
        specializations: signatures(),
    };
    static INITIALIZED: OnceLock<InitializedSharedNative<Family>> = OnceLock::new();
    static INITIALIZING: AtomicBool = AtomicBool::new(false);
    struct Winner;
    impl Drop for Winner {
        fn drop(&mut self) {
            INITIALIZING.store(false, Ordering::Release);
        }
    }
    pub(crate) const OUTPUT_DIMENSIONS: usize = 32;
    // This is a fact about PLAN and the compiled native ABI, independent of
    // device state and of the separately admitted initialized family.
    static SOURCE_QUALIFIED: OnceLock<bool> = OnceLock::new();
    pub(crate) fn source_qualified() -> bool {
        *SOURCE_QUALIFIED
            .get_or_init(|| PLAN.layout::<SharedNativeInitializationCustody>().is_ok())
    }
    pub(crate) fn static_storage_bytes() -> usize {
        // The common generator preamble and retirement queue are already owned
        // by the pointwise family's process baseline in this same domain.
        size_of::<OnceLock<InitializedSharedNative<Family>>>()
            + std::mem::size_of_val(&SOURCE_QUALIFIED)
            + size_of::<AtomicBool>()
            + size_of::<MetalKernelFamilyPlan<'static, 3, 1, 32>>()
            + std::mem::size_of_val(&TEMPLATES)
            + DEFINITION.name.len()
            + DEFINITION.source.len()
            + DEFINITION.inputs.iter().map(|s| s.len()).sum::<usize>()
            + DEFINITION.outputs.iter().map(|s| s.len()).sum::<usize>()
            + c"COLUMNS".to_bytes_with_nul().len()
            + c"GROUPED".to_bytes_with_nul().len()
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let native = Family::control_bytes::<3, 1>(2, 2)?;
        [
            size_of::<&Family>(),
            size_of::<[BorrowedKernelTemplate<'_>; 2]>(),
            size_of::<[BorrowedKernelOutput<'_>; 1]>(),
            size_of::<[i32; 2]>(),
            size_of::<Option<OriginalScopeObserver>>(),
            size_of::<Result<Array, Exception>>(),
        ]
        .into_iter()
        .try_fold(native, usize::checked_add)
    }
    pub(crate) fn validate_call(rank: usize) -> Result<(), Exception> {
        if let Some(observer) = OriginalScopeObserver::try_current()? {
            if rank == 0
                || rank > OUTPUT_DIMENSIONS
                || control_bytes().is_none()
                || INITIALIZED.get().is_none()
            {
                return Err(observer.capacity_error());
            }
        }
        Ok(())
    }
    pub(crate) fn apply(
        inputs: [&Array; 3],
        rows: i32,
        outputs: i32,
        columns: bool,
        grouped: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        use safemlx::fast::{CustomKernelConfig, MetalKernel};
        use std::cell::RefCell;
        thread_local! { static ORDINARY: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        let shape = [rows, outputs];
        let output = [BorrowedKernelOutput {
            shape: &shape,
            dtype: Dtype::Bfloat16,
        }];
        let templates = templates(columns, grouped);
        if let Some(observer) = OriginalScopeObserver::try_current()? {
            let family = INITIALIZED.get().ok_or_else(|| observer.capacity_error())?;
            let [value] = family.output().apply_fixed_device(
                inputs,
                output,
                &templates,
                [32, outputs, rows],
                [32, 1, 1],
                stream,
            )?;
            return Ok(value);
        }
        ORDINARY.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    DEFINITION.name,
                    DEFINITION.inputs,
                    DEFINITION.outputs,
                    DEFINITION.source,
                    DEFINITION.header,
                    DEFINITION.ensure_row_contiguous,
                    DEFINITION.atomic_outputs,
                )?);
            }
            let loan = cell.borrow();
            let kernel = loan.as_ref().expect("BF16 kernel initialized");
            if MetalKernel::fixed_control_bytes::<3, 1>(2, 2).is_some() {
                let [value] = kernel.apply_fixed_device(
                    inputs,
                    output,
                    &templates,
                    [32, outputs, rows],
                    [32, 1, 1],
                    stream,
                )?;
                Ok(value)
            } else {
                let config = CustomKernelConfig::new()
                    .with_template_arg_int("COLUMNS", i32::from(columns))
                    .with_template_arg_int("GROUPED", i32::from(grouped))
                    .with_grid([32, outputs, rows])
                    .with_thread_group([32, 1, 1])
                    .with_output_arg(shape, Dtype::Bfloat16);
                let mut values = kernel.apply_device(inputs, &config, stream)?;
                Ok(values.pop().expect("one declared BF16 output"))
            }
        })
    }
    #[derive(Debug)]
    pub(crate) struct Initializer;
    impl SharedNativeInitializer for Initializer {
        type Output = Family;
        type Error = KernelDefinitionError<SharedNativeInitializationCustody>;
        fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
            PLAN.layout::<SharedNativeInitializationCustody>()
                .map_err(|e| match e {
                    KernelDefinitionCause::Overflow => WorkingMemoryError::Overflow,
                    _ => WorkingMemoryError::UnknownBound,
                })?
                .required_bytes()
                .and_then(|n| n.checked_add(size_of::<Self>()))
                .and_then(|n| n.checked_add(size_of::<Winner>()))
                .and_then(|n| n.checked_add(size_of::<MlxBf16ProjectionError>()))
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
            PLAN.realize(custody)
        }
    }
    #[derive(Debug, thiserror::Error)]
    pub(crate) enum MlxBf16ProjectionError {
        #[error("BF16 projection admission: {0}")]
        Policy(#[source] WorkingMemoryError),
        #[error("BF16 projection construction: {0}")]
        Constructor(#[source] SharedNativeInitializationError<Initializer>),
        #[error("BF16 projection initialization is busy")]
        Busy,
    }
    pub(crate) fn prepare_admitted(pool: &WorkingMemoryPool) -> Result<(), MlxBf16ProjectionError> {
        if !pool.same_domain(&super::super::domain()) {
            return Err(MlxBf16ProjectionError::Policy(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        if let Some(owner) = INITIALIZED.get() {
            return owner
                .validate_pool(pool)
                .map_err(MlxBf16ProjectionError::Policy);
        }
        if INITIALIZING
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(MlxBf16ProjectionError::Busy);
        }
        let _winner = Winner;
        if let Some(owner) = INITIALIZED.get() {
            return owner
                .validate_pool(pool)
                .map_err(MlxBf16ProjectionError::Policy);
        }
        let owner = pool
            .initialize_shared_native(Initializer)
            .map_err(MlxBf16ProjectionError::Constructor)?;
        INITIALIZED
            .set(owner)
            .expect("exclusive BF16 projection initializer");
        Ok(())
    }
    #[cfg(test)]
    include!("bf16_projection_kernel/tests.rs");
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
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn control_bytes() -> Option<usize> {
    None
}
