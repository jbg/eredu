//! Finite admitted source families for the existing block-FP8 Metal workers.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
mod metal {
    use eredu_runtime::working_memory::{
        InitializedSharedNative, SharedNativeInitializationCustody,
        SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
        WorkingMemoryPool,
    };
    use safemlx::fast::{
        BorrowedKernelOutput, KernelDefinitionCause, KernelDefinitionError, KernelFamilyLayout,
        KernelInputClass, KernelInputSignature, KernelSpecialization, MetalKernelDefinitionPlan,
        MetalKernelFamilyPlan, PreparedMetalKernelFamily,
    };
    use safemlx::{Array, Dtype, OriginalScopeObserver, Stream, error::Exception};
    use std::{
        mem::{size_of, size_of_val},
        sync::{
            OnceLock,
            atomic::{AtomicBool, Ordering},
        },
    };

    type Family = PreparedMetalKernelFamily<SharedNativeInitializationCustody>;
    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Quantize,
        Tiled,
        Scalar,
        GroupedTiledF32,
        GroupedTiledF16,
        GroupedTiledBf16,
        GroupedScalarF32,
        GroupedScalarF16,
        GroupedScalarBf16,
    }
    const KINDS: [Kind; 9] = [
        Kind::Quantize,
        Kind::Tiled,
        Kind::Scalar,
        Kind::GroupedTiledF32,
        Kind::GroupedTiledF16,
        Kind::GroupedTiledBf16,
        Kind::GroupedScalarF32,
        Kind::GroupedScalarF16,
        Kind::GroupedScalarBf16,
    ];
    const HEADER: &str = include_str!("../nn/fp8/header.metal");
    const QUANTIZE: MetalKernelDefinitionPlan<'static, 1, 2> = MetalKernelDefinitionPlan {
        name: "block_fp8_activation_quantization",
        inputs: ["input"],
        outputs: ["quantized", "activation_scale"],
        source: concat!(
            "constexpr uint SCALE_BLOCK=128; const uint IN_DIM=input_shape[1]; const uint SCALE_COLS=(IN_DIM+127)/128;\n",
            include_str!("../nn/fp8/activation.metal")
        ),
        header: HEADER,
        ensure_row_contiguous: true,
        atomic_outputs: false,
    };
    const TILED: MetalKernelDefinitionPlan<'static, 4, 1> = MetalKernelDefinitionPlan {
        name: "block_fp8_linear_k16",
        inputs: ["input", "input_scale", "weight", "scale"],
        outputs: ["out"],
        source: concat!(
            "constexpr uint SCALE_BLOCK=128, OUT_TILE=16, REDUCTION_TILE=16; const uint IN_DIM=input_shape[1], OUT_DIM=weight_shape[0], SCALE_COLS=input_scale_shape[1];\n",
            include_str!("../nn/fp8/linear_tiled.metal")
        ),
        header: HEADER,
        ensure_row_contiguous: true,
        atomic_outputs: false,
    };
    const SCALAR: MetalKernelDefinitionPlan<'static, 4, 1> = MetalKernelDefinitionPlan {
        name: "block_fp8_linear_scalar",
        source: concat!(
            "constexpr uint SCALE_BLOCK=128; const uint IN_DIM=input_shape[1], OUT_DIM=weight_shape[0], SCALE_COLS=input_scale_shape[1];\n",
            include_str!("../nn/fp8/linear_scalar.metal")
        ),
        ..TILED
    };
    // The actual selected row partitions are lent as rank-four shape views.
    // Their storage remains the same bank; no host scalar upload or generated
    // specialization is needed for a new independent row-block width.
    const GROUPED_TILED: MetalKernelDefinitionPlan<'static, 5, 1> = MetalKernelDefinitionPlan {
        name: "block_fp8_grouped_linear_k16",
        inputs: ["input", "input_scale", "weight", "scale", "group_ids"],
        outputs: ["out"],
        source: concat!(
            "constexpr uint SCALE_BLOCK=128, OUT_TILE=16, REDUCTION_TILE=16; const uint IN_DIM=input_shape[1], OUT_DIM=weight_shape[1]*weight_shape[2], SCALE_OUT=scale_shape[1]*scale_shape[2], SCALE_COLS=scale_shape[3], ROW_WIDTH=weight_shape[2], ROW_SCALES=scale_shape[2];\n",
            include_str!("../nn/fp8/grouped_tiled.metal")
        ),
        header: HEADER,
        ensure_row_contiguous: true,
        atomic_outputs: false,
    };
    const GROUPED_SCALAR: MetalKernelDefinitionPlan<'static, 5, 1> = MetalKernelDefinitionPlan {
        name: "block_fp8_grouped_linear_scalar",
        source: concat!(
            "constexpr uint SCALE_BLOCK=128; const uint IN_DIM=input_shape[1], OUT_DIM=weight_shape[1]*weight_shape[2], SCALE_OUT=scale_shape[1]*scale_shape[2], SCALE_COLS=scale_shape[3], ROW_WIDTH=weight_shape[2], ROW_SCALES=scale_shape[2];\n",
            include_str!("../nn/fp8/grouped_scalar.metal")
        ),
        ..GROUPED_TILED
    };
    const FLOATS: [Dtype; 3] = [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16];
    const fn quantize_signatures() -> [KernelSpecialization<'static, 1, 2>; 6] {
        let mut values = [KernelSpecialization {
            inputs: [KernelInputSignature {
                dtype: Dtype::Float32,
                class: KernelInputClass::Small,
            }],
            outputs: [Dtype::Uint8, Dtype::Float32],
            templates: &[],
        }; 6];
        let mut i = 0;
        while i < 6 {
            values[i].inputs[0].dtype = FLOATS[i / 2];
            values[i].inputs[0].class = if i % 2 == 0 {
                KernelInputClass::Small
            } else {
                KernelInputClass::Device
            };
            i += 1;
        }
        values
    }
    const fn linear_signatures() -> [KernelSpecialization<'static, 4, 1>; 48] {
        let byte = KernelInputSignature {
            dtype: Dtype::Uint8,
            class: KernelInputClass::Small,
        };
        let float = KernelInputSignature {
            dtype: Dtype::Float32,
            class: KernelInputClass::Small,
        };
        let mut values = [KernelSpecialization {
            inputs: [byte, float, byte, float],
            outputs: [Dtype::Float32],
            templates: &[],
        }; 48];
        let mut i = 0;
        while i < 48 {
            values[i].inputs[3].dtype = FLOATS[i / 16];
            let mut j = 0;
            while j < 4 {
                values[i].inputs[j].class = if i & (1 << j) == 0 {
                    KernelInputClass::Small
                } else {
                    KernelInputClass::Device
                };
                j += 1;
            }
            i += 1;
        }
        values
    }
    const fn grouped_signatures(scale: Dtype) -> [KernelSpecialization<'static, 5, 1>; 32] {
        let small = KernelInputClass::Small;
        let mut values = [KernelSpecialization {
            inputs: [
                KernelInputSignature {
                    dtype: Dtype::Uint8,
                    class: small,
                },
                KernelInputSignature {
                    dtype: Dtype::Float32,
                    class: small,
                },
                KernelInputSignature {
                    dtype: Dtype::Uint8,
                    class: small,
                },
                KernelInputSignature {
                    dtype: Dtype::Float32,
                    class: small,
                },
                KernelInputSignature {
                    dtype: Dtype::Int32,
                    class: small,
                },
            ],
            outputs: [Dtype::Float32],
            templates: &[],
        }; 32];
        let mut i = 0;
        while i < 32 {
            values[i].inputs[3].dtype = scale;
            let mut j = 0;
            while j < 5 {
                values[i].inputs[j].class = if i & (1 << j) == 0 {
                    KernelInputClass::Small
                } else {
                    KernelInputClass::Device
                };
                j += 1;
            }
            i += 1;
        }
        values
    }
    static GROUPED_TILED_F32_PLAN: MetalKernelFamilyPlan<'static, 5, 1, 32> =
        MetalKernelFamilyPlan {
            definition: GROUPED_TILED,
            specializations: grouped_signatures(Dtype::Float32),
        };
    static GROUPED_TILED_F16_PLAN: MetalKernelFamilyPlan<'static, 5, 1, 32> =
        MetalKernelFamilyPlan {
            definition: GROUPED_TILED,
            specializations: grouped_signatures(Dtype::Float16),
        };
    static GROUPED_TILED_BF16_PLAN: MetalKernelFamilyPlan<'static, 5, 1, 32> =
        MetalKernelFamilyPlan {
            definition: GROUPED_TILED,
            specializations: grouped_signatures(Dtype::Bfloat16),
        };
    static GROUPED_SCALAR_F32_PLAN: MetalKernelFamilyPlan<'static, 5, 1, 32> =
        MetalKernelFamilyPlan {
            definition: GROUPED_SCALAR,
            specializations: grouped_signatures(Dtype::Float32),
        };
    static GROUPED_SCALAR_F16_PLAN: MetalKernelFamilyPlan<'static, 5, 1, 32> =
        MetalKernelFamilyPlan {
            definition: GROUPED_SCALAR,
            specializations: grouped_signatures(Dtype::Float16),
        };
    static GROUPED_SCALAR_BF16_PLAN: MetalKernelFamilyPlan<'static, 5, 1, 32> =
        MetalKernelFamilyPlan {
            definition: GROUPED_SCALAR,
            specializations: grouped_signatures(Dtype::Bfloat16),
        };
    static QUANTIZE_PLAN: MetalKernelFamilyPlan<'static, 1, 2, 6> = MetalKernelFamilyPlan {
        definition: QUANTIZE,
        specializations: quantize_signatures(),
    };
    static TILED_PLAN: MetalKernelFamilyPlan<'static, 4, 1, 48> = MetalKernelFamilyPlan {
        definition: TILED,
        specializations: linear_signatures(),
    };
    static SCALAR_PLAN: MetalKernelFamilyPlan<'static, 4, 1, 48> = MetalKernelFamilyPlan {
        definition: SCALAR,
        specializations: linear_signatures(),
    };
    static INITIALIZED: [OnceLock<InitializedSharedNative<Family>>; 9] =
        [const { OnceLock::new() }; 9];
    static INITIALIZING: AtomicBool = AtomicBool::new(false);
    static SOURCE_QUALIFIED: OnceLock<bool> = OnceLock::new();
    struct Winner;
    impl Drop for Winner {
        fn drop(&mut self) {
            INITIALIZING.store(false, Ordering::Release);
        }
    }
    fn layout(kind: Kind) -> Result<KernelFamilyLayout, KernelDefinitionCause> {
        match kind {
            Kind::Quantize => QUANTIZE_PLAN.layout::<SharedNativeInitializationCustody>(),
            Kind::Tiled => TILED_PLAN.layout::<SharedNativeInitializationCustody>(),
            Kind::Scalar => SCALAR_PLAN.layout::<SharedNativeInitializationCustody>(),
            Kind::GroupedTiledF32 => {
                GROUPED_TILED_F32_PLAN.layout::<SharedNativeInitializationCustody>()
            }
            Kind::GroupedTiledF16 => {
                GROUPED_TILED_F16_PLAN.layout::<SharedNativeInitializationCustody>()
            }
            Kind::GroupedTiledBf16 => {
                GROUPED_TILED_BF16_PLAN.layout::<SharedNativeInitializationCustody>()
            }
            Kind::GroupedScalarF32 => {
                GROUPED_SCALAR_F32_PLAN.layout::<SharedNativeInitializationCustody>()
            }
            Kind::GroupedScalarF16 => {
                GROUPED_SCALAR_F16_PLAN.layout::<SharedNativeInitializationCustody>()
            }
            Kind::GroupedScalarBf16 => {
                GROUPED_SCALAR_BF16_PLAN.layout::<SharedNativeInitializationCustody>()
            }
        }
    }
    pub(crate) fn source_qualified() -> bool {
        *SOURCE_QUALIFIED.get_or_init(|| KINDS.into_iter().all(|kind| layout(kind).is_ok()))
    }
    pub(crate) fn static_storage_bytes() -> usize {
        size_of::<[OnceLock<InitializedSharedNative<Family>>;9]>() + size_of::<AtomicBool>() + size_of::<OnceLock<bool>>() +
        size_of_val(&QUANTIZE_PLAN)+size_of_val(&TILED_PLAN)+size_of_val(&SCALAR_PLAN)+size_of_val(&GROUPED_TILED_F32_PLAN) + size_of_val(&GROUPED_TILED_F16_PLAN) + size_of_val(&GROUPED_TILED_BF16_PLAN)+size_of_val(&GROUPED_SCALAR_F32_PLAN) + size_of_val(&GROUPED_SCALAR_F16_PLAN) + size_of_val(&GROUPED_SCALAR_BF16_PLAN)+size_of_val(&KINDS)+size_of_val(&FLOATS)+
        QUANTIZE.name.len()+QUANTIZE.source.len()+QUANTIZE.header.len()+QUANTIZE.inputs.iter().map(|s|s.len()).sum::<usize>()+QUANTIZE.outputs.iter().map(|s|s.len()).sum::<usize>()+
        [TILED,SCALAR].iter().map(|d|d.name.len()+d.source.len()+d.header.len()+d.inputs.iter().map(|s|s.len()).sum::<usize>()+d.outputs[0].len()).sum::<usize>()+
        [GROUPED_TILED,GROUPED_SCALAR].iter().map(|d|d.name.len()+d.source.len()+d.header.len()+d.inputs.iter().map(|s|s.len()).sum::<usize>()+d.outputs[0].len()).sum::<usize>()+
        include_str!("../nn/fp8/grouped_tiled.metal").len()+include_str!("../nn/fp8/grouped_scalar.metal").len()+
        // The ordinary definitions borrow the shared bodies separately from
        // the original metadata prefixes. No process-owned source is omitted.
        include_str!("../nn/fp8/activation.metal").len()+include_str!("../nn/fp8/linear_tiled.metal").len()+include_str!("../nn/fp8/linear_scalar.metal").len()+size_of::<[f32;256]>()
    }
    fn projection_control_bytes<const INPUTS: usize>() -> Option<usize> {
        let native = Family::control_bytes::<1, 2>(0, 2)?
            .checked_add(Family::control_bytes::<INPUTS, 1>(0, 2)?)?;
        [
            size_of::<&Family>(),
            size_of::<Kind>(),
            size_of::<[BorrowedKernelOutput<'_>; 2]>(),
            size_of::<[BorrowedKernelOutput<'_>; 1]>(),
            3 * size_of::<[i32; 2]>(),
            2 * size_of::<[i32; 3]>(),
            size_of::<[&Array; INPUTS]>(),
            size_of::<[Array; 2]>(),
            size_of::<[Array; 1]>(),
            size_of::<Result<[Array; 1], Exception>>(),
            size_of::<Result<[Array; 2], Exception>>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<Option<OriginalScopeObserver>>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<bool>(),
            6 * size_of::<i32>(),
        ]
        .into_iter()
        .try_fold(native, usize::checked_add)
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        projection_control_bytes::<4>()
    }
    pub(crate) fn grouped_control_bytes() -> Option<usize> {
        projection_control_bytes::<5>()?
            .checked_add(size_of::<[Array; 2]>())?
            .checked_add(size_of::<[[i32; 4]; 2]>())?
            .checked_add(size_of::<[&Array; 5]>())?
            .checked_add(size_of::<[i32; 3]>())
    }
    pub(crate) fn validate_call() -> Result<(), Exception> {
        let observer = OriginalScopeObserver::require_current()?;
        if control_bytes().is_none() || INITIALIZED.iter().any(|s| s.get().is_none()) {
            return Err(observer.capacity_error());
        }
        Ok(())
    }
    pub(crate) fn quantize(
        input: &Array,
        rows: i32,
        width: i32,
        scale_cols: i32,
        stream: &Stream,
    ) -> Result<[Array; 2], Exception> {
        let observer = OriginalScopeObserver::require_current()?;
        let family = INITIALIZED[Kind::Quantize as usize]
            .get()
            .ok_or_else(|| observer.capacity_error())?;
        let values = [rows, width];
        let scales = [rows, scale_cols];
        let grid = rows
            .checked_mul(scale_cols)
            .and_then(|n| n.checked_mul(128))
            .ok_or_else(|| observer.invalid_input_error())?;
        family.output().apply_fixed_device(
            [input],
            [
                BorrowedKernelOutput {
                    shape: &values,
                    dtype: Dtype::Uint8,
                },
                BorrowedKernelOutput {
                    shape: &scales,
                    dtype: Dtype::Float32,
                },
            ],
            &[],
            [grid, 1, 1],
            [128, 1, 1],
            stream,
        )
    }
    pub(crate) fn linear(
        inputs: [&Array; 4],
        rows: i32,
        outputs: i32,
        tiled: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let observer = OriginalScopeObserver::require_current()?;
        let kind = if tiled { Kind::Tiled } else { Kind::Scalar };
        let family = INITIALIZED[kind as usize]
            .get()
            .ok_or_else(|| observer.capacity_error())?;
        let grid = if tiled {
            [
                (outputs / 16 + i32::from(outputs % 16 != 0))
                    .checked_mul(16)
                    .ok_or_else(|| observer.invalid_input_error())?,
                rows.checked_mul(16)
                    .ok_or_else(|| observer.invalid_input_error())?,
                1,
            ]
        } else {
            [
                rows.checked_mul(outputs)
                    .ok_or_else(|| observer.invalid_input_error())?,
                1,
                1,
            ]
        };
        let shape = [rows, outputs];
        let [output] = family.output().apply_fixed_device(
            inputs,
            [BorrowedKernelOutput {
                shape: &shape,
                dtype: Dtype::Float32,
            }],
            &[],
            grid,
            if tiled { [16, 16, 1] } else { [256, 1, 1] },
            stream,
        )?;
        Ok(output)
    }
    pub(crate) fn grouped_linear(
        inputs: [&Array; 5],
        rows: i32,
        outputs: i32,
        row_width: i32,
        tiled: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let observer = OriginalScopeObserver::require_current()?;
        if row_width <= 0 || outputs <= 0 || outputs % row_width != 0 {
            return Err(observer.invalid_input_error());
        }
        let partitions = outputs / row_width;
        let row_scales = row_width / 128 + i32::from(row_width % 128 != 0);
        let weight = inputs[2].reshape(
            &[inputs[2].dim(0), partitions, row_width, inputs[2].dim(2)],
            stream,
        )?;
        let scales = inputs[3].reshape(
            &[inputs[3].dim(0), partitions, row_scales, inputs[3].dim(2)],
            stream,
        )?;
        let inputs = [inputs[0], inputs[1], &weight, &scales, inputs[4]];

        // The family catalog is finite at 64 entries. Partition its actual
        // scale dtype while preserving all 32 non-scalar input-class combinations.
        let kind = match (tiled, inputs[3].dtype()) {
            (true, Dtype::Float32) => Kind::GroupedTiledF32,
            (true, Dtype::Float16) => Kind::GroupedTiledF16,
            (true, Dtype::Bfloat16) => Kind::GroupedTiledBf16,
            (false, Dtype::Float32) => Kind::GroupedScalarF32,
            (false, Dtype::Float16) => Kind::GroupedScalarF16,
            (false, Dtype::Bfloat16) => Kind::GroupedScalarBf16,
            _ => return Err(observer.invalid_input_error()),
        };
        let family = INITIALIZED[kind as usize]
            .get()
            .ok_or_else(|| observer.capacity_error())?;
        let grid = if tiled {
            [
                (outputs / 16 + i32::from(outputs % 16 != 0))
                    .checked_mul(16)
                    .ok_or_else(|| observer.invalid_input_error())?,
                rows.checked_mul(16)
                    .ok_or_else(|| observer.invalid_input_error())?,
                1,
            ]
        } else {
            [
                rows.checked_mul(outputs)
                    .ok_or_else(|| observer.invalid_input_error())?,
                1,
                1,
            ]
        };
        let shape = [rows, outputs];
        let [output] = family.output().apply_fixed_device(
            inputs,
            [BorrowedKernelOutput {
                shape: &shape,
                dtype: Dtype::Float32,
            }],
            &[],
            grid,
            if tiled { [16, 16, 1] } else { [256, 1, 1] },
            stream,
        )?;
        Ok(output)
    }
    #[derive(Debug)]
    pub(crate) struct Initializer(Kind);
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
                .and_then(|n| n.checked_add(size_of::<MlxFp8KernelError>()))
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
            match self.0 {
                Kind::Quantize => QUANTIZE_PLAN.realize(custody),
                Kind::Tiled => TILED_PLAN.realize(custody),
                Kind::Scalar => SCALAR_PLAN.realize(custody),
                Kind::GroupedTiledF32 => GROUPED_TILED_F32_PLAN.realize(custody),
                Kind::GroupedTiledF16 => GROUPED_TILED_F16_PLAN.realize(custody),
                Kind::GroupedTiledBf16 => GROUPED_TILED_BF16_PLAN.realize(custody),
                Kind::GroupedScalarF32 => GROUPED_SCALAR_F32_PLAN.realize(custody),
                Kind::GroupedScalarF16 => GROUPED_SCALAR_F16_PLAN.realize(custody),
                Kind::GroupedScalarBf16 => GROUPED_SCALAR_BF16_PLAN.realize(custody),
            }
        }
    }
    #[derive(Debug, thiserror::Error)]
    pub(crate) enum MlxFp8KernelError {
        #[error("FP8 kernel admission: {0}")]
        Policy(#[source] WorkingMemoryError),
        #[error("FP8 kernel construction: {0}")]
        Constructor(#[source] SharedNativeInitializationError<Initializer>),
        #[error("FP8 kernel initialization is busy")]
        Busy,
    }
    pub(crate) fn prepare_admitted(pool: &WorkingMemoryPool) -> Result<(), MlxFp8KernelError> {
        if !pool.same_domain(&super::super::domain()) {
            return Err(MlxFp8KernelError::Policy(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        if INITIALIZED.iter().all(|s| s.get().is_some()) {
            for s in &INITIALIZED {
                s.get()
                    .expect("completed FP8 family")
                    .validate_pool(pool)
                    .map_err(MlxFp8KernelError::Policy)?;
            }
            return Ok(());
        }
        if INITIALIZING
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(MlxFp8KernelError::Busy);
        }
        let _winner = Winner;
        for kind in KINDS {
            let slot = &INITIALIZED[kind as usize];
            if let Some(owner) = slot.get() {
                owner
                    .validate_pool(pool)
                    .map_err(MlxFp8KernelError::Policy)?;
            } else {
                slot.set(
                    pool.initialize_shared_native(Initializer(kind))
                        .map_err(MlxFp8KernelError::Constructor)?,
                )
                .expect("exclusive FP8 family initializer");
            }
        }
        Ok(())
    }
}
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) use metal::*;
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn static_storage_bytes() -> usize {
    std::mem::size_of::<[f32; 256]>()
}
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn source_qualified() -> bool {
    false
}
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn control_bytes() -> Option<usize> {
    None
}

#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn grouped_control_bytes() -> Option<usize> {
    None
}
