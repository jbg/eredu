//! Finite source/cache owners for the shared reproducible row arithmetic.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
mod metal {
    use eredu_runtime::working_memory::{
        InitializedSharedNative, SharedNativeInitializationCustody,
        SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
        WorkingMemoryPool,
    };
    use safemlx::fast::{
        KernelDefinitionError, KernelFamilyLayout, KernelInputClass, KernelInputSignature,
        KernelSpecialization, MetalKernelDefinitionPlan, MetalKernelFamilyPlan,
        PreparedMetalKernelFamily,
    };
    use safemlx::Dtype;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock,
    };

    pub(crate) type Family = PreparedMetalKernelFamily<SharedNativeInitializationCustody>;
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum RowKernel {
        Rms,
        Sum,
        Softmax,
    }
    impl RowKernel {
        fn index(self) -> usize {
            self as usize
        }
    }

    pub(crate) static RMS: MetalKernelDefinitionPlan<'static, 3, 1> = MetalKernelDefinitionPlan {
        name: "f32_weightless_rms",
        inputs: ["input", "squared", "epsilon"],
        outputs: ["output"],
        source: concat!(
            "uint row=thread_position_in_grid.x; if(row>=threads_per_grid.x) return;",
            "uint width=input_shape[input_ndim-1]; size_t base=size_t(row)*width;",
            "float variance=cascade_sum_f32(squared+base,width)/float(width);",
            "float inverse=1.0f/metal::precise::sqrt(variance+epsilon[0]);",
            "for(uint i=0;i<width;++i) output[base+i]=input[base+i]*inverse;"
        ),
        header: include_str!("../nn/rms_cascade.metal"),
        ensure_row_contiguous: true,
        atomic_outputs: false,
    };
    pub(crate) static SUM: MetalKernelDefinitionPlan<'static, 1, 1> = MetalKernelDefinitionPlan {
        name: "f32_sum_last",
        inputs: ["input"],
        outputs: ["output"],
        source: concat!(
            "uint row=thread_position_in_grid.x; if(row>=threads_per_grid.x) return;",
            "uint width=input_shape[input_ndim-1];",
            "output[row]=cascade_sum_f32(input+size_t(row)*width,width);"
        ),
        header: include_str!("../nn/rms_cascade.metal"),
        ensure_row_contiguous: true,
        atomic_outputs: false,
    };
    pub(crate) static SOFTMAX: MetalKernelDefinitionPlan<'static, 1, 1> = MetalKernelDefinitionPlan {
        name: "f32_softmax_last", inputs: ["input"], outputs: ["output"],
        source: concat!(
            "uint row=thread_position_in_grid.x; if(row>=threads_per_grid.x) return;",
            "uint width=input_shape[input_ndim-1]; size_t base=size_t(row)*width;",
            "float maximum=-INFINITY; for(uint i=0;i<width;++i) { float v=input[base+i]; maximum=(isnan(v)||isnan(maximum)) ? NAN : max(maximum,v); }",
            "float sums[4]={}; for(uint i=0;i<width;++i) { float v=stable_exp_f32(input[base+i]-maximum); output[base+i]=v; sums[i%4]+=v; }",
            "float inverse=1.0f/((sums[0]+sums[2])+(sums[1]+sums[3]));",
            "for(uint i=0;i<width;++i) output[base+i]*=inverse;"),
        header: include_str!("../nn/exp_f32.metal"),
        ensure_row_contiguous: true, atomic_outputs: false,
    };
    const fn input(class: KernelInputClass) -> KernelInputSignature {
        KernelInputSignature {
            dtype: Dtype::Float32,
            class,
        }
    }
    const fn row(class: KernelInputClass) -> KernelSpecialization<'static, 1, 1> {
        KernelSpecialization {
            inputs: [input(class)],
            outputs: [Dtype::Float32],
            templates: &[],
        }
    }
    const fn rms(class: KernelInputClass) -> KernelSpecialization<'static, 3, 1> {
        KernelSpecialization {
            inputs: [input(class), input(class), input(KernelInputClass::Small)],
            outputs: [Dtype::Float32],
            templates: &[],
        }
    }
    static RMS_FAMILY: MetalKernelFamilyPlan<'static, 3, 1, 2> = MetalKernelFamilyPlan {
        definition: RMS,
        specializations: [rms(KernelInputClass::Small), rms(KernelInputClass::Device)],
    };
    static SUM_FAMILY: MetalKernelFamilyPlan<'static, 1, 1, 2> = MetalKernelFamilyPlan {
        definition: SUM,
        specializations: [row(KernelInputClass::Small), row(KernelInputClass::Device)],
    };
    static SOFTMAX_FAMILY: MetalKernelFamilyPlan<'static, 1, 1, 2> = MetalKernelFamilyPlan {
        definition: SOFTMAX,
        specializations: [row(KernelInputClass::Small), row(KernelInputClass::Device)],
    };
    static INITIALIZED: [OnceLock<InitializedSharedNative<Family>>; 3] =
        [const { OnceLock::new() }; 3];
    static INITIALIZING: AtomicBool = AtomicBool::new(false);
    struct Winner;
    impl Drop for Winner {
        fn drop(&mut self) {
            INITIALIZING.store(false, Ordering::Release);
        }
    }
    fn layout(kind: RowKernel) -> Result<KernelFamilyLayout, safemlx::fast::KernelDefinitionCause> {
        match kind {
            RowKernel::Rms => RMS_FAMILY.layout::<SharedNativeInitializationCustody>(),
            RowKernel::Sum => SUM_FAMILY.layout::<SharedNativeInitializationCustody>(),
            RowKernel::Softmax => SOFTMAX_FAMILY.layout::<SharedNativeInitializationCustody>(),
        }
    }
    // Each slot describes one immutable compiled source family, never whether
    // a shader/cache has already been initialized for a request.
    static SOURCE_QUALIFIED: [OnceLock<bool>; 3] = [const { OnceLock::new() }; 3];
    pub(crate) fn source_qualified(kind: RowKernel) -> bool {
        *SOURCE_QUALIFIED[kind.index()].get_or_init(|| layout(kind).is_ok())
    }
    pub(crate) fn sum_source_qualified() -> bool {
        source_qualified(RowKernel::Sum)
    }
    pub(crate) fn sum_control_bytes(rank: usize) -> Option<usize> {
        control_bytes(RowKernel::Sum, rank)
    }
    pub(crate) fn rms_source_qualified() -> bool {
        source_qualified(RowKernel::Rms)
    }
    pub(crate) fn rms_control_bytes(rank: usize) -> Option<usize> {
        control_bytes(RowKernel::Rms, rank)
    }
    pub(crate) fn softmax_source_qualified() -> bool {
        source_qualified(RowKernel::Softmax)
    }
    pub(crate) fn softmax_control_bytes(rank: usize) -> Option<usize> {
        control_bytes(RowKernel::Softmax, rank)
    }
    pub(crate) fn family(kind: RowKernel) -> Option<&'static Family> {
        INITIALIZED[kind.index()]
            .get()
            .map(InitializedSharedNative::output)
    }
    pub(crate) const OUTPUT_DIMENSIONS: usize = 32;
    pub(crate) fn control_bytes(kind: RowKernel, rank: usize) -> Option<usize> {
        let native = match kind {
            RowKernel::Rms => Family::control_bytes::<3, 1>(0, rank)?,
            _ => Family::control_bytes::<1, 1>(0, rank)?,
        }
        .checked_add(safemlx::Stream::device_type_control_bytes()?)?;
        [
            std::mem::size_of::<RowKernel>(),
            std::mem::size_of::<&Family>(),
            std::mem::size_of::<[i32; OUTPUT_DIMENSIONS]>(),
            std::mem::size_of::<Option<safemlx::OriginalScopeObserver>>(),
            std::mem::size_of::<safemlx::Array>() * 3,
            std::mem::size_of::<Result<Option<safemlx::Array>, safemlx::error::Exception>>(),
            std::mem::size_of::<f32>() * 2,
            std::mem::size_of::<usize>() * 3,
        ]
        .into_iter()
        .try_fold(native, usize::checked_add)
    }
    pub(crate) fn validate_call(
        kind: RowKernel,
        rank: usize,
    ) -> Result<(), safemlx::error::Exception> {
        if let Some(observer) = safemlx::OriginalScopeObserver::try_current()? {
            if rank > OUTPUT_DIMENSIONS
                || control_bytes(kind, rank).is_none()
                || family(kind).is_none()
            {
                return Err(observer.capacity_error());
            }
        }
        Ok(())
    }
    pub(crate) fn apply<const I: usize>(
        kind: RowKernel,
        inputs: [&safemlx::Array; I],
        shape: &[i32],
        rows: i32,
        stream: &safemlx::Stream,
    ) -> Result<safemlx::Array, safemlx::error::Exception> {
        use safemlx::fast::{BorrowedKernelOutput, CustomKernelConfig, MetalKernel};
        use std::cell::RefCell;
        thread_local! {
            static ORDINARY: [RefCell<Option<MetalKernel>>; 3] = [const { RefCell::new(None) }; 3];
        }
        let output = [BorrowedKernelOutput {
            shape,
            dtype: Dtype::Float32,
        }];
        if let Some(observer) = safemlx::OriginalScopeObserver::try_current()? {
            let family = family(kind).ok_or_else(|| observer.capacity_error())?;
            let [output] =
                family.apply_fixed_device(inputs, output, &[], [rows, 1, 1], [32, 1, 1], stream)?;
            return Ok(output);
        }
        ORDINARY.with(|kernels| {
            let cell = &kernels[kind.index()];
            if cell.borrow().is_none() {
                fn build<const N: usize>(
                    p: &MetalKernelDefinitionPlan<'_, N, 1>,
                ) -> Result<MetalKernel, safemlx::error::Exception> {
                    MetalKernel::new(
                        p.name,
                        p.inputs,
                        p.outputs,
                        p.source,
                        p.header,
                        p.ensure_row_contiguous,
                        p.atomic_outputs,
                    )
                }
                *cell.borrow_mut() = Some(match kind {
                    RowKernel::Rms => build(&RMS),
                    RowKernel::Sum => build(&SUM),
                    RowKernel::Softmax => build(&SOFTMAX),
                }?);
            }
            let loan = cell.borrow();
            let kernel = loan.as_ref().expect("row kernel initialized");
            if MetalKernel::fixed_control_bytes::<I, 1>(0, shape.len()).is_some() {
                let [output] = kernel.apply_fixed_device(
                    inputs,
                    output,
                    &[],
                    [rows, 1, 1],
                    [32, 1, 1],
                    stream,
                )?;
                Ok(output)
            } else {
                let config = CustomKernelConfig::new()
                    .with_grid([rows, 1, 1])
                    .with_thread_group([32, 1, 1])
                    .with_output_arg(shape, Dtype::Float32);
                let mut outputs = kernel.apply_device(inputs, &config, stream)?;
                Ok(outputs.pop().expect("row kernel has one declared output"))
            }
        })
    }
    fn definition_bytes<const I: usize>(definition: &MetalKernelDefinitionPlan<'_, I, 1>) -> usize {
        std::mem::size_of_val(definition)
            + definition.name.len()
            + definition.source.len()
            + definition.header.len()
            + definition.inputs.iter().map(|v| v.len()).sum::<usize>()
            + definition.outputs.iter().map(|v| v.len()).sum::<usize>()
    }
    pub(crate) fn static_storage_bytes() -> usize {
        // The shared Metal preamble/retirement machinery is already counted by
        // the pointwise family in this same process baseline.
        std::mem::size_of_val(&INITIALIZED)
            + std::mem::size_of_val(&SOURCE_QUALIFIED)
            + std::mem::size_of_val(&INITIALIZING)
            + std::mem::size_of_val(&RMS_FAMILY)
            + std::mem::size_of_val(&SUM_FAMILY)
            + std::mem::size_of_val(&SOFTMAX_FAMILY)
            + definition_bytes(&RMS)
            + definition_bytes(&SUM)
            + definition_bytes(&SOFTMAX)
    }
    #[derive(Debug)]
    pub(crate) struct Initializer(RowKernel);
    impl SharedNativeInitializer for Initializer {
        type Output = Family;
        type Error = KernelDefinitionError<SharedNativeInitializationCustody>;
        fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
            layout(self.0)
                .map_err(|e| match e {
                    safemlx::fast::KernelDefinitionCause::Overflow => WorkingMemoryError::Overflow,
                    _ => WorkingMemoryError::UnknownBound,
                })?
                .required_bytes()
                .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
                .and_then(|n| n.checked_add(std::mem::size_of::<Winner>()))
                .and_then(|n| n.checked_add(std::mem::size_of::<MlxRowKernelError>()))
                .and_then(|n| {
                    n.checked_add(std::mem::size_of::<
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
                RowKernel::Rms => RMS_FAMILY.realize(custody),
                RowKernel::Sum => SUM_FAMILY.realize(custody),
                RowKernel::Softmax => SOFTMAX_FAMILY.realize(custody),
            }
        }
    }
    #[derive(Debug, thiserror::Error)]
    pub(crate) enum MlxRowKernelError {
        #[error("row kernel admission: {0}")]
        Policy(#[source] WorkingMemoryError),
        #[error("row kernel construction: {0}")]
        Constructor(#[source] SharedNativeInitializationError<Initializer>),
        #[error("row kernel initialization is busy")]
        Busy,
    }
    pub(crate) fn prepare_admitted(pool: &WorkingMemoryPool) -> Result<(), MlxRowKernelError> {
        if !pool.same_domain(&super::super::domain()) {
            return Err(MlxRowKernelError::Policy(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        if INITIALIZED.iter().all(|slot| slot.get().is_some()) {
            for slot in &INITIALIZED {
                slot.get()
                    .expect("immutable completed row family")
                    .validate_pool(pool)
                    .map_err(MlxRowKernelError::Policy)?;
            }
            return Ok(());
        }
        if INITIALIZING
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(MlxRowKernelError::Busy);
        }
        let _winner = Winner;
        for kind in [RowKernel::Rms, RowKernel::Sum, RowKernel::Softmax] {
            let slot = &INITIALIZED[kind.index()];
            if let Some(owner) = slot.get() {
                owner
                    .validate_pool(pool)
                    .map_err(MlxRowKernelError::Policy)?;
            } else {
                let owner = pool
                    .initialize_shared_native(Initializer(kind))
                    .map_err(MlxRowKernelError::Constructor)?;
                slot.set(owner).expect("exclusive row kernel initializer");
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use safemlx::{fast::BorrowedKernelOutput, Array, Device, DeviceType, Stream};

        #[test]
        fn row_families_use_actual_dimensions_and_match_scalar_equations() {
            if std::env::var_os("EREDU_REQUIRE_QUALIFIED_ROW_FAMILY").is_some() {
                assert!([RowKernel::Rms, RowKernel::Sum, RowKernel::Softmax]
                    .into_iter()
                    .all(source_qualified));
            }
            if !source_qualified(RowKernel::Rms) {
                return;
            }
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let rms = pool
                .initialize_shared_native(Initializer(RowKernel::Rms))
                .unwrap();
            let sum = pool
                .initialize_shared_native(Initializer(RowKernel::Sum))
                .unwrap();
            let softmax = pool
                .initialize_shared_native(Initializer(RowKernel::Softmax))
                .unwrap();
            let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
            let epsilon = Array::try_from_slice(&[0.0003f32], &[1]).unwrap();
            for (rows, width) in [(1, 1), (1, 3), (1, 4), (3, 4), (3, 9), (2, 32)] {
                let values: Vec<f32> = (0..rows * width)
                    .map(|i| (i % 13) as f32 * 0.25 - 1.5)
                    .collect();
                let shape = [1, rows, width];
                let input = Array::try_from_slice(&values, &shape).unwrap();
                let squared = input.square(&stream).unwrap();
                let [rms_output] = rms
                    .output()
                    .apply_fixed_device(
                        [&input, &squared, &epsilon],
                        [BorrowedKernelOutput {
                            shape: &shape,
                            dtype: Dtype::Float32,
                        }],
                        &[],
                        [rows, 1, 1],
                        [32, 1, 1],
                        &stream,
                    )
                    .unwrap();
                let reduced_shape = [1, rows, 1];
                let [sum_output] = sum
                    .output()
                    .apply_fixed_device(
                        [&input],
                        [BorrowedKernelOutput {
                            shape: &reduced_shape,
                            dtype: Dtype::Float32,
                        }],
                        &[],
                        [rows, 1, 1],
                        [32, 1, 1],
                        &stream,
                    )
                    .unwrap();
                let rms_values = rms_output.evaluated().unwrap().try_to_vec::<f32>().unwrap();
                let sum_values = sum_output.evaluated().unwrap().try_to_vec::<f32>().unwrap();
                for (r, row) in values.chunks(width as usize).enumerate() {
                    assert_eq!(sum_values[r], row.iter().sum::<f32>());
                    let variance =
                        row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / f64::from(width);
                    for (i, x) in row.iter().enumerate() {
                        let expected = f64::from(*x) / (variance + f64::from(0.0003f32)).sqrt();
                        assert!(
                            (f64::from(rms_values[r * width as usize + i]) - expected).abs() < 2e-6
                        );
                    }
                }
                if width >= 4 {
                    let [output] = softmax
                        .output()
                        .apply_fixed_device(
                            [&input],
                            [BorrowedKernelOutput {
                                shape: &shape,
                                dtype: Dtype::Float32,
                            }],
                            &[],
                            [rows, 1, 1],
                            [32, 1, 1],
                            &stream,
                        )
                        .unwrap();
                    let actual = output.evaluated().unwrap().try_to_vec::<f32>().unwrap();
                    for (r, row) in values.chunks(width as usize).enumerate() {
                        let denominator = row.iter().map(|x| f64::from(*x).exp()).sum::<f64>();
                        for (i, x) in row.iter().enumerate() {
                            assert!(
                                (f64::from(actual[r * width as usize + i])
                                    - f64::from(*x).exp() / denominator)
                                    .abs()
                                    < 2e-6
                            );
                        }
                    }
                }
            }
            stream.synchronize().unwrap();
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
pub(crate) fn rms_source_qualified() -> bool {
    false
}
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn rms_control_bytes(_: usize) -> Option<usize> {
    None
}

#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn softmax_source_qualified() -> bool {
    false
}
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn softmax_control_bytes(_: usize) -> Option<usize> {
    None
}

#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn sum_source_qualified() -> bool { false }
#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
pub(crate) fn sum_control_bytes(_: usize) -> Option<usize> { None }
