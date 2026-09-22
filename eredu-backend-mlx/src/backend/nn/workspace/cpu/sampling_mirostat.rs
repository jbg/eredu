//! The shared adaptive cutoff and probability read use ordinary CPU primitives.
use super::program::Program;
use super::*;
use safemlx::Dtype;

fn seed(p: &mut Program, dtype: Dtype) -> Option<()> {
    p.seeds = p.seeds.checked_add(1)?;
    p.bytes = p.bytes.checked_add(p.capacity(1, dtype)?)?;
    p.controls(if dtype == Dtype::Bool {
        basic::scalar_bool_control_bytes()?
    } else {
        basic::scalar_f32_control_bytes()?
    })
}
fn comparison(
    p: &mut Program,
    kind: CpuBinaryOperation,
    rank: usize,
    width: usize,
    left_rank: usize,
    right_rank: usize,
    left_count: usize,
    right_count: usize,
) -> Option<()> {
    p.cast(Dtype::Float32, Dtype::Float32, left_rank, left_count)?;
    p.cast(Dtype::Float32, Dtype::Float32, right_rank, right_count)?;
    p.broadcast(left_rank, rank)?;
    p.broadcast(right_rank, rank)?;
    let source = OperationEvent::cpu_binary_layout(kind, Dtype::Float32, rank, width, false)?;
    p.bytes = p.bytes.checked_add(
        p.capacity(width, Dtype::Bool)?
            .checked_mul(source.backing_births() as u64)?,
    )?;
    p.native.binary(source)
}
fn select(
    p: &mut Program,
    dtype: Dtype,
    rank: usize,
    width: usize,
    condition_count: usize,
    first_rank: usize,
    first_count: usize,
) -> Option<()> {
    p.cast(Dtype::Bool, Dtype::Bool, rank, condition_count)?;
    p.cast(dtype, dtype, first_rank, first_count)?;
    p.cast(dtype, dtype, rank, width)?;
    p.broadcast(rank, rank)?;
    p.broadcast(first_rank, rank)?;
    p.broadcast(rank, rank)?;
    p.copy(
        OperationEvent::cpu_typed_select_broadcast_layout(dtype, rank, width, false)?,
        3,
        width,
        dtype,
    )
}
fn full_bool(p: &mut Program, rank: usize, elements: usize) -> Option<()> {
    seed(p, Dtype::Bool)?;
    p.broadcast(0, rank)?;
    p.copy(
        OperationEvent::cpu_scalar_full_layout(Dtype::Bool, rank, elements, false)?,
        1,
        elements,
        Dtype::Bool,
    )
}

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    use WorkspaceSamplingOperation as S;
    let WorkspaceOperationKindView::Sampling(kind) = operation.kind else {
        return Ok(None);
    };
    if !matches!(kind, S::MirostatCutoff | S::TokenProbability) {
        return Ok(None);
    }
    let (Some([input]), Some([output])) = (operation.inputs.array(), operation.outputs.array())
    else {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU adaptive sampling needs one score source and output",
        ));
    };
    let rank = input.shape().len();
    if !(2..=3).contains(&rank)
        || input.dtype() != WorkspaceDtype::Float32
        || !input
            .representation()
            .is_some_and(|r| r.dtype() == WorkspaceFloatingType::Float32 && r.row_contiguous())
        || output.dtype() != WorkspaceDtype::Float32
    {
        return Ok(None);
    }
    let width = usize::try_from(input.shape()[rank - 1])?;
    if width <= 1 || width > i32::MAX as usize || input.elements()? != width as u64 {
        return Ok(None);
    }
    if match kind {
        S::MirostatCutoff => output.shape() != input.shape(),
        S::TokenProbability => !output.shape().is_empty(),
        _ => unreachable!(),
    } {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU adaptive sampling output geometry differs",
        ));
    }
    let mut p = Program::new(mechanism);
    let source = (|| {
        p.copy(
            OperationEvent::cpu_typed_softmax_layout(Dtype::Float32, true, rank, width, 1, false)?,
            1,
            width,
            Dtype::Float32,
        )?;
        if matches!(kind, S::TokenProbability) {
            // The final scalar is a view of this full Softmax row. Its output
            // allocation is that backing; Slice and scalar Reshape allocate none.
            p.slice(rank)?;
            p.reshape(rank, 0)?;
        } else {
            seed(&mut p, Dtype::Float32)?; // cutoff = exp2(-mu)
            p.copy(
                OperationEvent::cpu_maximum_row_layout(rank, width, false)?,
                1,
                1,
                Dtype::Float32,
            )?;
            comparison(
                &mut p,
                CpuBinaryOperation::Less,
                rank,
                width,
                rank,
                0,
                width,
                1,
            )?;
            p.copy(
                OperationEvent::cpu_arg_reduce_layout(rank, width, 1, false)?,
                1,
                1,
                Dtype::Uint32,
            )?;
            p.copy(
                OperationEvent::cpu_squeeze_layout(rank, false)?,
                1,
                0,
                Dtype::Uint32,
            )?;
            p.reshape(rank - 1, rank)?;
            full_bool(&mut p, rank, width)?;
            full_bool(&mut p, rank, 1)?;
            // put_along_axis's operands already agree in dtype and shape. The
            // same overwrite task copies the full Bool row before the update.
            p.copy(
                OperationEvent::cpu_scatter_axis_layout(Dtype::Uint32, rank, width, 1, false)?,
                3,
                width,
                Dtype::Bool,
            )?;
            comparison(&mut p, CpuBinaryOperation::Greater, rank, 1, 0, rank, 1, 1)?;
            select(&mut p, Dtype::Bool, rank, width, 1, rank, width)?;
            seed(&mut p, Dtype::Float32)?; // final masked value = -infinity
            select(&mut p, Dtype::Float32, rank, width, width, 0, 1)?;
        }
        Some(())
    })();
    if source.is_none() {
        return Ok(None);
    }
    let output_bytes = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(width as u64, 4)?)?;
    let scratch_bytes = p
        .bytes
        .checked_sub(output_bytes)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames = [
        crate::backend::runtime::generation::sampling_mirostat_control_bytes()
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<Program>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<(
            &mut Program,
            Dtype,
            usize,
            usize,
            usize,
            usize,
            usize,
            Option<()>,
        )>(),
        size_of::<(
            &mut Program,
            CpuBinaryOperation,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            Option<()>,
        )>(),
        size_of::<(&mut Program, Dtype, Option<()>)>(),
        size_of::<(&mut Program, usize, usize, Option<()>)>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<usize>() * 2,
        size_of::<u64>() * 2,
        size_of::<Option<()>>(),
    ];
    p.native.controls = frames
        .into_iter()
        .try_fold(
            p.native
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population: p.native,
        alias_input: None,
        output_bytes,
        scratch_bytes,
        rank,
        parameter_shells: 0,
        seeds: p.seeds,
        validations: 0,
    }))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use crate::{
        backend::{
            managed_memory::gpu_stream::PreparedExecutionStreams,
            runtime::generation::MlxSamplingBackend, MlxBackend, MlxDeviceIdentity,
        },
        MlxTensor,
    };
    use eredu_runtime::{PenaltyConfig, SamplingBackend};
    use safemlx::{Array, Device, DeviceType};

    #[test]
    fn adaptive_cutoff_matches_nonzero_probabilities_and_keeps_the_best_fallback() {
        if !crate::tests::support::native_process::enter("cpu-adaptive-sampler") {
            return;
        }
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, choice)
            .unwrap()
            .unwrap();
        let backend = MlxBackend::for_prepared_execution_plan(
            streams,
            MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None)
                .unwrap(),
        );
        let data = [-1.5f32, 0.5, 2.0, -0.25, 1.0, -2.0, 0.0, 0.75];
        for shape in [&[1, 8][..], &[1, 1, 8][..]] {
            let input = MlxTensor::from_array(Array::from_slice(&data, shape));
            input.as_array().evaluated().unwrap();
            for mu in [-20.0f32, 2.0, 20.0] {
                let context = WorkspaceContext::new(cpu);
                let source = WorkspaceTensor::existing(
                    context
                        .layout(shape, WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32,
                            true,
                        ))),
                    &context,
                )
                .unwrap();
                context.begin_span();
                let output =
                    eredu_runtime::working_memory::WorkspaceSamplingBackend::apply_mirostat(
                        &source,
                        &[],
                        PenaltyConfig::default(),
                        0.8,
                        mu,
                        &context,
                    )
                    .unwrap();
                let report = context.finish_report(&[output]).unwrap();
                assert!(report.unpriced_operations.is_empty(), "{:?}", report);
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                    &report, ordinary, cpu, &context,
                )
                .unwrap();
                let scaled = data.map(|value| value / 0.8);
                let maximum = scaled.into_iter().fold(f32::NEG_INFINITY, f32::max);
                let sum = scaled
                    .into_iter()
                    .map(|value| (value - maximum).exp())
                    .sum::<f32>();
                let probabilities = scaled.map(|value| (value - maximum).exp() / sum);
                let cutoff = (-mu).exp2();
                let best_only = cutoff > probabilities.into_iter().fold(0.0, f32::max);
                super::super::test_execution::run(
                    recipe,
                    &backend,
                    &[&input],
                    |stream| {
                        MlxSamplingBackend::apply_mirostat(
                            &input,
                            &[],
                            PenaltyConfig::default(),
                            0.8,
                            mu,
                            stream,
                        )
                        .unwrap()
                    },
                    |actual| {
                        let evaluated = actual.as_array().evaluated().unwrap();
                        let values = evaluated.as_slice::<f32>();
                        for index in 0..data.len() {
                            let masked = if best_only {
                                index != 2
                            } else {
                                probabilities[index] < cutoff
                            };
                            if masked {
                                assert_eq!(values[index], f32::NEG_INFINITY);
                            } else {
                                assert!((values[index] - scaled[index]).abs() < 1e-6);
                            }
                        }
                    },
                );
            }
        }
    }
}
