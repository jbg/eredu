//! CPU constructors and typed backings of the shared logits penalty worker.
use super::*;
use safemlx::Dtype;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::Sampling(WorkspaceSamplingOperation::Penalties {
        repetition,
        additive,
        ..
    }) = operation.kind
    else {
        return Ok(None);
    };
    let Some([input]) = operation.inputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU penalties need one logits input",
        ));
    };
    let Some([output]) = operation.outputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU penalties need one logits output",
        ));
    };
    if input.shape() != output.shape()
        || input.dtype() != WorkspaceDtype::Float32
        || output.dtype() != input.dtype()
        || (!*repetition && !*additive)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU penalties change their floating geometry",
        ));
    }
    let rank = input.shape().len();
    if !(2..=3).contains(&rank)
        || input.shape().iter().any(|&n| n <= 0)
        || input.representation()
            != Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            ))
    {
        return Ok(None);
    }
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize {
        return Ok(None);
    }
    let source = (|| {
        let mut population = CpuPopulation::default();
        // Full F32, full Bool, and F32 scalar backings have distinct capacities.
        let mut births = [0usize; 3];
        let mut seeds = 0usize;
        if *repetition {
            births[1] = 1; // the uploaded repetition mask
            births[2] = 3; // two independent penalty scalars and the zero predicate
            seeds = 4;
            for kind in [
                CpuBinaryOperation::Divide,
                CpuBinaryOperation::Multiply,
                CpuBinaryOperation::Greater,
            ] {
                for (r, n, bucket) in [(rank, count, 0), (0, 1, 2)] {
                    let cast = OperationEvent::cpu_cast_layout(
                        Dtype::Float32,
                        Dtype::Float32,
                        r,
                        n,
                        false,
                    )?;
                    births[bucket] = births[bucket].checked_add(cast.backing_births())?;
                    population.copy(cast, 1)?;
                    population.copy(
                        OperationEvent::cpu_broadcast_alias_layout(r, rank, false)?,
                        1,
                    )?;
                }
                let binary =
                    OperationEvent::cpu_binary_layout(kind, Dtype::Float32, rank, count, false)?;
                let bucket = usize::from(matches!(kind, CpuBinaryOperation::Greater));
                births[bucket] = births[bucket].checked_add(binary.backing_births())?;
                population.binary(binary)?;
            }
            // Both where calls have equal-shape Bool/F32/F32 operands. Their
            // Select workers read complete rows, with no scalar branch broadcast.
            for _ in 0..2 {
                for dtype in [Dtype::Bool, Dtype::Float32, Dtype::Float32] {
                    let cast = OperationEvent::cpu_cast_layout(dtype, dtype, rank, count, false)?;
                    let bucket = usize::from(dtype == Dtype::Bool);
                    births[bucket] = births[bucket].checked_add(cast.backing_births())?;
                    population.copy(cast, 1)?;
                    population.copy(
                        OperationEvent::cpu_broadcast_alias_layout(rank, rank, false)?,
                        1,
                    )?;
                }
                let select = OperationEvent::cpu_select_layout(Dtype::Float32, rank, count, false)?;
                births[0] = births[0].checked_add(select.backing_births())?;
                population.copy(select, 3)?;
            }
            population.maximum_captures = population.maximum_captures.max(5);
        }
        if *additive {
            births[0] = births[0].checked_add(1)?; // uploaded F32 frequency/presence vector
            seeds = seeds.checked_add(1)?;
            for _ in 0..2 {
                let cast = OperationEvent::cpu_cast_layout(
                    Dtype::Float32,
                    Dtype::Float32,
                    rank,
                    count,
                    false,
                )?;
                births[0] = births[0].checked_add(cast.backing_births())?;
                population.copy(cast, 1)?;
                population.copy(
                    OperationEvent::cpu_broadcast_alias_layout(rank, rank, false)?,
                    1,
                )?;
            }
            let subtract = OperationEvent::cpu_binary_layout(
                CpuBinaryOperation::Subtract,
                Dtype::Float32,
                rank,
                count,
                false,
            )?;
            births[0] = births[0].checked_add(subtract.backing_births())?;
            population.binary(subtract)?;
        }
        Some((population, births, seeds))
    })();
    let Some((mut population, births, seeds)) = source else {
        return Ok(None);
    };
    let full = mechanism.allocation.fixed_buffer_capacity(input.bytes()?)?;
    let mask = mechanism.allocation.fixed_buffer_capacity(count as u64)?;
    let scalar = mechanism.allocation.fixed_buffer_capacity(4)?;
    let total = facts::add(
        facts::add(
            facts::mul(full, births[0] as u64)?,
            facts::mul(mask, births[1] as u64)?,
        )?,
        facts::mul(scalar, births[2] as u64)?,
    )?;
    let scratch_bytes = total
        .checked_sub(full)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames = [
        crate::backend::runtime::generation::sampling_penalty_control_bytes()
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<CpuPopulation>(),
        size_of::<[usize; 3]>(),
        size_of::<Option<(CpuPopulation, [usize; 3], usize)>>(),
        size_of::<CpuCopyEvalLayout>() * 2,
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<[CpuBinaryOperation; 3]>(),
        size_of::<std::array::IntoIter<CpuBinaryOperation, 3>>(),
        size_of::<[(usize, usize, usize); 2]>(),
        size_of::<std::array::IntoIter<(usize, usize, usize), 2>>(),
        size_of::<[Dtype; 3]>(),
        size_of::<std::array::IntoIter<Dtype, 3>>(),
        size_of::<std::ops::Range<usize>>() * 2,
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<usize>() * 8,
        size_of::<u64>() * 4,
        size_of::<Dtype>(),
        size_of::<CpuBinaryOperation>(),
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<[WorkspaceLayoutView<'_>; 1]>() * 2,
        size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>() * 2,
    ];
    population.controls = frames
        .into_iter()
        .try_fold(
            population
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population,
        alias_input: None,
        output_bytes: full,
        scratch_bytes,
        rank,
        parameter_shells: 0,
        seeds,
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

    #[test]
    fn cpu_penalty_windows_cover_host_payload_and_complete_native_records() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let matmul =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), matmul);
        for shape in [&[1, 1][..], &[2, 17][..], &[1, 1, 65][..]] {
            for history_positions in [0, 1, 32] {
                for (repetition, additive) in [(true, false), (false, true), (true, true)] {
                    let context = WorkspaceContext::new(cpu);
                    let layout = context
                        .layout(shape, WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32,
                            true,
                        )));
                    let input = WorkspaceTensor::existing(layout, &context).unwrap();
                    let output = context
                        .execute(
                            WorkspaceOperationKind::Sampling(
                                WorkspaceSamplingOperation::Penalties {
                                    history_positions,
                                    repetition,
                                    additive,
                                },
                            ),
                            &[&input],
                            vec![input.layout().clone()],
                        )
                        .unwrap()
                        .remove(0);
                    assert_eq!(
                        output.layout().representation(),
                        input.layout().representation()
                    );
                    let report = context.finish_report(&[output]).unwrap();
                    assert!(
                        report.unpriced_operations.is_empty()
                            && report.unpriced_host_operations.is_empty()
                    );
                    assert_eq!(
                        report.host_workspace_bytes,
                        Some(input.layout().elements().unwrap() * 5 + history_positions as u64 * 4)
                    );
                    let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                    assert_eq!(
                        plan.seeds,
                        usize::from(repetition) * 4 + usize::from(additive)
                    );
                    assert_eq!(
                        plan.population.primitives,
                        usize::from(repetition) * 29 + usize::from(additive) * 5
                    );
                    assert_eq!(
                        plan.population.input_edges,
                        usize::from(repetition) * 36 + usize::from(additive) * 6
                    );
                    let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                        &report, ordinary, cpu, &context,
                    )
                    .unwrap();
                    assert!(recipe.graph_capacity > 0 && recipe.record_capacity > 0);
                    assert_eq!(recipe.kernels, 0);
                    let mut missing = report.operations[0].clone();
                    missing.inputs[0] = missing.inputs[0].clone().with_representation(None);
                    assert!(cpu.plan(missing.as_view()).unwrap().is_none());
                }
            }
        }
    }
}
