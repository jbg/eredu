//! Actual CPU workers of the shared two-cut correction program. The all-axis
//! sum retains its ordinary SIMD order; the mass branch remains in the driver.
use super::*;
use safemlx::{CpuUnaryOperation, Dtype};

pub(in crate::backend::nn::workspace) fn difference(
    rank: usize,
    columns: usize,
    rows: usize,
) -> Option<CpuPopulation> {
    let elements = columns.checked_mul(rows)?;
    if !(2..=3).contains(&rank) || elements <= 1 || columns == 0 || rows == 0 {
        return None;
    }
    let mut population = CpuPopulation::default();
    for _ in 0usize..2 {
        population.copy(
            OperationEvent::cpu_softmax_layout(rank, columns, rows, false)?,
            1,
        )?;
    }
    population.binary(OperationEvent::cpu_binary_layout(
        CpuBinaryOperation::Subtract,
        Dtype::Float32,
        rank,
        elements,
        false,
    )?)?;
    // The actual maximum creates one F32 zero scalar, already counted by the
    // shared constructor/physical recipe, then one scalar Broadcast alias.
    population.copy(
        OperationEvent::cpu_broadcast_alias_layout(0, rank, false)?,
        1,
    )?;
    population.binary(OperationEvent::cpu_binary_layout(
        CpuBinaryOperation::Maximum,
        Dtype::Float32,
        rank,
        elements,
        false,
    )?)?;
    population.copy(
        OperationEvent::cpu_all_sum_layout(rank, elements, false)?,
        1,
    )?;
    population.copy(OperationEvent::cpu_squeeze_layout(rank, false)?, 1)?;
    if population.births != 5 || population.primitives != 7 {
        return None;
    }
    controls(population)
}
pub(in crate::backend::nn::workspace) fn logarithm(rank: usize) -> Option<CpuPopulation> {
    if !(2..=3).contains(&rank) {
        return None;
    }
    let mut population = CpuPopulation::default();
    population.unary(OperationEvent::cpu_unary_layout(
        CpuUnaryOperation::Log,
        Dtype::Float32,
        rank,
        false,
    )?)?;
    if population.births != 1 || population.primitives != 1 {
        return None;
    }
    controls(population)
}
fn controls(mut population: CpuPopulation) -> Option<CpuPopulation> {
    let parts = [
        size_of::<CpuPopulation>() * 3,
        size_of::<Option<CpuPopulation>>(),
        size_of::<(usize, usize, usize)>(),
        size_of::<usize>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<Option<()>>(),
    ];
    population.controls = parts.into_iter().try_fold(
        population.controls.checked_add(size_of_val(&parts))?,
        usize::checked_add,
    )?;
    Some(population)
}
