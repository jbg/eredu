//! CPU populations of the shared explicit-key random workers. Shape-derived
//! draw storage is priced from the actual selected source; bodies stay ordinary.
use super::*;
use safemlx::CpuUnaryOperation;

pub(in crate::backend::nn::workspace) fn uniform_unit_interval()->Option<CpuPopulation>{
    uniform(1,1)
}
pub(in crate::backend::nn::workspace) fn categorical(rank:usize,columns:usize)->Option<CpuPopulation>{
    let mut population=split_and_views()?;
    population.add(categorical_draw(rank,columns)?)?;
    compose_controls(&mut population)?;
    Some(population)
}
pub(super) fn categorical_draw(rank:usize,columns:usize)->Option<CpuPopulation>{
    if !(2..=3).contains(&rank)||columns<=1{return None;}
    let mut population=uniform_draw(rank,columns)?;
    // Native random::gumbel retains its nested -log(-log(uniform)) order.
    for operation in [CpuUnaryOperation::Log,CpuUnaryOperation::Negative,
        CpuUnaryOperation::Log,CpuUnaryOperation::Negative] {
        population.unary(OperationEvent::cpu_unary_layout(operation,safemlx::Dtype::Float32,rank,false)?)?;
    }
    population.binary(OperationEvent::cpu_binary_layout(CpuBinaryOperation::Add,safemlx::Dtype::Float32,rank,columns,false)?)?;
    population.copy(OperationEvent::cpu_arg_reduce_layout(rank,columns,1,false)?,1)?;
    population.copy(OperationEvent::cpu_squeeze_layout(rank,false)?,1)?;
    if population.primitives!=18||population.births!=13{return None;}
    let parts=[size_of::<CpuPopulation>()*2,size_of::<Option<CpuPopulation>>(),
        size_of::<(usize,usize)>(),size_of::<[CpuUnaryOperation;4]>(),
        size_of::<std::array::IntoIter<CpuUnaryOperation,4>>(),size_of::<CpuUnaryOperation>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),size_of::<Option<()>>()];
    population.controls=parts.into_iter().try_fold(population.controls.checked_add(size_of_val(&parts))?,usize::checked_add)?;
    Some(population)
}
fn split_and_views()->Option<CpuPopulation>{
    let mut population=CpuPopulation::default();
    population.copy(OperationEvent::cpu_random_bits_layout(2,4,false)?,1)?;
    for _ in 0usize..2 {
        population.copy(OperationEvent::cpu_slice_layout(2, false, false)?,1)?;
        population.copy(OperationEvent::cpu_reshape_alias_layout(2,1,false)?,1)?;
    }
    let parts=[size_of::<CpuPopulation>(),size_of::<Option<CpuPopulation>>(),
        size_of::<std::ops::Range<usize>>(),size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),size_of::<Option<()>>()];
    population.controls=parts.into_iter().try_fold(population.controls.checked_add(size_of_val(&parts))?,usize::checked_add)?;
    Some(population)
}
fn compose_controls(population:&mut CpuPopulation)->Option<()> {
    let parts=[size_of::<CpuPopulation>()*2,size_of::<Option<CpuPopulation>>(),
        size_of::<(usize,usize)>(),size_of::<Option<()>>()];
    population.controls=parts.into_iter().try_fold(population.controls.checked_add(size_of_val(&parts))?,usize::checked_add)?;
    Some(())
}
fn uniform(rank:usize,elements:usize)->Option<CpuPopulation>{
    let mut population=split_and_views()?;
    population.add(uniform_draw(rank,elements)?)?;
    compose_controls(&mut population)?;
    Some(population)
}
pub(super) fn uniform_draw(rank:usize,elements:usize)->Option<CpuPopulation>{
    if !(1..=3).contains(&rank)||elements==0||elements>i32::MAX as usize{return None;}
    let mut population=CpuPopulation::default();
    // F32 low/high are eager scalars. Their casts and final F32 restoration are
    // identities. The actual scalar subtraction computes the range.
    population.binary(OperationEvent::cpu_binary_layout(CpuBinaryOperation::Subtract,safemlx::Dtype::Float32,0,1,false)?)?;
    population.copy(OperationEvent::cpu_random_bits_layout(rank,elements,false)?,1)?;
    population.copy(OperationEvent::cpu_cast_layout(safemlx::Dtype::Uint32,safemlx::Dtype::Float32,rank,elements,false)?,1)?;
    // maxval, nextafter(1,0), range, low broadcast to the actual draw shape.
    for operation in [CpuBinaryOperation::Divide,CpuBinaryOperation::Minimum,
        CpuBinaryOperation::Multiply,CpuBinaryOperation::Add] {
        population.copy(OperationEvent::cpu_broadcast_alias_layout(0,rank,false)?,1)?;
        population.binary(OperationEvent::cpu_binary_layout(operation,safemlx::Dtype::Float32,rank,elements,false)?)?;
    }
    if population.births!=7||population.primitives!=11{return None;}
    let parts=[size_of::<CpuPopulation>(),size_of::<Option<CpuPopulation>>(),size_of::<(usize,usize)>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<[CpuBinaryOperation;4]>(),size_of::<std::array::IntoIter<CpuBinaryOperation,4>>(),
        size_of::<std::ops::Range<usize>>(),size_of::<CpuBinaryOperation>(),size_of::<Option<()>>()];
    population.controls=parts.into_iter().try_fold(population.controls.checked_add(size_of_val(&parts))?,usize::checked_add)?;
    Some(population)
}
