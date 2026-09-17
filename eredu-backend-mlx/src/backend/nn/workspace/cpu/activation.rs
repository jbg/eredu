//! Source of the unchanged CPU-declined pointwise fallback in layers.rs.
use super::*;
use safemlx::{CpuUnaryOperation,Dtype};
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let silu=match operation.kind {
        WorkspaceOperationKindView::Elementwise("silu")=>true,
        WorkspaceOperationKindView::Elementwise("sigmoid")=>false,
        _=>return Ok(None),
    };
    if operation.inputs.len()!=1||operation.outputs.len()!=1 {
        return Err(MlxWorkspaceFactError::descriptor("CPU activation population differs"));
    }
    let input=operation.inputs.get(0).expect("activation input");
    let output=operation.outputs.get(0).expect("activation output");
    if input.shape()!=output.shape() {return Err(MlxWorkspaceFactError::descriptor("CPU activation output differs"));}
    let rank=input.shape().len();
    if rank>4||input.shape().iter().any(|&n|n<=0)||input.dtype()!=WorkspaceDtype::Float32||output.dtype()!=WorkspaceDtype::Float32 {
        return Ok(None);
    }
    let Some(representation)=input.representation() else{return Ok(None)};
    if !representation.row_contiguous() {return Ok(None);}
    let dtype=representation.dtype();
    let native=match dtype {
        WorkspaceFloatingType::Float32=>Dtype::Float32,
        WorkspaceFloatingType::Bfloat16=>Dtype::Bfloat16,
        _=>return Ok(None),
    };
    let count=usize::try_from(input.elements()?)?;
    if count>i32::MAX as usize {return Ok(None);}
    let source=(|| {
        let mut population=CpuPopulation::default();let mut scalars=0usize;
        if native==Dtype::Bfloat16 {population.copy(OperationEvent::cpu_cast_layout(native,Dtype::Float32,rank,count,false)?,1)?;}
        if silu {
            population.unary(OperationEvent::cpu_unary_layout(CpuUnaryOperation::Negative,Dtype::Float32,rank,false)?)?;
            population.unary(OperationEvent::cpu_unary_layout(CpuUnaryOperation::Exponential,Dtype::Float32,rank,false)?)?;
            // Existing exp(work).add(real F32 one), then work.divide(denominator).
            for (kind,right_scalar) in [(CpuBinaryOperation::Add,true),(CpuBinaryOperation::Divide,false)] {
                population.copy(OperationEvent::cpu_cast_layout(Dtype::Float32,Dtype::Float32,rank,count,false)?,1)?;
                let (right_rank,right_count)=if right_scalar {(0,1)}else{(rank,count)};
                let cast=OperationEvent::cpu_cast_layout(Dtype::Float32,Dtype::Float32,right_rank,right_count,false)?;
                if right_scalar {scalars=scalars.checked_add(cast.backing_births())?;}
                population.copy(cast,1)?;
                population.copy(OperationEvent::cpu_broadcast_alias_layout(rank,rank,false)?,1)?;
                population.copy(OperationEvent::cpu_broadcast_alias_layout(right_rank,rank,false)?,1)?;
                population.binary(OperationEvent::cpu_binary_layout(kind,Dtype::Float32,rank,count,false)?)?;
            }
        } else {
            population.unary(OperationEvent::cpu_unary_layout(CpuUnaryOperation::Sigmoid,Dtype::Float32,rank,false)?)?;
        }
        population.copy(OperationEvent::cpu_cast_layout(Dtype::Float32,native,rank,count,false)?,1)?;
        population.controls=population.controls.checked_add(
            crate::backend::nn::arithmetic::cpu_pointwise_probe_control_bytes()?)?;
        Some((population,scalars))
    })();
    let Some((mut population,scalar_births))=source else{return Ok(None)};
    let seeds=usize::from(silu);
    let full=mechanism.allocation.fixed_buffer_capacity(facts::mul(input.elements()?,4)?)?;
    let scalar=mechanism.allocation.fixed_buffer_capacity(4)?;
    let scratch_full=population.births.checked_sub(scalar_births).and_then(|n|n.checked_sub(1))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let scratch_bytes=facts::add(facts::mul(full,u64::try_from(scratch_full)?)?,facts::mul(scalar,
        u64::try_from(scalar_births.checked_add(seeds).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?)?)?)?;
    let frames=[size_of::<CpuPopulation>()*3,size_of::<Option<(CpuPopulation,usize)>>(),size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*2,
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<WorkspaceFloatingType>(),size_of::<Dtype>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),size_of::<[(CpuBinaryOperation,bool);2]>(),
        size_of::<std::array::IntoIter<(CpuBinaryOperation,bool),2>>(),size_of::<(CpuBinaryOperation,bool)>(),
        size_of::<(usize,usize)>(),size_of::<usize>()*7,size_of::<u64>()*3,
        size_of::<(safemlx::Array,&safemlx::Stream)>(),size_of::<safemlx::Array>()*2,
        size_of::<Result<safemlx::Array,safemlx::error::Exception>>(),size_of::<bool>()*2];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,|n,b|n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan { alias_input: None,dtype,population,output_bytes:full,scratch_bytes,rank,parameter_shells:0,seeds,validations:0}))
}
