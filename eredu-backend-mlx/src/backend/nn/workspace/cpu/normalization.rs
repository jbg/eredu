//! Actual CPU RMS fallbacks: shared row sum/divide and cast-before-gain.
use super::*;
use eredu_nn::NormalizationScale;
use safemlx::{CpuUnaryOperation, Dtype};
mod weightless_half;
mod grouped;
#[cfg(test)]
mod grouped_tests;

#[derive(Clone, Copy)]
enum Buffer { Full=0, Row=1, Vector=2, Scalar=3 }
#[derive(Clone, Copy, Default)]
struct Population { native: CpuPopulation, births: [usize;4] }
impl Population {
    fn birth(&mut self,buffer:Buffer,count:usize)->Option<()> {
        let slot=&mut self.births[buffer as usize];*slot=slot.checked_add(count)?;Some(())
    }
    fn copy(&mut self,source:CpuCopyEvalLayout,buffer:Option<Buffer>)->Option<()> {
        if let Some(buffer)=buffer {self.birth(buffer,source.backing_births())?;}
        self.native.copy(source,1)
    }
    fn cast(&mut self,from:Dtype,to:Dtype,rank:usize,count:usize,buffer:Buffer)->Option<()> {
        self.copy(OperationEvent::cpu_cast_layout(from,to,rank,count,false)?,Some(buffer))
    }
    fn alias(&mut self,from:usize,to:usize)->Option<()> {
        self.copy(OperationEvent::cpu_broadcast_alias_layout(from,to,false)?,None)
    }
    fn unary(&mut self,kind:CpuUnaryOperation,rank:usize,buffer:Buffer)->Option<()> {
        let source=OperationEvent::cpu_unary_layout(kind,Dtype::Float32,rank,false)?;
        self.birth(buffer,source.backing_births())?;self.native.unary(source)
    }
    fn binary(&mut self,kind:CpuBinaryOperation,dtype:Dtype,rank:usize,count:usize,
        left:Buffer,right:Buffer,right_rank:usize,right_count:usize,output:Buffer)->Option<()> {
        // Shared cast/broadcast frontends may return identity views. Count
        // their real constructors without assuming that optimization.
        self.cast(dtype,dtype,rank,count,left)?;
        self.cast(dtype,dtype,right_rank,right_count,right)?;
        self.alias(rank,rank)?;self.alias(right_rank,rank)?;
        let source=OperationEvent::cpu_binary_layout(kind,dtype,rank,count,false)?;
        self.birth(output,source.backing_births())?;self.native.binary(source)
    }
}
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    if let WorkspaceOperationKindView::ConstructedNormalization(spec)=operation.kind {
        if let Some(groups)=spec.groups {return grouped::inspect(operation,mechanism,spec,groups);}
    }
    let (learned,constructed)=match operation.kind {
        WorkspaceOperationKindView::Normalization("rms",None)=>(operation.inputs.len()==2,false),
        WorkspaceOperationKindView::ConstructedNormalization(spec)=>{
            spec.validate_fixed()?;
            match &spec.scale {
                NormalizationScale::Learned(_)=>(true,true),
                NormalizationScale::Unit=>(false,true),
                NormalizationScale::LearnedOffset{..}=>return Ok(None),
            }
        }
        _=>return Ok(None),
    };
    if operation.outputs.len()!=1||operation.inputs.len()!=1+usize::from(learned) {
        return Err(MlxWorkspaceFactError::descriptor("CPU RMS operand population differs"));
    }
    let input=operation.inputs.get(0).expect("RMS input");
    let output=operation.outputs.get(0).expect("RMS output");
    let rank=input.shape().len();
    if !(1..=4).contains(&rank)||input.shape().iter().any(|&n|n<=0) {return Ok(None);}
    if input.shape()!=output.shape() {return Err(MlxWorkspaceFactError::descriptor("CPU RMS output shape differs"));}
    let Some(input_dtype)=input.representation().map(|r|r.dtype()) else{return Ok(None)};
    let gain_dtype=if learned {let Some(gain)=operation.inputs.get(1).and_then(|gain|gain.representation())
        else{return Ok(None)};gain.dtype()}else{input_dtype};
    let promoted=super::super::representation::promote(input_dtype,gain_dtype);
    let native_dtype=|dtype|match dtype {
        WorkspaceFloatingType::Float32=>Dtype::Float32,
        WorkspaceFloatingType::Bfloat16=>Dtype::Bfloat16,
        WorkspaceFloatingType::Float16=>Dtype::Float16,
    };
    let input_native=native_dtype(input_dtype);let gain_native=native_dtype(gain_dtype);
    let native=native_dtype(promoted);
    // The constructed module returns the promoted result. The shared Tensor
    // adapter explicitly casts the completed normalization back to input dtype.
    let dtype=if learned&&!constructed {input_dtype}else{promoted};
    for (index,value) in operation.inputs.iter().enumerate() {
        if value.dtype()!=WorkspaceDtype::Float32||!value.representation().is_some_and(|r|
            if index==0 {r.last_axis_contiguous()} else {r.row_contiguous()}) {
            return Ok(None);
        }
    }
    if !input.representation().is_some_and(|r|r.row_contiguous()) && input.shape()[rank-1]<=1 {
        return Ok(None);
    }
    if output.dtype()!=WorkspaceDtype::Float32 {return Ok(None);}
    let width=usize::try_from(input.shape()[rank-1])?;
    if operation.inputs.get(1).is_some_and(|w|w.shape()!=[input.shape()[rank-1]]) {
        return Err(MlxWorkspaceFactError::descriptor("CPU RMS gain shape differs"));
    }
    if let WorkspaceOperationKindView::ConstructedNormalization(spec)=operation.kind {
        if spec.dimensions!=input.shape()[rank-1] {return Err(MlxWorkspaceFactError::descriptor("CPU RMS construction width differs"));}
    }
    let count=usize::try_from(input.elements()?)?;let rows=count/width;
    if count>i32::MAX as usize {return Ok(None);}
    let source=(|| {
        if !learned && input_dtype!=WorkspaceFloatingType::Float32 {
            return weightless_half::source(input_native,rank,width,rows,count);
        }
        let mut p=Population::default();
        if learned {
            // Only equal half inputs enter input_precision_rms's GPU probe.
            // CPU declines it; its actual unused cast still has a constructor.
            if input_dtype==gain_dtype&&input_dtype!=WorkspaceFloatingType::Float32 {
                p.cast(input_native,Dtype::Float32,rank,count,Buffer::Full)?;
            }
            p.cast(gain_native,native,1,width,Buffer::Vector)?;
            p.cast(input_native,Dtype::Float32,rank,count,Buffer::Full)?;
        }
        p.unary(CpuUnaryOperation::Square,rank,Buffer::Full)?;
        p.copy(OperationEvent::cpu_row_sum_layout(rank,width,rows,false)?,Some(Buffer::Row))?;
        // mean divides once by the width scalar; no reciprocal rewrite.
        p.binary(CpuBinaryOperation::Divide,Dtype::Float32,rank,rows,
            Buffer::Row,Buffer::Scalar,0,1,Buffer::Row)?;
        p.binary(CpuBinaryOperation::Add,Dtype::Float32,rank,rows,
            Buffer::Row,Buffer::Scalar,0,1,Buffer::Row)?;
        p.unary(CpuUnaryOperation::Rsqrt,rank,Buffer::Row)?;
        p.binary(CpuBinaryOperation::Multiply,Dtype::Float32,rank,count,
            Buffer::Full,Buffer::Row,rank,rows,Buffer::Full)?;
        p.cast(Dtype::Float32,native,rank,count,Buffer::Full)?;
        if learned {
            p.binary(CpuBinaryOperation::Multiply,native,rank,count,
                Buffer::Full,Buffer::Vector,1,width,Buffer::Full)?;
            if !constructed {p.cast(native,input_native,rank,count,Buffer::Full)?;}
        }
        p.native.controls=p.native.controls.checked_add(
            OperationEvent::cpu_rms_fallback_control_bytes(native,rank,width,rows)?)?;
        p.native.controls=p.native.controls.checked_add(safemlx::Stream::device_type_control_bytes()?)?;
        Some(p)
    })();
    let Some(mut source)=source else{return Ok(None)};
    if source.births.into_iter().try_fold(0usize,usize::checked_add)!=Some(source.native.births) {
        return Err(MlxWorkspaceFactError::descriptor("CPU RMS physical source population differs"));
    }
    let seeds=2usize; // separately owned actual mean width and epsilon scalars.
    let full=mechanism.allocation.fixed_buffer_capacity(facts::mul(input.elements()?,4)?)?;
    let row=mechanism.allocation.fixed_buffer_capacity(facts::mul(u64::try_from(rows)?,4)?)?;
    let vector=mechanism.allocation.fixed_buffer_capacity(facts::mul(u64::try_from(width)?,4)?)?;
    let scalar=mechanism.allocation.fixed_buffer_capacity(4)?;
    let total=source.births.into_iter().zip([full,row,vector,scalar]).try_fold(
        facts::mul(scalar,u64::try_from(seeds)?)?,|n,(births,bytes)|facts::add(n,facts::mul(bytes,u64::try_from(births)?)?))?;
    let scratch_bytes=total.checked_sub(full).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames=[size_of::<Population>()*3,size_of::<Option<Population>>(),size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*3,
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<WorkspaceFloatingType>()*4,size_of::<Dtype>()*3,
        size_of::<Option<WorkspaceRepresentation>>(),size_of::<WorkspaceRepresentation>(),
        size_of::<(CpuBinaryOperation,Dtype,usize,usize,Buffer,Buffer,usize,usize,Buffer)>(),
        size_of::<(Dtype,Dtype,usize,usize,Buffer)>(),size_of::<(CpuUnaryOperation,usize,Buffer)>(),
        size_of::<(&mut Population,CpuCopyEvalLayout,Option<Buffer>)>(),size_of::<(&mut Population,Buffer,usize)>(),
        size_of::<(&mut Population,usize,usize)>(),size_of::<Option<Buffer>>(),size_of::<Option<()>>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<(&safemlx::Array,&safemlx::Array,f32,&safemlx::Stream)>(),
        size_of::<(&safemlx::Array,f32,&safemlx::Stream)>(),size_of::<safemlx::Array>()*2,
        size_of::<Result<safemlx::Array,safemlx::error::Exception>>(),
        size_of::<Result<Option<safemlx::Array>,safemlx::error::Exception>>(),
        size_of::<usize>()*9,size_of::<u64>()*5,size_of::<[usize;4]>(),size_of::<[u64;4]>(),
        size_of::<std::iter::Zip<std::array::IntoIter<usize,4>,std::array::IntoIter<u64,4>>>(),
        size_of::<std::slice::Iter<i32>>(),size_of::<bool>()*3,
        size_of::<std::iter::Enumerate<eredu_nn::workspace::WorkspaceLayoutIter<'_>>>()];
    source.native.controls=frames.into_iter().try_fold(source.native.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,|n,b|n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan { alias_input: None,dtype,population:source.native,output_bytes:full,scratch_bytes,rank,
        parameter_shells:usize::from(learned),seeds,validations:0}))
}
