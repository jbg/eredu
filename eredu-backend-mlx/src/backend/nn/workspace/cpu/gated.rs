//! Compose the same activation and product source used by ordinary gating.
use super::*;
use eredu_nn::GatedProductActivation;
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::GatedProduct(policy)=operation.kind else{return Ok(None)};
    policy.validate_fixed()?;
    if policy.activation()!=GatedProductActivation::Silu||policy.sigmoid_multiplier()!=1.0||
        policy.gate_upper_bound().is_some()||policy.up_absolute_bound().is_some()||policy.up_offset()!=0.0 {
        return Ok(None);
    }
    if operation.inputs.len()!=2||operation.outputs.len()!=1 {
        return Err(MlxWorkspaceFactError::descriptor("CPU gated-product population differs"));
    }
    let gate=operation.inputs.get(0).expect("two gated inputs");
    let up=operation.inputs.get(1).expect("two gated inputs");
    if gate.shape()!=up.shape() {
        return Err(MlxWorkspaceFactError::descriptor("CPU gated-product shape differs"));
    }
    // These descriptive views borrow the actual source layouts; they grant no
    // tensor, backing or execution authority. The two existing planners retain
    // their precise dtype checks, and native Eval still validates actual roots.
    let activation=WorkspaceOperationView {kind:WorkspaceOperationKindView::Elementwise("silu"),
        inputs:operation.inputs.slice(0..1).expect("checked source slice"),outputs:operation.outputs};
    let product=WorkspaceOperationView {kind:WorkspaceOperationKindView::Elementwise("multiply"),
        inputs:operation.inputs,outputs:operation.outputs};
    let Some(activation)=super::activation::inspect(activation,mechanism)? else{return Ok(None)};
    let Some(mut product)=super::pointwise::inspect(product,mechanism)? else{return Ok(None)};
    if activation.dtype!=product.dtype {return Ok(None);}
    product.population.add(activation.population).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    product.scratch_bytes=facts::add(product.scratch_bytes,facts::add(activation.output_bytes,activation.scratch_bytes)?)?;
    product.seeds=product.seeds.checked_add(activation.seeds).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    product.rank=product.rank.max(activation.rank);
    let frames=[size_of::<WorkspaceOperationView<'_>>()*3,size_of::<WorkspaceLayoutView<'_>>()*2,
        size_of::<OperationPlan>()*2,size_of::<Option<OperationPlan>>()*2,size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<eredu_nn::GatedProductPolicy>(),size_of::<std::ops::Range<usize>>(),
        size_of::<(crate::MlxTensor,crate::MlxTensor,
            eredu_nn::GatedProductPolicy,&safemlx::Stream)>(),
        size_of::<safemlx::Array>()*3,size_of::<Result<crate::MlxTensor,eredu_nn::Error>>()];
    product.population.controls=frames.into_iter().try_fold(product.population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,|n,b|n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(product))
}
