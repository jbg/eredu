//! Shared state selection with exact source-aware tensor operations.
use super::*;
use crate::{external_assistant::ExternalOperationResult,speculative_execution::PreparedEmbeddedEvidence};
use eredu_core::{AttentionPolicy,SpeculativeBuffer};
use std::mem::{size_of,size_of_val};

pub(super) fn select<'t,M:ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>>(
    hidden:&M::Tensor,shared:impl ExactSizeIterator<Item=(AttentionPolicy,&'t M::Tensor,&'t M::Tensor)>,
    evidence:Option<&PreparedEmbeddedEvidence>,row:usize,cache_len:i32,context:M::Context<'_>,
)->Result<ExternalTargetState<M::Tensor>,M::Error> where M::Tensor:'t {
    let controls=[size_of::<ExternalTargetState<M::Tensor>>(),size_of::<ExternalOperationResult<M::Tensor>>(),
        size_of::<Result<ExternalTargetState<M::Tensor>,M::Error>>(),size_of_val(&shared),
        size_of::<(usize,i32,Option<&PreparedEmbeddedEvidence>)>(),size_of::<(M::Tensor,M::Tensor)>(),
        size_of::<SpeculativeBuffer<PreparedEmbeddedEvidence>>()];
    let _host=M::state_host_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add),context)?;
    let count=shared.len();
    let proof_count=count.checked_mul(2).and_then(|n|n.checked_add(1)).ok_or_else(||M::state_refusal(context))?;
    let mut proofs=M::state_buffer(proof_count,context)?;
    let mut rows=M::state_buffer(count,context)?;
    let end=row.checked_add(1).ok_or_else(||M::state_refusal(context))?;
    let hidden=M::tensor_range_with_source(hidden,1,row,end,evidence,ExternalAssistantTensorPlacement::Target,context)?;
    if let Some(proof)=hidden.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
    let frontier=usize::try_from(cache_len).map_err(|_|M::state_refusal(context))?;
    for (policy,keys,values) in shared {
        if rows.iter().any(|(prior,_)|*prior==policy){return Err(M::state_refusal(context));}
        let k_end=M::state_dimension(keys,2,context)?.min(frontier);
        let v_end=M::state_dimension(values,2,context)?.min(frontier);
        let keys=M::tensor_range_with_source(keys,2,0,k_end,evidence,ExternalAssistantTensorPlacement::Target,context)?;
        let values=M::tensor_range_with_source(values,2,0,v_end,evidence,ExternalAssistantTensorPlacement::Target,context)?;
        if let Some(proof)=keys.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
        if let Some(proof)=values.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
        rows.try_push((policy,(keys.output,values.output))).map_err(|_|M::state_refusal(context))?;
    }
    let shared_kv=SharedAssistantStates::from_prepared(M::freeze_state_values(rows,context)?);
    let mut priors=M::state_buffer(proofs.len(),context)?;
    for proof in &proofs{priors.try_push(proof).map_err(|_|M::state_refusal(context))?;}
    let source=M::join_tensor_sources_at(|visit|{visit(&hidden.output);for (k,v) in shared_kv.values(){visit(k);visit(v);}},&priors,ExternalAssistantTensorPlacement::Target,context)?;
    Ok(ExternalTargetState{hidden:hidden.output,shared_kv,cache_len,evidence:source})
}

pub(super) fn on_draft<M:ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>>(
    state:&ExternalTargetState<M::Tensor>,context:M::Context<'_>,
)->Result<ExternalTargetState<M::Tensor>,M::Error>{
    let parts=[size_of::<ExternalTargetState<M::Tensor>>(),size_of::<Result<ExternalTargetState<M::Tensor>,M::Error>>(),
        size_of::<ExternalOperationResult<M::Tensor>>(),size_of::<SpeculativeBuffer<PreparedEmbeddedEvidence>>(),
        size_of::<SpeculativeBuffer<&PreparedEmbeddedEvidence>>(),size_of::<(M::Tensor,M::Tensor)>()];
    let _host=M::state_host_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add),context)?;
    let count=state.shared_kv.len();
    let proof_count=count.checked_mul(2).and_then(|n|n.checked_add(1)).ok_or_else(||M::state_refusal(context))?;
    let mut proofs=M::state_buffer(proof_count,context)?;
    let mut rows=M::state_buffer(count,context)?;
    let hidden=M::transfer_tensor_with_source(&state.hidden,state.evidence.as_ref(),ExternalAssistantTransfer::TargetToDraft,context)?;
    if let Some(proof)=hidden.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
    for (policy,(k,v)) in &state.shared_kv {
        let k=M::transfer_tensor_with_source(k,state.evidence.as_ref(),ExternalAssistantTransfer::TargetToDraft,context)?;
        let v=M::transfer_tensor_with_source(v,state.evidence.as_ref(),ExternalAssistantTransfer::TargetToDraft,context)?;
        if let Some(proof)=k.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
        if let Some(proof)=v.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
        rows.try_push((*policy,(k.output,v.output))).map_err(|_|M::state_refusal(context))?;
    }
    let shared_kv=SharedAssistantStates::from_prepared(M::freeze_state_values(rows,context)?);
    let mut prior=M::state_buffer(proofs.len(),context)?;
    for proof in &proofs{prior.try_push(proof).map_err(|_|M::state_refusal(context))?;}
    let evidence=M::join_tensor_sources_at(|visit|{
        visit(&hidden.output);for (k,v) in shared_kv.values(){visit(k);visit(v);}
    },&prior,ExternalAssistantTensorPlacement::Draft,context)?;
    Ok(ExternalTargetState{hidden:hidden.output,shared_kv,cache_len:state.cache_len,evidence})
}
pub(super) fn copy_draft<M:ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>>(
    state:&AssistantState<M::Tensor>,context:M::Context<'_>,
)->Result<AssistantState<M::Tensor>,M::Error>{
    let parts=[size_of::<AssistantState<M::Tensor>>(),size_of::<Result<AssistantState<M::Tensor>,M::Error>>(),size_of::<SharedAssistantStates<M::Tensor>>()];
    let _host=M::state_host_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add),context)?;
    let hidden=M::tensor_alias_with_source(&state.hidden,state.evidence.as_ref(),ExternalAssistantTensorPlacement::Draft,context)?;
    Ok(AssistantState{shared_kv:state.shared_kv.clone(),kv_offset:state.kv_offset,hidden:hidden.output,evidence:hidden.evidence})
}

/// Actual hidden and policy K/V copies retain independent current-root receipts.
/// No raw HashMap clone or source-only relabeling can create an isolated state.
pub(super) fn control_copy<M:ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>>(
    state:&ExternalTargetState<M::Tensor>,context:M::Context<'_>)->Result<ExternalTargetState<M::Tensor>,M::Error>{
    let controls=[size_of::<ExternalTargetState<M::Tensor>>(),size_of::<ExternalOperationResult<M::Tensor>>(),
        size_of::<Result<ExternalTargetState<M::Tensor>,M::Error>>(),
        size_of::<SpeculativeBuffer<PreparedEmbeddedEvidence>>(),size_of::<(M::Tensor,M::Tensor)>()];
    let _host=M::state_host_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add),context)?;
    let count=state.shared_kv.len();
    let proof_count=count.checked_mul(2).and_then(|n|n.checked_add(1)).ok_or_else(||M::state_refusal(context))?;
    // Proofs precede native values so failure releases the populated array prefix first.
    let mut proofs=M::state_buffer(proof_count,context)?;
    let mut rows=M::state_buffer(count,context)?;
    let hidden=M::control_copy_tensor_with_source(&state.hidden,state.evidence.as_ref(),ExternalAssistantTensorPlacement::Target,context)?;
    if let Some(proof)=hidden.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
    for (policy,(keys,values)) in &state.shared_kv{
        let keys=M::control_copy_tensor_with_source(keys,state.evidence.as_ref(),ExternalAssistantTensorPlacement::Target,context)?;
        let values=M::control_copy_tensor_with_source(values,state.evidence.as_ref(),ExternalAssistantTensorPlacement::Target,context)?;
        if let Some(proof)=keys.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
        if let Some(proof)=values.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
        rows.try_push((*policy,(keys.output,values.output))).map_err(|_|M::state_refusal(context))?;
    }
    let shared_kv=SharedAssistantStates::from_prepared(M::freeze_state_values(rows,context)?);
    let mut prior=M::state_buffer(proofs.len(),context)?;
    for proof in &proofs{prior.try_push(proof).map_err(|_|M::state_refusal(context))?;}
    let evidence=M::join_tensor_sources_at(|visit|{
        visit(&hidden.output);for (keys,values) in shared_kv.values(){visit(keys);visit(values);}
    },&prior,ExternalAssistantTensorPlacement::Target,context)?;
    Ok(ExternalTargetState{hidden:hidden.output,shared_kv,cache_len:state.cache_len,evidence})
}
