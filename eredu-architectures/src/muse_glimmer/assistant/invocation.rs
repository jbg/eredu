//! Exact raw-context operation shared by ordinary and admitted external execution.
use super::*;
use crate::external_assistant::{MuseGlimmerAssistantArchitecture,invocation::ExternalAssistantOperation};
use crate::speculative_execution::PreparedEmbeddedEvidence;
use eredu_nn::{AttentionCache,DistributedNeuralBackend,GroupedNeuralBackend,
    workspace::{WorkspaceBackend,WorkspaceContext,WorkspaceTensor,WorkspaceMetadataError}};
use std::mem::{size_of,size_of_val};

/// Ordered target tap assembly and rolling raw suffix; encoding remains lazy.
pub struct RawContext;
/// Exact source inputs of one completed target span or accepted verification.
pub struct RawContextArguments<'a,T>{
    /// Previously retained raw suffix, if any.
    pub previous:Option<&'a T>,
    /// Actual ordered target tap values.
    pub states:&'a [T],
}
/// Paid alias-preserving cold projection of the actual raw inputs.
pub struct ProjectedRawContext{
    previous:Option<WorkspaceTensor>,
    states:Vec<WorkspaceTensor>,
    _context:WorkspaceContext,
}
impl ExternalAssistantOperation<MuseGlimmerAssistantArchitecture> for RawContext{
    type Arguments<'a,T:Tensor+'a>=RawContextArguments<'a,T>;
    type Output<T:Tensor>=T;
    type Projected=ProjectedRawContext;
    fn invocation_kind()->eredu_runtime::speculative::external_occurrence::ExternalInvocationKind{
        eredu_runtime::speculative::external_occurrence::ExternalInvocationKind::AssembleContext
    }
    fn workspace_module(config:&DFlashConfig,context:&WorkspaceContext)->Result<DFlash<WorkspaceBackend>,Error>{
        DFlash::from_config(config,context)
    }
    fn execute<B,C>(module:&mut DFlash<B>,arguments:RawContextArguments<'_,B::Tensor>,context:&<B::Tensor as Tensor>::Context)->Result<B::Tensor,Error>
    where B:GroupedNeuralBackend+DistributedNeuralBackend+Clone,C:AttentionCache<B::Tensor>{
        module.prepare_raw_context_span(arguments.previous,arguments.states,context)
    }
    fn execute_with_metadata<B,C>(module:&mut DFlash<B>,arguments:RawContextArguments<'_,B::Tensor>,context:&<B::Tensor as Tensor>::Context,metadata:&WorkspaceContext)->Result<B::Tensor,Error>
    where B:GroupedNeuralBackend+DistributedNeuralBackend+Clone,C:AttentionCache<B::Tensor>{
        module.prepare_raw_context_span_with_metadata(arguments.previous,arguments.states,context,ModuleMetadata::funded(metadata))
    }
    fn reborrow<'a,'state,T:Tensor>(arguments:&'a mut RawContextArguments<'state,T>)->RawContextArguments<'a,T> where 'state:'a{
        RawContextArguments{previous:arguments.previous,states:arguments.states}
    }
    fn visit_evidence<T:Tensor>(_:&RawContextArguments<'_,T>,_:&mut dyn FnMut(&PreparedEmbeddedEvidence)){}
    fn retain_evidence<T:Tensor>(_:&mut RawContextArguments<'_,T>,_:PreparedEmbeddedEvidence){}
    fn project<'a,T:Tensor>(arguments:&'a RawContextArguments<'_,T>,mut project:impl FnMut(&'a T)->Result<WorkspaceTensor,Error>,context:&WorkspaceContext)->Result<Self::Projected,Error>{
        let controls=[size_of::<Self::Projected>(),size_of::<Result<Self::Projected,Error>>(),size_of_val(&project),
            size_of::<Option<WorkspaceTensor>>(),size_of::<std::slice::Iter<'_,T>>()];
        context.charge_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add).ok_or(WorkspaceMetadataError::Overflow)?)?;
        let previous=arguments.previous.map(&mut project).transpose()?;
        let mut states=context.metadata_vec(arguments.states.len())?;
        for value in arguments.states{states.push(project(value)?);}
        Ok(ProjectedRawContext{previous,states,_context:context.clone()})
    }
    fn geometry(projected:&Self::Projected,context:&WorkspaceContext)->Result<eredu_core::InferenceGeometry,Error>{
        let source=projected.states.first().ok_or_else(||context.metadata_error(format_args!("raw context has no target taps")))?;
        let shape=source.shape();
        if shape.len()!=3||shape[0]<=0||shape[1]<=0{return Err(context.metadata_error(format_args!("raw context has invalid batch/positions")));}
        let geometry=eredu_core::InferenceGeometry{batch_size:shape[0] as u64,cached_positions:0,
            input_positions:shape[1] as u64,prefill_chunk_positions:shape[1] as u64,max_output_tokens:0,output:eredu_core::OutputDemand::Sequence};
        geometry.validate_fixed().map_err(|cause|context.metadata_source(cause))?;Ok(geometry)
    }
    fn trace(module:&mut DFlash<WorkspaceBackend>,projected:&mut Self::Projected,context:&WorkspaceContext)->Result<WorkspaceTensor,Error>{
        module.prepare_raw_context_span_with_metadata(projected.previous.as_ref(),&projected.states,context,ModuleMetadata::funded(context))
    }
    fn visit_projected(projected:&Self::Projected,visit:&mut dyn FnMut(&WorkspaceTensor)){
        if let Some(previous)=&projected.previous{visit(previous);}
        for value in &projected.states{visit(value);}
    }
    fn visit_arguments<T:Tensor>(arguments:&RawContextArguments<'_,T>,visit:&mut dyn FnMut(&T)){
        if let Some(previous)=arguments.previous{visit(previous);}
        for value in arguments.states{visit(value);}
    }
    fn visit_output<T:Tensor>(output:&T,visit:&mut dyn FnMut(&T)){visit(output);}
}

/// Encodes newly accepted target rows and appends the committed K/V window.
pub struct UpdateContext;
/// Actual committed context and pending raw rows for this update.
pub struct UpdateContextArguments<'a,T>{
    /// Previous committed context, lent without cloning its container.
    pub previous:Option<&'a DFlashContext<T>>,
    /// Raw ordered target rows not yet encoded.
    pub pending:&'a T,
    /// Actual target frontier after these rows.
    pub absolute_end:i32,
}
/// Finite source projection of one context append.
pub struct ProjectedUpdateContext{
    previous:Option<DFlashContext<WorkspaceTensor>>,
    pending:WorkspaceTensor,
    absolute_end:i32,
    _context:WorkspaceContext,
}
fn project_context<'a,T:Tensor>(source:&'a DFlashContext<T>,project:&mut impl FnMut(&'a T)->Result<WorkspaceTensor,Error>,context:&WorkspaceContext)->Result<DFlashContext<WorkspaceTensor>,Error>{
    let controls=[size_of::<DFlashContext<WorkspaceTensor>>(),size_of::<Result<DFlashContext<WorkspaceTensor>,Error>>(),
        size_of::<std::slice::Iter<'_,DFlashLayerContext<T>>>(),size_of::<DFlashLayerContext<WorkspaceTensor>>()];
    context.charge_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add).ok_or(WorkspaceMetadataError::Overflow)?)?;
    let encoded=project(&source.encoded)?;
    let mut layers=context.metadata_vec(source.layers.len())?;
    for layer in &source.layers{layers.push(DFlashLayerContext{keys:project(&layer.keys)?,values:project(&layer.values)?});}
    Ok(DFlashContext{encoded,layers,start:source.start,end:source.end})
}
fn visit_context<T>(value:&DFlashContext<T>,visit:&mut dyn FnMut(&T)){
    visit(&value.encoded);for layer in &value.layers{visit(&layer.keys);visit(&layer.values);}
}
fn geometry(input:&WorkspaceTensor,cached:i32,context:&WorkspaceContext)->Result<eredu_core::InferenceGeometry,Error>{
    let shape=input.shape();
    if shape.len()!=3||shape[0]<=0||shape[1]<=0||cached<0{return Err(context.metadata_error(format_args!("invalid fused assistant input/frontier geometry")));}
    let geometry=eredu_core::InferenceGeometry{batch_size:shape[0] as u64,cached_positions:cached as u64,
        input_positions:shape[1] as u64,prefill_chunk_positions:shape[1] as u64,max_output_tokens:0,output:eredu_core::OutputDemand::Sequence};
    geometry.validate_fixed().map_err(|cause|context.metadata_source(cause))?;Ok(geometry)
}
impl ExternalAssistantOperation<MuseGlimmerAssistantArchitecture> for UpdateContext{
    type Arguments<'a,T:Tensor+'a>=UpdateContextArguments<'a,T>;
    type Output<T:Tensor>=DFlashContext<T>;
    type Projected=ProjectedUpdateContext;
    fn invocation_kind()->eredu_runtime::speculative::external_occurrence::ExternalInvocationKind{eredu_runtime::speculative::external_occurrence::ExternalInvocationKind::UpdateContext}
    fn workspace_module(config:&DFlashConfig,context:&WorkspaceContext)->Result<DFlash<WorkspaceBackend>,Error>{DFlash::from_config(config,context)}
    fn execute<B,C>(module:&mut DFlash<B>,arguments:Self::Arguments<'_,B::Tensor>,context:&<B::Tensor as Tensor>::Context)->Result<Self::Output<B::Tensor>,Error>
    where B:GroupedNeuralBackend+DistributedNeuralBackend+Clone,C:AttentionCache<B::Tensor>{
        module.update_context_borrowed(arguments.previous,arguments.pending,arguments.absolute_end,context,ModuleMetadata::new::<B>(context))
    }
    fn execute_with_metadata<B,C>(module:&mut DFlash<B>,arguments:Self::Arguments<'_,B::Tensor>,context:&<B::Tensor as Tensor>::Context,metadata:&WorkspaceContext)->Result<Self::Output<B::Tensor>,Error>
    where B:GroupedNeuralBackend+DistributedNeuralBackend+Clone,C:AttentionCache<B::Tensor>{
        module.update_context_borrowed(arguments.previous,arguments.pending,arguments.absolute_end,context,ModuleMetadata::funded(metadata))
    }
    fn reborrow<'a,'state,T:Tensor>(arguments:&'a mut Self::Arguments<'state,T>)->Self::Arguments<'a,T> where 'state:'a{
        UpdateContextArguments{previous:arguments.previous,pending:arguments.pending,absolute_end:arguments.absolute_end}
    }
    fn visit_evidence<T:Tensor>(_:&Self::Arguments<'_,T>,_:&mut dyn FnMut(&PreparedEmbeddedEvidence)){}
    fn retain_evidence<T:Tensor>(_:&mut Self::Arguments<'_,T>,_:PreparedEmbeddedEvidence){}
    fn project<'a,T:Tensor>(arguments:&'a Self::Arguments<'_,T>,mut project:impl FnMut(&'a T)->Result<WorkspaceTensor,Error>,context:&WorkspaceContext)->Result<Self::Projected,Error>{
        let controls=[size_of::<Self::Projected>(),size_of::<Result<Self::Projected,Error>>(),size_of_val(&project)];
        context.charge_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add).ok_or(WorkspaceMetadataError::Overflow)?)?;
        let previous=arguments.previous.map(|prior|project_context(prior,&mut project,context)).transpose()?;
        let pending=project(arguments.pending)?;
        Ok(ProjectedUpdateContext{previous,pending,absolute_end:arguments.absolute_end,_context:context.clone()})
    }
    fn geometry(projected:&Self::Projected,context:&WorkspaceContext)->Result<eredu_core::InferenceGeometry,Error>{geometry(&projected.pending,0,context)}
    fn trace(module:&mut DFlash<WorkspaceBackend>,projected:&mut Self::Projected,context:&WorkspaceContext)->Result<Self::Output<WorkspaceTensor>,Error>{
        module.update_context_borrowed(projected.previous.as_ref(),&projected.pending,projected.absolute_end,context,ModuleMetadata::funded(context))
    }
    fn visit_projected(projected:&Self::Projected,visit:&mut dyn FnMut(&WorkspaceTensor)){
        if let Some(previous)=&projected.previous{visit_context(previous,visit);}visit(&projected.pending);
    }
    fn visit_arguments<T:Tensor>(arguments:&Self::Arguments<'_,T>,visit:&mut dyn FnMut(&T)){
        if let Some(previous)=arguments.previous{visit_context(previous,visit);}visit(arguments.pending);
    }
    fn visit_output<T:Tensor>(output:&Self::Output<T>,visit:&mut dyn FnMut(&T)){visit_context(output,visit);}
}

/// Executes the existing anchor-plus-mask block against immutable committed K/V.
pub struct FusedProposal;
/// Exact embeddings and committed context used by the fused block.
pub struct FusedProposalArguments<'a,T>{
    /// Target token embeddings for anchor and mask rows.
    pub embeddings:&'a T,
    /// Exact committed assistant context.
    pub committed:&'a DFlashContext<T>,
    /// Actual committed target frontier.
    pub absolute_end:i32,
}
/// Paid cold projection of the actual fused block.
pub struct ProjectedFusedProposal{
    embeddings:WorkspaceTensor,committed:DFlashContext<WorkspaceTensor>,absolute_end:i32,_context:WorkspaceContext,
}
impl ExternalAssistantOperation<MuseGlimmerAssistantArchitecture> for FusedProposal{
    type Arguments<'a,T:Tensor+'a>=FusedProposalArguments<'a,T>;
    type Output<T:Tensor>=T;
    type Projected=ProjectedFusedProposal;
    fn invocation_kind()->eredu_runtime::speculative::external_occurrence::ExternalInvocationKind{eredu_runtime::speculative::external_occurrence::ExternalInvocationKind::FusedProposal}
    fn workspace_module(config:&DFlashConfig,context:&WorkspaceContext)->Result<DFlash<WorkspaceBackend>,Error>{DFlash::from_config(config,context)}
    fn execute<B,C>(module:&mut DFlash<B>,arguments:Self::Arguments<'_,B::Tensor>,context:&<B::Tensor as Tensor>::Context)->Result<B::Tensor,Error>
    where B:GroupedNeuralBackend+DistributedNeuralBackend+Clone,C:AttentionCache<B::Tensor>{module.proposal_states(arguments.embeddings,arguments.committed,arguments.absolute_end,context)}
    fn execute_with_metadata<B,C>(module:&mut DFlash<B>,arguments:Self::Arguments<'_,B::Tensor>,context:&<B::Tensor as Tensor>::Context,metadata:&WorkspaceContext)->Result<B::Tensor,Error>
    where B:GroupedNeuralBackend+DistributedNeuralBackend+Clone,C:AttentionCache<B::Tensor>{
        module.proposal_states_with_metadata(arguments.embeddings,arguments.committed,arguments.absolute_end,context,ModuleMetadata::funded(metadata))
    }
    fn reborrow<'a,'state,T:Tensor>(arguments:&'a mut Self::Arguments<'state,T>)->Self::Arguments<'a,T> where 'state:'a{
        FusedProposalArguments{embeddings:arguments.embeddings,committed:arguments.committed,absolute_end:arguments.absolute_end}
    }
    fn visit_evidence<T:Tensor>(_:&Self::Arguments<'_,T>,_:&mut dyn FnMut(&PreparedEmbeddedEvidence)){}
    fn retain_evidence<T:Tensor>(_:&mut Self::Arguments<'_,T>,_:PreparedEmbeddedEvidence){}
    fn project<'a,T:Tensor>(arguments:&'a Self::Arguments<'_,T>,mut project:impl FnMut(&'a T)->Result<WorkspaceTensor,Error>,context:&WorkspaceContext)->Result<Self::Projected,Error>{
        let controls=[size_of::<Self::Projected>(),size_of::<Result<Self::Projected,Error>>(),size_of_val(&project)];
        context.charge_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add).ok_or(WorkspaceMetadataError::Overflow)?)?;
        let embeddings=project(arguments.embeddings)?;
        let committed=project_context(arguments.committed,&mut project,context)?;
        Ok(ProjectedFusedProposal{embeddings,committed,absolute_end:arguments.absolute_end,_context:context.clone()})
    }
    fn geometry(projected:&Self::Projected,context:&WorkspaceContext)->Result<eredu_core::InferenceGeometry,Error>{geometry(&projected.embeddings,projected.absolute_end,context)}
    fn trace(module:&mut DFlash<WorkspaceBackend>,projected:&mut Self::Projected,context:&WorkspaceContext)->Result<WorkspaceTensor,Error>{
        module.proposal_states_with_metadata(&projected.embeddings,&projected.committed,projected.absolute_end,context,ModuleMetadata::funded(context))
    }
    fn visit_projected(projected:&Self::Projected,visit:&mut dyn FnMut(&WorkspaceTensor)){visit(&projected.embeddings);visit_context(&projected.committed,visit);}
    fn visit_arguments<T:Tensor>(arguments:&Self::Arguments<'_,T>,visit:&mut dyn FnMut(&T)){visit(arguments.embeddings);visit_context(arguments.committed,visit);}
    fn visit_output<T:Tensor>(output:&T,visit:&mut dyn FnMut(&T)){visit(output);}
}
