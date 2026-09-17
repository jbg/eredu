//! Idle ranks borrow the same count/transport/provider producers as active EP.
use super::*;
use eredu_nn::workspace::{WorkspaceExpertInactiveWave,WorkspaceExpertTransfer,WorkspaceFloatingType};
pub(crate) struct ExpertInactiveWaveQuote {
    pub(crate) declaration:WorkspaceExpertInactiveWave,
    pub(crate) counts:ExpertCountQuote,
    pub(crate) transport:ExpertTransportQuote,
    pub(crate) provider:ExpertProviderQuote,
    pub(crate) aggregate:ExpertRegionAggregate,
    source:OriginalParallelSource,
}
impl ExpertInactiveWaveQuote {
    pub(crate) fn prepare(source:&OriginalParallelSource,declaration:WorkspaceExpertInactiveWave,
        mechanism:ResidentExecutionMechanisms)->Result<Self,Error>{
        let context=WorkspaceContext::new_with_metadata_funding(mechanism,source.funding().clone())?;
        context.charge_metadata(size_of::<(Self,Result<Self,Error>,WorkspaceContext,ExpertTransferGeometry)>())?;
        declaration.validate()?;
        let invalid=||context.metadata_error(format_args!("inactive expert wave has an incomplete retained source"));
        let dtype=match declaration.dtype{WorkspaceFloatingType::Float32=>safemlx::Dtype::Float32,
            WorkspaceFloatingType::Float16=>safemlx::Dtype::Float16,WorkspaceFloatingType::Bfloat16=>safemlx::Dtype::Bfloat16};
        let counts=ExpertCountQuote::prepare_for(source,mechanism,declaration.provider.expert_group,declaration.rank,declaration.peers)?;
        let provider=ExpertProviderQuote::prepare_for(source,mechanism,declaration.provider.groups())?;
        let geometry=ExpertTransferGeometry{group:declaration.provider.expert_group,rank:declaration.rank,peers:declaration.peers,
            selected_rows:declaration.selected_rows,input_width:declaration.input_width,output_width:declaration.output_width,
            input_dtype:dtype,scores_dtype:dtype,coefficients_dtype:dtype,output_dtype:dtype,
            bias_dtype:declaration.transfers.iter().any(|v|v==WorkspaceExpertTransfer::ReverseBias).then_some(dtype),transfers:declaration.transfers};
        let transport=ExpertTransportQuote::prepare_for(source,mechanism,geometry)?;
        let mut bytes=counts.backing.checked_add(provider.backing).ok_or_else(invalid)?;
        let mut births=counts.births.checked_add(provider.births).ok_or_else(invalid)?;
        for profile in transport.profiles(){
            bytes=bytes.checked_add(u64::try_from(profile.aggregate_backing().ok_or_else(invalid)?).map_err(|_|invalid())?).ok_or_else(invalid)?;
            births=births.checked_add(profile.births).ok_or_else(invalid)?;
        }
        // Integer metadata comes from the same initialized one-row seed and
        // exact empty Slice as the active path. Floating placeholders use the
        // allocator's shared typed zero constructor; these remain parent nodes.
        let metadata=declaration.transfers.iter().filter(|v|matches!(v,
            WorkspaceExpertTransfer::ForwardIndex|WorkspaceExpertTransfer::ReverseIndex)).count();
        let parent=WorkspaceContext::new_with_metadata_funding(mechanism,source.funding().clone())?;
        let seed=WorkspaceTensor::existing(parent.layout(&[1,1],WorkspaceDtype::Int32)?,&parent)?;
        parent.begin_span();let empty=seed.narrow_axis(0,0,0,&parent)?;
        let mut report=parent.finish_report(&[empty])?;
        if report.operations.len()!=1{return Err(invalid());}
        let empty_slice=report.operations.pop().ok_or_else(invalid)?;
        let mut extra_parents=context.metadata_vec(declaration.transfers.len().checked_sub(metadata).ok_or_else(invalid)?)?;
        for payload in declaration.transfers.iter(){
            let width=match payload{
                WorkspaceExpertTransfer::ForwardInput=>declaration.input_width,
                WorkspaceExpertTransfer::ForwardScores|WorkspaceExpertTransfer::ForwardCoefficients=>1,
                WorkspaceExpertTransfer::ReverseOutput|WorkspaceExpertTransfer::ReverseBias=>declaration.output_width,
                _=>continue,
            };
            let parent=WorkspaceContext::new_with_metadata_funding(mechanism,source.funding().clone())?;
            parent.begin_span();
            let prototype=eredu_nn::workspace::WorkspaceLayoutView::new(&[],WorkspaceDtype::Float32)?
                .with_representation(Some(WorkspaceRepresentation::new(declaration.dtype,true)));
            let value=WorkspaceTensor::zeros_from_prototype(&[0,width],prototype,&parent)?;
            let mut report=parent.finish_report(&[value])?;
            if report.operations.len()!=1{return Err(invalid());}
            extra_parents.push(report.operations.pop().ok_or_else(invalid)?);
        }
        let parent_completions=declaration.transfers.len().checked_add(metadata).ok_or_else(invalid)?;
        let aggregate=ExpertRegionAggregate{child_bytes:bytes,child_births:births,host_bytes:0,parent_completions,
            empty_slices:metadata,empty_slice,extra_parents,output_bytes:Vec::new()};
        Ok(Self{declaration,counts,transport,provider,aggregate,source:source.clone()})
    }
    pub(crate) fn source(&self)->&OriginalParallelSource{&self.source}
}
