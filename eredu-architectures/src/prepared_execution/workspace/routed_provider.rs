//! The retained selected bank table supplies whole addressable regions.
use super::*;
use crate::routed_text::RetainedRoutedBanks;
use eredu_runtime::{RoutedExpertProvider,RoutedExpertRequest,ParameterBankResidency,ParameterBankLoadOptions};

pub(super) struct EquationRoutedProvider {
    banks:Option<RetainedRoutedBanks>,options:Option<ParameterBankLoadOptions>,
}
impl EquationRoutedProvider {
    pub(super) fn resident()->Self{Self{banks:None,options:None}}
    pub(super) fn new(banks:RetainedRoutedBanks,residency:ParameterBankResidency,context:&WorkspaceContext)->Result<Self,Error>{
        context.charge_metadata(std::mem::size_of::<(Self,RetainedRoutedBanks,ParameterBankResidency,Result<Self,Error>)>())?;
        match residency {
            ParameterBankResidency::WithLayer=>Ok(Self{banks:Some(banks),options:None}),
            ParameterBankResidency::IndependentCache(options)=>Ok(Self{banks:Some(banks),options:Some(options)}),
            _=>Err(context.metadata_error(format_args!("selected bank residency has no equation provider"))),
        }
    }
    pub(super) fn is_addressable(&self)->bool{self.options.is_some()}
    fn addressable(&self,request:&mut RoutedExpertRequest<'_,'_,WorkspaceTensor>,context:&WorkspaceContext,kind:u8)->Result<Option<WorkspaceTensor>,Error>{
        let Some(options)=self.options else{return Ok(None)};
        context.charge_metadata(std::mem::size_of::<(Self,&Self,&RoutedExpertRequest<'_,'_,WorkspaceTensor>,
            eredu_nn::workspace::WorkspaceAddressableRegionView<'_>,Result<Option<WorkspaceTensor>,Error>)>()
            .checked_add(eredu_runtime::expert::AddressableChunkPlan::control_bytes())
            .and_then(|n|n.checked_add(eredu_nn::workspace::WorkspaceAddressableRegionView::control_bytes()?))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
        let bank=self.banks.as_ref().and_then(|banks|banks.get(&request.bank))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?;
        let source=bank.addressable_workspace_source(request,options,None,context)?;
        let actual=match source.kernel {eredu_nn::workspace::WorkspaceExpertKernel::Gated(_)=>0,
            eredu_nn::workspace::WorkspaceExpertKernel::Relu2(_)=>1,eredu_nn::workspace::WorkspaceExpertKernel::Linear(_)=>2};
        if actual!=kind{return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())}
        let input=request.input;let routes=request.routes;
        let (output,bias)=eredu_runtime::with_routed_unit_invocation(request.unit_observer.take(),
            eredu_runtime::RoutedUnitInvocation{input,origins:None,unit_coordinates:None},
            |mut observer|{
                let interested=observer.is_some();
                let mut observe=|source:eredu_nn::workspace::WorkspaceAddressableObservationView<'_>|observer.as_mut()
                    .ok_or_else(||eredu_nn::Error::backend_source(eredu_nn::GroupedUnitError::Unavailable))?
                    .observe_addressable_source(source);
                eredu_nn::workspace::record_addressable_region_with_observation(source,input,routes,context,
                    interested.then_some(&mut observe as &mut dyn for<'a> FnMut(eredu_nn::workspace::WorkspaceAddressableObservationView<'a>)->Result<eredu_nn::workspace::WorkspaceAddressableObservationSource,eredu_nn::Error>)).map(|out|out.into_parts())
            },std::convert::identity)?;
        if bias.is_some(){return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())}
        Ok(Some(output))
    }
}
impl RoutedExpertProvider<WorkspaceBackend> for EquationRoutedProvider {
    type Error=Error;
    fn forward_grouped(&mut self,bank:&mut <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        mut request:RoutedExpertRequest<'_,'_,WorkspaceTensor>,context:&WorkspaceContext)->Result<WorkspaceTensor,Error>{
        if let Some(value)=self.addressable(&mut request,context,0)?{return Ok(value)}
        <eredu_runtime::ResidentExpertProvider as RoutedExpertProvider<WorkspaceBackend>>::forward_grouped(&mut eredu_runtime::ResidentExpertProvider,bank,request,context)
    }
    fn forward_relu2_routed(&mut self,bank:&mut <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        mut request:RoutedExpertRequest<'_,'_,WorkspaceTensor>,context:&WorkspaceContext)->Result<WorkspaceTensor,Error>{
        if let Some(value)=self.addressable(&mut request,context,1)?{return Ok(value)}
        <eredu_runtime::ResidentExpertProvider as RoutedExpertProvider<WorkspaceBackend>>::forward_relu2_routed(&mut eredu_runtime::ResidentExpertProvider,bank,request,context)
    }
    fn forward_linear_routed(&mut self,bank:&mut <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::LinearGroups,
        mut request:RoutedExpertRequest<'_,'_,WorkspaceTensor>,context:&WorkspaceContext)->Result<WorkspaceTensor,Error>{
        if let Some(value)=self.addressable(&mut request,context,2)?{return Ok(value)}
        <eredu_runtime::ResidentExpertProvider as RoutedExpertProvider<WorkspaceBackend>>::forward_linear_routed(&mut eredu_runtime::ResidentExpertProvider,bank,request,context)
    }
}
