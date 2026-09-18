//! One retained constructor selects the resident or independently cached worker.
use super::*;
use crate::prepared_execution::workspace::routed_provider::EquationRoutedProvider;
use crate::routed_text::{RetainedRoutedBanks, partition_source::PartitionResidentProvider};
use eredu_runtime::{RoutedExpertProvider, TensorParallelRoutedExpertProvider};

pub(crate) enum Source {
    Resident(RetainedPartitionResidentSource),
    Addressable { banks: RetainedRoutedBanks, residency: eredu_runtime::ParameterBankResidency },
}
impl Source {
    pub(super) fn resident(&self) -> Option<RetainedPartitionResidentSource> {
        match self { Self::Resident(source) => Some(source.clone()), Self::Addressable { .. } => None }
    }
    pub(super) fn provider(&self, context:&WorkspaceContext)->Result<Provider<'_>,Error> {
        context.charge_metadata(std::mem::size_of::<(&Self,&WorkspaceContext,Provider<'_>,Result<Provider<'_>,Error>)>())?;
        match self {
            Self::Resident(source) => Ok(Provider::Resident(source.borrowed())),
            Self::Addressable { banks, residency } => Ok(Provider::Addressable(EquationRoutedProvider::new(banks.clone(),*residency,context)?)),
        }
    }
}
pub(super) enum Provider<'a> {
    Resident(eredu_runtime::BorrowedRoutedBankProviders<'a,PartitionResidentProvider>),
    Addressable(EquationRoutedProvider),
}
macro_rules! ordinary {
    ($name:ident,$module:ident) => {
        fn $name(&mut self,bank:&mut <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::$module,
            request:eredu_runtime::RoutedExpertRequest<'_,'_,WorkspaceTensor>,context:&WorkspaceContext)
            ->Result<WorkspaceTensor,Self::Error> {
            context.charge_metadata(std::mem::size_of::<(&mut Self, &mut <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::$module,
                eredu_runtime::RoutedExpertRequest<'_,'_,WorkspaceTensor>, &WorkspaceContext, Result<WorkspaceTensor,Self::Error>)>())
                .map_err(crate::RoutedTextExecutionError::from_error)?;
            match self { Self::Resident(provider)=>RoutedExpertProvider::<WorkspaceBackend>::$name(provider,bank,request,context).map_err(crate::RoutedTextExecutionError::from_error),
                Self::Addressable(provider)=>RoutedExpertProvider::<WorkspaceBackend>::$name(provider,bank,request,context).map_err(crate::RoutedTextExecutionError::from_error) }
        }
    };
}
macro_rules! parallel {
    ($name:ident,$module:ident) => {
        fn $name(&mut self,bank:&mut <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::$module,
            request:eredu_runtime::RoutedExpertRequest<'_,'_,WorkspaceTensor>,partitions:usize,context:&WorkspaceContext)
            ->Result<eredu_runtime::RoutedExpertTensorParallelOutput<WorkspaceTensor>,Self::Error> {
            context.charge_metadata(std::mem::size_of::<(&mut Self, &mut <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::$module,
                eredu_runtime::RoutedExpertRequest<'_,'_,WorkspaceTensor>, usize, &WorkspaceContext,
                Result<eredu_runtime::RoutedExpertTensorParallelOutput<WorkspaceTensor>,Self::Error>)>())
                .map_err(crate::RoutedTextExecutionError::from_error)?;
            match self { Self::Resident(provider)=>TensorParallelRoutedExpertProvider::<WorkspaceBackend>::$name(provider,bank,request,partitions,context).map_err(crate::RoutedTextExecutionError::from_error),
                Self::Addressable(provider)=>TensorParallelRoutedExpertProvider::<WorkspaceBackend>::$name(provider,bank,request,partitions,context).map_err(crate::RoutedTextExecutionError::from_error) }
        }
    };
}
impl RoutedExpertProvider<WorkspaceBackend> for Provider<'_> {
    type Error=crate::RoutedTextExecutionError;
    ordinary!(forward_grouped,GatedProductGroups);
    ordinary!(forward_compact_grouped,GatedProductGroups);
    ordinary!(forward_relu2_routed,Relu2Groups);
    ordinary!(forward_linear_routed,LinearGroups);
}
impl TensorParallelRoutedExpertProvider<WorkspaceBackend> for Provider<'_> {
    parallel!(forward_grouped_tensor_parallel,GatedProductGroups);
    parallel!(forward_compact_grouped_tensor_parallel,GatedProductGroups);
    parallel!(forward_relu2_routed_tensor_parallel,Relu2Groups);
}
