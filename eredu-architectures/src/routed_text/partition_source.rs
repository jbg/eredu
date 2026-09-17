//! Immutable resident bank source shared by ordinary construction and cold loans.
use super::*;
use std::sync::Arc;

pub(crate) enum PartitionResidentProvider {
    Empty,
    Resident(PartitionUnitProvider<PlannedResidentBank>),
}

#[derive(Clone)]
pub(crate) struct RetainedPartitionResidentSource {
    banks: Arc<eredu_runtime::RoutedBankProviders<PartitionResidentProvider>>,
}
impl RetainedPartitionResidentSource {
    pub(crate) fn prepare(
        selected: &BTreeMap<RoutedBankId, SelectedRoutedBank>, exchange: bool,
    ) -> Result<Self, RoutedTextExecutionError> {
        let banks = selected.iter().map(|(id, bank)| {
            let provider = if bank.plan().local_global_group_indices().is_empty() {
                PartitionResidentProvider::Empty
            } else {
                PartitionResidentProvider::Resident(PartitionUnitProvider::new(
                    bank, PlannedResidentBank::from_partitioned(bank, exchange)?,
                ))
            };
            Ok((*id, provider))
        }).collect::<Result<Vec<_>, RoutedTextExecutionError>>()?;
        let banks = eredu_runtime::RoutedBankProviders::new(banks)
            .map_err(|cause| RoutedTextExecutionError::Contract(cause.to_string()))?;
        Ok(Self { banks: Arc::new(banks) })
    }

    pub(crate) fn borrowed(&self) -> eredu_runtime::BorrowedRoutedBankProviders<'_, PartitionResidentProvider> {
        self.banks.borrowed()
    }

    pub(crate) fn ordinary<B>(&self) -> Result<crate::prepared_execution::PartitionBankProviders<B>, RoutedTextExecutionError>
    where B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        let banks = self.banks.banks().keys().map(|&id| {
            let handle = ResidentPartitionBank { source: self.clone(), id };
            let provider: Box<dyn eredu_runtime::TensorParallelRoutedExpertProvider<B, Error = RoutedTextExecutionError>> = Box::new(handle);
            (id, provider)
        });
        eredu_runtime::RoutedBankProviders::new(banks)
            .map_err(|cause| RoutedTextExecutionError::Contract(cause.to_string()))
    }
}

struct ResidentPartitionBank {
    source: RetainedPartitionResidentSource,
    id: RoutedBankId,
}
impl ResidentPartitionBank {
    fn provider(&self) -> Result<&PartitionResidentProvider, RoutedTextExecutionError> {
        self.source.banks.bank(self.id).ok_or_else(|| RoutedTextExecutionError::Contract(
            "retained partition provider lost its original bank identity".into(),
        ))
    }
}

impl<B: GroupedNeuralBackend> RoutedExpertProvider<B> for &PartitionResidentProvider {
    type Error = RoutedTextExecutionError;
    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match *self {
            PartitionResidentProvider::Empty => RoutedExpertProvider::<B>::forward_grouped(&mut EmptyPartitionRoutedExpertProvider, resident, request, context),
            PartitionResidentProvider::Resident(provider) => RoutedExpertProvider::<B>::forward_grouped(&mut {provider}, resident, request, context),
        }
    }

    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match *self {
            PartitionResidentProvider::Empty => RoutedExpertProvider::<B>::forward_compact_grouped(&mut EmptyPartitionRoutedExpertProvider, resident, request, context),
            PartitionResidentProvider::Resident(provider) => RoutedExpertProvider::<B>::forward_compact_grouped(&mut {provider}, resident, request, context),
        }
    }

    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match *self {
            PartitionResidentProvider::Empty => RoutedExpertProvider::<B>::forward_relu2_routed(&mut EmptyPartitionRoutedExpertProvider, resident, request, context),
            PartitionResidentProvider::Resident(provider) => RoutedExpertProvider::<B>::forward_relu2_routed(&mut {provider}, resident, request, context),
        }
    }

    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match *self {
            PartitionResidentProvider::Empty => RoutedExpertProvider::<B>::forward_linear_routed(&mut EmptyPartitionRoutedExpertProvider, resident, request, context),
            PartitionResidentProvider::Resident(provider) => RoutedExpertProvider::<B>::forward_linear_routed(&mut {provider}, resident, request, context),
        }
    }
}

impl<B: GroupedNeuralBackend> RoutedExpertProvider<B> for ResidentPartitionBank {
    type Error = RoutedTextExecutionError;
    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_grouped(&mut self.provider()?, resident, request, context)
    }

    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_compact_grouped(&mut self.provider()?, resident, request, context)
    }

    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_relu2_routed(&mut self.provider()?, resident, request, context)
    }

    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_linear_routed(&mut self.provider()?, resident, request, context)
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend> eredu_runtime::TensorParallelRoutedExpertProvider<B> for &PartitionResidentProvider {
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        match *self {
            PartitionResidentProvider::Empty => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(&mut EmptyPartitionRoutedExpertProvider, resident, request, partitions, context),
            PartitionResidentProvider::Resident(provider) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(&mut {provider}, resident, request, partitions, context),
        }
    }

    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        match *self {
            PartitionResidentProvider::Empty => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_compact_grouped_tensor_parallel(&mut EmptyPartitionRoutedExpertProvider, resident, request, partitions, context),
            PartitionResidentProvider::Resident(provider) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_compact_grouped_tensor_parallel(&mut {provider}, resident, request, partitions, context),
        }
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        match *self {
            PartitionResidentProvider::Empty => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(&mut EmptyPartitionRoutedExpertProvider, resident, request, partitions, context),
            PartitionResidentProvider::Resident(provider) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(&mut {provider}, resident, request, partitions, context),
        }
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend> eredu_runtime::TensorParallelRoutedExpertProvider<B> for ResidentPartitionBank {
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&partitions), std::mem::size_of_val(&context), std::mem::size_of::<Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(&mut self.provider()?, resident, request, partitions, context)
    }

    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&partitions), std::mem::size_of_val(&context), std::mem::size_of::<Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_compact_grouped_tensor_parallel(&mut self.provider()?, resident, request, partitions, context)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&partitions), std::mem::size_of_val(&context), std::mem::size_of::<Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(&mut self.provider()?, resident, request, partitions, context)
    }
}
