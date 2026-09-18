//! Shared native/cold addressable declaration and exact borrowed callback frame.
use super::*;
use eredu_nn::workspace::{
    WorkspaceAddressableRegionView, WorkspaceExpertKernel, WorkspaceMetadataError,
};

// Every generic field is a borrowed tensor/context/observer. There are no
// by-value backend tensors or provider objects in this frame. Both native
// execution and cold source construction use this same layout.
pub(super) struct Callback<'input, 'observer, 'context, P, T: Tensor> {
    pub request: Option<RoutedExpertRequest<'input, 'observer, T>>,
    pub error: Option<RoutedTextExecutionError>,
    pub chunks: eredu_runtime::expert::AddressableChunkPlan,
    pub partitions: Option<usize>,
    pub context: &'context T::Context,
    pub run: fn(
        &mut P,
        RoutedExpertRequest<'input, 'observer, T>,
        Option<usize>,
        eredu_runtime::expert::AddressableChunkPlan,
        Option<&eredu_nn::workspace::HostMetadataFunding>,
        &T::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<T>, RoutedTextExecutionError>,
}
pub(crate) fn callback_control_bytes<T: Tensor>() -> Option<usize> {
    std::mem::size_of::<Callback<'_, '_, '_, (), T>>()
        .checked_add(std::mem::size_of::<&mut Callback<'_, '_, '_, (), T>>())?
        // The actual adapter contains one owner reference, one accessor fn
        // pointer and one callback trait-object loan; no P or M by value.
        .checked_add(eredu_runtime::expert::BorrowedIndexedInvocation::<(), (), T>::control_bytes())
}
pub(super) fn declaration<'a>(
    owner_group: &'a str,
    bank: RoutedBankId,
    unit: usize,
    pass: eredu_runtime::ExpertPass,
    chunks: eredu_runtime::expert::AddressableChunkPlan,
    local_members: Option<&'a [usize]>,
    kernel: WorkspaceExpertKernel<'a>,
    partitions: Option<usize>,
    compact_scratch_bytes: u64,
    bulk_target_bytes: u64,
    callback_control_bytes: usize,
) -> Result<WorkspaceAddressableRegionView<'a>, WorkspaceMetadataError> {
    let source = WorkspaceAddressableRegionView {
        owner_group,
        bank: bank.value(),
        unit,
        prefill: matches!(pass, eredu_runtime::ExpertPass::Prefill),
        chunks: chunks.workspace_source(),
        local_members,
        kernel,
        tensor_partitions: partitions,
        compact_scratch_bytes,
        bulk_target_bytes,
        callback_control_bytes,
    };
    source.validate()?;
    Ok(source)
}
impl SelectedRoutedBank {
    /// Borrows the already selected physical equation and member-byte policy.
    /// This creates only a descriptor; the backend must bind its real bank source.
    pub(crate) fn addressable_workspace_source<'a, T: Tensor>(
        &'a self,
        request: &RoutedExpertRequest<'_, '_, T>,
        options: eredu_runtime::ParameterBankLoadOptions,
        partitions: Option<usize>,
        compact: bool,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<WorkspaceAddressableRegionView<'a>, eredu_nn::Error> {
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            &RoutedExpertRequest<'_, '_, T>,
            eredu_runtime::ParameterBankLoadOptions,
            Option<usize>,
            bool,
            WorkspaceAddressableRegionView<'_>,
            WorkspaceExpertKernel<'_>,
            Option<&[usize]>,
            eredu_nn::workspace::ExpertRegionInputShape,
            Option<u64>,
            Option<&PartitionSource>,
            &PartitionSource,
            eredu_runtime::expert::AddressableChunkPlan,
            Result<WorkspaceAddressableRegionView<'_>, eredu_nn::Error>,
            std::slice::Iter<'_, eredu_runtime::AddressableBankMember>,
            [usize; 4],
        )>())?;
        validate_route_cardinality(
            request.routes,
            if compact {
                1
            } else {
                *self
                    .routes_by_unit
                    .get(&request.layer)
                    .ok_or(WorkspaceMetadataError::Unqualified)?
            },
        )
        .map_err(|_| eredu_nn::Error::from(WorkspaceMetadataError::Unqualified))?;
        let (kernel, local_members) = match &self.plan {
            RoutedGroupedPlan::Linear(plan) => (
                WorkspaceExpertKernel::Linear(
                    plan.unit_spec(self.owner_group.as_str(), request.layer)
                        .ok_or(WorkspaceMetadataError::Unqualified)?,
                ),
                (!plan.unit_is_replicated(request.layer))
                    .then(|| plan.local_global_group_indices()),
            ),
            RoutedGroupedPlan::Gated(plan) => (
                WorkspaceExpertKernel::Gated(
                    plan.unit_spec(self.owner_group.as_str(), request.layer)
                        .ok_or(WorkspaceMetadataError::Unqualified)?,
                ),
                (!plan.unit_is_replicated(request.layer))
                    .then(|| plan.local_global_group_indices()),
            ),
            RoutedGroupedPlan::Relu2(plan) => (
                WorkspaceExpertKernel::Relu2(
                    plan.unit_spec(self.owner_group.as_str(), request.layer)
                        .ok_or(WorkspaceMetadataError::Unqualified)?,
                ),
                (!plan.unit_is_replicated(request.layer))
                    .then(|| plan.local_global_group_indices()),
            ),
        };
        let members = match kernel {
            WorkspaceExpertKernel::Linear(v) => v.group_count(),
            WorkspaceExpertKernel::Gated(v) => v.group_count(),
            WorkspaceExpertKernel::Relu2(v) => v.group_count(),
        };
        let geometry = eredu_nn::workspace::ExpertRegionInputShape::inspect(
            request.input.shape(),
            request.routes.group_indices().shape(),
        )?;
        let access = request.pass.parameter_bank_access();
        let maximum = match self.plan.partition_source() {
            Some(source) => {
                context.charge_metadata(
                    PartitionSource::maximum_control_bytes()
                        .ok_or(WorkspaceMetadataError::Overflow)?,
                )?;
                Some(source.maximum(request.layer, options)?)
            }
            None if access == eredu_runtime::ParameterBankAccess::Bulk => Some(
                self.addressable_members
                    .iter()
                    .filter(|member| member.key().unit() == request.layer)
                    .map(|member| member.selected_bytes())
                    .max()
                    .ok_or(WorkspaceMetadataError::Unqualified)?,
            ),
            None => None,
        };
        let chunks = eredu_runtime::expert::AddressableChunkPlan::new(
            usize::try_from(geometry.rows).map_err(|_| WorkspaceMetadataError::Overflow)?,
            usize::try_from(geometry.routes).map_err(|_| WorkspaceMetadataError::Overflow)?,
            usize::try_from(members).map_err(|_| WorkspaceMetadataError::Overflow)?,
            access,
            maximum,
            options.prefill_compact_bank_target_bytes(),
        )
        .map_err(eredu_nn::Error::backend_retained_source)?;
        declaration(
            self.owner_group.as_str(),
            request.bank,
            request.layer,
            request.pass,
            chunks,
            local_members,
            kernel,
            partitions,
            options.compact_bank_scratch_bytes(),
            options.prefill_compact_bank_target_bytes(),
            callback_control_bytes::<T>().ok_or(WorkspaceMetadataError::Overflow)?,
        )
        .map_err(Into::into)
    }
}

/// One construction-owned physical source shared by the selected bank and its
/// native/cold execution plans. Completion moves the checked adapter map here;
/// quotes can only borrow it and cannot initialize or revise it.
#[derive(Debug)]
pub(crate) struct PartitionSource {
    owner_group: String,
    options: eredu_runtime::ParameterBankLoadOptions,
    completed: std::sync::OnceLock<BTreeMap<ParameterBankKey, u64>>,
}
// Plan equality describes immutable selected semantics. Physical completion is
// authenticated independently by the actual bank and exact once-only map join.
impl PartialEq for PartitionSource {
    fn eq(&self, other: &Self) -> bool {
        self.owner_group == other.owner_group && self.options == other.options
    }
}
impl PartitionSource {
    pub(crate) fn new(
        owner_group: &eredu_runtime::ExecutionGroupId,
        options: eredu_runtime::ParameterBankLoadOptions,
    ) -> Self {
        Self {
            owner_group: owner_group.as_str().to_owned(),
            options,
            completed: std::sync::OnceLock::new(),
        }
    }
    pub(crate) fn complete(
        &self,
        bytes: BTreeMap<ParameterBankKey, u64>,
        options: eredu_runtime::ParameterBankLoadOptions,
    ) -> Result<(), RoutedTextExecutionError> {
        if options != self.options {
            return Err(RoutedTextExecutionError::Contract(
                "completed bank source options differ".into(),
            ));
        }
        match self.completed.set(bytes) {
            Ok(()) => Ok(()),
            Err(bytes) if self.completed.get() == Some(&bytes) => Ok(()),
            Err(_) => Err(RoutedTextExecutionError::Contract(
                "completed bank physical source differs".into(),
            )),
        }
    }
    pub(crate) fn require_completion(&self) -> Result<(), WorkspaceMetadataError> {
        self.completed
            .get()
            .ok_or(WorkspaceMetadataError::Unqualified)
            .map(|_| ())
    }
    pub(crate) fn completion_control_bytes() -> usize {
        std::mem::size_of::<(
            &Self,
            Option<&BTreeMap<ParameterBankKey, u64>>,
            Result<&BTreeMap<ParameterBankKey, u64>, WorkspaceMetadataError>,
            &BTreeMap<ParameterBankKey, u64>,
            Result<(), WorkspaceMetadataError>,
        )>()
    }
    pub(crate) fn maximum(
        &self,
        unit: usize,
        options: eredu_runtime::ParameterBankLoadOptions,
    ) -> Result<u64, WorkspaceMetadataError> {
        if options != self.options {
            return Err(WorkspaceMetadataError::Unqualified);
        }
        self.completed
            .get()
            .ok_or(WorkspaceMetadataError::Unqualified)?
            .iter()
            .filter(|(key, _)| key.unit() == unit)
            .map(|(_, bytes)| *bytes)
            .max()
            .ok_or(WorkspaceMetadataError::Unqualified)
    }
    pub(crate) fn declaration<'a>(
        &'a self,
        region: eredu_nn::workspace::WorkspaceExpertRegionView<'a>,
        local_members: &'a [usize],
    ) -> Result<WorkspaceAddressableRegionView<'a>, WorkspaceMetadataError> {
        let rows = region
            .maximum_received_rows()
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let maximum = self.maximum(region.unit, self.options)?;
        let access = if region.prefill {
            eredu_runtime::ParameterBankAccess::Bulk
        } else {
            eredu_runtime::ParameterBankAccess::Incremental
        };
        let chunks = eredu_runtime::expert::AddressableChunkPlan::new(
            rows,
            1,
            local_members.len(),
            access,
            Some(maximum),
            self.options.prefill_compact_bank_target_bytes(),
        )
        .map_err(|_| WorkspaceMetadataError::Unqualified)?;
        declaration(
            &self.owner_group,
            RoutedBankId::new(region.bank),
            region.unit,
            if region.prefill {
                eredu_runtime::ExpertPass::Prefill
            } else {
                eredu_runtime::ExpertPass::Decode
            },
            chunks,
            Some(local_members),
            region.kernel,
            region.tensor_partitions,
            self.options.compact_bank_scratch_bytes(),
            self.options.prefill_compact_bank_target_bytes(),
            callback_control_bytes::<eredu_nn::workspace::WorkspaceTensor>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        std::mem::size_of::<(
            &Self,
            eredu_nn::workspace::WorkspaceExpertRegionView<'_>,
            &[usize],
            usize,
            u64,
            Option<u64>,
            eredu_runtime::ParameterBankAccess,
            eredu_runtime::expert::AddressableChunkPlan,
            WorkspaceAddressableRegionView<'_>,
            Result<WorkspaceAddressableRegionView<'_>, WorkspaceMetadataError>,
        )>()
        .checked_add(eredu_runtime::expert::AddressableChunkPlan::control_bytes())?
        .checked_add(WorkspaceAddressableRegionView::control_bytes()?)?
        .checked_add(Self::maximum_control_bytes()?)
    }
    pub(crate) fn maximum_control_bytes() -> Option<usize> {
        Some(std::mem::size_of::<(
            &Self,
            usize,
            eredu_runtime::ParameterBankLoadOptions,
            Option<&BTreeMap<ParameterBankKey, u64>>,
            Result<&BTreeMap<ParameterBankKey, u64>, WorkspaceMetadataError>,
            &BTreeMap<ParameterBankKey, u64>,
            std::collections::btree_map::Iter<'_, ParameterBankKey, u64>,
            (&ParameterBankKey, &u64),
            &(&ParameterBankKey, &u64),
            &usize,
            Option<u64>,
            Result<u64, WorkspaceMetadataError>,
        )>())
    }
}
impl RoutedGroupedPlan {
    pub(crate) fn partition_source(&self) -> Option<&PartitionSource> {
        match self {
            Self::Gated(p) => p.addressable_source(),
            Self::Relu2(p) => p.addressable_source(),
            Self::Linear(p) => p.addressable_source(),
        }
    }
    pub(crate) fn retained_partition_source(&self) -> Option<&std::sync::Arc<PartitionSource>> {
        match self {
            Self::Gated(p) => p.retained_addressable_source(),
            Self::Relu2(p) => p.retained_addressable_source(),
            Self::Linear(p) => p.retained_addressable_source(),
        }
    }
    pub(crate) fn bind_partition_source(&mut self, source: std::sync::Arc<PartitionSource>) {
        match self {
            Self::Gated(p) => p.bind_addressable_source(source),
            Self::Relu2(p) => p.bind_addressable_source(source),
            Self::Linear(p) => p.bind_addressable_source(source),
        }
    }
}
impl SelectedRoutedBank {
    pub(crate) fn complete_partition_source(
        &self,
        bytes: BTreeMap<ParameterBankKey, u64>,
        options: eredu_runtime::ParameterBankLoadOptions,
    ) -> Result<(), RoutedTextExecutionError> {
        self.plan
            .partition_source()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(
                    "selected partition omitted its physical source owner".into(),
                )
            })?
            .complete(bytes, options)
    }
    pub(crate) fn adopt_partition_source(&mut self, actual: &Self) -> Result<(), String> {
        if self != actual {
            return Err("retained partition bank geometry differs".into());
        }
        self.adopt_execution_source(&actual.plan)
    }
    pub(crate) fn adopt_execution_source(
        &mut self,
        actual: &RoutedGroupedPlan,
    ) -> Result<(), String> {
        if &self.plan != actual {
            return Err("retained partition execution geometry differs".into());
        }
        match (
            self.plan.retained_partition_source(),
            actual.retained_partition_source(),
        ) {
            (Some(_), Some(source)) => self.plan.bind_partition_source(source.clone()),
            (None, None) => {}
            _ => return Err("retained partition bank source policy differs".into()),
        }
        Ok(())
    }
}

/// Repeated construction borrows the first published owner after comparing the
/// complete local bank geometry. It cannot publish into a discarded retry slot.
pub(crate) fn adopt_bank_sources(
    banks: &mut BTreeMap<RoutedBankId, SelectedRoutedBank>,
    actual: &BTreeMap<RoutedBankId, SelectedRoutedBank>,
) -> Result<(), String> {
    if banks.keys().ne(actual.keys()) {
        return Err("retained partition bank identities differ".into());
    }
    // Validate every bank before replacing any owner.
    if banks.iter().any(|(id, bank)| bank != &actual[id]) {
        return Err("retained partition bank geometry differs".into());
    }
    for (id, bank) in banks {
        bank.adopt_partition_source(&actual[id])?;
    }
    Ok(())
}
