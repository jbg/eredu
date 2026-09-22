//! Exact model evidence rows, consumed by the same two-side receipt worker.
use super::*;
use crate::composition::mlx::session::intervention::PreparedModelInterventions;
use crate::composition::mlx::session::model_session::text_quote::capture::partition::evidence::EvidenceSource;
use eredu_runtime::working_memory::OwnedPartitionFragmentHostPlan;

type Plans = Vec<(usize, [OwnedPartitionFragmentHostPlan; 2])>;
type Funded = Vec<(usize, [PreparedPartitionFragmentHostFunding; 2])>;
struct ModelOperation {
    source: SharedCapturePlan,
    rows: [RowSource; 2],
    scalars: [Option<WorkspaceFloatingType>; 2],
}
/// Actual cold source rows and immutable loaded publication. The returned plans
/// must enter the same model role through its typed evidence admission.
pub(in crate::composition::mlx) struct ModelEvidenceHostSource {
    loaded: LoadedPartitionCapture,
    parent: SharedCapturePlan,
    original: OriginalInterventionSource,
    invocation: PartitionCaptureInvocation,
    rank: usize,
    overlay: Option<String>,
    operations: Vec<Option<ModelOperation>>,
    metadata: HostMetadataFunding,
}
/// Once-consumed model Hosts. Program preparation takes the same operation slots
/// as the Text adapter; neither this object nor its labels issue another role.
pub(in crate::composition::mlx) struct ModelEvidenceHosts {
    loaded: LoadedPartitionCapture,
    parent: SharedCapturePlan,
    original: OriginalInterventionSource,
    invocation: PartitionCaptureInvocation,
    rank: usize,
    overlay: Option<String>,
    operations: Vec<Option<Operation>>,
    _source_metadata: HostMetadataFunding,
    _metadata: HostMetadataFunding,
}
fn source_controls() -> Option<usize> {
    let parts = [
        controls()?,
        size_of::<ModelEvidenceHostSource>() * 2,
        size_of::<ModelOperation>() * 2,
        size_of::<Plans>() * 2,
        size_of::<[OwnedPartitionFragmentHostPlan; 2]>() * 2,
        size_of::<Result<(ModelEvidenceHostSource, Plans), Error>>(),
        size_of::<(
            &LoadedPartitionCapture,
            usize,
            Option<&str>,
            &SharedCapturePlan,
            &PreparedModelInterventions,
            &HostMetadataFunding,
        )>(),
        size_of::<[Prototype; 2]>(),
        size_of::<[RowSource; 2]>(),
        size_of::<Option<&[Option<eredu_core::capture::CaptureSkipReason>; 2]>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn bind_controls() -> Option<usize> {
    let parts = [
        size_of::<ModelEvidenceHostSource>(),
        size_of::<ModelEvidenceHosts>() * 2,
        size_of::<Funded>(),
        size_of::<Result<ModelEvidenceHosts, Error>>(),
        size_of::<std::vec::IntoIter<(usize, [PreparedPartitionFragmentHostFunding; 2])>>(),
        size_of::<std::iter::Enumerate<std::vec::IntoIter<Option<ModelOperation>>>>(),
        size_of::<std::array::IntoIter<RowSource, 2>>(),
        size_of::<std::array::IntoIter<PreparedPartitionFragmentHostFunding, 2>>(),
        size_of::<
            std::iter::Zip<
                std::array::IntoIter<RowSource, 2>,
                std::array::IntoIter<PreparedPartitionFragmentHostFunding, 2>,
            >,
        >(),
        size_of::<HostRow>() * 2,
        size_of::<Option<Operation>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl ModelEvidenceHostSource {
    pub(in crate::composition::mlx) fn prepare(
        loaded: &LoadedPartitionCapture,
        rank: usize,
        overlay: Option<&str>,
        parent: &SharedCapturePlan,
        model: &PreparedModelInterventions,
        metadata: &HostMetadataFunding,
    ) -> Result<(Self, Plans), Error> {
        metadata.reserve_metadata(source_controls().ok_or_else(overflow)?)?;
        let original = model.source();
        let invocation = EvidenceSource::invocation(model).ok_or_else(unknown)?;
        let logical = invocation
            .logical()
            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
        let admission = original.plan().admission();
        let (artifact, execution, setup) = loaded.source_labels();
        if rank >= setup.participant_count()
            || admission.request() != parent.admission().request()
            || admission.text_origin() != parent.admission().text_origin()
            || admission.invocation_bounds() != parent.admission().invocation_bounds()
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let mut operations = metadata.metadata_vec(admission.plan().operations.len())?;
        let mut plans = metadata.metadata_vec(admission.plan().operations.len())?;
        for (operation, entry) in admission.plan().operations.iter().enumerate() {
            if entry.evidence == InterventionEvidence::None
                || !entry
                    .schedule
                    .includes(invocation.phase, invocation.prediction)
                || !EvidenceSource::reached(model, operation, metadata)?
            {
                operations.push(None);
                continue;
            }
            let source = original
                .plan()
                .evidence(operation)
                .ok_or_else(unknown)?
                .shared_geometry_source();
            if source.admission().plan().selections.len() != 2 {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            let mut context = PreparedPartitionCaptureRunIdentity::prepare_host_context_at(
                source,
                artifact,
                execution,
                setup,
                overlay,
                invocation.phase,
                invocation.prediction,
                Some(logical),
                invocation.receipt_window(),
                metadata,
            )
            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
            let before = super::prepare(
                loaded,
                parent,
                original,
                operation,
                source,
                &context,
                FrameGeometry::Invocation(invocation),
                rank,
                metadata,
            )?;
            context.selection_index = 1;
            let after = super::prepare(
                loaded,
                parent,
                original,
                operation,
                source,
                &context,
                FrameGeometry::Invocation(invocation),
                rank,
                metadata,
            )?;
            let [before, after] = [before, after];
            let [
                super::super::RowSource::Evidence(before_source),
                super::super::RowSource::Evidence(after_source),
            ] = [before.source, after.source]
            else {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            };
            let before = OwnedPartitionFragmentHostPlan::prepare(before.receipt, None)
                .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
            let after = OwnedPartitionFragmentHostPlan::prepare(after.receipt, None)
                .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
            plans.push((operation, [before, after]));
            let scalars = [
                model.partition_evidence_scalar(
                    invocation.phase,
                    invocation.prediction,
                    operation,
                    eredu_core::capture::InterventionEvidenceSide::Before,
                    metadata,
                )?,
                model.partition_evidence_scalar(
                    invocation.phase,
                    invocation.prediction,
                    operation,
                    eredu_core::capture::InterventionEvidenceSide::After,
                    metadata,
                )?,
            ];
            operations.push(Some(ModelOperation {
                source: source.clone(),
                rows: [before_source, after_source],
                scalars,
            }));
        }
        Ok((
            Self {
                loaded: loaded.clone(),
                parent: parent.clone(),
                original: original.clone(),
                invocation,
                rank,
                overlay: overlay
                    .map(|label| metadata.metadata_string(format_args!("{label}")))
                    .transpose()?,
                operations,
                metadata: metadata.clone(),
            },
            plans,
        ))
    }
    pub(in crate::composition::mlx) fn maximum_epoch_run_length(&self) -> Option<usize> {
        PreparedPartitionCaptureRunIdentity::epoch_run_name_length(
            self.loaded.source_labels().2,
            eredu_core::DistributedCommitEpoch::new(u64::MAX)?,
        )
    }
    pub(in crate::composition::mlx) fn epoch_metadata_bytes(&self) -> Option<usize> {
        PreparedPartitionCaptureRunIdentity::epoch_metadata_bytes(
            self.loaded.source_labels().2,
            eredu_core::DistributedCommitEpoch::new(u64::MAX)?,
        )
    }
    pub(in crate::composition::mlx) fn operation_indices(
        &self,
    ) -> impl Iterator<Item = usize> + '_ {
        self.operations
            .iter()
            .enumerate()
            .filter_map(|(index, row)| row.as_ref().map(|_| index))
    }
    pub(in crate::composition::mlx) fn bind(
        self,
        hosts: Funded,
        metadata: &HostMetadataFunding,
    ) -> Result<ModelEvidenceHosts, Error> {
        metadata.reserve_metadata(bind_controls().ok_or_else(overflow)?)?;
        if hosts.len() != self.operations.iter().flatten().count() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let mut actual = hosts.into_iter();
        let mut operations = metadata.metadata_vec(self.operations.len())?;
        for (operation, row) in self.operations.into_iter().enumerate() {
            let Some(row) = row else {
                operations.push(None);
                continue;
            };
            let (index, hosts) = actual.next().ok_or_else(unknown)?;
            if index != operation {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            let mut rows = metadata.metadata_vec(2)?;
            for (source, host) in row.rows.into_iter().zip(hosts) {
                rows.push(Some(HostRow {
                    source: super::super::RowSource::Evidence(source),
                    geometry: FrameGeometry::Invocation(self.invocation),
                    prediction: self.invocation.prediction,
                    fragments: host.original_receipt_limits().max_fragments,
                    host,
                }));
            }
            operations.push(Some(Operation {
                source: row.source,
                selection: None,
                hosts: rows,
            }));
        }
        Ok(ModelEvidenceHosts {
            loaded: self.loaded,
            parent: self.parent,
            original: self.original,
            invocation: self.invocation,
            rank: self.rank,
            overlay: self.overlay,
            operations,
            _source_metadata: self.metadata,
            _metadata: metadata.clone(),
        })
    }
}
impl ModelEvidenceHosts {
    pub(in crate::composition::mlx) fn prepare<'t, T: NativePartitionCaptureTransport>(
        &mut self,
        transport: &'t T,
        parent: &SharedCapturePlan,
        model: &PreparedModelInterventions,
        metadata: &HostMetadataFunding,
    ) -> Result<
        Vec<
            Option<eredu_runtime::capture::partition::PreparedPartitionInterventionEvidence<'t, T>>,
        >,
        Error,
    >
    where
        <T::Completion as eredu_core::Completion>::Error: Send + Sync + 'static,
    {
        metadata.reserve_metadata(prepare_controls::<T>().ok_or_else(overflow)?)?;
        if !self.parent.same_storage(parent)
            || !self.original.same_source(model.source())
            || EvidenceSource::invocation(model) != Some(self.invocation)
            || transport.capture_rank() != self.rank
            || transport.participant_count() != self.loaded.source_labels().2.participant_count()
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let (artifact, execution, setup) = self.loaded.source_labels();
        crate::composition::mlx::session::model_session::text_quote::capture::partition::evidence::prepare(
            transport,artifact,execution,setup,self.overlay.as_deref(),parent,model,
            self.invocation.phase,self.invocation.prediction,&mut self.operations,metadata)
    }
}

fn prepare_controls<T: NativePartitionCaptureTransport>() -> Option<usize> {
    use eredu_runtime::capture::partition::PreparedPartitionInterventionEvidence as E;
    let parts = [
        size_of::<(
            &mut ModelEvidenceHosts,
            &T,
            &SharedCapturePlan,
            &PreparedModelInterventions,
            &HostMetadataFunding,
        )>(),
        size_of::<Result<Vec<Option<E<'_, T>>>, Error>>(),
        size_of::<(&str, &str, eredu_runtime::CommunicationSessionIdentity)>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl ModelEvidenceHostSource {
    /// Borrow the exact cold receipts and local scalar facts before their Host
    /// plans move into the typed model admission. No receipt is reconstructed.
    pub(in crate::composition::mlx) fn receipts<'a>(
        &'a self,
        plans: &'a [(usize, [OwnedPartitionFragmentHostPlan; 2])],
    ) -> Result<
        impl Iterator<
            Item = (
                usize,
                usize,
                &'a PartitionCaptureReceiptPlan,
                Option<WorkspaceFloatingType>,
            ),
        > + 'a,
        Error,
    > {
        if plans.len() != self.operation_indices().count() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        for ((operation, pair), expected) in plans.iter().zip(self.operation_indices()) {
            let row = self
                .operations
                .get(*operation)
                .and_then(Option::as_ref)
                .ok_or_else(unknown)?;
            if *operation != expected {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            for (side, plan) in pair.iter().enumerate() {
                let receipt = plan.receipt();
                let context = receipt.context();
                if !receipt.shared_plan_source().same_storage(&row.source)
                    || context.selection_index != side
                    || context.phase != self.invocation.phase
                    || context.prediction != self.invocation.prediction
                    || !context
                        .matches_invocation(Some(self.invocation.physical), self.invocation.window)
                {
                    return Err(memory(WorkingMemoryError::IdentityMismatch));
                }
            }
        }
        Ok(plans.iter().flat_map(move |(operation, pair)| {
            pair.iter().enumerate().map(move |(side, plan)| {
                (
                    *operation,
                    side,
                    plan.receipt(),
                    self.operations[*operation]
                        .as_ref()
                        .expect("authenticated operation")
                        .scalars[side],
                )
            })
        }))
    }
    /// Constructor/binding populations for the same two-side worker. Transaction
    /// entry, receipt delivery, intervention wrapper and native callbacks remain
    /// separate queries owned by those workers.
    pub(in crate::composition::mlx) fn execution_metadata_bytes<
        T: NativePartitionCaptureTransport,
    >(
        &self,
    ) -> Result<usize, Error>
    where
        <T::Completion as eredu_core::Completion>::Error: Send + Sync + 'static,
    {
        use eredu_nn::workspace::WorkspaceContext as W;
        use eredu_runtime::capture::partition::PreparedPartitionInterventionEvidence as E;
        let (artifact, execution, setup) = self.loaded.source_labels();
        let mut bytes = bind_controls()
            .and_then(|bytes| bytes.checked_add(prepare_controls::<T>()?))
            .and_then(|bytes| {
                bytes.checked_add(super::super::super::evidence::model_control_bytes::<T>()?)
            })
            .and_then(|bytes| {
                bytes.checked_add(W::metadata_vec_bytes::<Option<Operation>>(
                    self.operations.len(),
                )?)
            })
            .and_then(|bytes| {
                bytes.checked_add(W::metadata_vec_bytes::<Option<E<'_, T>>>(
                    self.operations.len(),
                )?)
            })
            .ok_or_else(overflow)?;
        for row in self.operations.iter().flatten() {
            for part in [
                W::metadata_vec_bytes::<Option<HostRow>>(2),
                PreparedPartitionCaptureRunIdentity::host_context_metadata_bytes(
                    &row.source,
                    artifact,
                    execution,
                    setup,
                    self.overlay.as_deref(),
                ),
                PreparedPartitionCaptureProgram::<T>::selected_host_context_metadata_bytes(
                    &row.source,
                    artifact,
                    execution,
                    setup,
                    self.overlay.as_deref(),
                    2,
                ),
                PreparedPartitionCaptureProgram::<T>::projected_rows_metadata_bytes(2, 2, 0),
                PreparedModelInterventions::partition_evidence_scalar_control_bytes()
                    .and_then(|bytes| bytes.checked_mul(2)),
            ] {
                bytes = bytes
                    .checked_add(part.ok_or_else(overflow)?)
                    .ok_or_else(overflow)?;
            }
            for (side, source) in row.rows.iter().enumerate() {
                bytes = bytes
                    .checked_add(source.execution_metadata_bytes(
                        &row.source,
                        self.rank,
                        row.scalars[side],
                        &self.metadata,
                    )?)
                    .ok_or_else(overflow)?;
            }
        }
        Ok(bytes)
    }
    /// Same selected coordinate-source vote wrapper for this exact side.
    pub(in crate::composition::mlx) fn projected_execution_metadata_bytes<
        T: NativePartitionCaptureTransport,
    >(
        &self,
        operation: usize,
        side: usize,
    ) -> Result<usize, Error>
    where
        <T::Completion as eredu_core::Completion>::Error: Send + Sync + 'static,
    {
        let row = self
            .operations
            .get(operation)
            .and_then(Option::as_ref)
            .ok_or_else(unknown)?;
        row.rows
            .get(side)
            .ok_or_else(unknown)?
            .projected_execution_metadata_bytes::<T>(&row.source, &self.metadata)
    }
}
