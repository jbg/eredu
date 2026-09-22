//! Exact cold fragment sources transferred into the existing model Host bank.
use super::*;
mod protocol;
use crate::composition::mlx::session::model_session::partition_capture::LoadedPartitionCapture;
use eredu_runtime::working_memory::OwnedPartitionFragmentHostPlan;

/// Sealed architecture source mode; physical geometry does not select it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::composition::mlx) enum PartitionCaptureRowMode {
    Complete,
    Contiguous,
    Routed,
}

/// Loaded architecture projection rows for one independently admitted callback.
/// The returned Host quotations are consumed by its real model account; this
/// companion never creates another account, native budget or transport role.
pub(in crate::composition::mlx) struct PartitionCaptureInvocationSource {
    loaded: LoadedPartitionCapture,
    source: SharedCapturePlan,
    invocation: PartitionCaptureInvocation,
    rank: usize,
    overlay: Option<String>,
    producers: Vec<Option<usize>>,
    rows: Vec<Option<RowSource>>,
    protocol: Vec<(usize, usize)>,
    complete_receipts: Vec<PartitionCaptureReceiptPlan>,
    metadata: HostMetadataFunding,
}
impl PartitionCaptureInvocationSource {
    pub(in crate::composition::mlx) fn capture_source(&self) -> &SharedCapturePlan {
        &self.source
    }
    pub(in crate::composition::mlx) fn prepare(
        loaded: &LoadedPartitionCapture,
        rank: usize,
        overlay: Option<&str>,
        source: &SharedCapturePlan,
        invocation: PartitionCaptureInvocation,
        metadata: &HostMetadataFunding,
    ) -> Result<(Self, Vec<OwnedPartitionFragmentHostPlan>), Error> {
        let parts = [size_of::<Self>()*2, size_of::<Vec<OwnedPartitionFragmentHostPlan>>()*2,
            size_of::<OwnedPartitionFragmentHostPlan>()*2,
            size_of::<(Self,Vec<OwnedPartitionFragmentHostPlan>)>(),
            size_of::<Result<(Self,Vec<OwnedPartitionFragmentHostPlan>),Error>>(),
            size_of::<(&LoadedPartitionCapture,usize,Option<&str>,&SharedCapturePlan,PartitionCaptureInvocation,&HostMetadataFunding)>(),
            size_of::<Option<HostAdmission>>(),size_of::<std::vec::IntoIter<Frame<Prototype>>>(),
            size_of::<std::vec::IntoIter<Option<Prototype>>>(),
            size_of::<Vec<Option<RowSource>>>(),size_of::<Vec<Option<usize>>>(),
            eredu_architectures::component_partition::ComponentPartitionLayouts::complete_capture_source_control_bytes().ok_or_else(overflow)?,
        ];
        metadata.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(overflow)?,
        )?;
        let logical = invocation
            .logical()
            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
        source
            .admission()
            .geometry_at(invocation.phase, invocation.prediction, Some(logical))
            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
        source
            .admission()
            .geometry_at(
                invocation.phase,
                invocation.prediction,
                Some(invocation.physical),
            )
            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
        let (_, _, setup) = loaded.source_labels();
        if rank >= setup.participant_count() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let projected = HostAdmission::prepare_source(
            loaded,
            rank,
            overlay,
            source,
            FrameGeometry::Invocation(invocation),
            invocation.prediction,
            metadata,
        )?;
        let count = source.admission().plan().selections.len();
        let mut producers = metadata.metadata_vec(count)?;
        let mut protocol = metadata.metadata_vec(count)?;
        let mut complete_receipts = metadata.metadata_vec(count)?;
        let (artifact, execution, _) = loaded.source_labels();
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
        for (index, selection) in source.admission().plan().selections.iter().enumerate() {
            let producer = loaded
                .layouts()
                .complete_capture_source(&selection.path)
                .map(|value| value.producer());
            if producer.is_some_and(|value| value >= setup.participant_count()) {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            if let Some(producer) = producer.filter(|_| {
                selection
                    .schedule
                    .includes(invocation.phase, invocation.prediction)
            }) {
                context.selection_index = index;
                let mut ledger = CaptureLedger::new(source.admission());
                ledger.begin_step();
                let receipt = PartitionCaptureReceiptPlan::new_complete_shared_funded(
                    source,
                    &context,
                    producer,
                    setup.participant_count(),
                    PartitionCaptureReceiptLimits {
                        max_producers: setup.participant_count(),
                        max_fragments: 1,
                        max_record_bytes: source.admission().plan().limits.per_step.encoded_bytes,
                    },
                    metadata,
                    &mut ledger,
                )
                .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
                protocol.push((
                    index,
                    receipt
                        .maximum_payload_words()
                        .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?,
                ));
                complete_receipts.push(receipt);
            }
            producers.push(producer);
        }
        let mut plans = metadata.metadata_vec(count)?;
        let mut rows = metadata.metadata_vec(count)?;
        match projected {
            Some(projected) => {
                if projected.frames.len() != 1 {
                    return Err(memory(WorkingMemoryError::IdentityMismatch));
                }
                for frame in projected.frames {
                    if frame.prediction != invocation.prediction {
                        return Err(memory(WorkingMemoryError::IdentityMismatch));
                    }
                    for row in frame
                        .rows
                        .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
                    {
                        match row {
                            Some(row) => {
                                protocol.push((
                                    row.receipt.context().selection_index,
                                    row.receipt.maximum_payload_words().map_err(|cause| {
                                        Error::Neural(metadata.metadata_source(cause))
                                    })?,
                                ));
                                plans.push(
                                    OwnedPartitionFragmentHostPlan::prepare(row.receipt, None)
                                        .map_err(|cause| {
                                            Error::Neural(metadata.metadata_source(cause))
                                        })?,
                                );
                                rows.push(Some(row.source));
                            }
                            None => rows.push(None),
                        }
                    }
                }
            }
            None => rows.resize_with(count, || None),
        }
        let overlay = overlay
            .map(|value| metadata.metadata_string(format_args!("{value}")))
            .transpose()?;
        Ok((
            Self {
                loaded: loaded.clone(),
                source: source.clone(),
                invocation,
                rank,
                overlay,
                producers,
                rows,
                protocol,
                complete_receipts,
                metadata: metadata.clone(),
            },
            plans,
        ))
    }
    /// Reached receipt rows and their checked payload widths. Coordination is one
    /// frame-level occurrence; fixed per-row widths come from FrameKind itself.
    pub(in crate::composition::mlx) fn protocol_rows(
        &self,
    ) -> impl ExactSizeIterator<Item = (usize, usize)> + '_ {
        self.protocol.iter().copied()
    }
    /// Exact complete receipt prototypes retained by the same cold source worker.
    pub(in crate::composition::mlx) fn complete_receipts(
        &self,
    ) -> impl ExactSizeIterator<Item = &PartitionCaptureReceiptPlan> {
        self.complete_receipts.iter()
    }
    /// Mode of one reached selected row, authenticated by the retained source.
    pub(in crate::composition::mlx) fn protocol_row_mode(
        &self,
        index: usize,
    ) -> Option<PartitionCaptureRowMode> {
        if !self.protocol.iter().any(|(row, _)| *row == index) {
            return None;
        }
        if self.producers.get(index)?.is_some() {
            return Some(PartitionCaptureRowMode::Complete);
        }
        match self.rows.get(index)?.as_ref()? {
            RowSource::Contiguous(_) => Some(PartitionCaptureRowMode::Contiguous),
            RowSource::Routed(_) => Some(PartitionCaptureRowMode::Routed),
            RowSource::Evidence(_) => None,
        }
    }
    /// Finite epoch-label writer envelope from this exact loaded setup.
    pub(in crate::composition::mlx) fn epoch_metadata_bytes(&self) -> Option<usize> {
        let (_, _, setup) = self.loaded.source_labels();
        PreparedPartitionCaptureRunIdentity::epoch_metadata_bytes(
            setup,
            eredu_core::DistributedCommitEpoch::new(u64::MAX)?,
        )
    }
    /// Maximum run label length formatted by the actual positive u64 epoch.
    pub(in crate::composition::mlx) fn maximum_epoch_run_length(&self) -> Option<usize> {
        let (_, _, setup) = self.loaded.source_labels();
        PreparedPartitionCaptureRunIdentity::epoch_run_name_length(
            setup,
            eredu_core::DistributedCommitEpoch::new(u64::MAX)?,
        )
    }
    /// Actual selected projected-row execution wrappers; receipt, Program and
    /// transport callback populations are accounted by their owning workers.
    pub(in crate::composition::mlx) fn projected_execution_metadata_bytes<
        T: NativePartitionCaptureTransport,
    >(
        &self,
    ) -> Result<usize, Error>
    where
        <T::Completion as eredu_core::Completion>::Error: Send + Sync + 'static,
    {
        let mut bytes = 0usize;
        for row in self.rows.iter().flatten() {
            bytes = bytes
                .checked_add(
                    row.projected_execution_metadata_bytes::<T>(&self.source, &self.metadata)?,
                )
                .ok_or_else(overflow)?;
        }
        Ok(bytes)
    }
    fn bind_control_bytes<T>() -> Option<usize> {
        let parts = [
            size_of::<(
                Self,
                &[Option<WorkspaceFloatingType>],
                Vec<PreparedPartitionFragmentHostFunding>,
                T,
            )>(),
            size_of::<PartitionCaptureFrame<T>>() * 2,
            size_of::<Result<PartitionCaptureFrame<T>, Error>>(),
            size_of::<std::vec::IntoIter<PreparedPartitionFragmentHostFunding>>(),
            size_of::<std::vec::IntoIter<Option<RowSource>>>(),
            size_of::<HostRow>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Prospective execution metadata of these exact consumed rows and labels.
    /// Transport protocol callbacks retain their separate producer-owned census.
    pub(in crate::composition::mlx) fn execution_metadata_bytes<
        T: NativePartitionCaptureTransport,
        C: PartitionCaptureRunSource,
    >(
        &self,
        scalars: &[Option<WorkspaceFloatingType>],
        has_interventions: bool,
    ) -> Result<usize, Error>
    where
        <T::Completion as eredu_core::Completion>::Error: Send + Sync + 'static,
    {
        use eredu_nn::workspace::WorkspaceContext as W;
        if scalars.len() != self.producers.len() || self.rows.len() != self.producers.len() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let (artifact, execution, setup) = self.loaded.source_labels();
        let mut bytes = Self::bind_control_bytes::<T>()
            .ok_or_else(overflow)?
            .checked_add(W::metadata_string_bytes(artifact.len()).ok_or_else(overflow)?)
            .ok_or_else(overflow)?
            .checked_add(W::metadata_string_bytes(execution.len()).ok_or_else(overflow)?)
            .ok_or_else(overflow)?
            .checked_add(
                W::metadata_vec_bytes::<Option<HostRow>>(self.rows.len()).ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?
            .checked_add(
                W::metadata_vec_bytes::<Option<WorkspaceFloatingType>>(scalars.len())
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?
            .checked_add(
                PartitionCaptureFrame::<T>::program_control_bytes::<C>().ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        if self.source.admission().is_empty() && !has_interventions {
            return Ok(bytes);
        }
        for part in [
            PreparedPartitionCaptureRunIdentity::preparation_metadata_bytes(
                &self.source,
                artifact,
                execution,
                setup,
                self.overlay.as_deref(),
            ),
            W::metadata_vec_bytes::<Option<PartitionCaptureProducerSource>>(self.producers.len()),
            W::metadata_vec_bytes::<PreparedPartitionCaptureRow<'_>>(self.producers.len()),
            PreparedPartitionCaptureProgram::<T>::selected_for_run_metadata_bytes(
                &self.source,
                artifact,
                execution,
                setup,
                self.overlay.as_deref(),
                self.producers.len(),
            ),
        ] {
            bytes = bytes
                .checked_add(part.ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        let contiguous = self
            .rows
            .iter()
            .filter(|row| matches!(row, Some(RowSource::Contiguous(_))))
            .count();
        let routed = self
            .rows
            .iter()
            .filter(|row| matches!(row, Some(RowSource::Routed(_))))
            .count();
        bytes = bytes
            .checked_add(
                PreparedPartitionCaptureProgram::<T>::projected_rows_metadata_bytes(
                    self.rows.len(),
                    contiguous,
                    routed,
                )
                .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        for (index, row) in self.rows.iter().enumerate() {
            if let Some(row) = row {
                bytes = bytes
                    .checked_add(row.execution_metadata_bytes(
                        &self.source,
                        self.rank,
                        scalars[index],
                        &self.metadata,
                    )?)
                    .ok_or_else(overflow)?;
            }
        }
        Ok(bytes)
    }
    /// Attach the exact one-shot funded Host rows and genuine native role loan.
    /// Source equality is rechecked by each consumed Host at receipt issuance.
    pub(in crate::composition::mlx) fn bind<T: NativePartitionCaptureTransport>(
        self,
        scalars: &[Option<WorkspaceFloatingType>],
        hosts: Vec<PreparedPartitionFragmentHostFunding>,
        transport: T,
    ) -> Result<PartitionCaptureFrame<T>, Error> {
        let metadata = transport.prepare_capture_metadata()?;
        let result = (|| {
            metadata.reserve_metadata(Self::bind_control_bytes::<T>().ok_or_else(overflow)?)?;
            let (artifact, execution, setup) = self.loaded.source_labels();
            if transport.capture_rank() != self.rank
                || transport.participant_count() != setup.participant_count()
                || scalars.len() != self.producers.len()
                || self.rows.len() != self.producers.len()
                || hosts.len() != self.rows.iter().flatten().count()
            {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            let artifact = metadata.metadata_string(format_args!("{artifact}"))?;
            let execution = metadata.metadata_string(format_args!("{execution}"))?;
            let mut actual = hosts.into_iter();
            let mut rows = metadata.metadata_vec(self.rows.len())?;
            for row in self.rows {
                rows.push(match row {
                    Some(source) => {
                        let host = actual
                            .next()
                            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
                        Some(HostRow {
                            source,
                            geometry: FrameGeometry::Invocation(self.invocation),
                            prediction: self.invocation.prediction,
                            fragments: host.original_receipt_limits().max_fragments,
                            host,
                        })
                    }
                    None => None,
                });
            }
            let mut copied = metadata.metadata_vec(scalars.len())?;
            copied.extend_from_slice(scalars);
            Ok((artifact, execution, setup, rows, copied))
        })()
        .map_err(|cause| transport.retain_capture_preparation_error(cause))?;
        let (artifact, execution, setup, hosts, scalars) = result;
        Ok(PartitionCaptureFrame {
            artifact,
            execution,
            setup,
            overlay: self.overlay,
            source: self.source.storage_identity().clone(),
            producers: self.producers,
            hosts,
            evidence: Vec::new(),
            scalars,
            prediction: self.invocation.prediction,
            geometry: FrameGeometry::Invocation(self.invocation),
            consumed: false,
            source_metadata: Some(self.metadata),
            interventions: None,
            model_evidence: None,
            transport,
        })
    }
}
