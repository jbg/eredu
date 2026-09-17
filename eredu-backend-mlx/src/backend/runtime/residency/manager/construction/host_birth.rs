//! Source-accounted final host buffers and completed initial device aliases.
use super::*;
use eredu_checkpoint::recipe::EncodedRecipeRead;
use eredu_core::residency::{
    ResidencyAdmissionStorage, ResidencyProtection, ResidencyReservationRow,
};
use safemlx::{PreparedHostTransferPlan, PreparedInputArena, PreparedSubmissionGraphQuota};

fn overflow() -> WorkingMemoryError {
    WorkingMemoryError::Overflow
}
fn unknown() -> WorkingMemoryError {
    WorkingMemoryError::UnknownBound
}
fn usize_bytes(value: u64) -> Result<usize, WorkingMemoryError> {
    usize::try_from(value).map_err(|_| overflow())
}

impl OriginalManagerPlan {
    pub(super) fn host_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let runtime = crate::backend::managed_memory::input_allocator::borrow_admitted(&self.pool)?;
        let mut bytes = if let Some(foreground) = &self.foreground {
            self.host_units.iter().try_fold(0usize, |bytes, &unit| {
                let range = self.read_range(unit);
                if range.is_empty() {
                    return Ok(bytes);
                }
                let read = foreground
                    .read_source()
                    .read_slice(range)
                    .and_then(|read| read.read_layout())
                    .ok_or_else(unknown)?;
                bytes
                    .checked_add(read.required_bytes())
                    .ok_or_else(overflow)
            })?
        } else {
            let detached = EncodedRecipeRead::prepare_detached(
                self.reads.iter().map(|row| row.read.encoded()),
            )
            .ok_or_else(unknown)?;
            detached
                .required_bytes::<ManagerCustody>()
                .ok_or_else(overflow)?
                .checked_add(
                    detached
                        .read_layout::<ManagerCustody>()
                        .ok_or_else(unknown)?
                        .required_bytes(),
                )
                .ok_or_else(overflow)?
        };
        let mut add = |n: usize| {
            bytes = bytes.checked_add(n).ok_or_else(overflow)?;
            Ok::<_, WorkingMemoryError>(())
        };
        add(OriginalHostSources::storage_bytes(self)?)?;
        for layout in [
            Layout::array::<HostTransferBuffer>(self.host_reads.len()),
            Layout::array::<RetainedHostBuffer>(self.host_reads.len()),
            Layout::array::<&mut [u8]>(self.host_reads.len()),
            Layout::array::<(OffloadUnitId, ResidentHostOwner)>(self.host_units.len()),
            Layout::array::<OffloadUnitId>(self.units.len()),
            Layout::array::<ResidencyReservationRow>(self.units.len()),
        ] {
            add(layout.map_err(|_| overflow())?.size())?;
        }
        let maximum = self
            .units
            .iter()
            .map(|unit| unit.id().as_str().len())
            .max()
            .unwrap_or(0);
        add(ResidencyAdmissionStorage::requested_bytes(
            self.units.len(),
            self.units.len(),
            maximum,
        )
        .ok_or_else(overflow)?)?;
        for &index in &self.host_reads {
            let row = &self.reads[index];
            let plan = PreparedHostTransferPlan::new(
                &runtime,
                row.read.shape(),
                row.read.dtype(),
                self.array_handles(index).ok_or_else(unknown)?,
            )
            .map_err(|_| unknown())?;
            add(
                PreparedInputArena::layout::<ManagerCustody>(plan.metadata_bytes())
                    .map_err(|_| unknown())?
                    .total_bytes()
                    .ok_or_else(overflow)?,
            )?;
            add(plan.backing_bytes())?;
            add(plan.control_bytes().ok_or_else(overflow)?)?;
            add(usize_bytes(RetainedHostBuffer::storage_bytes()?)?)?;
        }
        // Initial-publication IDs cover the entire controller; only actual
        // initial source units own an immutable host row/name population.
        for unit in &self.units {
            add(unit.id().as_str().len())?;
        }
        for &unit_index in &self.host_units {
            let unit = &self.units[unit_index];
            add(unit.id().as_str().len())?;
            add(usize_bytes(ResidentHostOwner::storage_bytes()?)?)?;
            add(
                rows::Rows::<String, RetainedHostBuffer>::requested_layout(unit.bindings().len())
                    .ok_or_else(overflow)?
                    .size(),
            )?;
            let device = self.initial_units[1].contains(unit.id());
            if device {
                add(usize_bytes(ResidentArraysOwner::source_storage_bytes()?)?)?;
                add(
                    rows::Rows::<String, named_arrays::SourceArray>::requested_layout(
                        unit.bindings().len(),
                    )
                    .ok_or_else(overflow)?
                    .size(),
                )?;
            }
            for binding in unit.bindings() {
                add(binding
                    .name()
                    .len()
                    .checked_mul(1 + usize::from(device))
                    .ok_or_else(overflow)?)?;
            }
        }
        if let Some(target) = &self.target {
            if self.foreground.is_none() {
                add(usize_bytes(
                    super::super::HostCopyWorkspace::constructor_storage_bytes(
                        &target.selected_definitions,
                        |id, binding| self.shape_for(id, binding),
                    )
                    .map_err(|_| unknown())?,
                )?)?;
                add(super::super::HostCopyIdentity::requested_bytes(
                    &target.layout,
                    target.selected_definitions.iter(),
                )
                .ok_or_else(unknown)?)?;
                add(usize_bytes(
                    super::super::OriginalResidencySource::constructor_storage_bytes(
                        self.units.len(),
                        target.layout.len(),
                    )
                    .ok_or_else(unknown)?,
                )?)?;
            }
            if self.foreground.is_some() {
                let one = usize_bytes(ResidencyManager::original_foreground_operation_source_bytes(
                    &self.pool, &target.layout, &self.units, target.selected_ids.len(),
                )?)?;
                add(one)?;
                if target.dense_controller.is_some_and(|facts| facts.options.host_budget_bytes() > 0) {
                    // The same constructor runs again for actual Host depth. Its
                    // metadata bound uses full controller/row counts, independently
                    // of depth; no source buffer or worker is created by this pass.
                    add(one)?;
                }
            }
        }
        for n in [
            size_of::<ConstructionCause>(),
            size_of::<ResidencyAdmissionStorage>(),
            size_of::<
                Result<
                    ResidencyAdmissionStorage,
                    eredu_core::residency::PreparedResidencyAdmissionFailure,
                >,
            >(),
            size_of::<PreparedHostTransferPlan<'_>>(),
            size_of::<PreparedInputArena>(),
            size_of::<PreparedSubmissionGraphQuota<ManagerCustody>>(),
            size_of::<Vec<HostTransferBuffer>>(),
            size_of::<Vec<RetainedHostBuffer>>(),
            size_of::<Vec<&mut [u8]>>(),
            size_of::<source::ReadSourceOwner>(),
            size_of::<eredu_checkpoint::store::DetachedEncodedReadSlice<'_, ManagerCustody>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<[usize; 3]>(),
            size_of::<std::time::Instant>(),
            size_of::<std::time::Duration>(),
        ] {
            add(n)?;
        }
        Ok(bytes)
    }

    pub(super) fn construct_hosts(
        &self,
        control: &ResidencyController,
        custody: ManagerCustody,
    ) -> Result<(OriginalHostSources, u64, std::time::Duration), ConstructionCause> {
        let runtime = crate::backend::managed_memory::input_allocator::borrow_admitted(&self.pool)
            .map_err(ResidencyError::OriginalCache)?;
        let source = if let Some(foreground) = &self.foreground {
            source::ReadSourceOwner::Foreground(foreground.clone())
        } else {
            let detached = EncodedRecipeRead::prepare_detached(
                self.reads.iter().map(|row| row.read.encoded()),
            )
            .ok_or(ResidencyError::OriginalCache(unknown()))?
            .construct(custody.clone())?;
            source::ReadSourceOwner::Host(source::OriginalReadSources::new(
                &self.read_source_plan(),
                detached,
                custody.clone(),
            )?)
        };
        let mut buffers = Vec::with_capacity(self.host_reads.len());
        for &index in &self.host_reads {
            let row = &self.reads[index];
            let plan = PreparedHostTransferPlan::new(
                &runtime,
                row.read.shape(),
                row.read.dtype(),
                self.array_handles(index)
                    .ok_or(ResidencyError::OriginalOperationDomain)?,
            )
            .map_err(ResidencyError::OriginalHostInput)?;
            let preparation =
                PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes(), custody.clone())
                    .map_err(|error| {
                        let cause = error.cause();
                        drop(error);
                        ConstructionCause::Quota(cause)
                    })?;
            let arena = PreparedInputArena::try_allocate(preparation).map_err(|error| {
                let cause = error.cause();
                drop(error);
                ConstructionCause::Quota(cause)
            })?;
            buffers.push(
                plan.construct(&arena)
                    .map_err(ResidencyError::OriginalHostInput)?,
            );
        }
        let read_bytes = self.host_reads.iter().try_fold(0u64, |sum, &index| {
            let bytes = self.reads[index].read.encoded().output().byte_len();
            sum.checked_add(bytes)
                .ok_or(ResidencyError::OriginalCache(overflow()))
        })?;
        let read_started = std::time::Instant::now();
        {
            // These are the final native destinations. A failed read publishes
            // none of the partially initialized buffers and retains its cause.
            let mut outputs = Vec::with_capacity(buffers.len());
            for buffer in &mut buffers {
                outputs.push(
                    buffer
                        .as_bytes_mut()
                        .map_err(ResidencyError::OriginalNative)?,
                );
            }
            if self.foreground.is_none() {
                source
                    .read_slice(0..self.reads.len())
                    .ok_or(ResidencyError::OriginalOperationDomain)?
                    .read_many_into(&mut outputs)?;
            } else {
                for &unit in &self.host_units {
                    let range = self.read_range(unit);
                    if range.is_empty() {
                        continue;
                    }
                    let start = self.host_read_slots[range.start]
                        .ok_or(ResidencyError::OriginalOperationDomain)?;
                    let end = start
                        .checked_add(range.len())
                        .ok_or(ResidencyError::OriginalCache(overflow()))?;
                    source
                        .read_slice(range)
                        .ok_or(ResidencyError::OriginalOperationDomain)?
                        .read_many_into(
                            outputs
                                .get_mut(start..end)
                                .ok_or(ResidencyError::OriginalOperationDomain)?,
                        )?;
                }
            }
        }
        let read_duration = read_started.elapsed();
        let buffers = buffers
            .into_iter()
            .map(|buffer| RetainedHostBuffer::original(buffer.freeze(), custody.clone()))
            .collect();
        Ok((
            OriginalHostSources::new(self, source, buffers, control, custody)?,
            read_bytes,
            read_duration,
        ))
    }

    pub(super) fn publish_initial_storage(
        &self,
        manager: &mut ResidencyManager,
        custody: &ManagerCustody,
    ) -> Result<(), ConstructionCause> {
        let mut state = manager
            .inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        let ids = self
            .units
            .iter()
            .map(|unit| unit.id().clone())
            .collect::<Vec<_>>();
        let mut reservations = Vec::with_capacity(ids.len());
        let source = state.control.ledger().plan_source();
        let mut admission = ResidencyAdmissionStorage::try_new(
            source,
            ids.len(),
            ids.iter().map(|id| id.as_str().len()).max().unwrap_or(0),
        )?;
        for (tier_index, tier) in [MemoryTier::Host, MemoryTier::Device]
            .into_iter()
            .enumerate()
        {
            let selected = &self.initial_units[tier_index];
            super::super::transfer::aliases::prepare_owner_pins(&state, selected, tier)?;
            reservations.clear();
            for (input, id) in ids.iter().enumerate() {
                if !selected.contains(id) {
                    continue;
                }
                let host = manager
                    .inner
                    .sources
                    .prepared_host(id)
                    .ok_or(ResidencyError::StatePoisoned)?;
                let unit = state
                    .control
                    .unit(id)
                    .ok_or(ResidencyError::StatePoisoned)?;
                let bytes =
                    unit.bindings()
                        .iter()
                        .filter(|b| !b.is_alias())
                        .try_fold(0u64, |sum, b| {
                            let metadata = host
                                .buffers
                                .get(b.name())
                                .and_then(RetainedHostBuffer::prepared_metadata)
                                .ok_or(ResidencyError::StatePoisoned)?;
                            sum.checked_add(metadata.allocation().bytes() as u64)
                                .ok_or(ResidencyError::OriginalCache(overflow()))
                        })?;
                if bytes > 0 {
                    reservations.push(ResidencyReservationRow { input, bytes });
                }
            }
            admission = state.control.ledger_mut().reserve_copies_in(
                &ids,
                &reservations,
                tier,
                ResidencyProtection::sorted(&[]).expect("empty sorted protection"),
                admission,
            )?;
            for row in &reservations {
                let id = &ids[row.input];
                let host = manager
                    .inner
                    .sources
                    .prepared_host(id)
                    .ok_or(ResidencyError::StatePoisoned)?;
                let value = if tier == MemoryTier::Device {
                    let mut values = Vec::with_capacity(host.buffers.len());
                    for (name, buffer) in &host.buffers {
                        values.push((
                            name.clone(),
                            named_arrays::SourceArray {
                                value: buffer
                                    .try_prepared_source_array()
                                    .map_err(ResidencyError::OriginalHostInput)?,
                                source: buffer.clone(),
                            },
                        ));
                    }
                    Some(ResidentArraysOwner::from_source(
                        values.into_iter().collect(),
                        custody.clone(),
                    ))
                } else {
                    None
                };
                state
                    .control
                    .ledger_mut()
                    .publish_reserved(id, tier, row.bytes, None)
                    .map_err(ResidencyError::from)?;
                let storage = state
                    .storage
                    .get_mut(id)
                    .ok_or(ResidencyError::StatePoisoned)?;
                if let Some(value) = value {
                    storage.device = Some(value)
                } else {
                    storage.host = Some(host.clone())
                }
            }
            super::super::transfer::aliases::pin_owners(&mut state, selected, tier)?;
        }
        state.control.ledger_mut().mark_initialized();
        Ok(())
    }
}
