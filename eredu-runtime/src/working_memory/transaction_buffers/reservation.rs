//! Ordinary reservations construct their owned descriptors after admission.
use super::*;

impl MemoryLedger {
    pub(in crate::working_memory) fn reserve_standalone(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: Option<MemoryLimits>,
        residual: Option<(&DomainMemoryRequirements, residual::RegisteredStoragePin)>,
        handoffs: &[WorkingMemoryCapacityHandoff],
        source: Option<&dyn saved_source::SavedSourceValidation>,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        let state = &admission.state;
        let workspace = state
            .execution_workspace
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let geometry = workspace.geometry;
        let requested = geometry
            .cached_positions
            .checked_add(geometry.input_positions)
            .and_then(|positions| positions.checked_add(geometry.max_output_tokens))
            .ok_or(WorkingMemoryError::Overflow)?;
        if geometry.batch_size != state.assumptions.batch_size
            || requested != admission.requested_positions
            || requested != state.assumptions.requested_positions
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let state = state
            .physical_domains
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let workspace = workspace
            .physical_domains
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let full = [
            &state.decoder_state,
            &state.media_embeddings,
            &state.media_workspace,
            &workspace.activations,
            &workspace.attention,
            &workspace.vocabulary,
            &workspace.state_update,
            &workspace.materialization,
            &workspace.retained,
        ];
        for part in full {
            part.validate(self.topology())?;
        }
        for (domain, _) in self.topology().domains() {
            full.iter()
                .try_fold(DomainMemoryCharge::default(), |sum, part| {
                    sum.checked_add(part.get(domain)?)
                })?;
        }
        let residual_parts;
        let parts = match &residual {
            Some((requirements, _)) => {
                residual_parts = [*requirements];
                &residual_parts[..]
            }
            None => &full[..],
        };
        let mut projection = RequirementProjection {
            parts,
            headroom: &admission.additional_headroom,
            host_bytes: 0,
        };
        let limits_backing = u64::try_from(
            self.topology()
                .len()
                .checked_mul(std::mem::size_of::<MemoryLimit>())
                .ok_or(WorkingMemoryError::Overflow)?,
        )
        .map_err(|_| WorkingMemoryError::Overflow)?;
        let mut controls = u64::try_from(reservation_metadata::constructor_bytes_from_backing(
            self.topology(),
            projection.backing_bytes(self.topology())?,
            limits_backing,
        )?)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .checked_add(reservation_metadata::admission_backing_bytes(admission)?)
        .ok_or(WorkingMemoryError::Overflow)?;
        if let Some(source) = source {
            controls = controls
                .checked_add(
                    u64::try_from(source.pin_control_bytes()?)
                        .map_err(|_| WorkingMemoryError::Overflow)?,
                )
                .ok_or(WorkingMemoryError::Overflow)?;
            if residual.is_some() {
                controls = controls
                    .checked_add(
                        u64::try_from(residual::RegisteredStoragePin::pair_control_bytes(true)?)
                            .map_err(|_| WorkingMemoryError::Overflow)?,
                    )
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        projection.host_bytes = controls;
        let pending = self.accept_projected(
            execution,
            RequirementProjection {
                parts,
                headroom: projection.headroom,
                host_bytes: controls,
            },
            &admission.memory_limits,
            capacity.as_ref(),
            handoffs,
            controls,
            |usage| {
                if let Some(source) = source {
                    source.validate(self, usage)?;
                }
                Ok(())
            },
        )?;
        let ticket = pending.publish();
        ticket.status()?;
        // Ticket custody precedes every owned descriptor, including unwind.
        let retained_admission = admission.clone();
        let requirements = projection.materialize(self.topology())?;
        let mut limits = MemoryLimits::unlimited(self.topology());
        limits.resolve_named_in_place(
            self.topology(),
            &admission.memory_limits,
            capacity.as_ref(),
        )?;
        let borrowed_storage = match (residual.map(|(_, pin)| pin), source) {
            (Some(pin), Some(source)) => {
                Some(residual::RegisteredStoragePin::pair(pin, source.pin()))
            }
            (Some(pin), None) => Some(pin),
            (None, Some(source)) => Some(source.pin()),
            (None, None) => None,
        };
        {
            let mut usage = self
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            usage
                .funding
                .report_constructor_controls(ticket.id(), controls)?;
        }
        let value = Reservation {
            account_id: ticket.id(),
            pool: self.clone(),
            execution: execution.clone(),
            admission: retained_admission,
            geometry,
            requirements,
            capacity: Some(limits),
            funding: None,
            borrowed_storage,
            span_workspace: None,
            start: ControlMutex::new(text_preparation::RequestStart::Fresh),
            planning_metadata: None,
        };
        Ok(ticket.into_reservation(value))
    }
}
