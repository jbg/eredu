//! Atomic source-inclusive holds for one actual two-level decoder table group.
use super::*;

fn total_holds(holds: &[u64]) -> Result<u64, WorkingMemoryError> {
    if holds.is_empty() {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    holds.iter().try_fold(0u64, |sum, n| {
        sum.checked_add(*n).ok_or(WorkingMemoryError::Overflow)
    })
}
fn host_scope(pool: &WorkingMemoryPool, id: u64, held: u64) -> WorkingMemoryDecoderHostScope {
    WorkingMemoryDecoderHostScope {
        scope: Some(WorkingMemoryFundingScope {
            purpose: ScopePurpose::Host,
            pool: pool.clone(),
            id,
            active: true,
            borrowed_storage: None,
            capture_source: None,
            native_publication_identity: None,
        }),
        held,
    }
}

impl WorkingMemoryPool {
    // Only actual typed group plans can call this private numeric bridge. Every
    // source origin is checked with all holds before the one account commit.
    pub(in crate::working_memory) fn open_grouped_text_components_account<
        K: Ord + Send + 'static,
    >(
        &self,
        sampler: FundingSource<'_>,
        sampler_execution: &InferenceExecutionIdentity,
        decoders: &[DecoderCopySource<'_, K>],
        holds: &[u64],
        operands: &WorkingMemoryStorage<K>,
        complete_source: &WorkingMemoryStorage<K>,
        pin: RegisteredStoragePin,
        destination: &InferenceExecutionIdentity,
        bytes: u64,
        sampler_hold: u64,
        capacity: u64,
    ) -> Result<
        (
            WorkingMemoryFundingRun,
            WorkingMemorySamplerScope,
            Vec<WorkingMemoryDecoderHostScope>,
            WorkingMemoryFundingScope,
        ),
        WorkingMemoryError,
    > {
        if !self.same_domain(sampler.pool()) || decoders.len() != holds.len() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let decoder = total_holds(holds)?;
        // Allocation/destruction of descriptors stays outside the usage lock.
        // Exact capacity means filling the scope vector cannot grow after commit.
        let mut scopes = super::super::qualified_storage::vector(
            holds.len(),
            complete_source.source_preparation().is_some(),
        )?;
        let mut node = Some(AccountNode::empty());
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        sampler.validate(&usage, sampler_execution)?;
        for source in decoders {
            source.validate(self, &usage)?;
        }
        operands.validate_copy_source(self, &usage)?;
        complete_source.validate_copy_source(self, &usage)?;
        let id = commit_copy_account(
            self,
            &mut usage,
            &mut node,
            destination,
            bytes,
            capacity,
            CopyHostHolds::Grouped {
                sampler: sampler_hold,
                decoder,
                tables: holds.len(),
            },
        )?;
        drop(usage);
        for &held in holds {
            scopes.push(host_scope(self, id, held));
        }
        Ok((
            WorkingMemoryFundingRun {
                pool: self.clone(),
                id,
                open: true,
                handoff_taken: false,
                borrowed_storage: None,
            },
            WorkingMemorySamplerScope {
                scope: Some(WorkingMemoryFundingScope {
                    purpose: ScopePurpose::Host,
                    pool: self.clone(),
                    id,
                    active: true,
                    borrowed_storage: None,
                    capture_source: None,
                    native_publication_identity: None,
                }),
                held: sampler_hold,
            },
            scopes,
            WorkingMemoryFundingScope {
                purpose: ScopePurpose::Native,
                pool: self.clone(),
                id,
                active: true,
                borrowed_storage: Some(pin),
                capture_source: None,
                native_publication_identity: None,
            },
        ))
    }
}

impl WorkingMemoryFundingRun {
    // Existing fresh reservation only: no repeated stage and no new account.
    // The separate native scope retains all source and original residual pins.
    pub(in crate::working_memory) fn open_grouped_dense_prompt_scopes<K: Ord + Send + 'static>(
        &self,
        reservation: &WorkingMemoryReservation,
        sources: &[DecoderCopySource<'_, K>],
        holds: &[u64],
        complete_source: &WorkingMemoryStorage<K>,
        pins: RegisteredStoragePin,
    ) -> Result<
        (
            InferenceExecutionIdentity,
            Vec<WorkingMemoryDecoderHostScope>,
            WorkingMemoryFundingScope,
        ),
        WorkingMemoryError,
    > {
        if reservation.0.funding != Some(self.id)
            || !self.pool.same_domain(&reservation.0.pool)
            || sources.len() != holds.len()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let bytes = total_holds(holds)?;
        let additional_scopes = holds
            .len()
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let mut scopes = super::super::qualified_storage::vector(
            holds.len(),
            complete_source.source_preparation().is_some(),
        )?;
        let pins = if complete_source.source_preparation().is_some() {
            // The source constructor H prices this fixed pair. Its children
            // already exist; no variable Vec or additional single-pin wrapper.
            match &self.borrowed_storage {
                Some(borrowed) => RegisteredStoragePin::pair(pins, borrowed.clone()),
                None => pins,
            }
        } else {
            RegisteredStoragePin::aggregate(
                [Some(pins), self.borrowed_storage.clone()]
                    .into_iter()
                    .flatten(),
            )
        };
        let execution = reservation.0.execution.clone();
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::CopyRun(self).validate(&usage, &execution)?;
        for source in sources {
            source.validate(&self.pool, &usage)?;
        }
        complete_source.validate_copy_source(&self.pool, &usage)?;
        let state = usage
            .funding
            .get_mut(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.metadata_live {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        state.validate_span_spend(None)?;
        let available = state.spendable_remaining()?;
        if bytes > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: bytes,
                available_bytes: available,
            });
        }
        let held = state
            .host_held
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let (count, native_scopes) = state.scope_counts_after(additional_scopes, 1)?;
        state.host_held = held;
        state.scopes = count;
        state.native_scopes = native_scopes;
        drop(usage);
        for &held in holds {
            scopes.push(host_scope(&self.pool, self.id, held));
        }
        Ok((
            execution,
            scopes,
            WorkingMemoryFundingScope {
                purpose: ScopePurpose::Native,
                pool: self.pool.clone(),
                id: self.id,
                active: true,
                borrowed_storage: Some(pins),
                capture_source: None,
                native_publication_identity: None,
            },
        ))
    }
}
