//! Closed capture host custody, independent or inside one original parent account.
use super::*;
use crate::working_memory::CaptureTensorHostPlan;

/// Entirely private custody for one actual host destination. It contains no
/// observation, callback, native handle, source table or retained plan payload.
#[derive(Debug)]
pub(in crate::working_memory) enum CaptureTensorCustody {
    Single(CaptureHostCustody),
    Model(crate::working_memory::OriginalSpeculativeBudgetCustody),
    Scheduled(std::sync::Arc<CaptureHostCustody>),
    Speculative(crate::working_memory::OriginalSpeculativeNumericalBudgetCustody),
}

#[derive(Debug)]
pub(in crate::working_memory) struct CaptureHostCustody {
    host: WorkingMemoryDecoderHostScope,
    // Retains only the exact accounting identity. Independent construction
    // creates a private identity; parent construction borrows the original one.
    // This custody never exposes it as a new request or execution grant.
    execution: InferenceExecutionIdentity,
    // Independent host-only accounts deliberately close their hidden run at
    // creation. Parent construction/fill requires the original run and metadata
    // to remain live; immutable finished aliases need only retain this custody.
    requires_open_parent: bool,
}
impl CaptureTensorCustody {
    fn text_host(&self) -> Result<&CaptureHostCustody, WorkingMemoryError> {
        match self {
            Self::Single(host) => Ok(host),
            Self::Scheduled(host) => Ok(host),
            Self::Speculative(_) | Self::Model(_) => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
}
impl CaptureTensorCustody {
    // Only scheduled claims can clone custody, never a public allocator/callback.
    pub(in crate::working_memory) fn share_scheduled(&self) -> Self {
        match self {
            Self::Scheduled(host) => Self::Scheduled(host.clone()),
            Self::Speculative(host) => Self::Speculative(host.clone()),
            Self::Model(host) => Self::Model(host.clone()),
            Self::Single(_) => unreachable!("only scheduled claim construction shares custody"),
        }
    }
    pub(in crate::working_memory) fn same_schedule(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scheduled(a), Self::Scheduled(b)) => std::sync::Arc::ptr_eq(a, b),
            (Self::Speculative(a), Self::Speculative(b)) => a.same_account(b),
            (Self::Model(a), Self::Model(b)) => a.same_account(b),
            _ => false,
        }
    }
}
impl CaptureTensorCustody {
    pub(in crate::working_memory) fn validate_model(
        &self,
        expected: &crate::working_memory::OriginalSpeculativeBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Model(actual) if actual.same_account(expected) => self.validate(),
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    pub(in crate::working_memory) fn validate(&self) -> Result<(), WorkingMemoryError> {
        if let Self::Model(custody) = self {
            let pool = custody.pool();
            let usage = pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            return custody.validate_copy_source(pool, &usage);
        }
        if let Self::Speculative(custody) = self {
            let pool = custody.pool();
            let usage = pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            return custody.validate_copy_source(pool, &usage);
        }
        let pool = self.text_host()?.host.pool();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.text_host()?
            .host
            .validate(pool, &usage, &self.text_host()?.execution)?;
        if self.text_host()?.requires_open_parent {
            let id = self
                .text_host()?
                .host
                .scope
                .as_ref()
                .expect("validated host scope")
                .id;
            let state = usage
                .funding
                .get(&id)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if !state.run_open || !state.metadata_live {
                return Err(WorkingMemoryError::ExecutionFenced);
            }
        }
        Ok(())
    }
}
impl MemoryLedger {
    // Called only by the concrete F32 geometry worker. No generic scalar/closure
    // funding API is widened and no existing native scope is auto-certified.
    pub(in crate::working_memory) fn open_capture_tensor_account(
        &self,
        plan: &CaptureTensorHostPlan<'_>,
        limits: &crate::working_memory::CaptureTensorLimits,
    ) -> Result<CaptureTensorCustody, WorkingMemoryError> {
        let bytes = plan.initialization_peak_bytes();
        let projection = crate::working_memory::transaction_buffers::RequirementProjection {
            parts: &[],
            headroom: &limits.additional_headroom,
            host_bytes: bytes,
        };
        let controls = capture_controls(self, &projection)?;
        let accepted = PreparedCopyAccount::accept(
            self,
            &self.construction_identity(),
            projection,
            &limits.memory_limits,
            controls,
            CopyHostHolds::HostOnly(bytes),
            |_| Ok(()),
        )?;
        let execution = InferenceExecutionIdentity::default();
        let host = accepted.host(&execution, bytes)?;
        Ok(CaptureTensorCustody::Single(CaptureHostCustody {
            host,
            execution,
            requires_open_parent: false,
        }))
    }
}

pub(in crate::working_memory) fn capture_controls(
    pool: &MemoryLedger,
    projection: &crate::working_memory::transaction_buffers::RequirementProjection<'_>,
) -> Result<u64, WorkingMemoryError> {
    copy_domain_controls(pool, projection)?
        .checked_add(
            u64::try_from(copy_account_control_bytes(false, 1, false)?)
                .map_err(|_| WorkingMemoryError::Overflow)?,
        )
        .and_then(|n| {
            n.checked_add(crate::working_memory::qualified_storage::shared_bytes::<()>().ok()?)
        })
        .ok_or(WorkingMemoryError::Overflow)
}

impl WorkingMemoryFundingRun {
    // Only the closed geometry worker can request this hold. It never borrows
    // permission from a native scope or changes the request's logical stages.
    pub(in crate::working_memory) fn hold_capture_tensor(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: &CaptureTensorHostPlan<'_>,
    ) -> Result<CaptureTensorCustody, WorkingMemoryError> {
        self.hold_capture_destination(reservation, plan.initialization_peak_bytes())
    }

    // Only another closed plan can reach this path; no raw-byte public grant.
    pub(in crate::working_memory) fn hold_capture_step(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: &crate::working_memory::CaptureStepHostPlan<'_>,
    ) -> Result<CaptureTensorCustody, WorkingMemoryError> {
        self.hold_capture_destination(reservation, plan.initialization_peak_bytes())
    }

    // Common parent arithmetic for the two concrete host payload programs.
    // Private to this module: neither callers nor native scopes supply bytes.
    fn hold_capture_destination(
        &self,
        reservation: &WorkingMemoryReservation,
        bytes: u64,
    ) -> Result<CaptureTensorCustody, WorkingMemoryError> {
        if reservation.0.funding != Some(self.id) || !self.pool.same_ledger(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let execution = reservation.0.execution.clone();
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::CopyRun(self).validate(&usage, &execution)?;
        let state = usage
            .funding
            .get_mut(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        state.validate_span_spend(None)?;
        let available = state.spendable_remaining()?;
        if bytes > available {
            return Err(WorkingMemoryError::DomainAllowanceExceeded {
                domain: self.pool.topology().host_domain(),
                required_bytes: bytes,
                available_bytes: available,
            });
        }
        let held = state
            .host_held
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let scopes = state
            .scopes
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        state.host_held = held;
        state.scopes = scopes;
        drop(usage);
        Ok(CaptureTensorCustody::Single(CaptureHostCustody {
            host: WorkingMemoryDecoderHostScope {
                scope: Some(WorkingMemoryFundingScope {
                    purpose: ScopePurpose::Host,
                    pool: self.pool.clone(),
                    id: self.id,
                    active: true,
                    borrowed_storage: None,
                    capture_source: None,
                    native_publication_identity: None,
                    allocation_funding: None,
                }),
                held: bytes,
            },
            execution,
            requires_open_parent: true,
        }))
    }
}

/// Restores source-pin custody only while construction has not entered native
/// work. No key-owned bundle is dropped under Usage. Successful construction
/// disarms this guard; all later errors leave the larger bundle on native scope.
pub(in crate::working_memory) struct CaptureSourceRollback<'a> {
    scope: Option<&'a mut WorkingMemoryFundingScope>,
    previous: CaptureSourcePrevious,
}
enum CaptureSourcePrevious {
    Permanent(Option<RegisteredStoragePin>),
    Segment(super::capture_source::CaptureSourceEntries),
}
impl<'a> CaptureSourceRollback<'a> {
    pub(super) fn segment(
        scope: &'a mut WorkingMemoryFundingScope,
        previous: super::capture_source::CaptureSourceEntries,
    ) -> Self {
        Self {
            scope: Some(scope),
            previous: CaptureSourcePrevious::Segment(previous),
        }
    }
    pub(in crate::working_memory) fn commit(mut self) -> &'a mut WorkingMemoryFundingScope {
        self.scope.take().expect("construction scope")
    }
}
impl Drop for CaptureSourceRollback<'_> {
    fn drop(&mut self) {
        if let Some(scope) = self.scope.as_mut() {
            match &mut self.previous {
                CaptureSourcePrevious::Permanent(previous) => {
                    let added = std::mem::replace(&mut scope.borrowed_storage, previous.take());
                    drop(added);
                }
                CaptureSourcePrevious::Segment(previous) => {
                    // This exclusive scope loan prevents replacement of the slot.
                    let slot = scope
                        .capture_source
                        .as_mut()
                        .expect("bound capture channel");
                    let added = std::mem::replace(&mut slot.sources, std::mem::take(previous));
                    drop(added);
                }
            }
        }
    }
}
impl WorkingMemoryFundingRun {
    pub(in crate::working_memory) fn hold_capture_tensor_with_source<
        's,
        K: Clone + Ord + Send + Sync + 'static,
    >(
        &self,
        reservation: &WorkingMemoryReservation,
        native: &'s mut WorkingMemoryFundingScope,
        plan: &CaptureTensorHostPlan<'_>,
        source: &WorkingMemoryStorage<K>,
    ) -> Result<(CaptureTensorCustody, CaptureSourceRollback<'s>), WorkingMemoryError> {
        if reservation.0.funding != Some(self.id)
            || !self.pool.same_ledger(&reservation.0.pool)
            || !FundingSource::CopyRun(self).same_account(FundingSource::HostScope(native))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Every key clone and aggregate allocation precedes the account lock.
        let pins = RegisteredStoragePin::aggregate(
            [
                native.borrowed_storage.clone(),
                Some(RegisteredStoragePin::new(source.clone())),
            ]
            .into_iter()
            .flatten(),
        );
        let execution = reservation.0.execution.clone();
        let bytes = plan.initialization_peak_bytes();
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::CopyRun(self).validate(&usage, &execution)?;
        FundingSource::NativeScope(native).validate(&usage, &execution)?;
        source.validate_copy_source(&self.pool, &usage)?;
        let state = usage
            .funding
            .get_mut(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        state.validate_span_spend(None)?;
        let available = state.spendable_remaining()?;
        if bytes > available {
            return Err(WorkingMemoryError::DomainAllowanceExceeded {
                domain: self.pool.topology().host_domain(),
                required_bytes: bytes,
                available_bytes: available,
            });
        }
        let held = state
            .host_held
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let scopes = state
            .scopes
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        // Only infallible moves and scalar updates remain in this commit.
        let previous = native.borrowed_storage.replace(pins);
        state.host_held = held;
        state.scopes = scopes;
        drop(usage);
        let rollback = CaptureSourceRollback {
            scope: Some(native),
            previous: CaptureSourcePrevious::Permanent(previous),
        };
        Ok((
            CaptureTensorCustody::Single(CaptureHostCustody {
                host: WorkingMemoryDecoderHostScope {
                    scope: Some(WorkingMemoryFundingScope {
                        purpose: ScopePurpose::Host,
                        pool: self.pool.clone(),
                        id: self.id,
                        active: true,
                        borrowed_storage: None,
                        capture_source: None,
                        native_publication_identity: None,
                        allocation_funding: None,
                    }),
                    held: bytes,
                },
                execution,
                requires_open_parent: true,
            }),
            rollback,
        ))
    }
}
impl CaptureTensorCustody {
    pub(in crate::working_memory) fn validate_transfer<K: Ord + Send + 'static>(
        &self,
        native: &WorkingMemoryFundingScope,
        source: &WorkingMemoryStorage<K>,
    ) -> Result<(), WorkingMemoryError> {
        let host = self
            .text_host()?
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::ExecutionFenced)?;
        if !FundingSource::HostScope(host).same_account(FundingSource::HostScope(native)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let pool = self.text_host()?.host.pool();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.text_host()?
            .host
            .validate(pool, &usage, &self.text_host()?.execution)?;
        FundingSource::NativeScope(native).validate(&usage, &self.text_host()?.execution)?;
        source.validate_copy_source(pool, &usage)?;
        if let Some(segment) = &native.capture_source {
            segment.validate(pool, &usage)?;
        }
        let state = usage
            .funding
            .get(&host.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        state.validate_span_spend(Some(native))?;
        if !state.run_open || !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        Ok(())
    }
}

impl WorkingMemoryFundingRun {
    // Exact cumulative geometry is the only new path to this private arithmetic.
    pub(in crate::working_memory) fn hold_capture_run(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: &crate::working_memory::CaptureRunHostPlan<'_>,
    ) -> Result<CaptureTensorCustody, WorkingMemoryError> {
        let CaptureTensorCustody::Single(host) =
            self.hold_capture_destination(reservation, plan.initialization_peak_bytes())?
        else {
            unreachable!("fresh private host hold")
        };
        Ok(CaptureTensorCustody::Scheduled(std::sync::Arc::new(host)))
    }
}
impl CaptureTensorCustody {
    // Health check before settling a lazy source. Its actual complete source
    // origins do not exist yet and are bound by the later transfer constructor.
    pub(in crate::working_memory) fn validate_scheduled_native(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        // Preserve structural mismatch precedence before touching either pool.
        self.scheduled_native_account(native)?;
        let pool = self.text_host()?.host.pool();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_scheduled_native_locked(native, &usage)
    }

    pub(super) fn validate_scheduled_native_locked(
        &self,
        native: &WorkingMemoryFundingScope,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        let host = self.scheduled_native_account(native)?;
        let pool = self.text_host()?.host.pool();
        self.text_host()?
            .host
            .validate(pool, usage, &self.text_host()?.execution)?;
        FundingSource::NativeScope(native).validate(usage, &self.text_host()?.execution)?;
        let state = usage
            .funding
            .get(&host.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        state.validate_span_spend(Some(native))?;
        if !state.run_open || !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        Ok(())
    }

    fn scheduled_native_account(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<&WorkingMemoryFundingScope, WorkingMemoryError> {
        if !matches!(self, Self::Scheduled(_)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let host = self
            .text_host()?
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::ExecutionFenced)?;
        if !FundingSource::HostScope(host).same_account(FundingSource::HostScope(native)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(host)
    }

    // Adds source pins to exactly the exclusively borrowed native scope. The
    // full scheduled H was protected earlier; this commit creates no new hold,
    // scope or reservation. All key-owned construction/drop is outside Usage.
    pub(in crate::working_memory) fn bind_scheduled_source<
        's,
        K: Clone + Ord + Send + Sync + 'static,
    >(
        &self,
        native: &'s mut WorkingMemoryFundingScope,
        source: &WorkingMemoryStorage<K>,
    ) -> Result<CaptureSourceRollback<'s>, WorkingMemoryError> {
        if !matches!(self, Self::Scheduled(_)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let host = self
            .text_host()?
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::ExecutionFenced)?;
        if !FundingSource::HostScope(host).same_account(FundingSource::HostScope(native)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let pins = RegisteredStoragePin::aggregate(
            [
                native.borrowed_storage.clone(),
                Some(RegisteredStoragePin::new(source.clone())),
            ]
            .into_iter()
            .flatten(),
        );
        let pool = self.text_host()?.host.pool();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.text_host()?
            .host
            .validate(pool, &usage, &self.text_host()?.execution)?;
        FundingSource::NativeScope(native).validate(&usage, &self.text_host()?.execution)?;
        source.validate_copy_source(pool, &usage)?;
        let state = usage
            .funding
            .get(&host.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        state.validate_span_spend(Some(native))?;
        if !state.run_open || !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        let previous = native.borrowed_storage.replace(pins);
        drop(usage);
        Ok(CaptureSourceRollback {
            scope: Some(native),
            previous: CaptureSourcePrevious::Permanent(previous),
        })
    }
}

impl WorkingMemoryFundingRun {
    // One independently tagged host hold per fragment prevents same-shaped
    // completed terms from being exchanged between ranks or fragment ordinals.
    // All holds consume the original account; no native scope is created.
    pub(in crate::working_memory) fn hold_partition_fragment(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: &crate::working_memory::capture_run::FragmentHostPlan<'_>,
    ) -> Result<CaptureTensorCustody, WorkingMemoryError> {
        let CaptureTensorCustody::Single(host) =
            self.hold_capture_destination(reservation, plan.initialization_peak_bytes()?)?
        else {
            unreachable!("fresh host hold")
        };
        Ok(CaptureTensorCustody::Scheduled(std::sync::Arc::new(host)))
    }
    pub(in crate::working_memory) fn hold_partition_fragment_table(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: &crate::working_memory::PartitionFragmentHostPlan<'_>,
    ) -> Result<CaptureTensorCustody, WorkingMemoryError> {
        let CaptureTensorCustody::Single(host) =
            self.hold_capture_destination(reservation, plan.table_peak_bytes())?
        else {
            unreachable!("fresh host hold")
        };
        Ok(CaptureTensorCustody::Scheduled(std::sync::Arc::new(host)))
    }
}

impl CaptureTensorCustody {
    pub(in crate::working_memory) fn shared_host_control_bytes() -> Result<u64, WorkingMemoryError>
    {
        crate::working_memory::qualified_shared_bytes::<CaptureHostCustody>()
    }
}
