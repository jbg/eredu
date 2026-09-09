use super::*;

/// Inactive serial branch slot owned by one controlled scope. Cloning a handle
/// does not copy state. Release the slot explicitly when it is no longer needed.
#[derive(Debug, Clone)]
pub struct SpeculativeBranchHandle {
    owner: Arc<()>,
    pub(super) id: u64,
}
impl SpeculativeBranchHandle {
    /// Stable slot identity; exchange changes which logical run it holds.
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// Host-only identity and prefix for reconciling an Inspector's branch journal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeBranchInfo {
    /// Logical run identity, independent of the exchange slot.
    pub run_id: u64,
    /// Canonical generated prefix, including inherited tokens.
    pub token_ids: Vec<u32>,
    /// Saved lifecycle; terminal branches remain inspectable.
    pub status: SpeculativeRequestStatus,
}

pub(super) struct Branch<E: SpeculativeExecutor, S: SpeculativeSampling, C> {
    state: Rc<Saved<E, S, C>>,
    run_id: u64,
    _reservation: SnapshotReservation,
}

impl<'a, E, S, C, P> Session<'a, E, S, C, P>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    pub(super) fn owned_branch(
        &self,
        handle: &SpeculativeBranchHandle,
    ) -> Result<(), SpeculativeControlError> {
        if !Arc::ptr_eq(&self.owner, &handle.owner) || !self.branches.contains_key(&handle.id) {
            return Err(SpeculativeControlError::IncompatibleBranch);
        }
        Ok(())
    }

    pub(super) fn branch_info_inner(
        &self,
        handle: &SpeculativeBranchHandle,
    ) -> Result<SpeculativeBranchInfo, SpeculativeControlError> {
        self.owned_branch(handle)?;
        let branch = &self.branches[&handle.id];
        Ok(SpeculativeBranchInfo {
            run_id: branch.run_id,
            token_ids: branch.state.state.token_ids().to_vec(),
            status: branch.state.state.status(),
        })
    }

    pub(super) fn fork_inner(
        &mut self,
        handle: &SpeculativeSnapshotHandle,
    ) -> Result<SpeculativeBranchHandle, SpeculativeControlError> {
        self.healthy()?;
        self.owned(handle)?;
        if !self.can_snapshot() {
            return Err(SpeculativeControlError::NotQuiescent);
        }
        let id = self.next_branch;
        let next = id.checked_add(1).ok_or(ExecutionControlError::Overflow)?;
        let overhead = std::mem::size_of::<Branch<E, S, C>>() as u64 + 256;
        let reservation = self.budget.as_ref().expect("snapshot budget").reserve(
            SnapshotResourceKind::Branch,
            Some(SnapshotEstimate {
                retained_bytes: overhead,
                copy_bytes: 0,
            }),
        )?;
        // Immutable state can be shared until activation. The payload's original
        // reservation remains live even if the source snapshot is released.
        self.branches.insert(
            id,
            Branch {
                state: Rc::clone(&self.snapshots[&handle.id]),
                run_id: id,
                _reservation: reservation,
            },
        );
        self.next_branch = next;
        Ok(SpeculativeBranchHandle {
            owner: Arc::clone(&self.owner),
            id,
        })
    }

    pub(super) fn exchange_inner(
        &mut self,
        handle: &SpeculativeBranchHandle,
    ) -> Result<SpeculativeBranchInfo, SpeculativeControlError> {
        self.healthy()?;
        self.owned_branch(handle)?;
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        let incoming = Rc::clone(&self.branches[&handle.id].state);
        let incoming_run = self.branches[&handle.id].run_id;
        // Reserve both copies before replacing anything. Every switch prices
        // the actual outgoing state, including growth since its last activation.
        let outgoing = self.save_state(SnapshotResourceKind::BranchCopy)?;
        let _restore = self
            .budget
            .as_ref()
            .expect("branch budget")
            .reserve(SnapshotResourceKind::Restore, Some(incoming.estimate))?;
        let request = self
            .scheduler
            .requests
            .request_mut(self.id.ok_or(SpeculativeControlError::NotQuiescent)?)
            .expect("submitted request");
        self.failed = true;
        request.restore_control_snapshot(
            self.scheduler.executor,
            &incoming.state,
            self.scheduler.context,
        )?;
        let slot = self.branches.get_mut(&handle.id).expect("owned branch");
        slot.state = outgoing;
        slot.run_id = self.run_id;
        self.run_id = incoming_run;
        self.epoch = epoch;
        self.failed = false;
        Ok(SpeculativeBranchInfo {
            run_id: self.run_id,
            token_ids: self.token_ids().to_vec(),
            status: self.status(),
        })
    }
}
