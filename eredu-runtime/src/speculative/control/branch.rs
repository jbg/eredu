use super::*;

/// Inactive serial branch slot owned by one controlled scope. Cloning a handle
/// does not copy state. Release the slot explicitly when it is no longer needed.
#[derive(Debug, Clone)]
pub struct SpeculativeBranchHandle {
    owner: SpeculativeRequestIdentity,
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
    pub token_ids: SpeculativeValues<u32>,
    /// Saved lifecycle; terminal branches remain inspectable.
    pub status: SpeculativeRequestStatus,
}

pub(super) struct Branch<E: SpeculativeExecutor, S: SpeculativeSampling, C> {
    state: SavedOwner<E, S, C>,
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
        if !self.owner.same(&handle.owner) || !self.branches.contains_key(&handle.id) {
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
            token_ids: views::collect(branch.state.state.token_ids().iter().copied(),self.scheduler.executor,self.scheduler.context,
                std::mem::size_of::<SpeculativeBranchInfo>()+std::mem::size_of::<Result<SpeculativeBranchInfo,SpeculativeControlError>>())?,
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
        let overhead = self.branches.growth_bytes(self.scheduler.executor)?
            .checked_add(PendingSnapshotReservation::control_bytes().ok_or(ExecutionControlError::Overflow)?)
            .and_then(|n| n.checked_add(std::mem::size_of::<Branch<E,S,C>>()))
            .and_then(|n| u64::try_from(n).ok()).ok_or(ExecutionControlError::Overflow)?;
        let reservation = self.reserve_control(SnapshotResourceKind::Branch,
            SnapshotEstimate { retained_bytes: overhead, copy_bytes: 0 })?;
        self.branches.prepare_insert(self.scheduler.executor, self.scheduler.context)?;
        // Immutable state can be shared until activation. The payload's original
        // reservation remains live even if the source snapshot is released.
        self.branches.insert(
            id,
            Branch {
                state: self.snapshots[&handle.id].clone(),
                run_id: id,
                _reservation: reservation,
            },
        )?;
        self.next_branch = next;
        Ok(SpeculativeBranchHandle {
            owner: self.owner.clone(),
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
        let incoming = self.branches[&handle.id].state.clone();
        let incoming_run = self.branches[&handle.id].run_id;
        // Reserve both copies before replacing anything. Every switch prices
        // the actual outgoing state, including growth since its last activation.
        let outgoing = self.save_state(SnapshotResourceKind::BranchCopy)?;
        let _restore = self.reserve_control(SnapshotResourceKind::Restore, incoming.estimate)?;
        // Prepare the exact immutable incoming prefix before replacing either
        // active state or the branch slot. Refusal leaves both installed owners.
        let info=SpeculativeBranchInfo {
            run_id:incoming_run,
            token_ids:views::collect(incoming.state.token_ids().iter().copied(),self.scheduler.executor,self.scheduler.context,
                std::mem::size_of::<SpeculativeBranchInfo>()+std::mem::size_of::<Result<SpeculativeBranchInfo,SpeculativeControlError>>())?,
            status:incoming.state.status(),
        };
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
        Ok(info)
    }
}
