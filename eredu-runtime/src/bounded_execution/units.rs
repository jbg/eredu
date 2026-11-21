//! Portable execution of bounded parameter windows.

mod binding;
pub use binding::{BindBoundedPolicy, BindResidentPolicy};
mod resident;
pub use resident::{ResidentUnit, ResidentUnitError, ResidentUnitPolicy};
mod storage;
use eredu_core::residency::{NativeBoundedUnits, NativeResidentUnits};
pub use storage::{BoundUnitStorage, UnitStorageError};
mod selected;
pub use selected::{
    BoundedUnitBinding, SelectedLayerwiseError, SelectedLayerwisePolicy, SelectedUnitLease,
    SelectedUnitResources,
};

/// Derives the selected layerwise policy from native storage, transfers and leases.
pub type BoundParameterUnitPolicy<B, U, N, O, I> = SelectedLayerwisePolicy<
    ResidentUnitPolicy<
        B,
        N,
        <O as NativeResidentUnits<
            B,
            <N as eredu_core::residency::ResidencyStorageMechanisms>::Transfer,
        >>::Module<U>,
        storage::ResidentUnitRetention<N, O>,
        <O as eredu_core::residency::UnitRetirement>::Error,
    >,
    BoundUnitStorage<N, O>,
    <O as NativeBoundedUnits>::Lease<U>,
    <BoundUnitStorage<N, O> as BoundedUnitBinding>::DenseOwner,
    I,
>;

use std::{
    collections::{BTreeSet, VecDeque},
    ops::DerefMut,
    sync::Arc,
};

use crate::MaterializedUnit;
use eredu_core::{
    checkpoint_mechanisms::CheckpointParameterBackend,
    residency::{MemoryTier, OffloadUnitId},
    BackendFailure,
};
use eredu_nn::Parameterized;

mod population;
pub use population::SelectiveUnitParameters;

use crate::{
    dense::{
        sample_residency_memory, DenseStreamController, DenseStreamForwardGuard,
        DenseStreamGroupGuard, DenseStreamMechanisms, DenseTransferWindow,
    },
    residency::ResidencyWindowManager,
    ExecutionUnitAddressMap, ExecutionUnitLeaseAddress, ExecutionUnitStorageLocation,
    LayerwiseAcquireError,
};

fn install_bounded_unit<U, N>(
    native: &mut N,
    unit: U,
    transfer: N::Transfer,
    context: &N::Context,
) -> Result<N::Lease<U>, N::Error>
where
    N: BoundedUnitMechanisms,
    U: Parameterized<N::Tensor> + 'static,
{
    let mut lease = native.bind_lease(unit, transfer, context)?;
    let weights = MaterializedUnit::<N::Parameters>::try_from_named_weights(N::unit_weights::<U>(
        &mut lease,
    )?)
    .map_err(BackendFailure::from_error)?;
    population::install_unit::<N::Parameters, _>(&mut *lease, weights, native.population())?;
    Ok(lease)
}

/// Access to the installed residency owner alongside native bounded operations.
pub trait BoundedUnitMechanisms: NativeBoundedUnits {
    /// Portable policy used when populating units.
    fn population(&self) -> Option<&SelectiveUnitParameters<Self::Tensor>>;
    /// Records already admitted immutable replacements.
    fn publish_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, Self::Tensor>,
        active: bool,
    ) -> bool;

    /// Residency and dense scheduling controlled by portable execution.
    type Residency: DenseStreamMechanisms<Transfer = Self::Transfer>;

    /// Returns the installed storage primitives for this unit catalog.
    fn residency(&self) -> &Self::Residency;
    /// Whether the selected plan samples allocator and process memory, respectively.
    fn memory_sampling(&self) -> (bool, bool);
}

/// Portable bounded execution that cannot proceed in the current state.
#[derive(Debug, thiserror::Error)]
pub enum BoundedUnitError {
    /// A bounded run was started before the previous run settled.
    #[error("bounded execution already has an active forward")]
    ActiveForward,
    /// The operation needs a settled native unit frontier.
    #[error("{operation} requires an idle bounded execution")]
    Busy {
        /// Requested operation.
        operation: &'static str,
    },
    /// The returned transfer disagreed with the admitted invocation.
    #[error("dense transfer returned storage unit {actual}, expected {expected}")]
    WrongTransfer {
        /// Physical ordinal returned by the mechanism.
        actual: usize,
        /// Selected physical ordinal.
        expected: usize,
    },
}

/// Portable state for one dense run. The backend may defer destruction of this
/// state so guard cleanup occurs at an ordinary unlocked host boundary.
pub struct BoundedDenseRun<M: DenseStreamMechanisms> {
    controller: Arc<DenseStreamController<M>>,
    windows: Vec<Option<DenseTransferWindow<M>>>,
    forward: Option<DenseStreamForwardGuard<M>>,
    groups: Vec<Option<DenseStreamGroupGuard<M>>>,
    prefill: bool,
    aborted: bool,
}

impl<M: DenseStreamMechanisms> BoundedDenseRun<M> {
    /// Allocates an idle forward for the exact number of execution groups.
    fn new(controller: Arc<DenseStreamController<M>>, groups: usize) -> Self {
        Self {
            controller,
            windows: (0..groups).map(|_| None).collect(),
            groups: (0..groups).map(|_| None).collect(),
            forward: None,
            prefill: false,
            aborted: false,
        }
    }
}

/// Orders bounded acquisition, consumer retirement, and forward publication.
struct BoundedUnitDriver<L, D> {
    unit_ids: Vec<OffloadUnitId>,
    pinned: BTreeSet<OffloadUnitId>,
    depth: usize,
    pending: VecDeque<L>,
    dense: Option<D>,
}

impl<L, D, M> BoundedUnitDriver<L, D>
where
    D: DerefMut<Target = BoundedDenseRun<M>>,
    M: DenseStreamMechanisms,
{
    /// Binds already validated physical identities and lookahead to the idle driver.
    fn new(
        unit_ids: Vec<OffloadUnitId>,
        pinned: BTreeSet<OffloadUnitId>,
        depth: usize,
        dense: Option<D>,
    ) -> Self {
        Self {
            unit_ids,
            pinned,
            depth,
            pending: VecDeque::new(),
            dense,
        }
    }

    /// Publishes immutable replacements only while there is no active consumer.
    pub fn publish_replacements<U: 'static, N>(
        &mut self,
        native: &mut N,
        values: &std::collections::BTreeMap<String, N::Tensor>,
        active: bool,
    ) -> Result<bool, N::Error>
    where
        N: BoundedUnitMechanisms<Lease<U> = L, Residency = M, Transfer = M::Transfer>,
    {
        if !self.pending.is_empty()
            || self
                .dense
                .as_ref()
                .is_some_and(|dense| dense.forward.is_some())
        {
            return Err(BackendFailure::from_error(BoundedUnitError::Busy {
                operation: "parameter publication",
            })
            .into());
        }
        Ok(native.publish_replacements(values, active))
    }

    /// Starts a forward after retiring work left by the previous one.
    pub fn begin<U: 'static, N>(
        &mut self,
        native: &mut N,
        initial: &N::Tensor,
    ) -> Result<(), N::Error>
    where
        N: BoundedUnitMechanisms<Lease<U> = L, Residency = M, Transfer = M::Transfer>,
    {
        native.reclaim(false);
        self.drain::<U, N>(native)?;
        let Some(dense) = &mut self.dense else {
            return Ok(());
        };
        if dense.aborted {
            for window in &mut dense.windows {
                window.take();
            }
            for group in &mut dense.groups {
                group.take();
            }
            dense.forward.take();
            dense.aborted = false;
        }
        if dense.forward.is_some()
            || dense.windows.iter().any(Option::is_some)
            || dense.groups.iter().any(Option::is_some)
        {
            return Err(BackendFailure::from_error(BoundedUnitError::ActiveForward).into());
        }
        let prefill = native.token_width(initial) > 1;
        dense.forward = Some(
            dense
                .controller
                .forward_guard(prefill)
                .map_err(BackendFailure::from_error)?,
        );
        dense.prefill = prefill;
        Ok(())
    }

    /// Releases the tentative unit and defers lock-taking cleanup to the next begin.
    pub fn abort(&mut self, active: Option<L>) {
        drop(active);
        if let Some(dense) = &mut self.dense {
            dense.aborted = true;
        }
    }

    fn drain_one<U: 'static, N>(&mut self, native: &mut N) -> Result<(), N::Error>
    where
        N: BoundedUnitMechanisms<Lease<U> = L, Residency = M, Transfer = M::Transfer>,
    {
        if let Some(lease) = self.pending.pop_front() {
            native.retire::<U>(lease)?;
            native.reclaim_retired();
        }
        Ok(())
    }

    /// Establishes completion of every outstanding consumer in order.
    pub fn drain<U: 'static, N>(&mut self, native: &mut N) -> Result<(), N::Error>
    where
        N: BoundedUnitMechanisms<Lease<U> = L, Residency = M, Transfer = M::Transfer>,
    {
        while !self.pending.is_empty() {
            self.drain_one::<U, N>(native)?;
        }
        Ok(())
    }

    fn trim_window<U: 'static, N>(
        &self,
        native: &mut N,
        window: &[ExecutionUnitStorageLocation],
    ) -> Result<(), N::Error>
    where
        N: BoundedUnitMechanisms<Lease<U> = L, Residency = M, Transfer = M::Transfer>,
    {
        for (index, id) in self.unit_ids.iter().enumerate() {
            if !self.pinned.contains(id) && !window.iter().any(|unit| unit.ordinal() == index) {
                native
                    .residency()
                    .evict(id, MemoryTier::Device)
                    .map_err(BackendFailure::from_error)?;
            }
        }
        Ok(())
    }

    /// Runs an inspection against one unit and settles its exact native scope.
    pub fn inspect<U, N, E>(
        &mut self,
        native: &mut N,
        mapping: &ExecutionUnitAddressMap,
        invocation: ExecutionUnitLeaseAddress,
        build: impl FnOnce(&N::Context) -> Result<U, E>,
        operation: impl FnOnce(&mut U) -> Result<(), N::Error>,
        context: &N::Context,
    ) -> Result<bool, LayerwiseAcquireError<E, N::Error>>
    where
        U: Parameterized<N::Tensor> + 'static,
        N: BoundedUnitMechanisms<Lease<U> = L, Residency = M, Transfer = M::Transfer>,
        L: DerefMut<Target = U>,
    {
        let inspect = |error| LayerwiseAcquireError::Policy(error);
        native.reclaim(true);
        self.drain::<U, N>(native).map_err(inspect)?;
        if self.dense.as_ref().is_some_and(|dense| {
            dense.forward.is_some() || dense.windows.iter().any(Option::is_some)
        }) {
            return Err(inspect(
                BackendFailure::from_error(BoundedUnitError::Busy {
                    operation: "parameter inspection",
                })
                .into(),
            ));
        }
        let window = mapping
            .storage_window(invocation.invocation_ordinal(), self.depth)
            .expect("validated invocation");
        self.trim_window::<U, N>(native, &window).map_err(inspect)?;
        let unit = build(context).map_err(LayerwiseAcquireError::Architecture)?;
        let id = &self.unit_ids[invocation.storage_location().ordinal()];
        let transfer = native
            .residency()
            .acquire_many_with_transfer(&[(id.clone(), 1)], MemoryTier::Device)
            .map_err(|error| inspect(BackendFailure::from_error(error).into()))?;
        let mut lease = install_bounded_unit(native, unit, transfer, context).map_err(inspect)?;
        operation(&mut *lease).map_err(inspect)?;
        native
            .submit_inspection::<U>(&mut lease, context)
            .map_err(inspect)?;
        native.retire::<U>(lease).map_err(inspect)?;
        native.reclaim_retired();
        Ok(true)
    }

    /// Acquires and populates the physical unit selected for this invocation.
    pub fn acquire<U, N, E>(
        &mut self,
        native: &mut N,
        mapping: &ExecutionUnitAddressMap,
        invocation: ExecutionUnitLeaseAddress,
        build: impl FnOnce(&N::Context) -> Result<U, E>,
        context: &N::Context,
    ) -> Result<L, LayerwiseAcquireError<E, N::Error>>
    where
        U: Parameterized<N::Tensor> + 'static,
        N: BoundedUnitMechanisms<Lease<U> = L, Residency = M, Transfer = M::Transfer>,
    {
        let policy = LayerwiseAcquireError::Policy;
        native.reclaim(false);
        let unit = build(context).map_err(LayerwiseAcquireError::Architecture)?;
        while let Some(lease) = self.pending.front() {
            if !native.is_complete::<U>(lease).map_err(policy)? {
                break;
            }
            self.drain_one::<U, N>(native).map_err(policy)?;
        }
        if let Some(dense) = &mut self.dense {
            let storage = invocation.storage_location();
            let group = storage.address().group();
            let layout = mapping.storage_layout();
            let name = layout.group_id(group).expect("validated group").as_str();
            if dense.groups[group].is_none() {
                dense.groups[group] = Some(dense.controller.group_guard(name));
            }
            if dense.windows[group].is_none() {
                let sequence = mapping
                    .group_storage_ordinals(group)
                    .expect("validated group");
                dense.windows[group] = Some(
                    dense
                        .controller
                        .transfer_window(name, &self.unit_ids, sequence, dense.prefill)
                        .map_err(|error| policy(BackendFailure::from_error(error).into()))?,
                );
            }
            loop {
                let dense = self.dense.as_mut().expect("dense run");
                let window = dense.windows[group].as_mut().expect("open transfer window");
                match window.refill() {
                    Ok(()) => break,
                    Err(_) if !self.pending.is_empty() => {
                        self.drain_one::<U, N>(native).map_err(policy)?
                    }
                    Err(error) if window.can_advance_after(&error) => break,
                    Err(error) => return Err(policy(BackendFailure::from_error(error).into())),
                }
            }
            let window = self.dense.as_mut().expect("dense run").windows[group]
                .as_mut()
                .expect("open transfer window");
            let (step, transfer) = window
                .next()
                .map_err(|error| policy(BackendFailure::from_error(error).into()))?;
            let ordinal = step.ordinal();
            let actual = step.storage_index();
            let group_start = mapping.group_range(group).expect("validated group").start;
            let expected = storage.ordinal();
            if actual != expected || ordinal != invocation.invocation_ordinal() - group_start {
                return Err(policy(
                    BackendFailure::from_error(BoundedUnitError::WrongTransfer {
                        actual,
                        expected,
                    })
                    .into(),
                ));
            }
            return install_bounded_unit(native, unit, transfer, context).map_err(policy);
        }

        // A prior window can own an overlapping in-flight copy. Retire its
        // consumer before acquiring the next window.
        self.drain_one::<U, N>(native).map_err(policy)?;
        let window = mapping
            .storage_window(invocation.invocation_ordinal(), self.depth)
            .expect("validated invocation");
        self.trim_window::<U, N>(native, &window).map_err(policy)?;
        let requests = window
            .iter()
            .map(|unit| (self.unit_ids[unit.ordinal()].clone(), 1))
            .collect::<Vec<_>>();
        let transfer = loop {
            match native
                .residency()
                .acquire_many_with_transfer(&requests, MemoryTier::Device)
            {
                Ok(transfer) => break transfer,
                Err(_) if !self.pending.is_empty() => {
                    self.drain_one::<U, N>(native).map_err(policy)?
                }
                Err(error) => return Err(policy(BackendFailure::from_error(error).into())),
            }
        };
        install_bounded_unit(native, unit, transfer, context).map_err(policy)
    }

    /// Attaches consumer completion before advancing to a different group.
    pub fn complete<'a, U: 'static, N>(
        &mut self,
        native: &mut N,
        mapping: &ExecutionUnitAddressMap,
        invocation: ExecutionUnitLeaseAddress,
        mut lease: L,
        roots: impl Iterator<Item = &'a N::Tensor>,
    ) -> Result<(), N::Error>
    where
        N: BoundedUnitMechanisms<Lease<U> = L, Residency = M, Transfer = M::Transfer>,
        N::Tensor: 'a,
    {
        native.submit::<U>(&mut lease, roots)?;
        self.pending.push_back(lease);
        if self.dense.is_some() && mapping.is_group_end(invocation.invocation_ordinal()) {
            self.drain::<U, N>(native)?;
            let dense = self.dense.as_mut().expect("dense run");
            let group = invocation.storage_location().address().group();
            dense.windows[group].take();
            if let Some(owner) = dense.groups[group].take() {
                owner.complete().map_err(BackendFailure::from_error)?;
            }
        }
        Ok(())
    }

    /// Settles the output, retires consumers and commits the forward.
    pub fn finish<U: 'static, N>(
        &mut self,
        native: &mut N,
        output: &N::Tensor,
        context: &N::Context,
    ) -> Result<(), N::Error>
    where
        N: BoundedUnitMechanisms<Lease<U> = L, Residency = M, Transfer = M::Transfer>,
    {
        native.finish_output(output, context)?;
        self.drain::<U, N>(native)?;
        if let Some(dense) = &mut self.dense {
            for window in &mut dense.windows {
                window.take();
            }
            for group in &mut dense.groups {
                if let Some(owner) = group.take() {
                    owner.complete().map_err(BackendFailure::from_error)?;
                }
            }
            if let Some(forward) = dense.forward.take() {
                forward.complete().map_err(BackendFailure::from_error)?;
            }
        } else {
            let (allocator, process) = native.memory_sampling();
            sample_residency_memory(native.residency(), allocator, process)
                .map_err(BackendFailure::from_error)?;
        }
        Ok(())
    }
}
