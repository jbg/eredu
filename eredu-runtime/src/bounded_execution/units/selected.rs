//! Portable ownership of selected resident and bounded parameter execution.

use std::{
    collections::BTreeSet,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use eredu_core::checkpoint_mechanisms::unit_binding::ImmutableUnitStorage;
use eredu_core::{
    residency::{
        OffloadUnitId, ParameterConversionRetentionTrimReport, ParameterConversionTrimError,
    },
    resources::ResourceIdentity,
    BackendFailure,
};
use eredu_nn::{NeuralBackend, Tensor};

use super::{BoundedDenseRun, BoundedUnitDriver, BoundedUnitMechanisms};
use crate::{
    dense::DenseStreamController, DenseDiskStreamReport, ExecutionUnitAddress,
    ExecutionUnitAddressMap, ExecutionUnitLeaseAddress, ImmutableParameterPolicy,
    LayerwiseAcquireError, LayerwisePolicy, ResidencyReport, ResidentLayerGroupReport,
    UnitPolicyResidency, UnitResidencyTelemetry,
};

/// Reports retained storage facts for one selected policy.
pub trait SelectedUnitResources {
    /// Readback failure for the selected policy.
    type Error: std::error::Error + Send + Sync + 'static + From<BackendFailure>;
    /// Exact invocation and physical storage mapping.
    fn addresses(&self) -> &ExecutionUnitAddressMap;
    /// Logical residency of the selected units.
    fn residency_report(&self) -> Result<ResidencyReport, Self::Error>;
    /// Dense disk-stream counters for the installed mechanism, if applicable.
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Self::Error>;
    /// Per-group residency and transfers.
    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Self::Error>;
    /// Releases optional converted parameter storage.
    fn trim_parameter_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError>;
}

/// Native bounded storage already admitted for one physical parameter catalog.
pub trait BoundedUnitBinding: BoundedUnitMechanisms {
    /// Retains dense state until guard cleanup can execute outside native locks.
    type DenseOwner: DerefMut<Target = BoundedDenseRun<Self::Residency>>;

    /// Returns exact units, permanent pins, and selected lookahead.
    fn bounded_layout(&self) -> (Vec<OffloadUnitId>, BTreeSet<OffloadUnitId>, usize);
    /// Returns the portable controller retained with the admitted dense storage.
    fn dense_controller(&self) -> Option<&Arc<DenseStreamController<Self::Residency>>>;
    /// Provides ordinary-host retirement for a supplied portable dense run.
    fn retain_dense_run(&self, run: BoundedDenseRun<Self::Residency>) -> Self::DenseOwner;
}

/// Portable errors before native unit acquisition or release.
#[derive(Debug, thiserror::Error)]
pub enum SelectedLayerwiseError {
    /// A prior panic poisoned the shared execution owner.
    #[error("selected layerwise policy lock was poisoned")]
    Poisoned,
    /// This lease was acquired by a different selected policy.
    #[error("selected layerwise lease belongs to another policy")]
    ForeignLease,
    /// A lease was returned to a different residency kind.
    #[error("selected layerwise policy received a lease from another residency")]
    WrongResidency,
}

enum SelectedKind<R, N, L, D> {
    Resident(R),
    Bounded {
        driver: BoundedUnitDriver<L, D>,
        native: N,
    },
}

/// Shared selected execution and its portable bounded driver.
pub struct SelectedLayerwisePolicy<R, N, L, D, I> {
    inner: Arc<Mutex<SelectedKind<R, N, L, D>>>,
    addresses: ExecutionUnitAddressMap,
    identity: Arc<()>,
    immutable: Option<Arc<I>>,
}

impl<R, N, L, D, I> Clone for SelectedLayerwisePolicy<R, N, L, D, I> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            addresses: self.addresses.clone(),
            identity: Arc::clone(&self.identity),
            immutable: self.immutable.clone(),
        }
    }
}

impl<R, N, L, D, I> SelectedLayerwisePolicy<R, N, L, D, I>
where
    R: SelectedUnitResources,
    N: SelectedUnitResources<Error = R::Error>,
{
    /// Adopts the exact installed resident parameters.
    pub fn resident(policy: R) -> Self {
        Self {
            addresses: policy.addresses().clone(),
            identity: Arc::new(()),
            immutable: None,
            inner: Arc::new(Mutex::new(SelectedKind::Resident(policy))),
        }
    }

    /// Installs the portable driver over admitted native storage.
    pub fn bounded<U: 'static>(native: N) -> Self
    where
        N: BoundedUnitBinding<Lease<U> = L, DenseOwner = D>,
        D: DerefMut<Target = BoundedDenseRun<N::Residency>>,
    {
        let addresses = native.addresses().clone();
        let (ids, pinned, depth) = native.bounded_layout();
        let dense = native.dense_controller().map(|controller| {
            native.retain_dense_run(BoundedDenseRun::new(
                Arc::clone(controller),
                addresses.storage_layout().group_count(),
            ))
        });
        let driver = BoundedUnitDriver::new(ids, pinned, depth, dense);
        Self {
            addresses,
            identity: Arc::new(()),
            immutable: None,
            inner: Arc::new(Mutex::new(SelectedKind::Bounded { driver, native })),
        }
    }

    /// Releases optional converted parameter storage at an idle boundary.
    pub fn trim_parameter_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        let policy = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::from_error(SelectedLayerwiseError::Poisoned))?;
        match &*policy {
            SelectedKind::Resident(resident) => resident.trim_parameter_conversions(),
            SelectedKind::Bounded { native, .. } => native.trim_parameter_conversions(),
        }
    }

    /// Reports logical residency and transfers.
    pub fn residency_report(&self) -> Result<ResidencyReport, R::Error> {
        let policy = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::from_error(SelectedLayerwiseError::Poisoned))?;
        match &*policy {
            SelectedKind::Resident(resident) => resident.residency_report(),
            SelectedKind::Bounded { native, .. } => native.residency_report(),
        }
    }

    /// Reports dense disk-stream counters when selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, R::Error> {
        let policy = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::from_error(SelectedLayerwiseError::Poisoned))?;
        match &*policy {
            SelectedKind::Resident(resident) => resident.dense_stream_report(),
            SelectedKind::Bounded { native, .. } => native.dense_stream_report(),
        }
    }
}

/// Exclusive unit acquired for one exact invocation of a selected policy.
pub struct SelectedUnitLease<R, L> {
    address: ExecutionUnitLeaseAddress,
    identity: Arc<()>,
    inner: SelectedLeaseKind<R, L>,
}

enum SelectedLeaseKind<R, L> {
    Resident(R),
    Bounded(L),
}

impl<R: Deref, L: Deref<Target = R::Target>> Deref for SelectedUnitLease<R, L> {
    type Target = R::Target;
    fn deref(&self) -> &Self::Target {
        match &self.inner {
            SelectedLeaseKind::Resident(lease) => lease,
            SelectedLeaseKind::Bounded(lease) => lease,
        }
    }
}

impl<R: DerefMut, L: DerefMut<Target = R::Target>> DerefMut for SelectedUnitLease<R, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.inner {
            SelectedLeaseKind::Resident(lease) => lease,
            SelectedLeaseKind::Bounded(lease) => lease,
        }
    }
}

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

impl<B, U, R, N, L, D, I, E> LayerwisePolicy<B, U> for SelectedLayerwisePolicy<R, N, L, D, I>
where
    B: NeuralBackend,
    U: eredu_nn::Parameterized<B::Tensor> + 'static,
    R: LayerwisePolicy<B, U, Error = E> + SelectedUnitResources<Error = E>,
    N: BoundedUnitMechanisms<Lease<U> = L, Tensor = B::Tensor, Context = Context<B>, Error = E>
        + SelectedUnitResources<Error = E>,
    L: DerefMut<Target = U>,
    D: DerefMut<Target = BoundedDenseRun<N::Residency>>,
    E: From<BackendFailure>,
{
    type Lease = SelectedUnitLease<R::Lease, L>;
    type Error = E;

    fn permits_deferred_serial_groups(&self) -> bool {
        let Ok(policy) = self.inner.lock() else {
            return false;
        };
        match &*policy {
            SelectedKind::Resident(policy) => policy.permits_deferred_serial_groups(),
            SelectedKind::Bounded { .. } => false,
        }
    }

    fn resident_parameters_available(&self) -> bool {
        let Ok(policy) = self.inner.lock() else {
            return false;
        };
        match &*policy {
            SelectedKind::Resident(policy) => policy.resident_parameters_available(),
            SelectedKind::Bounded { .. } => false,
        }
    }

    fn visit_resident_units(&mut self, visitor: &mut impl FnMut(&mut U)) -> bool {
        let Ok(mut policy) = self.inner.lock() else {
            return false;
        };
        match &mut *policy {
            SelectedKind::Resident(policy) => policy.visit_resident_units(visitor),
            SelectedKind::Bounded { .. } => false,
        }
    }

    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, B::Tensor>,
        active: bool,
    ) -> Result<bool, E> {
        let mut policy = self
            .inner
            .lock()
            .map_err(|_| E::from(BackendFailure::from_error(SelectedLayerwiseError::Poisoned)))?;
        match &mut *policy {
            SelectedKind::Resident(policy) => policy.publish_parameter_replacements(values, active),
            SelectedKind::Bounded { driver, native } => {
                driver.publish_replacements::<U, _>(native, values, active)
            }
        }
    }

    fn parameter_telemetry(&self) -> Option<&dyn crate::layered::ParameterPolicyTelemetry> {
        Some(self)
    }

    fn begin(&mut self, initial: &B::Tensor, context: &Context<B>) -> Result<(), Self::Error> {
        let mut policy = self
            .inner
            .lock()
            .map_err(|_| E::from(BackendFailure::from_error(SelectedLayerwiseError::Poisoned)))?;
        match &mut *policy {
            SelectedKind::Resident(policy) => policy.begin(initial, context),
            SelectedKind::Bounded { driver, native } => driver.begin::<U, _>(native, initial),
        }
    }

    fn inspect_unit<A, F, V>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        operation: V,
        context: &Context<B>,
    ) -> Result<bool, LayerwiseAcquireError<A, Self::Error>>
    where
        F: FnOnce(&Context<B>) -> Result<U, A>,
        V: FnOnce(&mut U) -> Result<(), Self::Error>,
    {
        let local = self.addresses.resolve(ordinal, address).map_err(|error| {
            LayerwiseAcquireError::Policy(E::from(BackendFailure::from_error(error)))
        })?;
        let invocation = local;
        let mut policy = self.inner.lock().map_err(|_| {
            LayerwiseAcquireError::Policy(
                BackendFailure::from_error(SelectedLayerwiseError::Poisoned).into(),
            )
        })?;
        match &mut *policy {
            SelectedKind::Resident(policy) => {
                policy.inspect_unit(ordinal, address, build, operation, context)
            }
            SelectedKind::Bounded { driver, native } => driver.inspect(
                native,
                &self.addresses,
                invocation,
                build,
                operation,
                context,
            ),
        }
    }

    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        context: &Context<B>,
    ) {
        let active = active.and_then(|(_, _, lease)| {
            if !Arc::ptr_eq(&self.identity, &lease.identity) {
                return None;
            }
            Some((
                lease.address.invocation_ordinal(),
                lease.address.invocation_address(),
                lease.inner,
            ))
        });
        let Ok(mut policy) = self.inner.lock() else {
            drop(active);
            return;
        };
        match (&mut *policy, active) {
            (
                SelectedKind::Resident(policy),
                Some((ordinal, address, SelectedLeaseKind::Resident(lease))),
            ) => policy.abort(Some((ordinal, address, lease)), context),
            (
                SelectedKind::Bounded { driver, .. },
                Some((_, _, SelectedLeaseKind::Bounded(lease))),
            ) => driver.abort(Some(lease)),
            (SelectedKind::Resident(policy), _) => policy.abort(None, context),
            (SelectedKind::Bounded { driver, .. }, _) => driver.abort(None),
        }
    }

    fn acquire<A, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &Context<B>,
    ) -> Result<Self::Lease, LayerwiseAcquireError<A, Self::Error>>
    where
        F: FnOnce(&Context<B>) -> Result<U, A>,
    {
        let address = self.addresses.resolve(ordinal, address).map_err(|error| {
            LayerwiseAcquireError::Policy(E::from(BackendFailure::from_error(error)))
        })?;
        let mut policy = self.inner.lock().map_err(|_| {
            LayerwiseAcquireError::Policy(
                BackendFailure::from_error(SelectedLayerwiseError::Poisoned).into(),
            )
        })?;
        let lease = match &mut *policy {
            SelectedKind::Resident(policy) => policy
                .acquire(ordinal, address.invocation_address(), build, context)
                .map(SelectedLeaseKind::Resident),
            SelectedKind::Bounded { driver, native } => driver
                .acquire(native, &self.addresses, address, build, context)
                .map(SelectedLeaseKind::Bounded),
        }?;
        Ok(SelectedUnitLease {
            address,
            identity: Arc::clone(&self.identity),
            inner: lease,
        })
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a B::Tensor,
        state_values: StateValues,
        context_values: ContextValues,
        context: &Context<B>,
    ) -> Result<(), Self::Error>
    where
        B::Tensor: 'a,
        StateValues: Iterator<Item = &'a B::Tensor>,
        ContextValues: Iterator<Item = &'a B::Tensor>,
    {
        if !Arc::ptr_eq(&self.identity, &lease.identity) {
            self.abort(None, context);
            return Err(BackendFailure::from_error(SelectedLayerwiseError::ForeignLease).into());
        }
        if let Err(error) = lease.address.validate_return(ordinal, address) {
            self.abort(Some((ordinal, address, lease)), context);
            return Err(BackendFailure::from_error(error).into());
        }
        let invocation = lease.address;
        let mut policy = self
            .inner
            .lock()
            .map_err(|_| E::from(BackendFailure::from_error(SelectedLayerwiseError::Poisoned)))?;
        match (&mut *policy, lease.inner) {
            (SelectedKind::Resident(policy), SelectedLeaseKind::Resident(lease)) => policy
                .complete(
                    ordinal,
                    address,
                    lease,
                    output,
                    state_values,
                    context_values,
                    context,
                ),
            (SelectedKind::Bounded { driver, native }, SelectedLeaseKind::Bounded(lease)) => driver
                .complete::<U, _>(
                    native,
                    &self.addresses,
                    invocation,
                    lease,
                    std::iter::once(output)
                        .chain(state_values)
                        .chain(context_values),
                ),
            _ => Err(BackendFailure::from_error(SelectedLayerwiseError::WrongResidency).into()),
        }
    }

    fn finish(&mut self, output: &B::Tensor, context: &Context<B>) -> Result<(), Self::Error> {
        let mut policy = self
            .inner
            .lock()
            .map_err(|_| E::from(BackendFailure::from_error(SelectedLayerwiseError::Poisoned)))?;
        match &mut *policy {
            SelectedKind::Resident(policy) => policy.finish(output, context),
            SelectedKind::Bounded { driver, native } => {
                driver.finish::<U, _>(native, output, context)
            }
        }
    }
}

impl<R, N, L, D, I> UnitPolicyResidency for SelectedLayerwisePolicy<R, N, L, D, I>
where
    R: SelectedUnitResources,
    N: SelectedUnitResources<Error = R::Error>,
{
    fn residency(&self) -> Result<ResidencyReport, BackendFailure> {
        self.residency_report().map_err(BackendFailure::from_error)
    }
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.dense_stream_report()
            .map_err(BackendFailure::from_error)
    }
}

impl<R, N, L, D, I> crate::LocalParameterRetention for SelectedLayerwisePolicy<R, N, L, D, I>
where
    R: SelectedUnitResources,
    N: SelectedUnitResources<Error = R::Error>,
{
    fn trim_local_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.trim_parameter_conversions()
    }
}

impl<B: NeuralBackend, R, N, L, D, I: ImmutableUnitStorage<Context<B>>> ImmutableParameterPolicy<B>
    for SelectedLayerwisePolicy<R, N, L, D, I>
{
    type Certificate = I;

    fn set_immutable_certificate(&mut self, certificate: I) {
        self.immutable = Some(Arc::new(certificate));
    }

    fn immutable_parameters_identity(&self, context: &Context<B>) -> Option<&ResourceIdentity> {
        self.immutable.as_ref()?.identity(context)
    }
}

impl<R, N, L, D, I> UnitResidencyTelemetry for SelectedLayerwisePolicy<R, N, L, D, I>
where
    R: SelectedUnitResources,
    N: SelectedUnitResources<Error = R::Error>,
{
    type Error = R::Error;

    fn residency_report(&self) -> Result<ResidencyReport, Self::Error> {
        self.residency_report()
    }
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Self::Error> {
        self.dense_stream_report()
    }
    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Self::Error> {
        let policy = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::from_error(SelectedLayerwiseError::Poisoned))?;
        match &*policy {
            SelectedKind::Resident(policy) => policy.execution_group_reports(),
            SelectedKind::Bounded { native, .. } => native.execution_group_reports(),
        }
    }
}
