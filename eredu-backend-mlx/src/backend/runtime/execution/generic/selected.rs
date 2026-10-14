//! Shared native selected-policy handles used by execution and residency reporting.

use super::*;
use eredu_runtime::ResidencyReport;
use std::{
    ops::{Deref, DerefMut},
    sync::Mutex,
};

pub(crate) enum MlxSelectedLayerwisePolicyInner<U: 'static, P> {
    Resident(MlxResidentPolicy<U>),
    Bounded {
        policy: MlxLayerwisePolicy<U, P>,
        local_addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
    },
}

pub(crate) struct MlxSelectedLayerwisePolicy<U: 'static, P> {
    inner: Arc<Mutex<MlxSelectedLayerwisePolicyInner<U, P>>>,
    /// Exact architecture addresses paired with policy-local residency addresses.
    /// Pipeline cuts may retain a nonzero global unit index at local ordinal zero.
    parameter_locations: Arc<
        [(
            eredu_runtime::ExecutionUnitAddress,
            eredu_runtime::ExecutionUnitAddress,
        )],
    >,
}

impl<U: 'static, P> Clone for MlxSelectedLayerwisePolicy<U, P> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            parameter_locations: Arc::clone(&self.parameter_locations),
        }
    }
}

impl<U: 'static, P> MlxSelectedLayerwisePolicy<U, P> {
    fn parameter_locations(
        layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
    ) -> Result<
        Arc<
            [(
                eredu_runtime::ExecutionUnitAddress,
                eredu_runtime::ExecutionUnitAddress,
            )],
        >,
        Error,
    > {
        if layout.len() != addresses.len() {
            return Err(Error::ArchitectureModel(
                "prepared parameter addresses differ from policy units".into(),
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut locations = Vec::with_capacity(addresses.len());
        for (ordinal, address) in addresses.iter().copied().enumerate() {
            let local = layout.address(ordinal).expect("validated policy layout");
            if local.group() != address.group() || !seen.insert((address.group(), address.index()))
            {
                return Err(Error::ArchitectureModel(
                    "prepared parameter addresses differ from policy groups".into(),
                ));
            }
            locations.push((address, local));
        }
        Ok(locations.into())
    }

    pub(crate) fn resident(
        policy: MlxResidentPolicy<U>,
        layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
    ) -> Result<Self, Error> {
        Ok(Self {
            parameter_locations: Self::parameter_locations(layout, addresses)?,
            inner: Arc::new(Mutex::new(MlxSelectedLayerwisePolicyInner::Resident(
                policy,
            ))),
        })
    }

    pub(crate) fn bounded(
        policy: MlxLayerwisePolicy<U, P>,
        layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
    ) -> Result<Self, Error> {
        let parameter_locations = Self::parameter_locations(layout, addresses)?;
        let local_addresses = (0..layout.len())
            .map(|ordinal| {
                layout
                    .address(ordinal)
                    .expect("validated execution layout covers every ordinal")
            })
            .collect();
        Ok(Self {
            parameter_locations,
            inner: Arc::new(Mutex::new(MlxSelectedLayerwisePolicyInner::Bounded {
                policy,
                local_addresses,
            })),
        })
    }

    pub(crate) fn trim_parameter_conversions(
        &self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        let policy = self.inner.lock().map_err(|_| {
            eredu_core::BackendFailure::from_error(Error::ArchitectureModel(
                "selected layerwise policy lock was poisoned".into(),
            ))
        })?;
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.trim_parameter_conversions()
            }
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.residency_manager().trim_parameter_conversions()
            }
        }
    }

    pub(crate) fn residency_report(&self) -> Result<ResidencyReport, Error> {
        let policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy.residency_report(),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => policy.residency_report(),
        }
    }

    pub(crate) fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        let policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(_) => Ok(None),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => policy.dense_stream_report(),
        }
    }
}

pub(crate) enum MlxSelectedUnitLease<U: 'static> {
    Resident(MlxResidentUnit<U>),
    Bounded(MlxUnitLease<U>),
}

impl<U: 'static> Deref for MlxSelectedUnitLease<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Resident(lease) => lease,
            Self::Bounded(lease) => lease,
        }
    }
}

impl<U: 'static> DerefMut for MlxSelectedUnitLease<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Resident(lease) => lease,
            Self::Bounded(lease) => lease,
        }
    }
}

impl<U, P> eredu_runtime::LayerwisePolicy<MlxNeuralBackend, U> for MlxSelectedLayerwisePolicy<U, P>
where
    U: eredu_nn::Parameterized<MlxTensor> + 'static,
    P: MlxUnitPopulator<U>,
{
    type Lease = MlxSelectedUnitLease<U>;
    type Error = Error;

    fn permits_deferred_serial_groups(&self) -> bool {
        let Ok(policy) = self.inner.lock() else {
            return false;
        };
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.permits_deferred_serial_groups()
            }
            MlxSelectedLayerwisePolicyInner::Bounded { .. } => false,
        }
    }

    fn resident_parameters_available(&self) -> bool {
        let Ok(policy) = self.inner.lock() else {
            return false;
        };
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.resident_parameters_available()
            }
            MlxSelectedLayerwisePolicyInner::Bounded { .. } => false,
        }
    }

    fn visit_resident_units(&mut self, visitor: &mut impl FnMut(&mut U)) -> bool {
        let Ok(mut policy) = self.inner.lock() else {
            return false;
        };
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.visit_resident_units(visitor)
            }
            MlxSelectedLayerwisePolicyInner::Bounded { .. } => false,
        }
    }

    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, MlxTensor>,
        active: bool,
    ) -> Result<bool, Error> {
        let mut policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.publish_parameter_replacements(values, active)
            }
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.publish_parameter_replacements(values, active)
            }
        }
    }

    fn begin(&mut self, initial: &MlxTensor, context: &Stream) -> Result<(), Self::Error> {
        let mut policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy.begin(initial, context),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.begin(initial, context)
            }
        }
    }

    fn inspect_unit<E, F, V>(
        &mut self,
        ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
        build: F,
        operation: V,
        context: &Stream,
    ) -> Result<bool, eredu_runtime::LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&Stream) -> Result<U, E>,
        V: FnOnce(&mut U) -> Result<(), Self::Error>,
    {
        let &(expected, local) = self.parameter_locations.get(ordinal).ok_or_else(|| {
            eredu_runtime::LayerwiseAcquireError::Policy(Error::ArchitectureModel(
                "parameter unit ordinal is outside the selected policy".into(),
            ))
        })?;
        if expected != address {
            return Err(eredu_runtime::LayerwiseAcquireError::Policy(
                Error::ArchitectureModel(
                    "parameter unit address differs from the prepared owner".into(),
                ),
            ));
        }
        let mut policy = self.inner.lock().map_err(|_| {
            eredu_runtime::LayerwiseAcquireError::Policy(Error::ArchitectureModel(
                "selected layerwise policy lock was poisoned".into(),
            ))
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.inspect_unit(ordinal, local, build, operation, context)
            }
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.inspect_unit(ordinal, local, build, operation, context)
            }
        }
    }

    fn abort(
        &mut self,
        active: Option<(usize, eredu_runtime::ExecutionUnitAddress, Self::Lease)>,
        context: &Stream,
    ) {
        let Ok(mut policy) = self.inner.lock() else {
            drop(active);
            return;
        };
        match (&mut *policy, active) {
            (
                MlxSelectedLayerwisePolicyInner::Resident(policy),
                Some((ordinal, address, MlxSelectedUnitLease::Resident(lease))),
            ) => policy.abort(Some((ordinal, address, lease)), context),
            (
                MlxSelectedLayerwisePolicyInner::Bounded {
                    policy,
                    local_addresses,
                },
                Some((ordinal, address, MlxSelectedUnitLease::Bounded(lease))),
            ) => policy.abort(
                Some((
                    ordinal,
                    local_addresses.get(ordinal).copied().unwrap_or(address),
                    lease,
                )),
                context,
            ),
            (MlxSelectedLayerwisePolicyInner::Resident(policy), None) => {
                policy.abort(None, context)
            }
            (MlxSelectedLayerwisePolicyInner::Bounded { policy, .. }, None) => {
                policy.abort(None, context)
            }
            (MlxSelectedLayerwisePolicyInner::Resident(policy), Some(_)) => {
                policy.abort(None, context)
            }
            (MlxSelectedLayerwisePolicyInner::Bounded { policy, .. }, Some(_)) => {
                policy.abort(None, context)
            }
        }
    }

    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
        build: F,
        context: &Stream,
    ) -> Result<Self::Lease, eredu_runtime::LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&Stream) -> Result<U, E>,
    {
        let &(expected, local) = self.parameter_locations.get(ordinal).ok_or_else(|| {
            eredu_runtime::LayerwiseAcquireError::Policy(Error::ArchitectureModel(
                "execution unit ordinal is outside the selected policy".into(),
            ))
        })?;
        if expected != address {
            return Err(eredu_runtime::LayerwiseAcquireError::Policy(Error::ArchitectureModel(
                "execution unit address differs from the prepared owner".into(),
            )));
        }
        let mut policy = self.inner.lock().map_err(|_| {
            eredu_runtime::LayerwiseAcquireError::Policy(Error::ArchitectureModel(
                "selected layerwise policy lock was poisoned".into(),
            ))
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy
                .acquire(ordinal, local, build, context)
                .map(MlxSelectedUnitLease::Resident),
            MlxSelectedLayerwisePolicyInner::Bounded {
                policy,
                local_addresses: _,
            } => {
                policy.acquire(ordinal, local, build, context).map(|lease| {
                    #[cfg(test)]
                    crate::tests::support::path_instrumentation::bounded_unit_acquisition();
                    MlxSelectedUnitLease::Bounded(lease)
                })
            }
        }
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a MlxTensor,
        state_values: StateValues,
        context_values: ContextValues,
        context: &Stream,
    ) -> Result<(), Self::Error>
    where
        MlxTensor: 'a,
        StateValues: Iterator<Item = &'a MlxTensor>,
        ContextValues: Iterator<Item = &'a MlxTensor>,
    {
        let mut policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match (&mut *policy, lease) {
            (
                MlxSelectedLayerwisePolicyInner::Resident(policy),
                MlxSelectedUnitLease::Resident(lease),
            ) => policy.complete(
                ordinal,
                address,
                lease,
                output,
                state_values,
                context_values,
                context,
            ),
            (
                MlxSelectedLayerwisePolicyInner::Bounded {
                    policy,
                    local_addresses,
                },
                MlxSelectedUnitLease::Bounded(lease),
            ) => {
                let local = local_addresses.get(ordinal).copied().ok_or_else(|| {
                    Error::ArchitectureModel(format!(
                        "bounded partition unit ordinal {ordinal} has no local slot"
                    ))
                })?;
                if local.group() != address.group() {
                    return Err(Error::ArchitectureModel(format!(
                        "bounded partition unit group {} differs from local group {}",
                        address.group(),
                        local.group()
                    )));
                }
                policy.complete(
                    ordinal,
                    local,
                    lease,
                    output,
                    state_values,
                    context_values,
                    context,
                )
            }
            _ => Err(Error::ArchitectureModel(
                "selected layerwise policy received a lease from another residency".into(),
            )),
        }
    }

    fn finish(&mut self, output: &MlxTensor, context: &Stream) -> Result<(), Self::Error> {
        let mut policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy.finish(output, context),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.finish(output, context)
            }
        }
    }
}
