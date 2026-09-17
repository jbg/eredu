//! Descriptive counts of declared parameter owners, never a model byte bound.
#[cfg(test)]
mod tests;
use super::ErasedReplicatedTextExecutable;
use crate::backend::nn::shared::{
    add_parameter_count, visit_parameter_map, ParameterCountError, ParameterSourceCounter,
    ParameterSourceCounts,
};
use crate::MlxTensor;
use eredu_nn::Parameterized;
use eredu_runtime::replicated_session::RuntimeInspectionBoundary;
use safemlx::RuntimeCallGuard;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParameterOwnerRole {
    Static,
    PredictionInner,
    PredictionPlaceholder,
    PredictionReplacement,
    DisplacedOriginal,
    PublishedOverlay,
}
impl ParameterOwnerRole {
    fn index(self) -> usize {
        match self {
            Self::Static => 0,
            Self::PredictionInner => 1,
            Self::PredictionPlaceholder => 2,
            Self::PredictionReplacement => 3,
            Self::DisplacedOriginal => 4,
            Self::PublishedOverlay => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ParameterOwnerSourceError {
    #[error("selected execution has no borrowed static aggregate")]
    StaticUnavailable,
    #[error("selected executable has no parameter owner companion")]
    ExecutableUnavailable,
    #[error("selected prediction has no parameter owner companion")]
    PredictionUnavailable,
    #[error("parameter owner inspection boundary failed")]
    Boundary(#[source] RuntimeInspectionBoundary),
    #[error("parameter owner {role:?} module {module:?} failed")]
    Observation {
        role: ParameterOwnerRole,
        module: Option<usize>,
        #[source]
        source: ParameterCountError,
    },
    #[error("prediction owner occurrence count overflow")]
    OwnerCountOverflow,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParameterOwnerRoleCounts {
    pub(crate) parameters: ParameterSourceCounts,
    pub(crate) map_key_bytes: usize,
}

/// Counts of these declared components only. Prototypes, manager/source storage,
/// active lanes, banks, policy overrides and recovery roots are not covered by
/// the numerical rows. No scalar field grants readiness or submission authority.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParameterOwnerCounts {
    roles: [ParameterOwnerRoleCounts; 6],
    pub(crate) prediction_modules: usize,
    pub(crate) prediction_ids_present: usize,
    pub(crate) prediction_managers_installed: usize,
    pub(crate) pooling_prototypes: usize,
    pub(crate) model_prototypes: usize,
}
impl ParameterOwnerCounts {
    pub(crate) fn role(self, role: ParameterOwnerRole) -> ParameterOwnerRoleCounts {
        self.roles[role.index()]
    }

    pub(crate) fn observe_source<T: Parameterized<MlxTensor> + ?Sized>(
        &mut self,
        role: ParameterOwnerRole,
        module: Option<usize>,
        source: &T,
        guard: &mut RuntimeCallGuard,
    ) -> Result<(), ParameterOwnerSourceError> {
        let mut counter = ParameterSourceCounter::new(guard);
        counter.counts = self.roles[role.index()].parameters;
        counter.observe_source(source).map_err(|source| {
            ParameterOwnerSourceError::Observation {
                role,
                module,
                source,
            }
        })?;
        self.roles[role.index()].parameters = counter.counts;
        Ok(())
    }

    pub(crate) fn observe_map(
        &mut self,
        role: ParameterOwnerRole,
        module: Option<usize>,
        values: &BTreeMap<String, MlxTensor>,
        guard: &mut RuntimeCallGuard,
    ) -> Result<(), ParameterOwnerSourceError> {
        let mut counter = ParameterSourceCounter::new(guard);
        let prior = self.roles[role.index()];
        counter.counts = prior.parameters;
        let mut key_bytes = prior.map_key_bytes;
        visit_parameter_map(values, |key, value| {
            add_parameter_count(&mut key_bytes, key.len())?;
            counter.observe_auxiliary(value)
        })
        .map_err(|source| ParameterOwnerSourceError::Observation {
            role,
            module,
            source,
        })?;
        self.roles[role.index()] = ParameterOwnerRoleCounts {
            parameters: counter.counts,
            map_key_bytes: key_bytes,
        };
        Ok(())
    }

    pub(crate) fn prediction_module(
        &mut self,
        id: bool,
        manager: bool,
    ) -> Result<(), ParameterOwnerSourceError> {
        let mut modules = self.prediction_modules;
        let mut ids = self.prediction_ids_present;
        let mut managers = self.prediction_managers_installed;
        add_owner_count(&mut modules, 1)?;
        add_owner_count(&mut ids, usize::from(id))?;
        add_owner_count(&mut managers, usize::from(manager))?;
        self.prediction_modules = modules;
        self.prediction_ids_present = ids;
        self.prediction_managers_installed = managers;
        Ok(())
    }
    pub(crate) fn pooling_prototype(&mut self) -> Result<(), ParameterOwnerSourceError> {
        add_owner_count(&mut self.pooling_prototypes, 1)
    }
    pub(crate) fn model_prototype(&mut self) -> Result<(), ParameterOwnerSourceError> {
        add_owner_count(&mut self.model_prototypes, 1)
    }
}
fn add_owner_count(total: &mut usize, amount: usize) -> Result<(), ParameterOwnerSourceError> {
    add_parameter_count(total, amount).map_err(|_| ParameterOwnerSourceError::OwnerCountOverflow)
}

/// Borrows the actual erased executable; does not allocate another erased owner.
/// The executable must perform its fixed runtime fence before any observation.
pub(crate) struct NativeParameterOwnerSource<'source> {
    owner: &'source dyn ErasedReplicatedTextExecutable,
}
pub(crate) struct CountedNativeParameterOwnerSource<'source> {
    source: NativeParameterOwnerSource<'source>,
    counts: ParameterOwnerCounts,
}
impl<'source> NativeParameterOwnerSource<'source> {
    pub(crate) fn new(owner: &'source dyn ErasedReplicatedTextExecutable) -> Self {
        Self { owner }
    }
    pub(crate) fn count(
        self,
        guard: &mut RuntimeCallGuard,
    ) -> Result<CountedNativeParameterOwnerSource<'source>, ParameterOwnerSourceError> {
        let counts = self.owner.count_parameter_owners(guard)?;
        Ok(CountedNativeParameterOwnerSource {
            source: self,
            counts,
        })
    }
    pub(crate) fn owner(&self) -> &'source dyn ErasedReplicatedTextExecutable {
        self.owner
    }
}
impl<'source> CountedNativeParameterOwnerSource<'source> {
    pub(crate) const fn counts(&self) -> ParameterOwnerCounts {
        self.counts
    }
    pub(crate) fn source(&self) -> &NativeParameterOwnerSource<'source> {
        &self.source
    }
}
