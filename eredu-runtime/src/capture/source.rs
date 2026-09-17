//! Retain either the legacy Arc or the actual registered shared source owner.
use eredu_core::capture::{AdmittedCapturePlan, SharedCapturePlan};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) enum CapturePlanSource {
    Legacy(Arc<AdmittedCapturePlan>),
    Shared(SharedCapturePlan),
}
impl CapturePlanSource {
    pub(crate) fn shared(&self) -> Option<&SharedCapturePlan> {
        match self {
            Self::Shared(source) => Some(source),
            Self::Legacy(_) => None,
        }
    }
    // Existing legacy ownership tests inspect retirement of the actual raw Arc.
    #[cfg(test)]
    pub(crate) fn legacy_arc(&self) -> Option<&Arc<AdmittedCapturePlan>> {
        match self {
            Self::Legacy(plan) => Some(plan),
            Self::Shared(_) => None,
        }
    }
}
impl std::ops::Deref for CapturePlanSource {
    type Target = AdmittedCapturePlan;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Legacy(plan) => plan,
            Self::Shared(plan) => plan.admission(),
        }
    }
}
impl AsRef<AdmittedCapturePlan> for CapturePlanSource {
    fn as_ref(&self) -> &AdmittedCapturePlan {
        self
    }
}
impl From<Arc<AdmittedCapturePlan>> for CapturePlanSource {
    fn from(plan: Arc<AdmittedCapturePlan>) -> Self {
        Self::Legacy(plan)
    }
}
impl From<SharedCapturePlan> for CapturePlanSource {
    fn from(plan: SharedCapturePlan) -> Self {
        Self::Shared(plan)
    }
}
