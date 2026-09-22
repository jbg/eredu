//! One original enforced submission-tracking component, with no population guess.
use super::*;
use crate::working_memory::{InferenceTextPreparation, funding::RawSpanHostOwner};
use std::num::NonZeroU64;

/// Cold provider facts for one shared enforced tracking arena. The requested
/// ceiling and selected allocation remain distinct from graph-fit evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubmissionTrackingFacts {
    capacity: NonZeroU64,
    requested_ceiling: Option<NonZeroU64>,
    provider_bytes: u64,
}
impl SubmissionTrackingFacts {
    /// Native mechanisms report their actual complete arena/control allocation.
    /// This fact grants neither an allocation nor additional request authority.
    pub fn new(capacity: NonZeroU64, provider_bytes: u64) -> Result<Self, WorkingMemoryError> {
        Self::within_ceiling(capacity, capacity, provider_bytes)
    }
    /// Selects an actual arena within the unchanged requested policy ceiling.
    /// The provider must derive the selected size before original admission;
    /// these scalar facts grant no allocation or complete-producer guarantee.
    pub fn within_ceiling(
        requested_ceiling: NonZeroU64,
        capacity: NonZeroU64,
        provider_bytes: u64,
    ) -> Result<Self, WorkingMemoryError> {
        Self::for_policy(Some(requested_ceiling), capacity, provider_bytes)
    }
    /// Retains an optional original ceiling. None selects the provider's fully
    /// derived allocation; it never rewrites the original request configuration.
    /// The complete provider contribution is still charged by normal admission.
    pub fn for_policy(
        requested_ceiling: Option<NonZeroU64>,
        capacity: NonZeroU64,
        provider_bytes: u64,
    ) -> Result<Self, WorkingMemoryError> {
        if requested_ceiling.is_some_and(|ceiling| capacity > ceiling)
            || provider_bytes < capacity.get()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Self {
            capacity,
            requested_ceiling,
            provider_bytes,
        })
    }
    /// Original optional ceiling, retained for exact preparation identity.
    pub fn requested_ceiling(self) -> Option<NonZeroU64> {
        self.requested_ceiling
    }
    /// The actual selected component capacity, no larger than the requested ceiling.
    pub fn capacity(self) -> NonZeroU64 {
        self.capacity
    }
    /// The provider's actual physical arena plus named control representations.
    pub fn provider_bytes(self) -> u64 {
        self.provider_bytes
    }
}

/// One non-Clone Send capsule, constructed only from the original accepted span
/// and its actual untouched preparation. No source/Scope/quote backreference,
/// arbitrary guard attachment or completion operation is exposed.
#[derive(Debug)]
pub struct OriginalSubmissionTracking {
    facts: SubmissionTrackingFacts,
    _raw: RawSpanHostOwner,
}
impl OriginalSubmissionTracking {
    /// Read-only exact facts, already included in the original Q contribution.
    pub fn facts(&self) -> SubmissionTrackingFacts {
        self.facts
    }
}

impl PreparedTextControlWorkspace {
    /// Genuine control-only original Q: no fabricated capture or sequence claim.
    /// The plan and exact execution/request geometry remain the original source
    /// of authority. This cold constructor does not grant allocation permission.
    pub fn prepare_controls(
        geometry: InferenceGeometry,
        plan: &InferenceSpanWorkspacePlan,
        facts: TextHostControlFacts,
    ) -> Result<Self, WorkingMemoryError> {
        geometry
            .validate()
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        if plan.geometry() != geometry {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        facts.total_bytes()?;
        if let Some(funding) = plan.metadata_funding() {
            let shell = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
                .extend(std::alloc::Layout::new::<()>())
                .map_err(|_| WorkingMemoryError::Overflow)?
                .0
                .pad_to_align()
                .size();
            let bytes = [
                shell,
                size_of::<PreparedTextControlWorkspace>(),
                size_of::<TextControlBinding>(),
                size_of::<Result<PreparedTextControlWorkspace, WorkingMemoryError>>(),
                size_of::<TextHostControlFacts>(),
                size_of::<InferenceGeometry>(),
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
            funding
                .reserve_metadata(bytes)
                .map_err(crate::working_memory::reservation_metadata::funding_error)?;
        }
        Ok(Self {
            binding: TextControlBinding {
                identity: Arc::new(()),
                _metadata_funding: plan.metadata_funding(),
                sampling_extension: None,
                source: None,
                sequence: None,
                prefill_paths: None,
                capture_first: None,
                geometry,
                facts,
                preparation_scopes: None,
                prediction_scopes: None,
                prefill_scopes: None,
                tracking: None,
                graph_metadata: None,
                host_destinations: None,
                output_sources: None,
                native_storage: None,
                pins: None,
                publication: None,
                batches: None,
            },
            plan: plan.clone(),
        })
    }
    /// Add one shared physical component before sealing. No role count, output
    /// count, later reserve or refill is involved.
    pub fn with_submission_tracking(
        mut self,
        facts: SubmissionTrackingFacts,
    ) -> Result<Self, WorkingMemoryError> {
        cold_controls::<(
            Self,
            SubmissionTrackingFacts,
            Result<Self, WorkingMemoryError>,
        )>(self.plan.metadata_funding().as_ref())?;
        if self.binding.tracking.is_some() {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        let controls = u64::try_from(control_bytes().ok_or(WorkingMemoryError::Overflow)?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let added = facts
            .provider_bytes
            .checked_add(controls)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.binding.facts.admission = self
            .binding
            .facts
            .admission
            .map(|prior| prior.checked_add(added).ok_or(WorkingMemoryError::Overflow))
            .transpose()?;
        self.binding.tracking = Some(facts);
        Ok(self)
    }
}
impl OwnedTextSpanWorkspace {
    /// Move the exact accepted tracking contribution once. This checks the real
    /// private Preparing Arc while it is untouched. It does not bind a core run,
    /// issue a prompt/sampling stage, or authorize any native Scope.
    pub fn take_submission_tracking(
        &mut self,
        preparation: &InferenceTextPreparation,
    ) -> Result<Option<OriginalSubmissionTracking>, WorkingMemoryError> {
        let binding = self
            .workspace()
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let Some(facts) = binding.tracking else {
            return Ok(None);
        };
        if self.tracking_taken {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        let reservation = preparation.request().memory_reservation();
        if !reservation.0.same(&self.reservation().0)
            || preparation.request().geometry() != binding.geometry
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        preparation.validate_tracking_preparation(facts.requested_ceiling)?;
        self.publish_tracking(facts)
    }
    pub(super) fn take_extension_submission_tracking(
        &mut self,
        origin: &crate::working_memory::text_preparation::SamplingExtensionBinding,
    ) -> Result<Option<OriginalSubmissionTracking>, WorkingMemoryError> {
        origin.validate_pending()?;
        let binding = self
            .workspace()
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !binding
            .sampling_extension
            .as_ref()
            .is_some_and(|expected| expected.same(origin))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let Some(facts) = binding.tracking else {
            return Ok(None);
        };
        self.publish_tracking(facts)
    }
    fn publish_tracking(
        &mut self,
        facts: SubmissionTrackingFacts,
    ) -> Result<Option<OriginalSubmissionTracking>, WorkingMemoryError> {
        if self.tracking_taken {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        self.controls.validate_reservation(self.reservation())?;
        let original = OriginalSubmissionTracking {
            facts,
            _raw: self.controls.custody.raw().clone(),
        };
        self.tracking_taken = true;
        Ok(Some(original))
    }
}
fn control_bytes() -> Option<usize> {
    [
        size_of::<SubmissionTrackingFacts>(),
        size_of::<OriginalSubmissionTracking>(),
        size_of::<Option<OriginalSubmissionTracking>>(),
        size_of::<Result<Option<OriginalSubmissionTracking>, WorkingMemoryError>>(),
        size_of::<&InferenceTextPreparation>(),
        size_of::<&WorkingMemoryReservation>(),
        // Two short actual identity/state loans; no callbacks occur under them.
        crate::working_memory::text_preparation::tracking_validation_control_bytes()?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
