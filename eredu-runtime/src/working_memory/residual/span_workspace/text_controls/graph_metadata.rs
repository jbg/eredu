//! One original enforced graph-metadata component, with no population guess.
use super::*;
use crate::working_memory::{InferenceTextPreparation, funding::RawSpanHostOwner};
use std::num::NonZeroU64;

/// Cold provider facts for one shared enforced graph metadata arena. The selected
/// allocation and optional requested ceiling are distinct from producer-fit evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphMetadataFacts {
    capacity: NonZeroU64,
    requested_ceiling: Option<NonZeroU64>,
    provider_bytes: u64,
}
impl GraphMetadataFacts {
    /// Native mechanisms report their actual complete arena/control allocation.
    /// This fact grants neither an allocation nor additional request authority.
    pub fn new(capacity: NonZeroU64, provider_bytes: u64) -> Result<Self, WorkingMemoryError> {
        Self::within_ceiling(capacity, capacity, provider_bytes)
    }
    /// Selects a derived allocation within the unchanged original policy ceiling.
    /// These scalar facts do not themselves certify complete producer coverage.
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
    /// Original optional policy ceiling, retained for preparation identity.
    pub fn requested_ceiling(self) -> Option<NonZeroU64> {
        self.requested_ceiling
    }
    /// Actual selected capacity, no larger than the original requested ceiling.
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
pub struct OriginalGraphMetadata {
    facts: GraphMetadataFacts,
    _raw: RawSpanHostOwner,
}
impl OriginalGraphMetadata {
    /// Read-only exact facts, already included in the original Q contribution.
    pub fn facts(&self) -> GraphMetadataFacts {
        self.facts
    }
}

impl PreparedTextControlWorkspace {
    /// Add one shared physical component before sealing. No role count, output
    /// count, later reserve or refill is involved.
    pub fn with_graph_metadata(
        mut self,
        facts: GraphMetadataFacts,
    ) -> Result<Self, WorkingMemoryError> {
        cold_controls::<(Self, GraphMetadataFacts, Result<Self, WorkingMemoryError>)>(
            self.plan.metadata_funding().as_ref(),
        )?;
        if self.binding.graph_metadata.is_some() {
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
        self.binding.graph_metadata = Some(facts);
        Ok(self)
    }
}
impl OwnedTextSpanWorkspace {
    /// Move the exact accepted graph metadata contribution once. This checks the real
    /// private Preparing Arc while it is untouched. It does not bind a core run,
    /// issue a prompt/sampling stage, or authorize any native Scope.
    pub fn take_graph_metadata(
        &mut self,
        preparation: &InferenceTextPreparation,
    ) -> Result<Option<OriginalGraphMetadata>, WorkingMemoryError> {
        let binding = self
            .workspace()
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let Some(facts) = binding.graph_metadata else {
            return Ok(None);
        };
        if self.graph_taken {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        let reservation = preparation.request().memory_reservation();
        if !reservation.0.same(&self.reservation().0)
            || preparation.request().geometry() != binding.geometry
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        preparation.validate_graph_preparation(facts.requested_ceiling)?;
        self.publish_graph_metadata(facts)
    }
    pub(super) fn take_extension_graph_metadata(
        &mut self,
        origin: &crate::working_memory::text_preparation::SamplingExtensionBinding,
    ) -> Result<Option<OriginalGraphMetadata>, WorkingMemoryError> {
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
        let Some(facts) = binding.graph_metadata else {
            return Ok(None);
        };
        self.publish_graph_metadata(facts)
    }
    fn publish_graph_metadata(
        &mut self,
        facts: GraphMetadataFacts,
    ) -> Result<Option<OriginalGraphMetadata>, WorkingMemoryError> {
        if self.graph_taken {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        self.controls.validate_reservation(self.reservation())?;
        let original = OriginalGraphMetadata {
            facts,
            _raw: self.controls.custody.raw().clone(),
        };
        self.graph_taken = true;
        Ok(Some(original))
    }
}
fn control_bytes() -> Option<usize> {
    [
        size_of::<GraphMetadataFacts>(),
        size_of::<OriginalGraphMetadata>(),
        size_of::<Option<OriginalGraphMetadata>>(),
        size_of::<Result<Option<OriginalGraphMetadata>, WorkingMemoryError>>(),
        size_of::<&InferenceTextPreparation>(),
        size_of::<&WorkingMemoryReservation>(),
        // Two short actual identity/state loans; no callbacks occur under them.
        crate::working_memory::text_preparation::graph_validation_control_bytes()?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
