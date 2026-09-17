//! One-use startup target copied from an actual settled registered source.
use super::*;

/// The registered copy owns independent backing and its original copy Q/H.
/// No completed model-role account is invented for prior ordinary text state.
/// State and all retained copy metadata retire before the source context's H.
pub(crate) struct OriginalPredictionTarget {
    prepared: PreparedLane<OriginalResidentState>,
    frontier: u64,
    copy: OriginalPredictionCopyContext,
}
impl OriginalPredictionTarget {
    pub(crate) fn frontier(&self) -> u64 {
        self.frontier
    }

    pub(super) fn validate(
        &self,
        preparation: &OriginalSpeculativeSemanticPreparation,
        origin: &ReplicatedTextControlOrigin,
    ) -> Result<(), StartupCause> {
        self.copy.validate(preparation, origin)
    }

    /// The existing shared target-preparation agreement checks the converted
    /// state layout. This consumes the already completed startup copy exactly
    /// once; subsequent cache branches use the unchanged actual copy provider.
    pub(super) fn into_prepared<S: MlxStateMechanisms>(
        self,
        source: &S,
        preparation: &OriginalSpeculativeSemanticPreparation,
        origin: &ReplicatedTextControlOrigin,
    ) -> Result<PreparedLane<S>, StartupCause> {
        let parts = [
            size_of::<Self>(),
            size_of::<(
                &S,
                &OriginalSpeculativeSemanticPreparation,
                &ReplicatedTextControlOrigin,
            )>(),
            size_of::<Result<S, OriginalResidentState>>(),
            size_of::<PreparedLane<S>>(),
            size_of::<Result<PreparedLane<S>, StartupCause>>(),
            size_of::<Result<u64, std::num::TryFromIntError>>(),
        ];
        self.copy
            .reserve(parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
        self.validate(preparation, origin)?;
        if u64::try_from(source.offset()).ok() != Some(self.frontier) {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let value = match S::from_original_resident_copy(self.prepared.value) {
            Ok(value) => value,
            Err(unexpected) => {
                drop(unexpected);
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
        };
        Ok(PreparedLane {
            value,
            host: self.prepared.host,
        })
    }
}
impl OriginalPredictionLane {
    /// The session supplies its actual fixed, idle source loan and exact current
    /// origin. Source pinning and publication use the same independent copy
    /// worker as later target checkpoints, with no optional model-role witness.
    pub(crate) fn prepare_registered_target(
        &self,
        source: PreparedResidentDecoderCopy<'_>,
        preparation: &OriginalSpeculativeSemanticPreparation,
        origin: &ReplicatedTextControlOrigin,
        frontier: u64,
    ) -> Result<OriginalPredictionTarget, StartupCause> {
        let parts = [
            size_of::<OriginalPredictionTarget>(),
            size_of::<Result<OriginalPredictionTarget, StartupCause>>(),
            size_of::<PreparedLane<OriginalResidentState>>(),
            size_of::<OriginalPredictionCopyContext>(),
            size_of::<(
                &Self,
                PreparedResidentDecoderCopy<'_>,
                &OriginalSpeculativeSemanticPreparation,
                &ReplicatedTextControlOrigin,
                u64,
            )>(),
        ];
        self.copy
            .reserve(parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
        self.copy.validate(preparation, origin)?;
        let prepared = self.copy.target_source(source, None)?;
        Ok(OriginalPredictionTarget {
            prepared,
            frontier,
            copy: self.copy.clone(),
        })
    }
}
