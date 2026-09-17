//! External request startup for the same immutable tensor/copy provider.
use super::*;
use eredu_runtime::working_memory::{
    OriginalExternalSpeculativeSource, OriginalExternalSpeculativeStartup,
};

// The prepared stream and copy environment retire before this host owner. The
// actual External constructor account outlives every provider and escaped copy.
struct ExternalCopyCustody {
    _startup: OriginalExternalSpeculativeStartup,
    _funding: WorkspaceMetadataFunding,
}

impl OriginalEmbeddedCachePreparation {
    /// Constructs only the shared tensor service from an actual External startup.
    /// Embedded lane/target consumers remain unavailable because their one-use
    /// slots are empty. No target or assistant equation authority is issued.
    pub(crate) fn new_external(
        startup: OriginalExternalSpeculativeStartup,
        preparation: &OriginalSpeculativeSemanticPreparation,
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Self, Error> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<Self, StartupCause>>(),
            size_of::<OriginalPredictionCopyContext>(),
            size_of::<Result<OriginalPredictionCopyContext, StartupCause>>(),
            size_of::<ExternalCopyCustody>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<OriginalExternalSpeculativeStartup>(),
            size_of::<(&OriginalSpeculativeSemanticPreparation,
                &OriginalSpeculativeNumericalSources, &OriginalCopyEnvironment<'_>)>(),
            size_of::<(&PrefillRootsRuntime, MlxMetalWorkspaceMechanisms)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            HostPreparationAuthority::retention_bytes::<ExternalCopyCustody>()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            OriginalPredictionStartupContext::copy_error_control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        ];
        preparation.metadata_funding().reserve_metadata(parts.into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let result = (|| -> Result<Self, StartupCause> {
            preparation.validate(sources.pool(), sources.request().execution_identity())
                .map_err(memory)?;
            sources.validate_environment(environment)?;
            if startup.source() != OriginalExternalSpeculativeSource::Assistant
                || !startup.belongs_to_request(sources.request())
            {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            let host = HostPreparationAuthority::retain(ExternalCopyCustody {
                _startup: startup,
                _funding: preparation.metadata_funding().clone(),
            });
            let (roots, mechanisms) = sources.numerical_prerequisites();
            let copy = OriginalPredictionCopyContext::prepare(environment, roots,
                mechanisms, preparation, sources.target_origin(), &host)?;
            // The exact retained environment/stream is checked by the common
            // provider before every copy. These slots cannot fabricate an
            // Embedded lane or a completed target from External source facts.
            Ok(Self {
                lane: RefCell::new(None),
                target: RefCell::new(None),
                copy,
                identity: sources.request().source_identity(),
                _host: host,
            })
        })();
        result.map_err(|cause| OriginalPredictionStartupContext::failure(preparation, cause))
    }
}
