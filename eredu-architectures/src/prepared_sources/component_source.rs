//! Original component-source construction from the retained model selection.
use super::*;
use crate::component_partition::{
    ComponentPartitionError, ComponentPartitionLayouts, construction::Destination,
};
use crate::speculative_execution::SpeculativeActivationExecution;
use eredu_core::HostMetadataFunding;

#[cfg(test)]
mod tests;

/// Complete component declarations constructed under their original account.
/// This is descriptive source storage, not capture or native-work authority.
#[derive(Debug)]
pub struct PreparedComponentPartitionSource {
    layouts: ComponentPartitionLayouts,
    funding: Option<HostMetadataFunding>,
}
impl PreparedComponentPartitionSource {
    /// Borrow the exact complete topology and invocation declarations.
    pub fn layouts(&self) -> &ComponentPartitionLayouts {
        &self.layouts
    }
    /// Borrow the original account for enclosing source destinations.
    pub fn metadata_funding(&self) -> Option<&HostMetadataFunding> {
        self.funding.as_ref()
    }
}
/// A failed source constructor retains its actual account through diagnostics.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct ComponentPartitionSourceError {
    #[source]
    cause: ComponentPartitionError,
    _funding: Option<HostMetadataFunding>,
}

/// The original identity or capture-field producer failed before publication.
#[derive(Debug, thiserror::Error)]
pub enum CaptureDiscoverySourceError {
    /// Original catalog/support destination or semantic failure.
    #[error("{0}")]
    Capture(#[from] eredu_core::capture::CaptureError),
    /// Original filesystem identity preparation with its retained account.
    #[error("{0}")]
    Identity(#[from] eredu_core::artifact::ArtifactIdentityPreparationError),
}
impl CaptureDiscoverySourceError {
    /// Fixed funding refusal without an additional diagnostic producer.
    pub fn funding_error(&self) -> Option<eredu_core::HostMetadataFundingError> {
        match self {
            Self::Identity(error) => error.funding_error(),
            Self::Capture(eredu_core::capture::CaptureError::AdmissionStorage(
                eredu_core::capture::CaptureAdmissionStorageError::Funding(error),
            )) => Some(*error),
            _ => None,
        }
    }
}
impl ComponentPartitionSourceError {
    /// Fixed refusal before any diagnostic wrapper allocation.
    pub fn funding_error(&self) -> Option<eredu_core::HostMetadataFundingError> {
        match &self.cause {
            ComponentPartitionError::MetadataFunding(error) => Some(*error),
            ComponentPartitionError::Capture(
                eredu_core::capture::CaptureError::AdmissionStorage(
                    eredu_core::capture::CaptureAdmissionStorageError::Funding(error),
                ),
            ) => Some(*error),
            _ => None,
        }
    }
}

impl PreparedModelDiscovery {
    /// The ordinary and funded source publication boundary. The construction
    /// policy is retained inside the closed result and diagnostic; no existing
    /// caller layout can be adopted by this constructor. The optional actual
    /// prediction execution selects its already bound invocation declarations;
    /// no artifact is reopened and no native resource is created.
    pub fn compile_component_partition_source(
        &self,
        max_ranks: usize,
        execution: Option<&SpeculativeActivationExecution>,
        construction: eredu_core::capture::CaptureSourceConstruction<'_>,
    ) -> Result<Option<PreparedComponentPartitionSource>, ComponentPartitionSourceError> {
        let funding = construction.funding().cloned();
        let allocation = Destination(funding.as_ref());
        let result = (|| {
            allocation.controls::<(
                &Self,
                usize,
                Option<&SpeculativeActivationExecution>,
                PreparedComponentPartitionSource,
                ComponentPartitionSourceError,
                Option<HostMetadataFunding>,
                eredu_core::capture::CaptureSourceConstruction<'_>,
                Destination<'_>,
            )>()?;
            self.component_layouts_worker(max_ranks, execution, allocation)
        })();
        match result {
            Ok(Some(layouts)) => Ok(Some(PreparedComponentPartitionSource { layouts, funding })),
            Ok(None) => Ok(None),
            Err(cause) => Err(ComponentPartitionSourceError {
                cause,
                _funding: funding,
            }),
        }
    }
    pub(super) fn component_layouts_worker(
        &self,
        max_ranks: usize,
        execution: Option<&SpeculativeActivationExecution>,
        allocation: Destination<'_>,
    ) -> Result<Option<ComponentPartitionLayouts>, ComponentPartitionError> {
        allocation.controls::<(
            &Self,
            usize,
            Option<&SpeculativeActivationExecution>,
            Option<ComponentPartitionLayouts>,
        )>()?;
        let Some(selected) = &self.partition_selection else {
            return Ok(None);
        };
        let parameters = self.partition_parameters.as_ref().ok_or_else(|| {
            allocation.capture_invalid(format_args!(
                "component discovery has not been bound to the constructed partition"
            ))
        })?;
        let Some(target) = selected.component_partition_layouts_worker(
            &self.descriptor,
            parameters,
            max_ranks,
            allocation,
        )?
        else {
            return Ok(None);
        };
        let Some(execution) = execution else {
            return Ok(Some(target));
        };
        let prediction = self.prediction.as_ref().ok_or_else(|| {
            allocation.capture_unsupported(format_args!(
                "prepared sources have no selected prediction catalog"
            ))
        })?;
        let placement = prediction.placement.get().ok_or_else(|| {
            allocation.capture_invalid(format_args!(
                "prediction placement has not been bound by materialization"
            ))
        })?;
        target
            .with_prediction_worker(&prediction.descriptor, placement, execution, allocation)
            .map(Some)
    }
}
