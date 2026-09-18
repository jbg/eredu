//! Owned lexical copy-environment loan, using the existing prepared stream copy.
use super::{OriginalCopyEnvironment, OriginalCopyEnvironmentError, RetainedOriginalCopyEnvironment};
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::WorkingMemoryPool;
use safemlx::{PreparedStreamCopy, StreamCopyCause, StreamCopyError, StreamCopyPlan};
use std::{alloc::Layout, mem::{size_of, size_of_val}, sync::atomic::AtomicUsize};

// The same prepared-stream owner retains both the native constructor's host
// authority and the metadata account that paid its actual wrapper. A native
// refusal/quarantine carries this custody too, so it cannot outlive that H.
#[derive(Clone)]
pub(crate) struct CopyEnvironmentCustody {
    _host: HostPreparationAuthority,
    _funding: HostMetadataFunding,
}
impl std::fmt::Debug for CopyEnvironmentCustody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CopyEnvironmentCustody")
    }
}

/// Retains an independent copy of the exact admitted stream wrapper and its
/// immutable prerequisites. It creates no stream registration or model scope.
#[derive(Clone)]
pub(crate) struct PreparedOriginalCopyEnvironment {
    stream: PreparedStreamCopy<CopyEnvironmentCustody>,
    environment: RetainedOriginalCopyEnvironment,
    // Wrapper/source values retire before their exact constructor metadata.
    funding: HostMetadataFunding,
}
impl std::fmt::Debug for PreparedOriginalCopyEnvironment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PreparedOriginalCopyEnvironment")
    }
}

/// Inline query/refusal or the actual failed native wrapper with its custody.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PreparedOriginalCopyEnvironmentError {
    #[error(transparent)]
    Environment(#[from] OriginalCopyEnvironmentError),
    #[error(transparent)]
    Metadata(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Source(#[from] StreamCopyCause),
    #[error(transparent)]
    Stream(#[from] StreamCopyError<CopyEnvironmentCustody>),
}
fn overflow() -> HostMetadataFundingError { HostMetadataFundingError::Overflow }

impl PreparedOriginalCopyEnvironment {
    fn preparation_controls() -> Option<usize> {
        let parts = [
            size_of::<Self>(), size_of::<Result<Self, PreparedOriginalCopyEnvironmentError>>(),
            size_of::<RetainedOriginalCopyEnvironment>(),
            size_of::<Result<RetainedOriginalCopyEnvironment, OriginalCopyEnvironmentError>>(),
            size_of::<StreamCopyPlan<CopyEnvironmentCustody>>(),
            size_of::<Result<StreamCopyPlan<CopyEnvironmentCustody>, StreamCopyCause>>(),
            size_of::<Layout>(), size_of::<Result<(Layout, usize), std::alloc::LayoutError>>(),
            size_of::<Result<PreparedStreamCopy<CopyEnvironmentCustody>, StreamCopyError<CopyEnvironmentCustody>>>(),
            size_of::<(&OriginalCopyEnvironment<'_>, &HostPreparationAuthority, &HostMetadataFunding)>(),
            size_of::<HostMetadataFunding>(), size_of::<CopyEnvironmentCustody>(),
            size_of::<PreparedOriginalCopyEnvironmentError>(),
            size_of::<Option<usize>>(),
            OriginalCopyEnvironment::control_bytes()?,
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn prepare(
        environment: &OriginalCopyEnvironment<'_>,
        host: &HostPreparationAuthority,
        funding: &HostMetadataFunding,
    ) -> Result<Self, PreparedOriginalCopyEnvironmentError> {
        funding.reserve_metadata(Self::preparation_controls().ok_or_else(overflow)?)?;
        let retained = environment.retain_prerequisites()?;
        let plan = StreamCopyPlan::<CopyEnvironmentCustody>::capture(environment.stream())?;
        let shared = Layout::new::<[AtomicUsize; 2]>()
            .extend(plan.shared_body_layout()).map_err(|_| overflow())?
            .0.pad_to_align().size();
        let parts = [
            retained.control_bytes().ok_or_else(overflow)?,
            shared, plan.owner_node_layout().size(), plan.native_wrapper_bytes(),
            plan.control_bytes().ok_or_else(overflow)?,
            plan.source_comparison_control_bytes().ok_or_else(overflow)?,
        ];
        funding.reserve_metadata(parts.into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add).ok_or_else(overflow)?)?;
        let stream = plan.realize(CopyEnvironmentCustody {
            _host: host.clone(), _funding: funding.clone(),
        })?;
        Ok(Self { stream, environment: retained, funding: funding.clone() })
    }
    /// Same scalar source authentication controls as the existing copy context.
    pub(crate) fn control_bytes(&self) -> Option<usize> {
        self.environment.control_bytes()
    }
    pub(crate) fn loan(&self, pool: &WorkingMemoryPool)
        -> Result<OriginalCopyEnvironment<'_>, PreparedOriginalCopyEnvironmentError> {
        let parts = [
            size_of::<(&Self, &WorkingMemoryPool)>(),
            size_of::<Result<OriginalCopyEnvironment<'_>, PreparedOriginalCopyEnvironmentError>>(),
            size_of::<PreparedOriginalCopyEnvironmentError>(),
            self.environment.control_bytes().ok_or_else(overflow)?,
        ];
        self.funding.reserve_metadata(parts.into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add).ok_or_else(overflow)?)?;
        Ok(self.environment.loan(&self.stream, pool)?)
    }
}
