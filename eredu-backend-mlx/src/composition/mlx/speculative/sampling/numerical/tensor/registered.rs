//! Account-only evidence from an actual published copy or its completed view.
use super::*;
use crate::backend::array_copy::{RegisteredArrayCopy, RegisteredArrayCopyCustody};
use eredu_runtime::working_memory::{
    OriginalSpeculativeRequest, OriginalSpeculativeSourceIdentity,
};
use safemlx::{AllocationIdentity, StreamCopyPlan};

/// This is a particular already published allocation, not a registry grant.
/// No Array or native stream owner is retained by evidence. The tensor owner
/// drops its actual Array before this copy account and metadata custody.
#[derive(Clone)]
pub(crate) struct RegisteredTensorSource {
    view_budget: Option<(safemlx::OriginalBufferBudget, OriginalSpeculativeNumericalBudgetCustody)>,
    identity: OriginalSpeculativeSourceIdentity,
    allocation: AllocationIdentity,
    bytes: usize,
    stream: StreamCopyPlan<()>,
    copy: RegisteredArrayCopyCustody,
    funding: WorkspaceMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Metadata(#[from] safemlx::ArrayMetadataError),
    #[error(transparent)]
    Stream(#[from] safemlx::StreamCopyCause),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    _funding: WorkspaceMetadataFunding,
}
fn controls(funding: &WorkspaceMetadataFunding) -> Result<(), Error> {
    let parts = [
        size_of::<RegisteredTensorSource>(),
        size_of::<Option<safemlx::AllocationInfo>>(),
        size_of::<Result<Option<safemlx::AllocationInfo>, safemlx::ArrayMetadataError>>(),
        size_of::<StreamCopyPlan<()>>(),
        size_of::<Result<StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
        size_of::<Cause>(),
        size_of::<Failure>(),
        size_of::<Result<(), Error>>(),
        BackendFailure::source_retention_peak_bytes::<Failure>().ok_or_else(model::overflow)?,
    ];
    funding
        .reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(model::overflow)?,
        )
        .map_err(Error::WorkspacePlanning)
}
fn failure(cause: impl Into<Cause>, funding: &WorkspaceMetadataFunding) -> Error {
    Error::StorageSource(BackendFailure::from_error(Failure {
        cause: cause.into(),
        _funding: funding.clone(),
    }))
}
impl RegisteredTensorSource {
    pub(super) fn with_view_budget(&self,budget:safemlx::OriginalBufferBudget,custody:OriginalSpeculativeNumericalBudgetCustody)->Self {
        let mut source=self.clone();
        source.view_budget=Some((budget,custody));
        source
    }

    fn capture(
        array: &Array,
        copy: &RegisteredArrayCopyCustody,
        identity: OriginalSpeculativeSourceIdentity,
        stream: &Stream,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, Error> {
        controls(funding)?;
        copy.copy_retention()
            .validate_pool(identity.pool())
            .map_err(Error::PrefillControl)?;
        let info = array
            .try_allocation_info()
            .map_err(|cause| failure(cause, funding))?
            .ok_or_else(invalid)?;
        let stream = StreamCopyPlan::capture(stream).map_err(|cause| failure(cause, funding))?;
        funding
            .reserve_metadata(stream.control_bytes().ok_or_else(model::overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        Ok(Self {
            identity,
            allocation: info.identity(),
            bytes: info.bytes(),
            stream,
            copy: copy.clone(),
            view_budget: None,
            funding: funding.clone(),
        })
    }
    pub(in crate::composition::mlx::speculative::sampling::numerical) fn from_value(
        value: &OriginalNumericalValue,
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Self, Error> {
        sources.validate_environment(environment)?;
        controls(sources.metadata_funding())?;
        let stream = StreamCopyPlan::<()>::capture(environment.stream())
            .map_err(|cause| failure(cause, sources.metadata_funding()))?;
        sources
            .metadata_funding()
            .reserve_metadata(
                stream
                    .source_comparison_control_bytes()
                    .ok_or_else(model::overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let value = value.value();
        if !matches!(&value.provenance,Provenance::Registered(source)
            if SpeculativeNumericalSource::Registered(source).belongs_to_request(sources.request()))
            || value.original_budget.is_some()
            || !stream.matches_source(&value.stream)
        {
            return Err(invalid());
        }
        Self::capture(
            &value.array,
            value._copy.as_ref().ok_or_else(invalid)?,
            sources.request().source_identity(),
            environment.stream(),
            sources.metadata_funding(),
        )
    }
    /// Only an actual completed independent-copy result can create fresh copy
    /// evidence. A prior array identity is never reused for the new allocation.
    pub(crate) fn from_copy(
        copy: &RegisteredArrayCopy,
        identity: OriginalSpeculativeSourceIdentity,
        stream: &Stream,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, Error> {
        copy.validate_completed_stream(stream, funding)?;
        Self::capture(copy.array(), copy.custody(), identity, stream, funding)
    }
    pub(crate) fn validate(
        &self,
        request: &OriginalSpeculativeRequest,
        stream: &Stream,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<(), Error> {
        controls(funding)?;
        funding
            .reserve_metadata(
                self.stream
                    .source_comparison_control_bytes()
                    .ok_or_else(model::overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        if !self.identity.belongs_to_request(request) {
            return Err(invalid());
        }
        self.validate_stream(stream, funding)
    }
    /// Descriptive completed placement only. Request and backing validation
    /// remain separate, and no destination completion can be inferred here.
    pub(crate) fn matches_completed_stream(&self, stream: &Stream,
        funding: &WorkspaceMetadataFunding) -> Result<bool, Error> {
        controls(funding)?;
        let parts = [size_of::<(&Self, &Stream, &WorkspaceMetadataFunding)>(),
            size_of::<Result<bool, Error>>(),
            self.stream.source_comparison_control_bytes().ok_or_else(model::overflow)?];
        funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(model::overflow)?).map_err(Error::WorkspacePlanning)?;
        Ok(self.stream.matches_source(stream))
    }
    pub(crate) fn validate_stream(
        &self,
        stream: &Stream,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<(), Error> {
        controls(funding)?;
        funding
            .reserve_metadata(
                self.stream
                    .source_comparison_control_bytes()
                    .ok_or_else(model::overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        if !self.stream.matches_source(stream) {
            return Err(invalid());
        }
        self.copy
            .copy_retention()
            .validate_pool(self.identity.pool())
            .map_err(Error::PrefillControl)
    }
    /// Exact descriptive lookup; a different backing is not a failed source.
    pub(crate) fn matches_array(&self,array:&Array,funding:&WorkspaceMetadataFunding)->Result<bool,Error>{
        controls(funding)?;
        let info=array.try_allocation_info().map_err(|cause|failure(cause,funding))?.ok_or_else(invalid)?;
        Ok(info.identity()==self.allocation && info.bytes()==self.bytes)
    }
    pub(crate) fn same_backing(&self,other:&Self)->bool{
        self.allocation==other.allocation && self.bytes==other.bytes
    }
    pub(crate) fn validate_array(
        &self,
        array: &Array,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<(), Error> {
        controls(funding)?;
        let info = array
            .try_allocation_info()
            .map_err(|cause| failure(cause, funding))?
            .ok_or_else(invalid)?;
        if info.identity() != self.allocation || info.bytes() != self.bytes {
            return Err(invalid());
        }
        Ok(())
    }
    /// Historical evidence can outlive some of its tensors. Authenticate its
    /// request/stream first, then check only backing selected by this equation.
    /// The complete input binder still requires authority for every native row.
    pub(crate) fn validate_selected_projection(
        &self,
        native: &crate::backend::nn::workspace::ProjectedNativeStorage,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<(), Error> {
        controls(funding)?;
        funding.reserve_metadata(size_of::<Option<&Array>>())
            .map_err(Error::WorkspacePlanning)?;
        if let Some(array) = native.native_array(self.allocation) {
            self.validate_array(array, funding)?;
        }
        Ok(())
    }
    pub(crate) fn identity(&self) -> &OriginalSpeculativeSourceIdentity {
        &self.identity
    }
    pub(super) fn provenance(
        &self,
        sources: &OriginalSpeculativeNumericalSources,
    ) -> Result<OriginalSpeculativeRegisteredSource, Error> {
        sources
            .request()
            .bind_registered_copy_source(self.copy.copy_retention())
            .map_err(Error::PrefillControl)
    }
}
