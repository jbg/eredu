//! Borrowed prerequisites for one standalone parameter loan.
use crate::backend::OriginalCopyEnvironment;
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::InferenceExecutionIdentity;

/// Native parameter preparation forwarded by the shared portable loan driver.
/// Its fields are private: callers cannot infer this source from a stream value
/// or attach unrelated capacity. It grants no text execution authority.
pub struct MlxParameterPreparation<'a> {
    pub(crate) environment: &'a OriginalCopyEnvironment<'a>,
    pub(crate) funding: &'a HostMetadataFunding,
    pub(crate) execution: &'a InferenceExecutionIdentity,
    pub(crate) parameters: &'a [Option<&'a str>],
    pub(crate) mechanism: crate::backend::nn::workspace::ResidentExecutionMechanisms,
    pub(crate) sources:
        &'a std::cell::RefCell<Vec<crate::backend::nn::workspace::CompletedParameterSource>>,
}

impl MlxParameterPreparation<'_> {
    pub(super) fn retain_host_source(
        &self,
        receipt: crate::backend::runtime::residency::storage::RetainedAllocationReceipt<'_>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), crate::backend::Error> {
        let mut sources = self
            .sources
            .try_borrow_mut()
            .map_err(|_| crate::backend::Error::PrefillScopeReentrant)?;
        context
            .reserve_metadata_vec(&mut sources, 1)
            .map_err(crate::backend::Error::Neural)?;
        sources
            .push(crate::backend::nn::workspace::CompletedParameterSource::Host(receipt.retain()));
        Ok(())
    }
    pub(crate) fn retain_source(
        &self,
        source: crate::backend::submission_recovery::native_role::physical::CompletedNumericalSource,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), crate::backend::Error> {
        let mut sources = self
            .sources
            .try_borrow_mut()
            .map_err(|_| crate::backend::Error::PrefillScopeReentrant)?;
        context.charge_metadata(std::mem::size_of::<(
            std::slice::Iter<'_, crate::backend::nn::workspace::CompletedParameterSource>,
            &crate::backend::submission_recovery::native_role::physical::CompletedNumericalSource,
            bool,
        )>()).map_err(|cause| crate::backend::Error::Neural(cause.into()))?;
        if sources.iter().any(|previous| {
            matches!(previous, crate::backend::nn::workspace::CompletedParameterSource::Numerical(previous)
                if previous.account().same_account(source.account()) && previous.budget().same_budget(source.budget()))
        }) {
            return Ok(());
        }
        context
            .reserve_metadata_vec(&mut sources, 1)
            .map_err(crate::backend::Error::Neural)?;
        sources.push(source.into());
        Ok(())
    }
    /// Exact borrowed IDs chosen by the same query/overlay source producer.
    /// This view allocates no map and grants no residency or copy permission.
    pub(crate) fn selected_parameters(&self) -> impl Iterator<Item = &str> {
        self.parameters.iter().filter_map(|value| *value)
    }

    pub(crate) fn same_descriptor(
        context: &eredu_nn::workspace::WorkspaceContext,
        a: &crate::MlxTensor,
        b: &crate::MlxTensor,
    ) -> Result<bool, eredu_nn::Error> {
        context.charge_metadata(
            safemlx::Array::descriptor_comparison_control_bytes()
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        a.as_array()
            .try_descriptor()
            .map_err(|cause| context.metadata_source(cause))?
            .same_descriptor(b.as_array())
            .map_err(|cause| context.metadata_source(cause))
    }

    pub(crate) fn publication_failure(
        context: &eredu_nn::workspace::WorkspaceContext,
        cause: eredu_runtime::parameter_operations::ParameterPublicationError<
            crate::backend::Error,
        >,
    ) -> crate::backend::Error {
        use crate::backend::Error;
        use eredu_runtime::parameter_operations::{
            ParameterPublicationError, ParameterPublicationFailure,
        };
        match cause {
            ParameterPublicationError::Participant(cause) => cause,
            ParameterPublicationError::Value(cause) => Error::Neural(cause),
            ParameterPublicationError::Contract(ParameterPublicationFailure::Metadata(cause)) => {
                Error::Neural(cause.into())
            }
            ParameterPublicationError::Contract(cause) => {
                Error::Neural(context.metadata_source(cause))
            }
        }
    }

    pub(crate) fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>(),
            size_of::<Option<&Self>>(),
            size_of::<(&Self, &safemlx::Stream)>(),
            OriginalCopyEnvironment::control_bytes()?,
            size_of::<InferenceExecutionIdentity>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Result<(), crate::backend::Error>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

mod selected;
