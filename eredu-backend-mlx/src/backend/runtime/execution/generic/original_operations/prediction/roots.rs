//! Final module dependencies, prepared before the invocation's native Scope.
use super::*;
use eredu_architectures::prediction_extension::PreparedPredictionInvocationRoots;
use safemlx::{Array, PreparedArrayClone, PreparedNestedRoots};

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct RootFailure {
    #[source]
    cause: safemlx::error::Exception,
    funding: WorkspaceMetadataFunding,
}

pub(super) struct ModuleRoots {
    values: Vec<MlxTensor>,
    clones: Vec<PreparedArrayClone>,
    equation_limit: usize,
    validation_limit: usize,
    spent: bool,
    funding: WorkspaceMetadataFunding,
}
impl ModuleRoots {
    pub(super) fn control_bytes(equation: usize, validations: usize) -> Option<usize> {
        let count = equation.checked_add(validations)?;
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Vec<MlxTensor>, eredu_nn::Error>>(),
            size_of::<PreparedArrayClone>(),
            size_of::<Result<PreparedArrayClone, safemlx::PreparedArrayCloneCause>>(),
            size_of::<Option<eredu_nn::Error>>(),
            eredu_nn::Error::retained_source_control_bytes::<RootFailure>()?,
            size_of::<safemlx::OriginalScopeObserver>(),
            size_of::<Result<(), safemlx::error::Exception>>(),
            size_of::<(&mut Self, &mut dyn FnMut(&mut dyn FnMut(&MlxTensor)))>(),
            Layout::array::<MlxTensor>(count).ok()?.size(),
            Layout::array::<PreparedArrayClone>(count).ok()?.size(),
            count.checked_mul(
                PreparedArrayClone::control_bytes()?
                    .checked_add(Array::inspection_clone_handle_bytes())?,
            )?,
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    pub(super) fn prepare(
        equation: usize,
        validations: usize,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, Error> {
        let count = equation.checked_add(validations).ok_or_else(overflow)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|cause| planned_error(cause, funding))?;
        let mut clones = Vec::new();
        clones
            .try_reserve_exact(count)
            .map_err(|cause| planned_error(cause, funding))?;
        for _ in 0..count {
            clones.push(
                PreparedArrayClone::try_prepare_for_inspection()
                    .map_err(|cause| planned_error(cause, funding))?,
            );
        }
        Ok(Self {
            values,
            clones,
            equation_limit: equation,
            validation_limit: validations,
            spent: false,
            funding: funding.clone(),
        })
    }
    /// Copies validation handles into the same paid destination. The concrete
    /// helper exposes neither a TLS loan nor a caller callback.
    pub(super) fn append_validations(
        &mut self,
        values: &mut Vec<MlxTensor>,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<(), Error> {
        crate::backend::nn::tensor::append_active_token_validation_copies(
            &mut self.clones[self.equation_limit..],
            values,
            self.equation_limit
                .checked_add(self.validation_limit)
                .ok_or_else(overflow)?,
            observer,
        )
        .map_err(Error::from)
    }
    pub(super) fn recover_prefix(&mut self, values: &mut Vec<MlxTensor>) -> Result<(), Error> {
        if self.values.capacity() != 0 {
            if !values.is_empty() {
                return Err(identity());
            }
            std::mem::swap(values, &mut self.values);
        }
        Ok(())
    }
}
impl PreparedPredictionInvocationRoots<MlxTensor> for ModuleRoots {
    fn controls(&self, bytes: usize) -> Result<(), eredu_nn::Error> {
        self.funding.reserve_metadata(bytes)
            .map_err(|cause| eredu_nn::workspace::WorkspaceMetadataError::Funding(cause).into())
    }
    fn retain(
        &mut self,
        visit: &mut dyn FnMut(&mut dyn FnMut(&MlxTensor)),
    ) -> Result<Vec<MlxTensor>, eredu_nn::Error> {
        if self.spent {
            return Err(eredu_nn::Error::from(
                eredu_nn::workspace::WorkspaceMetadataError::Unqualified,
            ));
        }
        self.spent = true;
        let observer = safemlx::OriginalScopeObserver::require_current().map_err(|cause| {
            eredu_nn::Error::backend_retained_source(RootFailure {
                cause,
                funding: self.funding.clone(),
            })
        })?;
        let mut failure = None;
        visit(&mut |value| {
            if failure.is_some() {
                return;
            }
            let index = self.values.len();
            if index == self.equation_limit {
                failure = Some(eredu_nn::Error::from(
                    eredu_nn::workspace::WorkspaceMetadataError::Unqualified,
                ));
                return;
            }
            match self.clones[index].fill_in_original_scope(value.as_ref(), &observer) {
                Ok(value) => self.values.push(MlxTensor::from(value)),
                Err(cause) => {
                    failure = Some(eredu_nn::Error::backend_retained_source(RootFailure {
                        cause,
                        funding: self.funding.clone(),
                    }))
                }
            }
        });
        match failure {
            Some(cause) => Err(cause),
            None => Ok(std::mem::take(&mut self.values)),
        }
    }
}

pub(super) type NestedRoots = PreparedNestedRoots<OriginalOperationMetadataCustody>;
