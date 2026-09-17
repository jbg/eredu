//! Prepared forbidden-controller source and destination hooks. The ordinary
//! runtime remains the owner of tool policy and grammar activation semantics.
use super::*;
use eredu_core::speculative::ForbiddenControllerMutation;

impl PreparedForbiddenSource {
    fn into_controller(self) -> ConstraintController {
        let Self {
            inputs,
            history,
            validity,
            prefix,
            authority,
            funding,
        } = self;
        let controller = ConstraintController {
            runtime: ConstraintRuntime::PreparedForbidden {
                inputs,
                pending: prefix,
                original: None,
            },
            committed_tokens: history,
            validity,
            authority,
        };
        drop(funding); // actual buffers/history and controller retain the same account
        controller
    }
}
impl ConstraintController {
    /// Builds an independent bounded canonical destination from the actual
    /// forbidden source. Plain, Auto and Active are not accepted by this worker.
    pub(crate) fn prepare_forbidden_controller(
        &self,
        capacity: usize,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, ForbiddenSourceError> {
        // This outer move/publication uses no new Arc or payload allocation.
        let parts = [
            size_of::<Self>(),
            size_of::<ConstraintRuntime>(),
            size_of::<Result<Self, ForbiddenSourceError>>(),
            size_of::<PreparedForbiddenSource>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| ForbiddenSourceError {
                cause: Cause::Overflow,
                funding: funding.clone(),
            })?;
        funding
            .reserve_metadata(bytes)
            .map_err(|cause| ForbiddenSourceError {
                cause: Cause::Funding(cause),
                funding: funding.clone(),
            })?;
        self.prepare_forbidden_source_with_capacity(funding, capacity)
            .map(PreparedForbiddenSource::into_controller)
    }
    /// The backend only runs the neutral original input compiler; all trigger
    /// policy and current history are read from this actual facade controller.
    pub(crate) fn prepare_original_forbidden_controller<
        B: eredu_runtime::working_memory::OriginalChatBackend,
    >(
        &self,
        runtime: &eredu_core::ModelRuntime<B>,
        capacity: usize,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, ForbiddenSourceError> {
        self.prepare_original_forbidden_controller_with(capacity, funding, |plan| {
            B::compile_original_forbidden_source(runtime, plan)
        })
    }
    pub(super) fn prepare_original_forbidden_controller_with<F>(
        &self,
        capacity: usize,
        funding: &WorkspaceMetadataFunding,
        compile: F,
    ) -> Result<Self, ForbiddenSourceError>
    where
        F: FnOnce(
            eredu_core::speculative::PreparedForbiddenInputCopy<'_>,
        ) -> Result<
            eredu_runtime::working_memory::OriginalForbiddenSource,
            eredu_runtime::working_memory::OriginalForbiddenSourceError,
        >,
    {
        let retain = |cause| ForbiddenSourceError {
            cause,
            funding: funding.clone(),
        };
        let parts = [
            controls().ok_or_else(|| retain(Cause::Overflow))?,
            size_of::<F>(),
            size_of::<Self>(),
            size_of::<ConstraintRuntime>(),
            size_of::<Result<Self, ForbiddenSourceError>>(),
            size_of::<eredu_core::speculative::PreparedForbiddenInputCopy<'_>>(),
            size_of::<
                Result<
                    eredu_core::speculative::PreparedForbiddenInputCopy<'_>,
                    ForbiddenControllerError,
                >,
            >(),
            size_of::<eredu_runtime::working_memory::OriginalForbiddenSource>(),
            size_of::<
                Result<
                    eredu_runtime::working_memory::OriginalForbiddenSource,
                    eredu_runtime::working_memory::OriginalForbiddenSourceError,
                >,
            >(),
            ForbiddenControllerInputs::operation_control_bytes()
                .ok_or_else(|| retain(Cause::Overflow))?,
        ];
        funding
            .reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or_else(|| retain(Cause::Overflow))?,
            )
            .map_err(|cause| retain(Cause::Funding(cause)))?;
        let result = (|| {
            let ConstraintRuntime::Forbidden {
                vocabulary,
                trigger,
                pending,
            } = &self.runtime
            else {
                return Err(Cause::Source);
            };
            if capacity < self.committed_tokens.len() {
                return Err(Cause::Source);
            }
            let (layout, packed) = vocabulary.source();
            let plan = eredu_core::speculative::PreparedForbiddenInputCopy::new(
                packed,
                layout.len(),
                layout.maximum(),
                trigger,
            )?;
            let original = compile(plan)?;
            let bytes = PlainControllerHistory::copy_metadata_bytes(capacity)
                .and_then(|n| {
                    n.checked_add(HostPreparationAuthority::retention_bytes::<
                        WorkspaceMetadataFunding,
                    >()?)
                })
                .ok_or(Cause::Overflow)?;
            funding.reserve_metadata(bytes)?;
            let authority = HostPreparationAuthority::retain(funding.clone());
            let committed_tokens = self
                .committed_tokens
                .copy_prepared(capacity, authority.clone())?;
            Ok(Self {
                runtime: ConstraintRuntime::PreparedForbidden {
                    inputs: original.inputs().clone(),
                    pending: *pending,
                    original: Some(original),
                },
                committed_tokens,
                validity: self.validity.clone(),
                authority,
            })
        })();
        result.map_err(retain)
    }
    pub(in super::super) fn original_forbidden_source(
        &self,
    ) -> Option<&eredu_runtime::working_memory::OriginalForbiddenSource> {
        let ConstraintRuntime::PreparedForbidden { original, .. } = &self.runtime else {
            return None;
        };
        original.as_ref()
    }
    pub(in super::super) fn forbidden_source(&self) -> Option<ForbiddenControllerSource<'_>> {
        let ConstraintRuntime::PreparedForbidden {
            inputs,
            pending,
            original,
        } = &self.runtime
        else {
            return None;
        };
        let source = ForbiddenControllerSource::new(
            &self.committed_tokens,
            &self.validity,
            inputs,
            *pending,
        )
        .ok()?;
        Some(match original {
            Some(original) => {
                source.with_original_storage(eredu_core::OriginalTokenDomainWitness::new(original))
            }
            None => source,
        })
    }
    pub(in super::super) fn forbidden_copy_bytes(&self, capacity: usize) -> Option<usize> {
        let source = self.forbidden_source()?;
        if capacity < source.history().len() {
            return None;
        }
        let parts = [
            PlainControllerHistory::copy_metadata_bytes(capacity)?,
            ForbiddenControllerInputs::operation_control_bytes()?,
            size_of::<Self>(),
            size_of::<ConstraintRuntime>(),
            size_of::<SharedTokenFilter>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<ForbiddenControllerInputs>(),
            size_of::<Option<eredu_runtime::working_memory::OriginalForbiddenSource>>(),
            size_of::<Result<Self, ForbiddenControllerError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(in super::super) fn copy_forbidden(
        &self,
        capacity: usize,
        host: HostPreparationAuthority,
    ) -> Result<Self, ForbiddenControllerError> {
        let source = self
            .forbidden_source()
            .ok_or(ForbiddenControllerError::Unknown)?;
        let committed_tokens = self
            .committed_tokens
            .copy_prepared(capacity, host.clone())?;
        Ok(Self {
            runtime: ConstraintRuntime::PreparedForbidden {
                inputs: source.inputs().clone(),
                pending: source.prefix(),
                original: self.original_forbidden_source().cloned(),
            },
            committed_tokens,
            validity: self.validity.clone(),
            authority: host,
        })
    }
    pub(in super::super) fn forbidden_mutation(
        &mut self,
    ) -> Result<ForbiddenControllerMutation<'_>, ForbiddenControllerError> {
        let ConstraintRuntime::PreparedForbidden {
            inputs, pending, ..
        } = &mut self.runtime
        else {
            return Err(ForbiddenControllerError::Unknown);
        };
        ForbiddenControllerMutation::new(
            &mut self.committed_tokens,
            &self.validity,
            inputs,
            pending,
        )
    }
}
/// Formatting belongs only to the ordinary facade callback. Prepared hooks
/// preserve ForbiddenControllerError unchanged and never enter this formatter.
pub(in super::super) fn ordinary_error(
    cause: ForbiddenControllerError,
) -> crate::api::ConstraintError {
    match cause {
        ForbiddenControllerError::Forbidden(_) => crate::api::ConstraintError::fixed(
            "token would emit a tool-call activation trigger while tool_choice is None",
        ),
        other => crate::api::ConstraintError::new(other.to_string()),
    }
}
