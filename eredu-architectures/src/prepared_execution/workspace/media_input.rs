//! Closed ordinary projection from the actual B input, not arbitrary metadata.
use super::*;
use crate::media_plan::BoundPreparedMediaSemantics;
use eredu_runtime::{
    PreparedModelInput,
    input::{OriginalPreparedInputProjection, PreparedModelInputOwner},
    working_memory::WorkingMemoryUnquotedLease,
};

pub use eredu_runtime::input::PreparedMediaWorkspaceTensor;

/// Source-authenticated ordinary projection. Its private constructor visits only
/// actual native slots of the original B owner; it cannot become a native input.
pub struct OriginalMediaWorkspaceInput {
    pub(super) prepared: PreparedModelInput<WorkspaceTensor>,
    pub(super) original: BoundPreparedMediaSemantics,
    pub(super) tables: OriginalPreparedInputProjection,
    pub(super) source_storage: eredu_runtime::input::OriginalPreparedWorkspaceSource,
    pub(super) ordinary: Option<WorkingMemoryUnquotedLease>,
    // Projected part/identity storage outlives the constructing Context.
    funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}
impl std::fmt::Debug for OriginalMediaWorkspaceInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalMediaWorkspaceInput")
            .field("parts", &self.prepared.len())
            .finish_non_exhaustive()
    }
}
/// An owning ordinary diagnostic failure, retaining A/B/source and actual lease
/// through the error payload/control retirement. No original work was granted.
pub struct OriginalMediaWorkspaceInputError {
    cause: Error,
    tables: Option<OriginalPreparedInputProjection>,
    original: BoundPreparedMediaSemantics,
    ordinary: Option<WorkingMemoryUnquotedLease>,
    _funding: (
        Option<eredu_nn::workspace::HostMetadataFunding>,
        Option<eredu_nn::workspace::HostMetadataFunding>,
    ),
}
impl std::fmt::Debug for OriginalMediaWorkspaceInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalMediaWorkspaceInputError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for OriginalMediaWorkspaceInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for OriginalMediaWorkspaceInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
struct PlanFailure {
    cause: eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
    tables: OriginalPreparedInputProjection,
    ordinary: Option<WorkingMemoryUnquotedLease>,
    _funding: (
        Option<eredu_nn::workspace::HostMetadataFunding>,
        Option<eredu_nn::workspace::HostMetadataFunding>,
    ),
}
impl std::fmt::Debug for PlanFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.cause, f)
    }
}
impl std::fmt::Display for PlanFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for PlanFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl OriginalMediaWorkspaceInput {
    /// The exact completed B projection, retaining accounting independently of
    /// its native tensor owners. No source selection can be replaced here.
    pub fn source_storage(&self) -> &eredu_runtime::input::OriginalPreparedWorkspaceSource {
        &self.source_storage
    }

    fn charge_retained_metadata(&self, bytes: usize) -> Result<(), Error> {
        if let Some(funding) = &self.funding {
            funding
                .reserve_metadata(bytes)
                .map_err(eredu_nn::workspace::WorkspaceMetadataError::Funding)?;
        }
        Ok(())
    }
    pub(crate) fn reject(self, cause: Error) -> Error {
        let Some(bytes) =
            Error::retained_source_construction_bytes::<OriginalMediaWorkspaceInputError>()
        else {
            return eredu_nn::workspace::WorkspaceMetadataError::Overflow.into();
        };
        if let Err(cause) = self.charge_retained_metadata(bytes) {
            return cause;
        }
        let Self {
            prepared,
            original,
            tables,
            source_storage,
            ordinary,
            funding,
        } = self;
        drop(source_storage);
        drop(prepared);
        Error::backend_retained_source(OriginalMediaWorkspaceInputError {
            cause,
            tables: Some(tables),
            original,
            ordinary,
            _funding: (funding, None),
        })
    }
    pub(crate) fn plan_error(
        cause: eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
        tables: OriginalPreparedInputProjection,
        ordinary: Option<WorkingMemoryUnquotedLease>,
        funding: (
            Option<eredu_nn::workspace::HostMetadataFunding>,
            Option<eredu_nn::workspace::HostMetadataFunding>,
        ),
    ) -> Error {
        // The shared neutral error deallocates both its Arc and concrete Box
        // before this payload and its last ordinary lease retire.
        Error::backend_retained_source(PlanFailure {
            cause,
            tables,
            ordinary,
            _funding: funding,
        })
    }

    /// Existing table-based encoders keep their strict completed-table loan.
    pub(crate) fn construct_plan<P>(
        self,
        context: Option<&WorkspaceContext>,
        construct: impl FnOnce(PreparedModelInputOwner<WorkspaceTensor>, BoundPreparedMediaSemantics)
            -> Result<P, eredu_runtime::working_memory::OriginalCompositeSemanticStorageError>,
    ) -> Result<(
        P,
        (eredu_runtime::input::OriginalEncoderTableProjection,
         eredu_runtime::input::OriginalPreparedWorkspaceSource, Option<WorkingMemoryUnquotedLease>),
        (Option<eredu_nn::workspace::HostMetadataFunding>, Option<eredu_nn::workspace::HostMetadataFunding>),
    ), Error> {
        if !self.tables.has_encoder_tables() {
            return Err(self.reject(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into()));
        }
        let controls = std::mem::size_of::<(
            P, OriginalPreparedInputProjection, eredu_runtime::input::OriginalEncoderTableProjection,
            eredu_runtime::input::OriginalPreparedWorkspaceSource, Option<WorkingMemoryUnquotedLease>,
            (Option<eredu_nn::workspace::HostMetadataFunding>, Option<eredu_nn::workspace::HostMetadataFunding>),
        )>();
        match context {
            Some(context) => context.charge_metadata(controls)?,
            None => self.charge_retained_metadata(controls)?,
        }
        let (plan, (tables, storage, ordinary), funding) = self.construct_source_plan(context, construct)?;
        let tables = tables.into_encoder_tables().expect("checked immutable encoder tables");
        Ok((plan, (tables, storage, ordinary), funding))
    }

    /// Shares the original family plan worker while admitting its actual return
    /// and retained-error transports before consuming any source owner.
    pub(crate) fn construct_source_plan<P>(
        self,
        context: Option<&WorkspaceContext>,
        construct: impl FnOnce(
            PreparedModelInputOwner<WorkspaceTensor>,
            BoundPreparedMediaSemantics,
        ) -> Result<
            P,
            eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
        >,
    ) -> Result<
        (
            P,
            (
                OriginalPreparedInputProjection,
                eredu_runtime::input::OriginalPreparedWorkspaceSource,
                Option<WorkingMemoryUnquotedLease>,
            ),
            (
                Option<eredu_nn::workspace::HostMetadataFunding>,
                Option<eredu_nn::workspace::HostMetadataFunding>,
            ),
        ),
        Error,
    > {
        if context.is_some() || self.funding.is_some() {
            let controls = [
                std::mem::size_of::<Self>(),
                std::mem::size_of::<P>(),
                std::mem::size_of_val(&construct),
                std::mem::size_of::<PreparedModelInputOwner<WorkspaceTensor>>(),
                std::mem::size_of::<
                    Result<
                        (
                            P,
                            (
                                OriginalPreparedInputProjection,
                                eredu_runtime::input::OriginalPreparedWorkspaceSource,
                                Option<WorkingMemoryUnquotedLease>,
                            ),
                            (
                                Option<eredu_nn::workspace::HostMetadataFunding>,
                                Option<eredu_nn::workspace::HostMetadataFunding>,
                            ),
                        ),
                        Error,
                    >,
                >(),
                std::mem::size_of::<(
                    P,
                    OriginalPreparedInputProjection,
                    eredu_runtime::input::OriginalPreparedWorkspaceSource,
                    Option<WorkingMemoryUnquotedLease>,
                    (
                        Option<eredu_nn::workspace::HostMetadataFunding>,
                        Option<eredu_nn::workspace::HostMetadataFunding>,
                    ),
                )>(),
                std::mem::size_of::<
                    Result<P, eredu_runtime::working_memory::OriginalCompositeSemanticStorageError>,
                >(),
                Error::retained_source_construction_bytes::<PlanFailure>()
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
            ];
            let bytes = controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            match context {
                Some(context) => context.charge_metadata(bytes)?,
                None => self.charge_retained_metadata(bytes)?,
            }
        }
        let plan_funding = context.and_then(WorkspaceContext::metadata_funding);
        let (prepared, original, tables, source_storage, ordinary, source_funding) =
            self.into_parts();
        let funding = (source_funding, plan_funding);
        match construct(prepared.into(), original) {
            Ok(plan) => Ok((plan, (tables, source_storage, ordinary), funding)),
            Err(cause) => {
                drop(source_storage);
                Err(Self::plan_error(cause, tables, ordinary, funding))
            }
        }
    }

    pub(crate) fn reject_ingress_with_metadata(
        self,
        cause: eredu_runtime::media_prefill::MediaIngressError,
        context: &WorkspaceContext,
    ) -> Error {
        let Some(bytes) =
            Error::retained_source_construction_bytes::<OriginalMediaWorkspaceInputError>()
        else {
            return eredu_nn::workspace::WorkspaceMetadataError::Overflow.into();
        };
        if let Err(cause) = context.charge_metadata(bytes) {
            return cause.into();
        }
        let cause = context.metadata_source(cause);
        let Self {
            prepared,
            original,
            tables,
            source_storage,
            ordinary,
            funding,
        } = self;
        drop(source_storage);
        drop(prepared);
        Error::backend_retained_source(OriginalMediaWorkspaceInputError {
            cause,
            tables: Some(tables),
            original,
            ordinary,
            _funding: (funding, context.metadata_funding()),
        })
    }
    /// Projects the genuine original input under ordinary metadata custody.
    /// Actual I identity is checked before even constructing a projector.
    pub fn project<'a, T: PreparedMediaWorkspaceTensor>(
        source: &'a PreparedModelInputOwner<T>,
        original: BoundPreparedMediaSemantics,
        context: &'a WorkspaceContext,
        ordinary: &WorkingMemoryUnquotedLease,
    ) -> Result<Self, OriginalMediaWorkspaceInputError> {
        Self::project_with_custody(
            source,
            original,
            context,
            Some(ordinary.clone()),
            |original| {
                original
                    .source()
                    .validate_unquoted(ordinary)
                    .map_err(|cause| context.metadata_source(cause))
            },
        )
    }

    /// Projects the same exact B source under an already admitted host planning
    /// account. This performs no native allocation or execution and creates no
    /// ordinary exclusion lease. Source/pool identity and the complete native
    /// descriptor comparison are unchanged; funding survives every output/error.
    pub fn project_with_metadata<'a, T: PreparedMediaWorkspaceTensor>(
        source: &'a PreparedModelInputOwner<T>,
        original: BoundPreparedMediaSemantics,
        context: &'a WorkspaceContext,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<Self, OriginalMediaWorkspaceInputError> {
        Self::project_with_custody(source, original, context, None, |original| {
            if context.metadata_funding().is_none() {
                return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
            }
            original
                .source()
                .validate_pool(pool)
                .map_err(|cause| context.metadata_source(cause))
        })
    }

    fn project_with_custody<'a, T: PreparedMediaWorkspaceTensor>(
        source: &'a PreparedModelInputOwner<T>,
        original: BoundPreparedMediaSemantics,
        context: &'a WorkspaceContext,
        ordinary: Option<WorkingMemoryUnquotedLease>,
        validate: impl FnOnce(&BoundPreparedMediaSemantics) -> Result<(), Error>,
    ) -> Result<Self, OriginalMediaWorkspaceInputError> {
        let tables = source.original_projection();
        let result = (|| {
            let controls = [
                std::mem::size_of::<Self>(),
                std::mem::size_of::<OriginalMediaWorkspaceInputError>(),
                std::mem::size_of::<Result<Self, OriginalMediaWorkspaceInputError>>(),
            ];
            let bytes = controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            context.charge_metadata(bytes)?;
            validate(&original)?;
            let tables = tables.as_ref().ok_or_else(|| {
                context.metadata_error(format_args!(
                    "original media projection requires original B custody"
                ))
            })?;
            if !original.source().same_source(tables.source())
                || source.identity() != tables.identity()
            {
                return Err(context.metadata_error(format_args!(
                    "original media projection source differs from bound A/B",
                )));
            }
            source.project_workspace_with_metadata(context, &super::composite::TextInspector)
        })();
        match result {
            Ok((prepared, source_storage)) => Ok(Self {
                prepared,
                source_storage,
                original,
                tables: tables.expect("validated tables"),
                ordinary,
                funding: context.metadata_funding(),
            }),
            Err(cause) => Err(OriginalMediaWorkspaceInputError {
                cause,
                tables,
                original,
                ordinary,
                _funding: (None, context.metadata_funding()),
            }),
        }
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        PreparedModelInput<WorkspaceTensor>,
        BoundPreparedMediaSemantics,
        OriginalPreparedInputProjection,
        eredu_runtime::input::OriginalPreparedWorkspaceSource,
        Option<WorkingMemoryUnquotedLease>,
        Option<eredu_nn::workspace::HostMetadataFunding>,
    ) {
        (
            self.prepared,
            self.original,
            self.tables,
            self.source_storage,
            self.ordinary,
            self.funding,
        )
    }
}
