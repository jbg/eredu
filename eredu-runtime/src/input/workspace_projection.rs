//! Exact original prepared input projected through its ordinary tensor mechanism.
use super::{
    PreparedInputInspector, PreparedInputPart, PreparedModelInput, PreparedModelInputOwner,
};
use crate::working_memory::{MemoryLedger, PreparedInputHostCustody, WorkingMemoryError};
use eredu_nn::{
    workspace::{HostMetadataFunding, WorkspaceBorrowedStorage, WorkspaceContext, WorkspaceTensor},
    Error, Tensor,
};
use std::mem::{size_of, size_of_val};

/// Native tensor realization of an actual original slot's settled backing.
/// Implementations preserve aliases across the supplied source inventory. Unknown
/// backing stays unknown; projection performs no native allocation or evaluation.
pub trait PreparedMediaWorkspaceTensor: Tensor + Send + Sync {
    type Projection<'a>
    where
        Self: 'a;
    fn workspace_projection<'a>(context: &'a WorkspaceContext) -> Self::Projection<'a>;
    fn workspace_projection_with_count<'a>(
        context: &'a WorkspaceContext,
        _slots: usize,
    ) -> Result<Self::Projection<'a>, Error> {
        if context.uses_checked_metadata() {
            Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
        } else {
            Ok(Self::workspace_projection(context))
        }
    }
    fn project_workspace_slot<'a>(
        &'a self,
        projection: &mut Self::Projection<'a>,
    ) -> Result<WorkspaceTensor, Error>;
    /// Complete roots from exactly the slots already projected by this loan.
    /// No unrelated root or newly generated tensor may enter this selection.
    /// None preserves incomplete evidence; it never becomes source credit.
    fn finish_workspace_projection<'a>(
        _projection: Self::Projection<'a>,
        _context: &'a WorkspaceContext,
    ) -> Result<Option<WorkspaceBorrowedStorage>, Error> {
        Ok(None)
    }
}

/// A source-owned association between exact projected roots and its original B
/// account. The private constructor visits that actual prepared owner; equal
/// descriptors, a byte count or arbitrary roots cannot construct this witness.
/// A captured cache can also lend an account-only residence witness without
/// projected roots; that form cannot grant source credit or media ingress.
/// It contains no native tensor or payload backedge. It grants no native work.
pub struct OriginalPreparedWorkspaceSource {
    roots: Option<WorkspaceBorrowedStorage>,
    custody: PreparedInputHostCustody,
    funding: Option<HostMetadataFunding>,
}
impl OriginalPreparedWorkspaceSource {
    pub(super) fn cache_residence(custody: PreparedInputHostCustody) -> Self {
        Self {
            roots: None,
            custody,
            funding: None,
        }
    }

    /// Compares the actual original materialization account without inspecting
    /// payloads, cloning an owner or accepting equal logical descriptors.
    /// This is source evidence only and grants no work or additional binding.
    pub fn matches_prepared_owner<T>(&self, owner: &PreparedModelInputOwner<T>) -> bool {
        owner
            .workspace_custody_ref()
            .is_some_and(|custody| self.custody.same_account(custody))
    }
    /// True only for cache metadata constructed by this same accepted B
    /// materialization. Equal host contents/descriptors or another materialization
    /// in the same pool are insufficient. This performs no allocation or adoption.
    pub fn matches_cache_identity(&self, cache: &crate::SharedPreparedInputCacheIdentity) -> bool {
        cache.matches_original_account(&self.custody)
    }
    /// Exact source projection, or unknown backing that cannot receive credit.
    pub fn borrowed_storage(&self) -> Option<&WorkspaceBorrowedStorage> {
        self.roots.as_ref()
    }
    /// The B account's actual pool; no registry entry or second charge is created.
    pub fn pool(&self) -> &MemoryLedger {
        self.custody.pool()
    }
    pub(crate) fn account_pin(&self) -> PreparedInputHostCustody {
        self.custody.share()
    }
}
impl Clone for OriginalPreparedWorkspaceSource {
    fn clone(&self) -> Self {
        Self {
            roots: self.roots.clone(),
            custody: self.custody.share(),
            funding: self.funding.clone(),
        }
    }
}
impl std::fmt::Debug for OriginalPreparedWorkspaceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalPreparedWorkspaceSource")
            .field("complete", &self.roots.is_some())
            .finish_non_exhaustive()
    }
}

impl<T: PreparedMediaWorkspaceTensor> PreparedModelInputOwner<T> {
    /// Projects the sole actual payload of a completed original source. This
    /// is useful for copy programs that need no reconstructed input tree or
    /// semantic inspector. Metadata-bearing/multiple-part sources use the full
    /// projection above; arbitrary tensors cannot construct the B witness.
    pub fn project_single_payload_with_metadata<'a>(
        &'a self,
        context: &'a WorkspaceContext,
    ) -> Result<(WorkspaceTensor, OriginalPreparedWorkspaceSource), Error> {
        let [part] = self.parts() else {
            return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
        };
        if !part.metadata().is_empty() {
            return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
        }
        self.project_payload_with_metadata(0, context)
    }
    /// Projects one actual payload selected by its original ordered part index.
    /// Metadata slots and other parts are not credited to this payload loan.
    /// The source witness retains the same complete original B account.
    pub fn project_payload_with_metadata<'a>(
        &'a self,
        index: usize,
        context: &'a WorkspaceContext,
    ) -> Result<(WorkspaceTensor, OriginalPreparedWorkspaceSource), Error> {
        let custody = self
            .workspace_custody()
            .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?;
        let part = self
            .parts()
            .get(index)
            .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?;
        context.charge_metadata(size_of::<(
            T::Projection<'a>,
            WorkspaceTensor,
            OriginalPreparedWorkspaceSource,
            PreparedInputHostCustody,
            Option<WorkspaceBorrowedStorage>,
            Result<(WorkspaceTensor, OriginalPreparedWorkspaceSource), Error>,
        )>())?;
        let mut projection = T::workspace_projection_with_count(context, 1)?;
        let value = part
            .payload()
            .value()
            .project_workspace_slot(&mut projection)?;
        let roots = T::finish_workspace_projection(projection, context)?;
        validate_projected_requirements(
            roots
                .as_ref()
                .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?,
            &custody,
            context,
        )?;
        Ok((
            value,
            OriginalPreparedWorkspaceSource {
                roots,
                custody,
                funding: context.metadata_funding(),
            },
        ))
    }
    /// Projects this genuine B's ordered slots with one shared metadata worker.
    /// The inspector describes projected shapes; it never selects another source.
    /// The returned witness retains only accounting and metadata after native
    /// source owners retire. Keep funding through any escaping error as usual.
    pub fn project_workspace_with_metadata<'a>(
        &'a self,
        context: &'a WorkspaceContext,
        inspector: &impl PreparedInputInspector<WorkspaceTensor>,
    ) -> Result<
        (
            PreparedModelInput<WorkspaceTensor>,
            OriginalPreparedWorkspaceSource,
        ),
        Error,
    > {
        let custody = self
            .workspace_custody()
            .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?;
        let frames = [
            size_of::<T::Projection<'a>>(),
            size_of::<Vec<PreparedInputPart<WorkspaceTensor>>>(),
            size_of::<PreparedModelInput<WorkspaceTensor>>(),
            size_of::<OriginalPreparedWorkspaceSource>(),
            size_of::<
                Result<
                    (
                        PreparedModelInput<WorkspaceTensor>,
                        OriginalPreparedWorkspaceSource,
                    ),
                    Error,
                >,
            >(),
            size_of::<PreparedInputHostCustody>(),
            size_of::<Option<WorkspaceBorrowedStorage>>(),
            size_of::<(&Self, &WorkspaceContext, usize)>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(bytes)?;
        let slots = self
            .parts()
            .iter()
            .try_fold(0usize, |count, part| {
                count.checked_add(1)?.checked_add(part.metadata().len())
            })
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let mut projection = T::workspace_projection_with_count(context, slots)?;
        let mut parts = context.metadata_vec(self.len())?;
        for part in self.parts() {
            parts.push(part.map_with_metadata(context, |value| {
                value.project_workspace_slot(&mut projection)
            })?);
        }
        let prepared = PreparedModelInput::new_with_metadata(parts, context, |tensor| {
            inspector.identity_with_metadata(tensor, context)
        })?;
        if prepared.identity() != self.identity() {
            return Err(context.metadata_error(format_args!(
                "actual native slot projection changed original ordered descriptors"
            )));
        }
        let roots = T::finish_workspace_projection(projection, context)?;
        if let Some(roots) = &roots {
            validate_projected_requirements(roots, &custody, context)?;
        }
        Ok((
            prepared,
            OriginalPreparedWorkspaceSource {
                roots,
                custody,
                funding: context.metadata_funding(),
            },
        ))
    }
}

fn validate_projected_requirements(
    roots: &WorkspaceBorrowedStorage,
    custody: &PreparedInputHostCustody,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    let requirements = roots.requirements(context)?;
    requirements
        .validate(custody.pool().topology())
        .map_err(|e| context.metadata_source(e))?;
    for (domain, projected) in requirements.iter() {
        let retained = custody
            .requirements()
            .get(domain)
            .map_err(|e| context.metadata_source(e))?;
        if projected.total().map_err(|e| context.metadata_source(e))?
            > retained.total().map_err(|e| context.metadata_source(e))?
        {
            return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
        }
    }
    Ok(())
}
