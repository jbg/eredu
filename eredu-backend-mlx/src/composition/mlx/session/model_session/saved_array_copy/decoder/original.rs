//! Cold native/publication requirements consumed by the existing copy driver.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    array_copy::{
        OriginalCopyCause, OriginalCopyLayoutBuilder, OriginalCopyPlan, SavedHostCopyPlan,
    },
    runtime::cache::state::{SnapshotArraySources, SnapshotOperand},
};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(super) enum CopyPreparationCause {
    #[error(transparent)]
    Source(#[from] crate::backend::runtime::cache::state::SnapshotProjectionCause),
    #[error(transparent)]
    Native(#[from] OriginalCopyCause),
    #[error(transparent)]
    Environment(#[from] crate::backend::OriginalCopyEnvironmentError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
}

/// Source geometry only; no borrowed backend reference crosses the mutable
/// execution handoff. The same pinned source is recounted before admission.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct CopyRequirements {
    operands: usize,
    native: Option<NativeRequirements>,
    publication_controls: usize,
    controls: usize,
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct NativeRequirements {
    controls: usize,
    physical: usize,
    host_destinations: usize,
}
impl NativeRequirements {
    fn inspect(plan: &OriginalCopyPlan<'_>) -> Result<Self, WorkingMemoryError> {
        Ok(Self {
            controls: plan
                .control_bytes::<ScopeRetention>()
                .ok_or(WorkingMemoryError::Overflow)?,
            physical: plan.physical_bytes(),
            host_destinations: plan.host_destination_roots(),
        })
    }
}

impl CopyRequirements {
    pub(super) fn inspect(
        decoder: &PreparedResidentDecoderCopy<'_>,
        key: Option<&Array>,
        pending: Option<&Array>,
        backend: &MlxBackend<'_>,
        operands: usize,
    ) -> Result<Self, CopyPreparationCause> {
        let native = if operands == 0 {
            validate_empty(decoder, key, pending)?;
            None
        } else {
            let environment = backend.original_copy_environment()?;
            native_plan(decoder, key, pending, &environment)?
                .as_ref()
                .map(NativeRequirements::inspect)
                .transpose()?
        };
        let host_rows = native.map_or(0, |value| value.host_destinations);
        let roots = operands
            .checked_add(host_rows)
            .ok_or(WorkingMemoryError::Overflow)?;
        let publication =
            text_funding::SnapshotPublicationPlan::with_host(roots, host_rows, 0)?;
        let parts = [
            native.map_or(0, |value| value.controls),
            publication.control_bytes(),
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<CopyPreparationCause>(),
            size_of::<Result<Self, CopyPreparationCause>>(),
            size_of::<Result<Option<OriginalCopyPlan<'_>>, CopyPreparationCause>>(),
            size_of::<text_funding::SnapshotPublicationPlan>(),
            size_of::<Result<text_funding::SnapshotPublicationPlan, WorkingMemoryError>>(),
            size_of::<Option<crate::backend::array_copy::PreparedOriginalCopy>>(),
            size_of::<Option<crate::backend::array_copy::OriginalCopyExecution>>(),
            size_of::<ConstructionGuard<'_>>(),
            size_of::<Option<safemlx::PrefillRootsRuntime>>(),
            size_of::<Result<safemlx::PrefillRootsRuntime, Error>>(),
            size_of::<Option<OriginalCopyEnvironment<'_>>>(),
            size_of::<Option<OriginalCopyPlan<'_>>>(),
            size_of::<
                Result<
                    Option<crate::backend::array_copy::PreparedOriginalCopy>,
                    crate::backend::array_copy::OriginalCopyFailure,
                >,
            >(),
            size_of::<Result<(), Error>>(),
            size_of::<SavedHostCopyPlan>(),
            size_of::<Result<SavedHostCopyPlan, OriginalCopyCause>>(),
            size_of::<SnapshotOperand<'_>>(),
            size_of::<
                &mut dyn FnMut(
                    SnapshotOperand<'_>,
                ) -> Result<
                    (),
                    crate::backend::runtime::cache::state::SnapshotProjectionCause,
                >,
            >(),
            size_of::<Option<Error>>(),
            size_of::<Option<safemlx::OriginalScopeObserver>>(),
            size_of::<safemlx::OriginalScopeObserver>(),
            size_of::<Result<safemlx::EvaluatedArray<'_>, safemlx::error::Exception>>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            operands,
            native,
            publication_controls: publication.control_bytes(),
            controls,
        })
    }
    pub(super) fn control_bytes(self) -> usize {
        self.controls
    }
    pub(super) fn host_rows(self) -> usize {
        self.native.map_or(0, |value| value.host_destinations)
    }
    pub(super) fn has_native(self) -> bool {
        self.native.is_some()
    }
    pub(super) fn physical_extra(self, numerical_bytes: u64) -> Result<u64, WorkingMemoryError> {
        let native = self.native.map_or(0, |value| value.physical);
        let native = u64::try_from(native).map_err(|_| WorkingMemoryError::Overflow)?;
        Ok(native.saturating_sub(numerical_bytes))
    }
    pub(super) fn validate_native(
        self,
        plan: Option<&OriginalCopyPlan<'_>>,
    ) -> Result<(), WorkingMemoryError> {
        let actual = plan.map(NativeRequirements::inspect).transpose()?;
        if actual != self.native {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    pub(super) fn publication_plan(
        self,
    ) -> Result<text_funding::SnapshotPublicationPlan, WorkingMemoryError> {
        let host_rows = self.host_rows();
        let roots = self
            .operands
            .checked_add(host_rows)
            .ok_or(WorkingMemoryError::Overflow)?;
        let plan =
            text_funding::SnapshotPublicationPlan::with_host(roots, host_rows, 0)?;
        if plan.control_bytes() != self.publication_controls {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(plan)
    }
}

pub(super) fn native_plan<'a>(
    decoder: &PreparedResidentDecoderCopy<'_>,
    key: Option<&Array>,
    pending: Option<&Array>,
    environment: &'a OriginalCopyEnvironment<'_>,
) -> Result<Option<OriginalCopyPlan<'a>>, CopyPreparationCause> {
    let mut builder = OriginalCopyLayoutBuilder::new();
    let mut failure = None;
    let mut retain = |source: &Array| {
        if failure.is_none() {
            failure = builder.push_retained_source(source).err();
        }
    };
    if decoder.is_paged() {
        SnapshotArraySources::visit_operands(decoder, &mut |source| {
            if let SnapshotOperand::Array(array) = source {
                retain(array);
            }
            Ok(())
        })?;
    } else {
        decoder.visit_retained_arrays(&mut retain)?;
    }
    for source in key.into_iter().chain(pending) {
        retain(source);
    }
    if let Some(cause) = failure {
        return Err(cause.into());
    }
    SnapshotArraySources::visit_operands(decoder, &mut |source| {
        if failure.is_none() {
            failure = (|| match source {
                SnapshotOperand::Array(array) => builder.push_operand(array),
                SnapshotOperand::Host(host) => {
                    builder.push_host_operand(host)?;
                    builder.push_host_destination(&SavedHostCopyPlan::from_host(host, environment)?)
                }
            })()
            .err();
        }
        Ok(())
    })?;
    for source in key.into_iter().chain(pending) {
        if failure.is_none() {
            failure = builder.push_operand(source).err();
        }
    }
    if let Some(cause) = failure {
        return Err(cause.into());
    }
    builder.finish(environment).map_err(Into::into)
}

fn validate_empty(
    decoder: &PreparedResidentDecoderCopy<'_>,
    key: Option<&Array>,
    pending: Option<&Array>,
) -> Result<(), CopyPreparationCause> {
    let mut populated = key.is_some() || pending.is_some();
    SnapshotArraySources::visit_operands(decoder, &mut |_| {
        populated = true;
        Ok(())
    })?;
    if populated {
        return Err(WorkingMemoryError::IdentityMismatch.into());
    }
    Ok(())
}

/// Initial recovery roots are the full retained source set. Native construction
/// has begun before this worker, so each original clone consumes its paid shell.
pub(super) fn retain_sources(
    decoder: &PreparedResidentDecoderCopy<'_>,
    sampling: &TextArrayBinding<'_>,
    roots: &RefCell<Vec<Array>>,
    original: bool,
    native: Option<&crate::backend::nn::workspace::ProjectedNativeStorage>,
) -> Result<(), Error> {
    let mut failure = None;
    let mut retain = |array: &Array| {
        if failure.is_some() {
            return;
        }
        let copied = if original {
            array.try_clone_handle()
        } else {
            Ok(array.clone())
        };
        match copied {
            Ok(copy) => roots.borrow_mut().push(copy),
            Err(cause) => failure = Some(Error::from(cause)),
        }
    };
    if decoder.is_paged() {
        let native =
            native.ok_or_else(|| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        for (identity, _, _) in native.iter() {
            if let Some(array) = native.native_array(identity) {
                retain(array);
            } else if native.native_host(identity).is_none() {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
        }
    } else {
        decoder
            .visit_retained_arrays(&mut retain)
            .map_err(crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error)?;
        // The paged prepared inventory already includes these exact sampling
        // arrays. Add them here only for the ordinary direct decoder visitor.
        for source in sampling.key().into_iter().chain(sampling.pending()) {
            retain(source);
        }
    }
    match failure {
        Some(cause) => Err(cause),
        None => Ok(()),
    }
}

pub(super) fn destination_plan<'a>(
    native: &'a SavedResidentDecoderCopy,
    authority: &eredu_core::HostPreparationAuthority,
) -> Result<PreparedResidentDecoderCopy<'a>, Error> {
    native.prepare_copy_fixed().map_err(|cause| {
        cold_source::retain_failure(
            cold_source::SourcePreparationCause::Decoder(cause),
            authority,
        )
    })
}

struct ConstructionGuard<'a>(Option<&'a mut crate::backend::array_copy::OriginalCopyExecution>);
impl Drop for ConstructionGuard<'_> {
    fn drop(&mut self) {
        if let Some(execution) = self.0.as_mut() {
            execution.finish_construction();
        }
    }
}
/// Always remove the resident construction bank before the surrounding
/// SessionOperation seals or unwinds its recovery. The execution's budget
/// remains retained through destination publication after this local guard.
pub(super) fn during_construction<T>(
    execution: &mut Option<crate::backend::array_copy::OriginalCopyExecution>,
    copy: impl FnOnce() -> T,
) -> T {
    let _guard = ConstructionGuard(execution.as_mut());
    copy()
}

/// The isolated worker already completed the copy. Verify that same scope's
/// finished storage without starting another Eval or changing its graph bound.
pub(super) fn validate_completed_destination(array: &Array) -> Result<(), Error> {
    let observer = safemlx::OriginalScopeObserver::try_current()?
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    array.completed_in_original_scope(&observer)?;
    Ok(())
}
