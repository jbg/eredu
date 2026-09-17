//! One owned validation batch shared by completion and independent recovery.
use super::{Array, Error, SubmissionResourcesOwner, TokenValidationBatch};
use eredu_runtime::working_memory::{
    OriginalTextControlGuard, OriginalTextMetadataCustody, WorkingMemoryError,
};
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, ManuallyDrop},
    rc::{Rc, Weak},
};

struct CompletionRoots {
    token_validations: TokenValidationBatch,
    output: CompletionOutput,
    // Last: the producing request survives the batch, output and Rc allocation.
    _owner: SubmissionResourcesOwner,
}

// A completion owns this independent C shell after the execution Scope seals.
// Raw host custody follows the shell through final completion/observation drop.
pub(super) struct CompletionOutput {
    array: Option<Array>,
    _custody: Option<OriginalTextMetadataCustody>,
}

pub(super) struct PreparedCompletionOutput {
    slot: safemlx::PreparedArrayClone,
    custody: OriginalTextMetadataCustody,
}

#[derive(Default)]
pub(super) enum CompletionOutputIngress {
    #[default]
    Ordinary,
    Original(PreparedCompletionOutput),
    Spent,
}

impl CompletionOutputIngress {
    pub(super) fn prepare(controls: &OriginalTextControlGuard) -> Result<Self, Error> {
        output_control_bytes().ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let custody = controls.metadata_custody();
        let slot = safemlx::PreparedArrayClone::try_prepare_for_inspection()
            .map_err(crate::backend::runtime::residency::manager::ResidencyError::OriginalClone)?;
        Ok(Self::Original(PreparedCompletionOutput { slot, custody }))
    }

    // Move the selected prepared owner out before the no-hooks native call or
    // any destructor. Ordinary completion preserves its historical clone path.
    pub(super) fn take(&mut self) -> Self {
        if matches!(self, Self::Ordinary) {
            Self::Ordinary
        } else {
            std::mem::replace(self, Self::Spent)
        }
    }

    pub(super) fn finish(self, source: Option<&Array>) -> Result<CompletionOutput, Error> {
        match self {
            Self::Ordinary => Ok(CompletionOutput {
                array: source.cloned(),
                _custody: None,
            }),
            Self::Original(mut prepared) => {
                let array = source
                    .map(|source| prepared.slot.fill_for_inspection(source))
                    .transpose()
                    .map_err(
                        crate::backend::runtime::residency::manager::ResidencyError::OriginalClone,
                    )?;
                Ok(CompletionOutput {
                    array,
                    _custody: Some(prepared.custody),
                })
            }
            Self::Spent => Err(Error::PrefillScopeUnavailable),
        }
    }
}

// Completion construction owns the handoff until a real completion exists.
// On refusal, request finalization while the Recovery parameter still owns its
// scope ticket. Settlement, never this guard, decides when release is safe.
pub(super) struct ConstructionRelease<'a>(Option<&'a SubmissionResourcesOwner>);
impl<'a> ConstructionRelease<'a> {
    pub(super) fn new(owner: &'a SubmissionResourcesOwner) -> Self {
        Self(Some(owner))
    }
    pub(super) fn disarm(mut self) {
        self.0 = None;
    }
}
impl Drop for ConstructionRelease<'_> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            if std::thread::panicking() {
                owner.reject_unresolved();
            }
            owner.request_release();
        }
    }
}

fn output_control_bytes() -> Option<usize> {
    crate::backend::runtime::residency::storage::native_storage::Bank::shared_borrowed_owner_bytes(
    )?;
    [
        size_of::<PreparedCompletionOutput>(),
        size_of::<ConstructionRelease<'static>>(),
        size_of::<SubmissionResourcesOwner>(), // completion's alias before disarm
        size_of::<CompletionOutputIngress>(),
        size_of::<CompletionOutput>(),
        size_of::<Result<CompletionOutputIngress, Error>>(),
        size_of::<Result<CompletionOutput, Error>>(),
        size_of::<Result<super::MlxSessionCompletion, Error>>(),
        size_of::<std::cell::RefMut<'static, CompletionOutputIngress>>(),
        size_of::<OriginalTextMetadataCustody>(),
        size_of::<Option<&Array>>(),
        size_of::<Result<Option<Array>, safemlx::PreparedArrayCloneCause>>(),
    ]
    .into_iter()
    .try_fold(
        Array::inspection_clone_handle_bytes()
            .checked_add(safemlx::PreparedArrayClone::control_bytes()?)?,
        usize::checked_add,
    )
}

/// Closed shared population: no raw Rc or Weak may escape this owner.
pub(in crate::composition::mlx::session) struct CompletionRootsOwner(Option<Rc<CompletionRoots>>);
impl CompletionRootsOwner {
    pub(super) fn new(
        output: CompletionOutput,
        token_validations: TokenValidationBatch,
        owner: SubmissionResourcesOwner,
    ) -> Self {
        Self(Some(Rc::new(CompletionRoots {
            token_validations,
            output,
            _owner: owner,
        })))
    }

    pub(super) fn arrays(&self) -> impl Iterator<Item = &Array> {
        let roots = self.0.as_deref().expect("live completion roots");
        roots
            .output
            .array
            .iter()
            .chain(roots.token_validations.arrays())
    }

    pub(in crate::composition::mlx::session) fn validate_completed(
        &self,
    ) -> Result<(), safemlx::error::Exception> {
        self.0
            .as_deref()
            .expect("live completion roots")
            .token_validations
            .validate_completed()
    }
}
impl Clone for CompletionRootsOwner {
    fn clone(&self) -> Self {
        Self(Some(
            self.0.as_ref().expect("live completion roots").clone(),
        ))
    }
}
impl Drop for CompletionRootsOwner {
    fn drop(&mut self) {
        if let Some(roots) = self.0.take() {
            // The last strong exit removes the Rc block before dropping native
            // roots, validation messages and finally their accounting owner.
            drop(Rc::into_inner(roots));
        }
    }
}

/// Observation's roots remain independent of the caller's completion/token.
pub(in crate::composition::mlx::session) enum ObservationRoots {
    Model { roots: CompletionRootsOwner },
    Scalar { _array: Array },
    Ordinary { _arrays: Vec<Array> },
}

/// One fixed allocation per model completion, included in ModelExecution Q.
/// The validation batch remains separately priced. One prepared C clone shell
/// covers the completion's independent output alias, even after Scope sealing.
pub(super) fn control_bytes() -> Option<u64> {
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align();
    let block = header
        .extend(Layout::new::<CompletionRoots>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let bytes = [
        size_of::<CompletionRoots>(), // constructor aggregate
        size_of::<CompletionRoots>(), // Rc::new argument
        size_of::<CompletionRoots>(), // final into_inner payload
        size_of::<CompletionRootsOwner>(),
        size_of::<Option<CompletionRootsOwner>>(),
        size_of::<Option<Rc<CompletionRoots>>>(),
        size_of::<Rc<CompletionRoots>>(),
        size_of::<ManuallyDrop<Rc<CompletionRoots>>>(),
        size_of::<Weak<CompletionRoots>>(),
        size_of::<Result<CompletionRoots, Rc<CompletionRoots>>>(),
        size_of::<Option<CompletionRoots>>(),
    ]
    .into_iter()
    .try_fold(block, usize::checked_add)?;
    u64::try_from(bytes.checked_add(output_control_bytes()?)?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{
            nn::tensor::{validate_token_domain, TokenValidationScope},
            ExecutionContext,
        },
        composition::mlx::session::{
            model_session::model_submission, MlxSessionCompletionKind, SessionAuthority,
        },
    };

    #[test]
    fn observation_keeps_the_moved_validation_handles_after_completion_retirement() {
        let context = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        for token in [2, 7] {
            let scope = TokenValidationScope::begin().unwrap();
            validate_token_domain(&Array::from_int(token), 4, None, context.stream()).unwrap();
            let batch = scope.finish();
            let validation_handle = batch.arrays().next().unwrap().as_ptr().ctx;
            assert_eq!(batch.arrays().count(), 1);
            safemlx::transforms::async_eval_with_event(batch.arrays())
                .unwrap()
                .synchronize()
                .unwrap();
            let mut authority = SessionAuthority::new();
            let submission = model_submission(
                Array::from_slice(&[3.0_f32, 7.0], &[1, 2]),
                batch,
                false,
                authority.begin_submission().unwrap(),
            );
            let completion = submission.completion;
            drop(submission.output);
            let MlxSessionCompletionKind::Model { roots, owner, .. } = &completion.inner;
            assert_eq!(
                roots.arrays().nth(1).unwrap().as_ptr().ctx,
                validation_handle
            );
            let mut observation = owner
                .observation_recovery_with_prediction(
                    ObservationRoots::Model {
                        roots: roots.clone(),
                    },
                    None,
                )
                .unwrap();
            observation.seal();
            assert_eq!(Rc::strong_count(roots.0.as_ref().unwrap()), 2);
            drop(completion);
            assert!(authority.require_idle().is_err());
            let ObservationRoots::Model { roots } = &observation.retention()._roots else {
                panic!("model observation");
            };
            assert_eq!(Rc::strong_count(roots.0.as_ref().unwrap()), 1);
            assert_eq!(
                roots.arrays().nth(1).unwrap().as_ptr().ctx,
                validation_handle
            );
            assert_eq!(roots.validate_completed().is_ok(), token < 4);
            let status = observation.finish();
            assert!(status.settled && !status.failed && !status.blocked);
            authority.require_idle().unwrap();
        }
    }

    #[test]
    fn completion_construction_refusal_requests_release_before_recovery_retirement() {
        for reentrant in [false, true] {
            let mut authority = SessionAuthority::new();
            let owner = super::super::SubmissionResources::new(
                authority.begin_submission().unwrap(),
                Rc::new(Cell::new(false)),
            );
            let mut recovery = owner.recovery().unwrap();
            recovery.seal();
            owner
                .completion_output
                .replace(CompletionOutputIngress::Spent);
            let loan = reentrant.then(|| owner.completion_output.borrow_mut());
            let result = super::super::model_completion(
                None,
                TokenValidationBatch::default(),
                owner.clone(),
                recovery,
            );
            if reentrant {
                assert!(matches!(result, Err(Error::PrefillScopeReentrant)));
            } else {
                assert!(matches!(result, Err(Error::PrefillScopeUnavailable)));
            }
            drop(loan);
            assert!(owner.release_requested.get());
            assert!(owner.is_healthy());
            crate::backend::submission_recovery::wait_for_retirement(|| {
                owner.resources_releasable()
            });
            // This alias intentionally outlives failure and the Recovery. The
            // lease must be released through settlement, not last-Rc teardown.
            authority.require_idle().unwrap();
            assert!(owner.lease.borrow().is_none());
        }
    }
}
