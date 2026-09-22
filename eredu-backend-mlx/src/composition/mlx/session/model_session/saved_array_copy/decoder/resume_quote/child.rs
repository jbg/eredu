//! Actual derived child declarations, separate from immutable numerical copies.
use super::*;
use eredu_runtime::{capture::FundedCaptureCheckpoint, layered::PreparedCaptureSelection};

struct Capture {
    checkpoint: FundedCaptureCheckpoint,
    selection: PreparedCaptureSelection,
    // Box and aliases retire before their prospective control account.
    _funding: HostMetadataFunding,
}
pub(in crate::composition::mlx::session) struct ResumeCapture(Option<Box<Capture>>);
impl Drop for ResumeCapture {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            let capture = *owner;
            drop(capture);
        }
    }
}
impl ResumeCapture {
    pub(in crate::composition::mlx::session) fn checkpoint(&self) -> &FundedCaptureCheckpoint {
        &self.0.as_ref().expect("live child declaration").checkpoint
    }
    pub(in crate::composition::mlx::session) fn selection(&self) -> &PreparedCaptureSelection {
        &self.0.as_ref().expect("live child declaration").selection
    }
    pub(super) fn prepare(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        source: &CopiedTextComponents,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
        funding: &HostMetadataFunding,
    ) -> Result<Option<Self>, Error> {
        let inherited = source
            .capture_checkpoint()
            .and_then(|checkpoint| checkpoint.intervention_source());
        let edits = options.intervention.or_else(|| {
            options
                .session_id
                .and_then(|_| inherited.map(|source| source.plan().admission().plan()))
        });
        if options.capture_limits.is_none() && edits.is_none() {
            return Ok(None);
        }
        if options.kind != eredu_core::OriginalTextResumeKind::Branch {
            return Err(Error::PrefillControl(
                WorkingMemoryError::PreparationConfigurationMismatch,
            ));
        }
        let saved = source
            .capture_checkpoint()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let selected = source
            .capture_selection()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let controls = eredu_core::capture::PreparedCapturePlanCopy::inspection_control_bytes()
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Capture>()))
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<(
                    Self,
                    Option<Self>,
                    Result<Option<Self>, Error>,
                    Option<FundedCaptureCheckpoint>,
                    Result<
                        PreparedCaptureSelection,
                        eredu_runtime::layered::PreparedCaptureSelectionError,
                    >,
                )>())
            })
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        funding
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        let session = runtime.session();
        let pool = runtime.backend().memory_ledger();
        let partition = session.partition_capture_source();
        let mut revised = if let Some(limits) = options.capture_limits {
            let copy = eredu_core::capture::PreparedCapturePlanCopy::inspect_limit_revision(
                saved.source(),
                limits.clone(),
            )
            .map_err(|cause| planned_error(cause, Some(funding)))?;
            let declaration = pool
                .compile_capture_source(copy)
                .map_err(|cause| planned_error(cause, Some(funding)))?;
            Some(
                saved
                    .with_branch_capture_source(&declaration, pool, funding)
                    .map_err(|cause| planned_error(cause, Some(funding)))?,
            )
        } else {
            None
        };
        if let Some(edits) = edits {
            let child_session = options
                .session_id
                .or_else(|| inherited.map(|source| source.plan().admission().session_id()))
                .ok_or(Error::PrefillControl(
                    WorkingMemoryError::PreparationConfigurationMismatch,
                ))?;
            let declaration = session
                .original_intervention_declaration(funding)?
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            let previous = revised.as_ref().unwrap_or(saved);
            let edits = declaration.compile_source(
                edits,
                previous.source().admission(),
                child_session,
                pool,
            )?;
            revised = Some(
                previous
                    .with_branch_intervention_source(&edits, pool, funding)
                    .map_err(|cause| planned_error(cause, Some(funding)))?,
            );
        }
        let checkpoint =
            revised.ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let selection = if selected.is_prepared_media() {
            selected
                .paths()
                .prepare_media_capture_selection(checkpoint.source())
        } else {
            selected
                .paths()
                .prepare_capture_selection(checkpoint.source())
        }
        .map_err(|cause| planned_error(cause, Some(funding)))?;
        Ok(Some(Self(Some(Box::new(Capture {
            checkpoint,
            selection,
            _funding: funding.clone(),
        })))))
    }
}

pub(super) struct Source<'a, 'b> {
    pub(super) saved: &'a CopiedTextComponents,
    pub(super) capture: Option<&'b ResumeCapture>,
}
impl std::ops::Deref for Source<'_, '_> {
    type Target = CopiedTextComponents;
    fn deref(&self) -> &Self::Target {
        self.saved
    }
}
impl Source<'_, '_> {
    pub(super) fn capture_checkpoint(&self) -> Option<&FundedCaptureCheckpoint> {
        self.capture
            .map(ResumeCapture::checkpoint)
            .or_else(|| self.saved.capture_checkpoint())
    }
    pub(super) fn capture_selection(&self) -> Option<&PreparedCaptureSelection> {
        self.capture
            .map(ResumeCapture::selection)
            .or_else(|| self.saved.capture_selection())
    }
}
