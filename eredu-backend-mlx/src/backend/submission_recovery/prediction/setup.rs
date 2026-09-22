//! Configure the same consumed prediction Scope before registration or work.
use super::*;
use safemlx::{
    OriginalNativeControlError, PreparedPrefillFailure, RetainedPrefillFailure,
    ScopedSubmissionProgress,
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum PredictionSetupCause {
    #[error(transparent)]
    Backend(#[from] Error),
    #[error("original prediction observation unavailable: {0:?}")]
    Observation(ScopedSubmissionProgress),
}

/// One concrete source Box, paid in the role recipe, with accepted custody last.
/// No new permission, Scope or failure publication is created by this wrapper.
#[derive(Debug)]
pub(crate) struct PredictionSetupFailure {
    cause: PredictionSetupCause,
    _controls: OriginalTextControlGuard,
}
impl PredictionSetupFailure {
    pub(crate) fn cause(&self) -> &PredictionSetupCause {
        &self.cause
    }
}
impl std::fmt::Display for PredictionSetupFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for PredictionSetupFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
pub(super) fn failure(cause: PredictionSetupCause, controls: OriginalTextControlGuard) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(PredictionSetupFailure {
            cause,
            _controls: controls,
        }),
        false,
    )
}

pub(super) fn begin_original<T: PredictionRetention>(
    role: PredictionRole,
    retention: T,
    controls: &OriginalTextControlGuard,
) -> Result<Recovery<T>, PredictionSetupCause> {
    begin_original_with_model(role, retention, controls, None).map(|(recovery, _)| recovery)
}
pub(super) fn begin_original_with_model<T: PredictionRetention>(
    mut role: PredictionRole,
    mut retention: T,
    controls: &OriginalTextControlGuard,
    preparation: Option<super::super::prefill::ModelExecutionPreparation>,
) -> Result<
    (
        Recovery<T>,
        Option<super::super::prefill::ModelExecutionOwner>,
    ),
    PredictionSetupCause,
> {
    if role.is_host_sequence() {
        // This mode comes only from a quote which requested neither native
        // arena. Preserve its exact host Scope/recovery allocations and custody.
        // A missing requested arena is never reclassified as host-only.
        if role.quota.is_some()
            || role.graph.is_some()
            || role.native_storage.is_some()
            || role.sampling.is_some()
            || role.paged.is_some()
            || preparation.is_some()
        {
            return Err(Error::PrefillScopeUnavailable.into());
        }
        let operations = role.operations.take();
        let (native, recovery, _, _) = role.into_custody();
        retention.install_prediction_custody(recovery);
        let pending = PreparedRecovery::new(retention, native).map_err(|error| {
            let RecoveryPreparationError {
                cause,
                retention,
                custody,
            } = error;
            drop(retention);
            drop(custody);
            Error::PredictionScope(cause)
        })?;
        let mut active = pending.try_begin().map_err(|error| {
            let PreparedRecoveryError { cause, pending } = error;
            drop(pending);
            Error::PredictionScope(cause)
        })?;
        // The role owns host controls only. Bind the exact operation's
        // ordinary Work before callbacks can borrow its indexed/cache sources.
        active.configure_scope_with_retention(|scope, retention| {
            retention.configure_ordinary_scope(scope)
        }).map_err(Error::from)?;
        if let Some((registration, request)) = operations {
            let node = active.node.as_mut().expect("accepted host prediction node").node_mut();
            node.registration = registration.register_for_retirement(
                &request,
                node.probe
                    .as_ref()
                    .expect("accepted host prediction scope"),
            )?;
        }
        return Ok((active, None));
    }
    // An original role without either arena cannot fall back to an ordinary
    // producer. Check before any new failure/Scope/recovery allocation.
    if role.quota.is_none() {
        return Err(Error::from(OriginalNativeControlError::MissingRecord).into());
    }
    if role.graph.is_none() {
        return Err(Error::from(OriginalNativeControlError::MissingGraph).into());
    }
    let operations = role.operations.take();
    let paged = role.paged.take();
    let native_storage = role.native_storage.take();
    let (native, recovery, quota, graph) = role.into_custody();
    retention.install_prediction_custody(recovery);
    // Independent carrier custody survives the Scope and any escaping native
    // failure aliases. Construction preserves the actual supplied guard.
    let failure = PreparedPrefillFailure::try_new(controls.clone())
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            drop(owner);
            Error::PrefillRoots(cause.into())
        })?
        .try_allocate()
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            drop(owner);
            Error::PrefillRoots(cause.into())
        })?;
    // The collector and prediction use this SAME still-unbound carrier. Its
    // root vectors exist before accepting/configuring the ModelExecution scope.
    let model = preparation
        .map(|preparation| {
            super::super::prefill::ModelExecutionOwner::prepare(
                preparation,
                graph.as_ref().expect("graph checked above"),
                &failure,
            )
        })
        .transpose()?;
    let pending = match PreparedRecovery::new(retention, native) {
        Ok(pending) => pending.with_record_quota(quota).with_graph_quota(graph),
        Err(RecoveryPreparationError {
            cause,
            retention,
            custody,
        }) => {
            drop(retention);
            drop(custody);
            return Err(Error::PredictionScope(cause).into());
        }
    };
    let mut active = match pending.try_begin() {
        Ok(active) => active,
        Err(PreparedRecoveryError { cause, pending }) => {
            // A consumed prediction role has no replacement/retry allowance.
            // Never-started nodes keep their existing guarded retirement path.
            drop(pending);
            return Err(Error::PredictionScope(cause).into());
        }
    };
    let node = active
        .node
        .as_mut()
        .expect("accepted prediction node")
        .node_mut();
    let scope = node.probe
        .as_mut()
        .expect("accepted prediction scope");
    scope
        .enable_scoped_observation()
        .map_err(PredictionSetupCause::Observation)?;
    scope
        .require_original_native_controls()
        .map_err(Error::from)?;
    if let Some(model) = &model {
        model.bind_scope(scope)?;
    } else {
        failure.bind_original_scope(scope).map_err(Error::from)?;
    }
    scope
        .enable_original_native_controls()
        .map_err(Error::from)?;
    if let Some(bank) = native_storage {
        bank.bind_scope(scope, controls)?;
    }
    if let Some(model) = &model {
        model.observe_scope(scope)?;
    }
    // Registration sees only this already configured original Scope. Failure
    // leaves the same active empty node on its existing safe Drop path.
    if let Some((registration, request)) = &operations {
        node.registration = registration.register_for_retirement(request, scope)?;
    }
    if let Some(model) = &model {
        model.begin_host(scope)?;
    }
    let paged_scope = paged.map(|(paged, attempt)| {
        let (_, request) = operations.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
        let transient = model.as_ref().ok_or(Error::PrefillScopeUnavailable)?.transient_roots()?;
        paged.enter_decode(request, attempt, scope, transient)
    }).transpose()?;
    // bind_original_scope cloned the carrier into Scope. Its native owner and
    // every later escaping failure retain controls after this local disappears.
    drop(failure);
    if let Some(paged) = paged_scope { active.retention_mut().install_paged_scope(paged)?; }
    Ok((active, model))
}

pub(super) fn control_bytes() -> Option<u64> {
    use std::mem::size_of;
    let carrier = PreparedPrefillFailure::<OriginalTextControlGuard>::layout()
        .ok()?
        .total_bytes()?;
    let native = safemlx::OriginalNativeControlLayout::inspect().ok()?;
    let controls = [
        carrier,
        native.fixed_control_bytes,
        safemlx::original_scoped_evaluation_control_bytes()?,
        eredu_core::BackendFailure::source_retention_peak_bytes::<PredictionSetupFailure>()?,
        size_of::<PredictionSetupFailure>(),
        size_of::<PredictionSetupCause>(),
        size_of::<OriginalTextControlGuard>(),
        size_of::<RetainedPrefillFailure>(),
        size_of::<Result<(), ScopedSubmissionProgress>>(),
        size_of::<Result<(), OriginalNativeControlError>>(),
        size_of::<super::OperationRegistration>(),
        size_of::<Option<crate::backend::runtime::execution::generic::RegisteredOriginalScope>>(),
        size_of::<Result<Option<crate::backend::runtime::execution::generic::RegisteredOriginalScope>, Error>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    u64::try_from(controls).ok()
}

pub(crate) fn validate_layouts() -> Result<(), Error> {
    safemlx::OriginalNativeControlLayout::inspect()?;
    PreparedPrefillFailure::<OriginalTextControlGuard>::layout()
        .map_err(|cause| Error::PrefillRoots(cause.into()))?;
    Ok(())
}
