//! Closed immutable original policy and its actual paid host-copy constructor.
mod source;
use super::*;
pub(super) use source::{
    controller_source_control_bytes, grammar_source_control_bytes, validate_controller_source,
    validate_grammar_source,
};
mod adaptive;
mod control;
mod grammar;
use eredu_core::{
    capture::CaptureError, speculative::SpeculativeControlError, HostPreparationAuthority,
};
use eredu_nn::workspace::HostMetadataFundingError;
use eredu_runtime::generation::{
    PreparedControllerError, PreparedSpeculativeController, PreparedSpeculativeSamplerCopy,
};
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, size_of_val},
    rc::Rc,
};

const CAPTURE_REFUSAL: &str = "original sampler capture requires its admitted capture producer";
struct Owner<S> {
    source: S,
    capture_refusal: CaptureError,
    grammar_complete: bool,
    fixed_controller: bool,
    snapshot: Option<numerical::SnapshotContext>,
    funding: Option<eredu_nn::workspace::HostMetadataFunding>,
    // S and the refusal retire before this owner can return their host account.
    _host: HostPreparationAuthority,
}
pub(super) struct Original<S>(Option<Rc<Owner<S>>>);
impl<S> Clone for Original<S> {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(self.0.as_ref().expect("live sampler"))))
    }
}
impl<S> Drop for Original<S> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
impl<S> Original<S> {
    fn owner(&self) -> &Owner<S> {
        self.0.as_deref().expect("live sampler")
    }
}
pub(super) enum Policy<S> {
    Ordinary(S),
    Original(Original<S>),
}
impl<S: Clone> Clone for Policy<S> {
    fn clone(&self) -> Self {
        match self {
            Self::Ordinary(s) => Self::Ordinary(s.clone()),
            Self::Original(s) => Self::Original(s.clone()),
        }
    }
}
impl<S> Policy<S> {
    pub(super) fn source(&self) -> &S {
        match self {
            Self::Ordinary(s) => s,
            Self::Original(s) => &s.owner().source,
        }
    }
    pub(super) fn original_source(&self) -> Result<&S, Error> {
        match self {
            Self::Original(s) => Ok(&s.owner().source),
            Self::Ordinary(_) => Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )),
        }
    }
    pub(super) fn ordinary_mut(&mut self) -> Option<&mut S> {
        match self {
            Self::Ordinary(s) => Some(s),
            Self::Original(_) => None,
        }
    }
    pub(super) fn is_original(&self) -> bool {
        matches!(self, Self::Original(_))
    }
    fn is_fixed_controller(&self) -> bool {
        matches!(self,Self::Original(source) if source.owner().fixed_controller)
    }
    pub(super) fn grammar_complete(&self) -> Option<bool> {
        match self {
            Self::Original(s) => Some(s.owner().grammar_complete),
            Self::Ordinary(_) => None,
        }
    }
    fn snapshot_context(&self) -> Option<&numerical::SnapshotContext> {
        match self {
            Self::Original(s) => s.owner().snapshot.as_ref(),
            Self::Ordinary(_) => None,
        }
    }
    fn metadata_funding(&self) -> Option<&eredu_nn::workspace::HostMetadataFunding> {
        match self {
            Self::Original(source) => source.owner().funding.as_ref(),
            Self::Ordinary(_) => None,
        }
    }
    pub(super) fn capture_refusal(&self) -> Option<CaptureError> {
        match self {
            Self::Original(s) => Some(s.owner().capture_refusal.clone()),
            Self::Ordinary(_) => None,
        }
    }
    #[cfg(test)]
    pub(super) fn into_ordinary(self) -> S {
        match self {
            Self::Ordinary(s) => s,
            Self::Original(_) => panic!("original sampler has no owning policy escape"),
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct ControllerFailure {
    #[source]
    cause: PreparedControllerError,
    _host: HostPreparationAuthority,
}
fn controller_failure(
    cause: PreparedControllerError,
    host: &HostPreparationAuthority,
) -> eredu_core::BackendFailure {
    eredu_core::BackendFailure::from_error(ControllerFailure {
        cause,
        _host: host.clone(),
    })
}
enum CopyPlan<'a, S> {
    Standard(PreparedSpeculativeSamplerCopy<'a, S>),
    Controller(PreparedSpeculativeController<'a, S>),
}
impl<S> CopyPlan<'_, S> {
    fn metadata_bytes(&self) -> usize {
        match self {
            Self::Standard(plan) => plan.metadata_bytes(),
            Self::Controller(plan) => plan.copy_metadata_bytes(),
        }
    }
    fn validate_source(
        &self,
        pool: &eredu_runtime::working_memory::MemoryLedger,
    ) -> Result<(), Error> {
        if let Self::Controller(plan) = self {
            let source = plan.controller_source().map_err(|_| {
                Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )
            })?;
            validate_controller_source(source, pool)?;
        }
        Ok(())
    }
    fn copy(self, host: &HostPreparationAuthority) -> Result<S, eredu_core::BackendFailure> {
        match self {
            Self::Standard(plan) => Ok(plan.copy()),
            Self::Controller(plan) => plan
                .copy(host.clone())
                .map_err(|cause| controller_failure(cause, host)),
        }
    }
}
fn copy_plan<S: SpeculativeSampler<MlxSamplingBackend>>(source: &S) -> Option<CopyPlan<'_, S>> {
    if let Some(plan) = source.prepared_host_copy() {
        Some(CopyPlan::Standard(plan))
    } else {
        source.prepared_controller().map(CopyPlan::Controller)
    }
}
fn own_original<S>(
    source: S,
    host: HostPreparationAuthority,
    funding: Option<eredu_nn::workspace::HostMetadataFunding>,
    fixed_controller: bool,
    snapshot: Option<numerical::SnapshotContext>,
) -> Policy<S> {
    let capture_refusal =
        CaptureError::Unsupported(CAPTURE_REFUSAL.into()).retain_ordinary(host.clone());
    Policy::Original(Original(Some(Rc::new(Owner {
        source,
        capture_refusal,
        grammar_complete: false,
        fixed_controller,
        snapshot,
        funding,
        _host: host,
    }))))
}
fn copy_original<S>(
    plan: CopyPlan<'_, S>,
    host: HostPreparationAuthority,
    funding: Option<eredu_nn::workspace::HostMetadataFunding>,
    snapshot: Option<numerical::SnapshotContext>,
) -> Result<Policy<S>, eredu_core::BackendFailure> {
    let fixed_controller = matches!(&plan, CopyPlan::Controller(_));
    let source = plan.copy(&host)?;
    Ok(own_original(
        source,
        host,
        funding,
        fixed_controller,
        snapshot,
    ))
}
#[cfg(test)]
fn original<S>(
    plan: PreparedSpeculativeSamplerCopy<'_, S>,
    host: HostPreparationAuthority,
) -> Policy<S> {
    copy_original(CopyPlan::Standard(plan), host, None, None).expect("standard copy is infallible")
}
fn controls<S, E, L>(copy: usize) -> Option<usize> {
    let shared = Layout::new::<[Cell<usize>; 2]>()
        .extend(Layout::new::<Owner<S>>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let parts = [
        copy,
        controller_source_control_bytes()?,
        numerical::SnapshotContext::controls()?,
        adaptive::control_bytes::<S>()?,
        eredu_core::BackendFailure::source_retention_peak_bytes::<ControllerFailure>()?,
        size_of::<ControllerFailure>(),
        size_of::<Option<eredu_nn::workspace::HostMetadataFunding>>(),
        size_of::<CopyPlan<'_, S>>(),
        size_of::<Result<S, eredu_core::BackendFailure>>(),
        size_of::<Result<Policy<S>, eredu_core::BackendFailure>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<bool, Error>>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<SamplingPlacement>(),
        size_of::<(
            &mut MlxSpeculativeSampling<S, E, L>,
            &numerical::OriginalNumericalValue,
            u32,
            SamplingPlacement,
            SpeculativeExecutionStreams<'_>,
        )>(),
        shared,
        size_of::<Owner<S>>(),
        size_of::<Original<S>>(),
        size_of::<Policy<S>>(),
        size_of::<MlxSpeculativeSampling<S, E, L>>(),
        size_of::<HostPreparationAuthority>(),
        CAPTURE_REFUSAL.len(),
        size_of::<String>(),
        size_of::<CaptureError>(),
        usize::try_from(CaptureError::ordinary_retained_control_bytes()?).ok()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl<S: SpeculativeSampler<MlxSamplingBackend>> MlxSpeculativeSampling<S> {
    fn from_original_policy(inner: Policy<S>) -> Self {
        Self {
            inner,
            capture: None,
            original_capture: None,
            original_interventions: None,
            interventions: Vec::new(),
            memory_retention: None,
            error: std::marker::PhantomData,
        }
    }
    /// Ordinary construction stays unchanged. Original construction copies the
    /// exact known host source before wrapping it; it never adopts its old box.
    pub(crate) fn prepare(
        source: S,
        context: SpeculativeExecutionStreams<'_>,
        memory: Option<&NativeMemoryOwner>,
    ) -> Result<Self, Error> {
        let Some((sources, environment)) = context.original_numerical() else {
            let memory = memory.ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?;
            return Ok(Self::new(source).with_memory_owner(memory));
        };
        sources.validate_environment(environment)?;
        if source.prepared_grammar_controller().is_some() {
            let inner = grammar::prepare(&source, context)?;
            return Ok(Self::from_original_policy(inner));
        }
        let plan = copy_plan(&source).ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))?;
        let bytes = controls::<S, Exception, Array>(plan.metadata_bytes())
            .and_then(|bytes| {
                let parts = [
                    HostPreparationAuthority::retention_bytes::<
                        eredu_nn::workspace::HostMetadataFunding,
                    >()?,
                    size_of::<Result<Self, Error>>(),
                    size_of::<SpeculativeExecutionStreams<'_>>(),
                    size_of::<Option<&NativeMemoryOwner>>(),
                ];
                parts
                    .into_iter()
                    .try_fold(bytes.checked_add(size_of_val(&parts))?, usize::checked_add)
            })
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
        sources
            .metadata_funding()
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        plan.validate_source(environment.pool())?;
        let host = HostPreparationAuthority::retain(sources.metadata_funding().clone());
        let inner = copy_original(
            plan,
            host,
            Some(sources.metadata_funding().clone()),
            Some(numerical::SnapshotContext::prepare_for_context(context)?),
        )
        .map_err(Error::StorageSource)?;
        Ok(Self::from_original_policy(inner))
    }
}
impl<S, E, L> MlxSpeculativeSampling<S, E, L>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
{
    pub(super) fn original_snapshot_metadata(
        &self,
        target: Option<&MlxSpeculativeRandomState>,
        draft: Option<&MlxSpeculativeSeed>,
    ) -> Option<usize> {
        if !self.inner.is_original() || self.capture.is_some() || !self.interventions.is_empty() {
            return None;
        }
        let mut bytes = if self.inner.source().prepared_grammar_controller().is_some() {
            grammar::snapshot_metadata::<S, E, L>(&self.inner)?
        } else {
            controls::<S, E, L>(copy_plan(self.inner.source())?.metadata_bytes())?
        };
        if let Some(key) = target {
            if !key.memory_retention.owners().is_empty()
                || !self
                    .inner
                    .snapshot_context()?
                    .validate_key(key.value.original()?)
            {
                return None;
            }
            bytes = bytes.checked_add(numerical::SnapshotContext::key_controls()?)?;
        }
        if let Some(key) = draft {
            if !key.memory_retention.owners().is_empty()
                || !self
                    .inner
                    .snapshot_context()?
                    .validate_key(key.value.original()?)
            {
                return None;
            }
            bytes = bytes.checked_add(numerical::SnapshotContext::key_controls()?)?;
        }
        let parts = [
            size_of::<
                Result<
                    (
                        Self,
                        Option<MlxSpeculativeRandomState>,
                        Option<MlxSpeculativeSeed>,
                    ),
                    SpeculativeControlError,
                >,
            >(),
            size_of::<Option<&MlxSpeculativeRandomState>>(),
            size_of::<Option<&MlxSpeculativeSeed>>(),
            size_of::<Option<MlxSpeculativeRandomState>>(),
            size_of::<Option<MlxSpeculativeSeed>>(),
            size_of::<NativeMemoryRetention>(),
            size_of::<(
                Self,
                Option<MlxSpeculativeRandomState>,
                Option<MlxSpeculativeSeed>,
            )>(),
        ];
        parts
            .into_iter()
            .try_fold(bytes.checked_add(size_of_val(&parts))?, usize::checked_add)
    }
    pub(super) fn original_snapshot_bytes(
        &self,
        target: Option<&MlxSpeculativeRandomState>,
        draft: Option<&MlxSpeculativeSeed>,
    ) -> Option<u64> {
        let mut bytes = u64::try_from(self.original_snapshot_metadata(target, draft)?).ok()?;
        if let Some(source) = target {
            bytes = bytes.checked_add(
                self.inner
                    .snapshot_context()?
                    .key_bytes(source.value.original()?)?,
            )?;
        }
        if let Some(source) = draft {
            bytes = bytes.checked_add(
                self.inner
                    .snapshot_context()?
                    .key_bytes(source.value.original()?)?,
            )?;
        }
        Some(bytes)
    }
    pub(super) fn copy_snapshot(
        &self,
        target: Option<&MlxSpeculativeRandomState>,
        draft: Option<&MlxSpeculativeSeed>,
        host: HostPreparationAuthority,
    ) -> Result<
        (
            Self,
            Option<MlxSpeculativeRandomState>,
            Option<MlxSpeculativeSeed>,
        ),
        SpeculativeControlError,
    > {
        if !self.inner.is_original() && host.is_unmanaged() {
            return Ok((self.clone(), target.cloned(), draft.cloned()));
        }
        if host.is_unmanaged() || self.original_snapshot_metadata(target, draft).is_none() {
            return Err(SpeculativeControlError::Unsupported(
                "sampler snapshot needs a known paid policy and independent RNG copy",
            ));
        }
        let plan = copy_plan(self.inner.source());
        if plan.is_none() && self.inner.source().prepared_grammar_controller().is_none() {
            return Err(SpeculativeControlError::Unsupported(
                "sampler has no paid host copy",
            ));
        }
        let copy_key = |key: &numerical::OriginalNumericalKey| {
            self.inner
                .snapshot_context()
                .ok_or(SpeculativeControlError::Unsupported(
                    "original RNG source context unavailable",
                ))?
                .copy_key(key, &host)
                .map_err(|cause| numerical::SnapshotContext::failure(cause, &host))
        };
        let target = target
            .map(|source| {
                let key = source
                    .value
                    .original()
                    .ok_or(SpeculativeControlError::Unsupported(
                        "ordinary RNG source in original snapshot",
                    ))?;
                Ok::<_, SpeculativeControlError>(MlxSpeculativeRandomState {
                    value: KeyValue::Original(copy_key(key)?),
                    memory_retention: NativeMemoryRetention::default(),
                })
            })
            .transpose()?;
        let draft = draft
            .map(|source| {
                let key = source
                    .value
                    .original()
                    .ok_or(SpeculativeControlError::Unsupported(
                        "ordinary RNG source in original snapshot",
                    ))?;
                Ok::<_, SpeculativeControlError>(MlxSpeculativeSeed {
                    value: KeyValue::Original(copy_key(key)?),
                    memory_retention: NativeMemoryRetention::default(),
                })
            })
            .transpose()?;
        Ok((
            Self {
                inner: match plan {
                    Some(plan) => copy_original(
                        plan,
                        host,
                        self.inner.metadata_funding().cloned(),
                        self.inner.snapshot_context().cloned(),
                    ),
                    None => grammar::copy_snapshot::<S, E, L>(&self.inner, &host),
                }
                .map_err(SpeculativeControlError::Backend)?,
                capture: None,
                original_capture: self.original_capture.clone(),
                original_interventions: self.original_interventions.clone(),
                interventions: Vec::new(),
                memory_retention: None,
                error: std::marker::PhantomData,
            },
            target,
            draft,
        ))
    }
}

impl<S, E, L> MlxSpeculativeSampling<S, E, L>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
{
    /// Existing processed-value source validation precedes any provisional copy.
    /// Replacement occurs only after actual controller commitment succeeds.
    pub(super) fn commit_original(
        &mut self,
        value: &numerical::OriginalNumericalValue,
        token: u32,
        placement: SamplingPlacement,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<(), Error> {
        let (sources, environment) =
            context
                .original_numerical_for(placement)
                .ok_or(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ))?;
        self.inner.original_source()?;
        value.validate_consumer(sources)?;
        if self.inner.source().prepared_grammar_controller().is_some() {
            return self.commit_grammar_original(value, token, placement, context);
        }
        if self.inner.source().prepared_adaptive_commit().is_some() {
            return self.commit_adaptive_original(value, token, placement, context);
        }
        let Some(plan) = self.inner.source().prepared_controller() else {
            return numerical::commit_without_mutation(
                self.inner.source(),
                value,
                token,
                placement,
                context,
            );
        };
        numerical::validate_commit_value(value, token, placement, context)?;
        numerical::validate_controller_value(self.inner.source(), value, sources)?;
        CopyPlan::Controller(self.inner.source().prepared_controller().ok_or(
            Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound),
        )?)
        .validate_source(environment.pool())?;
        plan.validate_commit(token)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let bytes = controls::<S, E, L>(plan.commit_metadata_bytes())
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    eredu_nn::workspace::HostMetadataFunding,
                >()?)
            })
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
        let funding = value.validate_consumer(sources)?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let host = HostPreparationAuthority::retain(funding.clone());
        let source = plan
            .commit(token, host.clone())
            .map_err(|cause| Error::StorageSource(controller_failure(cause, &host)))?;
        let replacement = own_original(
            source,
            host,
            Some(funding.clone()),
            true,
            self.inner.snapshot_context().cloned(),
        );
        self.inner = replacement;
        Ok(())
    }
    /// Known plain prefix semantics, including the existing history checks.
    pub(super) fn original_prefix_is_complete(&self, history: &[u32]) -> Result<bool, Error> {
        self.inner.original_source()?;
        if self.inner.source().prepared_grammar_controller().is_some() {
            return self.grammar_prefix_original(history);
        }
        let funding = self.inner.metadata_funding().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))?;
        if let Some(plan) = self.inner.source().prepared_controller() {
            let parts = [
                std::mem::size_of::<Result<bool, PreparedControllerError>>(),
                std::mem::size_of_val(&plan),
                std::mem::size_of::<Result<bool, Error>>(),
                std::mem::size_of::<&[u32]>(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
            funding
                .reserve_metadata(bytes)
                .map_err(Error::WorkspacePlanning)?;
            plan.prefix_is_complete(history).map_err(|cause| {
                crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
            })
        } else {
            if self.inner.is_fixed_controller() {
                return Err(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                ));
            }
            self.inner.grammar_complete().ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::{ConfiguredTextSampler, GenerationSampler};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    struct Retires(Arc<AtomicBool>);
    impl Drop for Retires {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    fn authority(done: &Arc<AtomicBool>) -> HostPreparationAuthority {
        HostPreparationAuthority::retain(Retires(done.clone()))
    }
    #[test]
    fn original_sampler_aliases_snapshot_and_refusal_keep_independent_host_custody() {
        let source = ConfiguredTextSampler::Standard(
            GenerationSampler::new().with_generated_tokens([7, 9, 12]),
        );
        let source_done = Arc::new(AtomicBool::new(false));
        let copy_done = Arc::new(AtomicBool::new(false));
        let first = MlxSpeculativeSampling::<ConfiguredTextSampler> {
            inner: original(
                SpeculativeSampler::<MlxSamplingBackend>::prepared_host_copy(&source).unwrap(),
                authority(&source_done),
            ),
            capture: None,
            original_capture: None,
            original_interventions: None,
            interventions: Vec::new(),
            memory_retention: None,
            error: std::marker::PhantomData,
        };
        let alias = first.clone();
        let (copied, target, draft) = alias
            .copy_snapshot(None, None, authority(&copy_done))
            .unwrap();
        assert!(target.is_none() && draft.is_none());
        assert_eq!(copied.inner.source().history_len(), 3);
        assert_eq!(
            copied.inner.source().history_capacity(),
            source.history_capacity()
        );
        assert_eq!(copied.inner.grammar_complete(), Some(false));
        drop(first);
        assert!(!source_done.load(Ordering::SeqCst));
        drop(alias);
        assert!(source_done.load(Ordering::SeqCst));
        assert!(!copy_done.load(Ordering::SeqCst));
        let failure = copied.inner.capture_refusal().unwrap();
        let failure_alias = failure.clone();
        drop(copied);
        assert!(!copy_done.load(Ordering::SeqCst));
        assert!(matches!(failure.cause(), CaptureError::Unsupported(_)));
        drop(failure);
        assert!(!copy_done.load(Ordering::SeqCst));
        drop(failure_alias);
        assert!(copy_done.load(Ordering::SeqCst));
    }
}
