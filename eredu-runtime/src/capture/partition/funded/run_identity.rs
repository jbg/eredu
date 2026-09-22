//! Paid common setup and first actual forward identity, outside model state.
use super::*;
use std::alloc::Layout;
use std::sync::{Arc, OnceLock};

struct RunIdentity {
    artifact: String,
    execution: String,
    setup: crate::CommunicationSessionIdentity,
    overlay: Option<String>,
    first_epoch: OnceLock<DistributedCommitEpoch>,
    source: SharedCapturePlan,
    metadata: HostMetadataFunding,
}
/// Shared identity of one real capture run. The first reached forward survives
/// failed frames and restoration; a fresh fork binds its own later epoch. This
/// owns only descriptive source metadata, never native submission authority.
pub struct PreparedPartitionCaptureRunIdentity(Option<Arc<RunIdentity>>);
impl Clone for PreparedPartitionCaptureRunIdentity {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.inner())))
    }
}
impl Drop for PreparedPartitionCaptureRunIdentity {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // Every strong owner participates; no raw or weak handle escapes.
            // The final allocation retires before its payload releases funding.
            drop(Arc::into_inner(owner));
        }
    }
}
impl std::fmt::Debug for PreparedPartitionCaptureRunIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedPartitionCaptureRunIdentity")
            .field("setup", &self.inner().setup)
            .field("first_epoch", &self.inner().first_epoch.get())
            .finish_non_exhaustive()
    }
}
fn error(
    source: &SharedCapturePlan,
    metadata: &HostMetadataFunding,
    cause: Cause,
) -> PartitionCaptureProgramError {
    PartitionCaptureProgramError {
        cause,
        _source: source.clone(),
        _metadata: metadata.clone(),
    }
}
fn identity_control_bytes() -> Option<usize> {
    let allocation = Layout::new::<[usize; 2]>()
        .extend(Layout::new::<RunIdentity>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let parts = [
        allocation,
        size_of::<RunIdentity>() * 2,
        size_of::<Option<RunIdentity>>(),
        size_of::<Arc<RunIdentity>>(),
        size_of::<PreparedPartitionCaptureRunIdentity>() * 2,
        size_of::<Result<PreparedPartitionCaptureRunIdentity, PartitionCaptureProgramError>>(),
        size_of::<PartitionCaptureProgramError>(),
        size_of::<(
            &SharedCapturePlan,
            &str,
            &str,
            crate::CommunicationSessionIdentity,
            Option<&str>,
            &HostMetadataFunding,
        )>(),
        size_of::<Cause>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn context_control_bytes() -> Option<usize> {
    let parts = [
        size_of::<PartitionCaptureContext>() * 2,
        size_of::<Result<PartitionCaptureContext, Cause>>(),
        size_of::<Result<PartitionCaptureContext, PartitionCaptureProgramError>>(),
        size_of::<(
            &SharedCapturePlan,
            &str,
            &str,
            crate::CommunicationSessionIdentity,
            Option<&str>,
            CapturePhase,
            u64,
            &HostMetadataFunding,
        )>(),
        size_of::<(&PreparedPartitionCaptureRunIdentity, CapturePhase, u64)>(),
        size_of::<std::array::IntoIter<&str, 2>>(),
        size_of::<Option<&str>>(),
        size_of::<Option<CaptureInvocationShape>>(),
        size_of::<Option<eredu_core::capture::PartitionCaptureInvocationWindow>>(),
        size_of::<Result<CaptureInvocationShape, CaptureError>>(),
        size_of::<(
            &SharedCapturePlan,
            &str,
            &str,
            crate::CommunicationSessionIdentity,
            Option<&str>,
            CapturePhase,
            u64,
            Option<CaptureInvocationShape>,
            Option<eredu_core::capture::PartitionCaptureInvocationWindow>,
            &HostMetadataFunding,
        )>(),
        size_of::<(
            &PreparedPartitionCaptureRunIdentity,
            CapturePhase,
            u64,
            Option<CaptureInvocationShape>,
            Option<eredu_core::capture::PartitionCaptureInvocationWindow>,
            &HostMetadataFunding,
        )>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn bind_control_bytes() -> usize {
    size_of::<PreparedPartitionCaptureRunIdentity>() * 2
        + size_of::<Result<PreparedPartitionCaptureRunIdentity, PartitionCaptureProgramError>>()
        + size_of::<PartitionCaptureProgramError>()
        + size_of::<Cause>()
        + size_of::<(
            &mut crate::capture::FundedCaptureSession,
            &str,
            &str,
            crate::CommunicationSessionIdentity,
            Option<&str>,
            &HostMetadataFunding,
        )>()
}
fn formatted_label_length(value: impl std::fmt::Display) -> Option<usize> {
    struct Count(usize);
    impl std::fmt::Write for Count {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            self.0 = self.0.checked_add(value.len()).ok_or(std::fmt::Error)?;
            Ok(())
        }
    }
    let mut count = Count(0);
    std::fmt::write(&mut count, format_args!("{value}")).ok()?;
    Some(count.0)
}
fn formatted_label_bytes(value: impl std::fmt::Display) -> Option<usize> {
    eredu_nn::workspace::WorkspaceContext::metadata_string_bytes(formatted_label_length(value)?)
}
fn loaded_label_bytes(artifact: &str, execution: &str, overlay: Option<&str>) -> Option<usize> {
    [artifact, execution]
        .into_iter()
        .chain(overlay)
        .try_fold(0usize, |bytes, label| {
            if label.is_empty() || label.len() > 256 {
                return None;
            }
            bytes.checked_add(
                eredu_nn::workspace::WorkspaceContext::metadata_string_bytes(label.len())?,
            )
        })
}
fn context_label_bytes(
    source: &SharedCapturePlan,
    artifact: &str,
    execution: &str,
    setup: crate::CommunicationSessionIdentity,
    overlay: Option<&str>,
) -> Option<usize> {
    if setup.participant_count() == 0 {
        return None;
    }
    loaded_label_bytes(artifact, execution, overlay)?
        .checked_add(formatted_label_bytes(setup)?)?
        .checked_add(formatted_label_bytes(source.admission().identity())?)
}
impl PreparedPartitionCaptureRunIdentity {
    fn inner(&self) -> &Arc<RunIdentity> {
        self.0.as_ref().expect("live partition run identity")
    }
    /// Full first binding through `FundedCaptureSession`, including the fixed
    /// shared identity allocation and exact loaded label destinations. A reused
    /// binding only consumes the outer entry portion of this allowance.
    pub fn preparation_metadata_bytes(
        _source: &SharedCapturePlan,
        artifact: &str,
        execution: &str,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<&str>,
    ) -> Option<usize> {
        if setup.participant_count() == 0 {
            return None;
        }
        bind_control_bytes()
            .checked_add(identity_control_bytes()?)?
            .checked_add(loaded_label_bytes(artifact, execution, overlay)?)
    }
    /// Exact source-context frames and labels for the Host preparation entry.
    /// Independently admitted invocation axes use the same fixed transports.
    pub fn host_context_metadata_bytes(
        source: &SharedCapturePlan,
        artifact: &str,
        execution: &str,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<&str>,
    ) -> Option<usize> {
        context_control_bytes()?.checked_add(context_label_bytes(
            source, artifact, execution, setup, overlay,
        )?)
    }
    /// UTF-8 length written by the same run-name formatter, without allocation
    /// or binding an epoch. Receipt copies can use this exact label length.
    pub fn epoch_run_name_length(
        setup: crate::CommunicationSessionIdentity,
        epoch: DistributedCommitEpoch,
    ) -> Option<usize> {
        formatted_label_length(session::SessionRunName(setup, epoch))
    }
    /// The actual epoch-label writer destination and its formatter controls.
    /// This query neither binds the run's first epoch nor changes its custody.
    pub fn epoch_metadata_bytes(
        setup: crate::CommunicationSessionIdentity,
        epoch: DistributedCommitEpoch,
    ) -> Option<usize> {
        formatted_label_bytes(session::SessionRunName(setup, epoch))
    }
    fn prepare(
        source: &SharedCapturePlan,
        artifact: &str,
        execution: &str,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<&str>,
        metadata: &HostMetadataFunding,
    ) -> Result<Self, PartitionCaptureProgramError> {
        let fail = |cause| error(source, metadata, cause);
        let bytes = identity_control_bytes()
            .ok_or_else(|| fail(Cause::Source("run identity controls overflow")))?;
        metadata
            .reserve_metadata(bytes)
            .map_err(|cause| fail(cause.into()))?;
        if setup.participant_count() == 0
            || [artifact, execution]
                .into_iter()
                .chain(overlay)
                .any(|label| label.is_empty() || label.len() > 256)
        {
            return Err(fail(Cause::Source(
                "run identity differs from bounded source labels",
            )));
        }
        let artifact = metadata
            .metadata_string(format_args!("{artifact}"))
            .map_err(|cause| fail(cause.into()))?;
        let execution = metadata
            .metadata_string(format_args!("{execution}"))
            .map_err(|cause| fail(cause.into()))?;
        let overlay = overlay
            .map(|value| metadata.metadata_string(format_args!("{value}")))
            .transpose()
            .map_err(|cause| fail(cause.into()))?;
        Ok(Self(Some(Arc::new(RunIdentity {
            artifact,
            execution,
            setup,
            overlay,
            first_epoch: OnceLock::new(),
            source: source.clone(),
            metadata: metadata.clone(),
        }))))
    }
    fn matches(
        &self,
        source: &SharedCapturePlan,
        artifact: &str,
        execution: &str,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<&str>,
    ) -> bool {
        self.inner().source.same_storage(source)
            && self.inner().artifact == artifact
            && self.inner().execution == execution
            && self.inner().setup == setup
            && self.inner().overlay.as_deref() == overlay
    }
    pub(super) fn bind_epoch(&self, epoch: DistributedCommitEpoch) -> Result<String, Cause> {
        // First real attempt is consumed before any later destination failure.
        let first = *self.inner().first_epoch.get_or_init(|| epoch);
        self.inner()
            .metadata
            .metadata_string(format_args!(
                "{}",
                session::SessionRunName(self.inner().setup, first)
            ))
            .map_err(Into::into)
    }
    fn context(
        &self,
        phase: CapturePhase,
        prediction: u64,
        metadata: &HostMetadataFunding,
    ) -> Result<PartitionCaptureContext, Cause> {
        source_context(
            &self.inner().source,
            &self.inner().artifact,
            &self.inner().execution,
            self.inner().setup,
            self.inner().overlay.as_deref(),
            phase,
            prediction,
            metadata,
        )
    }
    fn context_at(
        &self,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        window: Option<eredu_core::capture::PartitionCaptureInvocationWindow>,
        metadata: &HostMetadataFunding,
    ) -> Result<PartitionCaptureContext, Cause> {
        let mut context = self.context(phase, prediction, metadata)?;
        self.inner()
            .source
            .admission()
            .geometry_at(phase, prediction, invocation)?;
        context.invocation = invocation;
        context.invocation_window = window;
        context.validate()?;
        Ok(context)
    }
    /// Descriptive pre-epoch context for original Host preparation. Uses the
    /// same loaded labels as the eventual program, but never binds a run epoch
    /// or grants receipt, native, transport or capture authority.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_host_context(
        source: &SharedCapturePlan,
        artifact: &str,
        execution: &str,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<&str>,
        phase: CapturePhase,
        prediction: u64,
        metadata: &HostMetadataFunding,
    ) -> Result<PartitionCaptureContext, PartitionCaptureProgramError> {
        source_context(
            source, artifact, execution, setup, overlay, phase, prediction, metadata,
        )
        .map_err(|cause| error(source, metadata, cause))
    }

    /// Descriptive context carrying the exact independently admitted physical axes.
    /// Shape validation creates no frame, receipt, transport or native authority.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_host_context_at(
        source: &SharedCapturePlan,
        artifact: &str,
        execution: &str,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<&str>,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        window: Option<eredu_core::capture::PartitionCaptureInvocationWindow>,
        metadata: &HostMetadataFunding,
    ) -> Result<PartitionCaptureContext, PartitionCaptureProgramError> {
        let mut context = Self::prepare_host_context(
            source, artifact, execution, setup, overlay, phase, prediction, metadata,
        )?;
        source
            .admission()
            .geometry_at(phase, prediction, invocation)
            .map_err(|cause| error(source, metadata, cause.into()))?;
        context.invocation = invocation;
        context.invocation_window = window;
        context
            .validate()
            .map_err(|cause| error(source, metadata, cause.into()))?;
        Ok(context)
    }
}
#[allow(clippy::too_many_arguments)]
fn source_context(
    source: &SharedCapturePlan,
    artifact: &str,
    execution: &str,
    setup: crate::CommunicationSessionIdentity,
    overlay: Option<&str>,
    phase: CapturePhase,
    prediction: u64,
    metadata: &HostMetadataFunding,
) -> Result<PartitionCaptureContext, Cause> {
    metadata.reserve_metadata(
        context_control_bytes().ok_or(Cause::Source("source context controls overflow"))?,
    )?;
    if setup.participant_count() == 0
        || [artifact, execution]
            .into_iter()
            .chain(overlay)
            .any(|label| label.is_empty() || label.len() > 256)
    {
        return Err(Cause::Source(
            "source context differs from bounded loaded labels",
        ));
    }
    Ok(PartitionCaptureContext {
        artifact_identity: metadata.metadata_string(format_args!("{artifact}"))?,
        execution_identity: metadata.metadata_string(format_args!("{execution}"))?,
        run_identity: metadata.metadata_string(format_args!("{setup}"))?,
        overlay_identity: overlay
            .map(|v| metadata.metadata_string(format_args!("{v}")))
            .transpose()?,
        capture_plan_identity: metadata
            .metadata_string(format_args!("{}", source.admission().identity()))?,
        selection_index: 0,
        phase,
        prediction,
        forward_epoch: 0,
        invocation: None,
        invocation_window: None,
    })
}
impl crate::capture::FundedCaptureSession {
    /// Bind the actual loaded setup/parameter labels once. The native caller
    /// must authenticate these facts; this constructor supplies no native grant.
    pub fn prepare_partition_run_identity(
        &mut self,
        artifact: &str,
        execution: &str,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<&str>,
        metadata: &HostMetadataFunding,
    ) -> Result<PreparedPartitionCaptureRunIdentity, PartitionCaptureProgramError> {
        let source = self.source();
        metadata
            .reserve_metadata(bind_control_bytes())
            .map_err(|cause| error(source, metadata, cause.into()))?;
        if let Some(run) = &self.partition_run {
            if !run.matches(source, artifact, execution, setup, overlay) {
                return Err(error(
                    source,
                    metadata,
                    Cause::Source("run identity changed after binding"),
                ));
            }
            return Ok(run.clone());
        }
        let run = PreparedPartitionCaptureRunIdentity::prepare(
            source, artifact, execution, setup, overlay, metadata,
        )?;
        self.partition_run = Some(run.clone());
        Ok(run)
    }
}
impl<'t, T: PartitionCaptureTransport> PreparedPartitionCaptureProgram<'t, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    /// Use the same receipt constructor with a paid live-run identity. Its first
    /// actual epoch is bound by prepare(), before any model/capture hook.
    #[allow(clippy::too_many_arguments)]
    pub fn new_for_run(
        transport: &'t T,
        source: &SharedCapturePlan,
        run: &PreparedPartitionCaptureRunIdentity,
        phase: CapturePhase,
        prediction: u64,
        rows: &[PartitionCaptureProducerSource],
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
    ) -> Result<Self, PartitionCaptureProgramError> {
        let fail = |cause| error(source, metadata, cause);
        metadata
            .reserve_metadata(
                size_of::<Self>() * 2
                    + size_of::<PreparedPartitionCaptureRunIdentity>()
                    + size_of::<Result<Self, PartitionCaptureProgramError>>()
                    + size_of::<PartitionCaptureContext>()
                    + size_of::<(
                        &T,
                        &SharedCapturePlan,
                        &PreparedPartitionCaptureRunIdentity,
                        CapturePhase,
                        u64,
                        &[PartitionCaptureProducerSource],
                        PartitionCaptureReceiptLimits,
                        &HostMetadataFunding,
                    )>(),
            )
            .map_err(|cause| fail(cause.into()))?;
        if !run.inner().source.same_storage(source)
            || run.inner().setup.participant_count() != transport.participant_count()
        {
            return Err(fail(Cause::Source(
                "run source or participant world differs",
            )));
        }
        let context = run.context(phase, prediction, metadata).map_err(fail)?;
        let mut program = Self::new(transport, source, &context, rows, limits, metadata)?;
        program.run_identity = Some(run.clone());
        Ok(program)
    }
}

impl<'t, T: PartitionCaptureTransport> PreparedPartitionCaptureProgram<'t, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    /// Same loaded run/epoch owner for complete and projected source rows.
    #[allow(clippy::too_many_arguments)]
    pub fn new_selected_for_run(
        transport: &'t T,
        source: &SharedCapturePlan,
        run: &PreparedPartitionCaptureRunIdentity,
        phase: CapturePhase,
        prediction: u64,
        rows: &[PreparedPartitionCaptureRow<'_>],
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
    ) -> Result<Self, PartitionCaptureProgramError> {
        Self::new_selected_for_run_at(
            transport, source, run, phase, prediction, None, None, rows, limits, metadata,
        )
    }
    fn selected_for_run_control_bytes() -> usize {
        size_of::<Self>() * 2
            + size_of::<PreparedPartitionCaptureRunIdentity>()
            + size_of::<Result<Self, PartitionCaptureProgramError>>()
            + size_of::<PartitionCaptureContext>()
            + size_of::<(
                &T,
                &SharedCapturePlan,
                &PreparedPartitionCaptureRunIdentity,
                CapturePhase,
                u64,
                Option<CaptureInvocationShape>,
                Option<eredu_core::capture::PartitionCaptureInvocationWindow>,
                &[PreparedPartitionCaptureRow<'_>],
                PartitionCaptureReceiptLimits,
                &HostMetadataFunding,
            )>()
    }
    /// Metadata consumed by `new_selected_for_run_at`, including its exact
    /// selected-row destinations and both source-context label copies. This
    /// does not prepare a run identity, receipt, projection, or native worker.
    #[allow(clippy::too_many_arguments)]
    pub fn selected_for_run_metadata_bytes(
        source: &SharedCapturePlan,
        artifact: &str,
        execution: &str,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<&str>,
        rows: usize,
    ) -> Option<usize> {
        if rows != source.admission().plan().selections.len() {
            return None;
        }
        let labels = context_label_bytes(source, artifact, execution, setup, overlay)?;
        Self::selected_for_run_control_bytes()
            .checked_add(context_control_bytes()?)?
            .checked_add(Self::selected_rows_metadata_bytes(rows)?)?
            .checked_add(labels.checked_mul(2)?)
    }
    /// Exact selected-row constructor with a previously prepared context.
    /// This includes its one label copy and actual row directories, excluding
    /// context preparation and any run/receipt/native source construction.
    pub fn selected_host_context_metadata_bytes(
        source: &SharedCapturePlan,
        artifact: &str,
        execution: &str,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<&str>,
        rows: usize,
    ) -> Option<usize> {
        if rows != source.admission().plan().selections.len() {
            return None;
        }
        Self::selected_rows_metadata_bytes(rows)?.checked_add(context_label_bytes(
            source, artifact, execution, setup, overlay,
        )?)
    }
    /// The same loaded run and original row owners with independently admitted axes.
    /// The actual scheduled frame must carry precisely this same invocation.
    #[allow(clippy::too_many_arguments)]
    pub fn new_selected_for_run_at(
        transport: &'t T,
        source: &SharedCapturePlan,
        run: &PreparedPartitionCaptureRunIdentity,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        window: Option<eredu_core::capture::PartitionCaptureInvocationWindow>,
        rows: &[PreparedPartitionCaptureRow<'_>],
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
    ) -> Result<Self, PartitionCaptureProgramError> {
        let fail = |cause| error(source, metadata, cause);
        metadata
            .reserve_metadata(Self::selected_for_run_control_bytes())
            .map_err(|cause| fail(cause.into()))?;
        if !run.inner().source.same_storage(source)
            || run.inner().setup.participant_count() != transport.participant_count()
        {
            return Err(fail(Cause::Source(
                "run source or participant world differs",
            )));
        }
        let context = run
            .context_at(phase, prediction, invocation, window, metadata)
            .map_err(fail)?;
        let mut result = Self::new_selected(transport, source, &context, rows, limits, metadata)?;
        result.run_identity = Some(run.clone());
        Ok(result)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;

#[cfg(test)]
mod metadata_query_tests {
    use super::*;
    use eredu_core::consensus::ConsensusTransport;
    use eredu_core::{
        DescriptionCompleteness, HostMetadataAccount, ObservationCatalog, ObservationSupportReport,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Singleton;
    impl ConsensusTransport for Singleton {
        type Error = std::convert::Infallible;
        fn participant_count(&self) -> usize {
            1
        }
        fn all_gather_words(&self, words: &[u32]) -> Result<Vec<u32>, Self::Error> {
            Ok(words.to_vec())
        }
    }
    #[derive(Debug)]
    struct Account {
        used: Arc<AtomicUsize>,
        limit: usize,
        retired: Option<Arc<AtomicUsize>>,
    }
    impl Drop for Account {
        fn drop(&mut self) {
            if let Some(retired) = &self.retired {
                retired.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            self.used
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                    used.checked_add(bytes).filter(|&total| total <= self.limit)
                })
                .map(|_| ())
                .map_err(|used| HostMetadataFundingError::Capacity {
                    required: bytes as u64,
                    available: (self.limit - used) as u64,
                })
        }
    }
    fn funding(bytes: usize) -> (HostMetadataFunding, Arc<AtomicUsize>, usize) {
        let baseline = HostMetadataFunding::constructor_bytes::<Account>().unwrap();
        let used = Arc::new(AtomicUsize::new(0));
        let funding = HostMetadataFunding::new(Account {
            used: used.clone(),
            limit: baseline + bytes,
            retired: None,
        })
        .unwrap();
        (funding, used, baseline)
    }
    fn source() -> (SharedCapturePlan, crate::CommunicationSessionIdentity) {
        let setup = crate::establish_communication_session(
            &Singleton,
            &crate::CommunicationManifest::new(1, 0, vec![], vec![]).unwrap(),
            Some([3; 32]),
        )
        .unwrap()
        .identity();
        let catalog = ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Complete,
            points: vec![],
        };
        let support = ObservationSupportReport {
            schema_version: 1,
            capture: Default::default(),
            points: vec![],
        };
        let usage = crate::capture::policy::frame_usage().unwrap();
        let plan = CapturePlan {
            schema_version: 1,
            selections: vec![],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage.checked_mul(2).unwrap(),
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &catalog,
            &support,
            &support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 2,
                max_predictions: 2,
            },
        )
        .unwrap();
        (SharedCapturePlan::new(plan), setup)
    }
    #[test]
    fn host_context_quote_funds_actual_unicode_labels_and_one_short_refuses() {
        let (source, setup) = source();
        let artifact = "模型";
        let execution = "selected worker";
        let overlay = Some("échantillon");
        let bytes = PreparedPartitionCaptureRunIdentity::host_context_metadata_bytes(
            &source, artifact, execution, setup, overlay,
        )
        .unwrap();
        for short in [false, true] {
            let (metadata, used, baseline) = funding(bytes - usize::from(short));
            let result = PreparedPartitionCaptureRunIdentity::prepare_host_context_at(
                &source,
                artifact,
                execution,
                setup,
                overlay,
                CapturePhase::Prefill,
                0,
                None,
                None,
                &metadata,
            );
            if short {
                assert!(result.is_err());
                assert!(used.load(Ordering::SeqCst) - baseline < bytes);
            } else {
                let context = result.unwrap();
                assert_eq!(context.artifact_identity, artifact);
                assert_eq!(context.overlay_identity.as_deref(), overlay);
                assert_eq!(context.run_identity, format!("{setup}"));
                assert_eq!(used.load(Ordering::SeqCst) - baseline, bytes);
            }
        }
        assert_eq!(
            PreparedPartitionCaptureRunIdentity::epoch_run_name_length(
                setup,
                DistributedCommitEpoch::FIRST
            ),
            Some(
                format!(
                    "{}",
                    session::SessionRunName(setup, DistributedCommitEpoch::FIRST)
                )
                .len()
            )
        );
    }
    #[test]
    fn run_identity_aliases_retain_source_account_and_first_epoch() {
        let (source, setup) = source();
        let baseline = HostMetadataFunding::constructor_bytes::<Account>().unwrap();
        let allowance = PreparedPartitionCaptureRunIdentity::preparation_metadata_bytes(
            &source,
            "loaded artifact",
            "actual worker",
            setup,
            None,
        )
        .unwrap()
        .checked_add(
            PreparedPartitionCaptureRunIdentity::epoch_metadata_bytes(
                setup,
                DistributedCommitEpoch::FIRST,
            )
            .unwrap(),
        )
        .unwrap();
        let used = Arc::new(AtomicUsize::new(0));
        let retired = Arc::new(AtomicUsize::new(0));
        let metadata = HostMetadataFunding::new(Account {
            used: used.clone(),
            limit: baseline + allowance,
            retired: Some(retired.clone()),
        })
        .unwrap();
        let run = PreparedPartitionCaptureRunIdentity::prepare(
            &source,
            "loaded artifact",
            "actual worker",
            setup,
            None,
            &metadata,
        )
        .unwrap();
        let spent = used.load(Ordering::SeqCst);
        let alias = run.clone();
        assert_eq!(used.load(Ordering::SeqCst), spent);
        drop(metadata);
        drop(run);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert!(alias.matches(&source, "loaded artifact", "actual worker", setup, None));
        let name = alias.bind_epoch(DistributedCommitEpoch::FIRST).unwrap();
        assert_eq!(
            name,
            format!(
                "{}",
                session::SessionRunName(setup, DistributedCommitEpoch::FIRST)
            )
        );
        drop(name);
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
        assert!(used.load(Ordering::SeqCst) <= baseline + allowance);
    }
}
