//! Paid common setup and first actual forward identity, outside model state.
use super::*;
use std::sync::{Arc, OnceLock};
use std::alloc::Layout;

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
#[derive(Clone)]
pub struct PreparedPartitionCaptureRunIdentity(Arc<RunIdentity>);
impl std::fmt::Debug for PreparedPartitionCaptureRunIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedPartitionCaptureRunIdentity")
            .field("setup", &self.0.setup).field("first_epoch", &self.0.first_epoch.get())
            .finish_non_exhaustive()
    }
}
fn error(source: &SharedCapturePlan, metadata: &HostMetadataFunding, cause: Cause)
    -> PartitionCaptureProgramError {
    PartitionCaptureProgramError { cause, _source: source.clone(), _metadata: metadata.clone() }
}
impl PreparedPartitionCaptureRunIdentity {
    fn prepare(source: &SharedCapturePlan, artifact: &str, execution: &str,
        setup: crate::CommunicationSessionIdentity, overlay: Option<&str>,
        metadata: &HostMetadataFunding) -> Result<Self, PartitionCaptureProgramError> {
        let fail = |cause| error(source, metadata, cause);
        let allocation = Layout::new::<[usize; 2]>().extend(Layout::new::<RunIdentity>())
            .map_err(|_| fail(Cause::Source("run identity allocation overflow")))?.0.pad_to_align().size();
        let parts = [allocation, size_of::<RunIdentity>() * 2, size_of::<Self>() * 2,
            size_of::<Result<Self, PartitionCaptureProgramError>>(), size_of::<PartitionCaptureProgramError>(),
            size_of::<(&SharedCapturePlan, &str, &str, crate::CommunicationSessionIdentity,
                Option<&str>, &HostMetadataFunding)>(), size_of::<Cause>()];
        let bytes = parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| fail(Cause::Source("run identity controls overflow")))?;
        metadata.reserve_metadata(bytes).map_err(|cause| fail(cause.into()))?;
        if setup.participant_count() == 0 || [artifact, execution].into_iter().chain(overlay)
            .any(|label| label.is_empty() || label.len() > 256) {
            return Err(fail(Cause::Source("run identity differs from bounded source labels")));
        }
        let artifact = metadata.metadata_string(format_args!("{artifact}"))
            .map_err(|cause| fail(cause.into()))?;
        let execution = metadata.metadata_string(format_args!("{execution}"))
            .map_err(|cause| fail(cause.into()))?;
        let overlay = overlay.map(|value| metadata.metadata_string(format_args!("{value}")))
            .transpose().map_err(|cause| fail(cause.into()))?;
        Ok(Self(Arc::new(RunIdentity { artifact, execution, setup, overlay,
            first_epoch: OnceLock::new(), source: source.clone(), metadata: metadata.clone() })))
    }
    fn matches(&self, source: &SharedCapturePlan, artifact: &str, execution: &str,
        setup: crate::CommunicationSessionIdentity, overlay: Option<&str>) -> bool {
        self.0.source.same_storage(source) && self.0.artifact == artifact
            && self.0.execution == execution && self.0.setup == setup
            && self.0.overlay.as_deref() == overlay
    }
    pub(super) fn bind_epoch(&self, epoch: DistributedCommitEpoch) -> Result<String, Cause> {
        // First real attempt is consumed before any later destination failure.
        let first = *self.0.first_epoch.get_or_init(|| epoch);
        self.0.metadata.metadata_string(format_args!("{}", session::SessionRunName(self.0.setup, first)))
            .map_err(Into::into)
    }
    fn context(&self, phase: CapturePhase, prediction: u64,
        metadata: &HostMetadataFunding) -> Result<PartitionCaptureContext, Cause> {
        source_context(&self.0.source,&self.0.artifact,&self.0.execution,self.0.setup,
            self.0.overlay.as_deref(),phase,prediction,metadata)
    }
    /// Descriptive pre-epoch context for original Host preparation. Uses the
    /// same loaded labels as the eventual program, but never binds a run epoch
    /// or grants receipt, native, transport or capture authority.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_host_context(source:&SharedCapturePlan,artifact:&str,execution:&str,
        setup:crate::CommunicationSessionIdentity,overlay:Option<&str>,phase:CapturePhase,
        prediction:u64,metadata:&HostMetadataFunding)->Result<PartitionCaptureContext,PartitionCaptureProgramError> {
        source_context(source,artifact,execution,setup,overlay,phase,prediction,metadata)
            .map_err(|cause|error(source,metadata,cause))
    }

}
#[allow(clippy::too_many_arguments)]
fn source_context(source:&SharedCapturePlan,artifact:&str,execution:&str,
    setup:crate::CommunicationSessionIdentity,overlay:Option<&str>,phase:CapturePhase,
    prediction:u64,metadata:&HostMetadataFunding)->Result<PartitionCaptureContext,Cause> {
    let parts=[size_of::<PartitionCaptureContext>()*2,size_of::<Result<PartitionCaptureContext,Cause>>(),
        size_of::<Result<PartitionCaptureContext,PartitionCaptureProgramError>>(),
        size_of::<(&SharedCapturePlan,&str,&str,crate::CommunicationSessionIdentity,Option<&str>,CapturePhase,u64,&HostMetadataFunding)>(),
        size_of::<(&PreparedPartitionCaptureRunIdentity,CapturePhase,u64)>(),
        size_of::<std::array::IntoIter<&str,2>>(),size_of::<Option<&str>>()];
    metadata.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(Cause::Source("source context controls overflow"))?)?;
    if setup.participant_count()==0||[artifact,execution].into_iter().chain(overlay)
        .any(|label|label.is_empty()||label.len()>256){
        return Err(Cause::Source("source context differs from bounded loaded labels"));
    }
    Ok(PartitionCaptureContext{
        artifact_identity:metadata.metadata_string(format_args!("{artifact}"))?,
        execution_identity:metadata.metadata_string(format_args!("{execution}"))?,
        run_identity:metadata.metadata_string(format_args!("{setup}"))?,
        overlay_identity:overlay.map(|v|metadata.metadata_string(format_args!("{v}"))).transpose()?,
        capture_plan_identity:metadata.metadata_string(format_args!("{}",source.admission().identity()))?,
        selection_index:0,phase,prediction,forward_epoch:0,invocation:None,
    })
}
impl crate::capture::FundedCaptureSession {
    /// Bind the actual loaded setup/parameter labels once. The native caller
    /// must authenticate these facts; this constructor supplies no native grant.
    pub fn prepare_partition_run_identity(&mut self, artifact: &str, execution: &str,
        setup: crate::CommunicationSessionIdentity, overlay: Option<&str>, metadata: &HostMetadataFunding)
        -> Result<PreparedPartitionCaptureRunIdentity, PartitionCaptureProgramError> {
        let source = self.source();
        metadata.reserve_metadata(size_of::<PreparedPartitionCaptureRunIdentity>() * 2
            + size_of::<Result<PreparedPartitionCaptureRunIdentity, PartitionCaptureProgramError>>()
            + size_of::<PartitionCaptureProgramError>() + size_of::<Cause>()
            + size_of::<(&mut Self, &str, &str, crate::CommunicationSessionIdentity, Option<&str>, &HostMetadataFunding)>())
            .map_err(|cause| error(source, metadata, cause.into()))?;
        if let Some(run) = &self.partition_run {
            if !run.matches(source, artifact, execution, setup, overlay) {
                return Err(error(source, metadata, Cause::Source("run identity changed after binding")));
            }
            return Ok(run.clone());
        }
        let run = PreparedPartitionCaptureRunIdentity::prepare(source, artifact, execution, setup, overlay, metadata)?;
        self.partition_run = Some(run.clone());
        Ok(run)
    }
}
impl<'t, T: PartitionCaptureTransport> PreparedPartitionCaptureProgram<'t, T>
where T::Error: Send + Sync + 'static, <T::Completion as Completion>::Error: Send + Sync + 'static {
    /// Use the same receipt constructor with a paid live-run identity. Its first
    /// actual epoch is bound by prepare(), before any model/capture hook.
    #[allow(clippy::too_many_arguments)]
    pub fn new_for_run(transport: &'t T, source: &SharedCapturePlan,
        run: &PreparedPartitionCaptureRunIdentity, phase: CapturePhase, prediction: u64,
        rows: &[PartitionCaptureProducerSource], limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding) -> Result<Self, PartitionCaptureProgramError> {
        let fail = |cause| error(source, metadata, cause);
        metadata.reserve_metadata(size_of::<Self>() * 2 + size_of::<PreparedPartitionCaptureRunIdentity>()
            + size_of::<Result<Self, PartitionCaptureProgramError>>() + size_of::<PartitionCaptureContext>()
            + size_of::<(&T, &SharedCapturePlan, &PreparedPartitionCaptureRunIdentity, CapturePhase, u64,
                &[PartitionCaptureProducerSource], PartitionCaptureReceiptLimits, &HostMetadataFunding)>())
            .map_err(|cause| fail(cause.into()))?;
        if !run.0.source.same_storage(source) || run.0.setup.participant_count() != transport.participant_count() {
            return Err(fail(Cause::Source("run source or participant world differs")));
        }
        let context = run.context(phase, prediction, metadata).map_err(fail)?;
        let mut program = Self::new(transport, source, &context, rows, limits, metadata)?;
        program.run_identity = Some(run.clone());
        Ok(program)
    }
}

impl<'t,T:PartitionCaptureTransport> PreparedPartitionCaptureProgram<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    /// Same loaded run/epoch owner for complete and projected source rows.
    #[allow(clippy::too_many_arguments)]
    pub fn new_selected_for_run(transport:&'t T,source:&SharedCapturePlan,
        run:&PreparedPartitionCaptureRunIdentity,phase:CapturePhase,prediction:u64,
        rows:&[PreparedPartitionCaptureRow<'_>],limits:PartitionCaptureReceiptLimits,
        metadata:&HostMetadataFunding)->Result<Self,PartitionCaptureProgramError> {
        let fail=|cause|error(source,metadata,cause);
        metadata.reserve_metadata(size_of::<Self>()*2+size_of::<PreparedPartitionCaptureRunIdentity>()
            +size_of::<Result<Self,PartitionCaptureProgramError>>()+size_of::<PartitionCaptureContext>()
            +size_of::<(&T,&SharedCapturePlan,&PreparedPartitionCaptureRunIdentity,CapturePhase,u64,
                &[PreparedPartitionCaptureRow<'_>],PartitionCaptureReceiptLimits,&HostMetadataFunding)>())
            .map_err(|cause|fail(cause.into()))?;
        if !run.0.source.same_storage(source)||run.0.setup.participant_count()!=transport.participant_count(){
            return Err(fail(Cause::Source("run source or participant world differs")));
        }
        let context=run.context(phase,prediction,metadata).map_err(fail)?;
        let mut result=Self::new_selected(transport,source,&context,rows,limits,metadata)?;
        result.run_identity=Some(run.clone());Ok(result)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
