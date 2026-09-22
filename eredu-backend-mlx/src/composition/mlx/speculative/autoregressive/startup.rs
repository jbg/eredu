//! Paid independent startup from the actual empty or populated canonical state.
use super::*;
use crate::backend::runtime::cache::state::{
    MlxHybridState, MlxKeyValueState, MlxPoolingAttentionState,
};
use crate::composition::mlx::replicated_text::ResidentResetProfile;
use eredu_core::HostPreparationAuthority;
use eredu_runtime::working_memory::{
    OriginalSpeculativeStartup, PreparedResidentEmptyState, ResidentEmptyStateError,
    WorkingMemoryError,
};
use safemlx::{StreamCopyError, StreamCopyPlan};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::atomic::AtomicUsize,
};

enum StatePlan<'a> {
    KeyValue(PreparedResidentEmptyState<'a, MlxKeyValueState>),
    Hybrid(PreparedResidentEmptyState<'a, MlxHybridState>),
    Pooling(PreparedResidentEmptyState<'a, MlxPoolingAttentionState>),
    Copy(crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'a>),
}
impl StatePlan<'_> {
    fn required_bytes(&self) -> Option<u64> {
        let (bytes, boxed) = match self {
            Self::KeyValue(plan) => (plan.required_bytes(), size_of::<MlxKeyValueState>()),
            Self::Hybrid(plan) => (plan.required_bytes(), size_of::<MlxHybridState>()),
            Self::Pooling(plan) => (plan.required_bytes(), size_of::<MlxPoolingAttentionState>()),
            // The shared source copy pays its tables, outer box and numerical
            // account independently. This startup keeps only its stream costs.
            Self::Copy(_) => (0, 0),
        };
        bytes.checked_add(u64::try_from(boxed).ok()?)
    }
    fn construct(
        self,
        host: &HostPreparationAuthority,
        sources: &AutoregressiveSourcePair,
        model: &Executable,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
    ) -> Result<MlxPredictionTargetState, Error> {
        match self {
            Self::KeyValue(plan) => plan
                .construct(host)
                .map(MlxPredictionTargetState::new)
                .map_err(|cause| sources.retain_startup_error(cause)),
            Self::Hybrid(plan) => plan
                .construct(host)
                .map(MlxPredictionTargetState::new)
                .map_err(|cause| sources.retain_startup_error(cause)),
            Self::Pooling(plan) => plan
                .construct(host)
                .map(MlxPredictionTargetState::new)
                .map_err(|cause| sources.retain_startup_error(cause)),
            Self::Copy(plan) => {
                let initialized = model.erased().prefill_roots_runtime()?;
                let mechanism = model
                    .workspace_mechanisms()
                    .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
                MlxPredictionTargetState::copy_original_source(
                    plan,
                    environment,
                    &initialized,
                    mechanism,
                    sources.metadata_funding(),
                    sources.request().limits().clone(),
                )
            }
        }
    }
}
#[derive(Debug, thiserror::Error)]
enum StartupCause {
    #[error(transparent)]
    State(#[from] ResidentEmptyStateError),
    #[error(transparent)]
    Stream(#[from] StreamCopyError<HostPreparationAuthority>),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct StartupFailure {
    #[source]
    cause: StartupCause,
    // The shared paid error erasure destroys its shell first, then this payload.
    _host: HostPreparationAuthority,
}

pub(super) fn prepare(
    model: &Executable,
    pass: AutoregressivePass,
    sources: &AutoregressiveSourcePair,
    environment: &crate::backend::OriginalCopyEnvironment<'_>,
) -> Result<MlxAutoregressiveState, Error> {
    sources.validate_environment(environment)?;
    let source_role = match pass {
        AutoregressivePass::TargetPrefill => AutoregressiveSource::Target,
        AutoregressivePass::DraftPrefill => AutoregressiveSource::Draft,
        _ => return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
    };
    sources.validate_source(model, source_role)?;
    let error =
        |cause| Error::PrefillControl(cause).at_speculative_stage("AR startup state source");
    let source_origin = model
        .erased()
        .resident_control_origin_fixed()
        .ok_or_else(|| error(WorkingMemoryError::UnknownBound))?
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let copy = model
        .erased()
        .prepare_resident_decoder_copy_fixed()
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let mut populated = false;
    copy.visit_retained_arrays(&mut |_| populated = true)
        .map_err(|cause| sources.retain_startup_error(cause))?;
    // An empty paged state still owns a manager. Its independent destination
    // uses the same admitted source-copy worker as a populated state.
    let plan = if populated || copy.is_paged() {
        StatePlan::Copy(copy)
    } else {
        match model.erased().resident_reset_profile() {
            Some(ResidentResetProfile::KeyValue) => StatePlan::KeyValue(
                PreparedResidentEmptyState::inspect(
                    model.erased().resident_reset_source().map_err(error)?,
                )
                .map_err(error)?,
            ),
            Some(ResidentResetProfile::Hybrid) => StatePlan::Hybrid(
                PreparedResidentEmptyState::inspect(
                    model
                        .erased()
                        .resident_hybrid_reset_source()
                        .map_err(error)?,
                )
                .map_err(error)?,
            ),
            Some(ResidentResetProfile::Pooling) => StatePlan::Pooling(
                PreparedResidentEmptyState::inspect(
                    model
                        .erased()
                        .resident_pooling_reset_source()
                        .map_err(error)?,
                )
                .map_err(error)?,
            ),
            None => return Err(error(WorkingMemoryError::UnknownBound)),
        }
    };
    let stream = StreamCopyPlan::<HostPreparationAuthority>::capture(environment.stream())
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let bytes = controls(&plan, &stream).ok_or_else(|| {
        Error::PrefillControl(WorkingMemoryError::UnknownBound)
            .at_speculative_stage("AR startup state and stream controls")
    })?;
    let accepted = sources
        .request()
        .reserve_startup(source_role, bytes)
        .map_err(|cause| sources.retain_startup_error(cause))?;
    // The runtime token contains only this request/source and its accepted host
    // ticket. It grants no native role; this one-shot constructor uses it solely
    // for the exact host payloads just priced above.
    let host = HostPreparationAuthority::retain(accepted);
    let original_copy = state_copy::StateCopyContext::prepare(model, sources, environment)?;
    let native = plan.construct(&host, sources, model, environment)?;
    let stream = stream.realize(host.clone()).map_err(|cause| {
        sources.retain_startup_error(StartupFailure {
            cause: cause.into(),
            _host: host.clone(),
        })
    })?;
    Ok(MlxAutoregressiveState {
        native,
        stream: StateStream::Original(stream),
        pool: environment.pool().clone(),
        source_origin: Some(source_origin),
        source_role,
        original_copy: Some(original_copy),
    })
}
fn controls(
    plan: &StatePlan<'_>,
    stream: &StreamCopyPlan<HostPreparationAuthority>,
) -> Option<u64> {
    let shared_stream = Layout::new::<[AtomicUsize; 2]>()
        .extend(stream.shared_body_layout())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let fields = [
        HostPreparationAuthority::retention_bytes::<OriginalSpeculativeStartup>()?,
        shared_stream,
        stream.owner_node_layout().size(),
        stream.native_wrapper_bytes(),
        stream.control_bytes()?,
        stream.source_comparison_control_bytes()?,
        size_of::<StatePlan<'_>>(),
        size_of::<MlxAutoregressiveState>(),
        size_of::<MlxPredictionTargetState>(),
        size_of::<StateStream>(),
        size_of::<StartupFailure>(),
        size_of::<StartupCause>(),
        size_of::<Result<MlxAutoregressiveState, Error>>(),
        size_of::<Result<MlxPredictionTargetState, ResidentEmptyStateError>>(),
        size_of::<
            Result<
                OriginalSpeculativeStartup,
                eredu_runtime::working_memory::SpeculativeRequestError,
            >,
        >(),
        size_of::<eredu_runtime::replicated_session::ReplicatedTextControlOrigin>(),
        size_of::<AutoregressiveSource>(),
        size_of::<AutoregressivePass>(),
        size_of::<MemoryLedger>(),
        size_of::<Error>(),
        6 * size_of::<usize>(),
    ];
    let controls = fields
        .into_iter()
        .try_fold(size_of_val(&fields), usize::checked_add)?;
    plan.required_bytes()?
        .checked_add(u64::try_from(controls).ok()?)
}
