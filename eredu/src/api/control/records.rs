//! Record specialization only; execution, snapshots and branches share one engine.
use super::*;
use crate::api::TraceLimits;
use std::sync::Arc;

pub const PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PreparedInputInstrumentation {
    Unobserved,
    Capture {
        plan: CapturePlan,
    },
    Intervention {
        capture: CapturePlan,
        intervention: eredu_core::intervention::InterventionPlan,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PreparedInstrumentationRecord {
    Unobserved,
    Captured {
        plan_id: String,
    },
    Intervened {
        capture_plan_id: String,
        intervention_plan_id: String,
    },
}

#[derive(Clone)]
pub(super) enum PromptRecord {
    Tokens(Arc<[u32]>),
    Prepared(SharedPromptAttribution),
}
impl PromptRecord {
    pub(super) fn range(&self, prediction: u64) -> Result<[u64; 2], ExecutionControlError> {
        match self {
            Self::Prepared(value) => value
                .attribution()
                .input_range(prediction)
                .map_err(|_| ExecutionControlError::Overflow),
            Self::Tokens(ids) => {
                let end = (ids.len() as u64)
                    .checked_add(prediction)
                    .ok_or(ExecutionControlError::Overflow)?;
                Ok(if prediction == 0 {
                    [0, end]
                } else {
                    [end - 1, end]
                })
            }
        }
    }
    pub(super) fn logical_bytes(&self) -> Option<u64> {
        match self {
            Self::Tokens(ids) => (ids.len() as u64).checked_mul(4),
            Self::Prepared(value) => value.logical_storage_bytes(),
        }
    }
    pub(super) fn tokens(&self) -> &[u32] {
        match self {
            Self::Tokens(ids) => ids,
            _ => unreachable!("legacy mode has real token source"),
        }
    }
    pub(super) fn prepared(&self) -> &SharedPromptAttribution {
        match self {
            Self::Prepared(value) => value,
            _ => unreachable!("prepared mode has actual attribution"),
        }
    }
}

#[derive(Clone)]
pub(super) struct RecordContext {
    pub run_id: String,
    pub artifact_identity: Option<String>,
    pub parameter_overlay_id: Option<String>,
    pub session_id: String,
    pub capture_plan_id: Option<String>,
    pub intervention_plan_id: Option<String>,
}
impl RecordContext {
    fn instrumentation(&self) -> PreparedInstrumentationRecord {
        match (&self.capture_plan_id, &self.intervention_plan_id) {
            (Some(capture), Some(intervention)) => PreparedInstrumentationRecord::Intervened {
                capture_plan_id: capture.clone(),
                intervention_plan_id: intervention.clone(),
            },
            (Some(plan), None) => PreparedInstrumentationRecord::Captured {
                plan_id: plan.clone(),
            },
            (None, None) => PreparedInstrumentationRecord::Unobserved,
            _ => unreachable!("interventions require actual capture installation"),
        }
    }
}

#[derive(Clone)]
pub(super) struct SnapshotInfo {
    /// Metadata version.
    pub schema_version: u32,
    /// Fresh snapshot identity.
    pub snapshot_id: String,
    /// Loaded session that owns the executable and prepared sources.
    pub session_id: String,
    /// Source artifact attribution, when present on the prepared request.
    pub artifact_identity: Option<String>,
    /// Immutable capture admission retained with this continuation.
    pub capture_plan_id: Option<String>,
    /// Immutable intervention admission retained with this continuation.
    pub intervention_plan_id: Option<String>,
    /// Exact immutable tokenizer fingerprint of this run.
    pub tokenizer_identity: [u8; 32],
    /// Initial generation policy identity, including text versus semantic mode,
    /// constraint blueprint, termination, tokenizer and seed. Later choices are retained
    /// exactly in the opaque saved state and attributed through control records.
    pub configuration_identity: [u8; 32],
    /// Prospective canonical choice saved before its ordinary commitment.
    pub pending_forced_token: Option<u32>,
    /// Prepared, paused, or normally completed state preserved by restoration.
    pub status: GenerationStatus,
    /// Delivered prefix paired with incremental semantic buffers.
    pub output: GenerationOutputCheckpoint,
    /// Complete charged logical native/controller/host state, excluding shared
    /// immutable weights/tokenizer data and allocator overhead.
    pub retained_bytes: u64,
}

#[derive(Clone)]
pub(super) struct BranchInfo {
    /// Metadata version.
    pub schema_version: u32,
    /// Fresh logical child run identity.
    pub run_id: String,
    /// Exact parent snapshot and inherited absolute/output boundary.
    pub parent: SnapshotInfo,
    /// Capture consumption already inherited before the child executes.
    pub inherited_capture_usage: CaptureUsage,
    /// Explicit child transport limit, independent of the parent's consumption.
    pub trace_limits: TraceLimits,
    /// Prospective sampler change, with inherited randomness unless explicitly reseeded.
    pub sampling_override: Option<SamplingOverride>,
    /// Effective sampler facts inherited from the exact saved boundary.
    pub sampling_before: SamplingStateFacts,
    /// Effective child sampler facts after the prospective override.
    pub sampling_after: SamplingStateFacts,
    /// Prospective immutable request re-admitted under this child's session.
    pub intervention_override: Option<eredu_core::intervention::InterventionPlan>,
}

pub(super) enum ControlEvent {
    Started {
        generation: ResolvedGenerationConfig,
        seed: u64,
    },
    SnapshotCreated {
        metadata: SnapshotInfo,
    },
    BranchStarted {
        lineage: BranchInfo,
        inherited_token_ids: Vec<u32>,
        inherited_semantics: Vec<SemanticEvent>,
    },
    Existing(ObservedGenerationEvent),
}
impl From<ObservedGenerationEvent> for ControlEvent {
    fn from(value: ObservedGenerationEvent) -> Self {
        Self::Existing(value)
    }
}

mod sealed {
    pub trait Sealed {}
}
/// Sealed metadata mode. It supplies records only, never generation policy.
#[doc(hidden)]
pub trait ControlRecordMode: sealed::Sealed {
    const VERSION: u32;
    fn retain_error(
        error: ControlledGenerationError,
        _: HostPreparationAuthority,
    ) -> ControlledGenerationError {
        error
    }
    fn snapshot_control_bytes() -> u64 {
        0
    }
    type Record: Serialize;
    type SnapshotMetadata: Clone;
    fn record(
        context: &RecordContext,
        prompt: &PromptRecord,
        event: ControlEvent,
        sequence: u64,
        epoch: u64,
        timing: GenerationTiming,
    ) -> Self::Record;
    fn snapshot(info: &SnapshotInfo, prompt: &PromptRecord) -> Self::SnapshotMetadata;
}
#[doc(hidden)]
pub struct LegacyText;
impl sealed::Sealed for LegacyText {}
#[doc(hidden)]
pub struct PreparedInputV2;
impl sealed::Sealed for PreparedInputV2 {}

impl SnapshotInfo {
    fn legacy(&self) -> GenerationSnapshotMetadata {
        GenerationSnapshotMetadata {
            schema_version: EXECUTION_CONTROL_SCHEMA_VERSION,
            snapshot_id: self.snapshot_id.clone(),
            session_id: self.session_id.clone(),
            artifact_identity: self.artifact_identity.clone(),
            capture_plan_id: self
                .capture_plan_id
                .clone()
                .expect("legacy captured source"),
            intervention_plan_id: self.intervention_plan_id.clone(),
            tokenizer_identity: self.tokenizer_identity,
            configuration_identity: self.configuration_identity,
            pending_forced_token: self.pending_forced_token,
            status: self.status,
            output: self.output.clone(),
            retained_bytes: self.retained_bytes,
        }
    }
}
impl BranchInfo {
    fn legacy(self) -> GenerationBranchMetadata {
        GenerationBranchMetadata {
            schema_version: EXECUTION_CONTROL_SCHEMA_VERSION,
            run_id: self.run_id,
            parent: self.parent.legacy(),
            inherited_capture_usage: self.inherited_capture_usage,
            trace_limits: self.trace_limits,
            sampling_override: self.sampling_override,
            sampling_before: self.sampling_before,
            sampling_after: self.sampling_after,
            intervention_override: self.intervention_override,
        }
    }
}
impl ControlRecordMode for LegacyText {
    const VERSION: u32 = EXECUTION_CONTROL_SCHEMA_VERSION;
    type Record = ControlledGenerationRecord;
    type SnapshotMetadata = GenerationSnapshotMetadata;
    fn snapshot(info: &SnapshotInfo, _: &PromptRecord) -> Self::SnapshotMetadata {
        info.legacy()
    }
    fn record(
        c: &RecordContext,
        prompt: &PromptRecord,
        event: ControlEvent,
        sequence: u64,
        epoch: u64,
        timing: GenerationTiming,
    ) -> Self::Record {
        let event = match event {
            ControlEvent::Started { generation, seed } => ObservedGenerationEvent::Started {
                prompt_token_ids: prompt.tokens().to_vec(),
                generation,
                seed,
            },
            ControlEvent::SnapshotCreated { metadata } => {
                ObservedGenerationEvent::SnapshotCreated {
                    metadata: metadata.legacy(),
                }
            }
            ControlEvent::BranchStarted {
                lineage,
                inherited_token_ids,
                inherited_semantics,
            } => ObservedGenerationEvent::BranchStarted {
                lineage: lineage.legacy(),
                prompt_token_ids: prompt.tokens().to_vec(),
                inherited_token_ids,
                inherited_semantics,
            },
            ControlEvent::Existing(event) => event,
        };
        ControlledGenerationRecord {
            schema_version: EXECUTION_CONTROL_SCHEMA_VERSION,
            sequence,
            epoch,
            timing,
            generation: ObservedGenerationRecord {
                schema_version: CAPTURE_SCHEMA_VERSION,
                run_id: c.run_id.clone(),
                artifact_identity: c.artifact_identity.clone(),
                session_id: c.session_id.clone(),
                capture_plan_id: c.capture_plan_id.clone().expect("legacy capture"),
                intervention_plan_id: c.intervention_plan_id.clone(),
                parameter_overlay_id: c.parameter_overlay_id.clone(),
                event,
            },
        }
    }
}

/// V2 snapshot metadata. A deserialized value is diagnostic, not a saved owner.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedGenerationSnapshotData<P = SharedPromptAttribution> {
    pub schema_version: u32,
    pub snapshot_id: String,
    pub session_id: String,
    pub artifact_identity: Option<String>,
    pub prompt_attribution: P,
    pub instrumentation: PreparedInstrumentationRecord,
    pub tokenizer_identity: [u8; 32],
    pub configuration_identity: [u8; 32],
    pub pending_forced_token: Option<u32>,
    pub status: GenerationStatus,
    pub output: GenerationOutputCheckpoint,
    pub retained_bytes: u64,
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedGenerationBranchMetadata<P = SharedPromptAttribution> {
    pub schema_version: u32,
    pub run_id: String,
    pub parent: PreparedGenerationSnapshotData<P>,
    pub inherited_capture_usage: CaptureUsage,
    pub trace_limits: super::super::TraceLimits,
    pub sampling_override: Option<SamplingOverride>,
    pub sampling_before: SamplingStateFacts,
    pub sampling_after: SamplingStateFacts,
    pub intervention_override: Option<eredu_core::intervention::InterventionPlan>,
}

/// Prepared-input records retain the actual ordinary metadata custody.
/// Use PreparedControlledWireRecord for deserialization of diagnostics.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedControlledRecordData<P = SharedPromptAttribution> {
    pub schema_version: u32,
    pub sequence: u64,
    pub epoch: u64,
    pub timing: GenerationTiming,
    pub run_id: String,
    pub artifact_identity: Option<String>,
    pub parameter_overlay_id: Option<String>,
    pub session_id: String,
    pub instrumentation: PreparedInstrumentationRecord,
    pub event: PreparedControlledGenerationEvent<P>,
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PreparedControlledGenerationEvent<P = SharedPromptAttribution> {
    Started {
        prompt_attribution: P,
        generation: ResolvedGenerationConfig,
        seed: u64,
    },
    SnapshotCreated {
        metadata: PreparedGenerationSnapshotData<P>,
    },
    BranchStarted {
        lineage: PreparedGenerationBranchMetadata<P>,
        prompt_attribution: P,
        inherited_token_ids: Vec<u32>,
        inherited_semantics: Vec<SemanticEvent>,
    },
    /// Existing token/lifecycle/sampling/semantic payload; never contains V1 startup/snapshot events.
    Progress { event: ObservedGenerationEvent },
}
/// Owned wire diagnostics cannot construct a backend input or restore a snapshot.
pub type PreparedControlledWireRecord = PreparedControlledRecordData<PreparedPromptAttribution>;
impl PreparedControlledWireRecord {
    pub fn from_json(input: &str) -> Result<Self, CaptureError> {
        let record: Self =
            serde_json::from_str(input).map_err(|e| CaptureError::Invalid(e.to_string()))?;
        if record.schema_version != PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION {
            return Err(CaptureError::Invalid(
                "unsupported prepared control schema".into(),
            ));
        }
        let invalid = || CaptureError::Invalid("inconsistent prepared control attribution".into());
        let source = |value: &PreparedPromptAttribution| {
            value
                .validate()
                .map_err(|e| CaptureError::Invalid(e.to_string()))
        };
        let snapshot = |value: &PreparedGenerationSnapshotData<PreparedPromptAttribution>| {
            if value.schema_version != PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION
                || value.session_id.is_empty()
            {
                return Err(invalid());
            }
            source(&value.prompt_attribution)
        };
        match &record.event {
            PreparedControlledGenerationEvent::Started {
                prompt_attribution, ..
            } => source(prompt_attribution)?,
            PreparedControlledGenerationEvent::SnapshotCreated { metadata } => {
                snapshot(metadata)?;
                if metadata.session_id != record.session_id
                    || metadata.instrumentation != record.instrumentation
                {
                    return Err(invalid());
                }
            }
            PreparedControlledGenerationEvent::BranchStarted {
                lineage,
                prompt_attribution,
                ..
            } => {
                source(prompt_attribution)?;
                snapshot(&lineage.parent)?;
                if lineage.schema_version != PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION
                    || lineage.run_id != record.run_id
                    || lineage.parent.prompt_attribution != *prompt_attribution
                {
                    return Err(invalid());
                }
            }
            PreparedControlledGenerationEvent::Progress { event }
                if matches!(
                    event,
                    ObservedGenerationEvent::Started { .. }
                        | ObservedGenerationEvent::BranchStarted { .. }
                        | ObservedGenerationEvent::SnapshotCreated { .. }
                ) =>
            {
                return Err(invalid())
            }
            _ => {}
        }
        Ok(record)
    }
}
impl ControlRecordMode for PreparedInputV2 {
    const VERSION: u32 = PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION;
    fn retain_error(
        error: ControlledGenerationError,
        host: HostPreparationAuthority,
    ) -> ControlledGenerationError {
        prepared::retained(error, host).into()
    }
    type Record = PreparedControlledGenerationRecord;
    type SnapshotMetadata = PreparedGenerationSnapshotMetadata;
    fn snapshot(info: &SnapshotInfo, prompt: &PromptRecord) -> Self::SnapshotMetadata {
        PreparedGenerationSnapshotMetadata::new(
            Self::snapshot_data(info, prompt),
            prompt.prepared().clone(),
        )
    }
    fn snapshot_control_bytes() -> u64 {
        // Exact concrete ArcInner layout; no allocator or erased-token claim.
        std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<PreparedSnapshotOwner>())
            .expect("finite metadata control layout")
            .0
            .pad_to_align()
            .size() as u64
    }
    fn record(
        c: &RecordContext,
        prompt: &PromptRecord,
        event: ControlEvent,
        sequence: u64,
        epoch: u64,
        timing: GenerationTiming,
    ) -> Self::Record {
        let event = match event {
            ControlEvent::Started { generation, seed } => {
                PreparedControlledGenerationEvent::Started {
                    prompt_attribution: prompt.prepared().clone(),
                    generation,
                    seed,
                }
            }
            ControlEvent::SnapshotCreated { metadata } => {
                PreparedControlledGenerationEvent::SnapshotCreated {
                    metadata: Self::snapshot_data(&metadata, prompt),
                }
            }
            ControlEvent::BranchStarted {
                lineage,
                inherited_token_ids,
                inherited_semantics,
            } => {
                let parent = Self::snapshot_data(&lineage.parent, prompt);
                PreparedControlledGenerationEvent::BranchStarted {
                    lineage: PreparedGenerationBranchMetadata {
                        schema_version: PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION,
                        run_id: lineage.run_id,
                        parent,
                        inherited_capture_usage: lineage.inherited_capture_usage,
                        trace_limits: lineage.trace_limits,
                        sampling_override: lineage.sampling_override,
                        sampling_before: lineage.sampling_before,
                        sampling_after: lineage.sampling_after,
                        intervention_override: lineage.intervention_override,
                    },
                    prompt_attribution: prompt.prepared().clone(),
                    inherited_token_ids,
                    inherited_semantics,
                }
            }
            ControlEvent::Existing(event) => PreparedControlledGenerationEvent::Progress { event },
        };
        PreparedControlledGenerationRecord::new(
            PreparedControlledRecordData {
                schema_version: PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION,
                sequence,
                epoch,
                timing,
                run_id: c.run_id.clone(),
                artifact_identity: c.artifact_identity.clone(),
                parameter_overlay_id: c.parameter_overlay_id.clone(),
                session_id: c.session_id.clone(),
                instrumentation: c.instrumentation(),
                event,
            },
            prompt.prepared().clone(),
        )
    }
}


struct PreparedRecordOwner {
    data: PreparedControlledRecordData,
    _custody: SharedPromptAttribution,
}
/// Closed ordinary record owner. Borrow data or serialize it; no raw shared owner
/// or consuming payload extraction can detach library storage from its custody.
/// ```compile_fail
/// use eredu::api::{PreparedControlledGenerationRecord, PreparedControlledRecordData};
/// fn detach(record: PreparedControlledGenerationRecord) -> PreparedControlledRecordData {
///     *record
/// }
/// ```
pub struct PreparedControlledGenerationRecord(Option<Arc<PreparedRecordOwner>>);
impl PreparedControlledGenerationRecord {
    fn new(data: PreparedControlledRecordData, custody: SharedPromptAttribution) -> Self {
        Self(Some(Arc::new(PreparedRecordOwner {
            data,
            _custody: custody,
        })))
    }
}
impl std::ops::Deref for PreparedControlledGenerationRecord {
    type Target = PreparedControlledRecordData;
    fn deref(&self) -> &Self::Target {
        &self.0.as_ref().expect("live V2 record").data
    }
}
impl Clone for PreparedControlledGenerationRecord {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live V2 record"))))
    }
}
impl Drop for PreparedControlledGenerationRecord {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl std::fmt::Debug for PreparedControlledGenerationRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}
impl PartialEq for PreparedControlledGenerationRecord {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Serialize for PreparedControlledGenerationRecord {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (**self).serialize(serializer)
    }
}

impl PreparedInputV2 {
    fn snapshot_data(info: &SnapshotInfo, prompt: &PromptRecord) -> PreparedGenerationSnapshotData {
        let instrumentation = match (&info.capture_plan_id, &info.intervention_plan_id) {
            (Some(capture), Some(intervention)) => PreparedInstrumentationRecord::Intervened {
                capture_plan_id: capture.clone(),
                intervention_plan_id: intervention.clone(),
            },
            (Some(plan), None) => PreparedInstrumentationRecord::Captured {
                plan_id: plan.clone(),
            },
            (None, None) => PreparedInstrumentationRecord::Unobserved,
            _ => unreachable!("installed intervention has a capture owner"),
        };
        PreparedGenerationSnapshotData {
            schema_version: PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION,
            snapshot_id: info.snapshot_id.clone(),
            session_id: info.session_id.clone(),
            artifact_identity: info.artifact_identity.clone(),
            prompt_attribution: prompt.prepared().clone(),
            instrumentation,
            tokenizer_identity: info.tokenizer_identity,
            configuration_identity: info.configuration_identity,
            pending_forced_token: info.pending_forced_token,
            status: info.status,
            output: info.output.clone(),
            retained_bytes: info.retained_bytes,
        }
    }
}

struct PreparedSnapshotOwner {
    data: PreparedGenerationSnapshotData,
    _custody: SharedPromptAttribution,
}
/// Closed ordinary snapshot metadata. Serialization yields diagnostics only;
/// cloning retains the same custody through all payload and shell retirement.
/// ```compile_fail
/// use eredu::api::{PreparedGenerationSnapshotMetadata, PreparedGenerationSnapshotData};
/// fn detach(metadata: PreparedGenerationSnapshotMetadata) -> PreparedGenerationSnapshotData {
///     *metadata
/// }
/// ```
pub struct PreparedGenerationSnapshotMetadata(Option<Arc<PreparedSnapshotOwner>>);
impl PreparedGenerationSnapshotMetadata {
    fn new(data: PreparedGenerationSnapshotData, custody: SharedPromptAttribution) -> Self {
        Self(Some(Arc::new(PreparedSnapshotOwner {
            data,
            _custody: custody,
        })))
    }
}
impl std::ops::Deref for PreparedGenerationSnapshotMetadata {
    type Target = PreparedGenerationSnapshotData;
    fn deref(&self) -> &Self::Target {
        &self.0.as_ref().expect("live V2 metadata").data
    }
}
impl Clone for PreparedGenerationSnapshotMetadata {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live V2 metadata"))))
    }
}
impl Drop for PreparedGenerationSnapshotMetadata {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl std::fmt::Debug for PreparedGenerationSnapshotMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}
impl PartialEq for PreparedGenerationSnapshotMetadata {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Serialize for PreparedGenerationSnapshotMetadata {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (**self).serialize(serializer)
    }
}
