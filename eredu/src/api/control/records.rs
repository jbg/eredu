//! One retained record family for every controlled input source.
use super::*;
use crate::api::TraceLimits;
use eredu_core::HostMetadataFunding;
use std::sync::Arc;
pub(super) mod construction;
mod media;
pub use construction::{RecordConstructionCause, RecordConstructionError};
#[cfg(test)]
mod tests;

pub const PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION: u32 = 2;

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
pub(crate) struct PromptRecord(SharedPromptAttribution);
impl PromptRecord {
    pub(super) fn range(&self, prediction: u64) -> Result<[u64; 2], ExecutionControlError> {
        self.0
            .attribution()
            .input_range(prediction)
            .map_err(|_| ExecutionControlError::Overflow)
    }
    pub(super) fn logical_bytes(&self) -> Option<u64> {
        self.0.logical_storage_bytes()
    }
    pub(crate) fn prepared(&self) -> &SharedPromptAttribution {
        &self.0
    }
    pub(super) fn new(value: SharedPromptAttribution) -> Self {
        Self(value)
    }
    pub(crate) fn from_tokens(
        ids: &[u32],
        funding: &HostMetadataFunding,
    ) -> Result<Self, RecordConstructionError> {
        use construction::{RecordConstructionCause as Cause, controls, string_capacity, vector};
        use std::mem::size_of;
        let result = (|| {
            controls(
                funding,
                &[
                    size_of::<Self>(),
                    size_of::<PreparedPromptAttribution>(),
                    size_of::<PreparedInputIdentity>(),
                    size_of::<InputPartDescriptor>(),
                    size_of::<InputTensorIdentity>(),
                    size_of::<PreparedPromptSegment>(),
                    size_of::<PreparedPromptSegmentPlan>(),
                    size_of::<PromptTokenAttribution>(),
                    size_of::<eredu_core::PreparedInputError>(),
                size_of::<Result<Self, RecordConstructionError>>(),
                size_of::<construction::Result<Self>>(),
                    size_of::<Result<PreparedInputIdentity, eredu_core::PreparedInputError>>(),
                    size_of::<HostPreparationAuthority>(),
                    size_of::<(u64, usize)>(),
                    eredu_core::cache::prompt_cache_token_fingerprint_control_bytes(),
                ],
            )?;
            let positions = u64::try_from(ids.len())
                .map_err(|_| eredu_core::HostMetadataFundingError::Overflow)?;
            if positions == 0 {
                return Err(Cause::Attribution);
            }
            let mut shape = vector(2, funding)?;
            shape.extend_from_slice(&[1, ids.len()]);
            let tensor = InputTensorIdentity::new(eredu_core::checkpoint::TensorDtype::U32, shape)
                .map_err(|_| Cause::Attribution)?;
            let part = InputPartDescriptor::new(
                InputModality::Text,
                InputPayloadKind::TokenIds,
                tensor,
                [],
            )
            .map_err(|_| Cause::Attribution)?;
            let mut parts = vector(1, funding)?;
            parts.push(part);
            let prepared = PreparedInputIdentity::new(parts).map_err(|_| Cause::Attribution)?;
            let mut segments = vector(1, funding)?;
            segments.push(PreparedPromptSegment {
                plan: PreparedPromptSegmentPlan {
                    source_part: 0,
                    modality: InputModality::Text,
                    payload: InputPayloadKind::TokenIds,
                    decoder_range: [0, positions],
                },
                tokens: PromptTokenAttribution::Canonical {
                    range: [0, positions],
                },
            });
            let mut canonical_token_ids = vector(ids.len(), funding)?;
            canonical_token_ids.extend_from_slice(ids);
            let mut semantic_content_identity = string_capacity(64, funding)?;
            assert!(eredu_core::cache::prompt_cache_token_fingerprint_into(
                ids,
                &mut semantic_content_identity
            ));
            funding.reserve_metadata(
                HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                    .ok_or(eredu_core::HostMetadataFundingError::Overflow)?,
            )?;
            let host = HostPreparationAuthority::retain(funding.clone());
            funding.reserve_metadata(
                SharedPromptAttribution::construction_bytes()
                    .ok_or(eredu_core::HostMetadataFundingError::Overflow)?,
            )?;
            SharedPromptAttribution::from_prepared(
                PreparedPromptAttribution {
                    schema_version: PREPARED_PROMPT_ATTRIBUTION_VERSION,
                    prepared,
                    semantic_content_identity,
                    opening_position: 0,
                    decoder_positions: positions,
                    batch: 1,
                    segments,
                    canonical_token_ids,
                },
                host,
            )
            .map(Self)
            .map_err(|_| Cause::Attribution)
        })();
        result.map_err(|cause| RecordConstructionError::retain(cause, funding))
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

#[derive(Clone)]
pub(super) struct SnapshotInfo {
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
    pub output: GenerationOutputCheckpointData,
    /// Complete charged logical native/controller/host state, excluding shared
    /// immutable weights/tokenizer data and allocator overhead.
    pub retained_bytes: u64,
}

#[derive(Clone)]
pub(super) struct BranchInfo {
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

/// Snapshot metadata. A deserialized value is diagnostic, not a saved owner.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationSnapshotData<P = SharedPromptAttribution> {
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
    pub output: GenerationOutputCheckpointData,
    pub retained_bytes: u64,
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationBranchMetadata<P = SharedPromptAttribution> {
    pub schema_version: u32,
    pub run_id: String,
    pub parent: GenerationSnapshotData<P>,
    pub inherited_capture_usage: CaptureUsage,
    pub trace_limits: super::super::TraceLimits,
    pub sampling_override: Option<SamplingOverride>,
    pub sampling_before: SamplingStateFacts,
    pub sampling_after: SamplingStateFacts,
    pub intervention_override: Option<eredu_core::intervention::InterventionPlan>,
}

/// Prepared-input records retain the actual ordinary metadata custody.
/// Use ControlledWireRecord for deserialization of diagnostics.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlledRecordData<P = SharedPromptAttribution> {
    pub schema_version: u32,
    pub sequence: u64,
    pub epoch: u64,
    pub timing: GenerationTiming,
    pub run_id: String,
    pub artifact_identity: Option<String>,
    pub parameter_overlay_id: Option<String>,
    pub session_id: String,
    pub instrumentation: PreparedInstrumentationRecord,
    pub event: ControlledGenerationEvent<P>,
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlledGenerationEvent<P = SharedPromptAttribution> {
    Started {
        prompt_attribution: P,
        generation: ResolvedGenerationConfig,
        seed: u64,
    },
    SnapshotCreated {
        metadata: GenerationSnapshotData<P>,
    },
    BranchStarted {
        lineage: GenerationBranchMetadata<P>,
        prompt_attribution: P,
        inherited_token_ids: Vec<u32>,
        inherited_semantics: Vec<SemanticEvent>,
    },
    /// Existing token/lifecycle/sampling/semantic payload; never contains startup/snapshot events.
    Progress {
        event: ObservedGenerationEvent,
    },
}
impl<P> ControlledGenerationEvent<P> {
    /// Borrows a committed-token, semantic, lifecycle or sampling event.
    pub fn progress(&self) -> Option<&ObservedGenerationEvent> {
        match self {
            Self::Progress { event } => Some(event),
            _ => None,
        }
    }
}
/// Owned wire diagnostics cannot construct a backend input or restore a snapshot.
pub type ControlledWireRecord = ControlledRecordData<PreparedPromptAttribution>;
impl ControlledWireRecord {
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
        let snapshot = |value: &GenerationSnapshotData<PreparedPromptAttribution>| {
            if value.schema_version != PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION
                || value.session_id.is_empty()
            {
                return Err(invalid());
            }
            source(&value.prompt_attribution)
        };
        match &record.event {
            ControlledGenerationEvent::Started {
                prompt_attribution, ..
            } => source(prompt_attribution)?,
            ControlledGenerationEvent::SnapshotCreated { metadata } => {
                snapshot(metadata)?;
                if metadata.session_id != record.session_id
                    || metadata.instrumentation != record.instrumentation
                {
                    return Err(invalid());
                }
            }
            ControlledGenerationEvent::BranchStarted {
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
            ControlledGenerationEvent::Progress { event }
                if matches!(event, ObservedGenerationEvent::Started { .. }) =>
            {
                return Err(invalid());
            }
            _ => {}
        }
        Ok(record)
    }
}
impl ControlledGenerationRecord {
    pub(super) fn snapshot_control_bytes() -> u64 {
        // Exact concrete ArcInner layout; no allocator or erased-token claim.
        std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<SnapshotOwner>())
            .expect("finite metadata control layout")
            .0
            .pad_to_align()
            .size() as u64
    }
    pub(super) fn record(
        c: &RecordContext,
        prompt: &PromptRecord,
        event: ControlEvent,
        sequence: u64,
        epoch: u64,
        timing: GenerationTiming,
        funding: &HostMetadataFunding,
    ) -> Result<ControlledGenerationRecord, RecordConstructionError> {
        use std::mem::size_of;
        let result = (|| {
            construction::controls(
                funding,
                &[
                    size_of::<ControlEvent>(),
                    size_of::<ControlledGenerationEvent>(),
                size_of::<ControlledRecordData>(),
                size_of::<construction::Result<Self>>(),
                size_of::<Result<Self, RecordConstructionError>>(),
                    size_of::<RecordOwner>(),
                    size_of::<PreparedInstrumentationRecord>(),
                    size_of::<SnapshotInfo>(),
                    size_of::<BranchInfo>(),
                    size_of::<GenerationSnapshotData>(),
                    size_of::<GenerationBranchMetadata>(),
                    size_of::<Arc<RecordOwner>>(),
                    size_of::<SharedPromptAttribution>(),
                    size_of::<GenerationTiming>(),
                    size_of::<(u64, u64, &RecordContext, &PromptRecord)>(),
                ],
            )?;
            let event = match event {
                ControlEvent::Started { generation, seed } => ControlledGenerationEvent::Started {
                    prompt_attribution: prompt.prepared().clone(),
                    generation,
                    seed,
                },
                ControlEvent::SnapshotCreated { metadata } => {
                    ControlledGenerationEvent::SnapshotCreated {
                        metadata: Self::into_snapshot_data(metadata, prompt)?,
                    }
                }
                ControlEvent::BranchStarted {
                    lineage,
                    inherited_token_ids,
                    inherited_semantics,
                } => {
                    let parent = Self::into_snapshot_data(lineage.parent, prompt)?;
                    ControlledGenerationEvent::BranchStarted {
                        lineage: GenerationBranchMetadata {
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
                ControlEvent::Existing(event) => ControlledGenerationEvent::Progress { event },
            };
            let data = ControlledRecordData {
                schema_version: PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION,
                sequence,
                epoch,
                timing,
                run_id: construction::string(&c.run_id, funding)?,
                artifact_identity: construction::optional(&c.artifact_identity, funding)?,
                parameter_overlay_id: construction::optional(&c.parameter_overlay_id, funding)?,
                session_id: construction::string(&c.session_id, funding)?,
                instrumentation: construction::instrumentation(
                    &c.capture_plan_id,
                    &c.intervention_plan_id,
                    funding,
                )?,
                event,
            };
            construction::shell::<RecordOwner>(funding)?;
            Ok(Self::new(data, prompt.prepared().clone(), funding.clone()))
        })();
        result.map_err(|cause| RecordConstructionError::retain(cause, funding))
    }

    fn into_snapshot_data(
        info: SnapshotInfo,
        prompt: &PromptRecord,
    ) -> construction::Result<GenerationSnapshotData> {
        let instrumentation = match (info.capture_plan_id, info.intervention_plan_id) {
            (Some(capture_plan_id), Some(intervention_plan_id)) => {
                PreparedInstrumentationRecord::Intervened {
                    capture_plan_id,
                    intervention_plan_id,
                }
            }
            (Some(plan_id), None) => PreparedInstrumentationRecord::Captured { plan_id },
            (None, None) => PreparedInstrumentationRecord::Unobserved,
            _ => return Err(RecordConstructionCause::Attribution),
        };
        Ok(GenerationSnapshotData {
            schema_version: PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION,
            snapshot_id: info.snapshot_id,
            session_id: info.session_id,
            artifact_identity: info.artifact_identity,
            prompt_attribution: prompt.prepared().clone(),
            instrumentation,
            tokenizer_identity: info.tokenizer_identity,
            configuration_identity: info.configuration_identity,
            pending_forced_token: info.pending_forced_token,
            status: info.status,
            output: info.output,
            retained_bytes: info.retained_bytes,
        })
    }
}

struct RecordOwner {
    data: ControlledRecordData,
    _custody: SharedPromptAttribution,
    // The payload and final shared shell retire before their original payer.
    _funding: HostMetadataFunding,
}
/// Closed ordinary record owner. Borrow data or serialize it; no raw shared owner
/// or consuming payload extraction can detach library storage from its custody.
/// ```compile_fail
/// use eredu::api::{ControlledGenerationRecord, ControlledRecordData};
/// fn detach(record: ControlledGenerationRecord) -> ControlledRecordData {
///     *record
/// }
/// ```
pub struct ControlledGenerationRecord(Option<Arc<RecordOwner>>);
impl ControlledGenerationRecord {
    fn new(
        data: ControlledRecordData,
        custody: SharedPromptAttribution,
        funding: HostMetadataFunding,
    ) -> Self {
        Self(Some(Arc::new(RecordOwner {
            data,
            _custody: custody,
            _funding: funding,
        })))
    }
}
impl std::ops::Deref for ControlledGenerationRecord {
    type Target = ControlledRecordData;
    fn deref(&self) -> &Self::Target {
        &self.0.as_ref().expect("live control record").data
    }
}
impl Clone for ControlledGenerationRecord {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live control record"),
        )))
    }
}
impl Drop for ControlledGenerationRecord {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl std::fmt::Debug for ControlledGenerationRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}
impl PartialEq for ControlledGenerationRecord {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Serialize for ControlledGenerationRecord {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (**self).serialize(serializer)
    }
}

impl ControlledGenerationRecord {
    fn snapshot_data(info: &SnapshotInfo, prompt: &PromptRecord) -> GenerationSnapshotData {
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
        GenerationSnapshotData {
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

struct SnapshotOwner {
    data: GenerationSnapshotData,
    reservation: Option<eredu_runtime::execution_control::SnapshotReservation>,
    destination: HostPreparationAuthority,
}
/// Closed ordinary snapshot metadata. Serialization yields diagnostics only;
/// cloning retains the same custody through all payload and shell retirement.
/// ```compile_fail
/// use eredu::api::{GenerationSnapshotMetadata, GenerationSnapshotData};
/// fn detach(metadata: GenerationSnapshotMetadata) -> GenerationSnapshotData {
///     *metadata
/// }
/// ```
pub struct GenerationSnapshotMetadata(Option<Arc<SnapshotOwner>>);
impl GenerationSnapshotMetadata {
    pub(super) fn retain_destination(
        &mut self,
        host: HostPreparationAuthority,
        reservation: eredu_runtime::execution_control::SnapshotReservation,
    ) {
        let owner = Arc::get_mut(self.0.as_mut().expect("live snapshot metadata"))
            .expect("snapshot metadata is unpublished");
        owner.destination = host;
        owner.reservation = Some(reservation);
    }
    pub(in crate::api::control) fn new(data: GenerationSnapshotData, destination: HostPreparationAuthority) -> Self {
        Self(Some(Arc::new(SnapshotOwner {
            data,
            destination,
            reservation: None,
        })))
    }
}
impl std::ops::Deref for GenerationSnapshotMetadata {
    type Target = GenerationSnapshotData;
    fn deref(&self) -> &Self::Target {
        &self.0.as_ref().expect("live snapshot metadata").data
    }
}
impl Clone for GenerationSnapshotMetadata {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live snapshot metadata"),
        )))
    }
}
impl Drop for GenerationSnapshotMetadata {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl std::fmt::Debug for GenerationSnapshotMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}
impl PartialEq for GenerationSnapshotMetadata {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Serialize for GenerationSnapshotMetadata {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (**self).serialize(serializer)
    }
}
