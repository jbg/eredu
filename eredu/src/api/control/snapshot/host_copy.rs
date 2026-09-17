//! Exact borrowed ordinary host source for the shared snapshot transaction.
use super::*;
use crate::api::observed::PreparedIdentity;
use eredu_runtime::execution_control::{PreparedTextHostCopy, TextHostCopyError};

pub(in crate::api::control) fn payload_bytes(
    pipeline: &CommittedTokenPipeline<PreparedChatTokenDecoder>,
    cursor: &CommittedGenerationCursor,
    semantic_prefix: &Vec<SemanticEvent>,
    prompt: &PromptRecord,
) -> Option<u64> {
    use crate::runtime::generation::storage::SnapshotStorage;
    pipeline
        .snapshot_storage_bytes()?
        .checked_add(cursor.snapshot_storage_bytes()?)?
        .checked_add(semantic_prefix.snapshot_bytes()?)?
        .checked_add(prompt.logical_bytes()?)
}

pub(super) struct CaptureHost<'a, M: ControlRecordMode> {
    pipeline: &'a CommittedTokenPipeline<PreparedChatTokenDecoder>,
    cursor: &'a CommittedGenerationCursor,
    delivery: &'a Delivery<M>,
    identity: PreparedIdentity<'static>,
    tokenizer_identity: [u8; 32],
    pending_forced_token: Option<u32>,
    status: GenerationStatus,
    prediction: u64,
    bytes: Option<u64>,
}
pub(super) struct CapturedHost<M: ControlRecordMode> {
    pub(super) pipeline: CommittedTokenPipeline<PreparedChatTokenDecoder>,
    pub(super) cursor: CommittedGenerationCursor,
    pub(super) metadata: M::SnapshotMetadata,
    pub(super) info: SnapshotInfo,
    pub(super) semantic_prefix: Vec<SemanticEvent>,
    pub(super) prompt: PromptRecord,
}
impl<'a, M: ControlRecordMode> CaptureHost<'a, M> {
    pub(super) fn new<B: TextSnapshotBackend>(
        pipeline: &'a CommittedTokenPipeline<PreparedChatTokenDecoder>,
        cursor: &'a CommittedGenerationCursor,
        delivery: &'a Delivery<M>,
        tokenizer_identity: [u8; 32],
        pending_forced_token: Option<u32>,
        status: GenerationStatus,
        prediction: u64,
    ) -> Self {
        let identity = PreparedIdentity::new("snapshot");
        let bytes = (|| {
            let template = &delivery.template;
            let dynamic = identity
                .bytes()?
                .checked_add(template.session_id.len() as u64)?
                .checked_add(
                    template
                        .artifact_identity
                        .as_ref()
                        .map_or(0, |v| v.len() as u64),
                )?
                .checked_add(
                    template
                        .capture_plan_id
                        .as_ref()
                        .map_or(0, |v| v.len() as u64),
                )?
                .checked_add(
                    template
                        .intervention_plan_id
                        .as_ref()
                        .map_or(0, |v| v.len() as u64),
                )?
                .checked_add(template.run_id.len() as u64)?;
            payload_bytes(
                pipeline,
                cursor,
                &delivery.semantic_prefix,
                &delivery.prompt,
            )?
            .checked_add(std::mem::size_of::<ControlledGenerationSnapshot<B, M>>() as u64)?
            .checked_add(M::snapshot_control_bytes())?
            // The private restore facts and public metadata own the same
            // text fields independently. This matches their actual workers.
            .checked_add(dynamic)?
            .checked_add(dynamic)
        })();
        Self {
            pipeline,
            cursor,
            delivery,
            identity,
            tokenizer_identity,
            pending_forced_token,
            status,
            prediction,
            bytes,
        }
    }
}
impl<M: ControlRecordMode> PreparedTextHostCopy for CaptureHost<'_, M> {
    type Copied = CapturedHost<M>;
    fn storage_bytes(&self) -> Option<u64> {
        self.bytes
    }
    fn copy(self, retained_bytes: u64) -> Result<Self::Copied, TextHostCopyError> {
        let pipeline = self.pipeline.fork().map_err(TextHostCopyError::Message)?;
        let cursor = self.cursor.clone();
        let template = &self.delivery.template;
        let info = SnapshotInfo {
            schema_version: EXECUTION_CONTROL_SCHEMA_VERSION,
            snapshot_id: self.identity.render(),
            session_id: template.session_id.clone(),
            artifact_identity: template.artifact_identity.clone(),
            capture_plan_id: template.capture_plan_id.clone(),
            intervention_plan_id: template.intervention_plan_id.clone(),
            tokenizer_identity: self.tokenizer_identity,
            configuration_identity: self.delivery.configuration_identity,
            pending_forced_token: self.pending_forced_token,
            status: self.status,
            output: GenerationOutputCheckpoint {
                run_id: template.run_id.clone(),
                epoch: self.delivery.epoch,
                next_sequence: self.delivery.sequence,
                next_prediction: self.prediction,
            },
            retained_bytes,
        };
        Ok(CapturedHost {
            pipeline,
            cursor,
            metadata: M::snapshot(&info, &self.delivery.prompt),
            info,
            semantic_prefix: self.delivery.semantic_prefix.clone(),
            prompt: self.delivery.prompt.clone(),
        })
    }
}

pub(super) struct RestoreHost<'a, B: TextSnapshotBackend, M: ControlRecordMode> {
    pub(super) snapshot: &'a ControlledGenerationSnapshot<B, M>,
}
pub(super) struct RestoredHost {
    pub(super) pipeline: CommittedTokenPipeline<PreparedChatTokenDecoder>,
    pub(super) cursor: CommittedGenerationCursor,
    pub(super) semantic_prefix: Vec<SemanticEvent>,
    pub(super) prompt: PromptRecord,
}
impl<B: TextSnapshotBackend, M: ControlRecordMode> PreparedTextHostCopy for RestoreHost<'_, B, M> {
    type Copied = RestoredHost;
    fn storage_bytes(&self) -> Option<u64> {
        payload_bytes(
            &self.snapshot.pipeline,
            &self.snapshot.cursor,
            &self.snapshot.semantic_prefix,
            &self.snapshot.prompt,
        )
    }
    fn copy(self, _: u64) -> Result<Self::Copied, TextHostCopyError> {
        Ok(RestoredHost {
            pipeline: self
                .snapshot
                .pipeline
                .fork()
                .map_err(TextHostCopyError::Message)?,
            cursor: self.snapshot.cursor.clone(),
            semantic_prefix: self.snapshot.semantic_prefix.clone(),
            // This is staged before native exchange too; no fallible clone is
            // left in the otherwise infallible host installation sequence.
            prompt: self.snapshot.prompt.clone(),
        })
    }
}
