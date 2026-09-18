//! Concrete record journal copied within the original semantic snapshot account.
use super::*;
use crate::api::observed::PreparedIdentity;
use eredu_runtime::execution_control::{PreparedTextHostJournal, TextHostCopyError};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    mem::{size_of, size_of_val},
};

pub(in crate::api::control) struct CaptureJournal<'a, B: TextSnapshotBackend> {
    delivery: &'a Delivery,
    identity: PreparedIdentity<'static>,
    tokenizer: [u8; 32],
    pending: Option<u32>,
    status: GenerationStatus,
    prediction: u64,
    backend: std::marker::PhantomData<fn() -> B>,
}
pub(in crate::api::control) struct CopiedJournal {
    pub(in crate::api::control) metadata: GenerationSnapshotMetadata,
    pub(in crate::api::control) info: SnapshotInfo,
    pub(in crate::api::control) semantic_prefix: Vec<SemanticEvent>,
    pub(in crate::api::control) prompt: PromptRecord,
    pub(in crate::api::control) destination: HostPreparationAuthority,
}
#[derive(Debug, thiserror::Error)]
#[error("record journal host allocation failed")]
struct CopyFailure {
    #[source]
    cause: TryReserveError,
    destination: HostPreparationAuthority,
}
fn string(source: &str) -> Result<String, TryReserveError> {
    let mut copied = String::new();
    copied.try_reserve_exact(source.len())?;
    copied.push_str(source);
    Ok(copied)
}
fn optional(source: &Option<String>) -> Result<Option<String>, TryReserveError> {
    source.as_deref().map(string).transpose()
}
fn event_bytes(event: &SemanticEvent) -> Option<usize> {
    match event {
        SemanticEvent::TextDelta(text) | SemanticEvent::ReasoningDelta(text) => {
            Some(text.snapshot_copy_bytes())
        }
        SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => {
            Some(json_fragment.snapshot_copy_bytes())
        }
        SemanticEvent::ToolCallStart { id, name, .. } => id
            .snapshot_copy_bytes()
            .checked_add(name.snapshot_copy_bytes()),
        SemanticEvent::ToolCallEnd | SemanticEvent::Finished { .. } => Some(0),
    }
}
fn copy_events(source: &[SemanticEvent]) -> Result<Vec<SemanticEvent>, TryReserveError> {
    let mut copied = Vec::new();
    copied.try_reserve_exact(source.len())?;
    // SemanticText's canonical Clone copies ordinary owned spelling and aliases
    // immutable prepared spelling. event_bytes queries that exact producer.
    for event in source {
        copied.push(event.clone());
    }
    Ok(copied)
}
fn events_bytes(source: &[SemanticEvent]) -> Option<usize> {
    source.iter().try_fold(
        Layout::array::<SemanticEvent>(source.len()).ok()?.size(),
        |bytes, event| bytes.checked_add(event_bytes(event)?),
    )
}
fn copy_controls<T>() -> Option<usize> {
    let parts = [
        size_of::<T>(),
        size_of::<CopyFailure>(),
        size_of::<CopiedJournal>(),
        size_of::<SnapshotInfo>(),
        size_of::<GenerationSnapshotData>(),
        size_of::<PreparedInstrumentationRecord>(),
        size_of::<GenerationOutputCheckpointData>(),
        size_of::<String>(),
        size_of::<Option<String>>(),
        size_of::<SemanticEvent>(),
        size_of::<Vec<SemanticEvent>>(),
        size_of::<std::slice::Iter<'_, SemanticEvent>>(),
        size_of::<Result<String, TryReserveError>>(),
        size_of::<Result<Vec<SemanticEvent>, TryReserveError>>(),
        size_of::<Result<CopiedJournal, TryReserveError>>(),
        size_of::<Result<CopiedJournal, TextHostCopyError>>(),
        size_of::<std::fmt::Arguments<'_>>(),
        size_of::<std::fmt::Result>(),
        size_of::<usize>(),
        size_of::<Layout>(),
        BackendFailure::source_retention_peak_bytes::<CopyFailure>()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl<'a, B: TextSnapshotBackend> CaptureJournal<'a, B> {
    pub(in crate::api::control) fn new(
        delivery: &'a Delivery,
        tokenizer: [u8; 32],
        pending: Option<u32>,
        status: GenerationStatus,
        prediction: u64,
    ) -> Self {
        Self {
            delivery,
            identity: PreparedIdentity::new("snapshot"),
            tokenizer,
            pending,
            status,
            prediction,
            backend: std::marker::PhantomData,
        }
    }
    fn text_bytes(&self) -> Option<usize> {
        let c = &self.delivery.template;
        [
            self.identity.bytes()?.try_into().ok()?,
            c.session_id.len(),
            c.run_id.len(),
            c.artifact_identity.as_ref().map_or(0, String::len),
            c.capture_plan_id.as_ref().map_or(0, String::len),
            c.intervention_plan_id.as_ref().map_or(0, String::len),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    fn copy_inner(
        self,
        retained: u64,
        host: &HostPreparationAuthority,
    ) -> Result<CopiedJournal, TryReserveError> {
        let c = &self.delivery.template;
        let mut snapshot_id = String::new();
        snapshot_id.try_reserve_exact(self.identity.bytes().unwrap().try_into().unwrap())?;
        self.identity
            .write_into(&mut snapshot_id)
            .expect("funded String writer");
        let info = SnapshotInfo {
            snapshot_id,
            session_id: string(&c.session_id)?,
            artifact_identity: optional(&c.artifact_identity)?,
            capture_plan_id: optional(&c.capture_plan_id)?,
            intervention_plan_id: optional(&c.intervention_plan_id)?,
            tokenizer_identity: self.tokenizer,
            configuration_identity: self.delivery.configuration_identity,
            pending_forced_token: self.pending,
            status: self.status,
            output: GenerationOutputCheckpointData {
                run_id: string(&c.run_id)?,
                epoch: self.delivery.epoch,
                next_sequence: self.delivery.sequence,
                next_prediction: self.prediction,
            },
            retained_bytes: retained,
        };
        let data = GenerationSnapshotData {
            schema_version: PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION,
            snapshot_id: string(&info.snapshot_id)?,
            session_id: string(&info.session_id)?,
            artifact_identity: optional(&info.artifact_identity)?,
            prompt_attribution: self.delivery.prompt.prepared().clone(),
            instrumentation: match (&info.capture_plan_id, &info.intervention_plan_id) {
                (Some(capture), Some(intervention)) => PreparedInstrumentationRecord::Intervened {
                    capture_plan_id: string(capture)?,
                    intervention_plan_id: string(intervention)?,
                },
                (Some(capture), None) => PreparedInstrumentationRecord::Captured {
                    plan_id: string(capture)?,
                },
                (None, None) => PreparedInstrumentationRecord::Unobserved,
                _ => unreachable!("interventions require admitted capture"),
            },
            tokenizer_identity: info.tokenizer_identity,
            configuration_identity: info.configuration_identity,
            pending_forced_token: info.pending_forced_token,
            status: info.status,
            output: GenerationOutputCheckpointData {
                run_id: string(&info.output.run_id)?,
                ..info.output
            },
            retained_bytes: retained,
        };
        let metadata = GenerationSnapshotMetadata::new(data, host.clone());
        Ok(CopiedJournal {
            metadata,
            info,
            semantic_prefix: copy_events(&self.delivery.semantic_prefix)?,
            prompt: self.delivery.prompt.clone(),
            destination: host.clone(),
        })
    }
}
impl<B: TextSnapshotBackend> PreparedTextHostJournal for CaptureJournal<'_, B> {
    type Copied = CopiedJournal;
    fn storage_bytes(&self) -> Option<u64> {
        let bytes: u64 = self
            .text_bytes()?
            .checked_mul(2)?
            .checked_add(size_of::<CopiedJournal>())?
            .checked_add(size_of::<GenerationSnapshotData>())?
            .try_into()
            .ok()?;
        bytes
            .checked_add(self.delivery.semantic_prefix.snapshot_bytes()?)?
            .checked_add(self.delivery.prompt.logical_bytes()?)
    }
    fn preparation_bytes(&self) -> Option<usize> {
        self.text_bytes()?
            .checked_mul(2)?
            .checked_add(events_bytes(&self.delivery.semantic_prefix)?)?
            .checked_add(
                usize::try_from(ControlledGenerationRecord::snapshot_control_bytes()).ok()?,
            )?
            .checked_add(copy_controls::<Self>()?)?
            .checked_add(size_of::<ControlledGenerationSnapshot<B>>())?
            .checked_add(size_of::<
                Result<ControlledGenerationSnapshot<B>, ControlledGenerationError>,
            >())
    }
    fn copy(
        self,
        retained: u64,
        host: &HostPreparationAuthority,
    ) -> Result<CopiedJournal, TextHostCopyError> {
        self.copy_inner(retained, host).map_err(|cause| {
            TextHostCopyError::Source(BackendFailure::from_error(CopyFailure {
                cause,
                destination: host.clone(),
            }))
        })
    }
}

pub(in crate::api::control) struct RestoreJournal<'a> {
    pub(in crate::api::control) semantic_prefix: &'a [SemanticEvent],
    pub(in crate::api::control) prompt: &'a PromptRecord,
}
pub(in crate::api::control) struct RestoredJournal {
    pub(in crate::api::control) semantic_prefix: Vec<SemanticEvent>,
    pub(in crate::api::control) prompt: PromptRecord,
    pub(in crate::api::control) destination: HostPreparationAuthority,
}
impl PreparedTextHostJournal for RestoreJournal<'_> {
    type Copied = RestoredJournal;
    fn storage_bytes(&self) -> Option<u64> {
        self.semantic_prefix
            .iter()
            .try_fold(size_of::<RestoredJournal>() as u64, |bytes, event| {
                bytes.checked_add(event.snapshot_bytes()?)
            })?
            .checked_add(self.prompt.logical_bytes()?)
    }
    fn preparation_bytes(&self) -> Option<usize> {
        events_bytes(self.semantic_prefix)?
            .checked_add(copy_controls::<Self>()?)
            .and_then(|bytes| bytes.checked_add(size_of::<RestoredJournal>()))
    }
    fn copy(
        self,
        _: u64,
        host: &HostPreparationAuthority,
    ) -> Result<RestoredJournal, TextHostCopyError> {
        let semantic_prefix = copy_events(self.semantic_prefix).map_err(|cause| {
            TextHostCopyError::Source(BackendFailure::from_error(CopyFailure {
                cause,
                destination: host.clone(),
            }))
        })?;
        Ok(RestoredJournal {
            semantic_prefix,
            prompt: self.prompt.clone(),
            destination: host.clone(),
        })
    }
}

pub(in crate::api::control) fn funded_events(
    source: &[SemanticEvent],
    funding: &HostMetadataFunding,
) -> Result<Vec<SemanticEvent>, RecordConstructionError> {
    let result = (|| {
        let bytes = events_bytes(source)
            .and_then(|bytes| bytes.checked_add(copy_controls::<&[SemanticEvent]>()?))
            .ok_or(HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        copy_events(source).map_err(|_| RecordConstructionCause::HostAllocation)
    })();
    result.map_err(|cause| RecordConstructionError::retain(cause, funding))
}

/// A child gets fresh independently cancellable cells within the same original
/// destination as its copied semantic journal. Aliases retain that destination.
pub(in crate::api::control) struct BranchJournal<'a> {
    pub(in crate::api::control) restored: RestoreJournal<'a>,
}
pub(in crate::api::control) struct BranchedJournal {
    pub(in crate::api::control) restored: RestoredJournal,
    pub(in crate::api::control) control: GenerationControlHandle,
}
impl PreparedTextHostJournal for BranchJournal<'_> {
    type Copied = BranchedJournal;
    fn storage_bytes(&self) -> Option<u64> {
        self.restored
            .storage_bytes()?
            .checked_add(u64::try_from(GenerationControlHandle::construction_bytes()?).ok()?)
    }
    fn preparation_bytes(&self) -> Option<usize> {
        self.restored
            .preparation_bytes()?
            .checked_add(GenerationControlHandle::construction_bytes()?)?
            .checked_add(size_of::<BranchedJournal>())?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<Result<BranchedJournal, TextHostCopyError>>())
    }
    fn copy(
        self,
        bytes: u64,
        host: &HostPreparationAuthority,
    ) -> Result<BranchedJournal, TextHostCopyError> {
        let restored = self.restored.copy(bytes, host)?;
        let control = GenerationControlHandle::new_retained(host.clone());
        Ok(BranchedJournal { restored, control })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc,atomic::{AtomicBool,Ordering}};
    #[derive(Debug,Default)]
    struct Budget(std::sync::atomic::AtomicUsize);
    impl eredu_core::HostMetadataAccount for Budget {
        fn reserve_metadata(&self,bytes:usize)->Result<(),HostMetadataFundingError> {
            self.0.fetch_update(Ordering::SeqCst,Ordering::SeqCst,|used|
                used.checked_add(bytes).filter(|total|*total<=1<<20))
                .map(|_|()).map_err(|used|HostMetadataFundingError::Capacity {
                    required:bytes as u64,available:((1<<20)-used) as u64})
        }
    }
    #[derive(Debug)]
    struct Retirement(Arc<AtomicBool>);
    impl Drop for Retirement {fn drop(&mut self){self.0.store(true,Ordering::SeqCst);}}

    #[test]
    fn child_journal_preserves_partial_semantics_and_escaped_control_custody() {
        let funding=HostMetadataFunding::new(Budget::default()).unwrap();
        let prompt=PromptRecord::from_tokens(&[7,11,19],&funding).unwrap();
        let mut source=vec![SemanticEvent::ReasoningDelta("partial λ".into()),
            SemanticEvent::ToolCallStart {index:0,id:"call-0".into(),name:"lookup".into()},
            SemanticEvent::ToolArgumentsDelta {index:0,json_fragment:"{\"city\":\"Par".into()}];
        let expected=serde_json::to_vec(&source).unwrap();
        let journal=BranchJournal {restored:RestoreJournal {semantic_prefix:&source,prompt:&prompt}};
        assert!(journal.preparation_bytes().unwrap()>source.len()*size_of::<SemanticEvent>());
        let retired=Arc::new(AtomicBool::new(false));
        let host=HostPreparationAuthority::retain(Retirement(retired.clone()));
        let logical=journal.storage_bytes().unwrap();
        let copied=journal.copy(logical,&host).unwrap();
        source.clear();
        assert_eq!(serde_json::to_vec(&copied.restored.semantic_prefix).unwrap(),expected);
        assert_eq!(copied.restored.prompt.prepared().attribution().complete_token_ids(),Some([7,11,19].as_slice()));
        let cancel=copied.control.cancellation().clone();
        copied.control.request_pause();
        assert!(copied.control.pause_requested());
        drop((copied,host,prompt));
        assert!(!retired.load(Ordering::SeqCst));
        cancel.cancel();assert!(cancel.is_cancelled());
        drop(cancel);assert!(retired.load(Ordering::SeqCst));
    }
}
