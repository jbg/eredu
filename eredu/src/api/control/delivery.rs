//! Bounded record publication over the canonical committed-token boundary.
use super::*;

pub(super) struct Delivery {
    pub(super) configuration_identity: [u8; 32],
    pub(super) template: RecordContext,
    pub(super) budget: TraceBudget,
    pub(super) control: GenerationControlHandle,
    pub(super) sequence: u64,
    pub(super) epoch: u64,
    pub(super) prediction: u64,
    pub(super) prompt: PromptRecord,
    pub(super) started: Instant,
    pub(super) preparation_elapsed: std::time::Duration,
    pub(super) timing: GenerationTiming,
    pub(super) closed: bool,
    pub(super) failure: Option<CaptureError>,
    pub(super) record_failure: Option<RecordConstructionError>,
    pub(super) semantic_prefix: Vec<SemanticEvent>,
    // Last, after every retained record producer and failure destination.
    pub(super) funding: HostMetadataFunding,
}
impl Delivery {
    pub(super) fn refuse_record(&mut self, cause: RecordConstructionCause) {
        self.record_failure = Some(RecordConstructionError::retain(cause, &self.funding));
        self.control.cancel();
        self.closed = true;
    }

    pub(super) fn prepare_semantic_prefix(
        &mut self,
        event: &SemanticEvent,
    ) -> Result<(), RecordConstructionCause> {
        let bytes = match event {
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
        .and_then(|bytes| {
            bytes.checked_add(size_of::<Option<SemanticEvent>>() + size_of::<SemanticEvent>())
        })
        .ok_or(HostMetadataFundingError::Overflow)?;
        self.funding.reserve_metadata(bytes)?;
        if self.semantic_prefix.len() == self.semantic_prefix.capacity() {
            let capacity = if self.semantic_prefix.capacity() == 0 {
                1
            } else {
                self.semantic_prefix
                    .capacity()
                    .checked_mul(2)
                    .ok_or(HostMetadataFundingError::Overflow)?
            };
            let bytes = std::alloc::Layout::array::<SemanticEvent>(capacity)
                .map_err(|_| HostMetadataFundingError::Overflow)?
                .size();
            self.funding.reserve_metadata(bytes)?;
            self.semantic_prefix
                .try_reserve_exact(capacity - self.semantic_prefix.len())
                .map_err(|_| RecordConstructionCause::HostAllocation)?;
        }
        Ok(())
    }

    pub(super) fn send(
        &mut self,
        event: impl Into<ControlEvent>,
        emit: &mut impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) {
        if self.closed || self.failure.is_some() || self.record_failure.is_some() {
            return;
        }
        let event = event.into();
        let semantic = match &event {
            ControlEvent::Existing(ObservedGenerationEvent::Semantic { event, .. }) => {
                if let Err(cause) = self.prepare_semantic_prefix(event) {
                    self.refuse_record(cause);
                    return;
                }
                Some(event.clone())
            }
            _ => None,
        };
        let record = ControlledGenerationRecord::record(
            &self.template,
            &self.prompt,
            event,
            self.sequence,
            self.epoch,
            self.timing,
            &self.funding,
        );
        let record = match record {
            Ok(record) => record,
            Err(error) => {
                self.record_failure = Some(error);
                self.control.cancel();
                self.closed = true;
                return;
            }
        };
        let Some(next) = self.sequence.checked_add(1) else {
            self.failure = Some(CaptureError::Overflow);
            self.control.cancel();
            return;
        };
        let controls = TraceBudget::counting_control_bytes()
            .ok_or(HostMetadataFundingError::Overflow)
            .and_then(|bytes| self.funding.reserve_metadata(bytes));
        if let Err(cause) = controls {
            self.refuse_record(cause.into());
            return;
        }
        if let Err(error) = self.budget.charge(&record) {
            self.failure = Some(error);
            self.control.cancel();
            return;
        }
        // Charge and advance before invoking user code, including a caught panic.
        self.sequence = next;
        if let Some(event) = semantic {
            self.semantic_prefix.push(event);
        }
        if emit(record).is_break() {
            self.closed = true;
            self.control.cancel();
        }
    }
    pub(super) fn token(
        &mut self,
        token: Option<u32>,
        forced: bool,
        captures: Option<SharedCapturedStep>,
        seconds: f64,
        emit: &mut impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) {
        let prediction_index = self.prediction;
        let input_range = match self.prompt.range(prediction_index) {
            Ok(range) => range,
            Err(_) => {
                self.failure = Some(CaptureError::Overflow);
                self.control.cancel();
                return;
            }
        };
        match token {
            Some(token_id) => {
                let Some(next) = self.prediction.checked_add(1) else {
                    self.failure = Some(CaptureError::Overflow);
                    self.control.cancel();
                    return;
                };
                self.prediction = next;
                self.send(
                    ObservedGenerationEvent::from_token_delivery(
                        token_id,
                        forced,
                        prediction_index,
                        input_range,
                        captures,
                        seconds,
                    ),
                    emit,
                );
            }
            None => {
                if let Some(captures) = captures {
                    self.send(
                        ObservedGenerationEvent::from_failed_delivery(
                            prediction_index,
                            input_range,
                            captures,
                            seconds,
                        ),
                        emit,
                    );
                }
            }
        }
    }
}
