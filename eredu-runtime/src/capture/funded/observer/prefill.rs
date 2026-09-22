//! One logical p0 frame across independently committed native prompt chunks.
//! Only inline progress is added; the original bank owns the one frame/ledger.
use super::*;
use crate::prefill::PrefillChunk;
use eredu_core::OutputDemand;

use crate::capture::prefill_transaction::{self as transaction, Finish, Phase as ChunkPhase};
pub(super) struct PrefillAggregation {
    prompt_end: u64,
    cached: u64,
    next: u64,
    chunk_end: u64,
    current_epoch: Option<DistributedCommitEpoch>,
    expected_epoch: Option<DistributedCommitEpoch>,
    phase: ChunkPhase,
    failed: bool,
}
impl<T, E: std::error::Error + Send + Sync + 'static, N> FundedCaptureObserver<'_, T, E, N> {
    pub(super) fn active_transaction(&self) -> bool {
        self.epoch.is_some()
            && self
                .prefill
                .as_ref()
                .is_none_or(|p| p.phase == ChunkPhase::Prepared && !p.failed)
    }
    pub(super) fn current_fragment_chunk(&self) -> Result<PrefillChunk, CaptureProtocolError> {
        let bound = self.bound.ok_or(CaptureProtocolError::Transaction)?;
        let p = self
            .prefill
            .as_ref()
            .ok_or(CaptureProtocolError::Transaction)?;
        if !self.active_transaction() {
            return Err(CaptureProtocolError::Transaction);
        }
        Ok(PrefillChunk {
            input: p.next..p.chunk_end,
            position: p
                .cached
                .checked_add(p.next)
                .ok_or(CaptureProtocolError::Geometry)?,
            output: bound
                .geometry()
                .output
                .for_chunk(p.chunk_end == p.prompt_end),
        })
    }
    pub(super) fn begin_chunk(
        &mut self,
        chunk: &PrefillChunk,
    ) -> Result<(), FundedCaptureError<E>> {
        let result = self.begin_chunk_inner(chunk);
        if result.is_err() {
            if let Some(progress) = &mut self.prefill {
                progress.failed = true;
            }
        }
        result
    }
    fn begin_chunk_inner(&mut self, chunk: &PrefillChunk) -> Result<(), FundedCaptureError<E>> {
        if self.prediction != 0 {
            return Err(CaptureProtocolError::Geometry.into());
        }
        let (prompt_end, cached, next) = match &self.prefill {
            Some(p) if p.phase == ChunkPhase::Between && !p.failed => {
                (p.prompt_end, p.cached, p.next)
            }
            Some(_) => return Err(CaptureProtocolError::Transaction.into()),
            None => {
                if self.epoch.is_some() {
                    return Err(CaptureProtocolError::Transaction.into());
                }
                let Frame::Before(Some(bank)) = &self.frame else {
                    return Err(CaptureProtocolError::Transaction.into());
                };
                let admission = bank.source().admission();
                let origin = admission
                    .text_origin()
                    .ok_or(CaptureProtocolError::Invocation)?;
                (
                    admission.request().prompt_tokens,
                    origin.cached_positions,
                    0,
                )
            }
        };
        if chunk.input.start != next
            || chunk.input.end <= next
            || chunk.input.end > prompt_end
            || cached.checked_add(next) != Some(chunk.position)
            || cached.checked_add(prompt_end).is_none()
        {
            return Err(CaptureProtocolError::Geometry.into());
        }
        let expected = if let Some(bound) = self.bound {
            let geometry = bound.geometry();
            if chunk.input.end
                != next
                    .saturating_add(geometry.prefill_chunk_positions)
                    .min(prompt_end)
            {
                return Err(CaptureProtocolError::Geometry.into());
            }
            geometry.output.for_chunk(chunk.input.end == prompt_end)
        } else {
            let sequence =
                CaptureObservationStep::new(&self.session.plan, CapturePhase::Prefill, 0)?
                    .requires_sequence_readout();
            // Full prompt tensor attribution is unchanged. No chunk may substitute
            // shorter physical axes for a selected full-prompt capture.
            if sequence && (next != 0 || chunk.input.end != prompt_end) {
                return Err(CaptureProtocolError::PrefillAttribution.into());
            }
            if sequence {
                OutputDemand::Sequence
            } else {
                OutputDemand::LastPosition.for_chunk(chunk.input.end == prompt_end)
            }
        };
        if chunk.output != expected {
            return Err(CaptureProtocolError::Geometry.into());
        }
        match &mut self.prefill {
            Some(p) => {
                p.chunk_end = chunk.input.end;
                p.expected_epoch = None;
                p.phase = ChunkPhase::Announced;
            }
            None => {
                self.prefill = Some(PrefillAggregation {
                    prompt_end,
                    cached,
                    next,
                    chunk_end: chunk.input.end,
                    current_epoch: None,
                    expected_epoch: None,
                    phase: ChunkPhase::Announced,
                    failed: false,
                })
            }
        }
        Ok(())
    }
    pub(super) fn prepare_retention(
        &mut self,
        context: &crate::inspection::PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<crate::inspection::PreparedPrefillChunkRetention>, FundedCaptureError<E>>
    {
        if self
            .bound
            .is_some_and(|bound| bound.geometry() != context.geometry())
        {
            return Err(CaptureProtocolError::Geometry.into());
        }
        let progress = self
            .prefill
            .as_mut()
            .ok_or(CaptureProtocolError::Transaction)?;
        let chunk = context.chunk();
        if progress.phase != ChunkPhase::Announced
            || progress.failed
            || progress.expected_epoch.is_some()
            || chunk.input.start != progress.next
            || chunk.input.end != progress.chunk_end
            || progress.cached.checked_add(progress.next) != Some(chunk.position)
        {
            return Err(CaptureProtocolError::Transaction.into());
        }
        progress.expected_epoch = Some(context.epoch());
        let policy = CaptureObservationStep::new(&self.session.plan, CapturePhase::Prefill, 0)?;
        if matches!(self.frame, Frame::Empty) && !policy.has_selected_prefill_hook() {
            return Ok(None);
        }
        let bootstrap = match &self.frame {
            Frame::Before(Some(bank)) => bank.prefill_source_bootstrap(),
            Frame::Active(step) => step.prefill_source_bootstrap(),
            _ => return Err(CaptureProtocolError::Transaction.into()),
        }
        .map_err(CaptureRunHostError::from)?;
        let evidence = bootstrap.intervention_source().is_some_and(|source| {
            source
                .plan()
                .admission()
                .plan()
                .operations
                .iter()
                .zip(source.plan().admission().points())
                .any(|(operation, point)| {
                    operation.schedule.includes(CapturePhase::Prefill, 0)
                        && crate::intervention::InterventionPrefillWindow::row_axis(point)
                        && operation.evidence
                            != eredu_core::intervention::InterventionEvidence::None
                })
        });
        if !policy.has_selected_prefill_hook() && !evidence {
            return Ok(None);
        }
        self.backend
            .prepare_prefill_chunk_retention(bootstrap, context)
    }
    pub(super) fn prepare_chunk(
        &mut self,
        epoch: DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), FundedCaptureError<E>> {
        let valid = self.prefill.as_ref().is_some_and(|p| {
            pass == crate::ExpertPass::Prefill
                && transaction::can_prepare(
                    p.phase,
                    p.failed,
                    p.current_epoch,
                    p.expected_epoch,
                    epoch,
                )
        });
        let result = if !valid {
            Err(CaptureProtocolError::Transaction.into())
        } else if self.epoch.is_none() {
            self.prepare_first(epoch, pass)
        } else if !matches!(self.frame, Frame::Active(_) | Frame::Empty)
            || self.session.transaction
                != self.epoch.map(|e| (e, CaptureTransactionStatus::Pending))
            || self
                .session
                .last_transaction_epoch
                .is_some_and(|previous| previous >= epoch)
        {
            Err(CaptureProtocolError::Transaction.into())
        } else {
            // Preserve the logical transaction's first epoch, while preventing
            // any later forward from replaying an already entered chunk epoch.
            self.session.last_transaction_epoch = Some(epoch);
            Ok(())
        };
        let progress = self.prefill.as_mut().expect("annotated prefill");
        if result.is_ok() {
            progress.current_epoch = Some(epoch);
            progress.phase = ChunkPhase::Prepared;
        } else {
            progress.failed = true;
        }
        result
    }
    pub(super) fn complete_chunk(
        &mut self,
        epoch: DistributedCommitEpoch,
    ) -> Result<(), FundedCaptureError<E>> {
        let p = self.prefill.as_ref().expect("annotated prefill");
        let valid = transaction::can_complete(p.phase, p.failed, p.current_epoch, epoch);
        let final_chunk = p.chunk_end == p.prompt_end;
        let result = if !valid {
            Err(CaptureProtocolError::Transaction.into())
        } else {
            self.complete_fragment_chunk(final_chunk).and_then(|()| {
                if final_chunk {
                    self.seal_delivery(self.epoch.expect("prepared logical epoch"))
                } else {
                    Ok(())
                }
            })
        };
        let p = self.prefill.as_mut().expect("annotated prefill");
        if result.is_ok() {
            p.phase = ChunkPhase::Complete;
        } else {
            p.failed = true;
        }
        result
    }
    pub(super) fn finish_chunk(&mut self, epoch: DistributedCommitEpoch, committed: bool) {
        let p = self.prefill.as_mut().expect("annotated prefill");
        match transaction::finish(
            &mut p.phase,
            &mut p.failed,
            p.current_epoch,
            epoch,
            committed,
        ) {
            Finish::Committed => p.next = p.chunk_end,
            Finish::Aborted => {
                if let Some(logical) = self.epoch {
                    self.publish_terminal(logical, false);
                }
            }
            Finish::Ignored | Finish::Invalid => {}
        }
    }
    pub(super) fn finish_prompt(&mut self, committed: bool) {
        let Some(p) = &mut self.prefill else { return };
        if p.phase == ChunkPhase::Finished {
            return;
        }
        let accepted =
            committed && !p.failed && p.phase == ChunkPhase::Between && p.next == p.prompt_end;
        p.phase = ChunkPhase::Finished;
        if let Some(logical) = self.epoch {
            self.publish_terminal(logical, accepted);
        }
    }
}
