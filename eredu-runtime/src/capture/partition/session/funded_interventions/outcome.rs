//! Successful world proof borrowed from the same original operation owner.
use super::*;
/// Constructed only after the existing world frame validates every member.
/// It cannot be cloned or created from a charge, boolean or native value.
pub(crate) struct PartitionInterventionOutcome<'a> {
    source:&'a OriginalInterventionSource,operation:usize,phase:CapturePhase,prediction:u64,
    baseline:CaptureUsage,local:CaptureUsage,global:CaptureUsage,member:bool,
}
impl PartitionInterventionOutcome<'_> {
    pub(crate) fn source(&self)->&OriginalInterventionSource {self.source}
    pub(crate) fn operation(&self)->usize {self.operation}
    pub(crate) fn coordinate(&self)->(CapturePhase,u64) {(self.phase,self.prediction)}
    pub(crate) fn member(&self)->bool {self.member}
    pub(crate) fn expected(&self)->Result<CaptureUsage,CaptureError> {
        self.baseline.checked_add(if self.member {self.local}else{CaptureUsage::default()})
    }
    pub(crate) fn final_charge(&self)->Result<CaptureUsage,CaptureError> {self.baseline.checked_add(self.global)}
    pub(crate) fn additional(&self)->CaptureUsage {self.global}
}
impl<T:PartitionCaptureTransport> PreparedPartitionIntervention<'_,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    /// Use the existing fixed 18-word world protocol only after model completion.
    /// Missing/dropped local loans reject the whole operation, including on peers.
    pub(crate) fn deliver(&mut self,baseline:CaptureUsage) -> Result<PartitionInterventionOutcome<'_>, PartitionInterventionSourceError> {
        let transport = self.transport;
        self.validate_transport()?;
        let result = (|| -> Result<(), Cause> {
            if self.delivered {
                return Err(Cause::Source("world receipt already spent"));
            }
            self.delivered = true;
            self.receipt.reserve_quota(self.receipt_usage)?;
            let accepted = self
                .source
                .invocations
                .iter()
                .all(|row| matches!(row.state, State::Absent | State::Accepted));
            let receipt = PartitionInterventionReceipt::new(
                self.rank,
                self.source.world,
                self.epoch,
                self.source.operation,
                &self.source.members,
                &self.source.descriptor,
                false,
            )?;
            let words = receipt.frame(accepted, 0)?;
            let frame = PartitionCaptureFrame::new(
                PartitionCaptureFrameKind::InterventionReceipt,
                self.rank,
                self.source.world,
                &words,
                PartitionInterventionReceipt::WORDS,
            )?;
            let gathered =
                super::super::exchange::gather_capture_words(transport, self.wait, &frame)?;
            receipt.validate(&gathered)?;
            Ok(())
        })();
        if let Err(Cause::Exchange(cause)) = &result {
            if !matches!(cause, PartitionCaptureExchangeError::PeerRejected { .. }) {
                transport.fail_capture_exchange(cause);
            }
        }
        result.map_err(|cause| self.source.error(cause))?;
        Ok(PartitionInterventionOutcome {source:&self.source.source,operation:self.source.operation,
            phase:self.source.phase,prediction:self.source.prediction,baseline,local:self.local,
            global:self.global,member:self.source.members.contains(&self.rank)})
    }
}
