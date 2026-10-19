//! Reattribution of undelivered output within the unchanged queue bounds.

use super::*;
use eredu_core::{execution_control::SnapshotEstimate, scheduler::RequestId};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotRetainedGrowth, SnapshotValue,
};

impl<D> Queue<D> {
    /// Charges future retained output before the enclosing neural commit.
    pub(super) fn validate_publication_retention(
        &self,
        output: &VoiceChatConversationOutput<D>,
    ) -> Result<(), VoiceChatPublicationError> {
        if let (Some(growth), Some(estimate)) = (&self.retained_growth, self.retained_estimator) {
            let bytes = estimate(self, Some(output)).map_err(VoiceChatPublicationError::Backend)?;
            growth.admit(bytes).map_err(|error| {
                VoiceChatPublicationError::Backend(BackendFailure::from_error(error))
            })?;
        }
        Ok(())
    }

    /// Preserves the live target's admission region across restore. Validation
    /// precedes installing either host or neural state.
    pub(in crate::runtime::voicechat) fn validate_snapshot_retention(
        &mut self,
        target: &Self,
    ) -> Result<(), VoiceChatPublicationError> {
        if let (Some(growth), Some(estimate)) = (&target.retained_growth, target.retained_estimator)
        {
            let bytes = estimate(self, None).map_err(VoiceChatPublicationError::Backend)?;
            growth.admit(bytes).map_err(|error| {
                VoiceChatPublicationError::Backend(BackendFailure::from_error(error))
            })?;
        }
        self.retained_growth = target.retained_growth.clone();
        self.retained_estimator = target.retained_estimator;
        Ok(())
    }

    pub(in crate::runtime::voicechat) fn retain_fork(
        &mut self,
        retention: Option<SnapshotReservation>,
    ) {
        self.fork_retention = retention;
    }

    pub(in crate::runtime::voicechat) fn validate_fork(
        &self,
        request: RequestId,
        invocation: VoiceChatPublicationCapacity,
    ) -> Result<(), VoiceChatPublicationError> {
        let mut growth = 0usize;
        for item in &self.output {
            if let Some(batch) = &item.output.tools {
                let extra = usize::try_from(
                    batch
                        .fork_id_growth(request)
                        .map_err(VoiceChatPublicationError::Backend)?,
                )
                .map_err(|_| VoiceChatPublicationError::Capacity("tool bytes"))?;
                if item
                    .charge
                    .0
                    .tool_bytes
                    .checked_add(extra)
                    .is_none_or(|bytes| bytes > invocation.tool_bytes)
                {
                    return Err(VoiceChatPublicationError::Capacity("tool bytes"));
                }
                growth = growth
                    .checked_add(extra)
                    .ok_or(VoiceChatPublicationError::Capacity("tool bytes"))?;
            }
        }
        if self
            .usage
            .queued
            .tool_bytes
            .checked_add(growth)
            .is_none_or(|bytes| bytes > self.limits.capacity.tool_bytes)
        {
            return Err(VoiceChatPublicationError::Capacity("tool bytes"));
        }
        Ok(())
    }
}

impl<D: SnapshotValue> Queue<D> {
    pub(in crate::runtime::voicechat) fn fork_estimate(
        &self,
        request: RequestId,
    ) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>(
            std::iter::once(self.estimate_snapshot_value()).chain(
                self.output
                    .iter()
                    .filter_map(|item| item.output.tools.as_ref())
                    .map(|batch| batch.fork_id_estimate(request)),
            ),
        )
    }

    /// Only the already independent, unpublished candidate is rewritten.
    pub(in crate::runtime::voicechat) fn fork_owner(
        mut self,
        request: RequestId,
        invocation: VoiceChatPublicationCapacity,
        retention: &SnapshotReservation,
    ) -> Result<Rc<RefCell<Self>>, VoiceChatPublicationError> {
        self.validate_fork(request, invocation)?;
        for item in &mut self.output {
            if let Some(batch) = &mut item.output.tools {
                batch
                    .rekey_for_fork(request)
                    .map_err(VoiceChatPublicationError::Backend)?;
                let bytes = invocation.validate_tools(Some(batch))?;
                self.usage.queued.tool_bytes -= item.charge.0.tool_bytes;
                self.usage.queued.tool_bytes += bytes;
                item.charge.0.tool_bytes = bytes;
            }
            item.output.playback = item
                .output
                .playback
                .map(|receipt| receipt.with_request(request));
        }
        self.delivered = Rc::new(());
        self.fork_retention = Some(retention.clone());
        self.retained_growth = Some(SnapshotRetainedGrowth::new(
            retention,
            self.estimate_snapshot_value()
                .map_err(VoiceChatPublicationError::Backend)?
                .retained_bytes,
        ));
        self.retained_estimator = Some(Self::prospective_retained_bytes);
        Ok(Rc::new(RefCell::new(self)))
    }

    fn prospective_retained_bytes(
        &self,
        output: Option<&VoiceChatConversationOutput<D>>,
    ) -> Result<u64, BackendFailure> {
        let current = self.estimate_snapshot_value()?.retained_bytes;
        let extra = output
            .map(SnapshotValue::estimate_snapshot_value)
            .transpose()?
            .map_or(0, |estimate| estimate.retained_bytes);
        current.checked_add(extra).ok_or_else(|| {
            BackendFailure::from_error(
                eredu_core::execution_control::ExecutionControlError::Overflow,
            )
        })
    }
}
