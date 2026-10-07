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

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_architectures::voicechat::{
        observation::OwnedHostOutput,
        schedule::{Progress, Step},
    };
    use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
    use eredu_runtime::execution_control::SnapshotBudget;

    fn diagnostic_output(words: usize) -> VoiceChatConversationOutput<Vec<u32>> {
        VoiceChatConversationOutput {
            frame: OwnedHostOutput {
                invocation_index: 0,
                step: Step::Drain,
                reminder_token: None,
                before: Progress::default(),
                after: Progress::default(),
                decisions: None,
                generated_codes: None,
                speech: None,
                transcription: None,
                diagnostics: vec![vec![7; words]],
            },
            turn: None,
            tools: None,
            playback: None,
            agent_text: None,
            transcription: None,
        }
    }

    #[test]
    fn fork_queue_admits_diagnostic_storage_before_publication_and_preserves_restore_charge() {
        let mut capacity = crate::runtime::voicechat::conversation::tests::capacity();
        capacity.diagnostics = 1;
        let source = Queue::<Vec<u32>>::new(VoiceChatPublicationLimits {
            max_publications: 1,
            capacity,
        })
        .unwrap();
        let initial = source.borrow().fork_estimate(RequestId::new(42)).unwrap();
        let budget = SnapshotBudget::new(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: initial.retained_bytes + 4096,
            cumulative_copy_bytes: initial.copy_bytes,
        });
        let retention = budget
            .reserve(SnapshotResourceKind::Branch, Some(initial))
            .unwrap();
        let fork = source
            .borrow()
            .copy_snapshot_value(&retention)
            .unwrap()
            .fork_owner(RequestId::new(42), capacity, &retention)
            .unwrap();
        let permit = Queue::reserve(&fork, capacity).unwrap();
        let before = budget.usage();
        let mut oversized = diagnostic_output(1);
        oversized.frame.diagnostics[0].reserve_exact(8192);
        assert!(matches!(
            permit.prepare(&oversized),
            Err(VoiceChatPublicationError::Backend(_))
        ));
        assert_eq!(
            budget.usage(),
            before,
            "refused diagnostics consume no retained or copy budget"
        );
        assert_eq!(fork.borrow().usage().queued_publications, 0);
        assert!(Queue::pop(&fork).unwrap().is_none());
        let small = diagnostic_output(2);
        let charge = permit.prepare(&small).unwrap();
        assert!(budget.usage().retained_bytes > before.retained_bytes);
        assert_eq!(
            budget.usage().cumulative_copy_bytes,
            before.cumulative_copy_bytes
        );
        permit.commit(small, charge);
        assert_eq!(
            Queue::pop(&fork).unwrap().unwrap().frame.diagnostics,
            vec![vec![7, 7]]
        );
        let charged = budget.usage();
        // Restore retains the live child's high-water region even though this
        // saved queue contains no payload. Later delivery cannot refund growth.
        let mut candidate = source.borrow().copy_snapshot_value(&retention).unwrap();
        candidate
            .validate_snapshot_retention(&fork.borrow())
            .unwrap();
        candidate.retain_fork(Some(retention.clone()));
        *fork.borrow_mut() = candidate;
        let permit = Queue::reserve(&fork, capacity).unwrap();
        permit.prepare(&diagnostic_output(2)).unwrap();
        assert_eq!(budget.usage(), charged);
        assert!(permit.prepare(&oversized).is_err());
        assert_eq!(budget.usage(), charged);
        assert_eq!(fork.borrow().usage().queued_publications, 0);
    }
}
