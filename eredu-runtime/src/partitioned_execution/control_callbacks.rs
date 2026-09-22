//! Named control callbacks shared by execution and prospective source visitors.
use super::*;
use crate::replicated_session::{
    ParallelControlCallbackVisitor, ParallelControlEvent, SessionTransactionControlOccurrence,
};

pub(super) fn require_bound<B, T, F>(
    run: F,
) -> impl FnOnce(Option<&B::CommunicationGroup>) -> Result<T, PartitionExecutionError>
where
    B: CommunicationBackend,
    F: FnOnce(Option<&B::CommunicationGroup>) -> Result<T, PartitionExecutionError>,
{
    move |bound| {
        let bound = bound.ok_or(PartitionExecutionError::CommunicationPolicyMismatch)?;
        run(Some(bound))
    }
}
fn visit_factory<B, Input, T, F, Q>(
    _factory: impl FnOnce(Input) -> F,
    occurrence: SessionTransactionControlOccurrence,
    visitor: &mut Q,
) -> Result<(), Q::Error>
where
    B: CommunicationBackend,
    Q: ParallelControlCallbackVisitor<B>,
    F: FnOnce(Option<&B::CommunicationGroup>) -> Result<T, PartitionExecutionError>,
{
    visitor.visit_group::<T, PartitionExecutionError, F>(occurrence)
}

impl<B, G, R, I> PartitionCommunication<B, G, R, I>
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    pub(super) fn model_control_callback<'a, V: PartitionCommitAgreement<B, G, R, I>>(
        &'a self,
        policy: &'a mut V,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        success: bool,
        executor: &'a B::Executor,
    ) -> impl FnOnce(
        Option<(
            &B::ParallelContext,
            &eredu_nn::workspace::HostMetadataFunding,
        )>,
    ) -> Result<bool, PartitionExecutionError>
    + use<'a, V, B, G, R, I> {
        move |prepared| {
            self.with_control_group(
                group,
                CommunicationOperation::FailureAgreement,
                ParallelControlEvent::Phase(phase),
                prepared,
                executor,
                self.phase_control_callback(policy, group, phase, success, executor, false),
            )
        }
    }
    pub(super) fn visit_model_control_callbacks<'a, V, Q>(
        &'a self,
        occurrence: crate::replicated_session::SessionModelControlOccurrence,
        visitor: &mut Q,
    ) -> Result<(), Q::Error>
    where
        V: PartitionCommitAgreement<B, G, R, I> + 'a,
        Q: ParallelControlCallbackVisitor<B>,
    {
        fn visit<B, Input, F, Q>(
            _: impl FnOnce(Input) -> F,
            occurrence: SessionTransactionControlOccurrence,
            visitor: &mut Q,
        ) -> Result<(), Q::Error>
        where
            B: CommunicationBackend,
            Q: ParallelControlCallbackVisitor<B>,
            F: FnOnce(
                Option<(
                    &B::ParallelContext,
                    &eredu_nn::workspace::HostMetadataFunding,
                )>,
            ) -> Result<bool, PartitionExecutionError>,
        {
            visitor.visit::<bool, PartitionExecutionError, F>(occurrence)
        }
        let ParallelControlEvent::Phase(phase) = occurrence.event() else {
            unreachable!()
        };
        visit::<B, _, _, Q>(
            |(policy, executor): (&'a mut V, &'a B::Executor)| {
                self.model_control_callback(
                    policy,
                    occurrence.declaration().group,
                    phase,
                    false,
                    executor,
                )
            },
            occurrence.callback(),
            visitor,
        )?;
        self.visit_control_callback::<V, Q>(occurrence.callback(), visitor)
    }
    pub(super) fn phase_control_callback<'a, V: PartitionCommitAgreement<B, G, R, I>>(
        &'a self,
        policy: &'a mut V,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        success: bool,
        executor: &'a B::Executor,
        recovery: bool,
    ) -> impl FnOnce(Option<&B::CommunicationGroup>) -> Result<bool, PartitionExecutionError>
    + use<'a, V, B, G, R, I> {
        move |bound| {
            if recovery {
                policy.agree_phase_after_prior_failure_with_group(
                    self, group, phase, success, executor, bound,
                )
            } else {
                policy.agree_phase_with_group(self, group, phase, success, executor, bound)
            }
        }
    }
    pub(super) fn commit_control_callback<'a, V: PartitionCommitAgreement<B, G, R, I>>(
        &'a self,
        policy: &'a mut V,
        group: CollectiveGroupId,
        epoch: DistributedCommitEpoch,
        executor: &'a B::Executor,
    ) -> impl FnOnce(
        Option<&B::CommunicationGroup>,
    ) -> Result<DistributedCommitOutcome, PartitionExecutionError>
    + use<'a, V, B, G, R, I> {
        move |bound| policy.commit_with_group(self, group, epoch, executor, bound)
    }
    pub(super) fn visit_control_callback<'a, V, Q>(
        &'a self,
        occurrence: SessionTransactionControlOccurrence,
        visitor: &mut Q,
    ) -> Result<(), Q::Error>
    where
        V: PartitionCommitAgreement<B, G, R, I> + 'a,
        Q: ParallelControlCallbackVisitor<B>,
    {
        match occurrence.event() {
            ParallelControlEvent::Phase(phase) => visit_factory::<B, _, _, _, Q>(
                |(policy, group, executor): (&'a mut V, CollectiveGroupId, &'a B::Executor)| {
                    require_bound::<B, _, _>(
                        self.phase_control_callback(policy, group, phase, false, executor, false),
                    )
                },
                occurrence,
                visitor,
            ),
            ParallelControlEvent::Commit => visit_factory::<B, _, _, _, Q>(
                |(policy, group, epoch, executor): (
                    &'a mut V,
                    CollectiveGroupId,
                    DistributedCommitEpoch,
                    &'a B::Executor,
                )| {
                    require_bound::<B, _, _>(
                        self.commit_control_callback(policy, group, epoch, executor),
                    )
                },
                occurrence,
                visitor,
            ),
        }
    }
}
