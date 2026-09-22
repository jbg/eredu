//! Existing original agreement source/role lent to a capture member vote.
use super::*;
use eredu_runtime::capture::partition::PartitionCaptureHookTransport;

impl<O: CaptureSourceOwner> OriginalCaptureTransport<O> {
    fn complete_member_vote(
        &self,
        members: &[usize],
        wait: BoundedCompletionWait,
        success: bool,
    ) -> Result<bool, Error> {
        let owner = &self.owner;
        let c = owner.custody();
        controls(c)?;
        owner.validate_active()?;
        c.funding()
            .reserve_metadata(Self::member_vote_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        self.estimate_capture_hook(members)
            .map_err(|cause| fail(CaptureCause::Protocol(cause.into()), c))?;
        if members.len() < 2 || !members.contains(&self.capture_rank()) {
            return Err(fail(CaptureCause::Identity, c));
        }
        let source = owner.communication_source()?;
        source.validate()?;
        if !self.authority.same_authority(source.authority)
            || !c.source().same_source(source.source())
            || self
                .authority
                .completion_policy()
                .is_none_or(|policy| policy.bounded_wait() != wait)
        {
            return Err(fail(CaptureCause::Identity, c));
        }
        let descriptor = MlxDistributedSession::capture_hook_source(
            source.source().manifest().groups(),
            members,
        )
        .ok_or_else(|| fail(CaptureCause::Identity, c))?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(descriptor.id(), CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), c.source(), c.funding()))?;
        let group = source
            .group(selected.order())
            .ok_or_else(|| fail(CaptureCause::Identity, c))?
            .0;
        let stream = group
            .retained_transport_stream()
            .ok_or_else(|| fail(CaptureCause::Identity, c))?;
        let phase = DistributedExecutionPhase::ObservationCoordination;
        let event = ParallelControlEvent::Phase(phase);
        // Existing request issuance consumes this occurrence before any native
        // role can start. Its selected Group comes from this same actual source.
        let invocation = match owner.prepare_agreement(event, Some(group)) {
            Ok(invocation) => invocation,
            Err(cause) => {
                owner.failed().set(true);
                return Err(cause);
            }
        };
        if owner.failed().get() || owner.running().replace(true) {
            owner.failed().set(true);
            return Err(fail(CaptureCause::Identity, c));
        }
        let _running = Running {
            running: owner.running(),
            failed: owner.failed(),
        };
        let result = owner
            .run_agreement(
                invocation,
                stream,
                member_vote_callback(MemberVote {
                    custody: c,
                    source: &source,
                    stream,
                    success,
                }),
            )
            .map_err(|cause| Error::with_original_control_source(cause, false))
            .and_then(|result| result);
        if result.is_err() {
            owner.failed().set(true);
        }
        result
    }
}
impl<O: CaptureSourceOwner> PartitionCaptureHookTransport for OriginalCaptureTransport<O> {
    // Original completion remains inside run_role; no synthetic native event or
    // completed output is inserted into the legacy submission-only interface.
    type HookOutput = ();
    fn estimate_capture_hook(&self, members: &[usize]) -> Result<CaptureUsage, CaptureError> {
        let usage = MlxDistributedSession::capture_hook_usage(self.participant_count(), members)?;
        if members.len() > 1
            && MlxDistributedSession::capture_hook_source(&self.groups, members).is_none()
        {
            return Err(CaptureError::Unsupported(
                "capture hook group has no selected exact failure agreement".into(),
            ));
        }
        Ok(usage)
    }
    fn complete_capture_hook(
        &self,
        members: &[usize],
        wait: BoundedCompletionWait,
        success: bool,
    ) -> Result<bool, PartitionCaptureExchangeError> {
        self.complete_member_vote(members, wait, success)
            .map_err(|cause| PartitionCaptureExchangeError::Backend(cause.into_backend_failure()))
    }
    fn submit_capture_hook(
        &self,
        _members: &[usize],
        _success: bool,
    ) -> Result<Submission<Self::HookOutput, MlxNeuralCommunicationCompletion>, Error> {
        let c = self.owner.custody();
        controls(c)?;
        Err(fail(CaptureCause::Identity, c))
    }
    fn resolve_capture_hook(&self, _output: Self::HookOutput) -> Result<bool, Error> {
        let c = self.owner.custody();
        controls(c)?;
        Err(fail(CaptureCause::Identity, c))
    }
}

struct MemberVote<'a, 'native, C: CaptureSourceCustody> {
    custody: &'a C,
    source: &'a OriginalCommunicationSource<'native>,
    stream: &'a Stream,
    success: bool,
}
impl<C: CaptureSourceCustody> MemberVote<'_, '_, C> {
    fn run(self, group: &Group) -> Result<bool, Error> {
        let c = self.custody;
        let source = self.source;
        let stream = self.stream;
        let phase = DistributedExecutionPhase::ObservationCoordination;

        let binding = group
            .original_control()
            .ok_or_else(|| fail(CaptureCause::Identity, c))?;
        let (output, completion) = binding
            .agree(group, self.success, stream)
            .map_err(|cause| {
                source.authority.fence_protocol_failure(
                    CommunicationOperation::FailureAgreement,
                    phase,
                    None,
                );
                cause
            })?;
        let output = source
            .authority
            .wait_with_error(
                Submission { output, completion },
                CommunicationOperation::FailureAgreement,
                phase,
                None,
                |cause: Error| PartitionExecutionError::PreparedCommunication {
                    operation: CommunicationOperation::FailureAgreement,
                    phase,
                    completion: true,
                    source: cause.into_backend_failure(),
                },
            )
            .map_err(|cause| fail(CaptureCause::Communication(cause), c))?;
        output.resolve()
    }
}
fn member_vote_callback<'a, 'native: 'a, C: CaptureSourceCustody>(
    vote: MemberVote<'a, 'native, C>,
) -> impl FnOnce(&Group, &HostMetadataFunding) -> Result<bool, Error> + 'a {
    move |group, _funding| vote.run(group)
}
fn member_callback_sizes<C: CaptureSourceCustody>() -> (usize, usize) {
    fn sizes<'a, 'native: 'a, C: CaptureSourceCustody, F>(
        _: impl FnOnce(MemberVote<'a, 'native, C>) -> F,
    ) -> (usize, usize)
    where
        F: FnOnce(&Group, &HostMetadataFunding) -> Result<bool, Error>,
    {
        (
            size_of::<F>(),
            super::owner::agreement_context_size::<bool, Error, F>(),
        )
    }
    sizes(member_vote_callback::<C>)
}
impl<O: CaptureSourceOwner> OriginalCaptureTransport<O> {
    fn member_vote_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<(&Self, &[usize], BoundedCompletionWait, bool)>(),
            size_of::<OriginalParallelControlInvocation>(),
            size_of::<ParallelControlEvent>(),
            size_of::<Result<OriginalParallelControlInvocation, Error>>(),
            size_of::<Result<bool, Error>>(),
            size_of::<
                Submission<
                    crate::backend::runtime::distributed::completion::OriginalCommunicationBool,
                    MlxNeuralCommunicationCompletion,
                >,
            >(),
            size_of::<Result<bool, eredu_core::BackendFailure>>(),
            size_of::<Result<Result<bool, Error>, eredu_core::BackendFailure>>(),
            size_of::<(&Group, &Stream)>(),
            size_of::<Option<&CommunicationGroupDescriptor>>(),
            failure_control_bytes()?,
            size_of::<MemberVote<'_, '_, O::Custody>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// One real local member vote, using the actual immutable status input and
    /// selected group. Callback controls come from the same named factories.
    pub(crate) fn member_vote_requirements(
        source: &OriginalCommunicationSource<'_>,
        inputs: &super::super::super::super::agreement::OriginalAgreementInputs,
        members: &[usize],
        source_loan: usize,
    ) -> Result<super::quote::GatherRequirements, Error> {
        use crate::backend::submission_recovery::native_role::{self, NativeRoleCapacity};
        reserve(
            source.funding(),
            &[
                size_of::<(
                    &OriginalCommunicationSource<'_>,
                    &super::super::super::super::agreement::OriginalAgreementInputs,
                    &[usize],
                    usize,
                )>(),
                size_of::<super::quote::GatherRequirements>(),
                size_of::<Result<super::quote::GatherRequirements, Error>>(),
                size_of::<safemlx::StreamCopyPlan<()>>(),
                size_of::<(usize, usize)>(),
                CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let unknown = || failure(Cause::Resource, source.source(), source.funding());
        let rank = source.source().manifest().rank();
        if members.len() < 2 || !members.contains(&rank) {
            return Err(unknown());
        }
        let descriptor = MlxDistributedSession::capture_hook_source(
            source.source().manifest().groups(),
            members,
        )
        .ok_or_else(unknown)?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(descriptor.id(), CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), source.source(), source.funding()))?;
        let group = source.group(selected.order()).ok_or_else(unknown)?.0;
        let stream = group.retained_transport_stream().ok_or_else(unknown)?;
        let comparison = safemlx::StreamCopyPlan::<()>::capture(stream)
            .map_err(|_| unknown())?
            .source_comparison_control_bytes()
            .ok_or_else(overflow)?;
        let vote = inputs.requirements_for_group(source, descriptor.id())?;
        let (callback, context) = member_callback_sizes::<O::Custody>();
        let capacity = NativeRoleCapacity {
            graph: vote.capacity.graph,
            records: vote.capacity.records,
            backing: vote.capacity.backing,
        };
        let validation =
            OriginalCommunicationSource::validation_control_bytes().ok_or_else(overflow)?;
        let parts = [
            control_bytes::<O::Custody>().ok_or_else(overflow)?,
            Self::member_vote_control_bytes().ok_or_else(overflow)?,
            source_loan,
            validation,
            OriginalParallelControlRequest::prepare_control_bytes().ok_or_else(overflow)?,
            source_loan,
            super::super::super::source::agreement_capacity_for_group_control_bytes()
                .ok_or_else(overflow)?,
            vote.capacity_metadata.checked_mul(2).ok_or_else(overflow)?,
            group
                .retention_copy_bytes()
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_add(size_of::<[usize; 1]>()))
                .ok_or_else(overflow)?,
            native_role::control_bytes::<OriginalParallelControlInvocation, O::Custody>(
                capacity, None,
            )
            .map_err(|_| unknown())?,
            native_role::callback_control_bytes::<bool, Error>(context).ok_or_else(overflow)?,
            OriginalParallelControlInvocation::context_control_bytes::<bool, Error>(callback)
                .ok_or_else(overflow)?,
            OriginalControlBinding::agreement_control_bytes().ok_or_else(overflow)?,
            OriginalControlBinding::loan_control_bytes().ok_or_else(overflow)?,
            OriginalControlBinding::validation_control_bytes().ok_or_else(overflow)?,
            comparison
                .checked_add(size_of::<[usize; 1]>())
                .ok_or_else(overflow)?,
            source_loan,
            source_loan,
            vote.execution_metadata,
        ];
        let metadata = parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        Ok(super::quote::GatherRequirements {
            capacity: vote.capacity,
            metadata,
        })
    }
}
