//! Exact speculative role/ordinal binding around the common retained ingress.
use super::*;
use crate::working_memory::{OriginalSpeculativePrefillSpan, OriginalSpeculativeRole};
use eredu_nn::workspace::{
    HostMetadataFunding, HostMetadataFundingError, WorkspaceContext, WorkspaceMetadataAllocation,
};

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: PrefillIngressArchitecture<B, M::State, Error = eredu_nn::Error>,
    D: MediaTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Retains the actual selected media cut for an already admitted independent
    /// prefill role. This is not an inference request and issues no span claim.
    pub fn prepare_speculative_media_prefill(
        &self,
        plan: A::IngressPlan,
        role: &OriginalSpeculativeRole,
        metadata: &WorkspaceContext,
    ) -> Result<
        PreparedMediaPrefill<A, B, M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        if metadata.metadata_funding().is_none() {
            return Err(ReplicatedTextSessionError::WorkingMemory(
                WorkingMemoryError::UnknownBound,
            ));
        }
        if !self.selected.exact_completion_available()
            || !self.mechanisms.supports_media_ingress_completion()
        {
            return Err(ReplicatedTextSessionError::WorkingMemory(
                WorkingMemoryError::CompletionUnavailable,
            ));
        }
        role.validate_prefill_geometry(A::ingress_geometry(&plan))
            .map_err(ReplicatedTextSessionError::WorkingMemory)?;
        let original = A::ingress_session_binding(&plan).ok_or(
            ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::IdentityMismatch),
        )?;
        let current = self
            .original_request_media_binding()
            .map_err(ReplicatedTextSessionError::WorkingMemory)?;
        if !original.matches(&current) {
            return Err(ReplicatedTextSessionError::WorkingMemory(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let cut = D::prepare_media_cut_with_metadata(&self.execution, &plan, metadata)
            .map_err(ReplicatedTextSessionError::Architecture)?;
        PreparedMediaPrefill::new_speculative(
            plan,
            cut,
            role.clone(),
            current.revision.clone(),
            metadata,
        )
        .map_err(ReplicatedTextSessionError::Architecture)
    }

    /// Executes one claimed media span using the same input transaction, selected
    /// encoder/decoder traversal, publication and completion vote. The supplied
    /// checkpoint and callback belong to the actual accepted speculative role.
    pub fn prefill_media_span_with_checkpoint_and_completion<F>(
        &mut self,
        source: &mut PreparedMediaPrefill<A, B, M::State>,
        span: &OriginalSpeculativePrefillSpan,
        context: &<B::Tensor as Tensor>::Context,
        checkpoint: M::StateCheckpoint,
        funding: &HostMetadataFunding,
        complete: F,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        M::Error: From<HostMetadataFundingError>,
        F: FnOnce(
            Option<&B::Tensor>,
            &M::State,
            &crate::media_prefill::RetainedMediaRoots<'_, B::Tensor>,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(), M::Error>,
    {
        let controls = std::mem::size_of::<(
            F,
            Option<M::StateCheckpoint>,
            Option<B::Tensor>,
            Result<
                Option<B::Tensor>,
                ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
            >,
            Result<
                (Option<B::Tensor>, M::StateCheckpoint, A::ForwardContext),
                ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
            >,
            crate::media_prefill::RetainedMediaRoots<'_, B::Tensor>,
            Result<(), WorkingMemoryError>,
        )>();
        funding
            .reserve_metadata(controls)
            .map_err(|cause| ReplicatedTextSessionError::Mechanism(cause.into()))?;
        let input = (|| {
            source
                .validate_speculative_span(span)
                .map_err(ReplicatedTextSessionError::WorkingMemory)?;
            let current = self
                .original_request_media_binding()
                .map_err(ReplicatedTextSessionError::WorkingMemory)?;
            source
                .validate_revision(&current.revision)
                .map_err(ReplicatedTextSessionError::WorkingMemory)?;
            if current.frontier() != span.chunk().position
                || !source
                    .semantic_binding()
                    .is_some_and(|original| original.same_origin(&current))
            {
                return Err(ReplicatedTextSessionError::WorkingMemory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
            Ok(span.chunk())
        })();
        let result =
            self.with_observation_transaction(&mut crate::NoopObserver, |session, observer| {
                let (output, checkpoint, forward) = session.execute_media_span_before_publication(
                    source,
                    input,
                    span.chunk().output,
                    context,
                    observer,
                    Some(checkpoint),
                )?;
                if output.is_some() != (span.chunk().output != eredu_core::OutputDemand::StateOnly)
                {
                    return session.rollback_failure(
                        checkpoint,
                        ReplicatedTextSessionError::WorkingMemory(
                            WorkingMemoryError::IdentityMismatch,
                        ),
                        context,
                    );
                }
                let (output, checkpoint, forward) = session
                    .publish_observed_output_transaction_with_readout(
                        output, checkpoint, forward, context,
                    )?;
                let completion =
                    complete(output.as_ref(), &session.state, &source.roots(), context)
                        .map_err(ReplicatedTextSessionError::Mechanism);
                session
                    .finish_publication(output, checkpoint, forward, context, observer, completion)
            });
        match result {
            Ok(output) => {
                source
                    .committed(self.state.inference_retention().revision())
                    .map_err(|cause| {
                        ReplicatedTextSessionError::Architecture(funding.metadata_source(cause))
                    })?;
                Ok(output)
            }
            Err(cause) => {
                source.failed();
                Err(cause)
            }
        }
    }
}
