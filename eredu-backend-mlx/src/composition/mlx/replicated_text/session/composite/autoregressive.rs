//! Independent lanes reuse the selected composite transaction and exact cache loan.
use super::*;

impl<A, D, P> CompletedComposite<A, D, P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        >,
{
    pub(super) fn autoregressive_forward_inner(
        &mut self,
        tokens: &Array,
        cache: &mut MlxPredictionTargetState,
        prefill: bool,
        demand: eredu_core::OutputDemand,
        stream: &Stream,
        mut completion: Option<&mut dyn AutoregressiveSequenceCompletion>,
    ) -> Result<Option<Array>, Error> {
        let metadata = completion.as_ref().map(|value| value.metadata_context());
        let funding = metadata
            .as_ref()
            .and_then(|context| context.metadata_funding());
        if let Some(context) = &metadata {
            context
                .charge_metadata(std::mem::size_of::<(
                    Option<super::super::mechanisms::StateCheckpoint<MlxHybridState>>,
                    MlxPredictionTargetState,
                    Result<MlxPredictionTargetState, Error>,
                    Result<Option<MlxTensor>, Error>,
                    Result<Option<Array>, Error>,
                    PreparedCompositeInput<'_, MlxTensor, A::InputPartPlan>,
                )>())
                .map_err(|cause| Error::Neural(cause.into()))?;
        }
        let checkpoint = match completion.as_mut() {
            Some(completion) => {
                let mut source = completion.take_checkpoint()?;
                if !source.is::<MlxHybridState>() {
                    return Err(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                    ));
                }
                Some(super::super::mechanisms::StateCheckpoint::independent(
                    source.take_state::<MlxHybridState>()?,
                ))
            }
            None => None,
        };
        let (prepared, admitted, _) = decode_input::prepare::<A>(
            &self.admission,
            &self.processor,
            tokens,
            metadata.as_ref(),
        )?;
        let paired = match metadata.as_ref() {
            Some(context) => {
                PreparedCompositeInput::new_with_metadata(&prepared, &admitted, context)
                    .map_err(Error::Neural)?
            }
            None => PreparedCompositeInput::new(&prepared, &admitted)
                .map_err(Error::ArchitectureModel)?,
        };
        let output = external_state::with_state_and_metadata(
            &mut self.session,
            cache,
            stream,
            funding.as_ref(),
            |session, _prior| {
                let result = match completion {
                    Some(completion) if prefill => session
                        .prefill_span_with_checkpoint_and_completion(
                            paired,
                            demand,
                            stream,
                            checkpoint.expect("original completion retains checkpoint"),
                            funding
                                .as_ref()
                                .expect("original completion retains funding"),
                            |output, state, stream| {
                                completion.complete(output.map(MlxTensor::as_array), state, stream)
                            },
                        ),
                    Some(completion) => session
                        .sequence_logits_with_checkpoint_and_completion(
                            paired,
                            eredu_runtime::ExpertPass::Decode,
                            stream,
                            checkpoint.expect("original completion retains checkpoint"),
                            |output, state, stream| {
                                completion.complete(Some(output.as_array()), state, stream)
                            },
                        )
                        .map(Some),
                    None => session
                        .sequence_logits(
                            paired,
                            if prefill {
                                eredu_runtime::ExpertPass::Prefill
                            } else {
                                eredu_runtime::ExpertPass::Decode
                            },
                            stream,
                        )
                        .map(Some),
                };
                result.map_err(|cause| match funding.as_ref() {
                    Some(funding) => Error::Neural(funding.metadata_source(cause)),
                    None => Error::Other(Box::new(cause)),
                })
            },
        )?;
        Ok(self.published(output.map(MlxTensor::into_array)))
    }
}

/// A and B keep their distinct owners: each selected decoder compiles its own
/// semantic A from the retained host I, while borrowing the completed immutable
/// native B through the enclosing packet. Binding happens with the actual lane
/// cache installed, before equation or ingress construction.
pub(super) fn prepare_media_semantics<A, D, P>(
    model: &mut CompletedComposite<A, D, P>,
    source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
    cache: &mut MlxPredictionTargetState,
    blueprint: &eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
    pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    funding: &eredu_nn::workspace::HostMetadataFunding,
    stream: &Stream,
) -> Result<eredu_architectures::media_plan::BoundPreparedMediaSemantics, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        >,
{
    use eredu_architectures::media_plan::{
        BoundPreparedMediaSemantics, MediaSemanticError, OriginalPreparedMediaSemantics,
        PreparedMediaSemanticCompile,
    };
    use eredu_nn::workspace::WorkspaceMetadataAllocation;
    use eredu_runtime::working_memory::{
        MediaSessionBinding, OriginalCompositeSemanticStorageError,
    };
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<PreparedMediaSemanticCompile<'_, '_>>(),
        size_of::<Result<PreparedMediaSemanticCompile<'_, '_>, MediaSemanticError>>(),
        size_of::<OriginalPreparedMediaSemantics<'_>>(),
        size_of::<Result<OriginalPreparedMediaSemantics<'_>, OriginalCompositeSemanticStorageError>>(
        ),
        size_of::<BoundPreparedMediaSemantics>(),
        size_of::<Result<BoundPreparedMediaSemantics, OriginalCompositeSemanticStorageError>>(),
        size_of::<Result<BoundPreparedMediaSemantics, Error>>(),
        size_of::<MediaSessionBinding>(),
        size_of::<
            Result<
                MediaSessionBinding,
                eredu_runtime::replicated_session::OriginalMediaBindingError,
            >,
        >(),
    ];
    funding
        .reserve_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                ))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let bind = model
        .bind_original_media
        .ok_or(Error::PrefillScopeUnavailable)?;
    source.validate_pool(pool).map_err(Error::PrefillControl)?;
    let original = blueprint
        .plan_original_media_semantics(source)
        .map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })?
        .compile(pool)
        .map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })?;
    let admission = &model.admission;
    external_state::with_state_and_metadata(
        &mut model.session,
        cache,
        stream,
        Some(funding),
        |session, _| {
            let binding = session
                .prepare_original_media_semantic_binding(funding)
                .map_err(|cause| {
                    crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
                })?;
            bind(admission, original, blueprint, source, binding).map_err(|cause| {
                crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
            })
        },
    )
}
