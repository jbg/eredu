//! The same original semantic/controller/input producer for one or many lanes.
use super::*;
use crate::api::portable::prepared_semantic::PreparedOriginalSpeculativeHost;
use eredu_core::{
    SpeculativeBuffer, SpeculativeGenerationLane, TextGenerationBackend, TextGenerationConfig,
};
use std::num::NonZeroUsize;

pub(super) enum PreparedPrompt<'a, P> {
    Text {
        text: &'a str,
        add_special_tokens: bool,
    },
    TokenIds(&'a [u32]),
    Media(P),
}

pub(super) struct PreparedLane<'a, B: TextGenerationBackend> {
    pub(super) host: PreparedOriginalSpeculativeHost<'a>,
    pub(super) input: PreparedPrompt<'a, B::Prompt>,
    pub(super) prefill_chunk_positions: Option<std::num::NonZeroU64>,
    pub(super) generation: TextGenerationConfig,
    pub(super) constraint: PreparedChatSpeculativeConstraint,
    pub(super) cancellation: GenerationCancellationToken,
    // The aggregate owns all its framework fields until they retire on error.
    pub(super) funding: HostMetadataFunding,
}
impl<'a, B: OriginalTokenizerBackend + SpeculativeGenerationBackend> PreparedLane<'a, B> {
    pub(super) fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    pub(super) fn finish(
        self,
        model: &LoadedModel<B>,
    ) -> Result<SpeculativeGenerationLane<'a, B, PreparedChatSpeculativeConstraint>, Cause> {
        reserve(
            &self.funding,
            sum(&[
                size_of::<Self>(),
                size_of::<&LoadedModel<B>>(),
                size_of::<Result<B::Prompt, BackendFailure>>(),
                size_of::<
                    Result<
                        SpeculativeGenerationLane<'a, B, PreparedChatSpeculativeConstraint>,
                        Cause,
                    >,
                >(),
            ]),
        )?;
        let prompt = match self.input {
            PreparedPrompt::Text {
                text,
                add_special_tokens,
            } => {
                let encoded = B::encode_original_text_ids(
                    &model.runtime,
                    self.host.preparation().tokenizer(),
                    text,
                    add_special_tokens,
                )?;
                if !encoded.matches_source(self.host.preparation().tokenizer()) {
                    return Err(TokenInputRejection::IdentityMismatch.into());
                }
                B::prepare_semantic_prompt(
                    &model.runtime,
                    self.host.preparation(),
                    &eredu_core::TokenIdsInputPlan::new(encoded.ids())?,
                    self.prefill_chunk_positions,
                )?
            }
            PreparedPrompt::TokenIds(ids) => B::prepare_semantic_prompt(
                &model.runtime,
                self.host.preparation(),
                &eredu_core::TokenIdsInputPlan::new(ids)?,
                self.prefill_chunk_positions,
            )?,
            PreparedPrompt::Media(prompt) => {
                B::validate_original_prepared_input_domain(
                    &model.runtime,
                    &prompt,
                    self.host.preparation().tokenizer(),
                )?;
                prompt
            }
        };
        Ok(self
            .host
            .into_lane(prompt, self.generation, self.constraint, self.cancellation)?)
    }
}

pub(super) fn buffer<T>(
    count: usize,
    funding: &HostMetadataFunding,
) -> Result<SpeculativeBuffer<T>, Cause> {
    reserve(
        funding,
        SpeculativeBuffer::<T>::retained_control_bytes(count)
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            })
            .and_then(|n| {
                n.checked_add(size_of::<(
                    &HostMetadataFunding,
                    usize,
                    Result<SpeculativeBuffer<T>, Cause>,
                )>())
            }),
    )?;
    Ok(SpeculativeBuffer::try_new_retained(
        count,
        HostPreparationAuthority::retain(funding.clone()),
    )?)
}

impl<B: OriginalTokenizerBackend + SpeculativeGenerationBackend> LoadedModel<B> {
    pub(super) fn prepare_managed_speculative_host<'a, F: FnMut(SemanticEvent) + 'a>(
        &self,
        source: &ManagedPlainTextSource,
        text: ManagedPlainTextRequest<'a>,
        maximum_draft: NonZeroUsize,
        cancellation: GenerationCancellationToken,
        on_event: F,
        retained: &mut Option<HostMetadataFunding>,
    ) -> Result<PreparedLane<'a, B>, Cause> {
        let limits = text.settings.inference.memory_limits.clone();
        let (generation, maximum) = self.resolve_text_generation_settings(text.settings.clone())?;
        let host = self.prepare_original_speculative_plain_host(
            source,
            &limits,
            maximum.get(),
            maximum_draft.get(),
            generation.sampling().temperature,
            &self.eos_token_ids,
            PreparedStops::Refs(text.stop_sequences),
            text.skip_special_tokens,
            on_event,
        )?;
        let funding = host.preparation().metadata_funding();
        // Retain before a reserve can fail. A later lane's own failed source
        // also remains inside its typed cause, independently of batch custody.
        if retained.is_none() {
            *retained = Some(funding.clone());
        }
        reserve(
            funding,
            sum(&[
                size_of::<PreparedLane<'a, B>>(),
                size_of::<Result<PreparedLane<'a, B>, Cause>>(),
                size_of::<(
                    ManagedPlainTextRequest<'a>,
                    NonZeroUsize,
                    GenerationCancellationToken,
                    F,
                )>(),
                size_of::<Cause>(),
                size_of::<ManagedPlainTextSpeculativeError>(),
                size_of::<Option<HostMetadataFunding>>(),
                size_of::<Option<SpeculativeControlError>>(),
                size_of::<PreparedChatSpeculativeConstraint>(),
            ])
            .and_then(|n| {
                n.checked_add(BackendFailure::source_retention_peak_bytes::<B::Error>()?)
            }),
        )?;
        reserve(
            funding,
            ConstraintController::text_metadata_bytes().and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            }),
        )?;
        let controller = ConstraintController::text_prepared(
            self.token_validity.clone(),
            HostPreparationAuthority::retain(funding.clone()),
        );
        let funding = funding.clone();
        Ok(PreparedLane {
            host,
            input: PreparedPrompt::Text {
                text: text.input,
                add_special_tokens: text.add_special_tokens,
            },
            prefill_chunk_positions: text.settings.inference.prefill_chunk_positions,
            generation,
            constraint: PreparedChatSpeculativeConstraint::new(controller),
            cancellation,
            funding,
        })
    }
}

impl<B: OriginalTokenizerBackend + SpeculativeGenerationBackend> LoadedModel<B> {
    pub(super) fn validate_speculative_settings(
        &self,
        source: &ManagedPlainTextSource,
        settings: impl Iterator<Item = crate::api::PreparedChatGenerationSettings>,
    ) -> Result<(), Cause> {
        if !source.original().matches_configuration(&self.tokenizer) {
            return Err(TokenInputRejection::IdentityMismatch.into());
        }
        for setting in settings {
            self.resolve_text_generation_settings(setting)?;
        }
        Ok(())
    }

    /// All hosts are admitted before the shared prompt stage. Single requests
    /// use a one-element iterator, so there is no unaccounted staging Vec.
    pub(super) fn prepare_speculative_batch<'a, I, P>(
        &self,
        drafting: SpeculativeDraft<'a, B::Drafter>,
        mut input: I,
        validation: Result<(), Cause>,
        scheduler: eredu_core::SpeculativeSchedulerOptions,
        retained: &mut Option<HostMetadataFunding>,
        mut prepare: P,
    ) -> Result<
        Option<
            eredu_core::SpeculativeGenerationBatchRequest<
                'a,
                B,
                B::Drafter,
                PreparedChatSpeculativeConstraint,
            >,
        >,
        Cause,
    >
    where
        I: ExactSizeIterator,
        P: FnMut(I::Item, &mut Option<HostMetadataFunding>) -> Result<PreparedLane<'a, B>, Cause>,
    {
        use eredu_core::run_preparation::TextPreparationStage as Stage;
        use eredu_core::SpeculativeGenerationBatchRequest as Request;
        let count = input.len();
        let local = (|| {
            validation?;
            scheduler.validate()?;
            let Some(first) = input.next() else {
                return Ok(None);
            };
            let first = prepare(first, retained)?;
            let funding = first.funding();
            reserve(
                funding,
                sum(&[
                    size_of::<I>(),
                    size_of::<I::Item>(),
                    size_of::<P>(),
                    size_of::<Option<PreparedLane<'a, B>>>(),
                    size_of::<Result<Option<SpeculativeBuffer<PreparedLane<'a, B>>>, Cause>>(),
                    size_of::<Request<'a, B, B::Drafter, PreparedChatSpeculativeConstraint>>(),
                    size_of::<
                        Result<
                            Option<Request<'a, B, B::Drafter, PreparedChatSpeculativeConstraint>>,
                            Cause,
                        >,
                    >(),
                    size_of::<(
                        SpeculativeDraft<'a, B::Drafter>,
                        eredu_core::SpeculativeSchedulerOptions,
                        usize,
                    )>(),
                ]),
            )?;
            let mut prepared = buffer(count, funding)?;
            prepared.try_push(first)?;
            for lane in input {
                prepared.try_push(prepare(lane, retained)?)?;
            }
            Ok(Some(prepared))
        })();
        let prepared =
            self.runtime
                .finish_text_preparation(Stage::Request, local, Cause::Backend)?;
        let Some(prepared) = prepared else {
            return Ok(None);
        };
        let funding = retained
            .as_ref()
            .expect("first prepared lane retains batch custody");
        let local = (|| {
            let mut lanes = buffer::<
                SpeculativeGenerationLane<'a, B, PreparedChatSpeculativeConstraint>,
            >(count, funding)?;
            for lane in prepared {
                lanes.try_push(lane.finish(self)?)?;
            }
            Ok(lanes)
        })();
        let lanes = self
            .runtime
            .finish_text_preparation(Stage::Prompt, local, Cause::Backend)?;
        Ok(Some(Request::new(
            drafting,
            lanes,
            self.tokenizer_fingerprint,
        )))
    }

    pub(super) fn execute_speculative_batch<'a>(
        &mut self,
        request: Option<
            eredu_core::SpeculativeGenerationBatchRequest<
                'a,
                B,
                B::Drafter,
                PreparedChatSpeculativeConstraint,
            >,
        >,
        scheduler: eredu_core::SpeculativeSchedulerOptions,
        retained: &Option<HostMetadataFunding>,
    ) -> Result<eredu_core::SpeculativeGenerationBatchOutput, Cause> {
        let Some(request) = request else {
            return Ok(eredu_core::SpeculativeGenerationBatchOutput::new(
                Vec::new(),
                Default::default(),
            ));
        };
        reserve(
            retained.as_ref().expect("prepared batch custody"),
            sum(&[
                size_of::<eredu_runtime::RunSpeculativeGeneration>(),
                size_of::<Result<eredu_core::SpeculativeGenerationBatchOutput, Cause>>(),
                size_of::<
                    Result<
                        eredu_core::SpeculativeGenerationBatchOutput,
                        ManagedPlainTextSpeculativeError,
                    >,
                >(),
                size_of::<
                    Result<
                        eredu_core::SpeculativeGenerationBatchOutput,
                        PreparedChatSpeculativeError,
                    >,
                >(),
            ]),
        )?;
        B::with_speculative_execution(
            &mut self.runtime,
            request,
            eredu_runtime::RunSpeculativeGeneration::new(scheduler),
        )
        .map_err(|error| Cause::Backend(B::into_backend_failure(error)))
    }
}
