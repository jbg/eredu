//! Independently optional Gemma vision/audio roots and exact decoder spans.
use super::*;
use eredu_runtime::media_prefill::{MediaIngressError, PrefillIngressArchitecture};
use eredu_runtime::{PreparedInputInspector, PreparedModelInput, SharedPreparedInputCacheIdentity};

/// Actual family admission and original ordered semantic input.
pub struct MediaPrefillPlan<T> {
    prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
    admitted: Option<crate::media_plan::AdmittedCompositeInput<Gemma4InputPartPlan>>,
    original: Option<crate::media_plan::BoundPreparedMediaSemantics>,
    workspace: Option<(
        eredu_runtime::input::OriginalPreparedInputProjection,
        eredu_runtime::input::OriginalPreparedWorkspaceSource,
        Option<eredu_runtime::working_memory::WorkingMemoryUnquotedLease>,
    )>,
    geometry: eredu_core::InferenceGeometry,
    fingerprint: String,
    identity: Option<SharedPreparedInputCacheIdentity>,
    // All source and native/metadata payload owners retire before their funding.
    _workspace_funding: (
        Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
        Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
    ),
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}
impl<T: Tensor> MediaPrefillPlan<T> {
    /// Applies unchanged Gemma admission, including independent optional roots.
    pub fn prepare(
        args: &FamilyConfig,
        prepared: PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        geometry.validate().map_err(Error::backend)?;
        let admitted = crate::media_plan::admit_gemma4_input(args, &prepared, inspector)
            .map_err(Error::backend)?;
        Self::prepare_admitted(args, prepared, admitted, geometry)
    }
    pub fn prepare_admitted(
        args: &FamilyConfig,
        prepared: PreparedModelInput<T>,
        admitted: crate::media_plan::AdmittedCompositeInput<Gemma4InputPartPlan>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        geometry.validate().map_err(Error::backend)?;
        PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::backend)?;
        if admitted.decoder_shape() != [geometry.batch_size, geometry.input_positions]
            || geometry.batch_size != 1
        {
            return Err(Error::backend(
                "Gemma retained source geometry differs from admission",
            ));
        }
        Ok(Self {
            prepared: prepared.into(),
            admitted: Some(admitted),
            original: None,
            workspace: None,
            geometry,
            fingerprint: args.architecture_fingerprint(),
            identity: None,
            _workspace_funding: (None, None),
            metadata: None,
        })
    }
    fn input(&self) -> Result<PreparedCompositeInput<'_, T, Gemma4InputPartPlan>, String> {
        self.input_with_diagnostic(str::to_owned)
    }
    fn input_with_diagnostic<E>(&self, diagnostic: impl FnOnce(&'static str) -> E)
        -> Result<PreparedCompositeInput<'_, T, Gemma4InputPartPlan>, E> {
        let input = match (&self.admitted, &self.original) {
            (Some(admitted), None) => PreparedCompositeInput::new_with_diagnostic(&self.prepared, admitted, diagnostic),
            (None, Some(original)) => PreparedCompositeInput::from_original_with_diagnostic(&self.prepared, original, diagnostic),
            _ => unreachable!("closed Gemma semantic plan"),
        }?;
        Ok(input.with_metadata_loan(self.metadata.as_ref()))
    }
    fn from_original(
        prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
        original: crate::media_plan::BoundPreparedMediaSemantics,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, eredu_runtime::working_memory::OriginalCompositeSemanticStorageError> {
        if !original.is_gemma() || geometry.batch_size != 1 || geometry.cached_positions != 0
            || geometry.input_positions != original.decoder_positions() as u64
            || prepared.len() != original.records().len() || geometry.validate_fixed().is_err() {
            return Err(original.reject(crate::media_plan::MediaSemanticError::input(
                "compiled Gemma media source geometry mismatch")));
        }
        Ok(Self {
            prepared, admitted: None, original: Some(original), workspace: None,
            geometry, fingerprint: String::new(), identity: None,
            _workspace_funding: (None, None), metadata: None,
        })
    }
    /// Associates the existing original whole-input identity.
    pub fn with_shared_cache_identity(
        mut self,
        identity: SharedPreparedInputCacheIdentity,
    ) -> Result<Self, Error> {
        if identity.prepared() != self.prepared.identity() {
            return Err(Error::backend(
                "Gemma cache identity differs from prepared input",
            ));
        }
        self.identity = Some(identity);
        Ok(self)
    }
}
/// Complete projected optional roots, independent of each decoder span.
pub struct MediaIngress<T> {
    pending: PreparedCompositeIngress<T>,
    vision: Option<T>,
    audio: Option<T>,
}
pub(super) fn visit_pending<'a, T>(
    pending: &'a PreparedCompositeIngress<T>,
    visitor: &mut dyn FnMut(&'a T),
) {
    for value in &pending.tokens {
        visitor(value);
    }
    for value in pending.projected.iter().flatten() {
        visitor(value);
    }
    if let Some(v) = &pending.vision {
        for value in [&v.patches, &v.positions, &v.valid, &v.key_mask] {
            visitor(value);
        }
    }
    if let Some(a) = &pending.audio {
        for value in [&a.features, &a.input_mask, &a.first_stage_mask] {
            visitor(value);
        }
    }
}
impl<B, S> PrefillIngressArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type IngressPlan = MediaPrefillPlan<B::Tensor>;
    type Ingress = MediaIngress<B::Tensor>;
    fn ingress_geometry(plan: &Self::IngressPlan) -> eredu_core::InferenceGeometry {
        plan.geometry
    }
    fn ingress_session_binding(plan: &Self::IngressPlan)
        -> Option<&eredu_runtime::working_memory::MediaSessionBinding> {
        plan.original.as_ref().map(|original| original.binding())
    }
    fn ingress_cache_identity(
        plan: &Self::IngressPlan,
    ) -> Option<SharedPreparedInputCacheIdentity> {
        plan.identity.clone()
    }
    fn validate_ingress_plan(&self, plan: &Self::IngressPlan) -> Result<(), Error> {
        if plan.original.is_none() && self.args.architecture_fingerprint() != plan.fingerprint {
            return Err(Error::backend(
                "Gemma media source belongs to another architecture",
            ));
        }
        plan.input().map_err(Error::backend)?;
        Ok(())
    }
    fn validate_ingress_plan_with_metadata(&self, plan: &Self::IngressPlan,
        context: &eredu_nn::workspace::WorkspaceContext) -> Result<(), Error> {
        if !context.uses_checked_metadata() {
            return <Self as PrefillIngressArchitecture<B,S>>::validate_ingress_plan(self, plan);
        }
        context.charge_metadata(std::mem::size_of::<(Self::IngressPlan,
            PreparedCompositeInput<'_, B::Tensor, Gemma4InputPartPlan>)>())?;
        if !plan.original.as_ref().is_some_and(|original| original.is_gemma()) {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        plan.input_with_diagnostic(|message| context.metadata_error(format_args!("{message}")))?;
        Ok(())
    }
    fn ingress_error_with_metadata(error: MediaIngressError,
        context: &eredu_nn::workspace::WorkspaceContext) -> Error { context.metadata_source(error) }
    fn ingress_error(error: MediaIngressError) -> Error {
        Error::backend_source(error)
    }
    fn begin_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_media_source(plan, None, false, state, context)
    }
    fn begin_ingress_received(
        &mut self,
        plan: &Self::IngressPlan,
        received: &B::Tensor,
        encoder_continuation: bool,
        state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_media_source(plan, Some(received), encoder_continuation, state, context)
    }
    fn retain_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Ingress, Error> {
        let metadata = forward::Metadata::new(plan.metadata.as_ref().or_else(|| B::construction_metadata(context)));
        forward::with_metadata(metadata, || {
        metadata.controls::<(Self::Ingress, PreparedCompositeIngress<B::Tensor>,
            [(bool, Option<&B::Tensor>, &str);2], i32)>()?;
        let pending = forward
            .pending_media
            .as_ref()
            .ok_or_else(|| metadata.error(format_args!("Gemma cut lacks original source")))?;
        for (audio, value, label) in [
            (false, forward.vision_output.as_ref(), "vision"),
            (true, forward.audio_output.as_ref(), "audio"),
        ] {
            let count = pending
                .tokens
                .iter()
                .zip(&pending.modalities)
                .zip(&pending.projected)
                .filter(|((_, m), p)| {
                    p.is_none()
                        && if audio {
                            **m == eredu_core::InputModality::Audio
                        } else {
                            matches!(
                                **m,
                                eredu_core::InputModality::Image | eredu_core::InputModality::Video
                            )
                        }
                })
                .try_fold(0i32, |n, ((t, _), _)| n.checked_add(t.dim(1)))
                .ok_or_else(|| metadata.error(format_args!("Gemma cut extent overflow")))?;
            forward::validate_component_with_metadata(label, value, count, self.args.text.hidden_size, metadata)?;
        }
        // Raw padded inputs, masks and placement metadata remain under the
        // first-span context through completion, not in future decoder spans.
        let original = forward.pending_media.as_mut().expect("validated source");
        let pending = PreparedCompositeIngress {
            tokens: std::mem::take(&mut original.tokens),
            modalities: std::mem::take(&mut original.modalities),
            projected: std::mem::take(&mut original.projected),
            vision: None,
            audio: None,
            metadata: original.metadata.clone(),
        };
        Ok(MediaIngress {
            pending,
            vision: forward.vision_output.take(),
            audio: forward.audio_output.take(),
        })
        })
    }

    fn begin_ingress_span(
        &mut self,
        plan: &Self::IngressPlan,
        ingress: &Self::Ingress,
        span: &eredu_runtime::prefill::PrefillChunk,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        let metadata = forward::Metadata::new(plan.metadata.as_ref().or_else(|| B::construction_metadata(context)));
        forward::with_metadata(metadata, || {
        metadata.controls::<(LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            Vec<PreparedPart<B::Tensor>>, Vec<B::Tensor>, Vec<B::Tensor>,
            PreparedPart<B::Tensor>, [Index;3], Option<B::Tensor>,
            i32,i32,i32,i32,i32,i32,i32,i32,i32)>()?;
        let start = i32::try_from(span.input.start).map_err(|cause| metadata.source(cause))?;
        let end = i32::try_from(span.input.end).map_err(|cause| metadata.source(cause))?;
        if end <= start || end as u64 > plan.geometry.input_positions {
            return Err(metadata.error(format_args!("Gemma span exceeds original input")));
        }
        let mut parts = metadata.vector(ingress.pending.tokens.len())?;
        let mut vision = metadata.vector(ingress.pending.tokens.len())?;
        let mut audio = metadata.vector(ingress.pending.tokens.len())?;
        let mut position = 0i32;
        let mut vision_position = 0i32;
        let mut audio_position = 0i32;
        for ((original, modality), projected) in ingress
            .pending
            .tokens
            .iter()
            .zip(&ingress.pending.modalities)
            .zip(&ingress.pending.projected)
        {
            let count = original.dim(1);
            let begin = (start - position).max(0).min(count);
            let stop = (end - position).max(0).min(count);
            let raw_vision = projected.is_none()
                && matches!(
                    modality,
                    eredu_core::InputModality::Image | eredu_core::InputModality::Video
                );
            let raw_audio = projected.is_none() && *modality == eredu_core::InputModality::Audio;
            if begin < stop {
                let tokens = original.index(&[Index::Full, Index::Range(begin, stop)], context)?;
                if let Some(projected) = projected {
                    parts.push(PreparedPart::Text {
                        tokens,
                        embeddings: projected.index(
                            &[Index::Full, Index::Range(begin, stop), Index::Full],
                            context,
                        )?,
                    });
                } else if raw_vision {
                    parts.push(PreparedPart::Vision { tokens });
                    vision.push(slice_component(
                        ingress.vision.as_ref().expect("validated vision"),
                        vision_position + begin,
                        stop - begin,
                        context,
                    )?);
                } else if raw_audio {
                    parts.push(PreparedPart::Audio { tokens });
                    audio.push(slice_component(
                        ingress.audio.as_ref().expect("validated audio"),
                        audio_position + begin,
                        stop - begin,
                        context,
                    )?);
                } else {
                    let embeddings = if let Some(parallel) = parallel {
                        B::vocabulary_parallel_lookup(
                            &mut self.static_modules.text.embeddings,
                            &tokens,
                            EmbeddingLookupPolicy::Strict,
                            parallel,
                            context,
                        )?
                    } else {
                        self.static_modules
                            .text
                            .embeddings
                            .forward(&tokens, context)?
                    };
                    let embeddings = embeddings
                        .multiply_scalar((self.args.text.hidden_size as f32).sqrt(), context)?;
                    parts.push(PreparedPart::Text { tokens, embeddings });
                }
            }
            position = position
                .checked_add(count)
                .ok_or_else(|| metadata.error(format_args!("Gemma span extent overflow")))?;
            if raw_vision {
                vision_position = vision_position
                    .checked_add(count)
                    .ok_or_else(|| metadata.error(format_args!("Gemma vision extent overflow")))?;
            }
            if raw_audio {
                audio_position = audio_position
                    .checked_add(count)
                    .ok_or_else(|| metadata.error(format_args!("Gemma audio extent overflow")))?;
            }
        }
        let join = |mut values: Vec<B::Tensor>| -> Result<Option<B::Tensor>, Error> {
            match values.len() {
                0 => Ok(None),
                1 => Ok(values.pop()),
                _ => B::Tensor::concatenate(&values, 1, context).map(Some),
            }
        };
        metadata.borrowed_controls(&join)?;
        let vision = join(vision)?;
        let audio = join(audio)?;
        let assembled = self.assemble_with_metadata(&parts, vision.as_ref(), audio.as_ref(), context, metadata)?;
        // Both the identity lookup and projected per-layer input are restricted
        // to this span, retaining ordinary semantic placeholder IDs.
        let per_layer_inputs =
            self.per_layer_inputs(&assembled.token_ids, &assembled.embeddings, context)?;
        let position_offset = self.partition_position_offset_with_metadata(state, metadata)?;
        Ok(LayeredForwardState {
            hidden: assembled.embeddings,
            context: ForwardContext {
                mask: None,
                position_offset,
                parts,
                per_layer_token_override: None,
                per_layer_inputs,
                shared: HashMap::new().into(),
                shared_pending: false,
                vision_state: None,
                vision_initial: None,
                vision_output: vision,
                audio_valid: None,
                audio_initial: None,
                audio_output: audio,
                pending_media: None,
                media_span: true,
                metadata: metadata.context().cloned(),
            },
        })
        })
    }
    fn visit_ingress_roots(ingress: &Self::Ingress, visitor: &mut dyn FnMut(&B::Tensor)) {
        visit_pending(&ingress.pending, visitor);
        for value in ingress.vision.iter().chain(ingress.audio.iter()) {
            visitor(value);
        }
    }
}
impl<B> LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    fn begin_media_source<S>(
        &self,
        plan: &MediaPrefillPlan<B::Tensor>,
        received: Option<&B::Tensor>,
        encoder_continuation: bool,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let metadata = forward::Metadata::new(plan.metadata.as_ref().or_else(|| B::construction_metadata(context)));
        forward::with_metadata(metadata, || {
        metadata.controls::<(LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            PreparedCompositeIngress<B::Tensor>, Option<VisionState<B::Tensor>>,
            Option<Vec<i32>>, Option<&B::Tensor>, i32, bool)>()?;
        self.validate_partition_state_with_metadata(state, metadata)?;
        let position_offset = self.partition_position_offset_with_metadata(state, metadata)?;
        if !state.layout().is_empty()
            && u64::try_from(position_offset).ok() != Some(plan.geometry.cached_positions)
        {
            return Err(metadata.error(format_args!("Gemma media source cached position mismatch")));
        }
        let pending = prepare_composite_ingress::<B>(
            plan.input_with_diagnostic(|message| metadata.error(format_args!("{message}")))?,
            context,
        )?;
        // No patch/audio projection here. A continuing owner reconstructs only
        // masks/rotary/valid extents; the actual group receives its hidden value.
        let vision_state = if encoder_continuation {
            pending
                .vision_input()
                .map(|input| {
                    self.static_modules
                        .vision
                        .as_ref()
                        .ok_or_else(|| metadata.error(format_args!("Gemma continuation has no vision metadata")))?
                        .prepare_state_with_metadata(input, context, metadata)
                })
                .transpose()?
        } else {
            None
        };
        let audio_valid = pending.audio.as_ref().map(|audio| {
            let mut valid = metadata.vector(audio.valid.len())?;
            valid.extend_from_slice(&audio.valid);
            Ok::<_, Error>(valid)
        }).transpose()?;
        let hidden = received
            .or(pending.vision.as_ref().map(|v| &v.patches))
            .or(pending.audio.as_ref().map(|a| &a.features))
            .or(pending.tokens.first())
            .ok_or_else(|| metadata.error(format_args!("Gemma source has no input")))?
            .clone();
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: None,
                position_offset,
                parts: Vec::new(),
                per_layer_token_override: None,
                per_layer_inputs: None,
                shared: HashMap::new().into(),
                shared_pending: false,
                vision_state,
                vision_initial: None,
                vision_output: None,
                audio_valid,
                audio_initial: None,
                audio_output: None,
                pending_media: Some(pending),
                media_span: false,
                metadata: metadata.context().cloned(),
            },
        })
        })
    }
}
impl<B, S> crate::composite_execution::CompositeMediaIngressArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn prepare_ingress_plan(
        admission: &Self::AdmissionConfig,
        input: PreparedModelInput<B::Tensor>,
        inspector: &impl PreparedInputInspector<B::Tensor>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self::IngressPlan, Error> {
        MediaPrefillPlan::prepare(admission, input, inspector, geometry)
    }
    fn prepare_ingress_plan_admitted(
        admission: &Self::AdmissionConfig,
        input: PreparedModelInput<B::Tensor>,
        admitted: crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        _inspector: &impl PreparedInputInspector<B::Tensor>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self::IngressPlan, Error> {
        MediaPrefillPlan::prepare_admitted(admission, input, admitted, geometry)
    }
    fn bind_original_media_semantics(
        admission: &Self::AdmissionConfig,
        original: crate::media_plan::OriginalPreparedMediaSemantics<'_>,
        blueprint: &crate::prepared_execution::PreparedInferenceBlueprint,
        source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
        binding: eredu_runtime::working_memory::MediaSessionBinding,
    ) -> Result<
        crate::media_plan::BoundPreparedMediaSemantics,
        eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
    > {
        original.bind_gemma(admission, blueprint, source, binding)
    }
    fn prepare_bound_original_ingress_plan(
        input: crate::processor_execution::OriginalHostLowering<B::Tensor>,
        original: crate::media_plan::BoundPreparedMediaSemantics,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<
        Self::IngressPlan,
        eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
    > {
        if !input.source().same_source(original.source()) {
            return Err(
                original.reject(crate::media_plan::MediaSemanticError::input(
                    "lowered source differs from compiled source",
                )),
            );
        }
        MediaPrefillPlan::from_original(input.into_prepared(), original, geometry)
    }
    fn prepare_bound_original_ingress_plan_with_metadata(
        input: crate::processor_execution::OriginalHostLowering<B::Tensor>,
        original: crate::media_plan::BoundPreparedMediaSemantics,
        geometry: eredu_core::InferenceGeometry,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self::IngressPlan, Error> {
        let mut plan = <Self as crate::composite_execution::CompositeMediaIngressArchitecture<
            B,
            S,
        >>::prepare_bound_original_ingress_plan(input, original, geometry)
        .map_err(|cause| context.metadata_source(cause))?;
        plan.metadata = Some(context.clone());
        Ok(plan)
    }
    fn prepare_original_workspace_ingress_plan(
        input: crate::prepared_execution::OriginalMediaWorkspaceInput,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self::IngressPlan, Self::Error>
    where
        B: eredu_nn::NeuralBackend<Tensor = eredu_nn::workspace::WorkspaceTensor>,
    {
        let (mut plan, workspace, funding) = input.construct_source_plan(None, |prepared, original| {
            MediaPrefillPlan::from_original(prepared, original, geometry)
        })?;
        plan.workspace = Some(workspace);
        plan._workspace_funding = funding;
        Ok(plan)
    }
    fn prepare_original_workspace_ingress_plan_with_metadata(
        input: crate::prepared_execution::OriginalMediaWorkspaceInput,
        geometry: eredu_core::InferenceGeometry,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self::IngressPlan, Self::Error>
    where
        B: eredu_nn::NeuralBackend<Tensor = eredu_nn::workspace::WorkspaceTensor>,
    {
        let (mut plan, workspace, funding) = input
            .construct_source_plan(Some(context), |prepared, original| {
                MediaPrefillPlan::from_original(prepared, original, geometry)
            })?;
        plan.workspace = Some(workspace);
        plan._workspace_funding = funding;
        plan.metadata = Some(context.clone());
        Ok(plan)
    }
    fn prepared_ingress_input(
        plan: &Self::IngressPlan,
    ) -> PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan> {
        plan.input().expect("retained Gemma admission")
    }
    fn media_group_collective_waves(
        &self,
        _plan: &Self::IngressPlan,
        group: usize,
        tensor_partitions: usize,
        pipeline_stages: usize,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        // Unlike ordinary eager context construction, retained ingress performs
        // no text lookup while either replicated encoder runs.
        Ok((group < 2 && tensor_partitions > 1 && pipeline_stages > 1)
            .then(|| vec![Vec::new(); pipeline_stages]))
    }
    fn media_primary_ingress_collectives(
        &self,
        plan: &Self::IngressPlan,
        span: &eredu_runtime::prefill::PrefillChunk,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, String> {
        let mut offset = 0u64;
        let mut positions = Vec::new();
        for part in plan.input()?.admitted().gemma_parts() {
            let (length, text) = match part {
                Gemma4InputPartPlan::TextTokens { positions } => (positions, true),
                Gemma4InputPartPlan::Projected { positions, .. } => (positions, false),
                Gemma4InputPartPlan::Vision { ingress, .. } => {
                    (ingress.decoder_positions as u64, false)
                }
                Gemma4InputPartPlan::Audio { ingress, .. } => {
                    (ingress.decoder_positions as u64, false)
                }
            };
            let end = offset
                .checked_add(length)
                .ok_or("Gemma collective extent overflow")?;
            let start = offset.max(span.input.start);
            let stop = end.min(span.input.end);
            if text && start < stop {
                positions.push(stop - start);
            }
            offset = end;
        }
        crate::composite_execution::segmented_token_ingress_collectives(
            positions,
            self.args.text.hidden_size,
            tensor_partitions,
        )
    }
}
