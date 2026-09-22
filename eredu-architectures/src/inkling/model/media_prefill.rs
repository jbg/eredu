//! Retained hMLP/dMel outputs with Inkling's original ordered token semantics.
use super::*;
use eredu_runtime::media_prefill::{MediaIngressError, PrefillIngressArchitecture};
use eredu_runtime::{PreparedInputInspector, PreparedModelInput, SharedPreparedInputCacheIdentity};

/// Exact admitted source, including independently optional image and audio parts.
pub struct MediaPrefillPlan<T> {
    prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
    admitted: Option<crate::media_plan::AdmittedCompositeInput<InklingInputPartPlan>>,
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
        Option<eredu_nn::workspace::HostMetadataFunding>,
        Option<eredu_nn::workspace::HostMetadataFunding>,
    ),
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}
impl<T: Tensor> MediaPrefillPlan<T> {
    /// Applies unchanged Inkling admission, including independent optional roots.
    pub fn prepare(
        args: &ModelArgs,
        prepared: PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        geometry.validate().map_err(Error::backend)?;
        let admitted = crate::media_plan::admit_inkling_input(args, &prepared, inspector)
            .map_err(Error::backend)?;
        Self::prepare_admitted(args, prepared, admitted, geometry)
    }
    pub fn prepare_admitted(
        args: &ModelArgs,
        prepared: PreparedModelInput<T>,
        admitted: crate::media_plan::AdmittedCompositeInput<InklingInputPartPlan>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        geometry.validate().map_err(Error::backend)?;
        PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::backend)?;
        if admitted.decoder_shape() != [geometry.batch_size, geometry.input_positions]
            || geometry.batch_size != 1
        {
            return Err(Error::backend(
                "Inkling retained source geometry differs from admission",
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
    fn input(&self) -> Result<PreparedCompositeInput<'_, T, InklingInputPartPlan>, String> {
        self.input_with_diagnostic(str::to_owned)
    }
    fn input_with_diagnostic<E>(
        &self,
        diagnostic: impl FnOnce(&'static str) -> E,
    ) -> Result<PreparedCompositeInput<'_, T, InklingInputPartPlan>, E> {
        let input = match (&self.admitted, &self.original) {
            (Some(admitted), None) => {
                PreparedCompositeInput::new_with_diagnostic(&self.prepared, admitted, diagnostic)
            }
            (None, Some(original)) => PreparedCompositeInput::from_original_with_diagnostic(
                &self.prepared,
                original,
                diagnostic,
            ),
            _ => unreachable!("closed Inkling semantic plan"),
        }?;
        Ok(input.with_metadata_loan(self.metadata.as_ref()))
    }
    fn from_original(
        prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
        original: crate::media_plan::BoundPreparedMediaSemantics,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, eredu_runtime::working_memory::OriginalCompositeSemanticStorageError> {
        if !original.is_inkling()
            || geometry.batch_size != 1
            || geometry.cached_positions != 0
            || geometry.input_positions != original.decoder_positions() as u64
            || prepared.len() != original.records().len()
            || geometry.validate_fixed().is_err()
        {
            return Err(
                original.reject(crate::media_plan::MediaSemanticError::input(
                    "compiled Inkling media source geometry mismatch",
                )),
            );
        }
        Ok(Self {
            prepared,
            admitted: None,
            original: Some(original),
            workspace: None,
            geometry,
            fingerprint: String::new(),
            identity: None,
            _workspace_funding: (None, None),
            metadata: None,
        })
    }
    /// Associates the existing original whole-input identity.
    pub fn with_shared_cache_identity(
        mut self,
        identity: SharedPreparedInputCacheIdentity,
    ) -> Result<Self, Error> {
        if identity.prepared() != self.prepared.identity() {
            return Err(Error::backend(
                "Inkling cache identity differs from prepared input",
            ));
        }
        self.identity = Some(identity);
        Ok(self)
    }
}
/// Independently optional completed encoders and original ordered source parts.
pub struct MediaIngress<T> {
    pending: PreparedInput<T>,
    vision: Option<T>,
    audio: Option<T>,
}

pub(super) fn visit_pending<'a, T>(pending: &'a PreparedInput<T>, visitor: &mut dyn FnMut(&'a T)) {
    for tokens in &pending.tokens {
        visitor(tokens);
    }
    for value in pending.projected.iter().flatten() {
        visitor(value);
    }
    for value in pending.images.iter().chain(pending.audio.iter()) {
        visitor(value);
    }
}

impl<B, S> PrefillIngressArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
{
    type IngressPlan = MediaPrefillPlan<B::Tensor>;
    type Ingress = MediaIngress<B::Tensor>;
    fn ingress_geometry(plan: &Self::IngressPlan) -> eredu_core::InferenceGeometry {
        plan.geometry
    }
    fn ingress_session_binding(
        plan: &Self::IngressPlan,
    ) -> Option<&eredu_runtime::working_memory::MediaSessionBinding> {
        plan.original.as_ref().map(|original| original.binding())
    }
    fn ingress_cache_identity(
        plan: &Self::IngressPlan,
    ) -> Option<SharedPreparedInputCacheIdentity> {
        plan.identity.clone()
    }
    fn ingress_execution_graph(
        &self,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(
            &Self,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            eredu_runtime::ArchitectureExecutionGraph<'_>,
        )>()?;
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(
            &self.execution_graph,
        ))
    }
    fn validate_ingress_plan(
        &self,
        plan: &Self::IngressPlan,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<(), Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(
            &Self,
            &Self::IngressPlan,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            PreparedCompositeInput<'_, B::Tensor, InklingInputPartPlan>,
            String,
        )>()?;
        let identity_metadata = crate::decoder::identity::Metadata::new(metadata_context);
        if let Some(original) = &plan.original {
            if !original.is_inkling() {
                return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
            }
        } else {
            let fingerprint = self
                .args
                .architecture_fingerprint_with_metadata(identity_metadata)?;
            if fingerprint != plan.fingerprint {
                return Err(metadata.error(format_args!(
                    "Inkling media source belongs to another architecture"
                )));
            }
        }
        plan.input_with_diagnostic(|message| metadata.error(format_args!("{message}")))?;
        Ok(())
    }
    fn ingress_error(
        error: MediaIngressError,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Error {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        if let Err(refusal) = metadata.controls::<(
            MediaIngressError,
            Option<&eredu_nn::workspace::WorkspaceContext>,
        )>() {
            return refusal;
        }
        metadata.source(error)
    }
    fn begin_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_media_source(plan, None, state, context)
    }
    fn begin_ingress_received(
        &mut self,
        plan: &Self::IngressPlan,
        received: &B::Tensor,
        _encoder_continuation: bool,
        state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        // hMLP continuation needs no extra positional state. Audio's zero-unit
        // group remains the existing completion equation on its actual owner.
        self.begin_media_source(plan, Some(received), state, context)
    }
    fn retain_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Ingress, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(plan.metadata.as_ref());
        metadata.controls::<(Self::Ingress, i32, usize)>()?;
        let pending = forward
            .pending_media
            .as_ref()
            .ok_or_else(|| Error::backend("Inkling cut lacks original source"))?;
        for (modality, value, label) in [
            (
                eredu_core::InputModality::Image,
                forward.vision_output.as_ref(),
                "image",
            ),
            (
                eredu_core::InputModality::Audio,
                forward.audio_output.as_ref(),
                "audio",
            ),
        ] {
            let count = pending
                .tokens
                .iter()
                .zip(&pending.modalities)
                .zip(&pending.projected)
                .filter(|((_, m), p)| **m == modality && p.is_none())
                .try_fold(0i32, |n, ((t, _), _)| n.checked_add(t.dim(1)))
                .ok_or_else(|| Error::backend("Inkling cut extent overflow"))?;
            validate_component_with_metadata(
                label,
                value,
                count,
                self.args.text_config.hidden_size,
                metadata,
            )?;
        }
        // Keep padded/concatenated raw inputs in the old first-span context
        // until its completion. Later spans retain only semantic parts and the
        // completed media outputs, without cloning these Vec populations.
        let original = forward.pending_media.as_mut().expect("validated source");
        let pending = PreparedInput {
            tokens: std::mem::take(&mut original.tokens),
            modalities: std::mem::take(&mut original.modalities),
            projected: std::mem::take(&mut original.projected),
            images: None,
            audio: None,
            audio_frames: original.audio_frames,
        };
        Ok(MediaIngress {
            pending,
            vision: forward.vision_output.take(),
            audio: forward.audio_output.take(),
        })
    }

    fn begin_ingress_span(
        &mut self,
        plan: &Self::IngressPlan,
        ingress: &Self::Ingress,
        span: &eredu_runtime::prefill::PrefillChunk,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        let metadata = crate::decoder::ModuleMetadata::destination(
            plan.metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(
            LayeredForwardState<B::Tensor, Self::ForwardContext>,
            Vec<B::Tensor>,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            bool,
            bool,
        )>()?;
        let start = i32::try_from(span.input.start).map_err(Error::backend)?;
        let end = i32::try_from(span.input.end).map_err(Error::backend)?;
        if end <= start || end as u64 > plan.geometry.input_positions {
            return Err(Error::backend("Inkling span exceeds original input"));
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
            let raw_image = projected.is_none() && *modality == eredu_core::InputModality::Image;
            let raw_audio = projected.is_none() && *modality == eredu_core::InputModality::Audio;
            if begin < stop {
                let tokens = original.index(&[Index::Full, Index::Range(begin, stop)], context)?;
                if let Some(projected) = projected {
                    parts.push(PreparedPart::Projected {
                        tokens,
                        embeddings: projected.index(
                            &[Index::Full, Index::Range(begin, stop), Index::Full],
                            context,
                        )?,
                    });
                } else if raw_image {
                    parts.push(PreparedPart::Image { tokens });
                    vision.push(slice_component(
                        ingress.vision.as_ref().expect("validated image"),
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
                            &mut self.static_modules.embeddings,
                            &tokens,
                            EmbeddingLookupPolicy::Strict,
                            parallel,
                            context,
                        )?
                    } else {
                        self.static_modules.embeddings.forward(&tokens, context)?
                    };
                    parts.push(PreparedPart::Text { tokens, embeddings });
                }
            }
            position = position
                .checked_add(count)
                .ok_or_else(|| Error::backend("Inkling span extent overflow"))?;
            if raw_image {
                vision_position = vision_position
                    .checked_add(count)
                    .ok_or_else(|| Error::backend("Inkling image extent overflow"))?;
            }
            if raw_audio {
                audio_position = audio_position
                    .checked_add(count)
                    .ok_or_else(|| Error::backend("Inkling audio extent overflow"))?;
            }
        }
        let join = |mut values: Vec<B::Tensor>| -> Result<Option<B::Tensor>, Error> {
            match values.len() {
                0 => Ok(None),
                1 => Ok(values.pop()),
                _ => B::Tensor::concatenate(&values, 1, context).map(Some),
            }
        };
        let vision = join(vision)?;
        let audio = join(audio)?;
        let hidden = self.assemble_with_metadata(
            &parts,
            vision.as_ref(),
            audio.as_ref(),
            context,
            metadata,
        )?;
        let tokens = ordered_tokens_with_metadata(&parts, context, metadata)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                parts,
                tokens,
                audio_input: None,
                audio_valid_frames: None,
                audio_output: audio,
                vision_output: vision,
                has_vision: false,
                target_hidden: None,
                pending_media: None,
                media_span: true,
            },
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
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
    {
        let metadata = crate::decoder::ModuleMetadata::destination(
            plan.metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            PreparedInput<B::Tensor>,
            B::Tensor,
            Option<B::Tensor>,
            i32,
            bool,
        )>()?;
        if !self.accepts_execution_state(state.layout())? {
            return Err(Error::backend("Inkling source state layout mismatch"));
        }
        if !state.layout().is_empty()
            && u64::try_from(eredu_nn::AttentionCache::<B::Tensor>::offset(
                state.layer(0).map_err(Error::backend)?,
            ))
            .ok()
                != Some(plan.geometry.cached_positions)
        {
            return Err(Error::backend("Inkling source cached position mismatch"));
        }
        let pending = prepare_input(
            plan.input_with_diagnostic(|cause| metadata.error(format_args!("{cause}")))?,
            context,
        )?;
        let tokens = B::Tensor::concatenate(&pending.tokens, 1, context)?;
        let hidden = received
            .or(pending.images.as_ref())
            .or(pending.audio.as_ref())
            .unwrap_or(&tokens)
            .clone();
        let audio_input = pending.audio.clone();
        let audio_valid_frames = audio_input.as_ref().map(|_| pending.audio_frames);
        let has_vision = pending.images.is_some();
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                parts: Vec::new(),
                tokens,
                audio_input,
                audio_valid_frames,
                audio_output: None,
                vision_output: None,
                has_vision,
                target_hidden: None,
                pending_media: Some(pending),
                media_span: false,
            },
        })
    }
}
impl<B, S> crate::composite_execution::CompositeMediaIngressArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
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
        original.bind_inkling(admission, blueprint, source, binding)
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
        let (mut plan, workspace, funding) = input
            .construct_source_plan(None, |prepared, original| {
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
        plan.input().expect("retained Inkling admission")
    }
    fn media_group_collective_waves(
        &self,
        plan: &Self::IngressPlan,
        group: usize,
        tensor_partitions: usize,
        pipeline_stages: usize,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        <Self as CompositeArchitecture<B,S>>::prepared_group_collective_waves(self,group,<Self as crate::composite_execution::CompositeMediaIngressArchitecture<B,S>>::prepared_ingress_input(plan),tensor_partitions,pipeline_stages,None).map_err(|error|error.to_string())
    }
    fn media_group_collective_waves_with_metadata(
        &self,
        plan: &Self::IngressPlan,
        group: usize,
        tensor_partitions: usize,
        pipeline_stages: usize,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, Error>
    {
        <Self as CompositeArchitecture<B, S>>::prepared_group_collective_waves(
            self,
            group,
            plan.input_with_diagnostic(|cause| context.metadata_error(format_args!("{cause}")))?,
            tensor_partitions,
            pipeline_stages,
            Some(context),
        )
    }
    fn media_primary_ingress_collectives(
        &self,
        plan: &Self::IngressPlan,
        span: &eredu_runtime::prefill::PrefillChunk,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, String> {
        media_ingress_waves(
            plan,
            span,
            self.args.text_config.hidden_size,
            tensor_partitions,
            crate::composite_execution::graph::Destination(None),
        )
        .map_err(|cause| cause.to_string())
    }
    fn media_primary_ingress_collectives_with_metadata(
        &self,
        plan: &Self::IngressPlan,
        span: &eredu_runtime::prefill::PrefillChunk,
        tensor_partitions: usize,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, Error> {
        media_ingress_waves(
            plan,
            span,
            self.args.text_config.hidden_size,
            tensor_partitions,
            crate::composite_execution::graph::Destination(Some(context)),
        )
    }
}
fn media_ingress_waves<T: Tensor>(
    plan: &MediaPrefillPlan<T>,
    span: &eredu_runtime::prefill::PrefillChunk,
    hidden: i32,
    tensor_partitions: usize,
    destination: crate::composite_execution::graph::Destination<'_>,
) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, Error> {
    destination.controls::<(Vec<u64>, u64, u64, u64, u64, bool, InklingInputPartPlan)>()?;
    let mut offset = 0u64;
    let mut positions = destination.vector(plan.prepared.len())?;
    for part in plan
        .input_with_diagnostic(|cause| destination.error(format_args!("{cause}")))?
        .admitted()
        .inkling_parts()
    {
        let (length, text) = match part {
            InklingInputPartPlan::TextTokens { positions } => (positions, true),
            InklingInputPartPlan::Projected { positions, .. } => (positions, false),
            InklingInputPartPlan::Media { ingress, .. } => (ingress.placeholder_count, false),
        };
        let end = offset
            .checked_add(length)
            .ok_or_else(|| destination.error(format_args!("Inkling collective extent overflow")))?;
        let start = offset.max(span.input.start);
        let stop = end.min(span.input.end);
        if text && start < stop {
            positions.push(stop - start);
        }
        offset = end;
    }
    crate::composite_execution::segmented_token_ingress_collectives_in(
        positions,
        hidden,
        tensor_partitions,
        destination,
    )
}
