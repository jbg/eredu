//! Authenticated Muse raster-media roots and exact decoder spans.
use super::*;
use eredu_runtime::media_prefill::{MediaIngressError, PrefillIngressArchitecture};
use eredu_runtime::{PreparedInputInspector, PreparedModelInput, SharedPreparedInputCacheIdentity};

/// Actual family admission and original ordered semantic input.
pub struct MediaPrefillPlan<T> {
    prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
    admitted: Option<crate::media_plan::AdmittedCompositeInput<MuseGlimmerInputPartPlan>>,
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
    /// Applies unchanged Muse admission, including independent optional roots.
    pub fn prepare(
        args: &DecoderConfig,
        prepared: PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        geometry.validate().map_err(Error::backend)?;
        let admitted = crate::media_plan::admit_muse_glimmer_input(args, &prepared, inspector)
            .map_err(Error::backend)?;
        Self::prepare_admitted(args, prepared, admitted, geometry)
    }
    pub fn prepare_admitted(
        args: &DecoderConfig,
        prepared: PreparedModelInput<T>,
        admitted: crate::media_plan::AdmittedCompositeInput<MuseGlimmerInputPartPlan>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        geometry.validate().map_err(Error::backend)?;
        PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::backend)?;
        if admitted.decoder_shape() != [geometry.batch_size, geometry.input_positions]
            || geometry.batch_size != 1
        {
            return Err(Error::backend(
                "Muse retained source geometry differs from admission",
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
    fn input(&self) -> Result<PreparedCompositeInput<'_, T, MuseGlimmerInputPartPlan>, String> {
        self.input_with_diagnostic(str::to_owned)
    }
    fn input_with_diagnostic<E>(
        &self,
        diagnostic: impl FnOnce(&'static str) -> E,
    ) -> Result<PreparedCompositeInput<'_, T, MuseGlimmerInputPartPlan>, E> {
        let input = match (&self.admitted, &self.original) {
            (Some(admitted), None) => {
                PreparedCompositeInput::new_with_diagnostic(&self.prepared, admitted, diagnostic)
            }
            (None, Some(original)) => PreparedCompositeInput::from_original_with_diagnostic(
                &self.prepared,
                original,
                diagnostic,
            ),
            _ => unreachable!("closed Muse semantic plan"),
        }?;
        Ok(input.with_metadata_loan(self.metadata.as_ref()))
    }
    fn from_original(
        prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
        original: crate::media_plan::BoundPreparedMediaSemantics,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, eredu_runtime::working_memory::OriginalCompositeSemanticStorageError> {
        if !original.is_muse()
            || geometry.batch_size != 1
            || geometry.cached_positions != 0
            || geometry.input_positions != original.decoder_positions() as u64
            || prepared.len() != original.records().len()
            || geometry.validate_fixed().is_err()
        {
            return Err(
                original.reject(crate::media_plan::MediaSemanticError::input(
                    "compiled Muse media source geometry mismatch",
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
                "Muse cache identity differs from prepared input",
            ));
        }
        self.identity = Some(identity);
        Ok(self)
    }
}
/// Complete normalized projected image/video rows plus original semantic tokens.
pub struct MediaIngress<T> {
    pending: PreparedCompositeIngress<T>,
    projected: T,
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
            PreparedCompositeInput<'_, B::Tensor, MuseGlimmerInputPartPlan>,
            String,
        )>()?;
        if let Some(original) = &plan.original {
            if !original.is_muse() {
                return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
            }
        } else {
            let fingerprint = match metadata_context {
                Some(context) => self.args.architecture_fingerprint_with_metadata(context)?,
                None => self.args.architecture_fingerprint(),
            };
            if fingerprint != plan.fingerprint {
                return Err(metadata.error(format_args!(
                    "Muse media source belongs to another architecture"
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
        let metadata = crate::decoder::ModuleMetadata::destination(plan.metadata.as_ref());
        metadata.controls::<(&MediaPrefillPlan<B::Tensor>,)>()?;
        self.validate_media_state(plan, state)?;
        let mut pending = prepare_composite_ingress::<B>(
            plan.input_with_diagnostic(|cause| metadata.error(format_args!("{cause}")))?,
            context,
        )?;
        let pixels = pending
            .pixels
            .take()
            .ok_or_else(|| metadata.error(format_args!("Muse media source has no patches")))?;
        let (hidden, vision) = self
            .static_modules
            .vision
            .as_mut()
            .ok_or_else(|| {
                metadata.error(format_args!(
                    "Muse media source has no selected vision owner"
                ))
            })?
            .begin(
                VisionInput {
                    pixels: &pixels,
                    grid: &pending.grid,
                },
                context,
            )?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: None,
                parts: Vec::new(),
                vision: Some(vision),
                pending_media: Some(pending),
                media_output: None,
                media_span: false,
            },
        })
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
        let metadata = crate::decoder::ModuleMetadata::destination(plan.metadata.as_ref());
        metadata.controls::<(&MediaPrefillPlan<B::Tensor>,)>()?;
        self.validate_media_state(plan, state)?;
        let mut pending = prepare_composite_ingress::<B>(
            plan.input_with_diagnostic(|cause| metadata.error(format_args!("{cause}")))?,
            context,
        )?;
        // The actual boundary is already embedded and permuted. Rebuild only
        // placement metadata for remaining encoder blocks, never patch lookup.
        pending.pixels = None;
        let vision = if encoder_continuation {
            Some(
                self.static_modules
                    .vision
                    .as_ref()
                    .ok_or_else(|| {
                        metadata.error(format_args!(
                            "Muse encoder continuation has no selected vision owner"
                        ))
                    })?
                    .continuation_state(&pending.grid, context)?,
            )
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden: received.clone(),
            context: ForwardContext {
                mask: None,
                parts: Vec::new(),
                vision,
                pending_media: Some(pending),
                media_output: None,
                media_span: false,
            },
        })
    }
    fn retain_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Ingress, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(plan.metadata.as_ref());
        metadata.controls::<(&MediaPrefillPlan<B::Tensor>,)>()?;
        let pending = forward.pending_media.as_ref().ok_or_else(|| {
            metadata.error(format_args!("Muse cut has no original semantic input"))
        })?;
        let projected = forward.media_output.as_ref().ok_or_else(|| {
            metadata.error(format_args!("Muse cut has no completed normalized media"))
        })?;
        let rows = pending
            .tokens
            .iter()
            .zip(&pending.media)
            .filter(|(_, media)| **media)
            .try_fold(0i32, |total, (tokens, _)| total.checked_add(tokens.dim(1)))
            .ok_or_else(|| {
                metadata.error(format_args!("Muse retained media row count overflows"))
            })?;
        if projected.shape() != [rows, self.args.hidden_size] {
            return Err(metadata.error(format_args!(
                "Muse cut has incomplete projected media geometry"
            )));
        }
        Ok(MediaIngress {
            pending: forward
                .pending_media
                .take()
                .expect("validated original input"),
            projected: forward
                .media_output
                .take()
                .expect("validated normalized output"),
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
        let metadata = crate::decoder::ModuleMetadata::destination(plan.metadata.as_ref());
        metadata.controls::<(&MediaPrefillPlan<B::Tensor>,)>()?;
        let start = i32::try_from(span.input.start)
            .map_err(|cause| metadata.error(format_args!("{cause}")))?;
        let end = i32::try_from(span.input.end)
            .map_err(|cause| metadata.error(format_args!("{cause}")))?;
        if start < 0 || end <= start || end as u64 > plan.geometry.input_positions {
            return Err(metadata.error(format_args!("Muse decoder span exceeds original input")));
        }
        metadata.controls::<(
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            i32,
            i32,
        )>()?;
        let mut parts = metadata.vector(ingress.pending.tokens.len())?;
        let mut media = metadata.vector(ingress.pending.tokens.len())?;
        let mut position = 0i32;
        let mut media_position = 0i32;
        for (original, is_media) in ingress.pending.tokens.iter().zip(&ingress.pending.media) {
            let count = original.dim(1);
            let begin = (start - position).max(0).min(count);
            let stop = (end - position).max(0).min(count);
            if begin < stop {
                let tokens = original.index(
                    &[eredu_nn::Index::Full, eredu_nn::Index::Range(begin, stop)],
                    context,
                )?;
                if *is_media {
                    parts.push(PreparedPart::Media { tokens });
                    media.push(ingress.projected.index(
                        &[
                            eredu_nn::Index::Range(media_position + begin, media_position + stop),
                            eredu_nn::Index::Full,
                        ],
                        context,
                    )?);
                } else {
                    let embeddings = if let Some(parallel) = parallel {
                        let value = B::vocabulary_parallel_lookup(
                            &mut self.static_modules.text.embeddings,
                            &tokens,
                            EmbeddingLookupPolicy::Strict,
                            parallel,
                            context,
                        )?;
                        self.static_modules
                            .text
                            .normalize_embeddings(&value, context)?
                    } else {
                        self.static_modules.text.embed(&tokens, context)?
                    };
                    parts.push(PreparedPart::Text { tokens, embeddings });
                }
            }
            position = position
                .checked_add(count)
                .ok_or_else(|| metadata.error(format_args!("Muse decoder position overflows")))?;
            if *is_media {
                media_position = media_position
                    .checked_add(count)
                    .ok_or_else(|| metadata.error(format_args!("Muse media position overflows")))?;
            }
        }
        let media = match media.len() {
            0 => None,
            1 => media.pop(),
            _ => Some(B::Tensor::concatenate(&media, 0, context)?),
        };
        // Reuse Muse's normalized media/token assembly. Each text block still
        // derives its own causal/sliding mask from the actual retained cache.
        let hidden = self.assemble(&parts, media.as_ref(), context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: None,
                parts,
                vision: None,
                pending_media: None,
                media_output: media,
                media_span: true,
            },
        })
    }
    fn visit_ingress_roots(ingress: &Self::Ingress, visitor: &mut dyn FnMut(&B::Tensor)) {
        visitor(&ingress.projected);
        for tokens in &ingress.pending.tokens {
            visitor(tokens);
        }
        if let Some(pixels) = &ingress.pending.pixels {
            visitor(pixels);
        }
    }
}
impl<B> LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    fn validate_media_state<S>(
        &self,
        plan: &MediaPrefillPlan<B::Tensor>,
        state: &mut S,
    ) -> Result<(), Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let metadata = crate::decoder::ModuleMetadata::destination(plan.metadata.as_ref());
        metadata.controls::<(&MediaPrefillPlan<B::Tensor>,)>()?;
        self.validate_partition_state(state)?;
        if !state.layout().is_empty()
            && u64::try_from(
                state
                    .layer(0)
                    .map_err(|cause| metadata.error(format_args!("{cause}")))?
                    .offset(),
            )
            .ok()
                != Some(plan.geometry.cached_positions)
        {
            return Err(metadata.error(format_args!(
                "Muse source cached position differs from actual selected state"
            )));
        }
        Ok(())
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
        original.bind_muse(admission, blueprint, source, binding)
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
        plan.input().expect("retained Muse admission")
    }
    fn media_group_collective_waves(
        &self,
        plan: &Self::IngressPlan,
        group: usize,
        tensor_partitions: usize,
        pipeline_stages: usize,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        <Self as CompositeArchitecture<B,S>>::prepared_group_collective_waves(self, group,
            <Self as crate::composite_execution::CompositeMediaIngressArchitecture<B,S>>::prepared_ingress_input(plan), tensor_partitions, pipeline_stages, None).map_err(|error|error.to_string())
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
            self.args.hidden_size,
            tensor_partitions,
            crate::composite_execution::graph::Destination(None),
        )
        .map_err(|error| error.to_string())
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
            self.args.hidden_size,
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
    destination.controls::<(
        u64,
        Vec<u64>,
        PreparedCompositeInput<'_, T, MuseGlimmerInputPartPlan>,
    )>()?;
    let input =
        plan.input_with_diagnostic(|message| destination.error(format_args!("{message}")))?;
    let mut positions = destination.vector(plan.prepared.len())?;
    let mut offset = 0u64;
    for part in input.admitted().muse_parts() {
        let (length, text) = match part {
            MuseInputPartRef::TextTokens { positions } => (positions, true),
            MuseInputPartRef::Vision {
                placeholder_count, ..
            } => (placeholder_count, false),
        };
        let end = offset
            .checked_add(length)
            .ok_or_else(|| destination.error(format_args!("Muse span position overflow")))?;
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
