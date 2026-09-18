//! Muse's normalized raster-media cut and ordinary one-dimensional decoder spans.
use super::*;
use eredu_runtime::media_prefill::{MediaIngressError, PrefillIngressArchitecture};
use eredu_runtime::{PreparedInputInspector, PreparedModelInput, SharedPreparedInputCacheIdentity};

/// One admitted ordered Muse input, retained without replacing its placeholders.
pub struct MediaPrefillPlan<T> {
    prepared: PreparedModelInput<T>,
    admitted: crate::media_plan::AdmittedCompositeInput<MuseGlimmerInputPartPlan>,
    geometry: eredu_core::InferenceGeometry,
    fingerprint: String,
    identity: Option<SharedPreparedInputCacheIdentity>,
}
impl<T: Tensor> MediaPrefillPlan<T> {
    /// Uses the existing Muse image/video admission and exact decoder extent.
    pub fn prepare(
        args: &DecoderConfig,
        prepared: PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        geometry.validate().map_err(Error::backend)?;
        let admitted = crate::media_plan::admit_muse_glimmer_input(args, &prepared, inspector)
            .map_err(Error::backend)?;
        if admitted.decoder_shape() != [geometry.batch_size, geometry.input_positions]
            || geometry.batch_size != 1
            || !admitted
                .parts()
                .iter()
                .any(|part| matches!(part, MuseGlimmerInputPartPlan::Vision { .. }))
        {
            return Err(Error::backend(
                "Muse retained media geometry differs from actual raw-media admission",
            ));
        }
        Ok(Self {
            prepared,
            admitted,
            geometry,
            fingerprint: args.architecture_fingerprint(),
            identity: None,
        })
    }
    /// Associates the existing exact whole-input identity; no new source proof.
    pub fn with_shared_cache_identity(
        mut self,
        identity: SharedPreparedInputCacheIdentity,
    ) -> Result<Self, Error> {
        if identity.prepared() != self.prepared.identity() {
            return Err(Error::backend(
                "Muse cache identity differs from original prepared input",
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
    fn ingress_cache_identity(
        plan: &Self::IngressPlan,
    ) -> Option<SharedPreparedInputCacheIdentity> {
        plan.identity.clone()
    }
    fn ingress_execution_graph(&self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>)
        -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, Option<&eredu_nn::workspace::WorkspaceContext>, eredu_runtime::ArchitectureExecutionGraph<'_>)>()?;
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(&self.execution_graph))
    }
    fn validate_ingress_plan(&self, plan: &Self::IngressPlan, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<(), Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, &Self::IngressPlan, Option<&eredu_nn::workspace::WorkspaceContext>,
            PreparedCompositeInput<'_, B::Tensor, MuseGlimmerInputPartPlan>, String)>()?;
        let identity_metadata = crate::decoder::identity::Metadata::new(metadata_context);
        let fingerprint = match metadata_context { Some(context) => self.args.architecture_fingerprint_with_metadata(context)?, None => self.args.architecture_fingerprint() };
        if fingerprint != plan.fingerprint {
            return Err(metadata.error(format_args!("Muse media source belongs to another architecture")));
        }
        PreparedCompositeInput::new_with_diagnostic(&plan.prepared, &plan.admitted,
            |message| metadata.error(format_args!("{message}")))?;
        Ok(())
    }
    fn ingress_error(error: MediaIngressError, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Error {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        if let Err(refusal) = metadata.controls::<(MediaIngressError, Option<&eredu_nn::workspace::WorkspaceContext>)>() {
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
        self.validate_media_state(plan, state)?;
        let mut pending = prepare_composite_ingress::<B>(
            PreparedCompositeInput::new(&plan.prepared, &plan.admitted).map_err(Error::backend)?,
            context,
        )?;
        let pixels = pending
            .pixels
            .take()
            .ok_or_else(|| Error::backend("Muse media source has no patches"))?;
        let (hidden, vision) = self
            .static_modules
            .vision
            .as_mut()
            .ok_or_else(|| Error::backend("Muse media source has no selected vision owner"))?
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
        self.validate_media_state(plan, state)?;
        let mut pending = prepare_composite_ingress::<B>(
            PreparedCompositeInput::new(&plan.prepared, &plan.admitted).map_err(Error::backend)?,
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
                        Error::backend("Muse encoder continuation has no selected vision owner")
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
        _plan: &Self::IngressPlan,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Ingress, Error> {
        let pending = forward
            .pending_media
            .as_ref()
            .ok_or_else(|| Error::backend("Muse cut has no original semantic input"))?;
        let projected = forward
            .media_output
            .as_ref()
            .ok_or_else(|| Error::backend("Muse cut has no completed normalized media"))?;
        let rows = pending
            .tokens
            .iter()
            .zip(&pending.media)
            .filter(|(_, media)| **media)
            .try_fold(0i32, |total, (tokens, _)| total.checked_add(tokens.dim(1)))
            .ok_or_else(|| Error::backend("Muse retained media row count overflows"))?;
        if projected.shape() != [rows, self.args.hidden_size] {
            return Err(Error::backend(
                "Muse cut has incomplete projected media geometry",
            ));
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
        let start = i32::try_from(span.input.start).map_err(Error::backend)?;
        let end = i32::try_from(span.input.end).map_err(Error::backend)?;
        if start < 0 || end <= start || end as u64 > plan.geometry.input_positions {
            return Err(Error::backend("Muse decoder span exceeds original input"));
        }
        let mut parts = Vec::new();
        let mut media = Vec::new();
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
                .ok_or_else(|| Error::backend("Muse decoder position overflows"))?;
            if *is_media {
                media_position = media_position
                    .checked_add(count)
                    .ok_or_else(|| Error::backend("Muse media position overflows"))?;
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
        self.validate_partition_state(state)?;
        if !state.layout().is_empty()
            && u64::try_from(state.layer(0).map_err(Error::backend)?.offset()).ok()
                != Some(plan.geometry.cached_positions)
        {
            return Err(Error::backend(
                "Muse source cached position differs from actual selected state",
            ));
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
    fn prepared_ingress_input(
        plan: &Self::IngressPlan,
    ) -> PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan> {
        PreparedCompositeInput::new(&plan.prepared, &plan.admitted)
            .expect("plan retains exact original admission")
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
    fn media_primary_ingress_collectives(
        &self,
        plan: &Self::IngressPlan,
        span: &eredu_runtime::prefill::PrefillChunk,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, String> {
        let mut offset = 0u64;
        let mut positions = Vec::new();
        for part in plan.admitted.parts() {
            let (length, text) = match part {
                MuseGlimmerInputPartPlan::TextTokens { positions } => (*positions, true),
                MuseGlimmerInputPartPlan::Vision { ingress, .. } => {
                    (ingress.placeholder_count, false)
                }
            };
            let end = offset
                .checked_add(length)
                .ok_or("Muse span position overflows")?;
            let start = offset.max(span.input.start);
            let stop = end.min(span.input.end);
            if text && start < stop {
                positions.push(stop - start);
            }
            offset = end;
        }
        crate::composite_execution::segmented_token_ingress_collectives(
            positions,
            self.args.hidden_size,
            tensor_partitions,
        )
    }
}
